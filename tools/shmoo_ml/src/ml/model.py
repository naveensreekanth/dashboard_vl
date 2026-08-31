"""
ShmooModel
----------
LightGBM binary classifier + RANSAC linear boundary extractor.
Supports single-die and multi-device (D0001 - D0010) boundary extraction,
performance ranking, high/low performer identification, and worst-case population guardbanding.
"""

import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError, Exception):
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ── Feature column lists ──────────────────────────────────────────────────────
FEATURE_COLS_BASE = [
    'VDD_V', 'Frequency_GHz',
    'vdd_freq_product', 'freq_per_volt',
    'vdd_squared', 'freq_squared',
    'vdd_norm', 'freq_norm',
]
FEATURE_COLS_OPTIONAL = [
    'Margin_GHz', 'Timing_ns', 'Current_mA',
    'Leakage_mA', 'fault_rate', 'pattern_fail_rate',
]

# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class ShmooResults:
    accuracy:                 float
    cv_accuracy:              float
    cv_std:                   float
    boundary_slope:           float
    boundary_intercept:       float
    boundary_r2:              float
    recommended_vdd:          float
    recommended_freq:         float
    voltage_margin_v:         float
    freq_margin_ghz:          float
    yield_by_vdd:             Dict[float, float]
    fmax_by_vdd:              Dict[float, float]
    critical_fault_patterns: List[dict]
    timing_fail_patterns:     List[dict]
    failure_code_dist:        Dict[str, int]
    n_pass:                   int
    n_fail:                   int
    predictions:              np.ndarray
    probabilities:            np.ndarray
    classification_report:    str
    # Multi-die additions
    is_multi_die:             bool = False
    die_results:              Dict[str, dict] = field(default_factory=dict)
    die_rankings:             List[dict] = field(default_factory=list)
    high_performer:           Optional[dict] = None
    low_performer:            Optional[dict] = None
    # Automated Shmoo plot classification & diagnostic attributes
    shmoo_plot_type:          str = "Normal Shmoo"
    shmoo_type_description:   str = ""
    # kept for plot overlay
    ransac: object = field(default=None, repr=False)


# ── Model class ───────────────────────────────────────────────────────────────
class ShmooModel:
    def __init__(self):
        self.lgbm:         Optional[lgb.LGBMClassifier] = None
        self.ransac:       Optional[RANSACRegressor]     = None
        self.feature_cols: Optional[List[str]]           = None
        self.results:      Optional[ShmooResults]        = None

    def train_and_evaluate(self, df: pd.DataFrame,
                           progress_cb=None) -> ShmooResults:
        """
        Full pipeline: feature selection → LightGBM CV + fit → RANSAC boundary → Multi-Die Ranking.
        progress_cb: optional callable(step: int, total: int, msg: str)
        """
        self.feature_cols = self._get_features(df)
        X = df[self.feature_cols].values.astype(np.float32)
        y = df['label'].values.astype(int)

        total_steps = 4

        # ── Step 1: Cross-validation ──────────────────────────────────────────
        if progress_cb:
            progress_cb(1, total_steps, "Running 5-fold cross-validation…")

        if HAS_LIGHTGBM:
            lgbm_params = self._default_params()
            tmp_clf = lgb.LGBMClassifier(**lgbm_params)
            self.lgbm = lgb.LGBMClassifier(**lgbm_params)
        else:
            tmp_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=42)
            self.lgbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=42)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(tmp_clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

        # ── Step 2: Final fit on full dataset ─────────────────────────────────
        if progress_cb:
            progress_cb(2, total_steps, "Training Gradient Boosting Model on full dataset…")

        self.lgbm.fit(X, y)
        preds = self.lgbm.predict(X)
        probs = self.lgbm.predict_proba(X)[:, 1]
        acc   = accuracy_score(y, preds)
        report = classification_report(y, preds, target_names=['FAIL', 'PASS'])

        # ── Step 3: RANSAC boundary extraction ───────────────────────────────
        if progress_cb:
            progress_cb(3, total_steps, "Extracting RANSAC pass/fail boundary…")

        boundary = self._extract_boundary(df)
        vdd_pts  = np.array(list(boundary.keys()),   dtype=np.float32).reshape(-1, 1)
        fmax_pts = np.array(list(boundary.values()), dtype=np.float32)

        self.ransac = RANSACRegressor(
            LinearRegression(),
            min_samples=max(0.5, min(0.9, 2.0 / len(vdd_pts))),
            residual_threshold=0.05,
            random_state=42,
        )
        self.ransac.fit(vdd_pts, fmax_pts)

        slope     = float(self.ransac.estimator_.coef_[0])
        intercept = float(self.ransac.estimator_.intercept_)
        fmax_pred = self.ransac.predict(vdd_pts)
        ss_res    = float(np.sum((fmax_pts - fmax_pred) ** 2))
        ss_tot    = float(np.sum((fmax_pts - fmax_pts.mean()) ** 2))
        r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        # ── Step 4: Metrics, multi-die ranking & operating point ───────────────
        if progress_cb:
            progress_cb(4, total_steps, "Computing margins, multi-device ranking & recommended OP…")

        vdd_vals = np.sort(df['VDD_V'].unique())
        rec_vdd  = float(np.percentile(vdd_vals, 55))
        rec_freq = float(self.ransac.predict([[rec_vdd]])[0]) * 0.90   # 10% guardband

        # Check multi-device / multi-die presence
        die_col = None
        for candidate in ['Die_ID', 'Test_ID', 'Wafer_ID']:
            if candidate in df.columns and df[candidate].nunique() > 1:
                die_col = candidate
                break

        die_results = {}
        die_rankings = []
        high_performer = None
        low_performer = None
        is_multi_die = False

        if die_col:
            is_multi_die = True
            unique_dies = sorted(df[die_col].unique())
            for die_id in unique_dies:
                die_df = df[df[die_col] == die_id]
                if len(die_df) < 5: continue
                d_boundary = self._extract_boundary(die_df)
                if not d_boundary: continue
                d_vdd = np.array(list(d_boundary.keys()), dtype=np.float32).reshape(-1, 1)
                d_fmax = np.array(list(d_boundary.values()), dtype=np.float32)

                try:
                    d_ransac = RANSACRegressor(
                        LinearRegression(),
                        min_samples=max(0.5, min(0.9, 2.0 / len(d_vdd))),
                        residual_threshold=0.05,
                        random_state=42,
                    )
                    d_ransac.fit(d_vdd, d_fmax)
                    d_slope = float(d_ransac.estimator_.coef_[0])
                    d_intercept = float(d_ransac.estimator_.intercept_)
                    d_fmax_pred = d_ransac.predict(d_vdd)
                    d_ss_res = float(np.sum((d_fmax - d_fmax_pred) ** 2))
                    d_ss_tot = float(np.sum((d_fmax - d_fmax.mean()) ** 2))
                    d_r2 = 1.0 - d_ss_res / d_ss_tot if d_ss_tot > 0 else 1.0
                except Exception:
                    d_slope, d_intercept, d_r2 = slope, intercept, r2

                fmax_at_nom = float(d_slope * rec_vdd + d_intercept)
                d_rec_freq = float(fmax_at_nom * 0.90)
                d_pass_rate = float((die_df['label'] == 1).mean())

                die_info = {
                    'die_id': str(die_id),
                    'slope': round(d_slope, 4),
                    'intercept': round(d_intercept, 4),
                    'r2': round(d_r2, 4),
                    'fmax_at_nom': round(fmax_at_nom, 3),
                    'recommended_vdd': round(rec_vdd, 3),
                    'recommended_freq': round(d_rec_freq, 3),
                    'pass_rate': round(d_pass_rate, 4),
                    'n_pass': int((die_df['label'] == 1).sum()),
                    'n_fail': int((die_df['label'] == 0).sum()),
                    'boundary': {float(k): float(v) for k, v in d_boundary.items()}
                }
                die_results[str(die_id)] = die_info
                die_rankings.append(die_info)

            if die_rankings:
                # Rank devices by maximum frequency at nominal VDD in descending order
                die_rankings.sort(key=lambda x: x['fmax_at_nom'], reverse=True)
                high_performer = die_rankings[0]
                low_performer = die_rankings[-1]

                # Population recommendation bounded by Low Performer (100% yield guarantee across all dies)
                rec_freq = min(rec_freq, low_performer['recommended_freq'])

        min_pass_vdd    = float(df[df['label'] == 1]['VDD_V'].min())
        v_margin        = rec_vdd - min_pass_vdd
        boundary_at_rec = float(slope * rec_vdd + intercept)
        f_margin        = boundary_at_rec - rec_freq

        yield_by_vdd = df.groupby('VDD_V')['label'].mean().to_dict()
        fmax_by_vdd  = {
            float(vdd): float(boundary[vdd])
            for vdd in sorted(df['VDD_V'].unique())
            if vdd in boundary
        }

        # ── Generalized Critical Functional Fault Extraction (Scan & M-BIST) ──
        HARD_FAULT_CODES_EXCLUDE = {'FREQ_MARGIN', 'NA'}
        critical_patterns: List[dict] = []

        if 'Failure_Code' in df.columns:
            hard_fail_df = df[~df['Failure_Code'].isin(HARD_FAULT_CODES_EXCLUDE)]
            if len(hard_fail_df):
                if 'Pattern_ID' in df.columns:
                    group_cols = ['Pattern_ID']
                    label_fn = lambda r: str(r['Pattern_ID'])
                elif {'March_Algorithm', 'Memory_Instance'}.issubset(df.columns):
                    group_cols = ['March_Algorithm', 'Memory_Instance']
                    label_fn = lambda r: f"{r['March_Algorithm']} / {r['Memory_Instance']}"
                else:
                    group_cols = None

                if group_cols:
                    counts = hard_fail_df.groupby(group_cols).size().sort_values(ascending=False)
                    for idx, c in counts.head(10).items():
                        if isinstance(idx, tuple):
                            sub_mask = np.ones(len(hard_fail_df), dtype=bool)
                            for gc, val in zip(group_cols, idx):
                                sub_mask &= (hard_fail_df[gc] == val)
                            sub_df = hard_fail_df[sub_mask]
                            row_dict = dict(zip(group_cols, idx))
                        else:
                            sub_df = hard_fail_df[hard_fail_df[group_cols[0]] == idx]
                            row_dict = {group_cols[0]: idx}
                        
                        mode_code = sub_df['Failure_Code'].mode()
                        fault_type = str(mode_code.iloc[0]) if len(mode_code) else 'HARD_DEFECT'
                        source_label = label_fn(row_dict)
                        critical_patterns.append({
                            'source': source_label,
                            'pattern': source_label,
                            'fail_count': int(c),
                            'fault_type': fault_type,
                        })

        fail_codes = {str(k): int(v)
                      for k, v in df['Failure_Code'].value_counts().items()}

        shmoo_plot_type, shmoo_type_description = self._classify_shmoo_type(df, boundary, fail_codes)

        self.results = ShmooResults(
            accuracy=float(acc),
            cv_accuracy=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            boundary_slope=slope,
            boundary_intercept=intercept,
            boundary_r2=float(r2),
            recommended_vdd=float(rec_vdd),
            recommended_freq=float(rec_freq),
            voltage_margin_v=float(v_margin),
            freq_margin_ghz=float(f_margin),
            yield_by_vdd={float(k): float(v) for k, v in yield_by_vdd.items()},
            fmax_by_vdd=fmax_by_vdd,
            critical_fault_patterns=critical_patterns,
            timing_fail_patterns=critical_patterns,
            failure_code_dist=fail_codes,
            n_pass=int((df['label'] == 1).sum()),
            n_fail=int((df['label'] == 0).sum()),
            predictions=preds,
            probabilities=probs,
            classification_report=report,
            is_multi_die=is_multi_die,
            die_results=die_results,
            die_rankings=die_rankings,
            high_performer=high_performer,
            low_performer=low_performer,
            shmoo_plot_type=shmoo_plot_type,
            shmoo_type_description=shmoo_type_description,
            ransac=self.ransac,
        )
        return self.results

    def save(self, path: str) -> None:
        joblib.dump({
            'lgbm':         self.lgbm,
            'ransac':       self.ransac,
            'feature_cols': self.feature_cols,
        }, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.lgbm         = data['lgbm']
        self.ransac       = data['ransac']
        self.feature_cols = data['feature_cols']

    def _get_features(self, df: pd.DataFrame) -> List[str]:
        cols = FEATURE_COLS_BASE.copy()
        for c in FEATURE_COLS_OPTIONAL:
            if c in df.columns and c not in cols:
                cols.append(c)
        return cols

    @staticmethod
    def _extract_boundary(df: pd.DataFrame) -> dict:
        """For each VDD level find Fmax of last PASS (O(N) groupby)."""
        boundary = {}
        for vdd, grp in df.groupby('VDD_V'):
            pass_rows = grp[grp['label'] == 1]
            if len(pass_rows):
                boundary[float(vdd)] = float(pass_rows['Frequency_GHz'].max())
        return boundary

    @staticmethod
    def _default_params() -> dict:
        return {
            'objective':        'binary',
            'metric':           'binary_error',
            'n_estimators':     300,
            'learning_rate':    0.05,
            'num_leaves':       63,
            'max_depth':        7,
            'min_child_samples': 10,
            'subsample':        0.8,
            'colsample_bytree': 0.8,
            'reg_alpha':        0.1,
            'reg_lambda':       0.1,
            'verbose':          -1,
            'n_jobs':           -1,
            'random_state':     42,
        }

    @staticmethod
    def _classify_shmoo_type(df: pd.DataFrame, boundary: dict, fail_codes: dict) -> tuple:
        """
        Automatically classifies the Shmoo plot pattern and identifies the root cause
        (e.g., Low-Voltage Wall vs High-Voltage Hold-Time Wall vs Reverse Speedpath vs Floor vs Finger vs Brick Wall vs Normal).
        """
        codes = set(fail_codes.keys())

        # 1. Brick Wall Shmoo
        if 'BRICK_WALL_INIT' in codes:
            return (
                "Brick Wall Shmoo (Bi-Stable Reset / Initialization Failure)",
                "The plot exhibits characteristic vertical 'brick wall' failing stripes. "
                "This occurs due to bi-stable initialization issues where uninitialized registers or flip-flops "
                "power up randomly into 0 or 1 states without a proper reset pulse, causing intermittent failures "
                "across frequencies regardless of supply voltage."
            )

        # 2. Wall Shmoo: Low VDD Wall vs High VDD Hold-Time Wall
        if 'LOW_VDD_WALL' in codes or 'HOLD_TIME_WALL' in codes:
            if 'LOW_VDD_WALL' in codes and 'HOLD_TIME_WALL' in codes:
                return (
                    "Double Wall Shmoo (Low Voltage Drive Cutoff & High Voltage Hold-Time Wall)",
                    "The plot is bounded by dual vertical walls: (1) Low-Voltage Wall (circuit fails below minimum supply "
                    "due to insufficient gate drive / IR drop), and (2) High-Voltage Wall (circuit fails at elevated VDD "
                    "irrespective of frequency due to fast-path clock skew and hold-time race conditions)."
                )
            elif 'LOW_VDD_WALL' in codes:
                return (
                    "Wall Shmoo (Low-Voltage Gate Drive Cutoff / Voltage Failure)",
                    "The plot exhibits a hard vertical failure wall at low supply voltages. "
                    "This occurs due to voltage failure: below a critical VDD threshold, transistor gate drive is insufficient "
                    "or IR drop collapses logic states, causing the circuit to fail across all operating frequencies."
                )
            else:
                return (
                    "Wall Shmoo (High-Voltage Hold-Time Race / Fast-Path Skew)",
                    "The plot exhibits a hard vertical failure wall at high supply voltages. "
                    "This occurs due to hold-time race conditions: elevated VDD causes circuits to run faster, resulting in "
                    "clock skew and data latching at incorrect clock edges regardless of operating frequency."
                )

        # 3. Reverse Speedpath Shmoo
        if 'REVERSE_SPEEDPATH_LEAKAGE' in codes:
            return (
                "Reverse Speedpath Shmoo (Dynamic Node Leakage at High VDD)",
                "The plot exhibits an inverted pass/fail boundary where maximum operating frequency (Fmax) decreases "
                "as supply voltage increases. This occurs because elevated VDD accentuates subthreshold and gate leakage "
                "on weak dynamic nodes, causing charge loss and severe RC delay degradation before cycle evaluation."
            )

        # 4. Floor Shmoo
        if 'FLOOR_LEAKAGE_FAIL' in codes:
            return (
                "Floor Shmoo (Low-Frequency Dynamic Node Discharge)",
                "The plot exhibits a horizontal failing floor at low frequencies. "
                "The circuit operates successfully at mid/high frequencies but fails at lower clock rates because "
                "extended clock periods give unrefreshed dynamic nodes enough time to leak away their stored charge."
            )

        # 5. Finger Shmoo
        if 'FINGER_RESONANCE_COUPLING' in codes:
            return (
                "Finger Shmoo (Aggressor-Victim Crosstalk Resonance)",
                "The plot displays distinct failing 'comb fingers' (notches) protruding into the pass region. "
                "This occurs due to inductive/capacitive crosstalk coupling where specific harmonic clock frequencies "
                "and pattern alignments trigger resonance between aggressor and victim signal traces."
            )

        # 6. Power & IR Drop Issue Shmoo
        if 'POWER_IR_DROP_FAIL' in codes:
            return (
                "Power & IR Drop Issue Shmoo (Supply Grid Rail Collapse)",
                "The plot shows widespread high-frequency failures caused by high switching current power consumption. "
                "As test frequency increases, dynamic switching power (P = C * V^2 * f) causes internal IR drop on the supply grid, "
                "collapsing internal VDD below nominal requirements and causing timing failure."
            )

        # 7. Hold Time Race Issue Shmoo
        if 'HOLD_TIME_VIOLATION' in codes:
            return (
                "Hold Time Race Issue Shmoo (Fast-Path Clock Skew Violation)",
                "The plot exhibits failures at high supply voltages (VDD > 1.05V) irrespective of operating frequency. "
                "Elevated supply voltage accelerates combinational logic propagation speed, causing data signals to race ahead "
                "of the capture clock edge and fail hold-time timing constraints."
            )

        # 8. Marginality Issue Shmoo
        if 'MARGINALITY_VDDL_FAIL' in codes or 'MARGINALITY_VDDH_FAIL' in codes or 'MARGINALITY_IR_DROP' in codes:
            return (
                "Marginality Issue Shmoo (VDDL / VDDH Voltage Rail Corner Failure)",
                "The plot shows failure clusters concentrated at extreme voltage corners (VDDL low-rail and VDDH high-rail). "
                "This occurs due to circuit marginality: low VDD fails setup timing under reduced drive, while high VDD fails "
                "under PLL jitter and rail droop."
            )

        # ── Spatial Pattern Heuristics (if generic failure codes like FREQ_MARGIN / TIMING are used) ──
        vdds = sorted(df['VDD_V'].unique())
        freqs = sorted(df['Frequency_GHz'].unique())

        # Check low voltage wall spatially
        if len(vdds) >= 4:
            low_v_val = vdds[min(2, len(vdds)-1)]
            low_v_df = df[df['VDD_V'] <= low_v_val]
            if len(low_v_df) and (low_v_df['label'] == 0).mean() > 0.90:
                return (
                    "Wall Shmoo (Low-Voltage Gate Drive Cutoff / Voltage Failure)",
                    f"The plot exhibits a hard vertical wall at low supply voltages (VDD < {low_v_val:.2f} V). "
                    "This is caused by voltage failure: below minimum gate drive, logic gates fail to switch properly regardless of frequency."
                )

        # Check floor shmoo spatially
        if len(freqs) >= 4:
            low_f_val = freqs[min(3, len(freqs)-1)]
            low_f_df = df[df['Frequency_GHz'] <= low_f_val]
            if len(low_f_df) and (low_f_df['label'] == 0).mean() > 0.85 and (df['label'] == 1).sum() > 0:
                return (
                    "Floor Shmoo (Low-Frequency Dynamic Node Discharge)",
                    f"The plot displays a failing floor at lower clock frequencies (Frequency < {low_f_val:.2f} GHz). "
                    "The circuit operates at higher frequencies but fails at low rates due to dynamic node leakage over long cycle times."
                )

        # Default: Normal Shmoo
        return (
            "Normal Shmoo (Well-Behaved Linear Speedpath)",
            "The plot exhibits a standard, well-behaved linear pass/fail boundary. "
            "Maximum operating frequency (Fmax) scales smoothly with supply voltage (VDD), indicating stable silicon "
            "where failures are purely frequency/voltage-limited."
        )
