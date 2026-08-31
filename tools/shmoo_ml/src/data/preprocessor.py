"""
ShmooPreprocessor
-----------------
Loads CSV / Excel, normalizes column names & string formatting, auto-derives
missing Pass/Fail status from failure codes/margins, auto-detects single vs.
multi-die datasets, and engineers features for the ML model.

Time complexity : O(N)  for all operations (N = number of rows)
Space complexity: O(N)  – one copy of the dataframe with extra columns
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Alias mapping for flexible schema ingestion
COLUMN_ALIASES = {
    'VDD_V': ['VDD_V', 'VDD', 'V', 'VDD(V)', 'VDD_VOLT', 'VOLTAGE'],
    'Frequency_GHz': ['FREQUENCY_GHZ', 'FREQUENCY', 'FREQ_GHZ', 'FREQ', 'FREQ(GHZ)', 'F_GHZ'],
    'Test_Result': ['TEST_RESULT', 'TEST_RESULTS', 'RESULT', 'PASS_FAIL', 'STATUS', 'TEST_STATUS', 'OUTCOME'],
    'Failure_Code': ['FAILURE_CODE', 'FAIL_CODE', 'FAILURECODE', 'FAILURE', 'ERROR_CODE', 'FAIL_TYPE'],
    'Point_ID': ['POINT_ID', 'POINTID', 'ID', 'POINT', 'INDEX'],
}

OPTIONAL_COLS = {
    'Lot_ID', 'Wafer_ID', 'Die_ID', 'Temperature_C',
    'Current_mA', 'Timing_ns', 'Leakage_mA',
    'Test_Time_ms', 'Pattern_ID', 'Test_ID', 'Margin_GHz',
    'March_Algorithm', 'Memory_Instance', 'Memory_Address',
}
DIE_ID_COLS = ['Lot_ID', 'Wafer_ID', 'Die_ID']


class ShmooPreprocessor:
    def __init__(self):
        self.is_multi_die: bool = False
        self.die_groups: list  = []
        self.df_raw: pd.DataFrame | None = None
        self.df_processed: pd.DataFrame | None = None

    def load(self, filepath: str) -> dict:
        """Load file, normalize schema, return metadata dict."""
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix in ('.xlsx', '.xls'):
            df = pd.read_excel(filepath)
        elif suffix == '.csv':
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: '{suffix}'. Use CSV or Excel.")

        df = self._normalize_columns(df)
        self._validate_and_derive_columns(df)
        self.df_raw = df
        return self._detect_structure(df)

    def process(self) -> pd.DataFrame:
        """Engineer features; call after load(). Returns processed DataFrame."""
        if self.df_raw is None:
            raise RuntimeError("Call load() before process().")

        df = self.df_raw.copy()

        # ── Target label ──────────────────────────────────────────────────────
        pass_mask = df['Test_Result'].isin(['PASS', 'PASSED', '1', 'TRUE', 'P'])
        df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
        df['label']       = pass_mask.astype(int)

        # ── Core engineered features ──────────────────────────────────────────
        df['vdd_freq_product'] = df['VDD_V'] * df['Frequency_GHz']
        df['freq_per_volt']    = df['Frequency_GHz'] / df['VDD_V']
        df['vdd_squared']      = df['VDD_V'] ** 2
        df['freq_squared']     = df['Frequency_GHz'] ** 2

        vdd_min, vdd_max   = float(df['VDD_V'].min()), float(df['VDD_V'].max())
        freq_min, freq_max = float(df['Frequency_GHz'].min()), float(df['Frequency_GHz'].max())

        denom_v = (vdd_max  - vdd_min)  or 1.0
        denom_f = (freq_max - freq_min) or 1.0

        df['vdd_norm']  = (df['VDD_V']           - vdd_min)  / denom_v
        df['freq_norm'] = (df['Frequency_GHz']   - freq_min) / denom_f

        # ── Optional numerical features – median-impute if missing values ─────
        for col in ('Margin_GHz', 'Timing_ns', 'Current_mA', 'Leakage_mA'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

        # ── Generalized Fault-Rate Feature (Works for Scan or M-BIST) ─────────
        if 'Pattern_ID' in df.columns:
            group_key = 'Pattern_ID'
        elif {'March_Algorithm', 'Memory_Instance'}.issubset(df.columns):
            group_key = ['March_Algorithm', 'Memory_Instance']
        else:
            group_key = None

        if group_key is not None:
            fail_rate = 1.0 - df.groupby(group_key)['label'].transform('mean')
            df['fault_rate'] = fail_rate
            df['pattern_fail_rate'] = fail_rate

        self.df_processed = df
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map case-insensitive column aliases to standard internal names."""
        column_map = {}
        upper_to_original = {col.upper(): col for col in df.columns}

        for target_name, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias.upper() in upper_to_original:
                    column_map[upper_to_original[alias.upper()]] = target_name
                    break

        return df.rename(columns=column_map)

    def _validate_and_derive_columns(self, df: pd.DataFrame) -> None:
        """Check essential columns and smart-derive missing values if possible."""
        # 1. Check mandatory geometric axes
        missing_axes = []
        if 'VDD_V' not in df.columns:
            missing_axes.append('VDD_V (Voltage)')
        if 'Frequency_GHz' not in df.columns:
            missing_axes.append('Frequency_GHz (Frequency)')

        if missing_axes:
            raise ValueError(
                f"Missing mandatory Shmoo axes: {missing_axes}.\n"
                f"Found columns in file: {list(df.columns)}"
            )

        # Ensure numeric values for VDD and Frequency
        df['VDD_V'] = pd.to_numeric(df['VDD_V'], errors='coerce')
        df['Frequency_GHz'] = pd.to_numeric(df['Frequency_GHz'], errors='coerce')

        # 2. Auto-generate Point_ID if missing
        if 'Point_ID' not in df.columns:
            df['Point_ID'] = np.arange(1, len(df) + 1)

        # 3. Clean Failure_Code if present
        if 'Failure_Code' in df.columns:
            df['Failure_Code'] = df['Failure_Code'].fillna('NA').astype(str).str.strip().str.upper()
            df['Failure_Code'] = df['Failure_Code'].replace({'NAN': 'NA', 'NONE': 'NA', '': 'NA', 'NULL': 'NA', 'N/A': 'NA'})
        else:
            df['Failure_Code'] = 'NA'

        # 4. Derive or clean Test_Result
        if 'Test_Result' in df.columns:
            df['Test_Result'] = df['Test_Result'].astype(str).str.strip().str.upper()
        else:
            # Smart-derive Test_Result from Failure_Code or Margin_GHz if Test_Result column is omitted
            if 'Failure_Code' in df.columns:
                is_pass = df['Failure_Code'].isin(['NA', 'NONE', '', 'NAN', '0', 'PASS', 'PASSED'])
                df['Test_Result'] = np.where(is_pass, 'PASS', 'FAIL')
            elif 'Margin_GHz' in df.columns:
                df['Test_Result'] = np.where(pd.to_numeric(df['Margin_GHz'], errors='coerce') >= 0, 'PASS', 'FAIL')
            else:
                raise ValueError(
                    "Missing 'Test_Result' column in dataset.\n"
                    "Please include a 'Test_Result' column (PASS/FAIL) or a 'Failure_Code' column."
                )

        # If Failure_Code was originally missing, set default code for FAIL rows
        if 'Failure_Code' not in df.columns or (df['Failure_Code'] == 'NA').all():
            fail_mask = df['Test_Result'] == 'FAIL'
            df['Failure_Code'] = np.where(fail_mask, 'FREQ_MARGIN', 'NA')

    def _detect_structure(self, df: pd.DataFrame) -> dict:
        """Auto-detect single vs. multi-die and build metadata."""
        id_cols = [c for c in DIE_ID_COLS if c in df.columns]

        if id_cols:
            n_groups = df.groupby(id_cols).ngroups
        else:
            n_groups = 1

        self.is_multi_die = n_groups > 1
        self.die_groups   = id_cols if self.is_multi_die else []

        test_result_clean = df['Test_Result'].astype(str).str.strip().str.upper()
        pass_mask = test_result_clean.isin(['PASS', 'PASSED', '1', 'TRUE', 'P'])
        fail_code_clean = df['Failure_Code'].fillna('NA').astype(str).str.strip().str.upper().replace({'NAN': 'NA', 'NONE': 'NA', '': 'NA', 'NULL': 'NA', 'N/A': 'NA'})

        return {
            'n_points':      len(df),
            'n_dies':        n_groups,
            'is_multi_die':  self.is_multi_die,
            'die_id_cols':   id_cols,
            'vdd_range':     [float(df['VDD_V'].min()),          float(df['VDD_V'].max())],
            'freq_range':    [float(df['Frequency_GHz'].min()),  float(df['Frequency_GHz'].max())],
            'pass_rate':     float(pass_mask.mean()),
            'n_pass':        int(pass_mask.sum()),
            'n_fail':        int((~pass_mask).sum()),
            'failure_codes': fail_code_clean.value_counts().to_dict(),
            # Lot / wafer / die info if present
            'lot_id':   str(df['Lot_ID'].iloc[0])   if 'Lot_ID'   in df.columns else 'N/A',
            'wafer_id': str(df['Wafer_ID'].iloc[0]) if 'Wafer_ID' in df.columns else 'N/A',
            'die_id':   str(df['Die_ID'].iloc[0])   if 'Die_ID'   in df.columns else 'N/A',
            'temp_c':   float(df['Temperature_C'].iloc[0]) if 'Temperature_C' in df.columns else None,
        }
