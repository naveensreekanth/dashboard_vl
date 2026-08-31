import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from ..config.settings import (
    ALL_MODEL_FEATURES,
    FEATURE_MATRIX_FORBIDDEN_COLS,
    MODEL_VERSION,
    ARTIFACTS_DIR,
    IDENTIFIER_COLS,
    TARGET_COL,
    MONTH_12_OUTCOMES_FILE
)
from ..data.ingestion import load_all_datasets, load_pre_retest_workbook
from ..data.preprocessing import prepare_xy
from ..data.validation import (
    assert_no_leakage_in_feature_matrix,
    format_pre_retest_validation_error,
    validate_pre_retest_upload,
)
from ..models.trainer import train_and_compare_models, train_final_deployment_model
from ..explainability.shap_explainer import ModelExplainer
from ..decision.decision_policy import (
    DOCX_REFERENCE_THRESHOLD,
    POLICY_LABEL,
    apply_decision_policy,
    apply_batch_decision_policy,
)
from .online_learning import (
    ADAPTED_PROB_COL,
    BASE_PROB_COL,
    RLSCalibrator,
)
from ..validation.outcome_validator import validate_recommendations_against_outcomes
from ..kpis.business_impact import (
    ESTIMATED_TIME_COL,
    attach_estimated_retest_times,
    build_retest_time_lookup,
    calculate_time_and_cost_impact,
    estimate_retest_time_seconds,
)

ARTIFACT_FILE = os.path.join(ARTIFACTS_DIR, "model_artifacts.pkl")


class MLService:
    """
    Shared ML Service Layer providing unified prediction, explainability,
    and evaluation functionality to both FastAPI and Streamlit.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model_name = "XGBoost"
        self.model_version = MODEL_VERSION
        self.datasets = load_all_datasets()
        self.comparison_results = None
        self.final_model = None
        self.explainer = None
        self.selection_reason = ""
        self.month_12_outcomes = None
        self.retest_time_lookup = self._build_retest_time_lookup()
        self.rls_calibrator = RLSCalibrator()
        self._initialize_pipeline()

    def _build_retest_time_lookup(self) -> Dict[str, Any]:
        frames = [
            self.datasets[key]
            for key in ("month_0", "month_6")
            if key in self.datasets and self.datasets[key] is not None
        ]
        if not frames:
            return build_retest_time_lookup(None)
        return build_retest_time_lookup(pd.concat(frames, ignore_index=True))

    def attach_estimated_retest_times(self, df: pd.DataFrame) -> pd.DataFrame:
        return attach_estimated_retest_times(df, self.retest_time_lookup)

    def estimate_retest_time_for_event(self, event: pd.DataFrame) -> float:
        times = estimate_retest_time_seconds(event, self.retest_time_lookup)
        if len(times) == 0:
            return 0.0
        return float(times.iloc[0])

    def get_cost_impact(self, df: pd.DataFrame, cost_per_hour: Optional[float] = None) -> Dict[str, Any]:
        framed = df if ESTIMATED_TIME_COL in df.columns else self.attach_estimated_retest_times(df)
        return calculate_time_and_cost_impact(framed, cost_per_hour=cost_per_hour)

    def _initialize_pipeline(self):
        """Initializes and trains or loads the pipeline artifacts."""
        df_m0 = self.datasets["month_0"]
        df_m6 = self.datasets["month_6"]
        df_combined = pd.concat([df_m0, df_m6], ignore_index=True)

        self.comparison_results = train_and_compare_models(df_m0, df_m6)
        self.model_name = self.comparison_results["best_model_name"]
        self.selection_reason = self.comparison_results.get("selection_reason", "")

        self.final_model = train_final_deployment_model(df_combined, model_name=self.model_name)
        self.explainer = ModelExplainer(self.final_model, df_combined)

        try:
            with open(ARTIFACT_FILE, "wb") as f:
                pickle.dump({
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "final_model": self.final_model,
                    "selection_reason": self.selection_reason,
                }, f)
        except Exception as e:
            print(f"Warning: Could not cache artifacts: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the active ML model."""
        return {
            "model": self.model_name,
            "version": self.model_version,
            "target": "RETEST_BENEFICIAL",
            "features": ALL_MODEL_FEATURES,
            "training_data": ["Month 0", "Month 6"],
            "inference_data": "Month 12",
            "calibration_enabled": True,
            "selection_reason": self.selection_reason,
            "decision_policy_label": POLICY_LABEL,
            "decision_policy_threshold": DOCX_REFERENCE_THRESHOLD,
        }

    def _reject_leakage_keys(self, event_dict: Dict[str, Any]) -> None:
        forbidden = set(FEATURE_MATRIX_FORBIDDEN_COLS)
        leakage_keys = [k for k in event_dict.keys() if k in forbidden]
        if leakage_keys:
            raise ValueError(f"Input rejected: Outcome leakage fields detected: {leakage_keys}")

    def predict_single_event(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates P(RETEST_BENEFICIAL) then applies the isolated DOCX-reference policy.
        Probability is not modified by the policy.
        """
        self._reject_leakage_keys(event_dict)

        df_single = pd.DataFrame([event_dict])
        X_clean, _ = prepare_xy(df_single, is_inference=True)
        assert_no_leakage_in_feature_matrix(X_clean.columns)
        prob = float(self.final_model.predict_proba(X_clean)[0, 1])
        base = float(np.clip(prob, 0.0, 1.0))
        adapted = float(self.rls_calibrator.adapt_probability(base))
        active = bool(self.rls_calibrator.is_active)
        final = adapted if active else base
        policy = apply_decision_policy(final)

        est_time = self.estimate_retest_time_for_event(df_single)
        predicted_time = est_time if policy["is_retest"] else 0.0
        result = {
            "probability_retest_beneficial": round(final, 4),
            "probability_percent": round(final * 100.0, 2),
            "probability_base": round(base, 4),
            "probability_adapted": round(adapted, 4),
            "online_adaptation_active": active,
            "recommendation": policy["recommendation"],
            "policy_label": policy["policy_label"],
            "policy_threshold": policy["policy_threshold"],
            "estimated_retest_time_sec": round(est_time, 2),
            "predicted_retest_time_sec": round(predicted_time, 2),
            "model": self.model_name,
            "version": self.model_version,
        }
        if event_dict.get("Device_ID") is not None:
            result["Device_ID"] = str(event_dict["Device_ID"])
        if event_dict.get("Failure_Event") is not None:
            result["Failure_Event"] = int(event_dict["Failure_Event"])
        return result

    def predict_batch_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch prediction for multiple events."""
        predictions = []
        for ev in events:
            predictions.append(self.predict_single_event(ev))
        return predictions

    def _score_pre_retest_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shared inference: drop outcome/leakage columns, score with the deployed model,
        then apply the isolated DOCX-reference policy. Does not train or mutate datasets.
        """
        df_work = df.copy()
        drop_cols = [c for c in FEATURE_MATRIX_FORBIDDEN_COLS if c in df_work.columns]
        if drop_cols:
            df_work = df_work.drop(columns=drop_cols)
        X, _ = prepare_xy(df_work, is_inference=True)
        assert_no_leakage_in_feature_matrix(X.columns)
        probs = self.final_model.predict_proba(X)[:, 1]
        base = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
        adapted = np.asarray(self.rls_calibrator.adapt_probability(base), dtype=float)
        active = bool(self.rls_calibrator.is_active)
        final = adapted if active else base

        df_out = df_work.copy()
        df_out[BASE_PROB_COL] = np.round(base, 4)
        df_out[ADAPTED_PROB_COL] = np.round(adapted, 4)
        df_out["P(RETEST_BENEFICIAL)"] = np.round(final, 4)
        df_out["Probability_%"] = np.round(final * 100.0, 2)
        recs = apply_batch_decision_policy(final)
        df_out["AI_Recommendation"] = recs["AI_Recommendation"].values
        df_out["Policy_Label"] = POLICY_LABEL
        df_out = self.attach_estimated_retest_times(df_out)
        return df_out

    def predict_pre_retest_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score an uploaded pre-retest event table with the existing inference pipeline.
        Does not use Ground_Truth / outcome columns as features and does not change
        the Month 12 sample dataset or the outcome validation pipeline.
        """
        report = validate_pre_retest_upload(df)
        if not report["is_valid"]:
            raise ValueError(format_pre_retest_validation_error(report))
        return self._score_pre_retest_frame(df)

    def load_and_predict_pre_retest_workbook(self, source) -> pd.DataFrame:
        """Read a pre-retest XLSX and run the existing prediction pipeline."""
        df = load_pre_retest_workbook(source)
        return self.predict_pre_retest_table(df)

    def get_month_12_batch_table(self) -> pd.DataFrame:
        """
        Month 12 inference from the inference workbook only.
        Adds P(RETEST_BENEFICIAL) and DOCX-reference AI_Recommendation.
        Does not require or use ground truth.
        """
        return self._score_pre_retest_frame(self.datasets["month_12"])

    def get_historical_validation_table(self) -> pd.DataFrame:
        """
        Month 6 temporal holdout: probabilities from the Month-0-trained selected model,
        recommendations from the isolated DOCX-reference policy, plus actual Ground_Truth.
        """
        df_m6 = self.datasets["month_6"].copy()
        name = self.model_name
        probs = self.comparison_results["results"][name]["val_probabilities_cal"]
        recs = apply_batch_decision_policy(probs)
        df_out = df_m6.copy()
        df_out["P(RETEST_BENEFICIAL)"] = np.round(probs, 4)
        df_out["AI_Recommendation"] = recs["AI_Recommendation"].values
        df_out["Policy_Label"] = POLICY_LABEL
        return df_out

    def get_historical_decision_kpis(self) -> Dict[str, Any]:
        """Post-outcome KPIs for Month 6 holdout using this model's DOCX-reference recommendations."""
        df = self.get_historical_validation_table()
        return validate_recommendations_against_outcomes(
            df[TARGET_COL], df["AI_Recommendation"], events=df
        )

    def get_reference_report_kpis(self) -> Dict[str, Any]:
        """
        Recompute historical/reference KPIs from the AI dataset's stored
        AI_Recommendation vs Ground_Truth. Not Month 12 performance.
        """
        df = self.datasets.get("ai_dataset")
        if df is None or "AI_Recommendation" not in df.columns:
            return {"has_outcomes": False}
        return validate_recommendations_against_outcomes(
            df[TARGET_COL], df["AI_Recommendation"], events=df
        )

    def month_12_outcomes_available(self) -> bool:
        return self.month_12_outcomes is not None and len(self.month_12_outcomes) > 0

    def load_month_12_outcomes(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load actual Month 12 outcomes separately for validation only.
        Never used as prediction input.
        """
        source = path or MONTH_12_OUTCOMES_FILE
        if not os.path.exists(source):
            raise FileNotFoundError(
                f"Month 12 outcome file not found: {source}. "
                "Outcomes must be provided separately and are not used for prediction."
            )
        df_gt = pd.read_excel(source, sheet_name=0)
        if TARGET_COL not in df_gt.columns:
            raise ValueError("Month 12 outcome file must contain Ground_Truth.")
        self.month_12_outcomes = df_gt
        return df_gt

    def get_month_12_validation_table(self) -> Optional[pd.DataFrame]:
        """Join Month 12 predictions with separately loaded outcomes. None if not loaded."""
        if not self.month_12_outcomes_available():
            return None
        preds = self.get_month_12_batch_table()
        gt = self.month_12_outcomes.copy()
        keys = [c for c in IDENTIFIER_COLS if c in preds.columns and c in gt.columns]
        if not keys:
            raise ValueError("Cannot join Month 12 outcomes: missing Device_ID/Failure_Event.")
        merged = preds.merge(
            gt[keys + [c for c in [TARGET_COL, "Retest_Result", "Final_Result"] if c in gt.columns]],
            on=keys,
            how="inner",
            suffixes=("", "_outcome"),
        )
        return merged

    def get_month_12_decision_kpis(self) -> Optional[Dict[str, Any]]:
        table = self.get_month_12_validation_table()
        if table is None:
            return None
        return validate_recommendations_against_outcomes(
            table[TARGET_COL], table["AI_Recommendation"], events=table
        )

    def get_online_learning_status(self) -> Dict[str, Any]:
        """Diagnostic status for the optional RLS calibration layer. Separate from Month 12 outcomes."""
        info = self.rls_calibrator.status()
        info["base_model"] = self.model_name
        return info

    def reset_online_learning(self) -> Dict[str, Any]:
        """Clear RLS state only. Does not touch the trained model, uploads, or outcomes."""
        self.rls_calibrator.reset()
        return self.get_online_learning_status()

    def adapt_probability(self, probabilities) -> np.ndarray:
        """Apply RLS calibration to base probabilities. Pass-through while inactive."""
        adapted = self.rls_calibrator.adapt_probability(probabilities)
        return np.asarray(adapted, dtype=float)

    def update_from_validated_outcomes(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Learn from an explicitly approved joined prediction/outcome frame.
        Uses base probability and Ground_Truth only. Does not set month_12_outcomes.
        """
        return self.rls_calibrator.update_from_validated_frame(df)
