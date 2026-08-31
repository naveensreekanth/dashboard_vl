import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple
from ..config.settings import RANDOM_SEED
from ..data.preprocessing import prepare_xy
from ..decision.decision_policy import DOCX_REFERENCE_THRESHOLD, POLICY_LABEL
from .logistic_model import build_logistic_model
from .xgboost_model import build_xgboost_model
from .gradient_boosting import build_gradient_boosting_model
from .calibration import calibrate_model, evaluate_calibration
from .evaluator import evaluate_probability_predictions, evaluate_at_reporting_cutoff


def _select_best_model(results: Dict[str, Any]) -> Tuple[str, str]:
    """
    Evidence-based selection using Month 6 temporal validation.
    Primary: ROC-AUC (discrimination). Tie-break: lower Brier Score (probability quality).
    """
    ranked = []
    for name, res in results.items():
        m = res["calibrated_metrics"]
        ranked.append((name, m["ROC-AUC"], -m["Brier Score"]))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_name = ranked[0][0]
    best = results[best_name]["calibrated_metrics"]
    reason = (
        f"{best_name} selected on Month 6 temporal holdout: "
        f"ROC-AUC={best['ROC-AUC']:.3f}, PR-AUC={best['PR-AUC']:.3f}, "
        f"Brier={best['Brier Score']:.4f}, Log Loss={best['Log Loss']:.4f}. "
        f"Operational recommendations use {POLICY_LABEL} (threshold={DOCX_REFERENCE_THRESHOLD:.2f}), "
        "not this selection rule."
    )
    return best_name, reason


def train_and_compare_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame
) -> Dict[str, Any]:
    """
    Trains Logistic Regression, XGBoost, and GradientBoosting on df_train (Month 0)
    and evaluates on df_val (Month 6) to produce an unbiased temporal comparison.
    """
    X_train, y_train = prepare_xy(df_train, is_inference=False)
    X_val, y_val = prepare_xy(df_val, is_inference=False)

    candidate_builders = {
        "Logistic Regression": lambda: build_logistic_model(random_state=RANDOM_SEED),
        "XGBoost": lambda: build_xgboost_model(random_state=RANDOM_SEED),
        "Gradient Boosting": lambda: build_gradient_boosting_model(random_state=RANDOM_SEED)
    }

    results = {}
    trained_models = {}

    for name, builder in candidate_builders.items():
        raw_model = builder()
        raw_model.fit(X_train, y_train)

        probs_val_raw = raw_model.predict_proba(X_val)[:, 1]
        raw_eval = evaluate_probability_predictions(y_val, probs_val_raw)
        raw_calib = evaluate_calibration(y_val, probs_val_raw)

        try:
            calibrated_model = calibrate_model(builder(), method="sigmoid", cv=5)
            calibrated_model.fit(X_train, y_train)
            probs_val_cal = calibrated_model.predict_proba(X_val)[:, 1]
            cal_eval = evaluate_probability_predictions(y_val, probs_val_cal)
            cal_eval_reporting = evaluate_at_reporting_cutoff(y_val, probs_val_cal)
            cal_calib = evaluate_calibration(y_val, probs_val_cal)
        except Exception:
            calibrated_model = raw_model
            probs_val_cal = probs_val_raw
            cal_eval = raw_eval
            cal_eval_reporting = evaluate_at_reporting_cutoff(y_val, probs_val_raw)
            cal_calib = raw_calib

        trained_models[name] = {
            "raw_model": raw_model,
            "calibrated_model": calibrated_model
        }

        results[name] = {
            "raw_metrics": raw_eval,
            "calibrated_metrics": cal_eval,
            "calibrated_metrics_reporting_cutoff": cal_eval_reporting,
            "raw_calibration": raw_calib,
            "calibrated_calibration": cal_calib,
            "val_probabilities_raw": probs_val_raw,
            "val_probabilities_cal": probs_val_cal
        }

    comparison_table_rows = []
    for model_name, res in results.items():
        m = res["calibrated_metrics"]
        comparison_table_rows.append({
            "Model": model_name,
            "Accuracy": f"{m['Accuracy']*100:.1f}%",
            "Precision": f"{m['Precision']*100:.1f}%",
            "Recall": f"{m['Recall']*100:.1f}%",
            "Specificity": f"{m['Specificity']*100:.1f}%",
            "F1": f"{m['F1']:.3f}",
            "ROC-AUC": f"{m['ROC-AUC']:.3f}",
            "PR-AUC": f"{m['PR-AUC']:.3f}",
            "Brier Score": f"{m['Brier Score']:.4f}",
            "Log Loss": f"{m['Log Loss']:.4f}"
        })

    df_comparison = pd.DataFrame(comparison_table_rows)
    best_name, selection_reason = _select_best_model(results)

    return {
        "models": trained_models,
        "results": results,
        "comparison_table": df_comparison,
        "best_model_name": best_name,
        "selection_reason": selection_reason,
        "operational_policy_label": POLICY_LABEL,
        "operational_threshold": DOCX_REFERENCE_THRESHOLD,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val
    }


def train_final_deployment_model(
    df_historical_combined: pd.DataFrame,
    model_name: str = "XGBoost"
) -> Any:
    """
    Trains the final calibrated model on all historical data (Month 0 + Month 6)
    for unseen Month 12 inference.
    """
    X_hist, y_hist = prepare_xy(df_historical_combined, is_inference=False)

    if model_name == "Logistic Regression":
        base_pipeline = build_logistic_model(random_state=RANDOM_SEED)
    elif model_name == "Gradient Boosting":
        base_pipeline = build_gradient_boosting_model(random_state=RANDOM_SEED)
    else:
        base_pipeline = build_xgboost_model(random_state=RANDOM_SEED)

    calibrated_pipeline = calibrate_model(base_pipeline, method="sigmoid", cv=5)
    calibrated_pipeline.fit(X_hist, y_hist)
    return calibrated_pipeline
