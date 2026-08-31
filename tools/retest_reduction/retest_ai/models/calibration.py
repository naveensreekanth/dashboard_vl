import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

def calibrate_model(
    base_estimator: Any,
    method: str = "sigmoid",
    cv: Any = 5
) -> CalibratedClassifierCV:
    """
    Wraps a base model/pipeline in CalibratedClassifierCV.
    cv can be an integer (e.g. 5) or 'prefit'.
    """
    return CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=cv
    )

def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5
) -> Dict[str, Any]:
    """
    Evaluates probability calibration using Brier score, Log Loss, and reliability diagram buckets.
    """
    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob, labels=[0, 1]))
    
    # Reliability curve points
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    
    # 5 Fixed Buckets matching DOCX report: [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    
    df_buckets = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df_buckets["bucket"] = pd.cut(df_buckets["y_prob"], bins=bins, labels=bin_labels, right=False)
    
    bucket_summary = []
    total_events = len(y_true)
    ece = 0.0  # Expected Calibration Error
    
    for label in bin_labels:
        subset = df_buckets[df_buckets["bucket"] == label]
        cnt = len(subset)
        if cnt > 0:
            mean_pred = float(subset["y_prob"].mean())
            obs_rate = float(subset["y_true"].mean())
            ece += (cnt / total_events) * abs(mean_pred - obs_rate)
        else:
            mean_pred = 0.0
            obs_rate = 0.0
            
        bucket_summary.append({
            "bucket": label,
            "event_count": cnt,
            "mean_predicted_prob": mean_pred,
            "observed_benefit_rate": obs_rate,
            "diff": mean_pred - obs_rate
        })
        
    return {
        "brier_score": brier,
        "log_loss": loss,
        "expected_calibration_error": float(ece),
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        },
        "bucket_table": bucket_summary
    }
