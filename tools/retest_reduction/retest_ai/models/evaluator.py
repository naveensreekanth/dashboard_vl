import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss
)

from ..config.settings import EVAL_REPORTING_CUTOFF
from ..decision.decision_policy import DOCX_REFERENCE_THRESHOLD, POLICY_LABEL


def evaluate_probability_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = None,
    threshold_label: str = None
) -> Dict[str, Any]:
    """
    Evaluates probability predictions against binary ground truth.

    threshold-free metrics always use the raw probabilities.
    Classification metrics require an explicit cutoff:
      - operational DOCX-reference policy (0.30) for recommendation scoring
      - EVAL_REPORTING_CUTOFF (0.5) only for labeled model-comparison reporting
    """
    if threshold is None:
        threshold = DOCX_REFERENCE_THRESHOLD
        threshold_label = threshold_label or POLICY_LABEL

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob, labels=[0, 1]))

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Brier Score": brier,
        "Log Loss": loss,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_events": len(y_true),
        "classification_threshold": float(threshold),
        "classification_threshold_label": threshold_label or f"cutoff={threshold}"
    }


def evaluate_at_reporting_cutoff(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Classification metrics at 0.5 for model comparison only. Not the operational policy."""
    return evaluate_probability_predictions(
        y_true,
        y_prob,
        threshold=EVAL_REPORTING_CUTOFF,
        threshold_label="Evaluation/reporting cutoff (0.5) — not the operational decision policy"
    )
