import numpy as np
import pandas as pd
from typing import Dict, Any, Union

from ..decision.decision_policy import RETEST_LABEL, DONT_RETEST_LABEL


def calculate_decision_quality(
    y_true: Union[np.ndarray, pd.Series, list],
    recommendations: Union[np.ndarray, pd.Series, list]
) -> Dict[str, Any]:
    """
    Computes standard ATE Retest decision-quality metrics from
    AI Recommendation labels vs actual Ground_Truth.

    TP: Correct Retest
    FP: Unnecessary Retest
    FN: Missed Opportunity
    TN: Correct Skip

    Operates only on explicit RETEST / DON'T RETEST labels.
    The 30% cutoff lives only in the isolated decision-policy module.
    """
    y_t = np.asarray(y_true)
    if y_t.dtype.type is np.str_ or y_t.dtype == object:
        y_t = np.where(np.asarray(y_t, dtype=object) == "RETEST_BENEFICIAL", 1, 0).astype(int)
    else:
        y_t = y_t.astype(int)

    recs = np.asarray(recommendations)
    if recs.dtype.type is np.str_ or recs.dtype == object:
        rec_set = set(pd.unique(recs))
        allowed = {RETEST_LABEL, DONT_RETEST_LABEL}
        if rec_set - allowed:
            raise ValueError(
                f"Recommendations must be {RETEST_LABEL!r} or {DONT_RETEST_LABEL!r}. "
                f"Got unexpected values: {sorted(rec_set - allowed)}"
            )
        y_p = np.where(recs == RETEST_LABEL, 1, 0).astype(int)
    else:
        raise ValueError(
            "Decision-quality KPIs require RETEST / DON'T RETEST labels. "
            "Do not pass raw probabilities here; apply the isolated decision policy first."
        )

    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    fp = int(np.sum((y_p == 1) & (y_t == 0)))
    fn = int(np.sum((y_p == 0) & (y_t == 1)))
    tn = int(np.sum((y_p == 0) & (y_t == 0)))
    total = int(len(y_t))

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_events": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "correct_retests": tp,
        "unnecessary_retests": fp,
        "missed_opportunities": fn,
        "correct_skips": tn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "unnecessary_retests_count": fp,
        "unnecessary_retests_pct": float(fp / total * 100.0) if total > 0 else 0.0,
        "missed_opportunities_count": fn,
        "missed_opportunities_pct": float(fn / total * 100.0) if total > 0 else 0.0,
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        }
    }
