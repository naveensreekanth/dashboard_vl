"""
Post-outcome validation.

Compares AI recommendation against actual Ground_Truth after the
retest/outcome is known. Never used as input to pre-retest prediction.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..kpis.decision_quality import calculate_decision_quality


def validate_recommendations_against_outcomes(
    y_true: Union[np.ndarray, pd.Series, list],
    recommendations: Union[np.ndarray, pd.Series, list],
    events: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Score RETEST / DON'T RETEST recommendations against actual outcomes.

    TP Correct Retest | FP Unnecessary Retest
    FN Missed Opportunity | TN Correct Skip
    """
    quality = calculate_decision_quality(y_true, recommendations)
    result = dict(quality)
    result["has_outcomes"] = True
    result["terminology"] = {
        "TP": "Correct Retest",
        "FP": "Unnecessary Retest",
        "FN": "Missed Opportunity",
        "TN": "Correct Skip",
    }

    if events is not None:
        df = events.copy()
        y_t = np.asarray(y_true)
        if y_t.dtype.type is np.str_ or y_t.dtype == object:
            y_bin = np.where(np.asarray(y_t, dtype=object) == "RETEST_BENEFICIAL", 1, 0)
        else:
            y_bin = y_t.astype(int)
        recs = np.asarray(recommendations)
        rec_bin = np.where(recs == "RETEST", 1, 0)
        df["_actual_beneficial"] = y_bin == 1
        df["_ai_retest"] = rec_bin == 1
        drop_cols = ["_actual_beneficial", "_ai_retest"]
        result["correct_retest_events"] = df[df["_ai_retest"] & df["_actual_beneficial"]].drop(
            columns=drop_cols, errors="ignore"
        )
        result["unnecessary_retest_events"] = df[df["_ai_retest"] & ~df["_actual_beneficial"]].drop(
            columns=drop_cols, errors="ignore"
        )
        result["missed_opportunity_events"] = df[~df["_ai_retest"] & df["_actual_beneficial"]].drop(
            columns=drop_cols, errors="ignore"
        )
        result["correct_skip_events"] = df[~df["_ai_retest"] & ~df["_actual_beneficial"]].drop(
            columns=drop_cols, errors="ignore"
        )

    return result
