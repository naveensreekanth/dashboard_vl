"""
Isolated recommendation layer.

The ML model outputs P(RETEST_BENEFICIAL) only.
This module converts that probability into RETEST / DON'T RETEST
using the 30% logic referenced in RETEST~2.docx.

Label: Reference / DOCX decision policy — subject to validation

This is NOT a scientifically established or permanently approved
production threshold. It can be replaced later without retraining
the ML model.
"""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Source of truth for the current prototype recommendation rule.
# Change this module only when a different production policy is approved.
DOCX_REFERENCE_THRESHOLD = 0.30
POLICY_LABEL = "Reference / DOCX decision policy — subject to validation"
RETEST_LABEL = "RETEST"
DONT_RETEST_LABEL = "DON'T RETEST"


def apply_decision_policy(
    probability: float,
    threshold: float = DOCX_REFERENCE_THRESHOLD,
    policy_label: str = POLICY_LABEL,
) -> Dict[str, Any]:
    """Convert a single probability into a recommendation. Does not alter the probability."""
    prob = float(np.clip(probability, 0.0, 1.0))
    cutoff = float(threshold)
    recommendation = RETEST_LABEL if prob >= cutoff else DONT_RETEST_LABEL
    return {
        "probability": prob,
        "probability_percent": round(prob * 100.0, 2),
        "recommendation": recommendation,
        "policy_threshold": cutoff,
        "policy_threshold_percent": round(cutoff * 100.0, 1),
        "policy_label": policy_label,
        "is_retest": recommendation == RETEST_LABEL,
    }


def apply_batch_decision_policy(
    probabilities: Union[np.ndarray, List[float], pd.Series],
    threshold: float = DOCX_REFERENCE_THRESHOLD,
    policy_label: str = POLICY_LABEL,
) -> pd.DataFrame:
    """Vectorized recommendation for a batch of probabilities. Does not alter probabilities."""
    probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    recs = np.where(probs >= float(threshold), RETEST_LABEL, DONT_RETEST_LABEL)
    return pd.DataFrame(
        {
            "P(RETEST_BENEFICIAL)": probs,
            "AI_Recommendation": recs,
            "Policy_Threshold": float(threshold),
            "Policy_Label": policy_label,
        }
    )
