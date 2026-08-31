"""
Optional Recursive Least Squares (RLS) probability calibration.

The primary classifier remains frozen. This layer learns a mapping from
base P(RETEST_BENEFICIAL) to observed post-retest Ground_Truth using
explicitly approved validated outcomes only.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..config.settings import (
    RLS_FORGETTING_FACTOR,
    RLS_INITIAL_P_SCALE,
    RLS_MIN_UPDATES_BEFORE_ACTIVE,
    TARGET_COL,
)

BASE_PROB_COL = "P_BASE_RETEST_BENEFICIAL"
ADAPTED_PROB_COL = "P_ADAPTED_RETEST_BENEFICIAL"
PUBLIC_PROB_COL = "P(RETEST_BENEFICIAL)"

POSITIVE_GROUND_TRUTH = "RETEST_BENEFICIAL"
NEGATIVE_GROUND_TRUTH_LABELS = frozenset({"PERSISTENT_FAILURE", "RETEST_NOT_BENEFICIAL"})

N_FEATURES = 2  # intercept + base probability


def encode_ground_truth(value: Any) -> Optional[int]:
    """Map a Ground_Truth label to 1 / 0. Return None for missing or unsupported values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    label = str(value).strip().upper()
    if not label or label in {"NAN", "NONE", "NULL"}:
        return None
    if label == POSITIVE_GROUND_TRUTH:
        return 1
    if label in NEGATIVE_GROUND_TRUTH_LABELS:
        return 0
    return None


def extract_base_probability_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Prefer the stored base probability; never use an already-adapted public column when base exists."""
    if BASE_PROB_COL in df.columns:
        return pd.to_numeric(df[BASE_PROB_COL], errors="coerce")
    if PUBLIC_PROB_COL in df.columns:
        return pd.to_numeric(df[PUBLIC_PROB_COL], errors="coerce")
    return None


def _normalize_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            as_float = float(text)
            if as_float.is_integer():
                return str(int(as_float))
        except ValueError:
            pass
    return text


class RLSCalibrator:
    """In-memory RLS calibration layer. Identity-initialized; inactive until enough updates."""

    def __init__(
        self,
        forgetting_factor: float = RLS_FORGETTING_FACTOR,
        min_updates_before_active: int = RLS_MIN_UPDATES_BEFORE_ACTIVE,
        initial_p_scale: float = RLS_INITIAL_P_SCALE,
    ):
        self.forgetting_factor = float(forgetting_factor)
        self.min_updates_before_active = int(min_updates_before_active)
        self.initial_p_scale = float(initial_p_scale)
        self.reset()

    def reset(self) -> None:
        """Clear weights, covariance, update count, and learned dataset fingerprints."""
        self.weights = np.array([[0.0], [1.0]], dtype=float)
        self.P = np.eye(N_FEATURES, dtype=float) * self.initial_p_scale
        self.update_count = 0
        self.initialized = True
        self.learned_fingerprints: set[str] = set()

    @property
    def is_active(self) -> bool:
        return self.update_count >= self.min_updates_before_active

    def status(self) -> Dict[str, Any]:
        return {
            "initialized": bool(self.initialized),
            "update_count": int(self.update_count),
            "activation_threshold": int(self.min_updates_before_active),
            "active": bool(self.is_active),
            "forgetting_factor": float(self.forgetting_factor),
            "learned_dataset_count": int(len(self.learned_fingerprints)),
        }

    def diagnostics(self) -> Dict[str, Any]:
        info = self.status()
        info["weights"] = [float(self.weights[0, 0]), float(self.weights[1, 0])]
        return info

    def adapt_probability(
        self, base_probability: Union[float, Sequence[float], np.ndarray, pd.Series]
    ) -> Union[float, np.ndarray]:
        """
        Return adapted probability when active; otherwise pass the base probability through.

        Always clips the returned value to [0, 1].
        """
        scalar = np.isscalar(base_probability) or (
            isinstance(base_probability, np.ndarray) and base_probability.ndim == 0
        )
        base = np.asarray(base_probability, dtype=float).reshape(-1)
        base = np.clip(base, 0.0, 1.0)
        if not self.is_active:
            out = base.copy()
        else:
            design = np.column_stack([np.ones(base.shape[0], dtype=float), base])
            raw = design @ self.weights.ravel()
            out = np.clip(raw, 0.0, 1.0)
        if scalar:
            return float(out[0]) if out.size else 0.0
        return out

    def update_from_validated_frame(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Recursively update RLS from an already-joined prediction/outcome frame.

        Uses base probability + Ground_Truth only. Duplicate dataset fingerprints
        are rejected without a second update.
        """
        result = {
            "learned": 0,
            "skipped": 0,
            "update_count": int(self.update_count),
            "active": bool(self.is_active),
            "already_learned": False,
            "fingerprint": None,
            "activation_threshold": int(self.min_updates_before_active),
        }
        if df is None or len(df) == 0:
            result["skipped"] = 0 if df is None else int(len(df))
            return result

        work = df.copy()
        base_series = extract_base_probability_series(work)
        if base_series is None or TARGET_COL not in work.columns:
            result["skipped"] = int(len(work))
            return result

        valid_rows: List[Dict[str, Any]] = []
        skipped = 0
        for idx in work.index:
            encoded = encode_ground_truth(work.at[idx, TARGET_COL])
            base_p = base_series.at[idx]
            if encoded is None or not np.isfinite(base_p):
                skipped += 1
                continue
            base_clipped = float(np.clip(float(base_p), 0.0, 1.0))
            valid_rows.append(
                {
                    "Device_ID": _normalize_id(work.at[idx, "Device_ID"] if "Device_ID" in work.columns else ""),
                    "Failure_Event": _normalize_id(
                        work.at[idx, "Failure_Event"] if "Failure_Event" in work.columns else ""
                    ),
                    "Ground_Truth": str(work.at[idx, TARGET_COL]).strip().upper(),
                    "base_probability": f"{base_clipped:.4f}",
                    "target": encoded,
                    "base": base_clipped,
                }
            )

        result["skipped"] = skipped
        if not valid_rows:
            return result

        fingerprint = self._fingerprint_rows(valid_rows)
        result["fingerprint"] = fingerprint
        if fingerprint in self.learned_fingerprints:
            result["already_learned"] = True
            return result

        for row in valid_rows:
            self._update_one(row["base"], row["target"])

        self.learned_fingerprints.add(fingerprint)
        result["learned"] = len(valid_rows)
        result["update_count"] = int(self.update_count)
        result["active"] = bool(self.is_active)
        return result

    def _update_one(self, base_probability: float, target: int) -> None:
        x = np.array([[1.0], [float(base_probability)]], dtype=float)
        lam = float(self.forgetting_factor)
        px = self.P @ x
        denom = lam + float(np.asarray(x.T @ px).reshape(-1)[0])
        if not np.isfinite(denom) or denom <= 0.0:
            return
        gain = px / denom
        prediction = float(np.asarray(x.T @ self.weights).reshape(-1)[0])
        error = float(target) - prediction
        self.weights = self.weights + gain * error
        self.P = (self.P - (gain @ x.T) @ self.P) / lam
        self.P = 0.5 * (self.P + self.P.T)
        self.update_count += 1

    @staticmethod
    def _fingerprint_rows(rows: Iterable[Dict[str, Any]]) -> str:
        keys = []
        for row in rows:
            keys.append(
                "|".join(
                    [
                        str(row.get("Device_ID", "")),
                        str(row.get("Failure_Event", "")),
                        str(row.get("Ground_Truth", "")),
                        str(row.get("base_probability", "")),
                    ]
                )
            )
        keys.sort()
        payload = "\n".join(keys).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
