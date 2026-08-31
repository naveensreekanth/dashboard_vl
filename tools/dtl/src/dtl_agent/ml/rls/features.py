"""Compact feature engineering for experimental RLS candidate scoring.

Uses the same Core candidate fields as CoreGRU (``CORE_CAND_NUM``) plus
deterministic sequence aggregates from the parameter's measurement channel.
Never includes forbidden simulation outcomes as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dtl_agent.features.core_engine import SEQUENCE_FEATURE_ORDER
from dtl_agent.ml.datasets.phase7_datasets import CORE_CAND_NUM
from dtl_agent.ml.unified_experiment import FORBIDDEN_INPUT_COLS

# Fixed Core parameters scored by CoreGRU temporal.
CORE_PARAMETERS = ("ir_drop", "thermal")

# Exact feature names in vector order (bias first).
RLS_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    # Candidate geometry (same conceptual fields as CORE_CAND_NUM)
    "candidate_limit",
    "current_limit",
    "candidate_delta",
    "candidate_delta_percent",
    "limit_minus_seq_mean",
    # Sequence aggregates for the scored parameter channel
    "seq_mean",
    "seq_std",
    "seq_min",
    "seq_max",
    "seq_last",
    "seq_delta",
    "seq_trend",
    "seq_p10",
    "seq_p90",
    # One-hots
    "param_ir_drop",
    "param_thermal",
    "dir_lower",
    "dir_upper",
    "tight_current",
    "tight_looser",
    "tight_tighter",
    "cat_center",
    "cat_edge",
    "cat_normal",
    "cat_scratch",
)


@dataclass(frozen=True)
class SequenceAggregates:
    mean: float
    std: float
    min: float
    max: float
    last: float
    delta: float
    trend: float
    p10: float
    p90: float


def _channel_index(parameter: str) -> int:
    if parameter not in SEQUENCE_FEATURE_ORDER:
        raise KeyError(f"parameter {parameter!r} not in SEQUENCE_FEATURE_ORDER")
    return list(SEQUENCE_FEATURE_ORDER).index(parameter)


def sequence_aggregates(seq: np.ndarray, parameter: str) -> SequenceAggregates:
    """Compact temporal summary of the parameter channel (length-200 sequence)."""
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(SEQUENCE_FEATURE_ORDER):
        raise ValueError(f"sequence shape {arr.shape} invalid")
    ch = arr[:, _channel_index(parameter)]
    n = ch.size
    head = max(1, n // 10)
    first = float(np.mean(ch[:head]))
    last_win = float(np.mean(ch[-head:]))
    return SequenceAggregates(
        mean=float(np.mean(ch)),
        std=float(np.std(ch)),
        min=float(np.min(ch)),
        max=float(np.max(ch)),
        last=float(ch[-1]),
        delta=float(ch[-1] - ch[0]),
        trend=float(last_win - first),
        p10=float(np.percentile(ch, 10)),
        p90=float(np.percentile(ch, 90)),
    )


def build_feature_vector(
    *,
    parameter: str,
    direction: str,
    tighten_or_loosen: str,
    lot_category: str,
    candidate_limit: float,
    current_limit: float,
    candidate_delta: float,
    candidate_delta_percent: float,
    agg: SequenceAggregates,
) -> np.ndarray:
    """Return float64 feature vector aligned with ``RLS_FEATURE_NAMES``."""
    param = str(parameter)
    direction = str(direction)
    tight = str(tighten_or_loosen)
    cat = str(lot_category)

    return np.array(
        [
            1.0,  # bias
            float(candidate_limit),
            float(current_limit),
            float(candidate_delta),
            float(candidate_delta_percent),
            float(candidate_limit) - agg.mean,
            agg.mean,
            agg.std,
            agg.min,
            agg.max,
            agg.last,
            agg.delta,
            agg.trend,
            agg.p10,
            agg.p90,
            1.0 if param == "ir_drop" else 0.0,
            1.0 if param == "thermal" else 0.0,
            1.0 if direction == "LOWER" else 0.0,
            1.0 if direction == "UPPER" else 0.0,
            1.0 if tight == "CURRENT" else 0.0,
            1.0 if tight == "LOOSER" else 0.0,
            1.0 if tight == "TIGHTER" else 0.0,
            1.0 if cat == "CENTER" else 0.0,
            1.0 if cat == "EDGE" else 0.0,
            1.0 if cat == "NORMAL" else 0.0,
            1.0 if cat == "SCRATCH" else 0.0,
        ],
        dtype=np.float64,
    )


def _precompute_aggregates(
    examples: pd.DataFrame,
    seq_store: dict[str, np.ndarray],
) -> dict[tuple[str, str], SequenceAggregates]:
    pairs = examples[["sequence_id", "parameter"]].drop_duplicates()
    out: dict[tuple[str, str], SequenceAggregates] = {}
    for sid, param in pairs.itertuples(index=False, name=None):
        sid_s, param_s = str(sid), str(param)
        if sid_s not in seq_store:
            raise KeyError(f"missing sequence_id {sid_s}")
        out[(sid_s, param_s)] = sequence_aggregates(seq_store[sid_s], param_s)
    return out


def build_feature_matrix(
    examples: pd.DataFrame,
    seq_store: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X, y=target_score, and example_id list.

    ``seq_store`` maps ``sequence_id`` → (200, 5) float arrays.
    """
    required = [
        "sequence_id",
        "parameter",
        "direction",
        "tighten_or_loosen",
        "lot_category",
        "target_score",
        "example_id",
        *CORE_CAND_NUM,
    ]
    missing = [c for c in required if c not in examples.columns]
    if missing:
        raise KeyError(f"examples missing columns: {missing}")

    n = len(examples)
    X = np.zeros((n, len(RLS_FEATURE_NAMES)), dtype=np.float64)
    y = examples["target_score"].to_numpy(dtype=np.float64)
    ids = examples["example_id"].astype(str).tolist()

    agg_cache = _precompute_aggregates(examples, seq_store)

    params = examples["parameter"].astype(str).to_numpy()
    dirs = examples["direction"].astype(str).to_numpy()
    tights = examples["tighten_or_loosen"].astype(str).to_numpy()
    cats = examples["lot_category"].astype(str).to_numpy()
    sids = examples["sequence_id"].astype(str).to_numpy()
    cand_lim = examples["candidate_limit"].to_numpy(dtype=np.float64)
    cur_lim = examples["current_limit"].to_numpy(dtype=np.float64)
    cand_d = examples["candidate_delta"].to_numpy(dtype=np.float64)
    cand_dp = examples["candidate_delta_percent"].to_numpy(dtype=np.float64)

    for i in range(n):
        agg = agg_cache[(sids[i], params[i])]
        X[i] = build_feature_vector(
            parameter=params[i],
            direction=dirs[i],
            tighten_or_loosen=tights[i],
            lot_category=cats[i],
            candidate_limit=float(cand_lim[i]),
            current_limit=float(cur_lim[i]),
            candidate_delta=float(cand_d[i]),
            candidate_delta_percent=float(cand_dp[i]),
            agg=agg,
        )
    return X, y, ids


def assert_no_forbidden_features(feature_names: tuple[str, ...] = RLS_FEATURE_NAMES) -> None:
    bad = set(feature_names) & FORBIDDEN_INPUT_COLS
    if bad:
        raise RuntimeError(f"RLS features intersect forbidden cols: {sorted(bad)}")
