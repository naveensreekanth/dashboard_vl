"""Phase 12.5D — UnifiedParameterGRURanker experiment (offline only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.loader import load_temporal_month
from dtl_agent.data.temporal.paths import temporal_data_root
from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml.models.parametric_encoder import ParametricMLPRanker
from dtl_agent.ml.models.unified_gru_ranker import (
    CORE_SCORE_PARAMETERS,
    PARAMETRIC_CONDITION_ORDER,
    PARAMETRIC_CONTEXT_DIM,
    PARAMETRIC_SCORE_PARAMETERS,
    UNIFIED_PARAMETER_VOCAB,
    UnifiedParameterGRURanker,
)
from dtl_agent.ml.unified_experiment import (
    FORBIDDEN_INPUT_COLS,
    UnifiedCandidateDataset,
    build_parametric_context_table,
    empty_parametric_context_row,
)
from dtl_agent.recommendation.pipeline import recommend

ROOT = default_project_root()
TEMPORAL_AVAILABLE = (temporal_data_root(ROOT) / "2026-01" / "actual_die" / "measurements.csv").is_file()

pytestmark = pytest.mark.skipif(
    not TEMPORAL_AVAILABLE,
    reason="data/3 months data package not present",
)


def test_supported_parameters_exclude_setup_hold_test_time():
    assert "setup_slack" not in UNIFIED_PARAMETER_VOCAB
    assert "hold_slack" not in UNIFIED_PARAMETER_VOCAB
    assert "test_time" not in UNIFIED_PARAMETER_VOCAB
    assert CORE_SCORE_PARAMETERS == frozenset({"ir_drop", "thermal"})
    assert PARAMETRIC_SCORE_PARAMETERS == frozenset(
        {
            "VMIN",
            "VMAX",
            "IDDQ",
            "SUPPLY_CURRENT",
            "CONTACT_RESISTANCE",
            "INTERCONNECT_RESISTANCE",
            "ON_RESISTANCE",
        }
    )
    assert set(UNIFIED_PARAMETER_VOCAB) == CORE_SCORE_PARAMETERS | PARAMETRIC_SCORE_PARAMETERS


def test_unified_model_forward_shape_and_no_nan():
    m = UnifiedParameterGRURanker()
    b = 4
    y = m(
        sequence=torch.randn(b, 200, 5),
        cand_num=torch.randn(b, 4),
        parametric_context=torch.randn(b, PARAMETRIC_CONTEXT_DIM),
        has_parametric_context=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        parameter_idx=torch.tensor([0, 2, 0, 5]),
        direction_idx=torch.tensor([0, 1, 0, 0]),
        tight_idx=torch.tensor([0, 1, 2, 0]),
    )
    assert y.shape == (b,)
    assert torch.isfinite(y).all()


def test_core_and_mlp_classes_still_importable_unchanged():
    assert CoreGRURanker is not None
    assert ParametricMLPRanker is not None
    # Instantiable with same ctor signature used in production inference
    CoreGRURanker(n_parameter=2, n_direction=1, n_tight=3)
    ParametricMLPRanker(norm_num_dim=8, n_parameter=7, n_direction=2, n_tight=3)


def test_parametric_context_not_pattern_tiled():
    month = load_temporal_month("2026-01", project_root=ROOT)
    ctx = build_parametric_context_table(month)
    assert len(ctx) == 1000 * 7  # dies × parametric params
    assert PARAMETRIC_CONTEXT_DIM == 8
    assert list(PARAMETRIC_CONDITION_ORDER) == [
        "COND_RT_NOM",
        "COND_HOT_NOM",
        "COND_RT_LOWV",
        "COND_HOT_HIGHV",
    ]
    # Exactly 4 values + 4 masks — not 200 pattern slots
    for i in range(4):
        assert f"ctx_val_{i}" in ctx.columns
        assert f"ctx_mask_{i}" in ctx.columns
    assert (ctx[[f"ctx_mask_{i}" for i in range(4)]] == 1.0).all().all()
    empty = empty_parametric_context_row()
    assert empty["has_parametric_context"] == 0.0
    assert all(empty[f"ctx_mask_{i}"] == 0.0 for i in range(4))


def test_dataset_masks_missing_context_and_blocks_leakage_cols():
    assert "objective_score" in FORBIDDEN_INPUT_COLS
    assert "simulated_yield" in FORBIDDEN_INPUT_COLS
    # Feature path must not include forbidden names
    feature_names = {
        "sequence",
        "cand_num",
        "parametric_context",
        "has_parametric_context",
        "parameter_idx",
        "direction_idx",
        "tight_idx",
    }
    assert feature_names.isdisjoint(FORBIDDEN_INPUT_COLS)


def test_unknown_parameter_raises():
    from dtl_agent.ml.datasets.phase7_datasets import CoreSequenceStore

    seq = pd.DataFrame(
        {
            "sequence_id": ["2026-01::L::D"] * 200,
            "pattern_id": list(range(200)),
            "ir_drop": np.zeros(200),
            "thermal": np.zeros(200),
            "setup_slack": np.zeros(200),
            "hold_slack": np.zeros(200),
            "test_time": np.zeros(200),
        }
    )
    store = CoreSequenceStore(seq)
    rows = pd.DataFrame(
        [
            {
                "sequence_id": "2026-01::L::D",
                "parameter": "NOT_A_PARAM",
                "direction": "UPPER",
                "tighten_or_loosen": "CURRENT",
                "candidate_limit": 1.0,
                "current_limit": 1.0,
                "candidate_delta": 0.0,
                "candidate_delta_percent": 0.0,
                "norm_candidate_limit": 0.0,
                "norm_current_limit": 0.0,
                "norm_candidate_delta": 0.0,
                "norm_candidate_delta_percent": 0.0,
                "ctx_val_0": 0,
                "ctx_val_1": 0,
                "ctx_val_2": 0,
                "ctx_val_3": 0,
                "norm_ctx_val_0": 0,
                "norm_ctx_val_1": 0,
                "norm_ctx_val_2": 0,
                "norm_ctx_val_3": 0,
                "ctx_mask_0": 0,
                "ctx_mask_1": 0,
                "ctx_mask_2": 0,
                "ctx_mask_3": 0,
                "has_parametric_context": 0.0,
                "target_score": 0.0,
                "example_id": "x",
            }
        ]
    )
    ds = UnifiedCandidateDataset(rows, store)
    with pytest.raises(KeyError):
        _ = ds[0]


def test_recommend_not_wired_to_unified_model():
    src = Path(recommend.__code__.co_filename).read_text(encoding="utf-8")
    assert "UnifiedParameterGRU" not in src
    assert "unified_parameter_gru" not in src
