"""Tests for optimized sensitivity experiment + fast DTL reuse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.ml.rls.eval_metrics import decide_all, ranking_metrics
from dtl_agent.ml.rls.fast_dtl import build_shared_group_pack, decide_all_fast, ranking_metrics_fast
from dtl_agent.ml.rls.features import assert_no_forbidden_features
from dtl_agent.ml.rls.hybrid_residual import hybrid_scores, scaled_hybrid_scores
from dtl_agent.ml.rls.jan_gru_shadow import PRODUCTION_CKPT_REL, SHADOW_DIR_REL
from dtl_agent.ml.rls.sensitivity_experiment import ALPHAS, OUT_REL
from dtl_agent.recommendation import pipeline as rec_pipeline

ROOT = default_project_root()


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "production_month": ["2026-03"] * 4,
            "lot_id": ["L1"] * 4,
            "die_id": ["D1"] * 4,
            "parameter": ["ir_drop"] * 4,
            "candidate_limit": [40.0, 50.0, 55.0, 60.0],
            "current_limit": [55.0] * 4,
            "tighten_or_loosen": ["TIGHTER", "TIGHTER", "CURRENT", "LOOSER"],
            "simulated_yield": [0.9, 1.0, 1.0, 0.8],
            "target_score": [0.7, 0.95, 0.9, 0.4],
            "example_id": ["a", "b", "c", "d"],
        }
    )


def test_alpha_zero_exact_gru_equivalence():
    g = np.array([0.2, 0.9, 0.5, 0.1])
    r = np.array([0.3, -0.4, 0.1, 0.2])
    h0 = scaled_hybrid_scores(g, r, alpha=0.0)
    np.testing.assert_allclose(h0, g)
    df = _toy_frame()
    df["gru_score"] = g
    df["hybrid0"] = h0
    d_g = decide_all(df, score_col="gru_score")
    d_h = decide_all(df, score_col="hybrid0")
    assert float(d_g.iloc[0]["recommended_limit"]) == float(d_h.iloc[0]["recommended_limit"])


def test_alpha_one_equals_full_hybrid_formula():
    g = np.array([0.2, 0.9, 0.5])
    r = np.array([0.1, -0.2, 0.05])
    np.testing.assert_allclose(scaled_hybrid_scores(g, r, alpha=1.0), hybrid_scores(g, r))


def test_fast_dtl_matches_decide_all_semantics():
    df = _toy_frame()
    scores = np.array([0.2, 0.95, 0.5, 0.1])
    df["ml_score"] = scores
    slow = decide_all(df, score_col="ml_score")
    pack = build_shared_group_pack(df)
    fast = decide_all_fast(pack, scores)
    assert abs(float(slow.iloc[0]["recommended_limit"]) - float(fast.iloc[0]["recommended_limit"])) < 1e-12
    assert str(slow.iloc[0]["decision"]) == str(fast.iloc[0]["decision"])


def test_fast_ranking_close_to_reference():
    df = _toy_frame()
    # Two groups
    df2 = pd.concat([df, df.assign(die_id="D2", example_id=["e", "f", "g", "h"])], ignore_index=True)
    scores = np.array([0.2, 0.95, 0.5, 0.1, 0.3, 0.8, 0.4, 0.05])
    df2["ml_score"] = scores
    ref = ranking_metrics(df2, score_col="ml_score")
    pack = build_shared_group_pack(df2)
    fast = ranking_metrics_fast(pack, scores)
    assert abs(ref["top1_candidate_agreement"] - fast["top1_candidate_agreement"]) < 1e-12
    assert abs(ref["topk_candidate_overlap"] - fast["topk_candidate_overlap"]) < 1e-12


def test_alphas_unchanged():
    assert tuple(ALPHAS) == (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)


def test_sensitivity_source_reuses_shared_pack_and_no_prod_wiring():
    src = Path(ROOT / "src/dtl_agent/ml/rls/sensitivity_experiment.py").read_text(encoding="utf-8")
    assert "build_shared_group_pack" in src
    assert "evaluate_scores" in src
    assert "pred_temporal_gru" not in src
    assert "for alpha in alphas" in src
    pipe = Path(rec_pipeline.__file__).read_text(encoding="utf-8")
    assert "sensitivity_experiment" not in pipe
    assert "fast_dtl" not in pipe
    assert SHADOW_DIR_REL.as_posix() != PRODUCTION_CKPT_REL.as_posix()
    assert "hybrid_sensitivity" in OUT_REL.as_posix()
    assert_no_forbidden_features()


def test_no_month_reload_pattern_in_alpha_loop():
    src = Path(ROOT / "src/dtl_agent/ml/rls/sensitivity_experiment.py").read_text(encoding="utf-8")
    # Alpha loop section should not call load_month_temporal_split or score_examples_with_gru
    # Split source around "Part B"
    part_b = src.split("Part B", 1)[-1]
    assert "load_month_temporal_split" not in part_b
    assert "score_examples_with_gru" not in part_b
    assert "build_feature_matrix" not in part_b
