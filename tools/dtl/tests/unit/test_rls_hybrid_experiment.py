"""Focused tests for GRU + RLS residual hybrid (shadow only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.ml.rls.data import MONTH_TEST, MONTH_TRAIN, MONTH_VAL, load_month_temporal_split
from dtl_agent.ml.rls.features import RLS_FEATURE_NAMES, assert_no_forbidden_features
from dtl_agent.ml.rls.hybrid_experiment import OUT_REL
from dtl_agent.ml.rls.hybrid_residual import (
    build_residual_targets,
    correction_diagnostics,
    hybrid_scores,
)
from dtl_agent.ml.rls.jan_gru_shadow import PRODUCTION_CKPT_REL, SHADOW_DIR_REL
from dtl_agent.ml.unified_experiment import FORBIDDEN_INPUT_COLS
from dtl_agent.recommendation import pipeline as rec_pipeline

ROOT = default_project_root()
TEMPORAL_JAN = (
    ROOT
    / "artifacts"
    / "temporal"
    / "2026-01"
    / "ml_dataset"
    / "train"
    / "core_candidate_examples.parquet"
)


def test_residual_target_construction():
    y = np.array([1.0, 2.0, 3.0])
    g = np.array([0.5, 1.5, 2.0])
    r = build_residual_targets(y, g)
    np.testing.assert_allclose(r, np.array([0.5, 0.5, 1.0]))


def test_hybrid_score_is_sum():
    g = np.array([1.0, 2.0])
    r = np.array([0.1, -0.2])
    h = hybrid_scores(g, r)
    np.testing.assert_allclose(h, np.array([1.1, 1.8]))


def test_zero_residual_equals_gru():
    g = np.array([0.3, 0.7, 1.1])
    z = np.zeros_like(g)
    np.testing.assert_allclose(hybrid_scores(g, z), g)


def test_forbidden_features_not_in_rls_vector():
    assert_no_forbidden_features()
    assert not (set(RLS_FEATURE_NAMES) & FORBIDDEN_INPUT_COLS)


def test_production_pipeline_and_paths_isolated():
    src = Path(rec_pipeline.__file__).read_text(encoding="utf-8")
    assert "RLSRegressor" not in src
    assert "hybrid_experiment" not in src
    assert "hybrid_residual" not in src
    assert "ml.rls" not in src
    assert SHADOW_DIR_REL.as_posix() != PRODUCTION_CKPT_REL.as_posix()
    assert "rls_experiment/hybrid" in (ROOT / OUT_REL).as_posix().replace("\\", "/")


def test_hybrid_experiment_source_uses_shadow_gru_only():
    src = Path(ROOT / "src" / "dtl_agent" / "ml" / "rls" / "hybrid_experiment.py").read_text(
        encoding="utf-8"
    )
    assert "pred_temporal_gru" not in src
    assert "load_jan_shadow_scorer" in src
    assert "build_residual_targets" in src
    assert "residual_target = target_score - gru_score" not in src  # doc may differ
    assert "target_score - gru_score" in src


def test_correction_diagnostics_improvement_flags():
    y = np.array([1.0, 2.0, 3.0])
    g = np.array([1.2, 1.0, 3.5])
    r = np.array([-0.2, 1.0, -0.5])  # perfect correction
    d = correction_diagnostics(target_score=y, gru_score=g, residual_pred=r)
    assert d["pct_candidates_improved"] == 1.0
    assert d["hybrid_mae"] < d["gru_mae"] + 1e-12


@pytest.mark.skipif(not TEMPORAL_JAN.is_file(), reason="temporal ml dataset missing")
def test_month_split_leakage_contract():
    data = load_month_temporal_split(ROOT)
    assert data.train["production_month"].eq(MONTH_TRAIN).all()
    assert data.validation["production_month"].eq(MONTH_VAL).all()
    assert data.test["production_month"].eq(MONTH_TEST).all()
    assert data.validation["example_id"].astype(str).tolist() == sorted(
        data.validation["example_id"].astype(str).tolist()
    )


@pytest.mark.skipif(not TEMPORAL_JAN.is_file(), reason="temporal ml dataset missing")
def test_gru_scores_independent_of_hybrid_combination():
    """Hybrid module must not mutate GRU base scores."""
    g = np.array([0.5, 0.8, 1.1])
    g_copy = g.copy()
    r = np.array([0.01, -0.02, 0.03])
    _ = hybrid_scores(g, r)
    np.testing.assert_allclose(g, g_copy)
