"""Phase 12.7 — GRU audit utilities (offline; no production changes)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.paths import temporal_artifact_root, temporal_data_root
from dtl_agent.ml.gru_audit import _score_stats
from dtl_agent.ml.unified_experiment import FORBIDDEN_INPUT_COLS as FORBIDDEN
from dtl_agent.recommendation.pipeline import recommend

ROOT = default_project_root()
TEMPORAL_AVAILABLE = (temporal_data_root(ROOT) / "2026-01" / "actual_die" / "measurements.csv").is_file()
AUDIT = temporal_artifact_root(ROOT) / "shared" / "gru_audit"

pytestmark = pytest.mark.skipif(not TEMPORAL_AVAILABLE, reason="temporal package missing")


def test_score_stats_detects_constant_and_nan():
    import numpy as np

    ok = _score_stats(np.array([0.1, 0.5, 0.9]))
    assert ok["valid"] is True
    assert ok["near_constant"] is False
    bad = _score_stats(np.array([1.0, 1.0, 1.0]))
    assert bad["near_constant"] is True
    assert bad["valid"] is False
    nan = _score_stats(np.array([1.0, np.nan]))
    assert nan["n_nan"] == 1
    assert nan["valid"] is False


def test_forbidden_features_unchanged_for_leakage_contract():
    assert "simulated_yield" in FORBIDDEN
    assert "objective_score" in FORBIDDEN


def test_recommend_not_modified_by_audit():
    src = Path(recommend.__code__.co_filename).read_text(encoding="utf-8")
    assert "gru_audit" not in src
    assert "UnifiedParameterGRU" not in src


@pytest.mark.skipif(not (AUDIT / "audit_summary.json").is_file(), reason="audit not run yet")
def test_audit_artifacts_exist_and_cover_models():
    required = [
        "score_summary.csv",
        "candidate_sensitivity.csv",
        "temporal_response.csv",
        "calibration_summary.csv",
        "residual_summary.csv",
        "ranking_stability.csv",
        "audit_summary.json",
    ]
    for name in required:
        assert (AUDIT / name).is_file(), name
    sc = pd.read_csv(AUDIT / "score_summary.csv")
    assert set(sc["model"].unique()) >= {
        "core_gru_temporal_v1",
        "unified_parameter_gru_v1",
    }
    assert sc["n_nan"].sum() == 0
    assert sc["valid"].all()
    sens = pd.read_csv(AUDIT / "candidate_sensitivity.csv")
    assert sens["sensitive"].all()
