"""Integration tests for Phase 8 recommendation pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.recommendation import Decision, RecommendationConfig, recommend

ROOT = Path(__file__).resolve().parents[2]


def _artifacts_ready() -> bool:
    return (
        (ROOT / "artifacts" / "ml" / "checkpoints" / "core_gru_best.pt").is_file()
        and (ROOT / "artifacts" / "ml_dataset" / "train" / "core_candidate_examples.parquet").is_file()
        and (ROOT / "artifacts" / "simulation" / "core" / "candidate_results.csv").is_file()
    )


@pytest.mark.integration
def test_linked_lot_core_and_parametric_independent():
    if not _artifacts_ready():
        pytest.skip("Phase 7/6/4 artifacts missing")
    # Known linked validation lot from Phase 7 split
    result = recommend(
        lot_id="DTL_NORM_004",
        die_id="DTL_NORM_004_D048",
        parameters=["ir_drop", "VMIN"],
        config=RecommendationConfig(TOP_N=5),
        project_root=ROOT,
    )
    assert result.core_available is True
    assert result.parametric_available is True
    params = {r.parameter: r for r in result.recommendations}
    assert "ir_drop" in params
    assert "VMIN" in params
    assert params["ir_drop"].model_id in {None, "core_gru"} or params["ir_drop"].decision in {
        Decision.REVIEW_REQUIRED,
        Decision.KEEP_CURRENT,
        Decision.RECOMMEND,
    }
    assert result.audit["joint_enabled"] is False
    assert all(r.evidence_origin == "SIMULATOR_DERIVED" for r in result.recommendations)
    # Reproducibility: same inputs → same decisions
    result2 = recommend(
        lot_id="DTL_NORM_004",
        die_id="DTL_NORM_004_D048",
        parameters=["ir_drop"],
        config=RecommendationConfig(TOP_N=5),
        project_root=ROOT,
    )
    assert result2.recommendations[0].decision == params["ir_drop"].decision
    assert result2.recommendations[0].recommended_limit == params["ir_drop"].recommended_limit


@pytest.mark.integration
def test_parametric_only_no_core_fabrication():
    if not _artifacts_ready():
        pytest.skip("Phase 7/6/4 artifacts missing")
    # Parametric-only lot from constants / split
    result = recommend(
        lot_id="DTL_PARAM_VMARGIN_003",
        die_id="DTL_PARAM_VMARGIN_003_DIE_041",
        parameters=["ir_drop", "VMIN"],
        config=RecommendationConfig(),
        project_root=ROOT,
    )
    assert result.core_available is False
    ir = next(r for r in result.recommendations if r.parameter == "ir_drop")
    assert ir.decision == Decision.REVIEW_REQUIRED
    assert "fabricat" in ir.explanation.get("policy_reason", "").lower() or ir.decision == Decision.REVIEW_REQUIRED
    vmin = next(r for r in result.recommendations if r.parameter == "VMIN")
    assert vmin.decision in {
        Decision.RECOMMEND,
        Decision.KEEP_CURRENT,
        Decision.REVIEW_REQUIRED,
    }


@pytest.mark.integration
def test_forbidden_path_rejects():
    result = recommend(
        lot_id="DTL_NORM_004",
        die_id="DTL_NORM_004_D048",
        parameters=["ir_drop"],
        project_root=ROOT,
        extra_paths=["data/core/evaluation/latent.csv"],
    )
    assert result.recommendations[0].decision == Decision.REJECT


@pytest.mark.integration
def test_unsupported_parameter_rejects():
    if not (ROOT / "artifacts" / "simulation" / "core" / "candidate_grid.csv").is_file():
        pytest.skip("simulation artifacts missing")
    result = recommend(
        lot_id="DTL_NORM_004",
        die_id="DTL_NORM_004_D048",
        parameters=["test_time"],
        project_root=ROOT,
    )
    assert result.recommendations[0].decision == Decision.REJECT
