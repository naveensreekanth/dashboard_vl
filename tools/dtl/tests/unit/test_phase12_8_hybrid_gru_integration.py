"""Phase 12.8 — Hybrid GRU integration into month-aware recommend()."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.loader import load_temporal_month
from dtl_agent.data.temporal.paths import TemporalPathError, validate_production_month
from dtl_agent.ml.datasets.phase7_datasets import CORE_CAND_NUM
from dtl_agent.ml.unified_experiment import FORBIDDEN_INPUT_COLS
from dtl_agent.recommendation import recommend
from dtl_agent.recommendation.routing import (
    HybridModelId,
    ROUTING_TABLE_DOC,
    model_for_parameter,
)
from dtl_agent.recommendation.temporal_config import (
    assert_month_simulation_isolated,
    temporal_recommendation_config,
)
from dtl_agent.recommendation.temporal_inference import TemporalHybridBundle

ROOT = default_project_root()
LOT = "DTL_NORM_001"
DIE = "DTL_NORM_001_D001"
SCORABLE = [
    "ir_drop",
    "thermal",
    "VMIN",
    "VMAX",
    "IDDQ",
    "SUPPLY_CURRENT",
    "CONTACT_RESISTANCE",
    "INTERCONNECT_RESISTANCE",
    "ON_RESISTANCE",
]
UNSUPPORTED = ["setup_slack", "hold_slack", "test_time"]


def test_routing_table_core_unified_unsupported():
    assert model_for_parameter("ir_drop", temporal=True) == HybridModelId.CORE_TEMPORAL
    assert model_for_parameter("thermal", temporal=True) == HybridModelId.CORE_TEMPORAL
    for p in (
        "VMIN",
        "VMAX",
        "IDDQ",
        "SUPPLY_CURRENT",
        "CONTACT_RESISTANCE",
        "INTERCONNECT_RESISTANCE",
        "ON_RESISTANCE",
    ):
        assert model_for_parameter(p, temporal=True) == HybridModelId.UNIFIED
    for p in UNSUPPORTED:
        assert model_for_parameter(p, temporal=True) == HybridModelId.UNSUPPORTED
    assert model_for_parameter("ir_drop", temporal=False) == HybridModelId.LEGACY_CORE
    assert model_for_parameter("VMIN", temporal=False) == HybridModelId.LEGACY_PARAMETRIC
    assert "ir_drop" in ROUTING_TABLE_DOC
    assert "ON_RESISTANCE" in ROUTING_TABLE_DOC


def test_sequence_identity_month_isolation():
    a = make_sequence_id(LOT, DIE, "2026-01")
    b = make_sequence_id(LOT, DIE, "2026-02")
    c = make_sequence_id(LOT, DIE, "2026-03")
    assert a == "2026-01::DTL_NORM_001::DTL_NORM_001_D001"
    assert b == "2026-02::DTL_NORM_001::DTL_NORM_001_D001"
    assert c == "2026-03::DTL_NORM_001::DTL_NORM_001_D001"
    assert len({a, b, c}) == 3
    assert make_sequence_id(LOT, DIE, None) == "DTL_NORM_001::DTL_NORM_001_D001"


def test_month_isolation_loader_and_invalid_months():
    jan = load_temporal_month("2026-01", project_root=ROOT)
    assert set(jan.actual_die["production_month"].astype(str).unique()) == {"2026-01"}
    assert set(jan.parametric["production_month"].astype(str).unique()) == {"2026-01"}
    with pytest.raises(TemporalPathError, match="Invalid production_month"):
        validate_production_month("2025-12")
    with pytest.raises(TemporalPathError):
        validate_production_month("2026-04")
    with pytest.raises(TemporalPathError):
        validate_production_month("random")
    with pytest.raises(ValueError, match="Invalid production_month"):
        recommend(
            lot_id=LOT,
            die_id=DIE,
            parameters=["ir_drop"],
            production_month="2025-12",
            project_root=ROOT,
        )


def test_simulation_isolation_config_paths():
    for month in ("2026-01", "2026-02", "2026-03"):
        assert_month_simulation_isolated(month, ROOT)
        cfg = temporal_recommendation_config(month)
        assert f"artifacts/temporal/{month}/simulation" in cfg.core_candidate_results_path
        assert "artifacts/simulation/" not in cfg.core_candidate_results_path.replace("\\", "/")
        assert cfg.core_candidate_results_path.endswith(
            f"artifacts/temporal/{month}/simulation/core/candidate_results.csv"
        ) or cfg.core_candidate_results_path.replace("\\", "/").endswith(
            f"artifacts/temporal/{month}/simulation/core/candidate_results.csv"
        )


def test_checkpoint_paths_for_hybrid_bundle():
    bundle = TemporalHybridBundle(ROOT)
    assert bundle.ensure_loaded()
    assert bundle.core_checkpoint_id is not None
    assert bundle.uni_checkpoint_id is not None
    assert "core_gru_temporal_v1.pt" in str(bundle.core_checkpoint_id).replace("\\", "/")
    assert "unified_parameter_gru_v1.pt" in str(bundle.uni_checkpoint_id).replace("\\", "/")
    assert "core_gru_best.pt" not in str(bundle.core_checkpoint_id)


def test_leakage_guard_feature_columns():
    feature_cols = list(CORE_CAND_NUM) + ["parameter", "direction", "tighten_or_loosen"]
    assert not (set(feature_cols) & FORBIDDEN_INPUT_COLS)
    for forbidden in (
        "simulated_yield",
        "violation_rate",
        "borderline_rate",
        "objective_score",
        "safety_result",
    ):
        assert forbidden in FORBIDDEN_INPUT_COLS or forbidden not in feature_cols


def test_temporal_recommend_model_routing_and_month_stamp():
    hybrid = TemporalHybridBundle(ROOT)
    assert hybrid.ensure_loaded()
    ir = recommend(
        lot_id=LOT,
        die_id=DIE,
        parameters=["ir_drop"],
        production_month="2026-01",
        project_root=ROOT,
        temporal_bundle=hybrid,
    ).recommendations[0]
    assert ir.model_used == "core_gru_temporal_v1"
    assert ir.production_month == "2026-01"
    assert "TEMPORAL_2026-01" in ir.evidence_origin

    vmin = recommend(
        lot_id=LOT,
        die_id=DIE,
        parameters=["VMIN"],
        production_month="2026-02",
        project_root=ROOT,
        temporal_bundle=hybrid,
    ).recommendations[0]
    assert vmin.model_used == "unified_parameter_gru_v1"
    assert vmin.production_month == "2026-02"


def test_unsupported_remain_non_scorable():
    hybrid = TemporalHybridBundle(ROOT)
    for p in UNSUPPORTED:
        rec = recommend(
            lot_id=LOT,
            die_id=DIE,
            parameters=[p],
            production_month="2026-01",
            project_root=ROOT,
            temporal_bundle=hybrid,
        ).recommendations[0]
        assert rec.decision.value == "REJECT"
        assert rec.explanation.get("policy_reason") == "unsupported_parameter"
        assert rec.model_used is None


def test_current_candidate_and_policy_semantics():
    hybrid = TemporalHybridBundle(ROOT)
    rec = recommend(
        lot_id=LOT,
        die_id=DIE,
        parameters=["ir_drop"],
        production_month="2026-01",
        project_root=ROOT,
        temporal_bundle=hybrid,
    ).recommendations[0]
    assert rec.decision.value in {"RECOMMEND", "KEEP_CURRENT", "REVIEW_REQUIRED", "REJECT"}
    if rec.decision.value == "RECOMMEND":
        assert rec.recommended_limit != rec.current_limit
        assert rec.explanation.get("policy_reason") == "max_simulated_yield_selected"
    if rec.decision.value == "KEEP_CURRENT":
        assert rec.recommended_limit == rec.current_limit
    if rec.safety_result.get("status") == "HARD_FAIL":
        assert rec.decision.value in {"REJECT", "REVIEW_REQUIRED", "KEEP_CURRENT"}


def test_legacy_production_month_none_unchanged_path():
    legacy = recommend(
        lot_id="DTL_NORM_004",
        die_id="DTL_NORM_004_D048",
        parameters=["ir_drop"],
        production_month=None,
        project_root=ROOT,
    )
    assert legacy.production_month is None
    rec = legacy.recommendations[0]
    assert rec.production_month is None
    # Legacy uses core_gru / not temporal checkpoint id for model_used when stamped
    assert rec.model_used in {None, "core_gru"}
    assert "TEMPORAL" not in rec.evidence_origin


def test_candidate_ranking_scoped_to_month_die_parameter():
    hybrid = TemporalHybridBundle(ROOT)
    month_data = load_temporal_month("2026-01", project_root=ROOT)
    scored, err, model_used = hybrid.score_parameter(
        production_month="2026-01",
        lot_id=LOT,
        die_id=DIE,
        parameter="ir_drop",
        month_data=month_data,
    )
    assert err is None and scored is not None
    assert model_used == "core_gru_temporal_v1"
    assert set(scored["parameter"].astype(str).unique()) == {"ir_drop"}
    assert (scored["lot_id"].astype(str) == LOT).all()
    assert (scored["die_id"].astype(str) == DIE).all()


def test_api_schema_production_month_validation():
    from dtl_agent.api.schemas import RecommendationRequest
    from pydantic import ValidationError

    ok = RecommendationRequest(lot_id="L", die_id="D", production_month="2026-01")
    assert ok.production_month == "2026-01"
    legacy = RecommendationRequest(lot_id="L", die_id="D")
    assert legacy.production_month is None
    with pytest.raises(ValidationError):
        RecommendationRequest(lot_id="L", die_id="D", production_month="2025-12")
    with pytest.raises(ValidationError):
        RecommendationRequest(lot_id="L", die_id="D", production_month="")
    with pytest.raises(ValidationError):
        RecommendationRequest(lot_id="L", die_id="D", production_month="random")
