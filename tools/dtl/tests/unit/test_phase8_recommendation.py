"""Unit tests for Phase 8 recommendation (spec §16 cases)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dtl_agent.recommendation.catalog import CandidateCatalog
from dtl_agent.recommendation.config import RecommendationConfig, load_recommendation_config
from dtl_agent.recommendation.context import detect_forbidden_request
from dtl_agent.recommendation.evidence import SimulationEvidenceLookup
from dtl_agent.recommendation.policy import EvaluatedCandidate, apply_recommendation_policy
from dtl_agent.recommendation.ranking import rank_candidates, select_top_n_plus_current
from dtl_agent.recommendation.safety import evaluate_safety
from dtl_agent.recommendation.schemas import (
    Decision,
    GateStatus,
    RankedCandidate,
    SafetyResult,
    SimulationEvidence,
)


def _cand(**kwargs) -> RankedCandidate:
    base = dict(
        parameter="ir_drop",
        test_id="T_IR_DROP_MV",
        lot_id="L1",
        die_id="D1",
        current_limit=25.0,
        candidate_limit=25.0,
        delta_absolute=0.0,
        delta_percent=0.0,
        direction="UPPER",
        tighten_or_loosen="CURRENT",
        unit="mV",
        source_status="SOURCE_CONFIRMED",
        ml_score=0.5,
        ml_rank=1,
        model_id="core_gru",
        catalog_valid=True,
    )
    base.update(kwargs)
    return RankedCandidate(**base)


def _ev(found=True, yield_=1.0, viol=0.0, **kwargs) -> SimulationEvidence:
    return SimulationEvidence(
        evidence_origin="SIMULATOR_DERIVED",
        population_level_aggregate=True,
        parameter="ir_drop",
        candidate_limit=kwargs.get("candidate_limit", 25.0),
        simulated_yield=yield_ if found else None,
        violation_rate=viol if found else None,
        borderline_rate=0.0 if found else None,
        found=found,
    )


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog():
    if not (ROOT / "artifacts" / "simulation" / "core" / "candidate_grid.csv").is_file():
        pytest.skip("simulation artifacts missing")
    return CandidateCatalog(ROOT, RecommendationConfig())


def test_top_n_from_config_not_hardcoded(catalog):
    df = pd.DataFrame(
        {
            "parameter": ["ir_drop"] * 7,
            "test_id": ["T_IR_DROP_MV"] * 7,
            "current_limit": [25.0] * 7,
            "candidate_limit": [20.0, 21.0, 22.0, 25.0, 30.0, 35.0, 40.0],
            "candidate_delta": [-5, -4, -3, 0, 5, 10, 15],
            "candidate_delta_percent": [-20, -16, -12, 0, 20, 40, 60],
            "direction": ["UPPER"] * 7,
            "tighten_or_loosen": ["TIGHTER", "TIGHTER", "TIGHTER", "CURRENT", "LOOSER", "LOOSER", "LOOSER"],
            "unit": ["mV"] * 7,
            "source_status": ["SOURCE_CONFIRMED"] * 7,
            "ml_score": [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7],
            "model_id": ["core_gru"] * 7,
        }
    )
    ranked = rank_candidates(df, lot_id="L", die_id="D", catalog=catalog)
    cfg3 = RecommendationConfig(TOP_N=3)
    sel3 = select_top_n_plus_current(ranked, cfg3)
    # Top 3 by score are 30,35,40 — CURRENT 25 must still be included
    limits = {c.candidate_limit for c in sel3}
    assert 25.0 in limits
    assert len(sel3) == 4  # top3 + current

    cfg2 = RecommendationConfig(TOP_N=2)
    sel2 = select_top_n_plus_current(ranked, cfg2)
    assert len(sel2) == 3  # top2 + current
    assert {c.candidate_limit for c in sel2} != limits


def test_ranking_assigns_ml_rank_descending(catalog):
    df = pd.DataFrame(
        {
            "parameter": ["ir_drop", "ir_drop"],
            "test_id": ["T_IR_DROP_MV", "T_IR_DROP_MV"],
            "current_limit": [25.0, 25.0],
            "candidate_limit": [25.0, 30.0],
            "candidate_delta": [0.0, 5.0],
            "candidate_delta_percent": [0.0, 20.0],
            "direction": ["UPPER", "UPPER"],
            "tighten_or_loosen": ["CURRENT", "LOOSER"],
            "unit": ["mV", "mV"],
            "source_status": ["SOURCE_CONFIRMED", "SOURCE_CONFIRMED"],
            "ml_score": [0.2, 0.9],
            "model_id": ["core_gru", "core_gru"],
        }
    )
    ranked = rank_candidates(df, lot_id="L", die_id="D", catalog=catalog)
    assert ranked[0].candidate_limit == 30.0
    assert ranked[0].ml_rank == 1
    assert ranked[1].ml_rank == 2


def test_safety_hard_fail_out_of_catalog(catalog):
    cand = _cand(candidate_limit=999.0, delta_absolute=974.0, tighten_or_loosen="LOOSER")
    ev = _ev(candidate_limit=999.0)
    res = evaluate_safety(
        candidate=cand,
        evidence=ev,
        catalog=catalog,
        config=RecommendationConfig(),
        domain="core",
    )
    assert res.status == GateStatus.HARD_FAIL
    assert any(c.name == "catalog_membership" and not c.passed for c in res.checks)


def test_safety_soft_fail_missing_simulation(catalog):
    cand = _cand(candidate_limit=25.0)
    ev = _ev(found=False)
    res = evaluate_safety(
        candidate=cand,
        evidence=ev,
        catalog=catalog,
        config=RecommendationConfig(),
        domain="core",
    )
    assert res.status == GateStatus.SOFT_FAIL


def test_safety_pass_in_catalog_with_evidence(catalog):
    cand = _cand(candidate_limit=25.0)
    ev = _ev(found=True)
    res = evaluate_safety(
        candidate=cand,
        evidence=ev,
        catalog=catalog,
        config=RecommendationConfig(),
        domain="core",
        context_complete=True,
        model_available=True,
    )
    assert res.status == GateStatus.PASS


def test_policy_no_safe_keeps_current():
    cur = _cand(candidate_limit=25.0, ml_score=0.1, ml_rank=2)
    loose = _cand(
        candidate_limit=40.0,
        delta_absolute=15.0,
        tighten_or_loosen="LOOSER",
        ml_score=0.99,
        ml_rank=1,
    )
    evaluated = [
        EvaluatedCandidate(
            loose,
            _ev(candidate_limit=40.0),
            SafetyResult(GateStatus.SOFT_FAIL, []),
        ),
        EvaluatedCandidate(
            cur,
            _ev(candidate_limit=25.0),
            SafetyResult(GateStatus.SOFT_FAIL, []),
        ),
    ]
    out = apply_recommendation_policy(evaluated=evaluated, current_limit=25.0)
    assert out.decision == Decision.KEEP_CURRENT


def test_policy_ml_high_safety_fail_cannot_recommend():
    high = _cand(candidate_limit=40.0, delta_absolute=15.0, tighten_or_loosen="LOOSER", ml_score=0.99)
    cur = _cand(candidate_limit=25.0, ml_score=0.1)
    evaluated = [
        EvaluatedCandidate(high, _ev(candidate_limit=40.0), SafetyResult(GateStatus.HARD_FAIL, [])),
        EvaluatedCandidate(cur, _ev(candidate_limit=25.0), SafetyResult(GateStatus.PASS, [])),
    ]
    out = apply_recommendation_policy(evaluated=evaluated, current_limit=25.0)
    assert out.decision == Decision.KEEP_CURRENT
    assert out.selected is not None
    assert out.selected.candidate_limit == 25.0


def test_policy_max_yield_wins_over_ml_rank():
    near = _cand(candidate_limit=26.0, delta_absolute=1.0, tighten_or_loosen="LOOSER", ml_score=0.5, ml_rank=2)
    far = _cand(candidate_limit=40.0, delta_absolute=15.0, tighten_or_loosen="LOOSER", ml_score=0.99, ml_rank=1)
    cur = _cand(candidate_limit=25.0, ml_score=0.4, ml_rank=3)
    evaluated = [
        EvaluatedCandidate(far, _ev(candidate_limit=40.0, yield_=0.90), SafetyResult(GateStatus.PASS, [])),
        EvaluatedCandidate(near, _ev(candidate_limit=26.0, yield_=0.99), SafetyResult(GateStatus.PASS, [])),
        EvaluatedCandidate(cur, _ev(candidate_limit=25.0, yield_=0.80), SafetyResult(GateStatus.PASS, [])),
    ]
    out = apply_recommendation_policy(evaluated=evaluated, current_limit=25.0)
    assert out.decision == Decision.RECOMMEND
    assert out.selected is not None
    assert out.selected.candidate_limit == 26.0
    assert out.reason == "max_simulated_yield_selected"


def test_policy_recommend_when_current_unsafe_max_yield():
    near = _cand(candidate_limit=26.0, delta_absolute=1.0, tighten_or_loosen="LOOSER", ml_score=0.5, ml_rank=2)
    far = _cand(candidate_limit=40.0, delta_absolute=15.0, tighten_or_loosen="LOOSER", ml_score=0.99, ml_rank=1)
    cur = _cand(candidate_limit=25.0, ml_score=0.4, ml_rank=3)
    evaluated = [
        EvaluatedCandidate(far, _ev(candidate_limit=40.0, yield_=0.90), SafetyResult(GateStatus.PASS, [])),
        EvaluatedCandidate(near, _ev(candidate_limit=26.0, yield_=0.99), SafetyResult(GateStatus.PASS, [])),
        EvaluatedCandidate(cur, _ev(candidate_limit=25.0), SafetyResult(GateStatus.SOFT_FAIL, [])),
    ]
    out = apply_recommendation_policy(evaluated=evaluated, current_limit=25.0)
    assert out.decision == Decision.RECOMMEND
    assert out.selected is not None
    assert out.selected.candidate_limit == 26.0


def test_policy_insufficient_evidence_review():
    out = apply_recommendation_policy(
        evaluated=[], current_limit=25.0, insufficient_evidence=True
    )
    assert out.decision == Decision.REVIEW_REQUIRED


def test_policy_hard_reject():
    out = apply_recommendation_policy(evaluated=[], current_limit=25.0, hard_reject=True)
    assert out.decision == Decision.REJECT


def test_forbidden_data_detection():
    assert detect_forbidden_request(["data/core/evaluation/foo.csv"]) is True
    assert detect_forbidden_request(["artifacts/ml/checkpoints/core_gru_best.pt"]) is False


def test_joint_disabled_in_default_config():
    cfg = RecommendationConfig()
    assert cfg.joint_enabled is False
    assert cfg.include_tree_baseline_diagnostic is False
    assert cfg.evidence_origin_label == "SIMULATOR_DERIVED"
    assert cfg.max_violation_rate_for_recommend is None


def test_simulation_lookup_never_invents():
    if not (ROOT / "artifacts" / "simulation" / "core" / "candidate_results.csv").is_file():
        pytest.skip("simulation artifacts missing")
    lookup = SimulationEvidenceLookup(ROOT, RecommendationConfig())
    miss = lookup.lookup(domain="core", parameter="ir_drop", candidate_limit=123456.0)
    assert miss.found is False
    assert miss.evidence_origin == "SIMULATOR_DERIVED"
    hit = lookup.lookup(domain="core", parameter="ir_drop", candidate_limit=25.0)
    assert hit.found is True


def test_layer3_threshold_applied_when_configured(catalog):
    cand = _cand(candidate_limit=25.0)
    ev = _ev(found=True, viol=0.5)
    cfg = RecommendationConfig(max_violation_rate_for_recommend=0.01)
    res = evaluate_safety(
        candidate=cand,
        evidence=ev,
        catalog=catalog,
        config=cfg,
        domain="core",
    )
    assert res.status == GateStatus.SOFT_FAIL
    assert any(c.name == "max_violation_rate" and not c.passed for c in res.checks)


def test_config_simulation_path_defaults_resolve_to_current_artifacts():
    cfg = RecommendationConfig()
    assert cfg.core_candidate_grid_path == "artifacts/simulation/core/candidate_grid.csv"
    assert cfg.core_candidate_results_path == "artifacts/simulation/core/candidate_results.csv"
    assert (
        cfg.parametric_candidate_grid_path
        == "artifacts/simulation/parametric/candidate_grid.csv"
    )
    assert (
        cfg.parametric_candidate_results_path
        == "artifacts/simulation/parametric/candidate_results.csv"
    )
    assert cfg.resolve_path(ROOT, cfg.core_candidate_grid_path).is_file()


def test_load_recommendation_config_preserves_null_layer3():
    cfg = load_recommendation_config(None)
    assert cfg.max_violation_rate_for_recommend is None
    assert cfg.looser_requires_stricter_gate is None


def test_catalog_valid_reflects_membership(catalog):
    df = pd.DataFrame(
        {
            "parameter": ["ir_drop", "ir_drop"],
            "test_id": ["T_IR_DROP_MV", "T_IR_DROP_MV"],
            "current_limit": [25.0, 25.0],
            "candidate_limit": [25.0, 999.0],
            "candidate_delta": [0.0, 974.0],
            "candidate_delta_percent": [0.0, 3896.0],
            "direction": ["UPPER", "UPPER"],
            "tighten_or_loosen": ["CURRENT", "LOOSER"],
            "unit": ["mV", "mV"],
            "source_status": ["SOURCE_CONFIRMED", "SOURCE_CONFIRMED"],
            "ml_score": [0.2, 0.9],
            "model_id": ["core_gru", "core_gru"],
        }
    )
    ranked = rank_candidates(df, lot_id="L", die_id="D", catalog=catalog)
    by_limit = {c.candidate_limit: c.catalog_valid for c in ranked}
    assert by_limit[25.0] is True
    assert by_limit[999.0] is False


def test_core_seq_features_matches_canonical_order():
    from dtl_agent.features.core_engine import SEQUENCE_FEATURE_ORDER
    from dtl_agent.ml.datasets.phase7_datasets import CORE_SEQ_FEATURES

    assert CORE_SEQ_FEATURES == list(SEQUENCE_FEATURE_ORDER)
