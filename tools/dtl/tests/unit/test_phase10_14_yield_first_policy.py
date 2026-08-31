"""Phase 10.14 yield-first policy: max simulated_yield, ML rank tie-break."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.recommendation.config import RecommendationConfig
from dtl_agent.recommendation.policy import EvaluatedCandidate, apply_recommendation_policy
from dtl_agent.recommendation.safety import evaluate_safety
from dtl_agent.recommendation.schemas import (
    REQUIRED_PARAM_CONDITIONS,
    Decision,
    GateStatus,
    RankedCandidate,
    SafetyResult,
    SimulationEvidence,
)

ROOT = Path(__file__).resolve().parents[2]


def _cand(**kwargs) -> RankedCandidate:
    base = dict(
        parameter="INTERCONNECT_RESISTANCE",
        test_id="T_INTERCONNECT_R",
        lot_id="L1",
        die_id="D1",
        current_limit=15.0,
        candidate_limit=15.0,
        delta_absolute=0.0,
        delta_percent=0.0,
        direction="UPPER",
        tighten_or_loosen="CURRENT",
        unit="ohm",
        source_status="SYNTHETIC_ASSUMED",
        ml_score=0.5,
        ml_rank=5,
        model_id="parametric_mlp",
        catalog_valid=True,
    )
    base.update(kwargs)
    return RankedCandidate(**base)


def _ev(limit: float, yield_: float | None = 0.5, found: bool = True) -> SimulationEvidence:
    return SimulationEvidence(
        evidence_origin="SIMULATOR_DERIVED",
        population_level_aggregate=True,
        parameter="INTERCONNECT_RESISTANCE",
        candidate_limit=limit,
        simulated_yield=yield_ if found else None,
        violation_rate=0.01 if found else None,
        borderline_rate=0.0 if found else None,
        found=found,
    )


def _pass() -> SafetyResult:
    return SafetyResult(GateStatus.PASS, [])


def test_max_yield_wins_over_better_ml_rank():
    """Mandatory: yield 99% rank #2 beats yield 98% rank #1."""
    a = _cand(candidate_limit=20.0, delta_absolute=5.0, tighten_or_loosen="LOOSER", ml_score=0.5, ml_rank=2)
    b = _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.9, ml_rank=1)
    cur = _cand(candidate_limit=15.0, ml_score=0.2, ml_rank=3)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(a, _ev(20.0, 0.99), _pass()),
            EvaluatedCandidate(b, _ev(25.0, 0.98), _pass()),
            EvaluatedCandidate(cur, _ev(15.0, 0.39), _pass()),
        ],
        current_limit=15.0,
    )
    assert out.decision == Decision.RECOMMEND
    assert out.selected is not None
    assert out.selected.candidate_limit == 20.0
    assert out.reason == "max_simulated_yield_selected"
    assert out.yield_tie is False


def test_ml_rank_breaks_equal_yield():
    a = _cand(candidate_limit=20.0, delta_absolute=5.0, tighten_or_loosen="LOOSER", ml_score=0.5, ml_rank=2)
    b = _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.9, ml_rank=1)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(a, _ev(20.0, 0.90), _pass()),
            EvaluatedCandidate(b, _ev(25.0, 0.90), _pass()),
        ],
        current_limit=15.0,
    )
    assert out.selected is not None
    assert out.selected.candidate_limit == 25.0
    assert out.yield_tie is True
    assert any("Tie: yes" in t for t in out.policy_trace)
    assert any("Tie-breaker: ML rank" in t for t in out.policy_trace)


def test_current_wins_when_max_yield():
    cur = _cand(candidate_limit=15.0, ml_score=0.2, ml_rank=5)
    alt = _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.9, ml_rank=1)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(cur, _ev(15.0, 0.99), _pass()),
            EvaluatedCandidate(alt, _ev(25.0, 0.80), _pass()),
        ],
        current_limit=15.0,
    )
    assert out.decision == Decision.KEEP_CURRENT
    assert out.selected is not None
    assert out.selected.candidate_limit == 15.0
    assert out.reason == "policy_selected_current"


def test_high_yield_safety_fail_excluded():
    a = _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.9, ml_rank=1)
    b = _cand(candidate_limit=20.0, delta_absolute=5.0, tighten_or_loosen="LOOSER", ml_score=0.5, ml_rank=2)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(a, _ev(25.0, 0.99), SafetyResult(GateStatus.SOFT_FAIL, [])),
            EvaluatedCandidate(b, _ev(20.0, 0.84), _pass()),
        ],
        current_limit=15.0,
    )
    assert out.selected is not None
    assert out.selected.candidate_limit == 20.0
    assert out.decision == Decision.RECOMMEND


def test_missing_simulation_evidence_excludes_candidate():
    a = _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.9, ml_rank=1)
    b = _cand(candidate_limit=20.0, delta_absolute=5.0, tighten_or_loosen="LOOSER", ml_score=0.5, ml_rank=2)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(a, _ev(25.0, 0.99, found=False), SafetyResult(GateStatus.SOFT_FAIL, [])),
            EvaluatedCandidate(b, _ev(20.0, 0.84), _pass()),
        ],
        current_limit=15.0,
    )
    assert out.selected is not None
    assert out.selected.candidate_limit == 20.0


def test_missing_condition_coverage_excludes_via_safety(catalog):
    cat = catalog
    high = _cand(
        candidate_limit=25.0,
        current_limit=15.0,
        delta_absolute=10.0,
        tighten_or_loosen="LOOSER",
        ml_rank=1,
    )
    low = _cand(
        candidate_limit=20.0,
        current_limit=15.0,
        delta_absolute=5.0,
        tighten_or_loosen="LOOSER",
        ml_rank=2,
    )
    high_ev = _ev(25.0, 0.99)
    low_ev = _ev(20.0, 0.84)
    incomplete = list(REQUIRED_PARAM_CONDITIONS)[:1]
    high_s = evaluate_safety(
        candidate=high,
        evidence=high_ev,
        catalog=cat,
        config=RecommendationConfig(),
        domain="parametric",
        conditions_present=incomplete,
        context_complete=True,
        model_available=True,
    )
    low_s = evaluate_safety(
        candidate=low,
        evidence=low_ev,
        catalog=cat,
        config=RecommendationConfig(),
        domain="parametric",
        conditions_present=list(REQUIRED_PARAM_CONDITIONS),
        context_complete=True,
        model_available=True,
    )
    assert high_s.status != GateStatus.PASS
    assert any(c.name == "condition_coverage" and not c.passed for c in high_s.checks)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(high, high_ev, high_s),
            EvaluatedCandidate(low, low_ev, low_s),
        ],
        current_limit=15.0,
    )
    assert out.selected is not None
    assert out.selected.candidate_limit == 20.0


def test_hard_constraint_reject():
    out = apply_recommendation_policy(evaluated=[], current_limit=15.0, hard_reject=True)
    assert out.decision == Decision.REJECT


def test_review_required():
    out = apply_recommendation_policy(evaluated=[], current_limit=15.0, insufficient_evidence=True)
    assert out.decision == Decision.REVIEW_REQUIRED


def test_recommend_and_keep_current_decisions():
    rec = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(
                _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_rank=1),
                _ev(25.0, 0.99),
                _pass(),
            ),
            EvaluatedCandidate(_cand(candidate_limit=15.0, ml_rank=5), _ev(15.0, 0.39), _pass()),
        ],
        current_limit=15.0,
    )
    assert rec.decision == Decision.RECOMMEND
    keep = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(_cand(candidate_limit=15.0, ml_rank=1), _ev(15.0, 0.99), _pass()),
            EvaluatedCandidate(
                _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_rank=2),
                _ev(25.0, 0.50),
                _pass(),
            ),
        ],
        current_limit=15.0,
    )
    assert keep.decision == Decision.KEEP_CURRENT


@pytest.fixture
def catalog():
    from dtl_agent.recommendation.catalog import CandidateCatalog

    if not (ROOT / "artifacts" / "simulation" / "parametric" / "candidate_grid.csv").is_file():
        pytest.skip("simulation artifacts missing")
    return CandidateCatalog(ROOT, RecommendationConfig())


@pytest.mark.integration
def test_acceptance_interconnect_15_to_25_yield_first():
    from dtl_agent.recommendation import Decision as RecDecision
    from dtl_agent.recommendation import recommend

    if not (ROOT / "artifacts" / "ml" / "checkpoints" / "parametric_mlp_best.pt").is_file():
        pytest.skip("checkpoints missing")
    result = recommend(
        lot_id="DTL_NORM_001",
        die_id="DTL_NORM_001_D001",
        parameters=["INTERCONNECT_RESISTANCE"],
        project_root=ROOT,
    )
    rec = next(r for r in result.recommendations if r.parameter == "INTERCONNECT_RESISTANCE")
    assert rec.current_limit == 15.0
    assert rec.decision == RecDecision.RECOMMEND
    assert rec.recommended_limit == 25.0
    assert rec.explanation["policy_reason"] == "max_simulated_yield_selected"
    assert rec.explanation["primary_criterion"] == "simulated_yield"
    assert rec.explanation["yield_tie"] is False
    assert "highest simulated yield" in rec.explanation["selection_rule"]
    sims = [
        s
        for s in result.audit.get("simulation_evidence_rows") or []
        if s.get("parameter") == "INTERCONNECT_RESISTANCE" and s.get("found")
    ]
    by_limit = {s["candidate_limit"]: s.get("simulated_yield") for s in sims}
    assert by_limit.get(25.0) is not None
    assert by_limit[25.0] == max(y for y in by_limit.values() if y is not None)
    assert rec.explanation.get("selected_simulated_yield") == by_limit[25.0]
    cands = [c for c in result.audit["candidate_set"] if c["parameter"] == "INTERCONNECT_RESISTANCE"]
    assert any(abs(c["candidate_limit"] - 15.0) < 1e-12 for c in cands)
