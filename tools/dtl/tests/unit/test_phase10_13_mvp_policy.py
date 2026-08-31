"""Phase 10.13 MVP policy: highest ML-ranked safety-passing candidate."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.recommendation.policy import EvaluatedCandidate, apply_recommendation_policy
from dtl_agent.recommendation.schemas import Decision, GateStatus, RankedCandidate, SafetyResult, SimulationEvidence

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


def _ev(limit: float, found: bool = True, yield_: float | None = None) -> SimulationEvidence:
    y = yield_ if yield_ is not None else (0.99 if found else None)
    return SimulationEvidence(
        evidence_origin="SIMULATOR_DERIVED",
        population_level_aggregate=True,
        parameter="INTERCONNECT_RESISTANCE",
        candidate_limit=limit,
        simulated_yield=y,
        violation_rate=0.01 if found else None,
        borderline_rate=0.0 if found else None,
        found=found,
    )


def test_case_a_ml1_safe_differs_from_current():
    c25 = _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.78, ml_rank=1)
    c15 = _cand(candidate_limit=15.0, ml_score=0.26, ml_rank=5)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(c25, _ev(25.0, yield_=0.99), SafetyResult(GateStatus.PASS, [])),
            EvaluatedCandidate(c15, _ev(15.0, yield_=0.39), SafetyResult(GateStatus.PASS, [])),
        ],
        current_limit=15.0,
    )
    assert out.decision == Decision.RECOMMEND
    assert out.selected is not None
    assert out.selected.candidate_limit == 25.0
    assert out.reason == "max_simulated_yield_selected"


def test_case_b_ml1_fails_next_safe_wins():
    c25 = _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.78, ml_rank=1)
    c20 = _cand(candidate_limit=20.0, delta_absolute=5.0, tighten_or_loosen="LOOSER", ml_score=0.66, ml_rank=2)
    c15 = _cand(candidate_limit=15.0, ml_score=0.26, ml_rank=5)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(c25, _ev(25.0, yield_=0.99), SafetyResult(GateStatus.HARD_FAIL, [])),
            EvaluatedCandidate(c20, _ev(20.0, yield_=0.84), SafetyResult(GateStatus.PASS, [])),
            EvaluatedCandidate(c15, _ev(15.0, yield_=0.39), SafetyResult(GateStatus.PASS, [])),
        ],
        current_limit=15.0,
    )
    assert out.decision == Decision.RECOMMEND
    assert out.selected is not None
    assert out.selected.candidate_limit == 20.0


def test_case_c_multiple_safe_highest_rank():
    evaluated = [
        EvaluatedCandidate(
            _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.78, ml_rank=1),
            _ev(25.0, yield_=0.9907),
            SafetyResult(GateStatus.PASS, []),
        ),
        EvaluatedCandidate(
            _cand(candidate_limit=20.0, delta_absolute=5.0, tighten_or_loosen="LOOSER", ml_score=0.66, ml_rank=2),
            _ev(20.0, yield_=0.8433),
            SafetyResult(GateStatus.PASS, []),
        ),
        EvaluatedCandidate(
            _cand(candidate_limit=18.0, delta_absolute=3.0, tighten_or_loosen="LOOSER", ml_score=0.56, ml_rank=3),
            _ev(18.0, yield_=0.6795),
            SafetyResult(GateStatus.PASS, []),
        ),
        EvaluatedCandidate(
            _cand(candidate_limit=16.0, delta_absolute=1.0, tighten_or_loosen="LOOSER", ml_score=0.41, ml_rank=4),
            _ev(16.0, yield_=0.4833),
            SafetyResult(GateStatus.PASS, []),
        ),
        EvaluatedCandidate(
            _cand(candidate_limit=15.0, ml_score=0.26, ml_rank=5),
            _ev(15.0, yield_=0.3944),
            SafetyResult(GateStatus.PASS, []),
        ),
    ]
    out = apply_recommendation_policy(evaluated=evaluated, current_limit=15.0)
    assert out.selected is not None
    assert out.selected.candidate_limit == 25.0
    assert out.decision == Decision.RECOMMEND


def test_case_d_ml1_is_current():
    cur = _cand(candidate_limit=15.0, ml_score=0.9, ml_rank=1)
    alt = _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.2, ml_rank=2)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(cur, _ev(15.0, yield_=0.99), SafetyResult(GateStatus.PASS, [])),
            EvaluatedCandidate(alt, _ev(25.0, yield_=0.80), SafetyResult(GateStatus.PASS, [])),
        ],
        current_limit=15.0,
    )
    assert out.decision == Decision.KEEP_CURRENT
    assert out.selected is not None
    assert out.selected.candidate_limit == 15.0
    assert out.reason == "policy_selected_current"


def test_case_e_no_safe_alternative():
    cur = _cand(candidate_limit=15.0, ml_score=0.26, ml_rank=5)
    alt = _cand(candidate_limit=25.0, delta_absolute=10.0, tighten_or_loosen="LOOSER", ml_score=0.78, ml_rank=1)
    out = apply_recommendation_policy(
        evaluated=[
            EvaluatedCandidate(alt, _ev(25.0), SafetyResult(GateStatus.SOFT_FAIL, [])),
            EvaluatedCandidate(cur, _ev(15.0), SafetyResult(GateStatus.PASS, [])),
        ],
        current_limit=15.0,
    )
    assert out.decision == Decision.KEEP_CURRENT
    assert out.selected is not None
    assert out.selected.candidate_limit == 15.0
    assert out.reason == "no_safe_candidate"
    assert any("No eligible alternative" in t for t in out.policy_trace)


def test_case_f_insufficient_evidence():
    out = apply_recommendation_policy(evaluated=[], current_limit=15.0, insufficient_evidence=True)
    assert out.decision == Decision.REVIEW_REQUIRED
    assert out.reason == "insufficient_evidence"


def test_case_g_hard_constraint():
    out = apply_recommendation_policy(evaluated=[], current_limit=15.0, hard_reject=True)
    assert out.decision == Decision.REJECT
    assert out.reason == "hard_constraint_failure"


@pytest.mark.integration
def test_acceptance_interconnect_15_to_25():
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
    cands = [c for c in result.audit["candidate_set"] if c["parameter"] == "INTERCONNECT_RESISTANCE"]
    top = min(cands, key=lambda c: c["ml_rank"])
    assert rec.current_limit == 15.0
    assert top["candidate_limit"] == 25.0
    assert rec.decision == RecDecision.RECOMMEND
    assert rec.recommended_limit == 25.0
    assert rec.explanation["policy_reason"] == "max_simulated_yield_selected"
    assert rec.explanation.get("action_text")
    assert "15" in rec.explanation["action_text"] and "25" in rec.explanation["action_text"]
    current_rows = [c for c in cands if abs(c["candidate_limit"] - 15.0) < 1e-12]
    assert current_rows, "CURRENT remains in candidate set"
