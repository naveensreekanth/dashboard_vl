"""Recommendation policy and decision selection."""

from __future__ import annotations

from dataclasses import dataclass

from dtl_agent.recommendation.schemas import (
    Decision,
    GateStatus,
    RankedCandidate,
    SafetyResult,
    SimulationEvidence,
)


@dataclass
class EvaluatedCandidate:
    candidate: RankedCandidate
    evidence: SimulationEvidence
    safety: SafetyResult


@dataclass
class PolicyResult:
    decision: Decision
    selected: RankedCandidate | None
    reason: str
    safe_set_size: int
    policy_trace: list[str]
    yield_tie: bool = False
    selected_yield: float | None = None


def _is_current(candidate: RankedCandidate, current_limit: float) -> bool:
    """Identify CURRENT by limit identity, not by |delta| (delta is not a selection key)."""
    return (
        candidate.tighten_or_loosen == "CURRENT"
        or abs(candidate.candidate_limit - current_limit) < 1e-12
    )


def _simulated_yield(e: EvaluatedCandidate) -> float:
    y = e.evidence.simulated_yield
    return float("-inf") if y is None else y


def apply_recommendation_policy(
    *,
    evaluated: list[EvaluatedCandidate],
    current_limit: float,
    insufficient_evidence: bool = False,
    hard_reject: bool = False,
) -> PolicyResult:
    trace: list[str] = []

    if hard_reject:
        return PolicyResult(
            decision=Decision.REJECT,
            selected=None,
            reason="hard_constraint_failure",
            safe_set_size=0,
            policy_trace=["REJECT: hard constraint failure"],
        )
    if insufficient_evidence:
        return PolicyResult(
            decision=Decision.REVIEW_REQUIRED,
            selected=None,
            reason="insufficient_evidence",
            safe_set_size=0,
            policy_trace=["REVIEW_REQUIRED: insufficient evidence/context"],
        )

    eligible = [e for e in evaluated if e.safety.status == GateStatus.PASS]
    trace.append(f"safe_set_size={len(eligible)}")

    current = next(
        (e for e in evaluated if _is_current(e.candidate, current_limit)),
        None,
    )

    if not eligible:
        sel = current.candidate if current else None
        return PolicyResult(
            decision=Decision.KEEP_CURRENT,
            selected=sel,
            reason="no_safe_candidate",
            safe_set_size=0,
            policy_trace=trace
            + [
                "KEEP_CURRENT: empty eligible set",
                "No eligible alternative candidate passed the required safety and evidence checks.",
            ],
        )

    # Primary: maximum simulated_yield. Secondary: best ML rank. |delta| is not used.
    ordered = sorted(
        eligible,
        key=lambda e: (-_simulated_yield(e), e.candidate.ml_rank, -e.candidate.ml_score),
    )
    winner = ordered[0]
    win_y = winner.evidence.simulated_yield
    tied = [e for e in eligible if e.evidence.simulated_yield == win_y]
    yield_tie = len(tied) > 1

    trace.append(
        "Selection rule: highest simulated yield among eligible candidates; "
        "ML rank used as tie-breaker"
    )
    trace.append("Primary criterion: simulated_yield")
    trace.append(f"Selected simulated yield: {win_y}")
    trace.append(f"ML rank: {winner.candidate.ml_rank}")
    if yield_tie:
        trace.append("Tie: yes")
        trace.append("Tie-breaker: ML rank")
        trace.append(f"Selected ML rank: {winner.candidate.ml_rank}")
    else:
        trace.append("Tie-breaker: not required")
    trace.append(
        f"winner_limit={winner.candidate.candidate_limit} rank={winner.candidate.ml_rank}"
    )

    non_current_eligible = [e for e in eligible if not _is_current(e.candidate, current_limit)]

    if _is_current(winner.candidate, current_limit):
        if not non_current_eligible:
            return PolicyResult(
                decision=Decision.KEEP_CURRENT,
                selected=winner.candidate,
                reason="no_safe_candidate",
                safe_set_size=len(eligible),
                policy_trace=trace
                + [
                    "KEEP_CURRENT: no eligible non-current candidate",
                    "No eligible alternative candidate passed the required safety and evidence checks.",
                ],
                yield_tie=yield_tie,
                selected_yield=win_y,
            )
        return PolicyResult(
            decision=Decision.KEEP_CURRENT,
            selected=winner.candidate,
            reason="policy_selected_current",
            safe_set_size=len(eligible),
            policy_trace=trace
            + ["KEEP_CURRENT: maximum simulated yield equals current limit"],
            yield_tie=yield_tie,
            selected_yield=win_y,
        )

    return PolicyResult(
        decision=Decision.RECOMMEND,
        selected=winner.candidate,
        reason="max_simulated_yield_selected",
        safe_set_size=len(eligible),
        policy_trace=trace
        + ["RECOMMEND: highest simulated yield among eligible candidates"],
        yield_tie=yield_tie,
        selected_yield=win_y,
    )
