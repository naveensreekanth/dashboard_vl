"""Deterministic Parametric optimizer."""

from __future__ import annotations

from dtl_agent.simulation.parametric.config import ObjectiveWeights
from dtl_agent.simulation.parametric.engine import ParametricCandidateResult


def select_best_candidate(
    results: list[ParametricCandidateResult],
    *,
    weights: ObjectiveWeights | None = None,
) -> ParametricCandidateResult:
    if not results:
        raise ValueError("no candidates to select")
    eps = weights.tie_epsilon if weights else 1e-12

    def key(r: ParametricCandidateResult) -> tuple[float, float, float]:
        return (-r.objective_score, abs(r.candidate_delta), r.candidate_limit)

    ranked = sorted(results, key=key)
    best = ranked[0]
    for r in results:
        r.selection_status = ""
    best.selection_status = "SELECTED_SIMULATED_OPTIMAL_CANDIDATE"
    for r in ranked[1:]:
        if abs(r.objective_score - best.objective_score) <= eps:
            r.selection_status = "TIED_OBJECTIVE_NOT_SELECTED"
        else:
            break
    return best


def baseline_result(results: list[ParametricCandidateResult]) -> ParametricCandidateResult | None:
    for r in results:
        if r.tighten_or_loosen == "CURRENT":
            return r
    return None
