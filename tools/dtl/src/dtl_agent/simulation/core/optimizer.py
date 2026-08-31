"""Deterministic Core optimizer over simulated candidates."""

from __future__ import annotations

from dataclasses import dataclass

from dtl_agent.simulation.core.config import CoreSimulationConfig, ObjectiveWeights
from dtl_agent.simulation.core.engine import CandidateSimulationResult


@dataclass(frozen=True)
class SelectionRule:
    """Prefer closest-to-current on objective ties (documented Phase 4 rule)."""

    name: str = "max_objective_then_min_abs_delta"


def select_best_candidate(
    results: list[CandidateSimulationResult],
    *,
    weights: ObjectiveWeights | None = None,
) -> CandidateSimulationResult:
    if not results:
        raise ValueError("no candidates to select")
    eps = weights.tie_epsilon if weights else 1e-12

    def key(r: CandidateSimulationResult) -> tuple[float, float, float]:
        # Maximize objective; then minimize |delta| to current; then prefer smaller candidate for stability
        return (-r.objective_score, abs(r.candidate_delta), r.candidate_limit)

    ranked = sorted(results, key=key)
    best = ranked[0]
    # Mark selection statuses
    for r in results:
        r.selection_status = ""
    best.selection_status = "SELECTED_SIMULATED_OPTIMAL_CANDIDATE"
    # Mark near-ties
    for r in ranked[1:]:
        if abs(r.objective_score - best.objective_score) <= eps:
            r.selection_status = "TIED_OBJECTIVE_NOT_SELECTED"
        else:
            break
    return best


def select_best_joint(
    results: list[CandidateSimulationResult],
    *,
    weights: ObjectiveWeights | None = None,
) -> CandidateSimulationResult:
    """Joint selection: max objective, then min L1 |Δ| to current pair, then notes string."""
    if not results:
        raise ValueError("no joint candidates")
    eps = weights.tie_epsilon if weights else 1e-12

    def key(r: CandidateSimulationResult) -> tuple[float, float, str]:
        # candidate_delta stores L1 distance for joint rows
        return (-r.objective_score, abs(r.candidate_delta), r.notes)

    ranked = sorted(results, key=key)
    best = ranked[0]
    for r in results:
        r.selection_status = ""
    best.selection_status = "SELECTED_SIMULATED_OPTIMAL_JOINT_CANDIDATE"
    for r in ranked[1:]:
        if abs(r.objective_score - best.objective_score) <= eps:
            r.selection_status = "TIED_OBJECTIVE_NOT_SELECTED"
        else:
            break
    return best


def baseline_result(
    results: list[CandidateSimulationResult],
) -> CandidateSimulationResult | None:
    for r in results:
        if r.tighten_or_loosen == "CURRENT":
            return r
    return None
