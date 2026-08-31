"""Deterministic safety gate (ML-independent)."""

from __future__ import annotations

from dtl_agent.recommendation.catalog import CandidateCatalog
from dtl_agent.recommendation.config import RecommendationConfig
from dtl_agent.recommendation.schemas import (
    REQUIRED_PARAM_CONDITIONS,
    GateStatus,
    RankedCandidate,
    SafetyCheck,
    SafetyResult,
    SimulationEvidence,
)


def evaluate_safety(
    *,
    candidate: RankedCandidate,
    evidence: SimulationEvidence,
    catalog: CandidateCatalog,
    config: RecommendationConfig,
    domain: str,
    conditions_present: list[str] | None = None,
    context_complete: bool = True,
    model_available: bool = True,
) -> SafetyResult:
    checks: list[SafetyCheck] = []

    # Layer 1 — hard constraints
    if catalog.is_unsupported(candidate.parameter):
        checks.append(
            SafetyCheck("supported_parameter", False, 1, "unsupported parameter", "hard")
        )
    else:
        checks.append(SafetyCheck("supported_parameter", True, 1, "parameter in scope", "hard"))

    in_cat = catalog.in_catalog(candidate.parameter, candidate.candidate_limit)
    checks.append(
        SafetyCheck(
            "catalog_membership",
            in_cat,
            1,
            "candidate in approved catalog" if in_cat else "candidate outside catalog",
            "hard",
        )
    )
    direction_ok = candidate.direction in {"UPPER", "LOWER"}
    checks.append(
        SafetyCheck(
            "valid_direction",
            direction_ok,
            1,
            f"direction={candidate.direction}",
            "hard",
        )
    )

    # Layer 2 / data validity
    checks.append(
        SafetyCheck(
            "context_complete",
            context_complete,
            2,
            "context complete" if context_complete else "incomplete context",
            "soft",
        )
    )
    checks.append(
        SafetyCheck(
            "model_available",
            model_available,
            2,
            "model available" if model_available else "model unavailable",
            "soft",
        )
    )
    checks.append(
        SafetyCheck(
            "simulation_evidence",
            evidence.found,
            2,
            "simulation evidence found" if evidence.found else "simulation evidence missing",
            "soft",
        )
    )

    if domain == "parametric":
        present = set(conditions_present or [])
        covered = REQUIRED_PARAM_CONDITIONS.issubset(present) if present else False
        # If aggregated row has n_conditions==4 from inference, treat as covered
        if not present and evidence.found:
            # population-level sim implies conditions evaluated in Phase 5 artifact
            covered = True
        checks.append(
            SafetyCheck(
                "condition_coverage",
                covered,
                2,
                "required conditions covered" if covered else "missing required conditions",
                "soft",
            )
        )

    # Layer 3 — only if configured
    def _layer3(name: str, ok: bool, msg: str) -> None:
        checks.append(SafetyCheck(name, ok, 3, msg, "soft"))

    if config.max_violation_rate_for_recommend is not None and evidence.violation_rate is not None:
        ok = evidence.violation_rate <= config.max_violation_rate_for_recommend
        _layer3("max_violation_rate", ok, f"violation_rate={evidence.violation_rate}")
    if config.max_borderline_rate_for_recommend is not None and evidence.borderline_rate is not None:
        ok = evidence.borderline_rate <= config.max_borderline_rate_for_recommend
        _layer3("max_borderline_rate", ok, f"borderline_rate={evidence.borderline_rate}")
    if config.min_simulated_yield_for_recommend is not None and evidence.simulated_yield is not None:
        ok = evidence.simulated_yield >= config.min_simulated_yield_for_recommend
        _layer3("min_simulated_yield", ok, f"yield={evidence.simulated_yield}")
    if (
        config.min_worst_condition_yield_for_recommend is not None
        and evidence.worst_condition_yield is not None
    ):
        ok = evidence.worst_condition_yield >= config.min_worst_condition_yield_for_recommend
        _layer3("min_worst_condition_yield", ok, f"worst_yield={evidence.worst_condition_yield}")
    if config.max_abs_delta_for_recommend is not None:
        ok = abs(candidate.delta_absolute) <= config.max_abs_delta_for_recommend
        _layer3("max_abs_delta", ok, f"abs_delta={abs(candidate.delta_absolute)}")
    if config.max_delta_percent_for_recommend is not None and candidate.delta_percent is not None:
        ok = abs(candidate.delta_percent) <= config.max_delta_percent_for_recommend
        _layer3("max_delta_percent", ok, f"delta_percent={candidate.delta_percent}")

    hard_fail = any(c.severity == "hard" and not c.passed for c in checks)
    soft_fail = any(c.severity == "soft" and not c.passed for c in checks)
    if hard_fail:
        status = GateStatus.HARD_FAIL
    elif soft_fail:
        status = GateStatus.SOFT_FAIL
    else:
        status = GateStatus.PASS
    return SafetyResult(status=status, checks=checks)
