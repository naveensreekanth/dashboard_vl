"""Explanation and audit writers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dtl_agent.recommendation.config import RecommendationConfig
from dtl_agent.recommendation.context import RecommendationContext
from dtl_agent.recommendation.policy import PolicyResult
from dtl_agent.recommendation.schemas import (
    Decision,
    EvidenceLevel,
    RankedCandidate,
    SafetyResult,
    SimulationEvidence,
)


def _synthetic_assumed_cap(config: RecommendationConfig) -> EvidenceLevel:
    """Cap for SYNTHETIC_ASSUMED sources; never exceeds MODERATE_EVIDENCE."""
    raw = config.synthetic_assumed_max_evidence_level
    try:
        level = EvidenceLevel(raw)
    except ValueError:
        return EvidenceLevel.MODERATE_EVIDENCE
    if level == EvidenceLevel.HIGH_EVIDENCE:
        return EvidenceLevel.MODERATE_EVIDENCE
    if level in {
        EvidenceLevel.MODERATE_EVIDENCE,
        EvidenceLevel.LOW_EVIDENCE,
        EvidenceLevel.INSUFFICIENT_EVIDENCE,
    }:
        return level
    return EvidenceLevel.MODERATE_EVIDENCE


def compute_evidence_level(
    *,
    config: RecommendationConfig,
    source_status: str,
    evidence: SimulationEvidence | None,
    decision: Decision,
) -> EvidenceLevel:
    if decision == Decision.REVIEW_REQUIRED or evidence is None or not evidence.found:
        return EvidenceLevel.INSUFFICIENT_EVIDENCE
    level = EvidenceLevel.HIGH_EVIDENCE
    if source_status == "SYNTHETIC_ASSUMED":
        level = _synthetic_assumed_cap(config)
    if evidence.population_level_aggregate and level == EvidenceLevel.HIGH_EVIDENCE:
        level = EvidenceLevel.MODERATE_EVIDENCE
    return level


def _fmt_limit(value: float | None, unit: str | None) -> str:
    if value is None:
        return "—"
    if unit:
        return f"{value} {unit}"
    return str(value)


def build_explanation(
    *,
    decision: Decision,
    current_limit: float,
    recommended_limit: float,
    selected: RankedCandidate | None,
    evidence: SimulationEvidence | None,
    safety: SafetyResult | None,
    policy: PolicyResult,
    config: RecommendationConfig,
) -> dict[str, Any]:
    passed = [c.name for c in (safety.checks if safety else []) if c.passed]
    failed = [c.name for c in (safety.checks if safety else []) if not c.passed]
    unit = selected.unit if selected is not None else None
    current_s = _fmt_limit(current_limit, unit)
    rec_s = _fmt_limit(recommended_limit, unit)
    selected_s = _fmt_limit(selected.candidate_limit, unit) if selected is not None else None

    ml_ranking_text = None
    safety_text = None
    decision_text = None
    action_text = None
    selection_text = "Highest simulated yield among eligible candidates."
    if policy.yield_tie:
        selection_text = (
            "Candidates had equivalent simulated yield, so ML ranking was used as the tie-breaker."
        )
    simulator_note = "Recommended based on simulator-derived evidence."

    if selected is not None:
        ml_ranking_text = f"ML rank {selected.ml_rank} (tie-breaker only; not the primary criterion)."

    if safety is not None and selected is not None:
        if not failed:
            safety_text = f"{selected_s} passed the required safety checks."
        else:
            safety_text = (
                f"{selected_s} did not pass all required safety checks: {', '.join(failed)}."
            )

    if decision == Decision.RECOMMEND and selected is not None:
        decision_text = (
            f"{selected_s} was selected for the highest simulated yield among eligible candidates."
        )
        action_text = f"Change DTL limit from {current_s} to {rec_s}."
    elif decision == Decision.KEEP_CURRENT:
        decision_text = f"Current DTL {current_s} remains selected."
        action_text = "Keep current DTL"
        if policy.reason == "no_safe_candidate":
            decision_text = (
                "No eligible alternative candidate passed the required safety and evidence checks."
            )
    elif decision == Decision.REVIEW_REQUIRED:
        decision_text = "Review required. Do not change DTL until required evidence is available."
        action_text = "Do not change DTL until required evidence is available."
    elif decision == Decision.REJECT:
        decision_text = "Candidate not eligible."
        action_text = "Do not change DTL."

    narrative = " ".join(
        t
        for t in (selection_text, safety_text, decision_text, action_text, simulator_note)
        if t
    )
    text = (
        f"{narrative} "
        f"Decision={decision.value}. "
        f"Current={current_limit}, recommended={recommended_limit}. "
        f"Reason={policy.reason}. "
        f"Evidence origin={config.evidence_origin_label}. "
        "Simulator objective_score is not production reliability or true optimality."
    )
    out: dict[str, Any] = {
        "text": text,
        "policy_reason": policy.reason,
        "policy_trace": policy.policy_trace,
        "safety_checks_passed": passed,
        "safety_checks_failed": failed,
        "ml_ranking_text": ml_ranking_text,
        "safety_text": safety_text,
        "decision_text": decision_text,
        "action_text": action_text,
        "selection_rule": (
            "highest simulated yield among eligible candidates; ML rank used as tie-breaker"
        ),
        "primary_criterion": "simulated_yield",
        "selected_simulated_yield": policy.selected_yield,
        "yield_tie": policy.yield_tie,
        "tie_breaker": "ML rank" if policy.yield_tie else "not required",
        "selection_text": selection_text,
        "simulator_note": simulator_note,
        "evidence_origin": config.evidence_origin_label,
        "disclaimer": (
            "SIMULATOR_DERIVED evidence only. Do not interpret objective_score as "
            "production reliability or true optimality."
        ),
        "tree_baseline_diagnostic": None,
    }
    if selected is not None:
        out["ml_rank"] = selected.ml_rank
        out["ml_score"] = selected.ml_score
        out["model_id"] = selected.model_id
        out["tighten_or_loosen"] = selected.tighten_or_loosen
    if evidence is not None:
        out["simulation"] = {
            "found": evidence.found,
            "simulated_yield": evidence.simulated_yield,
            "violation_rate": evidence.violation_rate,
            "borderline_rate": evidence.borderline_rate,
            "worst_condition_yield": evidence.worst_condition_yield,
            "objective_score": evidence.objective_score,
            "population_level_aggregate": evidence.population_level_aggregate,
        }
    if config.include_tree_baseline_diagnostic:
        out["tree_baseline_diagnostic"] = {
            "enabled": True,
            "note": "diagnostic/reference only; not decision authority",
        }
    return out


def build_audit_record(
    *,
    request_id: str,
    ctx: RecommendationContext,
    config: RecommendationConfig,
    parameters_requested: list[str],
    candidate_set: list[dict[str, Any]],
    ml_predictions: list[dict[str, Any]],
    simulation_rows: list[dict[str, Any]],
    safety_traces: list[dict[str, Any]],
    policy_traces: list[str],
    final_decisions: list[dict[str, Any]],
    checkpoint_ids: dict[str, str | None],
    simulation_config_version: str | None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "lot_id": ctx.lot_id,
        "die_id": ctx.die_id,
        "parameters_requested": parameters_requested,
        "core_available": ctx.core_available,
        "parametric_available": ctx.parametric_available,
        "cross_domain_available": ctx.cross_domain_available,
        "dataset_version": ctx.dataset_version_core,
        "dataset_version_parametric": ctx.dataset_version_parametric,
        "feature_registry_hash": ctx.feature_registry_hash,
        "ml_dataset_version": ctx.ml_dataset_version,
        "model_version": ctx.package_version,
        "checkpoint_id": checkpoint_ids,
        "simulation_config_version": simulation_config_version,
        "policy_config_version": config.policy_config_version,
        "TOP_N": config.TOP_N,
        "candidate_set": candidate_set,
        "ml_predictions": ml_predictions,
        "simulation_evidence_rows": simulation_rows,
        "safety_check_trace": safety_traces,
        "policy_trace": policy_traces,
        "final_decisions": final_decisions,
        "evidence_origin": config.evidence_origin_label,
        "include_tree_baseline_diagnostic": config.include_tree_baseline_diagnostic,
        "joint_enabled": config.joint_enabled,
        "context_errors": ctx.errors,
    }
