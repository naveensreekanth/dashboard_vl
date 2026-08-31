"""Phase 8 recommendation schemas and decision vocabulary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    RECOMMEND = "RECOMMEND"
    KEEP_CURRENT = "KEEP_CURRENT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECT = "REJECT"


class GateStatus(str, Enum):
    PASS = "PASS"
    SOFT_FAIL = "SOFT_FAIL"
    HARD_FAIL = "HARD_FAIL"


class EvidenceLevel(str, Enum):
    HIGH_EVIDENCE = "HIGH_EVIDENCE"
    MODERATE_EVIDENCE = "MODERATE_EVIDENCE"
    LOW_EVIDENCE = "LOW_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


CORE_PARAMETERS = frozenset({"ir_drop", "thermal"})
PARAMETRIC_PARAMETERS = frozenset(
    {
        "VMIN",
        "VMAX",
        "IDDQ",
        "SUPPLY_CURRENT",
        "CONTACT_RESISTANCE",
        "INTERCONNECT_RESISTANCE",
        "ON_RESISTANCE",
    }
)
UNSUPPORTED_CORE_PARAMETERS = frozenset({"setup_slack", "hold_slack", "test_time"})
REQUIRED_PARAM_CONDITIONS = frozenset(
    {"COND_RT_NOM", "COND_HOT_NOM", "COND_RT_LOWV", "COND_HOT_HIGHV"}
)


@dataclass
class RankedCandidate:
    parameter: str
    test_id: str
    lot_id: str
    die_id: str
    current_limit: float
    candidate_limit: float
    delta_absolute: float
    delta_percent: float | None
    direction: str
    tighten_or_loosen: str
    unit: str
    source_status: str
    ml_score: float
    ml_rank: int
    model_id: str
    catalog_valid: bool
    condition_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyCheck:
    name: str
    passed: bool
    layer: int
    message: str
    severity: str  # hard | soft | info

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyResult:
    status: GateStatus
    checks: list[SafetyCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "checks": [c.to_dict() for c in self.checks]}


@dataclass
class SimulationEvidence:
    evidence_origin: str
    population_level_aggregate: bool
    parameter: str
    candidate_limit: float
    simulated_yield: float | None = None
    simulated_fail_rate: float | None = None
    violation_rate: float | None = None
    borderline_rate: float | None = None
    risky_rate: float | None = None
    false_fail_proxy: float | None = None
    defective_proxy: float | None = None
    objective_score: float | None = None
    worst_condition_yield: float | None = None
    worst_condition_violation_rate: float | None = None
    evaluated_conditions: Any = None
    found: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["note"] = (
            "SIMULATOR_DERIVED evidence. objective_score is not production reliability "
            "or true optimality."
        )
        return d


@dataclass
class DTLRecommendation:
    request_id: str
    lot_id: str
    die_id: str
    parameter: str
    test_id: str
    unit: str
    direction: str
    current_limit: float
    recommended_limit: float
    decision: Decision
    ml_score: float | None
    ml_rank: int | None
    n_candidates: int
    model_id: str | None
    source_status: str
    simulation_evidence: dict[str, Any]
    safety_result: dict[str, Any]
    evidence_level: EvidenceLevel
    explanation: dict[str, Any]
    model_version: str
    checkpoint_id: str | None
    dataset_version: str
    feature_registry_hash: str | None
    simulation_config_version: str | None
    policy_config_version: str
    timestamp: str
    core_available: bool
    parametric_available: bool
    cross_domain_available: bool
    evidence_origin: str = "SIMULATOR_DERIVED"
    production_month: str | None = None
    model_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        d["evidence_level"] = self.evidence_level.value
        return d


@dataclass
class LotRecommendationResult:
    request_id: str
    lot_id: str
    die_id: str
    recommendations: list[DTLRecommendation]
    audit: dict[str, Any]
    core_available: bool
    parametric_available: bool
    cross_domain_available: bool
    production_month: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "lot_id": self.lot_id,
            "die_id": self.die_id,
            "production_month": self.production_month,
            "core_available": self.core_available,
            "parametric_available": self.parametric_available,
            "cross_domain_available": self.cross_domain_available,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "audit": self.audit,
        }
