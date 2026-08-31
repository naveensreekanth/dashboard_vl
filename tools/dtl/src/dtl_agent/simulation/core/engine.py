"""Core candidate-limit simulation engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from dtl_agent.features.margins import LimitSpec, proximity_class
from dtl_agent.simulation.core.candidates import CandidateLimit, classify_tighten_loosen
from dtl_agent.simulation.core.config import CoreSimulationConfig, ObjectiveWeights
from dtl_agent.simulation.core.die_index import CoreDieIndex, DieParamSeries


Proximity = Literal["SAFE", "BORDERLINE", "VIOLATION"]


def pattern_violates(value: float, *, direction: str, candidate: float) -> bool:
    if direction == "UPPER":
        return value > candidate
    if direction == "LOWER":
        return value < candidate
    raise ValueError(direction)


def die_fails_any_violation(series: DieParamSeries, *, direction: str, candidate: float) -> bool:
    return any(pattern_violates(v, direction=direction, candidate=candidate) for v in series.values)


def die_fails_violation_rate(
    series: DieParamSeries,
    *,
    direction: str,
    candidate: float,
    threshold: float,
) -> bool:
    if not series.values:
        return False
    n_viol = sum(1 for v in series.values if pattern_violates(v, direction=direction, candidate=candidate))
    return (n_viol / len(series.values)) >= threshold


def die_fails_consecutive(
    series: DieParamSeries,
    *,
    direction: str,
    candidate: float,
    consecutive_count: int,
) -> bool:
    run = 0
    for v in series.values:
        if pattern_violates(v, direction=direction, candidate=candidate):
            run += 1
            if run >= consecutive_count:
                return True
        else:
            run = 0
    return False


def die_fails(
    series: DieParamSeries,
    *,
    direction: str,
    candidate: float,
    policy: str,
    violation_rate_threshold: float,
    consecutive_count: int,
) -> bool:
    if policy == "ANY_VIOLATION":
        return die_fails_any_violation(series, direction=direction, candidate=candidate)
    if policy == "VIOLATION_RATE":
        return die_fails_violation_rate(
            series,
            direction=direction,
            candidate=candidate,
            threshold=violation_rate_threshold,
        )
    if policy in {"CONSECUTIVE", "CONSECUTIVE_VIOLATIONS"}:
        return die_fails_consecutive(
            series,
            direction=direction,
            candidate=candidate,
            consecutive_count=consecutive_count,
        )
    raise ValueError(f"unsupported die policy {policy!r}")


def die_proximity(
    series: DieParamSeries,
    *,
    direction: str,
    candidate: float,
    borderline_pct: float,
) -> Proximity:
    """Worst pattern proximity for the die (VIOLATION > BORDERLINE > SAFE)."""
    lim = LimitSpec(direction, candidate, "", "", "", "")
    worst: Proximity = "SAFE"
    for v in series.values:
        cls = proximity_class(v, lim, borderline_margin_percent=borderline_pct)
        if cls == "VIOLATION":
            return "VIOLATION"
        if cls == "BORDERLINE":
            worst = "BORDERLINE"
    return worst


@dataclass
class DieCandidateOutcome:
    lot_id: str
    die_id: str
    parameter: str
    candidate_limit: float
    simulated_fail: bool
    proximity: str
    source_status: str | None
    die_max: float
    die_min: float


@dataclass
class CandidateSimulationResult:
    domain: str
    parameter: str
    test_id: str
    candidate_limit: float
    current_limit: float
    direction: str
    unit: str
    source_status: str
    candidate_delta: float
    candidate_delta_percent: float | None
    tighten_or_loosen: str
    total_dies: int
    good_dies: int
    violating_dies: int
    simulated_yield: float
    simulated_fail_rate: float
    violation_rate: float
    borderline_rate: float
    risky_rate: float
    false_fail_proxy: float
    defective_proxy: float
    objective_score: float
    feasible: bool
    selection_status: str = ""
    die_policy: str = "ANY_VIOLATION"
    scope: str = "independent"  # independent | joint
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_objective(
    *,
    simulated_yield: float,
    defective_proxy: float,
    borderline_rate: float,
    false_fail_proxy: float,
    weights: ObjectiveWeights,
) -> float:
    """Agent-visible proxy objective (synthetic; configurable).

    Uses borderline_rate (limit proximity across dies), not reliability.
    Latent defective labels are never used; w_defective defaults to 0.
    """
    risk_term = weights.w_defective * defective_proxy + weights.w_risky * borderline_rate
    return (
        weights.yield_weight * simulated_yield
        - weights.lambda_risk * risk_term
        - weights.lambda_ff * false_fail_proxy
    )


def simulate_parameter_candidate(
    index: CoreDieIndex,
    candidate: CandidateLimit,
    config: CoreSimulationConfig,
) -> tuple[CandidateSimulationResult, list[DieCandidateOutcome]]:
    series_map = index.parameters()[candidate.parameter]
    outcomes: list[DieCandidateOutcome] = []
    total = 0
    viol = 0
    borderline_dies = 0
    false_fail = 0
    defective_proxy_n = 0  # source FAIL but simulated PASS
    for key in sorted(series_map):
        series = series_map[key]
        total += 1
        fails = die_fails(
            series,
            direction=candidate.direction,
            candidate=candidate.candidate_limit,
            policy=config.die_policy,
            violation_rate_threshold=config.violation_rate_threshold,
            consecutive_count=config.consecutive_count,
        )
        prox = die_proximity(
            series,
            direction=candidate.direction,
            candidate=candidate.candidate_limit,
            borderline_pct=config.borderline_margin_percent,
        )
        if fails:
            viol += 1
            if series.source_status == "PASS":
                false_fail += 1
        else:
            if prox == "BORDERLINE":
                borderline_dies += 1
            if series.source_status == "FAIL":
                defective_proxy_n += 1
        outcomes.append(
            DieCandidateOutcome(
                lot_id=series.lot_id,
                die_id=series.die_id,
                parameter=candidate.parameter,
                candidate_limit=candidate.candidate_limit,
                simulated_fail=fails,
                proximity=prox if fails else prox,
                source_status=series.source_status,
                die_max=series.die_max,
                die_min=series.die_min,
            )
        )
    good = total - viol
    sim_yield = good / total if total else 0.0
    fail_rate = viol / total if total else 0.0
    borderline_rate = borderline_dies / total if total else 0.0
    # risky := borderline among accepted dies (limit proximity, not reliability)
    accepted = good
    risky_rate = (borderline_dies / accepted) if accepted else 0.0
    ff_proxy = false_fail / total if total else 0.0
    def_proxy = defective_proxy_n / total if total else 0.0
    obj = compute_objective(
        simulated_yield=sim_yield,
        defective_proxy=def_proxy,
        borderline_rate=borderline_rate,
        false_fail_proxy=ff_proxy,
        weights=config.objective,
    )
    result = CandidateSimulationResult(
        domain="core",
        parameter=candidate.parameter,
        test_id=candidate.test_id,
        candidate_limit=candidate.candidate_limit,
        current_limit=candidate.current_limit,
        direction=candidate.direction,
        unit=candidate.unit,
        source_status=candidate.source_status,
        candidate_delta=candidate.delta_absolute,
        candidate_delta_percent=candidate.delta_percent,
        tighten_or_loosen=candidate.tighten_or_loosen,
        total_dies=total,
        good_dies=good,
        violating_dies=viol,
        simulated_yield=sim_yield,
        simulated_fail_rate=fail_rate,
        violation_rate=fail_rate,
        borderline_rate=borderline_rate,
        risky_rate=risky_rate,
        false_fail_proxy=ff_proxy,
        defective_proxy=def_proxy,
        objective_score=obj,
        feasible=True,
        die_policy=config.die_policy,
        scope="independent",
        notes=(
            "simulated_metric_yield ≠ source yield; "
            "borderline/risky are limit-proximity proxies (not reliability); "
            "defective_proxy = source_FAIL accepted (analysis only; default weight 0)"
        ),
    )
    return result, outcomes


def simulate_joint_candidate(
    index: CoreDieIndex,
    *,
    ir_candidate: CandidateLimit,
    thermal_candidate: CandidateLimit,
    config: CoreSimulationConfig,
) -> CandidateSimulationResult:
    """Joint IR OR Thermal die fail under multi_parameter_policy=OR."""
    if config.multi_parameter_policy.upper() != "OR":
        raise ValueError("Only OR joint policy is supported (matches disposition_rules)")
    keys = sorted(set(index.ir_drop) & set(index.thermal))
    total = len(keys)
    viol = 0
    borderline_dies = 0
    false_fail = 0
    defective_proxy_n = 0
    for key in keys:
        ir_s = index.ir_drop[key]
        th_s = index.thermal[key]
        ir_fail = die_fails(
            ir_s,
            direction=ir_candidate.direction,
            candidate=ir_candidate.candidate_limit,
            policy=config.die_policy,
            violation_rate_threshold=config.violation_rate_threshold,
            consecutive_count=config.consecutive_count,
        )
        th_fail = die_fails(
            th_s,
            direction=thermal_candidate.direction,
            candidate=thermal_candidate.candidate_limit,
            policy=config.die_policy,
            violation_rate_threshold=config.violation_rate_threshold,
            consecutive_count=config.consecutive_count,
        )
        fails = ir_fail or th_fail
        ir_prox = die_proximity(
            ir_s,
            direction=ir_candidate.direction,
            candidate=ir_candidate.candidate_limit,
            borderline_pct=config.borderline_margin_percent,
        )
        th_prox = die_proximity(
            th_s,
            direction=thermal_candidate.direction,
            candidate=thermal_candidate.candidate_limit,
            borderline_pct=config.borderline_margin_percent,
        )
        # die borderline if not failing but either param borderline
        is_border = (not fails) and (
            ir_prox == "BORDERLINE" or th_prox == "BORDERLINE"
        )
        if fails:
            viol += 1
            if ir_s.source_status == "PASS":
                false_fail += 1
        else:
            if is_border:
                borderline_dies += 1
            if ir_s.source_status == "FAIL":
                defective_proxy_n += 1
    good = total - viol
    sim_yield = good / total if total else 0.0
    fail_rate = viol / total if total else 0.0
    borderline_rate = borderline_dies / total if total else 0.0
    risky_rate = (borderline_dies / good) if good else 0.0
    ff_proxy = false_fail / total if total else 0.0
    def_proxy = defective_proxy_n / total if total else 0.0
    obj = compute_objective(
        simulated_yield=sim_yield,
        defective_proxy=def_proxy,
        borderline_rate=borderline_rate,
        false_fail_proxy=ff_proxy,
        weights=config.objective,
    )
    # Represent joint as composite row; scalar fields encode IR for schema stability;
    # full pair is always in notes and dedicated fields via notes parsing.
    ir_delta = ir_candidate.candidate_limit - ir_candidate.current_limit
    th_delta = thermal_candidate.candidate_limit - thermal_candidate.current_limit
    return CandidateSimulationResult(
        domain="core",
        parameter="ir_drop+thermal",
        test_id="T_IR_DROP_MV+T_THERMAL_C",
        candidate_limit=ir_candidate.candidate_limit,  # IR component (see notes for Thermal)
        current_limit=ir_candidate.current_limit,
        direction="JOINT_OR",
        unit="mV+°C",
        source_status="SOURCE_CONFIRMED",
        candidate_delta=abs(ir_delta) + abs(th_delta),  # L1 distance from current pair
        candidate_delta_percent=None,
        tighten_or_loosen="JOINT",
        total_dies=total,
        good_dies=good,
        violating_dies=viol,
        simulated_yield=sim_yield,
        simulated_fail_rate=fail_rate,
        violation_rate=fail_rate,
        borderline_rate=borderline_rate,
        risky_rate=risky_rate,
        false_fail_proxy=ff_proxy,
        defective_proxy=def_proxy,
        objective_score=obj,
        feasible=True,
        die_policy=config.die_policy,
        scope="joint",
        notes=(
            f"candidate_ir={ir_candidate.candidate_limit};"
            f"candidate_thermal={thermal_candidate.candidate_limit};"
            f"current_ir={ir_candidate.current_limit};"
            f"current_thermal={thermal_candidate.current_limit};"
            f"ir_class={classify_tighten_loosen(ir_candidate.direction, ir_candidate.current_limit, ir_candidate.candidate_limit)};"
            f"thermal_class={classify_tighten_loosen(thermal_candidate.direction, thermal_candidate.current_limit, thermal_candidate.candidate_limit)}"
        ),
    )
