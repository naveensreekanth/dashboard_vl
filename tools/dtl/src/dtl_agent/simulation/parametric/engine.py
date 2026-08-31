"""Parametric candidate simulation engine (condition-aware)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dtl_agent.features.margins import LimitSpec, proximity_class
from dtl_agent.simulation.parametric.candidates import CandidateLimit
from dtl_agent.simulation.parametric.config import ObjectiveWeights, ParametricSimulationConfig
from dtl_agent.simulation.parametric.die_index import DieConditionSeries, ParametricDieIndex


def measurement_violates(value: float, *, direction: str, candidate: float) -> bool:
    if direction == "UPPER":
        return value > candidate
    if direction == "LOWER":
        return value < candidate
    raise ValueError(direction)


def condition_fails_any_violation(series: DieConditionSeries, *, direction: str, candidate: float) -> bool:
    return any(measurement_violates(v, direction=direction, candidate=candidate) for v in series.values)


def condition_proximity(
    series: DieConditionSeries,
    *,
    direction: str,
    candidate: float,
    borderline_pct: float,
) -> str:
    lim = LimitSpec(direction, candidate, "", "", "", "")
    worst = "SAFE"
    for v in series.values:
        cls = proximity_class(v, lim, borderline_margin_percent=borderline_pct)
        if cls == "VIOLATION":
            return "VIOLATION"
        if cls == "BORDERLINE":
            worst = "BORDERLINE"
    return worst


@dataclass
class DieConditionOutcome:
    lot_id: str
    die_id: str
    condition_id: str
    parameter: str
    candidate_limit: float
    current_limit: float
    direction: str
    violation: bool
    borderline: bool
    proximity: str
    source_status: str | None
    value_max: float
    value_min: float
    temperature_c: str | None
    vdd_applied: str | None
    test_mode: str | None


@dataclass
class ParametricCandidateResult:
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
    worst_condition_yield: float
    worst_condition_fail_rate: float
    worst_condition_violation_rate: float
    false_fail_proxy: float
    defective_proxy: float
    objective_score: float
    feasible: bool
    missing_condition_dies: int
    evaluated_conditions: int
    selection_status: str = ""
    die_policy: str = "ANY_VIOLATION"
    scope: str = "independent"
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
    risk_term = weights.w_defective * defective_proxy + weights.w_risky * borderline_rate
    return (
        weights.yield_weight * simulated_yield
        - weights.lambda_risk * risk_term
        - weights.lambda_ff * false_fail_proxy
    )


def simulate_parameter_candidate(
    index: ParametricDieIndex,
    candidate: CandidateLimit,
    config: ParametricSimulationConfig,
) -> tuple[ParametricCandidateResult, list[DieConditionOutcome], list[dict[str, Any]]]:
    series_map = index.by_parameter[candidate.parameter]
    cond_stats: dict[str, dict[str, int]] = {
        c: {"total": 0, "viol": 0, "border": 0} for c in index.expected_conditions
    }
    by_die: dict[tuple[str, str], dict[str, DieConditionOutcome]] = {}
    outcomes: list[DieConditionOutcome] = []

    for key in sorted(series_map):
        series = series_map[key]
        fails = condition_fails_any_violation(
            series, direction=candidate.direction, candidate=candidate.candidate_limit
        )
        prox = condition_proximity(
            series,
            direction=candidate.direction,
            candidate=candidate.candidate_limit,
            borderline_pct=config.borderline_margin_percent,
        )
        cond_stats[series.condition_id]["total"] += 1
        if fails:
            cond_stats[series.condition_id]["viol"] += 1
        if (not fails) and prox == "BORDERLINE":
            cond_stats[series.condition_id]["border"] += 1
        meta = index.condition_meta.get(series.condition_id, {})
        out = DieConditionOutcome(
            lot_id=series.lot_id,
            die_id=series.die_id,
            condition_id=series.condition_id,
            parameter=candidate.parameter,
            candidate_limit=candidate.candidate_limit,
            current_limit=candidate.current_limit,
            direction=candidate.direction,
            violation=fails,
            borderline=((not fails) and prox == "BORDERLINE"),
            proximity=prox,
            source_status=series.source_status,
            value_max=series.value_max,
            value_min=series.value_min,
            temperature_c=meta.get("temperature_c"),
            vdd_applied=meta.get("vdd_applied"),
            test_mode=meta.get("test_mode"),
        )
        by_die.setdefault((series.lot_id, series.die_id), {})[series.condition_id] = out
        outcomes.append(out)

    total_dies = len(by_die)
    missing_condition_dies = 0
    violating_dies = 0
    borderline_dies = 0
    false_fail = 0
    defective_proxy_n = 0
    per_condition_rows: list[dict[str, Any]] = []

    for cond in index.expected_conditions:
        st = cond_stats.get(cond, {"total": 0, "viol": 0, "border": 0})
        total = st["total"]
        viol = st["viol"]
        good = total - viol
        per_condition_rows.append(
            {
                "domain": "parametric",
                "parameter": candidate.parameter,
                "condition_id": cond,
                "candidate_limit": candidate.candidate_limit,
                "current_limit": candidate.current_limit,
                "direction": candidate.direction,
                "unit": candidate.unit,
                "total_dies": total,
                "good_dies": good,
                "violating_dies": viol,
                "simulated_yield": (good / total) if total else 0.0,
                "simulated_fail_rate": (viol / total) if total else 0.0,
                "violation_rate": (viol / total) if total else 0.0,
                "borderline_rate": (st["border"] / total) if total else 0.0,
                "temperature_c": index.condition_meta.get(cond, {}).get("temperature_c"),
                "vdd_applied": index.condition_meta.get(cond, {}).get("vdd_applied"),
                "test_mode": index.condition_meta.get(cond, {}).get("test_mode"),
            }
        )

    for _, cond_map in by_die.items():
        has_all = all(c in cond_map for c in index.expected_conditions)
        if not has_all:
            missing_condition_dies += 1
            violating_dies += 1
            continue
        fails_any = any(cond_map[c].violation for c in index.expected_conditions)
        if fails_any:
            violating_dies += 1
            # conservative: count false-fail only if all source tags are pass-like
            if all((cond_map[c].source_status or "").upper().startswith("P") for c in index.expected_conditions):
                false_fail += 1
        else:
            if any(cond_map[c].borderline for c in index.expected_conditions):
                borderline_dies += 1
            if any((cond_map[c].source_status or "").upper().startswith("F") for c in index.expected_conditions):
                defective_proxy_n += 1

    good_dies = total_dies - violating_dies
    sim_yield = (good_dies / total_dies) if total_dies else 0.0
    fail_rate = (violating_dies / total_dies) if total_dies else 0.0
    borderline_rate = (borderline_dies / total_dies) if total_dies else 0.0
    risky_rate = (borderline_dies / good_dies) if good_dies else 0.0
    ff_proxy = (false_fail / total_dies) if total_dies else 0.0
    def_proxy = (defective_proxy_n / total_dies) if total_dies else 0.0
    obj = compute_objective(
        simulated_yield=sim_yield,
        defective_proxy=def_proxy,
        borderline_rate=borderline_rate,
        false_fail_proxy=ff_proxy,
        weights=config.objective,
    )
    cond_yields = [r["simulated_yield"] for r in per_condition_rows if r["total_dies"] > 0]
    cond_fails = [r["simulated_fail_rate"] for r in per_condition_rows if r["total_dies"] > 0]
    cond_viols = [r["violation_rate"] for r in per_condition_rows if r["total_dies"] > 0]
    result = ParametricCandidateResult(
        domain="parametric",
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
        total_dies=total_dies,
        good_dies=good_dies,
        violating_dies=violating_dies,
        simulated_yield=sim_yield,
        simulated_fail_rate=fail_rate,
        violation_rate=fail_rate,
        borderline_rate=borderline_rate,
        risky_rate=risky_rate,
        worst_condition_yield=min(cond_yields) if cond_yields else 0.0,
        worst_condition_fail_rate=max(cond_fails) if cond_fails else 0.0,
        worst_condition_violation_rate=max(cond_viols) if cond_viols else 0.0,
        false_fail_proxy=ff_proxy,
        defective_proxy=def_proxy,
        objective_score=obj,
        feasible=True,
        missing_condition_dies=missing_condition_dies,
        evaluated_conditions=len(index.expected_conditions),
        die_policy=config.die_policy,
        notes=(
            "SYNTHETIC_ASSUMED limits; borderline/risk are proximity indicators (not reliability); "
            "source labels are observational proxies only."
        ),
    )
    return result, outcomes, per_condition_rows
