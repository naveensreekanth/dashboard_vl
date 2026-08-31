"""Production-facing read-only DTL parametric test-time cost-savings estimator.

Consumes existing Phase 12.9 / analysis recommendation outputs only.
Does not call recommend(), modify limits/ranking/policy/safety/simulation,
or claim measured ATE savings.

Mechanism (M2 — adaptive parametric condition pruning counterfactual):
  baseline: all 4 parametric conditions execute
  DTL:      COND_RT_NOM always; remaining 3 skipped iff margin vs
            recommended_limit >= configured skip_threshold

Core parameters contribute 0 estimated seconds saved.
KEEP_CURRENT / REJECT contribute 0 estimated seconds saved.
Assumptions (condition_duration_s, skip_threshold, tester_cost_per_hour)
are configuration inputs — not measured dataset values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import json

import pandas as pd

from dtl_agent.config.constants import (
    CORE_EXPECTED_LIMITS,
    EXPECTED_CONDITION_IDS,
    PARAMETRIC_EXPECTED_LIMITS,
)
from dtl_agent.data.temporal.paths import parametric_root, temporal_artifact_root
from dtl_agent.recommendation.schemas import CORE_PARAMETERS, PARAMETRIC_PARAMETERS

FIRST_CONDITION_ID = "COND_RT_NOM"
N_BASELINE_CONDITIONS = len(EXPECTED_CONDITION_IDS)  # 4

_DIRECTION_BY_PARAMETER: dict[str, str] = {
    lim.parameter: lim.direction for lim in (*CORE_EXPECTED_LIMITS, *PARAMETRIC_EXPECTED_LIMITS)
}


@dataclass(frozen=True)
class CostSavingsAssumptions:
    """Configurable engineering assumptions — not measured from the dataset."""

    condition_duration_s: float = 0.05
    skip_threshold: float = 0.10
    tester_cost_per_hour: float = 25.0
    n_baseline_conditions: int = N_BASELINE_CONDITIONS
    first_condition_id: str = FIRST_CONDITION_ID

    def validate(self) -> None:
        if self.condition_duration_s < 0:
            raise ValueError("condition_duration_s must be >= 0")
        if self.skip_threshold < 0:
            raise ValueError("skip_threshold must be >= 0")
        if self.tester_cost_per_hour < 0:
            raise ValueError("tester_cost_per_hour must be >= 0")
        if self.n_baseline_conditions < 1:
            raise ValueError("n_baseline_conditions must be >= 1")


def parameter_direction(parameter: str) -> str | None:
    """Return UPPER/LOWER from repository catalog, or None if unknown."""
    return _DIRECTION_BY_PARAMETER.get(str(parameter))


def is_parametric_parameter(parameter: str) -> bool:
    return str(parameter) in PARAMETRIC_PARAMETERS


def is_core_parameter(parameter: str) -> bool:
    return str(parameter) in CORE_PARAMETERS


def compute_margin(
    *,
    direction: str,
    recommended_limit: float,
    measured_value: float,
) -> float:
    """Direction-specific margin vs recommended_limit.

    UPPER: margin = recommended_limit - measured_value
    LOWER: margin = measured_value - recommended_limit
    """
    d = str(direction).upper()
    if d == "UPPER":
        return float(recommended_limit) - float(measured_value)
    if d == "LOWER":
        return float(measured_value) - float(recommended_limit)
    raise ValueError(f"unsupported direction {direction!r}")


def skip_remaining_conditions(
    *,
    decision: str,
    parameter: str,
    direction: str | None,
    recommended_limit: float | None,
    measured_value: float | None,
    skip_threshold: float,
) -> bool:
    """True when counterfactual M2 would skip the remaining 3 conditions."""
    if str(decision) != "RECOMMEND":
        return False
    if not is_parametric_parameter(parameter):
        return False
    if direction is None or measured_value is None or recommended_limit is None:
        return False
    margin = compute_margin(
        direction=direction,
        recommended_limit=float(recommended_limit),
        measured_value=float(measured_value),
    )
    return margin >= float(skip_threshold)


def record_times(
    *,
    skip: bool,
    condition_duration_s: float,
    n_baseline_conditions: int = N_BASELINE_CONDITIONS,
) -> tuple[float, float, float]:
    """Return (baseline_time_s, dtl_time_s, estimated_seconds_saved)."""
    baseline = float(n_baseline_conditions) * float(condition_duration_s)
    dtl = float(condition_duration_s) if skip else baseline
    saved = baseline - dtl
    return baseline, dtl, saved


def seconds_to_cost(seconds_saved: float, tester_cost_per_hour: float) -> float:
    return (float(seconds_saved) / 3600.0) * float(tester_cost_per_hour)


@lru_cache(maxsize=8)
def _load_cond_rt_nom_index(project_root_str: str, production_month: str) -> dict[tuple[str, str, str], float]:
    """Index (lot_id, die_id, parameter) → COND_RT_NOM measurement_value."""
    path = parametric_root(production_month, Path(project_root_str)) / "measurements.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(
        path,
        usecols=["lot_id", "die_id", "parameter", "condition_id", "measurement_value"],
    )
    sub = df[df["condition_id"].astype(str) == FIRST_CONDITION_ID]
    out: dict[tuple[str, str, str], float] = {}
    for row in sub.itertuples(index=False):
        key = (str(row.lot_id), str(row.die_id), str(row.parameter))
        out[key] = float(row.measurement_value)
    return out


def clear_cost_savings_caches() -> None:
    _load_cond_rt_nom_index.cache_clear()


def _estimate_one(
    row: dict[str, Any],
    *,
    nom_measurement: float | None,
    assumptions: CostSavingsAssumptions,
) -> dict[str, Any]:
    parameter = str(row.get("parameter") or "")
    decision = str(row.get("decision") or "")
    direction = parameter_direction(parameter)
    current_limit = row.get("current_limit")
    recommended_limit = row.get("recommended_limit")

    current_f = float(current_limit) if current_limit is not None else None
    recommended_f = float(recommended_limit) if recommended_limit is not None else None

    # Core / non-RECOMMEND / missing measurement → no parametric skip savings
    skip = skip_remaining_conditions(
        decision=decision,
        parameter=parameter,
        direction=direction,
        recommended_limit=recommended_f,
        measured_value=nom_measurement,
        skip_threshold=assumptions.skip_threshold,
    )

    # Core parameters: M2 does not apply — report zero parametric time impact
    if is_core_parameter(parameter) or not is_parametric_parameter(parameter):
        baseline_s = 0.0
        dtl_s = 0.0
        saved_s = 0.0
        skip = False
    else:
        baseline_s, dtl_s, saved_s = record_times(
            skip=skip,
            condition_duration_s=assumptions.condition_duration_s,
            n_baseline_conditions=assumptions.n_baseline_conditions,
        )
        if decision != "RECOMMEND":
            # KEEP_CURRENT / REJECT: no counterfactual adoption of a new DTL
            skip = False
            dtl_s = baseline_s
            saved_s = 0.0

    margin: float | None = None
    if (
        is_parametric_parameter(parameter)
        and direction is not None
        and recommended_f is not None
        and nom_measurement is not None
    ):
        margin = compute_margin(
            direction=direction,
            recommended_limit=recommended_f,
            measured_value=float(nom_measurement),
        )

    cost = seconds_to_cost(saved_s, assumptions.tester_cost_per_hour)

    return {
        "die_id": row.get("die_id"),
        "lot_id": row.get("lot_id"),
        "production_month": row.get("production_month"),
        "parameter": parameter,
        "parameter_display": row.get("parameter_display") or parameter,
        "decision": decision,
        "current_limit": current_f,
        "recommended_limit": recommended_f,
        "nom_measurement": nom_measurement,
        "direction": direction,
        "margin": margin,
        "skip_remaining_conditions": bool(skip),
        "baseline_test_time_s": baseline_s,
        "predicted_dtl_test_time_s": dtl_s,
        "estimated_seconds_saved": saved_s,
        "predicted_cost_saving": cost,
        "missing_nom_measurement": is_parametric_parameter(parameter) and nom_measurement is None,
    }


def _load_recommendation_rows(project_root: Path) -> list[dict[str, Any]]:
    """Read Phase 12.9 recommendation artifacts without calling recommend()."""
    path = (
        temporal_artifact_root(project_root)
        / "shared"
        / "phase_12_9_analysis"
        / "three_month_recommendations.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing analysis recommendations artifact: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data.get("all_dies_rows") or [])
    if not rows:
        rows = list(data.get("rows") or [])
    return rows


def estimate_cost_savings(
    project_root: Path | str,
    *,
    assumptions: CostSavingsAssumptions | None = None,
    include_per_device: bool = True,
    recommendation_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build counterfactual cost-savings report from Phase 12.9 analysis artifacts.

    Does not call recommend() or alter recommendation fields.
    """
    root = Path(project_root)
    assumptions = assumptions or CostSavingsAssumptions()
    assumptions.validate()

    rows = recommendation_rows if recommendation_rows is not None else _load_recommendation_rows(root)

    # Preserve recommendation fields exactly (copy only for estimator output)
    per_device: list[dict[str, Any]] = []
    for row in rows:
        month = str(row.get("production_month") or "")
        lot_id = str(row.get("lot_id") or "")
        die_id = str(row.get("die_id") or "")
        parameter = str(row.get("parameter") or "")
        nom: float | None = None
        if is_parametric_parameter(parameter) and month:
            idx = _load_cond_rt_nom_index(str(root.resolve()), month)
            nom = idx.get((lot_id, die_id, parameter))
        per_device.append(_estimate_one(row, nom_measurement=nom, assumptions=assumptions))

    records_evaluated = len(per_device)
    eligible = [
        r
        for r in per_device
        if is_parametric_parameter(str(r["parameter"])) and str(r["decision"]) == "RECOMMEND"
    ]
    with_skip = [r for r in per_device if r["skip_remaining_conditions"]]
    zero_savings = [r for r in per_device if float(r["estimated_seconds_saved"]) <= 0]

    total_baseline = sum(float(r["baseline_test_time_s"]) for r in per_device)
    total_dtl = sum(float(r["predicted_dtl_test_time_s"]) for r in per_device)
    total_saved = sum(float(r["estimated_seconds_saved"]) for r in per_device)
    total_cost = sum(float(r["predicted_cost_saving"]) for r in per_device)

    per_record_saved = (total_saved / records_evaluated) if records_evaluated else 0.0
    per_eligible_saved = (total_saved / len(eligible)) if eligible else 0.0
    cost_per_record = (total_cost / records_evaluated) if records_evaluated else 0.0
    cost_per_1000 = cost_per_record * 1000.0
    tester_hours_saved = total_saved / 3600.0
    pct = ((total_saved / total_baseline) * 100.0) if total_baseline > 0 else 0.0

    payload: dict[str, Any] = {
        "status": "predicted",
        "is_measured_ate_saving": False,
        "label": "Predicted DTL Test-Time Cost Saving",
        "disclaimer": "Counterfactual estimate — not measured ATE savings.",
        "estimator": {
            "type": "counterfactual",
            "production_facing": True,
            "read_only": True,
            "mechanism": "M2_adaptive_parametric_condition_pruning",
            "label": "Predicted Cost Saving",
            "condition_duration_s": assumptions.condition_duration_s,
            "skip_threshold": assumptions.skip_threshold,
            "tester_cost_per_hour": assumptions.tester_cost_per_hour,
            "n_baseline_conditions": assumptions.n_baseline_conditions,
            "first_condition_id": assumptions.first_condition_id,
            "cost_source": "configured assumption",
            "duration_source": "configured assumption",
            "skip_threshold_source": "configured assumption",
            "formulas": {
                "UPPER_margin": "recommended_limit - nom_measurement",
                "LOWER_margin": "nom_measurement - recommended_limit",
                "skip_when": "decision==RECOMMEND and parametric and margin >= skip_threshold",
                "baseline_time_s": "n_baseline_conditions * condition_duration_s",
                "dtl_time_s": "condition_duration_s if skip else baseline_time_s",
                "estimated_seconds_saved": "baseline_time_s - dtl_time_s",
                "predicted_cost_saving": "(estimated_seconds_saved / 3600) * tester_cost_per_hour",
            },
            "assumptions": asdict(assumptions),
        },
        "aggregate": {
            "records_evaluated": records_evaluated,
            "eligible_records": len(eligible),
            "records_with_predicted_skip": len(with_skip),
            "records_with_zero_savings": len(zero_savings),
            "total_baseline_test_time_s": total_baseline,
            "total_dtl_test_time_s": total_dtl,
            "total_estimated_seconds_saved": total_saved,
            "estimated_seconds_saved_per_record": per_record_saved,
            "estimated_seconds_saved_per_eligible_record": per_eligible_saved,
            "predicted_time_saved_pct": pct,
            "tester_hours_saved": tester_hours_saved,
            "tester_cost_per_hour": assumptions.tester_cost_per_hour,
            "total_predicted_cost_saving": total_cost,
            "predicted_cost_saved_per_record": cost_per_record,
            "predicted_cost_saved_per_1000_records": cost_per_1000,
            "production_volume_supplied": False,
            "note": (
                "records_evaluated is the count of die×parameter×month recommendation "
                "rows scored; production volume was not fabricated."
            ),
        },
        "source": {
            "recommendations": "artifacts/temporal/shared/phase_12_9_analysis",
            "nom_measurements": "data/3 months data/{month}/parametric/measurements.csv",
            "recommendation_fields_unmodified": True,
        },
    }
    if include_per_device:
        payload["per_device"] = per_device
    return payload
