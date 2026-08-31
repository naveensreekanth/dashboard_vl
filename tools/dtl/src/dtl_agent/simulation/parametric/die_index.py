"""Candidate-independent die×condition summaries for Parametric simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from dtl_agent.canonical.dataset import CanonicalDataset


@dataclass
class DieConditionSeries:
    lot_id: str
    die_id: str
    condition_id: str
    source_status: str | None
    values: list[float] = field(default_factory=list)

    @property
    def value_max(self) -> float:
        return max(self.values) if self.values else float("nan")

    @property
    def value_min(self) -> float:
        return min(self.values) if self.values else float("nan")


@dataclass
class ParametricDieIndex:
    by_parameter: dict[str, dict[tuple[str, str, str], DieConditionSeries]]
    condition_meta: dict[str, dict[str, str]]
    expected_conditions: list[str]


def build_parametric_die_index(canonical: CanonicalDataset, parameters: set[str]) -> ParametricDieIndex:
    by_parameter: dict[str, dict[tuple[str, str, str], DieConditionSeries]] = {p: {} for p in parameters}
    condition_meta = {
        c.condition_id: {
            "temperature_c": c.temperature_c,
            "vdd_applied": c.vdd_applied,
            "test_mode": c.test_mode,
        }
        for c in canonical.get_conditions()
    }
    for rec in canonical.get_parametric_measurements(parameter=None):
        if rec.parameter not in parameters:
            continue
        key = (rec.lot_id, rec.die_id, rec.condition_id)
        src = rec.metadata.get("pass_fail_condition")
        if key not in by_parameter[rec.parameter]:
            by_parameter[rec.parameter][key] = DieConditionSeries(
                lot_id=rec.lot_id,
                die_id=rec.die_id,
                condition_id=rec.condition_id,
                source_status=src,
                values=[],
            )
        row = by_parameter[rec.parameter][key]
        row.values.append(rec.value)
        if row.source_status is None:
            row.source_status = src
    return ParametricDieIndex(
        by_parameter=by_parameter,
        condition_meta=condition_meta,
        expected_conditions=sorted(condition_meta),
    )
