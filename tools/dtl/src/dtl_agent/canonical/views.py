"""Lazy measurement views that preserve domain grain (no concatenation)."""

from __future__ import annotations

from collections.abc import Iterator

from dtl_agent.canonical.entities import (
    CoreMeasurementRecord,
    ParametricMeasurementRecord,
    row_get,
)
from dtl_agent.data.models.core import CoreDataset
from dtl_agent.data.models.parametric import ParametricDataset

# Columns projected for filtering / record construction (avoid full-row copies when streaming)
_CORE_PROJ = (
    "lot_id",
    "die_id",
    "pattern_id",
    "test_id",
    "test_name",
    "parameter",
    "measurement_value",
    "unit",
    "scenario_id",
    "scenario_family",
    "tester_id",
    "site_id",
    "pass_fail_pattern",
    "die_status",
    "production_sequence",
)

_PARAM_PROJ = (
    "lot_id",
    "die_id",
    "condition_id",
    "test_id",
    "parameter",
    "measurement_value",
    "unit",
    "scenario_id",
    "scenario_family",
    "tester_id",
    "site_id",
    "temperature_c",
    "vdd_applied",
    "test_mode",
    "limit_type",
    "pass_fail_condition",
    "dataset_version",
)


def _core_record(row: dict[str, str]) -> CoreMeasurementRecord:
    meta = {k: row_get(row, k) for k in _CORE_PROJ if k not in {
        "lot_id", "die_id", "pattern_id", "test_id", "parameter", "measurement_value", "unit"
    }}
    return CoreMeasurementRecord(
        lot_id=row_get(row, "lot_id"),
        die_id=row_get(row, "die_id"),
        pattern_id=row_get(row, "pattern_id"),
        test_id=row_get(row, "test_id"),
        parameter=row_get(row, "parameter"),
        value=float(row_get(row, "measurement_value")),
        unit=row_get(row, "unit"),
        metadata=meta,
    )


def _parametric_record(row: dict[str, str]) -> ParametricMeasurementRecord:
    meta = {k: row_get(row, k) for k in _PARAM_PROJ if k not in {
        "lot_id", "die_id", "condition_id", "test_id", "parameter", "measurement_value", "unit"
    }}
    return ParametricMeasurementRecord(
        lot_id=row_get(row, "lot_id"),
        die_id=row_get(row, "die_id"),
        condition_id=row_get(row, "condition_id"),
        test_id=row_get(row, "test_id"),
        parameter=row_get(row, "parameter"),
        value=float(row_get(row, "measurement_value")),
        unit=row_get(row, "unit"),
        metadata=meta,
    )


class CoreMeasurementView:
    """Streaming Core measurements at grain lot × die × pattern × test."""

    def __init__(self, dataset: CoreDataset) -> None:
        self._dataset = dataset

    @property
    def grain(self) -> str:
        return "lot × die × pattern × test"

    def iter(
        self,
        *,
        lot_id: str | None = None,
        die_id: str | None = None,
        pattern_id: str | None = None,
        test_id: str | None = None,
        parameter: str | None = None,
    ) -> Iterator[CoreMeasurementRecord]:
        for row in self._dataset.iter_measurements(columns=_CORE_PROJ):
            if lot_id is not None and row_get(row, "lot_id") != lot_id:
                continue
            if die_id is not None and row_get(row, "die_id") != die_id:
                continue
            if pattern_id is not None and row_get(row, "pattern_id") != pattern_id:
                continue
            if test_id is not None and row_get(row, "test_id") != test_id:
                continue
            if parameter is not None and row_get(row, "parameter") != parameter:
                continue
            yield _core_record(row)

    def count(
        self,
        *,
        lot_id: str | None = None,
        die_id: str | None = None,
        pattern_id: str | None = None,
        test_id: str | None = None,
        parameter: str | None = None,
    ) -> int:
        return sum(
            1
            for _ in self.iter(
                lot_id=lot_id,
                die_id=die_id,
                pattern_id=pattern_id,
                test_id=test_id,
                parameter=parameter,
            )
        )


class ParametricMeasurementView:
    """Streaming Parametric measurements at grain lot × die × condition × test."""

    def __init__(self, dataset: ParametricDataset) -> None:
        self._dataset = dataset

    @property
    def grain(self) -> str:
        return "lot × die × condition × test"

    def iter(
        self,
        *,
        lot_id: str | None = None,
        die_id: str | None = None,
        condition_id: str | None = None,
        test_id: str | None = None,
        parameter: str | None = None,
    ) -> Iterator[ParametricMeasurementRecord]:
        for row in self._dataset.iter_measurements(columns=_PARAM_PROJ):
            if lot_id is not None and row_get(row, "lot_id") != lot_id:
                continue
            if die_id is not None and row_get(row, "die_id") != die_id:
                continue
            if condition_id is not None and row_get(row, "condition_id") != condition_id:
                continue
            if test_id is not None and row_get(row, "test_id") != test_id:
                continue
            if parameter is not None and row_get(row, "parameter") != parameter:
                continue
            yield _parametric_record(row)

    def count(
        self,
        *,
        lot_id: str | None = None,
        die_id: str | None = None,
        condition_id: str | None = None,
        test_id: str | None = None,
        parameter: str | None = None,
    ) -> int:
        return sum(
            1
            for _ in self.iter(
                lot_id=lot_id,
                die_id=die_id,
                condition_id=condition_id,
                test_id=test_id,
                parameter=parameter,
            )
        )


def assert_not_concatenated(
    core_view: CoreMeasurementView,
    parametric_view: ParametricMeasurementView,
) -> bool:
    """Structural guard: views remain separate types/grains (never a single merged table)."""
    return (
        type(core_view) is not type(parametric_view)
        and core_view.grain != parametric_view.grain
        and callable(getattr(core_view, "iter", None))
        and callable(getattr(parametric_view, "iter", None))
    )
