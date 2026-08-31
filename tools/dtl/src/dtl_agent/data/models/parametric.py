"""Typed Parametric domain dataset handle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from dtl_agent.utils.csv_io import iter_csv_chunks, iter_csv_dicts


@dataclass
class ParametricDataset:
    root: Path
    version_metadata: dict[str, Any]
    lots: list[dict[str, str]]
    parts: list[dict[str, str]]
    conditions: list[dict[str, str]]
    test_catalog: list[dict[str, str]]
    current_limits: list[dict[str, str]]
    scenario_manifest: list[dict[str, str]]
    disposition_rules: dict[str, Any]
    limit_simulation_config: dict[str, Any]
    data_contract_text: str
    measurements_path: Path
    measurements_columns: list[str]
    measurements: list[dict[str, str]] | None = None
    domain: str = "parametric"
    _lot_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _die_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _lot_die_pairs: set[tuple[str, str]] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lot_ids = {r["lot_id"] for r in self.lots}
        self._die_ids = {r["die_id"] for r in self.parts}
        self._lot_die_pairs = {(r["lot_id"], r["die_id"]) for r in self.parts}

    @property
    def lot_ids(self) -> set[str]:
        return set(self._lot_ids)

    @property
    def die_ids(self) -> set[str]:
        return set(self._die_ids)

    @property
    def lot_die_pairs(self) -> set[tuple[str, str]]:
        return set(self._lot_die_pairs)

    @property
    def lot_count(self) -> int:
        return len(self._lot_ids)

    @property
    def die_count(self) -> int:
        return len(self._die_ids)

    @property
    def condition_count(self) -> int:
        return len(self.conditions)

    @property
    def dataset_version(self) -> str:
        return str(self.version_metadata.get("dataset_version", ""))

    def linked_lot_ids(self) -> set[str]:
        return {r["lot_id"] for r in self.lots if r.get("v1_link", "").lower() == "true"}

    def parametric_only_lot_ids(self) -> set[str]:
        return {r["lot_id"] for r in self.lots if r.get("v1_link", "").lower() != "true"}

    def iter_measurements(
        self,
        *,
        columns: Sequence[str] | None = None,
    ) -> Iterator[dict[str, str]]:
        if self.measurements is not None:
            for row in self.measurements:
                if columns is None:
                    yield row
                else:
                    yield {c: row.get(c, "") for c in columns}
            return
        yield from iter_csv_dicts(self.measurements_path, columns=columns)

    def iter_measurement_chunks(
        self,
        *,
        chunk_size: int = 50_000,
        columns: Sequence[str] | None = None,
    ) -> Iterator[list[dict[str, str]]]:
        if self.measurements is not None:
            rows = list(self.iter_measurements(columns=columns))
            for i in range(0, len(rows), chunk_size):
                yield rows[i : i + chunk_size]
            return
        yield from iter_csv_chunks(
            self.measurements_path, chunk_size=chunk_size, columns=columns
        )
