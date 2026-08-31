"""Compact per-die Core summaries for candidate simulation (IR / Thermal only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from dtl_agent.canonical.dataset import CanonicalDataset


@dataclass
class DieParamSeries:
    lot_id: str
    die_id: str
    source_status: str | None
    # pattern-ordered values (length 200 when complete)
    values: list[float] = field(default_factory=list)

    @property
    def die_max(self) -> float:
        return max(self.values) if self.values else float("nan")

    @property
    def die_min(self) -> float:
        return min(self.values) if self.values else float("nan")


@dataclass
class CoreDieIndex:
    """Candidate-independent die series for IR and Thermal."""

    ir_drop: dict[tuple[str, str], DieParamSeries]
    thermal: dict[tuple[str, str], DieParamSeries]

    def parameters(self) -> dict[str, dict[tuple[str, str], DieParamSeries]]:
        return {"ir_drop": self.ir_drop, "thermal": self.thermal}

    @property
    def die_ids(self) -> list[tuple[str, str]]:
        return sorted(set(self.ir_drop) | set(self.thermal))


def build_core_die_index(canonical: CanonicalDataset) -> CoreDieIndex:
    """One streaming pass: build ordered pattern value series for limit-eligible params."""
    status = {
        (d.lot_id, d.die_id): d.core_metadata.get("status")
        for d in canonical.dies.values()
        if d.in_core
    }
    # die -> pattern_id -> param -> value
    buckets: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for rec in canonical.get_core_measurements(parameter=None):
        if rec.parameter not in {"ir_drop", "thermal"}:
            continue
        key = (rec.lot_id, rec.die_id)
        pid = int(rec.pattern_id)
        buckets.setdefault(key, {}).setdefault(pid, {})[rec.parameter] = rec.value

    ir: dict[tuple[str, str], DieParamSeries] = {}
    th: dict[tuple[str, str], DieParamSeries] = {}
    for key, patterns in buckets.items():
        pids = sorted(patterns)
        ir_vals = [patterns[p]["ir_drop"] for p in pids if "ir_drop" in patterns[p]]
        th_vals = [patterns[p]["thermal"] for p in pids if "thermal" in patterns[p]]
        st = status.get(key)
        if ir_vals:
            ir[key] = DieParamSeries(key[0], key[1], st, ir_vals)
        if th_vals:
            th[key] = DieParamSeries(key[0], key[1], st, th_vals)
    return CoreDieIndex(ir_drop=ir, thermal=th)


def iter_series(index: CoreDieIndex, parameter: str) -> Iterator[DieParamSeries]:
    for key in sorted(index.parameters()[parameter]):
        yield index.parameters()[parameter][key]
