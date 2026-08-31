"""Parametric candidate generation and direction-aware classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dtl_agent.features.margins import LimitSpec


@dataclass(frozen=True)
class CandidateLimit:
    parameter: str
    test_id: str
    direction: str
    unit: str
    source_status: str
    current_limit: float
    candidate_limit: float

    @property
    def delta_absolute(self) -> float:
        return self.candidate_limit - self.current_limit

    @property
    def delta_percent(self) -> float | None:
        if abs(self.current_limit) < 1e-12:
            return None
        return 100.0 * self.delta_absolute / abs(self.current_limit)

    @property
    def tighten_or_loosen(self) -> str:
        return classify_tighten_loosen(self.direction, self.current_limit, self.candidate_limit)


def classify_tighten_loosen(direction: str, current: float, candidate: float) -> str:
    if abs(candidate - current) < 1e-12:
        return "CURRENT"
    if direction == "UPPER":
        return "TIGHTER" if candidate < current else "LOOSER"
    if direction == "LOWER":
        return "TIGHTER" if candidate > current else "LOOSER"
    raise ValueError(f"unsupported direction {direction!r}")


def generate_candidates(*, limit: LimitSpec, grid: Iterable[float]) -> list[CandidateLimit]:
    values = sorted({float(v) for v in grid})
    if float(limit.value) not in values:
        values.append(float(limit.value))
        values = sorted(values)
    return [
        CandidateLimit(
            parameter=limit.parameter,
            test_id=limit.test_id,
            direction=limit.direction,
            unit=limit.unit,
            source_status=limit.source_status,
            current_limit=float(limit.value),
            candidate_limit=v,
        )
        for v in values
    ]
