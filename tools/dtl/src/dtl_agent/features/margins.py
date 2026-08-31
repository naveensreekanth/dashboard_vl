"""Limit margin and proximity helpers (current-limit only; not reliability)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LimitSpec:
    direction: str  # UPPER | LOWER
    value: float
    unit: str
    source_status: str
    test_id: str
    parameter: str


def signed_margin(measurement: float, limit: LimitSpec) -> float:
    """Positive => on passing side of current limit."""
    if limit.direction == "UPPER":
        return limit.value - measurement
    if limit.direction == "LOWER":
        return measurement - limit.value
    raise ValueError(f"unsupported direction {limit.direction!r}")


def is_violation(measurement: float, limit: LimitSpec) -> bool:
    if limit.direction == "UPPER":
        return measurement > limit.value
    if limit.direction == "LOWER":
        return measurement < limit.value
    raise ValueError(f"unsupported direction {limit.direction!r}")


def proximity_class(
    measurement: float,
    limit: LimitSpec,
    *,
    borderline_margin_percent: float,
) -> str:
    """SAFE / BORDERLINE / VIOLATION using PERCENT_OF_LIMIT guard band (not reliability)."""
    if is_violation(measurement, limit):
        return "VIOLATION"
    pct = borderline_margin_percent / 100.0
    if limit.direction == "UPPER":
        band = limit.value * (1.0 - pct)
        if band < measurement <= limit.value:
            return "BORDERLINE"
        return "SAFE"
    if limit.direction == "LOWER":
        band = limit.value * (1.0 + pct)
        if limit.value <= measurement < band:
            return "BORDERLINE"
        return "SAFE"
    raise ValueError(f"unsupported direction {limit.direction!r}")


def normalized_margin(measurement: float, limit: LimitSpec) -> float | None:
    if abs(limit.value) < 1e-12:
        return None
    return signed_margin(measurement, limit) / abs(limit.value)


@dataclass
class MarginAggregate:
    count: int = 0
    violation_count: int = 0
    borderline_count: int = 0
    safe_count: int = 0
    margin_sum: float = 0.0
    margin_min: float = float("inf")
    margin_max: float = float("-inf")
    margins: list[float] | None = None

    def __post_init__(self) -> None:
        if self.margins is None:
            self.margins = []

    def add(self, measurement: float, limit: LimitSpec, *, borderline_pct: float) -> None:
        m = signed_margin(measurement, limit)
        cls = proximity_class(measurement, limit, borderline_margin_percent=borderline_pct)
        assert self.margins is not None
        self.count += 1
        self.margin_sum += m
        self.margin_min = min(self.margin_min, m)
        self.margin_max = max(self.margin_max, m)
        self.margins.append(m)
        if cls == "VIOLATION":
            self.violation_count += 1
        elif cls == "BORDERLINE":
            self.borderline_count += 1
        else:
            self.safe_count += 1

    def to_dict(self, prefix: str) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                f"{prefix}_margin_count": 0,
                f"{prefix}_margin_mean": None,
                f"{prefix}_margin_min": None,
                f"{prefix}_margin_max": None,
                f"{prefix}_violation_count": 0,
                f"{prefix}_violation_rate": None,
                f"{prefix}_borderline_count": 0,
                f"{prefix}_borderline_rate": None,
                f"{prefix}_safe_count": 0,
                f"{prefix}_safe_rate": None,
                f"{prefix}_fraction_within_guard_band": None,
            }
        assert self.margins is not None
        return {
            f"{prefix}_margin_count": self.count,
            f"{prefix}_margin_mean": self.margin_sum / self.count,
            f"{prefix}_margin_min": self.margin_min,
            f"{prefix}_margin_max": self.margin_max,
            f"{prefix}_violation_count": self.violation_count,
            f"{prefix}_violation_rate": self.violation_count / self.count,
            f"{prefix}_borderline_count": self.borderline_count,
            f"{prefix}_borderline_rate": self.borderline_count / self.count,
            f"{prefix}_safe_count": self.safe_count,
            f"{prefix}_safe_rate": self.safe_count / self.count,
            f"{prefix}_fraction_within_guard_band": self.borderline_count / self.count,
        }
