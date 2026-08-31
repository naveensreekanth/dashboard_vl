"""Numerically stable distribution statistics for feature engineering."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DistStats:
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    range: float
    p1: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    iqr: float
    cv: float | None


def _percentile_sorted(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def compute_dist_stats(values: list[float]) -> DistStats | None:
    if not values:
        return None
    n = len(values)
    s = sorted(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n  # population std for stability on full groups
    std = math.sqrt(var)
    med = _percentile_sorted(s, 50)
    p1 = _percentile_sorted(s, 1)
    p5 = _percentile_sorted(s, 5)
    p25 = _percentile_sorted(s, 25)
    p50 = med
    p75 = _percentile_sorted(s, 75)
    p95 = _percentile_sorted(s, 95)
    p99 = _percentile_sorted(s, 99)
    iqr = p75 - p25
    cv = (std / abs(mean)) if abs(mean) > 1e-12 else None
    return DistStats(
        count=n,
        mean=mean,
        median=med,
        std=std,
        min=s[0],
        max=s[-1],
        range=s[-1] - s[0],
        p1=p1,
        p5=p5,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
        p99=p99,
        iqr=iqr,
        cv=cv,
    )


def stats_to_dict(prefix: str, stats: DistStats | None) -> dict[str, float | int | None]:
    if stats is None:
        keys = [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "range",
            "p1",
            "p5",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "iqr",
            "cv",
        ]
        return {f"{prefix}_{k}": None for k in keys}
    return {
        f"{prefix}_count": stats.count,
        f"{prefix}_mean": stats.mean,
        f"{prefix}_median": stats.median,
        f"{prefix}_std": stats.std,
        f"{prefix}_min": stats.min,
        f"{prefix}_max": stats.max,
        f"{prefix}_range": stats.range,
        f"{prefix}_p1": stats.p1,
        f"{prefix}_p5": stats.p5,
        f"{prefix}_p25": stats.p25,
        f"{prefix}_p50": stats.p50,
        f"{prefix}_p75": stats.p75,
        f"{prefix}_p95": stats.p95,
        f"{prefix}_p99": stats.p99,
        f"{prefix}_iqr": stats.iqr,
        f"{prefix}_cv": stats.cv,
    }
