"""Machine-readable feature registry for Phase 3 → Phase 6 handoff."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FeatureSpec:
    feature_name: str
    domain: str  # core | parametric | cross_domain | sequence
    grain: str
    source_parameters: list[str]
    formula: str
    unit: str | None
    direction: str | None
    allowed_for_ml: bool
    candidate_dependent: bool
    evaluation_only: bool
    normalization_required: bool
    description: str
    leakage_status: str = "clean"  # clean | forbidden

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureRegistry:
    features: list[FeatureSpec] = field(default_factory=list)

    def add(self, spec: FeatureSpec) -> None:
        if spec.evaluation_only or spec.leakage_status == "forbidden":
            raise ValueError(f"refusing to register leaking feature: {spec.feature_name}")
        self.features.append(spec)

    def extend(self, specs: list[FeatureSpec]) -> None:
        for s in specs:
            self.add(s)

    def names(self) -> list[str]:
        return [f.feature_name for f in self.features]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_count": len(self.features),
            "features": [f.to_dict() for f in self.features],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def dist_feature_specs(
    *,
    prefix: str,
    domain: str,
    grain: str,
    parameter: str,
    unit: str | None,
) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for suffix, formula, desc in [
        ("count", "len(values)", "observation count"),
        ("mean", "mean(values)", "arithmetic mean"),
        ("median", "median(values)", "median"),
        ("std", "population_std(values)", "population standard deviation"),
        ("min", "min(values)", "minimum"),
        ("max", "max(values)", "maximum"),
        ("range", "max-min", "range"),
        ("p1", "percentile(1)", "1st percentile"),
        ("p5", "percentile(5)", "5th percentile"),
        ("p25", "percentile(25)", "25th percentile"),
        ("p50", "percentile(50)", "50th percentile"),
        ("p75", "percentile(75)", "75th percentile"),
        ("p95", "percentile(95)", "95th percentile"),
        ("p99", "percentile(99)", "99th percentile"),
        ("iqr", "p75-p25", "interquartile range"),
        ("cv", "std/|mean|", "coefficient of variation (None if mean~0)"),
    ]:
        specs.append(
            FeatureSpec(
                feature_name=f"{prefix}_{suffix}",
                domain=domain,
                grain=grain,
                source_parameters=[parameter],
                formula=formula,
                unit=unit if suffix not in {"count", "cv"} else None,
                direction=None,
                allowed_for_ml=True,
                candidate_dependent=False,
                evaluation_only=False,
                normalization_required=suffix
                in {"mean", "median", "std", "min", "max", "range", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "iqr"},
                description=desc,
            )
        )
    return specs


def margin_feature_specs(
    *,
    prefix: str,
    domain: str,
    grain: str,
    parameter: str,
    unit: str | None,
    direction: str,
) -> list[FeatureSpec]:
    out: list[FeatureSpec] = []
    for name, formula, desc, u in [
        ("margin_count", "count", "margin observation count", None),
        ("margin_mean", "mean(signed_margin)", "mean signed margin to current limit", unit),
        ("margin_min", "min(signed_margin)", "minimum signed margin", unit),
        ("margin_max", "max(signed_margin)", "maximum signed margin", unit),
        ("violation_count", "count(violations)", "violation count vs current limit", None),
        ("violation_rate", "violations/count", "violation rate vs current limit", None),
        ("borderline_count", "count(borderline)", "borderline proximity count (not reliability)", None),
        ("borderline_rate", "borderline/count", "borderline rate (limit proximity)", None),
        ("safe_count", "count(safe)", "safe proximity count", None),
        ("safe_rate", "safe/count", "safe rate", None),
        (
            "fraction_within_guard_band",
            "borderline/count",
            "fraction in configurable guard band (limit proximity only)",
            None,
        ),
    ]:
        out.append(
            FeatureSpec(
                feature_name=f"{prefix}_{name}",
                domain=domain,
                grain=grain,
                source_parameters=[parameter],
                formula=formula,
                unit=u,
                direction=direction,
                allowed_for_ml=True,
                candidate_dependent=False,
                evaluation_only=False,
                normalization_required=name.startswith("margin_"),
                description=desc,
            )
        )
    return out
