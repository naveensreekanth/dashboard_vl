"""Phase 5 Parametric simulation configuration."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dtl_agent.canonical.dataset import CanonicalDataset


@dataclass
class ObjectiveWeights:
    yield_weight: float = 1.0
    lambda_risk: float = 2.0
    w_defective: float = 0.0
    w_risky: float = 0.4
    lambda_ff: float = 0.15
    tie_epsilon: float = 1e-12


@dataclass
class ParametricSimulationConfig:
    version: str = "phase5_parametric_v1"
    die_policy: str = "ANY_VIOLATION"
    violation_rate_threshold: float = 0.01
    consecutive_count: int = 3
    condition_policy: str = "ALL_REQUIRED_CONDITIONS_PASS"
    borderline_margin_percent: float = 5.0
    objective: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    candidate_grids: dict[str, list[float]] = field(default_factory=dict)
    parameters: list[str] = field(
        default_factory=lambda: [
            "VMIN",
            "VMAX",
            "IDDQ",
            "SUPPLY_CURRENT",
            "CONTACT_RESISTANCE",
            "INTERCONNECT_RESISTANCE",
            "ON_RESISTANCE",
        ]
    )
    write_per_die_for_selected_only: bool = True
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_candidate_grids_from_source_rules(canonical: CanonicalDataset) -> dict[str, list[float]]:
    src = canonical.parametric.limit_simulation_config or {}
    raw = src.get("candidate_grids", {})
    out: dict[str, list[float]] = {}
    for key, values in raw.items():
        if not isinstance(values, list):
            continue
        uniq: list[float] = []
        seen: set[float] = set()
        for v in values:
            fv = float(v)
            if fv not in seen:
                seen.add(fv)
                uniq.append(fv)
        out[str(key)] = uniq
    return out


def build_parametric_simulation_config(canonical: CanonicalDataset) -> ParametricSimulationConfig:
    src = canonical.parametric.limit_simulation_config or {}
    disp = canonical.parametric.disposition_rules or {}
    gb_pct = float(src.get("objective", {}).get("acceptable_band_tol", 0.05)) * 100.0
    policy = str(disp.get("policy", "ANY_VIOLATION_ON_DIE_CONDITION"))
    if "ANY_VIOLATION" in policy:
        die_policy = "ANY_VIOLATION"
    else:
        die_policy = "ANY_VIOLATION"
    return ParametricSimulationConfig(
        die_policy=die_policy,
        borderline_margin_percent=gb_pct if gb_pct > 0 else 5.0,
        objective=ObjectiveWeights(),
        candidate_grids=default_candidate_grids_from_source_rules(canonical),
        notes={
            "source_disposition_vs_simulated": (
                "Source pass/fail fields are observational labels; simulated yield is candidate-dependent "
                "and not equal unless they coincide."
            ),
            "synthetic_limits_note": (
                "All parametric limits are SYNTHETIC_ASSUMED in this dataset and not production specifications."
            ),
            "objective_note": (
                "Synthetic objective uses agent-visible proxies only; no latent/eval labels."
            ),
        },
    )


def write_config(path: Path, config: ParametricSimulationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_config(path: Path) -> ParametricSimulationConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    obj = raw.get("objective", {})
    return ParametricSimulationConfig(
        version=raw.get("version", "phase5_parametric_v1"),
        die_policy=raw.get("die_policy", "ANY_VIOLATION"),
        violation_rate_threshold=float(raw.get("violation_rate_threshold", 0.01)),
        consecutive_count=int(raw.get("consecutive_count", 3)),
        condition_policy=raw.get("condition_policy", "ALL_REQUIRED_CONDITIONS_PASS"),
        borderline_margin_percent=float(raw.get("borderline_margin_percent", 5.0)),
        objective=ObjectiveWeights(**{k: obj[k] for k in ObjectiveWeights.__dataclass_fields__ if k in obj}),
        candidate_grids={k: list(map(float, v)) for k, v in raw.get("candidate_grids", {}).items()},
        parameters=list(raw.get("parameters", [])),
        write_per_die_for_selected_only=bool(raw.get("write_per_die_for_selected_only", True)),
        notes=deepcopy(raw.get("notes", {})),
    )
