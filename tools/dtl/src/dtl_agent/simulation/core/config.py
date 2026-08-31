"""Phase 4 Core simulation configuration (derived; does not modify source rules)."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dtl_agent.canonical.dataset import CanonicalDataset


@dataclass
class ObjectiveWeights:
    """Configurable synthetic objective weights (not a production business objective).

    Default form (agent-visible proxies only):
      yield_weight * simulated_yield
      - lambda_risk * (w_defective * defective_proxy + w_risky * borderline_rate)
      - lambda_ff * false_fail_proxy

    ``w_defective`` defaults to 0 because latent defective labels are forbidden;
    ``defective_proxy`` (source_FAIL accepted) remains computed for analysis only.
    """

    yield_weight: float = 1.0
    lambda_risk: float = 2.0
    w_defective: float = 0.0
    w_risky: float = 0.4
    lambda_ff: float = 0.15
    tie_epsilon: float = 1e-12


@dataclass
class CoreSimulationConfig:
    version: str = "phase4_core_v1"
    die_policy: str = "ANY_VIOLATION"
    violation_rate_threshold: float = 0.01
    consecutive_count: int = 3
    multi_parameter_policy: str = "OR"
    borderline_margin_percent: float = 5.0
    objective: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    # Explicit grids derived from source limit_simulation_config rationale (copied, not edited in place)
    candidate_grids: dict[str, list[float]] = field(default_factory=dict)
    parameters: list[str] = field(default_factory=lambda: ["ir_drop", "thermal"])
    write_per_die_for_selected_only: bool = True
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def default_candidate_grids_from_source_rules(canonical: CanonicalDataset) -> dict[str, list[float]]:
    """Build documented Core grids from source construction rationale + current limits.

    Does not invent Setup/Hold limits. Values follow the published grid-construction notes
    in ``limit_simulation_config.json`` (tighter / current / transition / looser).
    """
    ir = canonical.get_current_limit("core", test_id="T_IR_DROP_MV")
    th = canonical.get_current_limit("core", test_id="T_THERMAL_C")
    # From source rationale: IR 20-24, 25, 26-30, 35-72; Thermal 50-58, 60, 61-65, 70-92
    ir_grid = [
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
        float(ir.current_limit),
        26.0,
        27.0,
        28.0,
        29.0,
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
        55.0,
        60.0,
        65.0,
        70.0,
        72.0,
    ]
    th_grid = [
        50.0,
        52.0,
        54.0,
        56.0,
        58.0,
        float(th.current_limit),
        61.0,
        62.0,
        63.0,
        64.0,
        65.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
        92.0,
    ]
    # Deduplicate while preserving order
    def _uniq(vals: list[float]) -> list[float]:
        out: list[float] = []
        seen: set[float] = set()
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    return {"ir_drop": _uniq(ir_grid), "thermal": _uniq(th_grid)}


def build_core_simulation_config(canonical: CanonicalDataset) -> CoreSimulationConfig:
    src = canonical.core.limit_simulation_config
    disp = canonical.core.disposition_rules
    agg = src.get("aggregation", {}) if isinstance(src, dict) else {}
    gb = src.get("guard_band", {}) if isinstance(src, dict) else {}
    die_pol = agg.get("die_policy") or disp.get("die_level_aggregation", {}).get(
        "default_policy", "ANY_VIOLATION"
    )
    multi = agg.get("multi_parameter_policy") or disp.get("multi_parameter_aggregation", {}).get(
        "default", "OR"
    )
    return CoreSimulationConfig(
        die_policy=str(die_pol),
        violation_rate_threshold=float(agg.get("violation_rate_threshold", 0.01)),
        consecutive_count=int(agg.get("consecutive_count", 3)),
        multi_parameter_policy=str(multi),
        borderline_margin_percent=float(gb.get("borderline_margin_percent", 5.0)),
        objective=ObjectiveWeights(),
        candidate_grids=default_candidate_grids_from_source_rules(canonical),
        notes={
            "source_disposition_vs_simulated": (
                "source yield (scan/diagnosis) is NOT equal to simulated_metric_yield "
                "under candidate limits unless they happen to coincide."
            ),
            "objective_note": (
                "Synthetic configurable objective using agent-visible proxies only. "
                "Form: yield - lambda_risk*(w_defective*defective_proxy + w_risky*borderline_rate) "
                "- lambda_ff*false_fail_proxy. w_defective defaults to 0 (latent defective "
                "labels forbidden); otherwise accepting source_FAIL dies dominates and "
                "pathologically prefers yield=0. Not an industry production objective."
            ),
            "grid_source": (
                "Derived from data/core/rules/limit_simulation_config.json construction "
                "rationale; written to artifacts (source rules files are not modified)."
            ),
            "guard_band_note": (
                "Limit proximity indicator only — not reliability or customer-escape probability."
            ),
            "ml_contract": (
                "Candidate outcomes (context + candidate_limit + simulated metrics) are "
                "intended as Phase 6 supervision for the future GRU candidate ranker. "
                "No true_optimal / eval labels."
            ),
        },
    )


def write_config(path: Path, config: CoreSimulationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_config(path: Path) -> CoreSimulationConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    obj = raw.get("objective", {})
    return CoreSimulationConfig(
        version=raw.get("version", "phase4_core_v1"),
        die_policy=raw["die_policy"],
        violation_rate_threshold=float(raw["violation_rate_threshold"]),
        consecutive_count=int(raw["consecutive_count"]),
        multi_parameter_policy=raw["multi_parameter_policy"],
        borderline_margin_percent=float(raw["borderline_margin_percent"]),
        objective=ObjectiveWeights(**{k: obj[k] for k in ObjectiveWeights.__dataclass_fields__ if k in obj}),
        candidate_grids={k: list(map(float, v)) for k, v in raw.get("candidate_grids", {}).items()},
        parameters=list(raw.get("parameters", ["ir_drop", "thermal"])),
        write_per_die_for_selected_only=bool(raw.get("write_per_die_for_selected_only", True)),
        notes=deepcopy(raw.get("notes", {})),
    )
