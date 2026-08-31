"""Month-scoped parametric simulation for temporal package (Phase 12.5D support).

Writes under artifacts/temporal/{month}/simulation/parametric/ only.
Does not modify simulation formulas or legacy artifacts/simulation/parametric/.
"""

from __future__ import annotations

import json
from pathlib import Path

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.loader import TemporalMonthData, load_temporal_month
from dtl_agent.data.temporal.paths import (
    month_optimization_root,
    month_simulation_root,
    temporal_data_root,
    validate_production_month,
)
from dtl_agent.features.io_utils import file_sha256, write_csv_dicts, write_json
from dtl_agent.features.margins import LimitSpec
from dtl_agent.simulation.parametric.candidates import generate_candidates
from dtl_agent.simulation.parametric.config import ObjectiveWeights, ParametricSimulationConfig
from dtl_agent.simulation.parametric.die_index import (
    DieConditionSeries,
    ParametricDieIndex,
)
from dtl_agent.simulation.parametric.engine import simulate_parameter_candidate
from dtl_agent.simulation.parametric.optimizer import baseline_result, select_best_candidate
from dtl_agent.simulation.parametric.config import write_config

# Condition order matches UnifiedParameterGRURanker / Phase 12.5C.
_CONDITIONS = ("COND_RT_NOM", "COND_HOT_NOM", "COND_RT_LOWV", "COND_HOT_HIGHV")

_PARAM_SPECS: dict[str, dict] = {
    "VMIN": {
        "test_id": "T_VMIN",
        "direction": "UPPER",
        "unit": "V",
        "current": 0.85,
        "grid": [0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9, 0.92, 0.95, 1.0],
    },
    "VMAX": {
        "test_id": "T_VMAX",
        "direction": "LOWER",
        "unit": "V",
        "current": 1.15,
        "grid": [1.05, 1.08, 1.1, 1.12, 1.15, 1.18, 1.2, 1.22, 1.25],
    },
    "IDDQ": {
        "test_id": "T_IDDQ",
        "direction": "UPPER",
        "unit": "uA",
        "current": 50.0,
        "grid": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 100.0],
    },
    "SUPPLY_CURRENT": {
        "test_id": "T_SUPPLY_CURRENT",
        "direction": "UPPER",
        "unit": "mA",
        "current": 120.0,
        "grid": [90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 170.0, 200.0],
    },
    "CONTACT_RESISTANCE": {
        "test_id": "T_CONTACT_R",
        "direction": "UPPER",
        "unit": "ohm",
        "current": 5.0,
        "grid": [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0],
    },
    "INTERCONNECT_RESISTANCE": {
        "test_id": "T_INTERCONNECT_R",
        "direction": "UPPER",
        "unit": "ohm",
        "current": 15.0,
        "grid": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 25.0],
    },
    "ON_RESISTANCE": {
        "test_id": "T_ON_R",
        "direction": "UPPER",
        "unit": "ohm",
        "current": 25.0,
        "grid": [18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0, 40.0, 50.0],
    },
}


def build_parametric_die_index_from_temporal(month: TemporalMonthData) -> ParametricDieIndex:
    df = month.parametric
    params = set(_PARAM_SPECS)
    by_parameter: dict[str, dict[tuple[str, str, str], DieConditionSeries]] = {p: {} for p in params}
    condition_meta: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        param = str(r["parameter"])
        if param not in params:
            continue
        lot, die, cond = str(r["lot_id"]), str(r["die_id"]), str(r["condition_id"])
        key = (lot, die, cond)
        src = None if r.get("pass_fail_condition") is None else str(r["pass_fail_condition"])
        if key not in by_parameter[param]:
            by_parameter[param][key] = DieConditionSeries(lot, die, cond, src, [])
        by_parameter[param][key].values.append(float(r["measurement_value"]))
        if by_parameter[param][key].source_status is None:
            by_parameter[param][key].source_status = src
        if cond not in condition_meta:
            condition_meta[cond] = {
                "temperature_c": str(r.get("temperature_c", "")),
                "vdd_applied": str(r.get("vdd_applied", "")),
                "test_mode": str(r.get("test_mode", "")),
            }
    return ParametricDieIndex(
        by_parameter=by_parameter,
        condition_meta=condition_meta,
        expected_conditions=sorted(condition_meta) or list(_CONDITIONS),
    )


def temporal_parametric_config() -> ParametricSimulationConfig:
    grids = {p: list(spec["grid"]) for p, spec in _PARAM_SPECS.items()}
    return ParametricSimulationConfig(
        version="phase12_5d_temporal_parametric_v1",
        die_policy="ANY_VIOLATION",
        condition_policy="ALL_REQUIRED_CONDITIONS_PASS",
        borderline_margin_percent=5.0,
        objective=ObjectiveWeights(),
        candidate_grids=grids,
        parameters=list(_PARAM_SPECS.keys()),
        notes={"temporal": True, "legacy_simulation_artifacts": "not used"},
    )


def temporal_parametric_limits() -> dict[str, LimitSpec]:
    out: dict[str, LimitSpec] = {}
    for p, spec in _PARAM_SPECS.items():
        out[p] = LimitSpec(
            direction=spec["direction"],
            value=float(spec["current"]),
            unit=spec["unit"],
            source_status="SYNTHETIC_ASSUMED",
            test_id=spec["test_id"],
            parameter=p,
        )
    return out


def run_temporal_parametric_simulation(
    production_month: str,
    *,
    project_root: Path | None = None,
    month_data: TemporalMonthData | None = None,
) -> Path:
    """Simulate parametric candidates on one month; write temporal artifact tree."""
    root = project_root or default_project_root()
    month = validate_production_month(production_month)
    data = month_data or load_temporal_month(month, project_root=root)
    index = build_parametric_die_index_from_temporal(data)
    cfg = temporal_parametric_config()
    limits = temporal_parametric_limits()

    sim_dir = month_simulation_root(month, root) / "parametric"
    opt_dir = month_optimization_root(month, root) / "parametric"
    sim_dir.mkdir(parents=True, exist_ok=True)
    opt_dir.mkdir(parents=True, exist_ok=True)

    independent = {}
    selected = {}
    cand_rows = []
    result_rows = []
    for param in cfg.parameters:
        cands = generate_candidates(limit=limits[param], grid=cfg.candidate_grids[param])
        rows = []
        for c in cands:
            cand_rows.append(
                {
                    "parameter": c.parameter,
                    "test_id": c.test_id,
                    "direction": c.direction,
                    "unit": c.unit,
                    "source_status": c.source_status,
                    "current_limit": c.current_limit,
                    "candidate_limit": c.candidate_limit,
                    "delta_absolute": c.delta_absolute,
                    "delta_percent": c.delta_percent,
                    "tighten_or_loosen": c.tighten_or_loosen,
                }
            )
            res, _outcomes, _cond = simulate_parameter_candidate(index, c, cfg)
            rows.append(res)
            result_rows.append(res.to_dict())
        best = select_best_candidate(rows, weights=cfg.objective)
        independent[param] = rows
        selected[param] = best

    paths = {
        "simulation_config": sim_dir / "simulation_config.json",
        "candidate_grid": sim_dir / "candidate_grid.csv",
        "candidate_results": sim_dir / "candidate_results.csv",
        "selected_candidates": sim_dir / "selected_candidates.csv",
        "optimization_results": opt_dir / "optimization_results.csv",
        "optimization_summary": opt_dir / "optimization_summary.json",
    }
    write_config(paths["simulation_config"], cfg)
    write_csv_dicts(paths["candidate_grid"], cand_rows)
    write_csv_dicts(paths["candidate_results"], result_rows)
    sel_rows = [r.to_dict() for r in selected.values()]
    write_csv_dicts(paths["selected_candidates"], sel_rows)
    write_csv_dicts(paths["optimization_results"], sel_rows)
    write_json(
        paths["optimization_summary"],
        {
            "production_month": month,
            "baseline": {
                p: baseline_result(independent[p]).to_dict() if baseline_result(independent[p]) else None
                for p in independent
            },
            "selected": {p: selected[p].to_dict() for p in selected},
            "source": str(temporal_data_root(root)),
            "parametric_measurements_sha256": file_sha256(
                data.month_path / "parametric" / "measurements.csv"
            ),
        },
    )
    return paths["candidate_results"]
