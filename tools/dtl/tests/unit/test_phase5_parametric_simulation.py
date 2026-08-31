"""Phase 5 unit tests for Parametric simulation."""

from __future__ import annotations

from pathlib import Path

from dtl_agent.canonical.dataset import build_canonical_from_datasets
from dtl_agent.data.loaders.core_loader import CoreDataLoader
from dtl_agent.data.loaders.parametric_loader import ParametricDataLoader
from dtl_agent.features.margins import LimitSpec
from dtl_agent.simulation.parametric.candidates import classify_tighten_loosen, generate_candidates
from dtl_agent.simulation.parametric.config import ParametricSimulationConfig
from dtl_agent.simulation.parametric.die_index import build_parametric_die_index
from dtl_agent.simulation.parametric.engine import measurement_violates, simulate_parameter_candidate
from dtl_agent.simulation.parametric.optimizer import select_best_candidate
from tests.conftest import minimal_core_fixture, minimal_parametric_fixture, write_csv


def _augment_parametric(root: Path) -> None:
    rows = []
    conditions = [
        ("COND_RT_NOM", "25.0", "1.0", "NOMINAL"),
        ("COND_HOT_NOM", "85.0", "1.0", "HOT"),
        ("COND_RT_LOWV", "25.0", "0.9", "LOW_VDD"),
        ("COND_HOT_HIGHV", "85.0", "1.1", "HIGH_VDD"),
    ]
    # D1 mostly pass for both parameters, D2 violates at least one condition for each.
    for die, iddq_vals, vmax_vals in [
        ("LOT_A_D001", [40.0, 41.0, 39.0, 45.0], [1.20, 1.18, 1.16, 1.17]),
        ("LOT_A_D002", [40.0, 130.0, 39.0, 45.0], [1.20, 1.18, 1.10, 1.17]),
    ]:
        for (cid, t, v, mode), iddq, vmax in zip(conditions, iddq_vals, vmax_vals):
            rows.append(
                {
                    "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
                    "scenario_id": "SCEN_P_NORMAL",
                    "scenario_family": "normal",
                    "lot_id": "LOT_A",
                    "die_id": die,
                    "condition_id": cid,
                    "tester_id": "TESTER_A",
                    "site_id": "SITE_1",
                    "temperature_c": t,
                    "vdd_applied": v,
                    "test_mode": mode,
                    "test_id": "T_IDDQ",
                    "parameter": "IDDQ",
                    "measurement_value": str(iddq),
                    "unit": "uA",
                    "limit_type": "UPPER",
                    "generation_seed": "1",
                    "generator_version": "test",
                    "pass_fail_condition": "P" if iddq <= 50 else "F",
                }
            )
            rows.append(
                {
                    "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
                    "scenario_id": "SCEN_P_NORMAL",
                    "scenario_family": "normal",
                    "lot_id": "LOT_A",
                    "die_id": die,
                    "condition_id": cid,
                    "tester_id": "TESTER_A",
                    "site_id": "SITE_1",
                    "temperature_c": t,
                    "vdd_applied": v,
                    "test_mode": mode,
                    "test_id": "T_VMAX",
                    "parameter": "VMAX",
                    "measurement_value": str(vmax),
                    "unit": "V",
                    "limit_type": "LOWER",
                    "generation_seed": "1",
                    "generator_version": "test",
                    "pass_fail_condition": "P" if vmax >= 1.15 else "F",
                }
            )
    write_csv(root / "measurements.csv", rows)
    write_csv(
        root / "parts_dim.csv",
        [
            {
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D001",
                "scenario_id": "SCEN_P_NORMAL",
                "scenario_family": "normal",
                "tester_id": "TESTER_A",
                "site_id": "SITE_1",
                "v1_link": "True",
                "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
                "generation_seed": "1",
                "generator_version": "test",
            },
            {
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D002",
                "scenario_id": "SCEN_P_NORMAL",
                "scenario_family": "normal",
                "tester_id": "TESTER_A",
                "site_id": "SITE_1",
                "v1_link": "True",
                "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
                "generation_seed": "1",
                "generator_version": "test",
            },
            {
                "lot_id": "LOT_P_ONLY",
                "die_id": "LOT_P_ONLY_D001",
                "scenario_id": "SCEN_P_RES",
                "scenario_family": "resistance_degradation",
                "tester_id": "TESTER_B",
                "site_id": "SITE_2",
                "v1_link": "False",
                "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
                "generation_seed": "1",
                "generator_version": "test",
            },
        ],
    )


def _build_canonical(tmp_path: Path):
    core_root = minimal_core_fixture(tmp_path)
    par_root = minimal_parametric_fixture(tmp_path)
    _augment_parametric(par_root)
    core = CoreDataLoader(core_root).load(materialize_measurements=True)
    par = ParametricDataLoader(par_root).load(materialize_measurements=True)
    return build_canonical_from_datasets(core, par)


def test_upper_lower_logic() -> None:
    assert measurement_violates(51.0, direction="UPPER", candidate=50.0)
    assert not measurement_violates(49.0, direction="UPPER", candidate=50.0)
    assert measurement_violates(1.10, direction="LOWER", candidate=1.15)
    assert not measurement_violates(1.20, direction="LOWER", candidate=1.15)


def test_candidate_direction_classification() -> None:
    assert classify_tighten_loosen("UPPER", 50.0, 45.0) == "TIGHTER"
    assert classify_tighten_loosen("LOWER", 1.15, 1.20) == "TIGHTER"
    assert classify_tighten_loosen("LOWER", 1.15, 1.10) == "LOOSER"


def test_candidate_generation_includes_current() -> None:
    lim = LimitSpec("UPPER", 50.0, "uA", "SYNTHETIC_ASSUMED", "T_IDDQ", "IDDQ")
    cands = generate_candidates(limit=lim, grid=[40.0, 60.0])
    assert [c.candidate_limit for c in cands] == [40.0, 50.0, 60.0]


def test_condition_aware_simulation(tmp_path: Path) -> None:
    canonical = _build_canonical(tmp_path)
    index = build_parametric_die_index(canonical, {"IDDQ", "VMAX"})
    cfg = ParametricSimulationConfig(candidate_grids={"IDDQ": [50.0], "VMAX": [1.15]}, parameters=["IDDQ", "VMAX"])
    iddq_lim = LimitSpec("UPPER", 50.0, "uA", "SYNTHETIC_ASSUMED", "T_IDDQ", "IDDQ")
    v_cand = generate_candidates(limit=iddq_lim, grid=[50.0])[0]
    res, outcomes, cond_rows = simulate_parameter_candidate(index, v_cand, cfg)
    assert res.total_dies == 2
    assert res.violating_dies == 1
    assert res.simulated_yield == 0.5
    assert len(outcomes) == 8
    assert len(cond_rows) == 4


def test_vmax_lower_behavior(tmp_path: Path) -> None:
    canonical = _build_canonical(tmp_path)
    index = build_parametric_die_index(canonical, {"VMAX"})
    cfg = ParametricSimulationConfig(candidate_grids={"VMAX": [1.15, 1.20]}, parameters=["VMAX"])
    lim = LimitSpec("LOWER", 1.15, "V", "SYNTHETIC_ASSUMED", "T_VMAX", "VMAX")
    cands = generate_candidates(limit=lim, grid=[1.15, 1.20])
    r1, _, _ = simulate_parameter_candidate(index, cands[0], cfg)
    r2, _, _ = simulate_parameter_candidate(index, cands[1], cfg)
    assert r2.simulated_yield <= r1.simulated_yield


def test_tie_breaking(tmp_path: Path) -> None:
    canonical = _build_canonical(tmp_path)
    index = build_parametric_die_index(canonical, {"IDDQ"})
    cfg = ParametricSimulationConfig(candidate_grids={"IDDQ": [50.0, 55.0]}, parameters=["IDDQ"])
    lim = LimitSpec("UPPER", 50.0, "uA", "SYNTHETIC_ASSUMED", "T_IDDQ", "IDDQ")
    cands = generate_candidates(limit=lim, grid=[50.0, 55.0])
    rows = [simulate_parameter_candidate(index, c, cfg)[0] for c in cands]
    # force tie
    rows[0].objective_score = 0.5
    rows[1].objective_score = 0.5
    best = select_best_candidate(rows, weights=cfg.objective)
    assert best.candidate_limit == 50.0
