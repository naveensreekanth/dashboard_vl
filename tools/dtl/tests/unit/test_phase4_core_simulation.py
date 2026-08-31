"""Phase 4 unit tests for Core simulation + optimization."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.canonical.dataset import build_canonical_from_datasets
from dtl_agent.data.loaders.core_loader import CoreDataLoader
from dtl_agent.data.loaders.parametric_loader import ParametricDataLoader
from dtl_agent.features.margins import LimitSpec
from dtl_agent.simulation.core.candidates import (
    classify_tighten_loosen,
    generate_candidates,
)
from dtl_agent.simulation.core.config import CoreSimulationConfig, ObjectiveWeights
from dtl_agent.simulation.core.die_index import DieParamSeries, build_core_die_index
from dtl_agent.simulation.core.engine import (
    compute_objective,
    die_fails,
    die_proximity,
    pattern_violates,
    simulate_joint_candidate,
    simulate_parameter_candidate,
)
from dtl_agent.simulation.core.candidates import CandidateLimit
from dtl_agent.simulation.core.optimizer import select_best_candidate
from dtl_agent.simulation.core.die_index import CoreDieIndex
from tests.conftest import minimal_core_fixture, minimal_parametric_fixture, write_csv


def test_upper_lower_violation_logic() -> None:
    assert pattern_violates(26.0, direction="UPPER", candidate=25.0) is True
    assert pattern_violates(24.0, direction="UPPER", candidate=25.0) is False
    assert pattern_violates(1.10, direction="LOWER", candidate=1.15) is True
    assert pattern_violates(1.20, direction="LOWER", candidate=1.15) is False


def test_any_violation_policy() -> None:
    s = DieParamSeries("L", "D", "PASS", [10.0, 26.0, 11.0])
    assert die_fails(s, direction="UPPER", candidate=25.0, policy="ANY_VIOLATION", violation_rate_threshold=0.01, consecutive_count=3)
    s2 = DieParamSeries("L", "D", "PASS", [10.0, 20.0, 11.0])
    assert not die_fails(s2, direction="UPPER", candidate=25.0, policy="ANY_VIOLATION", violation_rate_threshold=0.01, consecutive_count=3)


def test_violation_rate_and_consecutive() -> None:
    s = DieParamSeries("L", "D", "PASS", [30.0, 10.0, 10.0, 10.0])  # 25% viol
    assert not die_fails(
        s, direction="UPPER", candidate=25.0, policy="VIOLATION_RATE",
        violation_rate_threshold=0.5, consecutive_count=3,
    )
    assert die_fails(
        s, direction="UPPER", candidate=25.0, policy="VIOLATION_RATE",
        violation_rate_threshold=0.2, consecutive_count=3,
    )
    s2 = DieParamSeries("L", "D", "PASS", [30.0, 31.0, 32.0, 10.0])
    assert die_fails(
        s2, direction="UPPER", candidate=25.0, policy="CONSECUTIVE_VIOLATIONS",
        violation_rate_threshold=0.01, consecutive_count=3,
    )


def test_candidate_generation_and_direction() -> None:
    lim = LimitSpec("UPPER", 25.0, "mV", "SOURCE_CONFIRMED", "T_IR_DROP_MV", "ir_drop")
    cands = generate_candidates(limit=lim, grid=[20.0, 25.0, 30.0, 25.0])
    assert [c.candidate_limit for c in cands] == [20.0, 25.0, 30.0]
    assert cands[0].tighten_or_loosen == "TIGHTER"
    assert cands[1].tighten_or_loosen == "CURRENT"
    assert cands[2].tighten_or_loosen == "LOOSER"
    assert classify_tighten_loosen("LOWER", 1.15, 1.20) == "TIGHTER"
    assert classify_tighten_loosen("LOWER", 1.15, 1.10) == "LOOSER"


def test_guard_band_classification() -> None:
    s = DieParamSeries("L", "D", "PASS", [24.0])  # within 5% of 25 => borderline band (23.75, 25]
    assert die_proximity(s, direction="UPPER", candidate=25.0, borderline_pct=5.0) == "BORDERLINE"
    s2 = DieParamSeries("L", "D", "PASS", [20.0])
    assert die_proximity(s2, direction="UPPER", candidate=25.0, borderline_pct=5.0) == "SAFE"
    s3 = DieParamSeries("L", "D", "PASS", [26.0])
    assert die_proximity(s3, direction="UPPER", candidate=25.0, borderline_pct=5.0) == "VIOLATION"


def test_objective_and_tie_break() -> None:
    w = ObjectiveWeights()
    a = compute_objective(simulated_yield=0.9, defective_proxy=0.0, borderline_rate=0.1, false_fail_proxy=0.0, weights=w)
    b = compute_objective(simulated_yield=0.8, defective_proxy=0.0, borderline_rate=0.0, false_fail_proxy=0.0, weights=w)
    assert a != b
    # Build fake results for tie-break: same objective, prefer closer to current
    from dtl_agent.simulation.core.engine import CandidateSimulationResult

    def make(cand: float, obj: float) -> CandidateSimulationResult:
        return CandidateSimulationResult(
            domain="core",
            parameter="ir_drop",
            test_id="T_IR_DROP_MV",
            candidate_limit=cand,
            current_limit=25.0,
            direction="UPPER",
            unit="mV",
            source_status="SOURCE_CONFIRMED",
            candidate_delta=cand - 25.0,
            candidate_delta_percent=None,
            tighten_or_loosen="TIGHTER" if cand < 25 else "LOOSER",
            total_dies=10,
            good_dies=9,
            violating_dies=1,
            simulated_yield=0.9,
            simulated_fail_rate=0.1,
            violation_rate=0.1,
            borderline_rate=0.0,
            risky_rate=0.0,
            false_fail_proxy=0.0,
            defective_proxy=0.0,
            objective_score=obj,
            feasible=True,
        )

    best = select_best_candidate([make(20.0, 0.5), make(24.0, 0.5), make(30.0, 0.4)], weights=w)
    assert best.candidate_limit == 24.0  # closer to 25 than 20


def _augment_core_with_patterns(core_root: Path) -> None:
    """Add a second die with known IR values for simulation unit tests."""
    rows = []
    # die1: all safe under 25
    for pid in range(1, 4):
        for tid, param, val in [
            ("T_IR_DROP_MV", "ir_drop", "20.0"),
            ("T_THERMAL_C", "thermal", "50.0"),
            ("T_SETUP_SLACK_PS", "setup_slack", "30.0"),
            ("T_HOLD_SLACK_PS", "hold_slack", "20.0"),
            ("T_TEST_TIME_MS", "test_time", "1.0"),
        ]:
            rows.append(
                {
                    "lot_id": "LOT_A",
                    "die_id": "LOT_A_D001",
                    "pattern_id": str(pid),
                    "test_id": tid,
                    "test_name": tid.replace("T_", ""),
                    "parameter": param,
                    "measurement_value": val,
                    "unit": "mV",
                    "scenario_id": "SCEN_NORMAL",
                    "scenario_family": "normal",
                    "tester_id": "TESTER_A",
                    "site_id": "1",
                    "pass_fail_pattern": "PASS",
                    "die_status": "PASS",
                    "generation_seed": "1",
                    "generator_version": "test",
                    "production_sequence": "1",
                }
            )
    # die2: IR pattern max 30 -> fails at 25
    write_csv(
        core_root / "parts_dim.csv",
        [
            {
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D001",
                "scenario_id": "SCEN_NORMAL",
                "scenario_family": "normal",
                "status": "PASS",
                "tester_id": "TESTER_A",
                "generation_seed": "1",
                "generator_version": "test",
                "production_sequence": "1",
            },
            {
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D002",
                "scenario_id": "SCEN_NORMAL",
                "scenario_family": "normal",
                "status": "PASS",
                "tester_id": "TESTER_A",
                "generation_seed": "1",
                "generator_version": "test",
                "production_sequence": "1",
            },
        ],
    )
    for pid in range(1, 4):
        ir = "30.0" if pid == 2 else "20.0"
        for tid, param, val in [
            ("T_IR_DROP_MV", "ir_drop", ir),
            ("T_THERMAL_C", "thermal", "50.0"),
            ("T_SETUP_SLACK_PS", "setup_slack", "30.0"),
            ("T_HOLD_SLACK_PS", "hold_slack", "20.0"),
            ("T_TEST_TIME_MS", "test_time", "1.0"),
        ]:
            rows.append(
                {
                    "lot_id": "LOT_A",
                    "die_id": "LOT_A_D002",
                    "pattern_id": str(pid),
                    "test_id": tid,
                    "test_name": tid.replace("T_", ""),
                    "parameter": param,
                    "measurement_value": val,
                    "unit": "mV",
                    "scenario_id": "SCEN_NORMAL",
                    "scenario_family": "normal",
                    "tester_id": "TESTER_A",
                    "site_id": "1",
                    "pass_fail_pattern": "PASS",
                    "die_status": "PASS",
                    "generation_seed": "1",
                    "generator_version": "test",
                    "production_sequence": "1",
                }
            )
    write_csv(core_root / "measurements.csv", rows)


@pytest.fixture
def sim_canonical(tmp_path: Path):
    core_root = minimal_core_fixture(tmp_path)
    _augment_core_with_patterns(core_root)
    core = CoreDataLoader(core_root).load(materialize_measurements=True)
    parametric = ParametricDataLoader(minimal_parametric_fixture(tmp_path)).load(
        materialize_measurements=True
    )
    return build_canonical_from_datasets(core, parametric)


def test_die_index_and_yield(sim_canonical) -> None:
    index = build_core_die_index(sim_canonical)
    assert len(index.ir_drop) == 2
    cfg = CoreSimulationConfig(candidate_grids={"ir_drop": [25.0, 35.0], "thermal": [60.0]})
    lim = LimitSpec("UPPER", 25.0, "mV", "SOURCE_CONFIRMED", "T_IR_DROP_MV", "ir_drop")
    c25 = generate_candidates(limit=lim, grid=[25.0])[0]
    res25, dies = simulate_parameter_candidate(index, c25, cfg)
    assert res25.total_dies == 2
    assert res25.violating_dies == 1
    assert res25.simulated_yield == 0.5
    c35 = CandidateLimit("ir_drop", "T_IR_DROP_MV", "UPPER", "mV", "SOURCE_CONFIRMED", 25.0, 35.0)
    res35, _ = simulate_parameter_candidate(index, c35, cfg)
    assert res35.simulated_yield == 1.0


def test_joint_or_policy(sim_canonical) -> None:
    index = build_core_die_index(sim_canonical)
    cfg = CoreSimulationConfig()
    ir = CandidateLimit("ir_drop", "T_IR_DROP_MV", "UPPER", "mV", "SOURCE_CONFIRMED", 25.0, 25.0)
    th = CandidateLimit("thermal", "T_THERMAL_C", "UPPER", "°C", "SOURCE_CONFIRMED", 60.0, 60.0)
    joint = simulate_joint_candidate(index, ir_candidate=ir, thermal_candidate=th, config=cfg)
    assert joint.scope == "joint"
    assert joint.violating_dies == 1  # die2 IR fails


def test_schema_fields() -> None:
    from dtl_agent.simulation.core.engine import CandidateSimulationResult

    fields = set(CandidateSimulationResult.__dataclass_fields__)
    for required in [
        "candidate_limit",
        "current_limit",
        "simulated_yield",
        "violation_rate",
        "objective_score",
        "false_fail_proxy",
        "feasible",
    ]:
        assert required in fields
