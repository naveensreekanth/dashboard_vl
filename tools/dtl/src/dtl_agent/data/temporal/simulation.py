"""Month-scoped Core simulation orchestration (Phase 12.3).

Writes under ``artifacts/temporal/{production_month}/simulation/`` only.
Never writes to legacy ``artifacts/simulation/``.
"""

from __future__ import annotations

from pathlib import Path

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.die_index import build_core_die_index_from_temporal
from dtl_agent.data.temporal.loader import TemporalMonthData, load_temporal_month
from dtl_agent.data.temporal.paths import (
    month_optimization_root,
    month_simulation_root,
    temporal_data_root,
    validate_production_month,
)
from dtl_agent.features.io_utils import file_sha256
from dtl_agent.features.margins import LimitSpec
from dtl_agent.simulation.core.config import CoreSimulationConfig, ObjectiveWeights
from dtl_agent.simulation.core.pipeline import (
    CoreSimulationArtifacts,
    run_core_simulation_optimization,
)

# Same engineering limits as temporal package metadata / Phase 4 Core defaults.
_TEMPORAL_IR_LIMIT = LimitSpec(
    direction="UPPER",
    value=25.0,
    unit="mV",
    source_status="SOURCE_CONFIRMED",
    test_id="T_IR_DROP_MV",
    parameter="ir_drop",
)
_TEMPORAL_THERMAL_LIMIT = LimitSpec(
    direction="UPPER",
    value=60.0,
    unit="°C",
    source_status="SOURCE_CONFIRMED",
    test_id="T_THERMAL_C",
    parameter="thermal",
)


def temporal_core_limits() -> dict[str, LimitSpec]:
    return {"ir_drop": _TEMPORAL_IR_LIMIT, "thermal": _TEMPORAL_THERMAL_LIMIT}


def temporal_core_simulation_config() -> CoreSimulationConfig:
    """Same grids / objective form as Phase 4 Core; formulas unchanged."""
    ir_grid = [
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
        25.0,
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
        60.0,
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
    return CoreSimulationConfig(
        version="phase12_3_temporal_core_v1",
        die_policy="ANY_VIOLATION",
        violation_rate_threshold=0.01,
        consecutive_count=3,
        multi_parameter_policy="OR",
        borderline_margin_percent=5.0,
        objective=ObjectiveWeights(),
        candidate_grids={"ir_drop": ir_grid, "thermal": th_grid},
        notes={
            "temporal": True,
            "production_month_role": "population filter / artifact path only",
            "legacy_simulation_artifacts": "not used",
        },
    )


def run_temporal_core_simulation(
    production_month: str,
    *,
    project_root: Path | None = None,
    month_data: TemporalMonthData | None = None,
    joint_search: str = "product",
) -> CoreSimulationArtifacts:
    """Simulate candidates on dies from a single production month only.

    Invokes ``run_core_simulation_optimization`` with a month-scoped die index and
    temporal output directories. Does not read or write ``artifacts/simulation/``.
    """
    root = project_root or default_project_root()
    month = validate_production_month(production_month)
    data = month_data or load_temporal_month(month, project_root=root)

    # Hard guard: never point at legacy simulation tree for temporal runs
    legacy_sim = root / "artifacts" / "simulation"
    sim_root = month_simulation_root(month, root)
    opt_root = month_optimization_root(month, root)
    if sim_root.resolve() == legacy_sim.resolve() or legacy_sim in sim_root.resolve().parents:
        raise RuntimeError("Temporal simulation must not write under artifacts/simulation/")

    index = build_core_die_index_from_temporal(data)
    actual_path = data.month_path / "actual_die" / "measurements.csv"
    checksums = {
        "temporal_actual_die": file_sha256(actual_path),
        "temporal_data_root": str(temporal_data_root(root)),
        "production_month": month,
    }

    return run_core_simulation_optimization(
        root,
        config=temporal_core_simulation_config(),
        joint_search=joint_search,
        die_index=index,
        limits=temporal_core_limits(),
        output_simulation_dir=sim_root / "core",
        output_optimization_dir=opt_root / "core",
        source_checksums=checksums,
        production_month=month,
    )
