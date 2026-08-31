"""Phase 12.3 — month-aware temporal loading, identity, and simulation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.die_index import build_core_die_index_from_temporal
from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.loader import TemporalLoaderError, load_temporal_month
from dtl_agent.data.temporal.paths import (
    ALLOWED_PRODUCTION_MONTHS,
    TemporalPathError,
    month_simulation_root,
    temporal_data_root,
    validate_production_month,
)
from dtl_agent.data.temporal.simulation import (
    run_temporal_core_simulation,
    temporal_core_limits,
    temporal_core_simulation_config,
)
from dtl_agent.features.margins import LimitSpec
from dtl_agent.recommendation.pipeline import recommend
from dtl_agent.simulation.core.candidates import generate_candidates
from dtl_agent.simulation.core.engine import simulate_parameter_candidate
from dtl_agent.simulation.core.pipeline import run_core_simulation_optimization


ROOT = default_project_root()
TEMPORAL_AVAILABLE = (temporal_data_root(ROOT) / "2026-01" / "actual_die" / "measurements.csv").is_file()

pytestmark = pytest.mark.skipif(
    not TEMPORAL_AVAILABLE,
    reason="data/3 months data package not present",
)


@pytest.fixture(scope="module")
def month_jan():
    return load_temporal_month("2026-01", project_root=ROOT)


@pytest.fixture(scope="module")
def month_feb():
    return load_temporal_month("2026-02", project_root=ROOT)


@pytest.fixture(scope="module")
def month_mar():
    return load_temporal_month("2026-03", project_root=ROOT)


def test_load_temporal_month_january_only(month_jan):
    assert set(month_jan.actual_die["production_month"].astype(str).unique()) == {"2026-01"}
    assert set(month_jan.parametric["production_month"].astype(str).unique()) == {"2026-01"}
    assert month_jan.production_month == "2026-01"
    assert "lot_id" in month_jan.actual_die.columns
    assert "die_id" in month_jan.actual_die.columns
    assert "pattern_id" in month_jan.actual_die.columns
    assert "test_id" in month_jan.actual_die.columns
    assert "pass_fail" in month_jan.actual_die.columns
    assert "condition_id" in month_jan.parametric.columns
    assert "pass_fail" in month_jan.parametric.columns


def test_load_temporal_month_february_only(month_feb):
    assert set(month_feb.actual_die["production_month"].astype(str).unique()) == {"2026-02"}
    assert set(month_feb.parametric["production_month"].astype(str).unique()) == {"2026-02"}


def test_load_temporal_month_march_only(month_mar):
    assert set(month_mar.actual_die["production_month"].astype(str).unique()) == {"2026-03"}
    assert set(month_mar.parametric["production_month"].astype(str).unique()) == {"2026-03"}


def test_same_die_independent_sequence_ids(month_jan, month_feb):
    jan_id = make_sequence_id("DTL_NORM_001", "DTL_NORM_001_D001", "2026-01")
    feb_id = make_sequence_id("DTL_NORM_001", "DTL_NORM_001_D001", "2026-02")
    assert jan_id == "2026-01::DTL_NORM_001::DTL_NORM_001_D001"
    assert feb_id == "2026-02::DTL_NORM_001::DTL_NORM_001_D001"
    assert jan_id != feb_id
    jan_ids = set(month_jan.die_identities)
    feb_ids = set(month_feb.die_identities)
    assert jan_id in jan_ids
    assert feb_id in feb_ids
    assert jan_ids.isdisjoint(feb_ids)


def test_invalid_month_fails_clearly():
    with pytest.raises(TemporalPathError, match="Invalid production_month"):
        validate_production_month("2025-12")
    with pytest.raises((TemporalPathError, TemporalLoaderError)):
        load_temporal_month("2025-12", project_root=ROOT)
    assert "2026-01" in ALLOWED_PRODUCTION_MONTHS


def test_temporal_mode_never_reads_legacy_simulation_artifacts(tmp_path, month_jan, monkeypatch):
    """Temporal sim writes only under artifacts/temporal and does not open legacy sim CSVs."""
    legacy = ROOT / "artifacts" / "simulation" / "core" / "candidate_results.csv"
    opened: list[Path] = []
    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        opened.append(Path(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)

    cfg = temporal_core_simulation_config()
    cfg.candidate_grids = {"ir_drop": [25.0], "thermal": [60.0]}
    index = build_core_die_index_from_temporal(month_jan)
    out_sim = tmp_path / "artifacts" / "temporal" / "2026-01" / "simulation" / "core"
    out_opt = tmp_path / "artifacts" / "temporal" / "2026-01" / "optimization" / "core"
    arts = run_core_simulation_optimization(
        tmp_path,
        config=cfg,
        die_index=index,
        limits=temporal_core_limits(),
        output_simulation_dir=out_sim,
        output_optimization_dir=out_opt,
        source_checksums={"production_month": "2026-01"},
        production_month="2026-01",
        joint_search="none",
    )
    assert arts.paths["candidate_results"].is_file()
    assert "temporal" in str(arts.paths["candidate_results"]).replace("\\", "/")
    for p in opened:
        resolved = str(p.resolve()).replace("\\", "/")
        assert "/artifacts/simulation/" not in resolved
    assert legacy not in {p.resolve() for p in opened if p.exists() or True}


def test_temporal_simulation_contains_only_selected_month_dies(tmp_path, month_jan):
    index = build_core_die_index_from_temporal(month_jan)
    expected = {(str(r.lot_id), str(r.die_id)) for r in
                month_jan.actual_die[["lot_id", "die_id"]].drop_duplicates().itertuples(index=False)}
    assert set(index.die_ids) == expected

    cfg = temporal_core_simulation_config()
    cfg.candidate_grids = {"ir_drop": [25.0], "thermal": [60.0]}
    out_sim = tmp_path / "artifacts" / "temporal" / "2026-01" / "simulation" / "core"
    out_opt = tmp_path / "artifacts" / "temporal" / "2026-01" / "optimization" / "core"
    arts = run_core_simulation_optimization(
        tmp_path,
        config=cfg,
        die_index=index,
        limits=temporal_core_limits(),
        output_simulation_dir=out_sim,
        output_optimization_dir=out_opt,
        source_checksums={"production_month": "2026-01"},
        production_month="2026-01",
        joint_search="none",
    )
    assert arts.paths["candidate_results"].parent == out_sim
    assert (ROOT / "artifacts" / "simulation") not in arts.paths["candidate_results"].parents
    for res in arts.independent_results["ir_drop"]:
        assert res.total_dies == len(expected)
        assert "simulated_yield" in res.to_dict()
        assert "violation_rate" in res.to_dict()
        assert "borderline_rate" in res.to_dict()
        assert "objective_score" in res.to_dict()
        assert "candidate_limit" in res.to_dict()
        assert res.parameter == "ir_drop"


def test_january_and_march_simulation_evidence_can_differ(month_jan, month_mar):
    idx_jan = build_core_die_index_from_temporal(month_jan)
    idx_mar = build_core_die_index_from_temporal(month_mar)
    assert set(idx_jan.die_ids) == set(idx_mar.die_ids)  # same lot/die labels
    assert len(idx_jan.die_ids) == 1000

    cfg = temporal_core_simulation_config()
    lim = temporal_core_limits()["ir_drop"]
    cand = generate_candidates(limit=lim, grid=[25.0])[0]
    res_jan, _ = simulate_parameter_candidate(idx_jan, cand, cfg)
    res_mar, _ = simulate_parameter_candidate(idx_mar, cand, cfg)
    assert res_jan.total_dies == res_mar.total_dies == 1000
    # Temporal drift: at least one evidence field differs for the same candidate
    differed = (
        res_jan.simulated_yield != res_mar.simulated_yield
        or res_jan.violation_rate != res_mar.violation_rate
        or res_jan.objective_score != res_mar.objective_score
        or res_jan.borderline_rate != res_mar.borderline_rate
    )
    assert differed, (
        f"Expected Jan vs Mar drift; jan_yield={res_jan.simulated_yield} "
        f"mar_yield={res_mar.simulated_yield}"
    )


def test_legacy_production_month_none_sequence_and_recommend():
    import inspect

    assert make_sequence_id("L1", "D1", None) == "L1::D1"
    assert make_sequence_id("L1", "D1") == "L1::D1"
    sig = inspect.signature(recommend)
    assert "production_month" in sig.parameters
    assert sig.parameters["production_month"].default is None
    # Phase 12.8: temporal month is wired; legacy None path stays separate.
    result = recommend(
        lot_id="DTL_NORM_001",
        die_id="DTL_NORM_001_D001",
        parameters=["ir_drop"],
        production_month="2026-01",
        project_root=ROOT,
    )
    assert result.production_month == "2026-01"
    assert len(result.recommendations) == 1
    rec = result.recommendations[0]
    assert rec.production_month == "2026-01"
    assert rec.model_used == "core_gru_temporal_v1"
    assert "SIMULATOR_DERIVED_TEMPORAL_2026-01" in rec.evidence_origin


def test_month_simulation_root_layout():
    path = month_simulation_root("2026-02", ROOT)
    assert path == ROOT / "artifacts" / "temporal" / "2026-02" / "simulation"
    assert "artifacts/simulation" not in str(path).replace("\\", "/")


def test_run_temporal_core_simulation_writes_temporal_tree(tmp_path, month_jan, monkeypatch):
    """End-to-end temporal entry writes under temporal artifact root only."""
    # Redirect project root artifact writes via monkeypatch of path helpers used by runner
    from dtl_agent.data.temporal import simulation as sim_mod

    def _sim_root(month: str, project_root=None):
        return tmp_path / "artifacts" / "temporal" / month / "simulation"

    def _opt_root(month: str, project_root=None):
        return tmp_path / "artifacts" / "temporal" / month / "optimization"

    monkeypatch.setattr(sim_mod, "month_simulation_root", _sim_root)
    monkeypatch.setattr(sim_mod, "month_optimization_root", _opt_root)

    # Shrink grid via patched config
    def _cfg():
        c = temporal_core_simulation_config()
        c.candidate_grids = {"ir_drop": [25.0], "thermal": [60.0]}
        return c

    monkeypatch.setattr(sim_mod, "temporal_core_simulation_config", _cfg)

    arts = run_temporal_core_simulation(
        "2026-01",
        project_root=ROOT,
        month_data=month_jan,
        joint_search="none",
    )
    cand_path = arts.paths["candidate_results"]
    assert cand_path.is_file()
    assert str(cand_path).replace("\\", "/").endswith(
        "artifacts/temporal/2026-01/simulation/core/candidate_results.csv"
    )
    legacy = ROOT / "artifacts" / "simulation" / "core" / "candidate_results.csv"
    assert cand_path.resolve() != legacy.resolve()
