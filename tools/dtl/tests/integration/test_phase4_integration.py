"""Phase 4 integration tests on real Core data."""

from __future__ import annotations

import pytest

from dtl_agent.canonical.dataset import build_canonical_dataset
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.features.io_utils import file_sha256
from dtl_agent.simulation.core.pipeline import (
    run_core_simulation_optimization,
    validate_phase4,
)
from dtl_agent.validation.pipeline import validate_bundle


@pytest.mark.integration
def test_real_core_simulation_optimization() -> None:
    root = default_project_root()
    if not (root / "data" / "core" / "measurements.csv").exists():
        pytest.skip("missing data")
    core_path = root / "data" / "core" / "measurements.csv"
    before = file_sha256(core_path)

    core = load_core(root, materialize_measurements=False)
    parametric = load_parametric(root, materialize_measurements=False)
    bundle = validate_bundle(core, parametric)
    canonical = build_canonical_dataset(bundle)
    artifacts = run_core_simulation_optimization(root, canonical=canonical)
    report = validate_phase4(artifacts, canonical)
    assert report["final_status"] == "PASS", report

    # Deterministic rerun
    artifacts2 = run_core_simulation_optimization(root, canonical=canonical)
    assert artifacts.selected["ir_drop"].candidate_limit == artifacts2.selected["ir_drop"].candidate_limit
    assert artifacts.selected["thermal"].candidate_limit == artifacts2.selected["thermal"].candidate_limit
    assert artifacts.paths["candidate_results"].read_text(encoding="utf-8") == artifacts2.paths[
        "candidate_results"
    ].read_text(encoding="utf-8")

    assert file_sha256(core_path) == before
    assert artifacts.runtime_seconds < 600  # sanity
    assert "setup_slack" not in artifacts.independent_results
