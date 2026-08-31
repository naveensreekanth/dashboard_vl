"""Phase 5 integration test on real Parametric data."""

from __future__ import annotations

import pytest

from dtl_agent.canonical.dataset import build_canonical_dataset
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.features.io_utils import file_sha256
from dtl_agent.simulation.parametric.pipeline import (
    run_parametric_simulation_optimization,
    validate_phase5,
)
from dtl_agent.validation.pipeline import validate_bundle


@pytest.mark.integration
def test_real_parametric_simulation_optimization() -> None:
    root = default_project_root()
    if not (root / "data" / "parametric" / "measurements.csv").exists():
        pytest.skip("missing data")
    param_path = root / "data" / "parametric" / "measurements.csv"
    before = file_sha256(param_path)

    core = load_core(root, materialize_measurements=False)
    parametric = load_parametric(root, materialize_measurements=False)
    bundle = validate_bundle(core, parametric)
    canonical = build_canonical_dataset(bundle)
    artifacts = run_parametric_simulation_optimization(root, canonical=canonical)
    report = validate_phase5(artifacts, canonical)
    assert report["final_status"] == "PASS", report

    artifacts2 = run_parametric_simulation_optimization(root, canonical=canonical)
    assert artifacts.paths["candidate_results"].read_text(encoding="utf-8") == artifacts2.paths[
        "candidate_results"
    ].read_text(encoding="utf-8")
    assert file_sha256(param_path) == before
    assert artifacts.runtime_seconds < 600
