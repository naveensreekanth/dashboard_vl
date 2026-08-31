"""Phase 6 integration tests on real artifacts/data."""

from __future__ import annotations

import pytest

from dtl_agent.canonical.dataset import build_canonical_dataset
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.features.io_utils import file_sha256
from dtl_agent.ml_dataset.pipeline import run_phase6_ml_dataset_assembly, validate_phase6
from dtl_agent.validation.pipeline import validate_bundle


@pytest.mark.integration
def test_phase6_end_to_end() -> None:
    root = default_project_root()
    if not (root / "data" / "core" / "measurements.csv").exists():
        pytest.skip("missing data")
    before_core = file_sha256(root / "data" / "core" / "measurements.csv")
    before_param = file_sha256(root / "data" / "parametric" / "measurements.csv")
    core = load_core(root, materialize_measurements=False)
    param = load_parametric(root, materialize_measurements=False)
    canonical = build_canonical_dataset(validate_bundle(core, param))
    artifacts = run_phase6_ml_dataset_assembly(root, canonical=canonical)
    report = validate_phase6(artifacts, canonical)
    assert report["final_status"] == "PASS", report
    assert artifacts.dataset_manifest["sequence_count"] == 1550
    assert artifacts.dataset_manifest["candidate_counts"]["core"] > 0
    assert artifacts.dataset_manifest["candidate_counts"]["parametric"] > 0
    assert file_sha256(root / "data" / "core" / "measurements.csv") == before_core
    assert file_sha256(root / "data" / "parametric" / "measurements.csv") == before_param
