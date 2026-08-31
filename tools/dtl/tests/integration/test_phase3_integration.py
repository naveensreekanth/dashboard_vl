"""Phase 3 integration tests on real allowlisted data."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.features.io_utils import file_sha256
from dtl_agent.features.pipeline import (
    run_feature_engineering,
    validate_phase3_features,
)


@pytest.mark.integration
def test_real_phase3_feature_engineering_and_gru_readiness() -> None:
    root = default_project_root()
    if not (root / "data" / "core" / "measurements.csv").exists():
        pytest.skip("real data missing")

    core_path = root / "data" / "core" / "measurements.csv"
    par_path = root / "data" / "parametric" / "measurements.csv"
    before = {str(core_path): file_sha256(core_path), str(par_path): file_sha256(par_path)}

    artifacts = run_feature_engineering(root)
    report = validate_phase3_features(artifacts)
    assert report["final_status"] == "PASS", report
    assert report["summary"]["gru_ready"] is True
    assert report["summary"]["sequence_contract"]["valid_sequences"] == 1550
    assert report["summary"]["sequence_contract"]["incomplete_sequences"] == 0
    assert report["summary"]["sequence_contract"]["feature_dimension"] == 5
    assert report["summary"]["cross_linked_die_features"] == 1550
    assert report["summary"]["registry_feature_count"] > 50

    # Determinism: second run yields identical die feature file content
    artifacts2 = run_feature_engineering(root)
    p1 = artifacts.paths["core_die"].read_text(encoding="utf-8")
    p2 = artifacts2.paths["core_die"].read_text(encoding="utf-8")
    assert p1 == p2

    after = {str(core_path): file_sha256(core_path), str(par_path): file_sha256(par_path)}
    assert before == after

    # Pattern ordering sample
    with artifacts.paths["core_pattern"].open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first = next(reader)
        assert first["pattern_id"] == "1"
        assert first["sequence_index"] == "0"
        assert first["ir_drop"] != ""
