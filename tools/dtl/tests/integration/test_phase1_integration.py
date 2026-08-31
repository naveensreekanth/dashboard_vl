"""Integration-style tests for linkage and real allowlisted datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import CoreDataLoader, load_core
from dtl_agent.data.loaders.parametric_loader import ParametricDataLoader, load_parametric
from dtl_agent.data.models.linkage import SharedLotDieIndex
from dtl_agent.data.repositories.allowlist_repository import AllowlistViolation
from dtl_agent.config.allowlists import CORE_ALLOWLIST, PARAMETRIC_ALLOWLIST
from dtl_agent.data.repositories.allowlist_repository import AllowlistRepository
from dtl_agent.validation.linkage_validator import validate_linkage
from dtl_agent.validation.pipeline import validate_bundle
from tests.conftest import minimal_core_fixture, minimal_parametric_fixture


def test_shared_lot_die_index_on_fixtures(tmp_path: Path) -> None:
    core = CoreDataLoader(minimal_core_fixture(tmp_path)).load(materialize_measurements=True)
    par = ParametricDataLoader(minimal_parametric_fixture(tmp_path)).load(
        materialize_measurements=True
    )
    index = SharedLotDieIndex.from_datasets(core, par)
    assert index.linked_lots == frozenset({"LOT_A"})
    assert index.parametric_only_lots == frozenset({"LOT_P_ONLY"})
    assert index.core_only_lots == frozenset()
    assert index.is_linked_die("LOT_A_D001")
    assert index.cross_domain_features_available(lot_id="LOT_A") is True
    assert index.cross_domain_features_available(lot_id="LOT_P_ONLY") is False


def test_linkage_validator_fails_when_counts_wrong(tmp_path: Path) -> None:
    core = CoreDataLoader(minimal_core_fixture(tmp_path)).load(materialize_measurements=True)
    par = ParametricDataLoader(minimal_parametric_fixture(tmp_path)).load(
        materialize_measurements=True
    )
    index = SharedLotDieIndex.from_datasets(core, par)
    summary = validate_linkage(index)
    # Fixture is intentionally tiny vs production expected counts
    assert summary.passed is False


def test_allowlist_rejects_ground_truth_basename(tmp_path: Path) -> None:
    core_root = minimal_core_fixture(tmp_path)
    repo = AllowlistRepository(core_root, CORE_ALLOWLIST, domain="core")
    with pytest.raises(AllowlistViolation):
        repo.resolve("ground_truth_optimal_limits.csv")
    repo_p = AllowlistRepository(
        minimal_parametric_fixture(tmp_path), PARAMETRIC_ALLOWLIST, domain="parametric"
    )
    with pytest.raises(AllowlistViolation):
        repo_p.resolve("split_assignments.csv")


@pytest.mark.integration
def test_real_data_loaders_and_linkage() -> None:
    root = default_project_root()
    if not (root / "data" / "core" / "measurements.csv").exists():
        pytest.skip("real agent-input data not present")
    core = load_core(root, materialize_measurements=False)
    parametric = load_parametric(root, materialize_measurements=False)
    assert core.lot_count == 31
    assert core.die_count == 1550
    assert parametric.lot_count == 43
    assert parametric.die_count == 2150
    assert parametric.condition_count == 4
    index = SharedLotDieIndex.from_datasets(core, parametric)
    assert len(index.linked_lots) == 31
    assert len(index.linked_dies) == 1550
    assert len(index.parametric_only_lots) == 12
    assert len(index.parametric_only_dies) == 600
    assert len(index.core_only_lots) == 0


@pytest.mark.integration
def test_real_data_full_phase1_validation() -> None:
    root = default_project_root()
    if not (root / "data" / "core" / "measurements.csv").exists():
        pytest.skip("real agent-input data not present")
    core = load_core(root, materialize_measurements=False)
    parametric = load_parametric(root, materialize_measurements=False)
    bundle = validate_bundle(core, parametric)
    assert bundle.ok, bundle.validation.to_dict()
    assert bundle.validation.final_status == "PASS"
