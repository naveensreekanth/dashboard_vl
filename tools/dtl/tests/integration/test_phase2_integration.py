"""Phase 2 integration tests against real allowlisted datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.canonical.dataset import build_canonical_dataset
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.validation.phase2 import run_phase2_validation, validate_canonical_dataset
from dtl_agent.validation.pipeline import validate_bundle


@pytest.mark.integration
def test_real_canonical_linkage_and_grains() -> None:
    root = default_project_root()
    if not (root / "data" / "core" / "measurements.csv").exists():
        pytest.skip("real agent-input data not present")
    core = load_core(root, materialize_measurements=False)
    parametric = load_parametric(root, materialize_measurements=False)
    bundle = validate_bundle(core, parametric)
    assert bundle.ok
    canonical = build_canonical_dataset(bundle)

    assert canonical.core_grain.description == "lot × die × pattern × test"
    assert canonical.parametric_grain.description == "lot × die × condition × test"
    assert canonical.measurements_are_separated()

    link = canonical.linkage.summary()
    assert link["linked_lot_count"] == 31
    assert link["linked_lot_die_pair_count"] == 1550
    assert link["parametric_only_lot_count"] == 12
    assert link["parametric_only_die_count"] == 600
    assert link["core_only_lot_count"] == 0

    assert len(canonical.get_conditions()) == 4
    ir = canonical.get_current_limit("core", test_id="T_IR_DROP_MV")
    assert ir.current_limit == 25.0
    assert ir.source_status == "SOURCE_CONFIRMED"
    vmin = canonical.get_current_limit("parametric", test_id="T_VMIN")
    assert vmin.current_limit == 0.85
    assert vmin.source_status == "SYNTHETIC_ASSUMED"

    # Linked die queryable; measurements remain separate grains
    pair = next(iter(sorted(canonical.linkage.linked_lot_die_pairs)))
    linked = canonical.get_linked_die(pair[0], pair[1])
    assert linked.cross_domain_available is True
    core_n = sum(
        1 for _ in canonical.get_core_measurements(lot_id=pair[0], die_id=pair[1])
    )
    par_n = sum(
        1
        for _ in canonical.get_parametric_measurements(lot_id=pair[0], die_id=pair[1])
    )
    assert core_n == 200 * 5  # 200 patterns × 5 tests
    assert par_n == 4 * 7  # 4 conditions × 7 parametric tests

    # Parametric-only lot has no core data
    only = sorted(canonical.linkage.parametric_only_lots)[0]
    assert canonical.cross_domain_available(only) is False
    assert canonical.has_core_data(only) is False


@pytest.mark.integration
def test_real_phase2_validation_pass() -> None:
    root = default_project_root()
    if not (root / "data" / "core" / "measurements.csv").exists():
        pytest.skip("real agent-input data not present")
    _canonical, report, _bundle = run_phase2_validation(root)
    assert report.passed, report.to_dict()
