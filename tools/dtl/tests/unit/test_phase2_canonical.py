"""Phase 2 unit tests for the canonical dual-grain layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.canonical.dataset import (
    CanonicalLookupError,
    build_canonical_from_datasets,
)
from dtl_agent.canonical.entities import CORE_GRAIN, PARAMETRIC_GRAIN
from dtl_agent.data.loaders.core_loader import CoreDataLoader
from dtl_agent.data.loaders.parametric_loader import ParametricDataLoader
from dtl_agent.data.models.linkage import SharedLotDieIndex
from tests.conftest import minimal_core_fixture, minimal_parametric_fixture


@pytest.fixture
def canonical_fixture(tmp_path: Path):
    core = CoreDataLoader(minimal_core_fixture(tmp_path)).load(
        materialize_measurements=True
    )
    parametric = ParametricDataLoader(minimal_parametric_fixture(tmp_path)).load(
        materialize_measurements=True
    )
    return build_canonical_from_datasets(core, parametric)


def test_canonical_core_entity_creation(canonical_fixture) -> None:
    lot = canonical_fixture.get_lot("LOT_A")
    assert lot.in_core is True
    assert lot.lot_id == "LOT_A"
    die = canonical_fixture.get_die("LOT_A", "LOT_A_D001")
    assert die.in_core is True
    assert die.core_metadata.get("status") == "PASS"


def test_canonical_parametric_entity_creation(canonical_fixture) -> None:
    lot = canonical_fixture.get_lot("LOT_P_ONLY")
    assert lot.in_parametric is True
    assert lot.in_core is False
    assert lot.cross_domain_available is False
    die = canonical_fixture.get_die("LOT_P_ONLY", "LOT_P_ONLY_D001")
    assert die.in_parametric is True
    assert die.cross_domain_available is False


def test_core_measurement_retrieval(canonical_fixture) -> None:
    rows = list(canonical_fixture.get_core_measurements(lot_id="LOT_A"))
    assert len(rows) == 1
    assert rows[0].pattern_id == "1"
    assert rows[0].test_id == "T_IR_DROP_MV"
    assert rows[0].natural_key == ("LOT_A", "LOT_A_D001", "1", "T_IR_DROP_MV")


def test_parametric_measurement_retrieval(canonical_fixture) -> None:
    rows = list(canonical_fixture.get_parametric_measurements(lot_id="LOT_A"))
    assert len(rows) == 1
    assert rows[0].condition_id == "COND_RT_NOM"
    assert rows[0].test_id == "T_IDDQ"
    assert rows[0].natural_key == ("LOT_A", "LOT_A_D001", "COND_RT_NOM", "T_IDDQ")


def test_condition_retrieval(canonical_fixture) -> None:
    conditions = canonical_fixture.get_conditions()
    assert len(conditions) == 4
    c = canonical_fixture.get_condition("COND_HOT_NOM")
    assert c.temperature_c == "85.0"
    assert c.vdd_applied == "1.0"


def test_test_catalog_retrieval(canonical_fixture) -> None:
    core_test = canonical_fixture.get_test_definition("core", "T_IR_DROP_MV")
    assert core_test.domain == "core"
    assert core_test.parameter == "ir_drop"
    par_test = canonical_fixture.get_test_definition("parametric", "T_IDDQ")
    assert par_test.domain == "parametric"
    assert par_test.dtl_eligible is True
    # Domains are separate — Cond_VDD only in parametric
    with pytest.raises(CanonicalLookupError):
        canonical_fixture.get_test_definition("core", "COND_VDD")


def test_current_limit_retrieval(canonical_fixture) -> None:
    ir = canonical_fixture.get_current_limit("core", test_id="T_IR_DROP_MV")
    assert ir.direction == "UPPER"
    assert ir.current_limit == 25.0
    assert ir.source_status == "SOURCE_CONFIRMED"
    iddq = canonical_fixture.get_current_limit("parametric", parameter="IDDQ")
    assert iddq.current_limit == 50.0
    assert iddq.source_status == "SYNTHETIC_ASSUMED"


def test_linked_lot_die_lookup(canonical_fixture) -> None:
    view = canonical_fixture.get_linked_die("LOT_A", "LOT_A_D001")
    assert view.cross_domain_available is True
    assert view.core_die is not None
    assert view.parametric_die is not None
    assert canonical_fixture.has_core_data("LOT_A", "LOT_A_D001")
    assert canonical_fixture.has_parametric_data("LOT_A", "LOT_A_D001")


def test_parametric_only_lot_lookup(canonical_fixture) -> None:
    assert "LOT_P_ONLY" in canonical_fixture.linkage.parametric_only_lots
    assert canonical_fixture.has_core_data("LOT_P_ONLY") is False
    assert canonical_fixture.has_parametric_data("LOT_P_ONLY") is True
    assert canonical_fixture.cross_domain_available("LOT_P_ONLY") is False


def test_core_only_behavior_when_no_parametric_die(canonical_fixture) -> None:
    # Fixture has no core-only lots; assert API still supports core-domain filters
    core_lots = list(canonical_fixture.iter_lots(domain="core"))
    assert any(l.lot_id == "LOT_A" for l in core_lots)
    assert canonical_fixture.has_core_data("LOT_A", "LOT_A_D001") is True


def test_cross_domain_availability_flag(canonical_fixture) -> None:
    assert canonical_fixture.cross_domain_available("LOT_A", "LOT_A_D001") is True
    assert canonical_fixture.cross_domain_available("LOT_P_ONLY", "LOT_P_ONLY_D001") is False


def test_natural_key_preservation(canonical_fixture) -> None:
    core = next(canonical_fixture.get_core_measurements())
    assert core.natural_key == (core.lot_id, core.die_id, core.pattern_id, core.test_id)
    par = next(canonical_fixture.get_parametric_measurements())
    assert par.natural_key == (par.lot_id, par.die_id, par.condition_id, par.test_id)


def test_grain_preservation(canonical_fixture) -> None:
    assert canonical_fixture.core_grain == CORE_GRAIN
    assert canonical_fixture.parametric_grain == PARAMETRIC_GRAIN
    assert canonical_fixture.core_measurements.grain == "lot × die × pattern × test"
    assert (
        canonical_fixture.parametric_measurements.grain
        == "lot × die × condition × test"
    )


def test_no_accidental_measurement_concatenation(canonical_fixture) -> None:
    assert canonical_fixture.measurements_are_separated() is True
    # Distinct view types / grains — no shared merged table attribute
    assert not hasattr(canonical_fixture, "measurements")
    assert canonical_fixture.core_measurements is not canonical_fixture.parametric_measurements


def test_shared_lot_die_index_reused(canonical_fixture) -> None:
    assert isinstance(canonical_fixture.linkage, SharedLotDieIndex)
    assert canonical_fixture.linkage.is_linked_lot_die("LOT_A", "LOT_A_D001")
