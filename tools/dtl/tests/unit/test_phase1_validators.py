"""Unit tests: loaders, schema, PK, limits, version, conditions, forbidden."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.config.allowlists import CORE_ALLOWLIST
from dtl_agent.config.constants import CORE_EXPECTED_LIMITS, PARAMETRIC_EXPECTED_LIMITS
from dtl_agent.data.loaders.core_loader import CoreDataLoader
from dtl_agent.data.loaders.parametric_loader import ParametricDataLoader
from dtl_agent.data.repositories.allowlist_repository import (
    AllowlistRepository,
    AllowlistViolation,
)
from dtl_agent.schemas import CORE_MEASUREMENTS_SCHEMA
from dtl_agent.validation.condition_validator import validate_conditions
from dtl_agent.validation.forbidden_validator import (
    find_forbidden_columns,
    validate_table_columns_not_forbidden,
)
from dtl_agent.validation.limit_validator import validate_expected_limits
from dtl_agent.validation.pk_validator import validate_primary_keys
from dtl_agent.validation.schema_validator import validate_columns_present
from dtl_agent.validation.version_validator import validate_version_metadata
from tests.conftest import minimal_core_fixture, minimal_parametric_fixture, write_csv


def test_core_loader_loads_allowlisted_fixture(tmp_path: Path) -> None:
    core_root = minimal_core_fixture(tmp_path)
    ds = CoreDataLoader(core_root).load(materialize_measurements=True)
    assert ds.lot_count == 1
    assert ds.die_count == 1
    assert ds.dataset_version == "DTL_DATASET_V1"
    assert ds.measurements is not None and len(ds.measurements) == 1


def test_parametric_loader_loads_allowlisted_fixture(tmp_path: Path) -> None:
    par_root = minimal_parametric_fixture(tmp_path)
    ds = ParametricDataLoader(par_root).load(materialize_measurements=True)
    assert ds.lot_count == 2
    assert ds.condition_count == 4
    assert ds.dataset_version == "DTL_PARAMETRIC_DATASET_V1"


def test_core_loader_rejects_non_allowlisted_path(tmp_path: Path) -> None:
    core_root = minimal_core_fixture(tmp_path)
    repo = AllowlistRepository(core_root, CORE_ALLOWLIST, domain="core")
    with pytest.raises(AllowlistViolation):
        repo.resolve("evaluation/ground_truth_optimal_limits.csv")


def test_schema_validation_fails_on_missing_column() -> None:
    result = validate_columns_present(
        table_name="measurements.csv",
        columns=["lot_id", "die_id"],
        schema=CORE_MEASUREMENTS_SCHEMA,
    )
    assert result.passed is False
    assert "pattern_id" in result.message


def test_pk_validation_detects_duplicates() -> None:
    rows = [
        {"lot_id": "L", "die_id": "D", "pattern_id": "1", "test_id": "T"},
        {"lot_id": "L", "die_id": "D", "pattern_id": "1", "test_id": "T"},
    ]
    result = validate_primary_keys(
        rows, ("lot_id", "die_id", "pattern_id", "test_id")
    )
    assert result.passed is False
    assert result.details["duplicate_pk_count"] == 1


def test_pk_validation_detects_missing_ids() -> None:
    rows = [{"lot_id": "L", "die_id": "", "pattern_id": "1", "test_id": "T"}]
    result = validate_primary_keys(
        rows, ("lot_id", "die_id", "pattern_id", "test_id")
    )
    assert result.passed is False
    assert result.details["missing_id_rows"] == 1


def test_limit_validation_core_ok() -> None:
    rows = [
        {
            "test_id": "T_IR_DROP_MV",
            "parameter": "ir_drop",
            "unit": "mV",
            "upper_limit": "25.0",
            "limit_direction": "UPPER",
            "source_status": "SOURCE_CONFIRMED",
        },
        {
            "test_id": "T_THERMAL_C",
            "parameter": "thermal",
            "unit": "°C",
            "upper_limit": "60.0",
            "limit_direction": "UPPER",
            "source_status": "SOURCE_CONFIRMED",
        },
    ]
    result = validate_expected_limits(
        rows,
        CORE_EXPECTED_LIMITS,
        domain="core",
        value_field="upper_limit",
        direction_field="limit_direction",
        source_field="source_status",
    )
    assert result.passed is True


def test_limit_validation_fails_wrong_value() -> None:
    rows = [
        {
            "test_id": "T_VMIN",
            "parameter": "VMIN",
            "unit": "V",
            "limit_value": "0.99",
            "limit_type": "UPPER",
            "source": "SYNTHETIC_ASSUMED",
        }
    ]
    # Incomplete set should fail for missing rows; also wrong value for VMIN
    result = validate_expected_limits(
        rows,
        PARAMETRIC_EXPECTED_LIMITS,
        domain="parametric",
        value_field="limit_value",
        direction_field="limit_type",
        source_field="source",
    )
    assert result.passed is False


def test_version_validation_mismatch_fails() -> None:
    checks = validate_version_metadata(
        {"dataset_version": "WRONG", "lot_count": 1},
        expected_version="DTL_DATASET_V1",
        expected_lot_count=1,
        observed_lot_count=1,
        domain="core",
    )
    assert any(c.name == "version_id:core" and not c.passed for c in checks)


def test_condition_validation_fails_on_bad_id() -> None:
    rows = [
        {
            "condition_id": "COND_BAD",
            "temperature_c": "25.0",
            "vdd_applied": "1.0",
            "test_mode": "NOMINAL",
            "description": "x",
        }
    ]
    checks = validate_conditions(rows)
    assert any(not c.passed for c in checks)


def test_forbidden_columns_detected() -> None:
    hits = find_forbidden_columns(["lot_id", "latent_quality", "measurement_value"])
    assert "latent_quality" in hits
    result = validate_table_columns_not_forbidden(
        table_name="evil.csv",
        columns=["true_optimal_upper_limit", "die_id"],
    )
    assert result.passed is False


def test_readme_forbidden_names_are_not_columns(tmp_path: Path) -> None:
    """README may document forbidden names; only data columns are rejected."""
    core_root = minimal_core_fixture(tmp_path)
    text = (core_root / "README_DATA_CONTRACT.md").read_text(encoding="utf-8")
    assert "latent_quality" in text
    ds = CoreDataLoader(core_root).load(materialize_measurements=True)
    result = validate_table_columns_not_forbidden(
        table_name="measurements", columns=ds.measurements_columns
    )
    assert result.passed is True


def test_duplicate_measurement_pk_fails_loader_validation(tmp_path: Path) -> None:
    core_root = minimal_core_fixture(tmp_path)
    # Append duplicate PK row without modifying production data (fixture only)
    write_csv(
        core_root / "measurements.csv",
        [
            {
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D001",
                "pattern_id": "1",
                "test_id": "T_IR_DROP_MV",
                "test_name": "IR_DROP_MV",
                "parameter": "ir_drop",
                "measurement_value": "20.0",
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
            },
            {
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D001",
                "pattern_id": "1",
                "test_id": "T_IR_DROP_MV",
                "test_name": "IR_DROP_MV",
                "parameter": "ir_drop",
                "measurement_value": "21.0",
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
            },
        ],
    )
    ds = CoreDataLoader(core_root).load(materialize_measurements=True)
    pk = validate_primary_keys(
        ds.iter_measurements(),
        ("lot_id", "die_id", "pattern_id", "test_id"),
    )
    assert pk.passed is False
