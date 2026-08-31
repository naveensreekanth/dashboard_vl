"""Schema contracts for CSV columns (required vs optional nullability)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableSchema:
    name: str
    required_columns: tuple[str, ...]
    # Columns that may be entirely empty in the supplied synthetic data
    nullable_columns: frozenset[str] = frozenset()


CORE_MEASUREMENTS_SCHEMA = TableSchema(
    name="measurements.csv",
    required_columns=(
        "lot_id",
        "die_id",
        "pattern_id",
        "test_id",
        "test_name",
        "parameter",
        "measurement_value",
        "unit",
        "scenario_id",
        "scenario_family",
        "tester_id",
        "site_id",
        "pass_fail_pattern",
        "die_status",
        "generation_seed",
        "generator_version",
        "production_sequence",
    ),
    nullable_columns=frozenset({"defect_type", "start_time", "source_log"}),
)

CORE_PARTS_SCHEMA = TableSchema(
    name="parts_dim.csv",
    required_columns=(
        "lot_id",
        "die_id",
        "scenario_id",
        "scenario_family",
        "status",
        "tester_id",
        "generation_seed",
        "generator_version",
        "production_sequence",
    ),
    nullable_columns=frozenset({"defect_type", "start_time", "end_time", "source_log"}),
)

CORE_LOTS_SCHEMA = TableSchema(
    name="lots_dim.csv",
    required_columns=(
        "lot_id",
        "scenario_id",
        "scenario_family",
        "tester_id",
        "total_dies",
        "production_sequence",
        "generation_seed",
        "generator_version",
    ),
    nullable_columns=frozenset({"defect_type", "start_time", "end_time"}),
)

CORE_TEST_CATALOG_SCHEMA = TableSchema(
    name="test_catalog.csv",
    required_columns=(
        "test_id",
        "test_name",
        "parameter",
        "unit",
        "direction",
        "source_status",
        "dtl_eligible",
        "optimization_priority",
    ),
)

CORE_CURRENT_LIMITS_SCHEMA = TableSchema(
    name="current_limits.csv",
    required_columns=(
        "test_id",
        "test_name",
        "parameter",
        "unit",
        "upper_limit",
        "limit_direction",
        "limit_type",
        "source_status",
        "active",
    ),
    nullable_columns=frozenset({"lower_limit", "nominal_value", "effective_date"}),
)

CORE_SCENARIO_MANIFEST_SCHEMA = TableSchema(
    name="scenario_manifest_public.csv",
    required_columns=(
        "scenario_id",
        "scenario_family",
        "lot_id",
        "production_sequence",
        "tester_id",
        "target_parameters",
        "generation_seed",
        "generator_version",
    ),
)

PARAMETRIC_MEASUREMENTS_SCHEMA = TableSchema(
    name="measurements.csv",
    required_columns=(
        "dataset_version",
        "scenario_id",
        "scenario_family",
        "lot_id",
        "die_id",
        "condition_id",
        "tester_id",
        "site_id",
        "temperature_c",
        "vdd_applied",
        "test_mode",
        "test_id",
        "parameter",
        "measurement_value",
        "unit",
        "limit_type",
        "generation_seed",
        "generator_version",
        "pass_fail_condition",
    ),
)

PARAMETRIC_PARTS_SCHEMA = TableSchema(
    name="parts_dim.csv",
    required_columns=(
        "lot_id",
        "die_id",
        "scenario_id",
        "scenario_family",
        "tester_id",
        "site_id",
        "v1_link",
        "dataset_version",
        "generation_seed",
        "generator_version",
    ),
)

PARAMETRIC_LOTS_SCHEMA = TableSchema(
    name="lots_dim.csv",
    required_columns=(
        "lot_id",
        "scenario_id",
        "scenario_family",
        "production_sequence",
        "tester_id",
        "v1_link",
        "total_dies",
        "generation_seed",
        "generator_version",
        "dataset_version",
    ),
)

PARAMETRIC_CONDITIONS_SCHEMA = TableSchema(
    name="conditions_dim.csv",
    required_columns=(
        "condition_id",
        "temperature_c",
        "vdd_applied",
        "test_mode",
        "description",
    ),
)

PARAMETRIC_TEST_CATALOG_SCHEMA = TableSchema(
    name="test_catalog.csv",
    required_columns=(
        "test_id",
        "parameter",
        "test_name",
        "unit",
        "limit_type",
        "dtl_eligible",
        "priority",
        "condition_dependent",
        "synthetic_source",
        "role",
    ),
)

PARAMETRIC_CURRENT_LIMITS_SCHEMA = TableSchema(
    name="current_limits.csv",
    required_columns=(
        "test_id",
        "parameter",
        "limit_type",
        "limit_value",
        "unit",
        "source",
        "note",
    ),
)

PARAMETRIC_SCENARIO_MANIFEST_SCHEMA = TableSchema(
    name="scenario_manifest_public.csv",
    required_columns=("scenario_id", "scenario_family", "lot_count", "mechanism"),
)
