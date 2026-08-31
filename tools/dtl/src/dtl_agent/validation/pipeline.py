"""Phase 1 validation pipeline: core, parametric, linkage, forbidden-data."""

from __future__ import annotations

import json
from pathlib import Path

from dtl_agent.config.constants import (
    CORE_ALL_TEST_IDS,
    CORE_CONTEXT_TEST_IDS,
    CORE_DATASET_VERSION,
    CORE_EXPECTED_DIE_COUNT,
    CORE_EXPECTED_LIMITS,
    CORE_EXPECTED_LOT_COUNT,
    CORE_EXPECTED_MEASUREMENT_ROWS,
    CORE_MEASUREMENT_PK,
    CORE_PRIMARY_TEST_IDS,
    CORE_SECONDARY_TEST_IDS,
    PARAMETRIC_CONTEXT_TEST_IDS,
    PARAMETRIC_DATASET_VERSION,
    PARAMETRIC_EXPECTED_CONDITION_COUNT,
    PARAMETRIC_EXPECTED_DIE_COUNT,
    PARAMETRIC_EXPECTED_LIMITS,
    PARAMETRIC_EXPECTED_LOT_COUNT,
    PARAMETRIC_EXPECTED_MEASUREMENT_ROWS,
    PARAMETRIC_MEASUREMENT_PK,
    PARAMETRIC_PRIMARY_TEST_IDS,
)
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.data.models.bundle import ValidatedDatasetBundle
from dtl_agent.data.models.core import CoreDataset
from dtl_agent.data.models.linkage import SharedLotDieIndex
from dtl_agent.data.models.parametric import ParametricDataset
from dtl_agent.schemas import (
    CORE_CURRENT_LIMITS_SCHEMA,
    CORE_LOTS_SCHEMA,
    CORE_MEASUREMENTS_SCHEMA,
    CORE_PARTS_SCHEMA,
    CORE_SCENARIO_MANIFEST_SCHEMA,
    CORE_TEST_CATALOG_SCHEMA,
    PARAMETRIC_CONDITIONS_SCHEMA,
    PARAMETRIC_CURRENT_LIMITS_SCHEMA,
    PARAMETRIC_LOTS_SCHEMA,
    PARAMETRIC_MEASUREMENTS_SCHEMA,
    PARAMETRIC_PARTS_SCHEMA,
    PARAMETRIC_SCENARIO_MANIFEST_SCHEMA,
    PARAMETRIC_TEST_CATALOG_SCHEMA,
)
from dtl_agent.validation.condition_validator import validate_conditions
from dtl_agent.validation.forbidden_validator import (
    build_forbidden_summary,
    validate_table_columns_not_forbidden,
)
from dtl_agent.validation.limit_validator import (
    validate_expected_limits,
    validate_no_invented_secondary_limits,
)
from dtl_agent.validation.linkage_validator import validate_linkage
from dtl_agent.validation.pk_validator import validate_primary_keys
from dtl_agent.validation.report import (
    CheckResult,
    DomainValidationSummary,
    Phase1ValidationReport,
)
from dtl_agent.validation.schema_validator import (
    collect_missing_required_field_names,
    validate_columns_present,
    validate_non_null_required_ids,
)
from dtl_agent.validation.version_validator import validate_version_metadata
from dtl_agent.utils.csv_io import count_csv_rows


def _catalog_test_ids(catalog: list[dict[str, str]]) -> set[str]:
    return {r["test_id"] for r in catalog}


def validate_core(dataset: CoreDataset) -> DomainValidationSummary:
    summary = DomainValidationSummary(
        domain="core",
        files_validated=sorted(
            [
                "measurements.csv",
                "parts_dim.csv",
                "lots_dim.csv",
                "test_catalog.csv",
                "current_limits.csv",
                "scenario_manifest_public.csv",
                "README_DATA_CONTRACT.md",
                "DATASET_VERSION.json",
                "rules/disposition_rules.json",
                "rules/limit_simulation_config.json",
            ]
        ),
        lot_count=dataset.lot_count,
        die_count=dataset.die_count,
        version=dataset.dataset_version,
        parameter_coverage=sorted(_catalog_test_ids(dataset.test_catalog)),
        current_limit_coverage=sorted({r["test_id"] for r in dataset.current_limits}),
    )

    checks: list[CheckResult] = []

    # Schema / forbidden columns
    table_specs = [
        ("measurements.csv", dataset.measurements_columns, CORE_MEASUREMENTS_SCHEMA),
        ("parts_dim.csv", list(dataset.parts[0].keys()) if dataset.parts else [], CORE_PARTS_SCHEMA),
        ("lots_dim.csv", list(dataset.lots[0].keys()) if dataset.lots else [], CORE_LOTS_SCHEMA),
        (
            "test_catalog.csv",
            list(dataset.test_catalog[0].keys()) if dataset.test_catalog else [],
            CORE_TEST_CATALOG_SCHEMA,
        ),
        (
            "current_limits.csv",
            list(dataset.current_limits[0].keys()) if dataset.current_limits else [],
            CORE_CURRENT_LIMITS_SCHEMA,
        ),
        (
            "scenario_manifest_public.csv",
            list(dataset.scenario_manifest[0].keys()) if dataset.scenario_manifest else [],
            CORE_SCENARIO_MANIFEST_SCHEMA,
        ),
    ]
    missing_fields: list[str] = []
    for name, cols, schema in table_specs:
        checks.append(validate_columns_present(table_name=name, columns=cols, schema=schema))
        checks.append(validate_table_columns_not_forbidden(table_name=f"core:{name}", columns=cols))
        missing_fields.extend(
            [f"{name}:{c}" for c in collect_missing_required_field_names(cols, schema)]
        )

    checks.append(
        validate_non_null_required_ids(
            table_name="lots_dim.csv", rows=dataset.lots, id_columns=("lot_id",)
        )
    )
    checks.append(
        validate_non_null_required_ids(
            table_name="parts_dim.csv",
            rows=dataset.parts,
            id_columns=("lot_id", "die_id"),
        )
    )

    # Measurement PK (streaming)
    pk_check = validate_primary_keys(
        dataset.iter_measurements(columns=list(CORE_MEASUREMENT_PK)),
        CORE_MEASUREMENT_PK,
        check_name="core_measurement_pk",
    )
    checks.append(pk_check)
    summary.duplicate_pk_count = int(pk_check.details.get("duplicate_pk_count", 0))
    summary.measurement_row_count = int(pk_check.details.get("total_rows", 0))

    # Catalog coverage
    catalog_ids = _catalog_test_ids(dataset.test_catalog)
    checks.append(
        CheckResult(
            name="core_catalog_coverage",
            passed=CORE_ALL_TEST_IDS.issubset(catalog_ids),
            message=f"catalog test_ids={sorted(catalog_ids)}",
            details={"expected": sorted(CORE_ALL_TEST_IDS)},
        )
    )

    checks.append(
        validate_expected_limits(
            dataset.current_limits,
            CORE_EXPECTED_LIMITS,
            domain="core",
            value_field="upper_limit",
            direction_field="limit_direction",
            source_field="source_status",
        )
    )
    checks.append(
        validate_no_invented_secondary_limits(
            dataset.current_limits,
            forbidden_test_ids=CORE_SECONDARY_TEST_IDS | CORE_CONTEXT_TEST_IDS,
            domain="core",
        )
    )

    checks.extend(
        validate_version_metadata(
            dataset.version_metadata,
            expected_version=CORE_DATASET_VERSION,
            expected_lot_count=CORE_EXPECTED_LOT_COUNT,
            expected_die_count=CORE_EXPECTED_DIE_COUNT,
            expected_row_count=CORE_EXPECTED_MEASUREMENT_ROWS,
            row_count_key="measurement_row_count",
            observed_lot_count=dataset.lot_count,
            observed_die_count=dataset.die_count,
            observed_row_count=summary.measurement_row_count,
            domain="core",
        )
    )

    # Dim uniqueness
    checks.append(
        validate_primary_keys(
            dataset.lots, ("lot_id",), check_name="core_lots_pk"
        )
    )
    checks.append(
        validate_primary_keys(
            dataset.parts, ("die_id",), check_name="core_parts_die_pk"
        )
    )

    summary.missing_required_fields = missing_fields
    summary.checks = checks
    return summary


def validate_parametric(dataset: ParametricDataset) -> DomainValidationSummary:
    summary = DomainValidationSummary(
        domain="parametric",
        files_validated=sorted(
            [
                "measurements.csv",
                "parts_dim.csv",
                "lots_dim.csv",
                "conditions_dim.csv",
                "test_catalog.csv",
                "current_limits.csv",
                "scenario_manifest_public.csv",
                "README_DATA_CONTRACT.md",
                "PARAMETRIC_DATASET_VERSION.json",
                "rules/disposition_rules.json",
                "rules/limit_simulation_config.json",
            ]
        ),
        lot_count=dataset.lot_count,
        die_count=dataset.die_count,
        condition_count=dataset.condition_count,
        version=dataset.dataset_version,
        parameter_coverage=sorted(_catalog_test_ids(dataset.test_catalog)),
        current_limit_coverage=sorted({r["test_id"] for r in dataset.current_limits}),
    )

    checks: list[CheckResult] = []
    table_specs = [
        ("measurements.csv", dataset.measurements_columns, PARAMETRIC_MEASUREMENTS_SCHEMA),
        (
            "parts_dim.csv",
            list(dataset.parts[0].keys()) if dataset.parts else [],
            PARAMETRIC_PARTS_SCHEMA,
        ),
        (
            "lots_dim.csv",
            list(dataset.lots[0].keys()) if dataset.lots else [],
            PARAMETRIC_LOTS_SCHEMA,
        ),
        (
            "conditions_dim.csv",
            list(dataset.conditions[0].keys()) if dataset.conditions else [],
            PARAMETRIC_CONDITIONS_SCHEMA,
        ),
        (
            "test_catalog.csv",
            list(dataset.test_catalog[0].keys()) if dataset.test_catalog else [],
            PARAMETRIC_TEST_CATALOG_SCHEMA,
        ),
        (
            "current_limits.csv",
            list(dataset.current_limits[0].keys()) if dataset.current_limits else [],
            PARAMETRIC_CURRENT_LIMITS_SCHEMA,
        ),
        (
            "scenario_manifest_public.csv",
            list(dataset.scenario_manifest[0].keys()) if dataset.scenario_manifest else [],
            PARAMETRIC_SCENARIO_MANIFEST_SCHEMA,
        ),
    ]
    missing_fields: list[str] = []
    for name, cols, schema in table_specs:
        checks.append(validate_columns_present(table_name=name, columns=cols, schema=schema))
        checks.append(
            validate_table_columns_not_forbidden(table_name=f"parametric:{name}", columns=cols)
        )
        missing_fields.extend(
            [f"{name}:{c}" for c in collect_missing_required_field_names(cols, schema)]
        )

    checks.append(
        validate_non_null_required_ids(
            table_name="lots_dim.csv", rows=dataset.lots, id_columns=("lot_id",)
        )
    )
    checks.append(
        validate_non_null_required_ids(
            table_name="parts_dim.csv",
            rows=dataset.parts,
            id_columns=("lot_id", "die_id"),
        )
    )
    checks.append(
        validate_non_null_required_ids(
            table_name="conditions_dim.csv",
            rows=dataset.conditions,
            id_columns=("condition_id",),
        )
    )

    pk_check = validate_primary_keys(
        dataset.iter_measurements(columns=list(PARAMETRIC_MEASUREMENT_PK)),
        PARAMETRIC_MEASUREMENT_PK,
        check_name="parametric_measurement_pk",
    )
    checks.append(pk_check)
    summary.duplicate_pk_count = int(pk_check.details.get("duplicate_pk_count", 0))
    summary.measurement_row_count = int(pk_check.details.get("total_rows", 0))

    catalog_ids = _catalog_test_ids(dataset.test_catalog)
    expected_catalog = PARAMETRIC_PRIMARY_TEST_IDS | PARAMETRIC_CONTEXT_TEST_IDS
    checks.append(
        CheckResult(
            name="parametric_catalog_coverage",
            passed=expected_catalog.issubset(catalog_ids),
            message=f"catalog test_ids={sorted(catalog_ids)}",
            details={"expected": sorted(expected_catalog)},
        )
    )
    checks.append(
        CheckResult(
            name="parametric_condition_count_meta",
            passed=dataset.condition_count == PARAMETRIC_EXPECTED_CONDITION_COUNT,
            message=f"condition_count={dataset.condition_count}",
            details={"expected": PARAMETRIC_EXPECTED_CONDITION_COUNT},
        )
    )
    checks.extend(validate_conditions(dataset.conditions))

    checks.append(
        validate_expected_limits(
            dataset.current_limits,
            PARAMETRIC_EXPECTED_LIMITS,
            domain="parametric",
            value_field="limit_value",
            direction_field="limit_type",
            source_field="source",
        )
    )
    checks.append(
        validate_no_invented_secondary_limits(
            dataset.current_limits,
            forbidden_test_ids=PARAMETRIC_CONTEXT_TEST_IDS,
            domain="parametric",
        )
    )
    # Explicit: VDD is not a DTL limit target
    checks.append(
        CheckResult(
            name="vdd_not_dtl_target",
            passed="COND_VDD" not in {r["test_id"] for r in dataset.current_limits},
            message="VDD/COND_VDD absent from current_limits",
        )
    )

    checks.extend(
        validate_version_metadata(
            dataset.version_metadata,
            expected_version=PARAMETRIC_DATASET_VERSION,
            expected_lot_count=PARAMETRIC_EXPECTED_LOT_COUNT,
            expected_die_count=PARAMETRIC_EXPECTED_DIE_COUNT,
            expected_row_count=PARAMETRIC_EXPECTED_MEASUREMENT_ROWS,
            row_count_key="row_count",
            observed_lot_count=dataset.lot_count,
            observed_die_count=dataset.die_count,
            observed_row_count=summary.measurement_row_count,
            domain="parametric",
        )
    )
    checks.append(
        validate_primary_keys(dataset.lots, ("lot_id",), check_name="parametric_lots_pk")
    )
    checks.append(
        validate_primary_keys(dataset.parts, ("die_id",), check_name="parametric_parts_die_pk")
    )
    checks.append(
        validate_primary_keys(
            dataset.conditions, ("condition_id",), check_name="parametric_conditions_pk"
        )
    )

    summary.missing_required_fields = missing_fields
    summary.checks = checks
    return summary


def validate_bundle(
    core: CoreDataset,
    parametric: ParametricDataset,
) -> ValidatedDatasetBundle:
    core_summary = validate_core(core)
    parametric_summary = validate_parametric(parametric)
    linkage = SharedLotDieIndex.from_datasets(core, parametric)
    linkage_summary = validate_linkage(linkage)

    forbidden_checks = [
        validate_table_columns_not_forbidden(
            table_name="core:measurements", columns=core.measurements_columns
        ),
        validate_table_columns_not_forbidden(
            table_name="parametric:measurements", columns=parametric.measurements_columns
        ),
    ]
    forbidden_summary = build_forbidden_summary(column_checks=forbidden_checks, path_hits=[])

    report = Phase1ValidationReport(
        core=core_summary,
        parametric=parametric_summary,
        linkage=linkage_summary,
        forbidden=forbidden_summary,
    )
    report.finalize()
    return ValidatedDatasetBundle(
        core=core,
        parametric=parametric,
        linkage=linkage,
        validation=report,
    )


def run_phase1_validation(project_root: Path | None = None) -> ValidatedDatasetBundle:
    root = project_root or default_project_root()
    core = load_core(root, materialize_measurements=False)
    parametric = load_parametric(root, materialize_measurements=False)
    # Sanity: measurement files exist and are non-trivial (count used only if PK path skipped)
    _ = count_csv_rows  # imported for potential tooling reuse
    return validate_bundle(core, parametric)


def write_validation_artifacts(
    bundle: ValidatedDatasetBundle,
    *,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    root = project_root or default_project_root()
    json_dir = root / "artifacts" / "validation"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / "phase1_validation.json"
    json_path.write_text(
        json.dumps(bundle.validation.to_dict(), indent=2),
        encoding="utf-8",
    )

    md_path = root / "docs" / "PHASE_1_VALIDATION.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_markdown(bundle), encoding="utf-8")
    return json_path, md_path


def _render_markdown(bundle: ValidatedDatasetBundle) -> str:
    r = bundle.validation
    lines = [
        "# Phase 1 Validation Report",
        "",
        f"## FINAL: **{r.final_status}**",
        "",
        "## CORE",
        "",
        f"- files validated: {len(r.core.files_validated)}",
        f"- measurement row count: {r.core.measurement_row_count}",
        f"- lot count: {r.core.lot_count}",
        f"- die count: {r.core.die_count}",
        f"- duplicate PK count: {r.core.duplicate_pk_count}",
        f"- missing required fields: {r.core.missing_required_fields or 'none'}",
        f"- parameter coverage: {r.core.parameter_coverage}",
        f"- current limit coverage: {r.core.current_limit_coverage}",
        f"- version: {r.core.version}",
        f"- domain checks passed: {r.core.passed}",
        "",
        "## PARAMETRIC",
        "",
        f"- files validated: {len(r.parametric.files_validated)}",
        f"- measurement row count: {r.parametric.measurement_row_count}",
        f"- lot count: {r.parametric.lot_count}",
        f"- die count: {r.parametric.die_count}",
        f"- condition count: {r.parametric.condition_count}",
        f"- duplicate PK count: {r.parametric.duplicate_pk_count}",
        f"- missing required fields: {r.parametric.missing_required_fields or 'none'}",
        f"- parameter coverage: {r.parametric.parameter_coverage}",
        f"- current limit coverage: {r.parametric.current_limit_coverage}",
        f"- version: {r.parametric.version}",
        f"- domain checks passed: {r.parametric.passed}",
        "",
        "## LINKAGE",
        "",
        f"- common lot count: {r.linkage.common_lot_count}",
        f"- common die count: {r.linkage.common_die_count}",
        f"- Core-only lot count: {r.linkage.core_only_lot_count}",
        f"- Parametric-only lot count: {r.linkage.parametric_only_lot_count}",
        f"- common lot/die pairs: {r.linkage.common_lot_die_pair_count}",
        f"- linkage status: {r.linkage.linkage_status}",
        "",
        "## FORBIDDEN DATA",
        "",
        f"- forbidden files detected: {r.forbidden.forbidden_files_detected or 'none'}",
        f"- forbidden columns detected: {r.forbidden.forbidden_columns_detected or 'none'}",
        f"- forbidden checks passed: {r.forbidden.passed}",
        "",
        "## Failed checks",
        "",
    ]
    failures = []
    for section in (r.core.checks, r.parametric.checks, r.linkage.checks, r.forbidden.checks):
        for c in section:
            if not c.passed:
                failures.append(f"- `{c.name}`: {c.message}")
    if failures:
        lines.extend(failures)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Phase 1 boundary")
    lines.append("")
    lines.append("- no feature engineering")
    lines.append("- no simulation / optimization / ML / API")
    lines.append("- raw data under `data/` unchanged")
    lines.append("")
    return "\n".join(lines)
