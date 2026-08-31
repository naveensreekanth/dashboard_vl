"""Table schema / required-column validation."""

from __future__ import annotations

from dtl_agent.schemas import TableSchema
from dtl_agent.validation.report import CheckResult


def validate_columns_present(
    *,
    table_name: str,
    columns: list[str],
    schema: TableSchema,
) -> CheckResult:
    missing = [c for c in schema.required_columns if c not in columns]
    return CheckResult(
        name=f"schema_columns:{table_name}",
        passed=not missing,
        message=(
            "required columns present"
            if not missing
            else f"missing required columns: {missing}"
        ),
        details={"missing": missing, "observed": columns},
    )


def validate_non_null_required_ids(
    *,
    table_name: str,
    rows: list[dict[str, str]],
    id_columns: tuple[str, ...],
) -> CheckResult:
    missing_examples: list[dict[str, str]] = []
    missing_count = 0
    for idx, row in enumerate(rows):
        bad = [c for c in id_columns if not str(row.get(c, "")).strip()]
        if bad:
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append({"row_index": str(idx), "columns": ",".join(bad)})
    return CheckResult(
        name=f"required_ids:{table_name}",
        passed=missing_count == 0,
        message=(
            "required identifiers present"
            if missing_count == 0
            else f"{missing_count} rows missing required identifiers"
        ),
        details={"missing_count": missing_count, "examples": missing_examples},
    )


def collect_missing_required_field_names(
    columns: list[str], schema: TableSchema
) -> list[str]:
    return [c for c in schema.required_columns if c not in columns]
