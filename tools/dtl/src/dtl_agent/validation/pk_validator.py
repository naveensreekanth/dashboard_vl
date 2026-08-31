"""Primary-key uniqueness validation (streaming-capable)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dtl_agent.validation.report import CheckResult


def validate_primary_keys(
    rows: Iterable[dict[str, str]],
    pk_columns: Sequence[str],
    *,
    check_name: str = "measurement_pk",
) -> CheckResult:
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    missing_id_rows = 0
    total = 0
    examples: list[str] = []

    for row in rows:
        total += 1
        values = tuple(str(row.get(c, "")).strip() for c in pk_columns)
        if any(v == "" for v in values):
            missing_id_rows += 1
            continue
        if values in seen:
            duplicates += 1
            if len(examples) < 5:
                examples.append(repr(values))
        else:
            seen.add(values)

    passed = duplicates == 0 and missing_id_rows == 0
    message = (
        f"PK unique over {total} rows"
        if passed
        else (
            f"PK failures: duplicates={duplicates}, "
            f"missing_id_rows={missing_id_rows}, total={total}"
        )
    )
    return CheckResult(
        name=check_name,
        passed=passed,
        message=message,
        details={
            "total_rows": total,
            "unique_keys": len(seen),
            "duplicate_pk_count": duplicates,
            "missing_id_rows": missing_id_rows,
            "pk_columns": list(pk_columns),
            "duplicate_examples": examples,
        },
    )
