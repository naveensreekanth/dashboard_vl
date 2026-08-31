"""Dataset version metadata validation."""

from __future__ import annotations

from typing import Any

from dtl_agent.validation.report import CheckResult


def validate_version_metadata(
    metadata: dict[str, Any],
    *,
    expected_version: str,
    expected_lot_count: int | None = None,
    expected_die_count: int | None = None,
    expected_row_count: int | None = None,
    row_count_key: str = "measurement_row_count",
    observed_lot_count: int | None = None,
    observed_die_count: int | None = None,
    observed_row_count: int | None = None,
    domain: str,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    version = str(metadata.get("dataset_version", ""))
    checks.append(
        CheckResult(
            name=f"version_id:{domain}",
            passed=version == expected_version,
            message=(
                f"version={version}"
                if version == expected_version
                else f"version {version!r} != {expected_version!r}"
            ),
            details={"metadata_version": version, "expected": expected_version},
        )
    )

    def _compare(name: str, meta_value: Any, observed: int | None, expected: int | None) -> None:
        problems: list[str] = []
        if expected is not None and meta_value is not None and int(meta_value) != expected:
            problems.append(f"metadata {meta_value} != expected {expected}")
        if observed is not None and meta_value is not None and int(meta_value) != observed:
            problems.append(f"metadata {meta_value} != observed {observed}")
        if observed is not None and expected is not None and observed != expected:
            problems.append(f"observed {observed} != expected {expected}")
        checks.append(
            CheckResult(
                name=name,
                passed=not problems,
                message="counts aligned" if not problems else "; ".join(problems),
                details={
                    "metadata": meta_value,
                    "observed": observed,
                    "expected": expected,
                },
            )
        )

    if expected_lot_count is not None or observed_lot_count is not None:
        _compare(
            f"version_lot_count:{domain}",
            metadata.get("lot_count"),
            observed_lot_count,
            expected_lot_count,
        )
    if expected_die_count is not None or observed_die_count is not None:
        _compare(
            f"version_die_count:{domain}",
            metadata.get("die_count"),
            observed_die_count,
            expected_die_count,
        )
    if expected_row_count is not None or observed_row_count is not None:
        _compare(
            f"version_row_count:{domain}",
            metadata.get(row_count_key),
            observed_row_count,
            expected_row_count,
        )
    return checks
