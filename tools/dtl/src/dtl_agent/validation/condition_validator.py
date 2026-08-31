"""Parametric condition dimension validation."""

from __future__ import annotations

from dtl_agent.config.constants import EXPECTED_CONDITION_IDS
from dtl_agent.validation.report import CheckResult


def validate_conditions(conditions: list[dict[str, str]]) -> list[CheckResult]:
    ids = {r["condition_id"] for r in conditions}
    checks = [
        CheckResult(
            name="condition_count",
            passed=len(conditions) == len(EXPECTED_CONDITION_IDS),
            message=f"condition_count={len(conditions)}",
            details={"count": len(conditions), "expected": len(EXPECTED_CONDITION_IDS)},
        ),
        CheckResult(
            name="condition_ids",
            passed=ids == set(EXPECTED_CONDITION_IDS),
            message=(
                "condition IDs match contract"
                if ids == set(EXPECTED_CONDITION_IDS)
                else f"unexpected condition IDs: observed={sorted(ids)}"
            ),
            details={"observed": sorted(ids), "expected": sorted(EXPECTED_CONDITION_IDS)},
        ),
    ]
    required = ("condition_id", "temperature_c", "vdd_applied", "test_mode")
    missing = 0
    for row in conditions:
        if any(not str(row.get(c, "")).strip() for c in required):
            missing += 1
    checks.append(
        CheckResult(
            name="condition_required_fields",
            passed=missing == 0,
            message=(
                "condition fields complete"
                if missing == 0
                else f"{missing} condition rows missing required fields"
            ),
            details={"missing_rows": missing},
        )
    )
    return checks
