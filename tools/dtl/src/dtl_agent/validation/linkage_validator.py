"""Core ↔ Parametric linkage validation."""

from __future__ import annotations

from dtl_agent.config.constants import (
    LINKED_DIE_COUNT,
    LINKED_LOT_COUNT,
    PARAMETRIC_ONLY_DIE_COUNT,
    PARAMETRIC_ONLY_LOT_COUNT,
)
from dtl_agent.data.models.linkage import SharedLotDieIndex
from dtl_agent.validation.report import CheckResult, LinkageValidationSummary


def validate_linkage(index: SharedLotDieIndex) -> LinkageValidationSummary:
    summary = LinkageValidationSummary(
        common_lot_count=len(index.linked_lots),
        common_die_count=len(index.linked_dies),
        core_only_lot_count=len(index.core_only_lots),
        parametric_only_lot_count=len(index.parametric_only_lots),
        common_lot_die_pair_count=len(index.linked_lot_die_pairs),
    )
    checks = [
        CheckResult(
            name="linked_lot_count",
            passed=len(index.linked_lots) == LINKED_LOT_COUNT,
            message=f"linked_lots={len(index.linked_lots)}",
            details={"expected": LINKED_LOT_COUNT},
        ),
        CheckResult(
            name="linked_die_count",
            passed=len(index.linked_dies) == LINKED_DIE_COUNT,
            message=f"linked_dies={len(index.linked_dies)}",
            details={"expected": LINKED_DIE_COUNT},
        ),
        CheckResult(
            name="parametric_only_lot_count",
            passed=len(index.parametric_only_lots) == PARAMETRIC_ONLY_LOT_COUNT,
            message=f"parametric_only_lots={len(index.parametric_only_lots)}",
            details={"expected": PARAMETRIC_ONLY_LOT_COUNT},
        ),
        CheckResult(
            name="parametric_only_die_count",
            passed=len(index.parametric_only_dies) == PARAMETRIC_ONLY_DIE_COUNT,
            message=f"parametric_only_dies={len(index.parametric_only_dies)}",
            details={"expected": PARAMETRIC_ONLY_DIE_COUNT},
        ),
        CheckResult(
            name="core_only_lots_empty",
            passed=len(index.core_only_lots) == 0,
            message=f"core_only_lots={len(index.core_only_lots)}",
            details={"lots": sorted(index.core_only_lots)},
        ),
        CheckResult(
            name="linked_pair_count",
            passed=len(index.linked_lot_die_pairs) == LINKED_DIE_COUNT,
            message=f"linked_lot_die_pairs={len(index.linked_lot_die_pairs)}",
            details={"expected": LINKED_DIE_COUNT},
        ),
    ]
    summary.checks = checks
    summary.linkage_status = "PASS" if summary.passed else "FAIL"
    return summary
