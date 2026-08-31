"""Validation report structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainValidationSummary:
    domain: str
    files_validated: list[str] = field(default_factory=list)
    measurement_row_count: int = 0
    lot_count: int = 0
    die_count: int = 0
    condition_count: int | None = None
    duplicate_pk_count: int = 0
    missing_required_fields: list[str] = field(default_factory=list)
    parameter_coverage: list[str] = field(default_factory=list)
    current_limit_coverage: list[str] = field(default_factory=list)
    version: str = ""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass
class LinkageValidationSummary:
    common_lot_count: int = 0
    common_die_count: int = 0
    core_only_lot_count: int = 0
    parametric_only_lot_count: int = 0
    common_lot_die_pair_count: int = 0
    linkage_status: str = "UNKNOWN"
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass
class ForbiddenDataSummary:
    forbidden_files_detected: list[str] = field(default_factory=list)
    forbidden_columns_detected: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            not self.forbidden_files_detected
            and not self.forbidden_columns_detected
            and all(c.passed for c in self.checks)
        )


@dataclass
class Phase1ValidationReport:
    core: DomainValidationSummary
    parametric: DomainValidationSummary
    linkage: LinkageValidationSummary
    forbidden: ForbiddenDataSummary
    final_status: str = "FAIL"

    @property
    def passed(self) -> bool:
        return self.final_status == "PASS"

    def finalize(self) -> None:
        ok = (
            self.core.passed
            and self.parametric.passed
            and self.linkage.passed
            and self.forbidden.passed
        )
        self.final_status = "PASS" if ok else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_status": self.final_status,
            "core": asdict(self.core),
            "parametric": asdict(self.parametric),
            "linkage": asdict(self.linkage),
            "forbidden": asdict(self.forbidden),
        }
