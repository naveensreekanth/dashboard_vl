"""Forbidden path and column protection for agent inputs."""

from __future__ import annotations

from pathlib import Path

from dtl_agent.config.allowlists import FORBIDDEN_COLUMN_NAMES, FORBIDDEN_PATH_FRAGMENTS
from dtl_agent.data.repositories.allowlist_repository import AllowlistRepository, AllowlistViolation
from dtl_agent.validation.report import CheckResult, ForbiddenDataSummary


def find_forbidden_columns(columns: list[str]) -> list[str]:
    lowered = {c.lower(): c for c in columns}
    hits: list[str] = []
    for forbidden in FORBIDDEN_COLUMN_NAMES:
        if forbidden.lower() in lowered:
            hits.append(lowered[forbidden.lower()])
        # also catch substring-style answer keys in column names
        for col_l, col in lowered.items():
            if forbidden.lower() in col_l and col not in hits:
                # avoid over-matching unrelated names; require token-ish presence
                if (
                    col_l == forbidden.lower()
                    or col_l.startswith(forbidden.lower())
                    or col_l.endswith(forbidden.lower())
                    or f"_{forbidden.lower()}" in col_l
                    or f"{forbidden.lower()}_" in col_l
                ):
                    hits.append(col)
    return sorted(set(hits))


def validate_table_columns_not_forbidden(
    *, table_name: str, columns: list[str]
) -> CheckResult:
    hits = find_forbidden_columns(columns)
    return CheckResult(
        name=f"forbidden_columns:{table_name}",
        passed=not hits,
        message=(
            "no forbidden answer-key columns"
            if not hits
            else f"forbidden columns present: {hits}"
        ),
        details={"forbidden_columns": hits},
    )


def validate_path_not_forbidden(path: str | Path) -> CheckResult:
    text = str(path).replace("\\", "/")
    hits = [frag for frag in FORBIDDEN_PATH_FRAGMENTS if frag.lower() in text.lower()]
    return CheckResult(
        name="forbidden_path",
        passed=not hits,
        message="path allowed" if not hits else f"forbidden path fragments: {hits}",
        details={"path": text, "hits": hits},
    )


def assert_allowlist_rejects_forbidden(
    repo: AllowlistRepository, relative_path: str
) -> CheckResult:
    try:
        repo.resolve(relative_path)
        return CheckResult(
            name=f"allowlist_reject:{relative_path}",
            passed=False,
            message=f"allowlist unexpectedly accepted {relative_path}",
        )
    except (AllowlistViolation, FileNotFoundError):
        return CheckResult(
            name=f"allowlist_reject:{relative_path}",
            passed=True,
            message=f"correctly rejected {relative_path}",
        )


def build_forbidden_summary(
    *,
    column_checks: list[CheckResult],
    path_hits: list[str],
) -> ForbiddenDataSummary:
    col_hits: list[str] = []
    for check in column_checks:
        col_hits.extend(check.details.get("forbidden_columns", []))
    summary = ForbiddenDataSummary(
        forbidden_files_detected=path_hits,
        forbidden_columns_detected=sorted(set(col_hits)),
        checks=column_checks,
    )
    summary.checks.append(
        CheckResult(
            name="forbidden_files_absent",
            passed=not path_hits,
            message=(
                "no forbidden files in loaded inputs"
                if not path_hits
                else f"forbidden files: {path_hits}"
            ),
            details={"files": path_hits},
        )
    )
    return summary
