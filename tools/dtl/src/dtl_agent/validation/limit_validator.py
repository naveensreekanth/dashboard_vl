"""Current-limit coverage and polarity validation."""

from __future__ import annotations

from dtl_agent.config.constants import ExpectedLimit
from dtl_agent.validation.report import CheckResult


def _normalize_unit(unit: str) -> str:
    """Normalize temperature unit variants to a canonical °C token."""
    u = unit.strip().replace("Â", "")
    if u in {"°C", "˚C", "C"}:
        return "°C"
    # Mojibake / replacement-char forms that still denote Celsius
    if "C" in u and len(u) <= 4 and "mV" not in u and "mA" not in u and "uA" not in u:
        if "ohm" not in u.lower() and "ps" not in u.lower() and "ms" not in u.lower():
            if u.endswith("C"):
                return "°C"
    return u


def validate_expected_limits(
    limit_rows: list[dict[str, str]],
    expected: tuple[ExpectedLimit, ...],
    *,
    domain: str,
    value_field: str,
    direction_field: str,
    source_field: str,
) -> CheckResult:
    by_test = {r["test_id"]: r for r in limit_rows}
    problems: list[str] = []
    covered: list[str] = []

    for exp in expected:
        row = by_test.get(exp.test_id)
        if row is None:
            problems.append(f"missing limit row for {exp.test_id}")
            continue
        covered.append(exp.test_id)
        direction = str(row.get(direction_field, "")).strip().upper()
        if direction != exp.direction:
            problems.append(
                f"{exp.test_id}: direction {direction!r} != expected {exp.direction!r}"
            )
        try:
            value = float(str(row.get(value_field, "")).strip())
        except ValueError:
            problems.append(f"{exp.test_id}: non-numeric {value_field}")
            continue
        if abs(value - exp.value) > 1e-9:
            problems.append(
                f"{exp.test_id}: value {value} != expected {exp.value}"
            )
        unit = _normalize_unit(str(row.get("unit", "")))
        exp_unit = _normalize_unit(exp.unit)
        if unit != exp_unit:
            # Allow exact match on raw unit if normalization is too aggressive
            raw = str(row.get("unit", "")).strip()
            if raw != exp.unit and unit != exp_unit:
                # For thermal, accept any unit containing C as degree C from source file
                if not (exp.test_id == "T_THERMAL_C" and "C" in raw):
                    problems.append(
                        f"{exp.test_id}: unit {raw!r} != expected {exp.unit!r}"
                    )
        source = str(row.get(source_field, "")).strip()
        if source and source != exp.source:
            problems.append(
                f"{exp.test_id}: source {source!r} != expected {exp.source!r}"
            )
        parameter = str(row.get("parameter", "")).strip()
        if parameter and parameter != exp.parameter:
            problems.append(
                f"{exp.test_id}: parameter {parameter!r} != expected {exp.parameter!r}"
            )

    return CheckResult(
        name=f"current_limits:{domain}",
        passed=not problems,
        message="current limits match contract" if not problems else "; ".join(problems),
        details={"covered_test_ids": covered, "problems": problems},
    )


def validate_no_invented_secondary_limits(
    limit_rows: list[dict[str, str]],
    *,
    forbidden_test_ids: frozenset[str],
    domain: str,
) -> CheckResult:
    present = sorted({r["test_id"] for r in limit_rows if r["test_id"] in forbidden_test_ids})
    return CheckResult(
        name=f"no_invented_limits:{domain}",
        passed=not present,
        message=(
            "no secondary/context limits invented"
            if not present
            else f"unexpected limits present for non-targets: {present}"
        ),
        details={"unexpected_test_ids": present},
    )
