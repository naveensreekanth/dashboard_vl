"""Load a single production month from ``data/3 months data/`` (Phase 12.3).

Does not concatenate months. Does not fall back to legacy ``data/core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.paths import (
    actual_die_root,
    month_root,
    parametric_root,
    validate_production_month,
)

REQUIRED_ACTUAL_DIE_COLUMNS = frozenset(
    {
        "production_month",
        "lot_id",
        "die_id",
        "pattern_id",
        "test_id",
        "parameter",
        "measurement_value",
        "pass_fail_pattern",
        "die_status",
    }
)

REQUIRED_PARAMETRIC_COLUMNS = frozenset(
    {
        "production_month",
        "lot_id",
        "die_id",
        "condition_id",
        "test_id",
        "parameter",
        "measurement_value",
        "pass_fail_condition",
    }
)


class TemporalLoaderError(ValueError):
    """Raised when temporal month data is missing, corrupt, or cross-contaminated."""


@dataclass
class TemporalMonthData:
    """Month-scoped frames compatible with downstream feature / simulation use."""

    production_month: str
    month_path: Path
    actual_die: pd.DataFrame
    parametric: pd.DataFrame
    parts_dim: pd.DataFrame | None

    @property
    def die_identities(self) -> list[str]:
        """Sorted unique month-prefixed sequence ids for actual_die population."""
        pairs = (
            self.actual_die[["lot_id", "die_id"]]
            .drop_duplicates()
            .sort_values(["lot_id", "die_id"])
        )
        return [
            make_sequence_id(str(r.lot_id), str(r.die_id), self.production_month)
            for r in pairs.itertuples(index=False)
        ]


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise TemporalLoaderError(f"Missing required {label}: {path}")
    return path


def _require_columns(df: pd.DataFrame, required: frozenset[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise TemporalLoaderError(f"Invalid {label} schema; missing columns: {missing}")


def _assert_month_isolation(df: pd.DataFrame, production_month: str, label: str) -> None:
    if df.empty:
        raise TemporalLoaderError(f"{label} is empty for production_month={production_month!r}")
    months = set(df["production_month"].astype(str).unique())
    if months != {production_month}:
        raise TemporalLoaderError(
            f"{label} contains unexpected production_month values {sorted(months)}; "
            f"expected only {production_month!r} (no silent repair)"
        )


def _assert_no_duplicate_die_keys(df: pd.DataFrame, production_month: str, label: str) -> None:
    """Detect unexpected duplicate temporal identities at die grain (lot×die×pattern×test)."""
    key_cols = ["lot_id", "die_id", "pattern_id", "test_id", "parameter"]
    dup = df.duplicated(subset=key_cols, keep=False)
    if dup.any():
        sample = df.loc[dup, key_cols].head(5).to_dict(orient="records")
        raise TemporalLoaderError(
            f"{label} has unexpected duplicate temporal measurement keys "
            f"for production_month={production_month!r}; sample={sample}"
        )


def _assert_no_duplicate_parametric_keys(
    df: pd.DataFrame, production_month: str, label: str
) -> None:
    key_cols = ["lot_id", "die_id", "condition_id", "test_id", "parameter"]
    dup = df.duplicated(subset=key_cols, keep=False)
    if dup.any():
        sample = df.loc[dup, key_cols].head(5).to_dict(orient="records")
        raise TemporalLoaderError(
            f"{label} has unexpected duplicate temporal parametric keys "
            f"for production_month={production_month!r}; sample={sample}"
        )


def _normalize_measurement_aliases(df: pd.DataFrame, *, pass_fail_src: str) -> pd.DataFrame:
    out = df.copy()
    if "pass_fail" not in out.columns:
        out["pass_fail"] = out[pass_fail_src]
    if "value" not in out.columns:
        out["value"] = out["measurement_value"]
    return out


def _filter_csv_die_rows(path: Path, *, lot_id: str, die_id: str) -> pd.DataFrame:
    """Stream-filter a measurements CSV to one lot/die without retaining the full file.

    Uses the stdlib csv reader (low transient RSS) then parses only the matched
    rows with pandas so dtypes match a normal ``pd.read_csv`` of those rows.
    """
    import csv
    from io import StringIO

    lot_s = str(lot_id)
    die_s = str(die_id)
    buf = StringIO()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
        try:
            lot_i = header.index("lot_id")
            die_i = header.index("die_id")
        except ValueError as exc:
            raise TemporalLoaderError(
                f"measurements CSV missing lot_id/die_id columns: {path}"
            ) from exc
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(header)
        n_match = 0
        for row in reader:
            if row[lot_i] == lot_s and row[die_i] == die_s:
                writer.writerow(row)
                n_match += 1
    if n_match == 0:
        return pd.DataFrame()
    buf.seek(0)
    return pd.read_csv(buf, low_memory=False)


def load_temporal_die(
    production_month: str,
    lot_id: str,
    die_id: str,
    *,
    project_root: Path | None = None,
    use_cache: bool = True,
) -> TemporalMonthData:
    """Load ONLY measurements for one lot/die within a production month.

    Same schema / aliases as ``load_temporal_month``, but rows are die-scoped so
    recommendation math sees the identical per-die subset without a ~500 MB frame.
    """
    from dtl_agent.config.paths import default_project_root
    from dtl_agent.data.temporal.month_cache import get_cached_die, put_cached_die

    month = validate_production_month(production_month)
    root = project_root if project_root is not None else default_project_root()
    if use_cache:
        hit = get_cached_die(root, month, lot_id, die_id)
        if hit is not None:
            return hit

    mroot = month_root(month, root)
    if not mroot.is_dir():
        raise TemporalLoaderError(
            f"Month directory does not exist: {mroot} "
            f"(production_month={month!r})"
        )

    actual_path = _require_file(
        actual_die_root(month, root) / "measurements.csv",
        "actual_die measurements.csv",
    )
    parametric_path = _require_file(
        parametric_root(month, root) / "measurements.csv",
        "parametric measurements.csv",
    )
    parts_path = parametric_root(month, root) / "parts_dim.csv"

    actual = _filter_csv_die_rows(actual_path, lot_id=lot_id, die_id=die_id)
    if actual.empty:
        raise TemporalLoaderError(
            f"No actual_die rows for lot_id={lot_id!r} die_id={die_id!r} "
            f"production_month={month!r}"
        )
    _require_columns(actual, REQUIRED_ACTUAL_DIE_COLUMNS, "actual_die")
    _assert_month_isolation(actual, month, "actual_die")
    _assert_no_duplicate_die_keys(actual, month, "actual_die")

    parametric = _filter_csv_die_rows(parametric_path, lot_id=lot_id, die_id=die_id)
    if not parametric.empty:
        _require_columns(parametric, REQUIRED_PARAMETRIC_COLUMNS, "parametric")
        _assert_month_isolation(parametric, month, "parametric")
        _assert_no_duplicate_parametric_keys(parametric, month, "parametric")
    else:
        parametric = pd.DataFrame(columns=sorted(REQUIRED_PARAMETRIC_COLUMNS))

    parts_dim: pd.DataFrame | None = None
    if parts_path.is_file():
        parts_dim = pd.read_csv(parts_path, low_memory=False)

    actual = _normalize_measurement_aliases(actual, pass_fail_src="pass_fail_pattern")
    if not parametric.empty:
        parametric = _normalize_measurement_aliases(
            parametric, pass_fail_src="pass_fail_condition"
        )

    data = TemporalMonthData(
        production_month=month,
        month_path=mroot,
        actual_die=actual,
        parametric=parametric,
        parts_dim=parts_dim,
    )
    if use_cache:
        return put_cached_die(root, month, lot_id, die_id, data)
    return data


def load_temporal_month(
    production_month: str,
    *,
    project_root: Path | None = None,
    use_cache: bool = True,
) -> TemporalMonthData:
    """Load ONLY ``data/3 months data/{production_month}/``.

    Returns structures with production_month, lot/die identity, measurements, and PASS/FAIL.

    When ``use_cache=True`` (default), at most one full-month package is retained
    in-process (LRU). Prefer ``load_temporal_die`` for API recommendation traffic.
    """
    from dtl_agent.config.paths import default_project_root
    from dtl_agent.data.temporal.month_cache import get_cached_month, put_cached_month

    month = validate_production_month(production_month)
    root = project_root if project_root is not None else default_project_root()
    if use_cache:
        hit = get_cached_month(root, month)
        if hit is not None:
            return hit

    mroot = month_root(month, root)
    if not mroot.is_dir():
        raise TemporalLoaderError(
            f"Month directory does not exist: {mroot} "
            f"(production_month={month!r})"
        )

    actual_path = _require_file(
        actual_die_root(month, root) / "measurements.csv",
        "actual_die measurements.csv",
    )
    parametric_path = _require_file(
        parametric_root(month, root) / "measurements.csv",
        "parametric measurements.csv",
    )
    parts_path = parametric_root(month, root) / "parts_dim.csv"

    actual = pd.read_csv(actual_path, low_memory=False)
    _require_columns(actual, REQUIRED_ACTUAL_DIE_COLUMNS, "actual_die")
    _assert_month_isolation(actual, month, "actual_die")
    _assert_no_duplicate_die_keys(actual, month, "actual_die")

    parametric = pd.read_csv(parametric_path, low_memory=False)
    _require_columns(parametric, REQUIRED_PARAMETRIC_COLUMNS, "parametric")
    _assert_month_isolation(parametric, month, "parametric")
    _assert_no_duplicate_parametric_keys(parametric, month, "parametric")

    parts_dim: pd.DataFrame | None = None
    if parts_path.is_file():
        parts_dim = pd.read_csv(parts_path, low_memory=False)

    actual = _normalize_measurement_aliases(actual, pass_fail_src="pass_fail_pattern")
    parametric = _normalize_measurement_aliases(
        parametric, pass_fail_src="pass_fail_condition"
    )

    data = TemporalMonthData(
        production_month=month,
        month_path=mroot,
        actual_die=actual,
        parametric=parametric,
        parts_dim=parts_dim,
    )
    if use_cache:
        return put_cached_month(root, month, data)
    return data


def temporal_month_summary(data: TemporalMonthData) -> dict[str, Any]:
    """Small diagnostic summary (no mutation)."""
    dies = data.actual_die[["lot_id", "die_id"]].drop_duplicates()
    return {
        "production_month": data.production_month,
        "actual_die_rows": int(len(data.actual_die)),
        "parametric_rows": int(len(data.parametric)),
        "n_dies": int(len(dies)),
        "n_lots": int(dies["lot_id"].nunique()),
    }
