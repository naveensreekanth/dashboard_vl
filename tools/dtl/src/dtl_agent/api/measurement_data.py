"""Read-only measurement / distribution / condition data access (Phase 10.11).

Serves existing canonical measurements and Phase 3 feature aggregates.
Does not call recommend(), ranking, or safety. Does not fabricate values.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dtl_agent.api.selector_data import load_selector_index
from dtl_agent.features.stats import compute_dist_stats
from dtl_agent.recommendation.schemas import CORE_PARAMETERS, PARAMETRIC_PARAMETERS

SOURCE_CLASSIFICATION = "SYNTHETIC"
DISCLAIMER = "Synthetic dataset measurement — not production silicon truth."
STATS_METHOD = "phase3_compute_dist_stats"
DEFAULT_PARAMETRIC_CONDITION = "COND_RT_NOM"
OBSERVED_RULE_CORE = "median_over_patterns"
OBSERVED_RULE_PARAMETRIC = "selected_condition"


class MeasurementLookupError(Exception):
    """Invalid request identity (lot/die/parameter/condition)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class DistSummary:
    n: int
    min: float
    median: float
    p95: float
    max: float
    stats_source: str


def _f(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _i(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _feature_prefix(domain: str, parameter: str) -> str:
    if domain == "core":
        return f"core_{parameter}"
    return f"param_{parameter.lower()}"


@lru_cache(maxsize=4)
def _dataset_versions(project_root: str) -> dict[str, str]:
    root = Path(project_root)
    out: dict[str, str] = {"core": "unknown", "parametric": "unknown"}
    core_path = root / "data" / "core" / "DATASET_VERSION.json"
    param_path = root / "data" / "parametric" / "PARAMETRIC_DATASET_VERSION.json"
    if core_path.is_file():
        data = json.loads(core_path.read_text(encoding="utf-8"))
        out["core"] = str(data.get("dataset_version") or "unknown")
    if param_path.is_file():
        data = json.loads(param_path.read_text(encoding="utf-8"))
        out["parametric"] = str(data.get("dataset_version") or "unknown")
    return out


@lru_cache(maxsize=4)
def _unit_catalog(project_root: str) -> dict[tuple[str, str], str]:
    """Map (domain, parameter) -> unit from test catalogs."""
    root = Path(project_root)
    units: dict[tuple[str, str], str] = {}
    for domain, rel in (
        ("core", Path("data") / "core" / "test_catalog.csv"),
        ("parametric", Path("data") / "parametric" / "test_catalog.csv"),
    ):
        path = root / rel
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                param = (row.get("parameter") or "").strip()
                unit = (row.get("unit") or "").strip()
                if param and unit:
                    units[(domain, param)] = unit
    return units


@lru_cache(maxsize=4)
def _conditions_dim(project_root: str) -> dict[str, dict[str, Any]]:
    path = Path(project_root) / "data" / "parametric" / "conditions_dim.csv"
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            cid = (row.get("condition_id") or "").strip()
            if not cid:
                continue
            out[cid] = {
                "condition_id": cid,
                "temperature_c": _f(row.get("temperature_c")),
                "vdd_applied": _f(row.get("vdd_applied")),
                "test_mode": (row.get("test_mode") or "").strip() or None,
                "description": (row.get("description") or "").strip() or None,
            }
    return out


@lru_cache(maxsize=4)
def _core_die_features(project_root: str) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(project_root) / "artifacts" / "features" / "core" / "die_features.csv"
    index: dict[tuple[str, str], dict[str, str]] = {}
    if not path.is_file():
        return index
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            lot = (row.get("lot_id") or "").strip()
            die = (row.get("die_id") or "").strip()
            if lot and die:
                index[(lot, die)] = row
    return index


@lru_cache(maxsize=4)
def _core_lot_parameter_features(project_root: str) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(project_root) / "artifacts" / "features" / "core" / "lot_parameter_features.csv"
    index: dict[tuple[str, str], dict[str, str]] = {}
    if not path.is_file():
        return index
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            lot = (row.get("lot_id") or "").strip()
            param = (row.get("parameter") or "").strip()
            if lot and param:
                index[(lot, param)] = row
    return index


@lru_cache(maxsize=4)
def _param_die_features(project_root: str) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(project_root) / "artifacts" / "features" / "parametric" / "die_features.csv"
    index: dict[tuple[str, str], dict[str, str]] = {}
    if not path.is_file():
        return index
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            lot = (row.get("lot_id") or "").strip()
            die = (row.get("die_id") or "").strip()
            if lot and die:
                index[(lot, die)] = row
    return index


@lru_cache(maxsize=4)
def _param_lot_features(project_root: str) -> dict[str, dict[str, str]]:
    path = Path(project_root) / "artifacts" / "features" / "parametric" / "lot_features.csv"
    index: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return index
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            lot = (row.get("lot_id") or "").strip()
            if lot:
                index[lot] = row
    return index


@lru_cache(maxsize=4)
def _param_lot_condition_features(
    project_root: str,
) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(project_root) / "artifacts" / "features" / "parametric" / "lot_condition_features.csv"
    index: dict[tuple[str, str], dict[str, str]] = {}
    if not path.is_file():
        return index
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            lot = (row.get("lot_id") or "").strip()
            cid = (row.get("condition_id") or "").strip()
            if lot and cid:
                index[(lot, cid)] = row
    return index


@lru_cache(maxsize=4)
def _param_measurements_index(
    project_root: str,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Index parametric measurements by (lot, die, parameter, condition)."""
    path = Path(project_root) / "data" / "parametric" / "measurements.csv"
    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not path.is_file():
        return index
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            lot = (row.get("lot_id") or "").strip()
            die = (row.get("die_id") or "").strip()
            param = (row.get("parameter") or "").strip()
            cid = (row.get("condition_id") or "").strip()
            if not (lot and die and param and cid):
                continue
            index[(lot, die, param, cid)] = {
                "measurement_value": _f(row.get("measurement_value")),
                "unit": (row.get("unit") or "").strip() or None,
                "pass_fail_condition": (row.get("pass_fail_condition") or "").strip() or None,
                "temperature_c": _f(row.get("temperature_c")),
                "vdd_applied": _f(row.get("vdd_applied")),
                "test_mode": (row.get("test_mode") or "").strip() or None,
            }
    return index


def _stats_from_feature_row(row: dict[str, str], prefix: str) -> DistSummary | None:
    n = _i(row.get(f"{prefix}_count"))
    mn = _f(row.get(f"{prefix}_min"))
    med = _f(row.get(f"{prefix}_median"))
    p95 = _f(row.get(f"{prefix}_p95"))
    mx = _f(row.get(f"{prefix}_max"))
    if n is None or n <= 0 or mn is None or med is None or p95 is None or mx is None:
        return None
    return DistSummary(
        n=n,
        min=mn,
        median=med,
        p95=p95,
        max=mx,
        stats_source="phase3_die_or_lot_features",
    )


def _stats_from_values(values: list[float], *, source: str) -> DistSummary | None:
    stats = compute_dist_stats(values)
    if stats is None:
        return None
    return DistSummary(
        n=stats.count,
        min=stats.min,
        median=stats.median,
        p95=stats.p95,
        max=stats.max,
        stats_source=source,
    )


def _stream_core_values(
    project_root: str,
    *,
    lot_id: str,
    parameter: str,
    die_id: str | None,
) -> list[float]:
    """Fallback streaming filter of core measurements (avoid on hot path)."""
    path = Path(project_root) / "data" / "core" / "measurements.csv"
    values: list[float] = []
    if not path.is_file():
        return values
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if (row.get("lot_id") or "").strip() != lot_id:
                continue
            if (row.get("parameter") or "").strip() != parameter:
                continue
            if die_id is not None and (row.get("die_id") or "").strip() != die_id:
                continue
            val = _f(row.get("measurement_value"))
            if val is not None:
                values.append(val)
    return values


def resolve_domain_and_validate(
    project_root: str,
    *,
    lot_id: str,
    die_id: str,
    parameter: str,
    condition_id: str | None = None,
    allow_condition_for_core: bool = False,
) -> str:
    """Validate lot/die/parameter(/condition). Return domain ``core``|``parametric``."""
    index = load_selector_index(project_root)
    if lot_id not in index.lots:
        raise MeasurementLookupError("Lot not found.")
    dies = index.dies_by_lot.get(lot_id)
    if dies is None or die_id not in dies:
        raise MeasurementLookupError("Die not found for lot.")

    if parameter in CORE_PARAMETERS:
        domain = "core"
    elif parameter in PARAMETRIC_PARAMETERS:
        domain = "parametric"
    else:
        raise MeasurementLookupError("Parameter not found.")

    params = index.params_by_lot_die.get((lot_id, die_id), ())
    if parameter not in params:
        raise MeasurementLookupError("Parameter not found for die.")

    if condition_id is not None:
        if domain == "core" and not allow_condition_for_core:
            raise MeasurementLookupError("Condition not supported for core parameters.")
        known = _conditions_dim(project_root)
        if condition_id not in known:
            raise MeasurementLookupError("Condition not found.")

    return domain


def get_unit(project_root: str, domain: str, parameter: str) -> str | None:
    return _unit_catalog(project_root).get((domain, parameter))


def get_dataset_version(project_root: str, domain: str) -> str:
    return _dataset_versions(project_root).get(domain, "unknown")


def get_selected_measurement(
    project_root: str,
    *,
    lot_id: str,
    die_id: str,
    parameter: str,
    condition_id: str | None = None,
) -> dict[str, Any]:
    domain = resolve_domain_and_validate(
        project_root,
        lot_id=lot_id,
        die_id=die_id,
        parameter=parameter,
        condition_id=condition_id,
    )
    unit = get_unit(project_root, domain, parameter)
    dataset_version = get_dataset_version(project_root, domain)
    base: dict[str, Any] = {
        "lot_id": lot_id,
        "die_id": die_id,
        "parameter": parameter,
        "domain": domain,
        "unit": unit,
        "source_classification": SOURCE_CLASSIFICATION,
        "dataset_version": dataset_version,
        "disclaimer": DISCLAIMER,
    }

    if domain == "core":
        if condition_id is not None:
            raise MeasurementLookupError("Condition not supported for core parameters.")
        base["observed_value_rule"] = OBSERVED_RULE_CORE
        row = _core_die_features(project_root).get((lot_id, die_id))
        prefix = _feature_prefix(domain, parameter)
        observed = _f(row.get(f"{prefix}_median")) if row else None
        if observed is None:
            values = _stream_core_values(
                project_root, lot_id=lot_id, parameter=parameter, die_id=die_id
            )
            stats = compute_dist_stats(values)
            observed = stats.median if stats else None
        if observed is None:
            return {**base, "observed_value": None, "found": False}
        return {**base, "observed_value": observed, "found": True}

    # Parametric
    selected = condition_id or DEFAULT_PARAMETRIC_CONDITION
    if selected not in _conditions_dim(project_root):
        raise MeasurementLookupError("Condition not found.")
    base["condition_id"] = selected
    base["observed_value_rule"] = OBSERVED_RULE_PARAMETRIC
    rec = _param_measurements_index(project_root).get((lot_id, die_id, parameter, selected))
    if rec is None or rec.get("measurement_value") is None:
        return {**base, "observed_value": None, "found": False}
    if unit is None and rec.get("unit"):
        base["unit"] = rec["unit"]
    return {**base, "observed_value": rec["measurement_value"], "found": True}


def get_distribution(
    project_root: str,
    *,
    lot_id: str,
    die_id: str,
    parameter: str,
    scope: str = "die",
    condition_id: str | None = None,
) -> dict[str, Any]:
    if scope not in {"die", "lot"}:
        raise MeasurementLookupError("Invalid scope. Use die or lot.")

    domain = resolve_domain_and_validate(
        project_root,
        lot_id=lot_id,
        die_id=die_id,
        parameter=parameter,
        condition_id=condition_id,
    )
    if domain == "core" and condition_id is not None:
        raise MeasurementLookupError("Condition not supported for core parameters.")

    unit = get_unit(project_root, domain, parameter)
    dataset_version = get_dataset_version(project_root, domain)
    base: dict[str, Any] = {
        "lot_id": lot_id,
        "die_id": die_id,
        "parameter": parameter,
        "domain": domain,
        "unit": unit,
        "scope": scope,
        "source_classification": SOURCE_CLASSIFICATION,
        "dataset_version": dataset_version,
        "stats_method": STATS_METHOD,
        "disclaimer": DISCLAIMER,
    }
    if condition_id is not None:
        base["condition_id"] = condition_id

    summary = _resolve_distribution(
        project_root,
        domain=domain,
        lot_id=lot_id,
        die_id=die_id,
        parameter=parameter,
        scope=scope,
        condition_id=condition_id,
    )
    if summary is None:
        return {
            **base,
            "n": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "found": False,
        }
    return {
        **base,
        "n": summary.n,
        "min": summary.min,
        "median": summary.median,
        "p95": summary.p95,
        "max": summary.max,
        "found": True,
        "stats_source": summary.stats_source,
    }


def _resolve_distribution(
    project_root: str,
    *,
    domain: str,
    lot_id: str,
    die_id: str,
    parameter: str,
    scope: str,
    condition_id: str | None,
) -> DistSummary | None:
    prefix = _feature_prefix(domain, parameter)

    if domain == "core":
        if scope == "die":
            row = _core_die_features(project_root).get((lot_id, die_id))
            if row:
                hit = _stats_from_feature_row(row, prefix)
                if hit is not None:
                    return hit
            values = _stream_core_values(
                project_root, lot_id=lot_id, parameter=parameter, die_id=die_id
            )
            return _stats_from_values(values, source="canonical_measurements_stream")
        # lot scope
        row = _core_lot_parameter_features(project_root).get((lot_id, parameter))
        if row:
            hit = _stats_from_feature_row(row, prefix)
            if hit is not None:
                return hit
        values = _stream_core_values(
            project_root, lot_id=lot_id, parameter=parameter, die_id=None
        )
        return _stats_from_values(values, source="canonical_measurements_stream")

    # Parametric
    if condition_id is not None:
        if scope == "die":
            values = _parametric_values(
                project_root,
                lot_id=lot_id,
                die_id=die_id,
                parameter=parameter,
                condition_id=condition_id,
            )
            return _stats_from_values(values, source="canonical_measurements")
        row = _param_lot_condition_features(project_root).get((lot_id, condition_id))
        if row:
            hit = _stats_from_feature_row(row, prefix)
            if hit is not None:
                return hit
        values = _parametric_values(
            project_root,
            lot_id=lot_id,
            die_id=None,
            parameter=parameter,
            condition_id=condition_id,
        )
        return _stats_from_values(values, source="canonical_measurements")

    # No condition filter: all conditions
    if scope == "die":
        row = _param_die_features(project_root).get((lot_id, die_id))
        if row:
            hit = _stats_from_feature_row(row, prefix)
            if hit is not None:
                return hit
        values = _parametric_values(
            project_root,
            lot_id=lot_id,
            die_id=die_id,
            parameter=parameter,
            condition_id=None,
        )
        return _stats_from_values(values, source="canonical_measurements")

    row = _param_lot_features(project_root).get(lot_id)
    if row:
        hit = _stats_from_feature_row(row, prefix)
        if hit is not None:
            return hit
    values = _parametric_values(
        project_root,
        lot_id=lot_id,
        die_id=None,
        parameter=parameter,
        condition_id=None,
    )
    return _stats_from_values(values, source="canonical_measurements")


def _parametric_values(
    project_root: str,
    *,
    lot_id: str,
    die_id: str | None,
    parameter: str,
    condition_id: str | None,
) -> list[float]:
    index = _param_measurements_index(project_root)
    values: list[float] = []
    for (lot, die, param, cid), rec in index.items():
        if lot != lot_id or param != parameter:
            continue
        if die_id is not None and die != die_id:
            continue
        if condition_id is not None and cid != condition_id:
            continue
        val = rec.get("measurement_value")
        if val is not None:
            values.append(float(val))
    return values


def get_conditions(
    project_root: str,
    *,
    lot_id: str,
    die_id: str,
    parameter: str,
) -> dict[str, Any]:
    domain = resolve_domain_and_validate(
        project_root,
        lot_id=lot_id,
        die_id=die_id,
        parameter=parameter,
        condition_id=None,
    )
    unit = get_unit(project_root, domain, parameter)
    dataset_version = get_dataset_version(project_root, domain)
    base: dict[str, Any] = {
        "lot_id": lot_id,
        "die_id": die_id,
        "parameter": parameter,
        "domain": domain,
        "unit": unit,
        "source_classification": SOURCE_CLASSIFICATION,
        "dataset_version": dataset_version,
        "disclaimer": DISCLAIMER,
    }

    if domain == "core":
        return {
            **base,
            "found": False,
            "reason": "not_condition_aware",
            "conditions": [],
        }

    dim = _conditions_dim(project_root)
    meas = _param_measurements_index(project_root)
    rows: list[dict[str, Any]] = []
    for cid in sorted(dim.keys()):
        meta = dim[cid]
        rec = meas.get((lot_id, die_id, parameter, cid))
        if rec is None:
            continue
        rows.append(
            {
                "condition_id": cid,
                "temperature_c": meta.get("temperature_c")
                if meta.get("temperature_c") is not None
                else rec.get("temperature_c"),
                "vdd_applied": meta.get("vdd_applied")
                if meta.get("vdd_applied") is not None
                else rec.get("vdd_applied"),
                "test_mode": meta.get("test_mode") or rec.get("test_mode"),
                "measurement_value": rec.get("measurement_value"),
                "unit": rec.get("unit") or unit,
                "pass_fail_condition": rec.get("pass_fail_condition"),
            }
        )

    if not rows:
        return {**base, "found": False, "reason": "no_condition_measurements", "conditions": []}
    return {**base, "found": True, "conditions": rows}
