"""Phase 13.1 — die-level temporal recommendation service (engine-backed, cached).

Uses existing ``recommend(production_month=...)`` only. Does not invent winners,
retrain, or alter policy/safety/ML. Results are cached under
``artifacts/temporal/shared/phase_13_1_die_recommendations/``.

Phase 13.3A: process-level reuse of TemporalHybridBundle, identity catalog,
and in-memory recommendation payloads keyed by month::lot::die::parameter.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.loader import load_temporal_die
from dtl_agent.data.temporal.paths import (
    actual_die_root,
    temporal_artifact_root,
    validate_production_month,
)
from dtl_agent.recommendation import recommend
from dtl_agent.recommendation.routing import model_for_parameter
from dtl_agent.recommendation.schemas import CORE_PARAMETERS
from dtl_agent.recommendation.temporal_inference import TemporalHybridBundle

CATEGORIES = ("NORMAL", "SCRATCH", "EDGE", "CENTER")
MONTH_ORDER = ("2026-01", "2026-02", "2026-03")
DISPLAY_TO_ENGINE = {
    "IR_DROP_MV": "ir_drop",
    "THERMAL_C": "thermal",
    "VMIN": "VMIN",
    "VMAX": "VMAX",
    "IDDQ": "IDDQ",
    "SUPPLY_CURRENT": "SUPPLY_CURRENT",
    "CONTACT_RESISTANCE": "CONTACT_RESISTANCE",
    "INTERCONNECT_RESISTANCE": "INTERCONNECT_RESISTANCE",
    "ON_RESISTANCE": "ON_RESISTANCE",
}
ENGINE_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_ENGINE.items()}
SCORABLE_ENGINE = tuple(DISPLAY_TO_ENGINE.values())

_lock = threading.Lock()
_bundle: TemporalHybridBundle | None = None
_identity_catalog: dict[str, Any] | None = None
_identity_catalog_root: str | None = None
# In-memory recommendation payloads: month::lot::die::parameter → payload
_rec_memory: dict[str, dict[str, Any]] = {}


def die_rec_root(project_root: Path) -> Path:
    return temporal_artifact_root(project_root) / "shared" / "phase_13_1_die_recommendations"


def _cache_key(
    project_root: Path, month: str, lot_id: str, die_id: str, engine_param: str
) -> str:
    """Strict isolation key — roots/months/lots/dies/parameters never share entries."""
    root_key = str(project_root.resolve())
    return f"{root_key}::{month}::{lot_id}::{die_id}::{engine_param}"


def _get_bundle(project_root: Path) -> TemporalHybridBundle:
    global _bundle
    with _lock:
        if _bundle is None or _bundle.project_root != project_root:
            _bundle = TemporalHybridBundle(project_root)
            if not _bundle.ensure_loaded():
                raise RuntimeError(f"Temporal hybrid bundle load failed: {_bundle.load_errors}")
        return _bundle


def resolve_parameter(parameter: str) -> tuple[str, str]:
    """Return (engine_name, display_name)."""
    p = str(parameter).strip()
    if p in DISPLAY_TO_ENGINE:
        return DISPLAY_TO_ENGINE[p], p
    if p in ENGINE_TO_DISPLAY:
        return p, ENGINE_TO_DISPLAY[p]
    raise ValueError(
        f"Unsupported or unknown parameter={parameter!r}; "
        f"scorable={sorted(DISPLAY_TO_ENGINE)}"
    )


def identity_catalog_path(project_root: Path) -> Path:
    return temporal_artifact_root(project_root) / "shared" / "identity_catalog.json"


def _catalog_from_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    lots = sorted({r["lot_id"] for r in rows})
    for lot in lots:
        cat = next(r["lot_category"] for r in rows if r["lot_id"] == lot)
        by_cat.setdefault(cat, []).append(lot)
    for c in by_cat:
        by_cat[c] = sorted(set(by_cat[c]))

    dies_by_lot: dict[str, list[str]] = {}
    for lot in lots:
        dies_by_lot[lot] = sorted({r["die_id"] for r in rows if r["lot_id"] == lot})

    return {
        "months": list(MONTH_ORDER),
        "categories": list(CATEGORIES),
        "lots_by_category": by_cat,
        "dies_by_lot": dies_by_lot,
        "identities": rows,
        "counts": {
            "lots": len(lots),
            "dies": len(rows),
            "dies_per_lot": {lot: len(dies_by_lot[lot]) for lot in lots},
            "by_category_dies": {
                c: sum(1 for r in rows if r["lot_category"] == c) for c in CATEGORIES
            },
        },
        "stable_across_months": True,
        "note": (
            "Lot/die identities are stable across 2026-01/02/03. "
            "Phase 12.9 analysis only precomputed 4 representative dies; "
            "die-level recommendations are produced on demand via recommend() and cached."
        ),
    }


def _build_identity_rows_from_jan(project_root: Path) -> list[dict[str, str]]:
    """Extract unique lot/die/category from Jan using slim column streaming only."""
    import pandas as pd

    path = actual_die_root("2026-01", project_root) / "measurements.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing identity source: {path}")
    seen: dict[tuple[str, str], str] = {}
    for chunk in pd.read_csv(
        path,
        usecols=["lot_id", "die_id", "lot_category"],
        chunksize=100_000,
        low_memory=False,
    ):
        for r in chunk.drop_duplicates().itertuples(index=False):
            key = (str(r.lot_id), str(r.die_id))
            if key not in seen:
                seen[key] = str(r.lot_category)
        del chunk
    rows = [
        {"lot_id": lot, "die_id": die, "lot_category": cat}
        for (lot, die), cat in sorted(seen.items())
    ]
    return rows


def load_identity_catalog(project_root: Path) -> dict[str, Any]:
    """Lot/die identity index for dashboard selectors (no full month DataFrames)."""
    global _identity_catalog, _identity_catalog_root
    root_key = str(project_root.resolve())
    with _lock:
        if _identity_catalog is not None and _identity_catalog_root == root_key:
            return _identity_catalog

    artifact = identity_catalog_path(project_root)
    rows: list[dict[str, str]] | None = None
    if artifact.is_file():
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            raw = payload.get("identities") if isinstance(payload, dict) else None
            if isinstance(raw, list) and raw:
                rows = [
                    {
                        "lot_id": str(r["lot_id"]),
                        "die_id": str(r["die_id"]),
                        "lot_category": str(r["lot_category"]),
                    }
                    for r in raw
                ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            rows = None

    if rows is None:
        rows = _build_identity_rows_from_jan(project_root)
        catalog = _catalog_from_rows(rows)
    else:
        catalog = _catalog_from_rows(rows)

    with _lock:
        _identity_catalog = catalog
        _identity_catalog_root = root_key
    return catalog


def _lot_category_from_catalog(
    project_root: Path, lot_id: str, die_id: str
) -> str | None:
    catalog = load_identity_catalog(project_root)
    for row in catalog["identities"]:
        if row["lot_id"] == lot_id and row["die_id"] == die_id:
            return str(row["lot_category"])
    return None


def _cache_path(
    project_root: Path, month: str, lot_id: str, die_id: str, engine_param: str
) -> Path:
    return (
        die_rec_root(project_root)
        / month
        / lot_id
        / die_id
        / f"{engine_param}.json"
    )


def _serialize_recommendation(rec, *, month: str, lot_category: str | None) -> dict[str, Any]:
    d = rec.to_dict()
    exp = d.get("explanation") or {}
    sim = d.get("simulation_evidence") or {}
    eng = str(d["parameter"])
    return {
        "production_month": month,
        "lot_category": lot_category,
        "lot_id": d["lot_id"],
        "die_id": d["die_id"],
        "sequence_id": make_sequence_id(d["lot_id"], d["die_id"], month),
        "parameter": eng,
        "parameter_display": ENGINE_TO_DISPLAY.get(eng, eng),
        "unit": d.get("unit"),
        "current_limit": d.get("current_limit"),
        "recommended_limit": d.get("recommended_limit"),
        "recommendation_delta": (
            None
            if d.get("recommended_limit") is None or d.get("current_limit") is None
            else float(d["recommended_limit"]) - float(d["current_limit"])
        ),
        "max_eligible_simulated_yield": exp.get("selected_simulated_yield", sim.get("simulated_yield")),
        "ml_score": d.get("ml_score"),
        "ml_rank": d.get("ml_rank"),
        "model_used": d.get("model_used") or d.get("model_id"),
        "model_expected": model_for_parameter(eng, temporal=True).value,
        "decision": d.get("decision"),
        "policy_reason": exp.get("policy_reason"),
        "yield_tie": bool(exp.get("yield_tie")),
        "tie_breaker": exp.get("tie_breaker"),
        "selection_text": exp.get("selection_text"),
        "explanation_text": exp.get("text"),
        "why_selected": exp.get("text") or exp.get("selection_text"),
        "safety_status": (d.get("safety_result") or {}).get("status"),
        "evidence_origin": d.get("evidence_origin"),
        "source": "recommend_engine_cached",
    }


def _candidates_from_audit(result, parameter: str) -> list[dict[str, Any]]:
    audit = result.audit or {}
    param = str(parameter)
    sims = [
        s
        for s in (audit.get("simulation_evidence_rows") or [])
        if str(s.get("parameter")) == param
    ]
    saf = [
        s
        for s in (audit.get("safety_check_trace") or [])
        if str(s.get("parameter")) == param
    ]
    cand_by = {
        float(c["candidate_limit"]): c
        for c in (audit.get("candidate_set") or [])
        if str(c.get("parameter")) == param and c.get("candidate_limit") is not None
    }
    out: list[dict[str, Any]] = []
    for i, sim in enumerate(sims):
        lim = float(sim["candidate_limit"])
        c = cand_by.get(lim, {})
        status = str((saf[i] if i < len(saf) else {}).get("status") or "")
        out.append(
            {
                "candidate_limit": lim,
                "simulated_yield": sim.get("simulated_yield"),
                "safety_status": status,
                "eligible": status.upper() == "PASS",
                "in_policy_gate_set": True,
                "ml_score": c.get("ml_score"),
                "ml_rank": c.get("ml_rank"),
                "is_current": str(c.get("tighten_or_loosen", "")).upper() == "CURRENT"
                or abs(float(c.get("candidate_delta", 1))) < 1e-12,
                "is_selected": False,
            }
        )
    return out


def _payload_valid(
    payload: dict[str, Any],
    *,
    month: str,
    lot_id: str,
    die_id: str,
    engine_param: str,
) -> bool:
    rec = payload.get("recommendation")
    if not isinstance(rec, dict):
        return False
    return (
        str(rec.get("production_month")) == month
        and str(rec.get("lot_id")) == lot_id
        and str(rec.get("die_id")) == die_id
        and str(rec.get("parameter")) == engine_param
    )


def get_die_recommendation(
    project_root: Path,
    *,
    production_month: str,
    lot_id: str,
    die_id: str,
    parameter: str,
    force_refresh: bool = False,
) -> dict[str, Any]:
    month = validate_production_month(production_month)
    engine_param, display = resolve_parameter(parameter)
    mem_key = _cache_key(project_root, month, lot_id, die_id, engine_param)
    path = _cache_path(project_root, month, lot_id, die_id, engine_param)

    if not force_refresh:
        with _lock:
            mem_hit = _rec_memory.get(mem_key)
        if mem_hit is not None and _payload_valid(
            mem_hit, month=month, lot_id=lot_id, die_id=die_id, engine_param=engine_param
        ):
            out = dict(mem_hit)
            out["cached"] = True
            out["cache_source"] = "memory"
            return out
        if path.is_file():
            try:
                disk_payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                disk_payload = None
            if disk_payload is not None and _payload_valid(
                disk_payload,
                month=month,
                lot_id=lot_id,
                die_id=die_id,
                engine_param=engine_param,
            ):
                disk_payload = dict(disk_payload)
                disk_payload["cached"] = True
                disk_payload["cache_source"] = "disk"
                with _lock:
                    _rec_memory[mem_key] = disk_payload
                return disk_payload

    # Validate identity via slim catalog (no full-month load).
    lot_category = _lot_category_from_catalog(project_root, lot_id, die_id)
    if lot_category is None:
        raise KeyError(f"Unknown die identity {lot_id}/{die_id} for month={month}")

    bundle = _get_bundle(project_root)
    result = recommend(
        lot_id=lot_id,
        die_id=die_id,
        parameters=[engine_param],
        production_month=month,
        project_root=project_root,
        temporal_bundle=bundle,
    )
    if not result.recommendations:
        raise RuntimeError("recommend() returned no recommendations")
    rec = result.recommendations[0]
    row = _serialize_recommendation(rec, month=month, lot_category=lot_category)
    cands = _candidates_from_audit(result, engine_param)
    # Mark selected
    if row.get("recommended_limit") is not None:
        for c in cands:
            if abs(float(c["candidate_limit"]) - float(row["recommended_limit"])) < 1e-12:
                c["is_selected"] = True
    payload = {
        "recommendation": row,
        "candidates": cands,
        "cached": False,
        "cache_path": str(path).replace("\\", "/"),
        "cache_source": "live",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with _lock:
        _rec_memory[mem_key] = payload
    return payload


def get_die_history(
    project_root: Path,
    *,
    lot_id: str,
    die_id: str,
    parameter: str,
) -> dict[str, Any]:
    engine_param, display = resolve_parameter(parameter)
    history = []
    for month in MONTH_ORDER:
        payload = get_die_recommendation(
            project_root,
            production_month=month,
            lot_id=lot_id,
            die_id=die_id,
            parameter=engine_param,
        )
        rec = payload["recommendation"]
        # Hard isolation: never accept a row from another month
        if str(rec.get("production_month")) != month:
            raise RuntimeError(
                f"History cache contamination: expected month={month}, "
                f"got {rec.get('production_month')}"
            )
        history.append(rec)
    return {
        "lot_id": lot_id,
        "die_id": die_id,
        "parameter": engine_param,
        "parameter_display": display,
        "history": history,
    }


def observed_summary(
    project_root: Path,
    *,
    lot_id: str,
    die_id: str,
) -> dict[str, Any]:
    """Compact observed means across months for available measurements."""
    out: dict[str, dict[str, float | None]] = {}
    params = ["ir_drop", "thermal", "VMIN", "VMAX", "IDDQ"]
    for month in MONTH_ORDER:
        data = load_temporal_die(month, lot_id, die_id, project_root=project_root)
        for param in params:
            src = data.actual_die if param in CORE_PARAMETERS else data.parametric
            if src.empty:
                sub = src
            else:
                sub = src[
                    (src["lot_id"].astype(str) == lot_id)
                    & (src["die_id"].astype(str) == die_id)
                    & (src["parameter"].astype(str) == param)
                ]
            key = ENGINE_TO_DISPLAY.get(param, param)
            out.setdefault(key, {})
            if sub.empty:
                out[key][month] = None
            else:
                out[key][month] = float(sub["measurement_value"].astype(float).mean())
    return {"lot_id": lot_id, "die_id": die_id, "observed_means": out}


def lot_die_browse(
    project_root: Path,
    *,
    production_month: str,
    lot_id: str,
    parameter: str,
    max_dies: int | None = None,
) -> dict[str, Any]:
    """Recommendations for dies in a lot (engine-backed, cached)."""
    month = validate_production_month(production_month)
    engine_param, display = resolve_parameter(parameter)
    catalog = load_identity_catalog(project_root)
    dies = catalog["dies_by_lot"].get(lot_id)
    if not dies:
        raise KeyError(f"Unknown lot_id={lot_id}")
    if max_dies is not None:
        dies = dies[: max(0, int(max_dies))]
    rows = []
    for die_id in dies:
        payload = get_die_recommendation(
            project_root,
            production_month=month,
            lot_id=lot_id,
            die_id=die_id,
            parameter=engine_param,
        )
        rows.append(payload["recommendation"])
    decisions = Counter(str(r.get("decision")) for r in rows)
    rec_vals = [float(r["recommended_limit"]) for r in rows if r.get("recommended_limit") is not None]
    return {
        "production_month": month,
        "lot_id": lot_id,
        "parameter": engine_param,
        "parameter_display": display,
        "dies": rows,
        "summary": {
            "dies": len(rows),
            "decision_counts": dict(decisions),
            "average_recommended_dtl": (sum(rec_vals) / len(rec_vals)) if rec_vals else None,
            "min_recommended_dtl": min(rec_vals) if rec_vals else None,
            "max_recommended_dtl": max(rec_vals) if rec_vals else None,
        },
    }


def cache_coverage(project_root: Path) -> dict[str, Any]:
    root = die_rec_root(project_root)
    if not root.is_dir():
        return {"cached_files": 0, "by_month": {}}
    files = list(root.rglob("*.json"))
    by_month: dict[str, int] = {}
    for f in files:
        parts = f.relative_to(root).parts
        if parts:
            by_month[parts[0]] = by_month.get(parts[0], 0) + 1
    return {"cached_files": len(files), "by_month": by_month}


_cost_savings_memory: dict[str, dict[str, Any]] = {}


def _cost_savings_cache_key(
    project_root: Path,
    production_month: str | None,
    lot_id: str,
    die_id: str,
    assumptions_key: tuple[float, float, float],
) -> str:
    root_key = str(project_root.resolve())
    month_key = production_month or "three-month"
    return f"{root_key}::{month_key}::{lot_id}::{die_id}::{assumptions_key}"


def get_die_recommendation_rows_for_cost_savings(
    project_root: Path,
    *,
    lot_id: str,
    die_id: str,
    production_month: str | None = None,
) -> list[dict[str, Any]]:
    months = [validate_production_month(production_month)] if production_month else list(MONTH_ORDER)

    # Check catalog first to validate die identity exists
    catalog = load_identity_catalog(project_root)
    lot_category = _lot_category_from_catalog(project_root, lot_id, die_id)
    if lot_category is None:
        raise KeyError(f"Unknown die identity {lot_id}/{die_id}")

    rows: list[dict[str, Any]] = []
    params = list(SCORABLE_ENGINE)

    for month in months:
        for engine_param in params:
            payload = get_die_recommendation(
                project_root,
                production_month=month,
                lot_id=lot_id,
                die_id=die_id,
                parameter=engine_param,
            )
            rec = payload.get("recommendation")
            if rec:
                rows.append(rec)
    return rows


def clear_die_level_process_caches() -> None:
    """Drop in-memory recommendation / identity / bundle handles (tests / reload)."""
    global _bundle, _identity_catalog, _identity_catalog_root
    with _lock:
        _bundle = None
        _identity_catalog = None
        _identity_catalog_root = None
        _rec_memory.clear()
        _cost_savings_memory.clear()
