"""Phase 13.1 — die-level three-month analysis API / service tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dtl_agent.api.app import create_app
from dtl_agent.api.die_level_service import (
    get_die_recommendation,
    load_identity_catalog,
    resolve_parameter,
)
from dtl_agent.api.settings import ServiceSettings

ROOT = Path(__file__).resolve().parents[2]


def _client() -> TestClient:
    app = create_app(ServiceSettings.from_env())
    return TestClient(app)


def test_identity_catalog_stable_structure():
    cat = load_identity_catalog(ROOT)
    assert cat["months"] == ["2026-01", "2026-02", "2026-03"]
    assert cat["categories"] == ["NORMAL", "SCRATCH", "EDGE", "CENTER"]
    assert cat["counts"]["lots"] == 20
    assert cat["counts"]["dies"] == 1000
    assert len(cat["lots_by_category"]["EDGE"]) == 5
    assert len(cat["dies_by_lot"]["DTL_EDGE_003"]) == 50
    assert "DTL_EDGE_003_D025" in cat["dies_by_lot"]["DTL_EDGE_003"]
    assert cat["stable_across_months"] is True


def test_resolve_parameter_and_unsupported():
    eng, disp = resolve_parameter("IR_DROP_MV")
    assert eng == "ir_drop" and disp == "IR_DROP_MV"
    with pytest.raises(ValueError):
        resolve_parameter("SETUP_SLACK_PS")


def test_identities_endpoint():
    client = _client()
    resp = client.get("/api/v1/analysis/three-month/identities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["counts"]["dies"] == 1000
    assert "DTL_NORM_001" in data["lots_by_category"]["NORMAL"]


def test_three_month_bundle_includes_die_level_identities():
    client = _client()
    resp = client.get("/api/v1/analysis/three-month")
    assert resp.status_code == 200
    assert resp.json()["die_level_identities"]["counts"]["lots"] == 20


def _fake_cache_payload(month: str, lot: str, die: str, param_display: str = "IR_DROP_MV"):
    return {
        "recommendation": {
            "production_month": month,
            "lot_category": "EDGE",
            "lot_id": lot,
            "die_id": die,
            "sequence_id": f"{month}::{lot}::{die}",
            "parameter": "ir_drop",
            "parameter_display": param_display,
            "unit": "mV",
            "current_limit": 25.0,
            "recommended_limit": 55.0,
            "max_eligible_simulated_yield": 1.0,
            "ml_score": 0.8,
            "ml_rank": 1,
            "model_used": "core_gru_temporal_v1",
            "model_expected": "core_gru_temporal_v1",
            "decision": "RECOMMEND",
            "policy_reason": "max_simulated_yield_selected",
            "yield_tie": False,
            "selection_text": "Unique max yield",
            "why_selected": "Unique max yield",
            "safety_status": "PASS",
            "source": "recommend_engine_cached",
        },
        "candidates": [
            {
                "candidate_limit": 55.0,
                "simulated_yield": 1.0,
                "safety_status": "PASS",
                "eligible": True,
                "ml_score": 0.8,
                "ml_rank": 1,
                "is_selected": True,
            }
        ],
        "cached": True,
    }


def test_die_recommendation_from_cache_no_engine(tmp_path, monkeypatch):
    """Integrity: cached payload identity must match request (no inventing)."""
    month, lot, die = "2026-01", "DTL_EDGE_003", "DTL_EDGE_003_D025"
    cache_root = tmp_path / "phase_13_1_die_recommendations"
    path = cache_root / month / lot / die / "ir_drop.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_fake_cache_payload(month, lot, die)), encoding="utf-8")

    monkeypatch.setattr(
        "dtl_agent.api.die_level_service.die_rec_root",
        lambda _root: cache_root,
    )
    payload = get_die_recommendation(
        ROOT,
        production_month=month,
        lot_id=lot,
        die_id=die,
        parameter="IR_DROP_MV",
    )
    rec = payload["recommendation"]
    assert rec["production_month"] == month
    assert rec["lot_id"] == lot
    assert rec["die_id"] == die
    assert rec["parameter_display"] == "IR_DROP_MV"
    assert rec["model_used"] == "core_gru_temporal_v1"
    assert payload["candidates"][0]["is_selected"] is True


def test_die_api_invalid_month_and_parameter():
    client = _client()
    bad_m = client.get(
        "/api/v1/analysis/three-month/dies",
        params={
            "production_month": "2025-12",
            "lot_id": "DTL_NORM_001",
            "die_id": "DTL_NORM_001_D001",
            "parameter": "IR_DROP_MV",
        },
    )
    assert bad_m.status_code == 422

    bad_p = client.get(
        "/api/v1/analysis/three-month/dies",
        params={
            "production_month": "2026-01",
            "lot_id": "DTL_NORM_001",
            "die_id": "DTL_NORM_001_D001",
            "parameter": "SETUP_SLACK_PS",
        },
    )
    assert bad_p.status_code == 422


def test_die_api_unknown_die():
    client = _client()
    resp = client.get(
        "/api/v1/analysis/three-month/dies",
        params={
            "production_month": "2026-01",
            "lot_id": "DTL_NORM_001",
            "die_id": "DTL_NORM_001_D999",
            "parameter": "IR_DROP_MV",
        },
    )
    assert resp.status_code == 404


def test_same_die_exists_across_months():
    from dtl_agent.data.temporal.loader import load_temporal_month

    for month in ("2026-01", "2026-02", "2026-03"):
        data = load_temporal_month(month, project_root=ROOT)
        ad = data.actual_die
        mask = (ad["lot_id"] == "DTL_NORM_001") & (ad["die_id"] == "DTL_NORM_001_D001")
        assert mask.any(), f"missing identity in {month}"


def test_die_history_month_isolation_from_cache(tmp_path, monkeypatch):
    lot, die = "DTL_NORM_001", "DTL_NORM_001_D001"
    cache_root = tmp_path / "phase_13_1_die_recommendations"
    for month, limit in (("2026-01", 50.0), ("2026-02", 72.0), ("2026-03", 55.0)):
        path = cache_root / month / lot / die / "ir_drop.json"
        path.parent.mkdir(parents=True)
        payload = _fake_cache_payload(month, lot, die)
        payload["recommendation"]["recommended_limit"] = limit
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "dtl_agent.api.die_level_service.die_rec_root",
        lambda _root: cache_root,
    )
    client = _client()
    hist = client.get(
        "/api/v1/analysis/three-month/die-history",
        params={"lot_id": lot, "die_id": die, "parameter": "IR_DROP_MV"},
    )
    assert hist.status_code == 200
    rows = hist.json()["history"]
    assert len(rows) == 3
    assert [r["production_month"] for r in rows] == ["2026-01", "2026-02", "2026-03"]
    assert [r["recommended_limit"] for r in rows] == [50.0, 72.0, 55.0]
    assert all(r["die_id"] == die for r in rows)
    for r in rows:
        assert r["lot_id"] == lot
        assert r["parameter_display"] == "IR_DROP_MV"


def test_cached_die_endpoint_via_api(tmp_path, monkeypatch):
    month, lot, die = "2026-02", "DTL_EDGE_003", "DTL_EDGE_003_D025"
    cache_root = tmp_path / "phase_13_1_die_recommendations"
    path = cache_root / month / lot / die / "ir_drop.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_fake_cache_payload(month, lot, die)), encoding="utf-8")
    monkeypatch.setattr(
        "dtl_agent.api.die_level_service.die_rec_root",
        lambda _root: cache_root,
    )
    client = _client()
    resp = client.get(
        "/api/v1/analysis/three-month/dies",
        params={
            "production_month": month,
            "lot_id": lot,
            "die_id": die,
            "parameter": "IR_DROP_MV",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommendation"]["die_id"] == die
    assert body["recommendation"]["production_month"] == month
    assert body["candidates"][0]["is_selected"] is True


def test_no_duplicate_die_recommendation_in_cache_payload(tmp_path, monkeypatch):
    month, lot, die = "2026-01", "DTL_NORM_001", "DTL_NORM_001_D001"
    cache_root = tmp_path / "phase_13_1_die_recommendations"
    path = cache_root / month / lot / die / "ir_drop.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_fake_cache_payload(month, lot, die)), encoding="utf-8")
    monkeypatch.setattr(
        "dtl_agent.api.die_level_service.die_rec_root",
        lambda _root: cache_root,
    )
    a = get_die_recommendation(
        ROOT, production_month=month, lot_id=lot, die_id=die, parameter="ir_drop"
    )
    b = get_die_recommendation(
        ROOT, production_month=month, lot_id=lot, die_id=die, parameter="ir_drop"
    )
    assert a["recommendation"]["recommended_limit"] == b["recommendation"]["recommended_limit"]
    assert path.is_file()


def test_lot_dies_uses_cache_only(tmp_path, monkeypatch):
    month, lot = "2026-01", "DTL_NORM_001"
    cache_root = tmp_path / "phase_13_1_die_recommendations"
    die = "DTL_NORM_001_D001"
    path = cache_root / month / lot / die / "ir_drop.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_fake_cache_payload(month, lot, die)), encoding="utf-8")
    monkeypatch.setattr(
        "dtl_agent.api.die_level_service.die_rec_root",
        lambda _root: cache_root,
    )
    monkeypatch.setattr(
        "dtl_agent.api.die_level_service.load_identity_catalog",
        lambda _root: {
            "dies_by_lot": {lot: [die, "DTL_NORM_001_D002"]},
            "lots_by_category": {"NORMAL": [lot]},
            "categories": ["NORMAL"],
            "months": ["2026-01", "2026-02", "2026-03"],
            "counts": {"lots": 1, "dies": 2},
            "stable_across_months": True,
            "note": "test",
            "identities": [],
        },
    )
    client = _client()
    resp = client.get(
        "/api/v1/analysis/three-month/lot-dies",
        params={
            "production_month": month,
            "lot_id": lot,
            "parameter": "IR_DROP_MV",
            "max_dies": 1,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["dies"] == 1
    assert data["dies"][0]["die_id"] == die
