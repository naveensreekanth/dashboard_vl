"""API tests for Phase 13.0 three-month analysis read endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dtl_agent.api.app import create_app
from dtl_agent.api.settings import ServiceSettings


def _client() -> TestClient:
    app = create_app(ServiceSettings.from_env())
    return TestClient(app)


def test_three_month_bundle_loads():
    client = _client()
    resp = client.get("/api/v1/analysis/three-month")
    assert resp.status_code == 200
    data = resp.json()
    assert "primary_recommendations" in data
    assert len(data["primary_recommendations"]) == 27
    assert data["allowed_months"] == ["2026-01", "2026-02", "2026-03"]
    assert "Simulated yield is not a guarantee" in data["disclaimer"]
    assert data["executive_summary"]["yield_first_proof_count"] >= 1


def test_three_month_recommendation_lookup_and_invalid_month():
    client = _client()
    ok = client.get(
        "/api/v1/analysis/three-month/recommendation",
        params={
            "production_month": "2026-01",
            "parameter_display": "IR_DROP_MV",
            "lot_id": "DTL_NORM_001",
            "die_id": "DTL_NORM_001_D001",
        },
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["recommendation"]["recommended_limit"] == 50.0
    assert body["recommendation"]["model_used"] == "core_gru_temporal_v1"
    assert len(body["three_month_history"]) == 3

    bad = client.get(
        "/api/v1/analysis/three-month/recommendation",
        params={"production_month": "2025-12", "parameter_display": "IR_DROP_MV"},
    )
    assert bad.status_code == 422

    unsupported = client.get(
        "/api/v1/analysis/three-month/recommendation",
        params={"production_month": "2026-01", "parameter_display": "SETUP_SLACK_PS"},
    )
    assert unsupported.status_code == 422
