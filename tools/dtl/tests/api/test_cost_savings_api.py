"""API tests for production-facing read-only /analysis/cost-savings estimator."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dtl_agent.api.app import create_app
from dtl_agent.api.settings import ServiceSettings


def _client() -> TestClient:
    return TestClient(create_app(ServiceSettings.from_env()))


def test_cost_savings_endpoint_read_only_shape():
    client = _client()
    resp = client.get(
        "/api/v1/analysis/cost-savings",
        params={
            "condition_duration_s": 0.05,
            "skip_threshold": 0.10,
            "tester_cost_per_hour": 25.0,
            "include_per_device": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "predicted"
    assert data["is_measured_ate_saving"] is False
    assert data["disclaimer"] == "Counterfactual estimate — not measured ATE savings."
    assert data["estimator"]["mechanism"] == "M2_adaptive_parametric_condition_pruning"
    assert data["estimator"]["production_facing"] is True
    assert data["estimator"]["read_only"] is True
    assert data["estimator"]["condition_duration_s"] == 0.05
    assert data["estimator"]["skip_threshold"] == 0.10
    assert data["estimator"]["tester_cost_per_hour"] == 25.0
    assert data["estimator"]["cost_source"] == "configured assumption"
    assert "assumptions" in data["estimator"]
    agg = data["aggregate"]
    assert agg["records_evaluated"] == 108
    assert agg["production_volume_supplied"] is False
    assert "per_device" in data
    assert len(data["per_device"]) == 108


def test_cost_savings_does_not_modify_recommendation_fields():
    client = _client()
    # Capture a Phase 12.9 recommendation before estimator
    bundle = client.get("/api/v1/analysis/three-month").json()
    sample = next(r for r in bundle["all_recommendations"] if r["parameter"] == "IDDQ")
    before = (sample["current_limit"], sample["recommended_limit"], sample["decision"])

    savings = client.get("/api/v1/analysis/cost-savings").json()
    match = next(
        r
        for r in savings["per_device"]
        if r["die_id"] == sample["die_id"]
        and r["lot_id"] == sample["lot_id"]
        and r["production_month"] == sample["production_month"]
        and r["parameter"] == "IDDQ"
    )
    assert (match["current_limit"], match["recommended_limit"], match["decision"]) == before

    # Bundle still unchanged after estimator call
    bundle2 = client.get("/api/v1/analysis/three-month").json()
    sample2 = next(
        r
        for r in bundle2["all_recommendations"]
        if r["die_id"] == sample["die_id"]
        and r["lot_id"] == sample["lot_id"]
        and r["production_month"] == sample["production_month"]
        and r["parameter"] == "IDDQ"
    )
    assert (sample2["current_limit"], sample2["recommended_limit"], sample2["decision"]) == before


def test_cost_savings_rejects_negative_assumptions():
    client = _client()
    resp = client.get("/api/v1/analysis/cost-savings", params={"tester_cost_per_hour": -1})
    assert resp.status_code == 422
