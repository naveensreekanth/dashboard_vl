"""Phase 9 health and readiness tests."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dtl_agent.recommendation.config import RecommendationConfig
from tests.api.conftest import ARTIFACTS_OK, ROOT


def test_health_returns_ok_without_io(session_client: TestClient) -> None:
    resp = session_client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_ready_when_artifacts_present(session_client: TestClient, session_app: FastAPI) -> None:
    resp = session_client.get("/api/v1/ready")
    assert resp.status_code == 200
    if session_app.state.ready:
        assert resp.json() == {"status": "ready"}
    else:
        data = resp.json()
        assert data["status"] == "not_ready"
        assert "reason" in data


def test_ready_not_ready_when_bundle_fails(session_app: FastAPI) -> None:
    prior_ready = session_app.state.ready
    prior_reason = session_app.state.ready_reason
    session_app.state.ready = False
    session_app.state.ready_reason = "MODEL_UNAVAILABLE"
    try:
        client = TestClient(session_app)
        resp = client.get("/api/v1/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "not_ready", "reason": "MODEL_UNAVAILABLE"}
    finally:
        session_app.state.ready = prior_ready
        session_app.state.ready_reason = prior_reason


def test_recommendations_503_when_not_ready(session_app: FastAPI) -> None:
    prior_ready = session_app.state.ready
    prior_reason = session_app.state.ready_reason
    session_app.state.ready = False
    session_app.state.ready_reason = "MODEL_UNAVAILABLE"
    try:
        client = TestClient(session_app)
        resp = client.post(
            "/api/v1/recommendations",
            json={"lot_id": "L1", "die_id": "D1", "parameters": ["ir_drop"]},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"]["code"] == "SERVICE_NOT_READY"
        assert "request_id" in data["error"]
    finally:
        session_app.state.ready = prior_ready
        session_app.state.ready_reason = prior_reason


@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_readiness_checks_config_paths_exist(session_app: FastAPI) -> None:
    cfg: RecommendationConfig = session_app.state.recommendation_config
    for rel in (
        cfg.core_candidate_grid_path,
        cfg.core_candidate_results_path,
        cfg.parametric_candidate_grid_path,
        cfg.parametric_candidate_results_path,
    ):
        assert (ROOT / rel).is_file()
