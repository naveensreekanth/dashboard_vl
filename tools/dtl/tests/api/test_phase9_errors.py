"""Phase 9 error sanitization tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tests.api.conftest import ARTIFACTS_OK, assert_no_sensitive_leaks


def test_service_not_ready_sanitized(session_client: TestClient, session_app) -> None:
    prior_ready = session_app.state.ready
    prior_reason = session_app.state.ready_reason
    session_app.state.ready = False
    session_app.state.ready_reason = "MODEL_UNAVAILABLE"
    try:
        resp = session_client.post(
            "/api/v1/recommendations",
            json={"lot_id": "L1", "die_id": "D1"},
        )
        assert resp.status_code == 503
        assert_no_sensitive_leaks(resp.text)
        assert resp.json()["error"]["code"] == "SERVICE_NOT_READY"
    finally:
        session_app.state.ready = prior_ready
        session_app.state.ready_reason = prior_reason


@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_recommendation_internal_error_sanitized(session_client: TestClient) -> None:
    if not session_client.app.state.ready:
        pytest.skip("service not ready")
    with patch("dtl_agent.api.routes.recommendations.recommend", side_effect=RuntimeError("boom")):
        resp = session_client.post(
            "/api/v1/recommendations",
            json={"lot_id": "DTL_NORM_004", "die_id": "DTL_NORM_004_D048", "parameters": ["ir_drop"]},
        )
    assert resp.status_code == 500
    assert_no_sensitive_leaks(resp.text)
    data = resp.json()
    assert data["error"]["code"] == "RECOMMENDATION_ERROR"
    assert data["error"]["request_id"]


def test_validation_error_includes_request_id(session_client: TestClient) -> None:
    resp = session_client.post("/api/v1/recommendations", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["request_id"]
