"""Phase 9 request validation tests."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tests.api.conftest import assert_no_sensitive_leaks


def test_missing_lot_id_returns_422(session_client: TestClient) -> None:
    resp = session_client.post("/api/v1/recommendations", json={"die_id": "D1"})
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert data["error"]["request_id"]
    assert_no_sensitive_leaks(resp.text)


def test_missing_die_id_returns_422(session_client: TestClient) -> None:
    resp = session_client.post("/api/v1/recommendations", json={"lot_id": "L1"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_empty_lot_id_returns_422(session_client: TestClient) -> None:
    resp = session_client.post(
        "/api/v1/recommendations",
        json={"lot_id": "", "die_id": "D1"},
    )
    assert resp.status_code == 422


def test_invalid_parameter_types_returns_422(session_client: TestClient) -> None:
    resp = session_client.post(
        "/api/v1/recommendations",
        json={"lot_id": "L1", "die_id": "D1", "parameters": "ir_drop"},
    )
    assert resp.status_code == 422


def test_optional_parameters_omitted_allowed(session_client: TestClient, session_app) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready")
    resp = session_client.post(
        "/api/v1/recommendations",
        json={"lot_id": "DTL_NORM_004", "die_id": "DTL_NORM_004_D048"},
    )
    assert resp.status_code in (200, 503)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "project_root",
        "checkpoint_path",
        "policy_config_path",
        "TOP_N",
        "joint_enabled",
        "max_violation_rate_for_recommend",
    ],
)
def test_forbidden_request_fields_rejected(
    session_client: TestClient, forbidden_field: str
) -> None:
    body = {
        "lot_id": "L1",
        "die_id": "D1",
        forbidden_field: "override",
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"
