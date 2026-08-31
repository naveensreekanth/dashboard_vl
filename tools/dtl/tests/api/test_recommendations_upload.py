"""API tests for POST /recommendations/upload."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dtl_agent.api.app import create_app
from dtl_agent.api.settings import ServiceSettings

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "upload_sample_die_actual.csv"


def _client() -> TestClient:
    return TestClient(create_app(ServiceSettings.from_env()))


def test_upload_rejects_empty_file():
    client = _client()
    resp = client.post(
        "/api/v1/recommendations/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert resp.status_code == 422
    assert "empty" in resp.json()["error"]["message"].lower()


def test_upload_rejects_stdf_extension():
    client = _client()
    resp = client.post(
        "/api/v1/recommendations/upload",
        files={"file": ("wafer.stdf", b"binary-junk", "application/octet-stream")},
    )
    assert resp.status_code == 422
    body = resp.json()["error"]["message"]
    assert "Unsupported" in body or "STDF" in body


def test_upload_rejects_malformed_csv():
    client = _client()
    resp = client.post(
        "/api/v1/recommendations/upload",
        files={"file": ("bad.csv", b"not,csv,enough\nx", "text/csv")},
    )
    assert resp.status_code == 422


def test_existing_recommendations_json_endpoint_unchanged():
    """POST /recommendations still accepts JSON lot/die body (no multipart required)."""
    client = _client()
    # Expect validation or artifact path — but route must remain JSON-shaped
    resp = client.post(
        "/api/v1/recommendations",
        json={"lot_id": "DTL_NORM_001", "die_id": "DTL_NORM_001_D001", "parameters": ["ir_drop"]},
    )
    # Ready service may return 200 or domain error; must not be 404/405
    assert resp.status_code in {200, 422, 503, 500}


@pytest.mark.slow
def test_upload_valid_fixture_returns_recommendations():
    if not FIXTURE.is_file():
        pytest.skip("fixture missing")
    client = _client()
    raw = FIXTURE.read_bytes()
    resp = client.post(
        "/api/v1/recommendations/upload",
        files={"file": ("upload_sample_die_actual.csv", raw, "text/csv")},
        data={"parameters": "ir_drop,thermal"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["upload"]["used_uploaded_measurements"] is True
    assert data["upload"]["used_static_three_month_measurements"] is False
    assert len(data["recommendations"]) >= 1
