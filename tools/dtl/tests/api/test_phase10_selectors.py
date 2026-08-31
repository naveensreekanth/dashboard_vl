"""Phase 10.9 selector endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_lots_populated_from_backend_dataset(session_client: TestClient) -> None:
    resp = session_client.get("/api/v1/lots")
    assert resp.status_code == 200
    data = resp.json()
    assert "lots" in data
    assert isinstance(data["lots"], list)
    assert len(data["lots"]) > 0


def test_select_lot_loads_only_dies_for_that_lot(session_client: TestClient) -> None:
    lots_resp = session_client.get("/api/v1/lots")
    lot_id = lots_resp.json()["lots"][0]

    dies_resp = session_client.get(f"/api/v1/lots/{lot_id}/dies")
    assert dies_resp.status_code == 200
    payload = dies_resp.json()
    assert payload["lot_id"] == lot_id
    assert isinstance(payload["dies"], list)
    assert len(payload["dies"]) > 0


def test_parameters_for_lot_die_are_data_driven(session_client: TestClient) -> None:
    lots_resp = session_client.get("/api/v1/lots")
    lot_id = lots_resp.json()["lots"][0]
    dies_resp = session_client.get(f"/api/v1/lots/{lot_id}/dies")
    die_id = dies_resp.json()["dies"][0]

    params_resp = session_client.get(f"/api/v1/lots/{lot_id}/dies/{die_id}/parameters")
    assert params_resp.status_code == 200
    payload = params_resp.json()
    assert payload["lot_id"] == lot_id
    assert payload["die_id"] == die_id
    assert isinstance(payload["parameters"], list)
    assert len(payload["parameters"]) > 0


def test_invalid_lot_handled_safely(session_client: TestClient) -> None:
    resp = session_client.get("/api/v1/lots/INVALID_LOT/dies")
    assert resp.status_code == 404
    assert resp.json()["error"]["message"] == "Lot not found."


def test_invalid_die_for_lot_handled_safely(session_client: TestClient) -> None:
    lots_resp = session_client.get("/api/v1/lots")
    lot_id = lots_resp.json()["lots"][0]
    resp = session_client.get(f"/api/v1/lots/{lot_id}/dies/INVALID_DIE/parameters")
    assert resp.status_code == 404
    assert resp.json()["error"]["message"] == "Die not found for lot."
