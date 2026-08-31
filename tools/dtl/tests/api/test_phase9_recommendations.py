"""Phase 9.1–9.3+ recommendation endpoint tests."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dtl_agent.recommendation import recommend
from tests.api.conftest import ARTIFACTS_OK, compare_direct_vs_api, rec_for_parameter


def test_post_recommendations_route_exists(session_app: FastAPI) -> None:
    paths = {getattr(r, "path", None) for r in session_app.routes}
    assert "/api/v1/recommendations" in paths


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_valid_core_recommendation_case_a(session_client: TestClient, session_app: FastAPI) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready")
    body = {
        "lot_id": "DTL_NORM_004",
        "die_id": "DTL_NORM_004_D048",
        "parameters": ["ir_drop"],
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 200
    data = resp.json()
    rec = rec_for_parameter(data, "ir_drop")
    assert rec["decision"] in {"KEEP_CURRENT", "RECOMMEND"}
    assert rec["model_id"] == "core_gru"
    assert rec["current_limit"] == 25.0
    if rec["decision"] == "KEEP_CURRENT":
        assert rec["recommended_limit"] == 25.0
    else:
        assert rec["recommended_limit"] is not None
        assert rec["recommended_limit"] != rec["current_limit"]
        assert rec["explanation"]["policy_reason"] == "max_simulated_yield_selected"
    assert data["request_id"]
    assert data["audit"]["request_id"] == data["request_id"]

    direct = recommend(
        lot_id=body["lot_id"],
        die_id=body["die_id"],
        parameters=body["parameters"],
        config=session_app.state.recommendation_config,
        project_root=session_app.state.project_root,
        model_bundle=session_app.state.model_bundle,
    ).to_dict()
    compare_direct_vs_api(direct=direct, api=data, parameter="ir_drop")


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_valid_parametric_recommendation_case_b(session_client: TestClient, session_app: FastAPI) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready")
    body = {
        "lot_id": "DTL_NORM_004",
        "die_id": "DTL_NORM_004_D048",
        "parameters": ["VMIN"],
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 200
    rec = rec_for_parameter(resp.json(), "VMIN")
    assert rec["model_id"] == "parametric_mlp"
    assert rec["decision"] in {"KEEP_CURRENT", "RECOMMEND"}


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_parametric_only_case_c_no_core_fabrication(session_client: TestClient, session_app: FastAPI) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready")
    body = {
        "lot_id": "DTL_PARAM_VMARGIN_003",
        "die_id": "DTL_PARAM_VMARGIN_003_DIE_041",
        "parameters": ["VMIN"],
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["core_available"] is False
    assert data["parametric_available"] is True
    rec = rec_for_parameter(data, "VMIN")
    assert rec["model_id"] == "parametric_mlp"
    assert rec["decision"] in {"KEEP_CURRENT", "RECOMMEND"}


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_safety_reject_test_time_case_d(session_client: TestClient, session_app: FastAPI) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready")
    body = {
        "lot_id": "DTL_NORM_004",
        "die_id": "DTL_NORM_004_D048",
        "parameters": ["test_time"],
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 200
    rec = rec_for_parameter(resp.json(), "test_time")
    assert rec["decision"] == "REJECT"


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_invalid_parameter_reject_case_g(session_client: TestClient, session_app: FastAPI) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready")
    body = {
        "lot_id": "DTL_NORM_004",
        "die_id": "DTL_NORM_004_D048",
        "parameters": ["INVALID_PARAMETER"],
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 200
    rec = rec_for_parameter(resp.json(), "INVALID_PARAMETER")
    assert rec["decision"] == "REJECT"


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_review_required_no_core_fabrication_case_i(session_client: TestClient, session_app: FastAPI) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready")
    body = {
        "lot_id": "DTL_PARAM_VMARGIN_003",
        "die_id": "DTL_PARAM_VMARGIN_003_DIE_041",
        "parameters": ["ir_drop"],
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 200
    rec = rec_for_parameter(resp.json(), "ir_drop")
    assert rec["decision"] == "REVIEW_REQUIRED"


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_current_protection_case_h(session_client: TestClient, session_app: FastAPI) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready")
    body = {
        "lot_id": "DTL_NORM_004",
        "die_id": "DTL_NORM_004_D048",
        "parameters": ["ir_drop"],
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 200
    data = resp.json()
    rec = rec_for_parameter(data, "ir_drop")
    assert rec["decision"] in {"KEEP_CURRENT", "RECOMMEND"}
    candidates = [c for c in data["audit"]["candidate_set"] if c["parameter"] == "ir_drop"]
    current_rows = [c for c in candidates if c["candidate_limit"] == rec["current_limit"]]
    assert current_rows, "CURRENT must remain in candidate set"


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_http_interconnect_yield_first_acceptance(session_client: TestClient, session_app: FastAPI) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready")
    body = {
        "lot_id": "DTL_NORM_001",
        "die_id": "DTL_NORM_001_D001",
        "parameters": ["INTERCONNECT_RESISTANCE"],
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 200
    rec = rec_for_parameter(resp.json(), "INTERCONNECT_RESISTANCE")
    assert rec["current_limit"] == 15.0
    assert rec["decision"] == "RECOMMEND"
    assert rec["recommended_limit"] == 25.0
    assert rec["explanation"]["policy_reason"] == "max_simulated_yield_selected"
    assert rec["explanation"]["policy_reason"] != "policy_selected_current"


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
def test_missing_simulation_no_fabrication_case_f() -> None:
    from dtl_agent.recommendation.config import RecommendationConfig
    from dtl_agent.recommendation.evidence import SimulationEvidenceLookup
    from tests.api.conftest import ROOT

    lookup = SimulationEvidenceLookup(ROOT, RecommendationConfig())
    miss = lookup.lookup(domain="core", parameter="ir_drop", candidate_limit=123456.789)
    assert miss.found is False
