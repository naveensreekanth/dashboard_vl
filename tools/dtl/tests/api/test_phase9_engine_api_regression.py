"""Formal direct engine vs API regression harness (Phase 9.8)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dtl_agent.recommendation import recommend
from tests.api.conftest import ARTIFACTS_OK, compare_direct_vs_api, rec_for_parameter

REGRESSION_CASES = [
    pytest.param(
        {
            "id": "A",
            "lot_id": "DTL_NORM_004",
            "die_id": "DTL_NORM_004_D048",
            "parameters": ["ir_drop"],
            "parameter": "ir_drop",
            "expected_decision": {"KEEP_CURRENT", "RECOMMEND"},
        },
        id="A_core_ir_drop",
    ),
    pytest.param(
        {
            "id": "B",
            "lot_id": "DTL_NORM_004",
            "die_id": "DTL_NORM_004_D048",
            "parameters": ["VMIN"],
            "parameter": "VMIN",
            "expected_decision": {"KEEP_CURRENT", "RECOMMEND"},
        },
        id="B_parametric_vmin",
    ),
    pytest.param(
        {
            "id": "C",
            "lot_id": "DTL_PARAM_VMARGIN_003",
            "die_id": "DTL_PARAM_VMARGIN_003_DIE_041",
            "parameters": ["VMIN"],
            "parameter": "VMIN",
            "expected_decision": {"KEEP_CURRENT", "RECOMMEND"},
        },
        id="C_parametric_only",
    ),
    pytest.param(
        {
            "id": "D",
            "lot_id": "DTL_NORM_004",
            "die_id": "DTL_NORM_004_D048",
            "parameters": ["test_time"],
            "parameter": "test_time",
            "expected_decision": "REJECT",
        },
        id="D_safety_reject",
    ),
    pytest.param(
        {
            "id": "G",
            "lot_id": "DTL_NORM_004",
            "die_id": "DTL_NORM_004_D048",
            "parameters": ["INVALID_PARAMETER"],
            "parameter": "INVALID_PARAMETER",
            "expected_decision": "REJECT",
        },
        id="G_invalid_parameter",
    ),
    pytest.param(
        {
            "id": "H",
            "lot_id": "DTL_NORM_004",
            "die_id": "DTL_NORM_004_D048",
            "parameters": ["ir_drop"],
            "parameter": "ir_drop",
            "expected_decision": {"KEEP_CURRENT", "RECOMMEND"},
        },
        id="H_current_protection",
    ),
    pytest.param(
        {
            "id": "I",
            "lot_id": "DTL_PARAM_VMARGIN_003",
            "die_id": "DTL_PARAM_VMARGIN_003_DIE_041",
            "parameters": ["ir_drop"],
            "parameter": "ir_drop",
            "expected_decision": "REVIEW_REQUIRED",
        },
        id="I_no_core_fabrication",
    ),
]


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="simulation artifacts missing")
@pytest.mark.parametrize("case", REGRESSION_CASES)
def test_engine_api_regression(case: dict, session_client: TestClient, session_app: FastAPI) -> None:
    if not session_app.state.ready:
        pytest.skip("service not ready for regression")
    body = {
        "lot_id": case["lot_id"],
        "die_id": case["die_id"],
        "parameters": case["parameters"],
    }
    resp = session_client.post("/api/v1/recommendations", json=body)
    assert resp.status_code == 200
    api_data = resp.json()
    rec = rec_for_parameter(api_data, case["parameter"])
    expected = case["expected_decision"]
    if isinstance(expected, (set, frozenset, list, tuple)):
        assert rec["decision"] in expected
    else:
        assert rec["decision"] == expected

    direct = recommend(
        lot_id=body["lot_id"],
        die_id=body["die_id"],
        parameters=body["parameters"],
        config=session_app.state.recommendation_config,
        project_root=session_app.state.project_root,
        model_bundle=session_app.state.model_bundle,
    ).to_dict()
    compare_direct_vs_api(direct=direct, api=api_data, parameter=case["parameter"])


def test_case_e_documented_skip() -> None:
    """Case E (empty safe-set) is not demonstrable with current artifacts."""
    pytest.skip("Case E NOT DEMONSTRABLE WITH CURRENT ARTIFACTS — see PHASE_8_E2E_VALIDATION.md")
