"""Phase 9 application foundation tests."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI

from tests.api.conftest import ROOT


def test_create_app_returns_fastapi_application(session_app: FastAPI) -> None:
    assert session_app.title == "DTL Agent API"
    assert hasattr(session_app.state, "model_bundle")
    assert hasattr(session_app.state, "recommendation_config")
    assert hasattr(session_app.state, "ready")


def test_router_registration(session_app: FastAPI) -> None:
    paths = {getattr(r, "path", None) for r in session_app.routes}
    assert "/api/v1/recommendations" in paths
    assert "/api/v1/health" in paths
    assert "/api/v1/ready" in paths
    assert "/api/v1/lots" in paths
    assert "/api/v1/lots/{lot_id}/dies" in paths
    assert "/api/v1/lots/{lot_id}/dies/{die_id}/parameters" in paths


def test_configuration_on_app_state(session_app: FastAPI, root) -> None:
    assert session_app.state.settings is not None
    assert session_app.state.project_root == root
    assert session_app.state.recommendation_config is not None
