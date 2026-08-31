"""Shared pytest fixtures and helpers for Phase 9 API tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dtl_agent.api.app import create_app
from dtl_agent.api.settings import ServiceSettings

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_OK = (ROOT / "artifacts" / "simulation" / "core" / "candidate_grid.csv").is_file()


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def session_app() -> FastAPI:
    return create_app(ServiceSettings(project_root=ROOT))


@pytest.fixture(scope="session")
def session_client(session_app: FastAPI) -> TestClient:
    return TestClient(session_app)


def _norm_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def rec_for_parameter(payload: dict[str, Any], parameter: str) -> dict[str, Any]:
    for r in payload["recommendations"]:
        if r["parameter"] == parameter:
            return r
    raise AssertionError(f"parameter {parameter!r} not in response")


def compare_direct_vs_api(
    *,
    direct: dict[str, Any],
    api: dict[str, Any],
    parameter: str,
) -> None:
    d = rec_for_parameter(direct, parameter)
    a = rec_for_parameter(api, parameter)
    assert d["decision"] == a["decision"]
    assert _norm_scalar(d["current_limit"]) == _norm_scalar(a["current_limit"])
    assert _norm_scalar(d["recommended_limit"]) == _norm_scalar(a["recommended_limit"])
    assert _norm_scalar(d["ml_score"]) == _norm_scalar(a["ml_score"])
    assert _norm_scalar(d["ml_rank"]) == _norm_scalar(a["ml_rank"])
    assert d["safety_result"] == a["safety_result"]
    assert d["evidence_level"] == a["evidence_level"]
    assert d["simulation_evidence"] == a["simulation_evidence"]
    d_cands = [c for c in direct["audit"]["candidate_set"] if c["parameter"] == parameter]
    a_cands = [c for c in api["audit"]["candidate_set"] if c["parameter"] == parameter]
    assert len(d_cands) == len(a_cands)
    for dc, ac in zip(
        sorted(d_cands, key=lambda x: (_norm_scalar(x.get("ml_rank")) or 0, x["candidate_limit"])),
        sorted(a_cands, key=lambda x: (_norm_scalar(x.get("ml_rank")) or 0, x["candidate_limit"])),
    ):
        assert dc["candidate_limit"] == ac["candidate_limit"]
        assert _norm_scalar(dc["ml_score"]) == _norm_scalar(ac["ml_score"])
        assert _norm_scalar(dc["ml_rank"]) == _norm_scalar(ac["ml_rank"])


def assert_no_sensitive_leaks(body: str) -> None:
    lowered = body.lower()
    forbidden = [
        "traceback",
        "file \"",
        "core_gru_best.pt",
        "parametric_mlp_best.pt",
        "model_load_error",
        "data/core/evaluation",
    ]
    for token in forbidden:
        assert token not in lowered, f"sensitive leak found: {token}"
