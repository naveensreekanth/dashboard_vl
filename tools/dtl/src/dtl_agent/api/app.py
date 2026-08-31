"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dtl_agent.api.errors import CONFIGURATION_ERROR, register_exception_handlers
from dtl_agent.api.logging_config import RequestLoggingMiddleware, configure_logging
from dtl_agent.api.readiness import check_readiness
from dtl_agent.api.routes import analysis, health, measurements, recommendations, selectors
from dtl_agent.api.settings import ServiceSettings
from dtl_agent.recommendation.config import load_recommendation_config
from dtl_agent.recommendation.inference import ModelBundle


def _build_app_state(svc: ServiceSettings) -> dict:
    project_root = svc.resolved_project_root()
    try:
        recommendation_config = load_recommendation_config(svc.policy_config_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "settings": svc,
            "project_root": project_root,
            "recommendation_config": None,
            "model_bundle": None,
            "ready": False,
            "ready_reason": CONFIGURATION_ERROR,
            "startup_error": str(type(exc).__name__),
        }

    # Construct ModelBundle but do NOT ensure_loaded() — training parquets stay on disk
    # until a legacy /recommendations route needs them. Three-month routes use
    # TemporalHybridBundle via die_level_service instead.
    model_bundle = ModelBundle(project_root, recommendation_config)
    ready, reason = check_readiness(project_root, recommendation_config)

    return {
        "settings": svc,
        "project_root": project_root,
        "recommendation_config": recommendation_config,
        "model_bundle": model_bundle,
        "ready": ready,
        "ready_reason": reason,
        "startup_error": None,
    }


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    """Build a FastAPI app with shared service config and ModelBundle on app state."""
    svc = settings or ServiceSettings.from_env()
    configure_logging(svc.log_level)
    state = _build_app_state(svc)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Option B: process starts even if not ready; /ready reflects state.
        yield

    app = FastAPI(
        title="DTL Agent API",
        version="0.9.0",
        description="Thin HTTP wrapper over the Phase 8 recommendation engine.",
        lifespan=lifespan,
    )

    app.state.settings = state["settings"]
    app.state.project_root = state["project_root"]
    app.state.recommendation_config = state["recommendation_config"]
    app.state.model_bundle = state["model_bundle"]
    app.state.ready = state["ready"]
    app.state.ready_reason = state["ready_reason"]

    origins = svc.parsed_cors_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Correlation-ID"],
        )

    app.add_middleware(RequestLoggingMiddleware)
    register_exception_handlers(app)

    prefix = svc.api_prefix.rstrip("/") or "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(selectors.router, prefix=prefix)
    app.include_router(measurements.router, prefix=prefix)
    app.include_router(recommendations.router, prefix=prefix)
    app.include_router(analysis.router, prefix=prefix)

    return app
