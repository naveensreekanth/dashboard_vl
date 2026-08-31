"""FastAPI dependencies for service-level configuration and shared model state."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, Query, Request, status

from dtl_agent.api.analysis_session import AnalysisSessionError, get_session
from dtl_agent.api.errors import (
    ModelUnavailableError,
    ServiceError,
    ServiceNotReadyError,
    VALIDATION_ERROR,
    new_request_id,
)
from dtl_agent.api.settings import ServiceSettings
from dtl_agent.recommendation.config import RecommendationConfig
from dtl_agent.recommendation.inference import ModelBundle


def get_settings(request: Request) -> ServiceSettings:
    return request.app.state.settings


def get_recommendation_config(request: Request) -> RecommendationConfig:
    return request.app.state.recommendation_config


def get_model_bundle(request: Request) -> ModelBundle:
    """Return legacy ModelBundle, loading checkpoints/parquets only on first use."""
    bundle = request.app.state.model_bundle
    if bundle is None:
        rid = getattr(request.state, "api_request_id", None) or new_request_id()
        request.state.api_request_id = rid
        raise ModelUnavailableError(request_id=rid)
    if not bundle.ensure_loaded():
        rid = getattr(request.state, "api_request_id", None) or new_request_id()
        request.state.api_request_id = rid
        raise ModelUnavailableError(request_id=rid)
    return bundle


def get_project_root(request: Request) -> Path:
    return request.app.state.project_root


def get_analysis_project_root(
    request: Request,
    analysis_session_id: Annotated[str | None, Query()] = None,
) -> Path:
    """Resolve project root for analysis routes.

    Without ``analysis_session_id`` → repository static root (backward compatible).
    With a valid session id → that session's uploaded sandbox only.
    """
    if analysis_session_id is None or not str(analysis_session_id).strip():
        return request.app.state.project_root
    try:
        sess = get_session(str(analysis_session_id).strip())
    except AnalysisSessionError as exc:
        raise ServiceError(
            code=VALIDATION_ERROR,
            message=str(exc),
            http_status=status.HTTP_404_NOT_FOUND,
        ) from exc
    request.state.analysis_session_id = sess.analysis_session_id
    request.state.analysis_session_provenance = sess.provenance
    return sess.root


def require_ready(request: Request) -> None:
    if not getattr(request.app.state, "ready", False):
        rid = getattr(request.state, "api_request_id", None) or new_request_id()
        request.state.api_request_id = rid
        reason = getattr(request.app.state, "ready_reason", None) or "SERVICE_NOT_READY"
        raise ServiceNotReadyError(reason=reason, request_id=rid)


SettingsDep = Annotated[ServiceSettings, Depends(get_settings)]
RecommendationConfigDep = Annotated[RecommendationConfig, Depends(get_recommendation_config)]
ModelBundleDep = Annotated[ModelBundle, Depends(get_model_bundle)]
ProjectRootDep = Annotated[Path, Depends(get_project_root)]
AnalysisProjectRootDep = Annotated[Path, Depends(get_analysis_project_root)]
ReadyDep = Annotated[None, Depends(require_ready)]
