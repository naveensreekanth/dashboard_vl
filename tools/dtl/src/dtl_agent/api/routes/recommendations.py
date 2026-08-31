"""Recommendation HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile, status

from dtl_agent.api.dependencies import (
    ModelBundleDep,
    ProjectRootDep,
    ReadyDep,
    RecommendationConfigDep,
)
from dtl_agent.api.errors import (
    ArtifactUnavailableError,
    RecommendationServiceError,
    ServiceError,
    VALIDATION_ERROR,
    new_request_id,
)
from dtl_agent.api.schemas import RecommendationRequest
from dtl_agent.api.upload_recommendation import (
    UploadRecommendationError,
    recommend_from_upload,
)
from dtl_agent.recommendation import recommend

router = APIRouter(tags=["recommendations"])


@router.post(
    "/recommendations",
    status_code=status.HTTP_200_OK,
    summary="Run Phase 8 advisory recommendation for one lot/die",
)
def post_recommendations(
    body: RecommendationRequest,
    request: Request,
    _ready: ReadyDep,
    config: RecommendationConfigDep,
    project_root: ProjectRootDep,
    model_bundle: ModelBundleDep,
) -> dict:
    """Thin wrapper over ``recommend()`` — domain decisions are returned in the body."""
    request.state.lot_id = body.lot_id
    request.state.die_id = body.die_id
    request.state.parameters = body.parameters

    try:
        result = recommend(
            lot_id=body.lot_id,
            die_id=body.die_id,
            parameters=body.parameters,
            config=config,
            project_root=project_root,
            model_bundle=model_bundle,
            production_month=body.production_month,
        )
    except ValueError as exc:
        rid = new_request_id()
        request.state.api_request_id = rid
        raise ServiceError(
            code=VALIDATION_ERROR,
            message=str(exc),
            http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            request_id=rid,
        ) from exc
    except FileNotFoundError as exc:
        rid = new_request_id()
        request.state.api_request_id = rid
        raise ArtifactUnavailableError(request_id=rid) from exc
    except Exception as exc:  # noqa: BLE001
        rid = new_request_id()
        request.state.api_request_id = rid
        raise RecommendationServiceError(request_id=rid) from exc

    payload = result.to_dict()
    request.state.engine_request_id = payload.get("request_id")
    request.state.api_request_id = payload.get("request_id")
    decisions = [r.get("decision") for r in payload.get("recommendations", [])]
    request.state.decisions = decisions
    if any(d == "REVIEW_REQUIRED" for d in decisions):
        request.state.has_review_required = True
    return payload


@router.post(
    "/recommendations/upload",
    status_code=status.HTTP_200_OK,
    summary=(
        "Upload temporal measurements CSV/ZIP and run existing recommend() "
        "on that uploaded data only (not static three-month artifacts)"
    ),
)
async def post_recommendations_upload(
    request: Request,
    _ready: ReadyDep,
    project_root: ProjectRootDep,
    file: UploadFile = File(..., description="actual_die measurements CSV or ZIP"),
    parametric_file: UploadFile | None = File(
        None, description="Optional parametric measurements CSV"
    ),
    parameters: str | None = Form(
        None,
        description="Optional comma-separated parameter list",
    ),
) -> dict:
    """Multipart upload → sandbox materialization → existing recommend() pipeline."""
    rid = new_request_id()
    request.state.api_request_id = rid

    raw = await file.read()
    param_raw: bytes | None = None
    param_name: str | None = None
    if parametric_file is not None and parametric_file.filename:
        param_raw = await parametric_file.read()
        param_name = parametric_file.filename

    param_list: list[str] | None = None
    if parameters and parameters.strip():
        param_list = [p.strip() for p in parameters.split(",") if p.strip()]

    try:
        payload = recommend_from_upload(
            filename=file.filename or "upload.csv",
            content=raw,
            source_root=project_root,
            parametric_filename=param_name,
            parametric_content=param_raw,
            parameters=param_list,
        )
    except UploadRecommendationError as exc:
        raise ServiceError(
            code=VALIDATION_ERROR,
            message=str(exc),
            http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            request_id=rid,
        ) from exc
    except FileNotFoundError as exc:
        raise ArtifactUnavailableError(request_id=rid) from exc
    except Exception as exc:  # noqa: BLE001
        raise RecommendationServiceError(request_id=rid) from exc

    request.state.lot_id = (payload.get("upload") or {}).get("lot_id")
    request.state.die_id = (payload.get("upload") or {}).get("die_id")
    request.state.engine_request_id = payload.get("request_id")
    request.state.api_request_id = payload.get("request_id") or rid
    decisions = [r.get("decision") for r in payload.get("recommendations", [])]
    request.state.decisions = decisions
    if any(d == "REVIEW_REQUIRED" for d in decisions):
        request.state.has_review_required = True
    return payload
