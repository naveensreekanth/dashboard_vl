"""Sanitized API error envelope and exception handlers (Phase 9.4)."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Stable error codes — only categories with real implementation cases.
VALIDATION_ERROR = "VALIDATION_ERROR"
SERVICE_NOT_READY = "SERVICE_NOT_READY"
CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
ARTIFACT_UNAVAILABLE = "ARTIFACT_UNAVAILABLE"
RECOMMENDATION_ERROR = "RECOMMENDATION_ERROR"
INTERNAL_ERROR = "INTERNAL_ERROR"


def new_request_id() -> str:
    return str(uuid.uuid4())


def error_envelope(
    code: str,
    message: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }


class ServiceError(Exception):
    """Base for service-layer failures mapped to sanitized HTTP responses."""

    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.request_id = request_id or new_request_id()


class ServiceNotReadyError(ServiceError):
    def __init__(self, reason: str = SERVICE_NOT_READY, request_id: str | None = None) -> None:
        super().__init__(
            code=SERVICE_NOT_READY,
            message="Recommendation service is not ready to process requests.",
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
            request_id=request_id,
        )
        self.reason = reason


class ConfigurationError(ServiceError):
    def __init__(self, message: str = "Service configuration is invalid.", request_id: str | None = None) -> None:
        super().__init__(
            code=CONFIGURATION_ERROR,
            message=message,
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id,
        )


class ModelUnavailableError(ServiceError):
    def __init__(self, request_id: str | None = None) -> None:
        super().__init__(
            code=MODEL_UNAVAILABLE,
            message="Required models are not available.",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id,
        )


class ArtifactUnavailableError(ServiceError):
    def __init__(self, request_id: str | None = None) -> None:
        super().__init__(
            code=ARTIFACT_UNAVAILABLE,
            message="Required recommendation artifacts are not available.",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id,
        )


class RecommendationServiceError(ServiceError):
    def __init__(self, request_id: str | None = None) -> None:
        super().__init__(
            code=RECOMMENDATION_ERROR,
            message="Recommendation service could not process the request.",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id,
        )


def _request_id_from_request(request: Request) -> str:
    existing = getattr(request.state, "api_request_id", None)
    if existing:
        return str(existing)
    rid = new_request_id()
    request.state.api_request_id = rid
    return rid


def _validation_message(errors: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for err in errors:
        loc = [str(x) for x in err.get("loc", ()) if x not in ("body",)]
        field = ".".join(loc) if loc else "request"
        msg = str(err.get("msg", "invalid value"))
        parts.append(f"{field}: {msg}")
    return "; ".join(parts) if parts else "Request validation failed."


def register_exception_handlers(app: FastAPI) -> None:
    """Register sanitized exception handlers on the FastAPI app."""

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = _request_id_from_request(request)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_envelope(
                VALIDATION_ERROR,
                _validation_message(list(exc.errors())),
                rid,
            ),
        )

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
        rid = exc.request_id or _request_id_from_request(request)
        return JSONResponse(
            status_code=exc.http_status,
            content=error_envelope(exc.code, exc.message, rid),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        rid = _request_id_from_request(request)
        code = VALIDATION_ERROR if exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY else INTERNAL_ERROR
        detail = exc.detail if isinstance(exc.detail, str) else "Request could not be processed."
        return JSONResponse(
            status_code=exc.status_code,
            content=error_envelope(code, detail, rid),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = _request_id_from_request(request)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_envelope(
                INTERNAL_ERROR,
                "Recommendation service could not process the request.",
                rid,
            ),
        )
