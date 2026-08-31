"""Structured service-level logging for the Phase 9 API (Phase 9.6)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("dtl_agent.api")


def configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.setLevel(numeric)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request metadata without sensitive payloads or filesystem paths."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id:
            request.state.correlation_id = correlation_id

        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        log_data: dict[str, Any] = {
            "endpoint": request.url.path,
            "method": request.method,
            "http_status": response.status_code,
            "latency_ms": latency_ms,
        }
        if correlation_id:
            log_data["correlation_id"] = correlation_id
        api_rid = getattr(request.state, "api_request_id", None)
        if api_rid:
            log_data["request_id"] = api_rid
        engine_rid = getattr(request.state, "engine_request_id", None)
        if engine_rid:
            log_data["engine_request_id"] = engine_rid
        for key in ("lot_id", "die_id", "parameters", "decisions"):
            val = getattr(request.state, key, None)
            if val is not None:
                log_data[key] = val

        if response.status_code >= 500:
            logger.error(json.dumps(log_data, default=str))
        elif getattr(request.state, "has_review_required", False):
            logger.warning(json.dumps(log_data, default=str))
        else:
            logger.info(json.dumps(log_data, default=str))

        return response
