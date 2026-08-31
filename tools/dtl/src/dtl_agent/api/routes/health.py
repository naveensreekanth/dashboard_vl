"""Health and readiness HTTP routes (Phase 9.5)."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
def get_health() -> dict[str, str]:
    """Confirm the API process is alive — no I/O or inference."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
def get_ready(request: Request) -> dict[str, str]:
    """Confirm the recommendation service is ready to serve requests."""
    if getattr(request.app.state, "ready", False):
        return {"status": "ready"}
    reason = getattr(request.app.state, "ready_reason", None) or "SERVICE_NOT_READY"
    return {"status": "not_ready", "reason": reason}
