"""In-process analysis sessions for uploaded Jan/Feb/Mar datasets.

Sessions map an opaque ``analysis_session_id`` to a temporary project_root
sandbox. No filesystem paths are exposed to API clients.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SESSION_TTL_SECONDS = 6 * 60 * 60  # 6 hours
_MAX_SESSIONS = 32


@dataclass
class AnalysisSession:
    analysis_session_id: str
    root: Path
    months: tuple[str, ...]
    created_at: float
    source_files: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: float | None = None) -> bool:
        ts = now if now is not None else time.time()
        return (ts - self.created_at) > SESSION_TTL_SECONDS


@dataclass
class JobStatus:
    analysis_session_id: str
    status: str  # queued | processing | completed | failed
    stage: str
    progress_pct: int
    created_at: float
    error: str | None = None
    result_meta: dict[str, Any] = field(default_factory=dict)


_lock = threading.Lock()
_sessions: dict[str, AnalysisSession] = {}
_job_status: dict[str, JobStatus] = {}


class AnalysisSessionError(ValueError):
    """Unknown / expired analysis session."""


def _purge_locked(now: float | None = None) -> None:
    ts = now if now is not None else time.time()
    dead = [sid for sid, s in _sessions.items() if s.is_expired(ts)]
    for sid in dead:
        sess = _sessions.pop(sid, None)
        _job_status.pop(sid, None)
        if sess is not None:
            shutil.rmtree(sess.root, ignore_errors=True)
    # Cap total sessions (oldest first)
    if len(_sessions) > _MAX_SESSIONS:
        ordered = sorted(_sessions.values(), key=lambda s: s.created_at)
        for sess in ordered[: max(0, len(_sessions) - _MAX_SESSIONS)]:
            _sessions.pop(sess.analysis_session_id, None)
            _job_status.pop(sess.analysis_session_id, None)
            shutil.rmtree(sess.root, ignore_errors=True)


def register_session(
    root: Path,
    *,
    months: tuple[str, ...],
    source_files: dict[str, str] | None = None,
    provenance: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> AnalysisSession:
    """Register an already-materialized sandbox as an analysis session."""
    sid = session_id or str(uuid.uuid4())
    sess = AnalysisSession(
        analysis_session_id=sid,
        root=root.resolve(),
        months=tuple(months),
        created_at=time.time(),
        source_files=dict(source_files or {}),
        provenance=dict(provenance or {}),
    )
    with _lock:
        _purge_locked()
        _sessions[sid] = sess
    return sess


def update_job_status(
    analysis_session_id: str,
    *,
    status: str,
    stage: str,
    progress_pct: int,
    error: str | None = None,
    result_meta: dict[str, Any] | None = None,
) -> JobStatus:
    """Update or initialize in-memory job status for an upload session."""
    with _lock:
        existing = _job_status.get(analysis_session_id)
        created_at = existing.created_at if existing else time.time()
        js = JobStatus(
            analysis_session_id=analysis_session_id,
            status=status,
            stage=stage,
            progress_pct=progress_pct,
            created_at=created_at,
            error=error,
            result_meta=dict(result_meta or (existing.result_meta if existing else {})),
        )
        _job_status[analysis_session_id] = js
        return js


def get_job_status(analysis_session_id: str) -> dict[str, Any]:
    """Retrieve job status or return clean failure if server restarted / session lost."""
    with _lock:
        _purge_locked()
        js = _job_status.get(analysis_session_id)
        if js is not None:
            res: dict[str, Any] = {
                "analysis_session_id": js.analysis_session_id,
                "status": js.status,
                "stage": js.stage,
                "progress_pct": js.progress_pct,
                "error": js.error,
            }
            if js.result_meta:
                res.update(js.result_meta)
            return res

        # Check if session already registered and ready
        sess = _sessions.get(analysis_session_id)
        if sess is not None and not sess.is_expired():
            prov = sess.provenance
            return {
                "analysis_session_id": analysis_session_id,
                "status": "completed",
                "stage": "Ready",
                "progress_pct": 100,
                "error": None,
                "months": list(sess.months),
                "used_uploaded_measurements": True,
                "used_static_three_month_measurements": False,
                "source_files": sess.source_files,
                "primary_die": prov.get("primary_die"),
                "scorable_parameters": prov.get("scorable_parameters"),
                "data_provenance": "Analysis generated from uploaded test data",
            }

    # If session is unknown/expired or server rebooted
    return {
        "analysis_session_id": analysis_session_id,
        "status": "failed",
        "stage": "Failed",
        "progress_pct": 0,
        "error": "Server restarted or session expired while processing. Please re-upload dataset.",
    }


def get_session(analysis_session_id: str) -> AnalysisSession:
    with _lock:
        _purge_locked()
        sess = _sessions.get(analysis_session_id)
        if sess is None or sess.is_expired():
            if sess is not None:
                _sessions.pop(analysis_session_id, None)
                _job_status.pop(analysis_session_id, None)
                shutil.rmtree(sess.root, ignore_errors=True)
            raise AnalysisSessionError(
                f"Unknown or expired analysis_session_id={analysis_session_id!r}"
            )
        return sess


def delete_session(analysis_session_id: str) -> bool:
    with _lock:
        sess = _sessions.pop(analysis_session_id, None)
        _job_status.pop(analysis_session_id, None)
    if sess is None:
        return False
    shutil.rmtree(sess.root, ignore_errors=True)
    return True


def clear_all_sessions() -> None:
    """Test / shutdown helper — remove all registered sessions and job statuses."""
    with _lock:
        items = list(_sessions.values())
        _sessions.clear()
        _job_status.clear()
    for sess in items:
        shutil.rmtree(sess.root, ignore_errors=True)
