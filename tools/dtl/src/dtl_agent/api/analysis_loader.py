"""Read-only loader for Phase 12.9 three-month analysis artifacts.

Does not run recommend(), retrain, or alter ML/policy/simulation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from dtl_agent.data.temporal.paths import temporal_artifact_root

ALLOWED_MONTHS = ("2026-01", "2026-02", "2026-03")
SCORABLE_DISPLAY = (
    "IR_DROP_MV",
    "THERMAL_C",
    "VMIN",
    "VMAX",
    "IDDQ",
    "SUPPLY_CURRENT",
    "CONTACT_RESISTANCE",
    "INTERCONNECT_RESISTANCE",
    "ON_RESISTANCE",
)
NON_SCORABLE = ("SETUP_SLACK_PS", "HOLD_SLACK_PS", "TEST_TIME_MS")


class AnalysisArtifactError(FileNotFoundError):
    """Raised when Phase 12.9 analysis artifacts are missing or unreadable."""


def analysis_dir(project_root: Path) -> Path:
    return temporal_artifact_root(project_root) / "shared" / "phase_12_9_analysis"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise AnalysisArtifactError(f"Missing analysis artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalysisArtifactError(f"Missing analysis artifact: {path}")
    df = pd.read_csv(path)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


@lru_cache(maxsize=8)
def load_three_month_bundle(project_root_str: str) -> dict[str, Any]:
    """Load the full Phase 12.9 presentation bundle (cached per process)."""
    root = Path(project_root_str)
    d = analysis_dir(root)
    if not d.is_dir():
        raise AnalysisArtifactError(f"Analysis directory missing: {d}")

    recommendations_json = _read_json(d / "three_month_recommendations.json")
    executive = _read_json(d / "executive_summary.json")
    policy_proofs = _read_json(d / "policy_proofs.json")

    upload_marker_path = temporal_artifact_root(root) / "shared" / "upload_session.json"
    upload_marker: dict[str, Any] | None = None
    if upload_marker_path.is_file():
        try:
            upload_marker = json.loads(upload_marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            upload_marker = None

    used_uploaded = bool(
        (upload_marker or {}).get("used_uploaded_measurements")
        or recommendations_json.get("used_uploaded_measurements")
        or executive.get("used_uploaded_measurements")
    )
    scorable = executive.get("scorable_parameters") or list(SCORABLE_DISPLAY)

    return {
        "source": (
            "uploaded_analysis_session"
            if used_uploaded
            else "artifacts/temporal/shared/phase_12_9_analysis"
        ),
        "disclaimer": (
            "Analysis generated from uploaded test data. Simulated yield is not a "
            "guarantee of production yield."
            if used_uploaded
            else (
                "This dashboard uses synthetic three-month production-like data for "
                "engineering validation and demonstration. Simulated yield is not a "
                "guarantee of production yield."
            )
        ),
        "allowed_months": list(ALLOWED_MONTHS),
        "scorable_parameters": list(scorable),
        "non_scorable_parameters": list(NON_SCORABLE),
        "non_scorable_note": (
            "Recommendation is currently unavailable for SETUP/HOLD/TEST_TIME "
            "because these parameters do not have a candidate/objective path."
        ),
        "executive_summary": executive,
        "primary_recommendations": recommendations_json.get("rows", []),
        "all_recommendations": recommendations_json.get("all_dies_rows", []),
        "candidate_explanations": _read_csv(d / "candidate_explanations.csv"),
        "temporal_changes": _read_csv(d / "temporal_changes.csv"),
        "same_die_analysis": _read_csv(d / "same_die_analysis.csv"),
        "model_traceability": _read_csv(d / "model_traceability.csv"),
        "policy_proofs": policy_proofs,
        "viz_recommended_dtl_by_month": _read_csv(d / "viz_recommended_dtl_by_month.csv"),
        "doc_reference": "docs/PHASE_12_9_THREE_MONTH_RECOMMENDATION_ANALYSIS.md",
        "artifact_reference": "artifacts/temporal/shared/phase_12_9_analysis/",
        "used_uploaded_measurements": used_uploaded,
        "used_static_three_month_measurements": not used_uploaded,
        "analysis_session_id": (upload_marker or {}).get("analysis_session_id"),
        "data_provenance": (
            "Analysis generated from uploaded test data"
            if used_uploaded
            else "Analysis generated from repository static three-month artifacts"
        ),
    }


def clear_analysis_cache() -> None:
    load_three_month_bundle.cache_clear()
