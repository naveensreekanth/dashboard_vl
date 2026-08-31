"""Readiness checks for the recommendation service (Phase 9.5).

Artifact presence only — does not load ModelBundle or training parquets.
Legacy ModelBundle is loaded lazily when /recommendations needs it.
"""

from __future__ import annotations

from pathlib import Path

from dtl_agent.api.errors import ARTIFACT_UNAVAILABLE
from dtl_agent.recommendation.config import RecommendationConfig

_ML_PARQUETS = (
    "train/core_candidate_examples.parquet",
    "train/parametric_candidate_examples.parquet",
    "sequences/core_sequences.parquet",
)


def _artifact_paths(project_root: Path, config: RecommendationConfig) -> list[Path]:
    return [
        config.resolve_path(project_root, config.core_candidate_grid_path),
        config.resolve_path(project_root, config.core_candidate_results_path),
        config.resolve_path(project_root, config.parametric_candidate_grid_path),
        config.resolve_path(project_root, config.parametric_candidate_results_path),
        config.resolve_path(project_root, config.core_checkpoint_path),
        config.resolve_path(project_root, config.parametric_checkpoint_path),
    ]


def check_readiness(
    project_root: Path,
    config: RecommendationConfig,
) -> tuple[bool, str | None]:
    """Return (ready, reason_code). Does not load models or run inference."""
    ml_root = project_root / "artifacts" / "ml_dataset"
    for rel in _ML_PARQUETS:
        if not (ml_root / rel).is_file():
            return False, ARTIFACT_UNAVAILABLE

    for path in _artifact_paths(project_root, config):
        if not path.is_file():
            return False, ARTIFACT_UNAVAILABLE

    return True, None
