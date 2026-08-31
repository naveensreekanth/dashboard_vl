"""Typed path helpers for the three-month temporal package (Phase 12.3).

Never silently fall back to legacy ``data/core`` / ``artifacts/simulation`` paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dtl_agent.config.paths import default_project_root

ProductionMonth = Literal["2026-01", "2026-02", "2026-03"]

ALLOWED_PRODUCTION_MONTHS: frozenset[str] = frozenset({"2026-01", "2026-02", "2026-03"})

TEMPORAL_DATA_DIRNAME = "3 months data"


class TemporalPathError(ValueError):
    """Raised when a temporal path or production_month is invalid."""


def validate_production_month(production_month: str) -> ProductionMonth:
    """Return ``production_month`` if allowed; otherwise raise clearly."""
    if production_month not in ALLOWED_PRODUCTION_MONTHS:
        raise TemporalPathError(
            f"Invalid production_month={production_month!r}; "
            f"allowed values are {sorted(ALLOWED_PRODUCTION_MONTHS)}"
        )
    return production_month  # type: ignore[return-value]


def temporal_data_root(project_root: Path | None = None) -> Path:
    """Root of the three-month package: ``data/3 months data/``."""
    root = project_root or default_project_root()
    return root / "data" / TEMPORAL_DATA_DIRNAME


def month_root(production_month: str, project_root: Path | None = None) -> Path:
    """``data/3 months data/{production_month}/``."""
    month = validate_production_month(production_month)
    return temporal_data_root(project_root) / month


def actual_die_root(production_month: str, project_root: Path | None = None) -> Path:
    """``data/3 months data/{production_month}/actual_die/``."""
    return month_root(production_month, project_root) / "actual_die"


def parametric_root(production_month: str, project_root: Path | None = None) -> Path:
    """``data/3 months data/{production_month}/parametric/``."""
    return month_root(production_month, project_root) / "parametric"


def temporal_artifact_root(project_root: Path | None = None) -> Path:
    """``artifacts/temporal/`` (never legacy ``artifacts/simulation``)."""
    root = project_root or default_project_root()
    return root / "artifacts" / "temporal"


def month_simulation_root(production_month: str, project_root: Path | None = None) -> Path:
    """``artifacts/temporal/{production_month}/simulation/``."""
    month = validate_production_month(production_month)
    return temporal_artifact_root(project_root) / month / "simulation"


def month_ml_dataset_root(production_month: str, project_root: Path | None = None) -> Path:
    """``artifacts/temporal/{production_month}/ml_dataset/``."""
    month = validate_production_month(production_month)
    return temporal_artifact_root(project_root) / month / "ml_dataset"


def month_optimization_root(production_month: str, project_root: Path | None = None) -> Path:
    """``artifacts/temporal/{production_month}/optimization/``."""
    month = validate_production_month(production_month)
    return temporal_artifact_root(project_root) / month / "optimization"


def shared_ml_dataset_root(project_root: Path | None = None) -> Path:
    """``artifacts/temporal/shared/ml_dataset/`` (pooled Option D store)."""
    return temporal_artifact_root(project_root) / "shared" / "ml_dataset"
