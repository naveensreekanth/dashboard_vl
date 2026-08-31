"""Path helpers for the DTL agent project root and data domains."""

from __future__ import annotations

from pathlib import Path


def default_project_root() -> Path:
    """Resolve project root as the parent of ``src/`` when installed editable, else CWD."""
    # src/dtl_agent/config/paths.py -> parents[3] = project root
    here = Path(__file__).resolve()
    candidate = here.parents[3]
    if (candidate / "data" / "core").is_dir() and (candidate / "data" / "parametric").is_dir():
        return candidate
    cwd = Path.cwd()
    if (cwd / "data" / "core").is_dir():
        return cwd
    return candidate


def core_data_dir(project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "data" / "core"


def parametric_data_dir(project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return root / "data" / "parametric"
