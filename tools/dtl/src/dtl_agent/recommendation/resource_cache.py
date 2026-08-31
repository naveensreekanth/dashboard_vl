"""Process-level reuse of month-scoped recommendation I/O (Phase 13.3A).

Caches CandidateCatalog and SimulationEvidenceLookup keyed by production_month
(+ project root + evidence paths). Does not change lookup/policy semantics.
"""

from __future__ import annotations

import threading
from pathlib import Path

from dtl_agent.recommendation.catalog import CandidateCatalog
from dtl_agent.recommendation.config import RecommendationConfig
from dtl_agent.recommendation.evidence import SimulationEvidenceLookup

_lock = threading.Lock()
_catalogs: dict[tuple[str, str, str, str], CandidateCatalog] = {}
_evidence: dict[tuple[str, str, str, str], SimulationEvidenceLookup] = {}


def _paths_key(project_root: Path, config: RecommendationConfig) -> tuple[str, str, str, str]:
    return (
        str(project_root.resolve()),
        str(config.core_candidate_grid_path),
        str(config.parametric_candidate_grid_path),
        str(config.core_candidate_results_path),
    )


def get_candidate_catalog(project_root: Path, config: RecommendationConfig) -> CandidateCatalog:
    key = _paths_key(project_root, config)
    with _lock:
        hit = _catalogs.get(key)
        if hit is not None:
            return hit
        cat = CandidateCatalog(project_root, config)
        _catalogs[key] = cat
        return cat


def get_evidence_lookup(
    project_root: Path, config: RecommendationConfig
) -> SimulationEvidenceLookup:
    # Include both results paths so core/parametric month isolation is preserved.
    key = (
        str(project_root.resolve()),
        str(config.core_candidate_results_path),
        str(config.parametric_candidate_results_path),
        str(config.evidence_origin_label),
    )
    with _lock:
        hit = _evidence.get(key)
        if hit is not None:
            return hit
        lookup = SimulationEvidenceLookup(project_root, config)
        _evidence[key] = lookup
        return lookup


def clear_recommendation_resource_cache() -> None:
    with _lock:
        _catalogs.clear()
        _evidence.clear()
