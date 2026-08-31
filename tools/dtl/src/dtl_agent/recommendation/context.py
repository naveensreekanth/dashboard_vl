"""Input context loader and routing flags for Phase 8."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dtl_agent.canonical.dataset import CanonicalDataset
from dtl_agent.config.allowlists import FORBIDDEN_PATH_FRAGMENTS
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.validation.pipeline import validate_bundle
from dtl_agent.canonical import build_canonical_dataset
from dtl_agent.version import __version__


@dataclass
class RecommendationContext:
    project_root: Path
    lot_id: str
    die_id: str
    core_available: bool
    parametric_available: bool
    cross_domain_available: bool
    is_parametric_only: bool
    dataset_version_core: str
    dataset_version_parametric: str
    ml_dataset_version: str | None
    feature_registry_hash: str | None
    package_version: str
    canonical: CanonicalDataset | None
    errors: list[str]
    forbidden_detected: bool


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def detect_forbidden_request(extra_paths: list[str] | None = None) -> bool:
    """Return True if any provided path fragment matches forbidden evaluation data."""
    for p in extra_paths or []:
        low = str(p).replace("\\", "/").lower()
        for frag in FORBIDDEN_PATH_FRAGMENTS:
            if frag.lower() in low:
                return True
    return False


def load_recommendation_context(
    *,
    lot_id: str,
    die_id: str,
    project_root: Path | None = None,
    extra_paths: list[str] | None = None,
    canonical: CanonicalDataset | None = None,
) -> RecommendationContext:
    root = project_root or default_project_root()
    errors: list[str] = []
    forbidden = detect_forbidden_request(extra_paths)

    ml_ver = _read_json(root / "artifacts" / "ml_dataset" / "ML_DATASET_VERSION.json")
    core_ver = _read_json(root / "data" / "core" / "DATASET_VERSION.json")
    par_ver = _read_json(root / "data" / "parametric" / "PARAMETRIC_DATASET_VERSION.json")

    canon = canonical
    if canon is None and not forbidden:
        try:
            core = load_core(root, materialize_measurements=False)
            param = load_parametric(root, materialize_measurements=False)
            bundle = validate_bundle(core, param)
            if not bundle.ok:
                errors.append("canonical_bundle_validation_failed")
            else:
                canon = build_canonical_dataset(bundle)
        except Exception as exc:  # noqa: BLE001 — fail soft into REVIEW path
            errors.append(f"context_load_error:{type(exc).__name__}")

    core_available = False
    parametric_available = False
    cross_domain_available = False
    is_parametric_only = False
    if canon is not None:
        try:
            core_available = bool(canon.has_core_data(lot_id, die_id))
        except Exception:  # noqa: BLE001
            core_available = False
            errors.append("core_availability_lookup_failed")
        try:
            parametric_available = bool(canon.has_parametric_data(lot_id, die_id))
        except Exception:  # noqa: BLE001
            parametric_available = False
            errors.append("parametric_availability_lookup_failed")
        cross_domain_available = bool(core_available and parametric_available)
        try:
            is_parametric_only = lot_id in canon.linkage.parametric_only_lots
        except Exception:  # noqa: BLE001
            is_parametric_only = parametric_available and not core_available

    if not core_available and not parametric_available:
        errors.append("lot_die_not_found_or_no_domain_data")

    return RecommendationContext(
        project_root=root,
        lot_id=str(lot_id),
        die_id=str(die_id),
        core_available=core_available,
        parametric_available=parametric_available,
        cross_domain_available=cross_domain_available,
        is_parametric_only=is_parametric_only,
        dataset_version_core=str(core_ver.get("dataset_version", "DTL_DATASET_V1")),
        dataset_version_parametric=str(
            par_ver.get("dataset_version", "DTL_PARAMETRIC_DATASET_V1")
        ),
        ml_dataset_version=str(ml_ver.get("version")) if ml_ver else None,
        feature_registry_hash=ml_ver.get("phase3_feature_registry_hash") if ml_ver else None,
        package_version=__version__,
        canonical=canon,
        errors=errors,
        forbidden_detected=forbidden,
    )
