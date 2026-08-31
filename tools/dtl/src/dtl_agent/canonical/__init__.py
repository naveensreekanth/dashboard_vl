"""Canonical dual-grain layer (Phase 2)."""

from dtl_agent.canonical.dataset import (
    CanonicalDataset,
    CanonicalLookupError,
    build_canonical_dataset,
    build_canonical_from_datasets,
)
from dtl_agent.canonical.entities import (
    CORE_GRAIN,
    PARAMETRIC_GRAIN,
    CanonicalCondition,
    CanonicalCurrentLimit,
    CanonicalDie,
    CanonicalLot,
    CanonicalTestDefinition,
    CoreMeasurementRecord,
    LinkedDieView,
    ParametricMeasurementRecord,
)

__all__ = [
    "CORE_GRAIN",
    "PARAMETRIC_GRAIN",
    "CanonicalCondition",
    "CanonicalCurrentLimit",
    "CanonicalDataset",
    "CanonicalDie",
    "CanonicalLot",
    "CanonicalLookupError",
    "CanonicalTestDefinition",
    "CoreMeasurementRecord",
    "LinkedDieView",
    "ParametricMeasurementRecord",
    "build_canonical_dataset",
    "build_canonical_from_datasets",
]
