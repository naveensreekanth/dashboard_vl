"""Dynamic Test Limit (DTL) Agent — public API."""

from dtl_agent.canonical import (
    CanonicalDataset,
    CanonicalLookupError,
    build_canonical_dataset,
    build_canonical_from_datasets,
)
from dtl_agent.data.loaders.core_loader import CoreDataLoader, load_core
from dtl_agent.data.loaders.parametric_loader import ParametricDataLoader, load_parametric
from dtl_agent.data.models.bundle import ValidatedDatasetBundle
from dtl_agent.data.models.linkage import SharedLotDieIndex
from dtl_agent.features import FeatureArtifacts, run_feature_engineering
from dtl_agent.ml_dataset import MLDatasetArtifacts, run_phase6_ml_dataset_assembly
from dtl_agent.simulation import (
    CoreSimulationArtifacts,
    ParametricSimulationArtifacts,
    run_core_simulation_optimization,
    run_parametric_simulation_optimization,
)
from dtl_agent.validation.pipeline import validate_bundle, validate_core, validate_parametric
from dtl_agent.validation.phase2 import validate_canonical_dataset
from dtl_agent.version import __version__

__all__ = [
    "CanonicalDataset",
    "CanonicalLookupError",
    "CoreDataLoader",
    "CoreSimulationArtifacts",
    "FeatureArtifacts",
    "MLDatasetArtifacts",
    "ParametricSimulationArtifacts",
    "ParametricDataLoader",
    "SharedLotDieIndex",
    "ValidatedDatasetBundle",
    "build_canonical_dataset",
    "build_canonical_from_datasets",
    "load_core",
    "load_parametric",
    "run_core_simulation_optimization",
    "run_parametric_simulation_optimization",
    "run_feature_engineering",
    "run_phase7_training",
    "run_phase6_ml_dataset_assembly",
    "Decision",
    "DTLRecommendation",
    "EvidenceLevel",
    "LotRecommendationResult",
    "RecommendationConfig",
    "recommend",
    "create_app",
    "validate_bundle",
    "validate_canonical_dataset",
    "validate_core",
    "validate_parametric",
    "__version__",
]


def __getattr__(name: str):
    """Lazy Phase 7/8 exports to avoid circular imports and heavy startup deps."""
    if name in {
        "Decision",
        "DTLRecommendation",
        "EvidenceLevel",
        "LotRecommendationResult",
        "RecommendationConfig",
        "recommend",
    }:
        from dtl_agent import recommendation as rec

        return getattr(rec, name)
    if name == "create_app":
        from dtl_agent.api import create_app

        return create_app
    if name == "run_phase7_training":
        from dtl_agent.ml.pipeline import run_phase7_training

        return run_phase7_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
