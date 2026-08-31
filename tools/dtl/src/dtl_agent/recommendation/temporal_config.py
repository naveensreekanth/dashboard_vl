"""Month-scoped recommendation config and context for Phase 12.8 hybrid temporal mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dtl_agent.data.temporal.loader import TemporalMonthData, load_temporal_die
from dtl_agent.data.temporal.paths import (
    month_simulation_root,
    validate_production_month,
)
from dtl_agent.recommendation.config import RecommendationConfig
from dtl_agent.recommendation.context import RecommendationContext
from dtl_agent.version import __version__


def temporal_recommendation_config(
    production_month: str,
    *,
    base: RecommendationConfig | None = None,
) -> RecommendationConfig:
    """Point catalog + simulation evidence at ``artifacts/temporal/{month}/simulation/`` only."""
    month = validate_production_month(production_month)
    # Paths are relative to project_root via RecommendationConfig.resolve_path
    sim = f"artifacts/temporal/{month}/simulation"
    cfg = base or RecommendationConfig()
    data = cfg.to_dict()
    data.update(
        {
            "core_candidate_grid_path": f"{sim}/core/candidate_grid.csv",
            "core_candidate_results_path": f"{sim}/core/candidate_results.csv",
            "parametric_candidate_grid_path": f"{sim}/parametric/candidate_grid.csv",
            "parametric_candidate_results_path": f"{sim}/parametric/candidate_results.csv",
            "core_checkpoint_path": "artifacts/temporal/shared/checkpoints/core_gru_temporal_v1.pt",
            # Parametric legacy MLP path unused in temporal hybrid (Unified GRU used instead)
            "parametric_checkpoint_path": (
                "artifacts/temporal/shared/checkpoints/unified_parameter_gru_v1.pt"
            ),
            "evidence_origin_label": f"SIMULATOR_DERIVED_TEMPORAL_{month}",
        }
    )
    return RecommendationConfig.from_dict(data)


@dataclass
class TemporalRecommendationContext(RecommendationContext):
    production_month: str = ""
    month_data: TemporalMonthData | None = None


def load_temporal_recommendation_context(
    *,
    production_month: str,
    lot_id: str,
    die_id: str,
    project_root: Path,
    extra_paths: list[str] | None = None,
) -> TemporalRecommendationContext:
    """Availability from month package only — never legacy ``data/core``.

    Loads a die-scoped slice (not the full ~500 MB month frame).
    """
    from dtl_agent.recommendation.context import detect_forbidden_request

    month = validate_production_month(production_month)
    forbidden = detect_forbidden_request(extra_paths)
    errors: list[str] = []
    month_data: TemporalMonthData | None = None
    if not forbidden:
        try:
            month_data = load_temporal_die(
                month, lot_id, die_id, project_root=project_root
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"temporal_month_load_error:{type(exc).__name__}:{exc}")

    core_available = False
    parametric_available = False
    if month_data is not None:
        ad = month_data.actual_die
        pr = month_data.parametric
        core_available = bool(
            ((ad["lot_id"].astype(str) == str(lot_id)) & (ad["die_id"].astype(str) == str(die_id))).any()
        )
        parametric_available = bool(
            (not pr.empty)
            and (
                (pr["lot_id"].astype(str) == str(lot_id))
                & (pr["die_id"].astype(str) == str(die_id))
            ).any()
        )

    return TemporalRecommendationContext(
        project_root=project_root,
        lot_id=lot_id,
        die_id=die_id,
        core_available=core_available,
        parametric_available=parametric_available,
        cross_domain_available=bool(core_available and parametric_available),
        is_parametric_only=bool(parametric_available and not core_available),
        dataset_version_core=f"temporal_{month}",
        dataset_version_parametric=f"temporal_{month}",
        ml_dataset_version="phase12_4_temporal_core_v1+phase12_5d_unified",
        feature_registry_hash=None,
        package_version=__version__,
        canonical=None,
        errors=errors,
        forbidden_detected=forbidden,
        production_month=month,
        month_data=month_data,
    )


def assert_month_simulation_isolated(production_month: str, project_root: Path) -> None:
    """Hard stop if temporal sim paths are missing (do not fall back to legacy)."""
    month = validate_production_month(production_month)
    root = month_simulation_root(month, project_root)
    for domain in ("core", "parametric"):
        results = root / domain / "candidate_results.csv"
        grid = root / domain / "candidate_grid.csv"
        if not results.is_file() or not grid.is_file():
            raise FileNotFoundError(
                f"Missing month-scoped simulation under {root / domain}; "
                f"refusing to use artifacts/simulation/ for production_month={month!r}"
            )
