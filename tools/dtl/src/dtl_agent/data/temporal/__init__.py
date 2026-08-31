"""Month-aware temporal data loading (Phase 12.3)."""

from __future__ import annotations

from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.loader import (
    TemporalMonthData,
    load_temporal_die,
    load_temporal_month,
)
from dtl_agent.data.temporal.paths import (
    ALLOWED_PRODUCTION_MONTHS,
    ProductionMonth,
    actual_die_root,
    month_ml_dataset_root,
    month_root,
    month_simulation_root,
    parametric_root,
    shared_ml_dataset_root,
    temporal_artifact_root,
    temporal_data_root,
    validate_production_month,
)

__all__ = [
    "ALLOWED_PRODUCTION_MONTHS",
    "ProductionMonth",
    "TemporalMonthData",
    "actual_die_root",
    "load_temporal_die",
    "load_temporal_month",
    "make_sequence_id",
    "month_ml_dataset_root",
    "month_root",
    "month_simulation_root",
    "parametric_root",
    "shared_ml_dataset_root",
    "temporal_artifact_root",
    "temporal_data_root",
    "validate_production_month",
]
