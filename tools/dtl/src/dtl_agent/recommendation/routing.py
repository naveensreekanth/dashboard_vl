"""Centralized parameter → ML model routing for hybrid temporal recommendations.

Single source of truth for Phase 12.8. Do not scatter model selection elsewhere.
"""

from __future__ import annotations

from enum import Enum

from dtl_agent.recommendation.schemas import (
    CORE_PARAMETERS,
    PARAMETRIC_PARAMETERS,
    UNSUPPORTED_CORE_PARAMETERS,
)


class HybridModelId(str, Enum):
    CORE_TEMPORAL = "core_gru_temporal_v1"
    UNIFIED = "unified_parameter_gru_v1"
    LEGACY_CORE = "core_gru"
    LEGACY_PARAMETRIC = "parametric_mlp"
    UNSUPPORTED = "unsupported"


# Canonical parameter names used by catalogs / recommend (not display IR_DROP_MV labels).
CORE_TEMPORAL_PARAMETERS = frozenset(CORE_PARAMETERS)  # ir_drop, thermal
UNIFIED_PARAMETERS = frozenset(PARAMETRIC_PARAMETERS)
NON_SCORABLE_PARAMETERS = frozenset(UNSUPPORTED_CORE_PARAMETERS)


def model_for_parameter(parameter: str, *, temporal: bool) -> HybridModelId:
    """Return which model scores ``parameter`` in legacy vs temporal mode."""
    p = str(parameter)
    if p in NON_SCORABLE_PARAMETERS:
        return HybridModelId.UNSUPPORTED
    if temporal:
        if p in CORE_TEMPORAL_PARAMETERS:
            return HybridModelId.CORE_TEMPORAL
        if p in UNIFIED_PARAMETERS:
            return HybridModelId.UNIFIED
        return HybridModelId.UNSUPPORTED
    if p in CORE_TEMPORAL_PARAMETERS:
        return HybridModelId.LEGACY_CORE
    if p in UNIFIED_PARAMETERS:
        return HybridModelId.LEGACY_PARAMETRIC
    return HybridModelId.UNSUPPORTED


def model_used_label(parameter: str, *, temporal: bool) -> str | None:
    mid = model_for_parameter(parameter, temporal=temporal)
    if mid == HybridModelId.UNSUPPORTED:
        return None
    return mid.value


ROUTING_TABLE_DOC = {
    "ir_drop": "core_gru_temporal_v1 (temporal) / core_gru (legacy)",
    "thermal": "core_gru_temporal_v1 (temporal) / core_gru (legacy)",
    "VMIN": "unified_parameter_gru_v1 (temporal) / parametric_mlp (legacy)",
    "VMAX": "unified_parameter_gru_v1 (temporal) / parametric_mlp (legacy)",
    "IDDQ": "unified_parameter_gru_v1 (temporal) / parametric_mlp (legacy)",
    "SUPPLY_CURRENT": "unified_parameter_gru_v1 (temporal) / parametric_mlp (legacy)",
    "CONTACT_RESISTANCE": "unified_parameter_gru_v1 (temporal) / parametric_mlp (legacy)",
    "INTERCONNECT_RESISTANCE": "unified_parameter_gru_v1 (temporal) / parametric_mlp (legacy)",
    "ON_RESISTANCE": "unified_parameter_gru_v1 (temporal) / parametric_mlp (legacy)",
    "setup_slack": "unsupported / non-scorable",
    "hold_slack": "unsupported / non-scorable",
    "test_time": "unsupported / non-scorable",
}
