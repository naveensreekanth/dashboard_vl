"""Canonical dual-grain entity types (Phase 2).

Natural keys are preserved; no invented measurement columns.
Domain-specific fields remain domain-scoped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CanonicalLot:
    """Lot identity with optional per-domain metadata payloads."""

    lot_id: str
    in_core: bool
    in_parametric: bool
    cross_domain_available: bool
    # Shared-ish fields (prefer parametric/core when present)
    scenario_id: str | None = None
    scenario_family: str | None = None
    production_sequence: str | None = None
    tester_id: str | None = None
    core_metadata: dict[str, str] = field(default_factory=dict)
    parametric_metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalDie:
    """Die identity at (lot_id, die_id) with domain-specific metadata."""

    lot_id: str
    die_id: str
    in_core: bool
    in_parametric: bool
    cross_domain_available: bool
    die_label: str | None = None
    wafer_id: str | None = None
    wafer_x: str | None = None
    wafer_y: str | None = None
    tester_id: str | None = None
    site_id: str | None = None
    scenario_id: str | None = None
    scenario_family: str | None = None
    v1_link: str | None = None
    core_metadata: dict[str, str] = field(default_factory=dict)
    parametric_metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalCondition:
    """Authoritative parametric condition (from conditions_dim)."""

    condition_id: str
    temperature_c: str
    vdd_applied: str
    test_mode: str
    description: str = ""


@dataclass(frozen=True)
class CanonicalTestDefinition:
    """Domain-scoped test/parameter definition (catalogs are not shared)."""

    domain: str  # "core" | "parametric"
    test_id: str
    parameter: str
    unit: str
    direction: str
    dtl_eligible: bool
    source_metadata: dict[str, str] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalCurrentLimit:
    """Domain-scoped current limit with source/status preserved."""

    domain: str
    test_id: str
    parameter: str
    direction: str
    current_limit: float
    unit: str
    source_status: str
    raw: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CoreMeasurementRecord:
    """Core grain: lot × die × pattern × test."""

    lot_id: str
    die_id: str
    pattern_id: str
    test_id: str
    parameter: str
    value: float
    unit: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def natural_key(self) -> tuple[str, str, str, str]:
        return (self.lot_id, self.die_id, self.pattern_id, self.test_id)


@dataclass(frozen=True)
class ParametricMeasurementRecord:
    """Parametric grain: lot × die × condition × test."""

    lot_id: str
    die_id: str
    condition_id: str
    test_id: str
    parameter: str
    value: float
    unit: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def natural_key(self) -> tuple[str, str, str, str]:
        return (self.lot_id, self.die_id, self.condition_id, self.test_id)


@dataclass(frozen=True)
class LinkedDieView:
    """Lot/die-level cross-domain view — not a measurement-row join."""

    lot_id: str
    die_id: str
    cross_domain_available: bool
    core_die: CanonicalDie | None
    parametric_die: CanonicalDie | None
    note: str = (
        "Relationship is at lot_id+die_id only; Core and Parametric "
        "measurement grains remain separate."
    )


@dataclass(frozen=True)
class GrainSpec:
    domain: str
    description: str
    natural_key_fields: tuple[str, ...]


CORE_GRAIN = GrainSpec(
    domain="core",
    description="lot × die × pattern × test",
    natural_key_fields=("lot_id", "die_id", "pattern_id", "test_id"),
)

PARAMETRIC_GRAIN = GrainSpec(
    domain="parametric",
    description="lot × die × condition × test",
    natural_key_fields=("lot_id", "die_id", "condition_id", "test_id"),
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def parse_bool_flag(value: str | None) -> bool:
    return _truthy(value)


def row_get(row: dict[str, Any], key: str, default: str = "") -> str:
    val = row.get(key, default)
    if val is None:
        return default
    return str(val)
