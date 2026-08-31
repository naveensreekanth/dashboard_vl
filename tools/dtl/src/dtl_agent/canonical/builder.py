"""Builders that map Phase 1 validated datasets into canonical entities."""

from __future__ import annotations

from dtl_agent.canonical.entities import (
    CanonicalCondition,
    CanonicalCurrentLimit,
    CanonicalDie,
    CanonicalLot,
    CanonicalTestDefinition,
    parse_bool_flag,
    row_get,
)
from dtl_agent.data.models.core import CoreDataset
from dtl_agent.data.models.linkage import SharedLotDieIndex
from dtl_agent.data.models.parametric import ParametricDataset


_CORE_DIE_SPECIFIC = (
    "defect_type",
    "diagnosis_classification",
    "soft_bin",
    "hard_bin",
    "bin_reason",
    "status",
    "patterns_executed",
    "patterns_passed",
    "patterns_failed",
    "die_index",
    "die_row",
    "die_col",
    "x1",
    "y1",
    "x2",
    "y2",
    "test_program",
    "device_name",
    "test_mode",
    "shift_cycles",
    "source_json",
    "generation_seed",
    "generator_version",
    "production_sequence",
)


def build_lots(
    core: CoreDataset, parametric: ParametricDataset, linkage: SharedLotDieIndex
) -> dict[str, CanonicalLot]:
    core_by_id = {row_get(r, "lot_id"): r for r in core.lots}
    par_by_id = {row_get(r, "lot_id"): r for r in parametric.lots}
    lots: dict[str, CanonicalLot] = {}
    for lot_id in sorted(core_by_id.keys() | par_by_id.keys()):
        c = core_by_id.get(lot_id)
        p = par_by_id.get(lot_id)
        in_core = c is not None
        in_par = p is not None
        cross = lot_id in linkage.linked_lots
        preferred = p or c or {}
        lots[lot_id] = CanonicalLot(
            lot_id=lot_id,
            in_core=in_core,
            in_parametric=in_par,
            cross_domain_available=cross,
            scenario_id=row_get(preferred, "scenario_id") or None,
            scenario_family=row_get(preferred, "scenario_family") or None,
            production_sequence=row_get(preferred, "production_sequence") or None,
            tester_id=row_get(preferred, "tester_id") or None,
            core_metadata=dict(c) if c else {},
            parametric_metadata=dict(p) if p else {},
        )
    return lots


def build_dies(
    core: CoreDataset, parametric: ParametricDataset, linkage: SharedLotDieIndex
) -> dict[tuple[str, str], CanonicalDie]:
    core_by_pair = {(row_get(r, "lot_id"), row_get(r, "die_id")): r for r in core.parts}
    par_by_pair = {
        (row_get(r, "lot_id"), row_get(r, "die_id")): r for r in parametric.parts
    }
    dies: dict[tuple[str, str], CanonicalDie] = {}
    for pair in sorted(core_by_pair.keys() | par_by_pair.keys()):
        lot_id, die_id = pair
        c = core_by_pair.get(pair)
        p = par_by_pair.get(pair)
        preferred = p or c or {}
        core_meta = {k: row_get(c, k) for k in _CORE_DIE_SPECIFIC if c} if c else {}
        dies[pair] = CanonicalDie(
            lot_id=lot_id,
            die_id=die_id,
            in_core=c is not None,
            in_parametric=p is not None,
            cross_domain_available=pair in linkage.linked_lot_die_pairs,
            die_label=row_get(preferred, "die_label") or None,
            wafer_id=row_get(preferred, "wafer_id") or None,
            wafer_x=row_get(preferred, "wafer_x") or None,
            wafer_y=row_get(preferred, "wafer_y") or None,
            tester_id=row_get(preferred, "tester_id") or None,
            site_id=row_get(preferred, "site_id") or None,
            scenario_id=row_get(preferred, "scenario_id") or None,
            scenario_family=row_get(preferred, "scenario_family") or None,
            v1_link=row_get(p, "v1_link") if p else None,
            core_metadata=core_meta if c else {},
            parametric_metadata=dict(p) if p else {},
        )
    return dies


def build_conditions(parametric: ParametricDataset) -> dict[str, CanonicalCondition]:
    out: dict[str, CanonicalCondition] = {}
    for row in parametric.conditions:
        cid = row_get(row, "condition_id")
        out[cid] = CanonicalCondition(
            condition_id=cid,
            temperature_c=row_get(row, "temperature_c"),
            vdd_applied=row_get(row, "vdd_applied"),
            test_mode=row_get(row, "test_mode"),
            description=row_get(row, "description"),
        )
    return out


def build_core_tests(core: CoreDataset) -> dict[str, CanonicalTestDefinition]:
    out: dict[str, CanonicalTestDefinition] = {}
    for row in core.test_catalog:
        tid = row_get(row, "test_id")
        out[tid] = CanonicalTestDefinition(
            domain="core",
            test_id=tid,
            parameter=row_get(row, "parameter"),
            unit=row_get(row, "unit"),
            direction=row_get(row, "direction"),
            dtl_eligible=parse_bool_flag(row_get(row, "dtl_eligible")),
            source_metadata={
                "source_status": row_get(row, "source_status"),
                "optimization_priority": row_get(row, "optimization_priority"),
                "category": row_get(row, "category"),
            },
            raw=dict(row),
        )
    return out


def build_parametric_tests(
    parametric: ParametricDataset,
) -> dict[str, CanonicalTestDefinition]:
    out: dict[str, CanonicalTestDefinition] = {}
    for row in parametric.test_catalog:
        tid = row_get(row, "test_id")
        out[tid] = CanonicalTestDefinition(
            domain="parametric",
            test_id=tid,
            parameter=row_get(row, "parameter"),
            unit=row_get(row, "unit"),
            direction=row_get(row, "limit_type"),
            dtl_eligible=parse_bool_flag(row_get(row, "dtl_eligible")),
            source_metadata={
                "synthetic_source": row_get(row, "synthetic_source"),
                "priority": row_get(row, "priority"),
                "role": row_get(row, "role"),
                "condition_dependent": row_get(row, "condition_dependent"),
            },
            raw=dict(row),
        )
    return out


def build_core_limits(core: CoreDataset) -> dict[str, CanonicalCurrentLimit]:
    out: dict[str, CanonicalCurrentLimit] = {}
    for row in core.current_limits:
        tid = row_get(row, "test_id")
        out[tid] = CanonicalCurrentLimit(
            domain="core",
            test_id=tid,
            parameter=row_get(row, "parameter"),
            direction=row_get(row, "limit_direction"),
            current_limit=float(row_get(row, "upper_limit")),
            unit=row_get(row, "unit"),
            source_status=row_get(row, "source_status"),
            raw=dict(row),
        )
    return out


def build_parametric_limits(
    parametric: ParametricDataset,
) -> dict[str, CanonicalCurrentLimit]:
    out: dict[str, CanonicalCurrentLimit] = {}
    for row in parametric.current_limits:
        tid = row_get(row, "test_id")
        out[tid] = CanonicalCurrentLimit(
            domain="parametric",
            test_id=tid,
            parameter=row_get(row, "parameter"),
            direction=row_get(row, "limit_type"),
            current_limit=float(row_get(row, "limit_value")),
            unit=row_get(row, "unit"),
            source_status=row_get(row, "source"),
            raw=dict(row),
        )
    return out
