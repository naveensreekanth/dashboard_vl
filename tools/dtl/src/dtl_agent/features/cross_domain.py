"""Cross-domain features for linked lot/die identities only (no measurement-row joins)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from dtl_agent.canonical.dataset import CanonicalDataset
from dtl_agent.features.io_utils import stable_float
from dtl_agent.features.registry import FeatureSpec


def _corr(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx < 1e-12 or deny < 1e-12:
        return None
    return num / (denx * deny)


@dataclass
class CrossDomainFeatureResult:
    linked_die_rows: list[dict[str, Any]] = field(default_factory=list)
    linked_lot_rows: list[dict[str, Any]] = field(default_factory=list)
    registry_specs: list[FeatureSpec] = field(default_factory=list)


def build_cross_domain_features(
    canonical: CanonicalDataset,
    *,
    core_die_rows: list[dict[str, Any]],
    parametric_die_rows: list[dict[str, Any]],
    core_lot_rows: list[dict[str, Any]],
    parametric_lot_rows: list[dict[str, Any]],
) -> CrossDomainFeatureResult:
    """Join only on lot_id+die_id / lot_id for linked entities — never measurement rows."""
    result = CrossDomainFeatureResult()
    core_die = {(r["lot_id"], r["die_id"]): r for r in core_die_rows}
    par_die = {(r["lot_id"], r["die_id"]): r for r in parametric_die_rows}
    core_lot = {r["lot_id"]: r for r in core_lot_rows}
    par_lot = {r["lot_id"]: r for r in parametric_lot_rows}

    pairs = [
        ("core_ir_drop_mean", "param_iddq_mean", "cross_ir_mean__iddq_mean"),
        ("core_thermal_mean", "param_iddq_mean", "cross_thermal_mean__iddq_mean"),
        ("core_ir_drop_mean", "param_contact_resistance_mean", "cross_ir_mean__contact_r_mean"),
        ("core_thermal_mean", "param_supply_current_mean", "cross_thermal_mean__supply_current_mean"),
        ("core_ir_drop_margin_min", "param_iddq_margin_min", "cross_ir_margin_min__iddq_margin_min"),
        ("core_thermal_margin_min", "param_vmax_margin_min", "cross_thermal_margin_min__vmax_margin_min"),
    ]

    # Linked dies
    for pair in sorted(canonical.linkage.linked_lot_die_pairs):
        c = core_die.get(pair)
        p = par_die.get(pair)
        if c is None or p is None:
            continue
        row: dict[str, Any] = {
            "lot_id": pair[0],
            "die_id": pair[1],
            "cross_domain_available": 1,
            "join_grain": "lot_id+die_id",
            "join_type": "entity_summary_not_measurement_row",
        }
        for ck, pk, out_name in pairs:
            cv = c.get(ck)
            pv = p.get(pk)
            row[ck] = cv
            row[pk] = pv
            if isinstance(cv, (int, float)) and isinstance(pv, (int, float)):
                row[out_name + "_product"] = stable_float(float(cv) * float(pv))
                row[out_name + "_diff"] = stable_float(float(cv) - float(pv))
            else:
                row[out_name + "_product"] = None
                row[out_name + "_diff"] = None
        result.linked_die_rows.append(row)

    # Linked lots + correlations across linked dies within lot
    for lot_id in sorted(canonical.linkage.linked_lots):
        c = core_lot.get(lot_id)
        p = par_lot.get(lot_id)
        if c is None or p is None:
            continue
        row = {
            "lot_id": lot_id,
            "cross_domain_available": 1,
            "join_grain": "lot_id",
            "join_type": "entity_summary_not_measurement_row",
            "linked_die_count": sum(1 for a, _ in canonical.linkage.linked_lot_die_pairs if a == lot_id),
        }
        # lot-level summary copies
        for key in (
            "core_ir_drop_mean",
            "core_thermal_mean",
            "core_ir_drop_violation_rate",
            "core_thermal_violation_rate",
        ):
            row[key] = c.get(key)
        for key in (
            "param_iddq_mean",
            "param_vmin_mean",
            "param_iddq_violation_rate",
            "param_contact_resistance_mean",
        ):
            row[key] = p.get(key)

        # die-level correlations within lot (descriptive)
        xs_ir: list[float] = []
        ys_iddq: list[float] = []
        xs_th: list[float] = []
        ys_res: list[float] = []
        for a, b in canonical.linkage.linked_lot_die_pairs:
            if a != lot_id:
                continue
            cd = core_die.get((a, b))
            pd = par_die.get((a, b))
            if not cd or not pd:
                continue
            if isinstance(cd.get("core_ir_drop_mean"), (int, float)) and isinstance(
                pd.get("param_iddq_mean"), (int, float)
            ):
                xs_ir.append(float(cd["core_ir_drop_mean"]))
                ys_iddq.append(float(pd["param_iddq_mean"]))
            if isinstance(cd.get("core_thermal_mean"), (int, float)) and isinstance(
                pd.get("param_contact_resistance_mean"), (int, float)
            ):
                xs_th.append(float(cd["core_thermal_mean"]))
                ys_res.append(float(pd["param_contact_resistance_mean"]))
        row["cross_ir_iddq_corr"] = stable_float(_corr(xs_ir, ys_iddq))
        row["cross_thermal_contact_r_corr"] = stable_float(_corr(xs_th, ys_res))
        result.linked_lot_rows.append(row)

    result.registry_specs = [
        FeatureSpec(
            feature_name="cross_ir_iddq_corr",
            domain="cross_domain",
            grain="linked lot",
            source_parameters=["ir_drop", "IDDQ"],
            formula="pearson_corr(core_ir_drop_mean, param_iddq_mean) over linked dies",
            unit=None,
            direction=None,
            allowed_for_ml=True,
            candidate_dependent=False,
            evaluation_only=False,
            normalization_required=False,
            description="Descriptive association only; not causal; no latent severity",
        ),
        FeatureSpec(
            feature_name="cross_thermal_contact_r_corr",
            domain="cross_domain",
            grain="linked lot",
            source_parameters=["thermal", "CONTACT_RESISTANCE"],
            formula="pearson_corr(core_thermal_mean, param_contact_resistance_mean)",
            unit=None,
            direction=None,
            allowed_for_ml=True,
            candidate_dependent=False,
            evaluation_only=False,
            normalization_required=False,
            description="Descriptive association only; not causal",
        ),
    ]
    return result
