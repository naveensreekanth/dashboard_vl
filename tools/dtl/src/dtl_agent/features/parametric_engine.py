"""Parametric condition-aware feature engineering (non-sequential / not Core GRU)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from dtl_agent.canonical.dataset import CanonicalDataset
from dtl_agent.features.io_utils import stable_float
from dtl_agent.features.margins import LimitSpec, MarginAggregate
from dtl_agent.features.registry import FeatureSpec, dist_feature_specs, margin_feature_specs
from dtl_agent.features.stats import compute_dist_stats, stats_to_dict

PARAM_GUARD_BAND_PCT = 5.0  # align with core config default; documented as proximity only


@dataclass
class ParametricFeatureResult:
    condition_rows: list[dict[str, Any]] = field(default_factory=list)  # die × condition
    die_rows: list[dict[str, Any]] = field(default_factory=list)
    lot_rows: list[dict[str, Any]] = field(default_factory=list)
    lot_condition_rows: list[dict[str, Any]] = field(default_factory=list)
    registry_specs: list[FeatureSpec] = field(default_factory=list)


def build_parametric_features(canonical: CanonicalDataset) -> ParametricFeatureResult:
    limits = {
        lim.parameter: LimitSpec(
            direction=lim.direction,
            value=lim.current_limit,
            unit=lim.unit,
            source_status=lim.source_status,
            test_id=lim.test_id,
            parameter=lim.parameter,
        )
        for lim in canonical.parametric_limits.values()
    }
    conditions = {c.condition_id: c for c in canonical.get_conditions()}

    # (lot, die, condition, param) -> value
    values: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for rec in canonical.get_parametric_measurements():
        values[(rec.lot_id, rec.die_id, rec.condition_id)][rec.parameter] = rec.value

    result = ParametricFeatureResult()

    # die × condition rows
    die_param_all: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    die_param_by_cond: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    lot_param_all: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    lot_param_by_cond: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    lot_dies: dict[str, set[str]] = defaultdict(set)

    for (lot_id, die_id, condition_id), params in sorted(values.items()):
        cond = conditions.get(condition_id)
        row: dict[str, Any] = {
            "lot_id": lot_id,
            "die_id": die_id,
            "condition_id": condition_id,
            "temperature_c": cond.temperature_c if cond else None,
            "vdd_applied": cond.vdd_applied if cond else None,
            "test_mode": cond.test_mode if cond else None,
            "cross_domain_available": int(
                canonical.cross_domain_available(lot_id, die_id)
            ),
        }
        lot_dies[lot_id].add(die_id)
        for param, value in sorted(params.items()):
            row[param] = stable_float(value)
            die_param_all[(lot_id, die_id)][param].append(value)
            die_param_by_cond[(lot_id, die_id)][param][condition_id] = value
            lot_param_all[lot_id][param].append(value)
            lot_param_by_cond[lot_id][condition_id][param].append(value)
            if param in limits:
                lim = limits[param]
                agg = MarginAggregate()
                agg.add(value, lim, borderline_pct=PARAM_GUARD_BAND_PCT)
                row.update(
                    {
                        k: stable_float(v) if isinstance(v, float) else v
                        for k, v in agg.to_dict(f"param_{param.lower()}").items()
                    }
                )
                row[f"param_{param.lower()}_current_limit"] = lim.value
                row[f"param_{param.lower()}_limit_direction"] = lim.direction
                row[f"param_{param.lower()}_limit_source"] = lim.source_status
        result.condition_rows.append(row)

    # Die-level: stats across conditions + deltas
    for (lot_id, die_id), param_map in sorted(die_param_all.items()):
        die_row: dict[str, Any] = {
            "lot_id": lot_id,
            "die_id": die_id,
            "condition_count": len(
                {cid for (l, d, cid) in values if l == lot_id and d == die_id}
            ),
            "cross_domain_available": int(canonical.cross_domain_available(lot_id, die_id)),
            "parametric_only": int(not canonical.has_core_data(lot_id, die_id)),
        }
        for param, vals in sorted(param_map.items()):
            prefix = f"param_{param.lower()}"
            stats = compute_dist_stats(vals)
            die_row.update(
                {
                    k: stable_float(v) if isinstance(v, float) else v
                    for k, v in stats_to_dict(prefix, stats).items()
                }
            )
            by_c = die_param_by_cond[(lot_id, die_id)][param]
            # Condition deltas (scientifically descriptive, not physical laws)
            if "COND_HOT_NOM" in by_c and "COND_RT_NOM" in by_c:
                die_row[f"{prefix}_delta_hot_minus_rt"] = stable_float(
                    by_c["COND_HOT_NOM"] - by_c["COND_RT_NOM"]
                )
            if "COND_RT_LOWV" in by_c and "COND_RT_NOM" in by_c:
                die_row[f"{prefix}_delta_lowv_minus_nom"] = stable_float(
                    by_c["COND_RT_LOWV"] - by_c["COND_RT_NOM"]
                )
            if "COND_HOT_HIGHV" in by_c and "COND_RT_NOM" in by_c:
                die_row[f"{prefix}_delta_hot_highv_minus_rt_nom"] = stable_float(
                    by_c["COND_HOT_HIGHV"] - by_c["COND_RT_NOM"]
                )
            if param in limits:
                agg = MarginAggregate()
                lim = limits[param]
                for v in vals:
                    agg.add(v, lim, borderline_pct=PARAM_GUARD_BAND_PCT)
                die_row.update(
                    {
                        k: stable_float(v) if isinstance(v, float) else v
                        for k, v in agg.to_dict(prefix).items()
                    }
                )
            for cid, v in sorted(by_c.items()):
                die_row[f"{prefix}__{cid}"] = stable_float(v)
        result.die_rows.append(die_row)

    # Lot-level
    for lot_id in sorted(lot_dies.keys()):
        lot_row: dict[str, Any] = {
            "lot_id": lot_id,
            "die_count": len(lot_dies[lot_id]),
            "condition_coverage": len(lot_param_by_cond[lot_id]),
            "cross_domain_available": int(canonical.cross_domain_available(lot_id)),
            "parametric_only": int(not canonical.has_core_data(lot_id)),
        }
        for param, vals in sorted(lot_param_all[lot_id].items()):
            prefix = f"param_{param.lower()}"
            stats = compute_dist_stats(vals)
            lot_row.update(
                {
                    k: stable_float(v) if isinstance(v, float) else v
                    for k, v in stats_to_dict(prefix, stats).items()
                }
            )
            if param in limits:
                agg = MarginAggregate()
                lim = limits[param]
                for v in vals:
                    agg.add(v, lim, borderline_pct=PARAM_GUARD_BAND_PCT)
                lot_row.update(
                    {
                        k: stable_float(v) if isinstance(v, float) else v
                        for k, v in agg.to_dict(prefix).items()
                    }
                )
        result.lot_rows.append(lot_row)

        for cid, pmap in sorted(lot_param_by_cond[lot_id].items()):
            lc: dict[str, Any] = {
                "lot_id": lot_id,
                "condition_id": cid,
                "temperature_c": conditions[cid].temperature_c if cid in conditions else None,
                "vdd_applied": conditions[cid].vdd_applied if cid in conditions else None,
            }
            for param, vals in sorted(pmap.items()):
                prefix = f"param_{param.lower()}"
                stats = compute_dist_stats(vals)
                lc.update(
                    {
                        k: stable_float(v) if isinstance(v, float) else v
                        for k, v in stats_to_dict(prefix, stats).items()
                    }
                )
            result.lot_condition_rows.append(lc)

    # Registry
    specs: list[FeatureSpec] = []
    for param, lim in sorted(limits.items(), key=lambda x: x[0]):
        prefix = f"param_{param.lower()}"
        specs.extend(
            dist_feature_specs(
                prefix=prefix,
                domain="parametric",
                grain="die|lot|die×condition|lot×condition",
                parameter=param,
                unit=lim.unit,
            )
        )
        specs.extend(
            margin_feature_specs(
                prefix=prefix,
                domain="parametric",
                grain="die|lot|die×condition",
                parameter=param,
                unit=lim.unit,
                direction=lim.direction,
            )
        )
        for delta in (
            "delta_hot_minus_rt",
            "delta_lowv_minus_nom",
            "delta_hot_highv_minus_rt_nom",
        ):
            specs.append(
                FeatureSpec(
                    feature_name=f"{prefix}_{delta}",
                    domain="parametric",
                    grain="die",
                    source_parameters=[param],
                    formula=delta,
                    unit=lim.unit,
                    direction=None,
                    allowed_for_ml=True,
                    candidate_dependent=False,
                    evaluation_only=False,
                    normalization_required=True,
                    description="Condition-to-condition delta (descriptive; not a physical law claim)",
                )
            )
    result.registry_specs = specs
    return result
