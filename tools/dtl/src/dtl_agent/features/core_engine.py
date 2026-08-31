"""Core domain feature engineering + GRU-ready pattern sequences."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from dtl_agent.canonical.dataset import CanonicalDataset
from dtl_agent.canonical.entities import CanonicalCurrentLimit
from dtl_agent.features.io_utils import stable_float
from dtl_agent.features.margins import LimitSpec, MarginAggregate, proximity_class, signed_margin
from dtl_agent.features.registry import (
    FeatureSpec,
    dist_feature_specs,
    margin_feature_specs,
)
from dtl_agent.features.stats import compute_dist_stats, stats_to_dict

# Fixed GRU step feature order (raw measurements)
SEQUENCE_FEATURE_ORDER = (
    "ir_drop",
    "thermal",
    "setup_slack",
    "hold_slack",
    "test_time",
)
EXPECTED_SEQUENCE_LENGTH = 200


def _limit_spec(lim: CanonicalCurrentLimit) -> LimitSpec:
    return LimitSpec(
        direction=lim.direction,
        value=lim.current_limit,
        unit=lim.unit,
        source_status=lim.source_status,
        test_id=lim.test_id,
        parameter=lim.parameter,
    )


def _guard_band_pct(canonical: CanonicalDataset) -> float:
    cfg = canonical.core.limit_simulation_config
    gb = cfg.get("guard_band", {}) if isinstance(cfg, dict) else {}
    return float(gb.get("borderline_margin_percent", 5.0))


@dataclass
class CoreFeatureResult:
    pattern_rows: list[dict[str, Any]] = field(default_factory=list)
    die_rows: list[dict[str, Any]] = field(default_factory=list)
    lot_rows: list[dict[str, Any]] = field(default_factory=list)
    lot_parameter_rows: list[dict[str, Any]] = field(default_factory=list)
    sequence_contract: dict[str, Any] = field(default_factory=dict)
    sequence_manifest: list[dict[str, Any]] = field(default_factory=list)
    registry_specs: list[FeatureSpec] = field(default_factory=list)


def build_core_features(canonical: CanonicalDataset) -> CoreFeatureResult:
    """Single streaming pass over Core measurements → pattern/die/lot features + sequences."""
    borderline_pct = _guard_band_pct(canonical)
    limits = {tid: _limit_spec(lim) for tid, lim in canonical.core_limits.items()}
    param_to_limit = {lim.parameter: lim for lim in limits.values()}

    # Accumulators keyed by die
    # die -> pattern_id(int) -> param -> value
    die_patterns: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    die_param_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    die_margin_aggs: dict[tuple[str, str], dict[str, MarginAggregate]] = defaultdict(dict)

    for rec in canonical.get_core_measurements():
        key = (rec.lot_id, rec.die_id)
        try:
            pid = int(rec.pattern_id)
        except ValueError as exc:
            raise ValueError(f"non-integer pattern_id={rec.pattern_id!r}") from exc
        die_patterns[key][pid][rec.parameter] = rec.value
        die_param_values[key][rec.parameter].append(rec.value)
        if rec.parameter in param_to_limit:
            lim = param_to_limit[rec.parameter]
            if rec.parameter not in die_margin_aggs[key]:
                die_margin_aggs[key][rec.parameter] = MarginAggregate()
            die_margin_aggs[key][rec.parameter].add(
                rec.value, lim, borderline_pct=borderline_pct
            )

    result = CoreFeatureResult()
    seq_lengths: list[int] = []
    incomplete = 0
    missing_steps_total = 0
    duplicate_steps = 0
    ordering_invalid = 0
    feature_dim = len(SEQUENCE_FEATURE_ORDER)
    valid_sequences = 0

    # Pass/fail from parts for lot summaries
    die_status = {
        (d.lot_id, d.die_id): d.core_metadata.get("status")
        for d in canonical.dies.values()
        if d.in_core
    }

    lot_param_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    lot_margin_aggs: dict[str, dict[str, MarginAggregate]] = defaultdict(dict)
    lot_die_counts: dict[str, int] = defaultdict(int)
    lot_pass: dict[str, int] = defaultdict(int)
    lot_fail: dict[str, int] = defaultdict(int)

    for (lot_id, die_id), patterns in sorted(die_patterns.items()):
        lot_die_counts[lot_id] += 1
        st = die_status.get((lot_id, die_id))
        if st == "PASS":
            lot_pass[lot_id] += 1
        elif st == "FAIL":
            lot_fail[lot_id] += 1

        pids = sorted(patterns.keys())
        seq_len = len(pids)
        seq_lengths.append(seq_len)
        expected = list(range(1, EXPECTED_SEQUENCE_LENGTH + 1))
        missing = [p for p in expected if p not in patterns]
        extras = [p for p in pids if p < 1 or p > EXPECTED_SEQUENCE_LENGTH]
        # duplicate pattern ids cannot exist in dict keys; track if any pattern missing params
        if missing or extras or seq_len != EXPECTED_SEQUENCE_LENGTH:
            incomplete += 1
            missing_steps_total += len(missing)
        if pids != expected[:seq_len] or (seq_len == EXPECTED_SEQUENCE_LENGTH and pids != expected):
            # ordering by numeric pattern_id
            if pids != sorted(pids):
                ordering_invalid += 1
        if seq_len == EXPECTED_SEQUENCE_LENGTH and not missing and pids == expected:
            valid_sequences += 1

        # Pattern-step rows for existing patterns only (no invented padding)
        for pid in pids:
            step = patterns[pid]
            row: dict[str, Any] = {
                "lot_id": lot_id,
                "die_id": die_id,
                "pattern_id": pid,
                "sequence_index": pid - 1,
                "valid_step": 1,
            }
            for feat in SEQUENCE_FEATURE_ORDER:
                row[feat] = stable_float(step.get(feat))
                row[f"{feat}_missing"] = 0 if feat in step else 1
            for param, lim in param_to_limit.items():
                if param in step:
                    m = signed_margin(step[param], lim)
                    cls = proximity_class(
                        step[param], lim, borderline_margin_percent=borderline_pct
                    )
                    row[f"{param}_margin"] = stable_float(m)
                    row[f"{param}_proximity_class"] = cls
                    row[f"{param}_violation_flag"] = 1 if cls == "VIOLATION" else 0
                else:
                    row[f"{param}_margin"] = None
                    row[f"{param}_proximity_class"] = None
                    row[f"{param}_violation_flag"] = None
            result.pattern_rows.append(row)

        # Die-level features
        die_row: dict[str, Any] = {
            "lot_id": lot_id,
            "die_id": die_id,
            "sequence_length": seq_len,
            "sequence_complete": int(seq_len == EXPECTED_SEQUENCE_LENGTH and not missing),
            "missing_pattern_count": len(missing),
            "source_die_status": st,
        }
        for param, values in sorted(die_param_values[(lot_id, die_id)].items()):
            prefix = f"core_{param}"
            stats = compute_dist_stats(values)
            die_row.update({k: stable_float(v) if isinstance(v, float) else v for k, v in stats_to_dict(prefix, stats).items()})
            lot_param_values[lot_id][param].extend(values)
        for param, agg in die_margin_aggs[(lot_id, die_id)].items():
            prefix = f"core_{param}"
            die_row.update({k: stable_float(v) if isinstance(v, float) else v for k, v in agg.to_dict(prefix).items()})
        result.die_rows.append(die_row)
        result.sequence_manifest.append(
            {
                "lot_id": lot_id,
                "die_id": die_id,
                "sequence_length": seq_len,
                "expected_length": EXPECTED_SEQUENCE_LENGTH,
                "complete": int(seq_len == EXPECTED_SEQUENCE_LENGTH and not missing),
                "missing_pattern_ids": ",".join(str(x) for x in missing),
                "feature_dim": feature_dim,
                "feature_order": ",".join(SEQUENCE_FEATURE_ORDER),
            }
        )

    # Lot-level margins from lot measurement pools (single accumulation)
    for lot_id, param_map in lot_param_values.items():
        for param, values in param_map.items():
            if param not in param_to_limit:
                continue
            agg = MarginAggregate()
            lim = param_to_limit[param]
            for v in values:
                agg.add(v, lim, borderline_pct=borderline_pct)
            lot_margin_aggs[lot_id][param] = agg

    # Lot-level + lot×parameter
    for lot_id in sorted(lot_die_counts.keys()):
        lot_row: dict[str, Any] = {
            "lot_id": lot_id,
            "die_count": lot_die_counts[lot_id],
            "pass_count": lot_pass[lot_id],
            "fail_count": lot_fail[lot_id],
            "source_pass_rate": (
                lot_pass[lot_id] / lot_die_counts[lot_id] if lot_die_counts[lot_id] else None
            ),
        }
        for param, values in sorted(lot_param_values[lot_id].items()):
            prefix = f"core_{param}"
            stats = compute_dist_stats(values)
            stats_map = {
                k: stable_float(v) if isinstance(v, float) else v
                for k, v in stats_to_dict(prefix, stats).items()
            }
            lot_row.update(stats_map)
            lp: dict[str, Any] = {
                "lot_id": lot_id,
                "parameter": param,
                "domain": "core",
                "has_current_limit": int(param in param_to_limit),
            }
            if param in param_to_limit:
                lim = param_to_limit[param]
                lp["current_limit"] = lim.value
                lp["limit_direction"] = lim.direction
                lp["unit"] = lim.unit
                lp["limit_source_status"] = lim.source_status
                lp["test_id"] = lim.test_id
            lp.update(stats_map)
            if param in lot_margin_aggs[lot_id]:
                lp.update(
                    {
                        k: stable_float(v) if isinstance(v, float) else v
                        for k, v in lot_margin_aggs[lot_id][param].to_dict(prefix).items()
                    }
                )
            result.lot_parameter_rows.append(lp)
        for param, agg in lot_margin_aggs[lot_id].items():
            prefix = f"core_{param}"
            lot_row.update(
                {
                    k: stable_float(v) if isinstance(v, float) else v
                    for k, v in agg.to_dict(prefix).items()
                }
            )
        result.lot_rows.append(lot_row)

    result.sequence_contract = {
        "domain": "core",
        "representation": "GRU-ready ordered pattern sequence (not a trained model)",
        "sequence_key": ["lot_id", "die_id"],
        "step_key": ["pattern_id"],
        "sequence_ordering": "pattern_id ascending integers 1..N",
        "expected_sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "observed_sequence_length_min": min(seq_lengths) if seq_lengths else None,
        "observed_sequence_length_max": max(seq_lengths) if seq_lengths else None,
        "observed_sequence_length_distribution": {
            str(k): seq_lengths.count(k) for k in sorted(set(seq_lengths))
        },
        "feature_dimension": feature_dim,
        "raw_feature_order": list(SEQUENCE_FEATURE_ORDER),
        "derived_per_step_features": [
            "ir_drop_margin",
            "thermal_margin",
            "ir_drop_proximity_class",
            "thermal_proximity_class",
            "ir_drop_violation_flag",
            "thermal_violation_flag",
        ],
        "padding_policy": "none — current dataset is complete; do not pad/truncate without justification",
        "missing_step_strategy": "document incompleteness in manifest; do not invent measurements",
        "masking_strategy": "valid_step flag + per-feature *_missing flags for future GRU mask",
        "normalization_strategy": {
            "method": "per-feature train-lot z-score or robust median/IQR (Phase 6/7)",
            "fitting_population": "training lots only — never fit on holdout/test",
            "do_not_normalize_away": "signed margins to current limits may be kept unscaled or jointly scaled with care",
            "phase3_status": "contract only; scalers not fitted here (leakage-safe)",
        },
        "candidate_dependent": False,
        "valid_sequences": valid_sequences,
        "incomplete_sequences": incomplete,
        "missing_steps_total": missing_steps_total,
        "duplicate_steps": duplicate_steps,
        "ordering_invalid_count": ordering_invalid,
        "number_of_sequences": len(seq_lengths),
        "guard_band_borderline_margin_percent": borderline_pct,
        "guard_band_note": "Limit proximity only — not reliability or escape probability",
    }

    # Registry specs
    specs: list[FeatureSpec] = []
    for param in sorted({p for d in die_param_values.values() for p in d}):
        unit = None
        for t in canonical.core_tests.values():
            if t.parameter == param:
                unit = t.unit
                break
        specs.extend(
            dist_feature_specs(
                prefix=f"core_{param}",
                domain="core",
                grain="die|lot|lot×parameter",
                parameter=param,
                unit=unit,
            )
        )
        if param in param_to_limit:
            lim = param_to_limit[param]
            specs.extend(
                margin_feature_specs(
                    prefix=f"core_{param}",
                    domain="core",
                    grain="die|lot|lot×parameter",
                    parameter=param,
                    unit=lim.unit,
                    direction=lim.direction,
                )
            )
    for feat in SEQUENCE_FEATURE_ORDER:
        specs.append(
            FeatureSpec(
                feature_name=f"seq_step_{feat}",
                domain="sequence",
                grain="pattern-step (lot×die×pattern)",
                source_parameters=[feat],
                formula="raw measurement at pattern step",
                unit=None,
                direction=None,
                allowed_for_ml=True,
                candidate_dependent=False,
                evaluation_only=False,
                normalization_required=True,
                description="GRU step raw feature",
            )
        )
    result.registry_specs = specs
    return result
