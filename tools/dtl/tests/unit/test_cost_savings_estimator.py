"""Unit tests for production-facing counterfactual DTL cost-savings estimator."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.analysis.cost_savings import (
    CostSavingsAssumptions,
    compute_margin,
    estimate_cost_savings,
    is_core_parameter,
    is_parametric_parameter,
    parameter_direction,
    record_times,
    seconds_to_cost,
    skip_remaining_conditions,
)

ROOT = Path(__file__).resolve().parents[2]


def test_upper_and_lower_margin():
    assert compute_margin(direction="UPPER", recommended_limit=100.0, measured_value=80.0) == 20.0
    assert compute_margin(direction="LOWER", recommended_limit=1.05, measured_value=1.20) == pytest.approx(0.15)


def test_skip_when_margin_meets_threshold():
    assert skip_remaining_conditions(
        decision="RECOMMEND",
        parameter="IDDQ",
        direction="UPPER",
        recommended_limit=100.0,
        measured_value=80.0,
        skip_threshold=0.10,
    )
    assert not skip_remaining_conditions(
        decision="RECOMMEND",
        parameter="IDDQ",
        direction="UPPER",
        recommended_limit=100.0,
        measured_value=99.95,
        skip_threshold=0.10,
    )


def test_lower_skip_and_no_skip():
    assert skip_remaining_conditions(
        decision="RECOMMEND",
        parameter="VMAX",
        direction="LOWER",
        recommended_limit=1.05,
        measured_value=1.20,
        skip_threshold=0.10,
    )
    assert not skip_remaining_conditions(
        decision="RECOMMEND",
        parameter="VMAX",
        direction="LOWER",
        recommended_limit=1.05,
        measured_value=1.08,
        skip_threshold=0.10,
    )


def test_baseline_and_skip_times():
    a = CostSavingsAssumptions(condition_duration_s=0.05)
    baseline, dtl, saved = record_times(skip=True, condition_duration_s=a.condition_duration_s)
    assert baseline == pytest.approx(0.20)
    assert dtl == pytest.approx(0.05)
    assert saved == pytest.approx(0.15)  # 3 conditions


def test_no_skip_times_equal():
    baseline, dtl, saved = record_times(skip=False, condition_duration_s=0.05)
    assert baseline == dtl == pytest.approx(0.20)
    assert saved == 0.0


def test_core_and_non_recommend_never_skip():
    assert is_core_parameter("ir_drop")
    assert not is_parametric_parameter("ir_drop")
    assert not skip_remaining_conditions(
        decision="RECOMMEND",
        parameter="ir_drop",
        direction="UPPER",
        recommended_limit=50.0,
        measured_value=10.0,
        skip_threshold=0.10,
    )
    assert not skip_remaining_conditions(
        decision="KEEP_CURRENT",
        parameter="IDDQ",
        direction="UPPER",
        recommended_limit=100.0,
        measured_value=10.0,
        skip_threshold=0.10,
    )
    assert not skip_remaining_conditions(
        decision="REJECT",
        parameter="IDDQ",
        direction="UPPER",
        recommended_limit=100.0,
        measured_value=10.0,
        skip_threshold=0.10,
    )


def test_missing_measurement_no_skip():
    assert not skip_remaining_conditions(
        decision="RECOMMEND",
        parameter="IDDQ",
        direction="UPPER",
        recommended_limit=100.0,
        measured_value=None,
        skip_threshold=0.10,
    )


def test_cost_conversion():
    # 0.15 s * $25/h / 3600 = 0.0010416...
    assert seconds_to_cost(0.15, 25.0) == pytest.approx(0.15 / 3600 * 25.0)


def test_directions_from_catalog():
    assert parameter_direction("IDDQ") == "UPPER"
    assert parameter_direction("VMAX") == "LOWER"
    assert parameter_direction("ir_drop") == "UPPER"


def test_estimate_from_synthetic_rows_no_volume_fabrication():
    rows = [
        {
            "die_id": "D1",
            "lot_id": "L1",
            "production_month": "2026-01",
            "parameter": "IDDQ",
            "parameter_display": "IDDQ",
            "decision": "RECOMMEND",
            "current_limit": 50.0,
            "recommended_limit": 100.0,
        },
        {
            "die_id": "D1",
            "lot_id": "L1",
            "production_month": "2026-01",
            "parameter": "ir_drop",
            "parameter_display": "IR_DROP_MV",
            "decision": "RECOMMEND",
            "current_limit": 25.0,
            "recommended_limit": 50.0,
        },
        {
            "die_id": "D2",
            "lot_id": "L1",
            "production_month": "2026-01",
            "parameter": "IDDQ",
            "parameter_display": "IDDQ",
            "decision": "KEEP_CURRENT",
            "current_limit": 50.0,
            "recommended_limit": 50.0,
        },
        {
            "die_id": "D3",
            "lot_id": "L1",
            "production_month": "2026-01",
            "parameter": "VMIN",
            "parameter_display": "VMIN",
            "decision": "REJECT",
            "current_limit": 0.85,
            "recommended_limit": None,
        },
    ]
    # Patch nom lookup by providing recommendation_rows and monkeypatching index
    from dtl_agent.analysis import cost_savings as cs

    original = cs._load_cond_rt_nom_index

    def fake_index(_root: str, _month: str):
        return {
            ("L1", "D1", "IDDQ"): 80.0,  # margin 20 >= 0.1 → skip
            ("L1", "D2", "IDDQ"): 10.0,
            ("L1", "D3", "VMIN"): 0.5,
        }

    cs._load_cond_rt_nom_index = fake_index  # type: ignore[assignment]
    try:
        out = estimate_cost_savings(
            ROOT,
            assumptions=CostSavingsAssumptions(
                condition_duration_s=0.05,
                skip_threshold=0.10,
                tester_cost_per_hour=25.0,
            ),
            recommendation_rows=rows,
        )
    finally:
        cs._load_cond_rt_nom_index = original  # type: ignore[assignment]

    assert out["is_measured_ate_saving"] is False
    assert out["status"] == "predicted"
    assert out["aggregate"]["records_evaluated"] == 4
    assert out["aggregate"]["production_volume_supplied"] is False
    assert "assumptions" in out["estimator"]
    assert out["estimator"]["cost_source"] == "configured assumption"

    by_param = {(r["die_id"], r["parameter"]): r for r in out["per_device"]}
    assert by_param[("D1", "IDDQ")]["skip_remaining_conditions"] is True
    assert by_param[("D1", "IDDQ")]["estimated_seconds_saved"] == pytest.approx(0.15)
    assert by_param[("D1", "ir_drop")]["estimated_seconds_saved"] == 0.0
    assert by_param[("D2", "IDDQ")]["estimated_seconds_saved"] == 0.0  # KEEP_CURRENT
    assert by_param[("D3", "VMIN")]["estimated_seconds_saved"] == 0.0  # REJECT

    # Recommendation limits must pass through unchanged
    assert by_param[("D1", "IDDQ")]["current_limit"] == 50.0
    assert by_param[("D1", "IDDQ")]["recommended_limit"] == 100.0


def test_full_artifact_estimate_runs():
    out = estimate_cost_savings(
        ROOT,
        assumptions=CostSavingsAssumptions(),
        include_per_device=True,
    )
    assert out["aggregate"]["records_evaluated"] == 108
    assert out["aggregate"]["eligible_records"] == 84  # 7 parametric × 12
    assert out["is_measured_ate_saving"] is False
    assert out["aggregate"]["total_predicted_cost_saving"] >= 0
    # Core rows present with zero savings
    core = [r for r in out["per_device"] if r["parameter"] in {"ir_drop", "thermal"}]
    assert core
    assert all(r["estimated_seconds_saved"] == 0.0 for r in core)
