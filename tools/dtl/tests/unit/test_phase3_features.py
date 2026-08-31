"""Phase 3 unit tests: features, margins, sequences, registry, leakage."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.canonical.dataset import build_canonical_from_datasets
from dtl_agent.config.allowlists import FORBIDDEN_COLUMN_NAMES
from dtl_agent.data.loaders.core_loader import CoreDataLoader
from dtl_agent.data.loaders.parametric_loader import ParametricDataLoader
from dtl_agent.features.core_engine import SEQUENCE_FEATURE_ORDER, build_core_features
from dtl_agent.features.cross_domain import build_cross_domain_features
from dtl_agent.features.io_utils import write_csv_dicts
from dtl_agent.features.margins import LimitSpec, is_violation, proximity_class, signed_margin
from dtl_agent.features.parametric_engine import build_parametric_features
from dtl_agent.features.registry import FeatureRegistry, FeatureSpec
from dtl_agent.features.stats import compute_dist_stats
from tests.conftest import minimal_core_fixture, minimal_parametric_fixture, write_csv


@pytest.fixture
def canonical_fixture(tmp_path: Path):
    core = CoreDataLoader(minimal_core_fixture(tmp_path)).load(materialize_measurements=True)
    parametric = ParametricDataLoader(minimal_parametric_fixture(tmp_path)).load(
        materialize_measurements=True
    )
    return build_canonical_from_datasets(core, parametric)


def test_distribution_features() -> None:
    stats = compute_dist_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats is not None
    assert stats.count == 5
    assert stats.mean == 3.0
    assert stats.median == 3.0
    assert stats.min == 1.0
    assert stats.max == 5.0


def test_margin_and_violation_upper_lower() -> None:
    up = LimitSpec("UPPER", 25.0, "mV", "SOURCE_CONFIRMED", "T_IR_DROP_MV", "ir_drop")
    lo = LimitSpec("LOWER", 1.15, "V", "SYNTHETIC_ASSUMED", "T_VMAX", "VMAX")
    assert signed_margin(20.0, up) == 5.0
    assert is_violation(26.0, up) is True
    assert is_violation(24.0, up) is False
    assert signed_margin(1.20, lo) == pytest.approx(0.05)
    assert is_violation(1.10, lo) is True
    assert proximity_class(24.0, up, borderline_margin_percent=5.0) in {
        "SAFE",
        "BORDERLINE",
        "VIOLATION",
    }


def test_core_features_and_pattern_ordering(canonical_fixture) -> None:
    result = build_core_features(canonical_fixture)
    assert len(result.die_rows) == 1
    assert result.pattern_rows[0]["pattern_id"] == 1
    assert result.pattern_rows[0]["sequence_index"] == 0
    assert "ir_drop" in result.pattern_rows[0]
    assert result.die_rows[0]["core_ir_drop_mean"] == pytest.approx(20.0)
    assert "core_ir_drop_violation_rate" in result.die_rows[0]


def test_sequence_feature_order_contract(canonical_fixture) -> None:
    result = build_core_features(canonical_fixture)
    assert result.sequence_contract["raw_feature_order"] == list(SEQUENCE_FEATURE_ORDER)
    assert result.sequence_contract["candidate_dependent"] is False
    assert result.sequence_contract["feature_dimension"] == 5


def test_parametric_condition_and_limit_features(canonical_fixture) -> None:
    result = build_parametric_features(canonical_fixture)
    assert len(result.condition_rows) >= 1
    assert {r["condition_id"] for r in result.condition_rows} <= {
        "COND_RT_NOM",
        "COND_HOT_NOM",
        "COND_RT_LOWV",
        "COND_HOT_HIGHV",
    }
    # Fixture only has one condition measurement row
    die = result.die_rows[0]
    assert "param_iddq_mean" in die
    only = [r for r in result.die_rows if r["die_id"] == "LOT_P_ONLY_D001"]
    # parametric-only die may have no measurements in fixture (only LOT_A measured)
    linked = [r for r in result.die_rows if r["die_id"] == "LOT_A_D001"][0]
    assert linked["cross_domain_available"] == 1


def test_cross_domain_linked_vs_parametric_only(canonical_fixture) -> None:
    core = build_core_features(canonical_fixture)
    par = build_parametric_features(canonical_fixture)
    cross = build_cross_domain_features(
        canonical_fixture,
        core_die_rows=core.die_rows,
        parametric_die_rows=par.die_rows,
        core_lot_rows=core.lot_rows,
        parametric_lot_rows=par.lot_rows,
    )
    assert len(cross.linked_die_rows) == 1
    assert cross.linked_die_rows[0]["join_type"] == "entity_summary_not_measurement_row"
    assert all(r["lot_id"] != "LOT_P_ONLY" for r in cross.linked_die_rows)


def test_feature_registry_rejects_eval_only() -> None:
    reg = FeatureRegistry()
    with pytest.raises(ValueError):
        reg.add(
            FeatureSpec(
                feature_name="bad",
                domain="core",
                grain="lot",
                source_parameters=[],
                formula="x",
                unit=None,
                direction=None,
                allowed_for_ml=False,
                candidate_dependent=False,
                evaluation_only=True,
                normalization_required=False,
                description="leak",
            )
        )


def test_leakage_protection_column_names(canonical_fixture) -> None:
    result = build_core_features(canonical_fixture)
    cols = set(result.die_rows[0]) | set(result.pattern_rows[0])
    for col in cols:
        for fb in FORBIDDEN_COLUMN_NAMES:
            assert fb.lower() not in col.lower()


def test_determinism_core_features(canonical_fixture) -> None:
    a = build_core_features(canonical_fixture)
    b = build_core_features(canonical_fixture)
    assert a.die_rows == b.die_rows
    assert a.pattern_rows == b.pattern_rows
    assert a.lot_rows == b.lot_rows


def test_grain_fields_present(canonical_fixture) -> None:
    core = build_core_features(canonical_fixture)
    assert {"lot_id", "die_id", "pattern_id", "sequence_index"} <= set(core.pattern_rows[0])
    par = build_parametric_features(canonical_fixture)
    assert {"lot_id", "die_id", "condition_id"} <= set(par.condition_rows[0])


def test_missing_step_documented_when_incomplete(canonical_fixture) -> None:
    result = build_core_features(canonical_fixture)
    # fixture has 1 pattern, expected 200
    assert result.sequence_manifest[0]["complete"] == 0
    assert result.sequence_contract["incomplete_sequences"] == 1
