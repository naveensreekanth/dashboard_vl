"""Focused unit tests for experimental RLS (no production path changes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.ml.rls.data import MONTH_TEST, MONTH_TRAIN, MONTH_VAL, load_month_temporal_split
from dtl_agent.ml.rls.eval_metrics import decide_all, ranking_metrics, regression_metrics
from dtl_agent.ml.rls.features import (
    RLS_FEATURE_NAMES,
    assert_no_forbidden_features,
    build_feature_matrix,
    build_feature_vector,
    sequence_aggregates,
)
from dtl_agent.ml.rls.regressor import RLSRegressor
from dtl_agent.ml.unified_experiment import FORBIDDEN_INPUT_COLS
from dtl_agent.recommendation import pipeline as rec_pipeline

ROOT = default_project_root()
TEMPORAL_JAN = (
    ROOT
    / "artifacts"
    / "temporal"
    / "2026-01"
    / "ml_dataset"
    / "train"
    / "core_candidate_examples.parquet"
)


def test_rls_init_predict_update_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    theta_true = np.array([1.0, -0.5, 0.25, 2.0])
    y = X @ theta_true + rng.normal(scale=0.01, size=50)

    m1 = RLSRegressor(4, forgetting_factor=0.999, delta=1.0)
    m1.fit(X, y)
    m2 = RLSRegressor(4, forgetting_factor=0.999, delta=1.0)
    m2.fit(X, y)
    np.testing.assert_allclose(m1.theta, m2.theta, rtol=1e-12, atol=1e-12)

    x_new = rng.normal(size=4)
    y_new = float(x_new @ theta_true)
    before = m1.predict_one(x_new)
    err = m1.update(x_new, y_new)
    after = m1.predict_one(x_new)
    assert err >= 0.0
    assert abs(after - y_new) <= abs(before - y_new) + 1e-9


def test_rls_forgetting_factor_and_repeated_updates():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=float)
    y_old = np.array([0.0, 1.0, 2.0])
    m_slow = RLSRegressor(2, forgetting_factor=1.0, delta=1.0)
    m_fast = RLSRegressor(2, forgetting_factor=0.9, delta=1.0)
    m_slow.fit(X, y_old)
    m_fast.fit(X, y_old)
    for _ in range(30):
        m_slow.update(np.array([1.0, 1.0]), 10.0)
        m_fast.update(np.array([1.0, 1.0]), 10.0)
    assert abs(m_fast.predict_one(np.array([1.0, 1.0])) - 10.0) < abs(
        m_slow.predict_one(np.array([1.0, 1.0])) - 10.0
    )


def test_rls_save_load(tmp_path: Path):
    m = RLSRegressor(3, forgetting_factor=0.99, delta=5.0)
    X = np.eye(3)
    y = np.array([1.0, 2.0, 3.0])
    m.fit(X, y)
    path = tmp_path / "rls.json"
    m.save(path)
    m2 = RLSRegressor.load(path)
    np.testing.assert_allclose(m.theta, m2.theta)
    np.testing.assert_allclose(m.P, m2.P)
    assert m2.forgetting_factor == 0.99
    assert m2.n_updates == m.n_updates


def test_rls_numerical_stability_ill_conditioned():
    x = np.array([1.0, 1.0 + 1e-12, 1.0])
    m = RLSRegressor(3, forgetting_factor=0.995, delta=1.0)
    for i in range(200):
        m.update(x, float(i % 5))
    assert np.isfinite(m.theta).all()
    assert np.isfinite(m.P).all()
    assert np.isfinite(m.predict_one(x))


def test_feature_vector_contract_and_no_forbidden():
    assert_no_forbidden_features()
    assert not (set(RLS_FEATURE_NAMES) & FORBIDDEN_INPUT_COLS)
    seq = np.zeros((200, 5), dtype=float)
    seq[:, 0] = np.linspace(0.1, 0.5, 200)
    agg = sequence_aggregates(seq, "ir_drop")
    vec = build_feature_vector(
        parameter="ir_drop",
        direction="UPPER",
        tighten_or_loosen="TIGHTER",
        lot_category="EDGE",
        candidate_limit=50.0,
        current_limit=55.0,
        candidate_delta=-5.0,
        candidate_delta_percent=-9.0,
        agg=agg,
    )
    assert vec.shape == (len(RLS_FEATURE_NAMES),)
    assert vec[0] == 1.0
    assert vec[list(RLS_FEATURE_NAMES).index("param_ir_drop")] == 1.0
    assert vec[list(RLS_FEATURE_NAMES).index("param_thermal")] == 0.0


def test_eval_metrics_regression_and_dtl():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 1.9, 2.5])
    m = regression_metrics(y, p)
    assert m["n"] == 3
    assert m["mae"] > 0
    df = pd.DataFrame(
        {
            "production_month": ["2026-03"] * 4,
            "lot_id": ["L1"] * 4,
            "die_id": ["D1"] * 4,
            "parameter": ["ir_drop"] * 4,
            "candidate_limit": [40.0, 50.0, 55.0, 60.0],
            "current_limit": [55.0] * 4,
            "tighten_or_loosen": ["TIGHTER", "TIGHTER", "CURRENT", "LOOSER"],
            "simulated_yield": [0.9, 0.95, 0.95, 0.8],
            "target_score": [0.7, 0.9, 0.85, 0.5],
            "ml_score": [0.2, 0.8, 0.7, 0.1],
        }
    )
    dec = decide_all(df, score_col="ml_score")
    assert len(dec) == 1
    assert dec.iloc[0]["recommended_limit"] in {50.0, 55.0}
    rk = ranking_metrics(df, score_col="ml_score")
    assert rk["n_groups"] == 1


@pytest.mark.skipif(not TEMPORAL_JAN.is_file(), reason="temporal ml dataset missing")
def test_temporal_month_split_shape():
    data = load_month_temporal_split(ROOT)
    assert data.train["production_month"].eq(MONTH_TRAIN).all()
    assert data.validation["production_month"].eq(MONTH_VAL).all()
    assert data.test["production_month"].eq(MONTH_TEST).all()
    assert len(data.train) > 0 and len(data.test) > 0
    assert set(data.train["production_month"].unique()) == {MONTH_TRAIN}


@pytest.mark.skipif(not TEMPORAL_JAN.is_file(), reason="temporal ml dataset missing")
def test_feature_matrix_small_sample():
    data = load_month_temporal_split(ROOT)
    sample = data.train.head(20)
    X, y, ids = build_feature_matrix(sample, data.seq_store)  # type: ignore[arg-type]
    assert X.shape == (20, len(RLS_FEATURE_NAMES))
    assert y.shape == (20,)
    assert len(ids) == 20
    assert np.isfinite(X).all()


def test_production_pipeline_not_wired_to_rls():
    src = Path(rec_pipeline.__file__).read_text(encoding="utf-8")
    assert "rls" not in src.lower()
    assert "RLSRegressor" not in src


def test_yield_tie_metrics_detect_ties():
    from dtl_agent.ml.rls.eval_metrics import yield_tie_dtl_metrics

    df = pd.DataFrame(
        {
            "production_month": ["2026-03"] * 4,
            "lot_id": ["L1"] * 4,
            "die_id": ["D1"] * 4,
            "parameter": ["ir_drop"] * 4,
            "candidate_limit": [40.0, 50.0, 55.0, 60.0],
            "current_limit": [55.0] * 4,
            "tighten_or_loosen": ["TIGHTER", "TIGHTER", "CURRENT", "LOOSER"],
            "simulated_yield": [0.9, 1.0, 1.0, 0.8],
            "target_score": [0.7, 0.9, 0.85, 0.5],
            "ml_score": [0.2, 0.95, 0.5, 0.1],
        }
    )
    m = yield_tie_dtl_metrics(df, score_col="ml_score")
    assert m["n_groups"] == 1
    assert m["yield_tie_rate"] == 1.0
    assert m["n_yield_tied_groups"] == 1
    # Top scores pick 50.0 among yield=1.0 ties
    assert abs(m["mean_selected_yield"] - 1.0) < 1e-9


def test_jan_shadow_path_not_production():
    from dtl_agent.ml.rls.jan_gru_shadow import PRODUCTION_CKPT_REL, SHADOW_DIR_REL, SHADOW_CKPT_NAME

    shadow = SHADOW_DIR_REL / SHADOW_CKPT_NAME
    assert shadow != PRODUCTION_CKPT_REL
    assert "rls_experiment" in shadow.as_posix()
    assert "jan_gru_shadow" in shadow.as_posix()


def test_equal_info_experiment_uses_shadow_gru_not_production_preds():
    src = Path(
        default_project_root()
        / "src"
        / "dtl_agent"
        / "ml"
        / "rls"
        / "equal_info_experiment.py"
    ).read_text(encoding="utf-8")
    assert "pred_temporal_gru" not in src
    assert "load_gru_test_predictions" not in src
    assert "load_jan_shadow_scorer" in src
    assert "train_jan_only_core_gru" in src


@pytest.mark.skipif(not TEMPORAL_JAN.is_file(), reason="temporal ml dataset missing")
def test_feb_validation_sorted_for_online_updates():
    data = load_month_temporal_split(ROOT)
    ids = data.validation["example_id"].astype(str).tolist()
    assert ids == sorted(ids)
    assert data.validation["production_month"].eq(MONTH_VAL).all()
