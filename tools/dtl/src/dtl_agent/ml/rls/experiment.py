"""Offline RLS vs CoreGRU experimental comparison (shadow only).

Does not modify production recommendation / GRU paths.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dtl_agent.config.paths import default_project_root
from dtl_agent.ml.rls.data import (
    MONTH_TEST,
    MONTH_TRAIN,
    MONTH_VAL,
    load_gru_test_predictions,
    load_month_temporal_split,
)
from dtl_agent.ml.rls.eval_metrics import (
    compare_dtl,
    decide_all,
    ranking_metrics,
    regression_metrics,
)
from dtl_agent.ml.rls.features import (
    RLS_FEATURE_NAMES,
    assert_no_forbidden_features,
    build_feature_matrix,
)
from dtl_agent.ml.rls.regressor import RLSRegressor

DEFAULT_LAMBDAS = (0.99, 0.995, 0.999, 1.0)
DEFAULT_DELTA = 10.0


@dataclass
class ExperimentConfig:
    forgetting_factor: float = 0.995
    delta: float = DEFAULT_DELTA
    lambdas_to_tune: tuple[float, ...] = DEFAULT_LAMBDAS
    output_dir: str = "artifacts/temporal/shared/rls_experiment"


def _select_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambdas: tuple[float, ...],
    delta: float,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    best_l = lambdas[0]
    best_mae = float("inf")
    for lam in lambdas:
        model = RLSRegressor(X_train.shape[1], forgetting_factor=lam, delta=delta)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        m = regression_metrics(y_val, pred)
        scores[str(lam)] = m["mae"]
        if m["mae"] < best_mae:
            best_mae = m["mae"]
            best_l = lam
    return best_l, scores


def _attach_scores(examples: pd.DataFrame, scores: np.ndarray, col: str) -> pd.DataFrame:
    out = examples.copy()
    out[col] = scores
    return out


def _join_key_cols() -> list[str]:
    return [
        "production_month",
        "lot_id",
        "die_id",
        "parameter",
        "candidate_limit",
    ]


def run_rls_experiment(
    *,
    root: Path | None = None,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Train RLS on Jan, tune on Feb, evaluate vs CoreGRU on Mar test-lot rows."""
    root = root or default_project_root()
    config = config or ExperimentConfig()
    assert_no_forbidden_features()

    out_dir = root / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_month_temporal_split(root)
    seq_store = data.seq_store  # type: ignore[assignment]

    t0 = time.perf_counter()
    X_tr, y_tr, _ = build_feature_matrix(data.train, seq_store)  # type: ignore[arg-type]
    X_va, y_va, _ = build_feature_matrix(data.validation, seq_store)  # type: ignore[arg-type]
    X_te, y_te, _ = build_feature_matrix(data.test, seq_store)  # type: ignore[arg-type]
    feat_s = time.perf_counter() - t0

    best_lambda, lambda_scores = _select_lambda(
        X_tr, y_tr, X_va, y_va, config.lambdas_to_tune, config.delta
    )

    model = RLSRegressor(
        n_features=X_tr.shape[1],
        forgetting_factor=best_lambda,
        delta=config.delta,
    )
    t1 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t1

    # Frozen RLS predictions
    t2 = time.perf_counter()
    pred_tr = model.predict(X_tr)
    pred_va = model.predict(X_va)
    pred_te = model.predict(X_te)
    # latency per row on test
    t_lat0 = time.perf_counter()
    _ = model.predict(X_te[: min(1000, len(X_te))])
    t_lat1 = time.perf_counter()
    n_lat = min(1000, len(X_te))
    latency_ms = 1000.0 * (t_lat1 - t_lat0) / max(n_lat, 1)
    pred_s = time.perf_counter() - t2

    tracemalloc.start()
    _ = model.predict(X_te[: min(500, len(X_te))])
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rls_metrics = {
        "train": regression_metrics(y_tr, pred_tr),
        "validation": regression_metrics(y_va, pred_va),
        "test_full_month": regression_metrics(y_te, pred_te),
    }

    test_scored = _attach_scores(data.test, pred_te, "rls_score")
    rls_rank_full = ranking_metrics(test_scored, score_col="rls_score")

    # --- Fair head-to-head vs CoreGRU on shared lot-split TEST lots for Mar ---
    gru_pred = load_gru_test_predictions(root)
    gru_mar = gru_pred[gru_pred["production_month"] == MONTH_TEST].copy()
    if "pred_temporal_gru" not in gru_mar.columns:
        raise RuntimeError("core_test_predictions missing pred_temporal_gru")

    # Restrict RLS test rows to the same keys as GRU held-out predictions.
    keys = _join_key_cols()
    cmp = test_scored.merge(
        gru_mar[keys + ["pred_temporal_gru"]].drop_duplicates(keys),
        on=keys,
        how="inner",
    )
    if cmp.empty:
        raise RuntimeError("No overlapping Mar rows between RLS test and GRU predictions")

    y_cmp = cmp["target_score"].to_numpy(dtype=float)
    rls_cmp = cmp["rls_score"].to_numpy(dtype=float)
    gru_cmp = cmp["pred_temporal_gru"].to_numpy(dtype=float)

    model_compare = {
        "n_rows": int(len(cmp)),
        "scope": "2026-03 lot-split TEST lots only (fair shared candidate set)",
        "rls": regression_metrics(y_cmp, rls_cmp),
        "gru": regression_metrics(y_cmp, gru_cmp),
        "rls_ranking": ranking_metrics(cmp.assign(ml_score=rls_cmp), score_col="rls_score"),
        "gru_ranking": ranking_metrics(
            cmp.assign(ml_score=gru_cmp), score_col="pred_temporal_gru"
        ),
        "score_correlation_rls_gru": float(np.corrcoef(rls_cmp, gru_cmp)[0, 1])
        if len(cmp) > 1
        else float("nan"),
    }

    # DTL policy on the same comparison frame
    dtl_rls = decide_all(cmp.assign(ml_score=cmp["rls_score"]), score_col="ml_score")
    dtl_gru = decide_all(cmp.assign(ml_score=cmp["pred_temporal_gru"]), score_col="ml_score")
    dtl_compare = compare_dtl(dtl_gru, dtl_rls, a_name="gru", b_name="rls")

    # Oracle: max simulated_yield among all candidates (no Top-N) for context
    oracle_rows = []
    for _, g in cmp.groupby(["production_month", "lot_id", "die_id", "parameter"], sort=False):
        win = g.sort_values("simulated_yield", ascending=False).iloc[0]
        oracle_rows.append(
            {
                "production_month": win["production_month"],
                "lot_id": win["lot_id"],
                "die_id": win["die_id"],
                "parameter": win["parameter"],
                "oracle_limit": float(win["candidate_limit"]),
                "oracle_yield": float(win["simulated_yield"]),
            }
        )
    oracle = pd.DataFrame(oracle_rows)
    dtl_rls_o = dtl_rls.merge(oracle, on=["production_month", "lot_id", "die_id", "parameter"])
    dtl_gru_o = dtl_gru.merge(oracle, on=["production_month", "lot_id", "die_id", "parameter"])
    dtl_compare["rls_vs_oracle_limit_agreement"] = float(
        np.mean(np.isclose(dtl_rls_o["recommended_limit"], dtl_rls_o["oracle_limit"]))
    )
    dtl_compare["gru_vs_oracle_limit_agreement"] = float(
        np.mean(np.isclose(dtl_gru_o["recommended_limit"], dtl_gru_o["oracle_limit"]))
    )
    dtl_compare["rls_mean_selected_yield"] = float(dtl_rls["simulated_yield"].mean())
    dtl_compare["gru_mean_selected_yield"] = float(dtl_gru["simulated_yield"].mean())

    # --- Online adaptation: update with Feb, re-score Mar comparison rows ---
    model_online = RLSRegressor(
        n_features=X_tr.shape[1],
        forgetting_factor=best_lambda,
        delta=config.delta,
    )
    model_online.fit(X_tr, y_tr)
    cmp_examples = (
        data.test.merge(cmp[keys].drop_duplicates(), on=keys, how="inner")
        .drop_duplicates(subset=["example_id"])
        .sort_values("example_id")
        .reset_index(drop=True)
    )
    X_cmp, y_cmp2, _ = build_feature_matrix(cmp_examples, seq_store)  # type: ignore[arg-type]
    before = model_online.predict(X_cmp)
    before_metrics = regression_metrics(y_cmp2, before)

    t_up0 = time.perf_counter()
    for i in range(len(X_va)):
        model_online.update(X_va[i], float(y_va[i]))
    online_update_s = time.perf_counter() - t_up0
    after = model_online.predict(X_cmp)
    after_metrics = regression_metrics(y_cmp2, after)

    online = {
        "updated_with": MONTH_VAL,
        "evaluated_on": f"{MONTH_TEST} lot-split TEST overlap",
        "before_feb_updates": before_metrics,
        "after_feb_updates": after_metrics,
        "mae_improvement": float(before_metrics["mae"] - after_metrics["mae"]),
        "online_update_seconds": online_update_s,
        "n_online_updates": int(len(X_va)),
    }

    # Save model (frozen Jan-trained)
    model_path = out_dir / "rls_core_jan_trained.json"
    model.save(model_path)
    online_path = out_dir / "rls_core_jan_plus_feb_online.json"
    model_online.save(online_path)

    # Persist scored comparison + DTL tables
    cmp_out = cmp.copy()
    cmp_out.to_parquet(out_dir / "mar_testlot_score_comparison.parquet", index=False)
    dtl_rls.to_csv(out_dir / "dtl_decisions_rls.csv", index=False)
    dtl_gru.to_csv(out_dir / "dtl_decisions_gru.csv", index=False)

    summary: dict[str, Any] = {
        "experiment": "rls_vs_core_gru_shadow",
        "production_path_unchanged": True,
        "split": {
            "train_month": MONTH_TRAIN,
            "validation_month": MONTH_VAL,
            "test_month": MONTH_TEST,
            "note": data.split_note,
            "train_rows": int(len(data.train)),
            "validation_rows": int(len(data.validation)),
            "test_rows": int(len(data.test)),
        },
        "target": {
            "name": "target_score",
            "definition": "objective_score from month-scoped Core simulation (same as CoreGRU)",
        },
        "features": {
            "names": list(RLS_FEATURE_NAMES),
            "n_features": len(RLS_FEATURE_NAMES),
        },
        "hyperparameters": {
            "forgetting_factor": best_lambda,
            "delta": config.delta,
            "lambda_validation_mae": lambda_scores,
        },
        "system": {
            "model_nbytes": model.model_nbytes(),
            "model_size_kb": round(model.model_nbytes() / 1024.0, 3),
            "inference_latency_ms_per_row": latency_ms,
            "feature_build_seconds": feat_s,
            "train_seconds": train_s,
            "predict_seconds": pred_s,
            "tracemalloc_peak_bytes_predict": int(peak),
            "n_updates_after_fit": model.n_updates,
        },
        "rls_month_metrics": rls_metrics,
        "rls_ranking_test_full_month": rls_rank_full,
        "model_compare_fair_testlots": model_compare,
        "dtl_compare_fair_testlots": dtl_compare,
        "online_adaptation": online,
        "artifacts": {
            "model": str(model_path.relative_to(root)),
            "model_online": str(online_path.relative_to(root)),
            "output_dir": str(out_dir.relative_to(root)),
        },
        "limitations": [
            "RLS trained on 2026-01 only; CoreGRU checkpoint was trained on multi-month lot-split train data.",
            "Head-to-head uses Mar lot-split TEST lots where pred_temporal_gru exists.",
            "Offline DTL eval mirrors Top-N + yield-first policy but does not invoke production pipeline.py.",
            "Core parameters only (ir_drop, thermal); UnifiedGRU parametric path not included.",
            "Sequence info enters RLS only as compact aggregates, not full 200-step dynamics.",
        ],
    }

    (out_dir / "RLS_EXPERIMENT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = run_rls_experiment()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
