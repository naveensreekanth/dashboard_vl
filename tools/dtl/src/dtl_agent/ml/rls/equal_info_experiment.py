"""Equal-information RLS vs Jan-only CoreGRU evaluation (shadow only).

Arm A: train both on January, tune on February, evaluate frozen on March.
Arm B: sequential RLS Jan→Feb online updates, evaluate March; GRU stays frozen.

Does not modify production recommendation / checkpoints.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dtl_agent.config.paths import default_project_root
from dtl_agent.features.io_utils import file_sha256, write_json
from dtl_agent.ml.rls.data import MONTH_TEST, MONTH_TRAIN, MONTH_VAL, load_month_temporal_split
from dtl_agent.ml.rls.eval_metrics import (
    compare_dtl_with_ties,
    decide_all,
    ranking_metrics,
    regression_metrics,
    yield_tie_dtl_metrics,
)
from dtl_agent.ml.rls.features import RLS_FEATURE_NAMES, assert_no_forbidden_features, build_feature_matrix
from dtl_agent.ml.rls.jan_gru_shadow import (
    PRODUCTION_CKPT_REL,
    SHADOW_DIR_REL,
    SHADOW_CKPT_NAME,
    load_jan_shadow_scorer,
    score_examples_with_gru,
    train_jan_only_core_gru,
)
from dtl_agent.ml.rls.regressor import RLSRegressor

DEFAULT_LAMBDAS = (0.99, 0.995, 0.999, 1.0)
DEFAULT_DELTA = 10.0
OUT_REL = Path("artifacts/temporal/shared/rls_experiment/equal_info")


def _select_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_df: pd.DataFrame,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    lambdas: tuple[float, ...],
    delta: float,
) -> tuple[float, dict[str, Any]]:
    """Primary: maximize Feb top-1; secondary: minimize Feb MAE; require stable online Arm B."""
    records: dict[str, Any] = {}
    ranked: list[tuple[tuple[float, float, float], float]] = []
    for lam in lambdas:
        model = RLSRegressor(X_train.shape[1], forgetting_factor=lam, delta=delta)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        reg = regression_metrics(y_val, pred)
        scored = val_df.copy()
        scored["rls_score"] = pred
        rank = ranking_metrics(scored, score_col="rls_score")
        top1 = float(rank["top1_candidate_agreement"])
        mae = float(reg["mae"])

        online = RLSRegressor(X_train.shape[1], forgetting_factor=lam, delta=delta)
        online.fit(X_train, y_train)
        for i in range(len(X_val)):
            online.update(X_val[i], float(y_val[i]))
        mar_pred = online.predict(X_test)
        online_stable = bool(np.isfinite(mar_pred).all())

        records[str(lam)] = {
            "mae": mae,
            "top1": top1,
            "ranking": rank,
            "regression": reg,
            "online_arm_b_stable": online_stable,
        }
        stable_key = (0.0 if online_stable else 1.0, -top1, mae)
        ranked.append((stable_key, lam))

    ranked.sort(key=lambda t: t[0])
    best_l = ranked[0][1]
    return best_l, records


def _system_stats(model: RLSRegressor, X_sample: np.ndarray) -> dict[str, Any]:
    t0 = time.perf_counter()
    _ = model.predict(X_sample)
    dt = time.perf_counter() - t0
    n = max(len(X_sample), 1)
    tracemalloc.start()
    _ = model.predict(X_sample)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "model_nbytes": model.model_nbytes(),
        "model_size_kb": round(model.model_nbytes() / 1024.0, 3),
        "inference_latency_ms_per_row": 1000.0 * dt / n,
        "tracemalloc_peak_bytes_predict": int(peak),
        "n_updates": model.n_updates,
    }


def _arm_metrics(scored: pd.DataFrame, *, score_col: str) -> dict[str, Any]:
    y = scored["target_score"].to_numpy(dtype=float)
    p = scored[score_col].to_numpy(dtype=float)
    return {
        "regression": regression_metrics(y, p),
        "ranking": ranking_metrics(scored, score_col=score_col),
        "yield_tie_dtl": yield_tie_dtl_metrics(scored, score_col=score_col),
        "dtl_decisions_n": int(
            len(decide_all(scored, score_col=score_col))
        ),
    }


def run_equal_info_experiment(
    *,
    root: Path | None = None,
    retrain_jan_gru: bool = True,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    delta: float = DEFAULT_DELTA,
) -> dict[str, Any]:
    root = root or default_project_root()
    assert_no_forbidden_features()
    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    prod_ckpt = root / PRODUCTION_CKPT_REL
    prod_hash_before = file_sha256(prod_ckpt) if prod_ckpt.is_file() else None

    # --- Jan-only shadow GRU ---
    if retrain_jan_gru or not (root / SHADOW_DIR_REL / SHADOW_CKPT_NAME).is_file():
        gru_art = train_jan_only_core_gru(root=root)
        gru_train_meta = {
            "retrained": True,
            "best": gru_art.best,
            "runtime_seconds": gru_art.runtime_seconds,
            "production_untouched": gru_art.production_untouched,
        }
    else:
        gru_train_meta = {"retrained": False, "loaded_existing_shadow": True}

    data = load_month_temporal_split(root)
    seq_store = data.seq_store  # type: ignore[assignment]

    # Same January rows for RLS as GRU used
    X_tr, y_tr, ids_tr = build_feature_matrix(data.train, seq_store)  # type: ignore[arg-type]
    X_va, y_va, _ = build_feature_matrix(data.validation, seq_store)  # type: ignore[arg-type]
    X_te, y_te, _ = build_feature_matrix(data.test, seq_store)  # type: ignore[arg-type]

    # Verify example_id alignment contract (Jan train shared with GRU dataset)
    train_ids = data.train["example_id"].astype(str).tolist()
    if ids_tr != train_ids:
        raise RuntimeError("RLS feature example_id order mismatch vs Jan train frame")
    if len(ids_tr) != len(data.train):
        raise RuntimeError("RLS feature row count mismatch vs Jan train")

    best_lambda, lambda_records = _select_lambda(
        X_tr, y_tr, data.validation, X_va, y_va, X_te, lambdas, delta
    )

    # Arm A: frozen RLS after Jan fit
    rls_static = RLSRegressor(
        n_features=X_tr.shape[1], forgetting_factor=best_lambda, delta=delta
    )
    t_fit = time.perf_counter()
    rls_static.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t_fit
    pred_te_rls = rls_static.predict(X_te)

    # Score March with Jan-only GRU (frozen)
    gru_model, vocabs, device = load_jan_shadow_scorer(root=root)
    t_gru = time.perf_counter()
    pred_te_gru = score_examples_with_gru(
        examples=data.test,
        seq_store=seq_store,  # type: ignore[arg-type]
        model=gru_model,
        vocabs=vocabs,
        device=device,
    )
    gru_score_s = time.perf_counter() - t_gru

    mar = data.test.copy()
    mar["rls_score"] = pred_te_rls
    mar["gru_score"] = pred_te_gru

    arm_a_rls = _arm_metrics(mar, score_col="rls_score")
    arm_a_gru = _arm_metrics(mar, score_col="gru_score")
    arm_a_compare = compare_dtl_with_ties(
        mar,
        mar,
        score_col_a="gru_score",
        score_col_b="rls_score",
        a_name="gru",
        b_name="rls",
    )

    # Arm B: online update with February, GRU remains frozen
    rls_online = RLSRegressor(
        n_features=X_tr.shape[1], forgetting_factor=best_lambda, delta=delta
    )
    rls_online.fit(X_tr, y_tr)
    before_mar = rls_online.predict(X_te)
    before_metrics = _arm_metrics(mar.assign(rls_score=before_mar), score_col="rls_score")

    # Deterministic Feb updates by example_id order (loader sorts validation frame)
    val_ids = data.validation["example_id"].astype(str).tolist()
    if not val_ids == sorted(val_ids):
        raise RuntimeError("February validation frame must be sorted by example_id for online updates")
    t_up = time.perf_counter()
    for i in range(len(X_va)):
        rls_online.update(X_va[i], float(y_va[i]))
    online_s = time.perf_counter() - t_up

    after_mar = rls_online.predict(X_te)
    mar_after = mar.copy()
    mar_after["rls_score"] = after_mar
    after_metrics = _arm_metrics(mar_after, score_col="rls_score")
    arm_b_compare = compare_dtl_with_ties(
        mar,  # GRU static scores
        mar_after,
        score_col_a="gru_score",
        score_col_b="rls_score",
        a_name="gru_frozen",
        b_name="rls_online",
    )
    static_vs_online = compare_dtl_with_ties(
        mar.assign(rls_score=before_mar),
        mar_after,
        score_col_a="rls_score",
        score_col_b="rls_score",
        a_name="rls_static",
        b_name="rls_online",
    )

    # Persist models / tables
    rls_static.save(out_dir / "rls_jan_static.json")
    rls_online.save(out_dir / "rls_jan_plus_feb_online.json")
    mar.to_parquet(out_dir / "march_scores_arm_a.parquet", index=False)
    mar_after.to_parquet(out_dir / "march_scores_arm_b.parquet", index=False)
    decide_all(mar, score_col="gru_score").to_csv(out_dir / "dtl_gru_arm_a.csv", index=False)
    decide_all(mar, score_col="rls_score").to_csv(out_dir / "dtl_rls_arm_a.csv", index=False)
    decide_all(mar_after, score_col="rls_score").to_csv(out_dir / "dtl_rls_arm_b.csv", index=False)

    prod_hash_after = file_sha256(prod_ckpt) if prod_ckpt.is_file() else None
    production_untouched = (
        prod_hash_before is not None
        and prod_hash_after is not None
        and prod_hash_before == prod_hash_after
    )

    summary: dict[str, Any] = {
        "experiment": "equal_info_rls_vs_jan_gru",
        "production_path_unchanged": True,
        "production_core_gru_temporal_v1_untouched": production_untouched,
        "decision_hierarchy": [
            "simulated_yield / safety",
            "selected DTL limit (esp. yield ties)",
            "candidate ranking",
            "predictive MAE/RMSE/bias",
            "latency / memory",
        ],
        "split": {
            "train_month": MONTH_TRAIN,
            "validation_month": MONTH_VAL,
            "test_month": MONTH_TEST,
            "train_rows": int(len(data.train)),
            "validation_rows": int(len(data.validation)),
            "test_rows": int(len(data.test)),
            "same_jan_rows_for_gru_and_rls": True,
            "note": data.split_note,
        },
        "target": {
            "name": "target_score",
            "definition": "objective_score from month-scoped Core simulation",
        },
        "features_rls": {"names": list(RLS_FEATURE_NAMES), "n": len(RLS_FEATURE_NAMES)},
        "online_label_realism": (
            "RLS online updates use February simulated target_score/objective_score "
            "after month simulation artifacts exist — same label family as GRU training, "
            "not live ATE PASS/FAIL."
        ),
        "hyperparameters": {
            "forgetting_factor": best_lambda,
            "delta": delta,
            "lambda_selection": {
                "rule": "maximize Feb top-1, then minimize Feb MAE",
                "records": lambda_records,
            },
        },
        "jan_gru_shadow": gru_train_meta,
        "arm_a_static_equal_info": {
            "description": (
                "Both models trained on January only; Feb used for GRU early-stop / "
                "RLS λ tune; March evaluation with both frozen. No Feb online updates."
            ),
            "gru": arm_a_gru,
            "rls": arm_a_rls,
            "dtl_compare_gru_vs_rls": arm_a_compare,
            "score_correlation": float(
                np.corrcoef(mar["gru_score"], mar["rls_score"])[0, 1]
            ),
        },
        "arm_b_sequential_rls": {
            "description": (
                "RLS starts from Jan fit (locked λ), online-updates with February "
                "target_score in example_id order; March re-scored. Jan-GRU remains frozen."
            ),
            "before_feb_updates_on_march": before_metrics,
            "after_feb_updates_on_march": after_metrics,
            "mae_improvement": float(
                before_metrics["regression"]["mae"] - after_metrics["regression"]["mae"]
            ),
            "top1_improvement": float(
                after_metrics["ranking"]["top1_candidate_agreement"]
                - before_metrics["ranking"]["top1_candidate_agreement"]
            ),
            "dtl_compare_frozen_gru_vs_online_rls": arm_b_compare,
            "dtl_compare_static_rls_vs_online_rls": static_vs_online,
            "online_update_seconds": online_s,
            "n_online_updates": int(len(X_va)),
        },
        "system": {
            "rls_static": _system_stats(rls_static, X_te[: min(1000, len(X_te))]),
            "rls_fit_seconds": fit_s,
            "gru_march_score_seconds": gru_score_s,
            "gru_shadow_checkpoint": str((SHADOW_DIR_REL / SHADOW_CKPT_NAME).as_posix()),
            "gru_shadow_checkpoint_bytes": int(
                (root / SHADOW_DIR_REL / SHADOW_CKPT_NAME).stat().st_size
            )
            if (root / SHADOW_DIR_REL / SHADOW_CKPT_NAME).is_file()
            else None,
        },
        "artifacts": {
            "output_dir": str(OUT_REL.as_posix()),
        },
        "limitations": [
            "Core parameters only (ir_drop, thermal).",
            "RLS uses sequence aggregates, not full 200-step tensors.",
            "Offline DTL mirror of Top-N + yield-first policy; not production pipeline.py.",
            "Month-temporal protocol differs from production lot-level split.",
        ],
    }

    write_json(out_dir / "EQUAL_INFO_EXPERIMENT_SUMMARY.json", summary)
    return summary


def main() -> None:
    summary = run_equal_info_experiment()
    # Print hierarchy-first digest
    a = summary["arm_a_static_equal_info"]
    digest = {
        "production_untouched": summary["production_core_gru_temporal_v1_untouched"],
        "lambda": summary["hyperparameters"]["forgetting_factor"],
        "arm_a_yield": {
            "gru": a["gru"]["yield_tie_dtl"]["mean_selected_yield"],
            "rls": a["rls"]["yield_tie_dtl"]["mean_selected_yield"],
        },
        "arm_a_oracle_limit_agreement": {
            "gru": a["gru"]["yield_tie_dtl"]["oracle_limit_agreement"],
            "rls": a["rls"]["yield_tie_dtl"]["oracle_limit_agreement"],
            "among_ties_gru": a["gru"]["yield_tie_dtl"]["oracle_limit_agreement_among_yield_ties"],
            "among_ties_rls": a["rls"]["yield_tie_dtl"]["oracle_limit_agreement_among_yield_ties"],
        },
        "arm_a_limit_agreement": a["dtl_compare_gru_vs_rls"]["recommended_limit_agreement"],
        "arm_a_ranking_top1": {
            "gru": a["gru"]["ranking"]["top1_candidate_agreement"],
            "rls": a["rls"]["ranking"]["top1_candidate_agreement"],
        },
        "arm_a_mae": {
            "gru": a["gru"]["regression"]["mae"],
            "rls": a["rls"]["regression"]["mae"],
        },
        "arm_b_mae_improvement": summary["arm_b_sequential_rls"]["mae_improvement"],
        "arm_b_top1_improvement": summary["arm_b_sequential_rls"]["top1_improvement"],
    }
    print(json.dumps(digest, indent=2))
    print("Full summary:", (default_project_root() / OUT_REL / "EQUAL_INFO_EXPERIMENT_SUMMARY.json"))


if __name__ == "__main__":
    main()
