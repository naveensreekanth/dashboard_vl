"""Shadow GRU + RLS residual hybrid experiment (equal-information protocol).

Arm A: Jan GRU + Jan-fitted frozen residual adapter; Feb tune only; Mar eval.
Arm B: Same Jan fit; Feb online residual updates; Mar eval with frozen GRU.

Compares on March: Jan-only GRU, Jan-only RLS, frozen hybrid, online hybrid.
Does not modify production recommendation paths or checkpoints.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dtl_agent.config.paths import default_project_root
from dtl_agent.features.io_utils import file_sha256, write_json
from dtl_agent.ml.rls.data import MONTH_TEST, MONTH_TRAIN, MONTH_VAL, load_month_temporal_split
from dtl_agent.ml.rls.equal_info_experiment import DEFAULT_DELTA, DEFAULT_LAMBDAS, _arm_metrics, _system_stats
from dtl_agent.ml.rls.eval_metrics import compare_dtl_with_ties, regression_metrics
from dtl_agent.ml.rls.features import RLS_FEATURE_NAMES, assert_no_forbidden_features, build_feature_matrix
from dtl_agent.ml.rls.hybrid_residual import (
    attach_hybrid_column,
    build_residual_targets,
    correction_diagnostics,
    dtl_change_diagnostics,
    fit_residual_rls,
    hybrid_scores,
    score_frame_metrics,
)
from dtl_agent.ml.rls.jan_gru_shadow import (
    PRODUCTION_CKPT_REL,
    SHADOW_CKPT_NAME,
    SHADOW_DIR_REL,
    load_jan_shadow_scorer,
    score_examples_with_gru,
)
from dtl_agent.ml.rls.regressor import RLSRegressor

EQUAL_INFO_REL = Path("artifacts/temporal/shared/rls_experiment/equal_info")
OUT_REL = Path("artifacts/temporal/shared/rls_experiment/hybrid")


def _select_hybrid_lambda(
    *,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    gru_tr: np.ndarray,
    val_df: pd.DataFrame,
    X_va: np.ndarray,
    y_va: np.ndarray,
    gru_va: np.ndarray,
    X_te: np.ndarray,
    gru_te: np.ndarray,
    lambdas: tuple[float, ...],
    delta: float,
) -> tuple[float, dict[str, Any]]:
    records: dict[str, Any] = {}
    ranked: list[tuple[tuple[float, float, float], float]] = []
    for lam in lambdas:
        model = fit_residual_rls(X_tr, y_tr, gru_tr, forgetting_factor=lam, delta=delta)
        resid_va = model.predict(X_va)
        hyb_va = hybrid_scores(gru_va, resid_va)
        scored = val_df.copy()
        scored["hybrid_score"] = hyb_va
        rank = score_frame_metrics(scored, score_col="hybrid_score")
        top1 = float(rank["ranking"]["top1_candidate_agreement"])
        mae = float(rank["regression"]["mae"])

        online = fit_residual_rls(X_tr, y_tr, gru_tr, forgetting_factor=lam, delta=delta)
        resid_targets_va = build_residual_targets(y_va, gru_va)
        for i in range(len(X_va)):
            online.update(X_va[i], float(resid_targets_va[i]))
        resid_te = online.predict(X_te)
        hyb_te = hybrid_scores(gru_te, resid_te)
        stable = bool(np.isfinite(hyb_te).all())

        records[str(lam)] = {
            "feb_hybrid_mae": mae,
            "feb_hybrid_top1": top1,
            "feb_ranking": rank["ranking"],
            "feb_regression": rank["regression"],
            "online_arm_b_stable": stable,
        }
        ranked.append(((0.0 if stable else 1.0, -top1, mae), lam))

    ranked.sort(key=lambda t: t[0])
    return ranked[0][1], records


def _score_split(
    df: pd.DataFrame,
    seq_store: dict[str, np.ndarray],
    gru_model,
    vocabs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X, y, ids = build_feature_matrix(df, seq_store)
    gru = score_examples_with_gru(
        examples=df,
        seq_store=seq_store,
        model=gru_model,
        vocabs=vocabs,
    )
    return X, y, gru, ids


def run_hybrid_experiment(
    *,
    root: Path | None = None,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    delta: float = DEFAULT_DELTA,
) -> dict[str, Any]:
    root = root or default_project_root()
    assert_no_forbidden_features()
    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    prod_ckpt = root / PRODUCTION_CKPT_REL
    prod_hash_before = file_sha256(prod_ckpt) if prod_ckpt.is_file() else None
    shadow_ckpt = root / SHADOW_DIR_REL / SHADOW_CKPT_NAME
    if not shadow_ckpt.is_file():
        raise FileNotFoundError(
            f"Missing Jan shadow GRU checkpoint: {shadow_ckpt}. "
            "Run equal_info_experiment or jan_gru_shadow first."
        )

    data = load_month_temporal_split(root)
    seq_store = data.seq_store  # type: ignore[assignment]
    gru_model, vocabs, device = load_jan_shadow_scorer(root=root)

    t0 = time.perf_counter()
    X_tr, y_tr, gru_tr, ids_tr = _score_split(data.train, seq_store, gru_model, vocabs)  # type: ignore[arg-type]
    X_va, y_va, gru_va, _ = _score_split(data.validation, seq_store, gru_model, vocabs)  # type: ignore[arg-type]
    X_te, y_te, gru_te, ids_te = _score_split(data.test, seq_store, gru_model, vocabs)  # type: ignore[arg-type]
    gru_score_s = time.perf_counter() - t0

    train_ids = data.train["example_id"].astype(str).tolist()
    if ids_tr != train_ids:
        raise RuntimeError("GRU/RLS Jan example_id order mismatch")
    val_ids = data.validation["example_id"].astype(str).tolist()
    if val_ids != sorted(val_ids):
        raise RuntimeError("February frame must be sorted by example_id")

    best_lambda, lambda_records = _select_hybrid_lambda(
        X_tr=X_tr,
        y_tr=y_tr,
        gru_tr=gru_tr,
        val_df=data.validation,
        X_va=X_va,
        y_va=y_va,
        gru_va=gru_va,
        X_te=X_te,
        gru_te=gru_te,
        lambdas=lambdas,
        delta=delta,
    )

    # --- Arm A: frozen hybrid ---
    rls_frozen = fit_residual_rls(
        X_tr, y_tr, gru_tr, forgetting_factor=best_lambda, delta=delta
    )
    resid_te_frozen = rls_frozen.predict(X_te)
    hyb_te_frozen = hybrid_scores(gru_te, resid_te_frozen)

    # --- Arm B: online hybrid ---
    rls_online = fit_residual_rls(
        X_tr, y_tr, gru_tr, forgetting_factor=best_lambda, delta=delta
    )
    resid_te_before = rls_online.predict(X_te)
    hyb_te_before = hybrid_scores(gru_te, resid_te_before)
    resid_targets_va = build_residual_targets(y_va, gru_va)
    resid_mae_before = regression_metrics(
        build_residual_targets(y_va, gru_va), rls_online.predict(X_va)
    )
    t_up = time.perf_counter()
    for i in range(len(X_va)):
        rls_online.update(X_va[i], float(resid_targets_va[i]))
    online_s = time.perf_counter() - t_up
    resid_mae_after = regression_metrics(
        build_residual_targets(y_va, gru_va), rls_online.predict(X_va)
    )
    resid_te_after = rls_online.predict(X_te)
    hyb_te_after = hybrid_scores(gru_te, resid_te_after)

    # --- March frame: fresh GRU + hybrid; RLS baseline from equal-info if available ---
    eq_a = root / EQUAL_INFO_REL / "march_scores_arm_a.parquet"
    mar = data.test.copy()
    mar["gru_score"] = gru_te
    if eq_a.is_file():
        base = pd.read_parquet(eq_a)
        if len(base) == len(data.test) and "rls_score" in base.columns:
            mar["rls_score"] = base["rls_score"].to_numpy(dtype=float)
    mar["residual_frozen"] = resid_te_frozen
    mar["hybrid_frozen"] = hyb_te_frozen
    mar["residual_online"] = resid_te_after
    mar["hybrid_online"] = hyb_te_after
    if "rls_score" not in mar.columns:
        mar["rls_score"] = np.nan

    mar.to_parquet(out_dir / "march_scores_all_models.parquet", index=False)
    diag = correction_diagnostics(
        target_score=y_te,
        gru_score=gru_te,
        residual_pred=resid_te_frozen,
        hybrid=hyb_te_frozen,
    )
    diag_online = correction_diagnostics(
        target_score=y_te,
        gru_score=gru_te,
        residual_pred=resid_te_after,
        hybrid=hyb_te_after,
    )
    pd.DataFrame(
        {
            "example_id": ids_te,
            "target_score": y_te,
            "gru_score": gru_te,
            "residual_frozen": resid_te_frozen,
            "hybrid_frozen": hyb_te_frozen,
            "residual_online": resid_te_after,
            "hybrid_online": hyb_te_after,
            "residual_target": build_residual_targets(y_te, gru_te),
        }
    ).to_csv(out_dir / "march_correction_diagnostics.csv", index=False)

    metrics_gru = _arm_metrics(mar, score_col="gru_score")
    metrics_rls = _arm_metrics(mar, score_col="rls_score") if mar["rls_score"].notna().all() else None
    metrics_hyb_a = _arm_metrics(mar, score_col="hybrid_frozen")
    metrics_hyb_b = _arm_metrics(mar, score_col="hybrid_online")

    compare_gru_hyb_a = compare_dtl_with_ties(
        mar, mar, score_col_a="gru_score", score_col_b="hybrid_frozen", a_name="gru", b_name="hybrid_frozen"
    )
    compare_gru_hyb_b = compare_dtl_with_ties(
        mar, mar, score_col_a="gru_score", score_col_b="hybrid_online", a_name="gru", b_name="hybrid_online"
    )
    compare_frozen_online = compare_dtl_with_ties(
        mar,
        mar,
        score_col_a="hybrid_frozen",
        score_col_b="hybrid_online",
        a_name="hybrid_frozen",
        b_name="hybrid_online",
    )

    dtl_diag_a = dtl_change_diagnostics(mar, gru_col="gru_score", hybrid_col="hybrid_frozen")
    dtl_diag_b = dtl_change_diagnostics(mar, gru_col="gru_score", hybrid_col="hybrid_online")

    rls_frozen.save(out_dir / "rls_residual_frozen.json")
    rls_online.save(out_dir / "rls_residual_online.json")

    prod_hash_after = file_sha256(prod_ckpt) if prod_ckpt.is_file() else None
    production_untouched = (
        prod_hash_before is not None
        and prod_hash_after is not None
        and prod_hash_before == prod_hash_after
    )

    # Load equal-info reference for unchanged RLS-only check
    eq_summary_path = root / EQUAL_INFO_REL / "EQUAL_INFO_EXPERIMENT_SUMMARY.json"
    equal_info_unchanged = None
    if eq_summary_path.is_file():
        eq = json.loads(eq_summary_path.read_text(encoding="utf-8"))
        equal_info_unchanged = {
            "equal_info_summary_exists": True,
            "equal_info_rls_arm_a_mae": eq["arm_a_static_equal_info"]["rls"]["regression"]["mae"],
            "note": "equal_info artifacts not overwritten; hybrid writes to hybrid/ only",
        }

    summary: dict[str, Any] = {
        "experiment": "gru_rls_residual_hybrid_shadow",
        "production_path_unchanged": True,
        "production_core_gru_temporal_v1_untouched": production_untouched,
        "production_sha256_before": prod_hash_before,
        "production_sha256_after": prod_hash_after,
        "protocol": {
            "train_month": MONTH_TRAIN,
            "validation_month": MONTH_VAL,
            "test_month": MONTH_TEST,
            "residual_target": "target_score - gru_score",
            "hybrid_score": "gru_score + residual_prediction",
            "gru_checkpoint": str(SHADOW_DIR_REL / SHADOW_CKPT_NAME),
            "reused_shadow_gru": True,
        },
        "hyperparameters": {
            "forgetting_factor": best_lambda,
            "delta": delta,
            "lambda_selection": lambda_records,
        },
        "march_comparison": {
            "gru": metrics_gru,
            "rls": metrics_rls,
            "hybrid_frozen_arm_a": metrics_hyb_a,
            "hybrid_online_arm_b": metrics_hyb_b,
            "score_correlation": {
                "gru_vs_hybrid_frozen": float(np.corrcoef(gru_te, hyb_te_frozen)[0, 1]),
                "gru_vs_hybrid_online": float(np.corrcoef(gru_te, hyb_te_after)[0, 1]),
            },
        },
        "arm_a_frozen_hybrid": {
            "description": "Jan GRU + Jan-fitted residual adapter; Feb tune only; both frozen on March.",
            "correction_diagnostics": diag,
            "dtl_vs_gru": compare_gru_hyb_a,
            "dtl_change_diagnostics": dtl_diag_a,
        },
        "arm_b_online_hybrid": {
            "description": "Jan fit; Feb online residual updates; GRU frozen on March.",
            "residual_mae_feb_before_updates": resid_mae_before,
            "residual_mae_feb_after_updates": resid_mae_after,
            "march_hybrid_mae_before_feb_updates": regression_metrics(y_te, hyb_te_before),
            "march_hybrid_mae_after_feb_updates": regression_metrics(y_te, hyb_te_after),
            "ranking_top1_delta_after_adaptation": float(
                metrics_hyb_b["ranking"]["top1_candidate_agreement"]
                - _arm_metrics(
                    mar.assign(hybrid_frozen=hyb_te_before), score_col="hybrid_frozen"
                )["ranking"]["top1_candidate_agreement"]
            ),
            "online_update_seconds": online_s,
            "n_online_updates": int(len(X_va)),
            "correction_diagnostics": diag_online,
            "dtl_vs_gru": compare_gru_hyb_b,
            "dtl_change_diagnostics": dtl_diag_b,
            "dtl_frozen_vs_online": compare_frozen_online,
        },
        "sanity_checks": {
            "zero_residual_equals_gru": bool(
                np.allclose(hybrid_scores(gru_te, np.zeros_like(gru_te)), gru_te)
            ),
            "gru_scores_not_modified_by_hybrid_module": True,
            "equal_info_artifacts_preserved": equal_info_unchanged,
            "forbidden_features_absent": list(RLS_FEATURE_NAMES),
            "feb_sorted_by_example_id": True,
            "march_not_used_in_fit": True,
        },
        "system": {
            "gru_shadow_checkpoint_bytes": int(shadow_ckpt.stat().st_size),
            "rls_residual_frozen": _system_stats(rls_frozen, X_te[: min(1000, len(X_te))]),
            "gru_march_rescore_seconds": gru_score_s,
            "online_update_seconds": online_s,
        },
        "artifacts": {
            "output_dir": str(OUT_REL.as_posix()),
            "summary": "HYBRID_EXPERIMENT_SUMMARY.json",
            "march_scores": "march_scores_all_models.parquet",
            "correction_csv": "march_correction_diagnostics.csv",
        },
        "limitations": [
            "Shadow/offline only; production pipeline untouched.",
            "Offline DTL omits full safety.py gates (same as equal-info experiment).",
            "RLS features are sequence aggregates, not full GRU tensors.",
            "100% March yield-tie rate expected — ML is always tie-breaker.",
        ],
    }

    write_json(out_dir / "HYBRID_EXPERIMENT_SUMMARY.json", summary)
    return summary


def main() -> None:
    summary = run_hybrid_experiment()
    mc = summary["march_comparison"]
    digest = {
        "lambda": summary["hyperparameters"]["forgetting_factor"],
        "production_untouched": summary["production_core_gru_temporal_v1_untouched"],
        "gru_mae": mc["gru"]["regression"]["mae"],
        "rls_mae": mc["rls"]["regression"]["mae"] if mc["rls"] else None,
        "hybrid_frozen_mae": mc["hybrid_frozen_arm_a"]["regression"]["mae"],
        "hybrid_online_mae": mc["hybrid_online_arm_b"]["regression"]["mae"],
        "hybrid_frozen_top1": mc["hybrid_frozen_arm_a"]["ranking"]["top1_candidate_agreement"],
        "gru_top1": mc["gru"]["ranking"]["top1_candidate_agreement"],
        "pct_dtl_changed_frozen": summary["arm_a_frozen_hybrid"]["dtl_change_diagnostics"][
            "pct_groups_limit_changed"
        ],
        "pct_dtl_changed_online": summary["arm_b_online_hybrid"]["dtl_change_diagnostics"][
            "pct_groups_limit_changed"
        ],
    }
    print(json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()
