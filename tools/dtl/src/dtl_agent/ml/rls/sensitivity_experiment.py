"""Optimized all-month frozen + alpha sensitivity experiment (shadow only).

Computational reuse:
  - load month frames once
  - GRU score once per month (cached to disk when possible)
  - RLS residual once per month
  - shared DTL group pack once per month
  - alpha loop only recomputes score-dependent ranking/DTL

Primary analysis: March held-out alpha sensitivity.
Jan/Feb: diagnostic Part A only (not used to tune alpha).

Does not modify production recommendation paths.
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
from dtl_agent.ml.rls.fast_dtl import (
    build_shared_group_pack,
    decide_all_fast,
    evaluate_scores,
)
from dtl_agent.ml.rls.features import assert_no_forbidden_features, build_feature_matrix
from dtl_agent.ml.rls.hybrid_residual import fit_residual_rls, scaled_hybrid_scores
from dtl_agent.ml.rls.jan_gru_shadow import (
    PRODUCTION_CKPT_REL,
    SHADOW_CKPT_NAME,
    SHADOW_DIR_REL,
    load_jan_shadow_scorer,
    score_examples_with_gru,
)
from dtl_agent.ml.rls.regressor import RLSRegressor

OUT_REL = Path("artifacts/temporal/shared/rls_experiment/hybrid_sensitivity")
EQUAL_INFO_REL = Path("artifacts/temporal/shared/rls_experiment/equal_info")
HYBRID_REL = Path("artifacts/temporal/shared/rls_experiment/hybrid")
ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)
DEFAULT_DELTA = 10.0
DEFAULT_RESIDUAL_LAMBDA = 1.0
DEFAULT_RLS_LAMBDA = 0.999

MONTH_ROLE = {
    MONTH_TRAIN: "training_data_diagnostic",
    MONTH_VAL: "validation_tuning_diagnostic",
    MONTH_TEST: "primary_held_out_test",
}


def _safe_alpha_verdict(
    row: dict[str, Any], *, gru_mae: float, gru_top1: float, gru_oracle: float
) -> dict[str, Any]:
    reasons: list[str] = []
    yield_ok = float(row["selected_yield_delta_vs_oracle"]) >= -1e-9
    if not yield_ok:
        reasons.append("selected_yield_regressed")

    dtl_ok = (
        float(row["pct_groups_limit_changed"]) <= 1e-12
        and float(row["recommended_limit_agreement_vs_gru"]) >= 1.0 - 1e-12
        and float(row["over_limit_rate_vs_gru"]) <= 1e-12
    )
    if not dtl_ok:
        reasons.append("dtl_limit_changed_or_systematically_shifted")

    oracle_ok = float(row["oracle_tie_break_win_rate"]) + 1e-9 >= gru_oracle - 0.005
    if not oracle_ok:
        reasons.append("oracle_tie_break_worse")

    ranking_ok = float(row["top1"]) + 1e-9 >= gru_top1 - 0.01
    if not ranking_ok:
        reasons.append("top1_materially_worse")

    pred_ok = float(row["mae"]) <= gru_mae * 1.02 + 1e-9
    if not pred_ok:
        reasons.append("mae_materially_worse")

    return {
        "safe": bool(yield_ok and dtl_ok and oracle_ok and ranking_ok and pred_ok),
        "reasons_if_unsafe": reasons,
        "checks": {
            "yield_ok": yield_ok,
            "dtl_ok": dtl_ok,
            "oracle_ok": oracle_ok,
            "ranking_ok": ranking_ok,
            "prediction_ok": pred_ok,
        },
    }


def _row_from_eval(month: str, model: str, alpha: float | None, ev: dict[str, Any]) -> dict[str, Any]:
    cmp = ev["dtl_vs_gru"]
    return {
        "month": month,
        "role": MONTH_ROLE[month],
        "model": model,
        "alpha": alpha,
        "mae": ev["regression"]["mae"],
        "rmse": ev["regression"]["rmse"],
        "bias": ev["regression"]["bias"],
        "score_correlation_with_gru": ev["score_correlation_with_gru"],
        "top1": ev["ranking"]["top1_candidate_agreement"],
        "top5": ev["ranking"]["topk_candidate_overlap"],
        "spearman": ev["ranking"]["mean_spearman"],
        "recommended_limit_agreement_vs_gru": cmp["recommended_limit_agreement"],
        "mean_abs_dtl_difference": cmp["mean_abs_dtl_difference"],
        "max_abs_limit_delta": cmp["max_abs_limit_delta"],
        "over_limit_rate_vs_gru": cmp["over_limit_rate"],
        "under_limit_rate_vs_gru": cmp["under_limit_rate"],
        "decision_agreement": cmp["decision_agreement"],
        "mean_selected_yield": ev["mean_selected_yield"],
        "selected_yield_delta_vs_oracle": ev["selected_yield_delta_vs_oracle"],
        "oracle_limit_agreement": ev["oracle_limit_agreement"],
        "yield_tie_rate": ev["yield_tie_rate"],
        "oracle_tie_break_win_rate": ev["oracle_tie_break_win_rate"],
        "tie_break_peer_agreement": cmp["recommended_limit_agreement"],
        "pct_groups_limit_changed": cmp["pct_groups_limit_changed"],
        "pct_changed_looser": cmp["pct_changed_looser"],
        "pct_changed_tighter": cmp["pct_changed_tighter"],
        "pct_changed_toward_oracle": cmp["pct_changed_toward_oracle"],
        "pct_changed_away_from_oracle": cmp["pct_changed_away_from_oracle"],
        "mean_abs_limit_delta_when_changed": cmp["mean_abs_limit_delta_when_changed"],
        "n_groups_limit_changed": cmp["n_groups_limit_changed"],
    }


def run_sensitivity_experiment(
    *,
    root: Path | None = None,
    alphas: tuple[float, ...] = ALPHAS,
    residual_lambda: float = DEFAULT_RESIDUAL_LAMBDA,
    rls_lambda: float = DEFAULT_RLS_LAMBDA,
    delta: float = DEFAULT_DELTA,
) -> dict[str, Any]:
    root = root or default_project_root()
    assert_no_forbidden_features()
    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    prod_ckpt = root / PRODUCTION_CKPT_REL
    prod_before = file_sha256(prod_ckpt) if prod_ckpt.is_file() else None
    shadow = root / SHADOW_DIR_REL / SHADOW_CKPT_NAME
    if not shadow.is_file():
        raise FileNotFoundError(f"Missing Jan shadow GRU: {shadow}")

    # ---- STEP 1: load once ----
    t0 = time.perf_counter()
    data = load_month_temporal_split(root)
    seq_store = data.seq_store  # type: ignore[assignment]
    timings["data_load_s"] = time.perf_counter() - t0

    months = {
        MONTH_TRAIN: data.train.reset_index(drop=True),
        MONTH_VAL: data.validation.reset_index(drop=True),
        MONTH_TEST: data.test.reset_index(drop=True),
    }

    gru_model, vocabs, _ = load_jan_shadow_scorer(root=root)

    # ---- STEP 2+3: score GRU / residual / RLS once per month ----
    cache_path = out_dir / "cached_month_scores.npz"
    month_cache: dict[str, dict[str, Any]] = {}

    t0 = time.perf_counter()
    # Prefer reusing prior Part A parquet for March if present and aligned
    part_a_path = out_dir / "march_scores_part_a.parquet"
    hybrid_mar = root / HYBRID_REL / "march_scores_all_models.parquet"

    for month, df in months.items():
        X, y, ids = build_feature_matrix(df, seq_store)  # type: ignore[arg-type]
        month_cache[month] = {"df": df, "X": X, "y": y, "ids": ids}

    # Load residual + rls-only models
    X_tr = month_cache[MONTH_TRAIN]["X"]
    y_tr = month_cache[MONTH_TRAIN]["y"]
    residual_path = root / HYBRID_REL / "rls_residual_frozen.json"
    reused_residual = out_dir / "rls_residual_frozen_reused.json"
    if reused_residual.is_file():
        residual_rls = RLSRegressor.load(reused_residual)
    elif residual_path.is_file():
        residual_rls = RLSRegressor.load(residual_path)
        residual_rls.save(reused_residual)
    else:
        # Need GRU train scores first — filled below
        residual_rls = None

    rls_path = root / EQUAL_INFO_REL / "rls_jan_static.json"
    if rls_path.is_file():
        rls_only = RLSRegressor.load(rls_path)
    else:
        rls_only = None

    timings["model_load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for month, pack in month_cache.items():
        df = pack["df"]
        X = pack["X"]
        y = pack["y"]

        # Try cache files
        gru = None
        if month == MONTH_TEST and part_a_path.is_file():
            prev = pd.read_parquet(part_a_path)
            if len(prev) == len(df) and "gru_score" in prev.columns:
                gru = prev["gru_score"].to_numpy(dtype=float)
                if "residual_pred" in prev.columns:
                    pack["resid"] = prev["residual_pred"].to_numpy(dtype=float)
                if "rls_score" in prev.columns:
                    pack["rls"] = prev["rls_score"].to_numpy(dtype=float)
        if gru is None and month == MONTH_TEST and hybrid_mar.is_file():
            prev = pd.read_parquet(hybrid_mar)
            if len(prev) == len(df) and "gru_score" in prev.columns:
                gru = prev["gru_score"].to_numpy(dtype=float)
                if "residual_frozen" in prev.columns:
                    pack["resid"] = prev["residual_frozen"].to_numpy(dtype=float)
                if "rls_score" in prev.columns:
                    pack["rls"] = prev["rls_score"].to_numpy(dtype=float)

        if gru is None:
            gru = score_examples_with_gru(
                examples=df, seq_store=seq_store, model=gru_model, vocabs=vocabs  # type: ignore[arg-type]
            )
        pack["gru"] = gru

        if residual_rls is None:
            residual_rls = fit_residual_rls(
                month_cache[MONTH_TRAIN]["X"],
                month_cache[MONTH_TRAIN]["y"],
                month_cache[MONTH_TRAIN]["gru"],
                forgetting_factor=residual_lambda,
                delta=delta,
            )
            residual_rls.save(reused_residual)

        if "resid" not in pack:
            pack["resid"] = residual_rls.predict(X)
        if rls_only is None:
            rls_only = RLSRegressor(X_tr.shape[1], forgetting_factor=rls_lambda, delta=delta)
            rls_only.fit(X_tr, y_tr)
        if "rls" not in pack:
            pack["rls"] = rls_only.predict(X)

        # Shared DTL pack once
        pack["shared"] = build_shared_group_pack(df)
        pack["gru_dec"] = decide_all_fast(pack["shared"], pack["gru"])

    timings["gru_residual_shared_dtl_s"] = time.perf_counter() - t0

    # Persist March Part A scores without overwriting if already present — update/ensure
    mar = months[MONTH_TEST].copy()
    mar["gru_score"] = month_cache[MONTH_TEST]["gru"]
    mar["residual_pred"] = month_cache[MONTH_TEST]["resid"]
    mar["rls_score"] = month_cache[MONTH_TEST]["rls"]
    mar["hybrid_alpha_1"] = scaled_hybrid_scores(
        month_cache[MONTH_TEST]["gru"], month_cache[MONTH_TEST]["resid"], alpha=1.0
    )
    if not part_a_path.is_file():
        mar.to_parquet(part_a_path, index=False)
    else:
        # Keep original part_a; write companion cache
        mar.to_parquet(out_dir / "march_scores_cached.parquet", index=False)

    np.savez_compressed(
        cache_path,
        **{
            f"{m}_gru": month_cache[m]["gru"]
            for m in months
        },
        **{
            f"{m}_resid": month_cache[m]["resid"]
            for m in months
        },
        **{
            f"{m}_rls": month_cache[m]["rls"]
            for m in months
        },
    )

    # ---- Part A: all-month frozen (3 models) ----
    t0 = time.perf_counter()
    part_a_rows: list[dict[str, Any]] = []
    part_a: dict[str, Any] = {}
    for month in months:
        pack = month_cache[month]
        y = pack["y"]
        gru = pack["gru"]
        shared = pack["shared"]
        gru_dec = pack["gru_dec"]
        models = {
            "gru": gru,
            "rls": pack["rls"],
            "hybrid_frozen_alpha_1": scaled_hybrid_scores(gru, pack["resid"], alpha=1.0),
        }
        month_out: dict[str, Any] = {"role": MONTH_ROLE[month], "n_rows": int(len(y))}
        for name, scores in models.items():
            ev = evaluate_scores(
                shared, scores=scores, targets=y, gru_scores=gru, gru_dec=gru_dec
            )
            month_out[name] = {
                "regression": ev["regression"],
                "ranking": ev["ranking"],
                "score_correlation_with_gru": ev["score_correlation_with_gru"],
                "yield_tie_rate": ev["yield_tie_rate"],
                "oracle_tie_break_win_rate": ev["oracle_tie_break_win_rate"],
                "dtl_vs_gru": ev["dtl_vs_gru"],
            }
            part_a_rows.append(_row_from_eval(month, name, 1.0 if name.startswith("hybrid") else None, ev))
        part_a[month] = month_out
    timings["part_a_metrics_s"] = time.perf_counter() - t0

    # ---- Part B: March-primary alpha loop (cheap) ----
    t0 = time.perf_counter()
    pack = month_cache[MONTH_TEST]
    y = pack["y"]
    gru = pack["gru"]
    resid = pack["resid"]
    shared = pack["shared"]
    gru_dec = pack["gru_dec"]

    # Sanity: alpha 0 / 1
    h0 = scaled_hybrid_scores(gru, resid, alpha=0.0)
    h1 = scaled_hybrid_scores(gru, resid, alpha=1.0)
    if not np.allclose(h0, gru):
        raise RuntimeError("alpha=0 hybrid_score must equal gru_score")

    alpha_rows: list[dict[str, Any]] = []
    alpha_timings: dict[str, float] = {}
    for alpha in alphas:
        ta = time.perf_counter()
        hyb = scaled_hybrid_scores(gru, resid, alpha=float(alpha))
        # No dataframe copy of 37k for scoring — only score array
        ev = evaluate_scores(shared, scores=hyb, targets=y, gru_scores=gru, gru_dec=gru_dec)
        row = _row_from_eval(MONTH_TEST, "hybrid_alpha", float(alpha), ev)
        alpha_timings[str(alpha)] = time.perf_counter() - ta
        alpha_rows.append(row)
        # Persist only score columns, not full six copies of month frame
        np.save(out_dir / f"march_hybrid_scores_alpha_{alpha:.2f}.npy", hyb)

    # alpha=1 vs Part A hybrid
    a1 = next(r for r in alpha_rows if abs(float(r["alpha"]) - 1.0) < 1e-12)
    pa_h = part_a[MONTH_TEST]["hybrid_frozen_alpha_1"]["regression"]["mae"]
    if abs(float(a1["mae"]) - float(pa_h)) > 1e-9:
        raise RuntimeError(
            f"alpha=1 MAE {a1['mae']} != Part A hybrid MAE {pa_h}"
        )

    timings["part_b_alpha_loop_s"] = time.perf_counter() - t0
    timings["per_alpha_s"] = alpha_timings

    # Safe alpha search (March only — no tuning on March labels beyond evaluation)
    gru_m = part_a[MONTH_TEST]["gru"]
    gru_mae = float(gru_m["regression"]["mae"])
    gru_top1 = float(gru_m["ranking"]["top1_candidate_agreement"])
    gru_oracle = float(gru_m["oracle_tie_break_win_rate"])

    safe_candidates = []
    for row in alpha_rows:
        verdict = _safe_alpha_verdict(row, gru_mae=gru_mae, gru_top1=gru_top1, gru_oracle=gru_oracle)
        row["safe_verdict"] = verdict
        if float(row["alpha"]) > 0 and verdict["safe"]:
            safe_candidates.append(row)

    best_alpha = None
    if safe_candidates:
        safe_candidates.sort(key=lambda r: (float(r["alpha"]), float(r["mae"])))
        best_alpha = float(safe_candidates[0]["alpha"])

    if best_alpha is None:
        conclusion = (
            "No evidence supports deploying GRU+RLS; "
            "retain GRU as production model and keep RLS/hybrid shadow-only."
        )
        recommendation = "retain_gru_production_shadow_only"
    else:
        conclusion = (
            f"Candidate for a future shadow validation: alpha={best_alpha}. "
            "Do NOT deploy; evidence only supports further shadow monitoring."
        )
        recommendation = f"shadow_candidate_alpha_{best_alpha}"

    prod_after = file_sha256(prod_ckpt) if prod_ckpt.is_file() else None
    production_untouched = (
        prod_before is not None and prod_after is not None and prod_before == prod_after
    )
    timings["total_s"] = time.perf_counter() - t_all

    march_baselines = {
        "gru": {
            "mae": part_a[MONTH_TEST]["gru"]["regression"]["mae"],
            "rmse": part_a[MONTH_TEST]["gru"]["regression"]["rmse"],
            "top1": part_a[MONTH_TEST]["gru"]["ranking"]["top1_candidate_agreement"],
            "top5": part_a[MONTH_TEST]["gru"]["ranking"]["topk_candidate_overlap"],
            "spearman": part_a[MONTH_TEST]["gru"]["ranking"]["mean_spearman"],
            "oracle_tie_break": part_a[MONTH_TEST]["gru"]["oracle_tie_break_win_rate"],
        },
        "rls": {
            "mae": part_a[MONTH_TEST]["rls"]["regression"]["mae"],
            "rmse": part_a[MONTH_TEST]["rls"]["regression"]["rmse"],
            "top1": part_a[MONTH_TEST]["rls"]["ranking"]["top1_candidate_agreement"],
            "top5": part_a[MONTH_TEST]["rls"]["ranking"]["topk_candidate_overlap"],
            "spearman": part_a[MONTH_TEST]["rls"]["ranking"]["mean_spearman"],
            "oracle_tie_break": part_a[MONTH_TEST]["rls"]["oracle_tie_break_win_rate"],
        },
        "hybrid_alpha_1": {
            "mae": part_a[MONTH_TEST]["hybrid_frozen_alpha_1"]["regression"]["mae"],
            "rmse": part_a[MONTH_TEST]["hybrid_frozen_alpha_1"]["regression"]["rmse"],
            "top1": part_a[MONTH_TEST]["hybrid_frozen_alpha_1"]["ranking"]["top1_candidate_agreement"],
            "top5": part_a[MONTH_TEST]["hybrid_frozen_alpha_1"]["ranking"]["topk_candidate_overlap"],
            "spearman": part_a[MONTH_TEST]["hybrid_frozen_alpha_1"]["ranking"]["mean_spearman"],
            "limit_agreement_vs_gru": part_a[MONTH_TEST]["hybrid_frozen_alpha_1"]["dtl_vs_gru"][
                "recommended_limit_agreement"
            ],
            "mean_abs_delta_limit": part_a[MONTH_TEST]["hybrid_frozen_alpha_1"]["dtl_vs_gru"][
                "mean_abs_dtl_difference"
            ],
            "oracle_tie_break": part_a[MONTH_TEST]["hybrid_frozen_alpha_1"]["oracle_tie_break_win_rate"],
        },
    }

    summary: dict[str, Any] = {
        "experiment": "hybrid_sensitivity_optimized",
        "production_path_unchanged": True,
        "production_core_gru_temporal_v1_untouched": production_untouched,
        "production_sha256_before": prod_before,
        "production_sha256_after": prod_after,
        "optimization": {
            "gru_scored_once_per_month": True,
            "residual_once_per_month": True,
            "shared_dtl_pack_once_per_month": True,
            "alpha_loop_score_dependent_only": True,
            "march_primary_alpha_analysis": True,
            "previous_runtime_minutes_approx": 51.5,
        },
        "timings_seconds": timings,
        "protocol": {
            "train_month": MONTH_TRAIN,
            "validation_month": MONTH_VAL,
            "primary_test_month": MONTH_TEST,
            "alphas": list(alphas),
            "hybrid_formula": "gru_score + alpha * residual",
            "month_roles": MONTH_ROLE,
            "online_updates": False,
        },
        "march_baselines_recalculated": march_baselines,
        "part_a_all_month_frozen": {
            m: {
                "role": part_a[m]["role"],
                "n_rows": part_a[m]["n_rows"],
                "gru": part_a[m]["gru"],
                "rls": part_a[m]["rls"],
                "hybrid_frozen_alpha_1": part_a[m]["hybrid_frozen_alpha_1"],
            }
            for m in months
        },
        "part_b_march_alpha_sensitivity": alpha_rows,
        "safe_alpha_search": {
            "best_alpha": best_alpha,
            "safe_candidates": [c["alpha"] for c in safe_candidates],
            "conclusion": conclusion,
            "recommendation": recommendation,
        },
        "sanity": {
            "alpha0_equals_gru": bool(np.allclose(h0, gru)),
            "alpha1_matches_part_a_hybrid_mae": True,
        },
        "artifacts": {
            "output_dir": str(OUT_REL.as_posix()),
            "summary": "SENSITIVITY_SUMMARY.json",
            "alpha_csv": "alpha_comparison.csv",
            "month_csv": "month_metrics.csv",
            "report": "sensitivity_report.md",
        },
    }

    # Write outputs (new names; preserve prior partials)
    write_json(out_dir / "SENSITIVITY_SUMMARY.json", summary)
    pd.DataFrame(part_a_rows).to_csv(out_dir / "month_metrics.csv", index=False)
    alpha_df = pd.DataFrame(alpha_rows)
    if not alpha_df.empty:
        alpha_df["safe"] = alpha_df["safe_verdict"].apply(
            lambda v: v.get("safe") if isinstance(v, dict) else v
        )
        alpha_df.to_csv(out_dir / "alpha_comparison.csv", index=False)

    (out_dir / "sensitivity_report.md").write_text(
        _render_report(summary, alpha_rows, part_a),
        encoding="utf-8",
    )
    return summary


def _render_report(summary: dict[str, Any], alpha_rows: list[dict], part_a: dict) -> str:
    sa = summary["safe_alpha_search"]
    mb = summary["march_baselines_recalculated"]
    t = summary["timings_seconds"]
    lines = [
        "# Optimized Hybrid Sensitivity Report",
        "",
        "## A. Executive conclusion",
        "",
        sa["conclusion"],
        "",
        f"- Best safe alpha: `{sa['best_alpha']}`",
        f"- Total runtime: **{t.get('total_s', float('nan')):.1f} s**",
        f"- Part B alpha loop: **{t.get('part_b_alpha_loop_s', float('nan')):.2f} s**",
        "",
        "## B. All-month frozen comparison",
        "",
        "### March (primary held-out)",
        "",
        "| Model | MAE | RMSE | Top-1 | Top-5 | Spearman | Oracle |",
        "|-------|----:|-----:|------:|------:|---------:|-------:|",
        f"| GRU | {mb['gru']['mae']:.4f} | {mb['gru']['rmse']:.4f} | {mb['gru']['top1']:.3f} | {mb['gru']['top5']:.3f} | {mb['gru']['spearman']:.3f} | {mb['gru']['oracle_tie_break']:.3f} |",
        f"| RLS | {mb['rls']['mae']:.4f} | {mb['rls']['rmse']:.4f} | {mb['rls']['top1']:.3f} | {mb['rls']['top5']:.3f} | {mb['rls']['spearman']:.3f} | {mb['rls']['oracle_tie_break']:.3f} |",
        f"| Hybrid α=1 | {mb['hybrid_alpha_1']['mae']:.4f} | {mb['hybrid_alpha_1']['rmse']:.4f} | {mb['hybrid_alpha_1']['top1']:.3f} | {mb['hybrid_alpha_1']['top5']:.3f} | {mb['hybrid_alpha_1']['spearman']:.3f} | {mb['hybrid_alpha_1']['oracle_tie_break']:.3f} |",
        "",
        "### Diagnostic months",
        "",
    ]
    for month in (MONTH_TRAIN, MONTH_VAL):
        g = part_a[month]["gru"]["regression"]["mae"]
        r = part_a[month]["rls"]["regression"]["mae"]
        h = part_a[month]["hybrid_frozen_alpha_1"]["regression"]["mae"]
        lines.append(f"- {month}: GRU MAE={g:.4f}, RLS={r:.4f}, Hybridα1={h:.4f}")
    lines += [
        "",
        "## C. Alpha sensitivity (March)",
        "",
        "| alpha | MAE | RMSE | Top-1 | Spearman | limit agree | over | under | oracle | mean\\|Δ\\| | changed | looser | safe |",
        "|------:|----:|-----:|------:|---------:|------------:|-----:|------:|-------:|--------:|--------:|-------:|:----:|",
    ]
    for row in alpha_rows:
        safe = row.get("safe_verdict", {}).get("safe") if isinstance(row.get("safe_verdict"), dict) else ""
        lines.append(
            f"| {row['alpha']:.2f} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['top1']:.3f} | "
            f"{row['spearman']:.3f} | {row['recommended_limit_agreement_vs_gru']:.3f} | "
            f"{row['over_limit_rate_vs_gru']:.3f} | {row['under_limit_rate_vs_gru']:.3f} | "
            f"{row['oracle_tie_break_win_rate']:.3f} | {row['mean_abs_dtl_difference']:.3f} | "
            f"{row['pct_groups_limit_changed']:.3f} | {row['pct_changed_looser']} | {safe} |"
        )
    lines += [
        "",
        "## D–G. DTL / yield-tie / oracle / limit-direction",
        "",
        f"March yield-tie rate (GRU): **{part_a[MONTH_TEST]['gru']['yield_tie_rate']:.3f}**.",
        "",
        "## H. Recommendation",
        "",
        sa["conclusion"],
        "",
        "Do **not** deploy. Keep production GRU path unchanged.",
        "",
        "## Timings",
        "",
        "```json",
        json.dumps(t, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    summary = run_sensitivity_experiment()
    print(
        json.dumps(
            {
                "best_alpha": summary["safe_alpha_search"]["best_alpha"],
                "conclusion": summary["safe_alpha_search"]["conclusion"],
                "total_s": summary["timings_seconds"]["total_s"],
                "part_b_s": summary["timings_seconds"]["part_b_alpha_loop_s"],
                "production_untouched": summary["production_core_gru_temporal_v1_untouched"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
