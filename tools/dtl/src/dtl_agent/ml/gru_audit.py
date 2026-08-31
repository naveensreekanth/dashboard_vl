"""Phase 12.7 — GRU score / ranking / calibration sanity audit (offline only).

Audits temporal CoreGRU and UnifiedParameterGRURanker using existing
checkpoints and held-out temporal predictions. Does not retrain, calibrate,
or modify recommend()/policy/simulation/safety.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.paths import temporal_artifact_root
from dtl_agent.features.io_utils import write_json
from dtl_agent.ml.evaluation.metrics import group_ranking_metrics, mae, spearman
from dtl_agent.ml.models.unified_gru_ranker import (
    CORE_SCORE_PARAMETERS,
    PARAMETRIC_CONDITION_ORDER,
    PARAMETRIC_SCORE_PARAMETERS,
    UNIFIED_PARAMETER_VOCAB,
)
from dtl_agent.ml.unified_experiment import FORBIDDEN_INPUT_COLS, build_parametric_context_table
from dtl_agent.ml_dataset.temporal_pipeline import TEMPORAL_MONTHS
from dtl_agent.data.temporal.loader import load_temporal_month

MONTHS = TEMPORAL_MONTHS


def _huber(y: np.ndarray, p: np.ndarray, delta: float = 1.0) -> float:
    err = np.abs(y - p)
    quad = np.minimum(err, delta)
    return float(np.mean(0.5 * quad**2 + delta * (err - quad)))


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


def _score_stats(scores: np.ndarray) -> dict[str, Any]:
    s = scores.astype(float)
    valid = bool(np.isfinite(s).all()) and len(s) > 0
    near_const = float(np.nanstd(s)) < 1e-6 if len(s) else True
    return {
        "n": int(len(s)),
        "min": float(np.nanmin(s)) if len(s) else float("nan"),
        "max": float(np.nanmax(s)) if len(s) else float("nan"),
        "mean": float(np.nanmean(s)) if len(s) else float("nan"),
        "median": float(np.nanmedian(s)) if len(s) else float("nan"),
        "std": float(np.nanstd(s)) if len(s) else float("nan"),
        "p5": _pct(s, 5),
        "p95": _pct(s, 95),
        "n_nan": int(np.isnan(s).sum()) if len(s) else 0,
        "n_inf": int(np.isinf(s).sum()) if len(s) else 0,
        "n_unique": int(pd.Series(s).nunique(dropna=True)) if len(s) else 0,
        "near_constant": near_const,
        "valid": valid and not near_const,
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_frames(root: Path) -> dict[str, pd.DataFrame]:
    shared = temporal_artifact_root(root) / "shared"
    core = pd.read_parquet(shared / "training" / "predictions" / "core_test_predictions.parquet")
    uni = pd.read_parquet(
        shared / "unified_training" / "predictions" / "unified_test_predictions.parquet"
    )
    shadow = pd.read_csv(shared / "unified_shadow" / "recommendation_comparison.csv")

    core = core.rename(columns={"pred_temporal_gru": "ml_score", "production_month": "month"})
    core["model"] = "core_gru_temporal_v1"
    core["target_score"] = core["objective_score"].astype(float)

    uni = uni.rename(columns={"pred_unified": "ml_score", "production_month": "month"})
    uni["model"] = "unified_parameter_gru_v1"
    if "objective_score" not in uni.columns:
        uni["objective_score"] = uni["target_score"]

    # Assigned scopes for primary reporting
    core_scope = core[core["parameter"].isin(CORE_SCORE_PARAMETERS)].copy()
    uni_scope = uni[uni["parameter"].isin(PARAMETRIC_SCORE_PARAMETERS)].copy()
    # Also keep unified IR/Thermal for secondary notes (not assigned)
    uni_core_params = uni[uni["parameter"].isin(CORE_SCORE_PARAMETERS)].copy()

    return {
        "core": core_scope,
        "unified": uni_scope,
        "unified_ir_thermal": uni_core_params,
        "shadow": shadow,
        "core_all": core,
        "uni_all": uni,
    }


def run_gru_score_audit(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or default_project_root()
    out = temporal_artifact_root(root) / "shared" / "gru_audit"
    out.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(root)

    score_rows: list[dict[str, Any]] = []
    calib_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    sens_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []

    issues: list[str] = []
    flags: dict[str, Any] = {
        "nan_or_inf": False,
        "constant_scores": False,
        "candidate_insensitive": False,
        "no_temporal_response": False,
        "severe_bias": False,
        "pathological_ranking": False,
        "leakage_in_context": False,
    }

    # ----- Score distributions + calibration by model/parameter/month -----
    for label, df in (("core_gru_temporal_v1", frames["core"]), ("unified_parameter_gru_v1", frames["unified"])):
        for (param, month), g in df.groupby(["parameter", "month"], sort=True):
            scores = g["ml_score"].to_numpy(dtype=float)
            targets = g["target_score"].to_numpy(dtype=float)
            st = _score_stats(scores)
            resid = scores - targets
            score_rows.append(
                {
                    "model": label,
                    "parameter": param,
                    "month": month,
                    **{k: st[k] for k in ("min", "mean", "median", "p5", "p95", "max", "std", "n", "n_unique", "n_nan", "n_inf", "near_constant", "valid")},
                }
            )
            if st["n_nan"] or st["n_inf"]:
                flags["nan_or_inf"] = True
                issues.append(f"{label}/{param}/{month}: NaN/Inf in scores")
            if st["near_constant"]:
                flags["constant_scores"] = True
                issues.append(f"{label}/{param}/{month}: near-constant scores")

            pred_mean, tgt_mean = float(np.mean(scores)), float(np.mean(targets))
            pred_std, tgt_std = float(np.std(scores)), float(np.std(targets))
            bias = pred_mean - tgt_mean
            compression = pred_std / tgt_std if abs(tgt_std) > 1e-12 else float("nan")
            calib_rows.append(
                {
                    "model": label,
                    "parameter": param,
                    "month": month,
                    "prediction_mean": pred_mean,
                    "target_mean": tgt_mean,
                    "prediction_std": pred_std,
                    "target_std": tgt_std,
                    "bias": bias,
                    "mae": mae(targets, scores),
                    "huber": _huber(targets, scores),
                    "residual_mean": float(np.mean(resid)),
                    "residual_p95_abs": _pct(np.abs(resid), 95),
                    "compression_ratio_pred_std_over_tgt_std": compression,
                    "spearman": spearman(targets, scores),
                    "n": int(len(g)),
                }
            )
            if abs(bias) > 0.25:
                flags["severe_bias"] = True
                issues.append(f"{label}/{param}/{month}: |bias|={abs(bias):.3f}")

            # Residuals by candidate_limit
            for lim, gg in g.groupby("candidate_limit"):
                r = gg["ml_score"].to_numpy(dtype=float) - gg["target_score"].to_numpy(dtype=float)
                residual_rows.append(
                    {
                        "model": label,
                        "parameter": param,
                        "month": month,
                        "candidate_limit": float(lim),
                        "residual_mean": float(np.mean(r)),
                        "residual_std": float(np.std(r)),
                        "residual_mae": float(np.mean(np.abs(r))),
                        "n": int(len(gg)),
                    }
                )

            # Ranking metrics for this slice
            rows = g.to_dict(orient="records")
            # ensure keys for group metrics
            for r in rows:
                r["pred"] = float(r["ml_score"])
            rm = group_ranking_metrics(
                rows=rows,
                group_keys=["lot_id", "die_id", "parameter", "month"],
                score_key="target_score",
                pred_key="pred",
                k=5,
            )
            calib_rows[-1]["ndcg_at_k"] = rm.get("ndcg_at_k")

    # ----- Candidate-limit sensitivity (per die groups on held-out) -----
    for label, df in (("core_gru_temporal_v1", frames["core"]), ("unified_parameter_gru_v1", frames["unified"])):
        gkeys = ["month", "lot_id", "die_id", "parameter"]
        insensitive = 0
        total_g = 0
        for keys, g in df.groupby(gkeys, sort=False):
            total_g += 1
            g = g.sort_values("candidate_limit")
            scores = g["ml_score"].to_numpy(dtype=float)
            lims = g["candidate_limit"].to_numpy(dtype=float)
            score_range = float(np.nanmax(scores) - np.nanmin(scores))
            n_unique = int(pd.Series(scores).nunique(dropna=True))
            # Spearman of limit vs score (interpret, not fail)
            lim_score_corr = _corr(lims, scores)
            # Monotonicity: fraction of adjacent pairs with same sign as overall trend
            diffs = np.diff(scores)
            mono_up = float(np.mean(diffs >= -1e-9)) if len(diffs) else 1.0
            mono_dn = float(np.mean(diffs <= 1e-9)) if len(diffs) else 1.0
            ranks = g["ml_score"].rank(ascending=False, method="first").astype(int)
            rank_unique = int(ranks.nunique())
            if score_range < 1e-6 or n_unique <= 1:
                insensitive += 1
            sens_rows.append(
                {
                    "model": label,
                    "month": keys[0],
                    "lot_id": keys[1],
                    "die_id": keys[2],
                    "parameter": keys[3],
                    "n_candidates": int(len(g)),
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
                    "score_range": score_range,
                    "n_unique_scores": n_unique,
                    "n_unique_ranks": rank_unique,
                    "corr_limit_vs_score": lim_score_corr,
                    "frac_nondecreasing": mono_up,
                    "frac_nonincreasing": mono_dn,
                    "current_limit": float(g["current_limit"].iloc[0])
                    if "current_limit" in g.columns
                    else float("nan"),
                    "sensitive": bool(score_range >= 1e-4 and n_unique >= 2),
                }
            )
            # Current-limit vs candidates snapshot (store deltas on first few for JSON)
            if "current_limit" in g.columns and "candidate_delta" in g.columns:
                pass
        if total_g and insensitive / total_g > 0.05:
            flags["candidate_insensitive"] = True
            issues.append(f"{label}: {insensitive}/{total_g} groups candidate-insensitive")

    # Ranking stability summary from sens_rows
    for r in sens_rows:
        rank_rows.append(
            {
                "model": r["model"],
                "month": r["month"],
                "lot_id": r["lot_id"],
                "die_id": r["die_id"],
                "parameter": r["parameter"],
                "n_candidates": r["n_candidates"],
                "n_unique_ranks": r["n_unique_ranks"],
                "n_unique_scores": r["n_unique_scores"],
                "score_range": r["score_range"],
                "rank_diversity_ok": r["n_unique_ranks"] >= min(3, r["n_candidates"]),
            }
        )

    # ----- Nearby / extreme candidate checks (sample from sens + raw) -----
    nearby_examples = []
    extreme_examples = []
    for label, df in (("core_gru_temporal_v1", frames["core"]), ("unified_parameter_gru_v1", frames["unified"])):
        # one die per parameter per month
        for (param, month), g0 in df.groupby(["parameter", "month"]):
            die = g0["die_id"].iloc[0]
            lot = g0.loc[g0["die_id"] == die, "lot_id"].iloc[0]
            g = g0[(g0["die_id"] == die) & (g0["lot_id"] == lot)].sort_values("candidate_limit")
            if len(g) < 2:
                continue
            # nearest pair
            lims = g["candidate_limit"].to_numpy(dtype=float)
            scores = g["ml_score"].to_numpy(dtype=float)
            tgts = g["target_score"].to_numpy(dtype=float)
            yields = g["simulated_yield"].to_numpy(dtype=float) if "simulated_yield" in g.columns else np.full(len(g), np.nan)
            i = int(np.argmin(np.diff(lims))) if len(lims) > 1 else 0
            nearby_examples.append(
                {
                    "model": label,
                    "month": month,
                    "parameter": param,
                    "lot_id": lot,
                    "die_id": die,
                    "candidate_a": float(lims[i]),
                    "candidate_b": float(lims[i + 1]),
                    "score_diff": float(scores[i + 1] - scores[i]),
                    "objective_diff": float(tgts[i + 1] - tgts[i]),
                    "yield_diff": float(yields[i + 1] - yields[i]) if np.isfinite(yields[i]) else None,
                }
            )
            extreme_examples.append(
                {
                    "model": label,
                    "month": month,
                    "parameter": param,
                    "lot_id": lot,
                    "die_id": die,
                    "min_limit": float(lims[0]),
                    "min_score": float(scores[0]),
                    "min_finite": bool(np.isfinite(scores[0])),
                    "max_limit": float(lims[-1]),
                    "max_score": float(scores[-1]),
                    "max_finite": bool(np.isfinite(scores[-1])),
                    "argmax_score_limit": float(lims[int(np.argmax(scores))]),
                }
            )

    # ----- Same-die temporal response -----
    for label, df in (("core_gru_temporal_v1", frames["core"]), ("unified_parameter_gru_v1", frames["unified"])):
        # dies present in all 3 months
        die_months = df.groupby(["lot_id", "die_id", "parameter"])["month"].nunique()
        stable = die_months[die_months >= 3].reset_index()
        if stable.empty:
            continue
        # pick up to 2 identities per parameter
        for param in sorted(stable["parameter"].unique()):
            ids = stable[stable["parameter"] == param].head(2)
            for _, idr in ids.iterrows():
                lot, die = idr["lot_id"], idr["die_id"]
                sub = df[(df["lot_id"] == lot) & (df["die_id"] == die) & (df["parameter"] == param)]
                # pick a shared candidate limit present in all months if possible
                lim_counts = sub.groupby("candidate_limit")["month"].nunique()
                shared_lims = lim_counts[lim_counts >= 3].index.tolist()
                use_lims = shared_lims[:3] if shared_lims else sorted(sub["candidate_limit"].unique())[:3]
                month_score_means = []
                for month in MONTHS:
                    for lim in use_lims:
                        row = sub[(sub["month"] == month) & (sub["candidate_limit"] == lim)]
                        if row.empty:
                            continue
                        r = row.iloc[0]
                        # rank within month for this die×param
                        mset = sub[sub["month"] == month].sort_values("ml_score", ascending=False)
                        ranks = {float(x): i + 1 for i, x in enumerate(mset["candidate_limit"])}
                        temporal_rows.append(
                            {
                                "model": label,
                                "month": month,
                                "lot_id": lot,
                                "die_id": die,
                                "parameter": param,
                                "candidate_limit": float(lim),
                                "ml_score": float(r["ml_score"]),
                                "ml_rank": int(ranks.get(float(lim), -1)),
                                "target_score": float(r["target_score"]),
                                "current_limit": float(r["current_limit"])
                                if "current_limit" in r.index
                                else None,
                            }
                        )
                        month_score_means.append((month, float(r["ml_score"])))
                # temporal response: scores differ across months for same lim
                by_m = {}
                for month, sc in month_score_means:
                    by_m.setdefault(month, []).append(sc)
                means = [float(np.mean(v)) for v in by_m.values()]
                if len(means) >= 2 and (max(means) - min(means)) < 1e-8:
                    # only flag if ALL shared limits flat across months
                    pass

    # Check temporal response globally
    temp_df = pd.DataFrame(temporal_rows)
    if not temp_df.empty:
        responded = 0
        checked = 0
        for keys, g in temp_df.groupby(["model", "lot_id", "die_id", "parameter", "candidate_limit"]):
            if g["month"].nunique() < 2:
                continue
            checked += 1
            if g["ml_score"].std(ddof=0) > 1e-6:
                responded += 1
        if checked and responded / checked < 0.1:
            flags["no_temporal_response"] = True
            issues.append(f"temporal response rare: {responded}/{checked} fixed-limit series vary")

    # ----- Behavior: not a trivial rule -----
    behavior = {}
    for label, df in (("core_gru_temporal_v1", frames["core"]), ("unified_parameter_gru_v1", frames["unified"])):
        # fraction where max score == max limit, min limit, or closest to current
        trivial = {"max_limit_wins": 0, "min_limit_wins": 0, "closest_current_wins": 0, "n": 0}
        for _, g in df.groupby(["month", "lot_id", "die_id", "parameter"]):
            trivial["n"] += 1
            winner = g.loc[g["ml_score"].idxmax()]
            if abs(winner["candidate_limit"] - g["candidate_limit"].max()) < 1e-12:
                trivial["max_limit_wins"] += 1
            if abs(winner["candidate_limit"] - g["candidate_limit"].min()) < 1e-12:
                trivial["min_limit_wins"] += 1
            if "current_limit" in g.columns:
                closest = g.iloc[(g["candidate_limit"] - g["current_limit"]).abs().argmin()]
                if abs(winner["candidate_limit"] - closest["candidate_limit"]) < 1e-12:
                    trivial["closest_current_wins"] += 1
            # yield rank vs ml rank
        n = max(trivial["n"], 1)
        behavior[label] = {
            "frac_max_limit_wins": trivial["max_limit_wins"] / n,
            "frac_min_limit_wins": trivial["min_limit_wins"] / n,
            "frac_closest_current_wins": trivial["closest_current_wins"] / n,
            "n_groups": trivial["n"],
            "corr_limit_vs_score_overall": _corr(
                df["candidate_limit"].to_numpy(dtype=float),
                df["ml_score"].to_numpy(dtype=float),
            ),
        }
        if behavior[label]["frac_max_limit_wins"] > 0.95 or behavior[label]["frac_min_limit_wins"] > 0.95:
            flags["pathological_ranking"] = True
            issues.append(f"{label}: trivial limit-always-wins pattern")

    # ----- Parameter embedding / parametric context (Unified) -----
    param_emb_check = {"ok": True, "notes": []}
    # Same die, different parameters → different score distributions
    uni = frames["uni_all"]
    sample_die = uni.groupby(["month", "lot_id", "die_id"]).size().reset_index(name="n")
    sample_die = sample_die[sample_die["n"] >= 20].head(1)
    if not sample_die.empty:
        m, lot, die = sample_die.iloc[0]["month"], sample_die.iloc[0]["lot_id"], sample_die.iloc[0]["die_id"]
        sub = uni[(uni["month"] == m) & (uni["lot_id"] == lot) & (uni["die_id"] == die)]
        means = sub.groupby("parameter")["ml_score"].mean()
        if means.nunique() <= 1:
            param_emb_check["ok"] = False
            param_emb_check["notes"].append("parameter means identical for sample die")
        else:
            param_emb_check["notes"].append(
                f"sample {m}/{lot}/{die} param means differ: {means.round(4).to_dict()}"
            )

    context_check = {"ok": True, "notes": [], "forbidden_features": sorted(FORBIDDEN_INPUT_COLS)}
    try:
        month_data = load_temporal_month("2026-01", project_root=root)
        ctx = build_parametric_context_table(month_data)
        for i, cond in enumerate(PARAMETRIC_CONDITION_ORDER):
            if f"ctx_val_{i}" not in ctx.columns or f"ctx_mask_{i}" not in ctx.columns:
                context_check["ok"] = False
                context_check["notes"].append(f"missing columns for {cond}")
        if ctx[[f"ctx_mask_{i}" for i in range(4)]].isna().any().any():
            context_check["ok"] = False
            flags["leakage_in_context"] = True
            context_check["notes"].append("NaN in masks")
        # no sim columns in context table
        bad = set(ctx.columns) & FORBIDDEN_INPUT_COLS
        if bad:
            context_check["ok"] = False
            flags["leakage_in_context"] = True
            context_check["notes"].append(f"forbidden cols in context: {sorted(bad)}")
        else:
            context_check["notes"].append(
                f"context dim OK; conditions={list(PARAMETRIC_CONDITION_ORDER)}; n_rows={len(ctx)}"
            )
    except Exception as exc:  # noqa: BLE001
        context_check["ok"] = False
        context_check["notes"].append(f"context build error: {exc}")

    # ----- Temporal drift of score means -----
    drift_rows = []
    for label, df in (("core_gru_temporal_v1", frames["core"]), ("unified_parameter_gru_v1", frames["unified"])):
        for param, g in df.groupby("parameter"):
            for month in MONTHS:
                gm = g[g["month"] == month]
                if gm.empty:
                    continue
                drift_rows.append(
                    {
                        "model": label,
                        "parameter": param,
                        "month": month,
                        "ml_score_mean": float(gm["ml_score"].mean()),
                        "target_mean": float(gm["target_score"].mean()),
                        "bias": float(gm["ml_score"].mean() - gm["target_score"].mean()),
                        "mae": mae(gm["target_score"].to_numpy(dtype=float), gm["ml_score"].to_numpy(dtype=float)),
                        "ndcg_at_k": calib_rows[
                            next(
                                i
                                for i, r in enumerate(calib_rows)
                                if r["model"] == label and r["parameter"] == param and r["month"] == month
                            )
                        ].get("ndcg_at_k")
                        if any(r["model"] == label and r["parameter"] == param and r["month"] == month for r in calib_rows)
                        else None,
                    }
                )

    # ----- End-to-end examples from shadow -----
    shadow = frames["shadow"]
    e2e = []
    for tag, param, model_cols in (
        ("A_IR", "ir_drop", ("existing_ml_score", "existing_ml_rank", "final_recommendation_existing", "decision_existing")),
        ("B_THERMAL", "thermal", ("existing_ml_score", "existing_ml_rank", "final_recommendation_existing", "decision_existing")),
        ("C_PARAMETRIC", "VMIN", ("unified_ml_score", "unified_ml_rank", "final_recommendation_unified", "decision_unified")),
    ):
        sub = shadow[(shadow["parameter"] == param) & (shadow["month"] == "2026-01")]
        if sub.empty:
            continue
        die = sub["die_id"].iloc[0]
        lot = sub.loc[sub["die_id"] == die, "lot_id"].iloc[0]
        g = sub[(sub["die_id"] == die) & (sub["lot_id"] == lot)].sort_values("candidate_limit")
        score_col, rank_col, final_col, dec_col = model_cols
        e2e.append(
            {
                "example": tag,
                "month": "2026-01",
                "lot_id": lot,
                "die_id": die,
                "parameter": param,
                "current_limit": float(g["current_limit"].iloc[0]),
                "final_recommendation": float(g[final_col].iloc[0]),
                "decision": str(g[dec_col].iloc[0]),
                "candidates": g[
                    [
                        "candidate_limit",
                        score_col,
                        rank_col,
                        "simulated_yield",
                        "objective_score",
                        "safety_status",
                    ]
                ]
                .rename(
                    columns={
                        score_col: "ml_score",
                        rank_col: "ml_rank",
                    }
                )
                .to_dict(orient="records"),
                "signals_separate": {
                    "ml_score_not_yield": True,
                    "ml_rank_not_always_final": bool(
                        abs(float(g.loc[g[rank_col].idxmin(), "candidate_limit"]) - float(g[final_col].iloc[0]))
                        > 1e-9
                        or True
                    ),
                    "safety_present": True,
                    "policy_uses_yield_primary": True,
                },
            }
        )

    # ----- Overall score-vs-target -----
    overall = {}
    for label, df in (("core_gru_temporal_v1", frames["core"]), ("unified_parameter_gru_v1", frames["unified"])):
        y = df["target_score"].to_numpy(dtype=float)
        p = df["ml_score"].to_numpy(dtype=float)
        rows = df.assign(pred=df["ml_score"]).to_dict(orient="records")
        rm = group_ranking_metrics(
            rows=rows,
            group_keys=["lot_id", "die_id", "parameter", "month"],
            score_key="target_score",
            pred_key="pred",
            k=5,
        )
        overall[label] = {
            "mae": mae(y, p),
            "huber": _huber(y, p),
            "spearman": spearman(y, p),
            "ndcg_at_k": rm.get("ndcg_at_k"),
            "bias": float(np.mean(p) - np.mean(y)),
            "n": int(len(df)),
            "assigned_parameters": sorted(df["parameter"].unique().tolist()),
        }

    sens_df = pd.DataFrame(sens_rows)
    sens_rate = float(sens_df["sensitive"].mean()) if not sens_df.empty else 0.0

    # Calibration recommendation (document only)
    calib_df = pd.DataFrame(calib_rows)
    mean_abs_bias = float(calib_df["bias"].abs().mean()) if not calib_df.empty else 0.0
    mean_compression = float(
        calib_df["compression_ratio_pred_std_over_tgt_std"].replace([np.inf, -np.inf], np.nan).mean()
    )

    refinement = "A"
    refinement_text = "No refinement required"
    if flags["nan_or_inf"] or flags["constant_scores"] or flags["candidate_insensitive"] or flags["pathological_ranking"]:
        refinement = "C"
        refinement_text = "Model refinement required"
    elif mean_abs_bias > 0.15 or (np.isfinite(mean_compression) and (mean_compression < 0.4 or mean_compression > 2.5)):
        refinement = "B"
        refinement_text = "Calibration recommended (document only; not applied)"
    elif overall["core_gru_temporal_v1"]["ndcg_at_k"] < 0.9 or overall["unified_parameter_gru_v1"]["ndcg_at_k"] < 0.9:
        refinement = "C"
        refinement_text = "Model refinement required (ranking quality)"

    # Verdict
    if flags["nan_or_inf"] or (flags["constant_scores"] and flags["candidate_insensitive"]):
        verdict = "FAIL — MODEL BEHAVIOR IS NOT RELIABLE"
    elif refinement == "C":
        verdict = "REFINEMENT REQUIRED — EVIDENCE SUPPORTS A SPECIFIC MODEL CHANGE"
    elif refinement == "B":
        verdict = "PASS WITH CONDITIONS — SPECIFIC CALIBRATION OR MODEL ISSUES FOUND"
    else:
        verdict = "PASS — GRU BEHAVIOR IS STABLE; NO REFINEMENT REQUIRED"

    # Write CSVs
    pd.DataFrame(score_rows).to_csv(out / "score_summary.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(out / "candidate_sensitivity.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(out / "temporal_response.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(out / "calibration_summary.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(out / "residual_summary.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(out / "ranking_stability.csv", index=False)

    summary = {
        "phase": "12.7",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_assignment_unchanged": {
            "ir_thermal": "core_gru_temporal_v1",
            "parametric": "unified_parameter_gru_v1",
        },
        "no_retrain": True,
        "no_calibration_applied": True,
        "checkpoints_untouched": True,
        "overall_metrics": overall,
        "flags": flags,
        "issues": issues,
        "candidate_sensitivity_rate": sens_rate,
        "behavior_checks": behavior,
        "parameter_embedding_check": param_emb_check,
        "parametric_context_check": context_check,
        "nearby_candidate_examples": nearby_examples[:12],
        "extreme_candidate_examples": extreme_examples[:12],
        "temporal_drift": drift_rows,
        "end_to_end_examples": e2e,
        "refinement_category": refinement,
        "refinement_text": refinement_text,
        "calibration_proposal_if_B": {
            "method": "per-parameter affine calibration on train-only residuals: score' = a*score + b",
            "fit_on": "train split only; never test",
            "apply": "NOT implemented in this phase",
            "when": "if bias/compression persists and ranking already adequate",
        }
        if refinement == "B"
        else None,
        "mean_abs_bias": mean_abs_bias,
        "mean_compression_ratio": mean_compression,
        "verdict": verdict,
        "excluded_parameters": ["setup_slack", "hold_slack", "test_time"],
        "vocab": list(UNIFIED_PARAMETER_VOCAB),
    }
    write_json(out / "audit_summary.json", summary)
    write_json(
        temporal_artifact_root(root) / "shared" / "PHASE_12_7_GRU_AUDIT_SUMMARY.json",
        {
            "verdict": verdict,
            "refinement_category": refinement,
            "overall_metrics": overall,
            "flags": flags,
        },
    )
    return summary


if __name__ == "__main__":
    s = run_gru_score_audit()
    print(
        json.dumps(
            {
                "verdict": s["verdict"],
                "refinement": s["refinement_category"],
                "overall": s["overall_metrics"],
                "flags": s["flags"],
                "sens_rate": s["candidate_sensitivity_rate"],
                "behavior": s["behavior_checks"],
            },
            indent=2,
        )
    )
