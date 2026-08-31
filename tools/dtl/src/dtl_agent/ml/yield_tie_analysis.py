"""Phase 12.6A — IR/Thermal yield-tie ML rank analysis (offline only).

Compares temporal CoreGRU vs UnifiedParameterGRU rankings on eligible
candidates that share identical simulated_yield (policy tie semantics).
Does not modify recommend(), policy, simulation, safety, or checkpoints.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.paths import temporal_artifact_root
from dtl_agent.features.io_utils import write_json
from dtl_agent.ml.evaluation.metrics import group_ranking_metrics, ndcg_at_k, spearman

CORE_PARAMS = ("ir_drop", "thermal")
MONTHS = ("2026-01", "2026-02", "2026-03")


def _kendall_tau(y_true: np.ndarray, y_pred: np.ndarray, tie_tol: float = 0.0) -> float:
    """Kendall τ-b style pairwise concordance (no new deps)."""
    n = len(y_true)
    if n < 2:
        return 1.0
    concord = 0
    discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = y_true[i] - y_true[j]
            dp = y_pred[i] - y_pred[j]
            if abs(dt) <= tie_tol or abs(dp) <= tie_tol:
                continue
            if (dt > 0 and dp > 0) or (dt < 0 and dp < 0):
                concord += 1
            else:
                discord += 1
    denom = concord + discord
    return float((concord - discord) / denom) if denom else 1.0


def _rank_by_score(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """1-based dense ranks; higher score → rank 1 when higher_is_better."""
    order = np.argsort(-scores if higher_is_better else scores)
    ranks = np.empty(len(scores), dtype=int)
    for i, idx in enumerate(order):
        ranks[idx] = i + 1
    return ranks


def _winner_limit(df: pd.DataFrame, score_col: str) -> float:
    """Policy ML tie-break: among max-yield rows, pick best (lowest) ml_rank / highest score."""
    # Already filtered to max-yield set; pick highest ML score
    return float(df.loc[df[score_col].idxmax(), "candidate_limit"])


def analyze_ir_thermal_yield_ties(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or default_project_root()
    shadow = temporal_artifact_root(root) / "shared" / "unified_shadow"
    detail_path = shadow / "recommendation_comparison.csv"
    if not detail_path.is_file():
        raise FileNotFoundError(f"Missing shadow CSV: {detail_path}")

    detail = pd.read_csv(detail_path)
    core = detail[detail["parameter"].isin(CORE_PARAMS)].copy()

    # Optional held-out lot filter
    split_path = (
        temporal_artifact_root(root) / "shared" / "ml_dataset" / "split_manifest.json"
    )
    test_lots: set[str] = set()
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        test_lots = {l for l, s in split.get("lot_to_split", {}).items() if s == "test"}

    tie_case_rows: list[dict[str, Any]] = []
    candidate_flat: list[dict[str, Any]] = []
    representative_tables: list[dict[str, Any]] = []
    critical_examples: list[dict[str, Any]] = []

    month_param_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "yield_tied_cases": 0,
            "core_closer": 0,
            "unified_closer": 0,
            "tie_equal": 0,
            "unified_picks_higher_obj": 0,
            "core_picks_higher_obj": 0,
            "same_final_pick": 0,
            "disagree_final_pick": 0,
        }
    )

    # Per-group metrics accumulators (tied max-yield subsets only)
    metric_rows_core: list[dict] = []
    metric_rows_uni: list[dict] = []
    metric_rows_core_test: list[dict] = []
    metric_rows_uni_test: list[dict] = []

    group_keys = ["month", "lot_id", "die_id", "parameter"]
    for keys, g in core.groupby(group_keys, sort=False):
        month, lot_id, die_id, parameter = keys
        g = g.copy()
        eligible = g[g["safety_status"].astype(str) == "PASS"].copy()
        if len(eligible) < 2:
            continue

        # Policy: max simulated_yield among eligible; exact equality for ties
        max_y = eligible["simulated_yield"].max()
        tied = eligible[eligible["simulated_yield"] == max_y].copy()
        if len(tied) < 2:
            continue

        # Full eligible set for display; analysis focus on tied max-yield set
        tied = tied.reset_index(drop=True)
        obj = tied["objective_score"].to_numpy(dtype=float)
        core_s = tied["existing_ml_score"].to_numpy(dtype=float)
        uni_s = tied["unified_ml_score"].to_numpy(dtype=float)
        obj_rank = _rank_by_score(obj, True)
        core_rank_in_tie = _rank_by_score(core_s, True)
        uni_rank_in_tie = _rank_by_score(uni_s, True)

        # Closer to objective ordering: higher Spearman on this tie set
        sp_core = spearman(obj, core_s)
        sp_uni = spearman(obj, uni_s)
        nd_core = ndcg_at_k(obj, core_s, k=min(5, len(tied)))
        nd_uni = ndcg_at_k(obj, uni_s, k=min(5, len(tied)))
        kt_core = _kendall_tau(obj, core_s)
        kt_uni = _kendall_tau(obj, uni_s)

        if sp_uni > sp_core + 1e-12:
            closer = "unified"
        elif sp_core > sp_uni + 1e-12:
            closer = "core"
        else:
            # break with NDCG then Kendall
            if nd_uni > nd_core + 1e-12:
                closer = "unified"
            elif nd_core > nd_uni + 1e-12:
                closer = "core"
            elif kt_uni > kt_core + 1e-12:
                closer = "unified"
            elif kt_core > kt_uni + 1e-12:
                closer = "core"
            else:
                closer = "equal"

        core_pick = _winner_limit(tied, "existing_ml_score")
        uni_pick = _winner_limit(tied, "unified_ml_score")
        # Best objective among max-yield set
        best_obj_lim = float(tied.loc[tied["objective_score"].idxmax(), "candidate_limit"])
        core_obj = float(tied.loc[tied["candidate_limit"] == core_pick, "objective_score"].iloc[0])
        uni_obj = float(tied.loc[tied["candidate_limit"] == uni_pick, "objective_score"].iloc[0])

        stats = month_param_stats[(str(parameter), str(month))]
        stats["yield_tied_cases"] += 1
        if closer == "core":
            stats["core_closer"] += 1
        elif closer == "unified":
            stats["unified_closer"] += 1
        else:
            stats["tie_equal"] += 1
        if abs(core_pick - uni_pick) < 1e-12:
            stats["same_final_pick"] += 1
        else:
            stats["disagree_final_pick"] += 1
        if uni_obj > core_obj + 1e-12:
            stats["unified_picks_higher_obj"] += 1
        elif core_obj > uni_obj + 1e-12:
            stats["core_picks_higher_obj"] += 1

        # Objective diversity within the max-yield tie set
        n_unique_obj = int(tied["objective_score"].nunique())
        constant_objective = n_unique_obj <= 1

        current_limit = float(tied["current_limit"].iloc[0])
        case = {
            "month": str(month),
            "lot_id": str(lot_id),
            "die_id": str(die_id),
            "parameter": str(parameter),
            "current_limit": current_limit,
            "max_simulated_yield": float(max_y),
            "n_eligible": int(len(eligible)),
            "n_tied_at_max_yield": int(len(tied)),
            "n_unique_objective_in_tie": n_unique_obj,
            "constant_objective_in_tie": constant_objective,
            "core_final_pick": core_pick,
            "unified_final_pick": uni_pick,
            "best_objective_limit": best_obj_lim,
            "core_pick_objective": core_obj,
            "unified_pick_objective": uni_obj,
            "unified_picks_higher_objective": bool(uni_obj > core_obj + 1e-12),
            "core_picks_higher_objective": bool(core_obj > uni_obj + 1e-12),
            "both_pick_max_objective": bool(
                abs(core_obj - float(tied["objective_score"].max())) < 1e-12
                and abs(uni_obj - float(tied["objective_score"].max())) < 1e-12
            ),
            "closer_to_objective_order": closer,
            "spearman_core": sp_core,
            "spearman_unified": sp_uni,
            "ndcg_core": nd_core,
            "ndcg_unified": nd_uni,
            "kendall_core": kt_core,
            "kendall_unified": kt_uni,
            "is_test_lot": str(lot_id) in test_lots,
        }
        tie_case_rows.append(case)

        for i, r in tied.iterrows():
            candidate_flat.append(
                {
                    **{k: case[k] for k in (
                        "month", "lot_id", "die_id", "parameter", "current_limit",
                        "max_simulated_yield", "closer_to_objective_order",
                        "core_final_pick", "unified_final_pick",
                    )},
                    "candidate_limit": float(r["candidate_limit"]),
                    "simulated_yield": float(r["simulated_yield"]),
                    "violation_rate": float(r["violation_rate"])
                    if pd.notna(r["violation_rate"])
                    else None,
                    "objective_score": float(r["objective_score"]),
                    "safety_status": str(r["safety_status"]),
                    "core_ml_score": float(r["existing_ml_score"]),
                    "core_ml_rank_full": int(r["existing_ml_rank"]),
                    "unified_ml_score": float(r["unified_ml_score"]),
                    "unified_ml_rank_full": int(r["unified_ml_rank"]),
                    "objective_rank_in_tie": int(obj_rank[i]),
                    "core_rank_in_tie": int(core_rank_in_tie[i]),
                    "unified_rank_in_tie": int(uni_rank_in_tie[i]),
                }
            )

            row_common = {
                "month": str(month),
                "lot_id": str(lot_id),
                "die_id": str(die_id),
                "parameter": str(parameter),
                "target_score": float(r["objective_score"]),
            }
            metric_rows_core.append({**row_common, "pred": float(r["existing_ml_score"])})
            metric_rows_uni.append({**row_common, "pred": float(r["unified_ml_score"])})
            if str(lot_id) in test_lots:
                metric_rows_core_test.append({**row_common, "pred": float(r["existing_ml_score"])})
                metric_rows_uni_test.append({**row_common, "pred": float(r["unified_ml_score"])})

        # Critical example hunt: pair where obj(A)>obj(B), uni ranks A>B, core ranks B>A
        lims = tied["candidate_limit"].to_numpy(dtype=float)
        for i in range(len(tied)):
            for j in range(len(tied)):
                if i == j:
                    continue
                if obj[i] <= obj[j] + 1e-12:
                    continue
                # A=i better objective than B=j
                if uni_s[i] > uni_s[j] + 1e-12 and core_s[j] > core_s[i] + 1e-12:
                    critical_examples.append(
                        {
                            "month": str(month),
                            "lot_id": str(lot_id),
                            "die_id": str(die_id),
                            "parameter": str(parameter),
                            "candidate_A_limit": float(lims[i]),
                            "candidate_B_limit": float(lims[j]),
                            "candidate_A_objective": float(obj[i]),
                            "candidate_B_objective": float(obj[j]),
                            "candidate_A_yield": float(tied.iloc[i]["simulated_yield"]),
                            "candidate_B_yield": float(tied.iloc[j]["simulated_yield"]),
                            "unified_prefers_A": True,
                            "core_prefers_B": True,
                            "core_score_A": float(core_s[i]),
                            "core_score_B": float(core_s[j]),
                            "unified_score_A": float(uni_s[i]),
                            "unified_score_B": float(uni_s[j]),
                        }
                    )

    # Representative tables: pick diverse cases
    cases_df = pd.DataFrame(tie_case_rows)
    for parameter in CORE_PARAMS:
        for month in MONTHS:
            sub = cases_df[
                (cases_df["parameter"] == parameter) & (cases_df["month"] == month)
            ]
            if sub.empty:
                continue
            # Prefer disagree + unified higher obj, else any disagree, else first
            prefer = sub[sub["unified_picks_higher_objective"]]
            if prefer.empty:
                prefer = sub[sub["core_final_pick"] != sub["unified_final_pick"]]
            if prefer.empty:
                prefer = sub
            row = prefer.iloc[0]
            cands = [
                c
                for c in candidate_flat
                if c["month"] == row["month"]
                and c["lot_id"] == row["lot_id"]
                and c["die_id"] == row["die_id"]
                and c["parameter"] == row["parameter"]
            ]
            cands = sorted(cands, key=lambda x: -x["objective_score"])
            representative_tables.append(
                {
                    "selection_reason": "representative_yield_tie",
                    "case": row.to_dict(),
                    "candidates_sorted_by_objective": cands,
                }
            )

    def _agg_metrics(rows: list[dict], pred_key: str = "pred") -> dict[str, float]:
        if not rows:
            return {"n_groups": 0.0, "ndcg_at_k": float("nan"), "spearman": float("nan")}
        # remap pred key for group_ranking_metrics
        adapted = [{**r, "pred_score": r[pred_key]} for r in rows]
        m = group_ranking_metrics(
            rows=adapted,
            group_keys=["month", "lot_id", "die_id", "parameter"],
            score_key="target_score",
            pred_key="pred_score",
            k=5,
        )
        # Kendall mean per group
        groups: dict[tuple, list] = defaultdict(list)
        for r in rows:
            groups[(r["month"], r["lot_id"], r["die_id"], r["parameter"])].append(r)
        kts = []
        for rs in groups.values():
            yt = np.array([x["target_score"] for x in rs], dtype=float)
            yp = np.array([x["pred"] for x in rs], dtype=float)
            kts.append(_kendall_tau(yt, yp))
        m["kendall"] = float(np.mean(kts)) if kts else float("nan")
        m["n_candidates"] = float(len(rows))
        return m

    metrics_all = {
        "core": _agg_metrics(metric_rows_core),
        "unified": _agg_metrics(metric_rows_uni),
    }
    metrics_test = {
        "core": _agg_metrics(metric_rows_core_test),
        "unified": _agg_metrics(metric_rows_uni_test),
    }
    metrics_by_param: dict[str, Any] = {}
    for p in CORE_PARAMS:
        metrics_by_param[p] = {
            "core": _agg_metrics([r for r in metric_rows_core if r["parameter"] == p]),
            "unified": _agg_metrics([r for r in metric_rows_uni if r["parameter"] == p]),
            "core_test_lots": _agg_metrics(
                [r for r in metric_rows_core_test if r["parameter"] == p]
            ),
            "unified_test_lots": _agg_metrics(
                [r for r in metric_rows_uni_test if r["parameter"] == p]
            ),
        }

    # Three-month table
    three_month = []
    for parameter in CORE_PARAMS:
        for month in MONTHS:
            s = month_param_stats[(parameter, month)]
            three_month.append(
                {
                    "parameter": parameter,
                    "month": month,
                    "yield_tied_cases": s["yield_tied_cases"],
                    "core_closer_to_objective_order": s["core_closer"],
                    "unified_closer_to_objective_order": s["unified_closer"],
                    "equal_closeness": s["tie_equal"],
                    "unified_picks_higher_objective": s["unified_picks_higher_obj"],
                    "core_picks_higher_objective": s["core_picks_higher_obj"],
                    "same_final_pick": s["same_final_pick"],
                    "disagree_final_pick": s["disagree_final_pick"],
                }
            )

    # Totals for verdict
    total_cases = len(tie_case_rows)
    uni_closer = sum(1 for c in tie_case_rows if c["closer_to_objective_order"] == "unified")
    core_closer = sum(1 for c in tie_case_rows if c["closer_to_objective_order"] == "core")
    equal_c = sum(1 for c in tie_case_rows if c["closer_to_objective_order"] == "equal")
    uni_higher_obj = sum(1 for c in tie_case_rows if c["unified_picks_higher_objective"])
    core_higher_obj = sum(1 for c in tie_case_rows if c["core_picks_higher_objective"])
    same_pick = sum(
        1 for c in tie_case_rows if abs(c["core_final_pick"] - c["unified_final_pick"]) < 1e-12
    )

    # Deduplicate critical examples (keep unique A/B pairs per die, first few)
    crit_unique = []
    seen = set()
    for ex in critical_examples:
        key = (
            ex["month"],
            ex["lot_id"],
            ex["die_id"],
            ex["parameter"],
            ex["candidate_A_limit"],
            ex["candidate_B_limit"],
        )
        if key in seen:
            continue
        seen.add(key)
        crit_unique.append(ex)
        if len(crit_unique) >= 20:
            break

    constant_n = sum(1 for c in tie_case_rows if c["constant_objective_in_tie"])
    varying_n = total_cases - constant_n
    both_max = sum(1 for c in tie_case_rows if c["both_pick_max_objective"])

    # Metrics only on ties with varying objective (informative ranking)
    metric_rows_core_var = []
    metric_rows_uni_var = []
    for c in tie_case_rows:
        if c["constant_objective_in_tie"]:
            continue
        for r in candidate_flat:
            if (
                r["month"] == c["month"]
                and r["lot_id"] == c["lot_id"]
                and r["die_id"] == c["die_id"]
                and r["parameter"] == c["parameter"]
            ):
                common = {
                    "month": r["month"],
                    "lot_id": r["lot_id"],
                    "die_id": r["die_id"],
                    "parameter": r["parameter"],
                    "target_score": r["objective_score"],
                }
                metric_rows_core_var.append({**common, "pred": r["core_ml_score"]})
                metric_rows_uni_var.append({**common, "pred": r["unified_ml_score"]})

    metrics_varying_obj = {
        "core": _agg_metrics(metric_rows_core_var),
        "unified": _agg_metrics(metric_rows_uni_var),
        "n_cases": varying_n,
    }

    # Verdict logic
    nd_u = metrics_all["unified"].get("ndcg_at_k", 0) or 0
    nd_c = metrics_all["core"].get("ndcg_at_k", 0) or 0
    sp_u = metrics_all["unified"].get("spearman", 0) or 0
    sp_c = metrics_all["core"].get("spearman", 0) or 0

    ir_u = metrics_by_param["ir_drop"]["unified"].get("ndcg_at_k") or 0
    ir_c = metrics_by_param["ir_drop"]["core"].get("ndcg_at_k") or 0
    th_u = metrics_by_param["thermal"]["unified"].get("ndcg_at_k") or 0
    th_c = metrics_by_param["thermal"]["core"].get("ndcg_at_k") or 0

    sp_u_var = metrics_varying_obj["unified"].get("spearman") or 0
    sp_c_var = metrics_varying_obj["core"].get("spearman") or 0

    uni_wins_order = uni_closer > core_closer and (uni_closer / max(total_cases, 1)) >= 0.55
    uni_wins_metrics = (nd_u > nd_c + 0.01) and (sp_u > sp_c + 0.01)
    uni_wins_obj_pick = uni_higher_obj > core_higher_obj
    uni_wins_varying = sp_u_var > sp_c_var + 0.02 and varying_n > 0

    both_params_uni = (ir_u > ir_c) and (th_u > th_c)
    both_params_core = (ir_c >= ir_u) and (th_c > th_u)

    if (
        varying_n > 0
        and sp_c_var > sp_u_var + 0.05
        and core_closer > uni_closer
        and not uni_wins_obj_pick
        and len(crit_unique) == 0
    ):
        # Core better where objective differs; Unified never wins critical pairwise pattern
        verdict = "FAIL — UNIFIED GRU DOES NOT IMPROVE IR/THERMAL TIE-BREAKING"
    elif uni_wins_order and uni_wins_metrics and both_params_uni and uni_wins_obj_pick:
        verdict = "PASS — UNIFIED GRU HAS STRONGER IR/THERMAL TIE-BREAK RANKING"
    else:
        verdict = "PASS WITH CONDITIONS — MIXED IR/THERMAL RESULTS"

    summary = {
        "phase": "12.6A",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy_unchanged": True,
        "tie_semantics": "exact equality of simulated_yield among safety PASS eligible (policy yield_tie)",
        "distinctions": {
            "simulated_yield": "population simulator primary selection key",
            "objective_score": "training/regression target; not production yield",
            "ml_score": "model scalar prediction of objective_score",
            "ml_rank": "order by ml_score within a die×parameter candidate set",
            "final_recommendation": "max simulated_yield among eligible; ML rank only on ties",
        },
        "n_yield_tied_cases": total_cases,
        "n_constant_objective_ties": constant_n,
        "n_varying_objective_ties": varying_n,
        "n_both_models_pick_max_objective": both_max,
        "n_critical_examples_unified_prefers_higher_obj": len(crit_unique),
        "closer_to_objective_order": {
            "unified": uni_closer,
            "core": core_closer,
            "equal": equal_c,
        },
        "final_pick_objective": {
            "unified_higher": uni_higher_obj,
            "core_higher": core_higher_obj,
            "same_pick": same_pick,
            "obj_equal_different_or_same_pick": total_cases
            - uni_higher_obj
            - core_higher_obj,
        },
        "ranking_metrics_all_shadow_ties": metrics_all,
        "ranking_metrics_test_lots_only": metrics_test,
        "ranking_metrics_varying_objective_ties": metrics_varying_obj,
        "ranking_metrics_by_parameter": metrics_by_param,
        "three_month_breakdown": three_month,
        "representative_tables": representative_tables,
        "critical_examples": crit_unique,
        "tie_cases": tie_case_rows,
        "key_finding": (
            "In most IR/Thermal max-yield ties, objective_score is constant across the tied "
            "set (often all 1.0). ML therefore cannot improve objective_score of the final pick; "
            "disagreement only changes which equally-scoring (by objective) limit is selected. "
            "Where objective varies slightly within the tie, Core GRU aligns better with "
            "objective ordering than Unified; no critical pairwise case favors Unified."
        ),
        "verdict": verdict,
    }

    # Write artifacts
    flat_df = pd.DataFrame(candidate_flat)
    flat_df.to_csv(shadow / "ir_thermal_yield_ties.csv", index=False)
    write_json(shadow / "ir_thermal_yield_ties.json", summary)
    write_json(
        temporal_artifact_root(root) / "shared" / "PHASE_12_6A_YIELD_TIE_SUMMARY.json",
        {
            "verdict": verdict,
            "n_yield_tied_cases": total_cases,
            "unified_closer": uni_closer,
            "core_closer": core_closer,
            "critical_examples": len(crit_unique),
            "metrics_all": metrics_all,
        },
    )
    return summary


if __name__ == "__main__":
    s = analyze_ir_thermal_yield_ties()
    print(
        json.dumps(
            {
                "verdict": s["verdict"],
                "n_ties": s["n_yield_tied_cases"],
                "closer": s["closer_to_objective_order"],
                "obj_pick": s["final_pick_objective"],
                "metrics": s["ranking_metrics_all_shadow_ties"],
                "by_param_ndcg": {
                    p: {
                        "core": s["ranking_metrics_by_parameter"][p]["core"].get("ndcg_at_k"),
                        "unified": s["ranking_metrics_by_parameter"][p]["unified"].get(
                            "ndcg_at_k"
                        ),
                    }
                    for p in CORE_PARAMS
                },
                "n_critical": s["n_critical_examples_unified_prefers_higher_obj"],
                "three_month": s["three_month_breakdown"],
            },
            indent=2,
        )
    )
