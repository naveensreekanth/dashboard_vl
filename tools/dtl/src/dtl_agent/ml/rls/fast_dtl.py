"""Fast offline DTL evaluation with score-independent preprocessing (shadow only).

Preserves the same decision semantics as ``eval_metrics.decide_group``:

    rank by score desc (mergesort / stable)
      → Top-N + CURRENT
      → finite simulated_yield eligibility
      → max simulated_yield
      → best ml_rank / score as tie-break
      → recommended limit

Does not modify production recommendation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dtl_agent.ml.rls.eval_metrics import TOP_N

GROUP_KEYS = ("production_month", "lot_id", "die_id", "parameter")


@dataclass
class SharedGroupPack:
    """Score-independent per-group candidate arrays."""

    # Group metadata (length = n_groups)
    production_month: np.ndarray
    lot_id: np.ndarray
    die_id: np.ndarray
    parameter: np.ndarray
    # Row index ranges into flat arrays: starts[i]:ends[i]
    starts: np.ndarray
    ends: np.ndarray
    # Flat candidate arrays (length = n_rows, group-contiguous)
    candidate_limit: np.ndarray
    current_limit: np.ndarray
    simulated_yield: np.ndarray
    is_current: np.ndarray
    target_score: np.ndarray
    # Oracle among full group (max yield; first max if ties)
    oracle_limit: np.ndarray
    oracle_yield: np.ndarray
    # Original dataframe row order mapping: pack flat index → original iloc
    # After build, rows are reordered group-contiguous; keep permute to map scores.
    permute: np.ndarray  # original positions in group-contiguous order


def build_shared_group_pack(df: pd.DataFrame) -> SharedGroupPack:
    """Build shared DTL structures once per month frame."""
    n = len(df)
    # Stable group assignment preserving first-seen group order
    keys = [df[k].astype(str).to_numpy() for k in GROUP_KEYS]
    # Build group labels via factorize on combined key
    combo = (
        keys[0]
        + "\0"
        + keys[1]
        + "\0"
        + keys[2]
        + "\0"
        + keys[3]
    )
    codes, uniques = pd.factorize(combo, sort=False)
    n_groups = len(uniques)

    # Contiguous permute: group by codes in first-seen order
    order = np.argsort(codes, kind="mergesort")
    codes_sorted = codes[order]
    # starts/ends
    starts = np.zeros(n_groups, dtype=np.int64)
    ends = np.zeros(n_groups, dtype=np.int64)
    # find boundaries
    change = np.flatnonzero(np.diff(codes_sorted)) + 1
    bounds = np.concatenate([[0], change, [n]])
    for gi in range(n_groups):
        starts[gi] = bounds[gi]
        ends[gi] = bounds[gi + 1]

    # Map unique order: codes_sorted[starts] gives group code; rebuild meta from first row
    cand = df["candidate_limit"].to_numpy(dtype=np.float64)[order]
    cur = df["current_limit"].to_numpy(dtype=np.float64)[order]
    yld = df["simulated_yield"].to_numpy(dtype=np.float64)[order]
    tgt = df["target_score"].to_numpy(dtype=np.float64)[order]
    tight = df["tighten_or_loosen"].astype(str).to_numpy()[order]
    is_cur = (tight == "CURRENT") | (np.abs(cand - cur) < 1e-12)

    pm = df["production_month"].astype(str).to_numpy()[order]
    lot = df["lot_id"].astype(str).to_numpy()[order]
    die = df["die_id"].astype(str).to_numpy()[order]
    param = df["parameter"].astype(str).to_numpy()[order]

    g_pm = np.empty(n_groups, dtype=object)
    g_lot = np.empty(n_groups, dtype=object)
    g_die = np.empty(n_groups, dtype=object)
    g_param = np.empty(n_groups, dtype=object)
    oracle_lim = np.empty(n_groups, dtype=np.float64)
    oracle_y = np.empty(n_groups, dtype=np.float64)

    for gi in range(n_groups):
        s, e = int(starts[gi]), int(ends[gi])
        g_pm[gi] = pm[s]
        g_lot[gi] = lot[s]
        g_die[gi] = die[s]
        g_param[gi] = param[s]
        yy = yld[s:e]
        # idxmax: first maximum
        j = int(np.argmax(yy))
        oracle_y[gi] = float(yy[j])
        oracle_lim[gi] = float(cand[s + j])

    return SharedGroupPack(
        production_month=g_pm,
        lot_id=g_lot,
        die_id=g_die,
        parameter=g_param,
        starts=starts,
        ends=ends,
        candidate_limit=cand,
        current_limit=cur,
        simulated_yield=yld,
        is_current=is_cur,
        target_score=tgt,
        oracle_limit=oracle_lim,
        oracle_yield=oracle_y,
        permute=order.astype(np.int64),
    )


def _decide_one(
    *,
    scores: np.ndarray,
    cand: np.ndarray,
    cur: np.ndarray,
    yld: np.ndarray,
    is_cur: np.ndarray,
) -> tuple[float, str, float, float, int]:
    """Return (recommended_limit, decision, selected_yield, ml_score, ml_rank)."""
    n = len(scores)
    # Stable argsort descending (mergesort)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)

    # Top-N indices in original local coords
    top_idx = order[: min(TOP_N, n)]
    cur_idx = np.flatnonzero(is_cur)
    gated_idx = np.unique(np.concatenate([top_idx, cur_idx]))

    finite = np.isfinite(yld[gated_idx])
    elig_idx = gated_idx[finite]
    current_limit = float(cur[0])

    if elig_idx.size == 0:
        if cur_idx.size:
            i = int(cur_idx[0])
        else:
            i = 0
        return current_limit, "KEEP_CURRENT", float(yld[i]), float(scores[i]), int(ranks[i])

    # Sort eligible: yield desc, ml_rank asc, score desc — stable
    elig_y = yld[elig_idx]
    elig_r = ranks[elig_idx]
    elig_s = scores[elig_idx]
    # lexsort keys are last-major: sort by score desc, rank asc, yield desc
    # numpy lexsort: last key is primary
    order_e = np.lexsort((-elig_s, elig_r, -elig_y))
    win_local = int(elig_idx[order_e[0]])
    rec = float(cand[win_local])
    if abs(rec - current_limit) < 1e-12:
        return current_limit, "KEEP_CURRENT", float(yld[win_local]), float(scores[win_local]), int(ranks[win_local])
    return rec, "RECOMMEND", float(yld[win_local]), float(scores[win_local]), int(ranks[win_local])


def decide_all_fast(pack: SharedGroupPack, scores_original_order: np.ndarray) -> pd.DataFrame:
    """Apply offline DTL policy using precomputed pack + score vector (original df order)."""
    scores = np.asarray(scores_original_order, dtype=np.float64)[pack.permute]
    rows: list[dict[str, Any]] = []
    for gi in range(len(pack.starts)):
        s, e = int(pack.starts[gi]), int(pack.ends[gi])
        rec, decision, sel_y, ml_s, ml_r = _decide_one(
            scores=scores[s:e],
            cand=pack.candidate_limit[s:e],
            cur=pack.current_limit[s:e],
            yld=pack.simulated_yield[s:e],
            is_cur=pack.is_current[s:e],
        )
        rows.append(
            {
                "production_month": pack.production_month[gi],
                "lot_id": pack.lot_id[gi],
                "die_id": pack.die_id[gi],
                "parameter": pack.parameter[gi],
                "current_limit": float(pack.current_limit[s]),
                "recommended_limit": rec,
                "decision": decision,
                "simulated_yield": sel_y,
                "ml_score": ml_s,
                "ml_rank": ml_r,
                "n_candidates": int(e - s),
                "oracle_limit": float(pack.oracle_limit[gi]),
                "oracle_yield": float(pack.oracle_yield[gi]),
            }
        )
    return pd.DataFrame(rows)


def ranking_metrics_fast(pack: SharedGroupPack, scores_original_order: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores_original_order, dtype=np.float64)[pack.permute]
    targets = pack.target_score
    top1 = []
    topk = []
    spears = []
    k = TOP_N
    for gi in range(len(pack.starts)):
        s, e = int(pack.starts[gi]), int(pack.ends[gi])
        if e - s < 2:
            continue
        yt = targets[s:e]
        yp = scores[s:e]
        cand = pack.candidate_limit[s:e]
        true_best = float(cand[int(np.argmax(yt))])
        pred_best = float(cand[int(np.argmax(yp))])
        top1.append(1.0 if abs(true_best - pred_best) < 1e-9 else 0.0)
        true_order = np.argsort(-yt, kind="mergesort")[:k]
        pred_order = np.argsort(-yp, kind="mergesort")[:k]
        true_top = set(cand[true_order].tolist())
        pred_top = set(cand[pred_order].tolist())
        topk.append(len(true_top & pred_top) / float(k))
        rt = pd.Series(yt).rank().to_numpy()
        rp = pd.Series(yp).rank().to_numpy()
        if np.std(rt) < 1e-12 or np.std(rp) < 1e-12:
            spears.append(0.0)
        else:
            spears.append(float(np.corrcoef(rt, rp)[0, 1]))
    return {
        "top1_candidate_agreement": float(np.mean(top1)) if top1 else float("nan"),
        "topk_candidate_overlap": float(np.mean(topk)) if topk else float("nan"),
        "mean_spearman": float(np.mean(spears)) if spears else float("nan"),
        "n_groups": int(len(top1)),
    }


def regression_fast(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "n": int(len(y)),
    }


def compare_decisions(
    gru_dec: pd.DataFrame,
    model_dec: pd.DataFrame,
) -> dict[str, Any]:
    """Compare two decision tables that already include oracle columns."""
    m = gru_dec.merge(
        model_dec,
        on=["production_month", "lot_id", "die_id", "parameter"],
        suffixes=("_gru", "_model"),
    )
    lim_g = m["recommended_limit_gru"].to_numpy(dtype=float)
    lim_m = m["recommended_limit_model"].to_numpy(dtype=float)
    diff = lim_m - lim_g
    agree = np.isclose(lim_g, lim_m, rtol=0.0, atol=1e-9)
    y_g = m["simulated_yield_gru"].to_numpy(dtype=float)
    y_m = m["simulated_yield_model"].to_numpy(dtype=float)
    o_lim = m["oracle_limit_gru"].to_numpy(dtype=float)  # same oracle either side
    # Yield-tie: from GRU gated perspective we approximate using selected yield==oracle
    # Better: use pack; here use decision tables only.
    # For tie rate we need pack; caller supplies yield_tie_rate separately if needed.

    oracle_agree_g = np.isclose(lim_g, o_lim, rtol=0.0, atol=1e-9)
    oracle_agree_m = np.isclose(lim_m, o_lim, rtol=0.0, atol=1e-9)

    changed = ~agree
    n_changed = int(np.sum(changed))
    looser = tighter = toward = away = 0
    if n_changed:
        dg = lim_g[changed]
        dm = lim_m[changed]
        oo = o_lim[changed]
        looser = int(np.sum(dm > dg + 1e-9))
        tighter = int(np.sum(dm < dg - 1e-9))
        toward = int(np.sum(np.abs(dm - oo) < np.abs(dg - oo) - 1e-9))
        away = int(np.sum(np.abs(dm - oo) > np.abs(dg - oo) + 1e-9))

    return {
        "n": int(len(m)),
        "recommended_limit_agreement": float(np.mean(agree)),
        "mean_abs_dtl_difference": float(np.mean(np.abs(diff))),
        "max_abs_limit_delta": float(np.max(np.abs(diff))) if len(diff) else float("nan"),
        "over_limit_rate": float(np.mean(diff > 1e-9)),
        "under_limit_rate": float(np.mean(diff < -1e-9)),
        "decision_agreement": float(
            np.mean(m["decision_gru"].astype(str) == m["decision_model"].astype(str))
        ),
        "mean_yield_gru": float(np.mean(y_g)),
        "mean_yield_model": float(np.mean(y_m)),
        "mean_selected_yield_minus_oracle": float(
            np.mean(y_m - m["oracle_yield_gru"].to_numpy(dtype=float))
        ),
        "oracle_limit_agreement_gru": float(np.mean(oracle_agree_g)),
        "oracle_limit_agreement_model": float(np.mean(oracle_agree_m)),
        "tie_break_oracle_win_rate_gru": float(np.mean(oracle_agree_g)),
        "tie_break_oracle_win_rate_model": float(np.mean(oracle_agree_m)),
        "pct_groups_limit_changed": float(np.mean(changed)),
        "n_groups_limit_changed": n_changed,
        "pct_changed_looser": float(looser / n_changed) if n_changed else float("nan"),
        "pct_changed_tighter": float(tighter / n_changed) if n_changed else float("nan"),
        "pct_changed_toward_oracle": float(toward / n_changed) if n_changed else float("nan"),
        "pct_changed_away_from_oracle": float(away / n_changed) if n_changed else float("nan"),
        "mean_abs_limit_delta_when_changed": (
            float(np.mean(np.abs(diff[changed]))) if n_changed else float("nan")
        ),
    }


def yield_tie_rate_fast(pack: SharedGroupPack, scores_original_order: np.ndarray) -> float:
    """Fraction of groups where gated Top-N+CURRENT has ≥2 candidates at max yield."""
    scores = np.asarray(scores_original_order, dtype=np.float64)[pack.permute]
    flags = []
    for gi in range(len(pack.starts)):
        s, e = int(pack.starts[gi]), int(pack.ends[gi])
        sc = scores[s:e]
        yld = pack.simulated_yield[s:e]
        is_cur = pack.is_current[s:e]
        order = np.argsort(-sc, kind="mergesort")
        top_idx = order[: min(TOP_N, e - s)]
        cur_idx = np.flatnonzero(is_cur)
        gated = np.unique(np.concatenate([top_idx, cur_idx]))
        gy = yld[gated]
        gy = gy[np.isfinite(gy)]
        if gy.size == 0:
            continue
        mx = float(np.max(gy))
        flags.append(1.0 if np.sum(np.isclose(gy, mx, atol=1e-12)) >= 2 else 0.0)
    return float(np.mean(flags)) if flags else float("nan")


def evaluate_scores(
    pack: SharedGroupPack,
    *,
    scores: np.ndarray,
    targets: np.ndarray,
    gru_scores: np.ndarray,
    gru_dec: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Full metric bundle for one score vector against shared pack + optional cached GRU decisions."""
    reg = regression_fast(targets, scores)
    rank = ranking_metrics_fast(pack, scores)
    model_dec = decide_all_fast(pack, scores)
    if gru_dec is None:
        gru_dec = decide_all_fast(pack, gru_scores)
    cmp = compare_decisions(gru_dec, model_dec)
    if len(scores) > 1 and np.std(scores) > 1e-12 and np.std(gru_scores) > 1e-12:
        corr = float(np.corrcoef(gru_scores, scores)[0, 1])
    else:
        corr = float("nan")
    ytr = yield_tie_rate_fast(pack, scores)
    return {
        "regression": reg,
        "ranking": rank,
        "score_correlation_with_gru": corr,
        "dtl_vs_gru": cmp,
        "decisions": model_dec,
        "yield_tie_rate": ytr,
        "mean_selected_yield": float(model_dec["simulated_yield"].mean()),
        "oracle_tie_break_win_rate": float(cmp["tie_break_oracle_win_rate_model"]),
        "oracle_limit_agreement": float(cmp["oracle_limit_agreement_model"]),
        "selected_yield_delta_vs_oracle": float(cmp["mean_selected_yield_minus_oracle"]),
    }
