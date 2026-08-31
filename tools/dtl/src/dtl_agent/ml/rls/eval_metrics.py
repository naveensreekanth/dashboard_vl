"""Offline DTL-policy evaluation shared by GRU and RLS (shadow only).

Replicates the production selection rule on scored candidate tables:

  1. Rank by ml_score descending
  2. Keep Top-N + CURRENT
  3. Among those, pick max simulated_yield; tie-break by best ml_rank

Does not call or modify ``recommendation/pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TOP_N = 5
GROUP_KEYS = ["production_month", "lot_id", "die_id", "parameter"]


@dataclass
class DTLDecision:
    production_month: str
    lot_id: str
    die_id: str
    parameter: str
    current_limit: float
    recommended_limit: float
    decision: str
    simulated_yield: float
    ml_score: float
    ml_rank: int
    n_candidates: int


def _is_current(row: pd.Series) -> bool:
    return str(row["tighten_or_loosen"]) == "CURRENT" or abs(
        float(row["candidate_limit"]) - float(row["current_limit"])
    ) < 1e-12


def decide_group(g: pd.DataFrame, *, score_col: str = "ml_score") -> DTLDecision:
    g = g.copy()
    g = g.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    g["ml_rank"] = np.arange(1, len(g) + 1)
    g["ml_score"] = g[score_col].astype(float)

    top = g.head(TOP_N)
    cur = g[g.apply(_is_current, axis=1)]
    gated = pd.concat([top, cur], ignore_index=True).drop_duplicates(
        subset=["candidate_limit"], keep="first"
    )

    # Eligible = gated rows with finite yield (simulation evidence present).
    eligible = gated[np.isfinite(gated["simulated_yield"].astype(float))].copy()
    current_limit = float(g.iloc[0]["current_limit"])
    if eligible.empty:
        cur_row = cur.iloc[0] if len(cur) else g.iloc[0]
        return DTLDecision(
            production_month=str(g.iloc[0]["production_month"]),
            lot_id=str(g.iloc[0]["lot_id"]),
            die_id=str(g.iloc[0]["die_id"]),
            parameter=str(g.iloc[0]["parameter"]),
            current_limit=current_limit,
            recommended_limit=current_limit,
            decision="KEEP_CURRENT",
            simulated_yield=float(cur_row["simulated_yield"]),
            ml_score=float(cur_row[score_col]),
            ml_rank=int(cur_row["ml_rank"]),
            n_candidates=len(g),
        )

    eligible = eligible.sort_values(
        ["simulated_yield", "ml_rank", score_col],
        ascending=[False, True, False],
        kind="mergesort",
    )
    win = eligible.iloc[0]
    rec_limit = float(win["candidate_limit"])
    if abs(rec_limit - current_limit) < 1e-12:
        decision = "KEEP_CURRENT"
        rec_limit = current_limit
    else:
        decision = "RECOMMEND"
    return DTLDecision(
        production_month=str(g.iloc[0]["production_month"]),
        lot_id=str(g.iloc[0]["lot_id"]),
        die_id=str(g.iloc[0]["die_id"]),
        parameter=str(g.iloc[0]["parameter"]),
        current_limit=current_limit,
        recommended_limit=rec_limit,
        decision=decision,
        simulated_yield=float(win["simulated_yield"]),
        ml_score=float(win[score_col]),
        ml_rank=int(win["ml_rank"]),
        n_candidates=len(g),
    )


def decide_all(df: pd.DataFrame, *, score_col: str = "ml_score") -> pd.DataFrame:
    rows: list[dict] = []
    for _, g in df.groupby(GROUP_KEYS, sort=False):
        d = decide_group(g, score_col=score_col)
        rows.append(d.__dict__)
    return pd.DataFrame(rows)


def compare_dtl(a: pd.DataFrame, b: pd.DataFrame, *, a_name: str, b_name: str) -> dict:
    """Compare two DTL decision tables on the same keys."""
    keys = GROUP_KEYS
    m = a.merge(b, on=keys, suffixes=(f"_{a_name}", f"_{b_name}"))
    if m.empty:
        return {"n": 0}
    lim_a = m[f"recommended_limit_{a_name}"].to_numpy(dtype=float)
    lim_b = m[f"recommended_limit_{b_name}"].to_numpy(dtype=float)
    agree = np.isclose(lim_a, lim_b, rtol=0.0, atol=1e-9)
    diff = lim_b - lim_a
    over = diff > 1e-9
    under = diff < -1e-9
    y_a = m[f"simulated_yield_{a_name}"].to_numpy(dtype=float)
    y_b = m[f"simulated_yield_{b_name}"].to_numpy(dtype=float)
    return {
        "n": int(len(m)),
        "recommended_limit_agreement": float(np.mean(agree)),
        "mean_abs_dtl_difference": float(np.mean(np.abs(diff))),
        "over_limit_rate": float(np.mean(over)),
        "under_limit_rate": float(np.mean(under)),
        "mean_yield_a": float(np.mean(y_a)),
        "mean_yield_b": float(np.mean(y_b)),
        "mean_yield_delta_b_minus_a": float(np.mean(y_b - y_a)),
        "decision_agreement": float(
            np.mean(m[f"decision_{a_name}"].astype(str) == m[f"decision_{b_name}"].astype(str))
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "n": int(len(y_true)),
    }


def ranking_metrics(df: pd.DataFrame, *, score_col: str, target_col: str = "target_score") -> dict:
    """Per-(lot,die,param,month) top-1 / top-k agreement vs target ranking."""
    top1 = []
    topk = []
    spears = []
    k = TOP_N
    for _, g in df.groupby(GROUP_KEYS, sort=False):
        if len(g) < 2:
            continue
        true_best = float(g.loc[g[target_col].idxmax(), "candidate_limit"])
        pred_best = float(g.loc[g[score_col].idxmax(), "candidate_limit"])
        top1.append(1.0 if abs(true_best - pred_best) < 1e-9 else 0.0)
        true_top = set(
            g.sort_values(target_col, ascending=False).head(k)["candidate_limit"].astype(float)
        )
        pred_top = set(
            g.sort_values(score_col, ascending=False).head(k)["candidate_limit"].astype(float)
        )
        topk.append(len(true_top & pred_top) / float(k))
        yt = g[target_col].to_numpy(dtype=float)
        yp = g[score_col].to_numpy(dtype=float)
        # Spearman via rank correlation
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


def _gated_frame(g: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    g = g.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    g = g.copy()
    g["ml_rank"] = np.arange(1, len(g) + 1)
    top = g.head(TOP_N)
    cur = g[g.apply(_is_current, axis=1)]
    gated = pd.concat([top, cur], ignore_index=True).drop_duplicates(
        subset=["candidate_limit"], keep="first"
    )
    return gated[np.isfinite(gated["simulated_yield"].astype(float))].copy()


def yield_tie_dtl_metrics(
    scored: pd.DataFrame,
    *,
    score_col: str,
) -> dict[str, float]:
    """Metrics where simulated_yield ties make ML rank decisive for DTL."""
    tie_flags: list[float] = []
    oracle_agree_all: list[float] = []
    oracle_agree_tied: list[float] = []
    yield_gap_vs_oracle: list[float] = []
    selected_yields: list[float] = []
    oracle_yields: list[float] = []

    for _, g in scored.groupby(GROUP_KEYS, sort=False):
        gated = _gated_frame(g, score_col=score_col)
        if gated.empty:
            continue
        max_y = float(gated["simulated_yield"].max())
        tied = gated[np.isclose(gated["simulated_yield"].astype(float), max_y, atol=1e-12)]
        is_tie = len(tied) >= 2
        tie_flags.append(1.0 if is_tie else 0.0)

        # Policy pick
        ordered = gated.sort_values(
            ["simulated_yield", "ml_rank", score_col],
            ascending=[False, True, False],
            kind="mergesort",
        )
        win = ordered.iloc[0]
        sel_lim = float(win["candidate_limit"])
        sel_y = float(win["simulated_yield"])
        selected_yields.append(sel_y)

        # Oracle among full group (max yield; tie → any max-yield limit)
        oracle_y = float(g["simulated_yield"].max())
        oracle_lim = float(
            g.loc[g["simulated_yield"].astype(float).idxmax(), "candidate_limit"]
        )
        oracle_yields.append(oracle_y)
        agree = 1.0 if abs(sel_lim - oracle_lim) < 1e-9 else 0.0
        oracle_agree_all.append(agree)
        if is_tie:
            oracle_agree_tied.append(agree)
        yield_gap_vs_oracle.append(sel_y - oracle_y)

    n = len(tie_flags)
    return {
        "n_groups": int(n),
        "yield_tie_rate": float(np.mean(tie_flags)) if n else float("nan"),
        "oracle_limit_agreement": float(np.mean(oracle_agree_all)) if n else float("nan"),
        "oracle_limit_agreement_among_yield_ties": (
            float(np.mean(oracle_agree_tied)) if oracle_agree_tied else float("nan")
        ),
        "n_yield_tied_groups": int(len(oracle_agree_tied)),
        "mean_selected_yield": float(np.mean(selected_yields)) if selected_yields else float("nan"),
        "mean_oracle_yield": float(np.mean(oracle_yields)) if oracle_yields else float("nan"),
        "mean_selected_yield_minus_oracle": (
            float(np.mean(yield_gap_vs_oracle)) if yield_gap_vs_oracle else float("nan")
        ),
        # Alias: among yield-tied gated groups, fraction picking oracle limit
        "tie_break_oracle_win_rate": (
            float(np.mean(oracle_agree_tied)) if oracle_agree_tied else float("nan")
        ),
    }


def compare_dtl_with_ties(
    scored_a: pd.DataFrame,
    scored_b: pd.DataFrame,
    *,
    score_col_a: str,
    score_col_b: str,
    a_name: str,
    b_name: str,
) -> dict:
    """Peer DTL compare plus yield-tie–conditioned limit agreement."""
    dtl_a = decide_all(scored_a, score_col=score_col_a)
    dtl_b = decide_all(scored_b, score_col=score_col_b)
    base = compare_dtl(dtl_a, dtl_b, a_name=a_name, b_name=b_name)
    ties_a = yield_tie_dtl_metrics(scored_a, score_col=score_col_a)
    ties_b = yield_tie_dtl_metrics(scored_b, score_col=score_col_b)

    # Among groups where gated yields tie for BOTH models' score orderings of Top-N,
    # compare whether they pick the same limit.
    tie_peer_agree: list[float] = []
    for keys, ga in scored_a.groupby(GROUP_KEYS, sort=False):
        gb = scored_b
        mask = True
        for k, v in zip(GROUP_KEYS, keys if isinstance(keys, tuple) else (keys,)):
            mask = mask & (gb[k] == v)
        gb = gb.loc[mask]
        if gb.empty:
            continue
        gated_a = _gated_frame(ga, score_col=score_col_a)
        gated_b = _gated_frame(gb, score_col=score_col_b)
        if gated_a.empty or gated_b.empty:
            continue
        max_a = float(gated_a["simulated_yield"].max())
        max_b = float(gated_b["simulated_yield"].max())
        tied_a = len(gated_a[np.isclose(gated_a["simulated_yield"].astype(float), max_a)]) >= 2
        tied_b = len(gated_b[np.isclose(gated_b["simulated_yield"].astype(float), max_b)]) >= 2
        if not (tied_a and tied_b):
            continue
        da = decide_group(ga, score_col=score_col_a)
        db = decide_group(gb, score_col=score_col_b)
        tie_peer_agree.append(
            1.0 if abs(da.recommended_limit - db.recommended_limit) < 1e-9 else 0.0
        )

    base["yield_tie_metrics_a"] = ties_a
    base["yield_tie_metrics_b"] = ties_b
    base["limit_agreement_among_mutual_yield_ties"] = (
        float(np.mean(tie_peer_agree)) if tie_peer_agree else float("nan")
    )
    base["n_mutual_yield_tied_groups"] = int(len(tie_peer_agree))
    base["tie_break_peer_agreement_rate"] = base["limit_agreement_among_mutual_yield_ties"]
    return base
