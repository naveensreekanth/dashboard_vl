"""GRU + RLS residual hybrid scoring (shadow only).

    hybrid_score = gru_score + rls_residual_prediction

RLS is trained on ``residual_target = target_score - gru_score``, not on
``target_score`` directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dtl_agent.ml.rls.eval_metrics import GROUP_KEYS, decide_all, ranking_metrics, regression_metrics
from dtl_agent.ml.rls.regressor import RLSRegressor


def build_residual_targets(
    target_score: np.ndarray,
    gru_score: np.ndarray,
) -> np.ndarray:
    """Residual label for the adapter: target - GRU base score."""
    y = np.asarray(target_score, dtype=np.float64).reshape(-1)
    g = np.asarray(gru_score, dtype=np.float64).reshape(-1)
    if y.shape != g.shape:
        raise ValueError("target_score and gru_score length mismatch")
    return y - g


def hybrid_scores(gru_score: np.ndarray, residual_prediction: np.ndarray) -> np.ndarray:
    """Combine frozen GRU base with RLS residual correction (alpha=1)."""
    return scaled_hybrid_scores(gru_score, residual_prediction, alpha=1.0)


def scaled_hybrid_scores(
    gru_score: np.ndarray,
    residual_prediction: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """``hybrid = gru + alpha * residual``. alpha=0 is pure GRU; alpha=1 is full hybrid."""
    g = np.asarray(gru_score, dtype=np.float64).reshape(-1)
    r = np.asarray(residual_prediction, dtype=np.float64).reshape(-1)
    if g.shape != r.shape:
        raise ValueError("gru_score and residual_prediction length mismatch")
    a = float(alpha)
    if not np.isfinite(a):
        raise ValueError("alpha must be finite")
    return g + a * r


def attach_hybrid_column(
    frame: pd.DataFrame,
    *,
    gru_col: str = "gru_score",
    residual_col: str = "residual_pred",
    out_col: str = "hybrid_score",
) -> pd.DataFrame:
    out = frame.copy()
    out[residual_col] = out[residual_col].astype(float)
    out[gru_col] = out[gru_col].astype(float)
    out[out_col] = hybrid_scores(out[gru_col].to_numpy(), out[residual_col].to_numpy())
    return out


def correction_diagnostics(
    *,
    target_score: np.ndarray,
    gru_score: np.ndarray,
    residual_pred: np.ndarray,
    hybrid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Candidate-level correction quality diagnostics."""
    y = np.asarray(target_score, dtype=np.float64)
    g = np.asarray(gru_score, dtype=np.float64)
    r = np.asarray(residual_pred, dtype=np.float64)
    h = np.asarray(hybrid, dtype=np.float64) if hybrid is not None else hybrid_scores(g, r)

    gru_err = np.abs(g - y)
    hyb_err = np.abs(h - y)
    improved = hyb_err < gru_err - 1e-12
    worsened = hyb_err > gru_err + 1e-12
    unchanged = ~(improved | worsened)

    return {
        "residual_mean": float(np.mean(r)),
        "residual_std": float(np.std(r)),
        "residual_mae": float(np.mean(np.abs(r - (y - g)))),
        "correction_mean": float(np.mean(r)),
        "correction_std": float(np.std(r)),
        "correction_abs_mean": float(np.mean(np.abs(r))),
        "correction_abs_p50": float(np.median(np.abs(r))),
        "correction_abs_p95": float(np.percentile(np.abs(r), 95)),
        "pct_candidates_improved": float(np.mean(improved)),
        "pct_candidates_worsened": float(np.mean(worsened)),
        "pct_candidates_unchanged_error": float(np.mean(unchanged)),
        "gru_mae": float(np.mean(gru_err)),
        "hybrid_mae": float(np.mean(hyb_err)),
        "n": int(len(y)),
    }


def dtl_change_diagnostics(
    scored: pd.DataFrame,
    *,
    gru_col: str = "gru_score",
    hybrid_col: str = "hybrid_score",
) -> dict[str, Any]:
    """Group-level DTL impact of hybrid vs GRU base."""
    gru_dec = decide_all(scored, score_col=gru_col)
    hyb_dec = decide_all(scored, score_col=hybrid_col)
    keys = GROUP_KEYS
    m = gru_dec.merge(hyb_dec, on=keys, suffixes=("_gru", "_hybrid"))

    lim_g = m["recommended_limit_gru"].to_numpy(dtype=float)
    lim_h = m["recommended_limit_hybrid"].to_numpy(dtype=float)
    changed = ~np.isclose(lim_g, lim_h, rtol=0.0, atol=1e-9)

    # Oracle limit per group from full candidate set
    oracle_rows: list[dict[str, float | str]] = []
    for _, g in scored.groupby(keys, sort=False):
        win = g.loc[g["simulated_yield"].astype(float).idxmax()]
        oracle_rows.append(
            {
                "production_month": str(g.iloc[0]["production_month"]),
                "lot_id": str(g.iloc[0]["lot_id"]),
                "die_id": str(g.iloc[0]["die_id"]),
                "parameter": str(g.iloc[0]["parameter"]),
                "oracle_limit": float(win["candidate_limit"]),
            }
        )
    oracle = pd.DataFrame(oracle_rows)
    mm = m.merge(oracle, on=keys)

    toward = 0
    away = 0
    looser = 0
    tighter = 0
    n_changed = int(np.sum(changed))
    for _, row in mm.loc[changed].iterrows():
        g_lim = float(row["recommended_limit_gru"])
        h_lim = float(row["recommended_limit_hybrid"])
        o_lim = float(row["oracle_limit"])
        d_g = abs(g_lim - o_lim)
        d_h = abs(h_lim - o_lim)
        if d_h < d_g - 1e-9:
            toward += 1
        elif d_h > d_g + 1e-9:
            away += 1
        if h_lim > g_lim + 1e-9:
            looser += 1
        elif h_lim < g_lim - 1e-9:
            tighter += 1

    return {
        "n_groups": int(len(m)),
        "pct_groups_limit_changed": float(np.mean(changed)),
        "n_groups_limit_changed": n_changed,
        "pct_changed_toward_oracle": float(toward / n_changed) if n_changed else float("nan"),
        "pct_changed_away_from_oracle": float(away / n_changed) if n_changed else float("nan"),
        "pct_changed_looser": float(looser / n_changed) if n_changed else float("nan"),
        "pct_changed_tighter": float(tighter / n_changed) if n_changed else float("nan"),
        "mean_abs_limit_delta_when_changed": (
            float(np.mean(np.abs(lim_h[changed] - lim_g[changed]))) if n_changed else float("nan")
        ),
    }


def score_frame_metrics(
    scored: pd.DataFrame,
    *,
    score_col: str,
    peer_col: str | None = None,
) -> dict[str, Any]:
    y = scored["target_score"].to_numpy(dtype=float)
    p = scored[score_col].to_numpy(dtype=float)
    out: dict[str, Any] = {
        "regression": regression_metrics(y, p),
        "ranking": ranking_metrics(scored, score_col=score_col),
    }
    if peer_col is not None:
        peer = scored[peer_col].to_numpy(dtype=float)
        if len(peer) > 1 and np.std(peer) > 1e-12 and np.std(p) > 1e-12:
            out["score_correlation"] = float(np.corrcoef(peer, p)[0, 1])
        else:
            out["score_correlation"] = float("nan")
    return out


def fit_residual_rls(
    X: np.ndarray,
    target_score: np.ndarray,
    gru_score: np.ndarray,
    *,
    forgetting_factor: float,
    delta: float,
) -> RLSRegressor:
    model = RLSRegressor(X.shape[1], forgetting_factor=forgetting_factor, delta=delta)
    resid = build_residual_targets(target_score, gru_score)
    model.fit(X, resid)
    return model
