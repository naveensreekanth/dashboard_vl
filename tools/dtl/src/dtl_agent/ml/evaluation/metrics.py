"""Ranking and regression metrics for Phase 7."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _dcg(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if rels.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0**rels - 1.0) / denom))


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    order = np.argsort(-y_pred)
    best = np.argsort(-y_true)
    dcg = _dcg(y_true[order], k)
    idcg = _dcg(y_true[best], k)
    if idcg <= 1e-12:
        return 1.0
    return float(dcg / idcg)


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tie_tol: float = 1e-6) -> float:
    n = len(y_true)
    if n < 2:
        return 1.0
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = y_true[i] - y_true[j]
            dp = y_pred[i] - y_pred[j]
            if abs(dt) <= tie_tol:
                total += 1
                correct += int(abs(dp) <= tie_tol)
            else:
                total += 1
                correct += int((dt > 0 and dp > 0) or (dt < 0 and dp < 0))
    return float(correct / total) if total else 1.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    # average ranks for ties
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = r
        i = j + 1
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 1.0
    rt = _rankdata(y_true)
    rp = _rankdata(y_pred)
    rt = rt - rt.mean()
    rp = rp - rp.mean()
    denom = np.sqrt(np.sum(rt**2) * np.sum(rp**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(rt * rp) / denom)


def group_ranking_metrics(
    *,
    rows: Iterable[dict],
    group_keys: list[str],
    score_key: str,
    pred_key: str,
    optimizer_choice_key: str | None = None,
    k: int = 5,
) -> dict[str, float]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_keys)].append(r)
    ndcgs = []
    pairs = []
    spears = []
    regrets = []
    y_regrets = []
    agree = []
    for rs in groups.values():
        yt = np.array([float(r[score_key]) for r in rs], dtype=float)
        yp = np.array([float(r[pred_key]) for r in rs], dtype=float)
        ndcgs.append(ndcg_at_k(yt, yp, min(k, len(rs))))
        pairs.append(pairwise_accuracy(yt, yp))
        spears.append(spearman(yt, yp))
        best_true = int(np.argmax(yt))
        best_pred = int(np.argmax(yp))
        regrets.append(float(yt[best_true] - yt[best_pred]))
        y_true = np.array([float(r.get("simulated_yield", 0.0)) for r in rs], dtype=float)
        y_regrets.append(float(y_true[best_true] - y_true[best_pred]))
        if optimizer_choice_key is not None:
            chosen = [i for i, r in enumerate(rs) if bool(r.get(optimizer_choice_key))]
            if chosen:
                agree.append(float(best_pred == chosen[0]))
    out = {
        "n_groups": float(len(groups)),
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else math.nan,
        "pairwise_accuracy": float(np.mean(pairs)) if pairs else math.nan,
        "spearman": float(np.mean(spears)) if spears else math.nan,
        "objective_regret": float(np.mean(regrets)) if regrets else math.nan,
        "yield_regret": float(np.mean(y_regrets)) if y_regrets else math.nan,
    }
    if agree:
        out["top1_optimizer_agreement"] = float(np.mean(agree))
    return out
