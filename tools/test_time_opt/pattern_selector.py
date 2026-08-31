"""
LSTM decides WHICH patterns to run and HOW MANY.

1) Embed each pattern with an LSTM (optional Dropout).
2) Rank patterns by diversity (farthest-point in embedding space).
3) Auto-choose count where extra patterns add little new diversity
   (and optionally stop earlier if a vector-RAM budget is set).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from lstm_compressor import PatternLSTMCompressor
from stil_stream import iter_pattern_scans


@dataclass
class SelectionResult:
    embeddings: np.ndarray
    pattern_ids: list[int]
    selected_ids: list[int]
    ranked_ids: list[int]
    n_total: int
    n_selected: int
    keep_ratio: float
    reason: str
    gains: list[float] = field(default_factory=list)


def embed_patterns(
    stil_path: str,
    max_patterns: int | None = None,
    hidden_size: int = 64,
    embed_dim: int = 32,
    dropout: float = 0.0,
    device: str = "cpu",
) -> tuple[list[int], np.ndarray]:
    model: PatternLSTMCompressor | None = None
    ids: list[int] = []
    emb_list: list[np.ndarray] = []

    for ps in iter_pattern_scans(stil_path, max_patterns=max_patterns):
        arr = ps.data
        if model is None:
            model = PatternLSTMCompressor(
                n_channels=arr.shape[1],
                hidden_size=hidden_size,
                embed_dim=embed_dim,
                num_layers=2,
                dropout=dropout,
            ).to(device)
            if dropout > 0:
                model.train()
            else:
                model.eval()
        with torch.set_grad_enabled(False):
            x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(device)
            x = (x - 1.0) / 1.0
            e = model(x).detach().cpu().numpy().reshape(-1)
        ids.append(ps.pattern_id)
        emb_list.append(e.astype(np.float32))

    if not emb_list:
        return [], np.zeros((0, embed_dim), dtype=np.float32)
    return ids, np.stack(emb_list, axis=0)


def diversity_ranking(
    embeddings: np.ndarray,
    always_include: list[int] | None = None,
) -> tuple[list[int], list[float]]:
    steps = list(iter_diversity_steps(embeddings, always_include=always_include))
    order = [s["picked_idx"] for s in steps]
    gains = [s["gain"] for s in steps]
    return order, gains


def iter_diversity_steps(
    embeddings: np.ndarray,
    always_include: list[int] | None = None,
    top_k_rivals: int = 3,
):
    """
    Yield farthest-point decisions one by one for live visualization.

    Each step includes:
      picked_idx, gain, selected_idxs,
      nearest_selected_idx, nearest_dist,
      rivals: top unused candidates by min-distance to selected set
    """
    n = embeddings.shape[0]
    if n == 0:
        return

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    x = embeddings / norms

    order: list[int] = []
    min_dist = np.full(n, np.inf, dtype=np.float64)

    def rivals_from(md: np.ndarray, exclude: list[int]) -> list[dict]:
        scores = md.copy()
        for e in exclude:
            scores[e] = -1.0
        out = []
        for _ in range(min(top_k_rivals, n - len(exclude))):
            j = int(np.argmax(scores))
            if scores[j] < 0:
                break
            out.append({"idx": j, "min_dist": float(scores[j])})
            scores[j] = -1.0
        return out

    # Seed picks
    seed: list[int] = []
    if always_include:
        for idx in always_include:
            if 0 <= idx < n and idx not in seed:
                seed.append(idx)

    if not seed:
        mean = x.mean(axis=0, keepdims=True)
        d0 = np.linalg.norm(x - mean, axis=1)
        seed.append(int(np.argmin(d0)))

    for si, idx in enumerate(seed):
        order.append(idx)
        # update distances after this seed
        min_dist = np.minimum(min_dist, np.linalg.norm(x - x[idx], axis=1))
        nearest_sel = None
        nearest_d = None
        if si > 0:
            # distance to previous seeds
            dists = [float(np.linalg.norm(x[idx] - x[p])) for p in order[:-1]]
            nearest_sel = order[int(np.argmin(dists))]
            nearest_d = float(min(dists))
        yield {
            "picked_idx": idx,
            "gain": float("inf") if si == 0 else float(nearest_d or 0.0),
            "selected_idxs": list(order),
            "nearest_selected_idx": nearest_sel,
            "nearest_dist": nearest_d,
            "rivals": rivals_from(min_dist, order),
            "reason": (
                "seed pattern (always include / start)"
                if si == 0
                else "seed pattern"
            ),
        }

    while len(order) < n:
        trial = min_dist.copy()
        trial[order] = -1.0
        nxt = int(np.argmax(trial))
        gain = float(trial[nxt])
        if gain < 0:
            break

        # nearest already-selected pattern in embedding space
        dists = [float(np.linalg.norm(x[nxt] - x[p])) for p in order]
        nearest_i = int(np.argmin(dists))
        nearest_sel = order[nearest_i]
        nearest_d = dists[nearest_i]
        rival_list = rivals_from(trial, order)

        order.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(x - x[nxt], axis=1))

        yield {
            "picked_idx": nxt,
            "gain": gain,
            "selected_idxs": list(order),
            "nearest_selected_idx": nearest_sel,
            "nearest_dist": nearest_d,
            "rivals": rival_list,
            "reason": (
                f"farthest from selected set "
                f"(min embedding distance={gain:.4f})"
            ),
        }


def auto_how_many(
    gains: list[float],
    n_total: int,
    min_frac: float = 0.2,
    max_frac: float = 0.85,
    stop_ratio: float = 0.08,
    patience: int = 3,
) -> tuple[int, str]:
    """
    Decide how many patterns to keep from diversity gains.
    Stops when marginal gain stays below stop_ratio * ref_gain.
    """
    if n_total <= 1:
        return n_total, "only one pattern available"

    min_k = max(1, int(round(n_total * min_frac)))
    max_k = max(min_k, int(round(n_total * max_frac)))

    # reference = best finite gain after the first pick
    finite = [g for g in gains[1:] if np.isfinite(g)]
    if not finite:
        k = min_k
        return k, f"auto keep {k}/{n_total} (min floor {min_frac:.0%})"

    ref = max(finite)
    thresh = stop_ratio * ref
    low_streak = 0
    chosen = max_k
    for k in range(min_k, max_k + 1):
        g = gains[k - 1] if k - 1 < len(gains) else 0.0
        if not np.isfinite(g):
            continue
        if g < thresh:
            low_streak += 1
            if low_streak >= patience:
                chosen = k
                return (
                    chosen,
                    f"auto keep {chosen}/{n_total} "
                    f"(diversity saturated: gain<{stop_ratio:.0%} of peak)",
                )
        else:
            low_streak = 0

    return (
        max_k,
        f"auto keep {max_k}/{n_total} (hit max keep {max_frac:.0%})",
    )


def apply_memory_budget(
    ranked_ids: list[int],
    cycles_by_id: dict[int, int],
    bytes_per_cycle: float,
    budget_mb: float,
) -> tuple[list[int], str]:
    """Take diversity-ranked patterns until vector RAM budget is filled."""
    if budget_mb <= 0:
        return ranked_ids, ""
    budget_bytes = budget_mb * (1024 ** 2)
    kept: list[int] = []
    used = 0.0
    for pid in ranked_ids:
        need = cycles_by_id.get(pid, 0) * bytes_per_cycle
        if kept and used + need > budget_bytes:
            break
        kept.append(pid)
        used += need
    if not kept and ranked_ids:
        kept = [ranked_ids[0]]
    return (
        kept,
        f"RAM budget {budget_mb:.2f} MB capped set to {len(kept)} patterns",
    )


def select_patterns_lstm(
    stil_path: str,
    max_patterns: int | None = None,
    keep_pattern0: bool = True,
    hidden_size: int = 64,
    embed_dim: int = 32,
    dropout: float = 0.0,
    device: str = "cpu",
    min_frac: float = 0.2,
    max_frac: float = 0.85,
    stop_ratio: float = 0.08,
    cycles_by_id: dict[int, int] | None = None,
    bytes_per_cycle: float = 0.0,
    budget_mb: float = 0.0,
) -> SelectionResult:
    """
    LSTM chooses which patterns and how many to run.
    """
    ids, emb = embed_patterns(
        stil_path,
        max_patterns=max_patterns,
        hidden_size=hidden_size,
        embed_dim=embed_dim,
        dropout=dropout,
        device=device,
    )
    n = len(ids)
    if n == 0:
        return SelectionResult(
            embeddings=emb,
            pattern_ids=[],
            selected_ids=[],
            ranked_ids=[],
            n_total=0,
            n_selected=0,
            keep_ratio=0.0,
            reason="no patterns",
        )

    always = []
    if keep_pattern0 and 0 in ids:
        always.append(ids.index(0))

    order_idx, gains = diversity_ranking(emb, always_include=always)
    ranked_ids = [ids[i] for i in order_idx]

    k, reason = auto_how_many(
        gains,
        n_total=n,
        min_frac=min_frac,
        max_frac=max_frac,
        stop_ratio=stop_ratio,
    )
    selected = ranked_ids[:k]

    if cycles_by_id is not None and budget_mb > 0 and bytes_per_cycle > 0:
        selected, budget_note = apply_memory_budget(
            selected, cycles_by_id, bytes_per_cycle, budget_mb
        )
        if budget_note:
            reason = f"{reason}; {budget_note}"

    # always keep pattern 0 if requested and present
    if keep_pattern0 and 0 in ids and 0 not in selected:
        selected = [0] + [p for p in selected if p != 0]

    return SelectionResult(
        embeddings=emb,
        pattern_ids=ids,
        selected_ids=sorted(set(selected)),
        ranked_ids=ranked_ids,
        n_total=n,
        n_selected=len(set(selected)),
        keep_ratio=len(set(selected)) / max(n, 1),
        reason=reason,
        gains=[float(g) if np.isfinite(g) else 0.0 for g in gains],
    )


# Back-compat wrapper used by ate_sim.py
def farthest_point_sample(
    embeddings: np.ndarray,
    n_select: int,
    always_include: list[int] | None = None,
) -> list[int]:
    order, _ = diversity_ranking(embeddings, always_include=always_include)
    return order[:n_select]


def select_patterns(
    stil_path: str,
    keep_ratio: float = 0.6,
    max_patterns: int | None = None,
    keep_pattern0: bool = True,
    hidden_size: int = 64,
    embed_dim: int = 32,
    dropout: float = 0.0,
    device: str = "cpu",
) -> SelectionResult:
    """Legacy fixed-ratio API → now routes through LSTM auto select with max_frac=keep_ratio."""
    return select_patterns_lstm(
        stil_path,
        max_patterns=max_patterns,
        keep_pattern0=keep_pattern0,
        hidden_size=hidden_size,
        embed_dim=embed_dim,
        dropout=dropout,
        device=device,
        min_frac=min(0.2, keep_ratio),
        max_frac=keep_ratio,
    )
