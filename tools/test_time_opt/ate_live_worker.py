"""
JSONL worker for the React/Node live dashboard.

LSTM decides WHICH patterns to run and HOW MANY (diversity saturation),
optionally capped by a vector-RAM budget.

Usage:
  python ate_live_worker.py --stil path.stil --dropout 0.3 --budget-mb 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psutil

from ate_stil_stats import extract_ate_profile
from ate_vector_model import AtePinModel
from stil_stream import iter_pattern_scans
from pattern_selector import (
    auto_how_many,
    apply_memory_budget,
    iter_diversity_steps,
)
from lstm_compressor import PatternLSTMCompressor

PROC = psutil.Process()


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def mb(n: float) -> float:
    return n / (1024 ** 2)


def rss_mb() -> float:
    return PROC.memory_info().rss / (1024 ** 2)


def bit_preview(arr, n: int = 234) -> list[int]:
    """
    One full scan-in chain as 0/1 for UI (default 234 = ScanLength).
    Uses the first scan-in channel; X→0.
    """
    if getattr(arr, "ndim", 1) == 2 and arr.shape[0] > 0:
        # (shifts, channels) → first channel's ScanLength bits
        chain = arr[:n, 0]
    else:
        chain = arr.reshape(-1)[:n]
    out: list[int] = []
    for v in chain:
        out.append(0 if int(v) != 1 else 1)
    while len(out) < n:
        out.append(0)
    return out[:n]


def bits_compare(a: list[int], b: list[int]) -> dict:
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i] == b[i])
    return {
        "n_same": same,
        "n_diff": n - same,
        "flip": [a[i] != b[i] for i in range(n)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stil", required=True)
    ap.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Neural-net Dropout probability in LSTM (0=off)",
    )
    ap.add_argument(
        "--budget-mb",
        type=float,
        default=0.0,
        help="Optional max ATE vector RAM (MB). 0 = LSTM diversity decides count only",
    )
    ap.add_argument(
        "--min-frac",
        type=float,
        default=0.2,
        help="Minimum fraction LSTM may keep",
    )
    ap.add_argument(
        "--max-frac",
        type=float,
        default=0.6,
        help="Maximum fraction LSTM may keep (0.6 ≈ 600 of 1000)",
    )
    ap.add_argument("--bits-per-pin", type=float, default=2.0)
    ap.add_argument("--period-ns", type=float, default=100.0)
    ap.add_argument("--max-patterns", type=int, default=0)
    ap.add_argument("--refresh-every", type=int, default=25)
    # legacy ignored (kept so old clients don't crash)
    ap.add_argument("--keep-ratio", type=float, default=None)
    args = ap.parse_args()

    stil_path = Path(args.stil)
    if not stil_path.exists():
        emit({"type": "error", "message": f"STIL not found: {stil_path}"})
        return 1

    dropout = min(max(args.dropout, 0.0), 0.9)
    max_patterns = args.max_patterns if args.max_patterns > 0 else None
    min_frac = args.min_frac
    max_frac = args.max_frac
    if args.keep_ratio is not None:
        # old UI sent keep-ratio as a cap
        max_frac = min(max_frac, max(args.keep_ratio, min_frac))

    try:
        profile = extract_ate_profile(stil_path)
        if max_patterns is not None:
            profile.patterns = [p for p in profile.patterns if p.pattern_id < max_patterns]
            profile.n_patterns = len(profile.patterns)
            profile.total_cycles = sum(p.n_cycles for p in profile.patterns)

        pin = AtePinModel(n_pins=profile.n_pins, bits_per_pin=args.bits_per_pin)
        cycles_by_id = {p.pattern_id: p.n_cycles for p in profile.patterns}
        full_peak_mb = mb(pin.bytes_for_cycles(profile.total_cycles))
        refresh = max(1, args.refresh_every)
        n_target = profile.n_patterns

        emit(
            {
                "type": "profile",
                "stil_name": stil_path.name,
                "stil_mb": round(mb(stil_path.stat().st_size), 3),
                "n_patterns": profile.n_patterns,
                "n_pins": profile.n_pins,
                "total_cycles": profile.total_cycles,
                "full_peak_mb": round(full_peak_mb, 4),
                "bytes_per_cycle": round(pin.bytes_per_cycle(), 4),
                "dropout": dropout,
                "budget_mb": args.budget_mb,
                "min_frac": min_frac,
                "max_frac": max_frac,
                "bits_per_pin": args.bits_per_pin,
                "period_ns": args.period_ns,
            }
        )

        # Progress while scanning for embeddings is inside select — approximate
        # by a pre-pass count for UI, then run LSTM select.
        emit(
            {
                "type": "progress",
                "phase": "embed",
                "step": 0,
                "total": n_target,
                "full_mb": 0.0,
                "lstm_mb": 0.0,
                "rss_mb": round(rss_mb(), 2),
                "message": "Verilumen agent embedding patterns to decide which/how many to run…",
            }
        )

        # Embed with live progress, then Verilumen decides which + how many.
        import numpy as np
        import torch

        # Fixed seed → same STIL file always gets the same ranking
        torch.manual_seed(42)
        np.random.seed(42)

        model = None
        ids: list[int] = []
        emb_list: list = []
        bits_by_id: dict[int, list[int]] = {}
        seen = 0
        cum_full = 0.0

        for ps in iter_pattern_scans(stil_path, max_patterns=max_patterns):
            if ps.pattern_id not in cycles_by_id:
                continue
            arr = ps.data
            if model is None:
                model = PatternLSTMCompressor(
                    n_channels=arr.shape[1],
                    hidden_size=64,
                    num_layers=2,
                    embed_dim=32,
                    dropout=dropout,
                )
                if dropout > 0:
                    model.train()
                else:
                    model.eval()
            with torch.set_grad_enabled(False):
                # Deterministic dropout path if ever enabled
                torch.manual_seed(42 + int(ps.pattern_id))
                x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
                x = (x - 1.0) / 1.0
                e = model(x).detach().cpu().numpy().reshape(-1)
            ids.append(ps.pattern_id)
            emb_list.append(e.astype(np.float32))
            bits_by_id[ps.pattern_id] = bit_preview(arr, n=234)
            seen += 1
            cum_full += pin.bytes_for_cycles(cycles_by_id[ps.pattern_id])
            if seen == 1 or seen % refresh == 0 or seen == n_target:
                emit(
                    {
                        "type": "progress",
                        "phase": "embed",
                        "step": seen,
                        "total": n_target,
                        "full_mb": round(mb(cum_full), 4),
                        "lstm_mb": 0.0,
                        "rss_mb": round(rss_mb(), 2),
                        "message": (
                            f"Verilumen agent embedding {seen}/{n_target} "
                            f"(dropout={dropout:.2f})"
                        ),
                    }
                )

        if not emb_list:
            emit({"type": "error", "message": "No patterns found in STIL"})
            return 1

        emb = np.stack(emb_list, axis=0)
        # Auto seed: pattern 0 if present, else first pattern
        seed_pid = 0 if 0 in ids else ids[0]
        always = [ids.index(seed_pid)]

        # Precompute full ranking to decide HOW MANY to keep, then replay
        # decisions live with comparison details for the UI.
        steps = list(iter_diversity_steps(emb, always_include=always, top_k_rivals=3))
        order_idx = [s["picked_idx"] for s in steps]
        gains = [s["gain"] for s in steps]
        ranked_ids = [ids[i] for i in order_idx]
        k, reason = auto_how_many(
            gains, n_total=len(ids), min_frac=min_frac, max_frac=max_frac
        )
        selected_list = ranked_ids[:k]
        if args.budget_mb > 0:
            selected_list, budget_note = apply_memory_budget(
                selected_list,
                cycles_by_id,
                pin.bytes_per_cycle(),
                args.budget_mb,
            )
            if budget_note:
                reason = f"{reason}; {budget_note}"
        if seed_pid not in selected_list:
            selected_list = [seed_pid] + [p for p in selected_list if p != seed_pid]
        selected = set(selected_list)

        import time

        normal_order = list(ids)
        emit(
            {
                "type": "progress",
                "phase": "select",
                "step": 0,
                "total": len(steps),
                "full_mb": round(full_peak_mb, 4),
                "lstm_mb": 0.0,
                "rss_mb": round(rss_mb(), 2),
                "message": "Verilumen agent comparing — watching pick order…",
                "reason": reason,
            }
        )

        chooser_every = 1
        for step_i, step in enumerate(steps, start=1):
            normal_pid = (
                normal_order[step_i - 1] if step_i <= len(normal_order) else None
            )
            lstm_pid = ids[step["picked_idx"]]
            kept = lstm_pid in selected
            nearest_pid = (
                ids[step["nearest_selected_idx"]]
                if step["nearest_selected_idx"] is not None
                else None
            )
            gain_f = float(step["gain"]) if np.isfinite(step["gain"]) else None
            # Compare 0/1 bits vs nearest already-kept pattern (or first kept)
            selected_so_far = [ids[i] for i in step["selected_idxs"]]
            ref_pid = nearest_pid
            if ref_pid is None and len(selected_so_far) > 1:
                # after seed is in selected_so_far, use previous keep as ref
                prev = [p for p in selected_so_far if p != lstm_pid]
                ref_pid = prev[-1] if prev else None
            ref_bits = bits_by_id.get(ref_pid) if ref_pid is not None else None

            def board_row(pid: int, winner: bool, min_dist: float, seed: bool = False):
                bits = bits_by_id.get(pid, [0] * 234)
                row = {
                    "pattern_id": pid,
                    "min_dist": round(float(min_dist), 4),
                    "winner": winner,
                    "seed": seed,
                    "bits": bits,
                }
                if ref_bits is not None:
                    cmp = bits_compare(bits, ref_bits)
                    row["n_same"] = cmp["n_same"]
                    row["n_diff"] = cmp["n_diff"]
                    row["flip"] = cmp["flip"]
                return row

            rivals = []
            for r in step.get("rivals", []):
                if r["idx"] == step["picked_idx"]:
                    continue
                rivals.append(
                    board_row(
                        ids[r["idx"]],
                        winner=False,
                        min_dist=float(r["min_dist"]),
                    )
                )
                if len(rivals) >= 3:
                    break

            if gain_f is None:
                winner_dist = max(
                    (float(r["min_dist"]) for r in rivals[:3]),
                    default=1.0,
                )
            else:
                winner_dist = gain_f
            compare_board = [
                board_row(
                    lstm_pid,
                    winner=True,
                    min_dist=winner_dist,
                    seed=gain_f is None,
                )
            ] + rivals[:3]

            emit(
                {
                    "type": "chooser",
                    "step": step_i,
                    "total": len(steps),
                    "k_target": len(selected),
                    "normal": {
                        "pattern_id": normal_pid,
                        "action": "run",
                        "note": "normal ATE order (next pattern in sequence)",
                    },
                    "lstm": {
                        "pattern_id": lstm_pid,
                        "action": "keep" if kept else "skip",
                        "gain": round(
                            float(step["gain"]) if np.isfinite(step["gain"]) else 0.0,
                            4,
                        ),
                        "reason": step["reason"],
                        "nearest_selected": nearest_pid,
                        "nearest_dist": (
                            round(float(step["nearest_dist"]), 4)
                            if step["nearest_dist"] is not None
                            else None
                        ),
                        "selected_so_far": selected_so_far[-8:],
                        "ref_pattern_id": ref_pid,
                        "ref_bits": ref_bits,
                        "compare_board": compare_board,
                        "note": "Verilumen agent diversity pick",
                    },
                    "message": (
                        f"#{step_i}: normal P{normal_pid} | Verilumen agent P{lstm_pid}"
                    ),
                }
            )
            # Slow for first 10 so the UI can be watched; speed up after
            if step_i <= 10:
                time.sleep(0.85)
            elif step_i % max(1, min(refresh, 3)) == 0:
                time.sleep(0.01)

        emit(
            {
                "type": "progress",
                "phase": "select",
                "step": len(selected),
                "total": len(ids),
                "full_mb": round(full_peak_mb, 4),
                "lstm_mb": 0.0,
                "rss_mb": round(rss_mb(), 2),
                "message": (
                    f"Verilumen agent chose {len(selected)}/{len(ids)} patterns — {reason}"
                ),
                "reason": reason,
            }
        )

        cum_f = 0.0
        cum_l = 0.0
        ordered = [p for p in profile.patterns if p.pattern_id in cycles_by_id]
        for i, p in enumerate(ordered, start=1):
            cum_f += pin.bytes_for_cycles(p.n_cycles)
            if p.pattern_id in selected:
                cum_l += pin.bytes_for_cycles(p.n_cycles)
            if i == 1 or i % refresh == 0 or i == len(ordered):
                emit(
                    {
                        "type": "progress",
                        "phase": "load",
                        "step": i,
                        "total": len(ordered),
                        "full_mb": round(mb(cum_f), 4),
                        "lstm_mb": round(mb(cum_l), 4),
                        "rss_mb": round(rss_mb(), 2),
                        "message": f"Vector memory load {i}/{len(ordered)}",
                    }
                )

        lstm_cycles = sum(cycles_by_id[i] for i in selected if i in cycles_by_id)
        full_ms = profile.total_cycles * args.period_ns / 1e6
        lstm_ms = lstm_cycles * args.period_ns / 1e6
        lstm_peak = mb(cum_l)

        discarded_ids = [pid for pid in ranked_ids if pid not in selected]
        # Explain why discarded: still valid, but past keep cut + similar to a kept one
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        x_emb = emb / norms
        id_to_idx = {pid: i for i, pid in enumerate(ids)}
        sel_idx = [id_to_idx[p] for p in selected_list if p in id_to_idx]
        sel_x = x_emb[sel_idx] if sel_idx else None
        discarded_details = []
        for d_rank, pid in enumerate(discarded_ids, start=1):
            i = id_to_idx.get(pid)
            if i is None or sel_x is None or len(sel_x) == 0:
                continue
            dists = np.linalg.norm(sel_x - x_emb[i], axis=1)
            nj = int(np.argmin(dists))
            nearest_pid = selected_list[nj]
            dist_v = float(dists[nj])
            overall_rank = ranked_ids.index(pid) + 1 if pid in ranked_ids else d_rank
            disc_bits = bits_by_id.get(pid, [0] * 234)
            kept_bits = bits_by_id.get(nearest_pid, [0] * 234)
            cmp = bits_compare(disc_bits, kept_bits)
            discarded_details.append(
                {
                    "pattern_id": pid,
                    "discard_rank": d_rank,
                    "diversity_rank": overall_rank,
                    "nearest_kept": nearest_pid,
                    "distance_to_nearest": round(dist_v, 4),
                    "bits": disc_bits,
                    "nearest_bits": kept_bits,
                    "flip": cmp["flip"],
                    "n_same": cmp["n_same"],
                    "n_diff": cmp["n_diff"],
                    "reason": (
                        f"P{pid} is a valid pattern; however, it was not selected "
                        f"for loading because its Verilumen agent embedding shows "
                        f"high similarity with the already retained pattern "
                        f"P{nearest_pid} (distance = {dist_v:.4f}). Since it "
                        f"provides minimal additional stimulus coverage, it was "
                        f"excluded to optimize vector RAM utilization."
                    ),
                }
            )

        selected_details = []
        for s_rank, pid in enumerate(selected_list, start=1):
            i = id_to_idx.get(pid)
            if i is None or sel_x is None or len(sel_x) == 0:
                continue
            # distance to nearest *other* kept pattern (seed has no peer)
            peer_mask = np.ones(len(sel_idx), dtype=bool)
            try:
                self_j = selected_list.index(pid)
                peer_mask[self_j] = False
            except ValueError:
                pass
            if not peer_mask.any():
                dist_v = 0.0
                nearest_pid = pid
            else:
                dists = np.linalg.norm(sel_x[peer_mask] - x_emb[i], axis=1)
                nj = int(np.argmin(dists))
                peer_ids = [p for k, p in enumerate(selected_list) if peer_mask[k]]
                nearest_pid = peer_ids[nj]
                dist_v = float(dists[nj])
            selected_details.append(
                {
                    "pattern_id": pid,
                    "keep_rank": s_rank,
                    "nearest_kept": nearest_pid,
                    "distance_to_nearest": round(dist_v, 4),
                }
            )

        emit(
            {
                "type": "done",
                "selected_n": len(selected),
                "total_n": len(ids),
                "full_peak_mb": round(full_peak_mb, 4),
                "lstm_peak_mb": round(lstm_peak, 4),
                "full_ms": round(full_ms, 4),
                "lstm_ms": round(lstm_ms, 4),
                "full_cycles": profile.total_cycles,
                "lstm_cycles": lstm_cycles,
                "saved_mb": round(full_peak_mb - lstm_peak, 4),
                "saved_pct": round(
                    100 * (full_peak_mb - lstm_peak) / max(full_peak_mb, 1e-9), 1
                ),
                "dropout": dropout,
                "keep_ratio": round(len(selected) / max(len(ids), 1), 4),
                "reason": reason,
                "selected_ids": list(selected_list),
                "selected_details": selected_details,
                "discarded_ids": discarded_ids,
                "discarded_details": discarded_details,
                "discard_example": discarded_details[0] if discarded_details else None,
                "selected_ids_head": list(selected_list)[:40],
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        emit({"type": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
