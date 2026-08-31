"""
Live RAM simulation: baseline (hold all pattern vectors) vs LSTM streaming.

Usage:
  python ram_sim.py --stil "C:\\Users\\Mohit\\Downloads\\Production_SCAN_stuck_at_1000pat.stil"
  python ram_sim.py --stil ... --max-patterns 50 --live
  python ram_sim.py --stil ... --max-patterns 200 --no-baseline
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from lstm_compressor import PatternLSTMCompressor
from stil_stream import estimate_full_load_bytes, iter_pattern_scans

PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / (1024 ** 2)


def mb(n_bytes: float) -> float:
    return n_bytes / (1024 ** 2)


def run_baseline(stil: Path, max_patterns: int | None, history: list) -> dict:
    """
    Naive approach: expand every pattern into float32 and keep ALL in a list.
    This is what blows up RAM as pattern count grows.
    """
    print("\n=== BASELINE: load all pattern vectors into RAM ===")
    t0 = time.perf_counter()
    rss0 = rss_mb()
    history.append({"mode": "baseline", "i": 0, "rss_mb": rss0, "held_mb": 0.0})

    held: list[np.ndarray] = []
    peak = rss0
    n = 0
    total_elems = 0

    for ps in iter_pattern_scans(stil, max_patterns=max_patterns):
        # Cast to float32 — typical ML / simulation path
        vec = ps.data.astype(np.float32)
        held.append(vec)
        total_elems += vec.size
        n += 1
        r = rss_mb()
        peak = max(peak, r)
        held_mb = total_elems * 4 / (1024 ** 2)
        history.append(
            {
                "mode": "baseline",
                "i": n,
                "rss_mb": r,
                "held_mb": held_mb,
                "pattern_id": ps.pattern_id,
            }
        )
        if n % 25 == 0 or n == 1:
            print(
                f"  patterns={n:4d}  RSS={r:8.1f} MB  "
                f"delta={r - rss0:+7.1f} MB  "
                f"held_tensors~{held_mb:7.1f} MB  shape={vec.shape}"
            )

    elapsed = time.perf_counter() - t0
    # Force retention so peak is meaningful until we clear
    retained_mb = sum(a.nbytes for a in held) / (1024 ** 2)
    result = {
        "patterns": n,
        "elapsed_s": elapsed,
        "rss_start_mb": rss0,
        "rss_peak_mb": peak,
        "rss_delta_mb": peak - rss0,
        "rss_end_mb": rss_mb(),
        "held_tensor_mb": retained_mb,
    }
    print(
        f"  DONE baseline: {n} patterns in {elapsed:.2f}s | "
        f"peak RSS={peak:.1f} MB (delta {peak - rss0:+.1f}) | "
        f"tensors held={retained_mb:.1f} MB"
    )
    # Release
    del held
    gc.collect()
    return result


def run_lstm_stream(
    stil: Path,
    max_patterns: int | None,
    history: list,
    hidden_size: int = 64,
    embed_dim: int = 32,
    store_embeddings: bool = True,
    device: str = "cpu",
) -> dict:
    """
    Optimized path: stream one pattern at a time through LSTM.
    Persistent memory ≈ model weights + optional compact embeddings (N × embed_dim),
    NOT N × T × F full vectors.
    """
    print("\n=== LSTM STREAM: fixed-state compression, one pattern at a time ===")
    t0 = time.perf_counter()
    rss0 = rss_mb()
    history.append({"mode": "lstm", "i": 0, "rss_mb": rss0, "held_mb": 0.0})

    model: PatternLSTMCompressor | None = None
    embeddings: list[np.ndarray] = []
    peak = rss0
    n = 0
    n_channels = None

    with torch.inference_mode():
        for ps in iter_pattern_scans(stil, max_patterns=max_patterns):
            arr = ps.data  # uint8 (T, C)
            if model is None:
                n_channels = arr.shape[1]
                model = PatternLSTMCompressor(
                    n_channels=n_channels,
                    hidden_size=hidden_size,
                    embed_dim=embed_dim,
                ).to(device)
                model.eval()
                print(
                    f"  model params={mb(model.param_nbytes()):.2f} MB | "
                    f"state/pattern~{model.state_nbytes()} bytes | "
                    f"channels={n_channels}"
                )

            # Only current pattern on device — not the full set
            x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(device)
            # normalize 0/1/2 → roughly [-1,1]
            x = (x - 1.0) / 1.0
            emb = model(x)
            if store_embeddings:
                embeddings.append(emb.detach().cpu().numpy().astype(np.float32).reshape(-1))

            del x, emb
            n += 1
            r = rss_mb()
            peak = max(peak, r)
            held_mb = (
                (n * embed_dim * 4) / (1024 ** 2) if store_embeddings else 0.0
            )
            history.append(
                {
                    "mode": "lstm",
                    "i": n,
                    "rss_mb": r,
                    "held_mb": held_mb,
                    "pattern_id": ps.pattern_id,
                }
            )
            if n % 25 == 0 or n == 1:
                print(
                    f"  patterns={n:4d}  RSS={r:8.1f} MB  "
                    f"delta={r - rss0:+7.1f} MB  "
                    f"embeddings~{held_mb:7.2f} MB  seq_len={arr.shape[0]}"
                )

    elapsed = time.perf_counter() - t0
    emb_mb = sum(e.nbytes for e in embeddings) / (1024 ** 2) if embeddings else 0.0
    result = {
        "patterns": n,
        "elapsed_s": elapsed,
        "rss_start_mb": rss0,
        "rss_peak_mb": peak,
        "rss_delta_mb": peak - rss0,
        "rss_end_mb": rss_mb(),
        "held_tensor_mb": emb_mb,
        "model_mb": mb(model.param_nbytes()) if model else 0.0,
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "n_channels": n_channels,
    }
    print(
        f"  DONE lstm: {n} patterns in {elapsed:.2f}s | "
        f"peak RSS={peak:.1f} MB (delta {peak - rss0:+.1f}) | "
        f"embeddings={emb_mb:.2f} MB | model={result['model_mb']:.2f} MB"
    )
    del embeddings, model
    gc.collect()
    return result


def plot_history(history: list, out_path: Path, live: bool = False) -> None:
    if not history:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    ax1, ax2 = axes

    for mode, color in (("baseline", "#c0392b"), ("lstm", "#2980b9")):
        pts = [h for h in history if h["mode"] == mode]
        if not pts:
            continue
        xs = [h["i"] for h in pts]
        ax1.plot(xs, [h["rss_mb"] for h in pts], color=color, label=f"{mode} RSS", lw=1.8)
        ax2.plot(
            xs,
            [h["held_mb"] for h in pts],
            color=color,
            label=f"{mode} held tensors",
            lw=1.8,
        )

    ax1.set_ylabel("Process RSS (MB)")
    ax1.set_title("Live RAM while running STIL patterns (includes Python/Torch base RSS)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Patterns processed")
    ax2.set_ylabel("Retained vector memory (MB)")
    ax2.set_title(
        "Vector memory kept in RAM "
        "(baseline = full scan tensors, LSTM = compact embeddings)"
    )
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved plot -> {out_path}")
    if live:
        plt.show()
    else:
        plt.close(fig)


def print_summary(est: dict, baseline: dict | None, lstm: dict | None) -> None:
    print("\n" + "=" * 64)
    print("SUMMARY - vector memory optimization (prototype)")
    print("=" * 64)
    print(f"STIL patterns seen (estimate pass): {est['patterns_seen']}")
    print(f"Example pattern shape (T, C):       {est['example_shape']}")
    print(
        f"If all kept as float32 tensors:     "
        f"~{mb(est['extrapolated_float32_bytes']):.1f} MB"
    )
    print(
        f"If all kept as uint8 bit-planes:    "
        f"~{mb(est['extrapolated_uint8_bytes']):.1f} MB"
    )
    if baseline:
        print(
            f"\nBaseline peak RSS:  {baseline['rss_peak_mb']:.1f} MB "
            f"(+{baseline['rss_delta_mb']:.1f} during run) | "
            f"held {baseline['held_tensor_mb']:.1f} MB | "
            f"{baseline['elapsed_s']:.2f}s"
        )
    if lstm:
        print(
            f"LSTM peak RSS:      {lstm['rss_peak_mb']:.1f} MB "
            f"(+{lstm['rss_delta_mb']:.1f} during run) | "
            f"held {lstm['held_tensor_mb']:.2f} MB | "
            f"{lstm['elapsed_s']:.2f}s | "
            f"model {lstm['model_mb']:.2f} MB"
        )
    if baseline and lstm and baseline["held_tensor_mb"] > 0:
        ratio = baseline["held_tensor_mb"] / max(lstm["held_tensor_mb"], 1e-9)
        print(f"\nHeld-vector reduction: {ratio:.1f}x "
              f"({baseline['held_tensor_mb']:.1f} -> {lstm['held_tensor_mb']:.2f} MB)")
        print(
            "Why: LSTM replaces (N*T*C) scan storage with (N*embed_dim) "
            "plus a fixed-size hidden/cell state while streaming."
        )

    # Project to larger N using measured avg pattern size
    avg_u8 = est.get("avg_pattern_bytes") or 0.0
    embed_dim = (lstm or {}).get("embed_dim", 32)
    print("\nScaling projection (vector payload only, not Torch base RSS):")
    print(f"{'N':>8}  {'baseline float32':>18}  {'LSTM embeddings':>16}  {'reduction':>10}")
    for n in (1_000, 10_000, 50_000, 100_000):
        base = mb(avg_u8 * n * 4)
        emb = mb(n * embed_dim * 4)
        red = base / max(emb, 1e-12)
        print(f"{n:8d}  {base:15.1f} MB  {emb:13.2f} MB  {red:9.0f}x")

    print(
        "\nNote: Process RSS includes Python+PyTorch (~hundreds of MB). "
        "Compare 'held' / delta / projection columns for vector-memory savings."
    )
    print("=" * 64)


def main() -> int:
    p = argparse.ArgumentParser(description="STIL pattern RAM simulation with LSTM")
    p.add_argument(
        "--stil",
        type=Path,
        default=Path(r"C:\Users\Mohit\Downloads\Production_SCAN_stuck_at_1000pat.stil"),
    )
    p.add_argument("--max-patterns", type=int, default=100,
                   help="Cap patterns for prototype speed (default 100). Use 1000 for full file.")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip full-load baseline (safer for large N)")
    p.add_argument("--no-store-embeddings", action="store_true",
                   help="Do not keep embeddings — pure streaming footprint")
    p.add_argument("--live", action="store_true", help="Show interactive matplotlib window")
    p.add_argument("--out", type=Path, default=Path("ram_sim_result.png"))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    if not args.stil.exists():
        print(f"STIL not found: {args.stil}", file=sys.stderr)
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"STIL: {args.stil} ({mb(args.stil.stat().st_size):.1f} MB on disk)")
    print(f"max_patterns={args.max_patterns}  device={args.device}")

    print("\n=== Quick estimate pass ===")
    est = estimate_full_load_bytes(args.stil, max_patterns=args.max_patterns)
    print(
        f"  patterns={est['patterns_seen']}  avg/pattern~{mb(est['avg_pattern_bytes']):.3f} MB "
        f"(uint8)  shape~{est['example_shape']}"
    )
    print(
        f"  extrapolated full float32 hold: ~{mb(est['extrapolated_float32_bytes']):.1f} MB"
    )

    history: list = []
    baseline = None
    if not args.no_baseline:
        baseline = run_baseline(args.stil, args.max_patterns, history)
        gc.collect()
        time.sleep(0.2)

    lstm = run_lstm_stream(
        args.stil,
        args.max_patterns,
        history,
        hidden_size=args.hidden_size,
        embed_dim=args.embed_dim,
        store_embeddings=not args.no_store_embeddings,
        device=args.device,
    )

    print_summary(est, baseline, lstm)
    plot_history(history, args.out, live=args.live)

    # Also dump CSV for further analysis
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("mode,i,rss_mb,held_mb,pattern_id\n")
        for h in history:
            f.write(
                f"{h['mode']},{h['i']},{h['rss_mb']:.3f},{h['held_mb']:.3f},"
                f"{h.get('pattern_id', '')}\n"
            )
    print(f"Saved metrics -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
