"""
ATE vector-memory optimization simulation.

Models: STIL -> compile -> Vector memory -> playback -> DUT pins

Compares:
  1) full_suite   — all cycles resident (baseline ATE vector RAM)
  2) lstm_subset  — LSTM diversity selection (fewer patterns / cycles)

Usage:
  python ate_sim.py --stil "C:\\Users\\Mohit\\Downloads\\Production_SCAN_stuck_at_1000pat.stil"
  python ate_sim.py --keep-ratio 0.6 --period-ns 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ate_stil_stats import extract_ate_profile
from ate_vector_model import AtePinModel, compare_strategies
from pattern_selector import select_patterns


def main() -> int:
    ap = argparse.ArgumentParser(description="ATE vector-memory optimization sim")
    ap.add_argument(
        "--stil",
        type=Path,
        default=Path(r"C:\Users\Mohit\Downloads\Production_SCAN_stuck_at_1000pat.stil"),
    )
    ap.add_argument("--max-patterns", type=int, default=None,
                    help="Optional cap for prototype speed")
    ap.add_argument("--keep-ratio", type=float, default=0.6,
                    help="Fraction of patterns kept by diversity selection (ATE subset)")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="Neural-net Dropout probability in LSTM (0=off)")
    ap.add_argument("--bits-per-pin", type=float, default=2.0,
                    help="Packed ATE bits per pin per cycle (default 2)")
    ap.add_argument("--period-ns", type=float, default=100.0,
                    help="Tester period for playback time estimate")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", type=Path, default=Path("ate_sim_result.json"))
    args = ap.parse_args()

    if not args.stil.exists():
        print(f"STIL not found: {args.stil}", file=sys.stderr)
        return 1

    print("=" * 68)
    print("ATE VECTOR MEMORY SIM  |  STIL -> compile -> vector memory -> DUT")
    print("=" * 68)
    print(f"STIL: {args.stil}")

    print("\n[1/3] Extracting TesterCycle / pin profile ...")
    profile = extract_ate_profile(args.stil)
    if args.max_patterns is not None:
        profile.patterns = [
            p for p in profile.patterns if p.pattern_id < args.max_patterns
        ]
        profile.n_patterns = len(profile.patterns)
        profile.total_cycles = sum(p.n_cycles for p in profile.patterns)

    pin = AtePinModel(n_pins=profile.n_pins, bits_per_pin=args.bits_per_pin)
    full_mb = pin.bytes_for_cycles(profile.total_cycles) / (1024 ** 2)
    print(
        f"  pins={profile.n_pins}  patterns={profile.n_patterns}  "
        f"total_cycles={profile.total_cycles:,}"
    )
    print(
        f"  bytes/cycle~{pin.bytes_per_cycle():.2f}  "
        f"full-suite vector RAM~{full_mb:.2f} MB "
        f"(bits_per_pin={args.bits_per_pin})"
    )
    print(
        f"  pattern0 cycles={profile.patterns[0].n_cycles if profile.patterns else 0}  "
        f"typical later~{profile.patterns[1].n_cycles if len(profile.patterns) > 1 else 0}"
    )

    print(
        f"\n[2/3] LSTM auto-select which + how many "
        f"(dropout={args.dropout}, max_frac={args.keep_ratio}) ..."
    )
    sel = select_patterns(
        str(args.stil),
        keep_ratio=args.keep_ratio,
        max_patterns=args.max_patterns,
        device=args.device,
        dropout=args.dropout,
    )
    selected = set(sel.selected_ids)
    print(
        f"  embedded={sel.n_total}  selected={sel.n_selected}  "
        f"({100 * sel.n_selected / max(sel.n_total, 1):.1f}% kept)"
    )
    if getattr(sel, "reason", None):
        print(f"  reason: {sel.reason}")

    print("\n[3/3] Comparing ATE strategies ...")
    results = compare_strategies(
        profile,
        selected_ids=selected,
        bits_per_pin=args.bits_per_pin,
        period_ns=args.period_ns,
    )

    baseline = results[0]
    print()
    print(
        f"{'strategy':<22} {'pats':>6} {'cycles':>10} "
        f"{'peak_MB':>10} {'play_ms':>10} {'vs_RAM':>8} {'vs_time':>8}"
    )
    print("-" * 68)
    for r in results:
        vs_ram = baseline.peak_vector_mb / max(r.peak_vector_mb, 1e-12)
        vs_time = baseline.playback_ms / max(r.playback_ms, 1e-12)
        print(
            f"{r.name:<22} {r.patterns_played:6d} {r.total_cycles:10,d} "
            f"{r.peak_vector_mb:10.2f} {r.playback_ms:10.2f} "
            f"{vs_ram:7.1f}x {vs_time:7.1f}x"
        )

    print()
    print("Notes:")
    print("  peak_MB = peak resident ATE vector memory")
    print("  play_ms = total_cycles * period_ns (continuous playback)")
    print("  LSTM subset keeps diverse patterns only -> less vector RAM and less time")

    payload = {
        "stil": str(args.stil),
        "profile": {
            "n_pins": profile.n_pins,
            "n_patterns": profile.n_patterns,
            "total_cycles": profile.total_cycles,
            "n_v_statements": profile.n_v_statements,
            "n_macros_total": profile.n_macros_total,
            "bytes_per_cycle": pin.bytes_per_cycle(),
            "bits_per_pin": args.bits_per_pin,
            "period_ns": args.period_ns,
        },
        "selection": {
            "keep_ratio": sel.keep_ratio,
            "n_total": sel.n_total,
            "n_selected": sel.n_selected,
            "selected_ids_head": sel.selected_ids[:40],
            "selected_ids_count": len(sel.selected_ids),
        },
        "strategies": [r.to_dict() for r in results],
        "per_pattern_cycles": [
            {
                "pattern_id": p.pattern_id,
                "n_cycles": p.n_cycles,
                "selected": p.pattern_id in selected,
                "vector_kb": pin.bytes_for_cycles(p.n_cycles) / 1024.0,
            }
            for p in profile.patterns
        ],
    }

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "patterns_played",
                "total_cycles",
                "peak_resident_cycles",
                "peak_vector_mb",
                "playback_ms",
                "reload_ms",
                "total_time_ms",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in w.fieldnames})
    print(f"Wrote {csv_path}")

    curve_path = args.out.with_name("ate_sim_curve.csv")
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "full_suite_cum_mb", "lstm_subset_cum_mb"])
        cum_full = 0.0
        cum_sel = 0.0
        for i, p in enumerate(profile.patterns, start=1):
            cum_full += pin.bytes_for_cycles(p.n_cycles)
            if p.pattern_id in selected:
                cum_sel += pin.bytes_for_cycles(p.n_cycles)
            w.writerow(
                [
                    i,
                    f"{cum_full / (1024 ** 2):.4f}",
                    f"{cum_sel / (1024 ** 2):.4f}",
                ]
            )
    print(f"Wrote {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
