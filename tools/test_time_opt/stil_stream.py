"""
Stream-parse Tessent STIL scan patterns without loading the whole file into RAM.

Extracts Macro "scan_edt_g1" scan-in chains, expands STIL RLE (\\rN C), and yields
one pattern at a time as a compact uint8 array: shape (n_shifts, n_channels).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# STIL run-length: \r191 0  or  \r15 1
_RLE = re.compile(r"\\r(\d+)\s+([01XHLZxhlz])")
_LITERAL = re.compile(r"[01XHLZxhlz]+")
_PATTERN_ANN = re.compile(
    r'Ann\s*\{\*\s*"cycle_number:\d+\s+pattern_numer:(\d+)\s+pattern_type:'
)
_MACRO_START = re.compile(r'Macro\s+"scan_edt_g1"\s*\{')
# Scan-in channel assignments (odd channels in the EDT pair list are typically SO)
_SCAN_IN_ASSIGN = re.compile(
    r'"([^"]*?(?:ETH_TXCLK|ETH_TXD0|ETH_TXD1|ETH_TXD2|ETH_TXD3|ETH_TXCTL|'
    r"ETH_MDCLK|SPI0_SCLK|SPI0_SCSN|SPI0_IO0_SDO|SPI0_IO1_SDI|SPI0_IO2_WP|"
    r"SPI0_IO3_RST|SPI1_SCSN|SPI1_SDO|SPI1_SDI|SPI2_SCLK|SPI2_SCSN|"
    r'SPI2_IO0_SDO|SPI2_IO1_SDI|SPI2_IO2_WP|SPI2_IO3_RST|SPI3_SCLK)[^"]*?)"\s*=\s*([^;]+);',
    re.IGNORECASE,
)


def expand_stil_bits(expr: str, target_len: int | None = None) -> list[int]:
    """Expand STIL bit expression with \\rN C run-length into 0/1/2(X) ints."""
    bits: list[int] = []
    pos = 0
    expr = expr.strip()
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        m = _RLE.match(expr, pos)
        if m:
            n, ch = int(m.group(1)), m.group(2).upper()
            val = 0 if ch == "0" else 1 if ch == "1" else 2
            bits.extend([val] * n)
            pos = m.end()
            continue
        m = _LITERAL.match(expr, pos)
        if m:
            for ch in m.group(0).upper():
                bits.append(0 if ch == "0" else 1 if ch == "1" else 2)
            pos = m.end()
            continue
        pos += 1

    if target_len is not None:
        if len(bits) < target_len:
            bits.extend([0] * (target_len - len(bits)))
        elif len(bits) > target_len:
            bits = bits[:target_len]
    return bits


@dataclass
class PatternScan:
    pattern_id: int
    # shape conceptually: (seq_len, n_channels) — each value in {0,1,2}
    data: "object"  # numpy ndarray uint8
    n_macros: int
    raw_bytes_estimate: int


def _extract_scan_ins(macro_body: str, chain_len: int = 234) -> list[list[int]]:
    """Parse scan-in assignments inside one Macro block → list of chains."""
    channels: list[list[int]] = []
    for m in _SCAN_IN_ASSIGN.finditer(macro_body):
        channels.append(expand_stil_bits(m.group(2), target_len=chain_len))
    return channels


def iter_pattern_scans(
    stil_path: str | Path,
    chain_len: int = 234,
    max_patterns: int | None = None,
) -> Iterator[PatternScan]:
    """
    Stream STIL file line-by-line. Yields one PatternScan per pattern_numer
    once its Macro load/unload blocks have been collected.

    Memory: O(one pattern) — never holds all patterns.
    """
    import numpy as np

    stil_path = Path(stil_path)
    current_pat: int | None = None
    macro_channels: list[list[list[int]]] = []  # macros × channels × bits
    in_macro = False
    macro_buf: list[str] = []
    brace_depth = 0
    yielded = 0

    def flush() -> PatternScan | None:
        nonlocal macro_channels, yielded
        if current_pat is None or not macro_channels:
            macro_channels = []
            return None
        # Stack macros along time: each macro is chain_len steps × n_ch
        # Use first macro's channel count; pad missing.
        n_ch = max(len(m) for m in macro_channels)
        seqs = []
        for mac in macro_channels:
            # (chain_len, n_ch)
            arr = np.zeros((chain_len, n_ch), dtype=np.uint8)
            for ci, chain in enumerate(mac):
                if ci >= n_ch:
                    break
                L = min(chain_len, len(chain))
                arr[:L, ci] = np.asarray(chain[:L], dtype=np.uint8)
            seqs.append(arr)
        data = np.concatenate(seqs, axis=0) if seqs else np.zeros((0, n_ch), dtype=np.uint8)
        raw = int(data.nbytes)
        ps = PatternScan(
            pattern_id=current_pat,
            data=data,
            n_macros=len(macro_channels),
            raw_bytes_estimate=raw,
        )
        macro_channels = []
        yielded += 1
        return ps

    with stil_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Track pattern id from annotations
            pm = _PATTERN_ANN.search(line)
            if pm:
                new_id = int(pm.group(1))
                if current_pat is not None and new_id != current_pat:
                    ps = flush()
                    if ps is not None:
                        yield ps
                        if max_patterns is not None and yielded >= max_patterns:
                            return
                current_pat = new_id

            if not in_macro and _MACRO_START.search(line):
                in_macro = True
                macro_buf = [line]
                brace_depth = line.count("{") - line.count("}")
                if brace_depth <= 0:
                    # single-line macro (unlikely)
                    body = "".join(macro_buf)
                    ch = _extract_scan_ins(body, chain_len)
                    if ch:
                        macro_channels.append(ch)
                    in_macro = False
                    macro_buf = []
                continue

            if in_macro:
                macro_buf.append(line)
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    body = "".join(macro_buf)
                    ch = _extract_scan_ins(body, chain_len)
                    if ch:
                        macro_channels.append(ch)
                    in_macro = False
                    macro_buf = []

    ps = flush()
    if ps is not None:
        yield ps


def estimate_full_load_bytes(
    stil_path: str | Path,
    max_patterns: int | None = None,
    sample: int = 5,
) -> dict:
    """Quick pass: sample a few patterns to extrapolate full-load RAM."""
    sizes = []
    n = 0
    last_shape = None
    for ps in iter_pattern_scans(stil_path, max_patterns=max_patterns):
        n += 1
        last_shape = ps.data.shape
        if len(sizes) < sample:
            sizes.append(ps.raw_bytes_estimate)
    avg = sum(sizes) / max(len(sizes), 1)
    return {
        "patterns_seen": n,
        "avg_pattern_bytes": avg,
        "extrapolated_uint8_bytes": avg * n,
        "extrapolated_float32_bytes": avg * n * 4,  # if cast to float for ML
        "example_shape": last_shape,
    }
