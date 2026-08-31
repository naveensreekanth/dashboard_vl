"""
Extract per-pattern ATE-oriented stats from a Tessent STIL file.

Uses TesterCycle annotations to attribute cycles to each pattern_numer.
Also counts Macro scan loads (EDT shift blocks).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


_PAT_ANN = re.compile(
    r"Ann\s*\{\*\s*Pattern:(\d+)\s+Vector:(\d+)\s+TesterCycle:(\d+)\s*\*\}"
)
_SIGNAL = re.compile(r'"[^"]+"\s+(In|Out|InOut)\b')
_MACRO = re.compile(r'Macro\s+"scan_edt_g1"\s*\{')
_PATTERN_NUM = re.compile(
    r'Ann\s*\{\*\s*"cycle_number:\d+\s+pattern_numer:(\d+)\s+pattern_type:'
)


@dataclass
class PatternAteStats:
    pattern_id: int
    cycle_start: int
    cycle_end: int  # exclusive
    n_cycles: int
    n_macros: int

    @property
    def expanded_bytes(self) -> int:
        """Filled by AtePinModel externally; placeholder 0 here."""
        return 0


@dataclass
class StilAteProfile:
    stil_path: str
    n_pins: int
    n_patterns: int
    total_cycles: int
    n_v_statements: int
    n_macros_total: int
    patterns: list[PatternAteStats]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def count_signals(stil_path: Path) -> int:
    n = 0
    in_signals = False
    with stil_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("Signals"):
                in_signals = True
                continue
            if in_signals:
                if line.strip().startswith("}"):
                    break
                n += len(_SIGNAL.findall(line))
    return n


def extract_ate_profile(stil_path: str | Path) -> StilAteProfile:
    stil_path = Path(stil_path)
    n_pins = count_signals(stil_path)

    pat_min: dict[int, int] = {}
    pat_max: dict[int, int] = {}
    macros_by_pat: dict[int, int] = {}
    current_pat: int | None = None
    max_cycle = 0
    n_v = 0
    n_macro = 0

    with stil_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "V {" in line:
                n_v += 1
            pm = _PATTERN_NUM.search(line)
            if pm:
                current_pat = int(pm.group(1))
            if _MACRO.search(line):
                n_macro += 1
                if current_pat is not None:
                    macros_by_pat[current_pat] = macros_by_pat.get(current_pat, 0) + 1
            m = _PAT_ANN.search(line)
            if m:
                p, _v, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
                max_cycle = max(max_cycle, c)
                if p not in pat_min:
                    pat_min[p] = c
                pat_max[p] = c

    total_cycles = max_cycle + 1
    starts = [(p, pat_min[p]) for p in sorted(pat_min)]
    patterns: list[PatternAteStats] = []
    for i, (p, s) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else total_cycles
        # Guard last pattern if annotation is a single marker
        n_cycles = max(0, end - s)
        if i == len(starts) - 1 and n_cycles < 10 and len(starts) > 1:
            # atypical trailing marker — use median of prior owned spans
            prior = [patterns[j].n_cycles for j in range(1, len(patterns))]
            if prior:
                prior_sorted = sorted(prior)
                n_cycles = prior_sorted[len(prior_sorted) // 2]
                end = s + n_cycles
        patterns.append(
            PatternAteStats(
                pattern_id=p,
                cycle_start=s,
                cycle_end=end,
                n_cycles=n_cycles,
                n_macros=macros_by_pat.get(p, 0),
            )
        )

    return StilAteProfile(
        stil_path=str(stil_path),
        n_pins=n_pins,
        n_patterns=len(patterns),
        total_cycles=sum(p.n_cycles for p in patterns),
        n_v_statements=n_v,
        n_macros_total=n_macro,
        patterns=patterns,
    )


def iter_owned_cycles(profile: StilAteProfile) -> Iterator[PatternAteStats]:
    yield from profile.patterns
