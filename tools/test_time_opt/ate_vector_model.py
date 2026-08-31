"""
ATE vector-memory model.

Models the REAL TEST path:
  STIL --compile--> Vector memory --playback--> DUT pins

Estimates peak resident vector RAM and playback time for:
  - full_suite: all pattern cycles resident
  - lstm_subset: play only an LSTM-selected pattern subset
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ate_stil_stats import PatternAteStats, StilAteProfile


@dataclass
class AtePinModel:
    """How many bits of vector memory one pin costs per cycle after compile."""

    n_pins: int
    bits_per_pin: float = 2.0  # drive + expect/mask packing (typical model)
    ate_overhead: float = 1.15  # sequencer / formatting overhead

    def bytes_per_cycle(self) -> float:
        return (self.n_pins * self.bits_per_pin / 8.0) * self.ate_overhead

    def bytes_for_cycles(self, n_cycles: int) -> float:
        return n_cycles * self.bytes_per_cycle()


@dataclass
class StrategyResult:
    name: str
    description: str
    patterns_played: int
    total_cycles: int
    peak_resident_cycles: int
    peak_vector_mb: float
    playback_ms: float
    reload_ms: float
    total_time_ms: float
    vector_mb_if_fully_expanded: float

    def to_dict(self) -> dict:
        return asdict(self)


def pattern_bytes(p: PatternAteStats, pin: AtePinModel) -> float:
    return pin.bytes_for_cycles(p.n_cycles)


def estimate_playback_ms(n_cycles: int, period_ns: float) -> float:
    """Wall time if tester runs continuously at `period_ns` per cycle."""
    return n_cycles * period_ns / 1e6


def strategy_full_suite(
    patterns: list[PatternAteStats],
    pin: AtePinModel,
    period_ns: float,
) -> StrategyResult:
    total_cycles = sum(p.n_cycles for p in patterns)
    peak = total_cycles
    pb = estimate_playback_ms(total_cycles, period_ns)
    peak_mb = pin.bytes_for_cycles(peak) / (1024 ** 2)
    return StrategyResult(
        name="full_suite",
        description="All patterns compiled and resident in ATE vector memory",
        patterns_played=len(patterns),
        total_cycles=total_cycles,
        peak_resident_cycles=peak,
        peak_vector_mb=peak_mb,
        playback_ms=pb,
        reload_ms=0.0,
        total_time_ms=pb,
        vector_mb_if_fully_expanded=peak_mb,
    )


def strategy_subset(
    patterns: list[PatternAteStats],
    selected_ids: set[int],
    pin: AtePinModel,
    period_ns: float,
    name: str,
    description: str,
) -> StrategyResult:
    subset = [p for p in patterns if p.pattern_id in selected_ids]
    subset.sort(key=lambda p: p.cycle_start)
    total_cycles = sum(p.n_cycles for p in subset)
    peak_cycles = total_cycles
    pb = estimate_playback_ms(total_cycles, period_ns)
    peak_mb = pin.bytes_for_cycles(peak_cycles) / (1024 ** 2)
    full_mb = pin.bytes_for_cycles(sum(p.n_cycles for p in patterns)) / (1024 ** 2)
    return StrategyResult(
        name=name,
        description=description,
        patterns_played=len(subset),
        total_cycles=total_cycles,
        peak_resident_cycles=peak_cycles,
        peak_vector_mb=peak_mb,
        playback_ms=pb,
        reload_ms=0.0,
        total_time_ms=pb,
        vector_mb_if_fully_expanded=full_mb,
    )


def compare_strategies(
    profile: StilAteProfile,
    selected_ids: set[int],
    bits_per_pin: float = 2.0,
    period_ns: float = 100.0,
) -> list[StrategyResult]:
    pin = AtePinModel(n_pins=profile.n_pins, bits_per_pin=bits_per_pin)
    patterns = profile.patterns
    return [
        strategy_full_suite(patterns, pin, period_ns),
        strategy_subset(
            patterns,
            selected_ids,
            pin,
            period_ns,
            name="lstm_subset",
            description="Play LSTM-selected pattern subset only",
        ),
    ]
