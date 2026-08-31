"""Build CoreDieIndex from a single temporal month (Phase 12.3)."""

from __future__ import annotations

import pandas as pd

from dtl_agent.data.temporal.loader import TemporalMonthData
from dtl_agent.simulation.core.die_index import CoreDieIndex, DieParamSeries


def build_core_die_index_from_temporal(month: TemporalMonthData) -> CoreDieIndex:
    """Construct IR/Thermal die series from month-scoped actual_die measurements only."""
    df = month.actual_die
    sim = df.loc[df["parameter"].isin(["ir_drop", "thermal"]), [
        "lot_id",
        "die_id",
        "pattern_id",
        "parameter",
        "measurement_value",
        "die_status",
    ]].copy()
    if sim.empty:
        raise ValueError(
            f"No ir_drop/thermal rows for production_month={month.production_month!r}"
        )

    status_map: dict[tuple[str, str], str | None] = {}
    for (lot, die), g in df.groupby(["lot_id", "die_id"], sort=False):
        st = g["die_status"].iloc[0]
        status_map[(str(lot), str(die))] = None if pd.isna(st) else str(st)

    sim["lot_id"] = sim["lot_id"].astype(str)
    sim["die_id"] = sim["die_id"].astype(str)
    sim["pattern_id"] = sim["pattern_id"].astype(int)
    sim = sim.sort_values(["lot_id", "die_id", "pattern_id"])

    ir: dict[tuple[str, str], DieParamSeries] = {}
    th: dict[tuple[str, str], DieParamSeries] = {}
    for (lot, die), g in sim.groupby(["lot_id", "die_id"], sort=False):
        key = (lot, die)
        st = status_map.get(key)
        ir_vals = (
            g.loc[g["parameter"] == "ir_drop", "measurement_value"].astype(float).tolist()
        )
        th_vals = (
            g.loc[g["parameter"] == "thermal", "measurement_value"].astype(float).tolist()
        )
        if ir_vals:
            ir[key] = DieParamSeries(lot, die, st, ir_vals)
        if th_vals:
            th[key] = DieParamSeries(lot, die, st, th_vals)
    return CoreDieIndex(ir_drop=ir, thermal=th)
