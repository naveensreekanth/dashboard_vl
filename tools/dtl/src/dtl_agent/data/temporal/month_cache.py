"""In-process caches for temporal month / die packages (Phase 13.3A + Render Free).

Full-month frames are capped at one resident entry (LRU). Prefer die-scoped
loads for recommendation traffic so Jan/Feb/Mar never sit in RAM together.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dtl_agent.data.temporal.loader import TemporalMonthData

# Up to 3 full-month packages resident in process memory (Jan/Feb/Mar).
_MAX_FULL_MONTHS = 3
# Die slices are tiny; keep a few for repeated dashboard clicks.
_MAX_DIE_SLICES = 8

_lock = threading.Lock()
_month_data: OrderedDict[tuple[str, str], TemporalMonthData] = OrderedDict()
_die_data: OrderedDict[tuple[str, str, str, str], TemporalMonthData] = OrderedDict()


def _month_key(project_root: Path, production_month: str) -> tuple[str, str]:
    return (str(project_root.resolve()), str(production_month))


def _die_key(
    project_root: Path, production_month: str, lot_id: str, die_id: str
) -> tuple[str, str, str, str]:
    return (str(project_root.resolve()), str(production_month), str(lot_id), str(die_id))


def get_cached_month(project_root: Path, production_month: str) -> TemporalMonthData | None:
    with _lock:
        key = _month_key(project_root, production_month)
        hit = _month_data.get(key)
        if hit is not None:
            _month_data.move_to_end(key)
        return hit


def put_cached_month(
    project_root: Path, production_month: str, data: TemporalMonthData
) -> TemporalMonthData:
    with _lock:
        key = _month_key(project_root, production_month)
        _month_data[key] = data
        _month_data.move_to_end(key)
        while len(_month_data) > _MAX_FULL_MONTHS:
            _month_data.popitem(last=False)
        return data


def get_cached_die(
    project_root: Path, production_month: str, lot_id: str, die_id: str
) -> TemporalMonthData | None:
    with _lock:
        key = _die_key(project_root, production_month, lot_id, die_id)
        hit = _die_data.get(key)
        if hit is not None:
            _die_data.move_to_end(key)
        return hit


def put_cached_die(
    project_root: Path,
    production_month: str,
    lot_id: str,
    die_id: str,
    data: TemporalMonthData,
) -> TemporalMonthData:
    with _lock:
        key = _die_key(project_root, production_month, lot_id, die_id)
        _die_data[key] = data
        _die_data.move_to_end(key)
        while len(_die_data) > _MAX_DIE_SLICES:
            _die_data.popitem(last=False)
        return data


def clear_temporal_month_cache() -> None:
    """Test / reload helper — drops month and die frames from process memory."""
    with _lock:
        _month_data.clear()
        _die_data.clear()


def cached_month_keys() -> list[tuple[str, str]]:
    with _lock:
        return list(_month_data.keys())


def cached_die_keys() -> list[tuple[str, str, str, str]]:
    with _lock:
        return list(_die_data.keys())
