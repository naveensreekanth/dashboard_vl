"""Phase 13.3A — recommendation cache / month isolation performance guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dtl_agent.api.die_level_service import (
    _cache_key,
    clear_die_level_process_caches,
    get_die_recommendation,
    get_die_history,
)
from dtl_agent.data.temporal.loader import load_temporal_month
from dtl_agent.data.temporal.month_cache import (
    cached_month_keys,
    clear_temporal_month_cache,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_temporal_month_cache()
    clear_die_level_process_caches()
    yield
    clear_temporal_month_cache()
    clear_die_level_process_caches()


def test_cache_key_includes_month_lot_die_parameter():
    assert (
        _cache_key(ROOT, "2026-01", "DTL_EDGE_003", "DTL_EDGE_003_D025", "ir_drop")
        == f"{str(ROOT.resolve())}::2026-01::DTL_EDGE_003::DTL_EDGE_003_D025::ir_drop"
    )
    assert _cache_key(ROOT, "2026-01", "L", "D", "ir_drop") != _cache_key(
        ROOT, "2026-02", "L", "D", "ir_drop"
    )
    assert _cache_key(ROOT, "2026-01", "L", "D1", "ir_drop") != _cache_key(
        ROOT, "2026-01", "L", "D2", "ir_drop"
    )
    assert _cache_key(ROOT, "2026-01", "L", "D", "ir_drop") != _cache_key(
        ROOT, "2026-01", "L", "D", "thermal"
    )


def test_month_data_cache_is_month_scoped():
    """Full-month cache is month-keyed and bounded to 3 resident months (LRU)."""
    clear_temporal_month_cache()
    a = load_temporal_month("2026-01", project_root=ROOT)
    b = load_temporal_month("2026-01", project_root=ROOT)
    assert a is b
    assert a.production_month == "2026-01"
    c = load_temporal_month("2026-02", project_root=ROOT)
    assert a is not c
    assert c.production_month == "2026-02"
    # Bounded cache: holds resident uploaded months without LRU thrashing.
    months = {m for _, m in cached_month_keys()}
    assert "2026-01" in months and "2026-02" in months
    d = load_temporal_month("2026-02", project_root=ROOT)
    assert d is c


def test_disk_cache_hit_skips_recommend(monkeypatch):
    """Cached path must not call recommend()."""
    month, lot, die, param = "2026-01", "DTL_EDGE_003", "DTL_EDGE_003_D025", "IR_DROP_MV"
    # Ensure a valid disk cache exists (generate once if missing)
    payload = get_die_recommendation(
        ROOT, production_month=month, lot_id=lot, die_id=die, parameter=param
    )
    assert payload["recommendation"]["production_month"] == month
    clear_die_level_process_caches()  # drop memory; keep disk

    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("recommend() must not run on disk cache hit")

    monkeypatch.setattr("dtl_agent.api.die_level_service.recommend", _boom)
    hit = get_die_recommendation(
        ROOT, production_month=month, lot_id=lot, die_id=die, parameter=param
    )
    assert called["n"] == 0
    assert hit["cached"] is True
    assert hit["recommendation"]["recommended_limit"] == payload["recommendation"][
        "recommended_limit"
    ]


def test_memory_cache_hit_after_disk(monkeypatch):
    month, lot, die, param = "2026-01", "DTL_EDGE_003", "DTL_EDGE_003_D025", "IR_DROP_MV"
    get_die_recommendation(
        ROOT, production_month=month, lot_id=lot, die_id=die, parameter=param
    )
    # Second call should use memory; even disk read is optional — recommend forbidden
    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("recommend() must not run on memory cache hit")

    monkeypatch.setattr("dtl_agent.api.die_level_service.recommend", _boom)
    hit = get_die_recommendation(
        ROOT, production_month=month, lot_id=lot, die_id=die, parameter=param
    )
    assert called["n"] == 0
    assert hit.get("cache_source") in {"memory", "disk"}


def test_history_months_isolated():
    hist = get_die_history(
        ROOT,
        lot_id="DTL_EDGE_003",
        die_id="DTL_EDGE_003_D025",
        parameter="IR_DROP_MV",
    )
    months = [h["production_month"] for h in hist["history"]]
    assert months == ["2026-01", "2026-02", "2026-03"]
    assert len({json.dumps(h, sort_keys=True) for h in hist["history"]}) >= 1
    for h, m in zip(hist["history"], months):
        assert h["lot_id"] == "DTL_EDGE_003"
        assert h["die_id"] == "DTL_EDGE_003_D025"
        assert h["production_month"] == m
