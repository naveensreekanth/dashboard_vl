"""Unit coverage for upload analysis session parsing / isolation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.api.analysis_session import clear_all_sessions, get_session, register_session
from dtl_agent.api.upload_analysis import parse_month_package
from dtl_agent.api.upload_recommendation import UploadRecommendationError

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "upload_sample_die_actual.csv"


def test_parse_month_package_accepts_fixture_for_jan():
    raw = FIXTURE.read_bytes()
    pkg = parse_month_package(filename="jan.csv", content=raw, expected_month="2026-01")
    assert pkg.production_month == "2026-01"
    assert not pkg.actual_die.empty


def test_parse_month_package_maps_different_month_to_expected_slot():
    raw = FIXTURE.read_bytes()
    pkg = parse_month_package(filename="feb.csv", content=raw, expected_month="2026-02")
    assert pkg.production_month == "2026-02"
    assert pkg.original_production_month == "2026-01"
    assert (pkg.actual_die["production_month"] == "2026-02").all()


def test_parse_month_package_rejects_empty():
    with pytest.raises(UploadRecommendationError, match="empty"):
        parse_month_package(filename="jan.csv", content=b"", expected_month="2026-01")


def test_session_registry_isolates_roots(tmp_path):
    clear_all_sessions()
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    sa = register_session(a, months=("2026-01", "2026-02", "2026-03"))
    sb = register_session(b, months=("2026-01", "2026-02", "2026-03"))
    assert sa.analysis_session_id != sb.analysis_session_id
    assert get_session(sa.analysis_session_id).root == a.resolve()
    assert get_session(sb.analysis_session_id).root == b.resolve()
    clear_all_sessions()
