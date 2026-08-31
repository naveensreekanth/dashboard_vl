"""Tests for upload → recommend adapter (validation + isolation from static data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtl_agent.api.upload_recommendation import (
    UploadRecommendationError,
    parse_upload_payload,
    recommend_from_upload,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "upload_sample_die_actual.csv"


def test_parse_valid_actual_die_csv():
    raw = FIXTURE.read_bytes()
    parsed = parse_upload_payload(filename="upload_sample_die_actual.csv", content=raw)
    assert parsed.lot_id == "DTL_NORM_001"
    assert parsed.die_id == "DTL_NORM_001_D001"
    assert parsed.production_month == "2026-01"
    assert not parsed.actual_die.empty
    assert parsed.parametric is None


def test_reject_empty_file():
    with pytest.raises(UploadRecommendationError, match="empty"):
        parse_upload_payload(filename="x.csv", content=b"")


def test_reject_invalid_extension():
    with pytest.raises(UploadRecommendationError, match="Unsupported file type"):
        parse_upload_payload(filename="chip.stdf", content=b"not-a-real-stdf")


def test_reject_malformed_csv():
    with pytest.raises(UploadRecommendationError, match="Malformed|missing required"):
        parse_upload_payload(filename="bad.csv", content=b"this,is,not,valid\n1,2")


def test_reject_missing_required_columns():
    csv = b"lot_id,die_id,production_month\nA,B,2026-01\n"
    with pytest.raises(UploadRecommendationError, match="missing required columns"):
        parse_upload_payload(filename="partial.csv", content=csv)


@pytest.mark.slow
def test_recommend_from_upload_uses_uploaded_data_not_static_cache():
    """End-to-end: materialize sandbox + existing recommend(); mark upload provenance."""
    if not FIXTURE.is_file():
        pytest.skip("fixture missing")
    raw = FIXTURE.read_bytes()
    out = recommend_from_upload(
        filename="upload_sample_die_actual.csv",
        content=raw,
        source_root=ROOT,
        parameters=["ir_drop", "thermal"],
    )
    assert out["upload"]["used_uploaded_measurements"] is True
    assert out["upload"]["used_static_three_month_measurements"] is False
    assert out["upload"]["lot_id"] == "DTL_NORM_001"
    assert out["upload"]["die_id"] == "DTL_NORM_001_D001"
    assert out["upload"]["production_month"] == "2026-01"
    assert "recommendations" in out
    assert len(out["recommendations"]) >= 1
    for rec in out["recommendations"]:
        assert rec["lot_id"] == "DTL_NORM_001"
        assert rec["die_id"] == "DTL_NORM_001_D001"
        assert rec["parameter"] in {"ir_drop", "thermal"}
