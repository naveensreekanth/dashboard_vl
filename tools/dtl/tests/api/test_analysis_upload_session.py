"""API tests for POST /analysis/upload and session-scoped analysis."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dtl_agent.api.analysis_session import clear_all_sessions, get_session
from dtl_agent.api.app import create_app
from dtl_agent.api.settings import ServiceSettings

ROOT = Path(__file__).resolve().parents[2]
LOT, DIE = "DTL_NORM_001", "DTL_NORM_001_D001"
MONTHS = ("2026-01", "2026-02", "2026-03")
OUT = ROOT / "artifacts" / "tmp_upload_analysis_api"
OUT.mkdir(parents=True, exist_ok=True)


def _client() -> TestClient:
    return TestClient(create_app(ServiceSettings.from_env()))


def _extract_die_csv(month: str) -> Path:
    src = ROOT / "data" / "3 months data" / month / "actual_die" / "measurements.csv"
    dest = OUT / f"{month.replace('-', '')}_{DIE}_actual.csv"
    if dest.exists():
        dest.unlink()
    with src.open(newline="", encoding="utf-8") as f, dest.open(
        "w", newline="", encoding="utf-8"
    ) as g:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(g, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["lot_id"] == LOT and row["die_id"] == DIE:
                writer.writerow(row)
    return dest


@pytest.fixture(autouse=True)
def _cleanup_sessions():
    yield
    clear_all_sessions()


def test_analysis_upload_rejects_missing_month():
    client = _client()
    jan = _extract_die_csv("2026-01")
    resp = client.post(
        "/api/v1/analysis/upload",
        files={
            "january": ("jan.csv", jan.read_bytes(), "text/csv"),
            "february": ("feb.csv", jan.read_bytes(), "text/csv"),
            # march missing
        },
    )
    assert resp.status_code == 422


def test_analysis_upload_cross_month_slot_mapping():
    """Verify UI slot month is authoritative for analysis regardless of internal month."""
    client = _client()
    jan_csv = _extract_die_csv("2026-01")
    feb_csv = _extract_die_csv("2026-02")
    mar_csv = _extract_die_csv("2026-03")

    # March data in January slot, January data in March slot
    resp = client.post(
        "/api/v1/analysis/upload",
        files={
            "january": ("mar.csv", mar_csv.read_bytes(), "text/csv"),
            "february": ("feb.csv", feb_csv.read_bytes(), "text/csv"),
            "march": ("jan.csv", jan_csv.read_bytes(), "text/csv"),
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    sid = body["analysis_session_id"]

    import time
    for _ in range(30):
        st = client.get(f"/api/v1/analysis/upload/status/{sid}")
        assert st.status_code == 200
        data = st.json()
        if data["status"] == "completed":
            break
        time.sleep(0.5)
    else:
        pytest.fail("Upload background job did not complete within 15 seconds")

    sess = get_session(sid)
    jan_path = sess.root / "data" / "3 months data" / "2026-01" / "actual_die" / "measurements.csv"
    assert jan_path.is_file()
    jan_text = jan_path.read_text(encoding="utf-8")
    assert "2026-01" in jan_text

    assert data["used_uploaded_measurements"] is True
    mappings = data.get("month_mappings") or {}
    assert mappings["2026-01"]["original_production_month"] == "2026-03"
    assert mappings["2026-01"]["analysis_month"] == "2026-01"
    assert mappings["2026-03"]["original_production_month"] == "2026-01"
    assert mappings["2026-03"]["analysis_month"] == "2026-03"


def test_analysis_upload_rejects_empty_file():
    client = _client()
    jan = _extract_die_csv("2026-01")
    feb = _extract_die_csv("2026-02")
    resp = client.post(
        "/api/v1/analysis/upload",
        files={
            "january": ("jan.csv", jan.read_bytes(), "text/csv"),
            "february": ("feb.csv", feb.read_bytes(), "text/csv"),
            "march": ("mar.csv", b"", "text/csv"),
        },
    )
    assert resp.status_code == 422


def test_static_three_month_still_works_without_session():
    client = _client()
    resp = client.get("/api/v1/analysis/three-month")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("used_uploaded_measurements") is not True
    assert data.get("primary_recommendations")


@pytest.mark.slow
def test_upload_session_analysis_isolation_and_cost_savings():
    client = _client()
    files = {m: _extract_die_csv(m) for m in MONTHS}
    resp = client.post(
        "/api/v1/analysis/upload",
        files={
            "january": ("jan.csv", files["2026-01"].read_bytes(), "text/csv"),
            "february": ("feb.csv", files["2026-02"].read_bytes(), "text/csv"),
            "march": ("mar.csv", files["2026-03"].read_bytes(), "text/csv"),
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    sid = body["analysis_session_id"]
    assert body["status"] in ("queued", "processing", "completed")
    assert body["used_uploaded_measurements"] is True
    assert body["used_static_three_month_measurements"] is False

    # Poll status endpoint until completed
    import time
    for _ in range(30):
        st = client.get(f"/api/v1/analysis/upload/status/{sid}")
        assert st.status_code == 200
        data = st.json()
        if data["status"] == "completed":
            break
        time.sleep(0.5)
    else:
        pytest.fail("Upload background job did not complete within 15 seconds")

    sess = get_session(sid)
    # Session sandbox must not be the repository root
    assert sess.root.resolve() != ROOT.resolve()
    # Must not contain a copy of repo static measurements path content for other dies
    for month in MONTHS:
        mpath = sess.root / "data" / "3 months data" / month / "actual_die" / "measurements.csv"
        assert mpath.is_file()
        text = mpath.read_text(encoding="utf-8")
        assert DIE in text
        # Isolation: only uploaded die
        assert "DTL_EDGE_001" not in text

    # Repo static files untouched / not used as session path
    repo_jan = (ROOT / "data" / "3 months data" / "2026-01" / "actual_die" / "measurements.csv").resolve()
    sess_jan = (
        sess.root / "data" / "3 months data" / "2026-01" / "actual_die" / "measurements.csv"
    ).resolve()
    assert sess_jan != repo_jan

    # No phase_13 precomputed cache copied into session
    assert not (sess.root / "artifacts" / "temporal" / "shared" / "phase_13_1_die_recommendations").exists()

    analysis = client.get(f"/api/v1/analysis/three-month?analysis_session_id={sid}")
    assert analysis.status_code == 200
    ab = analysis.json()
    assert ab["used_uploaded_measurements"] is True
    assert ab["used_static_three_month_measurements"] is False
    assert "uploaded" in (ab.get("data_provenance") or "").lower()
    assert ab["die_level_identities"]["dies_by_lot"][LOT] == [DIE]

    dies = client.get(
        "/api/v1/analysis/three-month/dies",
        params={
            "analysis_session_id": sid,
            "production_month": "2026-01",
            "lot_id": LOT,
            "die_id": DIE,
            "parameter": "ir_drop",
        },
    )
    assert dies.status_code == 200, dies.text
    assert dies.json()["used_uploaded_measurements"] is True
    assert dies.json()["recommendation"]["recommended_limit"] is not None

    hist = client.get(
        "/api/v1/analysis/three-month/die-history",
        params={
            "analysis_session_id": sid,
            "lot_id": LOT,
            "die_id": DIE,
            "parameter": "ir_drop",
        },
    )
    assert hist.status_code == 200
    assert len(hist.json()["history"]) == 3

    obs = client.get(
        "/api/v1/analysis/three-month/observed",
        params={"analysis_session_id": sid, "lot_id": LOT, "die_id": DIE},
    )
    assert obs.status_code == 200

    cost = client.get(
        "/api/v1/analysis/cost-savings",
        params={"analysis_session_id": sid, "include_per_device": False},
    )
    assert cost.status_code == 200
    cb = cost.json()
    assert cb["used_uploaded_measurements"] is True
    assert cb["disclaimer"] == "Counterfactual estimate — not measured ATE savings."

    # Unknown session must not fall back to static
    bad = client.get("/api/v1/analysis/three-month?analysis_session_id=does-not-exist")
    assert bad.status_code == 404


def test_upload_status_endpoint_unknown_session_returns_failed_state():
    client = _client()
    resp = client.get("/api/v1/analysis/upload/status/unknown-session-id")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "re-upload" in data["error"].lower()
