"""Phase 12.9 — Three-month recommendation analysis artifact checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.ml.phase12_9_analysis import (
    DISPLAY_NAME,
    MONTHS,
    SCORABLE_PARAMETERS,
    _explain_from_rec,
    _json_safe,
    analysis_output_dir,
)
from dtl_agent.recommendation.routing import HybridModelId, model_for_parameter

ROOT = default_project_root()
OUT = analysis_output_dir(ROOT)


@pytest.fixture(scope="module")
def artifacts_present() -> Path:
    required = [
        "three_month_recommendations.csv",
        "three_month_recommendations.json",
        "candidate_explanations.csv",
        "temporal_changes.csv",
        "same_die_analysis.csv",
        "model_traceability.csv",
        "executive_summary.json",
        "policy_proofs.json",
    ]
    missing = [n for n in required if not (OUT / n).is_file()]
    if missing:
        pytest.skip(
            f"Phase 12.9 artifacts missing ({missing}); run scripts/phase12_9_run_analysis.py"
        )
    return OUT


def test_primary_table_has_27_engine_rows(artifacts_present: Path):
    df = pd.read_csv(artifacts_present / "three_month_recommendations.csv")
    assert len(df) == 27
    assert set(df["production_month"].astype(str)) == set(MONTHS)
    assert set(df["parameter"].astype(str)) == set(SCORABLE_PARAMETERS)
    assert (df["die_id"] == "DTL_NORM_001_D001").all()
    # Values must look like engine outputs, not placeholders
    assert df["recommended_limit"].notna().all()
    assert df["decision"].isin(["RECOMMEND", "KEEP_CURRENT", "REVIEW_REQUIRED", "REJECT"]).all()


def test_model_routing_runtime_trace(artifacts_present: Path):
    mt = pd.read_csv(artifacts_present / "model_traceability.csv")
    assert len(mt) == 9
    assert mt["routing_ok"].all()
    for _, r in mt.iterrows():
        expected = model_for_parameter(str(r["parameter"]), temporal=True)
        assert expected != HybridModelId.UNSUPPORTED
        assert str(r["models_observed"]) == expected.value


def test_month_isolation_evidence_origin(artifacts_present: Path):
    df = pd.read_csv(artifacts_present / "three_month_recommendations.csv")
    for month in MONTHS:
        sub = df[df["production_month"] == month]
        assert (sub["evidence_origin"].astype(str) == f"SIMULATOR_DERIVED_TEMPORAL_{month}").all()
    proofs = json.loads((artifacts_present / "policy_proofs.json").read_text(encoding="utf-8"))
    for row in proofs["month_isolation_checks"]:
        assert row["uses_only_month_data"] is True
        assert row["uses_only_month_sim"] is True
        assert row["legacy_simulation_forbidden"] is True
        month = row["production_month"]
        assert row["data_root"].rstrip("/").endswith(f"/{month}")
        assert f"/temporal/{month}/" in row["simulation_root"] or row[
            "simulation_root"
        ].rstrip("/").endswith(f"/temporal/{month}/simulation")
        assert "/artifacts/simulation/" not in (row["simulation_root"] + "/")


def test_yield_first_and_tie_break_proofs(artifacts_present: Path):
    exe = json.loads((artifacts_present / "executive_summary.json").read_text(encoding="utf-8"))
    assert exe["yield_first_proof_count"] >= 1
    assert exe["ml_tie_break_proof_count"] >= 1
    yf = exe["yield_first_proof_example"]
    assert yf["winner"]["simulated_yield"] > yf["loser_higher_ml"]["simulated_yield"]
    assert yf["winner"]["ml_rank"] > yf["loser_higher_ml"]["ml_rank"]
    tb = exe["ml_tie_break_proof_example"]
    assert len(tb["tied_candidates"]) >= 2
    assert tb["tied_candidates"][0]["is_selected"] is True
    assert tb["tied_candidates"][0]["ml_rank"] == 1


def test_ir_drop_changes_across_months(artifacts_present: Path):
    ch = pd.read_csv(artifacts_present / "temporal_changes.csv")
    ir = ch[(ch["parameter"] == "ir_drop") & (ch["recommendation_changed"] == True)]  # noqa: E712
    assert len(ir) == 1
    assert float(ir.iloc[0]["jan_recommendation"]) == 50.0
    assert float(ir.iloc[0]["feb_recommendation"]) == 72.0
    assert float(ir.iloc[0]["mar_recommendation"]) == 55.0


def test_same_die_categories_present(artifacts_present: Path):
    df = pd.read_csv(artifacts_present / "same_die_analysis.csv")
    cats = set(df["lot_category"].astype(str).unique())
    assert {"NORMAL", "SCRATCH", "EDGE", "CENTER"}.issubset(cats)
    for die in (
        "DTL_NORM_001_D001",
        "DTL_SCRATCH_001_D001",
        "DTL_EDGE_001_D001",
        "DTL_CENTER_001_D001",
    ):
        assert die in set(df["die_id"].astype(str))
        seqs = set(df[df["die_id"] == die]["sequence_id"].astype(str))
        assert any(s.startswith("2026-01::") for s in seqs)
        assert any(s.startswith("2026-02::") for s in seqs)
        assert any(s.startswith("2026-03::") for s in seqs)


def test_explanation_cases_from_engine_fields():
    keep = {
        "decision": "KEEP_CURRENT",
        "unit": "mV",
        "recommended_limit": 25.0,
        "explanation": {"policy_reason": "policy_selected_current", "text": "KEEP_CURRENT: current wins"},
    }
    assert "current DTL" in _explain_from_rec(keep).lower() or "KEEP_CURRENT" in _explain_from_rec(keep)

    tie = {
        "decision": "RECOMMEND",
        "unit": "mV",
        "recommended_limit": 50.0,
        "explanation": {
            "policy_reason": "max_simulated_yield_selected",
            "yield_tie": True,
            "selection_text": "Candidates had equivalent simulated yield, so ML ranking was used as the tie-breaker.",
        },
    }
    text = _explain_from_rec(tie)
    assert "tie" in text.lower()

    uniq = {
        "decision": "RECOMMEND",
        "unit": "V",
        "recommended_limit": 1.0,
        "explanation": {
            "policy_reason": "max_simulated_yield_selected",
            "yield_tie": False,
            "selected_simulated_yield": 0.888,
        },
    }
    assert "maximum simulated yield" in _explain_from_rec(uniq).lower()


def test_json_safe_handles_numpy():
    import numpy as np

    payload = _json_safe({"a": np.int64(3), "b": np.float64(1.5), "c": [np.bool_(True)]})
    json.dumps(payload)


def test_display_names_cover_scorable():
    assert set(DISPLAY_NAME) == set(SCORABLE_PARAMETERS)


def test_live_recommend_matches_artifact_ir_jan(artifacts_present: Path):
    """Smoke: one live engine call matches stored primary artifact (no hardcoded invent)."""
    from dtl_agent.recommendation import recommend
    from dtl_agent.recommendation.temporal_inference import TemporalHybridBundle

    art = pd.read_csv(artifacts_present / "three_month_recommendations.csv")
    row = art[(art["parameter"] == "ir_drop") & (art["production_month"] == "2026-01")].iloc[0]
    hybrid = TemporalHybridBundle(ROOT)
    assert hybrid.ensure_loaded()
    live = recommend(
        lot_id="DTL_NORM_001",
        die_id="DTL_NORM_001_D001",
        parameters=["ir_drop"],
        production_month="2026-01",
        project_root=ROOT,
        temporal_bundle=hybrid,
    ).recommendations[0]
    assert live.recommended_limit == pytest.approx(float(row["recommended_limit"]))
    assert live.decision.value == str(row["decision"])
    assert live.model_used == str(row["model_used"])
    assert live.ml_rank == int(row["ml_rank"])
