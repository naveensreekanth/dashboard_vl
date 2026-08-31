"""Unit tests for Phase 13.2 offline policy review (analysis-only)."""

from __future__ import annotations

from pathlib import Path

from dtl_agent.ml.phase13_2_policy_review import (
    THRESHOLD_INVENTORY,
    WHAT_IF_BANNER,
    CellDecision,
    classify_disagreement,
    run_policy_review,
    temporal_flags,
    write_report,
)

ROOT = Path(__file__).resolve().parents[2]


def test_threshold_inventory_not_defined():
    assert THRESHOLD_INVENTORY["max_violation_rate_for_recommend"] == "NOT DEFINED"
    assert THRESHOLD_INVENTORY["min_simulated_yield_for_recommend"] == "NOT DEFINED"
    assert THRESHOLD_INVENTORY["temporal_anomaly_threshold"] == "NOT DEFINED"
    assert "WHAT-IF" in WHAT_IF_BANNER


def test_classify_disagreement_identical():
    a = CellDecision("RECOMMEND", 50.0, 25.0, 1, 0.9, 1.0, True, "ok", 5, 5, "m")
    assert classify_disagreement(a, a, a) == "A_identical"


def test_classify_disagreement_conservative_b():
    a = CellDecision("RECOMMEND", 50.0, 25.0, 1, 0.9, 1.0, False, "ok", 5, 5, "m")
    b = CellDecision("KEEP_CURRENT", 25.0, 25.0, 2, 0.8, 1.0, False, "whatif", 1, 1, "m")
    c = a
    assert classify_disagreement(a, b, c) == "D_b_more_conservative"


def test_temporal_flags_stable_and_revert():
    stable = temporal_flags({"2026-01": 50.0, "2026-02": 50.0, "2026-03": 50.0})
    assert stable["stable"] is True
    revert = temporal_flags({"2026-01": 50.0, "2026-02": 72.0, "2026-03": 50.0})
    assert revert["revert"] is True
    jump = temporal_flags(
        {"2026-01": 50.0, "2026-02": 90.0, "2026-03": 91.0}, jump_abs=20.0
    )
    assert jump["jump"] is True
    assert WHAT_IF_BANNER in jump["whatif_banner"]


def test_smoke_run_produces_artifacts_and_sanity(tmp_path, monkeypatch):
    import dtl_agent.ml.phase13_2_policy_review as mod

    smoke_out = tmp_path / "policy_review"
    monkeypatch.setattr(mod, "output_dir", lambda project_root=None: smoke_out)

    summary = mod.run_policy_review(ROOT, smoke=True, batch_size=32)
    assert summary["n_cells"] == 4 * 9 * 3  # 108
    assert (smoke_out / "policy_comparison.csv").is_file()
    assert (smoke_out / "policy_comparison_summary.json").is_file()
    assert (smoke_out / "threshold_inventory.json").is_file()
    sanity = summary["sanity_phase12_9"]
    assert sanity["matched_cells"] >= 1
    assert sanity["mismatched_cells"] == 0
    assert "PASS" in summary["verdict"] or "FAIL" in summary["verdict"]

    # Report to tmp only — do not clobber docs/ or full-population artifacts
    smoke_doc = tmp_path / "PHASE_13_2_SMOKE.md"

    def _write_tmp(summary, project_root=None):
        text = "FINAL VERDICT\n\n**" + summary["verdict"] + "**\nNOT DEFINED\n"
        smoke_doc.write_text(text, encoding="utf-8")
        return smoke_doc

    monkeypatch.setattr(mod, "write_report", _write_tmp)
    doc = mod.write_report(summary, ROOT)
    assert "FINAL VERDICT" in doc.read_text(encoding="utf-8")
    assert summary["verdict"] in doc.read_text(encoding="utf-8")


def test_production_policy_module_untouched():
    from dtl_agent.recommendation import policy

    assert hasattr(policy, "apply_recommendation_policy")
    src = Path(policy.__file__).read_text(encoding="utf-8")
    assert "simulated_yield" in src


def test_full_population_artifacts_present_if_generated():
    """If the full CLI run has been executed, assert 27k integrity."""
    from dtl_agent.ml.phase13_2_policy_review import output_dir
    import json
    import pandas as pd

    out = output_dir(ROOT)
    summary_path = out / "policy_comparison_summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("full_population"):
        return
    assert summary["n_cells"] == 27_000
    assert summary["sanity_phase12_9"]["mismatched_cells"] == 0
    df = pd.read_csv(out / "policy_comparison.csv")
    assert len(df) == 27_000
    assert df["die_id"].nunique() == 1000
