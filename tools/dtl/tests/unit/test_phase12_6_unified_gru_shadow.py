"""Phase 12.6 — Unified GRU recommendation shadow evaluation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.paths import month_simulation_root, temporal_data_root
from dtl_agent.ml.models.unified_gru_ranker import UNIFIED_PARAMETER_VOCAB
from dtl_agent.ml.unified_experiment import FORBIDDEN_INPUT_COLS
from dtl_agent.ml.unified_shadow import (
    EXCLUDED_SEQUENCE_ONLY,
    SHADOW_PARAMETERS,
    FORBIDDEN_MODEL_FEATURES,
    _cand_frame_from_results,
    _month_rec_config,
)
from dtl_agent.recommendation.catalog import CandidateCatalog
from dtl_agent.recommendation.evidence import SimulationEvidenceLookup
from dtl_agent.recommendation.pipeline import recommend
from dtl_agent.recommendation.policy import EvaluatedCandidate, apply_recommendation_policy
from dtl_agent.recommendation.safety import evaluate_safety
from dtl_agent.recommendation.schemas import GateStatus, RankedCandidate, SimulationEvidence

ROOT = default_project_root()
TEMPORAL_AVAILABLE = (temporal_data_root(ROOT) / "2026-01" / "actual_die" / "measurements.csv").is_file()
SHADOW_OUT = ROOT / "artifacts" / "temporal" / "shared" / "unified_shadow"

pytestmark = pytest.mark.skipif(
    not TEMPORAL_AVAILABLE,
    reason="data/3 months data package not present",
)


def test_shadow_parameters_are_nine_scorable_only():
    assert len(SHADOW_PARAMETERS) == 9
    assert set(SHADOW_PARAMETERS) == set(UNIFIED_PARAMETER_VOCAB)
    assert EXCLUDED_SEQUENCE_ONLY == frozenset({"setup_slack", "hold_slack", "test_time"})
    assert EXCLUDED_SEQUENCE_ONLY.isdisjoint(SHADOW_PARAMETERS)


def test_same_candidate_set_core_and_parametric_artifacts():
    for month in ("2026-01", "2026-02", "2026-03"):
        core = pd.read_csv(month_simulation_root(month, ROOT) / "core" / "candidate_results.csv")
        param = pd.read_csv(
            month_simulation_root(month, ROOT) / "parametric" / "candidate_results.csv"
        )
        for p in ("ir_drop", "thermal"):
            a = _cand_frame_from_results(
                month_simulation_root(month, ROOT) / "core" / "candidate_results.csv", p
            )
            assert set(a["parameter"].unique()) == {p}
            assert len(a) == len(core[core["parameter"] == p])
        for p in (
            "VMIN",
            "VMAX",
            "IDDQ",
            "SUPPLY_CURRENT",
            "CONTACT_RESISTANCE",
            "INTERCONNECT_RESISTANCE",
            "ON_RESISTANCE",
        ):
            a = _cand_frame_from_results(
                month_simulation_root(month, ROOT) / "parametric" / "candidate_results.csv", p
            )
            assert len(a) == len(param[param["parameter"] == p])


def test_same_simulation_and_safety_evidence_month_isolated():
    cfg_jan = _month_rec_config("2026-01", ROOT)
    cfg_feb = _month_rec_config("2026-02", ROOT)
    ev_jan = SimulationEvidenceLookup(ROOT, cfg_jan)
    ev_feb = SimulationEvidenceLookup(ROOT, cfg_feb)
    e1 = ev_jan.lookup(domain="core", parameter="ir_drop", candidate_limit=25.0)
    e2 = ev_feb.lookup(domain="core", parameter="ir_drop", candidate_limit=25.0)
    assert e1.found and e2.found
    assert "2026-01" in cfg_jan.core_candidate_results_path.replace("\\", "/")
    assert "2026-02" in cfg_feb.core_candidate_results_path.replace("\\", "/")
    assert "2026-01" not in cfg_feb.core_candidate_results_path.replace("\\", "/")

    catalog = CandidateCatalog(ROOT, cfg_jan)
    cand = RankedCandidate(
        parameter="ir_drop",
        test_id="T_IR_DROP_MV",
        lot_id="L",
        die_id="D",
        current_limit=25.0,
        candidate_limit=25.0,
        delta_absolute=0.0,
        delta_percent=0.0,
        direction="UPPER",
        tighten_or_loosen="CURRENT",
        unit="mV",
        source_status="SOURCE_CONFIRMED",
        ml_score=0.5,
        ml_rank=1,
        model_id="test",
        catalog_valid=True,
    )
    safety = evaluate_safety(
        candidate=cand,
        evidence=e1,
        catalog=catalog,
        config=cfg_jan,
        domain="core",
        context_complete=True,
        model_available=True,
    )
    assert safety.status == GateStatus.PASS


def test_unified_forbidden_features_exclude_simulation_outcomes():
    assert "simulated_yield" in FORBIDDEN_MODEL_FEATURES
    assert "objective_score" in FORBIDDEN_MODEL_FEATURES
    assert "violation_rate" in FORBIDDEN_MODEL_FEATURES
    assert FORBIDDEN_MODEL_FEATURES == FORBIDDEN_INPUT_COLS


def test_yield_primary_ml_only_tiebreak():
    def _rc(lim: float, rank: int, score: float) -> RankedCandidate:
        return RankedCandidate(
            parameter="ir_drop",
            test_id="T_IR_DROP_MV",
            lot_id="L",
            die_id="D",
            current_limit=25.0,
            candidate_limit=lim,
            delta_absolute=lim - 25.0,
            delta_percent=None,
            direction="UPPER",
            tighten_or_loosen="CURRENT" if abs(lim - 25.0) < 1e-12 else "TIGHTER",
            unit="mV",
            source_status="SOURCE_CONFIRMED",
            ml_score=score,
            ml_rank=rank,
            model_id="test",
            catalog_valid=True,
        )

    def _ev(lim: float, y: float) -> SimulationEvidence:
        return SimulationEvidence(
            evidence_origin="TEST",
            population_level_aggregate=True,
            parameter="ir_drop",
            candidate_limit=lim,
            simulated_yield=y,
            found=True,
        )

    class _Safe:
        status = GateStatus.PASS

    high_yield_low_ml = EvaluatedCandidate(
        candidate=_rc(20.0, rank=4, score=0.1),
        evidence=_ev(20.0, 0.99),
        safety=_Safe(),
    )
    low_yield_high_ml = EvaluatedCandidate(
        candidate=_rc(22.0, rank=1, score=0.9),
        evidence=_ev(22.0, 0.98),
        safety=_Safe(),
    )
    current = EvaluatedCandidate(
        candidate=_rc(25.0, rank=3, score=0.2),
        evidence=_ev(25.0, 0.95),
        safety=_Safe(),
    )
    pol = apply_recommendation_policy(
        evaluated=[high_yield_low_ml, low_yield_high_ml, current],
        current_limit=25.0,
    )
    assert pol.selected is not None
    assert abs(pol.selected.candidate_limit - 20.0) < 1e-12
    assert pol.selected.ml_rank == 4


def test_recommend_source_unchanged_by_shadow():
    src = Path(recommend.__code__.co_filename).read_text(encoding="utf-8")
    assert "UnifiedParameterGRU" not in src
    assert "unified_parameter_gru" not in src
    assert "unified_shadow" not in src


@pytest.mark.skipif(
    not (SHADOW_OUT / "recommendation_comparison.csv").is_file(),
    reason="shadow artifacts not generated yet",
)
def test_shadow_artifacts_cover_nine_parameters_three_months():
    detail = pd.read_csv(SHADOW_OUT / "recommendation_comparison.csv")
    assert set(detail["parameter"].unique()) == set(SHADOW_PARAMETERS)
    assert set(detail["month"].astype(str).unique()) == {"2026-01", "2026-02", "2026-03"}
    assert detail["existing_ml_score"].notna().all()
    assert detail["unified_ml_score"].notna().all()
    assert detail["simulated_yield"].notna().all()
    assert detail["safety_status"].notna().all()
    assert detail.groupby(["month", "parameter"])["candidate_limit"].nunique().min() >= 1
