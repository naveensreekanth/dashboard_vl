"""Phase 12.9 — Three-month recommendation analysis & explainability.

Read-only analysis over live ``recommend(production_month=...)`` outputs.
Does not modify models, checkpoints, policy, safety, yield, or catalogs.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.loader import load_temporal_month
from dtl_agent.data.temporal.paths import month_root, month_simulation_root, temporal_artifact_root
from dtl_agent.features.io_utils import write_json
from dtl_agent.recommendation import recommend
from dtl_agent.recommendation.routing import model_for_parameter
from dtl_agent.recommendation.temporal_inference import TemporalHybridBundle

MONTHS = ("2026-01", "2026-02", "2026-03")
MONTH_LABEL = {"2026-01": "Jan 2026", "2026-02": "Feb 2026", "2026-03": "Mar 2026"}

SCORABLE_PARAMETERS = (
    "ir_drop",
    "thermal",
    "VMIN",
    "VMAX",
    "IDDQ",
    "SUPPLY_CURRENT",
    "CONTACT_RESISTANCE",
    "INTERCONNECT_RESISTANCE",
    "ON_RESISTANCE",
)

DISPLAY_NAME = {
    "ir_drop": "IR_DROP_MV",
    "thermal": "THERMAL_C",
    "VMIN": "VMIN",
    "VMAX": "VMAX",
    "IDDQ": "IDDQ",
    "SUPPLY_CURRENT": "SUPPLY_CURRENT",
    "CONTACT_RESISTANCE": "CONTACT_RESISTANCE",
    "INTERCONNECT_RESISTANCE": "INTERCONNECT_RESISTANCE",
    "ON_RESISTANCE": "ON_RESISTANCE",
}

# Primary analysis die + representative lot categories present in all months.
PRIMARY_DIE = ("DTL_NORM_001", "DTL_NORM_001_D001", "NORMAL")
SAME_DIE_SET = (
    PRIMARY_DIE,
    ("DTL_SCRATCH_001", "DTL_SCRATCH_001_D001", "SCRATCH"),
    ("DTL_EDGE_001", "DTL_EDGE_001_D001", "EDGE"),
    ("DTL_CENTER_001", "DTL_CENTER_001_D001", "CENTER"),
)


def analysis_output_dir(project_root: Path | None = None) -> Path:
    root = project_root or default_project_root()
    return temporal_artifact_root(root) / "shared" / "phase_12_9_analysis"


def _json_safe(obj: Any) -> Any:
    """Convert numpy / pandas scalars for JSON dumps."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


def _safety_status(rec: dict[str, Any]) -> str:
    sr = rec.get("safety_result") or {}
    return str(sr.get("status") or "")


def _is_eligible_pass(status: str) -> bool:
    # Policy eligibility requires GateStatus.PASS only.
    return str(status).upper() == "PASS"


def _delta_pct(current: float, recommended: float) -> float | None:
    if current is None or abs(float(current)) < 1e-15:
        return None
    return float((recommended - current) / current * 100.0)


def _explain_from_rec(rec: dict[str, Any]) -> str:
    """Build Case A/B/C text from recorded engine fields only."""
    exp = rec.get("explanation") or {}
    decision = str(rec.get("decision") or "")
    reason = str(exp.get("policy_reason") or "")
    yield_tie = bool(exp.get("yield_tie"))
    text = exp.get("text")
    selection_text = exp.get("selection_text")
    unit = rec.get("unit") or ""
    lim = rec.get("recommended_limit")
    lim_s = f"{lim} {unit}".strip() if lim is not None else "selected limit"

    if decision == "KEEP_CURRENT":
        if text:
            return str(text)
        return (
            f"The current DTL ({lim_s}) remains selected because it is the winning "
            f"eligible candidate under the existing yield-first policy "
            f"(policy_reason={reason!r})."
        )
    if yield_tie or (selection_text and "tie" in str(selection_text).lower()):
        base = (
            f"Several eligible candidates had the same maximum simulated yield. "
            f"ML rank was used as the tie-breaker; {lim_s} had the best ML ranking "
            f"among the tied candidates (policy_reason={reason!r})."
        )
        if selection_text:
            return f"{selection_text} {base}"
        return base
    if reason == "max_simulated_yield_selected" or decision == "RECOMMEND":
        y = exp.get("selected_simulated_yield")
        y_s = f" (simulated_yield={y})" if y is not None else ""
        return (
            f"{lim_s} was selected because it was the eligible candidate with the "
            f"maximum simulated yield{y_s} (policy_reason={reason!r})."
        )
    if text:
        return str(text)
    return f"Decision={decision}; policy_reason={reason!r}."


def _candidate_rows_for_param(
    *,
    lot_id: str,
    die_id: str,
    month: str,
    parameter: str,
    rec: dict[str, Any],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Join gate-set sim/safety rows with ML scores from the scored candidate set.

    Pipeline audits simulation + safety only for ``select_top_n_plus_current``;
    those rows are parallel in parameter-filtered order. Full ``candidate_set``
    provides ML score/rank for every scored catalog candidate.
    """
    cand_by_limit = {
        float(c["candidate_limit"]): c
        for c in (audit.get("candidate_set") or [])
        if str(c.get("parameter")) == parameter and c.get("candidate_limit") is not None
    }
    param_sims = [
        s
        for s in (audit.get("simulation_evidence_rows") or [])
        if str(s.get("parameter")) == parameter
    ]
    param_saf = [
        s
        for s in (audit.get("safety_check_trace") or [])
        if str(s.get("parameter")) == parameter
    ]
    selected = float(rec["recommended_limit"]) if rec.get("recommended_limit") is not None else None
    current = float(rec["current_limit"]) if rec.get("current_limit") is not None else None
    out: list[dict[str, Any]] = []

    # Gate-evaluated candidates (primary explanation set)
    for i, sim in enumerate(param_sims):
        lim = float(sim["candidate_limit"])
        saf = param_saf[i] if i < len(param_saf) else {}
        status = str(saf.get("status") or "UNKNOWN")
        c = cand_by_limit.get(lim, {})
        out.append(
            {
                "production_month": month,
                "lot_id": lot_id,
                "die_id": die_id,
                "parameter": parameter,
                "parameter_display": DISPLAY_NAME[parameter],
                "candidate_limit": lim,
                "simulated_yield": sim.get("simulated_yield"),
                "safety_status": status,
                "eligible": _is_eligible_pass(status),
                "in_policy_gate_set": True,
                "ml_score": c.get("ml_score"),
                "ml_rank": c.get("ml_rank"),
                "is_current": current is not None and abs(lim - current) < 1e-12,
                "is_selected": selected is not None and abs(lim - selected) < 1e-12,
                "model_used": rec.get("model_used"),
                "decision": rec.get("decision"),
            }
        )

    gated_limits = {float(r["candidate_limit"]) for r in out}
    # Additional scored candidates (context; not in TOP_N gate set)
    for lim, c in sorted(cand_by_limit.items(), key=lambda kv: int(kv[1].get("ml_rank") or 999)):
        if lim in gated_limits:
            continue
        out.append(
            {
                "production_month": month,
                "lot_id": lot_id,
                "die_id": die_id,
                "parameter": parameter,
                "parameter_display": DISPLAY_NAME[parameter],
                "candidate_limit": lim,
                "simulated_yield": None,
                "safety_status": "NOT_IN_GATE_SET",
                "eligible": False,
                "in_policy_gate_set": False,
                "ml_score": c.get("ml_score"),
                "ml_rank": c.get("ml_rank"),
                "is_current": current is not None and abs(lim - current) < 1e-12,
                "is_selected": selected is not None and abs(lim - selected) < 1e-12,
                "model_used": rec.get("model_used"),
                "decision": rec.get("decision"),
            }
        )
    return out


def _find_yield_first_proof(candidate_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Eligible A has higher yield than B but worse ML rank; A still wins (selected)."""
    eligible = [r for r in candidate_rows if r.get("eligible") and r.get("simulated_yield") is not None]
    selected = next((r for r in eligible if r.get("is_selected")), None)
    if selected is None:
        return None
    for other in eligible:
        if other is selected:
            continue
        if float(selected["simulated_yield"]) <= float(other["simulated_yield"]):
            continue
        # selected has strictly higher yield
        sel_rank = selected.get("ml_rank")
        oth_rank = other.get("ml_rank")
        if sel_rank is None or oth_rank is None:
            continue
        if int(sel_rank) > int(oth_rank):
            # selected worse ML rank (higher number) but higher yield → yield-first proof
            return {
                "kind": "yield_first",
                "production_month": selected["production_month"],
                "parameter": selected["parameter"],
                "parameter_display": selected["parameter_display"],
                "lot_id": selected["lot_id"],
                "die_id": selected["die_id"],
                "winner": {
                    "candidate_limit": selected["candidate_limit"],
                    "simulated_yield": selected["simulated_yield"],
                    "ml_score": selected["ml_score"],
                    "ml_rank": selected["ml_rank"],
                },
                "loser_higher_ml": {
                    "candidate_limit": other["candidate_limit"],
                    "simulated_yield": other["simulated_yield"],
                    "ml_score": other["ml_score"],
                    "ml_rank": other["ml_rank"],
                },
                "statement": (
                    "Winner has higher simulated_yield but worse ML rank than the "
                    "comparison candidate; yield-first policy still selected the winner."
                ),
            }
    return None


def _find_tie_break_proof(candidate_rows: list[dict[str, Any]], rec: dict[str, Any]) -> dict[str, Any] | None:
    exp = rec.get("explanation") or {}
    if not exp.get("yield_tie"):
        return None
    eligible = [r for r in candidate_rows if r.get("eligible") and r.get("simulated_yield") is not None]
    if len(eligible) < 2:
        return None
    max_y = max(float(r["simulated_yield"]) for r in eligible)
    tied = [r for r in eligible if abs(float(r["simulated_yield"]) - max_y) < 1e-12]
    if len(tied) < 2:
        return None
    selected = next((r for r in tied if r.get("is_selected")), None)
    if selected is None:
        return None
    others = [r for r in tied if not r.get("is_selected")]
    return {
        "kind": "ml_tie_break",
        "production_month": selected["production_month"],
        "parameter": selected["parameter"],
        "parameter_display": selected["parameter_display"],
        "lot_id": selected["lot_id"],
        "die_id": selected["die_id"],
        "tied_yield": max_y,
        "tied_candidates": [
            {
                "candidate_limit": r["candidate_limit"],
                "simulated_yield": r["simulated_yield"],
                "ml_score": r["ml_score"],
                "ml_rank": r["ml_rank"],
                "is_selected": r.get("is_selected"),
            }
            for r in sorted(tied, key=lambda x: int(x.get("ml_rank") or 999))
        ],
        "winner_limit": selected["candidate_limit"],
        "statement": (
            "Multiple eligible candidates share the maximum simulated yield; "
            "ML rank (best among tied) determined the final selection."
        ),
        "other_count": len(others),
    }


def _measurement_summary(month: str, lot_id: str, die_id: str, parameter: str, project_root: Path) -> dict[str, Any]:
    data = load_temporal_month(month, project_root=project_root)
    ad = data.actual_die
    # core params live on actual_die; parametric on parametric table
    if parameter in {"ir_drop", "thermal"}:
        src = ad
    else:
        src = data.parametric
    sub = src[
        (src["lot_id"].astype(str) == lot_id)
        & (src["die_id"].astype(str) == die_id)
        & (src["parameter"].astype(str) == parameter)
    ]
    if sub.empty:
        return {"n": 0, "mean": None, "min": None, "max": None}
    vals = sub["measurement_value"].astype(float)
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def run_phase12_9_analysis(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or default_project_root()
    out_dir = analysis_output_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    hybrid = TemporalHybridBundle(root)
    if not hybrid.ensure_loaded():
        raise RuntimeError(f"Temporal hybrid bundle failed to load: {hybrid.load_errors}")

    recommendation_rows: list[dict[str, Any]] = []
    candidate_expl_rows: list[dict[str, Any]] = []
    same_die_rows: list[dict[str, Any]] = []
    yield_first_proofs: list[dict[str, Any]] = []
    tie_break_proofs: list[dict[str, Any]] = []
    month_isolation: list[dict[str, Any]] = []

    # Cache recommend results: (month, lot, die) -> LotRecommendationResult.to_dict()
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _get(month: str, lot_id: str, die_id: str) -> dict[str, Any]:
        key = (month, lot_id, die_id)
        if key not in cache:
            print(f"recommend {month} {lot_id}/{die_id} ...", flush=True)
            result = recommend(
                lot_id=lot_id,
                die_id=die_id,
                parameters=list(SCORABLE_PARAMETERS),
                production_month=month,
                project_root=root,
                temporal_bundle=hybrid,
            )
            cache[key] = result.to_dict()
        return cache[key]

    # Primary 27 + same-die set
    for lot_id, die_id, category in SAME_DIE_SET:
        for month in MONTHS:
            payload = _get(month, lot_id, die_id)
            assert payload.get("production_month") == month
            data_root_s = str(data_root).replace("\\", "/")
            sim_root_s = str(sim_root).replace("\\", "/")
            month_isolation.append(
                {
                    "production_month": month,
                    "lot_id": lot_id,
                    "die_id": die_id,
                    "data_root": data_root_s,
                    "simulation_root": sim_root_s,
                    "evidence_origin_sample": (
                        payload["recommendations"][0].get("evidence_origin")
                        if payload["recommendations"]
                        else None
                    ),
                    "uses_only_month_data": data_root_s.rstrip("/").endswith(f"/{month}")
                    or f"/3 months data/{month}" in data_root_s,
                    "uses_only_month_sim": f"/temporal/{month}/" in sim_root_s
                    or sim_root_s.rstrip("/").endswith(f"/temporal/{month}/simulation"),
                    "legacy_simulation_forbidden": "/artifacts/simulation" not in sim_root_s
                    and not sim_root_s.endswith("/artifacts/simulation"),
                }
            )
            audit = payload.get("audit") or {}
            for rec in payload.get("recommendations") or []:
                parameter = str(rec["parameter"])
                if parameter not in SCORABLE_PARAMETERS:
                    continue
                exp = rec.get("explanation") or {}
                cur = float(rec["current_limit"])
                rec_lim = float(rec["recommended_limit"])
                delta = rec_lim - cur
                y = exp.get("selected_simulated_yield")
                if y is None and isinstance(rec.get("simulation_evidence"), dict):
                    y = rec["simulation_evidence"].get("simulated_yield")
                row = {
                    "lot_category": category,
                    "lot_id": lot_id,
                    "die_id": die_id,
                    "sequence_id": make_sequence_id(lot_id, die_id, month),
                    "production_month": month,
                    "month_label": MONTH_LABEL[month],
                    "parameter": parameter,
                    "parameter_display": DISPLAY_NAME[parameter],
                    "unit": rec.get("unit"),
                    "current_limit": cur,
                    "recommended_limit": rec_lim,
                    "recommendation_delta": delta,
                    "recommendation_delta_percent": _delta_pct(cur, rec_lim),
                    "max_eligible_simulated_yield": y,
                    "ml_score": rec.get("ml_score"),
                    "ml_rank": rec.get("ml_rank"),
                    "model_used": rec.get("model_used"),
                    "model_expected": model_for_parameter(parameter, temporal=True).value,
                    "decision": rec.get("decision"),
                    "policy_reason": exp.get("policy_reason"),
                    "yield_tie": bool(exp.get("yield_tie")),
                    "tie_breaker": exp.get("tie_breaker"),
                    "selection_text": exp.get("selection_text"),
                    "explanation_text": exp.get("text"),
                    "why_selected": _explain_from_rec(rec),
                    "safety_status": _safety_status(rec),
                    "evidence_origin": rec.get("evidence_origin"),
                    "is_primary_die": category == "NORMAL" and die_id == PRIMARY_DIE[1],
                }
                recommendation_rows.append(row)

                cand_rows = _candidate_rows_for_param(
                    lot_id=lot_id,
                    die_id=die_id,
                    month=month,
                    parameter=parameter,
                    rec=rec,
                    audit=audit,
                )
                candidate_expl_rows.extend(cand_rows)

                yf = _find_yield_first_proof(cand_rows)
                if yf is not None:
                    yield_first_proofs.append(yf)
                tb = _find_tie_break_proof(cand_rows, rec)
                if tb is not None:
                    tie_break_proofs.append(tb)

                meas = _measurement_summary(month, lot_id, die_id, parameter, root)
                same_die_rows.append(
                    {
                        **{k: row[k] for k in (
                            "lot_category",
                            "lot_id",
                            "die_id",
                            "sequence_id",
                            "production_month",
                            "parameter",
                            "parameter_display",
                            "current_limit",
                            "recommended_limit",
                            "max_eligible_simulated_yield",
                            "ml_score",
                            "ml_rank",
                            "model_used",
                            "decision",
                            "why_selected",
                        )},
                        "observed_n": meas["n"],
                        "observed_mean": meas["mean"],
                        "observed_min": meas["min"],
                        "observed_max": meas["max"],
                    }
                )

    rec_df = pd.DataFrame(recommendation_rows)
    primary = rec_df[rec_df["is_primary_die"]].copy()
    assert len(primary) == 27, f"expected 27 primary rows, got {len(primary)}"

    # Temporal changes (primary die)
    change_rows: list[dict[str, Any]] = []
    for parameter in SCORABLE_PARAMETERS:
        sub = primary[primary["parameter"] == parameter].set_index("production_month")
        jan = sub.loc["2026-01"]
        feb = sub.loc["2026-02"]
        mar = sub.loc["2026-03"]
        recs = [float(jan["recommended_limit"]), float(feb["recommended_limit"]), float(mar["recommended_limit"])]
        changed = len({round(x, 12) for x in recs}) > 1
        change_rows.append(
            {
                "parameter": parameter,
                "parameter_display": DISPLAY_NAME[parameter],
                "jan_recommendation": float(jan["recommended_limit"]),
                "feb_recommendation": float(feb["recommended_limit"]),
                "mar_recommendation": float(mar["recommended_limit"]),
                "recommendation_changed": changed,
                "jan_yield": jan["max_eligible_simulated_yield"],
                "feb_yield": feb["max_eligible_simulated_yield"],
                "mar_yield": mar["max_eligible_simulated_yield"],
                "jan_ml_rank": jan["ml_rank"],
                "feb_ml_rank": feb["ml_rank"],
                "mar_ml_rank": mar["ml_rank"],
                "jan_decision": jan["decision"],
                "feb_decision": feb["decision"],
                "mar_decision": mar["decision"],
                "current_dtl_changed": len(
                    {
                        round(float(jan["current_limit"]), 12),
                        round(float(feb["current_limit"]), 12),
                        round(float(mar["current_limit"]), 12),
                    }
                )
                > 1,
                "yield_changed": len(
                    {
                        None if pd.isna(jan["max_eligible_simulated_yield"]) else round(float(jan["max_eligible_simulated_yield"]), 12),
                        None if pd.isna(feb["max_eligible_simulated_yield"]) else round(float(feb["max_eligible_simulated_yield"]), 12),
                        None if pd.isna(mar["max_eligible_simulated_yield"]) else round(float(mar["max_eligible_simulated_yield"]), 12),
                    }
                )
                > 1,
                "model_used": jan["model_used"],
            }
        )
        # Jan→Feb and Feb→Mar transitions
        for prev_m, next_m, prev, nxt in (
            ("2026-01", "2026-02", jan, feb),
            ("2026-02", "2026-03", feb, mar),
        ):
            if abs(float(prev["recommended_limit"]) - float(nxt["recommended_limit"])) > 1e-12:
                change_rows.append(
                    {
                        "parameter": parameter,
                        "parameter_display": DISPLAY_NAME[parameter],
                        "transition": f"{prev_m}→{next_m}",
                        "previous_dtl": float(prev["recommended_limit"]),
                        "new_dtl": float(nxt["recommended_limit"]),
                        "previous_yield": prev["max_eligible_simulated_yield"],
                        "new_yield": nxt["max_eligible_simulated_yield"],
                        "previous_ml_rank": prev["ml_rank"],
                        "new_ml_rank": nxt["ml_rank"],
                        "factual_note": (
                            "The recommendation changed because the month-specific candidate "
                            "simulation/evidence and ML ranking produced a different eligible winner."
                        ),
                    }
                )

    # Model traceability from runtime
    model_rows = []
    for parameter in SCORABLE_PARAMETERS:
        used = sorted(set(primary[primary["parameter"] == parameter]["model_used"].dropna().astype(str)))
        expected = model_for_parameter(parameter, temporal=True).value
        model_rows.append(
            {
                "parameter": parameter,
                "parameter_display": DISPLAY_NAME[parameter],
                "model_expected": expected,
                "models_observed": "|".join(used),
                "routing_ok": used == [expected],
            }
        )

    # Decision summary
    def _decision_breakdown(df: pd.DataFrame) -> dict[str, Any]:
        counts = Counter(df["decision"].astype(str))
        by_month = {
            m: Counter(df[df["production_month"] == m]["decision"].astype(str))
            for m in MONTHS
        }
        by_param = {
            DISPLAY_NAME[p]: Counter(df[df["parameter"] == p]["decision"].astype(str))
            for p in SCORABLE_PARAMETERS
        }
        by_model = {}
        for mid in sorted(df["model_used"].dropna().astype(str).unique()):
            by_model[mid] = Counter(df[df["model_used"] == mid]["decision"].astype(str))
        return {
            "total": int(len(df)),
            "counts": dict(counts),
            "by_month": {m: dict(by_month[m]) for m in MONTHS},
            "by_parameter": {k: dict(v) for k, v in by_param.items()},
            "by_model_family": {k: dict(v) for k, v in by_model.items()},
        }

    primary_decisions = _decision_breakdown(primary)
    all_decisions = _decision_breakdown(rec_df)

    changed_params = [
        r["parameter_display"] for r in change_rows if r.get("recommendation_changed") is True
    ]
    stable_params = [
        r["parameter_display"]
        for r in change_rows
        if r.get("recommendation_changed") is False and "transition" not in r
    ]

    executive = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "12.9",
        "primary_die": {"lot_id": PRIMARY_DIE[0], "die_id": PRIMARY_DIE[1]},
        "n_primary_recommendations": int(len(primary)),
        "months": list(MONTHS),
        "scorable_parameters": [DISPLAY_NAME[p] for p in SCORABLE_PARAMETERS],
        "decision_summary_primary": primary_decisions,
        "decision_summary_all_dies": all_decisions,
        "parameters_recommendation_changed": changed_params,
        "parameters_recommendation_stable": stable_params,
        "model_routing_verified": all(r["routing_ok"] for r in model_rows),
        "yield_first_proof_count": len(yield_first_proofs),
        "ml_tie_break_proof_count": len(tie_break_proofs),
        "yield_first_proof_example": yield_first_proofs[0] if yield_first_proofs else None,
        "ml_tie_break_proof_example": tie_break_proofs[0] if tie_break_proofs else None,
        "what_ml_does": (
            "The GRU scores candidate DTLs and produces ML rankings. The recommendation "
            "policy then uses maximum eligible simulated yield as the primary selection "
            "criterion and ML rank as the tie-breaker."
        ),
        "what_changed_summary": (
            f"On primary die {PRIMARY_DIE[1]}, recommended DTL changed across months for "
            f"{len(changed_params)} parameter(s): {', '.join(changed_params) or 'none'}; "
            f"stable for {len(stable_params)}: {', '.join(stable_params) or 'none'}."
        ),
        "limitations": [
            "Simulated yield is not guaranteed production yield.",
            "ML score is not simulated yield.",
            "ML rank is not automatically the final recommendation.",
            "SETUP/HOLD/TEST_TIME are not currently scorable DTL parameters.",
            "IR/Thermal use CoreGRU because the Phase 12.6A tie-break analysis favored Core.",
            "Parametric parameters use the Unified GRU.",
            "Three-month data is synthetic and must not be presented as real production data.",
            "Recommendations are based on available simulation/catalog/safety evidence.",
        ],
    }

    # Write artifacts
    primary.to_csv(out_dir / "three_month_recommendations.csv", index=False)
    write_json(
        out_dir / "three_month_recommendations.json",
        {
            "primary_die": {"lot_id": PRIMARY_DIE[0], "die_id": PRIMARY_DIE[1]},
            "rows": primary.to_dict(orient="records"),
            "all_dies_rows": rec_df.to_dict(orient="records"),
        },
    )
    pd.DataFrame(candidate_expl_rows).to_csv(out_dir / "candidate_explanations.csv", index=False)
    pd.DataFrame(change_rows).to_csv(out_dir / "temporal_changes.csv", index=False)
    pd.DataFrame(same_die_rows).to_csv(out_dir / "same_die_analysis.csv", index=False)
    pd.DataFrame(model_rows).to_csv(out_dir / "model_traceability.csv", index=False)
    write_json(out_dir / "executive_summary.json", executive)
    write_json(
        out_dir / "policy_proofs.json",
        {
            "yield_first_proofs": yield_first_proofs[:20],
            "ml_tie_break_proofs": tie_break_proofs[:20],
            "month_isolation_checks": month_isolation,
        },
    )

    # Optional simple viz data (CSV only — no misleading charts required)
    viz = primary[
        [
            "parameter_display",
            "production_month",
            "current_limit",
            "recommended_limit",
            "max_eligible_simulated_yield",
            "decision",
        ]
    ].copy()
    viz.to_csv(out_dir / "viz_recommended_dtl_by_month.csv", index=False)

    summary = {
        "output_dir": str(out_dir).replace("\\", "/"),
        "n_primary": int(len(primary)),
        "n_all_die_rows": int(len(rec_df)),
        "n_candidate_rows": len(candidate_expl_rows),
        "changed_params": changed_params,
        "stable_params": stable_params,
        "yield_first_proofs": len(yield_first_proofs),
        "tie_break_proofs": len(tie_break_proofs),
        "routing_ok": all(r["routing_ok"] for r in model_rows),
        "executive": executive,
        "primary_records": primary.to_dict(orient="records"),
        "change_rows": change_rows,
        "model_rows": model_rows,
        "yield_first_example": yield_first_proofs[0] if yield_first_proofs else None,
        "tie_break_example": tie_break_proofs[0] if tie_break_proofs else None,
        "same_die_sample": same_die_rows[:12],
    }
    write_json(
        out_dir / "analysis_run_summary.json",
        _json_safe({k: v for k, v in summary.items() if k != "executive"}),
    )
    return summary


def main() -> None:
    summary = run_phase12_9_analysis()
    print("output_dir", summary["output_dir"])
    print("primary", summary["n_primary"], "all", summary["n_all_die_rows"])
    print("changed", summary["changed_params"])
    print("stable", summary["stable_params"])
    print("yield_first_proofs", summary["yield_first_proofs"])
    print("tie_break_proofs", summary["tie_break_proofs"])
    print("routing_ok", summary["routing_ok"])


if __name__ == "__main__":
    main()
