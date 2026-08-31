"""Phase 8 recommendation orchestration pipeline."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dtl_agent.config.paths import default_project_root
from dtl_agent.recommendation.catalog import CandidateCatalog
from dtl_agent.recommendation.config import RecommendationConfig, load_recommendation_config
from dtl_agent.recommendation.context import load_recommendation_context
from dtl_agent.recommendation.evidence import SimulationEvidenceLookup
from dtl_agent.recommendation.explanation import (
    build_audit_record,
    build_explanation,
    compute_evidence_level,
)
from dtl_agent.recommendation.inference import (
    CoreGRUInferencer,
    ModelBundle,
    ParametricMLPInferencer,
)
from dtl_agent.recommendation.policy import EvaluatedCandidate, apply_recommendation_policy
from dtl_agent.recommendation.ranking import rank_candidates, select_top_n_plus_current
from dtl_agent.recommendation.routing import model_used_label
from dtl_agent.recommendation.safety import evaluate_safety
from dtl_agent.recommendation.schemas import (
    CORE_PARAMETERS,
    PARAMETRIC_PARAMETERS,
    Decision,
    DTLRecommendation,
    EvidenceLevel,
    LotRecommendationResult,
    GateStatus,
)
from dtl_agent.recommendation.temporal_config import (
    assert_month_simulation_isolated,
    load_temporal_recommendation_context,
    temporal_recommendation_config,
)
from dtl_agent.recommendation.temporal_inference import TemporalHybridBundle
from dtl_agent.data.temporal.paths import TemporalPathError, validate_production_month


def _default_parameters(core_available: bool, parametric_available: bool) -> list[str]:
    params: list[str] = []
    if core_available:
        params.extend(sorted(CORE_PARAMETERS))
    if parametric_available:
        params.extend(sorted(PARAMETRIC_PARAMETERS))
    return params


def _sim_config_version(root: Path, domain: str, production_month: str | None = None) -> str | None:
    if production_month is not None:
        path = (
            root
            / "artifacts"
            / "temporal"
            / production_month
            / "simulation"
            / domain
            / "simulation_config.json"
        )
    else:
        path = root / "artifacts" / "simulation" / domain / "simulation_config.json"
    if not path.is_file():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("version"))
    except Exception:  # noqa: BLE001
        return None


def recommend(
    *,
    lot_id: str,
    die_id: str,
    parameters: list[str] | None = None,
    config: RecommendationConfig | None = None,
    policy_config_path: Path | None = None,
    project_root: Path | None = None,
    extra_paths: list[str] | None = None,
    model_bundle: ModelBundle | None = None,
    production_month: str | None = None,
    temporal_bundle: TemporalHybridBundle | None = None,
) -> LotRecommendationResult:
    """Run the locked Phase 8 decision pipeline for one lot/die.

    ``production_month=None`` — legacy path (unchanged ModelBundle / artifacts/simulation).
    ``production_month`` in {2026-01,2026-02,2026-03} — hybrid temporal Core+Unified path.
    """
    root = project_root or default_project_root()
    temporal_mode = production_month is not None
    month: str | None = None
    if temporal_mode:
        try:
            month = validate_production_month(str(production_month))
        except TemporalPathError as exc:
            raise ValueError(str(exc)) from exc
        assert_month_simulation_isolated(month, root)
        base_cfg = config or load_recommendation_config(policy_config_path)
        cfg = temporal_recommendation_config(month, base=base_cfg)
        ctx = load_temporal_recommendation_context(
            production_month=month,
            lot_id=lot_id,
            die_id=die_id,
            project_root=root,
            extra_paths=extra_paths,
        )
        hybrid = temporal_bundle or TemporalHybridBundle(root)
    else:
        cfg = config or load_recommendation_config(policy_config_path)
        ctx = load_recommendation_context(
            lot_id=lot_id,
            die_id=die_id,
            project_root=root,
            extra_paths=extra_paths,
        )
        hybrid = None

    if cfg.joint_enabled:
        # Locked decision: joint must remain disabled in the decision path.
        cfg = RecommendationConfig.from_dict({**cfg.to_dict(), "joint_enabled": False})

    request_id = str(uuid.uuid4())

    if temporal_mode:
        from dtl_agent.recommendation.resource_cache import (
            get_candidate_catalog,
            get_evidence_lookup,
        )

        catalog = get_candidate_catalog(root, cfg)
        evidence_lookup = get_evidence_lookup(root, cfg)
    else:
        catalog = CandidateCatalog(root, cfg)
        evidence_lookup = SimulationEvidenceLookup(root, cfg)
    # Legacy ModelBundle is only loaded when non-temporal inferencers call ensure_loaded().
    bundle = model_bundle or ModelBundle(root, config or load_recommendation_config(policy_config_path))

    params = parameters or _default_parameters(ctx.core_available, ctx.parametric_available)
    recommendations: list[DTLRecommendation] = []
    audit_candidates: list[dict[str, Any]] = []
    audit_preds: list[dict[str, Any]] = []
    audit_sims: list[dict[str, Any]] = []
    audit_safety: list[dict[str, Any]] = []
    audit_policy: list[str] = []
    audit_finals: list[dict[str, Any]] = []
    sim_versions: list[str] = []

    ts = datetime.now(timezone.utc).isoformat()

    def _stamp(rec: DTLRecommendation, *, model_used: str | None = None) -> DTLRecommendation:
        rec.production_month = month
        if model_used is not None:
            rec.model_used = model_used
        elif rec.model_used is None and rec.parameter != "*":
            rec.model_used = model_used_label(rec.parameter, temporal=temporal_mode)
        return rec

    if ctx.forbidden_detected:
        rec = DTLRecommendation(
            request_id=request_id,
            lot_id=lot_id,
            die_id=die_id,
            parameter="*",
            test_id="",
            unit="",
            direction="",
            current_limit=float("nan"),
            recommended_limit=float("nan"),
            decision=Decision.REJECT,
            ml_score=None,
            ml_rank=None,
            n_candidates=0,
            model_id=None,
            source_status="",
            simulation_evidence={},
            safety_result={"status": GateStatus.HARD_FAIL.value, "checks": []},
            evidence_level=EvidenceLevel.INSUFFICIENT_EVIDENCE,
            explanation={
                "text": "REJECT: forbidden/evaluation data detected in request path",
                "policy_reason": "forbidden_data",
            },
            model_version=ctx.package_version,
            checkpoint_id=None,
            dataset_version=ctx.dataset_version_core,
            feature_registry_hash=ctx.feature_registry_hash,
            simulation_config_version=None,
            policy_config_version=cfg.policy_config_version,
            timestamp=ts,
            core_available=ctx.core_available,
            parametric_available=ctx.parametric_available,
            cross_domain_available=ctx.cross_domain_available,
            evidence_origin=cfg.evidence_origin_label,
            production_month=month,
            model_used=None,
        )
        recommendations.append(rec)
        audit = build_audit_record(
            request_id=request_id,
            ctx=ctx,
            config=cfg,
            parameters_requested=params,
            candidate_set=[],
            ml_predictions=[],
            simulation_rows=[],
            safety_traces=[],
            policy_traces=["REJECT: forbidden_data"],
            final_decisions=[rec.to_dict()],
            checkpoint_ids={},
            simulation_config_version=None,
        )
        return LotRecommendationResult(
            request_id=request_id,
            lot_id=lot_id,
            die_id=die_id,
            recommendations=recommendations,
            audit=audit,
            core_available=ctx.core_available,
            parametric_available=ctx.parametric_available,
            cross_domain_available=ctx.cross_domain_available,
            production_month=month,
        )

    core_inf = CoreGRUInferencer(bundle) if not temporal_mode else None
    param_inf = ParametricMLPInferencer(bundle) if not temporal_mode else None

    for parameter in params:
        domain = catalog.domain_for(parameter)

        if catalog.is_unsupported(parameter) or domain is None:
            cur = catalog.current_row(parameter)
            rec = DTLRecommendation(
                request_id=request_id,
                lot_id=lot_id,
                die_id=die_id,
                parameter=parameter,
                test_id=cur.test_id if cur else "",
                unit=cur.unit if cur else "",
                direction=cur.direction if cur else "",
                current_limit=cur.current_limit if cur else float("nan"),
                recommended_limit=cur.current_limit if cur else float("nan"),
                decision=Decision.REJECT,
                ml_score=None,
                ml_rank=None,
                n_candidates=0,
                model_id=None,
                source_status=cur.source_status if cur else "",
                simulation_evidence={},
                safety_result={
                    "status": GateStatus.HARD_FAIL.value,
                    "checks": [
                        {
                            "name": "supported_parameter",
                            "passed": False,
                            "layer": 1,
                            "message": "unsupported or unknown parameter",
                            "severity": "hard",
                        }
                    ],
                },
                evidence_level=EvidenceLevel.INSUFFICIENT_EVIDENCE,
                explanation={
                    "text": f"REJECT: unsupported parameter {parameter}",
                    "policy_reason": "unsupported_parameter",
                },
                model_version=ctx.package_version,
                checkpoint_id=None,
                dataset_version=ctx.dataset_version_core,
                feature_registry_hash=ctx.feature_registry_hash,
                simulation_config_version=None,
                policy_config_version=cfg.policy_config_version,
                timestamp=ts,
                core_available=ctx.core_available,
                parametric_available=ctx.parametric_available,
                cross_domain_available=ctx.cross_domain_available,
                evidence_origin=cfg.evidence_origin_label,
                production_month=month,
                model_used=None,
            )
            recommendations.append(rec)
            audit_finals.append(rec.to_dict())
            audit_policy.append(f"{parameter}:REJECT:unsupported")
            continue

        # Routing
        if domain == "core" and not ctx.core_available:
            cur = catalog.current_row(parameter)
            rec = _review_or_keep(
                request_id=request_id,
                ctx=ctx,
                cfg=cfg,
                parameter=parameter,
                cur_limit=cur.current_limit if cur else float("nan"),
                test_id=cur.test_id if cur else "",
                unit=cur.unit if cur else "",
                direction=cur.direction if cur else "",
                source_status=cur.source_status if cur else "",
                decision=Decision.REVIEW_REQUIRED,
                reason="core_unavailable_no_fabricated_sequence",
                ts=ts,
                production_month=month,
                model_used=model_used_label(parameter, temporal=temporal_mode),
            )
            recommendations.append(rec)
            audit_finals.append(rec.to_dict())
            audit_policy.append(f"{parameter}:REVIEW_REQUIRED:core_unavailable")
            continue

        if domain == "parametric" and not ctx.parametric_available:
            cur = catalog.current_row(parameter)
            rec = _review_or_keep(
                request_id=request_id,
                ctx=ctx,
                cfg=cfg,
                parameter=parameter,
                cur_limit=cur.current_limit if cur else float("nan"),
                test_id=cur.test_id if cur else "",
                unit=cur.unit if cur else "",
                direction=cur.direction if cur else "",
                source_status=cur.source_status if cur else "SYNTHETIC_ASSUMED",
                decision=Decision.REVIEW_REQUIRED,
                reason="parametric_unavailable",
                ts=ts,
                production_month=month,
                model_used=model_used_label(parameter, temporal=temporal_mode),
            )
            recommendations.append(rec)
            audit_finals.append(rec.to_dict())
            continue

        # Inference
        model_used = model_used_label(parameter, temporal=temporal_mode)
        if temporal_mode:
            assert hybrid is not None and month is not None
            month_data = getattr(ctx, "month_data", None)
            if month_data is None:
                scored, err = None, "temporal_month_data_unavailable"
            else:
                scored, err, model_used = hybrid.score_parameter(
                    production_month=month,
                    lot_id=lot_id,
                    die_id=die_id,
                    parameter=parameter,
                    month_data=month_data,
                )
            ckpt = (
                hybrid.core_checkpoint_id
                if model_used == "core_gru_temporal_v1"
                else hybrid.uni_checkpoint_id
            )
            conditions_present = None
            if scored is not None and "conditions_present" in scored.columns and len(scored):
                conditions_present = list(scored.iloc[0]["conditions_present"])
        elif domain == "core":
            assert core_inf is not None
            scored, err = core_inf.score_lot_die_parameter(
                lot_id=lot_id, die_id=die_id, parameter=parameter
            )
            ckpt = bundle.core_checkpoint_id
            conditions_present = None
        else:
            assert param_inf is not None
            scored, err = param_inf.score_lot_die_parameter(
                lot_id=lot_id, die_id=die_id, parameter=parameter
            )
            ckpt = bundle.param_checkpoint_id
            conditions_present = None
            if scored is not None and "conditions_present" in scored.columns and len(scored):
                conditions_present = list(scored.iloc[0]["conditions_present"])

        if err is not None or scored is None or scored.empty:
            cur = catalog.current_row(parameter)
            decision = Decision.REVIEW_REQUIRED
            reason = err or "inference_failed"
            rec = _review_or_keep(
                request_id=request_id,
                ctx=ctx,
                cfg=cfg,
                parameter=parameter,
                cur_limit=cur.current_limit if cur else float("nan"),
                test_id=cur.test_id if cur else "",
                unit=cur.unit if cur else "",
                direction=cur.direction if cur else "",
                source_status=cur.source_status if cur else "",
                decision=decision,
                reason=reason,
                ts=ts,
                checkpoint_id=ckpt,
                production_month=month,
                model_used=model_used,
            )
            recommendations.append(rec)
            audit_finals.append(rec.to_dict())
            audit_policy.append(f"{parameter}:{decision.value}:{reason}")
            continue

        ranked = rank_candidates(scored, lot_id=lot_id, die_id=die_id, catalog=catalog)
        for c in ranked:
            audit_candidates.append(c.to_dict())
            audit_preds.append(
                {
                    "parameter": c.parameter,
                    "candidate_limit": c.candidate_limit,
                    "ml_score": c.ml_score,
                    "ml_rank": c.ml_rank,
                    "model_id": c.model_id,
                    "model_used": model_used,
                    "production_month": month,
                }
            )

        selected_for_gate = select_top_n_plus_current(ranked, cfg)
        evaluated: list[EvaluatedCandidate] = []
        for cand in selected_for_gate:
            ev = evidence_lookup.lookup(
                domain=domain, parameter=parameter, candidate_limit=cand.candidate_limit
            )
            audit_sims.append(ev.to_dict())
            safety = evaluate_safety(
                candidate=cand,
                evidence=ev,
                catalog=catalog,
                config=cfg,
                domain=domain,
                conditions_present=conditions_present,
                context_complete=not ctx.errors,
                model_available=True,
            )
            audit_safety.append({"parameter": parameter, **safety.to_dict()})
            evaluated.append(EvaluatedCandidate(candidate=cand, evidence=ev, safety=safety))

        # Hard reject if any evaluated candidate is out-of-catalog HARD_FAIL and was requested
        # For normal catalog candidates this should not fire; out-of-range is REJECT at param level
        hard = any(e.safety.status == GateStatus.HARD_FAIL for e in evaluated)
        # If CURRENT itself hard-fails catalog, still KEEP if soft elsewhere — only REJECT when
        # the parameter/catalog path is invalid. Catalog-valid scored rows should PASS layer1.
        hard_reject = hard and all(
            any(ch.name == "catalog_membership" and not ch.passed for ch in e.safety.checks)
            for e in evaluated
        )

        current_limit = float(ranked[0].current_limit)
        policy = apply_recommendation_policy(
            evaluated=evaluated,
            current_limit=current_limit,
            insufficient_evidence=False,
            hard_reject=hard_reject,
        )
        audit_policy.extend([f"{parameter}:{t}" for t in policy.policy_trace])

        selected = policy.selected
        # Attach evidence/safety for selected (or CURRENT)
        sel_eval = None
        if selected is not None:
            sel_eval = next(
                (e for e in evaluated if abs(e.candidate.candidate_limit - selected.candidate_limit) < 1e-12),
                None,
            )
        if sel_eval is None and evaluated:
            sel_eval = next(
                (
                    e
                    for e in evaluated
                    if e.candidate.tighten_or_loosen == "CURRENT"
                    or abs(e.candidate.delta_absolute) < 1e-12
                ),
                evaluated[0],
            )

        recommended_limit = (
            selected.candidate_limit
            if selected is not None and policy.decision == Decision.RECOMMEND
            else current_limit
        )
        if policy.decision == Decision.KEEP_CURRENT:
            recommended_limit = current_limit

        evidence = sel_eval.evidence if sel_eval else None
        safety = sel_eval.safety if sel_eval else None
        source_status = selected.source_status if selected else ranked[0].source_status
        evidence_level = compute_evidence_level(
            config=cfg,
            source_status=source_status,
            evidence=evidence,
            decision=policy.decision,
        )
        explanation = build_explanation(
            decision=policy.decision,
            current_limit=current_limit,
            recommended_limit=recommended_limit,
            selected=selected,
            evidence=evidence,
            safety=safety,
            policy=policy,
            config=cfg,
        )
        sim_ver = _sim_config_version(root, domain, month)
        if sim_ver:
            sim_versions.append(sim_ver)

        rec = DTLRecommendation(
            request_id=request_id,
            lot_id=lot_id,
            die_id=die_id,
            parameter=parameter,
            test_id=ranked[0].test_id,
            unit=ranked[0].unit,
            direction=ranked[0].direction,
            current_limit=current_limit,
            recommended_limit=recommended_limit,
            decision=policy.decision,
            ml_score=selected.ml_score if selected else None,
            ml_rank=selected.ml_rank if selected else None,
            n_candidates=len(ranked),
            model_id=selected.model_id if selected else model_used,
            source_status=source_status,
            simulation_evidence=evidence.to_dict() if evidence else {},
            safety_result=safety.to_dict() if safety else {},
            evidence_level=evidence_level,
            explanation=explanation,
            model_version=ctx.package_version,
            checkpoint_id=ckpt,
            dataset_version=ctx.dataset_version_core
            if domain == "core"
            else ctx.dataset_version_parametric,
            feature_registry_hash=ctx.feature_registry_hash,
            simulation_config_version=sim_ver,
            policy_config_version=cfg.policy_config_version,
            timestamp=ts,
            core_available=ctx.core_available,
            parametric_available=ctx.parametric_available,
            cross_domain_available=ctx.cross_domain_available,
            evidence_origin=cfg.evidence_origin_label,
            production_month=month,
            model_used=model_used,
        )
        recommendations.append(rec)
        audit_finals.append(rec.to_dict())

    audit = build_audit_record(
        request_id=request_id,
        ctx=ctx,
        config=cfg,
        parameters_requested=params,
        candidate_set=audit_candidates,
        ml_predictions=audit_preds,
        simulation_rows=audit_sims,
        safety_traces=audit_safety,
        policy_traces=audit_policy,
        final_decisions=audit_finals,
        checkpoint_ids={
            "core_gru": bundle.core_checkpoint_id,
            "parametric_mlp": bundle.param_checkpoint_id,
        },
        simulation_config_version=",".join(sorted(set(sim_versions))) if sim_versions else None,
    )
    return LotRecommendationResult(
        request_id=request_id,
        lot_id=lot_id,
        die_id=die_id,
        recommendations=recommendations,
        audit=audit,
        core_available=ctx.core_available,
        parametric_available=ctx.parametric_available,
        cross_domain_available=ctx.cross_domain_available,
        production_month=month,
    )


def _review_or_keep(
    *,
    request_id: str,
    ctx,
    cfg: RecommendationConfig,
    parameter: str,
    cur_limit: float,
    test_id: str,
    unit: str,
    direction: str,
    source_status: str,
    decision: Decision,
    reason: str,
    ts: str,
    checkpoint_id: str | None = None,
    production_month: str | None = None,
    model_used: str | None = None,
) -> DTLRecommendation:
    return DTLRecommendation(
        request_id=request_id,
        lot_id=ctx.lot_id,
        die_id=ctx.die_id,
        parameter=parameter,
        test_id=test_id,
        unit=unit,
        direction=direction,
        current_limit=cur_limit,
        recommended_limit=cur_limit,
        decision=decision,
        ml_score=None,
        ml_rank=None,
        n_candidates=0,
        model_id=model_used,
        source_status=source_status,
        simulation_evidence={},
        safety_result={},
        evidence_level=EvidenceLevel.INSUFFICIENT_EVIDENCE,
        explanation={"text": f"{decision.value}: {reason}", "policy_reason": reason},
        model_version=ctx.package_version,
        checkpoint_id=checkpoint_id,
        dataset_version=ctx.dataset_version_core,
        feature_registry_hash=ctx.feature_registry_hash,
        simulation_config_version=None,
        policy_config_version=cfg.policy_config_version,
        timestamp=ts,
        core_available=ctx.core_available,
        parametric_available=ctx.parametric_available,
        cross_domain_available=ctx.cross_domain_available,
        evidence_origin=cfg.evidence_origin_label,
        production_month=production_month,
        model_used=model_used,
    )
