"""Phase 12.9 / Phase 13 three-month analysis API + upload analysis sessions."""

from __future__ import annotations

from fastapi import APIRouter, File, Query, UploadFile, status

from dtl_agent.analysis.cost_savings import (
    CostSavingsAssumptions,
    clear_cost_savings_caches,
    estimate_cost_savings,
)
from dtl_agent.api.analysis_loader import (
    ALLOWED_MONTHS,
    AnalysisArtifactError,
    SCORABLE_DISPLAY,
    clear_analysis_cache,
    load_three_month_bundle,
)
from dtl_agent.api.analysis_session import get_job_status
from dtl_agent.api.dependencies import AnalysisProjectRootDep, ProjectRootDep, ReadyDep
from dtl_agent.api.die_level_service import (
    _lot_category_from_catalog,
    cache_coverage,
    clear_die_level_process_caches,
    get_die_history,
    get_die_recommendation,
    get_die_recommendation_rows_for_cost_savings,
    load_identity_catalog,
    lot_die_browse,
    observed_summary,
    resolve_parameter,
)
from dtl_agent.api.errors import ArtifactUnavailableError, ServiceError, VALIDATION_ERROR
from dtl_agent.api.upload_analysis import create_upload_analysis_session, start_upload_analysis_job
from dtl_agent.api.upload_recommendation import UploadRecommendationError, read_upload_file
from dtl_agent.data.temporal.paths import TemporalPathError

router = APIRouter(tags=["analysis"])


def _validation(msg: str) -> ServiceError:
    return ServiceError(
        code=VALIDATION_ERROR,
        message=msg,
        http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


@router.post(
    "/analysis/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Jan/Feb/Mar measurements and start an async analysis session job",
)
async def post_analysis_upload(
    _: ReadyDep,
    project_root: ProjectRootDep,
    january: UploadFile = File(..., description="January 2026 actual_die CSV or ZIP"),
    february: UploadFile = File(..., description="February 2026 actual_die CSV or ZIP"),
    march: UploadFile = File(..., description="March 2026 actual_die CSV or ZIP"),
) -> dict:
    """Validate upload files (<0.5s) and start background analysis job.

    Returns HTTP 202 Accepted with analysis_session_id and status='queued'.
    Poll /api/v1/analysis/upload/status/{analysis_session_id} for progress.
    """
    try:
        files = {
            "january": (january.filename or "january.csv", read_upload_file(january.file)),
            "february": (february.filename or "february.csv", read_upload_file(february.file)),
            "march": (march.filename or "march.csv", read_upload_file(march.file)),
        }
        res = start_upload_analysis_job(
            files=files,
            source_root=project_root,
        )
        # Drop process caches so session-scoped roots are not mixed with static hits
        clear_analysis_cache()
        clear_cost_savings_caches()
        clear_die_level_process_caches()
        return res
    except UploadRecommendationError as exc:
        raise _validation(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(
            code="UPLOAD_ANALYSIS_ERROR",
            message=f"Upload analysis failed: {type(exc).__name__}: {exc}",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc


@router.get(
    "/analysis/upload/status/{analysis_session_id}",
    status_code=status.HTTP_200_OK,
    summary="Poll status of an uploaded analysis session background job",
)
def get_analysis_upload_status(analysis_session_id: str) -> dict:
    """Return status (queued | processing | completed | failed), stage, and progress_pct."""
    return get_job_status(analysis_session_id)


@router.get(
    "/analysis/three-month",
    status_code=status.HTTP_200_OK,
    summary="Phase 12.9 three-month recommendation analysis bundle (read-only)",
)
def get_three_month_analysis(project_root: AnalysisProjectRootDep) -> dict:
    """Serve analysis artifacts for static root or an upload analysis session."""
    try:
        bundle = load_three_month_bundle(str(project_root.resolve()))
    except AnalysisArtifactError as exc:
        raise ArtifactUnavailableError() from exc
    try:
        identities = load_identity_catalog(project_root)
        bundle["die_level_identities"] = {
            "months": identities["months"],
            "categories": identities["categories"],
            "lots_by_category": identities["lots_by_category"],
            "dies_by_lot": identities["dies_by_lot"],
            "counts": identities["counts"],
            "stable_across_months": identities["stable_across_months"],
            "note": identities["note"],
            "cache_coverage": cache_coverage(project_root),
        }
    except Exception:  # noqa: BLE001
        bundle["die_level_identities"] = None
    return bundle


@router.get(
    "/analysis/three-month/identities",
    status_code=status.HTTP_200_OK,
    summary="Lot/die identity catalog for temporal months (metadata only)",
)
def get_three_month_identities(project_root: AnalysisProjectRootDep) -> dict:
    try:
        data = load_identity_catalog(project_root)
        data["cache_coverage"] = cache_coverage(project_root)
        return data
    except Exception as exc:  # noqa: BLE001
        raise ArtifactUnavailableError() from exc


@router.get(
    "/analysis/three-month/dies",
    status_code=status.HTTP_200_OK,
    summary="Die-level recommendation via existing recommend() engine (cached)",
)
def get_three_month_die_recommendation(
    project_root: AnalysisProjectRootDep,
    production_month: str = Query(...),
    lot_id: str = Query(...),
    die_id: str = Query(...),
    parameter: str = Query(..., description="IR_DROP_MV or ir_drop"),
    force_refresh: bool = Query(False),
) -> dict:
    try:
        if production_month.strip() not in ALLOWED_MONTHS:
            raise _validation(
                f"Invalid production_month={production_month!r}; allowed: {', '.join(ALLOWED_MONTHS)}"
            )
        resolve_parameter(parameter)
        payload = get_die_recommendation(
            project_root,
            production_month=production_month.strip(),
            lot_id=lot_id.strip(),
            die_id=die_id.strip(),
            parameter=parameter.strip(),
            force_refresh=force_refresh,
        )
        rec = payload["recommendation"]
        if (
            rec["production_month"] != production_month.strip()
            or rec["lot_id"] != lot_id.strip()
            or rec["die_id"] != die_id.strip()
        ):
            raise RuntimeError("Die-level recommendation identity mismatch")
        # Provenance for upload sessions
        marker = project_root / "artifacts" / "temporal" / "shared" / "upload_session.json"
        if marker.is_file():
            payload["used_uploaded_measurements"] = True
            payload["used_static_three_month_measurements"] = False
        return payload
    except TemporalPathError as exc:
        raise _validation(str(exc)) from exc
    except ValueError as exc:
        raise _validation(str(exc)) from exc
    except KeyError as exc:
        raise ServiceError(
            code=VALIDATION_ERROR,
            message=str(exc),
            http_status=status.HTTP_404_NOT_FOUND,
        ) from exc
    except ServiceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(
            code="RECOMMENDATION_ERROR",
            message=f"Recommendation unavailable: {type(exc).__name__}: {exc}",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc


@router.get(
    "/analysis/three-month/die-history",
    status_code=status.HTTP_200_OK,
    summary="Same-die recommendation history across Jan/Feb/Mar",
)
def get_three_month_die_history(
    project_root: AnalysisProjectRootDep,
    lot_id: str = Query(...),
    die_id: str = Query(...),
    parameter: str = Query(...),
) -> dict:
    try:
        resolve_parameter(parameter)
        return get_die_history(
            project_root,
            lot_id=lot_id.strip(),
            die_id=die_id.strip(),
            parameter=parameter.strip(),
        )
    except (ValueError, TemporalPathError) as exc:
        raise _validation(str(exc)) from exc
    except KeyError as exc:
        raise ServiceError(
            code=VALIDATION_ERROR,
            message=str(exc),
            http_status=status.HTTP_404_NOT_FOUND,
        ) from exc


@router.get(
    "/analysis/three-month/observed",
    status_code=status.HTTP_200_OK,
    summary="Observed measurement means for a die across months (context only)",
)
def get_three_month_observed(
    project_root: AnalysisProjectRootDep,
    lot_id: str = Query(...),
    die_id: str = Query(...),
) -> dict:
    try:
        return observed_summary(project_root, lot_id=lot_id.strip(), die_id=die_id.strip())
    except Exception as exc:  # noqa: BLE001
        raise ArtifactUnavailableError() from exc


@router.get(
    "/analysis/three-month/lot-dies",
    status_code=status.HTTP_200_OK,
    summary="Browse all dies in a lot for one month/parameter (engine-backed, cached)",
)
def get_three_month_lot_dies(
    project_root: AnalysisProjectRootDep,
    production_month: str = Query(...),
    lot_id: str = Query(...),
    parameter: str = Query(...),
    max_dies: int | None = Query(None, ge=1, le=50),
) -> dict:
    try:
        if production_month.strip() not in ALLOWED_MONTHS:
            raise _validation(
                f"Invalid production_month={production_month!r}; allowed: {', '.join(ALLOWED_MONTHS)}"
            )
        resolve_parameter(parameter)
        return lot_die_browse(
            project_root,
            production_month=production_month.strip(),
            lot_id=lot_id.strip(),
            parameter=parameter.strip(),
            max_dies=max_dies,
        )
    except (ValueError, TemporalPathError) as exc:
        raise _validation(str(exc)) from exc
    except KeyError as exc:
        raise ServiceError(
            code=VALIDATION_ERROR,
            message=str(exc),
            http_status=status.HTTP_404_NOT_FOUND,
        ) from exc


@router.get(
    "/analysis/three-month/recommendation",
    status_code=status.HTTP_200_OK,
    summary="Lookup recommendation: Phase 12.9 artifact or live die-level engine",
)
def get_three_month_recommendation(
    project_root: AnalysisProjectRootDep,
    production_month: str = Query(..., description="2026-01 | 2026-02 | 2026-03"),
    parameter_display: str = Query(..., description="e.g. IR_DROP_MV"),
    lot_id: str = Query("DTL_NORM_001"),
    die_id: str = Query("DTL_NORM_001_D001"),
) -> dict:
    month = production_month.strip()
    if month not in ALLOWED_MONTHS:
        raise _validation(
            f"Invalid production_month={production_month!r}; "
            f"allowed: {', '.join(ALLOWED_MONTHS)} (no silent fallback)"
        )
    param = parameter_display.strip()
    if param not in SCORABLE_DISPLAY:
        raise _validation(
            f"Unsupported or unknown parameter_display={parameter_display!r}. "
            "SETUP/HOLD/TEST_TIME are currently non-scorable."
        )

    try:
        bundle = load_three_month_bundle(str(project_root.resolve()))
    except AnalysisArtifactError as exc:
        raise ArtifactUnavailableError() from exc

    rows = [
        r
        for r in bundle.get("all_recommendations") or []
        if str(r.get("production_month")) == month
        and str(r.get("parameter_display")) == param
        and str(r.get("lot_id")) == lot_id
        and str(r.get("die_id")) == die_id
    ]
    if not rows:
        try:
            live = get_die_recommendation(
                project_root,
                production_month=month,
                lot_id=lot_id,
                die_id=die_id,
                parameter=param,
            )
            hist = get_die_history(
                project_root, lot_id=lot_id, die_id=die_id, parameter=param
            )
            return {
                "recommendation": live["recommendation"],
                "candidates": live["candidates"],
                "three_month_history": hist["history"],
                "disclaimer": bundle["disclaimer"],
                "source": "recommend_engine_cached",
            }
        except KeyError as exc:
            raise ServiceError(
                code=VALIDATION_ERROR,
                message="No recommendation for the requested identity.",
                http_status=status.HTTP_404_NOT_FOUND,
            ) from exc

    rec = rows[0]
    candidates = [
        c
        for c in bundle.get("candidate_explanations") or []
        if str(c.get("production_month")) == month
        and str(c.get("parameter_display")) == param
        and str(c.get("lot_id")) == lot_id
        and str(c.get("die_id")) == die_id
    ]
    history = [
        r
        for r in bundle.get("all_recommendations") or []
        if str(r.get("parameter_display")) == param
        and str(r.get("lot_id")) == lot_id
        and str(r.get("die_id")) == die_id
    ]
    return {
        "recommendation": rec,
        "candidates": candidates,
        "three_month_history": history,
        "disclaimer": bundle["disclaimer"],
        "source": bundle["source"],
    }


@router.get(
    "/analysis/cost-savings",
    status_code=status.HTTP_200_OK,
    summary=(
        "Predicted DTL parametric test-time cost savings "
        "(read-only counterfactual estimate; selected die or session level)"
    ),
)
def get_cost_savings(
    project_root: AnalysisProjectRootDep,
    production_month: str | None = Query(
        None,
        description="Filter cost savings by production month (2026-01, 2026-02, 2026-03)",
    ),
    lot_id: str | None = Query(
        None,
        description="Filter cost savings by lot_id",
    ),
    die_id: str | None = Query(
        None,
        description="Filter cost savings by die_id",
    ),
    condition_duration_s: float = Query(
        0.05,
        ge=0.0,
        description="Configured ASSUMPTION: seconds per parametric condition (not measured)",
    ),
    skip_threshold: float = Query(
        0.10,
        ge=0.0,
        description="Configured ASSUMPTION: min margin vs recommended_limit to skip remaining conditions",
    ),
    tester_cost_per_hour: float = Query(
        25.0,
        ge=0.0,
        description="Configured ASSUMPTION: tester operating cost per hour (not from dataset)",
    ),
    include_per_device: bool = Query(
        True,
        description="Include per die×parameter×month detail rows",
    ),
) -> dict:
    """Read-only estimator over selected die or full session recommendations + COND_RT_NOM."""
    try:
        def _num(val: Any, default: float) -> float:
            if hasattr(val, "default"):
                return float(val.default)
            if val is None:
                return float(default)
            return float(val)

        assumptions = CostSavingsAssumptions(
            condition_duration_s=_num(condition_duration_s, 0.05),
            skip_threshold=_num(skip_threshold, 0.10),
            tester_cost_per_hour=_num(tester_cost_per_hour, 25.0),
        )

        def _str(val: Any) -> str | None:
            if hasattr(val, "default"):
                val = val.default
            if val is None or not str(val).strip() or str(val).startswith("annotation="):
                return None
            return str(val).strip()

        lot_s = _str(lot_id)
        die_s = _str(die_id)
        month_s = _str(production_month)

        if lot_s and die_s:
            recommendation_rows = get_die_recommendation_rows_for_cost_savings(
                project_root,
                lot_id=lot_s,
                die_id=die_s,
                production_month=month_s,
            )
        else:
            recommendation_rows = None

        payload = estimate_cost_savings(
            project_root,
            assumptions=assumptions,
            include_per_device=include_per_device,
            recommendation_rows=recommendation_rows,
        )

        marker = project_root / "artifacts" / "temporal" / "shared" / "upload_session.json"
        if marker.is_file():
            payload["used_uploaded_measurements"] = True
            payload["used_static_three_month_measurements"] = False
            payload["data_provenance"] = "Analysis generated from uploaded test data"
        else:
            payload["used_uploaded_measurements"] = False
            payload["used_static_three_month_measurements"] = True

        if lot_s and die_s:
            cat = _lot_category_from_catalog(project_root, lot_s, die_s)
            payload["label"] = "Predicted DTL Test-Time Cost Saving — Selected Die"
            payload["selected_scope"] = {
                "category": cat or "NORMAL",
                "lot_id": lot_s,
                "die_id": die_s,
                "production_month": month_s or "three-month",
            }

        return payload
    except (FileNotFoundError, KeyError) as exc:
        if isinstance(exc, KeyError):
            raise _validation(str(exc)) from exc
        raise ArtifactUnavailableError() from exc
    except ValueError as exc:
        raise _validation(str(exc)) from exc


@router.post(
    "/analysis/three-month/reload",
    status_code=status.HTTP_200_OK,
    summary="Clear in-process analysis artifact cache (admin/dev)",
    include_in_schema=False,
)
def reload_analysis_cache() -> dict:
    from dtl_agent.data.temporal.month_cache import clear_temporal_month_cache
    from dtl_agent.recommendation.resource_cache import clear_recommendation_resource_cache

    clear_analysis_cache()
    clear_cost_savings_caches()
    clear_die_level_process_caches()
    clear_temporal_month_cache()
    clear_recommendation_resource_cache()
    return {"ok": True}
