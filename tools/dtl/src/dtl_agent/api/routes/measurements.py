"""Read-only measurement, distribution, and condition routes (Phase 10.11)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from dtl_agent.api.measurement_data import (
    MeasurementLookupError,
    get_conditions,
    get_distribution,
    get_selected_measurement,
)

router = APIRouter(tags=["measurements"])


def _root(request: Request) -> str:
    return str(request.app.state.project_root.resolve())


def _raise_lookup(exc: MeasurementLookupError) -> None:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=exc.message,
    ) from exc


@router.get(
    "/dies/{die_id}/measurements",
    summary="Selected-die observed measurement (read-only, SYNTHETIC)",
)
def get_die_measurements(
    die_id: str,
    request: Request,
    lot_id: str = Query(..., description="Lot that owns the die"),
    parameter: str = Query(..., description="Measurement parameter"),
    condition_id: str | None = Query(
        None,
        description="Parametric condition; defaults to COND_RT_NOM when omitted",
    ),
) -> dict:
    try:
        return get_selected_measurement(
            _root(request),
            lot_id=lot_id,
            die_id=die_id,
            parameter=parameter,
            condition_id=condition_id,
        )
    except MeasurementLookupError as exc:
        _raise_lookup(exc)
        raise  # pragma: no cover


@router.get(
    "/dies/{die_id}/distribution",
    summary="Measurement distribution stats (read-only, SYNTHETIC)",
)
def get_die_distribution(
    die_id: str,
    request: Request,
    lot_id: str = Query(..., description="Lot that owns the die"),
    parameter: str = Query(..., description="Measurement parameter"),
    scope: str = Query("die", description="Population scope: die or lot"),
    condition_id: str | None = Query(
        None,
        description="Optional parametric condition filter",
    ),
) -> dict:
    try:
        return get_distribution(
            _root(request),
            lot_id=lot_id,
            die_id=die_id,
            parameter=parameter,
            scope=scope,
            condition_id=condition_id,
        )
    except MeasurementLookupError as exc:
        _raise_lookup(exc)
        raise  # pragma: no cover


@router.get(
    "/dies/{die_id}/conditions",
    summary="Parametric condition-level measurements (read-only)",
)
def get_die_conditions(
    die_id: str,
    request: Request,
    lot_id: str = Query(..., description="Lot that owns the die"),
    parameter: str = Query(..., description="Measurement parameter"),
) -> dict:
    try:
        return get_conditions(
            _root(request),
            lot_id=lot_id,
            die_id=die_id,
            parameter=parameter,
        )
    except MeasurementLookupError as exc:
        _raise_lookup(exc)
        raise  # pragma: no cover
