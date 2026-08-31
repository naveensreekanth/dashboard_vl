"""Read-only selector routes for lots, dies, and available parameters."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from dtl_agent.api.selector_data import load_selector_index

router = APIRouter(tags=["selectors"])


def _index(request: Request):
    root = str(request.app.state.project_root.resolve())
    return load_selector_index(root)


@router.get("/lots", summary="List available lots from canonical datasets")
def get_lots(request: Request) -> dict[str, list[str]]:
    index = _index(request)
    return {"lots": list(index.lots)}


@router.get("/lots/{lot_id}/dies", summary="List available dies for a lot")
def get_lot_dies(lot_id: str, request: Request) -> dict[str, object]:
    index = _index(request)
    dies = index.dies_by_lot.get(lot_id)
    if lot_id not in index.lots:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lot not found.",
        )
    return {"lot_id": lot_id, "dies": list(dies or ())}


@router.get(
    "/lots/{lot_id}/dies/{die_id}/parameters",
    summary="List available parameters for selected lot/die",
)
def get_lot_die_parameters(lot_id: str, die_id: str, request: Request) -> dict[str, object]:
    index = _index(request)
    dies = index.dies_by_lot.get(lot_id)
    if lot_id not in index.lots:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lot not found.",
        )
    if dies is None or die_id not in dies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Die not found for lot.",
        )
    params = index.params_by_lot_die.get((lot_id, die_id), ())
    return {"lot_id": lot_id, "die_id": die_id, "parameters": list(params)}
