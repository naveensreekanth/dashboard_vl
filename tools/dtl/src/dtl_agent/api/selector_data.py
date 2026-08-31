"""Read-only lot/die/parameter selector index for dashboard dropdowns."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dtl_agent.recommendation.schemas import CORE_PARAMETERS, PARAMETRIC_PARAMETERS


@dataclass(frozen=True)
class SelectorIndex:
    lots: tuple[str, ...]
    dies_by_lot: dict[str, tuple[str, ...]]
    params_by_lot_die: dict[tuple[str, str], tuple[str, ...]]


def _read_lot_ids(csv_path: Path) -> set[str]:
    lots: set[str] = set()
    if not csv_path.is_file():
        return lots
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            lot = (row.get("lot_id") or "").strip()
            if lot:
                lots.add(lot)
    return lots


def _read_parts(
    csv_path: Path,
    *,
    dies_by_lot: dict[str, set[str]],
    domain_by_lot_die: dict[tuple[str, str], set[str]],
    domain: str,
) -> None:
    if not csv_path.is_file():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            lot = (row.get("lot_id") or "").strip()
            die = (row.get("die_id") or "").strip()
            if not lot or not die:
                continue
            dies_by_lot.setdefault(lot, set()).add(die)
            domain_by_lot_die.setdefault((lot, die), set()).add(domain)


def _catalog_parameters(csv_path: Path, allowed: frozenset[str]) -> tuple[str, ...]:
    if not csv_path.is_file():
        return tuple(sorted(allowed))
    found: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            param = (row.get("parameter") or "").strip()
            eligible = (row.get("dtl_eligible") or "").strip().lower()
            if param in allowed and eligible in {"", "true", "1", "yes"}:
                found.add(param)
    return tuple(sorted(found or allowed))


@lru_cache(maxsize=4)
def load_selector_index(project_root: str) -> SelectorIndex:
    """Load lot/die/parameter selectors from canonical dim/catalog files.

    Does not scan raw measurement tables. Lots come from lots_dim, dies from
    parts_dim, and parameters from domain membership + test catalogs.
    """
    root = Path(project_root)
    data_core = root / "data" / "core"
    data_param = root / "data" / "parametric"

    lots: set[str] = set()
    lots.update(_read_lot_ids(data_core / "lots_dim.csv"))
    lots.update(_read_lot_ids(data_param / "lots_dim.csv"))

    dies_by_lot_mut: dict[str, set[str]] = {}
    domain_by_lot_die: dict[tuple[str, str], set[str]] = {}
    _read_parts(
        data_core / "parts_dim.csv",
        dies_by_lot=dies_by_lot_mut,
        domain_by_lot_die=domain_by_lot_die,
        domain="core",
    )
    _read_parts(
        data_param / "parts_dim.csv",
        dies_by_lot=dies_by_lot_mut,
        domain_by_lot_die=domain_by_lot_die,
        domain="parametric",
    )

    lots.update(dies_by_lot_mut.keys())
    core_params = _catalog_parameters(data_core / "test_catalog.csv", CORE_PARAMETERS)
    param_params = _catalog_parameters(data_param / "test_catalog.csv", PARAMETRIC_PARAMETERS)

    params_by_lot_die: dict[tuple[str, str], tuple[str, ...]] = {}
    for key, domains in domain_by_lot_die.items():
        params: list[str] = []
        if "core" in domains:
            params.extend(core_params)
        if "parametric" in domains:
            params.extend(param_params)
        params_by_lot_die[key] = tuple(dict.fromkeys(params))

    return SelectorIndex(
        lots=tuple(sorted(lots)),
        dies_by_lot={lot: tuple(sorted(dies)) for lot, dies in dies_by_lot_mut.items()},
        params_by_lot_die=params_by_lot_die,
    )
