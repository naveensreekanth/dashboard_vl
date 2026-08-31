"""Candidate catalog adapters from Phase 4/5 simulation grids."""



from __future__ import annotations



from dataclasses import dataclass

from pathlib import Path



import pandas as pd



from dtl_agent.recommendation.config import RecommendationConfig

from dtl_agent.recommendation.schemas import (

    CORE_PARAMETERS,

    PARAMETRIC_PARAMETERS,

    UNSUPPORTED_CORE_PARAMETERS,

)





@dataclass

class CatalogRow:

    parameter: str

    test_id: str

    direction: str

    unit: str

    source_status: str

    current_limit: float

    candidate_limit: float

    delta_absolute: float

    delta_percent: float | None

    tighten_or_loosen: str

    domain: str  # core | parametric





class CandidateCatalog:

    def __init__(self, project_root: Path, config: RecommendationConfig | None = None) -> None:

        self.project_root = project_root

        cfg = config or RecommendationConfig()

        self._core = pd.read_csv(

            cfg.resolve_path(project_root, cfg.core_candidate_grid_path)

        )

        self._param = pd.read_csv(

            cfg.resolve_path(project_root, cfg.parametric_candidate_grid_path)

        )



    def is_unsupported(self, parameter: str) -> bool:

        return parameter in UNSUPPORTED_CORE_PARAMETERS



    def domain_for(self, parameter: str) -> str | None:

        if parameter in CORE_PARAMETERS:

            return "core"

        if parameter in PARAMETRIC_PARAMETERS:

            return "parametric"

        return None



    def in_catalog(self, parameter: str, candidate_limit: float, tol: float = 1e-9) -> bool:

        domain = self.domain_for(parameter)

        if domain is None:

            return False

        df = self._core if domain == "core" else self._param

        sub = df[df["parameter"].astype(str) == parameter]

        return bool(((sub["candidate_limit"] - candidate_limit).abs() <= tol).any())



    def rows_for_parameter(self, parameter: str) -> list[CatalogRow]:

        domain = self.domain_for(parameter)

        if domain is None:

            return []

        df = self._core if domain == "core" else self._param

        sub = df[df["parameter"].astype(str) == parameter].copy()

        out: list[CatalogRow] = []

        for _, r in sub.iterrows():

            dp = r.get("delta_percent")

            out.append(

                CatalogRow(

                    parameter=str(r["parameter"]),

                    test_id=str(r["test_id"]),

                    direction=str(r["direction"]),

                    unit=str(r["unit"]),

                    source_status=str(r["source_status"]),

                    current_limit=float(r["current_limit"]),

                    candidate_limit=float(r["candidate_limit"]),

                    delta_absolute=float(r["delta_absolute"]),

                    delta_percent=None if pd.isna(dp) else float(dp),

                    tighten_or_loosen=str(r["tighten_or_loosen"]),

                    domain=domain,

                )

            )

        return out



    def current_row(self, parameter: str) -> CatalogRow | None:

        for row in self.rows_for_parameter(parameter):

            if row.tighten_or_loosen == "CURRENT" or abs(row.delta_absolute) < 1e-12:

                return row

        rows = self.rows_for_parameter(parameter)

        return rows[0] if rows else None
