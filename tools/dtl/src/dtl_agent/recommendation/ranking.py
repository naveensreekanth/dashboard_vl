"""Candidate ranking and Top-N selection."""



from __future__ import annotations



import pandas as pd



from dtl_agent.recommendation.catalog import CandidateCatalog

from dtl_agent.recommendation.config import RecommendationConfig

from dtl_agent.recommendation.schemas import RankedCandidate





def rank_candidates(

    scored_df: pd.DataFrame,

    *,

    lot_id: str,

    die_id: str,

    catalog: CandidateCatalog | None = None,

) -> list[RankedCandidate]:

    """Sort by ml_score desc and assign ml_rank starting at 1. CURRENT always retained.



    ``catalog_valid`` is audit metadata only; safety gate performs its own catalog check.

    """

    if scored_df is None or scored_df.empty:

        return []

    df = scored_df.copy()

    df = df.sort_values("ml_score", ascending=False, kind="mergesort").reset_index(drop=True)

    out: list[RankedCandidate] = []

    for i, r in df.iterrows():

        delta = float(r.get("candidate_delta", r.get("delta_absolute", 0.0)))

        dperc = r.get("candidate_delta_percent", r.get("delta_percent"))

        parameter = str(r["parameter"])

        candidate_limit = float(r["candidate_limit"])

        if catalog is not None:

            catalog_valid = catalog.in_catalog(parameter, candidate_limit)

        else:

            catalog_valid = True

        out.append(

            RankedCandidate(

                parameter=parameter,

                test_id=str(r.get("test_id", "")),

                lot_id=str(lot_id),

                die_id=str(die_id),

                current_limit=float(r["current_limit"]),

                candidate_limit=candidate_limit,

                delta_absolute=delta,

                delta_percent=None if dperc is None or (isinstance(dperc, float) and pd.isna(dperc)) else float(dperc),

                direction=str(r["direction"]),

                tighten_or_loosen=str(r["tighten_or_loosen"]),

                unit=str(r.get("unit", "")),

                source_status=str(r.get("source_status", "")),

                ml_score=float(r["ml_score"]),

                ml_rank=int(i) + 1,

                model_id=str(r.get("model_id", "")),

                catalog_valid=catalog_valid,

            )

        )

    return out





def select_top_n_plus_current(

    ranked: list[RankedCandidate], config: RecommendationConfig

) -> list[RankedCandidate]:

    """Advance Top-N from config plus CURRENT into evidence/gate evaluation."""

    top_n = int(config.TOP_N)

    selected = list(ranked[:top_n])

    current = [c for c in ranked if c.tighten_or_loosen == "CURRENT" or abs(c.delta_absolute) < 1e-12]

    seen = {(c.parameter, c.candidate_limit) for c in selected}

    for c in current:

        key = (c.parameter, c.candidate_limit)

        if key not in seen:

            selected.append(c)

            seen.add(key)

    return selected
