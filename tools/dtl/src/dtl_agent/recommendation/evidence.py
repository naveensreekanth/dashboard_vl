"""Simulation evidence lookup (artifact-first; never invent)."""



from __future__ import annotations



from pathlib import Path

from typing import Any



import pandas as pd



from dtl_agent.recommendation.config import RecommendationConfig

from dtl_agent.recommendation.schemas import SimulationEvidence





class SimulationEvidenceLookup:

    def __init__(self, project_root: Path, config: RecommendationConfig) -> None:

        self.config = config

        self.core = pd.read_csv(

            config.resolve_path(project_root, config.core_candidate_results_path)

        )

        self.param = pd.read_csv(

            config.resolve_path(project_root, config.parametric_candidate_results_path)

        )



    def lookup(self, *, domain: str, parameter: str, candidate_limit: float) -> SimulationEvidence:

        df = self.core if domain == "core" else self.param

        sub = df[

            (df["parameter"].astype(str) == str(parameter))

            & ((df["candidate_limit"] - float(candidate_limit)).abs() <= 1e-9)

        ]

        if sub.empty:

            return SimulationEvidence(

                evidence_origin=self.config.evidence_origin_label,

                population_level_aggregate=True,

                parameter=parameter,

                candidate_limit=float(candidate_limit),

                found=False,

            )

        r = sub.iloc[0].to_dict()



        def _f(key: str) -> float | None:

            v = r.get(key)

            if v is None or (isinstance(v, float) and pd.isna(v)):

                return None

            try:

                return float(v)

            except (TypeError, ValueError):

                return None



        return SimulationEvidence(

            evidence_origin=self.config.evidence_origin_label,

            population_level_aggregate=True,

            parameter=parameter,

            candidate_limit=float(candidate_limit),

            simulated_yield=_f("simulated_yield"),

            simulated_fail_rate=_f("simulated_fail_rate"),

            violation_rate=_f("violation_rate"),

            borderline_rate=_f("borderline_rate"),

            risky_rate=_f("risky_rate"),

            false_fail_proxy=_f("false_fail_proxy"),

            defective_proxy=_f("defective_proxy"),

            objective_score=_f("objective_score"),

            worst_condition_yield=_f("worst_condition_yield"),

            worst_condition_violation_rate=_f("worst_condition_violation_rate"),

            evaluated_conditions=r.get("evaluated_conditions"),

            found=True,

            raw={k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r.items()},

        )





def risk_inputs_from_evidence(ev: SimulationEvidence) -> dict[str, Any]:

    return {

        "simulated_yield": ev.simulated_yield,

        "violation_rate": ev.violation_rate,

        "borderline_rate": ev.borderline_rate,

        "worst_condition_yield": ev.worst_condition_yield,

        "worst_condition_violation_rate": ev.worst_condition_violation_rate,

        "objective_score": ev.objective_score,

        "found": ev.found,

        "evidence_origin": ev.evidence_origin,

    }
