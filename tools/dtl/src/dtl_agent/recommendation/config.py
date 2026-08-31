"""Phase 8 recommendation policy configuration."""



from __future__ import annotations



import json

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Any





POLICY_CONFIG_VERSION = "phase8_policy_v1"





@dataclass

class RecommendationConfig:

    """MVP config. Layer-3 numeric thresholds stay null until explicitly set."""



    TOP_N: int = 5

    core_checkpoint_path: str = "artifacts/ml/checkpoints/core_gru_best.pt"

    parametric_checkpoint_path: str = "artifacts/ml/checkpoints/parametric_mlp_best.pt"

    core_candidate_grid_path: str = "artifacts/simulation/core/candidate_grid.csv"

    core_candidate_results_path: str = "artifacts/simulation/core/candidate_results.csv"

    parametric_candidate_grid_path: str = "artifacts/simulation/parametric/candidate_grid.csv"

    parametric_candidate_results_path: str = (

        "artifacts/simulation/parametric/candidate_results.csv"

    )

    joint_enabled: bool = False

    include_tree_baseline_diagnostic: bool = False

    evidence_origin_label: str = "SIMULATOR_DERIVED"

    synthetic_assumed_max_evidence_level: str = "MODERATE_EVIDENCE"

    policy_config_version: str = POLICY_CONFIG_VERSION



    # Layer-3 proposed policy thresholds (unset = skip check)

    max_violation_rate_for_recommend: float | None = None

    max_borderline_rate_for_recommend: float | None = None

    min_simulated_yield_for_recommend: float | None = None

    min_worst_condition_yield_for_recommend: float | None = None

    max_abs_delta_for_recommend: float | None = None

    max_delta_percent_for_recommend: float | None = None

    # DEFERRED: semantics not approved — see PHASE_8_FINAL_IMPLEMENTATION_SPEC §Remaining policy decisions.

    # Present for future operator configuration only; not wired in safety.py or policy.py.

    looser_requires_stricter_gate: bool | None = None



    def resolve_path(self, project_root: Path, relative: str) -> Path:

        p = Path(relative)

        return p if p.is_absolute() else project_root / p



    def to_dict(self) -> dict[str, Any]:

        return asdict(self)



    @classmethod

    def from_dict(cls, data: dict[str, Any] | None) -> "RecommendationConfig":

        if not data:

            return cls()

        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]

        return cls(**{k: v for k, v in data.items() if k in allowed})





def load_recommendation_config(path: Path | None = None) -> RecommendationConfig:

    """Load policy config from an optional JSON file.



    When *path* is None or the file does not exist, returns defaults with all

    Layer-3 numeric thresholds unset (null).

    """

    if path is None or not path.is_file():

        return RecommendationConfig()

    data = json.loads(path.read_text(encoding="utf-8"))

    return RecommendationConfig.from_dict(data)
