"""Core GRU and Parametric MLP inference adapters for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dtl_agent.ml.datasets.phase7_datasets import (
    CORE_CAND_NUM,
    PARAM_CAND_NUM,
    PARAM_NORM_PREFIX,
    _cat_map,
    CoreSequenceStore,
)
from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml.models.parametric_encoder import ParametricMLPRanker
from dtl_agent.recommendation.config import RecommendationConfig


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass
class VocabBundle:
    core_param_map: dict[str, int]
    core_dir_map: dict[str, int]
    core_tight_map: dict[str, int]
    param_param_map: dict[str, int]
    param_dir_map: dict[str, int]
    param_tight_map: dict[str, int]
    param_cond_map: dict[str, int]
    param_mode_map: dict[str, int]
    param_norm_cols: list[str]


class ModelBundle:
    """Lazy-loaded models + train vocabs from Phase 6/7 artifacts."""

    def __init__(self, project_root: Path, config: RecommendationConfig) -> None:
        self.project_root = project_root
        self.config = config
        self.device = torch.device("cpu")
        self._ready = False
        self.vocabs: VocabBundle | None = None
        self.core_model: CoreGRURanker | None = None
        self.param_model: ParametricMLPRanker | None = None
        self.seq_store: CoreSequenceStore | None = None
        self.core_examples: pd.DataFrame | None = None
        self.param_examples: pd.DataFrame | None = None
        self.core_checkpoint_id: str | None = None
        self.param_checkpoint_id: str | None = None
        self.load_errors: list[str] = []

    def ensure_loaded(self) -> bool:
        if self._ready:
            return True
        try:
            ml_root = self.project_root / "artifacts" / "ml_dataset"
            train_core = pd.read_parquet(ml_root / "train" / "core_candidate_examples.parquet")
            train_param = pd.read_parquet(ml_root / "train" / "parametric_candidate_examples.parquet")
            seq_df = pd.read_parquet(ml_root / "sequences" / "core_sequences.parquet")
            self.seq_store = CoreSequenceStore(seq_df)
            self._ml_root = ml_root

            norm_cols = [c for c in train_param.columns if c.startswith(PARAM_NORM_PREFIX)]
            self.vocabs = VocabBundle(
                core_param_map=_cat_map(train_core["parameter"].astype(str).tolist()),
                core_dir_map=_cat_map(train_core["direction"].astype(str).tolist()),
                core_tight_map=_cat_map(train_core["tighten_or_loosen"].astype(str).tolist()),
                param_param_map=_cat_map(train_param["parameter"].astype(str).tolist()),
                param_dir_map=_cat_map(train_param["direction"].astype(str).tolist()),
                param_tight_map=_cat_map(train_param["tighten_or_loosen"].astype(str).tolist()),
                param_cond_map=_cat_map(train_param["condition_id"].astype(str).tolist()),
                param_mode_map=_cat_map(train_param["test_mode"].astype(str).tolist()),
                param_norm_cols=norm_cols,
            )

            core_ckpt = self.config.resolve_path(self.project_root, self.config.core_checkpoint_path)
            param_ckpt = self.config.resolve_path(
                self.project_root, self.config.parametric_checkpoint_path
            )
            if not core_ckpt.is_file():
                self.load_errors.append("core_checkpoint_missing")
            if not param_ckpt.is_file():
                self.load_errors.append("parametric_checkpoint_missing")
            if self.load_errors:
                return False

            self.core_model = CoreGRURanker(
                n_parameter=len(self.vocabs.core_param_map),
                n_direction=len(self.vocabs.core_dir_map),
                n_tight=len(self.vocabs.core_tight_map),
            )
            state_c = _load_checkpoint(core_ckpt)
            self.core_model.load_state_dict(state_c["model_state"])
            self.core_model.eval()
            self.core_checkpoint_id = str(core_ckpt)

            self.param_model = ParametricMLPRanker(
                norm_num_dim=len(norm_cols),
                n_parameter=len(self.vocabs.param_param_map),
                n_direction=len(self.vocabs.param_dir_map),
                n_tight=len(self.vocabs.param_tight_map),
                n_condition=len(self.vocabs.param_cond_map),
                n_mode=len(self.vocabs.param_mode_map),
            )
            state_p = _load_checkpoint(param_ckpt)
            self.param_model.load_state_dict(state_p["model_state"])
            self.param_model.eval()
            self.param_checkpoint_id = str(param_ckpt)
            self._ready = True
            return True
        except Exception as exc:  # noqa: BLE001
            self.load_errors.append(f"model_load_error:{type(exc).__name__}:{exc}")
            return False

    def find_core_examples(self, lot_id: str, die_id: str, parameter: str) -> pd.DataFrame:
        ml_root = getattr(self, "_ml_root", self.project_root / "artifacts" / "ml_dataset")
        parts = []
        filters = [
            ("lot_id", "==", str(lot_id)),
            ("die_id", "==", str(die_id)),
            ("parameter", "==", str(parameter)),
        ]
        for split in ("train", "validation", "test"):
            path = ml_root / split / "core_candidate_examples.parquet"
            try:
                sub = pd.read_parquet(path, filters=filters)
            except Exception:  # noqa: BLE001
                df = pd.read_parquet(path)
                sub = df[
                    (df["lot_id"].astype(str) == str(lot_id))
                    & (df["die_id"].astype(str) == str(die_id))
                    & (df["parameter"].astype(str) == str(parameter))
                ]
            if not sub.empty:
                parts.append(sub)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def find_param_examples(self, lot_id: str, die_id: str, parameter: str) -> pd.DataFrame:
        ml_root = getattr(self, "_ml_root", self.project_root / "artifacts" / "ml_dataset")
        parts = []
        filters = [
            ("lot_id", "==", str(lot_id)),
            ("die_id", "==", str(die_id)),
            ("parameter", "==", str(parameter)),
        ]
        for split in ("train", "validation", "test"):
            path = ml_root / split / "parametric_candidate_examples.parquet"
            try:
                sub = pd.read_parquet(path, filters=filters)
            except Exception:  # noqa: BLE001
                df = pd.read_parquet(path)
                sub = df[
                    (df["lot_id"].astype(str) == str(lot_id))
                    & (df["die_id"].astype(str) == str(die_id))
                    & (df["parameter"].astype(str) == str(parameter))
                ]
            if not sub.empty:
                parts.append(sub)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


class CoreGRUInferencer:
    model_id = "core_gru"

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def score_lot_die_parameter(
        self, *, lot_id: str, die_id: str, parameter: str
    ) -> tuple[pd.DataFrame | None, str | None]:
        """Return scored candidate rows or (None, error_code). Never fabricates sequences."""
        if not self.bundle.ensure_loaded():
            return None, "model_unavailable"
        assert self.bundle.core_model is not None
        assert self.bundle.vocabs is not None
        assert self.bundle.seq_store is not None

        sid = f"{lot_id}::{die_id}"
        if sid not in self.bundle.seq_store.mats:
            return None, "core_sequence_unavailable"

        sub = self.bundle.find_core_examples(lot_id, die_id, parameter).copy()
        if sub.empty:
            return None, "core_examples_missing"

        seq = self.bundle.seq_store.get(sid)
        v = self.bundle.vocabs
        scores: list[float] = []
        with torch.no_grad():
            for _, r in sub.iterrows():
                cand = np.array([float(r[c]) for c in CORE_CAND_NUM], dtype=np.float32)
                pred = self.bundle.core_model(
                    sequence=torch.from_numpy(np.array(seq, copy=True)).unsqueeze(0),
                    cand_num=torch.from_numpy(cand).unsqueeze(0),
                    parameter_idx=torch.tensor(
                        [v.core_param_map[str(r["parameter"])]], dtype=torch.long
                    ),
                    direction_idx=torch.tensor(
                        [v.core_dir_map[str(r["direction"])]], dtype=torch.long
                    ),
                    tight_idx=torch.tensor(
                        [v.core_tight_map[str(r["tighten_or_loosen"])]], dtype=torch.long
                    ),
                    cross_domain=torch.tensor(
                        [float(bool(r.get("cross_domain_available", False)))],
                        dtype=torch.float32,
                    ),
                )
                scores.append(float(pred.squeeze().cpu().numpy()))
        sub = sub.reset_index(drop=True)
        sub["ml_score"] = scores
        sub["model_id"] = self.model_id
        if "source_status" not in sub.columns:
            sub["source_status"] = "SOURCE_CONFIRMED"
        if "candidate_delta" not in sub.columns and "delta_absolute" in sub.columns:
            sub["candidate_delta"] = sub["delta_absolute"]
        return sub, None


class ParametricMLPInferencer:
    model_id = "parametric_mlp"

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def score_lot_die_parameter(
        self, *, lot_id: str, die_id: str, parameter: str
    ) -> tuple[pd.DataFrame | None, str | None]:
        if not self.bundle.ensure_loaded():
            return None, "model_unavailable"
        assert self.bundle.param_model is not None
        assert self.bundle.vocabs is not None

        sub = self.bundle.find_param_examples(lot_id, die_id, parameter).copy()
        if sub.empty:
            return None, "parametric_examples_missing"

        v = self.bundle.vocabs
        scores: list[float] = []
        with torch.no_grad():
            for _, r in sub.iterrows():
                cand = np.array([float(r[c]) for c in PARAM_CAND_NUM], dtype=np.float32)
                norm = np.array([float(r[c]) for c in v.param_norm_cols], dtype=np.float32)
                cond_num = np.array(
                    [float(r["temperature_c"]), float(r["vdd_applied"])], dtype=np.float32
                )
                pred = self.bundle.param_model(
                    norm_num=torch.from_numpy(norm).unsqueeze(0),
                    cand_num=torch.from_numpy(cand).unsqueeze(0),
                    cond_num=torch.from_numpy(cond_num).unsqueeze(0),
                    parameter_idx=torch.tensor(
                        [v.param_param_map[str(r["parameter"])]], dtype=torch.long
                    ),
                    direction_idx=torch.tensor(
                        [v.param_dir_map[str(r["direction"])]], dtype=torch.long
                    ),
                    tight_idx=torch.tensor(
                        [v.param_tight_map[str(r["tighten_or_loosen"])]], dtype=torch.long
                    ),
                    condition_idx=torch.tensor(
                        [v.param_cond_map[str(r["condition_id"])]], dtype=torch.long
                    ),
                    mode_idx=torch.tensor(
                        [v.param_mode_map[str(r["test_mode"])]], dtype=torch.long
                    ),
                )
                scores.append(float(pred.squeeze().cpu().numpy()))
        sub = sub.reset_index(drop=True)
        sub["ml_score"] = scores
        sub["model_id"] = self.model_id
        if "source_status" not in sub.columns:
            sub["source_status"] = "SYNTHETIC_ASSUMED"
        agg = sub.groupby("candidate_limit", as_index=False).agg(
            ml_score=("ml_score", "mean"),
            current_limit=("current_limit", "first"),
            candidate_delta=("candidate_delta", "first"),
            candidate_delta_percent=("candidate_delta_percent", "first"),
            direction=("direction", "first"),
            tighten_or_loosen=("tighten_or_loosen", "first"),
            unit=("unit", "first"),
            test_id=("test_id", "first"),
            source_status=("source_status", "first"),
            n_conditions=("condition_id", "nunique"),
            conditions_present=("condition_id", lambda s: sorted(set(map(str, s)))),
        )
        agg["parameter"] = parameter
        agg["lot_id"] = lot_id
        agg["die_id"] = die_id
        agg["model_id"] = self.model_id
        return agg, None
