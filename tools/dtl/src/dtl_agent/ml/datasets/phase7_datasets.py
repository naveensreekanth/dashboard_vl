"""PyTorch datasets and feature builders for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


from dtl_agent.features.core_engine import SEQUENCE_FEATURE_ORDER

CORE_SEQ_FEATURES = list(SEQUENCE_FEATURE_ORDER)
CORE_CAND_NUM = ["candidate_limit", "current_limit", "candidate_delta", "candidate_delta_percent"]
PARAM_CAND_NUM = ["candidate_limit", "current_limit", "candidate_delta", "candidate_delta_percent"]
PARAM_NORM_PREFIX = "norm_param_"


@dataclass
class CoreBatchRow:
    sequence: np.ndarray
    cand_num: np.ndarray
    parameter_idx: int
    direction_idx: int
    tight_idx: int
    cross_domain: float
    target: float
    metadata: dict[str, Any]


def _cat_map(values: list[str]) -> dict[str, int]:
    return {v: i for i, v in enumerate(sorted(set(values)))}


class CoreSequenceStore:
    def __init__(self, seq_df: pd.DataFrame) -> None:
        mats: dict[str, np.ndarray] = {}
        for sid, g in seq_df.groupby("sequence_id"):
            gg = g.sort_values("pattern_id")
            arr = gg[CORE_SEQ_FEATURES].to_numpy(dtype=np.float32)
            mats[str(sid)] = arr
        self.mats = mats

    def get(self, sid: str) -> np.ndarray:
        return self.mats[sid]


class CoreCandidateDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, seq_store: CoreSequenceStore) -> None:
        self.df = rows.reset_index(drop=True).copy()
        self.seq_store = seq_store
        self.param_map = _cat_map(self.df["parameter"].astype(str).tolist())
        self.dir_map = _cat_map(self.df["direction"].astype(str).tolist())
        self.tight_map = _cat_map(self.df["tighten_or_loosen"].astype(str).tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        r = self.df.iloc[idx]
        seq = self.seq_store.get(str(r["sequence_id"]))
        cand = np.array([float(r[c]) for c in CORE_CAND_NUM], dtype=np.float32)
        out: dict[str, torch.Tensor | str] = {
            "sequence": torch.from_numpy(np.array(seq, copy=True)),
            "cand_num": torch.from_numpy(cand),
            "parameter_idx": torch.tensor(self.param_map[str(r["parameter"])], dtype=torch.long),
            "direction_idx": torch.tensor(self.dir_map[str(r["direction"])], dtype=torch.long),
            "tight_idx": torch.tensor(self.tight_map[str(r["tighten_or_loosen"])], dtype=torch.long),
            "cross_domain": torch.tensor(float(bool(r.get("cross_domain_available", False))), dtype=torch.float32),
            "target": torch.tensor(float(r["target_score"]), dtype=torch.float32),
            "example_id": str(r["example_id"]),
        }
        return out


class ParametricTabularDataset(Dataset):
    def __init__(self, rows: pd.DataFrame) -> None:
        self.df = rows.reset_index(drop=True).copy()
        self.norm_cols = [c for c in self.df.columns if c.startswith(PARAM_NORM_PREFIX)]
        self.param_map = _cat_map(self.df["parameter"].astype(str).tolist())
        self.dir_map = _cat_map(self.df["direction"].astype(str).tolist())
        self.tight_map = _cat_map(self.df["tighten_or_loosen"].astype(str).tolist())
        self.cond_map = _cat_map(self.df["condition_id"].astype(str).tolist())
        self.mode_map = _cat_map(self.df["test_mode"].astype(str).tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        r = self.df.iloc[idx]
        cand = np.array([float(r[c]) for c in PARAM_CAND_NUM], dtype=np.float32)
        norm = np.array([float(r[c]) for c in self.norm_cols], dtype=np.float32)
        cond_num = np.array([float(r["temperature_c"]), float(r["vdd_applied"])], dtype=np.float32)
        out: dict[str, torch.Tensor | str] = {
            "norm_num": torch.from_numpy(norm),
            "cand_num": torch.from_numpy(cand),
            "cond_num": torch.from_numpy(cond_num),
            "parameter_idx": torch.tensor(self.param_map[str(r["parameter"])], dtype=torch.long),
            "direction_idx": torch.tensor(self.dir_map[str(r["direction"])], dtype=torch.long),
            "tight_idx": torch.tensor(self.tight_map[str(r["tighten_or_loosen"])], dtype=torch.long),
            "condition_idx": torch.tensor(self.cond_map[str(r["condition_id"])], dtype=torch.long),
            "mode_idx": torch.tensor(self.mode_map[str(r["test_mode"])], dtype=torch.long),
            "target": torch.tensor(float(r["target_score"]), dtype=torch.float32),
            "example_id": str(r["example_id"]),
        }
        return out


def load_phase6_ml_splits(root: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for split in ["train", "validation", "test"]:
        out[f"core_{split}"] = pd.read_parquet(root / split / "core_candidate_examples.parquet")
        out[f"param_{split}"] = pd.read_parquet(root / split / "parametric_candidate_examples.parquet")
    out["core_sequences"] = pd.read_parquet(root / "sequences" / "core_sequences.parquet")
    out["sequence_manifest"] = pd.read_parquet(root / "sequences" / "sequence_manifest.parquet")
    return out
