"""Temporal month-split data loading for the experimental RLS experiment.

Preferred leakage-safe split (supported by month packages):

    train → 2026-01 (all lot splits within the month)
    validation → 2026-02
    test → 2026-03

Note: the production GRU used a *lot-level* split pooled across months
(``split_manifest.json``). That is different. This module documents and uses
the month temporal split for the RLS experiment as requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dtl_agent.ml.datasets.phase7_datasets import CoreSequenceStore
from dtl_agent.ml.rls.features import CORE_PARAMETERS

MONTH_TRAIN = "2026-01"
MONTH_VAL = "2026-02"
MONTH_TEST = "2026-03"
SPLITS = ("train", "validation", "test")


@dataclass
class MonthSplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    seq_store: dict[str, object]
    split_note: str


def _month_examples(root: Path, month: str) -> pd.DataFrame:
    base = root / "artifacts" / "temporal" / month / "ml_dataset"
    frames = []
    for sp in SPLITS:
        path = base / sp / "core_candidate_examples.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["parameter"].isin(list(CORE_PARAMETERS))].copy()
    if not (df["target_score"].astype(float) == df["objective_score"].astype(float)).all():
        raise ValueError(f"{month}: target_score must equal objective_score")
    df = df.sort_values(["example_id"]).reset_index(drop=True)
    return df


def _load_month_sequences(root: Path, months: list[str]) -> dict[str, object]:
    mats: dict[str, object] = {}
    for month in months:
        path = (
            root
            / "artifacts"
            / "temporal"
            / month
            / "ml_dataset"
            / "sequences"
            / "core_sequences.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        store = CoreSequenceStore(pd.read_parquet(path))
        mats.update(store.mats)
    return mats


def load_month_temporal_split(root: Path) -> MonthSplitData:
    """Load Jan/Feb/Mar Core candidate examples + sequences."""
    train = _month_examples(root, MONTH_TRAIN)
    validation = _month_examples(root, MONTH_VAL)
    test = _month_examples(root, MONTH_TEST)
    seq_store = _load_month_sequences(root, [MONTH_TRAIN, MONTH_VAL, MONTH_TEST])
    note = (
        "Temporal month split: train=2026-01, validation=2026-02, test=2026-03. "
        "Each month frame pools that month's train/validation/test lot parquet files. "
        "This differs from the GRU lot-level pooled split in split_manifest.json."
    )
    return MonthSplitData(
        train=train,
        validation=validation,
        test=test,
        seq_store=seq_store,
        split_note=note,
    )


def load_gru_test_predictions(root: Path) -> pd.DataFrame:
    """Held-out CoreGRU predictions on lot-split test lots (all months)."""
    path = (
        root
        / "artifacts"
        / "temporal"
        / "shared"
        / "training"
        / "predictions"
        / "core_test_predictions.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)
