"""Phase 12.4 — temporal Core GRU ML dataset assembly tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.paths import shared_ml_dataset_root, temporal_data_root
from dtl_agent.features.core_engine import EXPECTED_SEQUENCE_LENGTH, SEQUENCE_FEATURE_ORDER
from dtl_agent.ml.datasets.phase7_datasets import CORE_CAND_NUM, CoreCandidateDataset, CoreSequenceStore
from dtl_agent.ml_dataset.temporal_pipeline import (
    TEMPORAL_MONTHS,
    build_temporal_lot_split_map,
    run_temporal_ml_dataset_assembly,
)

ROOT = default_project_root()
SHARED = shared_ml_dataset_root(ROOT)
TEMPORAL_AVAILABLE = (temporal_data_root(ROOT) / "2026-01" / "actual_die" / "measurements.csv").is_file()

pytestmark = pytest.mark.skipif(
    not TEMPORAL_AVAILABLE,
    reason="data/3 months data package not present",
)


@pytest.fixture(scope="module")
def shared_artifacts():
    """Use existing shared store if complete; otherwise assemble (may resimulate)."""
    manifest = SHARED / "dataset_manifest.json"
    train_pq = SHARED / "train" / "core_candidate_examples.parquet"
    if manifest.is_file() and train_pq.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))
    arts = run_temporal_ml_dataset_assembly(ROOT, force_resimulate=False)
    return arts.dataset_manifest


@pytest.fixture(scope="module")
def pooled_examples(shared_artifacts):
    frames = []
    for split in ("train", "validation", "test"):
        path = SHARED / split / "core_candidate_examples.parquet"
        assert path.is_file(), f"missing {path}"
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def sequences(shared_artifacts):
    path = SHARED / "sequences" / "core_sequences.parquet"
    assert path.is_file()
    return pd.read_parquet(path)


def test_all_three_months_produce_ml_examples(pooled_examples, shared_artifacts):
    months = set(pooled_examples["production_month"].astype(str).unique())
    assert months == set(TEMPORAL_MONTHS)
    for m in TEMPORAL_MONTHS:
        assert (pooled_examples["production_month"] == m).sum() > 0
        assert shared_artifacts["per_month"][m]["n_examples"] > 0


def test_month_exists_on_every_example(pooled_examples):
    assert pooled_examples["production_month"].notna().all()
    assert pooled_examples["production_month"].isin(TEMPORAL_MONTHS).all()


def test_sequence_id_is_month_prefixed(pooled_examples):
    assert pooled_examples["sequence_id"].astype(str).map(lambda s: s.count("::") == 2).all()
    assert pooled_examples.apply(
        lambda r: str(r["sequence_id"]).startswith(f"{r['production_month']}::"),
        axis=1,
    ).all()


def test_sequence_shape_200x5(pooled_examples, sequences):
    store = CoreSequenceStore(sequences)
    # one example per month
    for month in TEMPORAL_MONTHS:
        row = pooled_examples[pooled_examples["production_month"] == month].iloc[0]
        arr = store.get(str(row["sequence_id"]))
        assert arr.shape == (EXPECTED_SEQUENCE_LENGTH, len(SEQUENCE_FEATURE_ORDER))


def test_candidate_numeric_inputs_match_model(pooled_examples):
    for col in CORE_CAND_NUM:
        assert col in pooled_examples.columns
        assert pooled_examples[col].notna().all()
    ds = CoreCandidateDataset(
        pooled_examples.head(4),
        CoreSequenceStore(
            pd.read_parquet(SHARED / "sequences" / "core_sequences.parquet")
        ),
    )
    item = ds[0]
    assert tuple(item["cand_num"].shape) == (len(CORE_CAND_NUM),)


def test_target_score_equals_objective_score(pooled_examples):
    assert (
        pooled_examples["target_score"].astype(float).to_numpy()
        == pooled_examples["objective_score"].astype(float).to_numpy()
    ).all()


def test_no_missing_objective_scores(pooled_examples):
    assert pooled_examples["objective_score"].notna().all()
    assert pooled_examples["target_score"].notna().all()


def test_no_duplicate_temporal_candidate_examples(pooled_examples):
    dup = pooled_examples.duplicated(
        subset=["production_month", "lot_id", "die_id", "parameter", "candidate_limit"],
        keep=False,
    )
    assert not dup.any()


def test_train_val_test_lots_disjoint(shared_artifacts):
    split = json.loads((SHARED / "split_manifest.json").read_text(encoding="utf-8"))
    lot_to_split = split["lot_to_split"]
    train = {l for l, s in lot_to_split.items() if s == "train"}
    val = {l for l, s in lot_to_split.items() if s == "validation"}
    test = {l for l, s in lot_to_split.items() if s == "test"}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # same lot never appears in different splits in examples
    assert shared_artifacts["leakage_checks"]["train_val_disjoint"]
    assert shared_artifacts["leakage_checks"]["train_test_disjoint"]
    assert shared_artifacts["leakage_checks"]["val_test_disjoint"]


def test_all_months_represented_in_training(pooled_examples):
    train = pooled_examples[pooled_examples["split"] == "train"]
    assert set(train["production_month"].astype(str).unique()) == set(TEMPORAL_MONTHS)


def test_legacy_ml_dataset_untouched(shared_artifacts):
    legacy = ROOT / "artifacts" / "ml_dataset"
    summary = ROOT / "artifacts" / "temporal" / "PHASE_12_4_ASSEMBLY_SUMMARY.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        assert data.get("legacy_ml_dataset_untouched") is True
    # Shared store is not the legacy path
    assert SHARED.resolve() != legacy.resolve()
    assert "temporal" in str(SHARED).replace("\\", "/")
    # Writing to shared must not relocate legacy core train parquet
    legacy_train = legacy / "train" / "core_candidate_examples.parquet"
    if legacy_train.is_file():
        # Ensure temporal shared path is distinct file
        shared_train = SHARED / "train" / "core_candidate_examples.parquet"
        assert shared_train.resolve() != legacy_train.resolve()


def test_lot_split_stratified_helper():
    cats = {
        "DTL_NORM_001": "NORMAL",
        "DTL_NORM_002": "NORMAL",
        "DTL_NORM_003": "NORMAL",
        "DTL_EDGE_001": "EDGE",
        "DTL_EDGE_002": "EDGE",
        "DTL_EDGE_003": "EDGE",
    }
    split_map, manifest = build_temporal_lot_split_map(cats, seed=1)
    assert set(split_map) == set(cats)
    assert manifest["counts"]["train_lots"] == 2
    assert manifest["counts"]["validation_lots"] == 2
    assert manifest["counts"]["test_lots"] == 2
