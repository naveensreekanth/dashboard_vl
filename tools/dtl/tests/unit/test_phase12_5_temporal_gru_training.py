"""Phase 12.5 — temporal CoreGRURanker retraining tests."""

from __future__ import annotations

import json

import pytest
import torch

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.paths import shared_ml_dataset_root, temporal_artifact_root
from dtl_agent.ml.temporal_training import CHECKPOINT_NAME, CORE_TRAIN_CONFIG, EXPECTED_ARCHITECTURE

ROOT = default_project_root()
SHARED_ML = shared_ml_dataset_root(ROOT)
CKPT = temporal_artifact_root(ROOT) / "shared" / "checkpoints" / CHECKPOINT_NAME
TRAIN_DIR = temporal_artifact_root(ROOT) / "shared" / "training"
LEGACY_CKPT = ROOT / "artifacts" / "ml" / "checkpoints" / "core_gru_best.pt"

pytestmark = pytest.mark.skipif(
    not (SHARED_ML / "train" / "core_candidate_examples.parquet").is_file(),
    reason="temporal shared ML dataset missing",
)


@pytest.fixture(scope="module")
def training_complete():
    if not CKPT.is_file() or not (TRAIN_DIR / "metrics.json").is_file():
        pytest.skip("Phase 12.5 training artifacts not yet generated")
    return json.loads((TRAIN_DIR / "metrics.json").read_text(encoding="utf-8"))


def test_temporal_checkpoint_exists_separate_from_legacy(training_complete):
    assert CKPT.is_file()
    assert CKPT.resolve() != LEGACY_CKPT.resolve()
    assert "temporal" in str(CKPT).replace("\\", "/")
    assert CKPT.name == CHECKPOINT_NAME


def test_legacy_checkpoint_not_overwritten(training_complete):
    repro = json.loads((TRAIN_DIR / "reproducibility.json").read_text(encoding="utf-8"))
    assert repro["legacy_checkpoint_sha256_before"] == repro["legacy_checkpoint_sha256_after"]
    summary = json.loads(
        (temporal_artifact_root(ROOT) / "shared" / "PHASE_12_5_TRAINING_SUMMARY.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["legacy_checkpoint_untouched"] is True
    assert summary["wired_into_recommend"] is False


def test_architecture_unchanged(training_complete):
    arch = json.loads((TRAIN_DIR / "architecture.json").read_text(encoding="utf-8"))
    assert arch["model_class"] == "CoreGRURanker"
    assert arch["seq_input_dim"] == EXPECTED_ARCHITECTURE["seq_input_dim"]
    assert arch["sequence_length"] == 200
    assert arch["cand_num_dim"] == 4
    assert arch["production_month_is_gru_feature"] is False
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert tuple(state["model_state"]["gru.weight_ih_l0"].shape) == (192, 5)


def test_train_config_matches_phase7_core(training_complete):
    cfg = json.loads((TRAIN_DIR / "train_config.json").read_text(encoding="utf-8"))
    assert cfg["lr"] == CORE_TRAIN_CONFIG.lr
    assert cfg["batch_size"] == CORE_TRAIN_CONFIG.batch_size
    assert cfg["max_epochs"] == CORE_TRAIN_CONFIG.max_epochs
    assert cfg["patience"] == CORE_TRAIN_CONFIG.patience
    assert cfg["seed"] == CORE_TRAIN_CONFIG.seed


def test_metrics_and_per_month(training_complete):
    m = training_complete
    assert "test" in m and "validation" in m
    assert m["test"]["n_examples"] > 0
    for month in ("2026-01", "2026-02", "2026-03"):
        assert month in m["test_by_month"]
        assert m["test_by_month"][month]["n_examples"] > 0
        assert "huber_loss" in m["test_by_month"][month]
        assert "ndcg_at_k" in m["test_by_month"][month]


def test_ranking_not_all_constant(training_complete):
    sanity = training_complete["ranking_sanity"]
    assert sanity
    assert any(not r["constant_score"] for r in sanity)


def test_split_lots_disjoint(training_complete):
    split = training_complete["split"]
    tr, va, te = set(split["train_lots"]), set(split["validation_lots"]), set(split["test_lots"])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert split["counts"] == {"train": 12, "validation": 4, "test": 4}


def test_training_used_temporal_dataset_only(training_complete):
    ds = training_complete["dataset"]["root"].replace("\\", "/")
    assert "artifacts/temporal/shared/ml_dataset" in ds
