"""Phase 12.5 — retrain CoreGRURanker on temporal shared ML dataset (no architecture change).

Uses ONLY ``artifacts/temporal/shared/ml_dataset/``.
Writes checkpoints under ``artifacts/temporal/shared/checkpoints/``.
Does NOT overwrite ``artifacts/ml/checkpoints/core_gru_best.pt``.
Does NOT wire the new model into ``recommend()``.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.paths import shared_ml_dataset_root, temporal_artifact_root
from dtl_agent.features.core_engine import EXPECTED_SEQUENCE_LENGTH, SEQUENCE_FEATURE_ORDER
from dtl_agent.features.io_utils import file_sha256, write_json
from dtl_agent.ml.datasets.phase7_datasets import (
    CORE_CAND_NUM,
    CoreCandidateDataset,
    CoreSequenceStore,
)
from dtl_agent.ml.evaluation.metrics import group_ranking_metrics, mae, rmse
from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml.pipeline import _eval_model, _rows_with_pred
from dtl_agent.ml.training.trainer import TrainConfig, predict, train_regressor

# Existing Phase 7 Core GRU training hyperparameters (must not silently change).
CORE_TRAIN_CONFIG = TrainConfig(
    lr=8e-4,
    weight_decay=1e-4,
    batch_size=512,
    max_epochs=6,
    patience=2,
    huber_delta=1.0,
    seed=7,
)
TRAINING_SEED = 7
SPLIT_SEED = 12042026  # from Phase 12.4 dataset (not re-rolled here)
CHECKPOINT_NAME = "core_gru_temporal_v1.pt"
DATASET_VERSION = "phase12_4_temporal_core_v1"

EXPECTED_ARCHITECTURE = {
    "model_class": "CoreGRURanker",
    "seq_input_dim": 5,
    "sequence_length": 200,
    "gru_hidden": 64,
    "cand_num_dim": 4,
    "embed_dim": 8,
    "dropout": 0.2,
    "cand_num_fields": list(CORE_CAND_NUM),
    "sequence_channels": list(SEQUENCE_FEATURE_ORDER),
    "output": "scalar ml_score",
    "target": "target_score (= objective_score)",
    "production_month_is_gru_feature": False,
}


class TemporalTrainingError(RuntimeError):
    """Hard stop for temporal GRU training / validation failures."""


@dataclass
class TemporalTrainingArtifacts:
    checkpoint_path: Path
    training_root: Path
    metrics: dict[str, Any]
    architecture: dict[str, Any]
    best: dict[str, Any]
    history: list[dict[str, Any]]
    legacy_checkpoint_untouched: bool
    runtime_seconds: float


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_temporal_splits(shared: Path) -> dict[str, pd.DataFrame]:
    if not shared.is_dir():
        raise TemporalTrainingError(f"Missing temporal ML dataset: {shared}")
    legacy_hint = shared.parent.parent.parent / "ml_dataset"
    # hard guard: shared must be under artifacts/temporal
    if "temporal" not in str(shared).replace("\\", "/"):
        raise TemporalTrainingError(f"Refusing non-temporal dataset path: {shared}")

    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        path = shared / split / "core_candidate_examples.parquet"
        if not path.is_file():
            raise TemporalTrainingError(f"Missing {path}")
        out[split] = pd.read_parquet(path)
    seq_path = shared / "sequences" / "core_sequences.parquet"
    if not seq_path.is_file():
        raise TemporalTrainingError(f"Missing {seq_path}")
    out["sequences"] = pd.read_parquet(seq_path)
    return out


def _verify_examples(df: pd.DataFrame, *, split: str) -> None:
    required = {
        "production_month",
        "sequence_id",
        "candidate_limit",
        "current_limit",
        "candidate_delta",
        "candidate_delta_percent",
        "target_score",
        "objective_score",
        "lot_id",
        "die_id",
        "parameter",
        "example_id",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise TemporalTrainingError(f"{split} examples missing columns: {missing}")
    if df.empty:
        raise TemporalTrainingError(f"{split} examples empty")
    if df["production_month"].isna().any():
        raise TemporalTrainingError(f"{split}: null production_month")
    if not df["sequence_id"].astype(str).map(lambda s: s.count("::") == 2).all():
        raise TemporalTrainingError(f"{split}: sequence_id must be month::lot::die")
    if not np.allclose(
        df["target_score"].astype(float).to_numpy(),
        df["objective_score"].astype(float).to_numpy(),
    ):
        raise TemporalTrainingError(f"{split}: target_score must equal objective_score")


def _verify_split_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    tr = set(train["lot_id"].astype(str).unique())
    va = set(val["lot_id"].astype(str).unique())
    te = set(test["lot_id"].astype(str).unique())
    if not tr.isdisjoint(va) or not tr.isdisjoint(te) or not va.isdisjoint(te):
        raise TemporalTrainingError(
            f"Lot leakage detected: train∩val={tr & va}, train∩test={tr & te}, val∩test={va & te}"
        )
    return {
        "train_lots": sorted(tr),
        "validation_lots": sorted(va),
        "test_lots": sorted(te),
        "counts": {"train": len(tr), "validation": len(va), "test": len(te)},
        "disjoint": True,
        "split_seed_expected": SPLIT_SEED,
    }


def _verify_architecture(model: CoreGRURanker, train_ds: CoreCandidateDataset) -> dict[str, Any]:
    """Record architecture and STOP on mismatch vs locked CoreGRURanker contract."""
    item = train_ds[0]
    seq_shape = tuple(item["sequence"].shape)
    cand_shape = tuple(item["cand_num"].shape)
    if seq_shape != (EXPECTED_SEQUENCE_LENGTH, len(SEQUENCE_FEATURE_ORDER)):
        raise TemporalTrainingError(
            f"Architecture mismatch: sequence shape {seq_shape} != "
            f"({EXPECTED_SEQUENCE_LENGTH}, {len(SEQUENCE_FEATURE_ORDER)}). STOP."
        )
    if cand_shape != (len(CORE_CAND_NUM),):
        raise TemporalTrainingError(
            f"Architecture mismatch: cand_num shape {cand_shape} != ({len(CORE_CAND_NUM)},). STOP."
        )

    gru = model.gru
    arch = {
        **EXPECTED_ARCHITECTURE,
        "n_parameter": int(model.param_emb.num_embeddings),
        "n_direction": int(model.dir_emb.num_embeddings),
        "n_tight": int(model.tight_emb.num_embeddings),
        "observed_sequence_shape": list(seq_shape),
        "observed_cand_num_shape": list(cand_shape),
        "gru_input_size": int(gru.input_size),
        "gru_hidden_size": int(gru.hidden_size),
        "parameter_vocab": dict(train_ds.param_map),
        "direction_vocab": dict(train_ds.dir_map),
        "tighten_vocab": dict(train_ds.tight_map),
    }
    if arch["gru_input_size"] != 5 or arch["gru_hidden_size"] != 64:
        raise TemporalTrainingError(
            f"Architecture mismatch vs CoreGRURanker defaults: {arch}. STOP."
        )
    if arch["n_parameter"] < 1 or arch["n_tight"] < 1:
        raise TemporalTrainingError(f"Invalid embedding sizes: {arch}. STOP.")
    return arch


def _huber_loss_numpy(y: np.ndarray, p: np.ndarray, delta: float = 1.0) -> float:
    err = y - p
    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    lin = abs_err - quad
    return float(np.mean(0.5 * quad**2 + delta * lin))


def _ranking_sanity(
    df: pd.DataFrame,
    *,
    pred_col: str,
    n_groups: int = 6,
    seed: int = TRAINING_SEED,
) -> list[dict[str, Any]]:
    """Inspect candidate rankings for a few representative die×parameter groups."""
    keys = ["production_month", "lot_id", "die_id", "parameter"]
    groups = df.groupby(keys, sort=False)
    # Prefer groups that exist in all three months for the same lot/die/parameter
    rng = np.random.default_rng(seed)
    group_keys = list(groups.groups.keys())
    if not group_keys:
        return []
    # Stratify: pick 2 groups per month when possible
    picked: list[tuple] = []
    for month in ("2026-01", "2026-02", "2026-03"):
        month_keys = [k for k in group_keys if k[0] == month]
        if not month_keys:
            continue
        idx = rng.choice(len(month_keys), size=min(2, len(month_keys)), replace=False)
        for i in idx:
            picked.append(month_keys[int(i)])
        if len(picked) >= n_groups:
            break
    while len(picked) < min(n_groups, len(group_keys)):
        k = group_keys[int(rng.integers(0, len(group_keys)))]
        if k not in picked:
            picked.append(k)

    reports: list[dict[str, Any]] = []
    for key in picked[:n_groups]:
        g = groups.get_group(key).copy()
        g = g.sort_values("target_score", ascending=False).reset_index(drop=True)
        g["actual_rank"] = np.arange(1, len(g) + 1)
        g_pred = g.sort_values(pred_col, ascending=False).reset_index(drop=True)
        pred_rank_map = {
            eid: rank
            for rank, eid in enumerate(g_pred["example_id"].astype(str).tolist(), start=1)
        }
        g["predicted_rank"] = g["example_id"].astype(str).map(pred_rank_map)
        score_std = float(g[pred_col].std(ddof=0))
        rows = []
        for _, r in g.iterrows():
            rows.append(
                {
                    "candidate_limit": float(r["candidate_limit"]),
                    "target_score": float(r["target_score"]),
                    "predicted_ml_score": float(r[pred_col]),
                    "actual_rank": int(r["actual_rank"]),
                    "predicted_rank": int(r["predicted_rank"]),
                }
            )
        reports.append(
            {
                "production_month": key[0],
                "lot_id": key[1],
                "die_id": key[2],
                "parameter": key[3],
                "n_candidates": int(len(g)),
                "pred_score_std": score_std,
                "constant_score": bool(score_std < 1e-12),
                "rows": rows,
            }
        )
    return reports


def _eval_legacy_on_temporal(
    *,
    legacy_ckpt: Path,
    test_ds: CoreCandidateDataset,
    test_df: pd.DataFrame,
    arch: dict[str, Any],
    device: torch.device,
    forward_fn,
) -> dict[str, Any]:
    """Evaluate legacy checkpoint on temporal test if state-dict shapes match."""
    if not legacy_ckpt.is_file():
        return {"compatible": False, "reason": f"legacy checkpoint missing: {legacy_ckpt}"}
    state = torch.load(legacy_ckpt, map_location="cpu", weights_only=False)
    sd = state["model_state"]
    # Infer legacy embedding sizes from weights
    n_param = int(sd["param_emb.weight"].shape[0])
    n_dir = int(sd["dir_emb.weight"].shape[0])
    n_tight = int(sd["tight_emb.weight"].shape[0])
    if (
        n_param != arch["n_parameter"]
        or n_dir != arch["n_direction"]
        or n_tight != arch["n_tight"]
    ):
        return {
            "compatible": False,
            "reason": (
                "Embedding size mismatch between legacy checkpoint and temporal vocabs: "
                f"legacy(n_parameter={n_param}, n_direction={n_dir}, n_tight={n_tight}) vs "
                f"temporal(n_parameter={arch['n_parameter']}, n_direction={arch['n_direction']}, "
                f"n_tight={arch['n_tight']})"
            ),
            "legacy_config": state.get("config"),
            "legacy_best": state.get("best"),
        }
    # Also verify GRU input dim
    if tuple(sd["gru.weight_ih_l0"].shape) != (192, 5):
        return {
            "compatible": False,
            "reason": f"Unexpected GRU weight shape {tuple(sd['gru.weight_ih_l0'].shape)}",
        }

    model = CoreGRURanker(
        n_parameter=n_param,
        n_direction=n_dir,
        n_tight=n_tight,
    )
    model.load_state_dict(sd)
    model.to(device)
    y, p, ids = predict(
        model=model,
        loader=DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0),
        forward_fn=forward_fn,
        device=device,
    )
    pred_df = _rows_with_pred(test_df, ids, p, "pred_legacy_gru")
    metrics = _eval_model(pred_df, "pred_legacy_gru")
    metrics["huber_loss"] = _huber_loss_numpy(y, p, delta=CORE_TRAIN_CONFIG.huber_delta)
    return {
        "compatible": True,
        "reason": (
            "Same CoreGRURanker I/O contract; sequences are month-prefixed at the store "
            "but the model does not consume production_month. Evaluation is informative "
            "but the legacy model was trained on a different population (legacy data/core)."
        ),
        "legacy_checkpoint": str(legacy_ckpt),
        "legacy_best": state.get("best"),
        "test_metrics": metrics,
    }


def run_temporal_core_gru_training(
    project_root: Path | None = None,
) -> TemporalTrainingArtifacts:
    """Retrain CoreGRURanker on the Phase 12.4 temporal pooled dataset."""
    root = project_root or default_project_root()
    t0 = time.perf_counter()
    _seed_all(TRAINING_SEED)

    shared = shared_ml_dataset_root(root)
    legacy_ml = root / "artifacts" / "ml_dataset"
    legacy_ckpt = root / "artifacts" / "ml" / "checkpoints" / "core_gru_best.pt"
    legacy_ckpt_hash_before = file_sha256(legacy_ckpt) if legacy_ckpt.is_file() else None

    # Refuse accidental legacy dataset use
    if shared.resolve() == legacy_ml.resolve():
        raise TemporalTrainingError("Temporal shared path collides with legacy ml_dataset")

    splits = _load_temporal_splits(shared)
    train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]
    for name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        _verify_examples(df, split=name)
    split_info = _verify_split_leakage(train_df, val_df, test_df)

    # Confirm we did not load legacy ml_dataset paths
    for p in [
        shared / "train" / "core_candidate_examples.parquet",
        shared / "sequences" / "core_sequences.parquet",
    ]:
        if "artifacts/ml_dataset" in str(p.resolve()).replace("\\", "/"):
            raise TemporalTrainingError(f"Refusing legacy path: {p}")

    seq_store = CoreSequenceStore(splits["sequences"])
    # Temporal sanity: same lot/die across months are distinct sequence keys
    for lot, die in (("DTL_NORM_001", "DTL_NORM_001_D001"),):
        sids = [
            f"2026-01::{lot}::{die}",
            f"2026-02::{lot}::{die}",
            f"2026-03::{lot}::{die}",
        ]
        mats = [seq_store.get(s) for s in sids]
        for sid, arr in zip(sids, mats):
            if arr.shape != (200, 5):
                raise TemporalTrainingError(f"Bad sequence shape for {sid}: {arr.shape}")
        # Matrices should differ across months for at least one channel (temporal drift)
        if all(np.allclose(mats[0], m) for m in mats[1:]):
            raise TemporalTrainingError(
                "Temporal sequences identical across months for sanity die — unexpected"
            )

    train_ds = CoreCandidateDataset(train_df, seq_store)
    val_ds = CoreCandidateDataset(val_df, seq_store)
    test_ds = CoreCandidateDataset(test_df, seq_store)

    model = CoreGRURanker(
        seq_input_dim=5,
        gru_hidden=64,
        cand_num_dim=4,
        n_parameter=len(train_ds.param_map),
        n_direction=len(train_ds.dir_map),
        n_tight=len(train_ds.tight_map),
        embed_dim=8,
        dropout=0.2,
    )
    architecture = _verify_architecture(model, train_ds)
    print("[Phase12.5] CoreGRURanker architecture verified:", json.dumps(architecture, indent=2), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = temporal_artifact_root(root) / "shared"
    ckpt_dir = out_root / "checkpoints"
    train_dir = out_root / "training"
    for d in (ckpt_dir, train_dir, train_dir / "predictions"):
        d.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / CHECKPOINT_NAME
    if ckpt_path.resolve() == legacy_ckpt.resolve():
        raise TemporalTrainingError("Refusing to write temporal checkpoint over legacy path")

    forward_fn = lambda m, b: m(  # noqa: E731
        sequence=b["sequence"],
        cand_num=b["cand_num"],
        parameter_idx=b["parameter_idx"],
        direction_idx=b["direction_idx"],
        tight_idx=b["tight_idx"],
        cross_domain=b["cross_domain"],
    )

    cfg = CORE_TRAIN_CONFIG
    print(
        f"[Phase12.5] Training start device={device} "
        f"n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)} "
        f"cfg={cfg}",
        flush=True,
    )
    best, history = train_regressor(
        model=model,
        train_loader=DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
        ),
        val_loader=DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        ),
        forward_fn=forward_fn,
        checkpoint_path=ckpt_path,
        config=cfg,
        device=device,
    )
    print(f"[Phase12.5] Training done best={best}", flush=True)

    # Enrich checkpoint with temporal metadata (preserve model_state / best / config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    manifest = json.loads((shared / "dataset_manifest.json").read_text(encoding="utf-8"))
    state["temporal_metadata"] = {
        "phase": "12.5",
        "checkpoint_name": CHECKPOINT_NAME,
        "dataset_version": DATASET_VERSION,
        "dataset_root": str(shared),
        "dataset_manifest_hash": file_sha256(shared / "dataset_manifest.json"),
        "train_parquet_hash": file_sha256(shared / "train" / "core_candidate_examples.parquet"),
        "split_seed": SPLIT_SEED,
        "training_seed": TRAINING_SEED,
        "architecture": architecture,
        "normalization": {
            "sequence": "none (raw measurements — same as Phase 7 Core GRU)",
            "candidate_numeric": "none (raw CORE_CAND_NUM — same as Phase 7)",
            "target": "raw target_score / objective_score (no z-score)",
            "note": (
                "Phase 12.4 wrote train-only z-score stats for cand fields under "
                "normalization/normalization_stats.json for reproducibility, but Phase 7 "
                "CoreGRURanker training does not apply them; Phase 12.5 preserves that behavior."
            ),
            "stats_path": str(shared / "normalization" / "normalization_stats.json"),
        },
        "feature_schema": {
            "sequence_channels": list(SEQUENCE_FEATURE_ORDER),
            "cand_num": list(CORE_CAND_NUM),
            "embeddings": ["parameter", "direction", "tighten_or_loosen"],
            "cross_domain": True,
            "production_month": "metadata_only",
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    torch.save(state, ckpt_path)

    # Predictions
    def _predict_split(ds: CoreCandidateDataset, df: pd.DataFrame, col: str):
        y, p, ids = predict(
            model=model,
            loader=DataLoader(ds, batch_size=512, shuffle=False, num_workers=0),
            forward_fn=forward_fn,
            device=device,
        )
        pred_df = _rows_with_pred(df, ids, p, col)
        metrics = _eval_model(pred_df, col)
        metrics["huber_loss"] = _huber_loss_numpy(y, p, delta=cfg.huber_delta)
        metrics["n_examples"] = int(len(df))
        return pred_df, metrics

    train_pred, train_metrics = _predict_split(train_ds, train_df, "pred_temporal_gru")
    val_pred, val_metrics = _predict_split(val_ds, val_df, "pred_temporal_gru")
    test_pred, test_metrics = _predict_split(test_ds, test_df, "pred_temporal_gru")

    per_month: dict[str, Any] = {}
    for month in ("2026-01", "2026-02", "2026-03"):
        sub = test_pred[test_pred["production_month"].astype(str) == month].copy()
        if sub.empty:
            raise TemporalTrainingError(f"No test examples for month {month}")
        m = _eval_model(sub, "pred_temporal_gru")
        m["huber_loss"] = _huber_loss_numpy(
            sub["target_score"].to_numpy(dtype=float),
            sub["pred_temporal_gru"].to_numpy(dtype=float),
            delta=cfg.huber_delta,
        )
        m["n_examples"] = int(len(sub))
        per_month[month] = m

    ranking_sanity = _ranking_sanity(test_pred, pred_col="pred_temporal_gru")
    if any(r["constant_score"] for r in ranking_sanity):
        # Not necessarily a hard fail, but flag clearly
        pass

    legacy_comparison = _eval_legacy_on_temporal(
        legacy_ckpt=legacy_ckpt,
        test_ds=test_ds,
        test_df=test_df,
        arch=architecture,
        device=device,
        forward_fn=forward_fn,
    )

    metrics: dict[str, Any] = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "test_by_month": per_month,
        "best_checkpoint": {
            "path": str(ckpt_path),
            "best_epoch": best.get("epoch"),
            "best_val_loss": best.get("val_loss"),
        },
        "training_history": history,
        "split": split_info,
        "ranking_sanity": ranking_sanity,
        "legacy_comparison": legacy_comparison,
        "hyperparameters": asdict(cfg),
        "normalization": state["temporal_metadata"]["normalization"],
        "dataset": {
            "root": str(shared),
            "version": DATASET_VERSION,
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "n_test": int(len(test_df)),
        },
    }

    # Persist artifacts
    write_json(train_dir / "training_history.json", {"history": history, "best": best})
    write_json(train_dir / "metrics.json", metrics)
    write_json(train_dir / "architecture.json", architecture)
    write_json(train_dir / "train_config.json", asdict(cfg))
    write_json(
        train_dir / "reproducibility.json",
        {
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "dataset_manifest_hash": state["temporal_metadata"]["dataset_manifest_hash"],
            "train_parquet_hash": state["temporal_metadata"]["train_parquet_hash"],
            "checkpoint_sha256": file_sha256(ckpt_path),
            "legacy_checkpoint_sha256_before": legacy_ckpt_hash_before,
            "legacy_checkpoint_sha256_after": (
                file_sha256(legacy_ckpt) if legacy_ckpt.is_file() else None
            ),
        },
    )
    test_pred.to_parquet(train_dir / "predictions" / "core_test_predictions.parquet", index=False)
    val_pred.to_parquet(train_dir / "predictions" / "core_val_predictions.parquet", index=False)

    legacy_untouched = (
        legacy_ckpt_hash_before is not None
        and legacy_ckpt.is_file()
        and file_sha256(legacy_ckpt) == legacy_ckpt_hash_before
    )
    write_json(
        out_root / "PHASE_12_5_TRAINING_SUMMARY.json",
        {
            "checkpoint": str(ckpt_path),
            "best": best,
            "test_mae": test_metrics.get("mae"),
            "test_ndcg_at_k": test_metrics.get("ndcg_at_k"),
            "legacy_checkpoint_untouched": legacy_untouched,
            "wired_into_recommend": False,
        },
    )

    runtime = time.perf_counter() - t0
    return TemporalTrainingArtifacts(
        checkpoint_path=ckpt_path,
        training_root=train_dir,
        metrics=metrics,
        architecture=architecture,
        best=best,
        history=history,
        legacy_checkpoint_untouched=legacy_untouched,
        runtime_seconds=runtime,
    )
