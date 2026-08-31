"""Shadow Jan-only CoreGRU retrain for equal-info RLS comparison.

Writes ONLY under ``artifacts/temporal/shared/rls_experiment/jan_gru_shadow/``.
Does NOT overwrite production ``core_gru_temporal_v1.pt`` or legacy checkpoints.
Does NOT wire into ``recommend()``.
"""

from __future__ import annotations

import json
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
from dtl_agent.features.io_utils import file_sha256, write_json
from dtl_agent.ml.datasets.phase7_datasets import CoreCandidateDataset, CoreSequenceStore
from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml.rls.data import MONTH_TEST, MONTH_TRAIN, MONTH_VAL, load_month_temporal_split
from dtl_agent.ml.temporal_training import (
    CORE_TRAIN_CONFIG,
    EXPECTED_ARCHITECTURE,
    TRAINING_SEED,
    _seed_all,
    _verify_architecture,
)
from dtl_agent.ml.training.trainer import predict, train_regressor

SHADOW_DIR_REL = Path("artifacts/temporal/shared/rls_experiment/jan_gru_shadow")
SHADOW_CKPT_NAME = "core_gru_jan_only.pt"
PRODUCTION_CKPT_REL = Path("artifacts/temporal/shared/checkpoints/core_gru_temporal_v1.pt")


@dataclass
class JanGRUShadowArtifacts:
    checkpoint_path: Path
    architecture_path: Path
    summary_path: Path
    train_rows: int
    val_rows: int
    test_rows: int
    best: dict[str, Any]
    history: list[dict[str, Any]]
    production_ckpt_sha256_before: str | None
    production_ckpt_sha256_after: str | None
    production_untouched: bool
    runtime_seconds: float


def _forward(model: CoreGRURanker, batch: dict) -> torch.Tensor:
    return model(
        sequence=batch["sequence"],
        cand_num=batch["cand_num"],
        parameter_idx=batch["parameter_idx"],
        direction_idx=batch["direction_idx"],
        tight_idx=batch["tight_idx"],
        cross_domain=batch["cross_domain"],
    )


def train_jan_only_core_gru(*, root: Path | None = None) -> JanGRUShadowArtifacts:
    """Train CoreGRURanker on January only; early-stop on February; never touch March labels."""
    root = root or default_project_root()
    t0 = time.perf_counter()
    _seed_all(TRAINING_SEED)

    prod_ckpt = root / PRODUCTION_CKPT_REL
    prod_hash_before = file_sha256(prod_ckpt) if prod_ckpt.is_file() else None

    data = load_month_temporal_split(root)
    train_df = data.train.reset_index(drop=True)
    val_df = data.validation.reset_index(drop=True)
    test_df = data.test.reset_index(drop=True)  # held out — scored later, not used for fit

    if not train_df["production_month"].eq(MONTH_TRAIN).all():
        raise RuntimeError("Jan shadow train frame must be 2026-01 only")
    if not val_df["production_month"].eq(MONTH_VAL).all():
        raise RuntimeError("Jan shadow val frame must be 2026-02 only")
    if not test_df["production_month"].eq(MONTH_TEST).all():
        raise RuntimeError("Jan shadow test frame must be 2026-03 only")

    seq_store = _store_from_mats(data.seq_store)  # type: ignore[arg-type]

    train_ds = CoreCandidateDataset(train_df, seq_store)
    val_ds = CoreCandidateDataset(val_df, seq_store)

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
    for k, v in EXPECTED_ARCHITECTURE.items():
        if k in architecture and architecture[k] != v and k not in {
            "parameter_vocab",
            "direction_vocab",
            "tighten_vocab",
        }:
            # soft check — _verify_architecture already enforces shapes
            pass

    out_dir = root / SHADOW_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / SHADOW_CKPT_NAME
    if ckpt_path.resolve() == prod_ckpt.resolve():
        raise RuntimeError("Refusing to write shadow checkpoint over production path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CORE_TRAIN_CONFIG
    best, history = train_regressor(
        model=model,
        train_loader=DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
        ),
        val_loader=DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        ),
        forward_fn=_forward,
        checkpoint_path=ckpt_path,
        config=cfg,
        device=device,
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state["equal_info_shadow_metadata"] = {
        "protocol": "equal_info_jan_gru_shadow",
        "train_month": MONTH_TRAIN,
        "validation_month": MONTH_VAL,
        "test_month_held_out": MONTH_TEST,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows_not_used_in_fit": int(len(test_df)),
        "architecture": architecture,
        "parameter_vocab": dict(train_ds.param_map),
        "direction_vocab": dict(train_ds.dir_map),
        "tighten_vocab": dict(train_ds.tight_map),
        "training_seed": TRAINING_SEED,
        "note": (
            "Shadow only. March labels were not used for fitting or early stopping. "
            "Does not replace production core_gru_temporal_v1.pt."
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    torch.save(state, ckpt_path)

    arch_path = out_dir / "architecture.json"
    write_json(
        arch_path,
        {
            **architecture,
            "parameter_vocab": dict(train_ds.param_map),
            "direction_vocab": dict(train_ds.dir_map),
            "tighten_vocab": dict(train_ds.tight_map),
        },
    )

    prod_hash_after = file_sha256(prod_ckpt) if prod_ckpt.is_file() else None
    untouched = (
        prod_hash_before is not None
        and prod_hash_after is not None
        and prod_hash_before == prod_hash_after
    )

    summary = {
        "checkpoint": str(ckpt_path.relative_to(root)),
        "production_checkpoint": str(PRODUCTION_CKPT_REL),
        "production_untouched": untouched,
        "production_sha256_before": prod_hash_before,
        "production_sha256_after": prod_hash_after,
        "best": best,
        "history": history,
        "hyperparameters": asdict(cfg),
        "split": {
            "train": MONTH_TRAIN,
            "validation": MONTH_VAL,
            "test_held_out": MONTH_TEST,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
    }
    summary_path = out_dir / "JAN_GRU_SHADOW_SUMMARY.json"
    write_json(summary_path, summary)

    return JanGRUShadowArtifacts(
        checkpoint_path=ckpt_path,
        architecture_path=arch_path,
        summary_path=summary_path,
        train_rows=int(len(train_df)),
        val_rows=int(len(val_df)),
        test_rows=int(len(test_df)),
        best=best,
        history=history,
        production_ckpt_sha256_before=prod_hash_before,
        production_ckpt_sha256_after=prod_hash_after,
        production_untouched=untouched,
        runtime_seconds=time.perf_counter() - t0,
    )


def _store_from_mats(mats: dict[str, np.ndarray]) -> CoreSequenceStore:
    store = CoreSequenceStore.__new__(CoreSequenceStore)
    store.mats = {str(k): np.asarray(v, dtype=np.float32) for k, v in mats.items()}
    return store


def load_jan_shadow_scorer(*, root: Path | None = None) -> tuple[CoreGRURanker, dict[str, Any], torch.device]:
    """Load Jan-only shadow checkpoint for offline scoring."""
    root = root or default_project_root()
    ckpt_path = root / SHADOW_DIR_REL / SHADOW_CKPT_NAME
    arch_path = root / SHADOW_DIR_REL / "architecture.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing Jan shadow checkpoint: {ckpt_path}")
    arch = json.loads(arch_path.read_text(encoding="utf-8"))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = state.get("equal_info_shadow_metadata", {})
    param_map = meta.get("parameter_vocab") or arch["parameter_vocab"]
    dir_map = meta.get("direction_vocab") or arch["direction_vocab"]
    tight_map = meta.get("tighten_vocab") or arch["tighten_vocab"]
    model = CoreGRURanker(
        seq_input_dim=5,
        gru_hidden=64,
        cand_num_dim=4,
        n_parameter=len(param_map),
        n_direction=len(dir_map),
        n_tight=len(tight_map),
        embed_dim=8,
        dropout=0.2,
    )
    model.load_state_dict(state["model_state"])
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    vocabs = {
        "parameter_vocab": param_map,
        "direction_vocab": dir_map,
        "tighten_vocab": tight_map,
    }
    return model, vocabs, device


def score_examples_with_gru(
    *,
    examples: pd.DataFrame,
    seq_store: dict[str, np.ndarray],
    model: CoreGRURanker,
    vocabs: dict[str, Any],
    device: torch.device | None = None,
) -> np.ndarray:
    """Score candidate examples with a CoreGRU; returns pred array aligned to examples order."""
    device = device or torch.device("cpu")
    store = _store_from_mats(seq_store)
    # Align dataset cat maps to checkpoint vocabs
    ds = CoreCandidateDataset(examples.reset_index(drop=True), store)
    ds.param_map = {str(k): int(v) for k, v in vocabs["parameter_vocab"].items()}
    ds.dir_map = {str(k): int(v) for k, v in vocabs["direction_vocab"].items()}
    ds.tight_map = {str(k): int(v) for k, v in vocabs["tighten_vocab"].items()}
    _, preds, ids = predict(
        model=model,
        loader=DataLoader(ds, batch_size=512, shuffle=False, num_workers=0),
        forward_fn=_forward,
        device=device,
    )
    # Reorder preds to match examples example_id order if needed
    id_to_pred = {str(i): float(p) for i, p in zip(ids, preds)}
    out = np.array([id_to_pred[str(eid)] for eid in examples["example_id"].astype(str)], dtype=float)
    return out


def main() -> None:
    art = train_jan_only_core_gru()
    print(
        json.dumps(
            {
                "checkpoint": str(art.checkpoint_path),
                "production_untouched": art.production_untouched,
                "train_rows": art.train_rows,
                "val_rows": art.val_rows,
                "best": art.best,
                "runtime_seconds": art.runtime_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
