"""Phase 12.5D — unified temporal ML dataset + UnifiedParameterGRURanker training.

Offline experiment only. Does not wire into recommend(). Does not modify CoreGRURanker / MLP.
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
from torch.utils.data import DataLoader, Dataset

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.loader import load_temporal_month
from dtl_agent.data.temporal.parametric_simulation import run_temporal_parametric_simulation
from dtl_agent.data.temporal.paths import (
    month_simulation_root,
    shared_ml_dataset_root,
    temporal_artifact_root,
    validate_production_month,
)
from dtl_agent.features.core_engine import EXPECTED_SEQUENCE_LENGTH, SEQUENCE_FEATURE_ORDER
from dtl_agent.features.io_utils import file_sha256, write_json
from dtl_agent.ml.datasets.phase7_datasets import CORE_CAND_NUM, CoreSequenceStore, _cat_map
from dtl_agent.ml.evaluation.metrics import group_ranking_metrics, mae, rmse
from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml.models.unified_gru_ranker import (
    CORE_SCORE_PARAMETERS,
    PARAMETRIC_CONDITION_ORDER,
    PARAMETRIC_CONTEXT_DIM,
    PARAMETRIC_SCORE_PARAMETERS,
    UNIFIED_PARAMETER_VOCAB,
    UnifiedParameterGRURanker,
)
from dtl_agent.ml.pipeline import _rows_with_pred
from dtl_agent.ml.training.trainer import TrainConfig, predict, train_regressor
from dtl_agent.ml_dataset.pipeline import _det_example_id, _safe_mkdir, _write_parquet
from dtl_agent.ml_dataset.temporal_pipeline import SPLIT_SEED, TEMPORAL_MONTHS

TRAINING_SEED = 7
UNIFIED_TRAIN_CONFIG = TrainConfig(
    lr=8e-4,
    weight_decay=1e-4,
    batch_size=512,
    max_epochs=6,
    patience=2,
    huber_delta=1.0,
    seed=TRAINING_SEED,
)
CHECKPOINT_NAME = "unified_parameter_gru_v1.pt"
FORBIDDEN_INPUT_COLS = frozenset(
    {
        "simulated_yield",
        "violation_rate",
        "borderline_rate",
        "objective_score",
        "target_score",
        "risky_rate",
        "false_fail_proxy",
        "defective_proxy",
    }
)


class UnifiedExperimentError(RuntimeError):
    pass


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _huber(y: np.ndarray, p: np.ndarray, delta: float = 1.0) -> float:
    err = np.abs(y - p)
    quad = np.minimum(err, delta)
    return float(np.mean(0.5 * quad**2 + delta * (err - quad)))


def _eval_pred(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    y = df["target_score"].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    rows = df.to_dict(orient="records")
    rank = group_ranking_metrics(
        rows=rows,
        group_keys=["lot_id", "die_id", "parameter", "production_month"],
        score_key="target_score",
        pred_key=pred_col,
        k=5,
    )
    return {"mae": mae(y, p), "rmse": rmse(y, p), "huber_loss": _huber(y, p), **rank, "n_examples": int(len(df))}


def build_parametric_context_table(month_data) -> pd.DataFrame:
    """One row per lot×die×parameter with 4 condition values + masks (no tiling)."""
    df = month_data.parametric.copy()
    rows = []
    for (lot, die, param), g in df.groupby(["lot_id", "die_id", "parameter"], sort=False):
        by_cond = {
            str(r.condition_id): float(r.measurement_value)
            for r in g.itertuples(index=False)
        }
        rec: dict[str, Any] = {
            "lot_id": str(lot),
            "die_id": str(die),
            "parameter": str(param),
            "production_month": month_data.production_month,
        }
        vals = []
        masks = []
        for cond in PARAMETRIC_CONDITION_ORDER:
            if cond in by_cond:
                vals.append(by_cond[cond])
                masks.append(1.0)
            else:
                vals.append(0.0)
                masks.append(0.0)
        for i, cond in enumerate(PARAMETRIC_CONDITION_ORDER):
            rec[f"ctx_val_{i}"] = vals[i]
            rec[f"ctx_mask_{i}"] = masks[i]
        rec["has_parametric_context"] = 1.0 if sum(masks) > 0 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def empty_parametric_context_row() -> dict[str, float]:
    rec = {f"ctx_val_{i}": 0.0 for i in range(4)}
    rec.update({f"ctx_mask_{i}": 0.0 for i in range(4)})
    rec["has_parametric_context"] = 0.0
    return rec


def ensure_temporal_parametric_sims(project_root: Path, *, force: bool = False) -> dict[str, Path]:
    out = {}
    for m in TEMPORAL_MONTHS:
        path = month_simulation_root(m, project_root) / "parametric" / "candidate_results.csv"
        if force or not path.is_file():
            data = load_temporal_month(m, project_root=project_root)
            run_temporal_parametric_simulation(m, project_root=project_root, month_data=data)
        out[m] = path
    return out


def _load_split_map(project_root: Path) -> dict[str, str]:
    split_path = shared_ml_dataset_root(project_root) / "split_manifest.json"
    if not split_path.is_file():
        raise UnifiedExperimentError(f"Missing Phase 12.4 split_manifest: {split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data["lot_to_split"].items()}


def assemble_unified_ml_dataset(
    project_root: Path | None = None,
    *,
    force_param_sim: bool = False,
) -> Path:
    """Build artifacts/temporal/shared/unified_ml_dataset/ from Core + Parametric month sims."""
    root = project_root or default_project_root()
    out_root = temporal_artifact_root(root) / "shared" / "unified_ml_dataset"
    for sub in ("train", "validation", "test", "sequences", "normalization"):
        _safe_mkdir(out_root / sub)

    split_map = _load_split_map(root)
    ensure_temporal_parametric_sims(root, force=force_param_sim)

    # Reuse shared core sequences
    seq_src = shared_ml_dataset_root(root) / "sequences" / "core_sequences.parquet"
    if not seq_src.is_file():
        raise UnifiedExperimentError("Missing temporal shared core sequences")
    sequences = pd.read_parquet(seq_src)
    _write_parquet(sequences, out_root / "sequences" / "core_sequences.parquet")

    example_frames: list[pd.DataFrame] = []
    for month in TEMPORAL_MONTHS:
        month = validate_production_month(month)
        data = load_temporal_month(month, project_root=root)
        ctx_table = build_parametric_context_table(data)

        # --- Core examples from existing shared month examples or rebuild from sim ---
        core_cands = pd.read_csv(month_simulation_root(month, root) / "core" / "candidate_results.csv")
        dies = (
            data.actual_die[["lot_id", "die_id"]]
            .drop_duplicates()
            .assign(lot_id=lambda d: d["lot_id"].astype(str), die_id=lambda d: d["die_id"].astype(str))
        )
        dies["production_month"] = month
        dies["sequence_id"] = dies.apply(
            lambda r: make_sequence_id(r["lot_id"], r["die_id"], month), axis=1
        )
        dies["split"] = dies["lot_id"].map(split_map)
        dies = dies[dies["split"].notna()].copy()

        cand_cols = [
            "parameter",
            "test_id",
            "candidate_limit",
            "current_limit",
            "direction",
            "unit",
            "candidate_delta",
            "candidate_delta_percent",
            "tighten_or_loosen",
            "objective_score",
        ]
        core_cands = core_cands[core_cands["parameter"].isin(CORE_SCORE_PARAMETERS)][cand_cols].copy()
        dies_k = dies.assign(_k=1)
        core_k = core_cands.assign(_k=1)
        core_ex = dies_k.merge(core_k, on="_k", how="inner").drop(columns=["_k"])
        core_ex["domain"] = "core"
        core_ex["target_score"] = core_ex["objective_score"].astype(float)
        empty = empty_parametric_context_row()
        for k, v in empty.items():
            core_ex[k] = v
        example_frames.append(core_ex)

        # --- Parametric examples ---
        param_cands = pd.read_csv(
            month_simulation_root(month, root) / "parametric" / "candidate_results.csv"
        )
        param_cands = param_cands[param_cands["parameter"].isin(PARAMETRIC_SCORE_PARAMETERS)][
            cand_cols
        ].copy()
        param_k = param_cands.assign(_k=1)
        param_ex = dies_k.merge(param_k, on="_k", how="inner").drop(columns=["_k"])
        param_ex["domain"] = "parametric"
        param_ex["target_score"] = param_ex["objective_score"].astype(float)
        ctx_cols = (
            ["lot_id", "die_id", "parameter"]
            + [f"ctx_val_{i}" for i in range(4)]
            + [f"ctx_mask_{i}" for i in range(4)]
            + ["has_parametric_context"]
        )
        param_ex = param_ex.merge(ctx_table[ctx_cols], on=["lot_id", "die_id", "parameter"], how="left")
        for i in range(4):
            param_ex[f"ctx_val_{i}"] = param_ex[f"ctx_val_{i}"].fillna(0.0)
            param_ex[f"ctx_mask_{i}"] = param_ex[f"ctx_mask_{i}"].fillna(0.0)
        param_ex["has_parametric_context"] = param_ex["has_parametric_context"].fillna(0.0)
        example_frames.append(param_ex)

    pooled = pd.concat(example_frames, ignore_index=True)
    id_parts = (
        pooled["production_month"].astype(str)
        + "|"
        + pooled["split"].astype(str)
        + "|"
        + pooled["lot_id"].astype(str)
        + "|"
        + pooled["die_id"].astype(str)
        + "|"
        + pooled["parameter"].astype(str)
        + "|"
        + pooled["candidate_limit"].astype(str)
    )
    pooled["example_id"] = id_parts.map(lambda s: _det_example_id(s.split("|")))

    # Quality
    if pooled["objective_score"].isna().any() or pooled["target_score"].isna().any():
        raise UnifiedExperimentError("Missing objective/target scores")
    if not np.allclose(
        pooled["target_score"].astype(float), pooled["objective_score"].astype(float)
    ):
        raise UnifiedExperimentError("target_score must equal objective_score")
    dup = pooled.duplicated(
        subset=["production_month", "lot_id", "die_id", "parameter", "candidate_limit"],
        keep=False,
    )
    if dup.any():
        raise UnifiedExperimentError(f"Duplicate unified examples: {int(dup.sum())}")
    bad_sid = ~pooled["sequence_id"].astype(str).map(lambda s: s.count("::") == 2)
    if bad_sid.any():
        raise UnifiedExperimentError("sequence_id must be month::lot::die")

    # Leakage: ensure forbidden columns are not used as features later (kept for audit only)
    feature_cols = [
        "sequence_id",
        "parameter",
        "direction",
        "tighten_or_loosen",
        *CORE_CAND_NUM,
        *[f"ctx_val_{i}" for i in range(4)],
        *[f"ctx_mask_{i}" for i in range(4)],
        "has_parametric_context",
    ]
    if any(c in FORBIDDEN_INPUT_COLS for c in feature_cols):
        raise UnifiedExperimentError("Forbidden input column in feature set")

    # Normalization: per-parameter cand + context z-score on TRAIN only
    train = pooled[pooled["split"] == "train"]
    norm: dict[str, Any] = {"method": "zscore_train_only_per_parameter", "parameters": {}}
    numeric_ctx = [f"ctx_val_{i}" for i in range(4)]
    for param in sorted(pooled["parameter"].unique()):
        sub = train[train["parameter"] == param]
        feats = {}
        for c in list(CORE_CAND_NUM) + numeric_ctx:
            s = pd.to_numeric(sub[c], errors="coerce")
            mu = float(s.mean()) if s.notna().any() else 0.0
            sd = float(s.std(ddof=0)) if s.notna().any() else 1.0
            if abs(sd) < 1e-12:
                sd = 1.0
            feats[c] = {"mean": mu, "std": sd}
        norm["parameters"][str(param)] = feats

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for param, feats in norm["parameters"].items():
            mask = out["parameter"].astype(str) == param
            for c, st in feats.items():
                ncol = f"norm_{c}"
                if ncol not in out.columns:
                    out[ncol] = np.nan
                out.loc[mask, ncol] = (
                    pd.to_numeric(out.loc[mask, c], errors="coerce") - st["mean"]
                ) / st["std"]
        # fill any missing norm with raw
        for c in list(CORE_CAND_NUM) + numeric_ctx:
            ncol = f"norm_{c}"
            out[ncol] = out[ncol].fillna(pd.to_numeric(out[c], errors="coerce"))
        return out

    pooled = _apply(pooled)
    write_json(out_root / "normalization" / "normalization_stats.json", norm)

    for split in ("train", "validation", "test"):
        _write_parquet(
            pooled[pooled["split"] == split].sort_values("example_id"),
            out_root / split / "unified_candidate_examples.parquet",
        )

    manifest = {
        "dataset_name": "dtl_agent_unified_parameter_gru_dataset",
        "version": "phase12_5d_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "total_examples": int(len(pooled)),
        "train_examples": int((pooled["split"] == "train").sum()),
        "validation_examples": int((pooled["split"] == "validation").sum()),
        "test_examples": int((pooled["split"] == "test").sum()),
        "parameters": sorted(pooled["parameter"].astype(str).unique().tolist()),
        "split_seed": SPLIT_SEED,
        "parametric_context": {
            "condition_order": list(PARAMETRIC_CONDITION_ORDER),
            "dim": PARAMETRIC_CONTEXT_DIM,
            "construction": "4 condition values + 4 masks for scored parametric parameter; zeros/mask0 for Core",
        },
        "forbidden_input_features": sorted(FORBIDDEN_INPUT_COLS),
        "counts_by_parameter": {
            p: int((pooled["parameter"] == p).sum()) for p in sorted(pooled["parameter"].unique())
        },
        "counts_by_month": {
            m: int((pooled["production_month"] == m).sum()) for m in TEMPORAL_MONTHS
        },
    }
    write_json(out_root / "dataset_manifest.json", manifest)
    write_json(
        out_root / "split_manifest.json",
        json.loads((shared_ml_dataset_root(root) / "split_manifest.json").read_text(encoding="utf-8")),
    )
    return out_root


class UnifiedCandidateDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        seq_store: CoreSequenceStore,
        *,
        use_norm: bool = True,
        dir_map: dict[str, int] | None = None,
        tight_map: dict[str, int] | None = None,
    ) -> None:
        self.df = rows.reset_index(drop=True).copy()
        self.seq_store = seq_store
        self.use_norm = use_norm
        # Fixed global vocab (unknown → KeyError)
        self.param_map = {p: i for i, p in enumerate(UNIFIED_PARAMETER_VOCAB)}
        self.dir_map = dir_map or _cat_map(self.df["direction"].astype(str).tolist())
        self.tight_map = tight_map or _cat_map(self.df["tighten_or_loosen"].astype(str).tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        r = self.df.iloc[idx]
        param = str(r["parameter"])
        if param not in self.param_map:
            raise KeyError(f"Unknown parameter {param!r}; vocab={list(self.param_map)}")
        seq = self.seq_store.get(str(r["sequence_id"]))
        if self.use_norm:
            cand = np.array([float(r[f"norm_{c}"]) for c in CORE_CAND_NUM], dtype=np.float32)
            ctx_vals = [float(r[f"norm_ctx_val_{i}"]) for i in range(4)]
        else:
            cand = np.array([float(r[c]) for c in CORE_CAND_NUM], dtype=np.float32)
            ctx_vals = [float(r[f"ctx_val_{i}"]) for i in range(4)]
        masks = [float(r[f"ctx_mask_{i}"]) for i in range(4)]
        # zero-out missing condition values after norm
        ctx = np.array(
            [ctx_vals[i] * masks[i] for i in range(4)] + masks,
            dtype=np.float32,
        )
        return {
            "sequence": torch.from_numpy(np.array(seq, copy=True)),
            "cand_num": torch.from_numpy(cand),
            "parametric_context": torch.from_numpy(ctx),
            "has_parametric_context": torch.tensor(
                float(r["has_parametric_context"]), dtype=torch.float32
            ),
            "parameter_idx": torch.tensor(self.param_map[param], dtype=torch.long),
            "direction_idx": torch.tensor(self.dir_map[str(r["direction"])], dtype=torch.long),
            "tight_idx": torch.tensor(self.tight_map[str(r["tighten_or_loosen"])], dtype=torch.long),
            "target": torch.tensor(float(r["target_score"]), dtype=torch.float32),
            "example_id": str(r["example_id"]),
        }


@dataclass
class UnifiedExperimentArtifacts:
    dataset_root: Path
    checkpoint_path: Path
    metrics: dict[str, Any]
    runtime_seconds: float
    legacy_checkpoints_untouched: bool


def run_unified_parameter_gru_experiment(
    project_root: Path | None = None,
    *,
    force_rebuild_dataset: bool = False,
    force_param_sim: bool = False,
) -> UnifiedExperimentArtifacts:
    root = project_root or default_project_root()
    t0 = time.perf_counter()
    _seed_all(TRAINING_SEED)

    legacy_core = root / "artifacts" / "ml" / "checkpoints" / "core_gru_best.pt"
    temporal_core = (
        temporal_artifact_root(root) / "shared" / "checkpoints" / "core_gru_temporal_v1.pt"
    )
    h_legacy = file_sha256(legacy_core) if legacy_core.is_file() else None
    h_temporal = file_sha256(temporal_core) if temporal_core.is_file() else None

    ds_root = temporal_artifact_root(root) / "shared" / "unified_ml_dataset"
    train_pq = ds_root / "train" / "unified_candidate_examples.parquet"
    if force_rebuild_dataset or not train_pq.is_file():
        ds_root = assemble_unified_ml_dataset(root, force_param_sim=force_param_sim)

    train_df = pd.read_parquet(ds_root / "train" / "unified_candidate_examples.parquet")
    val_df = pd.read_parquet(ds_root / "validation" / "unified_candidate_examples.parquet")
    test_df = pd.read_parquet(ds_root / "test" / "unified_candidate_examples.parquet")
    seq_store = CoreSequenceStore(pd.read_parquet(ds_root / "sequences" / "core_sequences.parquet"))

    dir_map = _cat_map(
        train_df["direction"].astype(str).tolist()
        + val_df["direction"].astype(str).tolist()
        + test_df["direction"].astype(str).tolist()
    )
    tight_map = _cat_map(
        train_df["tighten_or_loosen"].astype(str).tolist()
        + val_df["tighten_or_loosen"].astype(str).tolist()
        + test_df["tighten_or_loosen"].astype(str).tolist()
    )
    train_ds = UnifiedCandidateDataset(
        train_df, seq_store, use_norm=True, dir_map=dir_map, tight_map=tight_map
    )
    val_ds = UnifiedCandidateDataset(
        val_df, seq_store, use_norm=True, dir_map=dir_map, tight_map=tight_map
    )
    test_ds = UnifiedCandidateDataset(
        test_df, seq_store, use_norm=True, dir_map=dir_map, tight_map=tight_map
    )

    model = UnifiedParameterGRURanker(
        n_parameter=len(UNIFIED_PARAMETER_VOCAB),
        n_direction=len(dir_map),
        n_tight=len(tight_map),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = temporal_artifact_root(root) / "shared" / "checkpoints"
    train_dir = temporal_artifact_root(root) / "shared" / "unified_training"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "predictions").mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / CHECKPOINT_NAME

    forward_fn = lambda m, b: m(  # noqa: E731
        sequence=b["sequence"],
        cand_num=b["cand_num"],
        parametric_context=b["parametric_context"],
        has_parametric_context=b["has_parametric_context"],
        parameter_idx=b["parameter_idx"],
        direction_idx=b["direction_idx"],
        tight_idx=b["tight_idx"],
    )

    print(
        f"[Phase12.5D] train n={len(train_df)} val={len(val_df)} test={len(test_df)} device={device}",
        flush=True,
    )
    best, history = train_regressor(
        model=model,
        train_loader=DataLoader(train_ds, batch_size=UNIFIED_TRAIN_CONFIG.batch_size, shuffle=True),
        val_loader=DataLoader(val_ds, batch_size=UNIFIED_TRAIN_CONFIG.batch_size, shuffle=False),
        forward_fn=forward_fn,
        checkpoint_path=ckpt_path,
        config=UNIFIED_TRAIN_CONFIG,
        device=device,
    )
    print(f"[Phase12.5D] best={best}", flush=True)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state["unified_metadata"] = {
        "phase": "12.5D",
        "parameter_vocab": list(UNIFIED_PARAMETER_VOCAB),
        "direction_vocab": dir_map,
        "tighten_vocab": tight_map,
        "condition_order": list(PARAMETRIC_CONDITION_ORDER),
        "parametric_context_dim": PARAMETRIC_CONTEXT_DIM,
        "embed_dim": 8,
        "forbidden_inputs": sorted(FORBIDDEN_INPUT_COLS),
        "dataset_root": str(ds_root),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    torch.save(state, ckpt_path)
    write_json(
        train_dir / "architecture.json",
        {
            "model_class": "UnifiedParameterGRURanker",
            "seq_input_dim": 5,
            "sequence_length": EXPECTED_SEQUENCE_LENGTH,
            "gru_hidden": 64,
            "cand_num_dim": 4,
            "parametric_context_dim": PARAMETRIC_CONTEXT_DIM,
            "embed_dim": 8,
            "parameter_vocab": {p: i for i, p in enumerate(UNIFIED_PARAMETER_VOCAB)},
            "direction_vocab": dir_map,
            "tighten_vocab": tight_map,
            "condition_order": list(PARAMETRIC_CONDITION_ORDER),
            "sequence_channels": list(SEQUENCE_FEATURE_ORDER),
            "production_month_is_gru_feature": False,
            "target": "target_score (= objective_score)",
            "wired_into_recommend": False,
        },
    )

    def _score(ds, df):
        y, p, ids = predict(
            model=model,
            loader=DataLoader(ds, batch_size=512, shuffle=False),
            forward_fn=forward_fn,
            device=device,
        )
        pred_df = _rows_with_pred(df, ids, p, "pred_unified")
        return pred_df, _eval_pred(pred_df, "pred_unified")

    train_pred, train_m = _score(train_ds, train_df)
    val_pred, val_m = _score(val_ds, val_df)
    test_pred, test_m = _score(test_ds, test_df)

    by_month = {
        m: _eval_pred(test_pred[test_pred["production_month"] == m], "pred_unified")
        for m in TEMPORAL_MONTHS
    }
    by_param = {
        p: _eval_pred(test_pred[test_pred["parameter"] == p], "pred_unified")
        for p in sorted(test_pred["parameter"].unique())
    }

    # Ranking sanity (non-constant)
    sanity = []
    gkeys = ["production_month", "lot_id", "die_id", "parameter"]
    groups = list(test_pred.groupby(gkeys, sort=False))
    rng = np.random.default_rng(TRAINING_SEED)
    for gi in rng.choice(len(groups), size=min(6, len(groups)), replace=False):
        key, g = groups[int(gi)]
        g = g.sort_values("target_score", ascending=False).copy()
        g["actual_rank"] = np.arange(1, len(g) + 1)
        gp = g.sort_values("pred_unified", ascending=False)
        pr = {eid: i + 1 for i, eid in enumerate(gp["example_id"].astype(str))}
        g["predicted_rank"] = g["example_id"].astype(str).map(pr)
        sanity.append(
            {
                "key": dict(zip(gkeys, key)),
                "pred_std": float(g["pred_unified"].std(ddof=0)),
                "constant": bool(g["pred_unified"].std(ddof=0) < 1e-12),
                "rows": g[
                    ["candidate_limit", "target_score", "pred_unified", "actual_rank", "predicted_rank"]
                ]
                .head(8)
                .to_dict(orient="records"),
            }
        )

    # Temporal same-die response
    temporal_ex = []
    lot, die, param = "DTL_NORM_001", "DTL_NORM_001_D001", "ir_drop"
    for month in TEMPORAL_MONTHS:
        sub = test_pred[
            (test_pred["production_month"] == month)
            & (test_pred["lot_id"] == lot)
            & (test_pred["die_id"] == die)
            & (test_pred["parameter"] == param)
        ].sort_values("pred_unified", ascending=False)
        if sub.empty:
            # may be train lot — try EDGE test lot
            continue
        temporal_ex.append(
            {
                "sequence_id": f"{month}::{lot}::{die}",
                "parameter": param,
                "top3": sub.head(3)[["candidate_limit", "pred_unified", "target_score"]].to_dict(
                    orient="records"
                ),
            }
        )
    if not temporal_ex:
        # pick a test lot from split
        split = json.loads((ds_root / "split_manifest.json").read_text(encoding="utf-8"))
        test_lot = next(l for l, s in split["lot_to_split"].items() if s == "test")
        die2 = f"{test_lot}_D001"
        for month in TEMPORAL_MONTHS:
            sub = test_pred[
                (test_pred["production_month"] == month)
                & (test_pred["lot_id"] == test_lot)
                & (test_pred["die_id"] == die2)
                & (test_pred["parameter"] == "ir_drop")
            ].sort_values("pred_unified", ascending=False)
            if sub.empty:
                continue
            temporal_ex.append(
                {
                    "sequence_id": f"{month}::{test_lot}::{die2}",
                    "parameter": "ir_drop",
                    "top3": sub.head(3)[["candidate_limit", "pred_unified", "target_score"]].to_dict(
                        orient="records"
                    ),
                }
            )

    # Baseline comparison on same test lots
    baselines = _baseline_compare(root, test_pred, seq_store, device)

    legacy_ok = True
    if h_legacy and legacy_core.is_file():
        legacy_ok = legacy_ok and file_sha256(legacy_core) == h_legacy
    if h_temporal and temporal_core.is_file():
        legacy_ok = legacy_ok and file_sha256(temporal_core) == h_temporal

    metrics = {
        "train": train_m,
        "validation": val_m,
        "test": test_m,
        "test_by_month": by_month,
        "test_by_parameter": by_param,
        "best": best,
        "history": history,
        "ranking_sanity": sanity,
        "temporal_examples": temporal_ex,
        "baselines": baselines,
        "hyperparameters": asdict(UNIFIED_TRAIN_CONFIG),
        "parameter_vocab": list(UNIFIED_PARAMETER_VOCAB),
        "wired_into_recommend": False,
    }
    write_json(train_dir / "metrics.json", metrics)
    write_json(train_dir / "training_history.json", {"history": history, "best": best})
    write_json(train_dir / "train_config.json", asdict(UNIFIED_TRAIN_CONFIG))
    test_pred.to_parquet(train_dir / "predictions" / "unified_test_predictions.parquet", index=False)
    write_json(
        temporal_artifact_root(root) / "shared" / "PHASE_12_5D_EXPERIMENT_SUMMARY.json",
        {
            "checkpoint": str(ckpt_path),
            "test_mae": test_m.get("mae"),
            "test_ndcg_at_k": test_m.get("ndcg_at_k"),
            "legacy_checkpoints_untouched": legacy_ok,
            "wired_into_recommend": False,
        },
    )

    return UnifiedExperimentArtifacts(
        dataset_root=ds_root,
        checkpoint_path=ckpt_path,
        metrics=metrics,
        runtime_seconds=time.perf_counter() - t0,
        legacy_checkpoints_untouched=legacy_ok,
    )


def _baseline_compare(
    root: Path,
    unified_test: pd.DataFrame,
    seq_store: CoreSequenceStore,
    device: torch.device,
) -> dict[str, Any]:
    """Compare unified vs temporal CoreGRU on IR/Thermal; document MLP non-comparability."""
    out: dict[str, Any] = {}
    core_ckpt = temporal_artifact_root(root) / "shared" / "checkpoints" / "core_gru_temporal_v1.pt"
    arch_path = temporal_artifact_root(root) / "shared" / "training" / "architecture.json"
    core_sub = unified_test[unified_test["parameter"].isin(CORE_SCORE_PARAMETERS)].copy()

    if core_ckpt.is_file() and arch_path.is_file() and not core_sub.empty:
        try:
            arch = json.loads(arch_path.read_text(encoding="utf-8"))
            state = torch.load(core_ckpt, map_location="cpu", weights_only=False)
            m = CoreGRURanker(
                n_parameter=int(state["model_state"]["param_emb.weight"].shape[0]),
                n_direction=int(state["model_state"]["dir_emb.weight"].shape[0]),
                n_tight=int(state["model_state"]["tight_emb.weight"].shape[0]),
            )
            m.load_state_dict(state["model_state"])
            m.eval().to(device)
            vparam = arch["parameter_vocab"]
            vdir = arch["direction_vocab"]
            vtight = arch["tighten_vocab"]
            scores: list[float] = []
            bs = 256
            with torch.no_grad():
                rows = list(core_sub.itertuples(index=False))
                for i in range(0, len(rows), bs):
                    chunk = rows[i : i + bs]
                    seqs = torch.stack(
                        [torch.from_numpy(np.array(seq_store.get(str(r.sequence_id)), copy=True)) for r in chunk]
                    ).to(device)
                    cand = torch.tensor(
                        [[float(getattr(r, c)) for c in CORE_CAND_NUM] for r in chunk],
                        dtype=torch.float32,
                        device=device,
                    )
                    pidx = torch.tensor(
                        [vparam[str(r.parameter)] for r in chunk], dtype=torch.long, device=device
                    )
                    didx = torch.tensor(
                        [vdir[str(r.direction)] for r in chunk], dtype=torch.long, device=device
                    )
                    tidx = torch.tensor(
                        [vtight[str(r.tighten_or_loosen)] for r in chunk],
                        dtype=torch.long,
                        device=device,
                    )
                    pred = m(
                        sequence=seqs,
                        cand_num=cand,
                        parameter_idx=pidx,
                        direction_idx=didx,
                        tight_idx=tidx,
                        cross_domain=torch.zeros(len(chunk), device=device),
                    )
                    scores.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
            core_sub["pred_core_temporal"] = scores
            out["core_ir_thermal"] = {
                "unified": _eval_pred(core_sub, "pred_unified"),
                "core_gru_temporal": _eval_pred(core_sub, "pred_core_temporal"),
                "n_examples": int(len(core_sub)),
            }
            for p in sorted(CORE_SCORE_PARAMETERS):
                subp = core_sub[core_sub["parameter"] == p]
                if subp.empty:
                    continue
                out[f"core_{p}"] = {
                    "unified": _eval_pred(subp, "pred_unified"),
                    "core_gru_temporal": _eval_pred(subp, "pred_core_temporal"),
                }
        except Exception as exc:  # noqa: BLE001
            out["core_ir_thermal"] = {"error": f"{type(exc).__name__}:{exc}"}

    param_sub = unified_test[unified_test["parameter"].isin(PARAMETRIC_SCORE_PARAMETERS)].copy()
    out["parametric"] = {
        "unified": _eval_pred(param_sub, "pred_unified") if not param_sub.empty else {},
        "mlp_comparable": False,
        "note": (
            "ParametricMLPRanker expects Phase-6 condition-grain rows + norm_param_* features from "
            "legacy artifacts/ml_dataset; not contract-compatible with temporal unified die×candidate "
            "rows without rebuilding MLP inputs. Unified parametric metrics reported; MLP remains "
            "the production parametric scorer."
        ),
    }
    return out


if __name__ == "__main__":
    arts = run_unified_parameter_gru_experiment(force_rebuild_dataset=True)
    print(
        {
            "checkpoint": str(arts.checkpoint_path),
            "dataset": str(arts.dataset_root),
            "runtime_s": round(arts.runtime_seconds, 1),
            "legacy_ok": arts.legacy_checkpoints_untouched,
            "test": arts.metrics.get("test"),
        }
    )
