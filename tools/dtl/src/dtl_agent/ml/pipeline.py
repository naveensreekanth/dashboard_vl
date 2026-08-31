"""Phase 7 GRU-based candidate ranker implementation + training."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from dtl_agent.config.paths import default_project_root
from dtl_agent.ml.datasets.phase7_datasets import (
    CORE_SEQ_FEATURES,
    CoreCandidateDataset,
    CoreSequenceStore,
    ParametricTabularDataset,
    load_phase6_ml_splits,
)
from dtl_agent.ml.evaluation.metrics import (
    group_ranking_metrics,
    mae,
    rmse,
)
from dtl_agent.ml.models.fusion import JointRanker
from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml.models.parametric_encoder import ParametricMLPRanker
from dtl_agent.ml.training.trainer import TrainConfig, predict, train_regressor


def _seed_all(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Phase7Artifacts:
    output_root: Path
    metrics: dict[str, Any]
    best_checkpoints: dict[str, str]
    runtime_seconds: float


class JointDataset(Dataset):
    def __init__(
        self,
        *,
        core_rows: pd.DataFrame,
        param_rows: pd.DataFrame,
        seq_store: CoreSequenceStore,
        core_vocab: dict[str, dict[str, int]],
        param_vocab: dict[str, dict[str, int]],
        param_norm_cols: list[str],
    ) -> None:
        self.seq_store = seq_store
        self.core_vocab = core_vocab
        self.param_vocab = param_vocab
        self.param_norm_cols = param_norm_cols

        c = core_rows.copy()
        c["domain_flag"] = "core"
        p = param_rows.copy()
        p["domain_flag"] = "parametric"
        self.df = pd.concat([c, p], ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        r = self.df.iloc[idx]
        is_core = str(r["domain_flag"]) == "core"
        cand = np.array(
            [float(r["candidate_limit"]), float(r["current_limit"]), float(r["candidate_delta"]), float(r["candidate_delta_percent"])],
            dtype=np.float32,
        )
        if is_core:
            seq = self.seq_store.get(str(r["sequence_id"]))
            norm_num = np.zeros(len(self.param_norm_cols), dtype=np.float32)
            cond_num = np.zeros(2, dtype=np.float32)
            parameter_idx = self.core_vocab["parameter"][str(r["parameter"])]
            direction_idx = self.core_vocab["direction"][str(r["direction"])]
            tight_idx = self.core_vocab["tight"][str(r["tighten_or_loosen"])]
            condition_idx = 0
            mode_idx = 0
            cross_domain = float(bool(r.get("cross_domain_available", False)))
            has_core = 1.0
        else:
            seq = np.zeros((200, 5), dtype=np.float32)
            norm_num = np.array([float(r[c]) for c in self.param_norm_cols], dtype=np.float32)
            cond_num = np.array([float(r["temperature_c"]), float(r["vdd_applied"])], dtype=np.float32)
            parameter_idx = self.param_vocab["parameter"][str(r["parameter"])]
            direction_idx = self.param_vocab["direction"][str(r["direction"])]
            tight_idx = self.param_vocab["tight"][str(r["tighten_or_loosen"])]
            condition_idx = self.param_vocab["condition"][str(r["condition_id"])]
            mode_idx = self.param_vocab["mode"][str(r["test_mode"])]
            cross_domain = 0.0
            has_core = 0.0
        return {
            "sequence": torch.from_numpy(np.array(seq, copy=True)),
            "cand_num": torch.from_numpy(cand),
            "norm_num": torch.from_numpy(norm_num),
            "cond_num": torch.from_numpy(cond_num),
            "parameter_idx": torch.tensor(parameter_idx, dtype=torch.long),
            "direction_idx": torch.tensor(direction_idx, dtype=torch.long),
            "tight_idx": torch.tensor(tight_idx, dtype=torch.long),
            "condition_idx": torch.tensor(condition_idx, dtype=torch.long),
            "mode_idx": torch.tensor(mode_idx, dtype=torch.long),
            "cross_domain": torch.tensor(cross_domain, dtype=torch.float32),
            "has_core": torch.tensor(has_core, dtype=torch.float32),
            "target": torch.tensor(float(r["target_score"]), dtype=torch.float32),
            "example_id": str(r["example_id"]),
        }


def _rows_with_pred(df: pd.DataFrame, ids: list[str], preds: np.ndarray, pred_col: str) -> pd.DataFrame:
    pmap = dict(zip(ids, preds.tolist()))
    out = df.copy()
    out[pred_col] = out["example_id"].map(pmap).astype(float)
    return out


def _group_keys_for_df(df: pd.DataFrame) -> list[str]:
    if "condition_id" in df.columns:
        return ["lot_id", "die_id", "parameter", "condition_id"]
    return ["lot_id", "die_id", "parameter"]


def _eval_model(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    y = df["target_score"].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    base = {"mae": mae(y, p), "rmse": rmse(y, p)}
    rows = df.to_dict(orient="records")
    rank = group_ranking_metrics(
        rows=rows,
        group_keys=_group_keys_for_df(df),
        score_key="target_score",
        pred_key=pred_col,
        k=5,
    )
    return base | rank


def run_phase7_training(project_root: Path | None = None) -> Phase7Artifacts:
    _seed_all(7)
    root = project_root or default_project_root()
    ml_root = root / "artifacts" / "ml"
    for p in ["checkpoints", "metrics", "predictions", "reports"]:
        (ml_root / p).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ds = load_phase6_ml_splits(root / "artifacts" / "ml_dataset")
    core_tr, core_va, core_te = ds["core_train"], ds["core_validation"], ds["core_test"]
    par_tr, par_va, par_te = ds["param_train"], ds["param_validation"], ds["param_test"]
    seq_store = CoreSequenceStore(ds["core_sequences"])

    print("[Phase7] Stage 1 Core GRU start", flush=True)
    # --- Stage 1: Core GRU ---
    core_train_ds = CoreCandidateDataset(core_tr, seq_store)
    core_val_ds = CoreCandidateDataset(core_va, seq_store)
    core_test_ds = CoreCandidateDataset(core_te, seq_store)
    core_model = CoreGRURanker(
        n_parameter=len(core_train_ds.param_map),
        n_direction=len(core_train_ds.dir_map),
        n_tight=len(core_train_ds.tight_map),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_core = TrainConfig(lr=8e-4, weight_decay=1e-4, batch_size=512, max_epochs=6, patience=2)
    core_ckpt = ml_root / "checkpoints" / "core_gru_best.pt"
    core_best, core_hist = train_regressor(
        model=core_model,
        train_loader=DataLoader(core_train_ds, batch_size=cfg_core.batch_size, shuffle=True, num_workers=0),
        val_loader=DataLoader(core_val_ds, batch_size=cfg_core.batch_size, shuffle=False, num_workers=0),
        forward_fn=lambda m, b: m(
            sequence=b["sequence"],
            cand_num=b["cand_num"],
            parameter_idx=b["parameter_idx"],
            direction_idx=b["direction_idx"],
            tight_idx=b["tight_idx"],
            cross_domain=b["cross_domain"],
        ),
        checkpoint_path=core_ckpt,
        config=cfg_core,
        device=device,
    )
    ytr, ptr, idtr = predict(
        model=core_model,
        loader=DataLoader(core_train_ds, batch_size=512, shuffle=False),
        forward_fn=lambda m, b: m(
            sequence=b["sequence"],
            cand_num=b["cand_num"],
            parameter_idx=b["parameter_idx"],
            direction_idx=b["direction_idx"],
            tight_idx=b["tight_idx"],
            cross_domain=b["cross_domain"],
        ),
        device=device,
    )
    yva, pva, idva = predict(
        model=core_model,
        loader=DataLoader(core_val_ds, batch_size=512, shuffle=False),
        forward_fn=lambda m, b: m(
            sequence=b["sequence"],
            cand_num=b["cand_num"],
            parameter_idx=b["parameter_idx"],
            direction_idx=b["direction_idx"],
            tight_idx=b["tight_idx"],
            cross_domain=b["cross_domain"],
        ),
        device=device,
    )
    yte, pte, idte = predict(
        model=core_model,
        loader=DataLoader(core_test_ds, batch_size=512, shuffle=False),
        forward_fn=lambda m, b: m(
            sequence=b["sequence"],
            cand_num=b["cand_num"],
            parameter_idx=b["parameter_idx"],
            direction_idx=b["direction_idx"],
            tight_idx=b["tight_idx"],
            cross_domain=b["cross_domain"],
        ),
        device=device,
    )
    core_train_pred = _rows_with_pred(core_tr, idtr, ptr, "pred_core_gru")
    core_val_pred = _rows_with_pred(core_va, idva, pva, "pred_core_gru")
    core_test_pred = _rows_with_pred(core_te, idte, pte, "pred_core_gru")
    print("[Phase7] Stage 1 Core GRU done", flush=True)

    print("[Phase7] Stage 2 Parametric encoder start", flush=True)
    # --- Stage 2: Parametric non-sequential ---
    par_tr_model = par_tr.sample(n=min(len(par_tr), 120_000), random_state=7).reset_index(drop=True)
    par_train_ds = ParametricTabularDataset(par_tr_model)
    par_val_ds = ParametricTabularDataset(par_va)
    par_test_ds = ParametricTabularDataset(par_te)
    par_model = ParametricMLPRanker(
        norm_num_dim=len(par_train_ds.norm_cols),
        n_parameter=len(par_train_ds.param_map),
        n_direction=len(par_train_ds.dir_map),
        n_tight=len(par_train_ds.tight_map),
        n_condition=len(par_train_ds.cond_map),
        n_mode=len(par_train_ds.mode_map),
    )
    par_ckpt = ml_root / "checkpoints" / "parametric_mlp_best.pt"
    cfg_param = TrainConfig(lr=8e-4, weight_decay=1e-4, batch_size=2048, max_epochs=2, patience=1)
    par_best, par_hist = train_regressor(
        model=par_model,
        train_loader=DataLoader(par_train_ds, batch_size=cfg_param.batch_size, shuffle=True, num_workers=0),
        val_loader=DataLoader(par_val_ds, batch_size=cfg_param.batch_size, shuffle=False, num_workers=0),
        forward_fn=lambda m, b: m(
            norm_num=b["norm_num"],
            cand_num=b["cand_num"],
            cond_num=b["cond_num"],
            parameter_idx=b["parameter_idx"],
            direction_idx=b["direction_idx"],
            tight_idx=b["tight_idx"],
            condition_idx=b["condition_idx"],
            mode_idx=b["mode_idx"],
        ),
        checkpoint_path=par_ckpt,
        config=cfg_param,
        device=device,
    )
    _, pptr_tr, pidtr = predict(
        model=par_model,
        loader=DataLoader(par_train_ds, batch_size=1024, shuffle=False),
        forward_fn=lambda m, b: m(
            norm_num=b["norm_num"],
            cand_num=b["cand_num"],
            cond_num=b["cond_num"],
            parameter_idx=b["parameter_idx"],
            direction_idx=b["direction_idx"],
            tight_idx=b["tight_idx"],
            condition_idx=b["condition_idx"],
            mode_idx=b["mode_idx"],
        ),
        device=device,
    )
    _, pptr_va, pidva = predict(
        model=par_model,
        loader=DataLoader(par_val_ds, batch_size=1024, shuffle=False),
        forward_fn=lambda m, b: m(
            norm_num=b["norm_num"],
            cand_num=b["cand_num"],
            cond_num=b["cond_num"],
            parameter_idx=b["parameter_idx"],
            direction_idx=b["direction_idx"],
            tight_idx=b["tight_idx"],
            condition_idx=b["condition_idx"],
            mode_idx=b["mode_idx"],
        ),
        device=device,
    )
    _, pptr_te, pidte = predict(
        model=par_model,
        loader=DataLoader(par_test_ds, batch_size=1024, shuffle=False),
        forward_fn=lambda m, b: m(
            norm_num=b["norm_num"],
            cand_num=b["cand_num"],
            cond_num=b["cond_num"],
            parameter_idx=b["parameter_idx"],
            direction_idx=b["direction_idx"],
            tight_idx=b["tight_idx"],
            condition_idx=b["condition_idx"],
            mode_idx=b["mode_idx"],
        ),
        device=device,
    )
    par_train_pred = _rows_with_pred(par_tr_model, pidtr, pptr_tr, "pred_param_mlp")
    par_val_pred = _rows_with_pred(par_va, pidva, pptr_va, "pred_param_mlp")
    par_test_pred = _rows_with_pred(par_te, pidte, pptr_te, "pred_param_mlp")
    print("[Phase7] Stage 2 Parametric encoder done", flush=True)

    print("[Phase7] Stage 3 Joint model start", flush=True)
    # --- Stage 3: Joint ---
    core_vocab = {"parameter": core_train_ds.param_map, "direction": core_train_ds.dir_map, "tight": core_train_ds.tight_map}
    param_vocab = {
        "parameter": par_train_ds.param_map,
        "direction": par_train_ds.dir_map,
        "tight": par_train_ds.tight_map,
        "condition": par_train_ds.cond_map,
        "mode": par_train_ds.mode_map,
    }
    # Joint training uses full Core train + sampled Parametric train for practical runtime.
    core_tr_joint = core_tr.sample(n=min(len(core_tr), 15_000), random_state=7).reset_index(drop=True)
    par_tr_joint = par_tr.sample(n=min(len(par_tr), 20_000), random_state=7).reset_index(drop=True)
    joint_train_ds = JointDataset(
        core_rows=core_tr_joint,
        param_rows=par_tr_joint,
        seq_store=seq_store,
        core_vocab=core_vocab,
        param_vocab=param_vocab,
        param_norm_cols=par_train_ds.norm_cols,
    )
    joint_val_ds = JointDataset(
        core_rows=core_va,
        param_rows=par_va,
        seq_store=seq_store,
        core_vocab=core_vocab,
        param_vocab=param_vocab,
        param_norm_cols=par_train_ds.norm_cols,
    )
    joint_test_ds = JointDataset(
        core_rows=core_te,
        param_rows=par_te,
        seq_store=seq_store,
        core_vocab=core_vocab,
        param_vocab=param_vocab,
        param_norm_cols=par_train_ds.norm_cols,
    )
    joint_model = JointRanker(norm_num_dim=len(par_train_ds.norm_cols))
    joint_ckpt = ml_root / "checkpoints" / "joint_ranker_best.pt"
    cfg_joint = TrainConfig(lr=7e-4, weight_decay=1e-4, batch_size=1024, max_epochs=2, patience=1)
    joint_best, joint_hist = train_regressor(
        model=joint_model,
        train_loader=DataLoader(joint_train_ds, batch_size=cfg_joint.batch_size, shuffle=True, num_workers=0),
        val_loader=DataLoader(joint_val_ds, batch_size=cfg_joint.batch_size, shuffle=False, num_workers=0),
        forward_fn=lambda m, b: m(b),
        checkpoint_path=joint_ckpt,
        config=cfg_joint,
        device=device,
    )
    _, jp_tr, jid_tr = predict(
        model=joint_model,
        loader=DataLoader(joint_train_ds, batch_size=1024, shuffle=False),
        forward_fn=lambda m, b: m(b),
        device=device,
    )
    _, jp_va, jid_va = predict(
        model=joint_model,
        loader=DataLoader(joint_val_ds, batch_size=1024, shuffle=False),
        forward_fn=lambda m, b: m(b),
        device=device,
    )
    _, jp_te, jid_te = predict(
        model=joint_model,
        loader=DataLoader(joint_test_ds, batch_size=1024, shuffle=False),
        forward_fn=lambda m, b: m(b),
        device=device,
    )

    # Build combined dfs for joint eval
    joint_train_df = pd.concat([core_tr, par_tr], ignore_index=True)
    joint_val_df = pd.concat([core_va, par_va], ignore_index=True)
    joint_test_df = pd.concat([core_te, par_te], ignore_index=True)
    joint_train_df = _rows_with_pred(joint_train_df, jid_tr, jp_tr, "pred_joint")
    joint_val_df = _rows_with_pred(joint_val_df, jid_va, jp_va, "pred_joint")
    joint_test_df = _rows_with_pred(joint_test_df, jid_te, jp_te, "pred_joint")
    print("[Phase7] Stage 3 Joint model done", flush=True)

    print("[Phase7] Baselines start", flush=True)
    # Lazy import: sklearn baselines are training-only and must not load at API startup.
    from dtl_agent.ml.baselines.baselines import train_and_predict_baselines

    # Baselines (sequence-free; computed per domain)
    # sequence-free baselines use deterministic train subsamples for runtime practicality.
    core_tr_base = core_tr.sample(n=min(len(core_tr), 25_000), random_state=7)
    par_tr_base = par_tr.sample(n=min(len(par_tr), 40_000), random_state=7)
    core_base_va = train_and_predict_baselines(train_df=core_tr_base, pred_df=core_va)
    core_base_te = train_and_predict_baselines(train_df=core_tr_base, pred_df=core_te)
    par_base_va = train_and_predict_baselines(train_df=par_tr_base, pred_df=par_va)
    par_base_te = train_and_predict_baselines(train_df=par_tr_base, pred_df=par_te)
    core_val_pred["pred_linear"] = core_base_va.linear_pred
    core_val_pred["pred_tree"] = core_base_va.tree_pred
    core_val_pred["pred_mlp_sf"] = core_base_va.mlp_pred
    core_test_pred["pred_linear"] = core_base_te.linear_pred
    core_test_pred["pred_tree"] = core_base_te.tree_pred
    core_test_pred["pred_mlp_sf"] = core_base_te.mlp_pred
    par_val_pred["pred_linear"] = par_base_va.linear_pred
    par_val_pred["pred_tree"] = par_base_va.tree_pred
    par_val_pred["pred_mlp_sf"] = par_base_va.mlp_pred
    par_test_pred["pred_linear"] = par_base_te.linear_pred
    par_test_pred["pred_tree"] = par_base_te.tree_pred
    par_test_pred["pred_mlp_sf"] = par_base_te.mlp_pred

    # Deterministic optimizer reference baseline (global selected candidate per parameter)
    core_sel = pd.read_csv(root / "artifacts" / "simulation" / "core" / "selected_candidates.csv")
    param_sel = pd.read_csv(root / "artifacts" / "simulation" / "parametric" / "selected_candidates.csv")
    core_pick = {str(r["parameter"]): float(r["candidate_limit"]) for _, r in core_sel.iterrows() if str(r.get("optimization_mode", "independent")) != "joint"}
    param_pick = {str(r["parameter"]): float(r["candidate_limit"]) for _, r in param_sel.iterrows()}
    for df, picks in [(core_val_pred, core_pick), (core_test_pred, core_pick), (par_val_pred, param_pick), (par_test_pred, param_pick)]:
        df["pred_optimizer_ref"] = -np.abs(df["candidate_limit"] - df["parameter"].map(picks))
    print("[Phase7] Baselines done", flush=True)

    # Metrics
    metrics = {
        "core": {
            "validation_gru": _eval_model(core_val_pred, "pred_core_gru"),
            "test_gru": _eval_model(core_test_pred, "pred_core_gru"),
            "validation_linear": _eval_model(core_val_pred, "pred_linear"),
            "test_linear": _eval_model(core_test_pred, "pred_linear"),
            "validation_tree": _eval_model(core_val_pred, "pred_tree"),
            "test_tree": _eval_model(core_test_pred, "pred_tree"),
            "validation_mlp_sf": _eval_model(core_val_pred, "pred_mlp_sf"),
            "test_mlp_sf": _eval_model(core_test_pred, "pred_mlp_sf"),
            "validation_optimizer_ref": _eval_model(core_val_pred, "pred_optimizer_ref"),
            "test_optimizer_ref": _eval_model(core_test_pred, "pred_optimizer_ref"),
        },
        "parametric": {
            "validation_mlp": _eval_model(par_val_pred, "pred_param_mlp"),
            "test_mlp": _eval_model(par_test_pred, "pred_param_mlp"),
            "validation_linear": _eval_model(par_val_pred, "pred_linear"),
            "test_linear": _eval_model(par_test_pred, "pred_linear"),
            "validation_tree": _eval_model(par_val_pred, "pred_tree"),
            "test_tree": _eval_model(par_test_pred, "pred_tree"),
            "validation_mlp_sf": _eval_model(par_val_pred, "pred_mlp_sf"),
            "test_mlp_sf": _eval_model(par_test_pred, "pred_mlp_sf"),
            "validation_optimizer_ref": _eval_model(par_val_pred, "pred_optimizer_ref"),
            "test_optimizer_ref": _eval_model(par_test_pred, "pred_optimizer_ref"),
        },
        "joint": {
            "validation_joint": _eval_model(joint_val_df, "pred_joint"),
            "test_joint": _eval_model(joint_test_df, "pred_joint"),
        },
        "training_history": {
            "core": core_hist,
            "parametric": par_hist,
            "joint": joint_hist,
        },
        "best_epochs": {"core": core_best, "parametric": par_best, "joint": joint_best},
    }

    # save predictions/metrics
    core_test_pred.to_parquet(ml_root / "predictions" / "core_test_predictions.parquet", index=False)
    par_test_pred.to_parquet(ml_root / "predictions" / "parametric_test_predictions.parquet", index=False)
    joint_test_df.to_parquet(ml_root / "predictions" / "joint_test_predictions.parquet", index=False)
    (ml_root / "metrics" / "phase7_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[Phase7] Metrics written", flush=True)

    # simple report
    report_lines = [
        "# Phase 7 GRU Results",
        "",
        "## Core (Validation/Test)",
        f"- GRU validation NDCG@K: {metrics['core']['validation_gru']['ndcg_at_k']:.4f}",
        f"- GRU test NDCG@K: {metrics['core']['test_gru']['ndcg_at_k']:.4f}",
        f"- Tree test NDCG@K: {metrics['core']['test_tree']['ndcg_at_k']:.4f}",
        "",
        "## Parametric (Validation/Test)",
        f"- Param MLP validation NDCG@K: {metrics['parametric']['validation_mlp']['ndcg_at_k']:.4f}",
        f"- Param MLP test NDCG@K: {metrics['parametric']['test_mlp']['ndcg_at_k']:.4f}",
        f"- Tree test NDCG@K: {metrics['parametric']['test_tree']['ndcg_at_k']:.4f}",
        "",
        "## Joint (Validation/Test)",
        f"- Joint validation NDCG@K: {metrics['joint']['validation_joint']['ndcg_at_k']:.4f}",
        f"- Joint test NDCG@K: {metrics['joint']['test_joint']['ndcg_at_k']:.4f}",
        "",
        "Synthetic objective caveat: model learns simulator-derived candidate quality, not production reliability truth.",
    ]
    (ml_root / "reports" / "phase7_results_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    runtime = time.perf_counter() - t0
    return Phase7Artifacts(
        output_root=ml_root,
        metrics=metrics,
        best_checkpoints={
            "core": str(core_ckpt),
            "parametric": str(par_ckpt),
            "joint": str(joint_ckpt),
        },
        runtime_seconds=runtime,
    )
