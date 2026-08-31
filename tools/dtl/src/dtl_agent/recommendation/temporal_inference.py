"""Hybrid temporal inference: CoreGRU temporal + UnifiedParameterGRU (Phase 12.8).

Does not modify CoreGRURanker / UnifiedParameterGRURanker classes or checkpoints.
Simulation outcomes are never fed as model inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.paths import month_simulation_root, temporal_artifact_root
from dtl_agent.ml.datasets.phase7_datasets import CORE_CAND_NUM, CoreSequenceStore
from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml.models.unified_gru_ranker import (
    CORE_SCORE_PARAMETERS,
    PARAMETRIC_CONDITION_ORDER,
    PARAMETRIC_SCORE_PARAMETERS,
    UNIFIED_PARAMETER_VOCAB,
    UnifiedParameterGRURanker,
)
from dtl_agent.ml.unified_experiment import (
    FORBIDDEN_INPUT_COLS,
    build_parametric_context_table,
    empty_parametric_context_row,
)
from dtl_agent.recommendation.routing import HybridModelId, model_for_parameter


class TemporalHybridInferenceError(RuntimeError):
    pass


def _cand_frame(path: Path, parameter: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["parameter"].astype(str) == str(parameter)].copy()
    if df.empty:
        raise TemporalHybridInferenceError(f"No candidates for {parameter} in {path}")
    if "candidate_delta" not in df.columns and "delta_absolute" in df.columns:
        df["candidate_delta"] = df["delta_absolute"]
    if "candidate_delta_percent" not in df.columns and "delta_percent" in df.columns:
        df["candidate_delta_percent"] = df["delta_percent"]
    # Leakage guard: scoring uses only cand numerics + cats — strip outcome cols from feature use
    return df.reset_index(drop=True)


class TemporalHybridBundle:
    """Lazy-loaded temporal Core + Unified scorers (month-aware sequences)."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._ready = False
        self.load_errors: list[str] = []
        self.core_model: CoreGRURanker | None = None
        self.uni_model: UnifiedParameterGRURanker | None = None
        self.core_arch: dict[str, Any] = {}
        self.uni_dir_map: dict[str, int] = {}
        self.uni_tight_map: dict[str, int] = {}
        self.uni_param_map = {p: i for i, p in enumerate(UNIFIED_PARAMETER_VOCAB)}
        self.uni_norm: dict[str, Any] = {}
        self.seq_store: CoreSequenceStore | None = None
        self.core_checkpoint_id: str | None = None
        self.uni_checkpoint_id: str | None = None
        self._deltas_by_month: dict[str, pd.DataFrame] = {}
        # Month + parameter scoped candidate frames (population evidence; never cross-month).
        self._cand_by_month_param: dict[tuple[str, str], pd.DataFrame] = {}

    def ensure_loaded(self) -> bool:
        if self._ready:
            return True
        try:
            shared = temporal_artifact_root(self.project_root) / "shared"
            core_ckpt = shared / "checkpoints" / "core_gru_temporal_v1.pt"
            uni_ckpt = shared / "checkpoints" / "unified_parameter_gru_v1.pt"
            arch_path = shared / "training" / "architecture.json"
            uni_arch = shared / "unified_training" / "architecture.json"
            seq_path = shared / "unified_ml_dataset" / "sequences" / "core_sequences.parquet"
            if not seq_path.is_file():
                seq_path = shared / "ml_dataset" / "sequences" / "core_sequences.parquet"
            norm_path = shared / "unified_ml_dataset" / "normalization" / "normalization_stats.json"

            for p, label in (
                (core_ckpt, "core_temporal_checkpoint"),
                (uni_ckpt, "unified_checkpoint"),
                (arch_path, "core_architecture"),
                (seq_path, "temporal_sequences"),
                (norm_path, "unified_normalization"),
            ):
                if not p.is_file():
                    self.load_errors.append(f"{label}_missing:{p}")
            if self.load_errors:
                return False

            self.core_arch = json.loads(arch_path.read_text(encoding="utf-8"))
            state_c = torch.load(core_ckpt, map_location="cpu", weights_only=False)
            self.core_model = CoreGRURanker(
                n_parameter=len(self.core_arch["parameter_vocab"]),
                n_direction=len(self.core_arch["direction_vocab"]),
                n_tight=len(self.core_arch["tighten_vocab"]),
            )
            self.core_model.load_state_dict(state_c["model_state"])
            self.core_model.eval()
            self.core_checkpoint_id = str(core_ckpt)

            if uni_arch.is_file():
                ua = json.loads(uni_arch.read_text(encoding="utf-8"))
                self.uni_dir_map = ua["direction_vocab"]
                self.uni_tight_map = ua["tighten_vocab"]
            else:
                meta = torch.load(uni_ckpt, map_location="cpu", weights_only=False).get(
                    "unified_metadata", {}
                )
                self.uni_dir_map = meta.get("direction_vocab", {"LOWER": 0, "UPPER": 1})
                self.uni_tight_map = meta.get(
                    "tighten_vocab", {"CURRENT": 0, "LOOSER": 1, "TIGHTER": 2}
                )

            state_u = torch.load(uni_ckpt, map_location="cpu", weights_only=False)
            self.uni_model = UnifiedParameterGRURanker(
                n_parameter=len(UNIFIED_PARAMETER_VOCAB),
                n_direction=len(self.uni_dir_map),
                n_tight=len(self.uni_tight_map),
            )
            self.uni_model.load_state_dict(state_u["model_state"])
            self.uni_model.eval()
            self.uni_checkpoint_id = str(uni_ckpt)
            self.uni_norm = json.loads(norm_path.read_text(encoding="utf-8"))
            self.seq_store = CoreSequenceStore(pd.read_parquet(seq_path))
            self._ready = True
            return True
        except Exception as exc:  # noqa: BLE001
            self.load_errors.append(f"temporal_bundle_load_error:{type(exc).__name__}:{exc}")
            return False

    def _norm_cand_ctx(
        self, parameter: str, cand_row: pd.Series, ctx: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        feats = self.uni_norm.get("parameters", {}).get(parameter, {})
        cand = []
        for c in CORE_CAND_NUM:
            raw = float(cand_row[c])
            st = feats.get(c, {"mean": 0.0, "std": 1.0})
            sd = st["std"] if abs(st["std"]) > 1e-12 else 1.0
            cand.append((raw - st["mean"]) / sd)
        ctx_vals, masks = [], []
        for i in range(4):
            raw = float(ctx.get(f"ctx_val_{i}", 0.0))
            mask = float(ctx.get(f"ctx_mask_{i}", 0.0))
            st = feats.get(f"ctx_val_{i}", {"mean": 0.0, "std": 1.0})
            sd = st["std"] if abs(st["std"]) > 1e-12 else 1.0
            ctx_vals.append(((raw - st["mean"]) / sd) * mask)
            masks.append(mask)
        return np.array(cand, dtype=np.float32), np.array(ctx_vals + masks, dtype=np.float32)

    def score_parameter(
        self,
        *,
        production_month: str,
        lot_id: str,
        die_id: str,
        parameter: str,
        month_data,
    ) -> tuple[pd.DataFrame | None, str | None, str]:
        """Return scored candidate frame, error code, and model_used label."""
        if not self.ensure_loaded():
            return None, "model_unavailable", "unavailable"
        assert self.seq_store is not None and self.core_model is not None and self.uni_model is not None

        mid = model_for_parameter(parameter, temporal=True)
        if mid == HybridModelId.UNSUPPORTED:
            return None, "unsupported_parameter", HybridModelId.UNSUPPORTED.value

        sid = make_sequence_id(lot_id, die_id, production_month)
        if sid not in self.seq_store.mats:
            return None, "core_sequence_unavailable", mid.value

        cand_key = (str(production_month), str(parameter))
        cand_df = self._cand_by_month_param.get(cand_key)
        if cand_df is None:
            sim_root = month_simulation_root(production_month, self.project_root)
            if parameter in CORE_SCORE_PARAMETERS:
                cand_path = sim_root / "core" / "candidate_results.csv"
            else:
                cand_path = sim_root / "parametric" / "candidate_results.csv"
            try:
                cand_df = _cand_frame(cand_path, parameter)
            except TemporalHybridInferenceError as exc:
                return None, str(exc), mid.value
            self._cand_by_month_param[cand_key] = cand_df

        # Explicit leakage check on feature construction
        feature_cols = list(CORE_CAND_NUM) + ["parameter", "direction", "tighten_or_loosen"]
        if any(c in FORBIDDEN_INPUT_COLS for c in feature_cols):
            return None, "leakage_guard_failed", mid.value

        seq = self.seq_store.get(sid)
        scores: list[float] = []

        N = len(cand_df)
        if N == 0:
            scores = []
        elif mid == HybridModelId.CORE_TEMPORAL:
            vparam = self.core_arch["parameter_vocab"]
            vdir = self.core_arch["direction_vocab"]
            vtight = self.core_arch["tighten_vocab"]
            with torch.no_grad():
                seq_b = torch.from_numpy(np.array(seq, copy=True)).unsqueeze(0).expand(N, -1, -1)
                cand_num_b = torch.from_numpy(
                    np.ascontiguousarray(cand_df[list(CORE_CAND_NUM)].to_numpy(dtype=np.float32))
                )
                param_idx_b = torch.tensor(
                    [vparam[str(p)] for p in cand_df["parameter"]], dtype=torch.long
                )
                dir_idx_b = torch.tensor(
                    [vdir[str(d)] for d in cand_df["direction"]], dtype=torch.long
                )
                tight_idx_b = torch.tensor(
                    [vtight[str(t)] for t in cand_df["tighten_or_loosen"]], dtype=torch.long
                )
                cross_b = torch.zeros(N, dtype=torch.float32)

                pred_b = self.core_model(
                    sequence=seq_b,
                    cand_num=cand_num_b,
                    parameter_idx=param_idx_b,
                    direction_idx=dir_idx_b,
                    tight_idx=tight_idx_b,
                    cross_domain=cross_b,
                )
                scores = [float(x) for x in pred_b.cpu().numpy()]
            model_id = HybridModelId.CORE_TEMPORAL.value
        else:
            ctx_table = build_parametric_context_table(month_data)
            ctx_rows = ctx_table[
                (ctx_table["lot_id"].astype(str) == str(lot_id))
                & (ctx_table["die_id"].astype(str) == str(die_id))
                & (ctx_table["parameter"].astype(str) == str(parameter))
            ]
            ctx = empty_parametric_context_row() if ctx_rows.empty else ctx_rows.iloc[0].to_dict()
            has_pc = float(ctx.get("has_parametric_context", 0.0))

            cand_n_list = []
            ctx_n_list = []
            dir_list = []
            tight_list = []
            for _, r in cand_df.iterrows():
                cn, ctxn = self._norm_cand_ctx(parameter, r, ctx)
                cand_n_list.append(cn)
                ctx_n_list.append(ctxn)
                dir_list.append(self.uni_dir_map[str(r["direction"])])
                tight_list.append(self.uni_tight_map[str(r["tighten_or_loosen"])])

            with torch.no_grad():
                seq_b = torch.from_numpy(np.array(seq, copy=True)).unsqueeze(0).expand(N, -1, -1)
                cand_num_b = torch.from_numpy(
                    np.ascontiguousarray(np.array(cand_n_list, dtype=np.float32))
                )
                ctx_num_b = torch.from_numpy(
                    np.ascontiguousarray(np.array(ctx_n_list, dtype=np.float32))
                )
                has_pc_b = torch.full((N,), has_pc, dtype=torch.float32)
                param_idx_b = torch.full((N,), self.uni_param_map[parameter], dtype=torch.long)
                dir_idx_b = torch.tensor(dir_list, dtype=torch.long)
                tight_idx_b = torch.tensor(tight_list, dtype=torch.long)

                pred_b = self.uni_model(
                    sequence=seq_b,
                    cand_num=cand_num_b,
                    parametric_context=ctx_num_b,
                    has_parametric_context=has_pc_b,
                    parameter_idx=param_idx_b,
                    direction_idx=dir_idx_b,
                    tight_idx=tight_idx_b,
                )
                scores = [float(x) for x in pred_b.cpu().numpy()]
            model_id = HybridModelId.UNIFIED.value

        out = cand_df.copy()
        out["ml_score"] = scores
        out["model_id"] = model_id
        out["lot_id"] = lot_id
        out["die_id"] = die_id
        if parameter in PARAMETRIC_SCORE_PARAMETERS:
            out["conditions_present"] = [list(PARAMETRIC_CONDITION_ORDER)] * len(out)
        if not np.isfinite(out["ml_score"].to_numpy(dtype=float)).all():
            return None, "non_finite_ml_score", model_id
        return out, None, model_id
