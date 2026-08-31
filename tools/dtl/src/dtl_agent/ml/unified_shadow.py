"""Phase 12.6 — Unified GRU recommendation shadow evaluation (offline only).

Compares existing CoreGRU / ParametricMLP vs UnifiedParameterGRURanker on the
same temporal candidates, simulation evidence, safety, and yield-first policy.

Does NOT wire UnifiedParameterGRU into recommend(). Does NOT mutate checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.loader import load_temporal_month
from dtl_agent.data.temporal.paths import month_simulation_root, temporal_artifact_root
from dtl_agent.features.io_utils import write_json
from dtl_agent.ml.datasets.phase7_datasets import (
    CORE_CAND_NUM,
    PARAM_CAND_NUM,
    CoreSequenceStore,
    _cat_map,
)
from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml.models.parametric_encoder import ParametricMLPRanker
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
from dtl_agent.ml_dataset.temporal_pipeline import TEMPORAL_MONTHS
from dtl_agent.recommendation.catalog import CandidateCatalog
from dtl_agent.recommendation.config import RecommendationConfig
from dtl_agent.recommendation.evidence import SimulationEvidenceLookup
from dtl_agent.recommendation.policy import EvaluatedCandidate, apply_recommendation_policy
from dtl_agent.recommendation.ranking import rank_candidates
from dtl_agent.recommendation.safety import evaluate_safety
from dtl_agent.recommendation.schemas import RankedCandidate

SHADOW_PARAMETERS = tuple(UNIFIED_PARAMETER_VOCAB)
EXCLUDED_SEQUENCE_ONLY = frozenset({"setup_slack", "hold_slack", "test_time"})
FORBIDDEN_MODEL_FEATURES = frozenset(FORBIDDEN_INPUT_COLS)


class UnifiedShadowError(RuntimeError):
    pass


def _month_rec_config(month: str, root: Path) -> RecommendationConfig:
    sim = month_simulation_root(month, root)
    return RecommendationConfig(
        core_candidate_grid_path=str(sim / "core" / "candidate_grid.csv"),
        core_candidate_results_path=str(sim / "core" / "candidate_results.csv"),
        parametric_candidate_grid_path=str(sim / "parametric" / "candidate_grid.csv"),
        parametric_candidate_results_path=str(sim / "parametric" / "candidate_results.csv"),
        evidence_origin_label=f"SIMULATOR_DERIVED_TEMPORAL_{month}",
    )


def _cand_frame_from_results(path: Path, parameter: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["parameter"].astype(str) == str(parameter)].copy()
    if df.empty:
        raise UnifiedShadowError(f"No candidates for {parameter} in {path}")
    if "candidate_delta" not in df.columns and "delta_absolute" in df.columns:
        df["candidate_delta"] = df["delta_absolute"]
    if "candidate_delta_percent" not in df.columns and "delta_percent" in df.columns:
        df["candidate_delta_percent"] = df["delta_percent"]
    return df.reset_index(drop=True)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_die_param_deltas(month_data) -> pd.DataFrame:
    """Die-level parametric deltas matching Phase-3 feature names used by MLP."""
    df = month_data.parametric.copy()
    rows = []
    for (lot, die), g in df.groupby(["lot_id", "die_id"], sort=False):
        rec: dict[str, Any] = {"lot_id": str(lot), "die_id": str(die)}
        for param in PARAMETRIC_SCORE_PARAMETERS:
            sub = g[g["parameter"].astype(str) == param]
            by_c = {
                str(r.condition_id): float(r.measurement_value)
                for r in sub.itertuples(index=False)
            }
            prefix = f"param_{param.lower()}"
            if "COND_HOT_NOM" in by_c and "COND_RT_NOM" in by_c:
                rec[f"{prefix}_delta_hot_minus_rt"] = by_c["COND_HOT_NOM"] - by_c["COND_RT_NOM"]
            if "COND_RT_LOWV" in by_c and "COND_RT_NOM" in by_c:
                rec[f"{prefix}_delta_lowv_minus_nom"] = by_c["COND_RT_LOWV"] - by_c["COND_RT_NOM"]
            if "COND_HOT_HIGHV" in by_c and "COND_RT_NOM" in by_c:
                rec[f"{prefix}_delta_hot_highv_minus_rt_nom"] = (
                    by_c["COND_HOT_HIGHV"] - by_c["COND_RT_NOM"]
                )
        rows.append(rec)
    return pd.DataFrame(rows)


def _condition_meta(month_data) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in month_data.parametric.itertuples(index=False):
        cid = str(r.condition_id)
        if cid not in out:
            out[cid] = {
                "temperature_c": float(r.temperature_c),
                "vdd_applied": float(r.vdd_applied),
                "test_mode": str(r.test_mode),
            }
    return out


class ShadowScorers:
    """Load existing + unified models once; score shared candidate frames."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.device = torch.device("cpu")
        self._load_existing_core_temporal()
        self._load_existing_mlp()
        self._load_unified()

    def _load_existing_core_temporal(self) -> None:
        ckpt = temporal_artifact_root(self.root) / "shared" / "checkpoints" / "core_gru_temporal_v1.pt"
        arch = temporal_artifact_root(self.root) / "shared" / "training" / "architecture.json"
        if not ckpt.is_file() or not arch.is_file():
            raise UnifiedShadowError("Missing temporal CoreGRU checkpoint/architecture")
        self.core_arch = json.loads(arch.read_text(encoding="utf-8"))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.core_model = CoreGRURanker(
            n_parameter=len(self.core_arch["parameter_vocab"]),
            n_direction=len(self.core_arch["direction_vocab"]),
            n_tight=len(self.core_arch["tighten_vocab"]),
        )
        self.core_model.load_state_dict(state["model_state"])
        self.core_model.eval()
        self.core_model_id = "core_gru_temporal_v1"
        self.core_ckpt = str(ckpt)

    def _load_existing_mlp(self) -> None:
        ml_root = self.root / "artifacts" / "ml_dataset"
        train_param = pd.read_parquet(ml_root / "train" / "parametric_candidate_examples.parquet")
        norm_cols = [c for c in train_param.columns if c.startswith(PARAM_NORM_PREFIX)]
        self.mlp_norm_cols = norm_cols
        self.mlp_param_map = _cat_map(train_param["parameter"].astype(str).tolist())
        self.mlp_dir_map = _cat_map(train_param["direction"].astype(str).tolist())
        self.mlp_tight_map = _cat_map(train_param["tighten_or_loosen"].astype(str).tolist())
        self.mlp_cond_map = _cat_map(train_param["condition_id"].astype(str).tolist())
        self.mlp_mode_map = _cat_map(train_param["test_mode"].astype(str).tolist())
        norm_path = ml_root / "normalization" / "normalization_stats.json"
        self.mlp_norm_stats = json.loads(norm_path.read_text(encoding="utf-8"))["parametric"][
            "features"
        ]
        ckpt = self.root / "artifacts" / "ml" / "checkpoints" / "parametric_mlp_best.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.mlp_model = ParametricMLPRanker(
            norm_num_dim=len(norm_cols),
            n_parameter=len(self.mlp_param_map),
            n_direction=len(self.mlp_dir_map),
            n_tight=len(self.mlp_tight_map),
            n_condition=len(self.mlp_cond_map),
            n_mode=len(self.mlp_mode_map),
        )
        self.mlp_model.load_state_dict(state["model_state"])
        self.mlp_model.eval()
        self.mlp_model_id = "parametric_mlp_best"
        self.mlp_ckpt = str(ckpt)

    def _load_unified(self) -> None:
        shared = temporal_artifact_root(self.root) / "shared"
        ckpt = shared / "checkpoints" / "unified_parameter_gru_v1.pt"
        arch_path = shared / "unified_training" / "architecture.json"
        if not ckpt.is_file():
            raise UnifiedShadowError(f"Missing unified checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta = state.get("unified_metadata", {})
        if arch_path.is_file():
            arch = json.loads(arch_path.read_text(encoding="utf-8"))
            self.uni_dir_map = arch["direction_vocab"]
            self.uni_tight_map = arch["tighten_vocab"]
        else:
            self.uni_dir_map = meta.get("direction_vocab", {"LOWER": 0, "UPPER": 1})
            self.uni_tight_map = meta.get(
                "tighten_vocab", {"CURRENT": 0, "LOOSER": 1, "TIGHTER": 2}
            )
        self.uni_param_map = {p: i for i, p in enumerate(UNIFIED_PARAMETER_VOCAB)}
        self.uni_model = UnifiedParameterGRURanker(
            n_parameter=len(UNIFIED_PARAMETER_VOCAB),
            n_direction=len(self.uni_dir_map),
            n_tight=len(self.uni_tight_map),
        )
        self.uni_model.load_state_dict(state["model_state"])
        self.uni_model.eval()
        self.uni_model_id = "unified_parameter_gru_v1"
        self.uni_ckpt = str(ckpt)
        norm_path = shared / "unified_ml_dataset" / "normalization" / "normalization_stats.json"
        self.uni_norm = json.loads(norm_path.read_text(encoding="utf-8"))

    def _norm_cand_ctx(
        self, parameter: str, cand_row: pd.Series, ctx: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        feats = self.uni_norm["parameters"].get(parameter, {})
        cand = []
        for c in CORE_CAND_NUM:
            raw = float(cand_row[c])
            st = feats.get(c, {"mean": 0.0, "std": 1.0})
            cand.append((raw - st["mean"]) / (st["std"] if abs(st["std"]) > 1e-12 else 1.0))
        ctx_vals = []
        masks = []
        for i in range(4):
            raw = float(ctx.get(f"ctx_val_{i}", 0.0))
            mask = float(ctx.get(f"ctx_mask_{i}", 0.0))
            st = feats.get(f"ctx_val_{i}", {"mean": 0.0, "std": 1.0})
            nv = (raw - st["mean"]) / (st["std"] if abs(st["std"]) > 1e-12 else 1.0)
            ctx_vals.append(nv * mask)
            masks.append(mask)
        return (
            np.array(cand, dtype=np.float32),
            np.array(ctx_vals + masks, dtype=np.float32),
        )

    def score_core_existing(self, *, seq: np.ndarray, cand_df: pd.DataFrame) -> list[float]:
        vparam = self.core_arch["parameter_vocab"]
        vdir = self.core_arch["direction_vocab"]
        vtight = self.core_arch["tighten_vocab"]
        scores: list[float] = []
        with torch.no_grad():
            seq_t = torch.from_numpy(np.array(seq, copy=True)).unsqueeze(0)
            for _, r in cand_df.iterrows():
                cand = np.array([float(r[c]) for c in CORE_CAND_NUM], dtype=np.float32)
                pred = self.core_model(
                    sequence=seq_t,
                    cand_num=torch.from_numpy(cand).unsqueeze(0),
                    parameter_idx=torch.tensor([vparam[str(r["parameter"])]], dtype=torch.long),
                    direction_idx=torch.tensor([vdir[str(r["direction"])]], dtype=torch.long),
                    tight_idx=torch.tensor([vtight[str(r["tighten_or_loosen"])]], dtype=torch.long),
                    cross_domain=torch.tensor([0.0], dtype=torch.float32),
                )
                scores.append(float(pred.squeeze().cpu().numpy()))
        return scores

    def score_mlp_existing(
        self,
        *,
        die_deltas: dict[str, float],
        cand_df: pd.DataFrame,
        cond_meta: dict[str, dict[str, Any]],
        parameter: str,
    ) -> list[float]:
        """Score each candidate; mean over 4 conditions (production MLP aggregation)."""
        norm_vec = []
        for col in self.mlp_norm_cols:
            feat = col[len("norm_") :]
            st = self.mlp_norm_stats.get(feat, {"mean": 0.0, "std": 1.0})
            raw = float(die_deltas.get(feat, 0.0))
            sd = st["std"] if abs(st["std"]) > 1e-12 else 1.0
            norm_vec.append((raw - st["mean"]) / sd)
        norm_arr = np.array(norm_vec, dtype=np.float32)

        out_scores: list[float] = []
        with torch.no_grad():
            for _, r in cand_df.iterrows():
                cand = np.array([float(r[c]) for c in PARAM_CAND_NUM], dtype=np.float32)
                cond_scores = []
                for cid in PARAMETRIC_CONDITION_ORDER:
                    meta = cond_meta[cid]
                    cond_num = np.array(
                        [float(meta["temperature_c"]), float(meta["vdd_applied"])],
                        dtype=np.float32,
                    )
                    pred = self.mlp_model(
                        norm_num=torch.from_numpy(norm_arr).unsqueeze(0),
                        cand_num=torch.from_numpy(cand).unsqueeze(0),
                        cond_num=torch.from_numpy(cond_num).unsqueeze(0),
                        parameter_idx=torch.tensor(
                            [self.mlp_param_map[parameter]], dtype=torch.long
                        ),
                        direction_idx=torch.tensor(
                            [self.mlp_dir_map[str(r["direction"])]], dtype=torch.long
                        ),
                        tight_idx=torch.tensor(
                            [self.mlp_tight_map[str(r["tighten_or_loosen"])]], dtype=torch.long
                        ),
                        condition_idx=torch.tensor([self.mlp_cond_map[cid]], dtype=torch.long),
                        mode_idx=torch.tensor(
                            [self.mlp_mode_map[str(meta["test_mode"])]], dtype=torch.long
                        ),
                    )
                    cond_scores.append(float(pred.squeeze().cpu().numpy()))
                out_scores.append(float(np.mean(cond_scores)))
        return out_scores

    def score_unified(
        self,
        *,
        seq: np.ndarray,
        cand_df: pd.DataFrame,
        ctx: dict[str, float],
        parameter: str,
    ) -> list[float]:
        scores: list[float] = []
        has_pc = float(ctx.get("has_parametric_context", 0.0))
        with torch.no_grad():
            seq_t = torch.from_numpy(np.array(seq, copy=True)).unsqueeze(0)
            for _, r in cand_df.iterrows():
                cand_n, ctx_n = self._norm_cand_ctx(parameter, r, ctx)
                pred = self.uni_model(
                    sequence=seq_t,
                    cand_num=torch.from_numpy(cand_n).unsqueeze(0),
                    parametric_context=torch.from_numpy(ctx_n).unsqueeze(0),
                    has_parametric_context=torch.tensor([has_pc], dtype=torch.float32),
                    parameter_idx=torch.tensor(
                        [self.uni_param_map[parameter]], dtype=torch.long
                    ),
                    direction_idx=torch.tensor(
                        [self.uni_dir_map[str(r["direction"])]], dtype=torch.long
                    ),
                    tight_idx=torch.tensor(
                        [self.uni_tight_map[str(r["tighten_or_loosen"])]], dtype=torch.long
                    ),
                )
                scores.append(float(pred.squeeze().cpu().numpy()))
        return scores


def _apply_path(
    *,
    scored_df: pd.DataFrame,
    lot_id: str,
    die_id: str,
    domain: str,
    catalog: CandidateCatalog,
    evidence: SimulationEvidenceLookup,
    config: RecommendationConfig,
) -> tuple[Any, list[RankedCandidate], list[EvaluatedCandidate]]:
    ranked = rank_candidates(scored_df, lot_id=lot_id, die_id=die_id, catalog=catalog)
    evaluated: list[EvaluatedCandidate] = []
    for cand in ranked:
        ev = evidence.lookup(
            domain=domain, parameter=cand.parameter, candidate_limit=cand.candidate_limit
        )
        safety = evaluate_safety(
            candidate=cand,
            evidence=ev,
            catalog=catalog,
            config=config,
            domain=domain,
            conditions_present=list(PARAMETRIC_CONDITION_ORDER)
            if domain == "parametric"
            else None,
            context_complete=True,
            model_available=True,
        )
        evaluated.append(EvaluatedCandidate(candidate=cand, evidence=ev, safety=safety))
    current_limit = float(scored_df["current_limit"].iloc[0])
    policy = apply_recommendation_policy(evaluated=evaluated, current_limit=current_limit)
    return policy, ranked, evaluated


def _select_dies(month_data, *, max_dies_per_lot: int = 50) -> pd.DataFrame:
    dies = (
        month_data.actual_die[["lot_id", "die_id"]]
        .drop_duplicates()
        .assign(lot_id=lambda d: d["lot_id"].astype(str), die_id=lambda d: d["die_id"].astype(str))
    )
    if max_dies_per_lot < 50:
        dies = dies.groupby("lot_id", group_keys=False).head(max_dies_per_lot)
    return dies.reset_index(drop=True)


def run_unified_recommendation_shadow(
    project_root: Path | None = None,
    *,
    max_dies_per_lot: int = 5,
    months: tuple[str, ...] = TEMPORAL_MONTHS,
) -> dict[str, Any]:
    """Shadow-evaluate recommendations. Default 5 dies/lot (100/month) for runtime."""
    root = project_root or default_project_root()
    out_dir = temporal_artifact_root(root) / "shared" / "unified_shadow"
    out_dir.mkdir(parents=True, exist_ok=True)

    watched = {
        "core_gru_best": root / "artifacts" / "ml" / "checkpoints" / "core_gru_best.pt",
        "parametric_mlp_best": root / "artifacts" / "ml" / "checkpoints" / "parametric_mlp_best.pt",
        "core_gru_temporal_v1": temporal_artifact_root(root)
        / "shared"
        / "checkpoints"
        / "core_gru_temporal_v1.pt",
        "unified_parameter_gru_v1": temporal_artifact_root(root)
        / "shared"
        / "checkpoints"
        / "unified_parameter_gru_v1.pt",
    }
    hashes_before = {k: _sha256_file(p) for k, p in watched.items()}

    scorer = ShadowScorers(root)
    seq_store = CoreSequenceStore(
        pd.read_parquet(
            temporal_artifact_root(root)
            / "shared"
            / "unified_ml_dataset"
            / "sequences"
            / "core_sequences.parquet"
        )
    )

    detail_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for month in months:
        month_data = load_temporal_month(month, project_root=root)
        cfg = _month_rec_config(month, root)
        catalog = CandidateCatalog(root, cfg)
        evidence = SimulationEvidenceLookup(root, cfg)
        dies = _select_dies(month_data, max_dies_per_lot=max_dies_per_lot)
        # Always include representative dies for temporal section
        extra = pd.DataFrame(
            [
                {"lot_id": "DTL_CENTER_001", "die_id": "DTL_CENTER_001_D001"},
                {"lot_id": "DTL_EDGE_003", "die_id": "DTL_EDGE_003_D001"},
            ]
        )
        dies = pd.concat([dies, extra], ignore_index=True).drop_duplicates()
        ctx_table = build_parametric_context_table(month_data)
        die_deltas = _build_die_param_deltas(month_data)
        cond_meta = _condition_meta(month_data)

        core_results = month_simulation_root(month, root) / "core" / "candidate_results.csv"
        param_results = (
            month_simulation_root(month, root) / "parametric" / "candidate_results.csv"
        )

        for parameter in SHADOW_PARAMETERS:
            domain = "core" if parameter in CORE_SCORE_PARAMETERS else "parametric"
            results_path = core_results if domain == "core" else param_results
            cand_df = _cand_frame_from_results(results_path, parameter)
            cand_limits = sorted(float(x) for x in cand_df["candidate_limit"].tolist())

            for _, die_r in dies.iterrows():
                lot_id, die_id = str(die_r["lot_id"]), str(die_r["die_id"])
                sid = make_sequence_id(lot_id, die_id, month)
                seq = seq_store.get(sid)

                if domain == "core":
                    existing_scores = scorer.score_core_existing(seq=seq, cand_df=cand_df)
                    existing_model = scorer.core_model_id
                    ctx = empty_parametric_context_row()
                else:
                    drow = die_deltas[
                        (die_deltas["lot_id"] == lot_id) & (die_deltas["die_id"] == die_id)
                    ]
                    deltas = drow.iloc[0].to_dict() if not drow.empty else {}
                    existing_scores = scorer.score_mlp_existing(
                        die_deltas=deltas,
                        cand_df=cand_df,
                        cond_meta=cond_meta,
                        parameter=parameter,
                    )
                    existing_model = scorer.mlp_model_id
                    ctx_rows = ctx_table[
                        (ctx_table["lot_id"] == lot_id)
                        & (ctx_table["die_id"] == die_id)
                        & (ctx_table["parameter"] == parameter)
                    ]
                    ctx = (
                        empty_parametric_context_row()
                        if ctx_rows.empty
                        else ctx_rows.iloc[0].to_dict()
                    )

                unified_scores = scorer.score_unified(
                    seq=seq, cand_df=cand_df, ctx=ctx, parameter=parameter
                )

                exist_scored = cand_df.copy()
                exist_scored["ml_score"] = existing_scores
                exist_scored["model_id"] = existing_model
                uni_scored = cand_df.copy()
                uni_scored["ml_score"] = unified_scores
                uni_scored["model_id"] = scorer.uni_model_id

                pol_e, ranked_e, eval_e = _apply_path(
                    scored_df=exist_scored,
                    lot_id=lot_id,
                    die_id=die_id,
                    domain=domain,
                    catalog=catalog,
                    evidence=evidence,
                    config=cfg,
                )
                pol_u, ranked_u, eval_u = _apply_path(
                    scored_df=uni_scored,
                    lot_id=lot_id,
                    die_id=die_id,
                    domain=domain,
                    catalog=catalog,
                    evidence=evidence,
                    config=cfg,
                )

                e_by_lim = {c.candidate_limit: c for c in ranked_e}
                u_by_lim = {c.candidate_limit: c for c in ranked_u}
                ev_by_lim = {e.candidate.candidate_limit: e for e in eval_e}

                current_limit = float(cand_df["current_limit"].iloc[0])
                exist_rec = (
                    float(pol_e.selected.candidate_limit) if pol_e.selected is not None else None
                )
                uni_rec = (
                    float(pol_u.selected.candidate_limit) if pol_u.selected is not None else None
                )
                same_final = (
                    exist_rec is not None
                    and uni_rec is not None
                    and abs(exist_rec - uni_rec) < 1e-9
                    and pol_e.decision == pol_u.decision
                )

                elig_yields = [
                    e.evidence.simulated_yield
                    for e in eval_e
                    if e.safety.status.value == "PASS" and e.evidence.simulated_yield is not None
                ]
                max_elig_yield = max(elig_yields) if elig_yields else None

                decision_rows.append(
                    {
                        "month": month,
                        "lot_id": lot_id,
                        "die_id": die_id,
                        "parameter": parameter,
                        "domain": domain,
                        "current_limit": current_limit,
                        "existing_model": existing_model,
                        "unified_model": scorer.uni_model_id,
                        "existing_recommendation": exist_rec,
                        "unified_recommendation": uni_rec,
                        "existing_decision": pol_e.decision.value,
                        "unified_decision": pol_u.decision.value,
                        "existing_ml_rank_of_winner": (
                            pol_e.selected.ml_rank if pol_e.selected else None
                        ),
                        "unified_ml_rank_of_winner": (
                            pol_u.selected.ml_rank if pol_u.selected else None
                        ),
                        "existing_top_ml_limit": ranked_e[0].candidate_limit if ranked_e else None,
                        "unified_top_ml_limit": ranked_u[0].candidate_limit if ranked_u else None,
                        "max_eligible_yield": max_elig_yield,
                        "yield_tie_existing": pol_e.yield_tie,
                        "yield_tie_unified": pol_u.yield_tie,
                        "same_final_dtl": bool(same_final),
                        "same_ml_top": bool(
                            ranked_e
                            and ranked_u
                            and abs(ranked_e[0].candidate_limit - ranked_u[0].candidate_limit)
                            < 1e-9
                        ),
                        "n_candidates": len(cand_limits),
                    }
                )

                for lim in cand_limits:
                    ec = e_by_lim[lim]
                    uc = u_by_lim[lim]
                    ev = ev_by_lim[lim]
                    detail_rows.append(
                        {
                            "month": month,
                            "lot_id": lot_id,
                            "die_id": die_id,
                            "parameter": parameter,
                            "current_limit": current_limit,
                            "existing_model": existing_model,
                            "unified_model": scorer.uni_model_id,
                            "candidate_limit": lim,
                            "existing_ml_score": ec.ml_score,
                            "existing_ml_rank": ec.ml_rank,
                            "unified_ml_score": uc.ml_score,
                            "unified_ml_rank": uc.ml_rank,
                            "simulated_yield": ev.evidence.simulated_yield,
                            "violation_rate": ev.evidence.violation_rate,
                            "borderline_rate": ev.evidence.borderline_rate,
                            "objective_score": ev.evidence.objective_score,
                            "safety_status": ev.safety.status.value,
                            "final_recommendation_existing": exist_rec,
                            "final_recommendation_unified": uni_rec,
                            "decision_existing": pol_e.decision.value,
                            "decision_unified": pol_u.decision.value,
                            "same_final_dtl": bool(same_final),
                        }
                    )

        print(f"[Phase12.6] completed month={month} dies={len(dies)}", flush=True)

    detail = pd.DataFrame(detail_rows)
    decisions = pd.DataFrame(decision_rows)
    detail.to_csv(out_dir / "recommendation_comparison.csv", index=False)

    hashes_after = {k: _sha256_file(p) for k, p in watched.items()}
    untouched = hashes_before == hashes_after

    summary = _build_summary(detail, decisions, hashes_before, untouched, scorer)
    write_json(out_dir / "recommendation_comparison.json", summary)
    write_json(
        temporal_artifact_root(root) / "shared" / "PHASE_12_6_SHADOW_SUMMARY.json",
        {
            "verdict": summary["verdict"],
            "recommendation_agreement_pct": summary["agreement"]["recommendation_agreement_pct"],
            "ml_top_agreement_pct": summary["agreement"]["ml_top_candidate_agreement_pct"],
            "checkpoints_untouched": untouched,
            "wired_into_recommend": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return summary


def _build_summary(
    detail: pd.DataFrame,
    decisions: pd.DataFrame,
    hashes: dict[str, str | None],
    untouched: bool,
    scorer: ShadowScorers,
) -> dict[str, Any]:
    n = len(decisions)
    rec_agree = float(decisions["same_final_dtl"].mean()) if n else 0.0
    ml_agree = float(decisions["same_ml_top"].mean()) if n else 0.0

    by_param: dict[str, Any] = {}
    for p, g in decisions.groupby("parameter"):
        by_param[str(p)] = {
            "recommendation_agreement_pct": float(g["same_final_dtl"].mean()) * 100.0,
            "ml_top_agreement_pct": float(g["same_ml_top"].mean()) * 100.0,
            "n": int(len(g)),
            "n_disagree_final": int((~g["same_final_dtl"]).sum()),
            "n_disagree_ml_top": int((~g["same_ml_top"]).sum()),
        }

    by_month: dict[str, Any] = {}
    for m, g in decisions.groupby("month"):
        by_month[str(m)] = {
            "recommendation_agreement_pct": float(g["same_final_dtl"].mean()) * 100.0,
            "ml_top_agreement_pct": float(g["same_ml_top"].mean()) * 100.0,
            "n": int(len(g)),
        }

    comparison_table = []
    for month in TEMPORAL_MONTHS:
        for parameter in SHADOW_PARAMETERS:
            g = decisions[(decisions["month"] == month) & (decisions["parameter"] == parameter)]
            if g.empty:
                continue
            exist_mode = g["existing_recommendation"].mode()
            uni_mode = g["unified_recommendation"].mode()
            e_rec = float(exist_mode.iloc[0]) if len(exist_mode) else None
            u_rec = float(uni_mode.iloc[0]) if len(uni_mode) else None
            comparison_table.append(
                {
                    "month": month,
                    "parameter": parameter,
                    "current_dtl": float(g["current_limit"].iloc[0]),
                    "existing_model_recommendation": e_rec,
                    "unified_gru_recommendation": u_rec,
                    "existing_ml_rank_median": float(g["existing_ml_rank_of_winner"].median()),
                    "unified_ml_rank_median": float(g["unified_ml_rank_of_winner"].median()),
                    "max_eligible_yield": float(g["max_eligible_yield"].iloc[0])
                    if g["max_eligible_yield"].notna().any()
                    else None,
                    "same_final_dtl_pct": float(g["same_final_dtl"].mean()) * 100.0,
                    "same_final_dtl_all_dies": bool(g["same_final_dtl"].all()),
                }
            )

    disagree = decisions[~decisions["same_final_dtl"]].copy()
    disagreement_cases = []
    for _, r in disagree.head(50).iterrows():
        sub = detail[
            (detail["month"] == r["month"])
            & (detail["lot_id"] == r["lot_id"])
            & (detail["die_id"] == r["die_id"])
            & (detail["parameter"] == r["parameter"])
        ]
        disagreement_cases.append(
            {
                "month": r["month"],
                "lot_id": r["lot_id"],
                "die_id": r["die_id"],
                "parameter": r["parameter"],
                "current_limit": r["current_limit"],
                "existing_recommendation": r["existing_recommendation"],
                "unified_recommendation": r["unified_recommendation"],
                "existing_decision": r["existing_decision"],
                "unified_decision": r["unified_decision"],
                "existing_top_ml_limit": r["existing_top_ml_limit"],
                "unified_top_ml_limit": r["unified_top_ml_limit"],
                "max_eligible_yield": r["max_eligible_yield"],
                "yield_tie_existing": bool(r["yield_tie_existing"]),
                "yield_tie_unified": bool(r["yield_tie_unified"]),
                "reason": (
                    "ML tie-break selected different candidates under equal max yield"
                    if r["yield_tie_existing"] or r["yield_tie_unified"]
                    else "policy paths diverged (inspect yields/safety)"
                ),
                "candidates": sub[
                    [
                        "candidate_limit",
                        "existing_ml_score",
                        "existing_ml_rank",
                        "unified_ml_score",
                        "unified_ml_rank",
                        "simulated_yield",
                        "safety_status",
                    ]
                ].to_dict(orient="records"),
            }
        )

    trends = []
    for parameter in SHADOW_PARAMETERS:
        row: dict[str, Any] = {"parameter": parameter}
        vals = []
        for month in TEMPORAL_MONTHS:
            g = decisions[(decisions["month"] == month) & (decisions["parameter"] == parameter)]
            if g.empty:
                row[month] = None
                continue
            mode = g["unified_recommendation"].mode()
            v = float(mode.iloc[0]) if len(mode) else None
            row[month] = v
            vals.append(v)
        uniq = {v for v in vals if v is not None}
        row["trend"] = "changes_across_months" if len(uniq) > 1 else "stable"
        trends.append(row)

    temporal_examples = []
    for lot, die in (
        ("DTL_CENTER_001", "DTL_CENTER_001_D001"),
        ("DTL_EDGE_003", "DTL_EDGE_003_D001"),
    ):
        for parameter in ("ir_drop", "VMIN", "IDDQ"):
            months_block = []
            for month in TEMPORAL_MONTHS:
                g = decisions[
                    (decisions["month"] == month)
                    & (decisions["lot_id"] == lot)
                    & (decisions["die_id"] == die)
                    & (decisions["parameter"] == parameter)
                ]
                if g.empty:
                    continue
                r = g.iloc[0]
                sub = detail[
                    (detail["month"] == month)
                    & (detail["lot_id"] == lot)
                    & (detail["die_id"] == die)
                    & (detail["parameter"] == parameter)
                ].sort_values("simulated_yield", ascending=False)
                months_block.append(
                    {
                        "month": month,
                        "existing_recommendation": r["existing_recommendation"],
                        "unified_recommendation": r["unified_recommendation"],
                        "same_final_dtl": bool(r["same_final_dtl"]),
                        "max_eligible_yield": r["max_eligible_yield"],
                        "top_candidates_by_yield": sub.head(5)[
                            [
                                "candidate_limit",
                                "simulated_yield",
                                "existing_ml_score",
                                "existing_ml_rank",
                                "unified_ml_score",
                                "unified_ml_rank",
                                "safety_status",
                            ]
                        ].to_dict(orient="records"),
                    }
                )
            if months_block:
                temporal_examples.append(
                    {
                        "lot_id": lot,
                        "die_id": die,
                        "parameter": parameter,
                        "months": months_block,
                    }
                )

    ml_vs_yield = []
    for _, r in decisions.head(30).iterrows():
        ml_eq = None
        if r["existing_top_ml_limit"] is not None and r["existing_recommendation"] is not None:
            ml_eq = abs(float(r["existing_top_ml_limit"]) - float(r["existing_recommendation"])) < 1e-9
        ml_vs_yield.append(
            {
                "month": r["month"],
                "parameter": r["parameter"],
                "die_id": r["die_id"],
                "ml_top_existing": r["existing_top_ml_limit"],
                "final_existing": r["existing_recommendation"],
                "ml_equals_final": ml_eq,
                "yield_tie": bool(r["yield_tie_existing"]),
                "note": (
                    "Final follows max yield; ML top equals final only if that candidate "
                    "also has max yield (or wins a yield tie)."
                ),
            }
        )

    classifications = {}
    for p, stats in by_param.items():
        if stats["recommendation_agreement_pct"] >= 99.0 and stats["n_disagree_final"] == 0:
            classifications[p] = {
                "status": "GREEN",
                "rationale": "Final DTL matches existing path on all evaluated dies; contract valid.",
            }
        elif stats["recommendation_agreement_pct"] >= 90.0:
            classifications[p] = {
                "status": "GREEN",
                "rationale": (
                    f"High final-DTL agreement ({stats['recommendation_agreement_pct']:.1f}%); "
                    f"ML-top agreement {stats['ml_top_agreement_pct']:.1f}%."
                ),
            }
        elif stats["recommendation_agreement_pct"] >= 70.0:
            classifications[p] = {
                "status": "YELLOW",
                "rationale": (
                    f"Material final-DTL disagreement "
                    f"({100 - stats['recommendation_agreement_pct']:.1f}%); review yield-tie cases."
                ),
            }
        else:
            classifications[p] = {
                "status": "YELLOW",
                "rationale": (
                    f"Frequent final disagreement ({stats['recommendation_agreement_pct']:.1f}% agree); "
                    "engineering review before integration."
                ),
            }

    any_red = any(v["status"] == "RED" for v in classifications.values())
    any_yellow = any(v["status"] == "YELLOW" for v in classifications.values())
    if any_red:
        verdict = "FAIL — unified model should not proceed to integration"
    elif any_yellow:
        verdict = "PASS WITH CONDITIONS — specific parameters/cases require review"
    else:
        verdict = "PASS WITH CONDITIONS — ready for integration planning; shadow-only (not wired)"

    return {
        "phase": "12.6",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "wired_into_recommend": False,
        "checkpoints_untouched": untouched,
        "checkpoint_hashes": hashes,
        "models": {
            "existing_core": scorer.core_model_id,
            "existing_parametric": scorer.mlp_model_id,
            "unified": scorer.uni_model_id,
        },
        "parameters_evaluated": list(SHADOW_PARAMETERS),
        "excluded_parameters": sorted(EXCLUDED_SEQUENCE_ONLY),
        "forbidden_model_input_features": sorted(FORBIDDEN_MODEL_FEATURES),
        "n_decision_rows": int(len(decisions)),
        "n_detail_rows": int(len(detail)),
        "agreement": {
            "recommendation_agreement_pct": rec_agree * 100.0,
            "ml_top_candidate_agreement_pct": ml_agree * 100.0,
            "by_parameter": by_param,
            "by_month": by_month,
        },
        "comparison_table": comparison_table,
        "disagreement_cases": disagreement_cases,
        "three_month_trends": trends,
        "temporal_examples": temporal_examples,
        "ml_score_vs_yield": ml_vs_yield,
        "parameter_classifications": classifications,
        "verdict": verdict,
        "policy": {
            "primary": "maximum simulated_yield among eligible (safety PASS)",
            "tie_break": "ml_rank then ml_score",
            "unchanged": True,
        },
    }


if __name__ == "__main__":
    summary = run_unified_recommendation_shadow()
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "recommendation_agreement_pct": summary["agreement"][
                    "recommendation_agreement_pct"
                ],
                "ml_top_agreement_pct": summary["agreement"]["ml_top_candidate_agreement_pct"],
                "n_decisions": summary["n_decision_rows"],
                "classifications": {
                    k: v["status"] for k, v in summary["parameter_classifications"].items()
                },
            },
            indent=2,
        )
    )
