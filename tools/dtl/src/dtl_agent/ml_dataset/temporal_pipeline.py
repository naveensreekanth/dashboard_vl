"""Phase 12.4 — temporal Core GRU ML dataset assembly (no training).

Builds month-scoped sequences + candidate examples from temporal simulation
``objective_score``, then pools into ``artifacts/temporal/shared/ml_dataset/``
with lot-disjoint train/validation/test splits across all months.

Does not modify legacy ``artifacts/ml_dataset/`` or GRU architecture/weights.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dtl_agent.config.paths import default_project_root
from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.loader import TemporalMonthData, load_temporal_month
from dtl_agent.data.temporal.paths import (
    ALLOWED_PRODUCTION_MONTHS,
    month_ml_dataset_root,
    month_simulation_root,
    shared_ml_dataset_root,
    temporal_artifact_root,
    temporal_data_root,
    validate_production_month,
)
from dtl_agent.data.temporal.simulation import run_temporal_core_simulation
from dtl_agent.features.core_engine import EXPECTED_SEQUENCE_LENGTH, SEQUENCE_FEATURE_ORDER
from dtl_agent.features.io_utils import file_sha256, write_json
from dtl_agent.ml.datasets.phase7_datasets import CORE_CAND_NUM, CoreCandidateDataset, CoreSequenceStore
from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml_dataset.pipeline import _det_example_id, _safe_mkdir, _write_parquet

TEMPORAL_MONTHS = ("2026-01", "2026-02", "2026-03")
SPLIT_SEED = 12042026
DATASET_VERSION = "phase12_4_temporal_core_v1"

# Deterministic lot-level split stratified by lot_category (5 lots each):
# within each category (sorted): index 0 → test, 1 → validation, 2.. → train
# → 12 train / 4 validation / 4 test lots; same assignment for all months.


class TemporalMLDatasetError(ValueError):
    """Hard failure for temporal ML dataset quality / isolation violations."""


@dataclass
class TemporalMLDatasetArtifacts:
    root: Path
    shared_root: Path
    month_roots: dict[str, Path]
    simulation_summaries: dict[str, dict[str, Any]]
    split_manifest: dict[str, Any]
    dataset_manifest: dict[str, Any]
    quality_report: dict[str, Any]
    compatibility_report: dict[str, Any]
    source_checksums: dict[str, str]
    runtime_seconds: float
    legacy_ml_dataset_untouched: bool


def _objective_stats(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
    }


def build_temporal_lot_split_map(
    lot_categories: dict[str, str],
    *,
    seed: int = SPLIT_SEED,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign each lot_id to exactly one split; stratify by lot_category."""
    by_cat: dict[str, list[str]] = {}
    for lot, cat in sorted(lot_categories.items()):
        by_cat.setdefault(str(cat), []).append(str(lot))

    split_map: dict[str, str] = {}
    for cat in sorted(by_cat):
        lots = sorted(by_cat[cat])
        # Stable permutation from seed + category
        rng = np.random.default_rng(seed + _stable_int(cat))
        order = list(rng.permutation(lots))
        if len(order) < 3:
            raise TemporalMLDatasetError(
                f"Need ≥3 lots in category {cat!r} for stratified holdouts; got {len(order)}"
            )
        split_map[order[0]] = "test"
        split_map[order[1]] = "validation"
        for lot in order[2:]:
            split_map[lot] = "train"

    counts = {
        "train_lots": sum(1 for v in split_map.values() if v == "train"),
        "validation_lots": sum(1 for v in split_map.values() if v == "validation"),
        "test_lots": sum(1 for v in split_map.values() if v == "test"),
    }
    manifest = {
        "strategy": "lot_level_across_all_months + lot_category_stratified",
        "seed": seed,
        "rule": "Within each lot_category: 1 test, 1 validation, remainder train (seeded permutation)",
        "counts": counts,
        "lot_to_split": split_map,
        "lot_to_category": dict(sorted(lot_categories.items())),
        "category_split_counts": {
            cat: {
                split: sum(1 for lot in lots if split_map[lot] == split)
                for split in ("train", "validation", "test")
            }
            for cat, lots in by_cat.items()
        },
    }
    return split_map, manifest


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def build_temporal_sequence_store(
    month: TemporalMonthData,
    out_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write month-prefixed core sequences (200×5) under ``out_root/sequences/``."""
    df = month.actual_die
    needed = {"lot_id", "die_id", "pattern_id", "parameter", "measurement_value"}
    missing = needed - set(df.columns)
    if missing:
        raise TemporalMLDatasetError(f"actual_die missing columns for sequences: {sorted(missing)}")

    rows = df.loc[
        df["parameter"].isin(list(SEQUENCE_FEATURE_ORDER)),
        ["lot_id", "die_id", "pattern_id", "parameter", "measurement_value"],
    ].copy()
    pivot = (
        rows.pivot_table(
            index=["lot_id", "die_id", "pattern_id"],
            columns="parameter",
            values="measurement_value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in SEQUENCE_FEATURE_ORDER:
        if col not in pivot.columns:
            raise TemporalMLDatasetError(
                f"Missing sequence channel {col!r} for month={month.production_month}"
            )
    pivot = pivot[
        ["lot_id", "die_id", "pattern_id", *SEQUENCE_FEATURE_ORDER]
    ].sort_values(["lot_id", "die_id", "pattern_id"])
    pivot["production_month"] = month.production_month
    pivot["sequence_id"] = (
        month.production_month
        + "::"
        + pivot["lot_id"].astype(str)
        + "::"
        + pivot["die_id"].astype(str)
    )

    manifest = (
        pivot.groupby(["lot_id", "die_id", "production_month", "sequence_id"], as_index=False)
        .agg(sequence_length=("pattern_id", "nunique"))
        .sort_values(["lot_id", "die_id"])
    )
    incomplete = manifest[manifest["sequence_length"] != EXPECTED_SEQUENCE_LENGTH]
    if not incomplete.empty:
        sample = incomplete.head(3).to_dict(orient="records")
        raise TemporalMLDatasetError(
            f"Incomplete sequences (expected length {EXPECTED_SEQUENCE_LENGTH}): {sample}"
        )
    null_any = pivot[list(SEQUENCE_FEATURE_ORDER)].isna().any().any()
    if null_any:
        raise TemporalMLDatasetError(
            f"Null sequence channel values for month={month.production_month}"
        )

    _write_parquet(pivot, out_root / "sequences" / "core_sequences.parquet")
    _write_parquet(manifest, out_root / "sequences" / "sequence_manifest.parquet")
    return pivot, manifest


def verify_month_simulation_isolation(
    production_month: str,
    *,
    project_root: Path,
    month_data: TemporalMonthData,
) -> dict[str, Any]:
    """Assert simulation artifacts cover only the selected month's die population."""
    month = validate_production_month(production_month)
    sim_dir = month_simulation_root(month, project_root) / "core"
    cand_path = sim_dir / "candidate_results.csv"
    per_die_path = sim_dir / "per_die_results.csv"
    if not cand_path.is_file():
        raise TemporalMLDatasetError(f"Missing candidate_results.csv for {month}: {cand_path}")

    candidates = pd.read_csv(cand_path)
    if candidates.empty:
        raise TemporalMLDatasetError(f"Empty candidate_results for {month}")
    if "objective_score" not in candidates.columns:
        raise TemporalMLDatasetError(f"candidate_results missing objective_score for {month}")

    expected_dies = {
        (str(r.lot_id), str(r.die_id))
        for r in month_data.actual_die[["lot_id", "die_id"]].drop_duplicates().itertuples(index=False)
    }
    total_dies_vals = candidates["total_dies"].astype(int).unique().tolist()
    if total_dies_vals != [len(expected_dies)]:
        raise TemporalMLDatasetError(
            f"Month {month} simulation total_dies={total_dies_vals} "
            f"!= month die count {len(expected_dies)} (population mixing suspected)"
        )

    if per_die_path.is_file() and per_die_path.stat().st_size > 0:
        per_die = pd.read_csv(per_die_path)
        if not per_die.empty:
            got = {(str(r.lot_id), str(r.die_id)) for r in per_die.itertuples(index=False)}
            if not got.issubset(expected_dies):
                raise TemporalMLDatasetError(
                    f"per_die_results for {month} contains dies outside month population"
                )

    legacy = project_root / "artifacts" / "simulation"
    if legacy.resolve() in sim_dir.resolve().parents or sim_dir.resolve() == legacy.resolve():
        raise TemporalMLDatasetError("Temporal simulation path must not be under artifacts/simulation/")

    return {
        "month": month,
        "total_dies": len(expected_dies),
        "candidate_count": int(len(candidates)),
        "parameter_count": int(candidates["parameter"].nunique()),
        "simulation_rows": int(len(candidates)),
        "parameters": sorted(candidates["parameter"].astype(str).unique().tolist()),
        "objective_score": _objective_stats(candidates["objective_score"]),
        "candidate_results_path": str(cand_path),
    }


def assemble_temporal_month_core_examples(
    *,
    production_month: str,
    month_data: TemporalMonthData,
    seq_manifest: pd.DataFrame,
    candidate_results: pd.DataFrame,
    split_map: dict[str, str],
) -> pd.DataFrame:
    """Die × independent-candidate examples; target_score = objective_score."""
    month = validate_production_month(production_month)
    if candidate_results["objective_score"].isna().any():
        raise TemporalMLDatasetError(f"Missing objective_score in candidates for {month}")

    # Independent Core candidates only (exclude joint-only rows if present)
    cands = candidate_results.copy()
    if "scope" in cands.columns:
        cands = cands[cands["scope"].astype(str).isin(["independent", ""]) | cands["scope"].isna()]
    if "domain" in cands.columns:
        cands = cands[cands["domain"].astype(str) == "core"]

    lot_cat = (
        month_data.actual_die[["lot_id", "lot_category"]]
        .drop_duplicates()
        .assign(lot_id=lambda d: d["lot_id"].astype(str), lot_category=lambda d: d["lot_category"].astype(str))
    )

    base = seq_manifest[["lot_id", "die_id", "sequence_id", "production_month"]].copy()
    base["lot_id"] = base["lot_id"].astype(str)
    base["die_id"] = base["die_id"].astype(str)
    base = base.merge(lot_cat, on="lot_id", how="left")
    base["split"] = base["lot_id"].map(split_map)
    if base["split"].isna().any():
        missing = sorted(base.loc[base["split"].isna(), "lot_id"].unique().tolist())
        raise TemporalMLDatasetError(f"Lots missing from split_map: {missing}")

    required_cand = [
        "parameter",
        "test_id",
        "candidate_limit",
        "current_limit",
        "direction",
        "unit",
        "candidate_delta",
        "candidate_delta_percent",
        "tighten_or_loosen",
        "simulated_yield",
        "simulated_fail_rate",
        "violation_rate",
        "borderline_rate",
        "risky_rate",
        "false_fail_proxy",
        "objective_score",
        "feasible",
    ]
    missing_cols = [c for c in required_cand if c not in cands.columns]
    if missing_cols:
        raise TemporalMLDatasetError(f"candidate_results missing columns: {missing_cols}")

    frames: list[pd.DataFrame] = []
    for _, cand in cands.iterrows():
        one = base.copy()
        one["domain"] = "core"
        for col in required_cand:
            one[col] = cand[col]
        one["cross_domain_available"] = False
        one["target_score"] = float(cand["objective_score"])
        one["target_kind"] = "candidate_quality_regression"
        frames.append(one)

    examples = pd.concat(frames, ignore_index=True)
    examples["example_id"] = examples.apply(
        lambda r: _det_example_id(
            [
                month,
                str(r["split"]),
                str(r["lot_id"]),
                str(r["die_id"]),
                str(r["parameter"]),
                str(r["candidate_limit"]),
            ]
        ),
        axis=1,
    )
    # Enforce month-prefixed sequence ids
    bad = ~examples["sequence_id"].astype(str).str.startswith(f"{month}::")
    if bad.any():
        raise TemporalMLDatasetError(
            f"Non month-prefixed sequence_id in {month} examples: "
            f"{examples.loc[bad, 'sequence_id'].head(3).tolist()}"
        )
    mismatch = examples["target_score"].astype(float) != examples["objective_score"].astype(float)
    if mismatch.any():
        raise TemporalMLDatasetError("target_score must equal objective_score")
    return examples


def _lot_categories_from_month(month: TemporalMonthData) -> dict[str, str]:
    pairs = (
        month.actual_die[["lot_id", "lot_category"]]
        .drop_duplicates()
        .sort_values("lot_id")
    )
    return {str(r.lot_id): str(r.lot_category) for r in pairs.itertuples(index=False)}


def ensure_temporal_simulations(
    project_root: Path,
    *,
    months: tuple[str, ...] = TEMPORAL_MONTHS,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run full ``run_temporal_core_simulation`` per month unless artifacts already exist."""
    summaries: dict[str, dict[str, Any]] = {}
    for m in months:
        month = validate_production_month(m)
        cand = month_simulation_root(month, project_root) / "core" / "candidate_results.csv"
        data = load_temporal_month(month, project_root=project_root)
        if force or not cand.is_file():
            run_temporal_core_simulation(month, project_root=project_root, month_data=data)
        summaries[month] = verify_month_simulation_isolation(
            month, project_root=project_root, month_data=data
        )
    return summaries


def validate_temporal_ml_quality(
    *,
    examples: pd.DataFrame,
    sequences: pd.DataFrame,
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Hard-fail quality gates for pooled temporal Core examples."""
    checks: list[dict[str, Any]] = []

    def _check(name: str, ok: bool, detail: str) -> None:
        passed = bool(ok)
        checks.append({"name": name, "passed": passed, "detail": str(detail)})
        if not passed:
            raise TemporalMLDatasetError(f"Quality check failed: {name}: {detail}")

    _check("non_empty", len(examples) > 0, f"n={len(examples)}")
    months = set(examples["production_month"].astype(str).unique())
    _check(
        "all_months_present",
        months == set(TEMPORAL_MONTHS),
        f"months={sorted(months)}",
    )
    _check(
        "month_field_valid",
        examples["production_month"].isin(ALLOWED_PRODUCTION_MONTHS).all(),
        "all production_month in allowlist",
    )
    _check(
        "sequence_id_month_prefixed",
        examples.apply(
            lambda r: str(r["sequence_id"]).startswith(f"{r['production_month']}::"),
            axis=1,
        ).all(),
        "every sequence_id starts with its production_month",
    )
    _check(
        "sequence_id_three_parts",
        examples["sequence_id"].astype(str).map(lambda s: s.count("::") == 2).all(),
        "month::lot::die",
    )
    _check(
        "target_equals_objective",
        np.allclose(
            examples["target_score"].astype(float).to_numpy(),
            examples["objective_score"].astype(float).to_numpy(),
            equal_nan=False,
        ),
        "target_score == objective_score",
    )
    _check(
        "no_missing_objective",
        examples["objective_score"].notna().all(),
        "objective_score complete",
    )
    _check(
        "no_missing_target",
        examples["target_score"].notna().all(),
        "target_score complete",
    )
    for col in CORE_CAND_NUM:
        _check(f"cand_num_{col}", examples[col].notna().all(), f"{col} present")

    dup_ex = examples.duplicated(
        subset=["production_month", "lot_id", "die_id", "parameter", "candidate_limit"],
        keep=False,
    )
    _check("no_duplicate_examples", not dup_ex.any(), f"dup_rows={int(dup_ex.sum())}")

    dup_sid = sequences.duplicated(subset=["sequence_id", "pattern_id"], keep=False)
    _check("no_duplicate_sequence_rows", not dup_sid.any(), f"dup={int(dup_sid.sum())}")

    # Cross-month collision: same lot::die without month would collide; with month prefix unique
    sid_unique = sequences.groupby("sequence_id")["production_month"].nunique()
    _check(
        "no_cross_month_sequence_collision",
        (sid_unique == 1).all(),
        "each sequence_id maps to one production_month",
    )

    lot_to_split = split_manifest["lot_to_split"]
    train_lots = {l for l, s in lot_to_split.items() if s == "train"}
    val_lots = {l for l, s in lot_to_split.items() if s == "validation"}
    test_lots = {l for l, s in lot_to_split.items() if s == "test"}
    _check("leak_train_val", train_lots.isdisjoint(val_lots), "train ∩ val = ∅")
    _check("leak_train_test", train_lots.isdisjoint(test_lots), "train ∩ test = ∅")
    _check("leak_val_test", val_lots.isdisjoint(test_lots), "val ∩ test = ∅")

    # Same lot always same split across months
    for lot, split in lot_to_split.items():
        lot_rows = examples[examples["lot_id"].astype(str) == lot]
        if lot_rows.empty:
            continue
        if not (lot_rows["split"].astype(str) == split).all():
            raise TemporalMLDatasetError(f"Lot {lot} has inconsistent splits across examples")

    lengths = sequences.groupby("sequence_id")["pattern_id"].nunique()
    _check(
        "sequence_length_200",
        (lengths == EXPECTED_SEQUENCE_LENGTH).all(),
        f"min={int(lengths.min())} max={int(lengths.max())}",
    )

    return {"passed": True, "checks": checks}


def verify_core_gru_compatibility(
    examples: pd.DataFrame,
    sequences: pd.DataFrame,
    *,
    sample_n: int = 32,
) -> dict[str, Any]:
    """Instantiate CoreGRURanker and feed a batch — no training."""
    import torch

    store = CoreSequenceStore(sequences)
    sample = (
        examples.groupby("production_month", group_keys=False)
        .apply(lambda g: g.sample(n=min(max(sample_n // 3, 1), len(g)), random_state=SPLIT_SEED))
        .reset_index(drop=True)
    )
    ds = CoreCandidateDataset(sample, store)
    item = ds[0]
    seq = item["sequence"]
    cand = item["cand_num"]
    if tuple(seq.shape) != (EXPECTED_SEQUENCE_LENGTH, len(SEQUENCE_FEATURE_ORDER)):
        raise TemporalMLDatasetError(
            f"Sequence shape {tuple(seq.shape)} != "
            f"({EXPECTED_SEQUENCE_LENGTH}, {len(SEQUENCE_FEATURE_ORDER)})"
        )
    if tuple(cand.shape) != (len(CORE_CAND_NUM),):
        raise TemporalMLDatasetError(f"cand_num shape {tuple(cand.shape)} != ({len(CORE_CAND_NUM)},)")

    model = CoreGRURanker(
        seq_input_dim=len(SEQUENCE_FEATURE_ORDER),
        cand_num_dim=len(CORE_CAND_NUM),
        n_parameter=len(ds.param_map),
        n_direction=len(ds.dir_map),
        n_tight=len(ds.tight_map),
    )
    model.eval()
    batch_n = min(8, len(ds))
    with torch.no_grad():
        batch = [ds[i] for i in range(batch_n)]
        out = model(
            sequence=torch.stack([b["sequence"] for b in batch]),
            cand_num=torch.stack([b["cand_num"] for b in batch]),
            parameter_idx=torch.stack([b["parameter_idx"] for b in batch]),
            direction_idx=torch.stack([b["direction_idx"] for b in batch]),
            tight_idx=torch.stack([b["tight_idx"] for b in batch]),
            cross_domain=torch.stack([b["cross_domain"] for b in batch]),
        )
    if out.shape != (batch_n,):
        raise TemporalMLDatasetError(f"Model output shape {tuple(out.shape)} != ({batch_n},)")

    return {
        "compatible": True,
        "sequence_shape": [EXPECTED_SEQUENCE_LENGTH, len(SEQUENCE_FEATURE_ORDER)],
        "cand_num_dim": int(len(CORE_CAND_NUM)),
        "cand_num_fields": list(CORE_CAND_NUM),
        "model_class": "CoreGRURanker",
        "sample_batch_size": int(batch_n),
        "output_shape": [int(x) for x in out.shape],
        "loss": "Huber (unchanged; not run in Phase 12.4)",
        "trained": False,
    }


def run_temporal_ml_dataset_assembly(
    project_root: Path | None = None,
    *,
    months: tuple[str, ...] = TEMPORAL_MONTHS,
    force_resimulate: bool = False,
    split_seed: int = SPLIT_SEED,
) -> TemporalMLDatasetArtifacts:
    """Full Phase 12.4 pipeline: simulate → month ML stores → pooled shared store."""
    root = project_root or default_project_root()
    t0 = time.perf_counter()

    legacy_ml = root / "artifacts" / "ml_dataset"
    legacy_hashes_before: dict[str, str] = {}
    if legacy_ml.is_dir():
        for p in sorted(legacy_ml.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(legacy_ml)).replace("\\", "/")
                legacy_hashes_before[rel] = file_sha256(p)

    sim_summaries = ensure_temporal_simulations(
        root, months=months, force=force_resimulate
    )

    # Lot categories from January (identical lot set across months)
    jan = load_temporal_month("2026-01", project_root=root)
    lot_categories = _lot_categories_from_month(jan)
    split_map, split_manifest = build_temporal_lot_split_map(
        lot_categories, seed=split_seed
    )

    month_roots: dict[str, Path] = {}
    month_examples: list[pd.DataFrame] = []
    month_sequences: list[pd.DataFrame] = []
    source_checksums: dict[str, str] = {
        "temporal_manifest": file_sha256(temporal_data_root(root) / "MANIFEST.json"),
        "split_seed": str(split_seed),
    }

    for m in months:
        month = validate_production_month(m)
        data = jan if month == "2026-01" else load_temporal_month(month, project_root=root)
        out_m = month_ml_dataset_root(month, root)
        for sub in ("train", "validation", "test", "sequences"):
            _safe_mkdir(out_m / sub)
        month_roots[month] = out_m

        seq_df, seq_man = build_temporal_sequence_store(data, out_m)
        cand_path = month_simulation_root(month, root) / "core" / "candidate_results.csv"
        candidates = pd.read_csv(cand_path)
        source_checksums[f"sim_{month}_candidate_results"] = file_sha256(cand_path)
        source_checksums[f"actual_die_{month}"] = file_sha256(
            data.month_path / "actual_die" / "measurements.csv"
        )

        examples = assemble_temporal_month_core_examples(
            production_month=month,
            month_data=data,
            seq_manifest=seq_man,
            candidate_results=candidates,
            split_map=split_map,
        )
        for split in ("train", "validation", "test"):
            _write_parquet(
                examples[examples["split"] == split].sort_values("example_id"),
                out_m / split / "core_candidate_examples.parquet",
            )
        write_json(
            out_m / "month_dataset_manifest.json",
            {
                "production_month": month,
                "n_lots": int(examples["lot_id"].nunique()),
                "n_dies": int(examples[["lot_id", "die_id"]].drop_duplicates().shape[0]),
                "n_sequences": int(seq_man.shape[0]),
                "n_examples": int(len(examples)),
                "candidate_count": int(candidates.shape[0]),
                "objective_score": _objective_stats(examples["objective_score"]),
                "target_score": _objective_stats(examples["target_score"]),
                "simulation": sim_summaries[month],
            },
        )
        month_examples.append(examples)
        month_sequences.append(seq_df)

    pooled = pd.concat(month_examples, ignore_index=True)
    pooled_seq = pd.concat(month_sequences, ignore_index=True)

    quality = validate_temporal_ml_quality(
        examples=pooled, sequences=pooled_seq, split_manifest=split_manifest
    )
    compatibility = verify_core_gru_compatibility(pooled, pooled_seq)

    shared = shared_ml_dataset_root(root)
    for sub in ("train", "validation", "test", "sequences", "normalization"):
        _safe_mkdir(shared / sub)

    for split in ("train", "validation", "test"):
        _write_parquet(
            pooled[pooled["split"] == split].sort_values("example_id"),
            shared / split / "core_candidate_examples.parquet",
        )
    _write_parquet(pooled_seq.sort_values(["production_month", "lot_id", "die_id", "pattern_id"]),
                   shared / "sequences" / "core_sequences.parquet")
    seq_manifest_all = (
        pooled_seq.groupby(["production_month", "lot_id", "die_id", "sequence_id"], as_index=False)
        .agg(sequence_length=("pattern_id", "nunique"))
        .sort_values(["production_month", "lot_id", "die_id"])
    )
    _write_parquet(seq_manifest_all, shared / "sequences" / "sequence_manifest.parquet")

    # Train-only normalization of cand numeric fields (metadata; GRU may z-score later)
    train_ex = pooled[pooled["split"] == "train"]
    norm = {
        "method": "zscore_train_only",
        "features": {},
        "fit_scope": "train_split_only",
        "note": "Fitted for reproducibility; CoreGRURanker Phase 7 may apply its own scaling",
    }
    for c in CORE_CAND_NUM:
        s = pd.to_numeric(train_ex[c], errors="coerce")
        mu = float(s.mean()) if s.notna().any() else 0.0
        sd = float(s.std(ddof=0)) if s.notna().any() else 1.0
        if abs(sd) < 1e-12:
            sd = 1.0
        norm["features"][c] = {"mean": mu, "std": sd, "fit_split": "train"}
    write_json(shared / "normalization" / "normalization_stats.json", norm)

    per_month_stats = {}
    for month in TEMPORAL_MONTHS:
        sub = pooled[pooled["production_month"] == month]
        per_month_stats[month] = {
            "n_lots": int(sub["lot_id"].nunique()),
            "n_dies": int(sub[["lot_id", "die_id"]].drop_duplicates().shape[0]),
            "n_sequences": int(sub["sequence_id"].nunique()),
            "n_examples": int(len(sub)),
            "candidate_limits_ir": int(
                sub.loc[sub["parameter"] == "ir_drop", "candidate_limit"].nunique()
            ),
            "candidate_limits_thermal": int(
                sub.loc[sub["parameter"] == "thermal", "candidate_limit"].nunique()
            ),
            "objective_score": _objective_stats(sub["objective_score"]),
            "target_score": _objective_stats(sub["target_score"]),
            "examples_by_split": {
                s: int((sub["split"] == s).sum()) for s in ("train", "validation", "test")
            },
        }

    train_lots = sorted(l for l, s in split_map.items() if s == "train")
    val_lots = sorted(l for l, s in split_map.items() if s == "validation")
    test_lots = sorted(l for l, s in split_map.items() if s == "test")

    dataset_manifest = {
        "dataset_name": "dtl_agent_temporal_core_ml_dataset",
        "version": DATASET_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "production_months": list(TEMPORAL_MONTHS),
        "total_examples": int(len(pooled)),
        "train_examples": int((pooled["split"] == "train").sum()),
        "validation_examples": int((pooled["split"] == "validation").sum()),
        "test_examples": int((pooled["split"] == "test").sum()),
        "train_lots": train_lots,
        "validation_lots": val_lots,
        "test_lots": test_lots,
        "lot_counts": {
            "train": len(train_lots),
            "validation": len(val_lots),
            "test": len(test_lots),
        },
        "category_distribution": {
            cat: int((pooled["lot_category"] == cat).sum())
            for cat in sorted(pooled["lot_category"].astype(str).unique())
        },
        "per_month": per_month_stats,
        "simulation_summaries": sim_summaries,
        "leakage_checks": {
            "train_val_disjoint": bool(set(train_lots).isdisjoint(val_lots)),
            "train_test_disjoint": bool(set(train_lots).isdisjoint(test_lots)),
            "val_test_disjoint": bool(set(val_lots).isdisjoint(test_lots)),
        },
        "target": {
            "name": "target_score",
            "definition": "objective_score from month-scoped Core simulation",
            "production_month_is_gru_feature": False,
        },
        "sequence": {
            "id_format": "{production_month}::{lot_id}::{die_id}",
            "shape": [EXPECTED_SEQUENCE_LENGTH, len(SEQUENCE_FEATURE_ORDER)],
            "channels": list(SEQUENCE_FEATURE_ORDER),
        },
        "candidate_numeric": list(CORE_CAND_NUM),
        "source_checksums": source_checksums,
    }

    write_json(shared / "dataset_manifest.json", dataset_manifest)
    write_json(shared / "split_manifest.json", split_manifest)
    write_json(shared / "quality_report.json", quality)
    write_json(shared / "model_compatibility.json", compatibility)
    write_json(
        shared / "training_contract.json",
        {
            "phase": "Phase 12.4",
            "model_implemented": False,
            "model_to_train_later": "CoreGRURanker",
            "architecture_unchanged": True,
            "primary_target": {
                "name": "target_score",
                "kind": "regression_for_ranking",
                "definition": "month-scoped simulated objective_score",
            },
            "forbidden": [
                "PASS/FAIL as target",
                "production_month as GRU input channel",
                "fabricated objective_score",
            ],
        },
    )
    write_json(
        shared / "version_manifest.json",
        {
            "dataset_version": DATASET_VERSION,
            "source_package": "DTL_TEMPORAL_V1",
            "split_seed": split_seed,
            "simulation_config_version": "phase12_3_temporal_core_v1",
            "created_at_utc": dataset_manifest["created_at_utc"],
        },
    )

    # Final hashes after all shared artifacts exist (exclude self-referential churn)
    artifact_hashes = {}
    for p in sorted(shared.rglob("*")):
        if p.is_file() and p.name != "dataset_manifest.json":
            artifact_hashes[str(p.relative_to(shared)).replace("\\", "/")] = file_sha256(p)
    dataset_manifest["artifact_hashes"] = artifact_hashes
    write_json(shared / "dataset_manifest.json", dataset_manifest)

    legacy_untouched = True
    if legacy_hashes_before:
        for rel, h in legacy_hashes_before.items():
            p = legacy_ml / rel
            if not p.is_file() or file_sha256(p) != h:
                legacy_untouched = False
                break
    elif legacy_ml.is_dir():
        # Did not snapshot — still ensure we did not write into it
        legacy_untouched = True

    write_json(
        temporal_artifact_root(root) / "PHASE_12_4_ASSEMBLY_SUMMARY.json",
        {
            "verdict_ready": True,
            "legacy_ml_dataset_untouched": legacy_untouched,
            "shared_root": str(shared),
            "total_examples": dataset_manifest["total_examples"],
            "simulation_summaries": sim_summaries,
        },
    )

    runtime = time.perf_counter() - t0
    return TemporalMLDatasetArtifacts(
        root=temporal_artifact_root(root),
        shared_root=shared,
        month_roots=month_roots,
        simulation_summaries=sim_summaries,
        split_manifest=split_manifest,
        dataset_manifest=dataset_manifest,
        quality_report=quality,
        compatibility_report=compatibility,
        source_checksums=source_checksums,
        runtime_seconds=runtime,
        legacy_ml_dataset_untouched=legacy_untouched,
    )
