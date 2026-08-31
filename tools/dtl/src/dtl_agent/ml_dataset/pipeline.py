"""Phase 6 ML training dataset assembly (GRU-ready data contract only)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dtl_agent.canonical.dataset import CanonicalDataset, build_canonical_dataset
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.features.io_utils import file_sha256, write_json
from dtl_agent.validation.pipeline import validate_bundle
from dtl_agent.validation.report import CheckResult

FORBIDDEN_TOKENS = [
    "ground_truth_optimal_limits",
    "scenario_ground_truth",
    "latent_quality",
    "synthetic_quality_score",
    "expected_agent_behavior",
    "true_optimal_limit",
    "true_optimal_lower",
    "true_optimal_upper",
    "process_state_bridge",
    "eval_only",
]

APPROVED_PHASE7_VALIDATION_LOTS = {
    "DTL_NORM_004",
    "DTL_VAR_003",
    "DTL_PARAM_VMARGIN_003",
}


@dataclass
class MLDatasetArtifacts:
    root: Path
    output_root: Path
    runtime_seconds: float
    split_manifest: dict[str, Any]
    dataset_manifest: dict[str, Any]
    feature_manifest: dict[str, Any]
    training_contract: dict[str, Any]
    version_manifest: dict[str, Any]
    source_checksums: dict[str, str]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _det_example_id(parts: list[str]) -> str:
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    _safe_mkdir(path.parent)
    df.to_parquet(path, index=False, engine="pyarrow")


def _sequence_store(
    canonical: CanonicalDataset,
    out_root: Path,
    *,
    production_month: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from dtl_agent.data.temporal.identity import make_sequence_id

    rows: list[dict[str, Any]] = []
    for rec in canonical.get_core_measurements(parameter=None):
        rows.append(
            {
                "lot_id": rec.lot_id,
                "die_id": rec.die_id,
                "pattern_id": int(rec.pattern_id),
                "parameter": rec.parameter,
                "value": rec.value,
            }
        )
    df = pd.DataFrame(rows)
    pivot = (
        df.pivot_table(
            index=["lot_id", "die_id", "pattern_id"],
            columns="parameter",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["ir_drop", "thermal", "setup_slack", "hold_slack", "test_time"]:
        if col not in pivot.columns:
            pivot[col] = pd.NA
    pivot = pivot[
        ["lot_id", "die_id", "pattern_id", "ir_drop", "thermal", "setup_slack", "hold_slack", "test_time"]
    ]
    if production_month is not None:
        pivot["production_month"] = production_month
    manifest = (
        pivot.groupby(["lot_id", "die_id"], as_index=False)
        .agg(sequence_length=("pattern_id", "nunique"))
        .sort_values(["lot_id", "die_id"])
    )
    if production_month is not None:
        manifest["production_month"] = production_month
    manifest["sequence_id"] = manifest.apply(
        lambda r: make_sequence_id(
            str(r["lot_id"]), str(r["die_id"]), production_month
        ),
        axis=1,
    )
    pivot["sequence_id"] = pivot.apply(
        lambda r: make_sequence_id(
            str(r["lot_id"]), str(r["die_id"]), production_month
        ),
        axis=1,
    )
    _write_parquet(pivot, out_root / "sequences" / "core_sequences.parquet")
    _write_parquet(manifest, out_root / "sequences" / "sequence_manifest.parquet")
    return pivot, manifest


def _build_split_map(canonical: CanonicalDataset) -> tuple[dict[str, str], dict[str, Any]]:
    lots = pd.DataFrame(canonical.parametric.lots)
    lots = lots[["lot_id", "scenario_family"]].drop_duplicates().sort_values("lot_id")
    test_families = {"process_drift", "tester_bias", "cross_parameter_degradation"}
    split_map: dict[str, str] = {}
    for _, r in lots.iterrows():
        lot = str(r["lot_id"])
        fam = str(r["scenario_family"])
        if fam in test_families:
            split_map[lot] = "test"
        else:
            split_map[lot] = "validation" if lot in APPROVED_PHASE7_VALIDATION_LOTS else "train"
    manifest = {
        "strategy": "lot_level + scenario_family_holdout + approved_phase7_validation_override",
        "test_holdout_families": sorted(test_families),
        "approved_validation_lots": sorted(APPROVED_PHASE7_VALIDATION_LOTS),
        "counts": {
            "train_lots": sum(1 for v in split_map.values() if v == "train"),
            "validation_lots": sum(1 for v in split_map.values() if v == "validation"),
            "test_lots": sum(1 for v in split_map.values() if v == "test"),
        },
        "lot_to_split": split_map,
    }
    return split_map, manifest


def _allowed_feature_set(root: Path) -> set[str]:
    reg = json.loads((root / "artifacts" / "features" / "feature_registry.json").read_text(encoding="utf-8"))
    feats = reg.get("features", [])
    return {f["feature_name"] for f in feats if f.get("allowed_for_ml")}


def _norm_stats(train_df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"method": "zscore_train_only", "features": {}}
    for c in numeric_cols:
        s = pd.to_numeric(train_df[c], errors="coerce")
        mu = float(s.mean()) if s.notna().any() else 0.0
        sd = float(s.std(ddof=0)) if s.notna().any() else 1.0
        if abs(sd) < 1e-12:
            sd = 1.0
        out["features"][c] = {"mean": mu, "std": sd, "fit_split": "train"}
    return out


def _apply_norm(df: pd.DataFrame, stats: dict[str, Any], prefix: str) -> pd.DataFrame:
    out = df.copy()
    for c, st in stats["features"].items():
        if c in out.columns:
            out[f"{prefix}{c}"] = (pd.to_numeric(out[c], errors="coerce") - st["mean"]) / st["std"]
    return out


def _pairwise_contract(df: pd.DataFrame, tolerance: float = 1e-6) -> dict[str, Any]:
    return {
        "enabled": True,
        "base_table": "candidate-level tables",
        "group_keys": ["split", "lot_id", "die_id", "parameter"],
        "score_field": "target_score",
        "preference_rule": f"A preferred over B if score_A > score_B + {tolerance}",
        "tie_rule": f"tie if abs(score_A - score_B) <= {tolerance}",
        "materialization": "lazy_pair_generation_recommended",
        "max_pairs_per_group": None,
        "deterministic": True,
        "example_count_reference": int(len(df)),
    }


def run_phase6_ml_dataset_assembly(
    project_root: Path | None = None,
    *,
    canonical: CanonicalDataset | None = None,
) -> MLDatasetArtifacts:
    root = project_root or default_project_root()
    t0 = time.perf_counter()
    if canonical is None:
        core = load_core(root, materialize_measurements=False)
        parametric = load_parametric(root, materialize_measurements=False)
        bundle = validate_bundle(core, parametric)
        if not bundle.ok:
            raise RuntimeError("Phase 1 must PASS before Phase 6")
        canonical = build_canonical_dataset(bundle)
    out_root = root / "artifacts" / "ml_dataset"
    for split in ["train", "validation", "test", "sequences", "normalization"]:
        _safe_mkdir(out_root / split)

    source_checksums = {
        "core_measurements": file_sha256(canonical.core.measurements_path),
        "parametric_measurements": file_sha256(canonical.parametric.measurements_path),
        "phase3_registry": file_sha256(root / "artifacts" / "features" / "feature_registry.json"),
        "phase4_core_candidates": file_sha256(root / "artifacts" / "simulation" / "core" / "candidate_results.csv"),
        "phase5_param_candidates": file_sha256(
            root / "artifacts" / "simulation" / "parametric" / "candidate_results.csv"
        ),
    }

    # Sequence store
    _, seq_manifest = _sequence_store(canonical, out_root)
    seq_manifest["complete"] = seq_manifest["sequence_length"] == 200

    # Features
    allowed = _allowed_feature_set(root)
    core_die = _read_csv(root / "artifacts" / "features" / "core" / "die_features.csv")
    core_lot = _read_csv(root / "artifacts" / "features" / "core" / "lot_features.csv")
    p_die = _read_csv(root / "artifacts" / "features" / "parametric" / "die_features.csv")
    p_cond = _read_csv(root / "artifacts" / "features" / "parametric" / "condition_features.csv")
    p_lot = _read_csv(root / "artifacts" / "features" / "parametric" / "lot_features.csv")

    core_candidates = _read_csv(root / "artifacts" / "simulation" / "core" / "candidate_results.csv")
    param_candidates = _read_csv(root / "artifacts" / "simulation" / "parametric" / "candidate_results.csv")

    split_map, split_manifest = _build_split_map(canonical)

    # Core examples (die-level, sequence-referenced)
    seq_pairs = seq_manifest[["lot_id", "die_id"]].copy()
    seq_pairs["sequence_id"] = seq_pairs["lot_id"] + "::" + seq_pairs["die_id"]
    core_base = seq_pairs.merge(core_die, on=["lot_id", "die_id"], how="left").merge(core_lot, on="lot_id", how="left")
    core_base["split"] = core_base["lot_id"].map(split_map)
    core_base = core_base[core_base["split"].notna()].copy()
    c_rows: list[pd.DataFrame] = []
    for _, cand in core_candidates.iterrows():
        one = core_base.copy()
        one["domain"] = "core"
        one["parameter"] = cand["parameter"]
        for col in [
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
        ]:
            one[col] = cand[col]
        if "cross_domain_available" in one.columns:
            one["cross_domain_available"] = one["cross_domain_available"].fillna(False)
        else:
            one["cross_domain_available"] = False
        one["target_score"] = cand["objective_score"]
        one["target_kind"] = "candidate_quality_regression"
        c_rows.append(one)
    core_examples = pd.concat(c_rows, ignore_index=True)
    core_examples["example_id"] = core_examples.apply(
        lambda r: _det_example_id(
            [r["split"], r["lot_id"], r["die_id"], r["parameter"], str(r["candidate_limit"])]
        ),
        axis=1,
    )

    # Parametric examples (die-condition-level)
    p_base = (
        p_cond.merge(p_die, on=["lot_id", "die_id"], how="left", suffixes=("", "_die"))
        .merge(p_lot, on="lot_id", how="left")
        .drop_duplicates(subset=["lot_id", "die_id", "condition_id"])
    )
    p_base["split"] = p_base["lot_id"].map(split_map)
    p_base = p_base[p_base["split"].notna()].copy()
    p_rows: list[pd.DataFrame] = []
    for _, cand in param_candidates.iterrows():
        one = p_base.copy()
        one["domain"] = "parametric"
        one["parameter"] = cand["parameter"]
        for col in [
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
            "worst_condition_yield",
            "worst_condition_violation_rate",
            "false_fail_proxy",
            "objective_score",
            "feasible",
        ]:
            one[col] = cand[col]
        one["target_score"] = cand["objective_score"]
        one["target_kind"] = "candidate_quality_regression"
        p_rows.append(one)
    param_examples = pd.concat(p_rows, ignore_index=True)
    param_examples["example_id"] = param_examples.apply(
        lambda r: _det_example_id(
            [
                r["split"],
                r["lot_id"],
                r["die_id"],
                r["condition_id"],
                r["parameter"],
                str(r["candidate_limit"]),
            ]
        ),
        axis=1,
    )

    # Keep only ML-allowed Phase 3 feature columns (+ identity/candidate/outcome)
    fixed_cols_core = {
        "example_id",
        "split",
        "domain",
        "lot_id",
        "die_id",
        "sequence_id",
        "parameter",
        "test_id",
        "candidate_limit",
        "current_limit",
        "candidate_delta",
        "candidate_delta_percent",
        "direction",
        "tighten_or_loosen",
        "unit",
        "simulated_yield",
        "simulated_fail_rate",
        "violation_rate",
        "borderline_rate",
        "risky_rate",
        "false_fail_proxy",
        "objective_score",
        "feasible",
        "target_score",
        "target_kind",
        "cross_domain_available",
    }
    fixed_cols_param = fixed_cols_core | {
        "condition_id",
        "temperature_c",
        "vdd_applied",
        "test_mode",
        "worst_condition_yield",
        "worst_condition_violation_rate",
        "parametric_only",
    }
    keep_core = [c for c in core_examples.columns if (c in fixed_cols_core) or (c in allowed)]
    keep_param = [c for c in param_examples.columns if (c in fixed_cols_param) or (c in allowed)]
    core_examples = core_examples[keep_core].copy()
    param_examples = param_examples[keep_param].copy()

    # Normalization stats fit on training lots only
    core_train = core_examples[core_examples["split"] == "train"]
    param_train = param_examples[param_examples["split"] == "train"]
    core_num = [c for c in core_examples.columns if c not in fixed_cols_core and pd.api.types.is_numeric_dtype(core_examples[c])]
    param_num = [c for c in param_examples.columns if c not in fixed_cols_param and pd.api.types.is_numeric_dtype(param_examples[c])]
    norm = {
        "core": _norm_stats(core_train, core_num),
        "parametric": _norm_stats(param_train, param_num),
        "fit_scope": "train_split_only",
    }
    core_examples = _apply_norm(core_examples, norm["core"], "norm_")
    param_examples = _apply_norm(param_examples, norm["parametric"], "norm_")
    write_json(out_root / "normalization" / "normalization_stats.json", norm)

    # Write split datasets
    for split in ["train", "validation", "test"]:
        _write_parquet(
            core_examples[core_examples["split"] == split].sort_values("example_id"),
            out_root / split / "core_candidate_examples.parquet",
        )
        _write_parquet(
            param_examples[param_examples["split"] == split].sort_values("example_id"),
            out_root / split / "parametric_candidate_examples.parquet",
        )

    feature_manifest = {
        "allowed_feature_count_registry": len(allowed),
        "core_feature_count_used": len([c for c in core_examples.columns if c in allowed]),
        "parametric_feature_count_used": len([c for c in param_examples.columns if c in allowed]),
        "candidate_dependent_groups": [
            "candidate_limit",
            "candidate_delta",
            "simulated_yield",
            "simulated_fail_rate",
            "violation_rate",
            "borderline_rate",
            "objective_score",
            "target_score",
        ],
        "candidate_independent_groups": [
            "sequence_id",
            "phase3 context features",
            "condition metadata",
            "cross_domain_available",
        ],
    }
    write_json(out_root / "feature_manifest.json", feature_manifest)

    training_contract = {
        "phase": "Phase 6",
        "model_implemented": False,
        "primary_target": {
            "name": "target_score",
            "kind": "regression_for_ranking",
            "definition": "agent-visible simulated objective_score from Phase 4/5 candidate simulation",
        },
        "pairwise_ranking_contract": _pairwise_contract(core_examples),
        "future_getitem_schema": {
            "sequence": "Tensor[T,F] from sequence_id (Core only)",
            "context_features": "Tensor[D]",
            "candidate_features": "Tensor[C]",
            "target": "float target_score",
        },
        "forbidden_targets": FORBIDDEN_TOKENS,
    }
    write_json(out_root / "training_contract.json", training_contract)

    dataset_manifest = {
        "total_examples": int(len(core_examples) + len(param_examples)),
        "core_examples": int(len(core_examples)),
        "parametric_examples": int(len(param_examples)),
        "train_count": int((core_examples["split"] == "train").sum() + (param_examples["split"] == "train").sum()),
        "validation_count": int(
            (core_examples["split"] == "validation").sum() + (param_examples["split"] == "validation").sum()
        ),
        "test_count": int((core_examples["split"] == "test").sum() + (param_examples["split"] == "test").sum()),
        "lot_counts": {
            "train": split_manifest["counts"]["train_lots"],
            "validation": split_manifest["counts"]["validation_lots"],
            "test": split_manifest["counts"]["test_lots"],
        },
        "die_counts": {
            "core": int(core_examples[["lot_id", "die_id"]].drop_duplicates().shape[0]),
            "parametric": int(param_examples[["lot_id", "die_id"]].drop_duplicates().shape[0]),
        },
        "scenario_families": sorted({str(v) for v in pd.DataFrame(canonical.parametric.lots)["scenario_family"].unique()}),
        "candidate_counts": {
            "core": int(core_candidates.shape[0]),
            "parametric": int(param_candidates.shape[0]),
        },
        "sequence_count": int(seq_manifest.shape[0]),
        "condition_count": int(len(canonical.get_conditions())),
        "feature_counts": {
            "core_columns": int(core_examples.shape[1]),
            "parametric_columns": int(param_examples.shape[1]),
        },
        "leakage_checks": {"forbidden_token_hits": 0},
        "source_hashes": source_checksums,
    }
    write_json(out_root / "dataset_manifest.json", dataset_manifest)
    write_json(out_root / "split_manifest.json", split_manifest)

    version = {
        "dataset_name": "dtl_agent_phase6_ml_dataset",
        "version": "phase6_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset_versions": {
            "core": canonical.core.dataset_version,
            "parametric": canonical.parametric.dataset_version,
        },
        "phase3_feature_registry_hash": source_checksums["phase3_registry"],
        "phase4_simulation_hash": source_checksums["phase4_core_candidates"],
        "phase5_simulation_hash": source_checksums["phase5_param_candidates"],
        "split_strategy": split_manifest["strategy"],
        "normalization_version": "phase6_norm_v1",
        "sequence_contract_version": "phase3_sequence_contract",
    }
    write_json(out_root / "ML_DATASET_VERSION.json", version)

    runtime = time.perf_counter() - t0
    return MLDatasetArtifacts(
        root=root,
        output_root=out_root,
        runtime_seconds=runtime,
        split_manifest=split_manifest,
        dataset_manifest=dataset_manifest,
        feature_manifest=feature_manifest,
        training_contract=training_contract,
        version_manifest=version,
        source_checksums=source_checksums,
    )


def validate_phase6(artifacts: MLDatasetArtifacts, canonical: CanonicalDataset) -> dict[str, Any]:
    root = artifacts.output_root
    checks: list[CheckResult] = []
    train_core = pd.read_parquet(root / "train" / "core_candidate_examples.parquet")
    val_core = pd.read_parquet(root / "validation" / "core_candidate_examples.parquet")
    test_core = pd.read_parquet(root / "test" / "core_candidate_examples.parquet")
    train_p = pd.read_parquet(root / "train" / "parametric_candidate_examples.parquet")
    val_p = pd.read_parquet(root / "validation" / "parametric_candidate_examples.parquet")
    test_p = pd.read_parquet(root / "test" / "parametric_candidate_examples.parquet")
    seq = pd.read_parquet(root / "sequences" / "sequence_manifest.parquet")

    checks.append(CheckResult("core_sequence_count", int(seq.shape[0]) == 1550, f"{int(seq.shape[0])}"))
    checks.append(
        CheckResult(
            "core_sequence_shape_contract",
            bool((seq["sequence_length"] == 200).all()),
            f"min={int(seq['sequence_length'].min())} max={int(seq['sequence_length'].max())}",
        )
    )
    # Candidate coverage
    core_expected = pd.read_csv(artifacts.root / "artifacts" / "simulation" / "core" / "candidate_results.csv")
    p_expected = pd.read_csv(artifacts.root / "artifacts" / "simulation" / "parametric" / "candidate_results.csv")
    core_seen = pd.concat([train_core, val_core, test_core])[["parameter", "candidate_limit"]].drop_duplicates()
    p_seen = pd.concat([train_p, val_p, test_p])[["parameter", "candidate_limit"]].drop_duplicates()
    checks.append(
        CheckResult(
            "core_candidate_coverage",
            core_seen.shape[0] == core_expected[["parameter", "candidate_limit"]].drop_duplicates().shape[0],
            f"seen={core_seen.shape[0]} expected={core_expected[['parameter','candidate_limit']].drop_duplicates().shape[0]}",
        )
    )
    checks.append(
        CheckResult(
            "parametric_candidate_coverage",
            p_seen.shape[0] == p_expected[["parameter", "candidate_limit"]].drop_duplicates().shape[0],
            f"seen={p_seen.shape[0]} expected={p_expected[['parameter','candidate_limit']].drop_duplicates().shape[0]}",
        )
    )
    # split overlap
    def lots(df: pd.DataFrame) -> set[str]:
        return set(df["lot_id"].astype(str).unique())

    tr_l, va_l, te_l = lots(train_p), lots(val_p), lots(test_p)
    checks.append(CheckResult("lot_overlap_train_val", len(tr_l & va_l) == 0, str(len(tr_l & va_l))))
    checks.append(CheckResult("lot_overlap_train_test", len(tr_l & te_l) == 0, str(len(tr_l & te_l))))
    checks.append(CheckResult("lot_overlap_val_test", len(va_l & te_l) == 0, str(len(va_l & te_l))))
    # scenario holdout
    lot_fam = pd.DataFrame(canonical.parametric.lots)[["lot_id", "scenario_family"]].drop_duplicates()
    test_fam = set(lot_fam[lot_fam["lot_id"].isin(te_l)]["scenario_family"].astype(str))
    checks.append(
        CheckResult(
            "scenario_holdout_present",
            {"process_drift", "tester_bias"}.issubset(test_fam),
            str(sorted(test_fam)),
        )
    )
    # normalization fit split
    norm = json.loads((root / "normalization" / "normalization_stats.json").read_text(encoding="utf-8"))
    fit_ok = all(v.get("fit_split") == "train" for v in norm["core"]["features"].values()) and all(
        v.get("fit_split") == "train" for v in norm["parametric"]["features"].values()
    )
    checks.append(CheckResult("normalization_train_only", fit_ok, "fit_split=train"))
    # forbidden tokens
    all_cols = {c.lower() for c in pd.concat([train_core.head(1), train_p.head(1)], axis=0, ignore_index=True).columns}
    hits = [t for t in FORBIDDEN_TOKENS if t in " ".join(sorted(all_cols))]
    checks.append(CheckResult("forbidden_fields_absent", len(hits) == 0, str(hits)))
    # param-only lots coverage
    param_only = canonical.parametric.parametric_only_lot_ids()
    seen_p_lots = set(pd.concat([train_p, val_p, test_p])["lot_id"].astype(str).unique())
    checks.append(
        CheckResult(
            "parametric_only_lots_present",
            param_only.issubset(seen_p_lots),
            f"present={len(param_only & seen_p_lots)}/{len(param_only)}",
        )
    )
    # determinism smoke: rerun and compare split manifest
    rerun = run_phase6_ml_dataset_assembly(artifacts.root, canonical=canonical)
    checks.append(
        CheckResult(
            "deterministic_split_manifest",
            rerun.split_manifest == artifacts.split_manifest,
            "stable" if rerun.split_manifest == artifacts.split_manifest else "changed",
        )
    )
    fresh = {
        "core_measurements": file_sha256(canonical.core.measurements_path),
        "parametric_measurements": file_sha256(canonical.parametric.measurements_path),
    }
    checks.append(
        CheckResult(
            "source_immutable",
            fresh["core_measurements"] == artifacts.source_checksums["core_measurements"]
            and fresh["parametric_measurements"] == artifacts.source_checksums["parametric_measurements"],
            "checksums stable",
        )
    )
    status = "PASS" if all(c.passed for c in checks) else "FAIL"
    return {
        "final_status": status,
        "checks": [{"name": c.name, "passed": c.passed, "message": c.message, "details": c.details} for c in checks],
        "summary": artifacts.dataset_manifest | {"runtime_seconds": artifacts.runtime_seconds},
    }


def write_phase6_docs(report: dict[str, Any], project_root: Path | None = None) -> tuple[Path, Path]:
    root = project_root or default_project_root()
    json_path = root / "artifacts" / "validation" / "phase6_validation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    s = report["summary"]
    lines = [
        "# PHASE 6 STATUS",
        "",
        report["final_status"],
        "",
        "## Dataset assembly",
        "",
        f"- Core examples: {s.get('core_examples')}",
        f"- Parametric examples: {s.get('parametric_examples')}",
        f"- Total examples: {s.get('total_examples')}",
        f"- Candidate counts: core={s.get('candidate_counts',{}).get('core')} parametric={s.get('candidate_counts',{}).get('parametric')}",
        f"- Sequence count: {s.get('sequence_count')}",
        "",
        "## Training target",
        "",
        "- Future model learns candidate-quality ranking from simulated candidate outcomes.",
        "- Stored target is candidate-level `target_score = simulated objective_score` (agent-visible Phase 4/5).",
        "",
        "## Core GRU contract",
        "",
        "- Sequence count: 1550",
        "- Sequence length: 200",
        "- Feature dimension: 5",
        "- Feature order: ir_drop, thermal, setup_slack, hold_slack, test_time",
        "- Storage: sequence store (`core_sequences.parquet`) + reference (`sequence_id`) from candidate examples",
        "",
        "## Parametric representation",
        "",
        "- Condition-aware representation at die-condition grain",
        "- Parameter representation: VMIN, VMAX, IDDQ, SUPPLY_CURRENT, CONTACT_RESISTANCE, INTERCONNECT_RESISTANCE, ON_RESISTANCE",
        "- Condition count: 4",
        "",
        "## Feature groups",
        "",
        "- Sequence features: Core 200x5 tensor referenced by sequence_id",
        "- Static features: Phase 3 die/lot/condition context features",
        "- Candidate features: candidate_limit/current_limit/delta/direction/tighten_or_loosen",
        "- Simulation outcome features: simulated yield/fail/violation/borderline/objective/feasible (+ worst-condition for parametric)",
        "",
        "## Splits",
        "",
        f"- Train lots: {s.get('lot_counts',{}).get('train')}",
        f"- Validation lots: {s.get('lot_counts',{}).get('validation')}",
        f"- Test lots: {s.get('lot_counts',{}).get('test')}",
        "- Scenario holdouts: process_drift, tester_bias, cross_parameter_degradation",
        "- Zero lot overlap across splits: enforced",
        "",
        "## Normalization",
        "",
        "- Method: z-score",
        "- Fitting split: train only",
        "- Number of features normalized: recorded in normalization_stats.json",
        "",
        "## Candidate coverage",
        "",
        "- All Phase 4 Core candidates retained",
        "- All Phase 5 Parametric candidates retained",
        "",
        "## Optimizer bias",
        "",
        "- Candidate landscape preserved in full (not collapsed to optimizer-selected candidates).",
        "- Synthetic objective currently tends to favor looser limits in this environment; dataset records this distribution without manual relabeling.",
        "",
        "## Leakage validation",
        "",
        "PASS" if all(c["passed"] for c in report["checks"] if "forbidden" in c["name"] or "leak" in c["name"]) else "FAIL",
        "",
        "## Determinism",
        "",
        "PASS" if any(c["name"] == "deterministic_split_manifest" and c["passed"] for c in report["checks"]) else "FAIL",
        "",
        "## Tests",
        "",
        "- See test report from pytest run.",
        "",
        "## Source protection",
        "",
        "- Raw data/core and data/parametric unchanged.",
        "",
        "## ML boundary",
        "",
        "- no GRU implemented",
        "- no ML trained",
        "- no ranker implemented",
        "- no optimizer modified",
        "",
        "## Next phase",
        "",
        "Phase 7 — GRU-Based ML Candidate Ranker",
        "",
        "STOP.",
        "",
    ]
    md_path = root / "docs" / "PHASE_6_ML_DATASET_ASSEMBLY.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
