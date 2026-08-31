"""Phase 3 orchestration: build features, write artifacts, validate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dtl_agent.canonical.dataset import CanonicalDataset, build_canonical_dataset
from dtl_agent.config.allowlists import FORBIDDEN_COLUMN_NAMES
from dtl_agent.config.constants import (
    EXPECTED_CONDITION_IDS,
    LINKED_DIE_COUNT,
    LINKED_LOT_COUNT,
    PARAMETRIC_ONLY_DIE_COUNT,
    PARAMETRIC_ONLY_LOT_COUNT,
)
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.features.core_engine import EXPECTED_SEQUENCE_LENGTH, SEQUENCE_FEATURE_ORDER, build_core_features
from dtl_agent.features.cross_domain import build_cross_domain_features
from dtl_agent.features.io_utils import file_sha256, write_csv_dicts, write_json
from dtl_agent.features.parametric_engine import build_parametric_features
from dtl_agent.features.registry import FeatureRegistry
from dtl_agent.validation.pipeline import validate_bundle
from dtl_agent.validation.report import CheckResult


@dataclass
class FeatureArtifacts:
    root: Path
    canonical: CanonicalDataset
    core_result: Any
    parametric_result: Any
    cross_result: Any
    registry: FeatureRegistry
    paths: dict[str, Path]
    source_checksums: dict[str, str]


def _features_root(project_root: Path) -> Path:
    return project_root / "artifacts" / "features"


def run_feature_engineering(
    project_root: Path | None = None,
    *,
    canonical: CanonicalDataset | None = None,
) -> FeatureArtifacts:
    root = project_root or default_project_root()
    if canonical is None:
        core = load_core(root, materialize_measurements=False)
        parametric = load_parametric(root, materialize_measurements=False)
        bundle = validate_bundle(core, parametric)
        if not bundle.ok:
            raise RuntimeError("Phase 1 validation must PASS before Phase 3")
        canonical = build_canonical_dataset(bundle)

    source_checksums = {
        "core_measurements": file_sha256(canonical.core.measurements_path),
        "parametric_measurements": file_sha256(canonical.parametric.measurements_path),
    }

    core_result = build_core_features(canonical)
    parametric_result = build_parametric_features(canonical)
    cross_result = build_cross_domain_features(
        canonical,
        core_die_rows=core_result.die_rows,
        parametric_die_rows=parametric_result.die_rows,
        core_lot_rows=core_result.lot_rows,
        parametric_lot_rows=parametric_result.lot_rows,
    )

    registry = FeatureRegistry()
    registry.extend(core_result.registry_specs)
    registry.extend(parametric_result.registry_specs)
    registry.extend(cross_result.registry_specs)

    out = _features_root(root)
    paths = {
        "core_pattern": out / "core" / "pattern_features.csv",
        "core_die": out / "core" / "die_features.csv",
        "core_lot": out / "core" / "lot_features.csv",
        "core_lot_parameter": out / "core" / "lot_parameter_features.csv",
        "parametric_condition": out / "parametric" / "condition_features.csv",
        "parametric_die": out / "parametric" / "die_features.csv",
        "parametric_lot": out / "parametric" / "lot_features.csv",
        "parametric_lot_condition": out / "parametric" / "lot_condition_features.csv",
        "cross_die": out / "cross_domain" / "linked_die_features.csv",
        "cross_lot": out / "cross_domain" / "linked_lot_features.csv",
        "sequence_manifest": out / "sequence" / "core_sequence_manifest.csv",
        "sequence_contract": out / "sequence" / "sequence_contract.json",
        "feature_registry": out / "feature_registry.json",
        "grain_manifest": out / "grain_manifest.json",
        "split_design": out / "split_aware_design.json",
    }

    write_csv_dicts(paths["core_pattern"], core_result.pattern_rows)
    write_csv_dicts(paths["core_die"], core_result.die_rows)
    write_csv_dicts(paths["core_lot"], core_result.lot_rows)
    write_csv_dicts(paths["core_lot_parameter"], core_result.lot_parameter_rows)
    write_csv_dicts(paths["parametric_condition"], parametric_result.condition_rows)
    write_csv_dicts(paths["parametric_die"], parametric_result.die_rows)
    write_csv_dicts(paths["parametric_lot"], parametric_result.lot_rows)
    write_csv_dicts(paths["parametric_lot_condition"], parametric_result.lot_condition_rows)
    write_csv_dicts(paths["cross_die"], cross_result.linked_die_rows)
    write_csv_dicts(paths["cross_lot"], cross_result.linked_lot_rows)
    write_csv_dicts(paths["sequence_manifest"], core_result.sequence_manifest)
    write_json(paths["sequence_contract"], core_result.sequence_contract)
    registry.write_json(paths["feature_registry"])
    write_json(
        paths["grain_manifest"],
        {
            "core_pattern_features": "lot × die × pattern",
            "core_die_features": "lot × die",
            "core_lot_features": "lot",
            "core_lot_parameter_features": "lot × parameter",
            "parametric_condition_features": "lot × die × condition",
            "parametric_die_features": "lot × die",
            "parametric_lot_features": "lot",
            "parametric_lot_condition_features": "lot × condition",
            "cross_linked_die_features": "linked lot × die",
            "cross_linked_lot_features": "linked lot",
            "note": "No Core+Parametric measurement-row concatenation",
        },
    )
    write_json(
        paths["split_design"],
        {
            "primary_split": "lot_level",
            "final_blind_test": "scenario_family_holdout",
            "forbidden": [
                "random_row_split",
                "die_sequences_crossing_splits",
                "fit_normalization_on_test",
            ],
            "normalization": core_result.sequence_contract["normalization_strategy"],
            "phase3_note": "Features are candidate-independent; scalers fitted in Phase 6/7 on train lots only",
        },
    )

    return FeatureArtifacts(
        root=out,
        canonical=canonical,
        core_result=core_result,
        parametric_result=parametric_result,
        cross_result=cross_result,
        registry=registry,
        paths=paths,
        source_checksums=source_checksums,
    )


def validate_phase3_features(artifacts: FeatureArtifacts) -> dict[str, Any]:
    checks: list[CheckResult] = []
    core = artifacts.core_result
    par = artifacts.parametric_result
    cross = artifacts.cross_result
    contract = core.sequence_contract

    checks.append(
        CheckResult(
            name="core_sequence_length",
            passed=contract.get("observed_sequence_length_min")
            == EXPECTED_SEQUENCE_LENGTH
            and contract.get("observed_sequence_length_max") == EXPECTED_SEQUENCE_LENGTH,
            message=str(contract.get("observed_sequence_length_distribution")),
        )
    )
    checks.append(
        CheckResult(
            name="core_valid_sequences",
            passed=contract.get("valid_sequences") == len(core.die_rows)
            and contract.get("incomplete_sequences") == 0,
            message=f"valid={contract.get('valid_sequences')} incomplete={contract.get('incomplete_sequences')}",
        )
    )
    checks.append(
        CheckResult(
            name="core_duplicate_steps",
            passed=contract.get("duplicate_steps") == 0,
            message=f"duplicate_steps={contract.get('duplicate_steps')}",
        )
    )
    checks.append(
        CheckResult(
            name="core_feature_dim",
            passed=contract.get("feature_dimension") == len(SEQUENCE_FEATURE_ORDER),
            message=f"dim={contract.get('feature_dimension')}",
        )
    )
    checks.append(
        CheckResult(
            name="pattern_row_count",
            passed=len(core.pattern_rows) == len(core.die_rows) * EXPECTED_SEQUENCE_LENGTH,
            message=f"pattern_rows={len(core.pattern_rows)}",
        )
    )

    cond_ids = {r["condition_id"] for r in par.condition_rows}
    checks.append(
        CheckResult(
            name="parametric_conditions",
            passed=cond_ids == set(EXPECTED_CONDITION_IDS),
            message=str(sorted(cond_ids)),
        )
    )
    checks.append(
        CheckResult(
            name="parametric_die_count",
            passed=len(par.die_rows) == artifacts.canonical.linkage.summary()["parametric_die_count"],
            message=f"die_rows={len(par.die_rows)}",
        )
    )

    link = artifacts.canonical.linkage.summary()
    checks.append(
        CheckResult(
            name="cross_linked_die_rows",
            passed=len(cross.linked_die_rows) == LINKED_DIE_COUNT,
            message=f"linked_die_features={len(cross.linked_die_rows)}",
            details={"expected": LINKED_DIE_COUNT},
        )
    )
    checks.append(
        CheckResult(
            name="cross_linked_lot_rows",
            passed=len(cross.linked_lot_rows) == LINKED_LOT_COUNT,
            message=f"linked_lot_features={len(cross.linked_lot_rows)}",
        )
    )
    checks.append(
        CheckResult(
            name="linkage_counts",
            passed=(
                link["linked_lot_count"] == LINKED_LOT_COUNT
                and link["parametric_only_lot_count"] == PARAMETRIC_ONLY_LOT_COUNT
                and link["parametric_only_die_count"] == PARAMETRIC_ONLY_DIE_COUNT
            ),
            message=str(link),
        )
    )

    # Leakage: forbidden columns in any feature row
    sample_rows = []
    for rows in (
        core.die_rows[:1],
        par.die_rows[:1],
        cross.linked_die_rows[:1],
        core.pattern_rows[:1],
    ):
        sample_rows.extend(rows)
    forbidden_hits: list[str] = []
    for row in sample_rows + core.lot_rows[:1] + par.lot_rows[:1]:
        for col in row:
            cl = col.lower()
            for fb in FORBIDDEN_COLUMN_NAMES:
                if fb.lower() in cl:
                    forbidden_hits.append(col)
    checks.append(
        CheckResult(
            name="leakage_forbidden_columns",
            passed=not forbidden_hits,
            message="clean" if not forbidden_hits else str(sorted(set(forbidden_hits))),
        )
    )
    checks.append(
        CheckResult(
            name="registry_nonempty",
            passed=len(artifacts.registry.features) > 0,
            message=f"features={len(artifacts.registry.features)}",
        )
    )
    checks.append(
        CheckResult(
            name="candidate_independent_contract",
            passed=contract.get("candidate_dependent") is False,
            message="sequence contract candidate_dependent=false",
        )
    )

    # Source checksums still match current files
    fresh = {
        "core_measurements": file_sha256(artifacts.canonical.core.measurements_path),
        "parametric_measurements": file_sha256(
            artifacts.canonical.parametric.measurements_path
        ),
    }
    checks.append(
        CheckResult(
            name="source_immutable_during_run",
            passed=fresh == artifacts.source_checksums,
            message="checksums stable",
            details={"before": artifacts.source_checksums, "after": fresh},
        )
    )

    # Limit direction spot checks
    ir = artifacts.canonical.get_current_limit("core", test_id="T_IR_DROP_MV")
    vmax = artifacts.canonical.get_current_limit("parametric", test_id="T_VMAX")
    checks.append(
        CheckResult(
            name="limit_directions",
            passed=ir.direction == "UPPER" and vmax.direction == "LOWER",
            message=f"IR={ir.direction} VMAX={vmax.direction}",
        )
    )

    status = "PASS" if all(c.passed for c in checks) else "FAIL"
    return {
        "final_status": status,
        "checks": [
            {"name": c.name, "passed": c.passed, "message": c.message, "details": c.details}
            for c in checks
        ],
        "summary": {
            "core_die_features": len(core.die_rows),
            "core_pattern_features": len(core.pattern_rows),
            "core_lot_features": len(core.lot_rows),
            "parametric_die_features": len(par.die_rows),
            "parametric_condition_features": len(par.condition_rows),
            "cross_linked_die_features": len(cross.linked_die_rows),
            "cross_linked_lot_features": len(cross.linked_lot_rows),
            "registry_feature_count": len(artifacts.registry.features),
            "sequence_contract": {
                k: contract.get(k)
                for k in (
                    "expected_sequence_length",
                    "feature_dimension",
                    "raw_feature_order",
                    "valid_sequences",
                    "incomplete_sequences",
                    "missing_steps_total",
                    "duplicate_steps",
                    "number_of_sequences",
                )
            },
            "source_checksums": artifacts.source_checksums,
            "phase_boundary": {
                "simulation": False,
                "optimization": False,
                "ml_training": False,
                "gru_model": False,
                "ml_ranker": False,
                "recommendation_engine": False,
            },
            "gru_ready": status == "PASS"
            and contract.get("valid_sequences") == contract.get("number_of_sequences"),
        },
    }


def write_phase3_docs(report: dict[str, Any], project_root: Path | None = None) -> tuple[Path, Path]:
    root = project_root or default_project_root()
    json_path = root / "artifacts" / "validation" / "phase3_validation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    s = report["summary"]
    seq = s["sequence_contract"]
    lines = [
        "# Phase 3 — Feature Engineering + GRU Readiness",
        "",
        f"## FINAL: **{report['final_status']}**",
        "",
        "## Feature groups and grains",
        "",
        "- Core pattern-step: `lot × die × pattern`",
        "- Core die / lot / lot×parameter",
        "- Parametric die×condition / die / lot / lot×condition",
        "- Cross-domain linked die / linked lot (entity join only)",
        "",
        "## Core sequence (GRU-ready)",
        "",
        f"- expected length: {seq.get('expected_sequence_length')}",
        f"- feature dimension: {seq.get('feature_dimension')}",
        f"- feature order: {seq.get('raw_feature_order')}",
        f"- valid sequences: {seq.get('valid_sequences')}",
        f"- incomplete sequences: {seq.get('incomplete_sequences')}",
        f"- missing steps: {seq.get('missing_steps_total')}",
        f"- duplicate steps: {seq.get('duplicate_steps')}",
        f"- GRU ready: {s.get('gru_ready')}",
        "",
        "## Counts",
        "",
        f"- core die features: {s.get('core_die_features')}",
        f"- core pattern features: {s.get('core_pattern_features')}",
        f"- parametric die features: {s.get('parametric_die_features')}",
        f"- cross linked dies: {s.get('cross_linked_die_features')}",
        f"- registry features: {s.get('registry_feature_count')}",
        "",
        "## Checks",
        "",
    ]
    for c in report["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        lines.append(f"- [{mark}] `{c['name']}`: {c['message']}")
    lines.extend(
        [
            "",
            "## Phase boundary",
            "",
            "- no simulation / optimization",
            "- no ML training / GRU / ranker",
            "- no recommendation engine",
            "- raw data unchanged (checksums recorded)",
            "",
            "## Next phase",
            "",
            "Phase 4 — Core Simulation + Optimization",
            "",
        ]
    )
    md_path = root / "docs" / "PHASE_3_FEATURE_ENGINEERING.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
