"""Phase 2 validation for the canonical dual-grain layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dtl_agent.canonical.dataset import CanonicalDataset, build_canonical_dataset
from dtl_agent.canonical.entities import CORE_GRAIN, PARAMETRIC_GRAIN
from dtl_agent.config.constants import (
    CORE_EXPECTED_LIMITS,
    EXPECTED_CONDITION_IDS,
    LINKED_DIE_COUNT,
    LINKED_LOT_COUNT,
    PARAMETRIC_EXPECTED_LIMITS,
    PARAMETRIC_ONLY_DIE_COUNT,
    PARAMETRIC_ONLY_LOT_COUNT,
)
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.data.models.bundle import ValidatedDatasetBundle
from dtl_agent.validation.pipeline import validate_bundle
from dtl_agent.validation.report import CheckResult


@dataclass
class Phase2ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    final_status: str = "FAIL"

    @property
    def passed(self) -> bool:
        return self.final_status == "PASS"

    def finalize(self) -> None:
        self.final_status = "PASS" if all(c.passed for c in self.checks) else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_status": self.final_status,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
        }


def validate_canonical_dataset(canonical: CanonicalDataset) -> Phase2ValidationReport:
    report = Phase2ValidationReport()
    checks = report.checks
    link = canonical.linkage.summary()

    # Grain
    checks.append(
        CheckResult(
            name="core_grain",
            passed=canonical.core_grain.description == CORE_GRAIN.description
            and canonical.core_grain.natural_key_fields
            == ("lot_id", "die_id", "pattern_id", "test_id"),
            message=canonical.core_grain.description,
        )
    )
    checks.append(
        CheckResult(
            name="parametric_grain",
            passed=canonical.parametric_grain.description == PARAMETRIC_GRAIN.description
            and canonical.parametric_grain.natural_key_fields
            == ("lot_id", "die_id", "condition_id", "test_id"),
            message=canonical.parametric_grain.description,
        )
    )
    checks.append(
        CheckResult(
            name="measurements_not_concatenated",
            passed=canonical.measurements_are_separated(),
            message="core and parametric measurement views remain separate",
        )
    )

    # Identity counts vs Phase 1 datasets
    checks.append(
        CheckResult(
            name="lot_identity_preserved",
            passed=len(canonical.lots)
            == len(canonical.linkage.core_lots | canonical.linkage.parametric_lots),
            message=f"canonical_lots={len(canonical.lots)}",
            details={"expected_union": len(canonical.linkage.core_lots | canonical.linkage.parametric_lots)},
        )
    )
    checks.append(
        CheckResult(
            name="die_identity_preserved",
            passed=len(canonical.dies)
            == len(
                canonical.linkage.core_lot_die_pairs
                | canonical.linkage.parametric_lot_die_pairs
            ),
            message=f"canonical_dies={len(canonical.dies)}",
        )
    )

    # Linkage expected production numbers
    checks.append(
        CheckResult(
            name="linked_lot_count",
            passed=link["linked_lot_count"] == LINKED_LOT_COUNT,
            message=f"linked_lots={link['linked_lot_count']}",
            details={"expected": LINKED_LOT_COUNT},
        )
    )
    checks.append(
        CheckResult(
            name="linked_lot_die_pairs",
            passed=link["linked_lot_die_pair_count"] == LINKED_DIE_COUNT,
            message=f"linked_pairs={link['linked_lot_die_pair_count']}",
            details={"expected": LINKED_DIE_COUNT},
        )
    )
    checks.append(
        CheckResult(
            name="parametric_only_lots",
            passed=link["parametric_only_lot_count"] == PARAMETRIC_ONLY_LOT_COUNT,
            message=f"parametric_only_lots={link['parametric_only_lot_count']}",
            details={"expected": PARAMETRIC_ONLY_LOT_COUNT},
        )
    )
    checks.append(
        CheckResult(
            name="parametric_only_dies",
            passed=link["parametric_only_die_count"] == PARAMETRIC_ONLY_DIE_COUNT,
            message=f"parametric_only_dies={link['parametric_only_die_count']}",
            details={"expected": PARAMETRIC_ONLY_DIE_COUNT},
        )
    )
    checks.append(
        CheckResult(
            name="core_only_lots_empty",
            passed=link["core_only_lot_count"] == 0,
            message=f"core_only_lots={link['core_only_lot_count']}",
        )
    )

    # Conditions
    cond_ids = {c.condition_id for c in canonical.get_conditions()}
    checks.append(
        CheckResult(
            name="conditions_present",
            passed=cond_ids == set(EXPECTED_CONDITION_IDS),
            message=f"conditions={sorted(cond_ids)}",
            details={"expected": sorted(EXPECTED_CONDITION_IDS)},
        )
    )

    # Limits
    for exp in CORE_EXPECTED_LIMITS:
        lim = canonical.get_current_limit("core", test_id=exp.test_id)
        ok = (
            lim.direction == exp.direction
            and abs(lim.current_limit - exp.value) < 1e-9
            and lim.source_status == exp.source
        )
        checks.append(
            CheckResult(
                name=f"core_limit:{exp.test_id}",
                passed=ok,
                message=(
                    f"{lim.direction} {lim.current_limit} {lim.unit} ({lim.source_status})"
                ),
            )
        )
    for exp in PARAMETRIC_EXPECTED_LIMITS:
        lim = canonical.get_current_limit("parametric", test_id=exp.test_id)
        ok = (
            lim.direction == exp.direction
            and abs(lim.current_limit - exp.value) < 1e-9
            and lim.source_status == exp.source
        )
        checks.append(
            CheckResult(
                name=f"parametric_limit:{exp.test_id}",
                passed=ok,
                message=(
                    f"{lim.direction} {lim.current_limit} {lim.unit} ({lim.source_status})"
                ),
            )
        )

    # Cross-domain flags on parametric-only
    param_only = sorted(canonical.linkage.parametric_only_lots)
    if param_only:
        sample = param_only[0]
        lot = canonical.get_lot(sample)
        checks.append(
            CheckResult(
                name="parametric_only_cross_domain_false",
                passed=(
                    lot.cross_domain_available is False
                    and canonical.cross_domain_available(sample) is False
                ),
                message=f"lot={sample} cross_domain_available={lot.cross_domain_available}",
            )
        )
    linked_sample = sorted(canonical.linkage.linked_lots)[0]
    linked_lot = canonical.get_lot(linked_sample)
    checks.append(
        CheckResult(
            name="linked_lot_cross_domain_true",
            passed=linked_lot.cross_domain_available is True,
            message=f"lot={linked_sample}",
        )
    )

    # Natural key sample from streaming (one die)
    linked_pair = next(iter(sorted(canonical.linkage.linked_lot_die_pairs)))
    core_recs = list(
        canonical.get_core_measurements(lot_id=linked_pair[0], die_id=linked_pair[1])
    )
    par_recs = list(
        canonical.get_parametric_measurements(
            lot_id=linked_pair[0], die_id=linked_pair[1]
        )
    )
    checks.append(
        CheckResult(
            name="core_natural_keys_on_sample_die",
            passed=all(
                r.natural_key == (r.lot_id, r.die_id, r.pattern_id, r.test_id)
                for r in core_recs
            )
            and len(core_recs) > 0,
            message=f"core_rows_for_die={len(core_recs)}",
        )
    )
    checks.append(
        CheckResult(
            name="parametric_natural_keys_on_sample_die",
            passed=all(
                r.natural_key == (r.lot_id, r.die_id, r.condition_id, r.test_id)
                for r in par_recs
            )
            and len(par_recs) > 0,
            message=f"parametric_rows_for_die={len(par_recs)}",
        )
    )
    # Grains differ: pattern_id vs condition_id fields
    checks.append(
        CheckResult(
            name="sample_die_grains_differ",
            passed=(
                hasattr(core_recs[0], "pattern_id")
                and hasattr(par_recs[0], "condition_id")
                and not hasattr(core_recs[0], "condition_id")
                and not hasattr(par_recs[0], "pattern_id")
            ),
            message="core has pattern_id; parametric has condition_id",
        )
    )

    # Source handles not replaced (same object identity from construction)
    checks.append(
        CheckResult(
            name="lazy_measurement_handles",
            passed=(
                canonical.core_measurements._dataset is canonical.core
                and canonical.parametric_measurements._dataset is canonical.parametric
            ),
            message="measurement views reference Phase 1 datasets (no full table copy)",
        )
    )

    report.summary = {
        **canonical.summary(),
        "sample_linked_die": {"lot_id": linked_pair[0], "die_id": linked_pair[1]},
        "sample_core_measurement_count": len(core_recs),
        "sample_parametric_measurement_count": len(par_recs),
        "phase_boundary": {
            "feature_engineering": False,
            "simulation": False,
            "optimizer": False,
            "ml": False,
            "recommendation_engine": False,
            "api": False,
        },
    }
    report.finalize()
    return report


def run_phase2_validation(project_root: Path | None = None) -> tuple[
    CanonicalDataset, Phase2ValidationReport, ValidatedDatasetBundle
]:
    root = project_root or default_project_root()
    core = load_core(root, materialize_measurements=False)
    parametric = load_parametric(root, materialize_measurements=False)
    bundle = validate_bundle(core, parametric)
    if not bundle.ok:
        raise RuntimeError("Phase 1 validation must PASS before Phase 2")
    canonical = build_canonical_dataset(bundle)
    report = validate_canonical_dataset(canonical)
    return canonical, report, bundle


def write_phase2_artifacts(
    report: Phase2ValidationReport,
    *,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    root = project_root or default_project_root()
    json_dir = root / "artifacts" / "validation"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / "phase2_validation.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    md_path = root / "docs" / "PHASE_2_CANONICAL_LAYER.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown(report: Phase2ValidationReport) -> str:
    s = report.summary
    link = s.get("linkage", {})
    lines = [
        "# Phase 2 — Canonical Dual-Grain Layer",
        "",
        f"## FINAL: **{report.final_status}**",
        "",
        "## Architecture",
        "",
        "CanonicalDataset wraps Phase 1 `CoreDataset` + `ParametricDataset` with:",
        "",
        "- **Core measurement view** — grain `lot × die × pattern × test` (lazy stream)",
        "- **Parametric measurement view** — grain `lot × die × condition × test` (lazy stream)",
        "- Indexed **Lot**, **Die**, **Condition**, **TestDefinition**, **CurrentLimit** entities",
        "- Reused Phase 1 **SharedLotDieIndex** (no measurement-row merge)",
        "",
        "No `canonical/measurements.csv` concatenation is created.",
        "",
        "## Grains",
        "",
        f"- Core: `{s.get('core_grain')}`",
        f"- Parametric: `{s.get('parametric_grain')}`",
        f"- Measurements separated: `{s.get('measurements_separated')}`",
        "",
        "## Linkage",
        "",
        f"- linked lots: {link.get('linked_lot_count')}",
        f"- linked lot/die pairs: {link.get('linked_lot_die_pair_count')}",
        f"- Core-only lots: {link.get('core_only_lot_count')}",
        f"- Parametric-only lots: {link.get('parametric_only_lot_count')}",
        f"- Parametric-only dies: {link.get('parametric_only_die_count')}",
        "",
        "## Counts",
        "",
        f"- lots: {s.get('lot_count')}",
        f"- dies: {s.get('die_count')}",
        f"- conditions: {s.get('condition_count')}",
        f"- core tests / limits: {s.get('core_test_count')} / {s.get('core_limit_count')}",
        f"- parametric tests / limits: {s.get('parametric_test_count')} / {s.get('parametric_limit_count')}",
        "",
        "## Performance",
        "",
        f"- measurement access: `{s.get('measurement_access')}`",
        "- Dim tables are indexed in memory; measurement CSVs are streamed on demand.",
        "",
        "## Checks",
        "",
    ]
    for c in report.checks:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"- [{mark}] `{c.name}`: {c.message}")
    lines.extend(
        [
            "",
            "## Phase boundary",
            "",
            "- no feature engineering",
            "- no simulation / optimization / ML",
            "- no recommendation engine / API",
            "- raw `data/` unchanged",
            "",
            "## Next phase",
            "",
            "Phase 3 — Analysis + Feature Engineering",
            "",
        ]
    )
    return "\n".join(lines)
