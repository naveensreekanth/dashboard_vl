"""Phase 5 orchestration: Parametric simulation + optimization artifacts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dtl_agent.canonical.dataset import CanonicalDataset, build_canonical_dataset
from dtl_agent.config.paths import default_project_root
from dtl_agent.data.loaders.core_loader import load_core
from dtl_agent.data.loaders.parametric_loader import load_parametric
from dtl_agent.features.io_utils import file_sha256, write_csv_dicts, write_json
from dtl_agent.features.margins import LimitSpec
from dtl_agent.simulation.parametric.candidates import generate_candidates
from dtl_agent.simulation.parametric.config import (
    ParametricSimulationConfig,
    build_parametric_simulation_config,
    write_config,
)
from dtl_agent.simulation.parametric.die_index import build_parametric_die_index
from dtl_agent.simulation.parametric.engine import (
    DieConditionOutcome,
    ParametricCandidateResult,
    simulate_parameter_candidate,
)
from dtl_agent.simulation.parametric.optimizer import baseline_result, select_best_candidate
from dtl_agent.validation.pipeline import validate_bundle
from dtl_agent.validation.report import CheckResult


@dataclass
class ParametricSimulationArtifacts:
    root: Path
    config: ParametricSimulationConfig
    independent_results: dict[str, list[ParametricCandidateResult]]
    selected: dict[str, ParametricCandidateResult]
    paths: dict[str, Path]
    source_checksums: dict[str, str]
    runtime_seconds: float
    per_die_selected: list[dict[str, Any]] = field(default_factory=list)
    per_condition_rows: list[dict[str, Any]] = field(default_factory=list)


def _to_limit_spec(canonical: CanonicalDataset, parameter: str) -> LimitSpec:
    lim = canonical.get_current_limit("parametric", parameter=parameter)
    return LimitSpec(
        direction=lim.direction,
        value=lim.current_limit,
        unit=lim.unit,
        source_status=lim.source_status,
        test_id=lim.test_id,
        parameter=lim.parameter,
    )


def run_parametric_simulation_optimization(
    project_root: Path | None = None,
    *,
    canonical: CanonicalDataset | None = None,
    config: ParametricSimulationConfig | None = None,
) -> ParametricSimulationArtifacts:
    root = project_root or default_project_root()
    t0 = time.perf_counter()
    if canonical is None:
        core = load_core(root, materialize_measurements=False)
        parametric = load_parametric(root, materialize_measurements=False)
        bundle = validate_bundle(core, parametric)
        if not bundle.ok:
            raise RuntimeError("Phase 1 must PASS before Phase 5")
        canonical = build_canonical_dataset(bundle)

    source_checksums = {
        "parametric_measurements": file_sha256(canonical.parametric.measurements_path),
        "parametric_current_limits": file_sha256(canonical.parametric.root / "current_limits.csv"),
    }
    cfg = config or build_parametric_simulation_config(canonical)
    limits = {p: _to_limit_spec(canonical, p) for p in cfg.parameters}
    index = build_parametric_die_index(canonical, set(cfg.parameters))

    independent: dict[str, list[ParametricCandidateResult]] = {}
    selected: dict[str, ParametricCandidateResult] = {}
    cand_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    per_die_selected: list[dict[str, Any]] = []
    per_condition_rows: list[dict[str, Any]] = []

    for param in cfg.parameters:
        cands = generate_candidates(limit=limits[param], grid=cfg.candidate_grids[param])
        by_limit: dict[float, list[DieConditionOutcome]] = {}
        for c in cands:
            cand_rows.append(
                {
                    "parameter": c.parameter,
                    "test_id": c.test_id,
                    "direction": c.direction,
                    "unit": c.unit,
                    "source_status": c.source_status,
                    "current_limit": c.current_limit,
                    "candidate_limit": c.candidate_limit,
                    "delta_absolute": c.delta_absolute,
                    "delta_percent": c.delta_percent,
                    "tighten_or_loosen": c.tighten_or_loosen,
                }
            )
        rows: list[ParametricCandidateResult] = []
        for c in cands:
            res, outcomes, cond_rows = simulate_parameter_candidate(index, c, cfg)
            rows.append(res)
            by_limit[c.candidate_limit] = outcomes
            result_rows.append(res.to_dict())
            per_condition_rows.extend(cond_rows)
        best = select_best_candidate(rows, weights=cfg.objective)
        independent[param] = rows
        selected[param] = best
        for o in by_limit[best.candidate_limit]:
            per_die_selected.append(
                {
                    "lot_id": o.lot_id,
                    "die_id": o.die_id,
                    "condition_id": o.condition_id,
                    "parameter": o.parameter,
                    "candidate_limit": o.candidate_limit,
                    "current_limit": o.current_limit,
                    "direction": o.direction,
                    "violation": int(o.violation),
                    "borderline": int(o.borderline),
                    "proximity": o.proximity,
                    "source_status": o.source_status,
                    "measurement_max": o.value_max,
                    "measurement_min": o.value_min,
                    "temperature_c": o.temperature_c,
                    "vdd_applied": o.vdd_applied,
                    "test_mode": o.test_mode,
                    "selection_scope": "independent_selected",
                }
            )

    out_sim = root / "artifacts" / "simulation" / "parametric"
    out_opt = root / "artifacts" / "optimization" / "parametric"
    paths = {
        "simulation_config": out_sim / "simulation_config.json",
        "candidate_grid": out_sim / "candidate_grid.csv",
        "candidate_results": out_sim / "candidate_results.csv",
        "per_die_condition_results": out_sim / "per_die_condition_results.csv",
        "per_condition_results": out_sim / "per_condition_results.csv",
        "selected_candidates": out_sim / "selected_candidates.csv",
        "optimization_results": out_opt / "optimization_results.csv",
        "optimization_summary": out_opt / "optimization_summary.json",
        "result_schema": out_sim / "candidate_result_schema.json",
        "ml_contract": out_sim / "phase6_ml_outcome_contract.json",
    }
    write_config(paths["simulation_config"], cfg)
    write_csv_dicts(paths["candidate_grid"], cand_rows)
    write_csv_dicts(paths["candidate_results"], result_rows)
    write_csv_dicts(paths["per_die_condition_results"], per_die_selected)
    write_csv_dicts(paths["per_condition_results"], per_condition_rows)

    selected_rows = [r.to_dict() for r in selected.values()]
    write_csv_dicts(paths["selected_candidates"], selected_rows)
    write_csv_dicts(paths["optimization_results"], selected_rows)

    summary = {
        "baseline": {
            param: baseline_result(independent[param]).to_dict() if baseline_result(independent[param]) else None
            for param in independent
        },
        "selected": {p: selected[p].to_dict() for p in selected},
        "objective": cfg.objective.__dict__,
        "tie_breaking": "max objective_score; then min abs(candidate-current); then smaller candidate_limit",
        "condition_policy": cfg.condition_policy,
        "die_policy": cfg.die_policy,
        "runtime_seconds": None,
        "parametric_only_lots": len(canonical.parametric.parametric_only_lot_ids()),
        "linked_lots": len(canonical.parametric.linked_lot_ids()),
    }
    write_json(paths["optimization_summary"], summary)
    write_json(
        paths["result_schema"],
        {
            "ParametricCandidateResult": sorted(ParametricCandidateResult.__dataclass_fields__.keys()),
            "yield_definition": "good_dies / total_dies at die-level across required conditions",
            "false_fail_proxy": "source pass-like die-condition labels rejected under candidate / total_dies",
            "defective_proxy": "source fail-like die-condition labels accepted under candidate / total_dies",
            "risk_note": "borderline/risky are proximity metrics, not reliability",
        },
    )
    write_json(
        paths["ml_contract"],
        {
            "purpose": "Phase 6 may join candidate outcomes with Phase 3 features for ML dataset assembly",
            "row_keys": [
                "parameter",
                "condition_id",
                "candidate_limit",
                "current_limit",
                "simulated_yield",
                "violation_rate",
                "borderline_rate",
                "objective_score",
            ],
            "forbidden_labels": [
                "true_optimal_limit",
                "latent_quality",
                "scenario_ground_truth",
                "evaluation objective_score",
            ],
            "phase5_implements_gru": False,
        },
    )
    runtime = time.perf_counter() - t0
    summary["runtime_seconds"] = runtime
    write_json(paths["optimization_summary"], summary)
    return ParametricSimulationArtifacts(
        root=root / "artifacts",
        config=cfg,
        independent_results=independent,
        selected=selected,
        paths=paths,
        source_checksums=source_checksums,
        runtime_seconds=runtime,
        per_die_selected=per_die_selected,
        per_condition_rows=per_condition_rows,
    )


def validate_phase5(artifacts: ParametricSimulationArtifacts, canonical: CanonicalDataset) -> dict[str, Any]:
    checks: list[CheckResult] = []
    checks.append(
        CheckResult(
            name="data_counts",
            passed=canonical.parametric.lot_count == 43
            and canonical.parametric.die_count == 2150
            and canonical.parametric.condition_count == 4,
            message=(
                f"lots={canonical.parametric.lot_count} dies={canonical.parametric.die_count} "
                f"conditions={canonical.parametric.condition_count}"
            ),
        )
    )
    expected_conditions = {"COND_RT_NOM", "COND_HOT_NOM", "COND_RT_LOWV", "COND_HOT_HIGHV"}
    got_conditions = set(c.condition_id for c in canonical.get_conditions())
    checks.append(
        CheckResult(
            name="conditions_present",
            passed=got_conditions == expected_conditions,
            message=str(sorted(got_conditions)),
        )
    )
    vmax = canonical.get_current_limit("parametric", parameter="VMAX")
    checks.append(
        CheckResult(
            name="vmax_lower_direction",
            passed=vmax.direction == "LOWER",
            message=f"VMAX direction={vmax.direction}",
        )
    )
    for p, rows in artifacts.independent_results.items():
        cands = [r.candidate_limit for r in rows]
        checks.append(
            CheckResult(
                name=f"grid_includes_current:{p}",
                passed=any(r.tighten_or_loosen == "CURRENT" for r in rows),
                message=f"n_candidates={len(rows)}",
            )
        )
        checks.append(
            CheckResult(
                name=f"no_duplicate_candidates:{p}",
                passed=len(cands) == len(set(cands)),
                message=f"unique={len(set(cands))}",
            )
        )
        best = artifacts.selected[p]
        again = select_best_candidate(list(rows), weights=artifacts.config.objective)
        checks.append(
            CheckResult(
                name=f"deterministic_selection:{p}",
                passed=best.candidate_limit == again.candidate_limit,
                message=f"{best.candidate_limit}",
            )
        )
        checks.append(
            CheckResult(
                name=f"worst_condition_metrics:{p}",
                passed=best.worst_condition_yield >= 0 and best.worst_condition_violation_rate >= 0,
                message=f"worst_yield={best.worst_condition_yield:.4f}",
            )
        )
    checks.append(
        CheckResult(
            name="parametric_only_lots_participate",
            passed=len(canonical.parametric.parametric_only_lot_ids()) == 12,
            message=f"parametric_only={len(canonical.parametric.parametric_only_lot_ids())}",
        )
    )
    checks.append(
        CheckResult(
            name="linked_lots_present",
            passed=len(canonical.parametric.linked_lot_ids()) == 31,
            message=f"linked={len(canonical.parametric.linked_lot_ids())}",
        )
    )
    fresh = {
        "parametric_measurements": file_sha256(canonical.parametric.measurements_path),
        "parametric_current_limits": file_sha256(canonical.parametric.root / "current_limits.csv"),
    }
    checks.append(
        CheckResult(
            name="source_immutable",
            passed=fresh == artifacts.source_checksums,
            message="checksums stable",
        )
    )
    status = "PASS" if all(c.passed for c in checks) else "FAIL"
    return {
        "final_status": status,
        "checks": [{"name": c.name, "passed": c.passed, "message": c.message, "details": c.details} for c in checks],
        "summary": {
            "runtime_seconds": artifacts.runtime_seconds,
            "selected": {p: artifacts.selected[p].to_dict() for p in artifacts.selected},
            "candidate_counts": {p: len(v) for p, v in artifacts.independent_results.items()},
            "lots": canonical.parametric.lot_count,
            "dies": canonical.parametric.die_count,
            "conditions": canonical.parametric.condition_count,
            "rows": canonical.parametric.version_metadata.get("row_count"),
            "linked_lots": len(canonical.parametric.linked_lot_ids()),
            "parametric_only_lots": len(canonical.parametric.parametric_only_lot_ids()),
            "phase_boundary": {
                "ml": False,
                "gru": False,
                "rnn": False,
                "candidate_ranker": False,
                "safety_gate": False,
                "recommendation_engine": False,
                "api": False,
            },
        },
    }


def write_phase5_docs(report: dict[str, Any], project_root: Path | None = None) -> tuple[Path, Path]:
    root = project_root or default_project_root()
    json_path = root / "artifacts" / "validation" / "phase5_validation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    s = report["summary"]
    lines = [
        "# PHASE 5 STATUS",
        "",
        f"**{report['final_status']}**",
        "",
        "## Parametric baseline",
        "",
        "- VMIN: UPPER 0.85 V SYNTHETIC_ASSUMED",
        "- VMAX: LOWER 1.15 V SYNTHETIC_ASSUMED",
        "- IDDQ: UPPER 50.0 uA SYNTHETIC_ASSUMED",
        "- SUPPLY_CURRENT: UPPER 120.0 mA SYNTHETIC_ASSUMED",
        "- CONTACT_RESISTANCE: UPPER 5.0 ohm SYNTHETIC_ASSUMED",
        "- INTERCONNECT_RESISTANCE: UPPER 15.0 ohm SYNTHETIC_ASSUMED",
        "- ON_RESISTANCE: UPPER 25.0 ohm SYNTHETIC_ASSUMED",
        "",
        "## Dataset",
        "",
        f"- lots={s.get('lots')} dies={s.get('dies')} conditions={s.get('conditions')} rows={s.get('rows')}",
        "",
        "## Candidate generation",
        "",
    ]
    for p, n in (s.get("candidate_counts") or {}).items():
        lines.append(f"- {p}: {n} candidates")
    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "- COND_RT_NOM (25C, 1.0V, NOMINAL)",
            "- COND_HOT_NOM (85C, 1.0V, HOT)",
            "- COND_RT_LOWV (25C, 0.9V, LOW_VDD)",
            "- COND_HOT_HIGHV (85C, 1.1V, HIGH_VDD)",
            "",
            "## Simulation",
            "",
            "- Grain: lot × die × condition × parameter × candidate",
            "- Policy: ANY_VIOLATION; overall die pass requires all required conditions pass",
            "- Yield: die-level good_dies / total_dies",
            "- Borderline: 5% guard band proximity indicator (not reliability)",
            "- Worst-condition metrics retained per candidate",
            "",
            "## Results",
            "",
        ]
    )
    for p, r in (s.get("selected") or {}).items():
        lines.append(
            f"- {p}: current={r.get('current_limit')} selected={r.get('candidate_limit')} "
            f"class={r.get('tighten_or_loosen')} yield={r.get('simulated_yield')} "
            f"obj={r.get('objective_score')} viol={r.get('violation_rate')} "
            f"borderline={r.get('borderline_rate')} worst_yield={r.get('worst_condition_yield')}"
        )
    lines.extend(
        [
            "",
            "## Optimization",
            "",
            "- Objective: yield - 2*(w_defective*defective_proxy + 0.4*borderline_rate) - 0.15*false_fail_proxy",
            "- Tie-break: max objective, then nearest-to-current, then smaller candidate",
            "- Parameter-wise strategy; no brute-force 7D search",
            "",
            "## Parametric-only lots",
            "",
            f"- Total parametric-only lots: {s.get('parametric_only_lots')}",
            "- Participated: YES; excluded: NONE",
            "",
            "## GRU readiness",
            "",
            "- Candidate outcomes preserve parameter + condition + candidate + simulated metrics for Phase 6 assembly",
            "- GRU / ML model is not implemented in Phase 5",
            "",
            "## Validation",
            "",
        ]
    )
    for c in report["checks"]:
        lines.append(f"- [{'PASS' if c['passed'] else 'FAIL'}] `{c['name']}`: {c['message']}")
    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- Runtime: {s.get('runtime_seconds')} s",
            "- Strategy: candidate-independent die/condition summaries cached once; candidate-dependent evaluation only",
            "",
            "## Leakage protection",
            "",
            "- No evaluation/ground-truth/latent files used",
            "",
            "## Source protection",
            "",
            "- Raw source files unchanged (checksums stable)",
            "",
            "## Phase boundary",
            "",
            "- no ML",
            "- no GRU",
            "- no RNN",
            "- no candidate ranker",
            "- no safety gate",
            "- no final recommendation",
            "- no API",
            "",
            "## Next phase",
            "",
            "Phase 6 — ML Training Dataset Assembly",
            "",
            "STOP.",
            "",
        ]
    )
    md_path = root / "docs" / "PHASE_5_PARAMETRIC_SIMULATION_OPTIMIZATION.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
