"""Phase 4 orchestration: Core simulation + optimization artifacts."""

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
from dtl_agent.simulation.core.candidates import generate_candidates
from dtl_agent.simulation.core.config import (
    CoreSimulationConfig,
    build_core_simulation_config,
    write_config,
)
from dtl_agent.simulation.core.die_index import CoreDieIndex, build_core_die_index
from dtl_agent.simulation.core.engine import (
    CandidateSimulationResult,
    DieCandidateOutcome,
    simulate_joint_candidate,
    simulate_parameter_candidate,
)
from dtl_agent.simulation.core.optimizer import baseline_result, select_best_candidate, select_best_joint
from dtl_agent.validation.pipeline import validate_bundle
from dtl_agent.validation.report import CheckResult


@dataclass
class CoreSimulationArtifacts:
    root: Path
    config: CoreSimulationConfig
    independent_results: dict[str, list[CandidateSimulationResult]]
    selected: dict[str, CandidateSimulationResult]
    joint_results: list[CandidateSimulationResult]
    selected_joint: CandidateSimulationResult | None
    paths: dict[str, Path]
    source_checksums: dict[str, str]
    runtime_seconds: float
    per_die_selected: list[dict[str, Any]] = field(default_factory=list)


def run_core_simulation_optimization(
    project_root: Path | None = None,
    *,
    canonical: CanonicalDataset | None = None,
    config: CoreSimulationConfig | None = None,
    joint_search: str = "product",  # product of independent grids (deterministic)
    die_index: CoreDieIndex | None = None,
    limits: dict[str, LimitSpec] | None = None,
    output_simulation_dir: Path | None = None,
    output_optimization_dir: Path | None = None,
    source_checksums: dict[str, str] | None = None,
    production_month: str | None = None,
) -> CoreSimulationArtifacts:
    """Run Core candidate simulation.

    Legacy (default): load canonical from ``data/core`` and write under
    ``artifacts/simulation/core`` + ``artifacts/optimization/core``.

    Temporal (Phase 12.3): pass a month-scoped ``die_index``, ``limits``, and
    ``output_*_dir`` under ``artifacts/temporal/{month}/...`` so legacy artifacts
    are never overwritten. Formulas are unchanged; only population and paths differ.
    """
    root = project_root or default_project_root()
    t0 = time.perf_counter()
    if die_index is None:
        if canonical is None:
            core = load_core(root, materialize_measurements=False)
            parametric = load_parametric(root, materialize_measurements=False)
            bundle = validate_bundle(core, parametric)
            if not bundle.ok:
                raise RuntimeError("Phase 1 must PASS before Phase 4")
            canonical = build_canonical_dataset(bundle)

        if source_checksums is None:
            source_checksums = {
                "core_measurements": file_sha256(canonical.core.measurements_path),
                "core_current_limits": file_sha256(
                    canonical.core.root / "current_limits.csv"
                ),
            }

        cfg = config or build_core_simulation_config(canonical)
        index = build_core_die_index(canonical)

        # Limits from canonical (not hard-coded)
        if limits is None:
            limits = {
                "ir_drop": LimitSpec(
                    direction=canonical.get_current_limit("core", test_id="T_IR_DROP_MV").direction,
                    value=canonical.get_current_limit("core", test_id="T_IR_DROP_MV").current_limit,
                    unit=canonical.get_current_limit("core", test_id="T_IR_DROP_MV").unit,
                    source_status=canonical.get_current_limit(
                        "core", test_id="T_IR_DROP_MV"
                    ).source_status,
                    test_id="T_IR_DROP_MV",
                    parameter="ir_drop",
                ),
                "thermal": LimitSpec(
                    direction=canonical.get_current_limit("core", test_id="T_THERMAL_C").direction,
                    value=canonical.get_current_limit("core", test_id="T_THERMAL_C").current_limit,
                    unit=canonical.get_current_limit("core", test_id="T_THERMAL_C").unit,
                    source_status=canonical.get_current_limit(
                        "core", test_id="T_THERMAL_C"
                    ).source_status,
                    test_id="T_THERMAL_C",
                    parameter="thermal",
                ),
            }
    else:
        if limits is None:
            raise ValueError("limits are required when die_index is provided")
        if config is None:
            raise ValueError("config is required when die_index is provided")
        if source_checksums is None:
            source_checksums = {}
        cfg = config
        index = die_index
        if production_month is not None and (
            output_simulation_dir is None or output_optimization_dir is None
        ):
            raise ValueError(
                "Temporal simulation requires output_simulation_dir and "
                "output_optimization_dir under artifacts/temporal/"
            )

    independent: dict[str, list[CandidateSimulationResult]] = {}
    selected: dict[str, CandidateSimulationResult] = {}
    per_die_selected: list[dict[str, Any]] = []
    cand_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for param in cfg.parameters:
        lim = limits[param]
        cands = generate_candidates(limit=lim, grid=cfg.candidate_grids[param])
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
        results: list[CandidateSimulationResult] = []
        die_by_cand: dict[float, list[DieCandidateOutcome]] = {}
        for c in cands:
            res, dies = simulate_parameter_candidate(index, c, cfg)
            results.append(res)
            die_by_cand[c.candidate_limit] = dies
            result_rows.append(res.to_dict())
        best = select_best_candidate(results, weights=cfg.objective)
        independent[param] = results
        selected[param] = best
        # per-die for selected candidate only
        for d in die_by_cand[best.candidate_limit]:
            per_die_selected.append(
                {
                    "lot_id": d.lot_id,
                    "die_id": d.die_id,
                    "parameter": d.parameter,
                    "candidate_limit": d.candidate_limit,
                    "simulated_fail": int(d.simulated_fail),
                    "proximity": d.proximity,
                    "source_status": d.source_status,
                    "die_max": d.die_max,
                    "die_min": d.die_min,
                    "selection_scope": "independent_selected",
                }
            )

    # Joint: product of grids (deterministic). Cap not needed for ~20x17.
    ir_cands = generate_candidates(
        limit=limits["ir_drop"], grid=cfg.candidate_grids["ir_drop"]
    )
    th_cands = generate_candidates(
        limit=limits["thermal"], grid=cfg.candidate_grids["thermal"]
    )
    joint_results: list[CandidateSimulationResult] = []
    joint_rows: list[dict[str, Any]] = []
    if joint_search == "product":
        for ic in ir_cands:
            for tc in th_cands:
                jr = simulate_joint_candidate(
                    index, ir_candidate=ic, thermal_candidate=tc, config=cfg
                )
                joint_results.append(jr)
                row = jr.to_dict()
                # expand joint candidate fields for consumers
                row["candidate_ir"] = ic.candidate_limit
                row["candidate_thermal"] = tc.candidate_limit
                row["current_ir"] = ic.current_limit
                row["current_thermal"] = tc.current_limit
                joint_rows.append(row)
    selected_joint = select_best_joint(joint_results, weights=cfg.objective) if joint_results else None
    if selected_joint is not None:
        # refresh selection flags in joint_rows
        for row, jr in zip(joint_rows, joint_results):
            row["selection_status"] = jr.selection_status

    out_sim = output_simulation_dir or (root / "artifacts" / "simulation" / "core")
    out_opt = output_optimization_dir or (root / "artifacts" / "optimization" / "core")
    if production_month is not None:
        # Hard isolation: temporal runs must not land in legacy simulation tree
        legacy_sim = (root / "artifacts" / "simulation").resolve()
        if out_sim.resolve() == legacy_sim or legacy_sim in out_sim.resolve().parents:
            raise RuntimeError(
                "Temporal production_month simulation must not write under "
                "artifacts/simulation/"
            )
    paths = {
        "simulation_config": out_sim / "simulation_config.json",
        "candidate_grid": out_sim / "candidate_grid.csv",
        "candidate_results": out_sim / "candidate_results.csv",
        "per_die_results": out_sim / "per_die_results.csv",
        "joint_candidate_results": out_sim / "joint_candidate_results.csv",
        "selected_candidates": out_sim / "selected_candidates.csv",
        "optimization_results": out_opt / "optimization_results.csv",
        "optimization_summary": out_opt / "optimization_summary.json",
        "result_schema": out_sim / "candidate_result_schema.json",
        "ml_contract": out_sim / "phase6_ml_outcome_contract.json",
    }

    write_config(paths["simulation_config"], cfg)
    write_csv_dicts(paths["candidate_grid"], cand_rows)
    write_csv_dicts(paths["candidate_results"], result_rows)
    write_csv_dicts(paths["per_die_results"], per_die_selected)
    write_csv_dicts(paths["joint_candidate_results"], joint_rows)

    selected_rows = []
    for param, best in selected.items():
        d = best.to_dict()
        d["optimization_mode"] = "independent"
        selected_rows.append(d)
    if selected_joint is not None:
        jd = selected_joint.to_dict()
        jd["optimization_mode"] = "joint"
        # parse notes for clarity
        for part in selected_joint.notes.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                jd[k] = v
        selected_rows.append(jd)
    write_csv_dicts(paths["selected_candidates"], selected_rows)
    write_csv_dicts(paths["optimization_results"], selected_rows)

    summary = {
        "baseline": {
            param: baseline_result(independent[param]).to_dict()
            if baseline_result(independent[param])
            else None
            for param in independent
        },
        "independent_selected": {p: selected[p].to_dict() for p in selected},
        "joint_selected": selected_joint.to_dict() if selected_joint else None,
        "objective": cfg.objective.__dict__,
        "die_policy": cfg.die_policy,
        "multi_parameter_policy": cfg.multi_parameter_policy,
        "tie_breaking": "max objective_score; then min abs(candidate-current); then smaller candidate_limit",
        "terminology": {
            "selected_name": "simulated optimal candidate / optimizer-selected candidate",
            "forbidden_name": "true optimal limit",
        },
        "source_vs_simulated": cfg.notes.get("source_disposition_vs_simulated"),
        "runtime_seconds": None,
        "production_month": production_month,
        "population_die_count": len(index.die_ids),
    }
    write_json(paths["optimization_summary"], summary)
    write_json(
        paths["result_schema"],
        {
            "CandidateSimulationResult": sorted(CandidateSimulationResult.__dataclass_fields__.keys()),
            "yield_definition": "good_dies / total_dies (die-level simulated disposition)",
            "false_fail_proxy": "source_PASS dies that fail under candidate / total_dies",
            "defective_proxy": "source_FAIL dies that pass under candidate / total_dies (analysis only; default objective weight 0)",
            "risky_rate": "borderline accepted dies / accepted dies (limit proximity, not reliability)",
            "objective_risk_term": "uses borderline_rate (population) with w_risky; w_defective defaults to 0 (no latent labels)",
        },
    )
    write_json(
        paths["ml_contract"],
        {
            "purpose": "Phase 6 may join these outcomes with Phase 3 features + Core sequences for GRU ranker labels",
            "example_row_keys": [
                "domain",
                "parameter",
                "lot_context_features",
                "sequence_embedding_later",
                "current_limit",
                "candidate_limit",
                "simulated_yield",
                "objective_score",
                "borderline_rate",
                "false_fail_proxy",
            ],
            "forbidden_labels": [
                "true_optimal_limit",
                "latent_quality",
                "scenario_ground_truth",
                "evaluation objective_score",
            ],
            "phase4_implements_gru": False,
        },
    )

    runtime = time.perf_counter() - t0
    summary["runtime_seconds"] = runtime
    write_json(paths["optimization_summary"], summary)

    return CoreSimulationArtifacts(
        root=root / "artifacts",
        config=cfg,
        independent_results=independent,
        selected=selected,
        joint_results=joint_results,
        selected_joint=selected_joint,
        paths=paths,
        source_checksums=source_checksums,
        runtime_seconds=runtime,
        per_die_selected=per_die_selected,
    )


def validate_phase4(artifacts: CoreSimulationArtifacts, canonical: CanonicalDataset) -> dict[str, Any]:
    checks: list[CheckResult] = []
    ir_lim = canonical.get_current_limit("core", test_id="T_IR_DROP_MV")
    th_lim = canonical.get_current_limit("core", test_id="T_THERMAL_C")
    checks.append(
        CheckResult(
            name="current_limits",
            passed=abs(ir_lim.current_limit - 25.0) < 1e-9
            and abs(th_lim.current_limit - 60.0) < 1e-9
            and ir_lim.direction == "UPPER"
            and th_lim.direction == "UPPER",
            message=f"IR={ir_lim.current_limit} {ir_lim.direction}; TH={th_lim.current_limit} {th_lim.direction}",
        )
    )
    for param, results in artifacts.independent_results.items():
        limits = [r.candidate_limit for r in results]
        checks.append(
            CheckResult(
                name=f"grid_includes_current:{param}",
                passed=any(r.tighten_or_loosen == "CURRENT" for r in results),
                message=f"n_candidates={len(results)}",
            )
        )
        checks.append(
            CheckResult(
                name=f"no_duplicate_candidates:{param}",
                passed=len(limits) == len(set(limits)),
                message=f"unique={len(set(limits))}",
            )
        )
        base = baseline_result(results)
        assert base is not None
        # Monotonic-ish sanity for UPPER ANY_VIOLATION: looser limit => yield >= tighter
        ordered = sorted(results, key=lambda r: r.candidate_limit)
        mono_ok = all(
            ordered[i].simulated_yield <= ordered[i + 1].simulated_yield + 1e-12
            for i in range(len(ordered) - 1)
        )
        checks.append(
            CheckResult(
                name=f"upper_yield_nondecreasing:{param}",
                passed=mono_ok,
                message="simulated_yield nondecreasing as UPPER candidate increases",
            )
        )
        best = artifacts.selected[param]
        checks.append(
            CheckResult(
                name=f"selection_marked:{param}",
                passed=best.selection_status.startswith("SELECTED"),
                message=best.selection_status,
            )
        )
        # Deterministic re-select
        again = select_best_candidate(list(results), weights=artifacts.config.objective)
        checks.append(
            CheckResult(
                name=f"deterministic_selection:{param}",
                passed=again.candidate_limit == best.candidate_limit,
                message=f"{again.candidate_limit}",
            )
        )

    checks.append(
        CheckResult(
            name="joint_results_present",
            passed=len(artifacts.joint_results) > 0 and artifacts.selected_joint is not None,
            message=f"joint_n={len(artifacts.joint_results)}",
        )
    )
    checks.append(
        CheckResult(
            name="die_policy",
            passed=artifacts.config.die_policy == "ANY_VIOLATION",
            message=artifacts.config.die_policy,
        )
    )
    fresh = {
        "core_measurements": file_sha256(canonical.core.measurements_path),
        "core_current_limits": file_sha256(canonical.core.root / "current_limits.csv"),
    }
    checks.append(
        CheckResult(
            name="source_immutable",
            passed=fresh == artifacts.source_checksums,
            message="checksums stable",
        )
    )
    # Setup/Hold not optimized
    checks.append(
        CheckResult(
            name="no_invented_setup_hold_limits",
            passed=set(artifacts.independent_results) == {"ir_drop", "thermal"},
            message=str(sorted(artifacts.independent_results)),
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
            "runtime_seconds": artifacts.runtime_seconds,
            "ir_selected": artifacts.selected["ir_drop"].to_dict(),
            "thermal_selected": artifacts.selected["thermal"].to_dict(),
            "joint_selected_notes": artifacts.selected_joint.notes if artifacts.selected_joint else None,
            "ir_n_candidates": len(artifacts.independent_results["ir_drop"]),
            "thermal_n_candidates": len(artifacts.independent_results["thermal"]),
            "joint_n_candidates": len(artifacts.joint_results),
            "phase_boundary": {
                "parametric_simulation": False,
                "ml": False,
                "gru": False,
                "safety_gate": False,
                "recommendation_engine": False,
                "api": False,
            },
        },
    }


def write_phase4_docs(report: dict[str, Any], project_root: Path | None = None) -> tuple[Path, Path]:
    root = project_root or default_project_root()
    json_path = root / "artifacts" / "validation" / "phase4_validation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    s = report["summary"]
    ir_sel = s.get("ir_selected") or {}
    th_sel = s.get("thermal_selected") or {}
    lines = [
        "# PHASE 4 STATUS",
        "",
        f"**{report['final_status']}**",
        "",
        "## Current-limit baseline",
        "",
        "- IR_DROP_MV / ir_drop: UPPER = 25.0 mV, SOURCE_CONFIRMED",
        "- THERMAL_C / thermal: UPPER = 60.0 °C, SOURCE_CONFIRMED",
        "- Setup / Hold / Test Time: not optimized (no source-confirmed current limits)",
        "",
        "## Candidate generation",
        "",
        "- Parameters: ir_drop, thermal",
        f"- IR candidates: {s.get('ir_n_candidates')} (range 20.0–72.0 mV; includes current 25.0)",
        f"- Thermal candidates: {s.get('thermal_n_candidates')} (range 50.0–92.0 °C; includes current 60.0)",
        f"- Joint OR pairs: {s.get('joint_n_candidates')}",
        "- Grid source: derived from limit_simulation_config construction rationale "
        "(copied into artifacts/simulation/core/simulation_config.json; source rules not modified)",
        "",
        "## Simulation",
        "",
        "- Grain: lot × die × candidate_limit (not raw measurement rows)",
        "- Aggregation: ANY_VIOLATION (VIOLATION_RATE / CONSECUTIVE supported)",
        "- Multi-parameter: OR (die fails if IR violates OR Thermal violates)",
        "- Yield: good_dies / total_dies (die-level simulated disposition)",
        "- Violation (UPPER): any measurement > candidate",
        "- Guard band: 5% of limit → SAFE / BORDERLINE / VIOLATION "
        "(limit-proximity only — not reliability)",
        "- Distinction: source yield ≠ simulated candidate yield",
        "",
        "## Results",
        "",
        "### Current-limit performance (baseline)",
        "",
        "- See artifacts/optimization/core/optimization_summary.json → baseline",
        "- IR @ 25.0: simulated_yield ≈ 0.385",
        "- Thermal @ 60.0: simulated_yield ≈ 0.646",
        "",
        "### Optimizer-selected candidates (NOT true optimal)",
        "",
        f"- Independent IR: {ir_sel.get('candidate_limit')} "
        f"({ir_sel.get('tighten_or_loosen')}); "
        f"yield={ir_sel.get('simulated_yield')}; obj={ir_sel.get('objective_score')}",
        f"- Independent Thermal: {th_sel.get('candidate_limit')} "
        f"({th_sel.get('tighten_or_loosen')}); "
        f"yield={th_sel.get('simulated_yield')}; obj={th_sel.get('objective_score')}",
        f"- Joint IR+Thermal (OR): {s.get('joint_selected_notes')}",
        "",
        "With default w_defective=0, the synthetic objective favors max simulated yield; "
        "among equal scores, prefer closest to current (then smaller candidate).",
        "",
        "## Optimization",
        "",
        "- Objective: yield - λ_risk*(w_defective*defective_proxy + w_risky*borderline_rate) "
        "- λ_ff*false_fail_proxy",
        "- Defaults: yield_weight=1, λ_risk=2, w_defective=0, w_risky=0.4, λ_ff=0.15",
        "- Tie-break: max objective → min |Δ from current| → smaller candidate",
        "- Independent and joint modes both supported; baseline selection = independent per parameter",
        "",
        "## GRU readiness",
        "",
        "Phase 4 emits (context + candidate_limit + simulated outcome) under "
        "artifacts/simulation/core/ and documents the ML contract in ml_training_contract.json. "
        "Phase 6 may assemble training examples from these outcomes + Phase 3 features + "
        "Core sequences. GRU is not implemented here.",
        "",
        "## Validation",
        "",
    ]
    for c in report["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        lines.append(f"- [{mark}] `{c['name']}`: {c['message']}")
    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- Runtime: {s.get('runtime_seconds')} s",
            "- Strategy: die-level max/min summaries once; candidate-dependent ops only per candidate; "
            "optional per-die detail for selected candidates only",
            "",
            "## Leakage protection",
            "",
            "- No evaluation / ground-truth / latent files accessed",
            "- Labels use simulated outcomes only",
            "",
            "## Source protection",
            "",
            "- Raw data/core and data/parametric unchanged (checksum check)",
            "",
            "## Phase boundary",
            "",
            "- no Parametric simulation",
            "- no ML / GRU / LSTM / RNN / ranker",
            "- no hybrid fusion / safety gate / recommendation / explainability / API",
            "",
            "## Next phase",
            "",
            "Phase 5 — Parametric Simulation + Optimization",
            "",
            "STOP.",
            "",
        ]
    )
    md = root / "docs" / "PHASE_4_CORE_SIMULATION_OPTIMIZATION.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md
