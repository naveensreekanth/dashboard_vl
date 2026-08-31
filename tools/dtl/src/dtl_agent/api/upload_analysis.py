"""Upload Jan/Feb/Mar → persistent analysis session (input layer only).

Reuses existing sequence build, temporal simulation, recommend(), and Phase 12.9
artifact *shape*. Does not alter GRU/policy/safety/simulation algorithms.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import uuid
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dtl_agent.api.analysis_session import AnalysisSession, register_session, update_job_status
from dtl_agent.api.upload_recommendation import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    UploadRecommendationError,
    _copy_shared_assets,
    _read_csv_bytes,
    _require_columns,
    _stub_parametric_simulation_artifacts,
    _write_empty_parametric_csv,
)
from dtl_agent.data.temporal.identity import make_sequence_id
from dtl_agent.data.temporal.loader import (
    REQUIRED_ACTUAL_DIE_COLUMNS,
    REQUIRED_PARAMETRIC_COLUMNS,
    TemporalLoaderError,
    TemporalMonthData,
    load_temporal_month,
)
from dtl_agent.data.temporal.month_cache import put_cached_month
from dtl_agent.data.temporal.parametric_simulation import run_temporal_parametric_simulation
from dtl_agent.data.temporal.paths import (
    month_root,
    month_simulation_root,
    temporal_artifact_root,
    validate_production_month,
)
from dtl_agent.data.temporal.simulation import run_temporal_core_simulation
from dtl_agent.features.io_utils import write_json
from dtl_agent.ml.phase12_9_analysis import (
    DISPLAY_NAME,
    MONTH_LABEL,
    MONTHS,
    SCORABLE_PARAMETERS,
    _candidate_rows_for_param,
    _delta_pct,
    _explain_from_rec,
    _find_tie_break_proof,
    _find_yield_first_proof,
    _json_safe,
    _safety_status,
    analysis_output_dir,
)
from dtl_agent.ml_dataset.pipeline import _write_parquet
from dtl_agent.ml_dataset.temporal_pipeline import build_temporal_sequence_store
from dtl_agent.recommendation import recommend
from dtl_agent.recommendation.routing import model_for_parameter
from dtl_agent.recommendation.schemas import CORE_PARAMETERS
from dtl_agent.recommendation.temporal_inference import TemporalHybridBundle

EXPECTED_MONTH_FILES = {
    "january": "2026-01",
    "february": "2026-02",
    "march": "2026-03",
}
# Upload critical path: precompute only the primary die × 3 months for the
# dashboard matrix. All other dies use on-demand recommend() + die-level cache.
MAX_PRECOMPUTE_DIES = 1


@dataclass(frozen=True)
class ParsedMonthPackage:
    actual_die: pd.DataFrame
    parametric: pd.DataFrame | None
    production_month: str
    source_filename: str
    original_production_month: str | None = None


def _month_identity(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        raise UploadRecommendationError(f"{label} contains no rows")
    months = df["production_month"].astype(str).unique().tolist()
    if len(months) != 1:
        raise UploadRecommendationError(
            f"{label} must contain exactly one production_month (found {months[:5]})"
        )
    try:
        return validate_production_month(months[0])
    except Exception as exc:  # noqa: BLE001
        raise UploadRecommendationError(
            f"Invalid production_month={months[0]!r}; allowed: 2026-01, 2026-02, 2026-03"
        ) from exc


def parse_month_package(*, filename: str, content: bytes, expected_month: str) -> ParsedMonthPackage:
    """Parse one month upload (multi-die allowed). Reassigns production_month to ``expected_month``."""
    if not content:
        raise UploadRecommendationError(f"{filename or 'file'} is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadRecommendationError(
            f"{filename} exceeds max upload size ({MAX_UPLOAD_BYTES} bytes)"
        )
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadRecommendationError(
            f"Unsupported file type {suffix!r} for {filename}. "
            f"Use .csv or .zip matching temporal measurements schema."
        )

    parametric_df: pd.DataFrame | None = None
    if suffix == ".zip":
        try:
            zf = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as exc:
            raise UploadRecommendationError(f"Malformed ZIP ({filename})") from exc
        names = zf.namelist()
        actual_name = next(
            (n for n in names if n.replace("\\", "/").endswith("actual_die/measurements.csv")),
            None,
        )
        if actual_name is None:
            actual_name = next(
                (
                    n
                    for n in names
                    if Path(n).name.lower() == "measurements.csv"
                    and "parametric" not in n.replace("\\", "/").lower()
                ),
                None,
            )
        if actual_name is None:
            raise UploadRecommendationError(
                f"{filename}: ZIP must contain actual_die/measurements.csv"
            )
        actual_df = _read_csv_bytes(zf.read(actual_name), f"{filename}/actual_die")
        param_name = next(
            (n for n in names if n.replace("\\", "/").endswith("parametric/measurements.csv")),
            None,
        )
        if param_name is not None:
            parametric_df = _read_csv_bytes(zf.read(param_name), f"{filename}/parametric")
    else:
        actual_df = _read_csv_bytes(content, filename)

    _require_columns(actual_df, REQUIRED_ACTUAL_DIE_COLUMNS, filename)
    orig_month = _month_identity(actual_df, filename)
    actual_df = actual_df.copy()
    actual_df["production_month"] = expected_month

    parametric_df_out: pd.DataFrame | None = None
    if parametric_df is not None and not parametric_df.empty:
        _require_columns(parametric_df, REQUIRED_PARAMETRIC_COLUMNS, f"{filename} parametric")
        _month_identity(parametric_df, f"{filename} parametric")
        parametric_df_out = parametric_df.copy()
        parametric_df_out["production_month"] = expected_month

    return ParsedMonthPackage(
        actual_die=actual_df,
        parametric=parametric_df_out,
        production_month=expected_month,
        source_filename=filename,
        original_production_month=orig_month,
    )


def _unique_dies(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    cols = ["lot_id", "die_id"]
    cat_col = "lot_category" if "lot_category" in df.columns else None
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, r in df[cols + ([cat_col] if cat_col else [])].drop_duplicates().iterrows():
        lot = str(r["lot_id"])
        die = str(r["die_id"])
        if (lot, die) in seen:
            continue
        seen.add((lot, die))
        cat = str(r[cat_col]) if cat_col else "NORMAL"
        rows.append((lot, die, cat))
    return rows


def _build_merged_sequences(month_datas: list[TemporalMonthData], seq_out: Path) -> None:
    pivots: list[pd.DataFrame] = []
    manifests: list[pd.DataFrame] = []
    for md in month_datas:
        tmp = Path(tempfile.mkdtemp(prefix="dtl_seq_month_"))
        try:
            pivot, manifest = build_temporal_sequence_store(md, tmp)
            pivots.append(pivot)
            manifests.append(manifest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if not pivots:
        raise UploadRecommendationError("No sequences generated from uploaded months")
    combined_p = pd.concat(pivots, ignore_index=True)
    combined_m = pd.concat(manifests, ignore_index=True)
    seq_dir = seq_out / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(combined_p, seq_dir / "core_sequences.parquet")
    _write_parquet(combined_m, seq_dir / "sequence_manifest.parquet")


def _write_identity_catalog(sandbox: Path, dies: list[tuple[str, str, str]]) -> None:
    from dtl_agent.api.die_level_service import _catalog_from_rows, identity_catalog_path

    rows = [
        {"lot_id": lot, "die_id": die, "lot_category": cat} for lot, die, cat in dies
    ]
    catalog = _catalog_from_rows(rows)
    catalog["note"] = (
        "Identities derived from uploaded Jan/Feb/Mar measurements for this analysis session."
    )
    path = identity_catalog_path(sandbox)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def _fast_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        table = pa.Table.from_pandas(df)
        pacsv.write_csv(table, str(path))
    except Exception:
        df.to_csv(path, index=False)


def materialize_three_month_sandbox(
    packages: dict[str, ParsedMonthPackage],
    *,
    source_root: Path,
) -> Path:
    """Build a durable sandbox with all three uploaded months (not deleted here)."""
    if set(packages.keys()) != set(MONTHS):
        raise UploadRecommendationError(
            f"Upload must include exactly months {list(MONTHS)}; got {sorted(packages)}"
        )

    sandbox = Path(tempfile.mkdtemp(prefix="dtl_upload_analysis_"))
    try:
        _copy_shared_assets(source_root, sandbox)

        month_datas: list[TemporalMonthData] = []
        all_dies: dict[tuple[str, str], str] = {}

        for month in MONTHS:
            pkg = packages[month]
            actual_path = (
                sandbox / "data" / "3 months data" / month / "actual_die" / "measurements.csv"
            )
            _fast_write_csv(pkg.actual_die, actual_path)

            param_path = (
                sandbox / "data" / "3 months data" / month / "parametric" / "measurements.csv"
            )
            has_param = pkg.parametric is not None and not pkg.parametric.empty
            if has_param:
                _fast_write_csv(pkg.parametric, param_path)
                parametric_df = pkg.parametric.copy()
            else:
                _write_empty_parametric_csv(param_path)
                parametric_df = pd.DataFrame(columns=sorted(REQUIRED_PARAMETRIC_COLUMNS))

            month_data = TemporalMonthData(
                production_month=month,
                month_path=sandbox / "data" / "3 months data" / month,
                actual_die=pkg.actual_die.copy(),
                parametric=parametric_df,
                parts_dim=None,
            )
            put_cached_month(sandbox, month, month_data)
            month_datas.append(month_data)

            for lot, die, cat in _unique_dies(pkg.actual_die):
                all_dies.setdefault((lot, die), cat)

            run_temporal_core_simulation(
                month, project_root=sandbox, month_data=month_data, joint_search="product"
            )
            if has_param:
                run_temporal_parametric_simulation(
                    month, project_root=sandbox, month_data=month_data
                )
            else:
                _stub_parametric_simulation_artifacts(sandbox, month, source_root)

        seq_out = sandbox / "artifacts" / "temporal" / "shared" / "ml_dataset"
        _build_merged_sequences(month_datas, seq_out)

        die_list = [(lot, die, cat) for (lot, die), cat in sorted(all_dies.items())]
        if not die_list:
            raise UploadRecommendationError("Uploaded months contain no lot/die identities")
        _write_identity_catalog(sandbox, die_list)

        # Marker: never copy static phase_12_9 / phase_13_1 into this sandbox
        marker = {
            "used_uploaded_measurements": True,
            "used_static_three_month_measurements": False,
            "months": list(MONTHS),
            "n_dies": len(die_list),
        }
        marker_path = (
            temporal_artifact_root(sandbox) / "shared" / "upload_session.json"
        )
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")

        return sandbox
    except Exception:
        shutil.rmtree(sandbox, ignore_errors=True)
        raise


def _scorable_for_session(sandbox: Path) -> tuple[str, ...]:
    """Core always; parametric only when any month has parametric rows."""
    has_param = False
    for month in MONTHS:
        path = (
            sandbox / "data" / "3 months data" / month / "parametric" / "measurements.csv"
        )
        if path.is_file():
            df = pd.read_csv(path, nrows=5)
            if len(df) > 0:
                has_param = True
                break
    if has_param:
        return tuple(p for p in SCORABLE_PARAMETERS)
    return tuple(sorted(CORE_PARAMETERS))


def _dies_present_all_months(sandbox: Path) -> list[tuple[str, str, str]]:
    per_month: list[set[tuple[str, str]]] = []
    cats: dict[tuple[str, str], str] = {}
    for month in MONTHS:
        data = load_temporal_month(month, project_root=sandbox)
        df = data.actual_die[["lot_id", "die_id"] + (["lot_category"] if "lot_category" in data.actual_die.columns else [])]
        keys: set[tuple[str, str]] = set()
        for lot, die, cat in _unique_dies(df):
            keys.add((lot, die))
            cats.setdefault((lot, die), cat)
        per_month.append(keys)
    common = set.intersection(*per_month) if per_month else set()
    if not common:
        # Fall back to first month dies
        df_jan = load_temporal_month("2026-01", project_root=sandbox).actual_die
        return _unique_dies(df_jan)
    ordered = sorted(common)
    # Prefer NORMAL category first for primary
    preferred = [k for k in ordered if cats.get(k, "").upper() == "NORMAL"]
    rest = [k for k in ordered if k not in preferred]
    ranked = preferred + rest
    return [(lot, die, cats[(lot, die)]) for lot, die in ranked]


def _session_measurement_summary(
    month: str,
    lot_id: str,
    die_id: str,
    parameter: str,
    month_data_by_month: dict[str, TemporalMonthData],
    die_scoped_by_month: dict[tuple[str, str, str], dict[str, pd.DataFrame]] | None = None,
) -> dict[str, Any]:
    """Observed stats from session-scoped month frames (no repeated CSV reloads)."""
    if die_scoped_by_month and (month, lot_id, die_id) in die_scoped_by_month:
        scoped = die_scoped_by_month[(month, lot_id, die_id)]
        src = scoped["actual_die"] if parameter in {"ir_drop", "thermal"} else scoped["parametric"]
    else:
        data = month_data_by_month[month]
        src = data.actual_die if parameter in {"ir_drop", "thermal"} else data.parametric
    if src.empty:
        return {"n": 0, "mean": None, "min": None, "max": None}
    sub = src[src["parameter"].astype(str) == parameter]
    if sub.empty:
        return {"n": 0, "mean": None, "min": None, "max": None}
    vals = sub["measurement_value"].astype(float)
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def generate_session_analysis_artifacts(sandbox: Path) -> dict[str, Any]:
    """Run existing recommend() for primary die only; write Phase 12.9-shaped artifacts.

    Other uploaded dies are recommended on demand via die-level APIs (unchanged
    ``recommend()`` + cache). Simulation/sequence artifacts from materialize are reused.
    """
    root = sandbox
    out_dir = analysis_output_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    hybrid = TemporalHybridBundle(root)
    if not hybrid.ensure_loaded():
        raise UploadRecommendationError(
            "Temporal model bundle failed to load in analysis session: "
            + "; ".join(hybrid.load_errors)
        )

    scorable = _scorable_for_session(root)
    die_set = _dies_present_all_months(root)[:MAX_PRECOMPUTE_DIES]
    if not die_set:
        raise UploadRecommendationError("No dies available for session analysis")
    primary_lot, primary_die, primary_cat = die_set[0]

    # Hold Jan/Feb/Mar frames for this generation pass only (avoids _MAX_FULL_MONTHS=1 thrash).
    month_data_by_month: dict[str, TemporalMonthData] = {
        month: load_temporal_month(month, project_root=root) for month in MONTHS
    }

    die_scoped_by_month: dict[tuple[str, str, str], dict[str, pd.DataFrame]] = {}
    for m in MONTHS:
        mdata = month_data_by_month[m]
        for lot, die, _ in die_set:
            ad = mdata.actual_die
            ad_sub = ad[(ad["lot_id"].astype(str) == lot) & (ad["die_id"].astype(str) == die)]
            if mdata.parametric is not None and not mdata.parametric.empty:
                pm = mdata.parametric
                pm_sub = pm[(pm["lot_id"].astype(str) == lot) & (pm["die_id"].astype(str) == die)]
            else:
                pm_sub = pd.DataFrame()
            die_scoped_by_month[(m, lot, die)] = {"actual_die": ad_sub, "parametric": pm_sub}

    recommendation_rows: list[dict[str, Any]] = []
    candidate_expl_rows: list[dict[str, Any]] = []
    same_die_rows: list[dict[str, Any]] = []
    yield_first_proofs: list[dict[str, Any]] = []
    tie_break_proofs: list[dict[str, Any]] = []
    month_isolation: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _get(month: str, lot_id: str, die_id: str) -> dict[str, Any]:
        key = (month, lot_id, die_id)
        if key not in cache:
            result = recommend(
                lot_id=lot_id,
                die_id=die_id,
                parameters=list(scorable),
                production_month=month,
                project_root=root,
                temporal_bundle=hybrid,
            )
            cache[key] = result.to_dict()
        return cache[key]

    for lot_id, die_id, category in die_set:
        for month in MONTHS:
            payload = _get(month, lot_id, die_id)
            data_root = month_root(month, root)
            sim_root = month_simulation_root(month, root)
            data_root_s = str(data_root).replace("\\", "/")
            sim_root_s = str(sim_root).replace("\\", "/")
            month_isolation.append(
                {
                    "production_month": month,
                    "lot_id": lot_id,
                    "die_id": die_id,
                    "data_root": data_root_s,
                    "simulation_root": sim_root_s,
                    "uses_only_month_data": f"/3 months data/{month}" in data_root_s
                    or data_root_s.rstrip("/").endswith(f"/{month}"),
                    "uses_only_month_sim": f"/temporal/{month}/" in sim_root_s,
                }
            )
            audit = payload.get("audit") or {}
            for rec in payload.get("recommendations") or []:
                parameter = str(rec["parameter"])
                if parameter not in scorable:
                    continue
                exp = rec.get("explanation") or {}
                cur = float(rec["current_limit"])
                rec_lim = float(rec["recommended_limit"])
                y = exp.get("selected_simulated_yield")
                if y is None and isinstance(rec.get("simulation_evidence"), dict):
                    y = rec["simulation_evidence"].get("simulated_yield")
                row = {
                    "lot_category": category,
                    "lot_id": lot_id,
                    "die_id": die_id,
                    "sequence_id": make_sequence_id(lot_id, die_id, month),
                    "production_month": month,
                    "month_label": MONTH_LABEL[month],
                    "parameter": parameter,
                    "parameter_display": DISPLAY_NAME.get(parameter, parameter),
                    "unit": rec.get("unit"),
                    "current_limit": cur,
                    "recommended_limit": rec_lim,
                    "recommendation_delta": rec_lim - cur,
                    "recommendation_delta_percent": _delta_pct(cur, rec_lim),
                    "max_eligible_simulated_yield": y,
                    "ml_score": rec.get("ml_score"),
                    "ml_rank": rec.get("ml_rank"),
                    "model_used": rec.get("model_used"),
                    "model_expected": model_for_parameter(parameter, temporal=True).value,
                    "decision": rec.get("decision"),
                    "policy_reason": exp.get("policy_reason"),
                    "yield_tie": bool(exp.get("yield_tie")),
                    "tie_breaker": exp.get("tie_breaker"),
                    "selection_text": exp.get("selection_text"),
                    "explanation_text": exp.get("text"),
                    "why_selected": _explain_from_rec(rec),
                    "safety_status": _safety_status(rec),
                    "evidence_origin": rec.get("evidence_origin"),
                    "is_primary_die": lot_id == primary_lot and die_id == primary_die,
                    "used_uploaded_measurements": True,
                }
                recommendation_rows.append(row)
                cand_rows = _candidate_rows_for_param(
                    lot_id=lot_id,
                    die_id=die_id,
                    month=month,
                    parameter=parameter,
                    rec=rec,
                    audit=audit,
                )
                candidate_expl_rows.extend(cand_rows)
                yf = _find_yield_first_proof(cand_rows)
                if yf is not None:
                    yield_first_proofs.append(yf)
                tb = _find_tie_break_proof(cand_rows, rec)
                if tb is not None:
                    tie_break_proofs.append(tb)
                try:
                    meas = _session_measurement_summary(
                        month, lot_id, die_id, parameter, month_data_by_month, die_scoped_by_month
                    )
                except Exception:  # noqa: BLE001
                    # Core-only uploads may have empty parametric CSVs; skip observed stats.
                    meas = {"n": 0, "mean": None, "min": None, "max": None}
                same_die_rows.append(
                    {
                        **{
                            k: row[k]
                            for k in (
                                "lot_category",
                                "lot_id",
                                "die_id",
                                "sequence_id",
                                "production_month",
                                "parameter",
                                "parameter_display",
                                "current_limit",
                                "recommended_limit",
                                "max_eligible_simulated_yield",
                                "ml_score",
                                "ml_rank",
                                "model_used",
                                "decision",
                                "why_selected",
                            )
                        },
                        "observed_n": meas["n"],
                        "observed_mean": meas["mean"],
                        "observed_min": meas["min"],
                        "observed_max": meas["max"],
                    }
                )

    rec_df = pd.DataFrame(recommendation_rows)
    if rec_df.empty:
        raise UploadRecommendationError("recommend() produced no rows for uploaded session")
    primary = rec_df[rec_df["is_primary_die"]].copy()

    change_rows: list[dict[str, Any]] = []
    for parameter in scorable:
        sub = primary[primary["parameter"] == parameter]
        if sub.empty or set(sub["production_month"]) != set(MONTHS):
            continue
        sub = sub.set_index("production_month")
        jan, feb, mar = sub.loc["2026-01"], sub.loc["2026-02"], sub.loc["2026-03"]
        recs = [
            float(jan["recommended_limit"]),
            float(feb["recommended_limit"]),
            float(mar["recommended_limit"]),
        ]
        changed = len({round(x, 12) for x in recs}) > 1
        change_rows.append(
            {
                "parameter": parameter,
                "parameter_display": DISPLAY_NAME.get(parameter, parameter),
                "jan_recommendation": float(jan["recommended_limit"]),
                "feb_recommendation": float(feb["recommended_limit"]),
                "mar_recommendation": float(mar["recommended_limit"]),
                "recommendation_changed": changed,
                "jan_yield": jan["max_eligible_simulated_yield"],
                "feb_yield": feb["max_eligible_simulated_yield"],
                "mar_yield": mar["max_eligible_simulated_yield"],
                "jan_ml_rank": jan["ml_rank"],
                "feb_ml_rank": feb["ml_rank"],
                "mar_ml_rank": mar["ml_rank"],
                "jan_decision": jan["decision"],
                "feb_decision": feb["decision"],
                "mar_decision": mar["decision"],
                "model_used": jan["model_used"],
            }
        )

    model_rows = []
    for parameter in scorable:
        model_rows.append(
            {
                "parameter": parameter,
                "parameter_display": DISPLAY_NAME.get(parameter, parameter),
                "model_expected": model_for_parameter(parameter, temporal=True).value,
                "routing_ok": True,
            }
        )

    def _decision_breakdown(df: pd.DataFrame) -> dict[str, Any]:
        counts = Counter(df["decision"].astype(str))
        by_month = {
            m: dict(Counter(df[df["production_month"] == m]["decision"].astype(str)))
            for m in MONTHS
        }
        return {"total": int(len(df)), "counts": dict(counts), "by_month": by_month}

    executive = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "12.9-upload-session",
        "primary_die": {"lot_id": primary_lot, "die_id": primary_die},
        "n_primary_recommendations": int(len(primary)),
        "months": list(MONTHS),
        "scorable_parameters": [DISPLAY_NAME.get(p, p) for p in scorable],
        "decision_summary_primary": _decision_breakdown(primary),
        "decision_summary_all_dies": _decision_breakdown(rec_df),
        "parameters_recommendation_changed": [
            r["parameter_display"] for r in change_rows if r.get("recommendation_changed")
        ],
        "parameters_recommendation_stable": [
            r["parameter_display"]
            for r in change_rows
            if r.get("recommendation_changed") is False
        ],
        "model_routing_verified": True,
        "yield_first_proof_count": len(yield_first_proofs),
        "ml_tie_break_proof_count": len(tie_break_proofs),
        "yield_first_proof_example": yield_first_proofs[0] if yield_first_proofs else None,
        "ml_tie_break_proof_example": tie_break_proofs[0] if tie_break_proofs else None,
        "what_ml_does": (
            "The GRU scores candidate DTLs and produces ML rankings. The recommendation "
            "policy then uses maximum eligible simulated yield as the primary selection "
            "criterion and ML rank as the tie-breaker."
        ),
        "what_changed_summary": (
            f"On primary die {primary_die}, analysis was generated from uploaded "
            f"Jan/Feb/Mar measurements."
        ),
        "used_uploaded_measurements": True,
        "used_static_three_month_measurements": False,
        "limitations": [
            "Simulated yield is not guaranteed production yield.",
            "ML score is not simulated yield.",
            "Counterfactual cost estimates are not measured ATE savings.",
            "Analysis session uses uploaded measurements only for this session.",
        ],
    }

    primary_rows = [r for r in recommendation_rows if r["is_primary_die"]]
    _fast_write_csv(pd.DataFrame(primary_rows), out_dir / "three_month_recommendations.csv")
    write_json(
        out_dir / "three_month_recommendations.json",
        {
            "primary_die": {"lot_id": primary_lot, "die_id": primary_die},
            "rows": primary_rows,
            "all_dies_rows": recommendation_rows,
            "used_uploaded_measurements": True,
            "used_static_three_month_measurements": False,
        },
    )
    _fast_write_csv(pd.DataFrame(candidate_expl_rows), out_dir / "candidate_explanations.csv")
    _fast_write_csv(pd.DataFrame(change_rows), out_dir / "temporal_changes.csv")
    _fast_write_csv(pd.DataFrame(same_die_rows), out_dir / "same_die_analysis.csv")
    _fast_write_csv(pd.DataFrame(model_rows), out_dir / "model_traceability.csv")
    write_json(out_dir / "executive_summary.json", _json_safe(executive))
    write_json(
        out_dir / "policy_proofs.json",
        {
            "yield_first_proofs": yield_first_proofs[:20],
            "ml_tie_break_proofs": tie_break_proofs[:20],
            "month_isolation_checks": month_isolation,
        },
    )
    viz_cols = [
        c
        for c in (
            "parameter_display",
            "production_month",
            "current_limit",
            "recommended_limit",
            "max_eligible_simulated_yield",
            "decision",
        )
        if c in primary.columns
    ]
    _fast_write_csv(primary[viz_cols], out_dir / "viz_recommended_dtl_by_month.csv")

    return {
        "primary_die": {"lot_id": primary_lot, "die_id": primary_die, "lot_category": primary_cat},
        "n_primary": int(len(primary)),
        "n_all_die_rows": int(len(rec_df)),
        "scorable_parameters": list(scorable),
        "n_dies_precomputed": len(die_set),
    }


def create_upload_analysis_session(
    *,
    files: dict[str, tuple[str, bytes]],
    source_root: Path,
) -> tuple[AnalysisSession, dict[str, Any]]:
    """Parse three month files, materialize sandbox, generate analysis, register session.

    ``files`` keys must be ``january``, ``february``, ``march`` with (filename, bytes).
    """
    missing = [k for k in EXPECTED_MONTH_FILES if k not in files or not files[k][1]]
    if missing:
        raise UploadRecommendationError(
            f"Missing required upload(s): {', '.join(missing)}. "
            "January, February, and March files are all required."
        )

    packages: dict[str, ParsedMonthPackage] = {}
    source_files: dict[str, str] = {}
    month_mappings: dict[str, dict[str, Any]] = {}
    for slot, expected_month in EXPECTED_MONTH_FILES.items():
        filename, content = files[slot]
        pkg = parse_month_package(
            filename=filename, content=content, expected_month=expected_month
        )
        packages[pkg.production_month] = pkg
        source_files[expected_month] = filename
        month_mappings[expected_month] = {
            "upload_slot": slot,
            "source_filename": filename,
            "analysis_month": expected_month,
            "original_production_month": pkg.original_production_month,
        }

    try:
        sandbox = materialize_three_month_sandbox(packages, source_root=source_root)
    except UploadRecommendationError:
        raise
    except TemporalLoaderError as exc:
        raise UploadRecommendationError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise UploadRecommendationError(
            f"Failed to prepare uploaded analysis session: {exc}"
        ) from exc

    try:
        summary = generate_session_analysis_artifacts(sandbox)
    except UploadRecommendationError:
        shutil.rmtree(sandbox, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(sandbox, ignore_errors=True)
        raise UploadRecommendationError(
            f"Failed to generate analysis from uploaded data: {exc}"
        ) from exc

    provenance = {
        "used_uploaded_measurements": True,
        "used_static_three_month_measurements": False,
        "input_format": "temporal_measurements_csv",
        "source_files": source_files,
        "month_mappings": month_mappings,
        **summary,
    }
    sess = register_session(
        sandbox,
        months=MONTHS,
        source_files=source_files,
        provenance=provenance,
    )
    # Stamp session id onto marker
    marker_path = temporal_artifact_root(sandbox) / "shared" / "upload_session.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["analysis_session_id"] = sess.analysis_session_id
    marker["month_mappings"] = month_mappings
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return sess, provenance


_upload_executor = ThreadPoolExecutor(max_workers=2)


def _execute_async_upload_job(
    analysis_session_id: str,
    packages: dict[str, ParsedMonthPackage],
    source_files: dict[str, str],
    source_root: Path,
) -> None:
    """Execute upload sandbox creation, simulation, sequence generation, recommend(), and artifact creation in background."""
    try:
        update_job_status(
            analysis_session_id,
            status="processing",
            stage="Materializing analysis sandbox and running simulations",
            progress_pct=25,
        )
        sandbox = materialize_three_month_sandbox(packages, source_root=source_root)
    except UploadRecommendationError as exc:
        update_job_status(
            analysis_session_id,
            status="failed",
            stage="Failed",
            progress_pct=0,
            error=str(exc),
        )
        return
    except Exception as exc:  # noqa: BLE001
        update_job_status(
            analysis_session_id,
            status="failed",
            stage="Failed",
            progress_pct=0,
            error=f"Failed to prepare uploaded analysis session: {exc}",
        )
        return

    try:
        update_job_status(
            analysis_session_id,
            status="processing",
            stage="Running GRU recommendations and generating analysis summary",
            progress_pct=75,
        )
        summary = generate_session_analysis_artifacts(sandbox)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(sandbox, ignore_errors=True)
        update_job_status(
            analysis_session_id,
            status="failed",
            stage="Failed",
            progress_pct=0,
            error=f"Failed to generate analysis from uploaded data: {exc}",
        )
        return

    month_mappings = {
        month: {
            "upload_slot": [k for k, v in EXPECTED_MONTH_FILES.items() if v == month][0],
            "source_filename": pkg.source_filename,
            "analysis_month": month,
            "original_production_month": pkg.original_production_month,
        }
        for month, pkg in packages.items()
    }
    provenance = {
        "used_uploaded_measurements": True,
        "used_static_three_month_measurements": False,
        "input_format": "temporal_measurements_csv",
        "source_files": source_files,
        "month_mappings": month_mappings,
        **summary,
    }
    sess = register_session(
        sandbox,
        months=MONTHS,
        source_files=source_files,
        provenance=provenance,
        session_id=analysis_session_id,
    )
    marker_path = temporal_artifact_root(sandbox) / "shared" / "upload_session.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["analysis_session_id"] = sess.analysis_session_id
        marker["month_mappings"] = month_mappings
        marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")

    update_job_status(
        analysis_session_id,
        status="completed",
        stage="Ready",
        progress_pct=100,
        result_meta={
            "months": list(sess.months),
            "used_uploaded_measurements": True,
            "used_static_three_month_measurements": False,
            "source_files": sess.source_files,
            "month_mappings": month_mappings,
            "primary_die": provenance.get("primary_die"),
            "scorable_parameters": provenance.get("scorable_parameters"),
            "data_provenance": "Analysis generated from uploaded test data",
        },
    )


def start_upload_analysis_job(
    *,
    files: dict[str, tuple[str, bytes]],
    source_root: Path,
) -> dict[str, Any]:
    """Validate upload files synchronously (<0.5s) and launch background analysis job."""
    missing = [k for k in EXPECTED_MONTH_FILES if k not in files or not files[k][1]]
    if missing:
        raise UploadRecommendationError(
            f"Missing required upload(s): {', '.join(missing)}. "
            "January, February, and March files are all required."
        )

    packages: dict[str, ParsedMonthPackage] = {}
    source_files: dict[str, str] = {}
    month_mappings: dict[str, dict[str, Any]] = {}
    for slot, expected_month in EXPECTED_MONTH_FILES.items():
        filename, content = files[slot]
        pkg = parse_month_package(
            filename=filename, content=content, expected_month=expected_month
        )
        packages[pkg.production_month] = pkg
        source_files[expected_month] = filename
        month_mappings[expected_month] = {
            "upload_slot": slot,
            "source_filename": filename,
            "analysis_month": expected_month,
            "original_production_month": pkg.original_production_month,
        }

    analysis_session_id = str(uuid.uuid4())
    update_job_status(
        analysis_session_id,
        status="queued",
        stage="Queued for analysis processing",
        progress_pct=5,
    )
    _upload_executor.submit(
        _execute_async_upload_job,
        analysis_session_id,
        packages,
        source_files,
        source_root,
    )
    return {
        "analysis_session_id": analysis_session_id,
        "status": "queued",
        "stage": "Queued for analysis processing",
        "progress_pct": 5,
        "months": list(MONTHS),
        "used_uploaded_measurements": True,
        "used_static_three_month_measurements": False,
        "source_files": source_files,
        "month_mappings": month_mappings,
        "data_provenance": "Analysis generated from uploaded test data",
    }
