"""Upload → materialize temporal sandbox → existing recommend() path.

Does not modify GRU/policy/safety/simulation algorithms. Builds the on-disk
package shape that ``recommend(production_month=...)`` already expects, using
ONLY the uploaded measurements for sequences + simulation population.

Supported upload formats (repository-native):
  - CSV matching temporal ``actual_die/measurements.csv`` schema
  - Optional second CSV matching temporal ``parametric/measurements.csv``
  - ZIP containing ``actual_die/measurements.csv`` and optionally
    ``parametric/measurements.csv``

STDF binary is not supported (no parser in repository).
"""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from dtl_agent.data.temporal.loader import (
    REQUIRED_ACTUAL_DIE_COLUMNS,
    REQUIRED_PARAMETRIC_COLUMNS,
    TemporalLoaderError,
    TemporalMonthData,
)
from dtl_agent.data.temporal.paths import validate_production_month
from dtl_agent.data.temporal.parametric_simulation import run_temporal_parametric_simulation
from dtl_agent.data.temporal.simulation import run_temporal_core_simulation
from dtl_agent.ml_dataset.temporal_pipeline import build_temporal_sequence_store
from dtl_agent.recommendation import recommend
from dtl_agent.recommendation.schemas import CORE_PARAMETERS, PARAMETRIC_PARAMETERS
from dtl_agent.recommendation.temporal_inference import TemporalHybridBundle

ALLOWED_EXTENSIONS = frozenset({".csv", ".zip"})
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MiB
_SHARED_COPY_DIRS = (
    Path("artifacts") / "temporal" / "shared" / "checkpoints",
    Path("artifacts") / "temporal" / "shared" / "training",
    Path("artifacts") / "temporal" / "shared" / "unified_training",
    Path("artifacts") / "temporal" / "shared" / "unified_ml_dataset" / "normalization",
)


class UploadRecommendationError(ValueError):
    """User-facing validation / materialization failure (HTTP 422)."""


@dataclass(frozen=True)
class ParsedUpload:
    actual_die: pd.DataFrame
    parametric: pd.DataFrame | None
    production_month: str
    lot_id: str
    die_id: str
    source_filename: str


def _read_csv_bytes(data: bytes, label: str) -> pd.DataFrame:
    if not data:
        raise UploadRecommendationError(f"{label} is empty")
    try:
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    except Exception as exc:  # noqa: BLE001
        raise UploadRecommendationError(f"Malformed CSV ({label}): {exc}") from exc


def _require_columns(df: pd.DataFrame, required: frozenset[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise UploadRecommendationError(
            f"{label} missing required columns: {missing}. "
            f"Upload must match the temporal measurements CSV schema."
        )


def _single_die_identity(df: pd.DataFrame, label: str) -> tuple[str, str, str]:
    if df.empty:
        raise UploadRecommendationError(f"{label} contains no rows")
    lots = df["lot_id"].astype(str).unique().tolist()
    dies = df["die_id"].astype(str).unique().tolist()
    months = df["production_month"].astype(str).unique().tolist()
    if len(lots) != 1 or len(dies) != 1:
        raise UploadRecommendationError(
            f"{label} must contain exactly one lot_id/die_id "
            f"(found lots={lots[:5]}, dies={dies[:5]})"
        )
    if len(months) != 1:
        raise UploadRecommendationError(
            f"{label} must contain exactly one production_month (found {months[:5]})"
        )
    try:
        month = validate_production_month(months[0])
    except Exception as exc:  # noqa: BLE001
        raise UploadRecommendationError(
            f"Invalid production_month={months[0]!r}; "
            f"allowed: 2026-01, 2026-02, 2026-03"
        ) from exc
    return lots[0], dies[0], month


def parse_upload_payload(
    *,
    filename: str,
    content: bytes,
    parametric_filename: str | None = None,
    parametric_content: bytes | None = None,
) -> ParsedUpload:
    """Parse multipart upload into validated temporal DataFrames."""
    if not content:
        raise UploadRecommendationError("Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadRecommendationError(
            f"Upload exceeds maximum size of {MAX_UPLOAD_BYTES} bytes"
        )

    name = (filename or "").strip()
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadRecommendationError(
            f"Unsupported file type {suffix!r}. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)} "
            f"(temporal actual_die measurements CSV or ZIP). "
            f"STDF binary is not supported by this repository."
        )

    parametric_df: pd.DataFrame | None = None

    if suffix == ".zip":
        try:
            zf = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as exc:
            raise UploadRecommendationError("Malformed ZIP archive") from exc
        names = [n for n in zf.namelist() if not n.endswith("/")]
        actual_name = next(
            (n for n in names if n.replace("\\", "/").endswith("actual_die/measurements.csv")),
            None,
        )
        if actual_name is None:
            actual_name = next(
                (n for n in names if Path(n).name.lower() == "measurements.csv"
                 and "parametric" not in n.replace("\\", "/").lower()),
                None,
            )
        if actual_name is None:
            raise UploadRecommendationError(
                "ZIP must contain actual_die/measurements.csv "
                "(or a top-level measurements.csv for actual_die)"
            )
        actual_df = _read_csv_bytes(zf.read(actual_name), "actual_die/measurements.csv")
        param_name = next(
            (n for n in names if n.replace("\\", "/").endswith("parametric/measurements.csv")),
            None,
        )
        if param_name is not None:
            parametric_df = _read_csv_bytes(zf.read(param_name), "parametric/measurements.csv")
    else:
        actual_df = _read_csv_bytes(content, name)
        if parametric_content is not None and parametric_filename:
            if not parametric_content:
                raise UploadRecommendationError("Parametric upload is empty")
            if len(parametric_content) > MAX_UPLOAD_BYTES:
                raise UploadRecommendationError("Parametric upload exceeds maximum size")
            if Path(parametric_filename).suffix.lower() != ".csv":
                raise UploadRecommendationError("Parametric upload must be a .csv file")
            parametric_df = _read_csv_bytes(parametric_content, parametric_filename)

    _require_columns(actual_df, REQUIRED_ACTUAL_DIE_COLUMNS, "actual_die")
    if "pattern_id" not in actual_df.columns:
        raise UploadRecommendationError("actual_die CSV must include pattern_id")
    lot_id, die_id, month = _single_die_identity(actual_df, "actual_die")

    if parametric_df is not None:
        _require_columns(parametric_df, REQUIRED_PARAMETRIC_COLUMNS, "parametric")
        plot, pdid, pmonth = _single_die_identity(parametric_df, "parametric")
        if (plot, pdid, pmonth) != (lot_id, die_id, month):
            raise UploadRecommendationError(
                "parametric CSV lot_id/die_id/production_month must match actual_die"
            )

    return ParsedUpload(
        actual_die=actual_df,
        parametric=parametric_df,
        production_month=month,
        lot_id=lot_id,
        die_id=die_id,
        source_filename=name,
    )


def _copy_shared_assets(source_root: Path, sandbox: Path) -> None:
    for rel in _SHARED_COPY_DIRS:
        src = source_root / rel
        dst = sandbox / rel
        if not src.is_dir():
            raise UploadRecommendationError(
                f"Missing required shared asset directory for scoring: {rel}"
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    # Unified sequence path fallback: ensure parent exists for our written sequences
    (sandbox / "artifacts" / "temporal" / "shared" / "ml_dataset" / "sequences").mkdir(
        parents=True, exist_ok=True
    )


def _write_empty_parametric_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=sorted(REQUIRED_PARAMETRIC_COLUMNS)).to_csv(path, index=False)


def _stub_parametric_simulation_artifacts(sandbox: Path, month: str, source_root: Path) -> None:
    """Ensure month parametric grid/results exist when parametric data was not uploaded.

    Copies catalog grids from the source project (limit catalogs, not die measurements).
    Results are regenerated as empty-population placeholders so assert_month_simulation_isolated
    passes without using source month die yields.
    """
    src_grid = (
        source_root
        / "artifacts"
        / "temporal"
        / month
        / "simulation"
        / "parametric"
        / "candidate_grid.csv"
    )
    dst_dir = sandbox / "artifacts" / "temporal" / month / "simulation" / "parametric"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_grid.is_file():
        raise UploadRecommendationError(
            f"Cannot stub parametric simulation; missing source grid {src_grid}"
        )
    shutil.copy2(src_grid, dst_dir / "candidate_grid.csv")
    grid = pd.read_csv(dst_dir / "candidate_grid.csv")
    # Placeholder results: same candidates, zero dies — not used when only core params requested
    rows = []
    for _, r in grid.iterrows():
        rows.append(
            {
                **{c: r[c] for c in grid.columns if c in r.index},
                "total_dies": 0,
                "good_dies": 0,
                "violating_dies": 0,
                "simulated_yield": 0.0,
                "simulated_fail_rate": 1.0,
                "objective_score": 0.0,
                "feasible": True,
                "selection_status": "",
                "notes": "upload_stub_no_parametric_measurements",
            }
        )
    pd.DataFrame(rows).to_csv(dst_dir / "candidate_results.csv", index=False)


def materialize_upload_sandbox(
    parsed: ParsedUpload,
    *,
    source_root: Path,
) -> Path:
    """Create a temporary project_root that recommend() can load."""
    sandbox = Path(tempfile.mkdtemp(prefix="dtl_upload_"))
    try:
        _copy_shared_assets(source_root, sandbox)

        month = parsed.production_month
        actual_path = (
            sandbox
            / "data"
            / "3 months data"
            / month
            / "actual_die"
            / "measurements.csv"
        )
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        parsed.actual_die.to_csv(actual_path, index=False)

        param_path = (
            sandbox
            / "data"
            / "3 months data"
            / month
            / "parametric"
            / "measurements.csv"
        )
        if parsed.parametric is not None and not parsed.parametric.empty:
            param_path.parent.mkdir(parents=True, exist_ok=True)
            parsed.parametric.to_csv(param_path, index=False)
        else:
            _write_empty_parametric_csv(param_path)

        month_data = TemporalMonthData(
            production_month=month,
            month_path=sandbox / "data" / "3 months data" / month,
            actual_die=parsed.actual_die.copy(),
            parametric=(
                parsed.parametric.copy()
                if parsed.parametric is not None and not parsed.parametric.empty
                else pd.DataFrame(columns=sorted(REQUIRED_PARAMETRIC_COLUMNS))
            ),
            parts_dim=None,
        )

        seq_out = sandbox / "artifacts" / "temporal" / "shared" / "ml_dataset"
        build_temporal_sequence_store(month_data, seq_out)

        run_temporal_core_simulation(
            month, project_root=sandbox, month_data=month_data, joint_search="product"
        )

        if parsed.parametric is not None and not parsed.parametric.empty:
            run_temporal_parametric_simulation(
                month, project_root=sandbox, month_data=month_data
            )
        else:
            _stub_parametric_simulation_artifacts(sandbox, month, source_root)

        return sandbox
    except Exception:
        shutil.rmtree(sandbox, ignore_errors=True)
        raise


def recommend_from_upload(
    *,
    filename: str,
    content: bytes,
    source_root: Path,
    parametric_filename: str | None = None,
    parametric_content: bytes | None = None,
    parameters: list[str] | None = None,
) -> dict[str, Any]:
    """Parse upload, materialize sandbox, run existing recommend(), cleanup."""
    parsed = parse_upload_payload(
        filename=filename,
        content=content,
        parametric_filename=parametric_filename,
        parametric_content=parametric_content,
    )

    has_param = parsed.parametric is not None and not parsed.parametric.empty
    if parameters is None:
        params = sorted(CORE_PARAMETERS)
        if has_param:
            params = sorted(CORE_PARAMETERS | PARAMETRIC_PARAMETERS)
    else:
        params = list(parameters)
        if any(p in PARAMETRIC_PARAMETERS for p in params) and not has_param:
            raise UploadRecommendationError(
                "Parametric parameters requested but no parametric measurements were uploaded"
            )

    sandbox: Path | None = None
    try:
        try:
            sandbox = materialize_upload_sandbox(parsed, source_root=source_root)
        except UploadRecommendationError:
            raise
        except TemporalLoaderError as exc:
            raise UploadRecommendationError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise UploadRecommendationError(
                f"Failed to prepare uploaded data for recommendation: {exc}"
            ) from exc

        bundle = TemporalHybridBundle(sandbox)
        if not bundle.ensure_loaded():
            raise UploadRecommendationError(
                "Temporal model bundle failed to load in upload sandbox: "
                + "; ".join(bundle.load_errors)
            )

        result = recommend(
            lot_id=parsed.lot_id,
            die_id=parsed.die_id,
            parameters=params,
            project_root=sandbox,
            production_month=parsed.production_month,
            temporal_bundle=bundle,
        )
        payload = result.to_dict()
        payload["upload"] = {
            "source_filename": parsed.source_filename,
            "production_month": parsed.production_month,
            "lot_id": parsed.lot_id,
            "die_id": parsed.die_id,
            "parameters": params,
            "used_uploaded_measurements": True,
            "used_static_three_month_measurements": False,
            "parametric_uploaded": has_param,
            "input_format": "temporal_measurements_csv",
        }
        return payload
    finally:
        if sandbox is not None:
            shutil.rmtree(sandbox, ignore_errors=True)


def read_upload_file(file_obj: BinaryIO) -> bytes:
    data = file_obj.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data
