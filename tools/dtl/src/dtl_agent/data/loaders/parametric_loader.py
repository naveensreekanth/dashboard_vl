"""Parametric allowlisted data loader."""

from __future__ import annotations

from pathlib import Path

from dtl_agent.config.allowlists import PARAMETRIC_ALLOWLIST
from dtl_agent.config.paths import default_project_root, parametric_data_dir
from dtl_agent.data.models.parametric import ParametricDataset
from dtl_agent.data.repositories.allowlist_repository import AllowlistRepository
from dtl_agent.utils.csv_io import load_csv_dicts, load_json, read_csv_header, read_text


class ParametricDataLoader:
    """Load only allowlisted Parametric agent-input files."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or parametric_data_dir()).resolve()
        self.repo = AllowlistRepository(self.root, PARAMETRIC_ALLOWLIST, domain="parametric")

    def load(self, *, materialize_measurements: bool = False) -> ParametricDataset:
        paths = self.repo.require_all()
        measurements_path = paths["measurements.csv"]
        measurements: list[dict[str, str]] | None = None
        if materialize_measurements:
            measurements = load_csv_dicts(measurements_path)
        return ParametricDataset(
            root=self.root,
            version_metadata=load_json(paths["PARAMETRIC_DATASET_VERSION.json"]),
            lots=load_csv_dicts(paths["lots_dim.csv"]),
            parts=load_csv_dicts(paths["parts_dim.csv"]),
            conditions=load_csv_dicts(paths["conditions_dim.csv"]),
            test_catalog=load_csv_dicts(paths["test_catalog.csv"]),
            current_limits=load_csv_dicts(paths["current_limits.csv"]),
            scenario_manifest=load_csv_dicts(paths["scenario_manifest_public.csv"]),
            disposition_rules=load_json(paths["rules/disposition_rules.json"]),
            limit_simulation_config=load_json(paths["rules/limit_simulation_config.json"]),
            data_contract_text=read_text(paths["README_DATA_CONTRACT.md"]),
            measurements_path=measurements_path,
            measurements_columns=read_csv_header(measurements_path),
            measurements=measurements,
        )


def load_parametric(
    project_root: Path | None = None,
    *,
    materialize_measurements: bool = False,
) -> ParametricDataset:
    root = parametric_data_dir(project_root or default_project_root())
    return ParametricDataLoader(root).load(materialize_measurements=materialize_measurements)
