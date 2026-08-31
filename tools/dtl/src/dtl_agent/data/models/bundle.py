"""Validated multi-domain bundle for Phase 2 handoff."""

from __future__ import annotations

from dataclasses import dataclass

from dtl_agent.data.models.core import CoreDataset
from dtl_agent.data.models.linkage import SharedLotDieIndex
from dtl_agent.data.models.parametric import ParametricDataset
from dtl_agent.validation.report import Phase1ValidationReport


@dataclass
class ValidatedDatasetBundle:
    core: CoreDataset
    parametric: ParametricDataset
    linkage: SharedLotDieIndex
    validation: Phase1ValidationReport

    @property
    def ok(self) -> bool:
        return self.validation.passed
