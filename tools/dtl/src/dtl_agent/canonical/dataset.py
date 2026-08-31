"""CanonicalDataset — dual-grain public API for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from dtl_agent.canonical.builder import (
    build_conditions,
    build_core_limits,
    build_core_tests,
    build_dies,
    build_lots,
    build_parametric_limits,
    build_parametric_tests,
)
from dtl_agent.canonical.entities import (
    CORE_GRAIN,
    PARAMETRIC_GRAIN,
    CanonicalCondition,
    CanonicalCurrentLimit,
    CanonicalDie,
    CanonicalLot,
    CanonicalTestDefinition,
    CoreMeasurementRecord,
    GrainSpec,
    LinkedDieView,
    ParametricMeasurementRecord,
)
from dtl_agent.canonical.views import (
    CoreMeasurementView,
    ParametricMeasurementView,
    assert_not_concatenated,
)
from dtl_agent.data.models.bundle import ValidatedDatasetBundle
from dtl_agent.data.models.core import CoreDataset
from dtl_agent.data.models.linkage import SharedLotDieIndex
from dtl_agent.data.models.parametric import ParametricDataset


class CanonicalLookupError(KeyError):
    """Raised when a canonical entity identity is not found."""


@dataclass
class CanonicalDataset:
    """Dual-grain canonical layer over validated Core + Parametric datasets.

    Performance:
    - Lot/die/test/limit/condition dims are indexed in memory (small).
    - Measurement tables are NOT duplicated; access is lazy via domain views
      that stream from Phase 1 ``CoreDataset`` / ``ParametricDataset`` handles.
    """

    core: CoreDataset
    parametric: ParametricDataset
    linkage: SharedLotDieIndex
    lots: dict[str, CanonicalLot]
    dies: dict[tuple[str, str], CanonicalDie]
    conditions: dict[str, CanonicalCondition]
    core_tests: dict[str, CanonicalTestDefinition]
    parametric_tests: dict[str, CanonicalTestDefinition]
    core_limits: dict[str, CanonicalCurrentLimit]
    parametric_limits: dict[str, CanonicalCurrentLimit]
    core_measurements: CoreMeasurementView = field(init=False, repr=False)
    parametric_measurements: ParametricMeasurementView = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.core_measurements = CoreMeasurementView(self.core)
        self.parametric_measurements = ParametricMeasurementView(self.parametric)

    # ----- grains -----
    @property
    def core_grain(self) -> GrainSpec:
        return CORE_GRAIN

    @property
    def parametric_grain(self) -> GrainSpec:
        return PARAMETRIC_GRAIN

    def measurements_are_separated(self) -> bool:
        return assert_not_concatenated(self.core_measurements, self.parametric_measurements)

    # ----- lot / die -----
    def get_lot(self, lot_id: str) -> CanonicalLot:
        try:
            return self.lots[lot_id]
        except KeyError as exc:
            raise CanonicalLookupError(f"unknown lot_id={lot_id!r}") from exc

    def iter_lots(self, *, domain: str | None = None) -> Iterator[CanonicalLot]:
        for lot in self.lots.values():
            if domain == "core" and not lot.in_core:
                continue
            if domain == "parametric" and not lot.in_parametric:
                continue
            yield lot

    def get_die(self, lot_id: str, die_id: str) -> CanonicalDie:
        key = (lot_id, die_id)
        try:
            return self.dies[key]
        except KeyError as exc:
            raise CanonicalLookupError(
                f"unknown die lot_id={lot_id!r} die_id={die_id!r}"
            ) from exc

    def iter_dies(self, *, lot_id: str | None = None, domain: str | None = None) -> Iterator[CanonicalDie]:
        for die in self.dies.values():
            if lot_id is not None and die.lot_id != lot_id:
                continue
            if domain == "core" and not die.in_core:
                continue
            if domain == "parametric" and not die.in_parametric:
                continue
            yield die

    def get_linked_die(self, lot_id: str, die_id: str) -> LinkedDieView:
        pair = (lot_id, die_id)
        die_entity = self.dies.get(pair)
        if die_entity is None:
            raise CanonicalLookupError(
                f"no die entity for lot_id={lot_id!r} die_id={die_id!r}"
            )
        return LinkedDieView(
            lot_id=lot_id,
            die_id=die_id,
            cross_domain_available=self.linkage.is_linked_lot_die(lot_id, die_id),
            core_die=die_entity if die_entity.in_core else None,
            parametric_die=die_entity if die_entity.in_parametric else None,
        )

    def has_core_data(self, lot_id: str, die_id: str | None = None) -> bool:
        if die_id is None:
            return lot_id in self.linkage.core_lots
        return (lot_id, die_id) in self.linkage.core_lot_die_pairs

    def has_parametric_data(self, lot_id: str, die_id: str | None = None) -> bool:
        if die_id is None:
            return lot_id in self.linkage.parametric_lots
        return (lot_id, die_id) in self.linkage.parametric_lot_die_pairs

    def cross_domain_available(self, lot_id: str, die_id: str | None = None) -> bool:
        if die_id is None:
            return self.linkage.cross_domain_features_available(lot_id=lot_id)
        return self.linkage.cross_domain_features_available(lot_id=lot_id, die_id=die_id)

    # ----- conditions / tests / limits -----
    def get_conditions(self) -> list[CanonicalCondition]:
        return [self.conditions[k] for k in sorted(self.conditions)]

    def get_condition(self, condition_id: str) -> CanonicalCondition:
        try:
            return self.conditions[condition_id]
        except KeyError as exc:
            raise CanonicalLookupError(f"unknown condition_id={condition_id!r}") from exc

    def get_test_definition(self, domain: str, test_id: str) -> CanonicalTestDefinition:
        catalog = self._tests_for_domain(domain)
        try:
            return catalog[test_id]
        except KeyError as exc:
            raise CanonicalLookupError(
                f"unknown test domain={domain!r} test_id={test_id!r}"
            ) from exc

    def iter_test_definitions(self, domain: str) -> Iterator[CanonicalTestDefinition]:
        yield from self._tests_for_domain(domain).values()

    def get_current_limit(
        self, domain: str, *, test_id: str | None = None, parameter: str | None = None
    ) -> CanonicalCurrentLimit:
        limits = self._limits_for_domain(domain)
        if test_id is not None:
            try:
                return limits[test_id]
            except KeyError as exc:
                raise CanonicalLookupError(
                    f"unknown limit domain={domain!r} test_id={test_id!r}"
                ) from exc
        if parameter is not None:
            for lim in limits.values():
                if lim.parameter == parameter:
                    return lim
            raise CanonicalLookupError(
                f"unknown limit domain={domain!r} parameter={parameter!r}"
            )
        raise ValueError("provide test_id or parameter")

    def iter_current_limits(self, domain: str) -> Iterator[CanonicalCurrentLimit]:
        yield from self._limits_for_domain(domain).values()

    # ----- measurements (lazy) -----
    def get_core_measurements(
        self,
        *,
        lot_id: str | None = None,
        die_id: str | None = None,
        pattern_id: str | None = None,
        test_id: str | None = None,
        parameter: str | None = None,
    ) -> Iterator[CoreMeasurementRecord]:
        return self.core_measurements.iter(
            lot_id=lot_id,
            die_id=die_id,
            pattern_id=pattern_id,
            test_id=test_id,
            parameter=parameter,
        )

    def get_parametric_measurements(
        self,
        *,
        lot_id: str | None = None,
        die_id: str | None = None,
        condition_id: str | None = None,
        test_id: str | None = None,
        parameter: str | None = None,
    ) -> Iterator[ParametricMeasurementRecord]:
        return self.parametric_measurements.iter(
            lot_id=lot_id,
            die_id=die_id,
            condition_id=condition_id,
            test_id=test_id,
            parameter=parameter,
        )

    def summary(self) -> dict[str, object]:
        return {
            "core_grain": self.core_grain.description,
            "parametric_grain": self.parametric_grain.description,
            "measurements_separated": self.measurements_are_separated(),
            "lot_count": len(self.lots),
            "die_count": len(self.dies),
            "condition_count": len(self.conditions),
            "core_test_count": len(self.core_tests),
            "parametric_test_count": len(self.parametric_tests),
            "core_limit_count": len(self.core_limits),
            "parametric_limit_count": len(self.parametric_limits),
            "linkage": self.linkage.summary(),
            "measurement_access": "lazy_streaming_no_full_copy",
        }

    def _tests_for_domain(self, domain: str) -> dict[str, CanonicalTestDefinition]:
        if domain == "core":
            return self.core_tests
        if domain == "parametric":
            return self.parametric_tests
        raise ValueError(f"domain must be 'core' or 'parametric', got {domain!r}")

    def _limits_for_domain(self, domain: str) -> dict[str, CanonicalCurrentLimit]:
        if domain == "core":
            return self.core_limits
        if domain == "parametric":
            return self.parametric_limits
        raise ValueError(f"domain must be 'core' or 'parametric', got {domain!r}")


def build_canonical_dataset(bundle: ValidatedDatasetBundle) -> CanonicalDataset:
    """Construct the canonical layer from a Phase 1 validated bundle."""
    if not bundle.ok:
        raise ValueError("Cannot build CanonicalDataset from a failing Phase 1 bundle")
    return CanonicalDataset(
        core=bundle.core,
        parametric=bundle.parametric,
        linkage=bundle.linkage,
        lots=build_lots(bundle.core, bundle.parametric, bundle.linkage),
        dies=build_dies(bundle.core, bundle.parametric, bundle.linkage),
        conditions=build_conditions(bundle.parametric),
        core_tests=build_core_tests(bundle.core),
        parametric_tests=build_parametric_tests(bundle.parametric),
        core_limits=build_core_limits(bundle.core),
        parametric_limits=build_parametric_limits(bundle.parametric),
    )


def build_canonical_from_datasets(
    core: CoreDataset,
    parametric: ParametricDataset,
    linkage: SharedLotDieIndex | None = None,
) -> CanonicalDataset:
    """Build without requiring a full Phase 1 report (tests / tooling)."""
    index = linkage or SharedLotDieIndex.from_datasets(core, parametric)
    return CanonicalDataset(
        core=core,
        parametric=parametric,
        linkage=index,
        lots=build_lots(core, parametric, index),
        dies=build_dies(core, parametric, index),
        conditions=build_conditions(parametric),
        core_tests=build_core_tests(core),
        parametric_tests=build_parametric_tests(parametric),
        core_limits=build_core_limits(core),
        parametric_limits=build_parametric_limits(parametric),
    )
