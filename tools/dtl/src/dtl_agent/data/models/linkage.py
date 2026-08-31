"""Cross-domain lot/die linkage index (no measurement merge)."""

from __future__ import annotations

from dataclasses import dataclass

from dtl_agent.data.models.core import CoreDataset
from dtl_agent.data.models.parametric import ParametricDataset


@dataclass(frozen=True)
class SharedLotDieIndex:
    """Index of shared and domain-only lot/die identities."""

    core_lots: frozenset[str]
    parametric_lots: frozenset[str]
    linked_lots: frozenset[str]
    core_only_lots: frozenset[str]
    parametric_only_lots: frozenset[str]
    core_dies: frozenset[str]
    parametric_dies: frozenset[str]
    linked_dies: frozenset[str]
    core_only_dies: frozenset[str]
    parametric_only_dies: frozenset[str]
    linked_lot_die_pairs: frozenset[tuple[str, str]]
    core_lot_die_pairs: frozenset[tuple[str, str]]
    parametric_lot_die_pairs: frozenset[tuple[str, str]]

    @classmethod
    def from_datasets(
        cls, core: CoreDataset, parametric: ParametricDataset
    ) -> SharedLotDieIndex:
        core_lots = frozenset(core.lot_ids)
        par_lots = frozenset(parametric.lot_ids)
        linked_lots = frozenset(core_lots & par_lots)
        core_dies = frozenset(core.die_ids)
        par_dies = frozenset(parametric.die_ids)
        linked_dies = frozenset(core_dies & par_dies)
        core_pairs = frozenset(core.lot_die_pairs)
        par_pairs = frozenset(parametric.lot_die_pairs)
        linked_pairs = frozenset(core_pairs & par_pairs)
        return cls(
            core_lots=core_lots,
            parametric_lots=par_lots,
            linked_lots=linked_lots,
            core_only_lots=frozenset(core_lots - par_lots),
            parametric_only_lots=frozenset(par_lots - core_lots),
            core_dies=core_dies,
            parametric_dies=par_dies,
            linked_dies=linked_dies,
            core_only_dies=frozenset(core_dies - par_dies),
            parametric_only_dies=frozenset(par_dies - core_dies),
            linked_lot_die_pairs=linked_pairs,
            core_lot_die_pairs=core_pairs,
            parametric_lot_die_pairs=par_pairs,
        )

    def is_linked_lot(self, lot_id: str) -> bool:
        return lot_id in self.linked_lots

    def is_parametric_only_lot(self, lot_id: str) -> bool:
        return lot_id in self.parametric_only_lots

    def is_linked_die(self, die_id: str) -> bool:
        return die_id in self.linked_dies

    def has_lot_die_pair(self, lot_id: str, die_id: str) -> bool:
        """True if (lot_id, die_id) exists in either domain."""
        pair = (lot_id, die_id)
        return pair in self.core_lot_die_pairs or pair in self.parametric_lot_die_pairs

    def is_linked_lot_die(self, lot_id: str, die_id: str) -> bool:
        return (lot_id, die_id) in self.linked_lot_die_pairs

    def lot_die_pairs_for_lot(self, lot_id: str) -> frozenset[tuple[str, str]]:
        return frozenset(
            p
            for p in (self.core_lot_die_pairs | self.parametric_lot_die_pairs)
            if p[0] == lot_id
        )

    def cross_domain_features_available(
        self, *, lot_id: str | None = None, die_id: str | None = None
    ) -> bool:
        """True when Core+Parametric identity overlap exists (optionally for a lot/die)."""
        if lot_id is not None and die_id is not None:
            return (lot_id, die_id) in self.linked_lot_die_pairs
        if lot_id is None:
            return bool(self.linked_lots) and bool(self.linked_lot_die_pairs)
        if lot_id not in self.linked_lots:
            return False
        return any(pair[0] == lot_id for pair in self.linked_lot_die_pairs)

    def summary(self) -> dict[str, int]:
        return {
            "core_lot_count": len(self.core_lots),
            "parametric_lot_count": len(self.parametric_lots),
            "linked_lot_count": len(self.linked_lots),
            "core_only_lot_count": len(self.core_only_lots),
            "parametric_only_lot_count": len(self.parametric_only_lots),
            "core_die_count": len(self.core_dies),
            "parametric_die_count": len(self.parametric_dies),
            "linked_die_count": len(self.linked_dies),
            "core_only_die_count": len(self.core_only_dies),
            "parametric_only_die_count": len(self.parametric_only_dies),
            "linked_lot_die_pair_count": len(self.linked_lot_die_pairs),
        }
