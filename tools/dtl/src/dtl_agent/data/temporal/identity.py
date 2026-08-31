"""Sequence identity helpers for legacy and temporal modes (Phase 12.3)."""

from __future__ import annotations


def make_sequence_id(
    lot_id: str,
    die_id: str,
    production_month: str | None = None,
) -> str:
    """Build sequence identity.

    - Temporal: ``{production_month}::{lot_id}::{die_id}``
    - Legacy (``production_month is None``): ``{lot_id}::{die_id}``
    """
    if production_month is None:
        return f"{lot_id}::{die_id}"
    return f"{production_month}::{lot_id}::{die_id}"
