"""Thin HTTP transport schemas for Phase 9 (structural validation only)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_PRODUCTION_MONTHS = frozenset({"2026-01", "2026-02", "2026-03"})


class RecommendationRequest(BaseModel):
    """Minimum client request — no recommendation policy or path fields."""

    model_config = ConfigDict(extra="forbid")

    lot_id: str = Field(min_length=1)
    die_id: str = Field(min_length=1)
    parameters: list[str] | None = None
    # Optional; omit or null → legacy recommend path. Invalid months must not fall back.
    production_month: str | None = None

    @field_validator("production_month")
    @classmethod
    def _validate_production_month(cls, value: str | None) -> str | None:
        if value is None:
            return None
        month = str(value).strip()
        if not month:
            raise ValueError(
                "production_month must be one of 2026-01, 2026-02, 2026-03 "
                "(empty/null omit for legacy mode)"
            )
        if month not in ALLOWED_PRODUCTION_MONTHS:
            raise ValueError(
                f"Invalid production_month={value!r}; allowed: 2026-01, 2026-02, 2026-03 "
                "(no silent fallback)"
            )
        return month
