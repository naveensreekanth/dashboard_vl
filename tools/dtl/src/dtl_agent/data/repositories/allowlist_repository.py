"""Allowlist-enforcing path resolution for a single data domain root."""

from __future__ import annotations

from pathlib import Path

from dtl_agent.config.allowlists import FORBIDDEN_PATH_FRAGMENTS


class AllowlistViolation(ValueError):
    """Raised when a path is outside the domain allowlist or is forbidden."""


class AllowlistRepository:
    """Resolve files only via an explicit relative allowlist (no recursive scan)."""

    def __init__(self, root: Path, allowlist: frozenset[str], *, domain: str) -> None:
        self.root = root.resolve()
        self.allowlist = allowlist
        self.domain = domain

    def resolve(self, relative_path: str) -> Path:
        rel = relative_path.replace("\\", "/")
        if rel not in self.allowlist:
            raise AllowlistViolation(
                f"{self.domain}: '{rel}' is not on the agent-input allowlist"
            )
        self._assert_not_forbidden(rel)
        path = (self.root / rel).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise AllowlistViolation(
                f"{self.domain}: resolved path escapes domain root: {path}"
            ) from exc
        if not path.exists():
            raise FileNotFoundError(f"{self.domain}: allowlisted file missing: {rel}")
        if not path.is_file():
            raise AllowlistViolation(f"{self.domain}: not a file: {rel}")
        return path

    def require_all(self) -> dict[str, Path]:
        return {rel: self.resolve(rel) for rel in sorted(self.allowlist)}

    @staticmethod
    def _assert_not_forbidden(relative_path: str) -> None:
        normalized = relative_path.replace("\\", "/").lower()
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            frag = fragment.lower()
            if frag in normalized or normalized.endswith(frag.rstrip("/")):
                raise AllowlistViolation(
                    f"Forbidden agent-input path rejected: {relative_path}"
                )
