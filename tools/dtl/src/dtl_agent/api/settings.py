"""Service-level settings for the Phase 9 HTTP API (not recommendation policy)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dtl_agent.config.paths import default_project_root


@dataclass(frozen=True)
class ServiceSettings:
    """API/service configuration loaded from environment."""

    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"
    project_root: Path | None = None
    policy_config_path: Path | None = None
    cors_allowed_origins: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        root_raw = os.environ.get("DTL_PROJECT_ROOT", "").strip()
        policy_raw = os.environ.get("DTL_POLICY_CONFIG_PATH", "").strip()
        port_raw = os.environ.get("API_PORT", "8000").strip()
        return cls(
            host=os.environ.get("API_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=int(port_raw) if port_raw.isdigit() else 8000,
            api_prefix=os.environ.get("API_PREFIX", "/api/v1").strip() or "/api/v1",
            project_root=Path(root_raw) if root_raw else None,
            policy_config_path=Path(policy_raw) if policy_raw else None,
            cors_allowed_origins=os.environ.get("CORS_ALLOWED_ORIGINS", "").strip(),
            log_level=os.environ.get("LOG_LEVEL", "INFO").strip() or "INFO",
        )

    def resolved_project_root(self) -> Path:
        return self.project_root or default_project_root()

    def parsed_cors_origins(self) -> list[str]:
        if not self.cors_allowed_origins:
            return []
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]
