from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings, read from BOOKREC_* environment variables."""

    artifacts_dir: Path = Path("artifacts")
    api_key: str | None = None
    default_n: int = 10
    max_n: int = 100
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            artifacts_dir=Path(env.get("BOOKREC_ARTIFACTS_DIR", "artifacts")),
            api_key=env.get("BOOKREC_API_KEY") or None,
            default_n=int(env.get("BOOKREC_DEFAULT_N", "10")),
            max_n=int(env.get("BOOKREC_MAX_N", "100")),
            log_level=env.get("BOOKREC_LOG_LEVEL", "INFO").upper(),
        )
