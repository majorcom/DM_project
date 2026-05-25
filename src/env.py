"""Small .env loader used by the notebook without an extra dependency."""

from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_env(project_root: str | Path, filename: str = ".env", override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a project .env file into os.environ."""

    env_path = Path(project_root) / filename
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value)
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value

    return loaded
