from __future__ import annotations

import os
from pathlib import Path


def get_app_data_dir() -> Path:
    env = os.environ.get("AUTOREGRESSX_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    home = Path.home()
    return (home / ".autoregressex").resolve()


def get_runs_dir() -> Path:
    return get_app_data_dir() / "runs"


def ensure_runs_dir() -> Path:
    d = get_runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d
