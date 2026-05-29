from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Locate the repo root regardless of the current working directory."""
    env = os.environ.get("ZO_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists() or (parent / "Justfile").exists():
            return parent
    return Path.cwd()


def experiments_dir() -> Path:
    """The shared run store. Override with ZO_EXPERIMENTS_DIR (e.g. cluster scratch)."""
    env = os.environ.get("ZO_EXPERIMENTS_DIR")
    base = Path(env).expanduser().resolve() if env else repo_root() / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base
