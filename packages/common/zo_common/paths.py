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
    """The shared run store. Override with ZO_EXPERIMENTS_DIR (e.g. cluster scratch).

    Default: ``~/.cache/zo-experiments`` (outside the repo) so scratch stays disposable.
    Set ``ZO_EXPERIMENTS_DIR=experiments`` in ``.env`` to use the legacy in-repo path.
    """
    env = os.environ.get("ZO_EXPERIMENTS_DIR")
    if env:
        base = Path(env).expanduser().resolve()
    else:
        base = Path.home() / ".cache" / "zo-experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base


def generated_data_dir() -> Path:
    """Shared store for the generated data corpus (splits, negatives, CoT, text examples).

    Override with ZO_DATA_DIR so every worktree / cluster job reads the identical frozen
    corpus (e.g. point it at shared scratch). Defaults to ``<repo>/data/generated``.
    """
    env = os.environ.get("ZO_DATA_DIR")
    base = Path(env).expanduser().resolve() if env else repo_root() / "data" / "generated"
    base.mkdir(parents=True, exist_ok=True)
    return base
