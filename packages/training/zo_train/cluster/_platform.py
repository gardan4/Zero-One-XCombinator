"""OS helpers — keep judge/cluster CLIs working on Windows, macOS, and Leonardo Linux."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def is_windows() -> bool:
    return sys.platform == "win32" or os.name == "nt"


def has_slurm() -> bool:
    return shutil.which("sbatch") is not None


def has_ssh_tools() -> bool:
    return shutil.which("ssh") is not None and shutil.which("scp") is not None


def posix_path(path: str | Path) -> str:
    """Normalize to forward slashes (safe inside SLURM bash scripts)."""
    return Path(path).as_posix()


def to_cluster_path(local: Path | str, cluster_repo: str) -> str:
    """Map a local repo-relative path to ``<cluster_repo>/…`` for sbatch scripts.

    When preparing jobs from Windows/macOS, ``ZO_CLUSTER_REPO_DIR`` in ``.env`` must be the
    **Leonardo** repo path (e.g. ``$HOME/Zero-One-Philyr``), not a local ``C:\\…`` path.
    """
    from zo_common.paths import repo_root

    local = Path(local)
    repo = cluster_repo.rstrip("/")
    try:
        rel = local.resolve().relative_to(repo_root().resolve())
        return f"{repo}/{rel.as_posix()}"
    except ValueError:
        return posix_path(local)


def to_cluster_model_path(model: str, cluster_repo: str) -> str:
    """Ensure model paths in sbatch are HF ids or cluster POSIX paths, never ``C:\\…``."""
    from zo_train.cluster._slurm import is_hf_repo_id

    if is_hf_repo_id(model):
        return model
    if model.startswith("$") or model.startswith("/leonardo") or model.startswith("/scratch"):
        return model.replace("\\", "/")
    p = Path(os.path.expandvars(model))
    if p.is_absolute():
        return to_cluster_path(p, cluster_repo)
    return model.replace("\\", "/")


def local_model_cache_dir() -> Path:
    """Cross-platform default for staging HF weights on a laptop (not Leonardo ``$SCRATCH``)."""
    explicit = os.environ.get("ZO_MODEL_CACHE_DIR")
    if explicit:
        return Path(os.path.expandvars(explicit)).expanduser()
    if os.environ.get("SCRATCH"):
        return Path(os.environ["SCRATCH"]) / "zo-models"
    return Path.home() / ".cache" / "zo-models"
