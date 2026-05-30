"""OS helpers — cluster CLIs on Windows, macOS, and Leonardo Linux."""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path


def is_windows() -> bool:
    return sys.platform == "win32" or os.name == "nt"


def has_slurm() -> bool:
    return shutil.which("sbatch") is not None


def has_ssh_tools() -> bool:
    return shutil.which("ssh") is not None and shutil.which("scp") is not None


def has_putty_tools() -> bool:
    return shutil.which("plink") is not None and shutil.which("pscp") is not None


def posix_path(path: str | Path) -> str:
    return Path(path).as_posix()


def expand_env_refs(text: str) -> str:
    """Expand ``${VAR}`` and ``$VAR`` using ``os.environ`` (works on Windows, unlike expandvars)."""

    def brace(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(0))

    def plain(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(0))

    text = re.sub(r"\$\{([^}]+)\}", brace, text)
    return re.sub(r"\$([A-Z_][A-Z0-9_]*)", plain, text)


def remote_expand(path: str) -> str:
    """Expand ``$HOME`` / ``$SCRATCH`` using ``ZO_CLUSTER_HOME`` / ``ZO_CLUSTER_SCRATCH``."""
    home = os.environ.get("ZO_CLUSTER_HOME")
    scratch = os.environ.get("ZO_CLUSTER_SCRATCH")
    if home:
        path = path.replace("$HOME", home)
    if scratch:
        path = path.replace("$SCRATCH", scratch)
    return path.replace("\\", "/")


def to_cluster_path(local: Path | str, cluster_repo: str) -> str:
    """Map a local repo-relative path to ``<cluster_repo>/…`` for remote scripts."""
    from zo_common.paths import repo_root

    local = Path(local)
    repo = cluster_repo.rstrip("/")
    try:
        rel = local.resolve().relative_to(repo_root().resolve())
        return f"{repo}/{rel.as_posix()}"
    except ValueError:
        return posix_path(local)
