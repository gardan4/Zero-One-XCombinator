"""SSH/SCP helpers shared by zo-cluster submit and Leonardo smoke orchestration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from zo_common.paths import repo_root

from zo_train.cluster._platform import (
    expand_env_refs,
    has_putty_tools,
    has_ssh_tools,
    remote_expand,
)


def load_dotenv(path: Path | None = None) -> None:
    env_file = path or (repo_root() / ".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, val.strip().strip('"').strip("'"))


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def cluster_repo_dir() -> str:
    raw = env("ZO_CLUSTER_REPO_DIR", "$HOME/Zero-One-Philyr") or "$HOME/Zero-One-Philyr"
    return remote_expand(expand_env_refs(raw))


def expand_cluster_path(path: str) -> str:
    return remote_expand(expand_env_refs(path))


def ssh_target() -> str | None:
    host, user = env("ZO_CLUSTER_HOST"), env("ZO_CLUSTER_USER")
    if host and user:
        return f"{user}@{host}"
    return None


def ensure_remote_path_vars(target: str | None = None) -> None:
    """Fill ``ZO_CLUSTER_HOME`` / ``ZO_CLUSTER_SCRATCH`` via one SSH probe when unset."""
    if env("ZO_CLUSTER_HOME") and env("ZO_CLUSTER_SCRATCH"):
        return
    target = target or ssh_target()
    if not target or not (has_ssh_tools() or has_putty_tools()):
        return
    try:
        out = ssh_capture(target, 'printf "HOME=%s\\nSCRATCH=%s\\n" "$HOME" "${SCRATCH:-}"')
    except subprocess.CalledProcessError:
        return
    for line in out.splitlines():
        if line.startswith("HOME=") and line[5:]:
            os.environ.setdefault("ZO_CLUSTER_HOME", line[5:])
        elif line.startswith("SCRATCH=") and line[8:]:
            os.environ.setdefault("ZO_CLUSTER_SCRATCH", line[8:])


def _use_putty() -> bool:
    password = env("ZO_CLUSTER_PASSWORD")
    return bool(password and has_putty_tools())


def ssh_run(
    target: str,
    command: str,
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    hostkey = env("ZO_CLUSTER_HOSTKEY")
    if _use_putty():
        args = ["plink", "-batch", "-ssh"]
        if hostkey:
            args += ["-hostkey", hostkey]
        args += ["-pw", env("ZO_CLUSTER_PASSWORD", ""), target, command]
    else:
        args = [
            "ssh",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=4",
            target,
            command,
        ]
    return subprocess.run(
        args,
        check=check,
        capture_output=capture_output,
        text=text,
    )


def ssh_capture(target: str, command: str) -> str:
    result = ssh_run(target, command, capture_output=True)
    return (result.stdout or "").strip()


def scp_download(local: Path, target: str, remote: str) -> None:
    """Download ``target:remote`` → ``local`` (mirror of ``scp_upload``)."""
    local.parent.mkdir(parents=True, exist_ok=True)
    hostkey = env("ZO_CLUSTER_HOSTKEY")
    if _use_putty():
        args = ["pscp", "-batch"]
        if hostkey:
            args += ["-hostkey", hostkey]
        args += ["-pw", env("ZO_CLUSTER_PASSWORD", ""), f"{target}:{remote}", str(local)]
    else:
        args = [
            "scp",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=4",
            f"{target}:{remote}",
            str(local),
        ]
    subprocess.run(args, check=True)


def scp_upload(local: Path, target: str, remote: str) -> None:
    hostkey = env("ZO_CLUSTER_HOSTKEY")
    if _use_putty():
        args = ["pscp", "-batch"]
        if hostkey:
            args += ["-hostkey", hostkey]
        args += ["-pw", env("ZO_CLUSTER_PASSWORD", ""), str(local), f"{target}:{remote}"]
    else:
        args = [
            "scp",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=4",
            str(local),
            f"{target}:{remote}",
        ]
    subprocess.run(args, check=True)


def write_sbatch(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


_SYNC_EXCLUDES = (
    ".git",
    ".env",
    ".venv",
    "apps/frontend/node_modules",
    "experiments",
    "hf_cache",
    "slurm_logs",
    "wandb",
)


def sync_repo_to_cluster(target: str, remote_repo: str) -> None:
    """Tar-based repo sync (Windows 10+, macOS, Linux ÔÇö no rsync required)."""
    import tarfile
    import tempfile

    root = repo_root()
    ssh_run(target, f"mkdir -p '{remote_repo}'", check=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = Path(tmp.name)
    try:
        with tarfile.open(tar_path, "w") as tar:
            for item in root.iterdir():
                if item.name in _SYNC_EXCLUDES:
                    continue
                tar.add(item, arcname=item.name)
        scp_upload(tar_path, target, f"{remote_repo}/repo.tar")
        ssh_run(
            target,
            f"cd '{remote_repo}' && tar -xf repo.tar && rm -f repo.tar && "
            "find scripts -name '*.sh' -exec sed -i 's/\\r$//' {} + 2>/dev/null || true",
            check=True,
        )
    finally:
        tar_path.unlink(missing_ok=True)


def push_cluster_env(target: str, remote_repo: str) -> None:
    """Copy safe subset of local ``.env`` to the cluster repo (secrets for HF/W&B)."""
    import tempfile

    env_path = repo_root() / ".env"
    if not env_path.is_file():
        return
    lines: list[str] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("ZO_CLUSTER_PASSWORD=") or s.startswith("ZO_CLUSTER_HOSTKEY="):
            continue
        key = s.split("=", 1)[0]
        if key.startswith("ZO_") or key in ("HF_TOKEN", "HF_HOME") or key.startswith("WANDB_"):
            lines.append(s)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".env", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = Path(tmp.name)
    try:
        scp_upload(tmp_path, target, f"{remote_repo}/.env")
    finally:
        tmp_path.unlink(missing_ok=True)
