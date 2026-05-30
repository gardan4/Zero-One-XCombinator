"""Cross-platform Leonardo smoke finetune orchestration (Windows / macOS / Linux)."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import typer

from zo_common.paths import repo_root

from zo_train.cluster._remote import (
    cluster_repo_dir,
    ensure_remote_path_vars,
    env,
    load_dotenv,
    push_cluster_env,
    ssh_capture,
    ssh_run,
    ssh_target,
    sync_repo_to_cluster,
)


def _require_env(*keys: str) -> None:
    missing = [k for k in keys if not env(k)]
    if missing:
        typer.secho(f"Set in .env: {', '.join(missing)}", fg="red")
        raise typer.Exit(1)


def run_leonardo_smoke(
    *,
    config: str = "packages/training/configs/leonardo_smoke_hf.yaml",
    dry_run: bool = False,
    skip_sync: bool = False,
    skip_prestage: bool = False,
    wait_upload: bool = False,
) -> None:
    load_dotenv()
    _require_env("ZO_CLUSTER_HOST", "ZO_CLUSTER_USER", "ZO_CLUSTER_REPO_DIR", "HF_TOKEN")
    ensure_remote_path_vars()
    target = ssh_target()
    if not target:
        typer.secho("Could not build SSH target.", fg="red")
        raise typer.Exit(1)

    remote_repo = cluster_repo_dir()

    if dry_run:
        typer.secho("Dry-run — skipping sync/prestage; rendering sbatch only.", fg="yellow")
        _submit(config, dry_run=True)
        return

    if not skip_sync:
        typer.secho(f"==> Sync repo to {target}:{remote_repo}", fg="cyan")
        sync_repo_to_cluster(target, remote_repo)
        push_cluster_env(target, remote_repo)

    if not skip_prestage:
        typer.secho("==> Pre-stage GPU environment and base model on login node", fg="cyan")
        ssh_run(target, f"bash '{remote_repo}/scripts/leonardo_remote_prestage.sh'", check=True)

    typer.secho("==> Submit Leonardo smoke finetune", fg="cyan")
    run_id, job_id = _submit(config, dry_run=False)

    if wait_upload and run_id:
        _wait_and_upload(target, remote_repo, run_id, job_id)


def _submit(config: str, *, dry_run: bool) -> tuple[str | None, str | None]:
    cmd = ["uv", "run", "zo-cluster", "submit", "--config", config]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, cwd=repo_root(), capture_output=True, text=True)
    out = (result.stdout or "") + (result.stderr or "")
    typer.echo(out.strip())
    if result.returncode != 0:
        raise typer.Exit(result.returncode)
    run_m = re.search(r"run (\d{8}_\d{6}_\S+)", out)
    job_m = re.search(r"submitted SLURM job (\d+)", out)
    return (run_m.group(1) if run_m else None, job_m.group(1) if job_m else None)


def _wait_and_upload(
    target: str,
    remote_repo: str,
    run_id: str,
    job_id: str | None,
    *,
    poll_seconds: int = 30,
) -> None:
    typer.secho(f"==> Waiting for SLURM job {job_id or '(any)'}", fg="cyan")
    while True:
        q = ssh_capture(target, "squeue --me --noheader 2>/dev/null || true")
        if job_id:
            if job_id not in q:
                break
        elif not q.strip():
            break
        typer.echo(f"  still running… ({time.strftime('%H:%M:%S')})")
        time.sleep(poll_seconds)

    log_glob = f"{remote_repo}/slurm_logs/{run_id}-*.out"
    tail = ssh_capture(
        target,
        f"ls -1t {log_glob} 2>/dev/null | head -1 | xargs -r tail -n 80",
    )
    if tail:
        typer.echo(tail)

    typer.secho("==> Upload artifacts to Hugging Face (login node)", fg="cyan")
    ssh_run(
        target,
        f"bash '{remote_repo}/scripts/leonardo_upload_artifact.sh' '{run_id}'",
        check=True,
    )
