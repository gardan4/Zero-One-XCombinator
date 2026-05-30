from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from zo_common import ExperimentConfig, new_run, update_run
from zo_common.registry import run_dir

from zo_train.cluster._remote import (
    cluster_repo_dir,
    ensure_remote_path_vars,
    env,
    expand_cluster_path,
    load_dotenv,
    scp_upload,
    ssh_run,
    ssh_target,
    write_sbatch,
)

app = typer.Typer(help="Submit training jobs to the Leonardo SLURM cluster.", no_args_is_help=True)


@dataclass(frozen=True)
class SubmitResult:
    run_id: str
    sbatch_path: Path
    job_id: str | None = None
    rendered_only: bool = False


def _cluster_experiments_dir(repo_dir: str) -> str:
    value = env("ZO_CLUSTER_EXPERIMENTS_DIR", f"{repo_dir}/experiments") or f"{repo_dir}/experiments"
    return expand_cluster_path(value)


def _resolve_config(config: str) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(config)
    if isinstance(cfg.model, str) and cfg.model:
        cfg.model = expand_cluster_path(cfg.model)
    return cfg


def _render(subcommand: str, run_id: str, cluster_config_path: str) -> str:
    from jinja2 import Template

    repo_dir = cluster_repo_dir()
    gpus = int(env("ZO_SLURM_GPUS_PER_NODE", "1"))
    mem = env("ZO_SLURM_MEM") or f"{120 * gpus}GB"
    cpus = int(env("ZO_SLURM_CPUS") or str(8 * gpus))
    ctx = dict(
        job_name=run_id,
        account=env("ZO_SLURM_ACCOUNT", ""),
        partition=env("ZO_SLURM_PARTITION", "boost_usr_prod"),
        reservation=env("ZO_SLURM_RESERVATION", ""),
        qos=env("ZO_SLURM_QOS", ""),
        nodes=int(env("ZO_SLURM_NODES", "1")),
        gpus_per_node=gpus,
        mem=mem,
        cpus=cpus,
        time=env("ZO_SLURM_TIME", "02:00:00"),
        repo_dir=repo_dir,
        experiments_dir=_cluster_experiments_dir(repo_dir),
        hf_home=expand_cluster_path(env("HF_HOME") or f"{repo_dir}/hf_cache"),
        proxy=env("ZO_CLUSTER_PROXY", ""),
        subcommand=subcommand,
        config_path=cluster_config_path,
        run_id=run_id,
    )
    tpl = (Path(__file__).parent / "slurm" / "train.sbatch.j2").read_text()
    return Template(tpl).render(**ctx)


def submit_run(
    config: str,
    kind: str = "sft",
    *,
    dry_run: bool = False,
) -> SubmitResult:
    """Render (and optionally submit) a SLURM training job. Returns run metadata."""
    load_dotenv()
    ensure_remote_path_vars()
    cfg = _resolve_config(config)
    host, user = env("ZO_CLUSTER_HOST"), env("ZO_CLUSTER_USER")
    repo_dir = cluster_repo_dir()
    experiments_dir = _cluster_experiments_dir(repo_dir)

    run = new_run(cfg.name, kind, config=cfg.model_dump(), cluster=host, tags=["cluster"])
    local_dir = run_dir(run.id)
    cfg.to_yaml(local_dir / "config.yaml")

    remote_dir = f"{experiments_dir}/{run.id}"
    sbatch = _render(kind, run.id, f"{remote_dir}/config.yaml")
    sbatch_path = local_dir / "job.sbatch"
    write_sbatch(sbatch_path, sbatch)

    if dry_run or not (host and user):
        update_run(run.id, status="queued", notes="rendered, not submitted")
        return SubmitResult(run_id=run.id, sbatch_path=sbatch_path, rendered_only=True)

    target = f"{user}@{host}"
    ssh_run(target, f"mkdir -p {remote_dir}", check=True)
    scp_upload(local_dir / "meta.json", target, f"{remote_dir}/meta.json")
    scp_upload(local_dir / "config.yaml", target, f"{remote_dir}/config.yaml")
    scp_upload(sbatch_path, target, f"{remote_dir}/job.sbatch")
    result = ssh_run(
        target,
        f"cd {repo_dir} && sbatch {remote_dir}/job.sbatch",
        capture_output=True,
    )
    job_id = (
        result.stdout.strip().split()[-1]
        if result.returncode == 0 and result.stdout.strip()
        else None
    )
    update_run(run.id, status="queued" if job_id else "failed", slurm_job_id=job_id)
    if not job_id:
        raise RuntimeError((result.stdout + result.stderr).strip() or "sbatch failed")
    return SubmitResult(run_id=run.id, sbatch_path=sbatch_path, job_id=job_id)


@app.command()
def submit(
    config: str = typer.Option(..., "--config", "-c"),
    kind: str = typer.Option("sft", help="sft | grpo"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render the sbatch but don't ssh."),
) -> None:
    result = submit_run(config, kind, dry_run=dry_run)
    typer.echo(f"run {result.run_id} -> {result.sbatch_path}")
    if result.rendered_only:
        if not (env("ZO_CLUSTER_HOST") and env("ZO_CLUSTER_USER")):
            typer.secho(
                "ZO_CLUSTER_HOST / ZO_CLUSTER_USER not set — wrote sbatch locally only.",
                fg="yellow",
            )
        raise typer.Exit()
    typer.secho(f"submitted SLURM job {result.job_id}  (watch: `just cluster-watch`)", fg="green")


@app.command("leonardo-smoke")
def leonardo_smoke_cmd(
    config: str = typer.Option(
        "packages/training/configs/leonardo_smoke_hf.yaml",
        "--config",
        "-c",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render sbatch only; skip sync/prestage."),
    skip_sync: bool = typer.Option(False, help="Skip repo sync (already on cluster)."),
    skip_prestage: bool = typer.Option(False, help="Skip login-node prestage."),
    wait_upload: bool = typer.Option(False, "--wait-upload", help="Poll job and upload to HF."),
) -> None:
    """Full Leonardo smoke pipeline from Windows, macOS, or Linux (Python + ssh only)."""
    from zo_train.cluster.leonardo_smoke import run_leonardo_smoke

    run_leonardo_smoke(
        config=config,
        dry_run=dry_run,
        skip_sync=skip_sync,
        skip_prestage=skip_prestage,
        wait_upload=wait_upload,
    )


@app.command()
def watch() -> None:
    """Show your SLURM queue on the cluster."""
    load_dotenv()
    target = ssh_target()
    if not target:
        typer.secho("Set ZO_CLUSTER_HOST and ZO_CLUSTER_USER in .env.", fg="red")
        raise typer.Exit(1)
    ssh_run(target, "squeue --me --long")


if __name__ == "__main__":
    app()
