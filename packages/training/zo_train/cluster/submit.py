from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer
from zo_common import ExperimentConfig, new_run, update_run
from zo_common.paths import repo_root
from zo_common.registry import run_dir

app = typer.Typer(help="Submit training jobs to the Leonardo SLURM cluster.", no_args_is_help=True)


def _load_dotenv() -> None:
    """Best-effort: populate os.environ from <repo>/.env (no extra dependency)."""
    env_file = repo_root() / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def _env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def _render(subcommand: str, run_id: str, cluster_config_path: str) -> str:
    from jinja2 import Template

    repo_dir = _env("ZO_CLUSTER_REPO_DIR", "$HOME/Zero-One-Philyr")
    ctx = dict(
        job_name=run_id,
        account=_env("ZO_SLURM_ACCOUNT", ""),
        partition=_env("ZO_SLURM_PARTITION", "boost_usr_prod"),
        qos=_env("ZO_SLURM_QOS", ""),
        nodes=int(_env("ZO_SLURM_NODES", "1")),
        gpus_per_node=int(_env("ZO_SLURM_GPUS_PER_NODE", "4")),
        cpus=int(_env("ZO_SLURM_CPUS", "8")),
        time=_env("ZO_SLURM_TIME", "02:00:00"),
        repo_dir=repo_dir,
        experiments_dir=_env("ZO_CLUSTER_EXPERIMENTS_DIR", f"{repo_dir}/experiments"),
        hf_home=_env("HF_HOME") or f"{repo_dir}/hf_cache",
        subcommand=subcommand,
        config_path=cluster_config_path,
        run_id=run_id,
    )
    tpl = (Path(__file__).parent / "slurm" / "train.sbatch.j2").read_text()
    return Template(tpl).render(**ctx)


@app.command()
def submit(
    config: str = typer.Option(..., "--config", "-c"),
    kind: str = typer.Option("sft", help="sft | grpo"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render the sbatch but don't ssh."),
) -> None:
    _load_dotenv()
    cfg = ExperimentConfig.from_yaml(config)
    host, user = _env("ZO_CLUSTER_HOST"), _env("ZO_CLUSTER_USER")
    repo_dir = _env("ZO_CLUSTER_REPO_DIR", "$HOME/Zero-One-Philyr")

    run = new_run(cfg.name, kind, config=cfg.model_dump(), cluster=host, tags=["cluster"])
    local_dir = run_dir(run.id)
    cfg.to_yaml(local_dir / "config.yaml")

    remote_dir = f"{repo_dir}/experiments/{run.id}"
    sbatch = _render(kind, run.id, f"{remote_dir}/config.yaml")
    (local_dir / "job.sbatch").write_text(sbatch)
    typer.echo(f"run {run.id} -> {local_dir / 'job.sbatch'}")

    if dry_run or not (host and user):
        if not (host and user):
            typer.secho(
                "ZO_CLUSTER_HOST / ZO_CLUSTER_USER not set — wrote sbatch locally only.",
                fg="yellow",
            )
        update_run(run.id, status="queued", notes="rendered, not submitted")
        raise typer.Exit()

    target = f"{user}@{host}"
    subprocess.run(["ssh", target, f"mkdir -p {remote_dir}"], check=True)
    subprocess.run(
        ["scp", str(local_dir / "config.yaml"), f"{target}:{remote_dir}/config.yaml"], check=True
    )
    subprocess.run(
        ["scp", str(local_dir / "job.sbatch"), f"{target}:{remote_dir}/job.sbatch"], check=True
    )
    result = subprocess.run(
        ["ssh", target, f"cd {repo_dir} && sbatch {remote_dir}/job.sbatch"],
        capture_output=True,
        text=True,
    )
    typer.echo((result.stdout + result.stderr).strip())
    job_id = (
        result.stdout.strip().split()[-1]
        if result.returncode == 0 and result.stdout.strip()
        else None
    )
    update_run(run.id, status="queued" if job_id else "failed", slurm_job_id=job_id)
    if job_id:
        typer.secho(f"submitted SLURM job {job_id}  (watch: `just cluster-watch`)", fg="green")


@app.command()
def watch() -> None:
    """Show your SLURM queue on the cluster."""
    _load_dotenv()
    host, user = _env("ZO_CLUSTER_HOST"), _env("ZO_CLUSTER_USER")
    if not (host and user):
        typer.secho("Set ZO_CLUSTER_HOST and ZO_CLUSTER_USER in .env.", fg="red")
        raise typer.Exit(1)
    subprocess.run(["ssh", f"{user}@{host}", "squeue --me --long"])


if __name__ == "__main__":
    app()
