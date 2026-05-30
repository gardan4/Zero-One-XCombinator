from __future__ import annotations

import subprocess

import typer
from zo_common import ExperimentConfig, new_run, update_run
from zo_common.registry import run_dir

from zo_train.cluster._slurm import cluster_env, ensure_cluster_env, render_template, slurm_context
from zo_train.cluster.judge import judge_eval, judge_serve, judge_setup, stage_model

app = typer.Typer(help="Submit training / inference jobs to the Leonardo SLURM cluster.", no_args_is_help=True)


def _render_train(subcommand: str, run_id: str, cluster_config_path: str) -> str:
    ctx = slurm_context(
        job_name=run_id,
        subcommand=subcommand,
        config_path=cluster_config_path,
        run_id=run_id,
    )
    return render_template("train.sbatch.j2", **ctx)


@app.command()
def submit(
    config: str = typer.Option(..., "--config", "-c"),
    kind: str = typer.Option("sft", help="sft | grpo"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render the sbatch but don't ssh."),
) -> None:
    ensure_cluster_env()
    cfg = ExperimentConfig.from_yaml(config)
    host, user = cluster_env("ZO_CLUSTER_HOST"), cluster_env("ZO_CLUSTER_USER")
    repo_dir = cluster_env("ZO_CLUSTER_REPO_DIR", "$HOME/Zero-One-Philyr")

    run = new_run(cfg.name, kind, config=cfg.model_dump(), cluster=host, tags=["cluster"])
    local_dir = run_dir(run.id)
    cfg.to_yaml(local_dir / "config.yaml")

    remote_dir = f"{repo_dir}/experiments/{run.id}"
    sbatch = _render_train(kind, run.id, f"{remote_dir}/config.yaml")
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
    ensure_cluster_env()
    host, user = cluster_env("ZO_CLUSTER_HOST"), cluster_env("ZO_CLUSTER_USER")
    if not (host and user):
        typer.secho("Set ZO_CLUSTER_HOST and ZO_CLUSTER_USER in .env.", fg="red")
        raise typer.Exit(1)
    subprocess.run(["ssh", f"{user}@{host}", "squeue --me --long"])


app.command("judge-setup")(judge_setup)
app.command("stage-model")(stage_model)
app.command("judge-eval")(judge_eval)
app.command("judge-serve")(judge_serve)


if __name__ == "__main__":
    app()
