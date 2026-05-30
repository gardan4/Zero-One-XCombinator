from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import typer
from zo_common import ExperimentConfig, new_run, update_run
from zo_common.registry import get_run, run_dir

from zo_train.cluster._remote import (
    cluster_repo_dir,
    ensure_remote_path_vars,
    env,
    expand_cluster_path,
    load_dotenv,
    scp_download,
    scp_upload,
    ssh_run,
    ssh_target,
    write_sbatch,
)
from zo_train.cluster._slurm import ensure_cluster_env, render_template, slurm_context
from zo_train.cluster.judge import judge_eval, judge_serve, judge_setup, stage_model

app = typer.Typer(
    help="Submit training / inference jobs to the Leonardo SLURM cluster.",
    no_args_is_help=True,
)


@dataclass(frozen=True)
class SubmitResult:
    run_id: str
    sbatch_path: Path
    job_id: str | None = None
    rendered_only: bool = False


def _cluster_experiments_dir(repo_dir: str) -> str:
    value = (
        env("ZO_CLUSTER_EXPERIMENTS_DIR", f"{repo_dir}/experiments") or f"{repo_dir}/experiments"
    )
    return expand_cluster_path(value)


def _resolve_config(config: str) -> ExperimentConfig:
    load_dotenv()
    cfg = ExperimentConfig.from_yaml(config)
    if isinstance(cfg.model, str) and cfg.model:
        cfg.model = expand_cluster_path(cfg.model)
    if isinstance(cfg.output_dir, str) and cfg.output_dir:
        cfg.output_dir = expand_cluster_path(cfg.output_dir)
    return cfg


def _effective_kind(cfg: ExperimentConfig, kind: str | None) -> str:
    """YAML ``kind`` wins; CLI ``--kind`` is an optional override."""
    return kind or cfg.kind


def _slurm_overrides_from_cfg(cfg: ExperimentConfig) -> dict:
    """Per-run SLURM overrides declared in a config's ``extra`` block (e.g. a multi-GPU FSDP job).

    ``gpus_per_node`` switches the sbatch to ``accelerate launch`` and auto-rescales mem/cpus; the
    ``slurm_*`` keys pin individual SLURM directives (time/cpus/mem/nodes) for that run, overriding
    the .env defaults — e.g. the FSDP 7B hero wants more cpus and a longer wall-clock than the
    single-GPU default.
    """
    ov: dict = {}
    gpn = cfg.extra.get("gpus_per_node")
    if gpn:
        ov["gpus_per_node"] = int(gpn)
    acc = cfg.extra.get("accelerate_config")
    if acc:
        ov["accelerate_config"] = str(acc)
    for extra_key, ctx_key in (
        ("slurm_time", "time"),
        ("slurm_cpus", "cpus"),
        ("slurm_mem", "mem"),
        ("slurm_nodes", "nodes"),
    ):
        val = cfg.extra.get(extra_key)
        if val:
            ov[ctx_key] = val
    return ov


def _render_train(
    subcommand: str, run_id: str, cluster_config_path: str, **slurm_overrides: object
) -> str:
    repo_dir = cluster_repo_dir()
    ctx = slurm_context(
        job_name=run_id,
        repo_dir=repo_dir,
        experiments_dir=_cluster_experiments_dir(repo_dir),
        hf_home=expand_cluster_path(env("HF_HOME") or f"{repo_dir}/hf_cache"),
        subcommand=subcommand,
        config_path=cluster_config_path,
        run_id=run_id,
        **slurm_overrides,
    )
    return render_template("train.sbatch.j2", **ctx)


def pull_run_from_cluster(run_id: str) -> None:
    """SCP ``meta.json`` + ``metrics.jsonl`` from the cluster run dir to the local registry."""
    ensure_cluster_env()
    ensure_remote_path_vars()
    meta = get_run(run_id)
    if meta is None:
        raise typer.BadParameter(f"run {run_id!r} not found locally")
    target = ssh_target()
    if not target:
        raise typer.BadParameter("Set ZO_CLUSTER_HOST and ZO_CLUSTER_USER in .env.")
    repo_dir = cluster_repo_dir()
    remote_dir = f"{_cluster_experiments_dir(repo_dir)}/{run_id}"
    local_dir = run_dir(run_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "metrics.jsonl"):
        remote = f"{remote_dir}/{name}"
        try:
            scp_download(local_dir / name, target, remote)
        except subprocess.CalledProcessError:
            if name == "meta.json":
                raise
    typer.secho(f"pulled cluster state for {run_id} → {local_dir}", fg="green")


def submit_run(
    config: str,
    kind: str | None = None,
    *,
    dry_run: bool = False,
) -> SubmitResult:
    """Render (and optionally submit) a SLURM training job."""
    ensure_cluster_env()
    ensure_remote_path_vars()
    cfg = _resolve_config(config)
    effective_kind = _effective_kind(cfg, kind)
    from zo_train.preflight import validate_experiment

    validate_experiment(cfg, cluster=True)

    host, user = env("ZO_CLUSTER_HOST"), env("ZO_CLUSTER_USER")
    repo_dir = cluster_repo_dir()
    experiments_dir = _cluster_experiments_dir(repo_dir)

    cfg_tags = list(cfg.extra.get("tags") or [])
    run_tags = ["cluster", *cfg_tags]
    run = new_run(cfg.name, effective_kind, config=cfg.model_dump(), cluster=host, tags=run_tags)
    local_dir = run_dir(run.id)
    cfg.to_yaml(local_dir / "config.yaml")

    remote_dir = f"{experiments_dir}/{run.id}"
    sbatch = _render_train(
        effective_kind, run.id, f"{remote_dir}/config.yaml", **_slurm_overrides_from_cfg(cfg)
    )
    sbatch_path = local_dir / "job.sbatch"
    write_sbatch(sbatch_path, sbatch)

    cluster_note = f"cluster run dir: {remote_dir} (pull with `just cluster-pull-run {run.id}`)"
    if dry_run or not (host and user):
        update_run(
            run.id, status="created", notes=f"rendered sbatch, not submitted. {cluster_note}"
        )
        return SubmitResult(run_id=run.id, sbatch_path=sbatch_path, rendered_only=True)

    target = f"{user}@{host}"
    ssh_run(target, f"mkdir -p {remote_dir}", check=True)
    scp_upload(local_dir / "meta.json", target, f"{remote_dir}/meta.json")
    scp_upload(local_dir / "config.yaml", target, f"{remote_dir}/config.yaml")
    scp_upload(sbatch_path, target, f"{remote_dir}/job.sbatch")
    update_run(run.id, notes=cluster_note)
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
    kind: str | None = typer.Option(None, help="Override YAML kind (default: read from config)."),
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


@app.command("pull-run")
def pull_run_cmd(
    run_id: str = typer.Argument(..., help="Run id to pull meta/metrics for from the cluster."),
) -> None:
    """Sync ``meta.json`` + ``metrics.jsonl`` from cluster scratch to the local run store."""
    pull_run_from_cluster(run_id)


@app.command("leonardo-smoke")
def leonardo_smoke_cmd(
    config: str = typer.Option(
        "packages/training/configs/leonardo_smoke_hf.yaml",
        "--config",
        "-c",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Render sbatch only; skip sync/prestage."
    ),
    skip_sync: bool = typer.Option(False, help="Skip repo sync (already on cluster)."),
    skip_prestage: bool = typer.Option(False, help="Skip login-node prestage."),
    submit_only: bool = typer.Option(False, help="Skip sync/prestage; submit SLURM job only."),
    wait_upload: bool = typer.Option(False, "--wait-upload", help="Poll job and upload to HF."),
) -> None:
    """Full Leonardo smoke pipeline from Windows, macOS, or Linux (Python + ssh only)."""
    from zo_train.cluster.leonardo_smoke import run_leonardo_smoke

    run_leonardo_smoke(
        config=config,
        dry_run=dry_run,
        skip_sync=skip_sync,
        skip_prestage=skip_prestage,
        submit_only=submit_only,
        wait_upload=wait_upload,
    )


@app.command()
def watch() -> None:
    """Show your SLURM queue on the cluster."""
    ensure_cluster_env()
    target = ssh_target()
    if not target:
        typer.secho("Set ZO_CLUSTER_HOST and ZO_CLUSTER_USER in .env.", fg="red")
        raise typer.Exit(1)
    ssh_run(target, "squeue --me --long")


app.command("judge-setup")(judge_setup)
app.command("stage-model")(stage_model)
app.command("judge-eval")(judge_eval)
app.command("judge-serve")(judge_serve)


if __name__ == "__main__":
    app()
