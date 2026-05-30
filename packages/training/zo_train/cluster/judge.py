"""Judge-friendly Leonardo inference: one-time setup, model staging, batch eval, optional vLLM serve.

Designed for reproducibility — a judge clones the repo on a Leonardo login node, fills ``.env``,
runs ``just judge-setup`` once, then ``just judge-eval``. No manual SLURM editing required.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer
from zo_common import new_run, update_run
from zo_common.paths import repo_root
from zo_common.registry import run_dir

from zo_train.cluster._platform import (
    has_slurm,
    has_ssh_tools,
    to_cluster_model_path,
    to_cluster_path,
)
from zo_train.cluster._slurm import (
    cluster_env,
    ensure_cluster_env,
    is_hf_repo_id,
    render_template,
    resolve_infer_model,
    slurm_context,
    staged_model_path,
)


def _repo() -> Path:
    return repo_root()


def _cluster_repo() -> str:
    return cluster_env("ZO_CLUSTER_REPO_DIR", "$HOME/Zero-One-Philyr") or "$HOME/Zero-One-Philyr"


def _cluster_path(local: Path) -> str:
    """Map a local repo-relative path to the cluster repo dir (for sbatch scripts)."""
    return to_cluster_path(local, _cluster_repo())


def _write_sbatch(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _eval_dir() -> Path:
    return Path(cluster_env("ZO_JUDGE_EVAL_DIR", "extras/eval_local") or "extras/eval_local")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    typer.echo("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or _repo(), check=True)


def _ssh_target() -> str | None:
    host, user = cluster_env("ZO_CLUSTER_HOST"), cluster_env("ZO_CLUSTER_USER")
    if host and user:
        return f"{user}@{host}"
    return None


def _on_login_node() -> bool:
    return cluster_env("ZO_CLUSTER_ON_LOGIN", "").lower() in ("1", "true", "yes")


def _submit_sbatch(sbatch_path: Path, *, local: bool) -> str | None:
    repo_dir = _cluster_repo()
    use_local = (local or _on_login_node()) and has_slurm()
    if use_local:
        result = subprocess.run(
            ["sbatch", str(sbatch_path)],
            cwd=_repo(),
            capture_output=True,
            text=True,
        )
    elif has_ssh_tools() and _ssh_target():
        target = _ssh_target()
        remote = f"{repo_dir}/{sbatch_path.relative_to(_repo()).as_posix()}"
        subprocess.run(["scp", str(sbatch_path), f"{target}:{remote}"], check=True)
        result = subprocess.run(
            ["ssh", target, f"cd {repo_dir} && sbatch {remote}"],
            capture_output=True,
            text=True,
        )
    else:
        return None
    typer.echo((result.stdout + result.stderr).strip())
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    return parts[-1] if parts else None


def judge_setup(
    family: str = typer.Option("MOSFET", help="Family for the local eval hold-out set."),
    skip_data: bool = typer.Option(False, help="Skip datagen + make-local-eval (inputs already present)."),
) -> None:
    """One-time prep on a Leonardo login node: GPU deps, corpus, local eval inputs."""
    ensure_cluster_env()
    typer.secho("Installing GPU stack (run once on login node)...", fg="cyan")
    _run(["uv", "sync", "--extra", "gpu"])

    eval_dir = _eval_dir()
    if skip_data and (eval_dir / "gold.json").exists():
        typer.secho(f"Eval inputs already at {eval_dir} — skipping datagen.", fg="yellow")
    else:
        typer.secho("Generating deterministic corpus + local eval set...", fg="cyan")
        _run(["uv", "run", "python", "-m", "zo_train.datagen", "--build"])
        eval_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "uv",
                "run",
                "zo-track",
                "make-local-eval",
                "--family",
                family,
                "--out",
                str(eval_dir),
            ]
        )
    typer.secho(
        f"Setup complete. Eval inputs: {eval_dir}\n"
        "Next: `just judge-stage` (if using HF weights) then `just judge-eval`.",
        fg="green",
    )


def stage_model(
    model: str = typer.Option(None, "--model", "-m", help="HF repo id (default: ZO_INFER_MODEL from .env)."),
    out: str = typer.Option(None, help="Local directory (default: $SCRATCH/zo-models/<repo-slug>)."),
) -> None:
    """Download a Hugging Face checkpoint to shared storage (login node — has internet)."""
    ensure_cluster_env()
    hf_id = (model or cluster_env("ZO_INFER_MODEL") or "").strip()
    if not hf_id or not is_hf_repo_id(hf_id):
        typer.secho("Pass --model org/name or set ZO_INFER_MODEL in .env.", fg="red")
        raise typer.Exit(1)

    dest = Path(os.path.expandvars(out or staged_model_path(hf_id))).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    typer.secho(f"Downloading {hf_id} → {dest}", fg="cyan")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise typer.BadParameter("Run `just judge-setup` (or `uv sync --extra gpu`) first.") from e

    token = cluster_env("HF_TOKEN") or cluster_env("HUGGINGFACE_HUB_TOKEN")
    snapshot_download(
        repo_id=hf_id,
        local_dir=str(dest),
        token=token,
        local_dir_use_symlinks=False,
    )
    typer.secho(f"Staged at {dest}\nAdd to .env:  ZO_INFER_MODEL_PATH={dest}", fg="green")


def _eval_dir_for(eval_set: str, eval_dir: str | None) -> Path:
    if eval_dir:
        return Path(eval_dir)
    if eval_set == "kickoff":
        return Path("data/industrial-infineon/eval")
    return Path(cluster_env("ZO_JUDGE_EVAL_DIR", "extras/eval_local") or "extras/eval_local")


def judge_eval(
    model: str = typer.Option(None, "--model", "-m", help="HF repo or local checkpoint path."),
    predictor: str = typer.Option("hf", help="hf (batch transformers) | llm (needs vLLM)."),
    version: str = typer.Option(None, help="Repro version tag (default: infer model slug)."),
    eval_dir: str = typer.Option(None, help="Dir with eval_input_*.csv (+ gold.json for labeled)."),
    eval_set: str = typer.Option("local", help="eval-set tag: local|kickoff"),
    tasks: str = typer.Option("nextstep,completion,anomaly", help="Task subset."),
    tags: str = typer.Option("judge,repro,split:id,real-run,reportable", help="Comma-separated run tags."),
    train_run: str = typer.Option(None, "--train-run", help="Training run id to tag and link."),
    time: str = typer.Option(None, help="SLURM wall time (default ZO_SLURM_INFER_TIME or 00:30:00)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render sbatch only."),
    local: bool = typer.Option(False, "--local", help="sbatch on this login node."),
    stage: bool = typer.Option(True, "--stage/--no-stage", help="Auto-download HF weights if missing."),
    self_check: bool = typer.Option(False, "--self-check", help="Run official eval_metrics.py (needs gold)."),
    promote: str = typer.Option(None, "--promote", help="Copy results to extras/results/<slug>/ on compute node."),
) -> None:
    """Submit a GPU batch job that runs track eval and writes scored results."""
    ensure_cluster_env()
    eval_path = _eval_dir_for(eval_set, eval_dir)
    valid = eval_path / "eval_input_valid.csv"
    anomaly = eval_path / "eval_input_anomaly.csv"
    gold = eval_path / "gold.json"
    for p in (valid, anomaly):
        if not p.exists():
            typer.secho(f"Missing {p}. Run `just judge-setup` or use --eval-set kickoff.", fg="red")
            raise typer.Exit(1)
    if eval_set != "kickoff" and not gold.exists():
        typer.secho(
            f"Missing {gold} for labeled eval. Run `just judge-setup` or `just local-eval FAMILY`.",
            fg="red",
        )
        raise typer.Exit(1)
    if self_check and not gold.exists():
        typer.secho("--self-check requires gold.json in the eval dir.", fg="red")
        raise typer.Exit(1)

    hf_raw = (model or cluster_env("ZO_INFER_MODEL") or "").strip()
    if stage and not dry_run and hf_raw and is_hf_repo_id(hf_raw):
        dest = Path(os.path.expandvars(staged_model_path(hf_raw))).expanduser()
        if not (dest / "config.json").exists() and not list(dest.glob("*.safetensors")):
            typer.secho(f"Model not staged at {dest} — downloading...", fg="yellow")
            stage_model(model=hf_raw, out=str(dest))
        os.environ.setdefault("ZO_INFER_MODEL_PATH", str(dest))

    try:
        model_path = to_cluster_model_path(
            os.path.expandvars(resolve_infer_model(model)), _cluster_repo()
        )
    except ValueError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1) from e

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if train_run and f"train-run:{train_run}" not in tag_list:
        tag_list.append(f"train-run:{train_run}")
    ver = (version or Path(model_path).name).strip()
    mref = hf_raw if is_hf_repo_id(hf_raw) else None

    run = new_run(
        f"judge-eval@{ver}",
        "eval",
        config={
            "model": model_path,
            "predictor": predictor,
            "version": ver,
            "model_ref": mref,
            "eval_set": eval_set,
            "eval_dir": str(eval_path),
            "tasks": tasks,
            "train_run_id": train_run,
        },
        tags=tag_list,
        cluster=cluster_env("ZO_CLUSTER_HOST"),
    )
    out_dir = run_dir(run.id) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = slurm_context(
        job_name=f"zo-infer-{run.id[-8:]}",
        time=time or cluster_env("ZO_SLURM_INFER_TIME", "00:30:00"),
        predictor=predictor,
        model_path=model_path,
        valid_csv=_cluster_path(valid),
        anomaly_csv=_cluster_path(anomaly),
        gold_json=_cluster_path(gold) if gold.exists() else "",
        tasks=tasks,
        tags=",".join(tag_list),
        version=ver,
        model_ref=mref or "",
        eval_set=eval_set,
        train_run_id=train_run or "",
        out_dir=_cluster_path(out_dir),
        run_id=run.id,
        self_check=self_check,
        promote=promote or "",
        run_proxy=eval_set == "kickoff",
    )
    sbatch = render_template("infer.sbatch.j2", **ctx)
    sbatch_path = run_dir(run.id) / "infer.sbatch"
    _write_sbatch(sbatch_path, sbatch)
    typer.echo(f"run {run.id} → {sbatch_path}")

    if dry_run:
        update_run(run.id, status="queued", notes="rendered infer.sbatch (dry-run)")
        typer.secho("Dry-run — sbatch not submitted.", fg="yellow")
        raise typer.Exit()

    job_id = _submit_sbatch(sbatch_path, local=local)
    if not job_id:
        hint = (
            "Wrote sbatch locally. On Leonardo login node run:\n"
            f"  export ZO_CLUSTER_ON_LOGIN=1\n"
            f"  sbatch {sbatch_path.relative_to(_repo()).as_posix()}"
        )
        if not has_slurm() and has_ssh_tools() and _ssh_target():
            hint = (
                "No local sbatch — submit remotely with:\n"
                f"  uv run zo-cluster judge-eval --eval-dir {eval_path.as_posix()}"
            )
        elif not has_slurm():
            hint = (
                "No sbatch on this machine (expected on Windows/macOS). Either:\n"
                "  • SSH to Leonardo and run with --local, or\n"
                "  • Use --dry-run here, copy infer.sbatch to the cluster, or\n"
                "  • Run local inference: uv run zo-track predict -p hf --model …"
            )
        typer.secho(hint, fg="yellow")
        update_run(run.id, status="queued", notes="rendered; submit manually on login node")
        raise typer.Exit()

    update_run(run.id, status="queued", slurm_job_id=job_id)
    typer.secho(
        f"Submitted SLURM job {job_id}. Watch: `just cluster-watch`\n"
        f"Results: {out_dir}\n"
        f"Logs:    tail -f slurm_logs/zo-infer-{run.id[-8:]}-{job_id}.out",
        fg="green",
    )


def judge_serve(
    model: str = typer.Option(None, "--model", "-m"),
    port: int = typer.Option(8001, help="vLLM port (SSH tunnel to this)."),
    time: str = typer.Option(None, help="SLURM wall time (default 04:00:00)."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    local: bool = typer.Option(False, "--local"),
    stage: bool = typer.Option(True, "--stage/--no-stage"),
) -> None:
    """Optional live demo: start vLLM on a GPU node."""
    ensure_cluster_env()
    hf_raw = (model or cluster_env("ZO_INFER_MODEL") or "").strip()
    if stage and hf_raw and is_hf_repo_id(hf_raw):
        dest = Path(os.path.expandvars(staged_model_path(hf_raw))).expanduser()
        if not (dest / "config.json").exists():
            stage_model(model=hf_raw, out=str(dest))
        os.environ.setdefault("ZO_INFER_MODEL_PATH", str(dest))

    try:
        model_path = to_cluster_model_path(
            os.path.expandvars(resolve_infer_model(model)), _cluster_repo()
        )
    except ValueError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1) from e

    target = _ssh_target() or "user@login01-ext.leonardo.cineca.it"
    ctx = slurm_context(
        job_name="zo-serve",
        time=time or cluster_env("ZO_SLURM_SERVE_TIME", "04:00:00"),
        model_path=model_path,
        port=port,
        ssh_target=target,
    )
    sbatch = render_template("serve.sbatch.j2", **ctx)
    serve_path = _repo() / "slurm_logs" / "judge-serve.sbatch"
    _write_sbatch(serve_path, sbatch)
    typer.echo(f"Wrote {serve_path}")

    if dry_run:
        raise typer.Exit()

    job_id = _submit_sbatch(serve_path, local=local)
    if job_id:
        typer.secho(
            f"Serve job {job_id} on port {port}.\n"
            f"Tunnel:  ssh -L {port}:<compute-node>:{port} {target}\n"
            f"Then:    ZO_MODEL_BASE_URL=http://localhost:{port}/v1 just track \"-p llm ...\"",
            fg="green",
        )
