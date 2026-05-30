from __future__ import annotations

import typer
from zo_common import ExperimentConfig, new_run, update_run
from zo_common.env import load_dotenv
from zo_common.registry import get_run, run_dir

app = typer.Typer(help="Train models (SFT / GRPO) for Zero One.", no_args_is_help=True)


@app.callback()
def _main() -> None:
    """Load <repo>/.env (WANDB_API_KEY, HF_TOKEN, ...) before any command, so W&B logging and
    HF auth work from .env without exporting by hand. Real env vars still win."""
    load_dotenv()


def _prepare(config: str, kind: str, run_id: str | None):
    cfg = ExperimentConfig.from_yaml(config)
    cfg.kind = kind  # type: ignore[assignment]
    existing = get_run(run_id) if run_id else None
    if existing is not None:
        update_run(existing.id, config=cfg.model_dump())
        run = existing
    else:
        run = new_run(cfg.name, kind, config=cfg.model_dump(), tags=list(cfg.extra.get("tags", [])))
    cfg.to_yaml(run_dir(run.id) / "config.yaml")
    return cfg, run


@app.command()
def wandb_smoke(
    run_id: str = typer.Option("wandb-smoke", "--run-id", help="Name for the W&B smoke run."),
) -> None:
    """Log one tiny W&B run from the current environment (verify connectivity, e.g. via the proxy)."""
    import os

    import wandb

    mode = os.environ.get("WANDB_MODE", "online")
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "XCombinator"),
        project=os.environ.get("WANDB_PROJECT", "XCombinator"),
        name=run_id,
        mode=mode,
        settings=wandb.Settings(init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "120"))),
    )
    run.log({"smoke/ok": 1, "smoke/step": 1})
    run.finish()
    typer.secho(f"logged W&B smoke run {run_id} in {mode} mode", fg="green")


@app.command()
def sft(
    config: str = typer.Option(..., "--config", "-c", help="Path to an experiment YAML."),
    run_id: str = typer.Option(None, "--run-id", help="Attach to an existing run (cluster jobs)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate metrics without torch."),
) -> None:
    cfg, run = _prepare(config, "sft", run_id)
    typer.secho(f"run {run.id}", fg="green")
    from zo_train.sft import run_sft

    try:
        run_sft(cfg, run.id, dry_run=dry_run)
    except Exception as e:
        update_run(run.id, status="failed", notes=str(e))
        raise


@app.command()
def grpo(
    config: str = typer.Option(..., "--config", "-c", help="Path to an experiment YAML."),
    run_id: str = typer.Option(None, "--run-id", help="Attach to an existing run (cluster jobs)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate metrics without torch."),
) -> None:
    cfg, run = _prepare(config, "grpo", run_id)
    typer.secho(f"run {run.id}", fg="green")
    from zo_train.rl import run_grpo

    try:
        run_grpo(cfg, run.id, dry_run=dry_run)
    except Exception as e:
        update_run(run.id, status="failed", notes=str(e))
        raise


if __name__ == "__main__":
    app()
