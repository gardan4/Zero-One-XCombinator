"""Shared W&B logging helpers — no-op when W&B is unavailable."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from zo_common.wandb_schema import prefixed_metrics

_active_run = None
_run_id: str | None = None


def wandb_enabled() -> bool:
    mode = os.environ.get("WANDB_MODE", "").replace("\r", "").strip().lower()
    if mode == "disabled":
        return False
    if not os.environ.get("WANDB_API_KEY"):
        return False
    return True


def _import_wandb():
    try:
        import wandb

        return wandb
    except ImportError:
        return None


def init_run(
    run_id: str,
    job_type: str,
    *,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    group: str | None = None,
    name: str | None = None,
    resume: str = "allow",
) -> Any:
    """Start or resume a W&B run keyed by registry run_id. Returns run or None."""
    global _active_run, _run_id
    if not wandb_enabled():
        return None
    wandb = _import_wandb()
    if wandb is None:
        return None
    if _active_run is not None and _run_id == run_id:
        return _active_run
    cfg = dict(config or {})
    cfg["job_type"] = job_type
    tag_list = list(tags or [])
    _active_run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY") or None,
        project=os.environ.get("WANDB_PROJECT", "XCombinator"),
        id=run_id,
        name=name or run_id,
        tags=tag_list,
        group=group or os.environ.get("WANDB_RUN_GROUP"),
        config=cfg,
        resume=resume,
        reinit=True,
    )
    _run_id = run_id
    return _active_run


def log_metrics(
    metrics: dict[str, Any],
    *,
    step: int | None = None,
    prefix: str = "train",
    use_eval_map: bool = False,
) -> None:
    if not _active_run and not wandb_enabled():
        return
    wandb = _import_wandb()
    if wandb is None or wandb.run is None:
        return
    if use_eval_map:
        payload = prefixed_metrics(metrics, prefix)
    else:
        payload = prefixed_metrics(metrics, prefix) if prefix != "train" else {
            f"train/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))
        }
    if payload:
        wandb.log(payload, step=step)


def log_config_update(updates: dict[str, Any]) -> None:
    wandb = _import_wandb()
    if wandb is None or wandb.run is None:
        return
    wandb.config.update(updates, allow_val_change=True)


def log_artifact(
    path_or_dir: str | Path,
    *,
    name: str,
    artifact_type: str,
    aliases: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str = "",
) -> str | None:
    """Upload a file or directory as a W&B artifact. Returns artifact version name or None."""
    if not wandb_enabled():
        return None
    wandb = _import_wandb()
    if wandb is None or wandb.run is None:
        return None
    p = Path(path_or_dir)
    if not p.exists():
        return None
    art = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {}, description=description)
    if p.is_dir():
        art.add_dir(str(p))
    else:
        art.add_file(str(p))
    wandb.log_artifact(art, aliases=aliases or ["latest"])
    return art.name


def log_hf_model(
    hub_model_id: str,
    *,
    revision: str | None = None,
    run_url: str | None = None,
) -> None:
    """Record HF model identity on the active W&B run."""
    if not wandb_enabled():
        return
    wandb = _import_wandb()
    if wandb is None or wandb.run is None:
        return
    payload = {
        "hf/repo": hub_model_id,
        "hf/url": f"https://huggingface.co/{hub_model_id}",
    }
    if revision:
        payload["hf/revision"] = revision
    if run_url:
        payload["wandb/run_url"] = run_url
    wandb.log(payload)
    wandb.config.update({"hub_model_id": hub_model_id, "hf_revision": revision}, allow_val_change=True)


def log_hf_to_training_run(
    run_id: str,
    hub_model_id: str,
    *,
    revision: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """After HF upload, log model identity back to the training W&B run."""
    if not wandb_enabled():
        return
    init_run(run_id, "train", tags=tags or [], config=config or {}, resume="allow")
    log_hf_model(hub_model_id, revision=revision)
    finish_run()


def finish_run(exit_code: int = 0) -> None:
    global _active_run, _run_id
    wandb = _import_wandb()
    if wandb is not None and wandb.run is not None:
        wandb.finish(exit_code=exit_code)
    _active_run = None
    _run_id = None


def pytest_auto_tags() -> list[str]:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return ["test"]
    return []
