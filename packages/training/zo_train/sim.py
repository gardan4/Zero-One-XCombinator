from __future__ import annotations

import os
import random
import time

from zo_common import ExperimentConfig, append_metric, run_dir, update_run


def _maybe_wandb(run_id: str):
    """Init W&B iff WANDB_API_KEY is set and `wandb` is importable — else None (no-op).

    Lets a `--dry-run` push a simulated curve to W&B so you can VERIFY the integration with no
    GPU. Real (non-dry) runs log to W&B via the trainer's `report_to=["wandb"]` instead.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY") or None,
        project=os.environ.get("WANDB_PROJECT", "XCombinator"),
        name=run_id,
        id=run_id,
        resume="allow",
        config={"dry_run": True},
    )
    return wandb


def simulate_training(
    run_id: str,
    cfg: ExperimentConfig | None = None,
    total_steps: int = 20,
    sleep: float = 0.0,
) -> None:
    """Write a believable metric curve WITHOUT torch.

    Lets you test the whole pipeline (registry -> backend -> dashboard, and W&B if a key is set)
    on a laptop before burning cluster time. `just train <cfg>` calls this when --dry-run is set.
    """
    if cfg is not None:
        from zo_train.preflight import validate_experiment

        validate_experiment(cfg, cluster=False)
        max_steps = int(cfg.extra.get("max_steps", -1))
        if max_steps > 0:
            total_steps = min(total_steps, max_steps)
    update_run(run_id, status="running")
    wb = _maybe_wandb(run_id)
    loss = 2.5
    for step in range(1, total_steps + 1):
        loss = max(0.2, loss * 0.92 + random.uniform(-0.02, 0.02))
        scalars = {"loss": round(loss, 4), "learning_rate": 2e-5, "reward": round(1 - loss / 2.5, 4)}
        append_metric(run_id, step=step, **scalars)
        if wb:
            wb.log(scalars, step=step)
        if sleep:
            time.sleep(sleep)
    if wb:
        wb.finish()
    update_run(run_id, status="completed")
