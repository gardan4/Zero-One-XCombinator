from __future__ import annotations

import random
import time

from zo_common import ExperimentConfig, append_metric, update_run
from zo_common.wandb_runs import finish_run, init_run, log_metrics, wandb_enabled


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
    if wandb_enabled():
        init_run(run_id, "train", tags=["smoke", "dry-run"], config={"dry_run": True})
    loss = 2.5
    for step in range(1, total_steps + 1):
        loss = max(0.2, loss * 0.92 + random.uniform(-0.02, 0.02))
        scalars = {"loss": round(loss, 4), "learning_rate": 2e-5, "reward": round(1 - loss / 2.5, 4)}
        append_metric(run_id, step=step, **scalars)
        log_metrics(scalars, step=step, prefix="train")
        if sleep:
            time.sleep(sleep)
    if wandb_enabled():
        finish_run(exit_code=0)
    update_run(run_id, status="completed")
