from __future__ import annotations

import random
import time

from zo_common import append_metric, update_run


def simulate_training(run_id: str, total_steps: int = 20, sleep: float = 0.0) -> None:
    """Write a believable metric curve WITHOUT torch.

    Lets you test the whole pipeline (registry -> backend -> dashboard) on a laptop
    before burning cluster time. `just train <cfg>` calls this when --dry-run is set.
    """
    update_run(run_id, status="running")
    loss = 2.5
    for step in range(1, total_steps + 1):
        loss = max(0.2, loss * 0.92 + random.uniform(-0.02, 0.02))
        append_metric(
            run_id,
            step=step,
            loss=round(loss, 4),
            learning_rate=2e-5,
            reward=round(1 - loss / 2.5, 4),
        )
        if sleep:
            time.sleep(sleep)
    update_run(run_id, status="completed")
