"""Shared building blocks for Zero One Philyr: experiment config + run registry.

Everything that produces results (training, eval, agent runs) writes through the
registry here, and the backend reads from it. That shared contract is what lets us
launch and inspect many runs in parallel.
"""

from zo_common.config import ExperimentConfig
from zo_common.registry import (
    RunMeta,
    append_metric,
    get_run,
    list_runs,
    new_run,
    read_metrics,
    run_dir,
    save_meta,
    update_run,
)

__all__ = [
    "ExperimentConfig",
    "RunMeta",
    "append_metric",
    "get_run",
    "list_runs",
    "new_run",
    "read_metrics",
    "run_dir",
    "save_meta",
    "update_run",
]
