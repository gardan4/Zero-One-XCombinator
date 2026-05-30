from __future__ import annotations

from typing import Any

from zo_common import append_metric, new_run, update_run
from zo_common.llm import chat, content
from zo_common.registry import get_run
from zo_common.wandb_runs import finish_run, init_run, log_metrics, pytest_auto_tags, wandb_enabled

from zo_eval.tasks import TaskSpec, score


def run_eval(
    task: TaskSpec,
    model: str,
    base_url: str | None = None,
    run_id: str | None = None,
    limit: int | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    items = task.load_items()
    if limit:
        items = items[:limit]

    run = get_run(run_id) if run_id else None
    if run is None:
        run = new_run(
            f"{task.name}@{model}",
            "eval",
            config={"task": task.name, "model": model, "metric": task.metric},
            tags=["eval"],
        )
    update_run(run.id, status="running")
    eval_tags = (run.tags or []) + pytest_auto_tags() + ["eval"]
    if wandb_enabled():
        init_run(run.id, "eval", tags=eval_tags, config={"task": task.name, "model": model})

    total, n = 0.0, 0
    for i, item in enumerate(items):
        messages = []
        if task.system:
            messages.append({"role": "system", "content": task.system})
        messages.append({"role": "user", "content": item.prompt})
        try:
            pred = content(chat(messages, model=model, base_url=base_url, temperature=temperature))
        except Exception:
            pred = ""
        s = score(task.metric, pred, item.answer, task.pattern)
        total += s
        n += 1
        row = {"item_score": s, "running_acc": round(total / n, 4)}
        append_metric(run.id, step=i, **row)
        log_metrics(row, step=i, prefix="eval")

    accuracy = round(total / max(n, 1), 4)
    if wandb_enabled():
        log_metrics({"accuracy": accuracy, "n": n}, step=n, prefix="eval")
        finish_run(exit_code=0)

    accuracy = round(total / max(n, 1), 4)
    update_run(
        run.id,
        status="completed",
        metrics={"accuracy": accuracy, "n": n, "task": task.name, "model": model},
    )
    return {"accuracy": accuracy, "n": n, "run_id": run.id}
