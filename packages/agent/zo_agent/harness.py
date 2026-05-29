from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from zo_common import append_metric, new_run, update_run
from zo_common.registry import get_run

from zo_agent.rollout import run_episode


class Scenario(BaseModel):
    name: str
    system: str | None = None
    task: str
    success_type: str = "contains"  # contains | exact | numeric
    success_value: str = ""
    repeat: int = 1


class ScenarioSuite(BaseModel):
    name: str
    model: str | None = None
    scenarios: list[Scenario] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ScenarioSuite:
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))


def _num(s: str | None) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    return float(m.group()) if m else None


def _judge(success_type: str, final: str, value: str) -> float:
    f, v = (final or "").strip().lower(), (value or "").strip().lower()
    if success_type == "contains":
        return float(v in f)
    if success_type == "exact":
        return float(f == v)
    if success_type == "numeric":
        a, b = _num(final), _num(value)
        return float(a is not None and b is not None and abs(a - b) < 1e-6)
    return 0.0


def run_suite(
    suite: ScenarioSuite,
    model: str,
    base_url: str | None = None,
    run_id: str | None = None,
    max_steps: int = 6,
) -> dict[str, Any]:
    run = get_run(run_id) if run_id else None
    if run is None:
        run = new_run(
            f"{suite.name}@{model}",
            "agent",
            config={"suite": suite.name, "model": model},
            tags=["agent"],
        )
    update_run(run.id, status="running")

    success, total, steps_sum, tools_sum = 0.0, 0, 0, 0
    for sc in suite.scenarios:
        for _ in range(max(sc.repeat, 1)):
            ep = run_episode(sc.system, sc.task, model, base_url=base_url, max_steps=max_steps)
            s = _judge(sc.success_type, ep["final"], sc.success_value)
            success += s
            total += 1
            steps_sum += ep["steps"]
            tools_sum += ep["tool_calls"]
            append_metric(
                run.id,
                step=total,
                scenario=sc.name,
                success=s,
                steps=ep["steps"],
                tool_calls=ep["tool_calls"],
            )

    metrics = {
        "success_rate": round(success / max(total, 1), 4),
        "n": total,
        "avg_steps": round(steps_sum / max(total, 1), 2),
        "avg_tool_calls": round(tools_sum / max(total, 1), 2),
        "model": model,
    }
    update_run(run.id, status="completed", metrics=metrics)
    return {**metrics, "run_id": run.id}
