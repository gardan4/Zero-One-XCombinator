from __future__ import annotations

import os
import subprocess
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from zo_common.paths import repo_root
from zo_common.registry import get_run, read_metrics, run_dir
from zo_eval.examples_trace import compare_example_traces, load_examples
from zo_eval.reporting import build_compare_row
from zo_eval.run_store import compare_report_from_store, default_source, filter_runs, get_store

router = APIRouter(tags=["runs"])

_CM_KEYS = ("tp", "fp", "tn", "fn")


def _runs_from_source(source: str | None):
    store = get_store(source)
    return store.list_runs()


@router.get("/runs")
def runs(
    source: str = Query(default_source(), description="local | wandb | repo"),
    only_reportable: bool = Query(True),
    include_tests: bool = Query(False),
    include_proxy: bool = Query(False),
) -> list[dict[str, Any]]:
    all_runs = _runs_from_source(source)
    filtered = filter_runs(
        all_runs,
        only_reportable=only_reportable,
        include_tests=include_tests,
        include_proxy=include_proxy,
    )
    return [r.model_dump() for r in filtered]


@router.get("/compare")
def compare(
    tag: Annotated[list[str], Query()] = [],  # noqa: B006
    source: str = Query(default_source()),
) -> list[dict[str, Any]]:
    """Runs whose tags contain ALL of the given ``tag`` filters (legacy projection)."""
    wanted = [t for t in tag if t]
    out: list[dict[str, Any]] = []
    for r in filter_runs(_runs_from_source(source), wanted_tags=wanted or None, only_reportable=False):
        if wanted and not all(w in r.tags for w in wanted):
            continue
        out.append(
            {"id": r.id, "name": r.name, "kind": r.kind, "status": r.status, "tags": r.tags, "metrics": r.metrics}
        )
    return out


@router.get("/compare/report")
def compare_report(
    tag: Annotated[list[str], Query()] = [],  # noqa: B006
    kind: str | None = None,
    role: str | None = None,
    split: str | None = None,
    family: str | None = None,
    suite: str | None = None,
    eval_set: str | None = None,
    model_ref: str | None = None,
    source: str = Query(default_source(), description="local | wandb | repo"),
    only_reportable: bool = Query(True),
    include_tests: bool = Query(False),
    include_proxy: bool = Query(False),
    refresh: bool = Query(False, description="Bypass W&B TTL cache"),
) -> dict[str, Any]:
    """Normalized comparison rows with model identity, structured metrics, artifact links."""
    wanted = [t for t in tag if t]
    return compare_report_from_store(
        source,
        force_refresh=refresh,
        wanted_tags=wanted or None,
        only_reportable=only_reportable,
        include_tests=include_tests,
        include_proxy=include_proxy,
        kind=kind,
        role=role,
        split=split,
        family=family,
        suite=suite,
        eval_set=eval_set,
        model_ref=model_ref,
    )


@router.post("/compare/refresh")
def compare_refresh(source: str = Query("wandb")) -> dict[str, Any]:
    """Force-refresh W&B-backed comparison cache."""
    from zo_eval.run_store import WandbRunStore

    store = get_store(source)
    if isinstance(store, WandbRunStore):
        store.invalidate_cache()
    report = compare_report_from_store(source, force_refresh=True, only_reportable=True)
    return {"refreshed": True, "count": report["count"], "source": source}


@router.get("/compare/examples")
def compare_examples(
    run_a: str = Query(...),
    run_b: str = Query(...),
    task: str = Query("nextstep"),
    mode: str = Query("different"),
    limit: int = Query(50, ge=1, le=200),
    source: str = Query(default_source()),
) -> dict[str, Any]:
    """Side-by-side per-example comparison from ``examples.jsonl`` artifacts."""
    store = get_store(source)
    meta_a = store.get_run(run_a)
    meta_b = store.get_run(run_b)
    pa = run_dir(run_a) / "results" / "examples.jsonl"
    pb = run_dir(run_b) / "results" / "examples.jsonl"
    if not pa.exists():
        raise HTTPException(status_code=404, detail=f"no examples.jsonl for run {run_a}")
    if not pb.exists():
        raise HTTPException(status_code=404, detail=f"no examples.jsonl for run {run_b}")
    examples = compare_example_traces(pa, pb, task=task, mode=mode, limit=limit)
    return {
        "run_a": build_compare_row(meta_a) if meta_a else {"run_id": run_a},
        "run_b": build_compare_row(meta_b) if meta_b else {"run_id": run_b},
        "task": task,
        "mode": mode,
        "examples": examples,
        "count": len(examples),
    }


@router.get("/runs/{run_id}")
def run_detail(run_id: str, source: str = Query(default_source())) -> dict[str, Any]:
    meta = get_store(source).get_run(run_id) or get_run(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="no such run")
    data = meta.model_dump()
    data["compare_row"] = build_compare_row(meta)
    return data


@router.get("/runs/{run_id}/confusion")
def run_confusion(run_id: str, source: str = Query(default_source())) -> dict[str, int]:
    meta = get_store(source).get_run(run_id) or get_run(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="no such run")
    m = meta.metrics or {}
    return {k: int(m.get(f"cm_{k}", 0) or 0) for k in _CM_KEYS}


@router.get("/runs/{run_id}/metrics")
def run_metrics(run_id: str) -> list[dict[str, Any]]:
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="no such run")
    return read_metrics(run_id)


@router.get("/runs/{run_id}/examples")
def run_examples(
    run_id: str,
    task: str = Query("nextstep"),
    outcome: str | None = Query(None, description="correct|wrong"),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    path = run_dir(run_id) / "results" / "examples.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="examples.jsonl not found for this run")
    by_task = load_examples(path)
    rows = list((by_task.get(task) or {}).values())
    if outcome == "correct":
        rows = [r for r in rows if r.get("correct") is True]
    elif outcome == "wrong":
        rows = [r for r in rows if r.get("correct") is False]
    meta = get_run(run_id)
    total = len(rows)
    return {
        "run_id": run_id,
        "task": task,
        "outcome": outcome,
        "examples": rows[:limit],
        "count": len(rows[:limit]),
        "total": total,
        "model": (build_compare_row(meta)["model"] if meta else None),
    }


@router.get("/runs/{run_id}/artifacts")
def run_artifacts(run_id: str, source: str = Query(default_source())) -> dict[str, Any]:
    meta = get_store(source).get_run(run_id) or get_run(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="no such run")
    row = build_compare_row(meta)
    return row.get("artifacts") or {}


@router.get("/runs/{run_id}/logs")
def run_logs(run_id: str, tail: int = 200) -> dict[str, list[str]]:
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="no such run")
    logs_dir = run_dir(run_id) / "logs"
    lines: list[str] = []
    if logs_dir.exists():
        for f in sorted(logs_dir.glob("*")):
            if f.is_file():
                lines += f.read_text(errors="ignore").splitlines()
    return {"lines": lines[-tail:]}


class LaunchRequest(BaseModel):
    kind: str
    config: str | None = None
    task: str | None = None
    scenario: str | None = None
    model: str = "default"
    dry_run: bool = False


@router.post("/launch")
def launch(req: LaunchRequest) -> dict[str, Any]:
    if not os.environ.get("ZO_ALLOW_LAUNCH"):
        raise HTTPException(
            status_code=403,
            detail="launch disabled; set ZO_ALLOW_LAUNCH=1 to enable (trusted/localhost only)",
        )
    if req.kind in ("sft", "grpo"):
        if not req.config:
            raise HTTPException(status_code=400, detail="config required for training")
        cmd = ["uv", "run", "zo-train", req.kind, "--config", req.config]
        if req.dry_run:
            cmd.append("--dry-run")
    elif req.kind == "eval":
        if not req.task:
            raise HTTPException(status_code=400, detail="task required for eval")
        cmd = ["uv", "run", "zo-eval", "run", "--task", req.task, "--model", req.model]
    elif req.kind == "agent":
        if not req.scenario:
            raise HTTPException(status_code=400, detail="scenario required for agent")
        cmd = ["uv", "run", "zo-agent", "run", "--scenario", req.scenario, "--model", req.model]
    else:
        raise HTTPException(status_code=400, detail=f"unknown kind {req.kind!r}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"status": "started", "pid": proc.pid, "cmd": " ".join(cmd)}
