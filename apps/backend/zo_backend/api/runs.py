from __future__ import annotations

import os
import subprocess
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from zo_common.paths import repo_root
from zo_common.registry import get_run, list_runs, read_metrics, run_dir

router = APIRouter(tags=["runs"])

# Confusion-matrix scalars live flat on a run's metrics summary (track.py writes cm_tp/fp/tn/fn).
_CM_KEYS = ("tp", "fp", "tn", "fn")


@router.get("/runs")
def runs() -> list[dict[str, Any]]:
    return [r.model_dump() for r in list_runs()]


@router.get("/compare")
def compare(tag: Annotated[list[str], Query()] = []) -> list[dict[str, Any]]:  # noqa: B006 — FastAPI reads the list from the query string, never mutates this default
    """Runs whose tags contain ALL of the given ``tag`` filters (repeatable query param).

    Powers the comparison dashboard (Stream 4): e.g. ``/api/compare?tag=eval:anomaly&tag=stream:4``
    returns every anomaly run from stream 4, ID and OOD alike (split:id|ood is itself a tag the
    frontend groups by). No filters → all runs. Read-only; a trimmed projection of ``RunMeta``.
    """
    wanted = [t for t in tag if t]
    out: list[dict[str, Any]] = []
    for r in list_runs():
        if all(w in r.tags for w in wanted):
            out.append(
                {"id": r.id, "name": r.name, "kind": r.kind, "status": r.status, "tags": r.tags, "metrics": r.metrics}
            )
    return out


@router.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    meta = get_run(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="no such run")
    return meta.model_dump()


@router.get("/runs/{run_id}/confusion")
def run_confusion(run_id: str) -> dict[str, int]:
    """The anomaly confusion matrix ``{tp,fp,tn,fn}`` from the run's flat ``cm_*`` metrics.

    Missing keys default to 0, so a run without anomaly metrics returns an all-zero matrix rather
    than erroring (the dashboard renders an empty grid). 404 only if the run itself is unknown.
    """
    meta = get_run(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="no such run")
    m = meta.metrics or {}
    return {k: int(m.get(f"cm_{k}", 0) or 0) for k in _CM_KEYS}


@router.get("/runs/{run_id}/metrics")
def run_metrics(run_id: str) -> list[dict[str, Any]]:
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="no such run")
    return read_metrics(run_id)


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
    kind: str  # sft | grpo | eval | agent
    config: str | None = None
    task: str | None = None
    scenario: str | None = None
    model: str = "default"
    dry_run: bool = False


@router.post("/launch")
def launch(req: LaunchRequest) -> dict[str, Any]:
    """Best-effort: spawn a run in the background. Poll /api/runs to see it appear.

    Disabled by default: this runs arbitrary `uv run` subprocesses, so exposing it on a
    non-localhost interface (ZO_API_HOST=0.0.0.0) would be remote code execution. Opt in with
    ZO_ALLOW_LAUNCH=1 only on a trusted machine you control.
    """
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
