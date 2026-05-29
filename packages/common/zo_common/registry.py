from __future__ import annotations

import json
import secrets
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from zo_common.paths import experiments_dir


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in text.strip())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")[:40] or "run"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


class RunMeta(BaseModel):
    id: str
    name: str
    kind: str
    status: str = "created"  # created|queued|running|completed|failed|cancelled
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    git_branch: str | None = None
    git_sha: str | None = None
    cluster: str | None = None
    slurm_job_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)  # latest summary scalars


def run_dir(run_id: str) -> Path:
    return experiments_dir() / run_id


def _meta_path(run_id: str) -> Path:
    return run_dir(run_id) / "meta.json"


def save_meta(meta: RunMeta) -> RunMeta:
    d = run_dir(meta.id)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    meta.updated_at = _now()
    _meta_path(meta.id).write_text(meta.model_dump_json(indent=2))
    return meta


def new_run(
    name: str,
    kind: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    cluster: str | None = None,
) -> RunMeta:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    # Random suffix: timestamps are only second-granular, so two runs started in the same
    # second (parallel worktrees / a shared ZO_EXPERIMENTS_DIR on cluster scratch) would
    # otherwise collide and clobber each other's meta.json.
    meta = RunMeta(
        id=f"{ts}_{kind}_{_slug(name)}_{secrets.token_hex(3)}",
        name=name,
        kind=kind,
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_sha=_git("rev-parse", "--short", "HEAD"),
        tags=tags or [],
        cluster=cluster,
        config=config or {},
    )
    return save_meta(meta)


def get_run(run_id: str) -> RunMeta | None:
    p = _meta_path(run_id)
    return RunMeta.model_validate_json(p.read_text()) if p.exists() else None


def list_runs() -> list[RunMeta]:
    runs: list[RunMeta] = []
    for child in experiments_dir().iterdir():
        meta_file = child / "meta.json"
        if child.is_dir() and meta_file.exists():
            try:
                runs.append(RunMeta.model_validate_json(meta_file.read_text()))
            except Exception:
                continue
    return sorted(runs, key=lambda r: r.created_at, reverse=True)


def update_run(run_id: str, **fields: Any) -> RunMeta | None:
    meta = get_run(run_id)
    if meta is None:
        return None
    data = meta.model_dump()
    for k, v in fields.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k] = {**data[k], **v}
        else:
            data[k] = v
    return save_meta(RunMeta(**data))


def append_metric(run_id: str, step: int | None = None, **metrics: Any) -> None:
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {"t": _now()}
    if step is not None:
        row["step"] = step
    row.update(metrics)
    with (d / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    update_run(run_id, metrics=metrics)  # keep a quick-look summary on the meta


def read_metrics(run_id: str) -> list[dict[str, Any]]:
    p = run_dir(run_id) / "metrics.jsonl"
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
