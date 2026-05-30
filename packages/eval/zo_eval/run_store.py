"""Unified run store: local registry, W&B, or repo-promoted results."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from zo_common.paths import repo_root
from zo_common.registry import RunMeta, get_run, list_runs
from zo_common.wandb_schema import SHOW_BY_DEFAULT, should_hide_by_default

from zo_eval import track_metrics as M
from zo_eval.reporting import build_compare_row

_CACHE_TTL = float(os.environ.get("ZO_WANDB_CACHE_TTL", "120"))
_wandb_cache: dict[str, tuple[float, list[RunMeta]]] = {}


def default_source() -> str:
    return os.environ.get("ZO_RESULTS_SOURCE", "local").lower()


class RunStore(ABC):
    @abstractmethod
    def list_runs(self) -> list[RunMeta]: ...

    @abstractmethod
    def get_run(self, run_id: str) -> RunMeta | None: ...


class LocalRunStore(RunStore):
    def list_runs(self) -> list[RunMeta]:
        return list_runs()

    def get_run(self, run_id: str) -> RunMeta | None:
        return get_run(run_id)


class RepoResultsStore(RunStore):
    """Read promoted pitch results from extras/results/INDEX.json."""

    def __init__(self, index_path: Path | None = None) -> None:
        self.index_path = index_path or (repo_root() / "extras" / "results" / "INDEX.json")
        self.results_root = self.index_path.parent

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data["entries"]
        if isinstance(data, dict):
            return [
                {"slug": slug, **entry}
                for slug, entry in data.items()
                if isinstance(entry, dict)
            ]
        return []

    def _entry_to_meta(self, entry: dict[str, Any]) -> RunMeta | None:
        slug = entry.get("slug") or entry.get("name")
        if not slug:
            return None
        results_dir = self.results_root / slug
        manifest_path = results_dir / "manifest.json"
        metrics: dict[str, Any] = {}
        config: dict[str, Any] = {"source": "repo-promoted", "slug": slug}
        tags = list(entry.get("tags") or [])
        tags.append("report:final")
        if "GITHUB FINAL" not in tags:
            tags.append("source:github-final")

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                config.update({k: v for k, v in manifest.items() if k != "metrics"})
                tags = list(dict.fromkeys(tags + (manifest.get("tags") or [])))
            except (json.JSONDecodeError, OSError):
                pass

        report_path = results_dir / "metrics_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                task_maps = {
                    "nextstep": M._FLAT_NEXT,
                    "completion": M._FLAT_COMPL,
                    "anomaly": M._FLAT_ANOM,
                }
                for section, flat_map in task_maps.items():
                    task = (report.get("tasks") or {}).get(section) or {}
                    for group, scores in (task.get("by_family") or {}).items():
                        suffix = "" if group == "overall" else f"_{group}"
                        for raw, flat in flat_map.items():
                            v = scores.get(raw)
                            if isinstance(v, (int, float)):
                                metrics[f"{flat}{suffix}"] = round(float(v), 4)
            except (json.JSONDecodeError, OSError):
                pass

        run_id = entry.get("run_id") or f"repo:{slug}"
        return RunMeta(
            id=run_id,
            name=entry.get("name") or slug,
            kind=entry.get("kind") or "eval",
            status="completed",
            tags=tags,
            config=config,
            metrics=metrics,
            notes=entry.get("notes") or "",
        )

    def list_runs(self) -> list[RunMeta]:
        out: list[RunMeta] = []
        for entry in self._load_index():
            meta = self._entry_to_meta(entry)
            if meta:
                out.append(meta)
        return out

    def get_run(self, run_id: str) -> RunMeta | None:
        for meta in self.list_runs():
            if meta.id == run_id or meta.config.get("slug") == run_id.replace("repo:", ""):
                return meta
        return None


class WandbRunStore(RunStore):
    """Server-side W&B query with in-memory TTL cache."""

    def __init__(self) -> None:
        self._entity = os.environ.get("WANDB_ENTITY")
        self._project = os.environ.get("WANDB_PROJECT", "XCombinator")

    def _api(self):
        try:
            import wandb

            return wandb.Api()
        except ImportError as exc:
            raise RuntimeError("wandb not installed; run `uv sync --extra gpu`") from exc

    def _wandb_run_to_meta(self, wr) -> RunMeta:  # noqa: ANN001
        tags = list(wr.tags or [])
        config = dict(wr.config or {})
        summary = dict(wr.summary or {})
        metrics: dict[str, Any] = {}
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                key = k.replace("/", "_") if k.startswith(("eval/", "proxy/", "train/")) else k
                if k.startswith("eval/"):
                    key = k.split("/", 1)[1]
                metrics[key] = float(v)
            elif k in ("top1", "em", "anomaly_f1"):
                metrics[k] = float(v)

        job_type = config.get("job_type") or wr.job_type or "eval"
        return RunMeta(
            id=wr.id or wr.name,
            name=wr.name or wr.id,
            kind=str(job_type),
            status=wr.state or "finished",
            created_at=str(getattr(wr, "created_at", "") or ""),
            tags=tags,
            config=config,
            metrics=metrics,
            notes=str(config.get("notes") or ""),
            git_sha=config.get("git_sha"),
            git_branch=config.get("git_branch"),
            slurm_job_id=config.get("slurm_job_id"),
        )

    def list_runs(self, *, force_refresh: bool = False) -> list[RunMeta]:
        cache_key = f"{self._entity}/{self._project}"
        now = time.time()
        if not force_refresh and cache_key in _wandb_cache:
            ts, cached = _wandb_cache[cache_key]
            if now - ts < _CACHE_TTL:
                return cached

        if not os.environ.get("WANDB_API_KEY"):
            return []

        api = self._api()
        path = f"{self._entity}/{self._project}" if self._entity else self._project
        runs = api.runs(path, order="-created_at")
        out = [self._wandb_run_to_meta(wr) for wr in runs]
        _wandb_cache[cache_key] = (now, out)
        return out

    def get_run(self, run_id: str) -> RunMeta | None:
        for meta in self.list_runs():
            if meta.id == run_id:
                return meta
        if not os.environ.get("WANDB_API_KEY"):
            return None
        try:
            api = self._api()
            path = f"{self._entity}/{self._project}/{run_id}" if self._entity else f"{self._project}/{run_id}"
            wr = api.run(path)
            return self._wandb_run_to_meta(wr)
        except Exception:
            return None

    def invalidate_cache(self) -> None:
        cache_key = f"{self._entity}/{self._project}"
        _wandb_cache.pop(cache_key, None)


def get_store(source: str | None = None) -> RunStore:
    src = (source or default_source()).lower()
    if src == "wandb":
        return WandbRunStore()
    if src == "repo":
        return RepoResultsStore()
    return LocalRunStore()


def filter_runs(
    runs: list[RunMeta],
    *,
    wanted_tags: list[str] | None = None,
    only_reportable: bool = True,
    include_tests: bool = False,
    include_proxy: bool = False,
    kind: str | None = None,
    role: str | None = None,
    split: str | None = None,
    family: str | None = None,
    suite: str | None = None,
    eval_set: str | None = None,
    model_ref: str | None = None,
) -> list[RunMeta]:
    out: list[RunMeta] = []
    for r in runs:
        if wanted_tags and not all(t in r.tags for t in wanted_tags):
            continue
        ts = set(r.tags)
        if not include_tests and ts & {"test", "smoke", "debug"} and not (ts & SHOW_BY_DEFAULT):
            continue
        if not include_proxy and "proxy-only" in ts and not (ts & SHOW_BY_DEFAULT):
            continue
        if only_reportable and not (ts & SHOW_BY_DEFAULT or "report:final" in ts):
            continue
        if only_reportable and should_hide_by_default(r.tags):
            continue
        if kind and r.kind != kind:
            continue
        row = build_compare_row(r)
        pt = row.get("parsed_tags") or {}
        m = row.get("model") or {}
        d = row.get("dataset") or {}
        if role and pt.get("role") != role:
            continue
        if split and pt.get("split") != split:
            continue
        if family and pt.get("family") != family:
            continue
        if suite and pt.get("suite") != suite:
            continue
        if eval_set and d.get("eval_set") != eval_set:
            continue
        if model_ref and m.get("model_ref") != model_ref:
            continue
        out.append(r)
    return out


def compare_report_from_store(
    source: str | None = None,
    *,
    force_refresh: bool = False,
    **filters,
) -> dict[str, Any]:
    from zo_eval.reporting import METRIC_SPECS, metric_deltas

    store = get_store(source)
    if isinstance(store, WandbRunStore) and force_refresh:
        store.invalidate_cache()
    runs = store.list_runs() if not isinstance(store, WandbRunStore) else store.list_runs(force_refresh=force_refresh)
    filtered = filter_runs(runs, **filters)
    rows = [build_compare_row(r) for r in filtered]
    src = (source or default_source()).lower()
    for row, meta in zip(rows, filtered, strict=False):
        row["data_source"] = {"local": "LOCAL CACHE", "wandb": "W&B EVAL", "repo": "GITHUB FINAL"}.get(src, src.upper())
        if "proxy-only" in row.get("tags", []):
            row["data_source"] = "W&B PROXY" if src == "wandb" else row["data_source"]
        if src == "repo":
            slug = meta.config.get("slug")
            if isinstance(slug, str):
                results_dir = repo_root() / "extras" / "results" / slug
                row["artifacts"] = _artifact_links(results_dir)
                report = results_dir / "metrics_report.json"
                if report.exists():
                    try:
                        row["metrics_structured"] = json.loads(report.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        pass
    return {
        "source": source or default_source(),
        "metric_specs": METRIC_SPECS,
        "rows": rows,
        "deltas_vs_baseline": metric_deltas(rows) if rows else {},
        "count": len(rows),
    }


def _artifact_links(results_dir: Path) -> dict[str, str | None]:
    def _p(name: str) -> str | None:
        p = results_dir / name
        return str(p) if p.exists() else None

    return {
        "results_dir": str(results_dir),
        "metrics_report_json": _p("metrics_report.json"),
        "metrics_report_md": _p("metrics_report.md"),
        "manifest_json": _p("manifest.json"),
        "examples_jsonl": _p("examples.jsonl"),
        "proxy_report_json": _p("proxy_report.json"),
        "official_scores_txt": _p("official_scores.txt"),
        "nextstep_csv": _p("nextstep.csv"),
        "completion_csv": _p("completion.csv"),
        "anomaly_csv": _p("anomaly.csv"),
    }
