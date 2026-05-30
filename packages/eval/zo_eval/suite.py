"""YAML-driven eval matrix — run tagged predict jobs for baseline vs model comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from zo_eval.reporting import METRIC_SPECS, build_compare_row, format_suite_report_markdown, metric_deltas
from zo_eval.track import run_track


def load_suite(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "runs" not in data:
        raise ValueError(f"suite YAML must have a top-level 'runs' list: {path}")
    return data


def _merge_tags(base: list[str] | None, extra: list[str] | None) -> list[str]:
    out: list[str] = []
    for t in (base or []) + (extra or []):
        t = str(t).strip()
        if t and t not in out:
            out.append(t)
    return out


def run_suite(
    suite_path: str | Path,
    *,
    build_predictor,
    valid_csv: str | None = None,
    anomaly_csv: str | Path | None = None,
    gold: str | None = None,
    eval_set: str | None = None,
    suite_tags: list[str] | None = None,
    model_override: str | None = None,
    model_ref_override: str | None = None,
    stop_on_error: bool = False,
) -> dict:
    """Execute every run spec in a suite file; return summary dict."""
    spec = load_suite(suite_path)
    defaults: dict[str, Any] = spec.get("defaults") or {}
    valid_csv = valid_csv or defaults.get("valid_csv")
    anomaly_csv = anomaly_csv or defaults.get("anomaly_csv")
    gold = gold or defaults.get("gold")
    eval_set = eval_set or defaults.get("eval_set", "local")
    global_tags = _merge_tags(defaults.get("tags"), suite_tags)

    results: list[dict] = []
    for i, run_spec in enumerate(spec["runs"]):
        name = run_spec.get("name") or f"run_{i}"
        predictor_kind = run_spec["predictor"]
        version = run_spec.get("version") or name
        tags = _merge_tags(global_tags, run_spec.get("tags"))
        tags.append(f"suite:{Path(suite_path).stem}")
        tags.append(f"suite-run:{name}")

        try:
            run_model = model_override or run_spec.get("model") or defaults.get("model", "default")
            mref = model_ref_override or run_spec.get("model_ref") or defaults.get("model_ref")
            if run_model != "default" and not mref and "/" in run_model:
                mref = run_model
            pred = build_predictor(
                predictor_kind,
                run_spec.get("train_families"),
                int(run_spec.get("order", defaults.get("order", 3))),
                run_model,
                run_spec.get("base_url", defaults.get("base_url")),
            )
            res = run_track(
                pred,
                valid_csv=run_spec.get("valid_csv") or valid_csv,
                anomaly_csv=run_spec.get("anomaly_csv") or anomaly_csv,
                gold=run_spec.get("gold") or gold,
                tasks=tuple(run_spec.get("tasks", defaults.get("tasks", ["nextstep", "completion", "anomaly"]))),
                tags=tags,
                version=version,
                model_ref=mref,
                eval_set=run_spec.get("eval_set") or eval_set,
                self_check=bool(run_spec.get("self_check", defaults.get("self_check", False))),
                run_proxy=bool(run_spec.get("run_proxy", defaults.get("run_proxy", True))),
                promote=run_spec.get("promote") or defaults.get("promote"),
            )
            results.append({"name": name, "status": "ok", **res})
        except Exception as e:
            results.append({"name": name, "status": "error", "error": str(e)})
            if stop_on_error:
                break

    from zo_common.registry import get_run

    rows: list[dict] = []
    for r in results:
        if r.get("status") != "ok" or not r.get("run_id"):
            continue
        meta = get_run(r["run_id"])
        if meta:
            rows.append(build_compare_row(meta))

    summary = {
        "suite": str(suite_path),
        "eval_set": eval_set,
        "runs": results,
        "rows": rows,
        "metric_specs": METRIC_SPECS,
        "deltas_vs_baseline": metric_deltas(rows) if rows else {},
    }
    out_root = Path(defaults.get("summary_dir") or "extras/results") / f"suite_{Path(suite_path).stem}"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    (out_root / "suite_report.md").write_text(format_suite_report_markdown(summary), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(out_root / "suite_report.md")
    return summary
