"""Normalized reporting schema for eval runs — model identity, metrics, artifacts, comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zo_common.registry import RunMeta, run_dir

# Headline metrics for dashboards (flat registry keys + structured report keys).
METRIC_SPECS: list[dict[str, Any]] = [
    # next-step
    {"key": "top1", "label": "Top-1", "task": "nextstep", "direction": "higher", "format": "pct"},
    {"key": "top3", "label": "Top-3", "task": "nextstep", "direction": "higher", "format": "pct"},
    {"key": "top5", "label": "Top-5", "task": "nextstep", "direction": "higher", "format": "pct"},
    {"key": "mrr", "label": "MRR", "task": "nextstep", "direction": "higher", "format": "dec"},
    # completion
    {"key": "em", "label": "Exact match", "task": "completion", "direction": "higher", "format": "pct"},
    {"key": "ned", "label": "NED", "task": "completion", "direction": "lower", "format": "dec"},
    {"key": "token_acc", "label": "Token acc", "task": "completion", "direction": "higher", "format": "pct"},
    {"key": "block_acc", "label": "Block acc", "task": "completion", "direction": "higher", "format": "pct"},
    # anomaly
    {"key": "anomaly_acc", "label": "Binary acc", "task": "anomaly", "direction": "higher", "format": "pct"},
    {"key": "anomaly_p", "label": "Precision", "task": "anomaly", "direction": "higher", "format": "pct"},
    {"key": "anomaly_r", "label": "Recall", "task": "anomaly", "direction": "higher", "format": "pct"},
    {"key": "anomaly_f1", "label": "F1", "task": "anomaly", "direction": "higher", "format": "pct"},
    {"key": "anomaly_auc", "label": "ROC-AUC", "task": "anomaly", "direction": "higher", "format": "dec"},
    {"key": "rule_attr_acc", "label": "Rule attr", "task": "anomaly", "direction": "higher", "format": "pct"},
]

HEADLINE_KEYS = [s["key"] for s in METRIC_SPECS]


def tag_value(tags: list[str], prefix: str) -> str | None:
    for t in tags:
        if t.startswith(f"{prefix}:"):
            return t.split(":", 1)[1]
    return None


def parse_tags(tags: list[str]) -> dict[str, str | None]:
    """Extract common prefixed tags into a stable dict."""
    prefixes = (
        "version",
        "predictor",
        "model-ref",
        "eval-set",
        "train-run",
        "split",
        "family",
        "role",
        "suite",
        "suite-run",
    )
    out: dict[str, str | None] = {p.replace("-", "_"): None for p in prefixes}
    for p in prefixes:
        out[p.replace("-", "_")] = tag_value(tags, p)
    out["raw_tags"] = ",".join(tags)
    return out


@dataclass
class ModelIdentity:
    display: str
    model_ref: str | None
    predictor: str | None
    version: str | None
    train_run_id: str | None
    role: str | None
    served_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetContext:
    eval_set: str | None
    split: str | None
    family: str | None
    suite: str | None
    suite_run: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactLinks:
    results_dir: str
    metrics_report_json: str | None = None
    metrics_report_md: str | None = None
    manifest_json: str | None = None
    examples_jsonl: str | None = None
    proxy_report_json: str | None = None
    official_scores_txt: str | None = None
    nextstep_csv: str | None = None
    completion_csv: str | None = None
    anomaly_csv: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_ref_from_tag(tags: list[str]) -> str | None:
    raw = tag_value(tags, "model-ref")
    if raw:
        return raw.replace("--", "/")
    return None


def build_model_identity(
    *,
    name: str,
    config: dict[str, Any] | None,
    tags: list[str],
    predictor: str | None = None,
    version: str | None = None,
    model_ref: str | None = None,
    train_run_id: str | None = None,
) -> ModelIdentity:
    cfg = config or {}
    parsed = parse_tags(tags)
    mref = model_ref or cfg.get("model_ref") or model_ref_from_tag(tags)
    pred = predictor or cfg.get("predictor") or parsed.get("predictor")
    ver = version or cfg.get("version") or parsed.get("version")
    tr = train_run_id or cfg.get("train_run_id") or parsed.get("train_run")
    role = parsed.get("role")
    served = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    display = mref or served or pred or name
    if ver and display != ver:
        display = f"{display} ({ver})"
    return ModelIdentity(
        display=display,
        model_ref=mref,
        predictor=pred,
        version=ver,
        train_run_id=tr,
        role=role,
        served_model=served if served and served != "default" else None,
    )


def build_dataset_context(
    *,
    tags: list[str],
    eval_set: str | None = None,
    config: dict[str, Any] | None = None,
) -> DatasetContext:
    parsed = parse_tags(tags)
    cfg = config or {}
    return DatasetContext(
        eval_set=eval_set or cfg.get("eval_set") or parsed.get("eval_set"),
        split=parsed.get("split"),
        family=parsed.get("family"),
        suite=parsed.get("suite"),
        suite_run=parsed.get("suite_run"),
    )


def artifact_links_for_run(run_id: str) -> ArtifactLinks:
    rd = run_dir(run_id) / "results"
    def _p(name: str) -> str | None:
        p = rd / name
        return str(p) if p.exists() else None

    return ArtifactLinks(
        results_dir=str(rd),
        metrics_report_json=_p("metrics_report.json"),
        metrics_report_md=_p("metrics_report.md"),
        manifest_json=_p("manifest.json"),
        examples_jsonl=_p("examples.jsonl"),
        proxy_report_json=_p("proxy_report.json"),
        official_scores_txt=_p("official_scores.txt"),
        nextstep_csv=_p("nextstep.csv"),
        completion_csv=_p("completion.csv"),
        anomaly_csv=_p("anomaly.csv"),
    )


def load_structured_metrics_report(run_id: str) -> dict[str, Any] | None:
    p = run_dir(run_id) / "results" / "metrics_report.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_compare_row(meta: RunMeta) -> dict[str, Any]:
    """Normalized comparison row for API / suite summaries."""
    model = build_model_identity(
        name=meta.name,
        config=meta.config,
        tags=meta.tags,
    )
    dataset = build_dataset_context(tags=meta.tags, config=meta.config)
    artifacts = artifact_links_for_run(meta.id)
    structured = load_structured_metrics_report(meta.id)
    flat = {k: v for k, v in (meta.metrics or {}).items() if isinstance(v, (int, float))}
    return {
        "run_id": meta.id,
        "run_name": meta.name,
        "kind": meta.kind,
        "status": meta.status,
        "created_at": meta.created_at,
        "notes": meta.notes or "",
        "tags": meta.tags,
        "parsed_tags": parse_tags(meta.tags),
        "model": model.to_dict(),
        "dataset": dataset.to_dict(),
        "metrics_flat": flat,
        "metrics_structured": structured,
        "artifacts": artifacts.to_dict(),
        "git_branch": meta.git_branch,
        "git_sha": meta.git_sha,
    }


def metric_deltas(
    rows: list[dict[str, Any]],
    *,
    baseline_role: str = "baseline",
    keys: list[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Compute mean deltas vs runs tagged role:baseline (or predictor ngram/freq)."""
    keys = keys or HEADLINE_KEYS
    baselines = [r for r in rows if (r.get("model") or {}).get("role") == baseline_role]
    if not baselines:
        baselines = [
            r
            for r in rows
            if (r.get("model") or {}).get("predictor") in ("ngram", "freq", "oracle")
        ]
    if not baselines:
        return {}
    base_means: dict[str, float] = {}
    for k in keys:
        vals = [r["metrics_flat"].get(k) for r in baselines if isinstance(r["metrics_flat"].get(k), (int, float))]
        if vals:
            base_means[k] = sum(vals) / len(vals)
    out: dict[str, dict[str, float | None]] = {}
    for r in rows:
        rid = r["run_id"]
        out[rid] = {}
        for k in keys:
            v = r["metrics_flat"].get(k)
            b = base_means.get(k)
            if isinstance(v, (int, float)) and b is not None:
                out[rid][k] = round(float(v) - b, 4)
            else:
                out[rid][k] = None
    return out


def format_suite_report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Eval suite — {Path(summary.get('suite', '?')).stem}",
        "",
        f"- **Eval set:** `{summary.get('eval_set', '?')}`",
        "",
        "| Run | Model | Role | Split | Top-1 | F1 | AUC |",
        "|-----|-------|------|-------|-------|----|-----|",
    ]
    for row in summary.get("rows") or []:
        m = row.get("model") or {}
        d = row.get("dataset") or {}
        flat_metrics = row.get("metrics_flat") or {}

        def _f(k: str, _flat=flat_metrics) -> str:
            v = _flat.get(k)
            return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        lines.append(
            f"| {row.get('run_name', '?')} | `{m.get('display', '?')}` | "
            f"{m.get('role') or '—'} | {d.get('split') or '—'} | "
            f"{_f('top1')} | {_f('anomaly_f1')} | {_f('anomaly_auc')} |"
        )
    lines.append("")
    return "\n".join(lines)
