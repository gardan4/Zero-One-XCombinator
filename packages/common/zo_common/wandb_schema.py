"""W&B run schema: tags, metric prefixes, artifact types, and visibility rules."""

from __future__ import annotations

from typing import Any

# Visibility tags — dashboard defaults show only real-run / reportable.
TAG_REAL_RUN = "real-run"
TAG_REPORTABLE = "reportable"
TAG_TEST = "test"
TAG_SMOKE = "smoke"
TAG_DEBUG = "debug"
TAG_PROXY_ONLY = "proxy-only"
TAG_REPORT_FINAL = "report:final"

HIDE_BY_DEFAULT = frozenset({TAG_TEST, TAG_SMOKE, TAG_DEBUG, TAG_PROXY_ONLY})
SHOW_BY_DEFAULT = frozenset({TAG_REAL_RUN, TAG_REPORTABLE})

JOB_TYPES = frozenset({"train", "eval", "inference"})

ARTIFACT_EVAL_RESULTS = "eval-results"
ARTIFACT_SUBMISSION = "submission-results"

# Metric key prefixes for W&B logging
PREFIX_TRAIN = "train"
PREFIX_EVAL = "eval"
PREFIX_PROXY = "proxy"
PREFIX_LIKELIHOOD = "likelihood"

# Flat registry keys → W&B eval/* keys (labeled eval)
EVAL_METRIC_MAP: dict[str, str] = {
    "top1": "eval/top1",
    "top3": "eval/top3",
    "top5": "eval/top5",
    "mrr": "eval/mrr",
    "em": "eval/em",
    "ned": "eval/ned",
    "token_acc": "eval/token_acc",
    "block_acc": "eval/block_acc",
    "anomaly_acc": "eval/anomaly_acc",
    "anomaly_p": "eval/anomaly_p",
    "anomaly_r": "eval/anomaly_r",
    "anomaly_f1": "eval/anomaly_f1",
    "anomaly_auc": "eval/anomaly_auc",
    "rule_attr_acc": "eval/rule_attr_acc",
}

PROXY_METRIC_MAP: dict[str, str] = {
    "proxy_rank1_vocab": "proxy/rank1_vocab",
    "proxy_grammar_valid": "proxy/grammar_valid",
    "proxy_oracle_acc": "proxy/oracle_acc",
}


def prefixed_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    """Map flat metrics to W&B namespaced keys."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        if k.startswith("proxy_"):
            out[PROXY_METRIC_MAP.get(k, f"{PREFIX_PROXY}/{k}")] = float(v)
        elif k in EVAL_METRIC_MAP:
            out[EVAL_METRIC_MAP[k]] = float(v)
        elif k.startswith("nll__") or k.startswith("ppl__"):
            metric, checkpoint, family = k.split("__", 2)
            out[f"{PREFIX_LIKELIHOOD}/{metric}/{checkpoint}/{family}"] = float(v)
        else:
            out[f"{prefix}/{k}"] = float(v)
    return out


def merge_tags(*tag_lists: list[str] | None, extra: list[str] | None = None) -> list[str]:
    out: list[str] = []
    for lst in tag_lists:
        if not lst:
            continue
        for t in lst:
            t = str(t).strip()
            if t and t not in out:
                out.append(t)
    if extra:
        for t in extra:
            t = str(t).strip()
            if t and t not in out:
                out.append(t)
    return out


def is_reportable(tags: list[str]) -> bool:
    ts = set(tags)
    if ts & HIDE_BY_DEFAULT and not (ts & SHOW_BY_DEFAULT):
        return False
    return bool(ts & SHOW_BY_DEFAULT) or TAG_REPORT_FINAL in ts


def should_hide_by_default(tags: list[str]) -> bool:
    ts = set(tags)
    if ts & SHOW_BY_DEFAULT:
        return False
    return bool(ts & HIDE_BY_DEFAULT)


def validate_run_tags(
    tags: list[str],
    *,
    job_type: str,
    require_real: bool = False,
    require_version: bool = False,
) -> list[str]:
    """Return warnings for missing recommended tags (does not raise)."""
    warnings: list[str] = []
    ts = set(tags)
    if require_real and TAG_REAL_RUN not in ts and TAG_TEST not in ts:
        warnings.append("missing tag 'real-run' (add it for production runs)")
    if require_version and not any(t.startswith("version:") for t in tags):
        warnings.append("missing tag 'version:<id>'")
    if job_type == "eval" and not any(t.startswith("predictor:") for t in tags):
        warnings.append("eval runs should include predictor:<kind>")
    return warnings
