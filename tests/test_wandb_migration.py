"""Tests for W&B schema, tagging, and run store filters."""

from __future__ import annotations

import json
from pathlib import Path

from zo_common.registry import RunMeta
from zo_common.wandb_runs import wandb_enabled
from zo_common.wandb_schema import (
    TAG_PROXY_ONLY,
    TAG_REAL_RUN,
    TAG_TEST,
    is_reportable,
    merge_tags,
    prefixed_metrics,
    should_hide_by_default,
    validate_run_tags,
)
from zo_eval.run_store import RepoResultsStore, filter_runs
from zo_eval.wandb_publish import apply_eval_tags


def test_wandb_disabled_when_mode_disabled(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert wandb_enabled() is False


def test_wandb_disabled_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert wandb_enabled() is False


def test_normalize_env_value_strips_api_key():
    from zo_common.env import normalize_env_value

    assert normalize_env_value("WANDB_API_KEY", "  abc123  ") == "abc123"
    assert normalize_env_value("ZO_FOO", "  bar  ") == "bar"
    assert normalize_env_value("WANDB_MODE", "online\r") == "online"


def test_wandb_enabled_ignores_crlf_mode(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_MODE", "online\r")
    assert wandb_enabled() is True


def test_merge_tags_dedupes():
    assert merge_tags(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_prefixed_metrics_eval_and_proxy():
    m = prefixed_metrics({"top1": 0.9, "proxy_rank1_vocab": 0.5}, "eval")
    assert m["eval/top1"] == 0.9
    assert m["proxy/rank1_vocab"] == 0.5


def test_should_hide_test_runs():
    assert should_hide_by_default([TAG_TEST]) is True
    assert should_hide_by_default([TAG_REAL_RUN, TAG_TEST]) is False
    assert is_reportable([TAG_REAL_RUN]) is True


def test_apply_eval_tags_proxy_and_debug():
    tags = apply_eval_tags(["source:dashboard-inference"], has_gold=False)
    assert TAG_PROXY_ONLY in tags
    assert "debug" in tags


def test_validate_run_tags_warnings():
    warns = validate_run_tags([], job_type="eval", require_real=True)
    assert any("real-run" in w for w in warns)


def test_filter_runs_hides_tests_by_default():
    runs = [
        RunMeta(id="a", name="real", kind="eval", tags=["real-run", "reportable"], metrics={"top1": 0.5}),
        RunMeta(id="b", name="test", kind="eval", tags=["test"], metrics={"top1": 1.0}),
        RunMeta(id="c", name="untagged", kind="eval", tags=[], metrics={"top1": 0.8}),
    ]
    out = filter_runs(runs, only_reportable=True, include_tests=False)
    assert len(out) == 1
    assert out[0].id == "a"


def test_repo_store_empty_index():
    store = RepoResultsStore(index_path=Path("/nonexistent/INDEX.json"))
    assert store.list_runs() == []


def test_repo_store_reads_promote_results_index_shape(tmp_path: Path):
    results = tmp_path / "final"
    results.mkdir()
    (results / "metrics_report.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "nextstep": {"by_family": {"overall": {"top1": 0.7, "top3": 0.9}}},
                    "anomaly": {"by_family": {"overall": {"f1": 0.6}}},
                }
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "INDEX.json"
    index.write_text(
        json.dumps(
            {
                "final": {
                    "path": str(results),
                    "run_id": "run-final",
                    "tags": ["report:final", "real-run"],
                }
            }
        ),
        encoding="utf-8",
    )

    [meta] = RepoResultsStore(index_path=index).list_runs()
    assert meta.id == "run-final"
    assert meta.config["slug"] == "final"
    assert meta.metrics["top1"] == 0.7
    assert meta.metrics["anomaly_f1"] == 0.6
