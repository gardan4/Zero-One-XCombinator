"""Tests for reporting schema, example traces, and comparison helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from zo_common.registry import RunMeta
from zo_eval.examples_trace import TraceCollector, compare_example_traces
from zo_eval.predict import extract_reasoning
from zo_eval.reporting import (
    METRIC_SPECS,
    build_compare_row,
    build_model_identity,
    metric_deltas,
    parse_tags,
    tag_value,
)
from zo_eval.submission import ValidInput


def test_metric_specs_cover_headline_keys():
    keys = {s["key"] for s in METRIC_SPECS}
    assert "top1" in keys
    assert "anomaly_auc" in keys
    assert "rule_attr_acc" in keys


def test_parse_tags_and_model_identity():
    tags = ["split:id", "family:MOSFET", "role:baseline", "model-ref:Org--my-model", "predictor:ngram"]
    parsed = parse_tags(tags)
    assert parsed["split"] == "id"
    assert parsed["family"] == "MOSFET"
    assert tag_value(tags, "model-ref") == "Org--my-model"
    ident = build_model_identity(
        name="track:ngram:v1",
        config={"predictor": "ngram", "version": "v1"},
        tags=tags,
        model_ref="Org/my-model",
    )
    assert ident.role == "baseline"
    assert ident.model_ref == "Org/my-model"


def test_trace_collector_scores_nextstep():
    col = TraceCollector()
    col.add(
        example_id="ex1",
        task="nextstep",
        family="MOSFET",
        input_payload={"partial_sequence": ["A"]},
        prediction=["B", "C"],
        gold="B",
        trace={"source": "test"},
    )
    assert col.rows[0]["correct"] is True
    assert col.rows[0]["rank"] == 1


def test_compare_example_traces_different():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pa = tmp / "a.jsonl"
        pb = tmp / "b.jsonl"
        row_a = {
            "example_id": "ex1",
            "task": "nextstep",
            "family": "MOSFET",
            "prediction": ["A"],
            "gold": "B",
            "correct": False,
            "trace": {},
        }
        row_b = {**row_a, "prediction": ["B"], "correct": True}
        pa.write_text(json.dumps(row_a) + "\n", encoding="utf-8")
        pb.write_text(json.dumps(row_b) + "\n", encoding="utf-8")
        out = compare_example_traces(pa, pb, task="nextstep", mode="different")
        assert len(out) == 1
        assert out[0]["example_id"] == "ex1"


def test_extract_reasoning():
    raw = "<think>because clean</think>\nAnswer: SHIP LOT"
    assert extract_reasoning(raw) == "because clean"


def test_metric_deltas():
    rows = [
        {
            "run_id": "a",
            "metrics_flat": {"top1": 0.5, "anomaly_f1": 0.8},
            "model": {"role": "baseline"},
        },
        {
            "run_id": "b",
            "metrics_flat": {"top1": 0.7, "anomaly_f1": 0.9},
            "model": {"role": "finetuned"},
        },
    ]
    deltas = metric_deltas(rows)
    assert deltas["b"]["top1"] == 0.2


def test_build_compare_row_from_meta(tmp_path, monkeypatch):
    from zo_common import registry as reg

    monkeypatch.setattr(reg, "experiments_dir", lambda: tmp_path)
    meta = RunMeta(
        id="20260101_120000_eval_test_abcd12",
        name="track:ngram:v1",
        kind="eval",
        tags=["split:id", "predictor:ngram", "version:v1"],
        config={"predictor": "ngram", "version": "v1", "eval_set": "local"},
        metrics={"top1": 0.42, "anomaly_f1": 0.55},
    )
    row = build_compare_row(meta)
    assert row["run_id"] == meta.id
    assert row["model"]["predictor"] == "ngram"
    assert row["metrics_flat"]["top1"] == 0.42
