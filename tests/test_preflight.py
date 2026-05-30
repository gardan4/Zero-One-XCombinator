"""Preflight validation tests — GPU-free."""

from __future__ import annotations

import pytest
from zo_common import ExperimentConfig


def test_preflight_rejects_missing_dataset(tmp_path, monkeypatch):
    from zo_train.preflight import validate_experiment

    monkeypatch.chdir(tmp_path)
    cfg = ExperimentConfig(
        name="t",
        kind="sft",
        model="/scratch/base",
        dataset="data/generated/MOSFET_sft_lm.jsonl",
    )
    with pytest.raises(ValueError, match="no local files found"):
        validate_experiment(cfg, cluster=False)


def test_preflight_rejects_hf_hub_on_cluster():
    from zo_train.preflight import validate_experiment

    cfg = ExperimentConfig(
        name="t",
        kind="sft",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        dataset=None,
    )
    with pytest.raises(ValueError, match="HuggingFace hub id"):
        validate_experiment(cfg, cluster=True)


def test_cot_anomaly_rows_include_text():
    from zo_train.datagen import anomaly_example

    row = anomaly_example("MOSFET", ["A", "B"], False, rules=["R1"], explain=True)
    assert "text" in row
    assert row["text"] == row["prompt"] + row["completion"]
