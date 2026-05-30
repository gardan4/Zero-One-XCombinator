"""Cluster judge-inference helpers — GPU-free unit tests."""

from __future__ import annotations

from pathlib import Path


def test_is_hf_repo_id():
    from zo_train.cluster._slurm import is_hf_repo_id

    assert is_hf_repo_id("XCombinator/sft-fab-all")
    assert not is_hf_repo_id("/scratch/zo-models/foo")
    assert not is_hf_repo_id("$SCRATCH/zo-models/foo")


def test_resolve_infer_model_prefers_path(monkeypatch):
    from zo_train.cluster._slurm import resolve_infer_model

    monkeypatch.setenv("ZO_INFER_MODEL", "XCombinator/sft-fab-all")
    monkeypatch.setenv("ZO_INFER_MODEL_PATH", "/scratch/zo-models/XCombinator--sft-fab-all")
    assert resolve_infer_model() == "/scratch/zo-models/XCombinator--sft-fab-all"


def test_render_infer_sbatch(tmp_path, monkeypatch):
    monkeypatch.setenv("ZO_CLUSTER_REPO_DIR", "/home/user/Zero-One-Philyr")
    monkeypatch.setenv("ZO_CLUSTER_EXPERIMENTS_DIR", "/scratch/zo-experiments")
    monkeypatch.setenv("ZO_SLURM_RESERVATION", "s_tra_ncc")

    from zo_train.cluster._slurm import render_template, slurm_context

    sbatch = render_template(
        "infer.sbatch.j2",
        **slurm_context(
            job_name="zo-infer-test",
            time="00:30:00",
            predictor="hf",
            model_path="/scratch/zo-models/XCombinator--sft-fab-all",
            valid_csv="/home/user/Zero-One-Philyr/extras/eval_local/eval_input_valid.csv",
            anomaly_csv="/home/user/Zero-One-Philyr/extras/eval_local/eval_input_anomaly.csv",
            gold_json="/home/user/Zero-One-Philyr/extras/eval_local/gold.json",
            tasks="nextstep,completion,anomaly",
            tags="judge,repro",
            version="sft-fab-all-v1",
            model_ref="XCombinator/sft-fab-all",
            eval_set="local",
            out_dir="/scratch/zo-experiments/run123/results",
            run_id="20260530_120000_eval_judge_abc123",
        ),
    )
    assert "#SBATCH --reservation=s_tra_ncc" in sbatch
    assert "zo-track predict" in sbatch
    assert '--version "sft-fab-all-v1"' in sbatch
    assert '--run-id "20260530_120000_eval_judge_abc123"' in sbatch
    assert "/scratch/zo-models/XCombinator--sft-fab-all" in sbatch
    assert "--predictor hf" in sbatch
