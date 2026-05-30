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
            track_batch_size="16",
        ),
    )
    assert "#SBATCH --reservation=s_tra_ncc" in sbatch
    assert "zo-track predict" in sbatch
    assert '--version "sft-fab-all-v1"' in sbatch
    assert '--run-id "20260530_120000_eval_judge_abc123"' in sbatch
    assert 'export ZO_TRACK_BATCH_SIZE="16"' in sbatch
    assert "_trim_env" in sbatch
    assert "WANDB_ENTITY" in sbatch
    assert "/scratch/zo-models/XCombinator--sft-fab-all" in sbatch
    assert "--predictor hf" in sbatch


def test_prepare_cluster_for_job_noop_without_ssh(monkeypatch):
    from zo_train.cluster._remote import prepare_cluster_for_job

    monkeypatch.delenv("ZO_CLUSTER_HOST", raising=False)
    monkeypatch.delenv("ZO_CLUSTER_USER", raising=False)
    monkeypatch.delenv("ZO_CLUSTER_ON_LOGIN", raising=False)
    prepare_cluster_for_job()  # should not raise


def test_prepare_cluster_for_job_skip_flag(monkeypatch):
    from zo_train.cluster._remote import cluster_prep_skipped, prepare_cluster_for_job

    monkeypatch.setenv("ZO_CLUSTER_HOST", "u@host")
    called = {"n": 0}
    monkeypatch.setattr(
        "zo_train.cluster._remote.push_cluster_env",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    assert cluster_prep_skipped(skip=True)
    assert cluster_prep_skipped(skip=False) is False
    monkeypatch.setenv("ZO_CLUSTER_SKIP_PREP", "1")
    assert cluster_prep_skipped(skip=False)
    monkeypatch.delenv("ZO_CLUSTER_SKIP_PREP")
    prepare_cluster_for_job(skip=True)
    assert called["n"] == 0
