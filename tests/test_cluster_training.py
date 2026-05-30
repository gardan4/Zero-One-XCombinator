"""Cluster path expansion and remote helper tests."""

from __future__ import annotations


def test_expand_env_refs_braces(monkeypatch):
    from zo_train.cluster._platform import expand_env_refs

    monkeypatch.setenv("ZO_SMOKE_BASE_MODEL_DIR", "$SCRATCH/hf-local/foo")
    assert expand_env_refs("${ZO_SMOKE_BASE_MODEL_DIR}") == "$SCRATCH/hf-local/foo"


def test_remote_expand_scratch(monkeypatch):
    from zo_train.cluster._platform import remote_expand

    monkeypatch.setenv("ZO_CLUSTER_SCRATCH", "/leonardo_scratch/large/user/foo")
    assert remote_expand("$SCRATCH/zo-experiments") == "/leonardo_scratch/large/user/foo/zo-experiments"


def test_expand_cluster_path_chain(monkeypatch):
    from zo_train.cluster._remote import expand_cluster_path

    monkeypatch.setenv("ZO_SMOKE_BASE_MODEL_DIR", "$SCRATCH/hf-local/Qwen2.5-0.5B-Instruct")
    monkeypatch.setenv("ZO_CLUSTER_SCRATCH", "/leonardo_scratch/large/usertrain/u/hf")
    out = expand_cluster_path("${ZO_SMOKE_BASE_MODEL_DIR}")
    assert out == "/leonardo_scratch/large/usertrain/u/hf/hf-local/Qwen2.5-0.5B-Instruct"


def test_resolve_config_expands_model(tmp_path, monkeypatch):
    from zo_train.cluster.submit import _resolve_config

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg_dir = repo / "packages" / "training" / "configs"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "test.yaml"
    cfg_file.write_text(
        "name: t\nkind: sft\nmodel: ${ZO_SMOKE_BASE_MODEL_DIR}\ndataset: null\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ZO_SMOKE_BASE_MODEL_DIR", "$SCRATCH/hf-local/base")
    monkeypatch.setenv("ZO_CLUSTER_SCRATCH", "/scratch/u")
    monkeypatch.chdir(repo)
    cfg = _resolve_config(str(cfg_file))
    assert cfg.model == "/scratch/u/hf-local/base"


def test_submit_dry_run_renders_sbatch(tmp_path, monkeypatch):
    from zo_train.cluster.submit import submit

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "packages" / "training" / "configs").mkdir(parents=True)
    cfg = repo / "packages" / "training" / "configs" / "leonardo_smoke_hf.yaml"
    cfg.write_text(
        "name: smoke\nkind: sft\nmodel: /scratch/base\ndataset: null\n",
        encoding="utf-8",
    )
    (repo / "experiments").mkdir()
    (repo / ".env").write_text(
        "ZO_CLUSTER_REPO_DIR=/leonardo/home/user/Zero-One-Philyr\n"
        "ZO_CLUSTER_EXPERIMENTS_DIR=/leonardo_scratch/user/zo-experiments\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("zo_common.paths.repo_root", lambda: repo)
    monkeypatch.setattr("zo_train.cluster._remote.repo_root", lambda: repo)

    import typer

    try:
        submit(config=str(cfg), kind="sft", dry_run=True)
    except typer.Exit:
        pass

    runs = list((repo / "experiments").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    sbatch = (run_dir / "job.sbatch").read_text(encoding="utf-8")
    assert "/leonardo/home/user/Zero-One-Philyr" in sbatch
    cfg_out = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "/scratch/base" in cfg_out
