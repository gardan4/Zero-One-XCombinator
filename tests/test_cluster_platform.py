"""Cross-platform path helpers for cluster/judge CLIs."""

from __future__ import annotations

from pathlib import Path


def test_to_cluster_path_maps_repo_relative(tmp_path, monkeypatch):
    from zo_train.cluster._platform import to_cluster_path

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "extras" / "eval_local").mkdir(parents=True)
    f = repo / "extras" / "eval_local" / "gold.json"
    f.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("zo_common.paths.repo_root", lambda: repo)
    out = to_cluster_path(f, "$HOME/Zero-One-Philyr")
    assert out == "$HOME/Zero-One-Philyr/extras/eval_local/gold.json"


def test_to_cluster_model_path_hf_id():
    from zo_train.cluster._platform import to_cluster_model_path

    assert to_cluster_model_path("XCombinator/foo", "$HOME/r") == "XCombinator/foo"


def test_to_cluster_model_path_windows_absolute(tmp_path, monkeypatch):
    from zo_train.cluster._platform import to_cluster_model_path

    repo = tmp_path / "repo"
    repo.mkdir()
    ckpt = repo / "models" / "ckpt"
    ckpt.mkdir(parents=True)
    monkeypatch.setattr("zo_common.paths.repo_root", lambda: repo)
    out = to_cluster_model_path(str(ckpt), "/leonardo/home/user/Zero-One-Philyr")
    assert out == "/leonardo/home/user/Zero-One-Philyr/models/ckpt"


def test_local_model_cache_dir_no_scratch(monkeypatch):
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.delenv("ZO_MODEL_CACHE_DIR", raising=False)
    from zo_train.cluster._platform import local_model_cache_dir

    p = local_model_cache_dir()
    assert "zo-models" in str(p)
