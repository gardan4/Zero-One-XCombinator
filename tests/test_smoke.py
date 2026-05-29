"""Light smoke tests — no torch / GPU stack required.

Exercise the shared run-registry contract and config round-trip so a broken
laptop install is caught before anyone burns cluster time. Run: `just test`.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ZO_EXPERIMENTS_DIR", str(tmp_path))
    from zo_common import registry

    run = registry.new_run(name="smoke test", kind="sft")
    assert run.id.endswith("_sft_smoke-test")

    registry.append_metric(run.id, step=0, loss=1.0)
    registry.append_metric(run.id, step=1, loss=0.4)

    rows = registry.read_metrics(run.id)
    assert len(rows) == 2
    assert rows[-1]["step"] == 1
    assert rows[-1]["loss"] == 0.4

    got = registry.get_run(run.id)
    assert got is not None
    assert got.metrics.get("loss") == 0.4  # latest summary kept on meta.json

    assert any(r.id == run.id for r in registry.list_runs())


def test_config_roundtrip(tmp_path):
    from zo_common.config import ExperimentConfig

    cfg = ExperimentConfig(name="t", kind="grpo", extra={"num_generations": 4})
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)

    loaded = ExperimentConfig.from_yaml(path)
    assert loaded.name == "t"
    assert loaded.kind == "grpo"
    assert loaded.extra["num_generations"] == 4


def test_example_configs_parse():
    from zo_common.config import ExperimentConfig

    configs = sorted((REPO_ROOT / "packages" / "training" / "configs").glob("*.yaml"))
    assert configs, "expected example training configs to exist"
    for path in configs:
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.name
