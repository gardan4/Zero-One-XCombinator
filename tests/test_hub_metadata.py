"""HF artifact metadata (training_manifest.json + model card)."""

import json
import tempfile
from pathlib import Path

from zo_common.config import ExperimentConfig
from zo_common.hub_metadata import (
    build_training_manifest,
    collect_tags,
    render_model_card,
    write_hub_artifact_metadata,
)


def test_collect_tags_merges_sources():
    cfg = ExperimentConfig(
        name="t",
        extra={"tags": ["sft", "lofo:mosfet"], "hub_tags": ["report-final", "sft"]},
    )
    tags = collect_tags(cfg, {"tags": ["cluster", "sft"]})
    assert tags == ["sft", "lofo:mosfet", "report-final", "cluster"]


def test_manifest_and_model_card():
    cfg = ExperimentConfig(
        name="leonardo-sft-fab-lofo-mosfet",
        learning_rate=1e-5,
        epochs=2,
        lora=False,
        extra={
            "tags": ["leonardo", "full-ft", "lofo:mosfet"],
            "hub_tags": ["attempt-2"],
            "base_model_hub_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "hub_notes": "Best LOFO MOSFET run",
        },
    )
    manifest = build_training_manifest(
        run_id="20260530_train_test_abc",
        cfg=cfg,
        run_meta={"git_sha": "deadbeef", "git_branch": "main"},
        hub_model_id="XCombinator/sft-lofo-mosfet",
    )
    assert manifest["run_id"] == "20260530_train_test_abc"
    assert "lofo:mosfet" in manifest["tags"]
    assert "attempt-2" in manifest["tags"]
    assert manifest["training"]["learning_rate"] == 1e-5

    card = render_model_card(manifest)
    assert "tags:" in card
    assert "lofo:mosfet" in card
    assert "training_manifest.json" in card
    assert "1e-05" in card or "1e-5" in card


def test_write_hub_artifact_metadata_files():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "artifacts"
        cfg = ExperimentConfig(name="smoke", extra={"tags": ["smoke"]})
        path = write_hub_artifact_metadata(
            out,
            "fake_run_id",
            cfg,
            hub_model_id="XCombinator/test-model",
            notes="manual note",
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["hub_model_id"] == "XCombinator/test-model"
        assert (out / "README.md").exists()
        assert "manual note" in (out / "README.md").read_text()
