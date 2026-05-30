"""Tests for hf_hub_util and minimal inference bootstrap."""

from __future__ import annotations

import json


def test_hub_has_full_weights_local_adapter_only(tmp_path):
    from zo_common.hf_hub_util import hub_has_full_weights

    d = tmp_path / "lora"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"x")
    assert hub_has_full_weights(str(d)) is False


def test_hub_has_full_weights_local_full(tmp_path):
    from zo_common.hf_hub_util import hub_has_full_weights

    d = tmp_path / "full"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    assert hub_has_full_weights(str(d)) is True


def test_ensure_inference_deps_exits_when_torch_missing(monkeypatch):
    from zo_common import hub_inference

    monkeypatch.setattr(hub_inference, "_can_import", lambda name: False)
    try:
        hub_inference.ensure_inference_deps()
        raised = False
    except SystemExit as e:
        raised = True
        assert "pip install" in str(e)
    assert raised
