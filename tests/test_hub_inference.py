"""Hub inference helper tests."""

from __future__ import annotations

import json

import pytest


def test_hub_model_spec_parse():
    from zo_common.hub_inference import HubModelSpec

    assert HubModelSpec.parse("XCombinator/sft-fab-all").repo_id == "XCombinator/sft-fab-all"
    lora = HubModelSpec.parse("Qwen/Qwen2.5-0.5B-Instruct:XCombinator/leonardo-smoke-qwen-0.5b-lora")
    assert lora.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert lora.repo_id == "XCombinator/leonardo-smoke-qwen-0.5b-lora"


def test_resolve_hub_spec_reads_adapter_config(tmp_path, monkeypatch):
    from zo_common.hub_inference import HubModelSpec, resolve_hub_spec

    adapter_dir = tmp_path / "lora"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("zo_common.hub_inference.hub_has_full_weights", lambda *a, **k: False)
    spec = resolve_hub_spec(HubModelSpec(local_path=str(adapter_dir)))
    assert spec.base_model == "Qwen/Qwen2.5-0.5B-Instruct"


def test_normalize_leonardo_scratch_base():
    from zo_common.hub_inference import _normalize_base_model

    assert _normalize_base_model("/leonardo_scratch/.../Qwen2.5-0.5B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"


def test_hub_chat_fn_shape():
    from zo_common.hub_inference import HubInferenceClient, hub_chat_fn

    class Fake(HubInferenceClient):
        def __init__(self):
            self.max_new_tokens = 64

        def chat(self, messages, **kwargs):
            return "VALID."

    resp = hub_chat_fn(Fake())([{"role": "user", "content": "hi"}], max_tokens=8)
    assert resp["choices"][0]["message"]["content"] == "VALID."


@pytest.mark.integration
def test_xcombinator_model_live():
    import os

    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN required")
    from zo_common.hub_inference import HubInferenceClient

    client = HubInferenceClient("XCombinator/leonardo-smoke-qwen-0.5b-lora")
    out = client.complete("Reply with the word OK only.", max_new_tokens=8)
    assert isinstance(out, str) and len(out.strip()) > 0
