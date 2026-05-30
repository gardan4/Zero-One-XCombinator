"""Featherless inference helpers — unit tests (GPU-free) + optional live integration."""

from __future__ import annotations

import pytest


def test_message_text_prefers_content():
    from zo_common.llm import message_text

    resp = {"choices": [{"message": {"content": "VALID.", "reasoning": "thinking..."}}]}
    assert message_text(resp) == "VALID."


def test_message_text_falls_back_to_reasoning():
    from zo_common.llm import message_text

    resp = {"choices": [{"message": {"content": "", "reasoning": "INVALID because SHIP LOT is too early."}}]}
    assert "INVALID" in message_text(resp)


def test_resolve_full_model():
    from zo_common.featherless import HFModelRef, resolve_featherless_model

    assert resolve_featherless_model(HFModelRef(full="XCombinator/sft-fab-all")) == "XCombinator/sft-fab-all"


def test_resolve_merged_override():
    from zo_common.featherless import HFModelRef, resolve_featherless_model

    ref = HFModelRef(
        base="Qwen/Qwen2.5-1.5B-Instruct",
        lora="XCombinator/sft-cot-lora",
        merged="XCombinator/sft-cot-merged",
    )
    assert resolve_featherless_model(ref) == "XCombinator/sft-cot-merged"


def test_resolve_lora_only_raises(monkeypatch):
    from zo_common.featherless import HFModelRef, resolve_featherless_model

    monkeypatch.setattr(
        "zo_common.featherless.hub_has_full_weights",
        lambda repo_id, token=None: False,
    )
    ref = HFModelRef(base="Qwen/Qwen2.5-1.5B-Instruct", lora="XCombinator/sft-cot-lora")
    with pytest.raises(ValueError, match="adapter weights only"):
        resolve_featherless_model(ref)


def test_hub_has_full_weights_detects_adapter_only(monkeypatch):
    from zo_common.featherless import hub_has_full_weights

    monkeypatch.setattr(
        "zo_common.featherless._hf_list_files",
        lambda repo_id, token=None: ["adapter_config.json", "adapter_model.safetensors"],
    )
    assert hub_has_full_weights("org/lora-only") is False


def test_hub_has_full_weights_detects_merged(monkeypatch):
    from zo_common.featherless import hub_has_full_weights

    monkeypatch.setattr(
        "zo_common.featherless._hf_list_files",
        lambda repo_id, token=None: ["config.json", "model.safetensors", "tokenizer.json"],
    )
    assert hub_has_full_weights("org/full-ckpt") is True


def test_parse_featherless_model_spec():
    from zo_eval.predict_llm import _parse_featherless_model

    full = _parse_featherless_model("Qwen/Qwen2.5-1.5B-Instruct")
    assert full.full == "Qwen/Qwen2.5-1.5B-Instruct"

    lora = _parse_featherless_model("Qwen/Qwen2.5-1.5B-Instruct:XCombinator/sft-cot")
    assert lora.base == "Qwen/Qwen2.5-1.5B-Instruct"
    assert lora.lora == "XCombinator/sft-cot"

    triple = _parse_featherless_model("Qwen/Qwen2.5-1.5B-Instruct:XCombinator/lora:XCombinator/merged")
    assert triple.merged == "XCombinator/merged"


@pytest.mark.integration
def test_featherless_live_completion_and_reasoning():
    import os

    if not os.environ.get("FEATHERLESS_API_KEY") and not os.environ.get("HF_TOKEN"):
        pytest.skip("FEATHERLESS_API_KEY or HF_TOKEN required for live Featherless test")

    from zo_common.featherless import FeatherlessClient, HFModelRef, resolve_featherless_model
    from zo_train.datagen import SEP

    full_id = resolve_featherless_model(HFModelRef(full="Qwen/Qwen2.5-1.5B-Instruct"))
    client = FeatherlessClient(full_id)
    prompt = (
        f"Product family: MOSFET\nProcess so far: {SEP.join(['RECEIVE WAFER LOT'])}\n\n"
        "Reply with one fab step name only."
    )
    out = client.complete(prompt, max_tokens=32)
    assert isinstance(out, str) and len(out.strip()) > 0

    reason = FeatherlessClient("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    anomaly = reason.complete(
        "Is RECEIVE WAFER LOT | SHIP LOT valid? Answer VALID or INVALID only.",
        max_tokens=256,
    )
    assert isinstance(anomaly, str) and len(anomaly.strip()) > 0

    merged_id = resolve_featherless_model(
        HFModelRef(base="Qwen/Qwen2.5-1.5B-Instruct", lora="Qwen/Qwen2.5-1.5B-Instruct")
    )
    merged = FeatherlessClient(merged_id).complete(prompt, max_tokens=32)
    assert isinstance(merged, str) and len(merged.strip()) > 0
