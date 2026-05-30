"""Base (un-fine-tuned) LLM predictor: rich-context prompt + ranked next-step parsing.

GPU-free — the served predictor takes an injected ``chat_fn``, so no model/server is loaded.
"""

from __future__ import annotations

from zo_eval.predict_llm import ServedLLMPredictor, _family_context, _ns_prompt_base
from zo_eval.predictors import PREDICTOR_KINDS, build_predictor
from zo_eval.submission import ValidInput


def test_base_kinds_registered():
    assert "base" in PREDICTOR_KINDS and "base-hf" in PREDICTOR_KINDS


def test_base_factory_routes_to_style_base():
    # `base` is the served path (no network until a request is made), so this is safe offline.
    p = build_predictor("base", model="Qwen/Qwen2.5-7B-Instruct")
    assert p.name == "base" and p.style == "base"


def test_family_context_loads_relevant_data():
    cand, ref, desc = _family_context("MOSFET")
    assert "SHIP LOT" in cand and "RECEIVE WAFER LOT" in cand  # the legal step vocabulary
    assert len(ref) > 50  # a full canonical reference recipe
    assert desc.get("SPIN COAT PHOTORESIST", {}).get("description")  # NL step descriptions


def test_base_nextstep_prompt_has_all_relevant_context():
    item = ValidInput(
        "v1", "MOSFET", 0.6, ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "SPIN COAT PHOTORESIST"]
    )
    prompt = _ns_prompt_base(item)
    assert "Product family: MOSFET" in prompt
    assert "Allowed process steps" in prompt and "- SHIP LOT" in prompt  # candidate vocab
    assert "Meaning of the most recent steps" in prompt  # recent-step descriptions
    assert "Process so far (3 steps)" in prompt and "SPIN COAT PHOTORESIST" in prompt


def test_base_nextstep_parses_ranked_candidates():
    def fake_chat(messages, **kw):
        # The base prompt asks for up to 5 pipe-separated candidates, most likely first.
        return {"choices": [{"message": {"content": "DEVELOP PHOTORESIST | SOFT BAKE | HARD BAKE"}}]}

    p = ServedLLMPredictor(chat_fn=fake_chat, style="base")
    assert p.name == "base"
    item = ValidInput("v1", "MOSFET", 0.6, ["SPIN COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1"])
    ranked = p.next_step(item)
    assert ranked[0] == "DEVELOP PHOTORESIST"  # rank-1 is the headline metric
    assert "SOFT BAKE" in ranked and len(ranked) <= 5


def test_base_reference_recipe_toggle(monkeypatch):
    monkeypatch.setenv("ZO_BASE_PROMPT_REFERENCE", "0")
    _family_context.cache_clear()
    try:
        _, ref, _ = _family_context("MOSFET")
        assert ref == ()  # reference recipe dropped when measuring learned ordering
    finally:
        _family_context.cache_clear()  # don't leak the toggled state to other tests
