"""Tests for rules-in-context zero-shot LLM baseline."""

from __future__ import annotations

from zo_eval.predict_llm import RulesContextLLMPredictor
from zo_eval.rules_context import build_messages, build_rules_digest, load_generation_rules
from zo_eval.submission import AnomalyInput, ValidInput
from zo_train.datagen import ALL_RULES as DATAGEN_RULES

MAX_DIGEST_CHARS = 12000


def test_load_generation_rules_nonempty():
    text = load_generation_rules()
    assert "Process Sequence Generation Rules" in text
    assert "RULE_DEP_NO_CLEAN" in text


def test_digest_contains_all_rules_and_backbone():
    digest = build_rules_digest()
    for rule in DATAGEN_RULES:
        assert rule in digest
    assert "Shared process backbone" in digest
    assert "Lithography block template" in digest
    assert len(digest) < MAX_DIGEST_CHARS


def test_digest_family_specific():
    mos = build_rules_digest("MOSFET")
    assert "Active product family: MOSFET" in mos
    assert "EPITAXIAL DEPOSITION" in mos
    all_fam = build_rules_digest()
    assert "Family-specific prep blocks:" in all_fam


def test_build_messages_shape():
    vi = ValidInput("e1", "IGBT", 0.6, ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"])
    msgs = build_messages("nextstep", vi)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "IGBT" in msgs[1]["content"]
    assert "RECEIVE WAFER LOT" in msgs[1]["content"]
    assert "RULE_DEP_NO_CLEAN" in msgs[0]["content"]

    cp = build_messages("completion", vi)
    assert "Complete the remaining" in cp[1]["content"]

    ai = AnomalyInput("a1", "IC", ["RECEIVE WAFER LOT", "SHIP LOT"])
    an = build_messages("anomaly", ai)
    assert "VALID." in an[1]["content"]
    for rule in DATAGEN_RULES:
        assert rule in an[1]["content"] or rule in an[0]["content"]


def test_zeroshot_predictor_smoke():
    def fake_chat(messages, **kw):
        user = messages[-1]["content"]
        assert messages[0]["role"] == "system"
        if "next process step" in user.lower():
            c = "SPIN COAT PHOTORESIST | SOFT BAKE | DEVELOP PHOTORESIST"
        elif "Complete the remaining" in user:
            c = "DEPOSIT METAL 1 | SHIP LOT"
        else:
            c = "INVALID. RULE_DEP_NO_CLEAN"
        return {"choices": [{"message": {"content": c}}]}

    p = RulesContextLLMPredictor(chat_fn=fake_chat)
    vi = ValidInput("v1", "MOSFET", 0.6, ["RECEIVE WAFER LOT"])
    ns = p.next_step(vi)
    assert ns[0] == "SPIN COAT PHOTORESIST"
    assert len(ns) <= 5

    cp = p.complete(vi)
    assert cp == ["DEPOSIT METAL 1", "SHIP LOT"]

    iv, score, rule = p.anomaly(AnomalyInput("a1", "MOSFET", ["RECEIVE WAFER LOT", "SHIP LOT"]))
    assert iv == 0 and rule == "RULE_DEP_NO_CLEAN"
    assert 0.0 <= score <= 1.0


def test_build_predictor_llm_zeroshot_registered():
    from zo_eval.predictors import build_predictor

    p = build_predictor("llm-zeroshot", model="default")
    assert p.name == "llm-zeroshot"

    p_hf = build_predictor("llm-zeroshot", model="Qwen/Qwen2.5-1.5B-Instruct")
    assert p_hf.name == "llm-zeroshot"
    assert p_hf.backend == "hf"
