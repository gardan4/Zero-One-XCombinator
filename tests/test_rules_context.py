"""Tests for rules-in-context zero-shot LLM baseline."""

from __future__ import annotations

from pathlib import Path

from zo_eval.predict_llm import RulesContextLLMPredictor
from zo_eval.rules_context import build_messages, build_rules_digest, build_step_vocab_digest, load_generation_rules
from zo_eval.submission import AnomalyInput, ValidInput
from zo_train.datagen import ALL_RULES as DATAGEN_RULES
from zo_train.fab import all_steps

MAX_DIGEST_CHARS = 16000


def test_v5_snapshot_preserved_for_revert():
    snap = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "eval"
        / "zo_eval"
        / "prompt_snapshots"
        / "zeroshot_rules_v5.md"
    )
    assert snap.is_file()


def test_shared_base_in_zeroshot_system():
    from zo_train.prompts import load_system_general

    vi = ValidInput("e1", "MOSFET", 0.6, ["RECEIVE WAFER LOT"])
    sys = build_messages("nextstep", vi)[0]["content"]
    assert load_system_general().splitlines()[0] in sys
    assert "TASK — Next-step prediction" in sys


def test_step_vocab_in_nextstep_prompt():
    vi = ValidInput("e1", "MOSFET", 0.6, ["RECEIVE WAFER LOT"])
    sys = build_messages("nextstep", vi)[0]["content"]
    assert "Allowed fab step names for MOSFET" in sys
    assert "MEASURE VIA CD" in sys
    assert len(build_step_vocab_digest("MOSFET")) > 500
    assert len(all_steps("MOSFET")) >= 100


def test_anomaly_has_v5_guidance_without_vocab_list():
    an = build_messages("anomaly", AnomalyInput("a1", "MOSFET", ["RECEIVE WAFER LOT", "SHIP LOT"]))
    sys = an[0]["content"]
    assert "Default: VALID" in sys
    assert "Most sequences are VALID" in sys
    assert "Allowed fab step names" not in sys
    assert "Forbidden patterns" in sys


def test_nextstep_warns_against_block_labels():
    vi = ValidInput("e1", "MOSFET", 0.6, ["A"])
    sys = build_messages("nextstep", vi)[0]["content"]
    assert "_BLOCK" in sys
    assert "RULE_* identifiers are anomaly violation labels" in sys


def test_load_generation_rules_nonempty():
    text = load_generation_rules()
    assert "Process Sequence Generation Rules" in text
    assert "RULE_DEP_NO_CLEAN" in text


def test_digest_contains_all_rules_and_backbone():
    digest = build_rules_digest("MOSFET")
    for rule in DATAGEN_RULES:
        assert rule in digest
    assert len(digest) < MAX_DIGEST_CHARS


def test_zeroshot_predictor_smoke():
    def fake_chat(messages, **kw):
        user = messages[-1]["content"]
        if "next allowed step" in user.lower():
            c = "Answer: SPIN COAT PHOTORESIST | SOFT BAKE"
        elif "Complete the route" in user:
            c = "DEPOSIT METAL 1 | SHIP LOT"
        else:
            c = "VALID."
        return {"choices": [{"message": {"content": c}}]}

    p = RulesContextLLMPredictor(chat_fn=fake_chat)
    vi = ValidInput("v1", "MOSFET", 0.6, ["RECEIVE WAFER LOT"])
    assert p.next_step(vi)[0] == "SPIN COAT PHOTORESIST"
    iv, _, rule = p.anomaly(AnomalyInput("a1", "MOSFET", ["RECEIVE WAFER LOT", "SHIP LOT"]))
    assert iv == 1 and rule is None
