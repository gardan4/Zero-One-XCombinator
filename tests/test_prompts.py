"""Tests for shared fab task prompts (instruct SFT + finetuned eval)."""

from __future__ import annotations

from pathlib import Path

import pytest
from zo_eval.submission import AnomalyInput, ValidInput
from zo_train.datagen import (
    anomaly_example,
    completion_example,
    nextstep_example,
)
from zo_train.prompts import (
    DEFAULT_SYSTEM_GENERAL,
    PromptItem,
    build_base_system,
    build_instruct_messages,
    build_json_completion_nextstep,
    build_messages,
    build_sft_user,
    load_system_general,
    use_system_prompt,
)


def test_base_system_contains_general_and_task():
    sys = build_base_system("nextstep")
    assert load_system_general() in sys
    assert "Next-step prediction" in sys
    assert "OUTPUT FORMAT" in sys
    assert '"steps"' in sys
    assert "Process grammar reference" not in sys


def test_anomaly_base_lists_rule_ids():
    sys = build_base_system("anomaly")
    assert "RULE_DEP_NO_CLEAN" in sys
    assert "Allowed rule IDs" in sys


def test_sft_user_matches_datagen_prompts():
    prefix = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"]
    vi = ValidInput("e1", "MOSFET", 0.6, prefix)
    assert build_sft_user("nextstep", vi) == nextstep_example("MOSFET", prefix, "")["prompt"]
    assert build_sft_user("completion", vi) == completion_example("MOSFET", prefix, [])["prompt"]
    ai = AnomalyInput("a1", "MOSFET", prefix)
    assert build_sft_user("anomaly", ai) == anomaly_example("MOSFET", prefix, True)["prompt"]


def test_numbered_user_nextstep():
    vi = ValidInput("e1", "MOSFET", 0.6, ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"])
    user = build_sft_user("nextstep", vi)
    assert "1. RECEIVE WAFER LOT" in user
    assert 'Last executed step (#2): "LOT IDENTIFICATION"' in user


def test_datagen_nextstep_json_completion():
    ex = nextstep_example("MOSFET", ["A"], "B")
    assert '"steps": ["B"]' in ex["completion"]
    assert '"reasoning": ""' in ex["completion"]


def test_build_messages_has_system_and_user():
    vi = ValidInput("e1", "MOSFET", 0.6, ["A"])
    msgs = build_messages("nextstep", vi)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert DEFAULT_SYSTEM_GENERAL.splitlines()[0] in msgs[0]["content"]


def test_instruct_messages_includes_assistant():
    comp = build_json_completion_nextstep("B")
    row = build_instruct_messages(
        "nextstep",
        PromptItem("MOSFET", partial_sequence=["A"]),
        comp,
    )
    assert row[-1]["role"] == "assistant"
    assert row[-1]["content"] == comp


def test_system_prompt_path_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    custom = tmp_path / "system.txt"
    custom.write_text("CUSTOM SYSTEM PROMPT", encoding="utf-8")
    monkeypatch.setenv("ZO_SYSTEM_PROMPT_PATH", str(custom))
    assert load_system_general() == "CUSTOM SYSTEM PROMPT"
    assert "CUSTOM SYSTEM PROMPT" in build_base_system("completion")


def test_prompt_legacy_disables_system(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZO_PROMPT_LEGACY", "1")
    assert use_system_prompt() is False
    vi = ValidInput("e1", "MOSFET", 0.6, ["A"])
    msgs = build_messages("nextstep", vi)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
