"""Unit tests for Featherless eval helper (no live API calls)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from zo_eval.featherless import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    FeatherlessConfigError,
    configure_llm_env,
    default_model,
)


def test_default_model_from_env(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_MODEL", raising=False)
    assert default_model() == DEFAULT_MODEL
    monkeypatch.setenv("FEATHERLESS_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    assert default_model() == "Qwen/Qwen2.5-7B-Instruct"


def test_configure_llm_env_uses_api_key(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "rc_test_key")
    monkeypatch.setenv("FEATHERLESS_BASE_URL", DEFAULT_BASE_URL)
    url, key = configure_llm_env()
    assert url == DEFAULT_BASE_URL
    assert key == "rc_test_key"
    assert os.environ["ZO_MODEL_BASE_URL"] == DEFAULT_BASE_URL
    assert os.environ["ZO_MODEL_API_KEY"] == "rc_test_key"


def test_fetch_api_key_requires_credentials(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    monkeypatch.delenv("FEATHERLESS_EMAIL", raising=False)
    monkeypatch.delenv("FEATHERLESS_PASSWORD", raising=False)
    from zo_eval.featherless import fetch_api_key_from_login

    with pytest.raises(FeatherlessConfigError):
        fetch_api_key_from_login()


def test_run_featherless_eval_builds_predictor(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "eval_input_valid.csv").write_text(
        "EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE\n"
        "valid_0000,MOSFET,0.6,RECEIVE WAFER LOT\n",
        encoding="utf-8",
    )
    (eval_dir / "eval_input_anomaly.csv").write_text(
        "EXAMPLE_ID,FAMILY,SEQUENCE\na1,MOSFET,RECEIVE WAFER LOT|SHIP LOT\n",
        encoding="utf-8",
    )
    (eval_dir / "gold.json").write_text(
        '{"next":{"valid_0000":"SPIN COAT PHOTORESIST"},"completion":{"valid_0000":["SHIP LOT"]},'
        '"anomaly":{"a1":{"valid":0,"rule":"RULE_SHIP_BEFORE_TEST"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("FEATHERLESS_API_KEY", "rc_test")
    monkeypatch.setenv("ZO_TRACK_USE_WANDB", "0")

    fake_result = {"run_id": "test-run", "out_dir": str(tmp_path / "out")}

    with patch("zo_eval.track.run_track", return_value=fake_result) as mock_track:
        from zo_eval.featherless import run_featherless_eval

        res = run_featherless_eval(
            version="test-v1",
            eval_dir=eval_dir,
            tasks=("nextstep",),
            self_check=False,
            wandb_log=False,
        )
    assert res["run_id"] == "test-run"
    mock_track.assert_called_once()
    predictor = mock_track.call_args[0][0]
    assert predictor.name == "llm-zeroshot"
    assert predictor.model == DEFAULT_MODEL

    def fake_chat(messages, **kw):
        user = messages[-1]["content"]
        if "Last executed step" in user:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"reasoning": "next", "steps": ["SPIN COAT PHOTORESIST"]}'
                            )
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": "OK"}}]}

    predictor._chat = fake_chat
    from zo_eval.submission import ValidInput

    steps, _ = predictor.next_step_with_trace(
        ValidInput("valid_0000", "MOSFET", 0.6, ["RECEIVE WAFER LOT"])
    )
    assert steps[0] == "SPIN COAT PHOTORESIST"
