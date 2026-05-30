"""Tests for JSON structured answer extraction."""

from zo_eval.predict import extract_answer, extract_reasoning, parse_anomaly


def test_extract_answer_json_steps():
    raw = '{"reasoning": "via block next", "steps": ["MEASURE VIA CD", "DEPOSIT BARRIER METAL"]}'
    assert extract_answer(raw) == "MEASURE VIA CD | DEPOSIT BARRIER METAL"
    assert "via block" in (extract_reasoning(raw) or "")


def test_extract_answer_json_anomaly_valid():
    raw = '{"reasoning": "no rule hit", "valid": true, "rule": null}'
    assert extract_answer(raw) == "VALID."
    assert parse_anomaly(raw) == (1, None)


def test_extract_answer_json_anomaly_invalid():
    raw = '{"reasoning": "ship before test", "valid": false, "rule": "RULE_SHIP_BEFORE_TEST"}'
    assert parse_anomaly(raw) == (0, "RULE_SHIP_BEFORE_TEST")


def test_extract_answer_fenced_json():
    raw = 'Here is my answer:\n```json\n{"steps": ["SHIP LOT"]}\n```'
    assert extract_answer(raw) == "SHIP LOT"


def test_legacy_answer_fallback():
    assert extract_answer("Answer: DEPOSIT METAL 1 | SHIP LOT") == "DEPOSIT METAL 1 | SHIP LOT"
