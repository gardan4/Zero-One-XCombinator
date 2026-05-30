"""Tests for Featherless JSON repair in zo_common.llm."""

from zo_common.llm import _parse_chat_response, content, message_text


class _FakeResp:
    def __init__(self, text: str):
        self.text = text


def test_repair_duplicate_content_without_comma():
    raw = (
        '{"choices":[{"message":{"role":"assistant","content":"OK""content":""},'
        '"finish_reason":"stop"}]}'
    )
    data = _parse_chat_response(_FakeResp(raw))
    assert content(data) == "OK"


def test_merge_split_duplicate_content():
    raw = (
        '{"choices":[{"message":{"role":"assistant","content":"HE","content":"LLO"},'
        '"finish_reason":"stop"}]}'
    )
    data = _parse_chat_response(_FakeResp(raw))
    assert content(data) == "HELLO"


def test_merge_split_json_across_content_keys():
    """DeepSeek JSON answers split as ``{"`` + ``reasoning": ...`` across duplicate keys."""
    raw = (
        '{"choices":[{"message":{"role":"assistant","content":"{\\"","content":"reasoning\\": \\"ok\\", \\"steps\\": [\\"A\\"]}"},'
        '"finish_reason":"stop"}]}'
    )
    data = _parse_chat_response(_FakeResp(raw))
    assert '"reasoning"' in content(data)
    assert "A" in content(data)


def test_message_text_falls_back_to_reasoning():
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "Answer: SPIN COAT PHOTORESIST",
                }
            }
        ]
    }
    assert message_text(resp) == "Answer: SPIN COAT PHOTORESIST"
