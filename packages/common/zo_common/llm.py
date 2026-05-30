"""Minimal OpenAI-compatible chat client (httpx).

Points at ZO_MODEL_BASE_URL — typically a vLLM server hosting your fine-tuned model
(`just serve <model>`), but works with any OpenAI-style endpoint. Shared by eval + agent.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx


def _base_url() -> str:
    return os.environ.get("ZO_MODEL_BASE_URL", "http://localhost:8001/v1")


def _api_key(base_url: str | None = None) -> str:
    url = base_url or _base_url()
    if "featherless.ai" in url:
        fk = os.environ.get("FEATHERLESS_API_KEY")
        if fk:
            return fk
    return os.environ.get("ZO_MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"


_CONTENT_FIELD_RE = re.compile(r'"content"\s*:\s*"((?:\\.|[^"\\])*)"')


def _decode_json_string(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def _balanced_brace_span(text: str, open_idx: int) -> tuple[int, int] | None:
    """Return ``(start, end_exclusive)`` for the ``{...}`` block opened at ``open_idx``."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return open_idx, i + 1
    return None


def _merged_assistant_content(text: str) -> str | None:
    """Join duplicate ``content`` string values inside ``choices[0].message``."""
    msg_key = '"message"'
    idx = text.find(msg_key)
    if idx < 0:
        return None
    brace = text.find("{", idx + len(msg_key))
    span = _balanced_brace_span(text, brace)
    if not span:
        return None
    blob = text[span[0] : span[1]]
    parts = [_decode_json_string(m.group(1)) for m in _CONTENT_FIELD_RE.finditer(blob)]
    if len(parts) <= 1:
        return parts[0] if parts else None
    return "".join(parts)


def _merge_duplicate_content_fields(text: str) -> str:
    """Rewrite ``choices[0].message`` so duplicate ``content`` keys become one merged value."""
    msg_key = '"message"'
    idx = text.find(msg_key)
    if idx < 0:
        return text
    brace = text.find("{", idx + len(msg_key))
    span = _balanced_brace_span(text, brace)
    if not span:
        return text
    blob = text[span[0] : span[1]]
    parts = [_decode_json_string(m.group(1)) for m in _CONTENT_FIELD_RE.finditer(blob)]
    if len(parts) <= 1:
        return text
    merged = "".join(parts)
    escaped = json.dumps(merged)[1:-1]
    replacement = f'"message":{{"role":"assistant","content":"{escaped}"}}'
    return text[:idx] + replacement + text[span[1] :]


def _repair_featherless_json(text: str) -> str:
    """Best-effort fix for malformed Featherless JSON (duplicate/unquoted content keys)."""
    if text.count('"content"') > 1:
        merged = _merge_duplicate_content_fields(text)
        if merged != text:
            return merged
    return re.sub(
        r'"content"\s*:\s*"((?:\\.|[^"\\])*)"\s*"content"\s*:\s*"(?:\\.|[^"\\])*"',
        r'"content":"\1"',
        text,
    )


def _parse_chat_response(resp: httpx.Response) -> dict[str, Any]:
    text = resp.text
    if text.count('"content"') <= 1:
        return json.loads(text)

    merged_content = _merged_assistant_content(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_repair_featherless_json(text))

    if merged_content is not None:
        content_val = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if merged_content != content_val:
            data["choices"][0]["message"]["content"] = merged_content
    return data


def chat(
    messages: list[dict[str, Any]],
    model: str = "default",
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    url = (base_url or _base_url()).rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    # Only sent when set, to keep the default request lean. logprobs/top_logprobs power the
    # anomaly SCORE + ROC-AUC (Stream 4); n>1 gives sampled candidates for next-step top-k.
    for key, val in (("logprobs", logprobs), ("top_logprobs", top_logprobs), ("n", n)):
        if val is not None:
            payload[key] = val
    resolved_url = base_url or _base_url()
    headers = {"Authorization": f"Bearer {api_key or _api_key(resolved_url)}"}
    retries = 5 if "featherless.ai" in resolved_url else 1
    last_resp: httpx.Response | None = None
    for attempt in range(retries):
        last_resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        if last_resp.status_code not in (429, 502, 503, 504) or attempt >= retries - 1:
            break
        # 429 = Featherless concurrency cap; back off longer before retry.
        delay = min(60.0, 5.0 * (2.0**attempt)) if last_resp.status_code == 429 else min(30.0, 2.0**attempt * 5.0)
        time.sleep(delay)
    assert last_resp is not None
    last_resp.raise_for_status()
    return _parse_chat_response(last_resp)


def content(resp: dict[str, Any]) -> str:
    return message_text(resp)


def message_text(resp: dict[str, Any], *, include_reasoning: bool = True) -> str:
    """Assistant text from a chat completion; falls back to ``reasoning`` for thinking models."""
    msg = resp["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if text:
        return text
    if include_reasoning:
        return (msg.get("reasoning") or "").strip()
    return ""


def token_logprobs(resp: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-token logprobs from an OpenAI/vLLM chat response (``choices[0].logprobs.content``).

    Each item is ``{token, logprob, top_logprobs:[{token, logprob}, ...]}``. Empty list if the
    server didn't return logprobs (so callers can fall back to a verdict-based score, never None).
    """
    try:
        return resp["choices"][0]["logprobs"]["content"] or []
    except (KeyError, TypeError, IndexError):
        return []
