"""Minimal OpenAI-compatible chat client (httpx).

Points at ZO_MODEL_BASE_URL — typically a vLLM server hosting your fine-tuned model
(`just serve <model>`), but works with any OpenAI-style endpoint. Shared by eval + agent.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


def _base_url() -> str:
    return os.environ.get("ZO_MODEL_BASE_URL", "http://localhost:8001/v1")


def _api_key() -> str:
    return os.environ.get("ZO_MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"


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
    headers = {"Authorization": f"Bearer {api_key or _api_key()}"}
    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def content(resp: dict[str, Any]) -> str:
    return resp["choices"][0]["message"].get("content") or ""


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
