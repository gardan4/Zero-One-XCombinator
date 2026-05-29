from __future__ import annotations

import json
from typing import Any

from zo_common.llm import chat

from zo_agent.tools import dispatch, specs


def run_episode(
    system: str | None,
    task: str,
    model: str,
    base_url: str | None = None,
    max_steps: int = 6,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """One tool-using episode. Returns final answer + a structured trace."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": task})

    trace: list[dict[str, Any]] = []
    tool_calls = 0
    final = ""
    step = 0
    for step in range(max_steps):
        resp = chat(
            messages,
            model=model,
            tools=specs(),
            tool_choice="auto",
            base_url=base_url,
            temperature=temperature,
        )
        msg = resp["choices"][0]["message"]
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = msg.get("tool_calls") or []
        if not calls:
            final = msg.get("content") or ""
            break
        for tc in calls:
            tool_calls += 1
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = dispatch(fn, args)
            trace.append({"step": step, "tool": fn, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})

    return {"final": final, "steps": step + 1, "tool_calls": tool_calls, "trace": trace}
