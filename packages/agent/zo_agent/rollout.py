from __future__ import annotations

import json
import re
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
    """One tool-using episode. Returns final answer + a structured trace.

    Autonomous path: the model decides when to call tools. Small models are unreliable at emitting
    OpenAI ``tool_calls``, so for the deterministic demo prefer :func:`run_episode_scripted`.
    """
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


# --------------------------------------------------------------------------- scripted demo path


_FAMILY_RE = re.compile(r"\b(MOSFET|IGBT|IC)\b", re.IGNORECASE)


def extract_recipe(task: str) -> tuple[str, str]:
    """Pull the (pipe-joined recipe, family) out of a repair task string.

    The recipe is the longest ``|``-joined run in the task (steps never contain ``|``); the family
    is the first MOSFET/IGBT/IC token if present, else "". Robust to phrasing like
    ``"Diagnose and fix this fab recipe: A | B | C"``.
    """
    fam_m = _FAMILY_RE.search(task or "")
    family = fam_m.group(1).upper() if fam_m else ""
    # Candidate recipes: any substring containing at least one pipe; pick the one with the most
    # pipe-delimited parts. Splitting the whole task on newlines first avoids swallowing prose.
    best = ""
    for chunk in re.split(r"\n+", task or ""):
        if "|" in chunk:
            # Trim a leading "label:" prefix (e.g. "recipe:") from the first piped chunk.
            cand = chunk.split(":", 1)[-1] if ":" in chunk.split("|", 1)[0] else chunk
            if cand.count("|") >= best.count("|"):
                best = cand
    return best.strip(), family


def run_episode_scripted(
    system: str | None,
    task: str,
    model: str,
    base_url: str | None = None,
    max_steps: int = 6,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Deterministic diagnose → explain → repair → re-validate loop for the repair copilot.

    Unlike :func:`run_episode`, the control flow is code-driven (no reliance on the model emitting
    ``tool_calls``); the model/stub is used **only** inside ``suggest_repair``. Returns the same
    trace dict shape ``{final, steps, tool_calls, trace}`` so it drops straight into ``run_suite``.

    ``system``/``model``/``base_url``/``temperature`` are accepted for signature parity with
    ``run_episode``; the repair brain reads ``ZO_MODEL_BASE_URL`` (or an injected stub via
    ``tools.set_repair_fn``), so they're currently informational here.
    """
    recipe, family = extract_recipe(task)
    trace: list[dict[str, Any]] = []
    tool_calls = 0

    def _call(step: int, fn: str, args: dict[str, Any]) -> str:
        nonlocal tool_calls
        tool_calls += 1
        result = dispatch(fn, args)
        trace.append({"step": step, "tool": fn, "args": args, "result": result})
        return result

    # 1) diagnose
    verdict = _call(0, "validate_recipe", {"sequence": recipe})
    if verdict == "VALID":
        # Already valid — echo it back as the answer (still counts under the "validates" judge).
        return {"final": recipe, "steps": 1, "tool_calls": tool_calls, "trace": trace}

    # 2) explain (for the trace / a human-readable diagnosis)
    _call(1, "explain_violation", {"sequence": recipe})

    # 3) repair (verifier-in-the-loop lives inside suggest_repair) + 4) re-validate
    repaired = _call(2, "suggest_repair", {"sequence": recipe, "family": family})
    _call(3, "validate_recipe", {"sequence": repaired})

    return {"final": repaired, "steps": 4, "tool_calls": tool_calls, "trace": trace}
