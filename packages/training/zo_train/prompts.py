"""Shared fab task prompts for instruct SFT, finetuned eval, and zero-shot eval.

Single source of truth: JSON output schema, numbered user input, behavioral guidance.
Optional rules/grammar digest is appended only when ``system_extra`` is passed (e.g.
``ZO_RULES_IN_CONTEXT=1`` via ``zo_eval.rules_context``).

Set ``ZO_PROMPT_LEGACY=1`` to disable the system role (user-only prompts for old checkpoints).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from zo_train.datagen import ALL_RULES

if TYPE_CHECKING:
    from zo_eval.submission import AnomalyInput, ValidInput


@dataclass
class PromptItem:
    """Minimal input for prompt builders (avoids zo_eval import from datagen)."""

    family: str
    partial_sequence: list[str] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)
    completion_fraction: float = 0.0


# ---------------------------------------------------------------------------
# Swappable base — edit here or point ZO_SYSTEM_PROMPT_PATH at a text file.
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_GENERAL = """\
You are a semiconductor wafer fabrication process-sequence assistant (Infineon Industrial AI track).

Process routes are ordered lists of concrete fab step names from a fixed uppercase vocabulary
(roughly 120 steps). User messages list steps in numbered execution order; you reply with one
JSON object per task (see OUTPUT FORMAT below).

Product families MOSFET, IGBT, and IC share a backbone flow but differ in early preparation blocks.
Each message is exactly one graded task — follow only that task's instructions below.
"""

_RULE_IDS = ", ".join(ALL_RULES)

# --- Task definitions --------------------------------------------------------

_TASK_NEXTSTEP = """\
TASK — Next-step prediction

You receive a product family and a partial process sequence (steps already executed, in order).
Predict the single fab step that immediately follows the last step, plus up to four alternates
(ranked best-first). Predict the very next step in the sequence only — not the start of a
later process phase. Use only concrete fab step names from the fixed vocabulary."""

_TASK_COMPLETION = """\
TASK — Sequence completion

You receive a product family and a partial sequence (prefix of a full route).
Predict all remaining suffix steps, in order, to complete the route.
Do not repeat any step from the prefix. Use only concrete fab step names from the fixed vocabulary."""

_TASK_ANOMALY = """\
TASK — Anomaly detection

You receive a product family and a full process sequence (numbered in execution order).
Decide whether the sequence is VALID or violates exactly one forbidden process-logic rule.
Work through your rule checks in "reasoning" first, then set "valid" and "rule"."""

_TASKS: dict[str, str] = {
    "nextstep": _TASK_NEXTSTEP,
    "completion": _TASK_COMPLETION,
    "anomaly": _TASK_ANOMALY,
}

# --- Output format (scorer reads JSON) ---------------------------------------

_OUTPUT_INTRO = """\
OUTPUT FORMAT — mandatory

Reply with one JSON object only (optional ```json fence). Put brief reasoning inside the JSON.
The automated scorer parses only this JSON — ignore any prose outside it."""

_OUTPUT_NEXTSTEP = """\
Schema:
{"reasoning": "<why this step follows the grammar>", "steps": ["BEST", "ALT2", "ALT3"]}

- "steps": array of 1–5 strings, ranked best-first; each string must be an exact fab step name.
- "steps"[0] is the single immediate next step after the last executed step; later entries are alternates for that same position only.
- Do not include steps already present in the partial sequence.
- Do not use grammar phase labels (…_BLOCK, PREFIX, SUFFIX) or RULE_* IDs as steps."""

_OUTPUT_COMPLETION = """\
Schema:
{"reasoning": "<brief plan for the suffix>", "steps": ["STEP1", "STEP2", "..."]}

- "steps": array of suffix steps only (after the prefix), in order; exact fab step names.
- Do not repeat prefix steps. No grammar labels or RULE_* IDs."""

_OUTPUT_ANOMALY = f"""\
Schema when valid:
{{"reasoning": "<your rule-by-rule analysis>", "valid": true, "rule": null}}

Schema when invalid:
{{"reasoning": "<your rule-by-rule analysis>", "valid": false, "rule": "RULE_<ID>"}}

- Write "reasoning" first: briefly walk through the forbidden rules that matter, then conclude valid or invalid.
- After reasoning, set "valid" and "rule" to match your conclusion.
- "rule": exactly one ID when valid is false; null when valid is true.
- Allowed rule IDs: {_RULE_IDS}."""

_OUTPUT_BY_TASK: dict[str, str] = {
    "nextstep": _OUTPUT_NEXTSTEP,
    "completion": _OUTPUT_COMPLETION,
    "anomaly": _OUTPUT_ANOMALY,
}

# --- Behavioral guidance (not full rule text) --------------------------------

_PREDICTION_MISTAKES = """\
Common mistakes to avoid:
- Skipping ahead to a later process phase instead of the very next step.
- Using grammar phase labels (…_BLOCK, TEST_SUITE, SUFFIX, PREFIX) as step predictions.
- Using RULE_* identifiers as steps — those label anomaly violations, not fab steps.
- Misspelling step names — copy exactly from the fab vocabulary."""

_ANOMALY_GUIDANCE = """\
Anomaly decision rules:
1. RULE_* IDs name violation patterns — a sequence is valid when that pattern does not apply.
2. Check the sequence against each forbidden process-logic rule — consider each one carefully.
3. If any forbidden rule applies, set valid=false and "rule" to that rule ID.
4. Set valid=true only when none of the forbidden rules apply.
5. When valid=false, set "rule" to exactly one rule ID (not a step name).
6. Reason first in "reasoning", then set valid and rule to match."""


def load_system_general() -> str:
    """Return the shared base system prompt (file override or default constant)."""
    path = os.environ.get("ZO_SYSTEM_PROMPT_PATH", "").strip()
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_GENERAL


def use_system_prompt() -> bool:
    """False when ``ZO_PROMPT_LEGACY=1`` (user-only prompts for pre-system checkpoints)."""
    return os.environ.get("ZO_PROMPT_LEGACY", "").strip().lower() not in ("1", "true", "yes")


def normalize_task(task: str) -> str:
    task = task.strip().lower()
    if task not in _TASKS:
        raise ValueError(f"unknown task {task!r}; expected nextstep|completion|anomaly")
    return task


def _guidance_block(task: str) -> str:
    task = normalize_task(task)
    if task in ("nextstep", "completion"):
        return _PREDICTION_MISTAKES
    if task == "anomaly":
        return _ANOMALY_GUIDANCE
    return ""


def build_system(task: str, *, rules_extra: str | None = None) -> str:
    """Full system prompt: role + task + JSON schema + guidance (+ optional rules digest)."""
    task = normalize_task(task)
    parts = [
        load_system_general().rstrip(),
        _TASKS[task],
        _OUTPUT_INTRO,
        _OUTPUT_BY_TASK[task],
        _guidance_block(task),
    ]
    if rules_extra:
        parts.append(rules_extra.strip())
    return "\n\n".join(p for p in parts if p)


def build_base_system(task: str) -> str:
    """Alias for ``build_system`` without rules digest — used in tests and legacy callers."""
    return build_system(task)


def _format_numbered_steps(steps: list[str]) -> str:
    """One step per line with 1-based index (execution order)."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def build_sft_user(task: str, item: PromptItem | ValidInput | AnomalyInput) -> str:
    """Numbered user message aligned with ``datagen.*_example`` prompt text (train/eval parity)."""
    task = normalize_task(task)
    family = getattr(item, "family", "MOSFET")
    if task == "nextstep":
        steps = list(item.partial_sequence)  # type: ignore[union-attr]
        numbered = _format_numbered_steps(steps)
        n = len(steps)
        last = steps[-1] if steps else "?"
        return (
            f"Product family: {family}\n"
            f"Partial sequence (numbered in execution order):\n{numbered}\n\n"
            f'Last executed step (#{n}): "{last}"\n'
            "Respond with the JSON object described in OUTPUT FORMAT."
        )
    if task == "completion":
        numbered = _format_numbered_steps(list(item.partial_sequence))  # type: ignore[union-attr]
        return (
            f"Product family: {family}\n"
            f"Partial sequence (prefix, numbered in execution order):\n{numbered}\n\n"
            "Respond with the JSON object described in OUTPUT FORMAT (suffix steps only)."
        )
    numbered = _format_numbered_steps(list(item.sequence))  # type: ignore[union-attr]
    return (
        f"Product family: {family}\n"
        f"Full sequence (numbered in execution order):\n{numbered}\n\n"
        "Reason through your analysis in JSON first, then give valid and rule.\n"
        "Respond with the JSON object described in OUTPUT FORMAT."
    )


def build_json_completion_nextstep(steps: str | list[str], *, reasoning: str = "") -> str:
    """JSON assistant label for next-step SFT/eval."""
    step_list = [steps] if isinstance(steps, str) else list(steps)
    return json.dumps({"reasoning": reasoning, "steps": step_list}, ensure_ascii=False)


def build_json_completion_completion(steps: list[str], *, reasoning: str = "") -> str:
    """JSON assistant label for sequence-completion SFT/eval."""
    return json.dumps({"reasoning": reasoning, "steps": list(steps)}, ensure_ascii=False)


def build_json_completion_anomaly(
    is_valid: bool,
    rule: str | None = None,
    *,
    reasoning: str | None = None,
) -> str:
    """JSON assistant label for anomaly SFT/eval."""
    if is_valid:
        text = reasoning or "no forbidden rule applies"
        return json.dumps({"reasoning": text, "valid": True, "rule": None}, ensure_ascii=False)
    rule_id = (rule or "RULE_UNKNOWN").strip().upper()
    text = reasoning or f"violates {rule_id}"
    return json.dumps({"reasoning": text, "valid": False, "rule": rule_id}, ensure_ascii=False)


def build_messages(
    task: str,
    item: PromptItem | ValidInput | AnomalyInput,
    *,
    system_extra: str | None = None,
) -> list[dict[str, str]]:
    """System + user chat messages for SFT, finetuned eval, and zero-shot (no assistant role)."""
    system = build_system(task, rules_extra=system_extra)
    user = build_sft_user(task, item)
    if not use_system_prompt():
        return [{"role": "user", "content": user}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_instruct_messages(
    task: str,
    item: PromptItem | ValidInput | AnomalyInput,
    completion: str,
    *,
    system_extra: str | None = None,
) -> list[dict[str, str]]:
    """Full SFT row: system + user + assistant completion."""
    msgs = build_messages(task, item, system_extra=system_extra)
    msgs.append({"role": "assistant", "content": completion})
    return msgs
