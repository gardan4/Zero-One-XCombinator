"""Shared fab task prompts for instruct SFT and finetuned eval.

The base system prompt is a single swappable string (``DEFAULT_SYSTEM_GENERAL`` or
``ZO_SYSTEM_PROMPT_PATH``). Per-task instructions sit below it. Zero-shot eval appends a
rules/grammar digest on top via ``zo_eval.rules_context`` — finetuned models rely on SFT
for process logic and only get this base stack.

Set ``ZO_PROMPT_LEGACY=1`` to disable the system role (user-only prompts for old checkpoints).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from zo_train.datagen import ALL_RULES, SEP

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
(roughly 120 steps). Inputs and outputs use " | " between step names.

Product families MOSFET, IGBT, and IC share a backbone flow but differ in early preparation blocks.
Each message is exactly one graded task — follow only that task's instructions below.
"""

_TASK_NEXTSTEP = """\
TASK — Next-step prediction

Input: product family + partial sequence (steps already executed, in order).
Output: the single step that should come next, plus up to four ranked alternates (ranks 2–5).

Reply with one line only: up to 5 exact step names from the fab vocabulary, separated by " | ".
Do not explain, number ranks, or repeat steps already in the partial sequence.
"""

_TASK_COMPLETION = """\
TASK — Sequence completion

Input: product family + partial sequence (a prefix of a full route).
Output: only the remaining suffix steps — everything that still needs to happen after the prefix.

Reply with one line only: suffix step names in execution order, separated by " | ".
Do not repeat prefix steps or add commentary.
"""

_RULE_IDS = ", ".join(ALL_RULES)

_TASK_ANOMALY = f"""\
TASK — Anomaly detection

Input: product family + full process sequence from start to finish.
Output: whether the sequence is valid or violates exactly one forbidden process-logic rule.

Reply with exactly one line:
VALID.
or
INVALID. RULE_<ID>

Allowed rule IDs when INVALID (use exactly one): {_RULE_IDS}.
Do not invent rule IDs or mark INVALID for missing optional steps alone.
"""

_TASKS: dict[str, str] = {
    "nextstep": _TASK_NEXTSTEP,
    "completion": _TASK_COMPLETION,
    "anomaly": _TASK_ANOMALY,
}


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


def build_base_system(task: str) -> str:
    """``GENERAL + TASK[task]`` — shared by finetuned eval and instruct SFT."""
    task = normalize_task(task)
    return f"{load_system_general().rstrip()}\n\n{_TASKS[task].rstrip()}"


def build_sft_user(task: str, item: PromptItem | ValidInput | AnomalyInput) -> str:
    """User message aligned with ``datagen.*_example`` prompt text (train/eval parity)."""
    task = normalize_task(task)
    family = getattr(item, "family", "MOSFET")
    if task == "nextstep":
        prefix = SEP.join(item.partial_sequence)  # type: ignore[union-attr]
        return (
            f"Product family: {family}\n"
            f"Process so far: {prefix}\n\n"
            f"Next process step?"
        )
    if task == "completion":
        prefix = SEP.join(item.partial_sequence)  # type: ignore[union-attr]
        return (
            f"Product family: {family}\n"
            f"Partial process sequence: {prefix}\n\n"
            f"Complete the remaining steps in order:"
        )
    seq = SEP.join(item.sequence)  # type: ignore[union-attr]
    return (
        f"Product family: {family}\n"
        f"Process sequence: {seq}\n\n"
        f"Is this a valid process sequence? If not, name the violated rule."
    )


def build_messages(
    task: str,
    item: PromptItem | ValidInput | AnomalyInput,
    *,
    system_extra: str | None = None,
) -> list[dict[str, str]]:
    """System + user chat messages for finetuned eval / instruct SFT (no assistant role)."""
    system = build_base_system(task)
    if system_extra:
        system = f"{system.rstrip()}\n\n{system_extra.strip()}"
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
