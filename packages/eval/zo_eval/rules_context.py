"""Rules-in-context prompt assembly for the zero-shot LLM baseline.

Uses the shared base system prompt from ``zo_train.prompts`` plus a task-specific digest
(grammar, allowed steps, forbidden rules) that finetuned models are expected to learn from data.

Snapshots: ``prompt_snapshots/zeroshot_rules_v5.md``, ``zeroshot_rules_v7.md``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from zo_train.datagen import ALL_RULES, SEP
from zo_train.fab import all_steps, default_data_dir
from zo_train.prompts import build_base_system, load_system_general

from zo_eval.submission import AnomalyInput, ValidInput

_GRAMMAR_NOTE = """\
Grammar note: labels like PREFIX, PASSIVATION_BLOCK, TEST_SUITE, and SUFFIX name phases in the rules
doc — they are not fab steps. Predictions must use concrete step names from the allowed list.
"""

_PREDICTION_MISTAKES = """\
Submission-line reminders (next-step and completion):
- Use only names from the allowed step list — copy spelling exactly.
- Do not put grammar phase labels (…_BLOCK, TEST_SUITE, SUFFIX, PREFIX) in the submission line.
- RULE_* identifiers are anomaly violation labels, not process steps — never use them as predictions.
"""

_ANOMALY_GUIDANCE = """\
Anomaly guidance:
1. Default: VALID unless a listed forbidden rule clearly applies.
2. Most sequences are VALID — answer INVALID only when one specific rule matches.
3. If INVALID, cite exactly one rule ID from the forbidden list (not a fab step name).
"""

_FAMILY_PREP = {
    "MOSFET": (
        "MOSFET family prep: SUBSTRATE CHECK → EPITAXY PREP → EPITAXIAL DEPOSITION → "
        "MEASURE EPITAXY THICKNESS → MEASURE RESISTIVITY → EPITAXY ANNEAL → WAFER SURFACE CLEAN."
    ),
    "IGBT": (
        "IGBT family prep: EPITAXIAL WAFER CHECK → MEASURE EPITAXY THICKNESS → MEASURE RESISTIVITY → "
        "[EPITAXIAL REWORK CHECK] → EPITAXIAL LAYER PREP. Uses 6 litho levels in process cycles."
    ),
    "IC": (
        "IC family prep: WAFER CLEAN PRE-GRIND → GRINDING WAFER BACKSIDE → MEASURE GEOMETRY → "
        "ETCH WET BACKSIDE → RINSE → DRY WAFER BACKSIDE → BACKSIDE CLEAN → MEASURE BACKSIDE ROUGHNESS. "
        "Backside grind happens early; BACKSIDE_BLOCK covers final clean/protection only."
    ),
}

_LITHO_TEMPLATE = """\
Lithography block template (every litho cycle):
SPIN COAT PHOTORESIST → SOFT BAKE → ALIGN MASK LEVEL N → EXPOSE LITHO LEVEL N →
[POST EXPOSE BAKE] → DEVELOP PHOTORESIST → pattern inspection → [HARD BAKE] → etch → STRIP RESIST →
CLEAN AFTER ETCH (mandatory after etch before next deposition).
"""

_BACKBONE = """\
Shared process backbone (all families):
PREFIX → PRE_PROCESS_CLEAN → FAMILY_SPECIFIC_PREP → FIRST_OXIDATION → PROCESS_CYCLES
→ ILD_BLOCK → VIA_BLOCK → METAL_BLOCK → PASSIVATION_BLOCK → BACKSIDE_BLOCK
→ FINAL_INSPECTION → TEST_SUITE (includes WAFER SORT TEST before ship) → SUFFIX (LOT RELEASE → SHIP LOT).
"""


def generation_rules_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_data_dir()) / "generation_rules.md"


def load_generation_rules(data_dir: Path | None = None) -> str:
    path = generation_rules_path(data_dir)
    return path.read_text(encoding="utf-8")


def _extract_between(text: str, start: str, end: str | None) -> str:
    """Return markdown between ``## start`` header and next ``##`` at same level (or EOF)."""
    pat_start = re.compile(rf"^##\s+{re.escape(start)}\s*$", re.MULTILINE)
    m = pat_start.search(text)
    if not m:
        return ""
    start_idx = m.end()
    if end:
        pat_end = re.compile(rf"^##\s+{re.escape(end)}\s*$", re.MULTILINE)
        m_end = pat_end.search(text, start_idx)
        chunk = text[start_idx : m_end.start()] if m_end else text[start_idx:]
    else:
        chunk = text[start_idx:]
    return chunk.strip()


def _condense_forbidden(section: str) -> str:
    """One line per RULE_* from section 3."""
    lines: list[str] = ["Forbidden patterns (check the sequence against these only):"]
    for rule in ALL_RULES:
        pat = re.compile(rf"###\s+{re.escape(rule)}\s*\n\*\*(.+?)\*\*", re.DOTALL)
        m = pat.search(section)
        if m:
            desc = re.sub(r"\s+", " ", m.group(1)).strip()
            lines.append(f"- {rule}: {desc}")
        else:
            lines.append(f"- {rule}")
    lines.append(f"Allowed rule IDs for INVALID answers: {', '.join(ALL_RULES)}")
    return "\n".join(lines)


@lru_cache(maxsize=8)
def build_step_vocab_digest(family: str) -> str:
    """All allowed fab steps for *family* (from corpus + canonical recipe)."""
    fam = (family or "").strip().upper()
    steps = sorted(all_steps(fam))
    if not steps:
        return ""
    body = "\n".join(f"- {s}" for s in steps)
    return (
        f"Allowed fab step names for {fam} ({len(steps)} steps — predictions must use these exact names):\n"
        f"{body}"
    )


@lru_cache(maxsize=8)
def build_grammar_digest(family: str | None = None) -> str:
    """Process grammar reference shared by all tasks (no forbidden-pattern list)."""
    parts = [
        "Process grammar reference:",
        _GRAMMAR_NOTE,
        "",
        _BACKBONE,
        "",
        _LITHO_TEMPLATE,
        "",
    ]
    fam = (family or "").strip().upper()
    if fam in _FAMILY_PREP:
        parts.extend([f"Active product family: {fam}", _FAMILY_PREP[fam], ""])
    else:
        parts.extend(
            [
                "Family-specific prep blocks:",
                _FAMILY_PREP["MOSFET"],
                _FAMILY_PREP["IGBT"],
                _FAMILY_PREP["IC"],
                "",
            ]
        )
    return "\n".join(parts)


@lru_cache(maxsize=1)
def build_anomaly_rules_digest() -> str:
    """Forbidden-pattern rules — included only on the anomaly task."""
    text = load_generation_rules()
    forbidden_raw = _extract_between(text, "3. Forbidden Patterns", "4. Variation Axes")
    return _condense_forbidden(forbidden_raw)


@lru_cache(maxsize=8)
def build_rules_digest(family: str | None = None) -> str:
    """Full digest for tests / debugging (grammar + vocab + anomaly rules)."""
    fam = (family or "MOSFET").upper()
    return "\n\n".join(
        [build_grammar_digest(fam), build_step_vocab_digest(fam), build_anomaly_rules_digest()]
    )


def build_zeroshot_extra(family: str, task: str) -> str:
    """Rules/grammar context appended for zero-shot only (not used in instruct SFT)."""
    task = task.strip().lower()
    fam = (family or "MOSFET").upper()
    parts = [build_grammar_digest(fam)]
    if task in ("nextstep", "completion"):
        vocab = build_step_vocab_digest(fam)
        if vocab:
            parts.extend([vocab, _PREDICTION_MISTAKES])
    if task == "anomaly":
        parts.extend([_ANOMALY_GUIDANCE, build_anomaly_rules_digest()])
    return "\n\n".join(p for p in parts if p)


def _system_content(family: str, task: str) -> str:
    task = task.strip().lower()
    extra = build_zeroshot_extra(family, task)
    return f"{build_base_system(task).rstrip()}\n\n{extra}"


def _user_nextstep(item: ValidInput) -> str:
    prefix = SEP.join(item.partial_sequence)
    last = item.partial_sequence[-1] if item.partial_sequence else "?"
    return (
        f"Product family: {item.family}\n"
        f"Partial sequence:\n{prefix}\n\n"
        f"What is the next allowed step after \"{last}\"?\n"
        "Submission line: up to 5 ranked step names from the allowed list, separated by \" | \"."
    )


def _user_completion(item: ValidInput) -> str:
    prefix = SEP.join(item.partial_sequence)
    return (
        f"Product family: {item.family}\n"
        f"Partial sequence:\n{prefix}\n\n"
        "Complete the route.\n"
        "Submission line: remaining suffix steps from the allowed list, separated by \" | \"."
    )


def _user_anomaly(item: AnomalyInput) -> str:
    seq = SEP.join(item.sequence)
    return (
        f"Product family: {item.family}\n"
        f"Full sequence:\n{seq}\n\n"
        "Classify valid vs forbidden-rule violation.\n"
        "Submission line: VALID. or INVALID. RULE_<ID>"
    )


def build_messages(task: str, item: ValidInput | AnomalyInput) -> list[dict[str, str]]:
    """Zero-shot: shared base system prompt + rules digest + zeroshot user framing."""
    family = getattr(item, "family", "MOSFET")
    task = task.strip().lower()
    if task == "nextstep":
        user = _user_nextstep(item)  # type: ignore[arg-type]
    elif task == "completion":
        user = _user_completion(item)  # type: ignore[arg-type]
    elif task == "anomaly":
        user = _user_anomaly(item)  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown task {task!r}")
    return [
        {"role": "system", "content": _system_content(family, task)},
        {"role": "user", "content": user},
    ]


# Re-export for tests that patch or introspect the shared base prompt.
__all__ = [
    "build_anomaly_rules_digest",
    "build_grammar_digest",
    "build_messages",
    "build_rules_digest",
    "build_step_vocab_digest",
    "build_zeroshot_extra",
    "generation_rules_path",
    "load_generation_rules",
    "load_system_general",
]
