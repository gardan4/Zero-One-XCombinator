"""Rules-in-context prompt assembly for the zero-shot LLM baseline.

Extracts a condensed digest from ``generation_rules.md`` (no hand-maintained copy) and
builds system+user chat messages for the three track tasks.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from zo_train.datagen import ALL_RULES, SEP
from zo_train.fab import default_data_dir

from zo_eval.submission import AnomalyInput, ValidInput

_OUTPUT_CONTRACT = """\
Output format (strict):
- Use EXACT uppercase step names as they appear in fab sequences.
- Join multiple steps with " | " (space-pipe-space).
- Next-step: reply with up to 5 ranked candidates, most likely first, pipe-separated.
- Completion: reply with remaining steps only, in order, pipe-separated.
- Anomaly: reply with "VALID." or "INVALID. RULE_<ID>" using one of the 10 rule IDs below.
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
PREFIX (RECEIVE WAFER LOT → LOT IDENTIFICATION → inspection)
→ initial measurements → PRE_PROCESS_CLEAN (RCA1/RCA2/HF DIP)
→ FAMILY_SPECIFIC_PREP → FIRST_OXIDATION (THERMAL OXIDATION)
→ PROCESS_CYCLES (3–6 litho–etch–implant cycles) → ILD_BLOCK → VIA_BLOCK → METAL_BLOCK
→ PASSIVATION_BLOCK (DEPOSIT PASSIVATION → CURE PASSIVATION → pad window litho/etch)
→ BACKSIDE_BLOCK → FINAL_INSPECTION → TEST_SUITE (must include WAFER SORT TEST before ship)
→ SUFFIX (LOT RELEASE → SHIP LOT).
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
    lines: list[str] = ["Forbidden patterns (anomaly detection):"]
    for rule in ALL_RULES:
        pat = re.compile(rf"###\s+{re.escape(rule)}\s*\n\*\*(.+?)\*\*", re.DOTALL)
        m = pat.search(section)
        if m:
            desc = re.sub(r"\s+", " ", m.group(1)).strip()
            lines.append(f"- {rule}: {desc}")
        else:
            lines.append(f"- {rule}")
    return "\n".join(lines)


@lru_cache(maxsize=8)
def build_rules_digest(family: str | None = None) -> str:
    """Condensed rules digest (~1.5–2.5k tokens) from vendored ``generation_rules.md``."""
    text = load_generation_rules()
    forbidden_raw = _extract_between(text, "3. Forbidden Patterns", "4. Variation Axes")
    parts = [
        "Semiconductor fab process sequence rules (Infineon track).",
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
    parts.append(_condense_forbidden(forbidden_raw))
    parts.extend(["", _OUTPUT_CONTRACT])
    return "\n".join(parts)


def _system_content(family: str) -> str:
    return (
        "You are a semiconductor fabrication process expert. "
        "Apply the process grammar and forbidden-pattern rules below exactly.\n\n"
        + build_rules_digest(family)
    )


def _user_nextstep(item: ValidInput) -> str:
    prefix = SEP.join(item.partial_sequence)
    return (
        f"Product family: {item.family}\n"
        f"Process so far: {prefix}\n\n"
        "What is the next process step? "
        "Reply with up to 5 ranked candidate step names (exact uppercase), separated by |."
    )


def _user_completion(item: ValidInput) -> str:
    prefix = SEP.join(item.partial_sequence)
    return (
        f"Product family: {item.family}\n"
        f"Partial process sequence: {prefix}\n\n"
        "Complete the remaining steps in order. "
        "Reply with only the steps after the partial sequence, pipe-separated ( | )."
    )


def _user_anomaly(item: AnomalyInput) -> str:
    seq = SEP.join(item.sequence)
    rules_list = ", ".join(ALL_RULES)
    return (
        f"Product family: {item.family}\n"
        f"Process sequence: {seq}\n\n"
        "Is this a valid process sequence? If invalid, name the violated rule.\n"
        f"Valid rule IDs: {rules_list}\n"
        'Reply "VALID." or "INVALID. RULE_<ID>".'
    )


def build_messages(task: str, item: ValidInput | AnomalyInput) -> list[dict[str, str]]:
    """Return ``[{role: system}, {role: user}]`` for *task* in nextstep|completion|anomaly."""
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
        {"role": "system", "content": _system_content(family)},
        {"role": "user", "content": user},
    ]
