"""Model-agnostic prediction layer for the 3 track tasks + the output normalizer.

A ``Predictor`` turns organizer-format eval inputs (``submission.ValidInput`` / ``AnomalyInput``)
into raw predictions; the ``track`` driver writes those to submission CSVs and scores them. The same
interface backs the n-gram/oracle baselines (``baselines.py``), the served LLM (``predict_llm.py``),
and any future model — so the whole pipeline is written once.

**The normalizer is the load-bearing part.** Model output is free text ("the next step is to
develop the photoresist") but the eval vocabulary is exact strings ("DEVELOP PHOTORESIST"). ``snap``
maps free text onto the exact vocab; without it Top-1 ≈ 0 even from a good model.

Separator note: training prompts join steps with ``" | "`` (``datagen.SEP``); submission/eval CSVs
use ``"|"`` (``submission.STEP_SEP``). Build prompts with ``" | "``; parse/write with ``"|"``.
``snap(..., strict=False)`` passes unknown-but-plausible tokens through — needed for the hidden OOD
family, where the correct step may not be in our known vocab.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from zo_train.datagen import ALL_RULES
from zo_train.fab import FAMILIES, all_steps

from zo_eval.submission import AnomalyInput, ValidInput
from zo_eval.track_metrics import _levenshtein

_RULES = set(ALL_RULES)


# --------------------------------------------------------------------------- vocabulary


_VOCAB: set[str] | None = None


def vocab() -> set[str]:
    """The exact step vocabulary (canonical ∪ observed across all families). Cached."""
    global _VOCAB
    if _VOCAB is None:
        steps: set[str] = set()
        for fam in FAMILIES:
            steps |= all_steps(fam)
        _VOCAB = steps
    return _VOCAB


def normalize_family(fam: str) -> str:
    """Map an eval-input family string to our canonical family token (case-insensitive)."""
    up = (fam or "").strip().upper()
    return up if up in FAMILIES else up


# --------------------------------------------------------------------------- normalizer

_LEAD_NOISE = re.compile(r"^[\s\-\*•\d\.\)\:]+")
_STEP_PREFIX = re.compile(r"(?i)^(the\s+)?(next\s+)?step\s*(is|would be|:)?\s*")


def _clean(text: str) -> str:
    t = (text or "").strip()
    t = _STEP_PREFIX.sub("", t)
    t = t.strip().strip("\"'`").strip()
    t = _LEAD_NOISE.sub("", t)
    return t.strip().strip("\"'`.").strip()


def snap(text: str, vocab_set: set[str] | None = None, strict: bool = True, max_word_edits: int = 2) -> str | None:
    """Map free text → the nearest exact vocab step.

    Order: exact → longest contained vocab step (handles verbose output) → nearest by word-level
    edit distance (≤ ``max_word_edits``). ``strict=True`` returns ``None`` if nothing is close;
    ``strict=False`` returns the cleaned, upper-cased text (pass-through for OOD novel steps).
    """
    vocab_set = vocab_set or vocab()
    raw = _clean(text)
    if not raw:
        return None
    up = raw.upper()
    if up in vocab_set:
        return up
    contained = [s for s in vocab_set if s in up]
    if contained:
        return max(contained, key=len)
    if not strict:
        return up  # OOD: preserve a novel-but-plausible step rather than mis-snapping it
    up_words = up.split()
    best, best_d = None, 10**9
    for s in vocab_set:
        d = _levenshtein(up_words, s.split())
        if d < best_d:
            best, best_d = s, d
    return best if best_d <= max_word_edits else None


def parse_pipe_list(text: str, vocab_set: set[str] | None = None, strict: bool = True) -> list[str]:
    """Split a model's step list on ``|`` / newlines / numbering, snap each, drop misses."""
    vocab_set = vocab_set or vocab()
    out: list[str] = []
    for part in re.split(r"\s*\|\s*|\n+", text or ""):
        s = snap(part, vocab_set, strict=strict)
        if s:
            out.append(s)
    return out


def extract_answer(text: str) -> str:
    """Strip a ``<think>…</think>`` rationale; return the post-'Answer:' text or the last line."""
    t = re.sub(r"(?is)<think>.*?</think>", "", text or "")
    m = re.search(r"(?is)(?:final\s+answer|answer)\s*:\s*(.+)$", t)
    if m:
        return m.group(1).strip()
    lines = [ln for ln in t.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else t.strip()


def parse_anomaly(text: str) -> tuple[int, str | None]:
    """Free text → (is_valid 1/0, predicted_rule or None). 'INVALID'/'not valid' ⇒ 0."""
    up = (text or "").upper()
    invalid = bool(re.search(r"\bINVALID\b|\bNOT\s+VALID\b", up))
    m = re.search(r"RULE_[A-Z_]+", up)
    rule = m.group() if (m and m.group() in _RULES) else None
    return (0 if invalid else 1, rule)


# --------------------------------------------------------------------------- interface


@runtime_checkable
class Predictor(Protocol):
    """Every model/baseline implements this; the ``track`` driver consumes it."""

    name: str

    def next_step(self, item: ValidInput) -> list[str]:
        """≤5 ranked next-step candidates (normalized)."""
        ...

    def complete(self, item: ValidInput) -> list[str]:
        """The steps AFTER the cut (normalized); excludes the given partial."""
        ...

    def anomaly(self, item: AnomalyInput) -> tuple[int, float | None, str | None]:
        """(is_valid 1/0, score=P(valid)∈[0,1] or None, predicted_rule or None)."""
        ...


def _smoke() -> None:  # pragma: no cover - manual
    v = vocab()
    assert "DEVELOP PHOTORESIST" in v and "OXIDE ETCH DRY" in v
    cases = {
        "The next step is to DEVELOP PHOTORESIST.": "DEVELOP PHOTORESIST",
        "develop photoresist": "DEVELOP PHOTORESIST",
        "- OXIDE ETCH DRY": "OXIDE ETCH DRY",
        "1. SPIN COAT PHOTORESIST": "SPIN COAT PHOTORESIST",
        '"SHIP LOT"': "SHIP LOT",
    }
    for text, want in cases.items():
        got = snap(text)
        assert got == want, f"snap({text!r}) = {got!r}, want {want!r}"
    pl = parse_pipe_list("SPIN COAT PHOTORESIST | SOFT BAKE | ALIGN MASK LEVEL 1")
    assert pl == ["SPIN COAT PHOTORESIST", "SOFT BAKE", "ALIGN MASK LEVEL 1"], pl
    assert parse_anomaly("INVALID. RULE_DEP_NO_CLEAN at step 5") == (0, "RULE_DEP_NO_CLEAN")
    assert parse_anomaly("This sequence is VALID.") == (1, None)
    assert extract_answer("<think>reasoning here</think>\nAnswer: SHIP LOT") == "SHIP LOT"
    assert snap("ZZZ QQQ WWW", strict=True) is None  # far from all vocab → no match
    # OOD: a near-but-novel step passes through lenient mode instead of mis-snapping
    assert snap("DEPOSIT UNOBTANIUM LAYER", strict=False) == "DEPOSIT UNOBTANIUM LAYER"
    print(f"predict.py smoke OK — vocab={len(v)} steps")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
