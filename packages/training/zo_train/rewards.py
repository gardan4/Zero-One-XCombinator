"""Verifiable rewards for GRPO on the Industrial AI (Infineon) track.

The reward **is** the product in RLVR — these functions turn a free-text model completion
into a scalar the GRPO trainer maximizes. We have a *free, perfect verifier*
(``grammar.validate_sequence``), so the reward is grounded in real process logic, not a
learned judge:

    completion text → extract_answer (strip <think>) → parse_pipe_list → validate_sequence
    → dense reward  ``1 - n_violations / n_steps``  (not a brittle 0/1).

Two reward *shapes* share this pipeline — they are the {outcome, process} axis of the GRPO reward
ablation: ``reward_validate`` (outcome — penalize *every* rule break, ``1 - n_viol/n``) and
``reward_process`` (process — credit a *long legal run from the first step*,
``longest_valid_prefix / n``). They diverge when violations cluster: one early break tanks the
process reward but barely dents the outcome reward.

Both rewards are **prefix-aware**: the continue-the-route tasks (completion / next-step) ask the
model to extend a partial route, so a generation is only legal *relative to the steps it follows*.
trl hands the reward the ``prompts`` it sampled from; ``_prefix_from_prompt`` recovers that prefix
and we validate ``prefix + completion`` together (the anti-hack length floor still fires on the
*completion* length). With no prompt the reward scores the completion in isolation — the original
behaviour, kept for the unit tests. Scoring in isolation was the bug behind the first GRPO run: a
1-step next-step answer hit the short-stub floor, every sample in the group got the same ``-0.99``,
so the group std (and the GRPO advantage) was zero and nothing learned.

**Reward hacking is the main risk** (the model finds a degenerate output that scores high
without learning anything). A length-1 sequence like ``RECEIVE WAFER LOT`` is *trivially
valid* — zero violations — so a naive ``1 - n_viol/n`` reward would pay the model to emit
one token forever. We guard against that explicitly:

- **empty / too-short** (``< MIN_STEPS``) completions get a hard floor penalty, scaled so a
  short-but-valid stub can never out-score a genuine long sequence;
- a **small** progress bonus is paid only for advancing toward / reaching ``SHIP LOT`` (the
  canonical sequence terminus), so the model can't win by parroting a single safe step. The
  bonus is deliberately tiny relative to the validity term — it breaks ties, it doesn't drive
  the objective.

Pure stdlib + the vendored grammar; **no torch**. ``parse_pipe_list`` / ``extract_answer`` are
imported lazily from ``zo_eval.predict`` (the single canonical parser/normalizer) so this module
still imports on a laptop and we don't duplicate the separator/normalization logic. The lazy
import also avoids a package-load cycle (``zo_eval`` already depends on ``zo_train``).

trl reward contract (see ``rl.py``)::

    def reward(completions: list[str], **kwargs) -> list[float]

``completions`` may also arrive as chat dicts (``[{"role","content"}, ...]``); ``_text`` flattens
both shapes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from zo_train.grammar import validate_sequence

# Tuning knobs (kept module-level so a config/test can reason about them).
MIN_STEPS = 5  # completions shorter than this are degenerate for our ~100-step sequences
EMPTY_PENALTY = -1.0  # no parseable steps at all
SHORT_PENALTY = -1.0  # parseable but trivially short (anti-hack: blocks the 1-step exploit)
SHIP_BONUS = 0.1  # small tie-breaker for reaching SHIP LOT; never the dominant term
PROGRESS_BONUS = 0.05  # smaller still: advancing into the ship/test tail without finishing
SHIP_LOT = "SHIP LOT"
# Backbone tail markers — being here means the sequence is "advancing toward" ship.
_TAIL_MARKERS = ("WAFER SORT TEST", "FINAL VISUAL INSPECTION", "CURE PASSIVATION")
# Marker *stems* that precede the prefix route in a GRPO prompt (see datagen's
# {completion,nextstep}_example and their *_context_example siblings). We match the stem (no colon)
# and then skip the header punctuation. Supports numbered lines (unified JSON prompts) and legacy
# pipe-joined layouts. Most-specific stem first.
_NUMBERED_STEP = re.compile(r"^\s*\d+\.\s+(.+)$")
_PREFIX_MARKERS = (
    "Partial sequence (prefix, numbered in execution order)",
    "Partial sequence (numbered in execution order)",
    "Partial process sequence",
    "Process so far",
)


def _parse_numbered_block(text: str) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        m = _NUMBERED_STEP.match(line)
        if m:
            steps.append(m.group(1).strip())
        elif steps and line.strip():
            break
    return steps


def _text(completion: object) -> str:
    """Flatten a trl completion (plain string OR a list of chat-message dicts) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, Sequence):
        return "".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in completion)
    return str(completion)


def _steps(completion: object) -> list[str]:
    """completion → cleaned, snapped list of exact-vocab steps (strict; drops free text)."""
    # Lazy import: keeps this module load-cycle-free and laptop-importable.
    from zo_eval.predict import extract_answer, parse_pipe_list

    answer = extract_answer(_text(completion))
    return parse_pipe_list(answer, strict=True)


def _prefix_from_prompt(prompt: object) -> list[str]:
    """Recover the pipe-joined prefix route embedded in a GRPO prompt (``[]`` if none found).

    The continue-the-route prompts place the prefix right after a marker stem (``_PREFIX_MARKERS``),
    either inline (``"Process so far: A | B | C"``) or under a header on the next line
    (``"Process so far (74 steps):\\nA | B | C"``). We find the marker, skip past its header
    punctuation (the first ``:``), then return the first following line that parses to >= 1 step — the
    prefix is always a single SEP-joined line. Validating ``prefix + completion`` lets the reward judge
    a continuation *in the context it must be legal in*. Returns ``[]`` when no marker is present, so
    the reward gracefully falls back to isolation scoring. ``parse_pipe_list`` snaps to the exact step
    vocabulary, so a header line like ``"(74 steps)"`` simply parses to nothing and is skipped.
    """
    if not prompt:
        return []
    from zo_eval.predict import parse_pipe_list

    text = _text(prompt)
    for marker in _PREFIX_MARKERS:
        idx = text.find(marker)
        if idx == -1:
            continue
        after = text[idx + len(marker) :]
        colon = after.find(":")  # skip the header (covers "<marker>:" and "<marker> (N steps):")
        if colon != -1:
            after = after[colon + 1 :]
        block = after.split("\n\n", 1)[0]
        numbered = _parse_numbered_block(block)
        if numbered:
            return numbered
        # Legacy: single SEP-joined line within this block.
        for line in block.splitlines():
            steps = parse_pipe_list(line, strict=True)
            if steps:
                return steps
    return []


def _tail_bonus(steps: list[str]) -> float:
    """Small tie-breaker bonus for a route that reaches (``SHIP LOT``) / approaches the terminus."""
    if SHIP_LOT in steps:
        return SHIP_BONUS  # reached the terminus
    if any(m in steps for m in _TAIL_MARKERS):
        return PROGRESS_BONUS  # advancing into the end-of-line tail
    return 0.0


def _score_one(steps: list[str], prefix: list[str] | None = None) -> float:
    """Dense validity reward for one completion, validated after ``prefix``, with anti-hack guards.

    ``steps`` is the model's completion; ``prefix`` is the route it continues (recovered from the
    prompt). The validity term scores ``prefix + steps`` together, but the anti-hack length floor
    fires on the *completion* length — a short continuation is the degenerate exploit no matter how
    long the given prefix is. ``validate_sequence`` only looks backward, so a legal prefix introduces
    no violations of its own; we count only the violations at/after the prefix boundary and divide by
    the completion length, which keeps the full dynamic range (an all-broken continuation → ~0, not
    ~prefix_len/full_len). With ``prefix=[]`` this is exactly the original ``1 - n_violations/n``.
    """
    n = len(steps)
    if n == 0:
        return EMPTY_PENALTY
    if n < MIN_STEPS:
        # Trivially short — even if perfectly valid, this is the degenerate exploit. Floor it
        # (with a tiny upward nudge by length so the gradient still prefers 4 steps over 1,
        # but it stays well below any non-degenerate sequence's score).
        return SHORT_PENALTY + 0.01 * n

    prefix = prefix or []
    full = prefix + steps
    p = len(prefix)
    # Only violations the continuation introduces (step_index >= p); a clean prefix contributes none,
    # and this stays robust if the recovered prefix is itself imperfect.
    new_violations = sum(1 for v in validate_sequence(full) if v.step_index >= p)
    base = 1.0 - new_violations / n  # dense: every fixed rule helps, not all-or-nothing
    return base + _tail_bonus(full)


def _longest_valid_prefix(steps: list[str]) -> int:
    """Length of the longest leading run of steps that breaks no rule.

    ``validate_sequence`` only ever looks *backward* within fixed windows, so its violation set
    grows monotonically as the prefix grows: ``steps[:k]`` is rule-clean for every ``k`` up to the
    index of the first violating step and dirty thereafter. So the longest valid prefix is simply
    the earliest ``step_index`` among the violations (or the full length when there are none) — one
    ``validate_sequence`` call, no O(n²) prefix sweep.
    """
    violations = validate_sequence(steps)
    if not violations:
        return len(steps)
    return min(v.step_index for v in violations)


def _score_one_process(steps: list[str], prefix: list[str] | None = None) -> float:
    """Process reward for one completion: how far the *continuation* extends the legal run.

    ``_longest_valid_prefix(prefix + steps)`` is the absolute index of the first rule break; we credit
    the part of that legal run contributed by the completion — ``(lvp - len(prefix)) / n`` — so a
    continuation that stays legal to the end scores ~1.0 and one that breaks at the boundary scores ~0.
    Same anti-hack length floor on the completion as ``_score_one`` (a prefix reward is, if anything,
    *more* temptable by a short all-valid stub). With ``prefix=[]`` this reduces to the original
    ``longest_valid_prefix / n``.
    """
    n = len(steps)
    if n == 0:
        return EMPTY_PENALTY
    if n < MIN_STEPS:
        return SHORT_PENALTY + 0.01 * n
    prefix = prefix or []
    full = prefix + steps
    legal_into_completion = max(0, _longest_valid_prefix(full) - len(prefix))
    base = legal_into_completion / n
    return base + _tail_bonus(full)


def _prefixes_for(completions, prompts) -> list[list[str]]:
    """One prefix route per completion (parsed from the paired prompt; ``[]`` when no prompts).

    trl repeats each prompt ``num_generations`` times, so ``len(prompts) == len(completions)``; we
    still pad/truncate to ``len(completions)`` so the downstream ``zip`` is provably aligned (a
    misalignment would pair completions with the wrong prefix — worse than a crash).
    """
    if prompts is None:
        return [[] for _ in completions]
    prefixes = [_prefix_from_prompt(p) for p in prompts]
    if len(prefixes) != len(completions):
        prefixes = (prefixes + [[]] * len(completions))[: len(completions)]
    return prefixes


def reward_validate(completions, prompts=None, **kwargs) -> list[float]:
    """Dense, verifier-grounded reward over a batch of completions (trl contract).

    Prefix-aware: when trl passes the ``prompts`` it sampled from, each completion is validated as
    ``prefix + completion`` (prefix recovered from the prompt), so a continuation is judged in the
    context it must be legal in — not in isolation. Per completion: ``1 - new_violations/n`` for
    non-degenerate outputs; a hard penalty for empty / ``< MIN_STEPS`` completions; a small bonus for
    reaching/approaching ``SHIP LOT``. With no prompts this is the original isolation reward. Range
    is roughly ``[-1, 1.1]``.
    """
    prefixes = _prefixes_for(completions, prompts)
    return [_score_one(_steps(c), pre) for c, pre in zip(completions, prefixes, strict=True)]


def reward_process(completions, prompts=None, **kwargs) -> list[float]:
    """Dense, verifier-grounded PROCESS reward over a batch of completions (trl contract).

    Prefix-aware like ``reward_validate``: scores ``(longest_valid_prefix(prefix+completion) -
    len(prefix)) / n`` — the fraction of the *continuation* that extends the legal run from step 0.
    Same empty / ``< MIN_STEPS`` floor and ``SHIP LOT`` tail bonus. Pairs with ``reward_validate`` as
    the {process, outcome} axis of the GRPO reward ablation. With no prompts this reduces to the
    original ``longest_valid_prefix / n``. Range ≈ ``[-1, 1.1]``.
    """
    prefixes = _prefixes_for(completions, prompts)
    return [
        _score_one_process(_steps(c), pre)
        for c, pre in zip(completions, prefixes, strict=True)
    ]


def _has_clean_think(text: str) -> bool:
    """A single, well-formed ``<think>…</think>`` block with non-empty content, then more text."""
    import re

    m = re.search(r"(?is)<think>(.*?)</think>(.*)", text)
    if not m:
        return False
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False  # malformed / repeated tags → no bonus
    return bool(m.group(1).strip()) and bool(m.group(2).strip())


def reward_format(completions, **kwargs) -> list[float]:
    """Small shaping bonus for a clean ``<think>…</think>`` rationale followed by a parseable answer.

    ``1.0`` only when the completion both (a) has exactly one well-formed think block with a
    non-empty final answer after it, and (b) that answer parses to ≥1 in-vocab step; ``0.0``
    otherwise. Pairs with ``reward_validate`` (``reward: validate+format``) to encourage the
    "think then answer" shape without rewarding the *content* of the rationale (which we can't
    verify).
    """
    out: list[float] = []
    for c in completions:
        text = _text(c)
        ok = _has_clean_think(text) and len(_steps(c)) >= 1
        out.append(1.0 if ok else 0.0)
    return out


# Registry so rl.py can resolve a config string like "validate" or "validate+format".
REWARDS = {
    "validate": reward_validate,
    "process": reward_process,
    "format": reward_format,
}


def select_rewards(spec: str):
    """Resolve a ``cfg.extra['reward']`` spec (e.g. ``"validate+format"``) to a list of funcs."""
    names = [p.strip() for p in (spec or "validate").split("+") if p.strip()]
    try:
        return [REWARDS[n] for n in names]
    except KeyError as e:  # pragma: no cover - config typo guard
        raise ValueError(
            f"Unknown reward {e.args[0]!r}; choose from {sorted(REWARDS)} "
            f"(combine with '+', e.g. 'validate+format')."
        ) from e
