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


def _score_one(steps: list[str]) -> float:
    """Dense validity reward for one parsed completion, with anti-hack guards."""
    n = len(steps)
    if n == 0:
        return EMPTY_PENALTY
    if n < MIN_STEPS:
        # Trivially short — even if perfectly valid, this is the degenerate exploit. Floor it
        # (with a tiny upward nudge by length so the gradient still prefers 4 steps over 1,
        # but it stays well below any non-degenerate sequence's score).
        return SHORT_PENALTY + 0.01 * n

    violations = validate_sequence(steps)
    base = 1.0 - len(violations) / n  # dense: every fixed rule helps, not all-or-nothing

    bonus = 0.0
    if SHIP_LOT in steps:
        bonus = SHIP_BONUS  # reached the terminus
    elif any(m in steps for m in _TAIL_MARKERS):
        bonus = PROGRESS_BONUS  # advancing into the end-of-line tail
    return base + bonus


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


def _score_one_process(steps: list[str]) -> float:
    """Process reward for one completion: how far the route stays legal before the first break.

    Same anti-hack guards as ``_score_one`` — a prefix reward is, if anything, *more* temptable by a
    short all-valid stub, so the empty / too-short floors are essential. Differs from ``_score_one``
    only in the base term: ``longest_valid_prefix / n`` (a long clean run from the start) instead of
    ``1 - n_viol/n`` (few total breaks).

    NOTE: validates the parsed completion *in isolation* — correct for the next-step task. For the
    completion task (continue a partial route) the prompt's prefix must be prepended first; that
    prompt-aware path lands with the completion-GRPO configs (via the trl ``prompts`` kwarg).
    """
    n = len(steps)
    if n == 0:
        return EMPTY_PENALTY
    if n < MIN_STEPS:
        return SHORT_PENALTY + 0.01 * n
    base = _longest_valid_prefix(steps) / n
    bonus = 0.0
    if SHIP_LOT in steps:
        bonus = SHIP_BONUS  # reached the terminus
    elif any(m in steps for m in _TAIL_MARKERS):
        bonus = PROGRESS_BONUS  # advancing into the end-of-line tail
    return base + bonus


def reward_validate(completions, **kwargs) -> list[float]:
    """Dense, verifier-grounded reward over a batch of completions (trl contract).

    Per completion: parse → ``1 - n_violations/n_steps`` for non-degenerate outputs; a hard
    penalty for empty / ``< MIN_STEPS`` completions; a small bonus for reaching/approaching
    ``SHIP LOT``. Range is roughly ``[-1, 1.1]``.
    """
    return [_score_one(_steps(c)) for c in completions]


def reward_process(completions, **kwargs) -> list[float]:
    """Dense, verifier-grounded PROCESS reward over a batch of completions (trl contract).

    Per completion: ``longest_valid_prefix / n_steps`` for non-degenerate outputs (the fraction of
    the route that stays rule-legal from the first step until the first violation), the same hard
    floor for empty / ``< MIN_STEPS`` completions, and the same small ``SHIP LOT`` tail bonus.
    Pairs with ``reward_validate`` as the {process, outcome} axis of the GRPO reward ablation.
    Range ≈ ``[-1, 1.1]``.
    """
    return [_score_one_process(_steps(c)) for c in completions]


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
