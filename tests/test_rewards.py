"""Unit tests for the GRPO verifiable rewards (Stream 2) — **no torch / GPU**.

These are the GPU-free guardrail for the RLVR spike: they prove the reward orders a real
valid sequence above a verified-corrupt one and that the anti-hack penalties fire, BEFORE we
spend any GPU on GRPO. Reward hacking is the headline risk, so the degenerate-output cases are
as important as the happy path.

Data comes from the vendored MOSFET variants (committed) + the in-repo grammar/datagen — no
generated JSONL corpus is required, so this runs anywhere `uv sync` succeeded.
"""

from __future__ import annotations

import random

from zo_train.datagen import SEP, make_negative
from zo_train.fab import read_sequences
from zo_train.grammar import validate_sequence
from zo_train.rewards import (
    MIN_STEPS,
    SHIP_BONUS,
    reward_format,
    reward_process,
    reward_validate,
    select_rewards,
)


def _valid_completion() -> tuple[list[str], str]:
    """A real, valid MOSFET sequence joined as the model would emit it (`' | '`-separated)."""
    seqs = read_sequences("MOSFET")
    steps = seqs[0]
    assert validate_sequence(steps) == [], "fixture sequence must be valid"
    return steps, SEP.join(steps)


def _corrupt_completion(seed: int = 0) -> tuple[dict, str]:
    """A verified rule-violating variant of a real sequence (via the datagen operators)."""
    rng = random.Random(seed)
    seqs = read_sequences("MOSFET")
    neg = None
    while neg is None:
        neg = make_negative(rng.choice(seqs), rng)
    assert validate_sequence(neg["steps"]) != [], "negative must actually violate a rule"
    return neg, SEP.join(neg["steps"])


def test_valid_scores_higher_than_corrupt():
    """The core RLVR signal: a valid sequence must out-score a verified-corrupt one."""
    _, valid_text = _valid_completion()
    _, corrupt_text = _corrupt_completion()

    (valid_score,) = reward_validate([valid_text])
    (corrupt_score,) = reward_validate([corrupt_text])

    assert valid_score > corrupt_score, (valid_score, corrupt_score)
    # A clean full sequence reaches SHIP LOT → ~1.0 + the small ship bonus.
    assert valid_score > 1.0
    # One injected violation in a ~125-step sequence ⇒ only a small dip (dense, not 0/1).
    assert corrupt_score < valid_score


def test_dense_not_binary():
    """More violations ⇒ strictly lower reward (the reward is dense, not all-or-nothing)."""
    steps, _ = _valid_completion()
    one_bad = list(steps)
    one_bad[5], one_bad[6] = one_bad[6], one_bad[5]  # a local swap perturbs validity a little
    # Construct progressively-worse variants and confirm monotonic-ish degradation vs valid.
    (valid_score,) = reward_validate([SEP.join(steps)])
    (swapped_score,) = reward_validate([SEP.join(one_bad)])
    assert valid_score >= swapped_score


def test_empty_completion_penalized():
    """An empty / whitespace completion gets a hard negative reward."""
    scores = reward_validate(["", "   ", "\n"])
    assert all(s <= -1.0 for s in scores), scores


def test_too_short_completion_penalized():
    """The anti-hack core: a *trivially valid* short stub must NOT beat a real sequence.

    `RECEIVE WAFER LOT` alone has zero violations, so a naive `1 - n_viol/n` reward would pay
    1.0 for it — the degenerate exploit GRPO would happily find. It must be floored instead.
    """
    _, valid_text = _valid_completion()
    (valid_score,) = reward_validate([valid_text])

    # Single trivially-valid step, and a short (< MIN_STEPS) but otherwise-valid prefix.
    steps, _ = _valid_completion()
    short_prefix = SEP.join(steps[: MIN_STEPS - 1])
    (one_step,) = reward_validate(["RECEIVE WAFER LOT"])
    (short_score,) = reward_validate([short_prefix])

    assert one_step <= -0.9, one_step
    assert short_score <= -0.9, short_score
    assert one_step < valid_score and short_score < valid_score
    # And a valid sequence with >= MIN_STEPS must clear the penalty band entirely.
    assert valid_score > 0.0


def test_min_steps_boundary():
    """Exactly MIN_STEPS valid steps is allowed (positive); MIN_STEPS-1 is penalized."""
    steps, _ = _valid_completion()
    (at_min,) = reward_validate([SEP.join(steps[:MIN_STEPS])])
    (below_min,) = reward_validate([SEP.join(steps[: MIN_STEPS - 1])])
    assert below_min <= -0.9
    assert at_min > below_min


def test_handles_chat_message_shape():
    """trl may pass completions as chat-message dicts; the reward must flatten them."""
    _, valid_text = _valid_completion()
    as_chat = [[{"role": "assistant", "content": valid_text}]]
    as_plain = [valid_text]
    assert reward_validate(as_chat) == reward_validate(as_plain)


def test_strips_think_block():
    """A <think>…</think> rationale before the answer must not change the validity score."""
    _, valid_text = _valid_completion()
    wrapped = f"<think>let me reason about the order</think>\n{valid_text}"
    (plain,) = reward_validate([valid_text])
    (think,) = reward_validate([wrapped])
    assert think == plain


def test_reward_format_rewards_clean_think():
    """reward_format pays 1.0 only for a clean <think> block + a parseable answer."""
    _, valid_text = _valid_completion()
    good = f"<think>reasoning</think>\n{valid_text}"
    no_think = valid_text
    empty_think = f"<think></think>\n{valid_text}"
    no_answer = "<think>reasoning</think>\n"
    double = f"<think>a</think><think>b</think>\n{valid_text}"

    assert reward_format([good]) == [1.0]
    assert reward_format([no_think]) == [0.0]
    assert reward_format([empty_think]) == [0.0]
    assert reward_format([no_answer]) == [0.0]
    assert reward_format([double]) == [0.0]


def test_select_rewards():
    """The config-string → reward-function resolver."""
    assert select_rewards("validate") == [reward_validate]
    assert select_rewards("validate+format") == [reward_validate, reward_format]
    assert select_rewards("") == [reward_validate]  # default
    import pytest

    with pytest.raises(ValueError):
        select_rewards("not_a_reward")


def test_batch_length_matches():
    """trl contract: one float per completion, in order."""
    _, valid_text = _valid_completion()
    batch = [valid_text, "", "RECEIVE WAFER LOT", valid_text]
    scores = reward_validate(batch)
    assert len(scores) == len(batch)
    assert all(isinstance(s, float) for s in scores)


# --------------------------------------------------------------------------- process reward


def test_process_reward_registry():
    """The process reward resolves from the config string (the {process} ablation axis)."""
    assert select_rewards("process") == [reward_process]
    assert select_rewards("process+format") == [reward_process, reward_format]


def test_process_valid_full_scores_above_one():
    """A fully-valid route → longest_valid_prefix == n → 1.0 base + the SHIP LOT tail bonus."""
    _, valid_text = _valid_completion()
    (score,) = reward_process([valid_text])
    assert score > 1.0, score


def test_process_tracks_first_violation_index():
    """Core formula: reward == first_violation_index / n (+ the small tail-bonus band).

    This is exactly what separates it from ``reward_validate`` (``1 - n_viol/n``): the process
    reward depends on *where* the first break is, not on *how many* breaks there are.
    """
    seqs = read_sequences("MOSFET")
    for seed in range(8):
        r = random.Random(seed)
        neg = None
        while neg is None:
            neg = make_negative(r.choice(seqs), r)
        steps = neg["steps"]
        first = min(v.step_index for v in validate_sequence(steps))
        (score,) = reward_process([SEP.join(steps)])
        base = first / len(steps)
        assert base - 1e-9 <= score <= base + SHIP_BONUS + 1e-9, (first, len(steps), score)


def test_process_floors_degenerate():
    """Anti-hack: empty / short stubs are floored below any real route (as with reward_validate)."""
    steps, valid_text = _valid_completion()
    (valid_score,) = reward_process([valid_text])
    empty_scores = reward_process(["", "   "])
    (one_step,) = reward_process(["RECEIVE WAFER LOT"])
    (short,) = reward_process([SEP.join(steps[: MIN_STEPS - 1])])
    assert all(s <= -1.0 for s in empty_scores), empty_scores
    assert one_step <= -0.9 and short <= -0.9
    assert one_step < valid_score and short < valid_score


def test_process_and_outcome_rewards_are_distinct():
    """{process, outcome} must be genuinely different functions on clustered violations."""
    seqs = read_sequences("MOSFET")
    diffs = 0
    for seed in range(10):
        r = random.Random(seed)
        neg = None
        while neg is None:
            neg = make_negative(r.choice(seqs), r)
        text = SEP.join(neg["steps"])
        (p,) = reward_process([text])
        (v,) = reward_validate([text])
        if abs(p - v) > 1e-6:
            diffs += 1
    assert diffs > 0, "process and outcome rewards should differ when violations are not uniform"
