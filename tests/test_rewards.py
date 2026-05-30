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

from zo_train.datagen import SEP, completion_example, make_negative, nextstep_example
from zo_train.fab import read_sequences
from zo_train.grammar import validate_sequence
from zo_train.rewards import (
    MIN_STEPS,
    SHIP_BONUS,
    _prefix_from_prompt,
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


# ----------------------------------------------------- prefix-aware (completion-task) rewards
#
# The first GRPO run was a no-op: the reward scored each completion *in isolation*, so a 1-step
# next-step answer hit the `< MIN_STEPS` floor and every sample in the group got the same -0.99 —
# zero std ⇒ zero GRPO advantage ⇒ no learning. The fix validates `prefix + completion` together,
# with the prefix recovered from the trl `prompts` kwarg. These tests pin that behaviour.


def _split_route(frac: float = 0.5) -> tuple[str, list[str], list[str]]:
    """A real, valid MOSFET route split into (family, prefix, rest) at `frac` (both >= MIN_STEPS)."""
    seqs = read_sequences("MOSFET")
    steps = seqs[0]
    assert validate_sequence(steps) == [], "fixture route must be valid"
    cut = max(MIN_STEPS, min(int(len(steps) * frac), len(steps) - MIN_STEPS))
    return "MOSFET", steps[:cut], steps[cut:]


def test_prefix_recovered_from_both_prompt_formats():
    """`_prefix_from_prompt` round-trips the prefix out of BOTH task prompt formats.

    This is the guard that keeps the reward coupled-but-safe to the datagen prompt wording: if either
    template changes so the marker/parse breaks, this fails loudly instead of the reward silently
    reverting to (degenerate) isolation scoring.
    """
    fam, prefix, rest = _split_route()
    comp_prompt = completion_example(fam, prefix, rest)["prompt"]
    ns_prompt = nextstep_example(fam, prefix, rest[0])["prompt"]
    assert _prefix_from_prompt(comp_prompt) == prefix
    assert _prefix_from_prompt(ns_prompt) == prefix
    assert _prefix_from_prompt("no marker in here") == []
    assert _prefix_from_prompt(None) == []


def test_prefix_recovered_from_context_prompt_layout():
    """The base-model *context* prompts put the prefix on the line AFTER a "(N steps):" header.

    `_prefix_from_prompt` must handle that layout too — a silently-empty prefix means the reward
    reverts to isolation scoring (the degenerate case). Written as a literal string so this doesn't
    couple to datagen's in-flight `*_context_example` helpers.
    """
    _fam, prefix, _rest = _split_route()
    ctx_prompt = (
        "You are given a sequence of process steps.\n"
        f"\nProcess so far ({len(prefix)} steps):\n{SEP.join(prefix)}\n"
        "\nPredict the most likely NEXT step.\n"
    )
    assert _prefix_from_prompt(ctx_prompt) == prefix


def test_completion_reward_validates_in_context():
    """A continuation legal *after its prefix* out-scores a verified-illegal one (the core signal)."""
    fam, prefix, rest = _split_route()
    prompt = completion_example(fam, prefix, rest)["prompt"]
    good = SEP.join(rest)  # the real continuation → completes the route
    bad_steps = list(reversed(rest))  # reversed route → breaks ordering rules early
    assert validate_sequence(prefix + bad_steps) != [], "reversed continuation must violate a rule"

    (good_score,) = reward_validate([good], prompts=[prompt])
    (bad_score,) = reward_validate([SEP.join(bad_steps)], prompts=[prompt])
    assert good_score > bad_score, (good_score, bad_score)
    # The genuine continuation reaches SHIP LOT → near-perfect base + the tail bonus.
    assert good_score > 1.0


def test_completion_group_has_reward_variance():
    """Regression for the degenerate run: a group of differing continuations must NOT all score equal.

    With isolation scoring these all collapsed to the floor (zero variance). Prefix-aware, they spread
    out — which is exactly what gives GRPO a non-zero advantage to learn from.
    """
    fam, prefix, rest = _split_route()
    prompt = completion_example(fam, prefix, rest)["prompt"]
    early_stop = rest[: max(MIN_STEPS, len(rest) // 3)]  # valid, but stops before SHIP LOT
    variants = [
        SEP.join(rest),  # perfect → ~1.1 (base 1.0 + ship bonus)
        SEP.join(early_stop),  # valid prefix of the route, no terminus → ~1.0
        SEP.join(list(reversed(rest))),  # many early violations → well below 1.0
    ]
    scores = reward_validate(variants, prompts=[prompt] * len(variants))
    assert len({round(s, 6) for s in scores}) >= 2, scores  # genuine spread, not a constant
    assert scores[0] > 1.0  # the perfect continuation clears 1.0 (base 1.0 + ship bonus)
    assert scores[1] > 0.9  # the valid early-stop is high — nowhere near the -0.99 short-stub floor


def test_process_reward_prefix_aware_credits_continuation():
    """Process reward credits the legal run *into the continuation*; an early break tanks it."""
    fam, prefix, rest = _split_route()
    prompt = completion_example(fam, prefix, rest)["prompt"]
    (full_legal,) = reward_process([SEP.join(rest)], prompts=[prompt])
    # Reverse the continuation → it breaks at/near the boundary → almost no legal run into the tail.
    (early_break,) = reward_process([SEP.join(list(reversed(rest)))], prompts=[prompt])
    assert full_legal > early_break, (full_legal, early_break)
    assert full_legal > 1.0  # full legal run (base 1.0) + ship tail bonus


def test_prefix_aware_backward_compatible():
    """No prompts ⇒ identical to isolation scoring (every pre-existing reward test relies on this)."""
    _, valid_text = _valid_completion()
    assert reward_validate([valid_text]) == reward_validate([valid_text], prompts=None)
    assert reward_process([valid_text]) == reward_process([valid_text], prompts=None)


def test_nextstep_single_step_still_floored_documents_scope():
    """Scope guard: the completion arm does NOT rescue next-step — a 1-step answer is still floored.

    The anti-hack length guard fires on the *completion* length, so a ~1-step next-step answer hits
    the floor whether or not a prefix is supplied. That's deliberate: next-step needs a different
    reward shape (a boundary "is the appended step legal in context?" signal), a separate optional
    arm. This pins the boundary of the prefix-aware *completion* fix so it isn't mistaken for a
    next-step fix.
    """
    fam, prefix, _rest = _split_route()
    next_step = prefix[-1]  # the model emits a single step as its "next" answer
    short_prefix = prefix[:-1]
    ns_prompt = nextstep_example(fam, short_prefix, next_step)["prompt"]

    (isolated,) = reward_validate([next_step])  # no prompt → 1 step < MIN_STEPS → floored
    (in_context,) = reward_validate([next_step], prompts=[ns_prompt])  # still 1-step completion
    assert isolated <= -0.9 and in_context <= -0.9  # floored either way (guard is on completion len)
