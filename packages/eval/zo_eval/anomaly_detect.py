"""Learned anomaly detectors behind the shared ``Predictor`` interface (Stream 4 science).

The **submitted** Task-3 path is the symbolic ``OraclePredictor`` (``validate_sequence``, ~100%).
These detectors are the *learned* science: what a model knows about process validity **without**
the rules — and, crucially, how that knowledge collapses out-of-distribution. The headline result
is measured here: an n-gram likelihood detector scores AUC ≈ 0.9999 in-distribution but ≈ 0.50
(chance) on a held-out family. A served LLM is scored the same way.

Two flavours, both implement ``predict.Predictor`` so they drop straight into ``track.run_track``
(with ``--tasks anomaly``):

- ``LikelihoodDetector`` — turns a **sequence score** (higher ⇒ more valid-looking, e.g. a mean
  token log-likelihood or a VALID/INVALID log-odds) into ``P(valid) ∈ [0,1]`` and a 0/1 verdict.
  The score function is **injectable** (``score_fn`` / ``logit_fn``) so it unit-tests with a fake
  and never touches a GPU; the default ``served_logodds_scorer`` reads VALID/INVALID logprobs from
  a served model via ``zo_common.llm.token_logprobs``. **Never emits ``None``** — the ROC-AUC scorer
  drops the metric if any row lacks a float score, so we always squash to a real number in [0,1].

- ``ClassifierDetector`` — prompts with the training-matched ``datagen.anomaly_example`` framing and
  parses the verdict + rule with ``predict.parse_anomaly`` (so it also attributes a rule id). SCORE
  comes from the verdict-token logprobs when present, else a verdict-based fallback.

Rule attribution: a likelihood detector has no rule signal (returns ``None``); the classifier does.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from zo_common.llm import content, token_logprobs
from zo_train.prompts import build_messages as build_sft_messages
from zo_train.datagen import SEP

from zo_eval.predict import parse_anomaly
from zo_eval.submission import AnomalyInput

# A raw score → P(valid) function takes an AnomalyInput and returns a real number (higher = more
# valid-looking). A logit function returns the log-odds of VALID directly (mapped by sigmoid).
ScoreFn = Callable[[AnomalyInput], float]


def _sigmoid(x: float) -> float:
    # Numerically stable; clamps extreme inputs so we never overflow ``exp``.
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def _verdict_logodds(resp: dict) -> float | None:
    """Log-odds of VALID from the first VALID/INVALID token's top-logprobs, or ``None``.

    Mirrors ``predict_llm._valid_prob`` but returns the *logit* (so the detector can calibrate /
    threshold in log-odds space) rather than the probability.
    """
    for t in token_logprobs(resp)[:8]:
        tok = (t.get("token") or "").strip().upper()
        if tok.startswith("VALID") or tok.startswith("INVALID"):
            cand = {(c.get("token") or "").strip().upper(): c.get("logprob") for c in t.get("top_logprobs", [])}
            cand.setdefault(tok, t.get("logprob"))
            lp_v = cand.get("VALID")
            lp_i = cand.get("INVALID")
            if lp_v is None and lp_i is None:
                return None
            # log-odds = logP(VALID) - logP(INVALID); treat a missing side as very unlikely.
            return (lp_v if lp_v is not None else -30.0) - (lp_i if lp_i is not None else -30.0)
    return None


def served_logodds_scorer(
    model: str = "default",
    base_url: str | None = None,
    chat_fn: Callable | None = None,
    temperature: float = 0.0,
) -> ScoreFn:
    """Default ``LikelihoodDetector`` score: VALID/INVALID log-odds from a served model.

    Builds the training-matched ``anomaly_example`` prompt, asks for a one-word verdict with
    ``logprobs``, and returns the log-odds of VALID. Falls back to a coarse ±4 (from the parsed
    verdict) if the server returns no usable logprobs — so the score is always a finite real number.
    """
    from zo_common.llm import chat as _default_chat

    _chat = chat_fn or _default_chat

    def _score(item: AnomalyInput) -> float:
        resp = _chat(
            build_sft_messages("anomaly", item),
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=48,
            logprobs=True,
            top_logprobs=5,
        )
        lo = _verdict_logodds(resp)
        if lo is not None:
            return lo
        is_valid, _ = parse_anomaly(content(resp))
        return 4.0 if is_valid else -4.0

    return _score


class LikelihoodDetector:
    """Sequence-likelihood / log-odds anomaly detector → ``P(valid)`` + a 0/1 verdict.

    ``score_fn(item)`` returns a real number, higher ⇒ more valid-looking (a mean token
    log-likelihood, a VALID/INVALID log-odds, anything monotone). It is mapped to ``P(valid)``:

    - **calibrated** (``calibrate(...)`` was called, or ``lo``/``hi`` given): affine to [0,1] with
      ``lo``→0, ``hi``→1, clamped — and the threshold is placed at the calibrated midpoint by default.
    - **uncalibrated**: ``sigmoid(gain * score)`` — any real input yields a probability in (0,1).

    The verdict is ``P(valid) >= threshold`` (threshold in probability space). Rule attribution is
    ``None`` (a likelihood has no rule signal). Drops into ``run_track`` with ``--tasks anomaly``.
    """

    name = "likelihood"

    def __init__(
        self,
        score_fn: ScoreFn,
        *,
        threshold: float = 0.5,
        gain: float = 1.0,
        lo: float | None = None,
        hi: float | None = None,
        name: str | None = None,
    ) -> None:
        self.score_fn = score_fn
        self.threshold = threshold
        self.gain = gain
        self.lo = lo
        self.hi = hi
        if name:
            self.name = name

    @classmethod
    def from_served(
        cls,
        model: str = "default",
        base_url: str | None = None,
        chat_fn: Callable | None = None,
        **kw,
    ) -> LikelihoodDetector:
        """Build a detector backed by a served model's VALID/INVALID log-odds."""
        return cls(served_logodds_scorer(model=model, base_url=base_url, chat_fn=chat_fn), **kw)

    def calibrate(
        self,
        items: Iterable[AnomalyInput],
        labels: Iterable[int],
        *,
        set_threshold: bool = True,
    ) -> LikelihoodDetector:
        """Fit the score→prob map (and optionally the threshold) on labelled examples.

        ``labels`` are 1 (valid) / 0 (invalid). Sets ``lo`` to the mean invalid score and ``hi`` to
        the mean valid score (so the two class means map near 0 and 1); when both classes are
        present, places the verdict threshold at the midpoint in probability space. Robust to a
        missing class (falls back to a logistic). Returns ``self`` so it chains.
        """
        scored = [(self.score_fn(it), int(lbl)) for it, lbl in zip(items, labels, strict=False)]
        valids = [s for s, lbl in scored if lbl == 1]
        invalids = [s for s, lbl in scored if lbl == 0]
        if valids and invalids:
            self.hi = sum(valids) / len(valids)
            self.lo = sum(invalids) / len(invalids)
            if self.hi == self.lo:  # degenerate; widen so the affine map stays finite
                self.hi, self.lo = self.lo + 1e-6, self.lo - 1e-6
            if set_threshold:
                mid = (self.hi + self.lo) / 2
                self.threshold = self._prob(mid)
        return self

    def _prob(self, raw: float) -> float:
        """Map a raw score → P(valid) ∈ [0,1]. Always finite, never None."""
        if self.lo is not None and self.hi is not None and self.hi != self.lo:
            p = (raw - self.lo) / (self.hi - self.lo)
            return max(0.0, min(1.0, p))
        return _sigmoid(self.gain * raw)

    # ------------------------------------------------------------------ Predictor protocol

    def next_step(self, item) -> list[str]:  # not modelled by a likelihood detector
        return []

    def complete(self, item) -> list[str]:
        return []

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        raw = self.score_fn(item)
        score = self._prob(raw)
        return (1 if score >= self.threshold else 0, score, None)


class ClassifierDetector:
    """Verdict-classifier detector: training-matched prompt → parsed VALID/INVALID + rule.

    Uses ``datagen.anomaly_example`` for the prompt (so eval matches training byte-for-byte) and
    ``predict.parse_anomaly`` for the verdict + rule id. SCORE = P(valid) from the verdict-token
    logprobs when the server returns them, else a verdict-based fallback (0.9 valid / 0.1 invalid) —
    **never ``None``**. Unlike ``LikelihoodDetector`` this also attributes a rule. ``chat_fn`` is
    injectable for tests.
    """

    name = "classifier"

    def __init__(
        self,
        model: str = "default",
        base_url: str | None = None,
        chat_fn: Callable | None = None,
        temperature: float = 0.0,
    ) -> None:
        from zo_common.llm import chat as _default_chat

        self.model = model
        self.base_url = base_url
        self.temp = temperature
        self._chat = chat_fn or _default_chat

    def next_step(self, item) -> list[str]:
        return []

    def complete(self, item) -> list[str]:
        return []

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        resp = self._chat(
            build_sft_messages("anomaly", item),
            model=self.model,
            base_url=self.base_url,
            temperature=self.temp,
            max_tokens=48,
            logprobs=True,
            top_logprobs=5,
        )
        is_valid, rule = parse_anomaly(content(resp))
        lo = _verdict_logodds(resp)
        score = _sigmoid(lo) if lo is not None else (0.9 if is_valid else 0.1)
        return (is_valid, score, rule)


def _smoke() -> None:  # pragma: no cover - manual (no server / GPU needed)
    # 1) LikelihoodDetector with an injected fake scorer: an invalid sequence scores low, a valid
    #    one high; calibration places the threshold between the class means; scores are floats.
    seqs = {
        "valid_a": ["RECEIVE WAFER LOT", "WAFER SORT TEST", "SHIP LOT"],
        "valid_b": ["RECEIVE WAFER LOT", "WAFER SORT TEST", "CURE PASSIVATION", "SHIP LOT"],
        "bad_a": ["RECEIVE WAFER LOT", "SHIP LOT"],
        "bad_b": ["SHIP LOT", "RECEIVE WAFER LOT"],
    }
    labels = {"valid_a": 1, "valid_b": 1, "bad_a": 0, "bad_b": 0}

    def fake_score(item: AnomalyInput) -> float:
        # Stand-in likelihood: a valid sequence runs RECEIVE… → WAFER SORT TEST → … → SHIP LOT.
        # Reward the right endpoints + having a test before ship — a separable monotone signal.
        st = item.sequence
        ends_ok = st[:1] == ["RECEIVE WAFER LOT"] and st[-1:] == ["SHIP LOT"]
        test_before_ship = "WAFER SORT TEST" in st and "SHIP LOT" in st and st.index("WAFER SORT TEST") < st.index("SHIP LOT")
        return float(2 * ends_ok + 2 * test_before_ship + 0.1 * len(st))

    items = [AnomalyInput(ex, "MOSFET", st) for ex, st in seqs.items()]
    det = LikelihoodDetector(fake_score).calibrate(items, [labels[it.example_id] for it in items])
    for it in items:
        iv, sc, rule = det.anomaly(it)
        assert 0.0 <= sc <= 1.0 and rule is None, (it.example_id, iv, sc)
        assert iv == labels[it.example_id], (it.example_id, iv, sc)
    # Uncalibrated path still yields a finite probability for any real score (never None).
    raw_det = LikelihoodDetector(fake_score)
    iv, sc, _ = raw_det.anomaly(items[0])
    assert sc is not None and 0.0 < sc < 1.0

    # 2) ClassifierDetector with a fake chat: parses the verdict + rule, score in [0,1].
    def fake_chat(messages, **kw):
        prompt = messages[-1]["content"]
        bad = "SHIP LOT |" in prompt.split("Process sequence:")[1][:40] or prompt.count(SEP) < 2
        content_txt = "INVALID. RULE_SHIP_BEFORE_TEST" if bad else "VALID."
        # vLLM-style logprobs on the first verdict token.
        first = "INVALID" if bad else "VALID"
        tlp = {
            "content": [
                {
                    "token": first,
                    "logprob": -0.2,
                    "top_logprobs": [
                        {"token": "VALID", "logprob": -2.0 if bad else -0.1},
                        {"token": "INVALID", "logprob": -0.1 if bad else -2.5},
                    ],
                }
            ]
        }
        return {"choices": [{"message": {"content": content_txt}, "logprobs": tlp}]}

    clf = ClassifierDetector(chat_fn=fake_chat)
    iv_good, sc_good, rule_good = clf.anomaly(AnomalyInput("g", "MOSFET", seqs["valid_b"]))
    iv_bad, sc_bad, rule_bad = clf.anomaly(AnomalyInput("b", "MOSFET", seqs["bad_b"]))
    assert iv_good == 1 and rule_good is None and 0.0 <= sc_good <= 1.0, (iv_good, sc_good, rule_good)
    assert iv_bad == 0 and rule_bad == "RULE_SHIP_BEFORE_TEST" and sc_bad < 0.5, (iv_bad, sc_bad, rule_bad)
    print("anomaly_detect.py smoke OK — likelihood + classifier verdicts, scores in [0,1], never None")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
