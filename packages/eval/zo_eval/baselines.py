"""GPU-free predictors: an n-gram baseline, a frequency floor, and the symbolic oracle.

These exercise the whole eval-input → submission → score pipeline **today** (no model needed) and
are the honest baselines-to-beat:
- ``NGramPredictor`` fit on the LOFO *train* families is the source of the headline ID→OOD drop
  (next-step top-1 ≈0.77 in-distribution → ≈0.47 on a held-out family).
- ``OraclePredictor`` (``validate_sequence``) is the **submitted** anomaly path — ~100%, free; the
  learned detector is the science, not this.
- ``FreqPredictor`` is the dumbest floor, to make the "logic vs memorization" delta visible.

All three satisfy ``predict.Predictor``. The driver decides which task(s) to run per predictor, so
the oracle/freq predictors return ``[]`` for tasks they don't model.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from zo_train.fab import FAMILIES, read_sequences
from zo_train.grammar import validate_sequence

from zo_eval.predict import normalize_family
from zo_eval.submission import AnomalyInput, ValidInput

_LOGFLOOR = math.log(1e-6)


class _NGram:
    """Back-off n-gram over raw step sequences (no special tokens)."""

    def __init__(self, order: int = 3) -> None:
        self.order = order
        self.ctx: dict[int, dict[tuple, Counter]] = {k: defaultdict(Counter) for k in range(1, order)}
        self.uni: Counter = Counter()

    def fit(self, seqs: list[list[str]]) -> _NGram:
        for s in seqs:
            for i, tok in enumerate(s):
                self.uni[tok] += 1
                for k in range(1, self.order):
                    if i - k >= 0:
                        self.ctx[k][tuple(s[i - k : i])][tok] += 1
        return self

    def _dist(self, hist: list[str]) -> Counter:
        for k in range(self.order - 1, 0, -1):
            if len(hist) >= k:
                c = self.ctx[k].get(tuple(hist[-k:]))
                if c:
                    return c
        return self.uni

    def topk(self, hist: list[str], k: int = 5) -> list[str]:
        # Accumulate distinct candidates across back-off levels (high→low) + unigram, so we
        # always return k even at deterministic positions where one level has few continuations.
        levels = [
            self.ctx[kk].get(tuple(hist[-kk:]))
            for kk in range(self.order - 1, 0, -1)
            if len(hist) >= kk and self.ctx[kk].get(tuple(hist[-kk:]))
        ]
        levels.append(self.uni)
        out: list[str] = []
        seen: set[str] = set()
        for c in levels:
            for tok, _ in c.most_common():
                if tok not in seen:
                    seen.add(tok)
                    out.append(tok)
                    if len(out) >= k:
                        return out
        return out

    def logprob(self, hist: list[str], tok: str) -> float:
        c = self._dist(hist)
        tot = sum(c.values())
        p = (c.get(tok, 0) / tot) if tot else 0.0
        return math.log(p) if p > 0 else _LOGFLOOR

    def mean_logprob(self, steps: list[str]) -> float:
        if len(steps) < 2:
            return 0.0
        return sum(self.logprob(steps[:i], steps[i]) for i in range(1, len(steps))) / (len(steps) - 1)


class NGramPredictor:
    """Per-family back-off n-gram with a pooled fallback (used for OOD families)."""

    name = "ngram"

    def __init__(self, train_families: list[str] | None = None, order: int = 3) -> None:
        fams = list(train_families) if train_families else list(FAMILIES)
        self.per_fam: dict[str, _NGram] = {}
        pooled: list[list[str]] = []
        for fam in fams:
            seqs = read_sequences(fam)
            self.per_fam[fam] = _NGram(order).fit(seqs)
            pooled += seqs
        self.pooled = _NGram(order).fit(pooled)
        # Anomaly calibration: the mean-logprob distribution over (valid) pooled sequences.
        lps = [self.pooled.mean_logprob(s) for s in pooled]
        self.lp_mean = sum(lps) / len(lps)
        self.lp_std = (sum((x - self.lp_mean) ** 2 for x in lps) / len(lps)) ** 0.5 or 1.0

    def _model(self, family: str) -> _NGram:
        return self.per_fam.get(normalize_family(family), self.pooled)

    def next_step(self, item: ValidInput) -> list[str]:
        return self._model(item.family).topk(item.partial_sequence, 5)

    def complete(self, item: ValidInput) -> list[str]:
        m = self._model(item.family)
        steps = list(item.partial_sequence)
        frac = item.completion_fraction or 0.7
        full_est = int(len(steps) / frac) if frac else len(steps) + 40
        cap = max(8, min(220, int((full_est - len(steps)) * 1.5)))
        out: list[str] = []
        for _ in range(cap):
            nxt = m.topk(steps, 1)
            if not nxt:
                break
            tok = nxt[0]
            out.append(tok)
            steps.append(tok)
            if tok == "SHIP LOT":
                break
        return out

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        # P(valid): high mean-logprob (near the valid-corpus mean) → valid; (mean - 3σ) → 0.
        lp = self.pooled.mean_logprob(item.sequence)
        z = (lp - (self.lp_mean - 3 * self.lp_std)) / (3 * self.lp_std)
        score = max(0.0, min(1.0, z))
        return (1 if score >= 0.5 else 0, score, None)


class OraclePredictor:
    """The submitted anomaly path: validate_sequence is a ~100% oracle (free). Not an ML result."""

    name = "oracle"

    def next_step(self, item: ValidInput) -> list[str]:
        return []

    def complete(self, item: ValidInput) -> list[str]:
        return []

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        viol = validate_sequence(item.sequence)
        if viol:
            return (0, 0.0, sorted({v.rule for v in viol})[0])
        return (1, 1.0, None)


class FreqPredictor:
    """Dumbest floor: always predicts the globally most-frequent steps. Makes the delta visible."""

    name = "freq"

    def __init__(self, train_families: list[str] | None = None) -> None:
        fams = list(train_families) if train_families else list(FAMILIES)
        c: Counter = Counter()
        for fam in fams:
            for s in read_sequences(fam):
                c.update(s)
        self.top5 = [t for t, _ in c.most_common(5)]

    def next_step(self, item: ValidInput) -> list[str]:
        return list(self.top5)

    def complete(self, item: ValidInput) -> list[str]:
        return []

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        return (1, 0.5, None)


def _smoke() -> None:  # pragma: no cover - manual
    import random

    from zo_train.datagen import make_negative

    rng = random.Random(0)
    ng = NGramPredictor()
    seqs = read_sequences("MOSFET")
    s = seqs[0]
    cut = int(len(s) * 0.6)
    vi = ValidInput("v1", "MOSFET", 0.6, s[:cut])
    nexts = ng.next_step(vi)
    comp = ng.complete(vi)
    print(f"ngram next_step (5): {nexts[:3]}…  | true next: {s[cut]!r} | hit: {s[cut] in nexts}")
    print(f"ngram complete: {len(comp)} steps, ends {comp[-1]!r}")
    assert len(nexts) == 5 and comp

    ora = OraclePredictor()
    neg = make_negative(seqs[1], rng)
    assert ora.anomaly(AnomalyInput("a1", "MOSFET", neg["steps"]))[0] == 0
    assert ora.anomaly(AnomalyInput("a2", "MOSFET", seqs[2]))[0] == 1
    av = ng.anomaly(AnomalyInput("a3", "MOSFET", seqs[2]))
    an = ng.anomaly(AnomalyInput("a4", "MOSFET", neg["steps"]))
    print(f"ngram anomaly score — valid: {av[1]:.3f}  invalid: {an[1]:.3f}")
    print("baselines.py smoke OK")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
