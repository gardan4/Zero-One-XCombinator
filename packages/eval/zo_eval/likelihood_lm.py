"""Likelihood eval for the **pure-LM** fab-sequence checkpoints (Industrial AI / Infineon).

The four already-trained Qwen2.5-1.5B checkpoints (all-families + 3 leave-one-family-out folds)
are plain next-token language models over *valid routes*: they were SFT'd on the raw-LM text
``datagen.lm_example`` emits (``"Product family: {fam}\\nProcess sequence: A | B | ..."``), **not**
on an instruction format. Prompting them instruction-style ("Is this valid? name the rule") gives
nonsense — a documented failure mode. The correct probe for what they learned is **likelihood**:
how surprised the model is by a sequence under its own next-token distribution.

So everything here scores by NLL on the *exact* training framing (byte-for-byte
``lm_example(...)["text"]``) rather than by generation:

- ``seq_nll`` — mean per-token teacher-forced NLL of a family-conditioned sequence.
- ``LMLikelihoodDetector`` — wraps ``score = -seq_nll`` into the existing
  :class:`zo_eval.anomaly_detect.LikelihoodDetector` so a held-out family's anomaly AUC falls out of
  ``track.run_track(..., tasks=("anomaly",))`` with **no new scoring code** (lower NLL ⇒ more
  valid-looking ⇒ higher score). This is the headline ID-vs-OOD result for the LM path.
- ``next_step_tf`` — teacher-forced Task-1: top-5 next steps by next-token argmax over the prefix's
  LM framing, decoded + snapped via :func:`zo_eval.predict.parse_pipe_list`. Wrapped as a
  ``Predictor.next_step`` so it plugs into ``run_track`` unchanged.
- ``id_ood_nll_matrix`` — for each checkpoint × family, mean held-out NLL / perplexity over that
  family's **test** split (from :func:`zo_train.datagen.make_splits`). The ID→OOD table; logged via
  ``new_run`` / ``append_metric`` so the dashboard Compare page reads it.

torch is imported **lazily inside functions**, so this module imports on a laptop with no torch and
unit-tests the whole plumbing GPU-free with an injected fake score (see ``tests/test_likelihood_lm.py``).
**The real forward-pass paths (``seq_nll`` / ``next_step_tf`` / ``id_ood_nll_matrix``) require torch +
the model weights and are meant to run on the cluster against pulled checkpoints**, not on the laptop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from zo_train.datagen import SEP, lm_example, make_splits
from zo_train.fab import FAMILIES, read_sequences

from zo_eval.anomaly_detect import LikelihoodDetector
from zo_eval.predict import parse_pipe_list, vocab
from zo_eval.submission import AnomalyInput, ValidInput

# A scorer maps a family + step list to a real-valued log-likelihood-like score (higher ⇒ more
# valid-looking). Injectable so the detector / matrix unit-test with a fake and never touch a GPU.
SeqScoreFn = Callable[[str, Sequence[str]], float]


# --------------------------------------------------------------------------------------
# Core: teacher-forced negative log-likelihood of the exact training framing.
# --------------------------------------------------------------------------------------


def seq_nll(model: Any, tok: Any, family: str, steps: Sequence[str]) -> float:
    """Mean per-token teacher-forced NLL of ``lm_example(family, steps)["text"]`` (nats/token).

    Builds the *exact* raw-LM string the checkpoints trained on (so likelihood is measured on the
    same framing, byte-for-byte), then sums the log-probability the model assigns to each actual
    next token and divides by the number of scored tokens. Lower ⇒ the model finds the sequence
    more "valid-looking". torch is imported here so the module stays import-clean without it.

    Standard causal shift: logits at position ``t`` predict token ``t+1``; we score positions
    ``0..n-2`` against targets ``1..n-1``. Returns ``+inf`` for a degenerate (≤1 token) input.
    """
    import torch

    text = lm_example(family, list(steps))["text"]
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"]
    device = getattr(model, "device", None)
    if device is not None:
        input_ids = input_ids.to(device)
    if input_ids.shape[1] < 2:
        return float("inf")

    with torch.no_grad():
        logits = model(input_ids).logits  # (1, T, V)

    # Shift: predict token t+1 from the logits at position t.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    nll = -token_lp.mean().item()
    return float(nll)


def seq_perplexity(model: Any, tok: Any, family: str, steps: Sequence[str]) -> float:
    """``exp(seq_nll)`` — perplexity per token (overflows to ``inf`` gracefully)."""
    nll = seq_nll(model, tok, family, steps)
    try:
        return math.exp(nll)
    except OverflowError:
        return float("inf")


def make_nll_scorer(model: Any, tok: Any) -> SeqScoreFn:
    """Bind a loaded ``(model, tok)`` into a ``(family, steps) -> -seq_nll`` score function.

    Negated so higher = more valid-looking, matching ``LikelihoodDetector``'s sign convention.
    """

    def _score(family: str, steps: Sequence[str]) -> float:
        return -seq_nll(model, tok, family, steps)

    return _score


# --------------------------------------------------------------------------------------
# Anomaly detector: reuse the existing LikelihoodDetector, supply an LM score function.
# --------------------------------------------------------------------------------------


class LMLikelihoodDetector(LikelihoodDetector):
    """``LikelihoodDetector`` whose sequence score is ``-seq_nll`` from a pure-LM checkpoint.

    Subclasses the existing detector so all of its score→P(valid) mapping, ``calibrate`` and the
    ``Predictor`` protocol are reused verbatim — the *only* new thing is where the per-item score
    comes from (the LM's mean token log-likelihood instead of a served VALID/INVALID log-odds).
    Drops straight into ``track.run_track(..., tasks=("anomaly",))`` to yield the held-out-family
    anomaly AUC with no new scoring code.

    ``score_fn`` here is a ``(family, steps) -> float`` callable (default: ``make_nll_scorer`` over
    ``model``/``tok``); it's adapted to the base class's ``AnomalyInput -> float`` interface. Pass an
    injected fake ``score_fn`` to exercise the whole plumbing GPU-free.
    """

    name = "lm-likelihood"

    def __init__(
        self,
        model: Any = None,
        tok: Any = None,
        *,
        score_fn: SeqScoreFn | None = None,
        threshold: float = 0.5,
        gain: float = 1.0,
        lo: float | None = None,
        hi: float | None = None,
        name: str | None = None,
    ) -> None:
        if score_fn is None:
            if model is None or tok is None:
                raise ValueError("LMLikelihoodDetector needs either score_fn= or (model, tok).")
            score_fn = make_nll_scorer(model, tok)
        self._seq_score = score_fn

        # Adapt the (family, steps) scorer to the base class's AnomalyInput -> float signature.
        def _item_score(item: AnomalyInput) -> float:
            return float(score_fn(item.family, item.sequence))

        super().__init__(
            _item_score,
            threshold=threshold,
            gain=gain,
            lo=lo,
            hi=hi,
            name=name or self.name,
        )

    @classmethod
    def from_client(cls, client: Any, **kw: Any) -> LMLikelihoodDetector:
        """Build a detector from a ``HubInferenceClient`` (loads weights via ``local_files_only``).

        Triggers the client's lazy load, then reads its ``model``/``tokenizer`` to score by NLL.
        Cluster-only (needs torch + the pulled checkpoint).
        """
        model, tok = _client_model_tok(client)
        return cls(model, tok, **kw)


def lm_likelihood_detector_from_client(client: Any, **kw: Any) -> LMLikelihoodDetector:
    """Module-level alias for :meth:`LMLikelihoodDetector.from_client` (convenience)."""
    return LMLikelihoodDetector.from_client(client, **kw)


# --------------------------------------------------------------------------------------
# Task-1: teacher-forced next-step from the LM's next-token distribution.
# --------------------------------------------------------------------------------------


def next_step_tf(
    model: Any, tok: Any, family: str, prefix: Sequence[str], *, k: int = 5
) -> list[str]:
    """Top-``k`` next steps by the LM's next-token distribution over the prefix's training framing.

    Frames the prefix exactly as in training — ``"Product family: {fam}\\nProcess sequence: s1 | s2
    | ... | "`` (note the trailing ``" | "`` separator the model emits *before* the next step) — and
    greedily decodes one step per beam-free top-token path: at each position we take the
    argmax/top token, append it, and continue until a separator/newline/EOS or a step-length cap.
    The candidate strings are snapped to the exact vocab with ``parse_pipe_list`` and de-duplicated,
    so the result is ≤``k`` ranked, normalized next-step candidates (rank-1 = the model's greedy
    continuation). torch is imported lazily.

    This is "teacher-forced" in that the *prefix* is given verbatim from gold and only the single
    next step is generated — the standard Task-1 protocol.
    """
    import torch

    # Exact training framing up to (and including) the separator that precedes the next step.
    base = f"Product family: {family}\nProcess sequence: "
    if prefix:
        base += SEP.join(prefix) + SEP
    enc = tok(base, return_tensors="pt")
    input_ids = enc["input_ids"]
    device = getattr(model, "device", None)
    if device is not None:
        input_ids = input_ids.to(device)

    # Tokens that terminate a step (the " | " separator and newline render to these substrings).
    eos_id = getattr(tok, "eos_token_id", None)

    def _greedy_top_tokens(n: int) -> list[int]:
        """The ``n`` most-likely *first* next tokens at the current position (rank seeds)."""
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
        topv = torch.topk(logits.float(), k=min(n, logits.shape[-1]), dim=-1)
        return topv.indices[0].tolist()

    # Seed up to k candidate continuations from the top-k first tokens, then greedily extend each
    # until a separator/newline/EOS. Decoding per-seed keeps it simple and deterministic.
    seeds = _greedy_top_tokens(k)
    candidates: list[str] = []
    max_step_tokens = 16
    for seed in seeds:
        ids = input_ids.clone()
        next_id = seed
        produced: list[int] = []
        for _ in range(max_step_tokens):
            if eos_id is not None and next_id == eos_id:
                break
            piece = tok.decode([next_id])
            if "|" in piece or "\n" in piece:
                break
            produced.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
            with torch.no_grad():
                logits = model(ids).logits[:, -1, :]
            next_id = int(torch.argmax(logits.float(), dim=-1)[0].item())
        text = tok.decode(produced).strip()
        if text:
            candidates.append(text)

    # Snap each candidate to the exact vocab (strict) and de-dup, preserving rank order.
    out: list[str] = []
    for cand in candidates:
        for snapped in parse_pipe_list(cand, vocab(), strict=True):
            if snapped not in out:
                out.append(snapped)
    return out[:k]


class LMNextStepPredictor:
    """``Predictor`` exposing teacher-forced ``next_step`` from a pure-LM checkpoint.

    Only Task-1 is modelled (``complete`` / ``anomaly`` return empties / a neutral verdict); for the
    anomaly task use ``LMLikelihoodDetector`` instead. ``score_fn`` is injectable as a
    ``(family, prefix) -> [steps]`` callable so the plumbing unit-tests without a model; the default
    runs ``next_step_tf`` over ``model``/``tok`` (cluster-only).
    """

    name = "lm-nextstep"

    def __init__(
        self,
        model: Any = None,
        tok: Any = None,
        *,
        k: int = 5,
        next_fn: Callable[[str, Sequence[str]], list[str]] | None = None,
    ) -> None:
        self.k = k
        if next_fn is None:
            if model is None or tok is None:
                raise ValueError("LMNextStepPredictor needs either next_fn= or (model, tok).")

            def next_fn(family: str, prefix: Sequence[str]) -> list[str]:
                return next_step_tf(model, tok, family, prefix, k=k)

        self._next_fn = next_fn

    def next_step(self, item: ValidInput) -> list[str]:
        return self._next_fn(item.family, list(item.partial_sequence))[: self.k]

    def complete(self, item: ValidInput) -> list[str]:  # not modelled by the next-step probe
        return []

    def anomaly(self, item: AnomalyInput) -> tuple[int, float | None, str | None]:
        return (1, 0.5, None)


# --------------------------------------------------------------------------------------
# ID→OOD held-out NLL / perplexity matrix across checkpoints × families.
# --------------------------------------------------------------------------------------


def family_test_nll(
    score_fn: SeqScoreFn,
    family: str,
    *,
    splits: dict | None = None,
    limit: int | None = None,
    data_dir: Any = None,
) -> dict[str, float]:
    """Mean held-out NLL + perplexity for one family over its **test** split.

    ``score_fn(family, steps)`` returns ``-nll`` (higher = more valid-looking), so we negate to get
    NLL back. The test indices come from :func:`make_splits` (deterministic per-family split), so
    no example a model trained on leaks in. ``limit`` caps sequences (for a quick cluster pass).
    """
    splits = splits or make_splits(data_dir=data_dir)
    seqs = read_sequences(family, data_dir)
    test_idx = splits["per_family"][family]["test"]
    if limit is not None:
        test_idx = test_idx[:limit]
    nlls = [-float(score_fn(family, seqs[i])) for i in test_idx]
    if not nlls:
        return {"mean_nll": float("nan"), "perplexity": float("nan"), "n": 0}
    mean_nll = sum(nlls) / len(nlls)
    try:
        ppl = math.exp(mean_nll)
    except OverflowError:
        ppl = float("inf")
    return {"mean_nll": mean_nll, "perplexity": ppl, "n": len(nlls)}


def id_ood_nll_matrix(
    checkpoints: dict[str, str],
    families: list[str] | None = None,
    *,
    score_fns: dict[str, SeqScoreFn] | None = None,
    limit: int | None = None,
    data_dir: Any = None,
    log: bool = True,
    run_name: str = "lm-likelihood:id-ood-matrix",
    tags: list[str] | None = None,
) -> dict:
    """Held-out NLL / perplexity for each checkpoint × family — the headline ID→OOD table.

    ``checkpoints`` maps a label (e.g. ``"all-families"`` or ``"lofo-IGBT"``) to a model reference
    (a local HF checkpoint dir / repo id understood by ``HubInferenceClient``). For each one we load
    the weights once and score every family's **test** split by NLL; for a LOFO checkpoint the
    *left-out* family is its OOD column and the rest are ID — the surprise gap across that boundary
    is the result. ``score_fns`` injects per-label scorers for a GPU-free unit test (bypasses model
    loading entirely).

    Returns ``{"families", "checkpoints", "matrix": {label: {family: {mean_nll, perplexity, n}}}}``.
    When ``log`` is set, writes one ``append_metric`` row per checkpoint (flattened
    ``nll__<label>__<family>`` / ``ppl__<label>__<family>`` keys) under a ``new_run`` so the
    dashboard Compare page can read the table. Logging never blocks the computation — a registry
    error is swallowed and recorded in the result.
    """
    families = families or list(FAMILIES)
    splits = make_splits(data_dir=data_dir)
    matrix: dict[str, dict[str, dict[str, float]]] = {}

    for label, ref in checkpoints.items():
        if score_fns and label in score_fns:
            score_fn = score_fns[label]
        else:
            score_fn = _scorer_for_checkpoint(ref)
        matrix[label] = {
            fam: family_test_nll(score_fn, fam, splits=splits, limit=limit, data_dir=data_dir)
            for fam in families
        }

    result: dict[str, Any] = {
        "families": families,
        "checkpoints": dict(checkpoints),
        "matrix": matrix,
    }

    if log:
        try:
            from zo_common import append_metric, new_run, update_run

            run = new_run(
                run_name,
                "eval",
                config={
                    "kind": "lm-likelihood-id-ood",
                    "checkpoints": dict(checkpoints),
                    "families": families,
                    "limit": limit,
                },
                tags=(tags or []) + ["lm-likelihood", "id-ood-matrix"],
            )
            for step, (label, by_fam) in enumerate(matrix.items()):
                flat: dict[str, float] = {"checkpoint": label}  # type: ignore[dict-item]
                for fam, cell in by_fam.items():
                    flat[f"nll__{label}__{fam}"] = round(float(cell["mean_nll"]), 6)
                    flat[f"ppl__{label}__{fam}"] = round(float(cell["perplexity"]), 4)
                append_metric(run.id, step=step, **flat)
            update_run(run.id, status="completed")
            result["run_id"] = run.id
        except Exception as exc:  # logging must never break the science
            result["log_error"] = repr(exc)

    return result


# --------------------------------------------------------------------------------------
# HubInferenceClient adapters (cluster-only — need torch + pulled weights).
# --------------------------------------------------------------------------------------


def _client_model_tok(client: Any) -> tuple[Any, Any]:
    """Trigger a ``HubInferenceClient``'s lazy load and return ``(model, tokenizer)``.

    ``HubInferenceClient`` loads with ``local_files_only`` for on-disk checkpoints and stores the
    model/tokenizer on private attrs; we go through ``_load()`` so we get the same instance it would
    generate with. The loaded model carries its own ``.device``, so the scorers move inputs onto it.
    """
    if hasattr(client, "_load"):
        client._load()
    model = getattr(client, "_model", None) or getattr(client, "model", None)
    tok = getattr(client, "_tok", None) or getattr(client, "tokenizer", None)
    if model is None or tok is None:
        raise ValueError(f"Could not get model/tokenizer from {client!r}")
    return model, tok


def _scorer_for_checkpoint(ref: str) -> SeqScoreFn:
    """Load a checkpoint via ``HubInferenceClient`` and return its ``(family, steps) -> -nll``."""
    from zo_common.hub_inference import HubInferenceClient

    client = HubInferenceClient(ref)
    model, tok = _client_model_tok(client)
    return make_nll_scorer(model, tok)


# --------------------------------------------------------------------------------------
# Smoke (manual, GPU-FREE): mirror anomaly_detect._smoke with an injected fake LM scorer.
# --------------------------------------------------------------------------------------


def _smoke() -> None:  # pragma: no cover - manual (no server / GPU / torch needed)
    # A fake "LM score": valid routes start RECEIVE… → … → SHIP LOT with a test before ship.
    # Returns -nll (higher = more valid-looking), so a separable monotone stand-in for the model.
    def fake_score(family: str, steps: Sequence[str]) -> float:
        st = list(steps)
        ends_ok = st[:1] == ["RECEIVE WAFER LOT"] and st[-1:] == ["SHIP LOT"]
        test_before_ship = (
            "WAFER SORT TEST" in st
            and "SHIP LOT" in st
            and st.index("WAFER SORT TEST") < st.index("SHIP LOT")
        )
        return float(2 * ends_ok + 2 * test_before_ship + 0.1 * len(st))

    seqs = {
        "valid_a": ["RECEIVE WAFER LOT", "WAFER SORT TEST", "SHIP LOT"],
        "valid_b": ["RECEIVE WAFER LOT", "WAFER SORT TEST", "CURE PASSIVATION", "SHIP LOT"],
        "bad_a": ["RECEIVE WAFER LOT", "SHIP LOT"],
        "bad_b": ["SHIP LOT", "RECEIVE WAFER LOT"],
    }
    labels = {"valid_a": 1, "valid_b": 1, "bad_a": 0, "bad_b": 0}
    items = [AnomalyInput(ex, "MOSFET", st) for ex, st in seqs.items()]

    det = LMLikelihoodDetector(score_fn=fake_score).calibrate(
        items, [labels[it.example_id] for it in items]
    )
    for it in items:
        iv, sc, rule = det.anomaly(it)
        assert 0.0 <= sc <= 1.0 and rule is None, (it.example_id, iv, sc)
        assert iv == labels[it.example_id], (it.example_id, iv, sc)

    # Matrix plumbing (GPU-free via an injected score_fn; logging off so no registry writes).
    fams = ["MOSFET"]
    out = id_ood_nll_matrix(
        {"fake-ckpt": "n/a"}, fams, score_fns={"fake-ckpt": fake_score}, log=False
    )
    cell = out["matrix"]["fake-ckpt"]["MOSFET"]
    assert set(cell) == {"mean_nll", "perplexity", "n"} and cell["n"] > 0, cell
    print("likelihood_lm.py smoke OK — LM detector verdicts + ID/OOD matrix structure (no torch)")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
