"""LLM-backed predictors behind the shared ``Predictor`` interface.

- ``ServedLLMPredictor`` queries an OpenAI-compatible endpoint (vLLM via ``just serve``) using
  ``zo_common.llm.chat`` and pipes output through the ``predict`` normalizer. This is the path
  used to score a fine-tuned checkpoint (Streams 1/2). A ``chat_fn`` can be injected for testing.
- ``HFGeneratePredictor`` runs ``transformers.generate`` locally (lazy import) — for laptop/CPU
  smoke or environments where vLLM won't install.

Prompts mirror the training framing (``datagen.SEP`` = ``" | "``) so eval matches training; the
normalizer maps free text back onto the exact vocab and the ``"|"`` submission separator.

Anomaly SCORE: derived from VALID/INVALID token logprobs when the server returns them, else a
verdict-based fallback.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

from zo_common.llm import chat as _default_chat
from zo_common.llm import content, token_logprobs
from zo_train.datagen import (
    anomaly_context_example,
    anomaly_example,
    completion_context_example,
    completion_example,
    nextstep_context_example,
    nextstep_example,
)

from zo_eval.examples_trace import trace_from_llm_response
from zo_eval.predict import extract_answer, parse_anomaly, parse_pipe_list, vocab
from zo_eval.rules_context import build_messages
from zo_eval.submission import AnomalyInput, ValidInput


def _unique_vocab_ranked(raw: str, vocabulary: set[str], *, limit: int = 5) -> list[str]:
    ranked = parse_pipe_list(raw, vocabulary, strict=True)
    out: list[str] = []
    for step in ranked:
        if step not in out:
            out.append(step)
    return out[:limit]


# Prompts MUST match the instruction-tuning framing byte-for-byte (datagen.*_example), or a
# small instruct model won't follow them — divergent prompts collapsed next-step/anomaly to chance.
def _ns_prompt(item: ValidInput) -> str:
    return nextstep_example(item.family, list(item.partial_sequence), "")["prompt"]


def _cp_prompt(item: ValidInput) -> str:
    return completion_example(item.family, list(item.partial_sequence), [])["prompt"]


def _an_prompt(item: AnomalyInput) -> str:
    return anomaly_example(item.family, list(item.sequence), True)["prompt"]


# --- BASE-model prompts: the SFT framing above is too terse for an un-fine-tuned model, which has
# never seen this vocabulary. ``style="base"`` instead spells out all the relevant context
# (datagen.*_context_example): the family, the legal step vocabulary, recent-step descriptions, and
# (optionally) a reference recipe. The per-family context is loaded once from the data and cached.
@lru_cache(maxsize=8)
def _family_context(family: str) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """``(candidate_steps, reference_recipe, descriptions)`` for base prompting.

    Empty for an unseen OOD family (Task 4) — the prompt then simply omits those blocks, and the
    normalizer's lenient mode lets novel steps through. Set ``ZO_BASE_PROMPT_REFERENCE=0`` to drop
    the reference recipe (recommended when measuring learned ordering rather than recipe lookup).
    """
    from zo_train.datagen import load_descriptions
    from zo_train.fab import all_steps, canonical_steps

    fam = (family or "").upper()
    try:
        candidates: tuple[str, ...] = tuple(sorted(all_steps(fam)))
        reference: tuple[str, ...] = tuple(canonical_steps(fam))
        descriptions = load_descriptions(fam)
    except Exception:  # unseen family / missing CSVs → run without family-specific context
        candidates, reference, descriptions = (), (), {}
    if os.environ.get("ZO_BASE_PROMPT_REFERENCE", "1") == "0":
        reference = ()
    return candidates, reference, descriptions


def _ns_prompt_base(item: ValidInput) -> str:
    cand, ref, desc = _family_context(item.family)
    return nextstep_context_example(
        item.family, list(item.partial_sequence), "", candidates=cand, reference=ref, descriptions=desc
    )["prompt"]


def _cp_prompt_base(item: ValidInput) -> str:
    cand, ref, desc = _family_context(item.family)
    return completion_context_example(
        item.family, list(item.partial_sequence), [], candidates=cand, reference=ref, descriptions=desc
    )["prompt"]


def _an_prompt_base(item: AnomalyInput) -> str:
    _, _, desc = _family_context(item.family)
    return anomaly_context_example(item.family, list(item.sequence), True, descriptions=desc)["prompt"]


def _valid_prob(resp: dict) -> float | None:
    """P(valid) from the first VALID/INVALID token's logprobs; None if unavailable."""
    for t in token_logprobs(resp)[:8]:
        tok = (t.get("token") or "").strip().upper()
        if tok.startswith("VALID") or tok.startswith("INVALID"):
            cand = {(c.get("token") or "").strip().upper(): c.get("logprob") for c in t.get("top_logprobs", [])}
            cand.setdefault(tok, t.get("logprob"))
            pv = math.exp(cand["VALID"]) if cand.get("VALID") is not None else 0.0
            pi = math.exp(cand["INVALID"]) if cand.get("INVALID") is not None else 0.0
            return pv / (pv + pi) if (pv + pi) > 0 else None
    return None


class ServedLLMPredictor:
    """Queries a served (fine-tuned) model over the OpenAI-compatible endpoint."""

    name = "llm"

    def __init__(
        self,
        model: str = "default",
        base_url: str | None = None,
        temperature: float = 0.0,
        chat_fn=None,
        style: str = "sft",
    ):
        self.model = model
        self.base_url = base_url
        self.temp = temperature
        self._chat = chat_fn or _default_chat
        self.vocab = vocab()
        # "sft" → terse training framing (default); "base" → rich-context prompt for an
        # un-fine-tuned model. The name drives the predictor: tag so /compare keeps them distinct.
        self.style = style
        self.name = "base" if style == "base" else "llm"

    def _ask(self, prompt: str, max_tokens: int, **kw) -> dict:
        return self._chat(
            [{"role": "user", "content": prompt}],
            model=self.model,
            base_url=self.base_url,
            temperature=self.temp,
            max_tokens=max_tokens,
            **kw,
        )

    def next_step(self, item: ValidInput) -> list[str]:
        ranks, _ = self.next_step_with_trace(item)
        return ranks

    def next_step_with_trace(self, item: ValidInput) -> tuple[list[str], dict]:
        prompt = _ns_prompt_base(item) if self.style == "base" else _ns_prompt(item)
        resp = self._ask(prompt, 64 if self.style == "base" else 24)
        raw = content(resp)
        ranked = parse_pipe_list(raw, self.vocab, strict=True)
        out: list[str] = []
        for s in ranked:
            if s not in out:
                out.append(s)
        trace = {**trace_from_llm_response(raw), "model": self.model}
        return out[:5], trace

    def complete(self, item: ValidInput) -> list[str]:
        steps, _ = self.complete_with_trace(item)
        return steps

    def complete_with_trace(self, item: ValidInput) -> tuple[list[str], dict]:
        resp = self._ask(_cp_prompt_base(item) if self.style == "base" else _cp_prompt(item), 1024)
        raw = content(resp)
        txt = extract_answer(raw)
        steps = parse_pipe_list(txt, self.vocab, strict=True)
        return steps, {**trace_from_llm_response(raw), "model": self.model}

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        result, _ = self.anomaly_with_trace(item)
        return result

    def anomaly_with_trace(self, item: AnomalyInput) -> tuple[tuple[int, float, str | None], dict]:
        prompt = _an_prompt_base(item) if self.style == "base" else _an_prompt(item)
        resp = self._ask(prompt, 64, logprobs=True, top_logprobs=5)
        raw = content(resp)
        is_valid, rule = parse_anomaly(raw)
        lp_score = _valid_prob(resp)
        score = lp_score if lp_score is not None else (0.9 if is_valid else 0.1)
        trace = {
            **trace_from_llm_response(raw),
            "model": self.model,
            "valid_prob_from_logprobs": lp_score,
        }
        return (is_valid, score, rule), trace


class RulesContextLLMPredictor:
    """Zero-shot baseline: process rules in system context, frozen base instruct model.

    Unlike ``ServedLLMPredictor`` / ``HFGeneratePredictor``, prompts come from
    ``rules_context.build_messages`` (generation_rules digest), not SFT ``datagen`` framing.
    """

    name = "llm-zeroshot"

    def __init__(
        self,
        model: str = "default",
        base_url: str | None = None,
        temperature: float = 0.0,
        chat_fn=None,
        *,
        backend: str = "served",
        device: str | None = None,
        batch_size: int | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.temp = temperature
        self.backend = backend.strip().lower()
        self.vocab = vocab()
        self.batch_size = batch_size
        if self.backend == "hf":
            from zo_common.hub_inference import HubInferenceClient

            self._hf = HubInferenceClient(model, device=device, max_new_tokens=256, batch_size=batch_size)
            self._chat = None
        else:
            self._hf = None
            self._chat = chat_fn or _default_chat

    def _ask(self, task: str, item: ValidInput | AnomalyInput, max_tokens: int, **kw) -> dict | str:
        messages = build_messages(task, item)
        if self.backend == "hf":
            assert self._hf is not None
            return self._hf.chat(messages, max_new_tokens=max_tokens, temperature=self.temp, **kw)
        assert self._chat is not None
        return self._chat(
            messages,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temp,
            max_tokens=max_tokens,
            **kw,
        )

    def _content(self, resp: dict | str) -> str:
        return resp if isinstance(resp, str) else content(resp)

    def next_step(self, item: ValidInput) -> list[str]:
        ranks, _ = self.next_step_with_trace(item)
        return ranks

    def next_step_with_trace(self, item: ValidInput) -> tuple[list[str], dict]:
        raw = self._content(self._ask("nextstep", item, 64))
        out = _unique_vocab_ranked(raw, self.vocab)
        trace = {**trace_from_llm_response(raw), "model": self.model, "backend": self.backend}
        return out, trace

    def next_step_batch(self, items: list[ValidInput]) -> list[tuple[list[str], dict]]:
        if self.backend != "hf":
            return [self.next_step_with_trace(item) for item in items]
        assert self._hf is not None
        messages = [build_messages("nextstep", item) for item in items]
        raws = self._hf.chat_batch(
            messages,
            max_new_tokens=64,
            temperature=self.temp,
            batch_size=self.batch_size,
        )
        return [
            (
                _unique_vocab_ranked(raw, self.vocab),
                {**trace_from_llm_response(raw), "model": self.model, "backend": self.backend, "batched": True},
            )
            for raw in raws
        ]

    def complete(self, item: ValidInput) -> list[str]:
        steps, _ = self.complete_with_trace(item)
        return steps

    def complete_with_trace(self, item: ValidInput) -> tuple[list[str], dict]:
        raw = self._content(self._ask("completion", item, 1024))
        txt = extract_answer(raw)
        steps = parse_pipe_list(txt, self.vocab, strict=True)
        return steps, {**trace_from_llm_response(raw), "model": self.model, "backend": self.backend}

    def complete_batch(self, items: list[ValidInput]) -> list[tuple[list[str], dict]]:
        if self.backend != "hf":
            return [self.complete_with_trace(item) for item in items]
        assert self._hf is not None
        messages = [build_messages("completion", item) for item in items]
        raws = self._hf.chat_batch(
            messages,
            max_new_tokens=1024,
            temperature=self.temp,
            batch_size=self.batch_size,
        )
        return [
            (
                parse_pipe_list(extract_answer(raw), self.vocab, strict=True),
                {**trace_from_llm_response(raw), "model": self.model, "backend": self.backend, "batched": True},
            )
            for raw in raws
        ]

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        result, _ = self.anomaly_with_trace(item)
        return result

    def anomaly_with_trace(self, item: AnomalyInput) -> tuple[tuple[int, float, str | None], dict]:
        resp = self._ask("anomaly", item, 64, logprobs=True, top_logprobs=5)
        raw = self._content(resp)
        is_valid, rule = parse_anomaly(raw)
        lp_score = _valid_prob(resp) if isinstance(resp, dict) else None
        score = lp_score if lp_score is not None else (0.9 if is_valid else 0.1)
        trace = {
            **trace_from_llm_response(raw),
            "model": self.model,
            "backend": self.backend,
            "valid_prob_from_logprobs": lp_score,
        }
        return (is_valid, score, rule), trace

    def anomaly_batch(self, items: list[AnomalyInput]) -> list[tuple[tuple[int, float, str | None], dict]]:
        if self.backend != "hf":
            return [self.anomaly_with_trace(item) for item in items]
        assert self._hf is not None
        messages = [build_messages("anomaly", item) for item in items]
        raws = self._hf.chat_batch(
            messages,
            max_new_tokens=64,
            temperature=self.temp,
            batch_size=self.batch_size,
        )
        out: list[tuple[tuple[int, float, str | None], dict]] = []
        for raw in raws:
            is_valid, rule = parse_anomaly(raw)
            result = (is_valid, 0.9 if is_valid else 0.1, rule)
            trace = {
                **trace_from_llm_response(raw),
                "model": self.model,
                "backend": self.backend,
                "batched": True,
                "valid_prob_from_logprobs": None,
            }
            out.append((result, trace))
        return out


class HFGeneratePredictor:
    """Local ``transformers.generate`` predictor (lazy import; for CPU/GPU smoke without a server)."""

    name = "hf"

    def __init__(self, model: str, device: str | None = None, max_new_tokens: int = 256, style: str = "sft"):
        from zo_common.hub_inference import HubInferenceClient

        self._client = HubInferenceClient(model, device=device, max_new_tokens=max_new_tokens)
        self.vocab = vocab()
        self.style = style
        self.name = "base-hf" if style == "base" else "hf"

    def _gen(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return self._client.generate(prompt, max_new_tokens=max_new_tokens or self._client.max_new_tokens)

    def next_step(self, item: ValidInput) -> list[str]:
        prompt = _ns_prompt_base(item) if self.style == "base" else _ns_prompt(item)
        max_tok = 64 if self.style == "base" else 24
        return _unique_vocab_ranked(self._gen(prompt, max_tok), self.vocab)

    def next_step_batch(self, items: list[ValidInput]) -> list[list[str]]:
        max_tok = 64 if self.style == "base" else 24
        prompts = [
            [{"role": "user", "content": _ns_prompt_base(it) if self.style == "base" else _ns_prompt(it)}]
            for it in items
        ]
        raws = self._client.chat_batch(prompts, max_new_tokens=max_tok)
        return [_unique_vocab_ranked(raw, self.vocab) for raw in raws]

    def complete(self, item: ValidInput) -> list[str]:
        prompt = _cp_prompt_base(item) if self.style == "base" else _cp_prompt(item)
        return parse_pipe_list(extract_answer(self._gen(prompt, 1024)), self.vocab, strict=True)

    def complete_batch(self, items: list[ValidInput]) -> list[list[str]]:
        prompts = [
            [
                {
                    "role": "user",
                    "content": _cp_prompt_base(it) if self.style == "base" else _cp_prompt(it),
                }
            ]
            for it in items
        ]
        raws = self._client.chat_batch(prompts, max_new_tokens=1024)
        return [parse_pipe_list(extract_answer(raw), self.vocab, strict=True) for raw in raws]

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        prompt = _an_prompt_base(item) if self.style == "base" else _an_prompt(item)
        is_valid, rule = parse_anomaly(self._gen(prompt, 64))
        return (is_valid, 0.9 if is_valid else 0.1, rule)

    def anomaly_batch(self, items: list[AnomalyInput]) -> list[tuple[int, float, str | None]]:
        prompts = [
            [
                {
                    "role": "user",
                    "content": _an_prompt_base(it) if self.style == "base" else _an_prompt(it),
                }
            ]
            for it in items
        ]
        raws = self._client.chat_batch(prompts, max_new_tokens=64)
        out = []
        for raw in raws:
            is_valid, rule = parse_anomaly(raw)
            out.append((is_valid, 0.9 if is_valid else 0.1, rule))
        return out


def _smoke() -> None:  # pragma: no cover - manual (no server/GPU needed)
    def fake_chat(messages, **kw):
        prompt = messages[-1]["content"]
        if "Next process step" in prompt:
            c = "DEPOSIT BARRIER METAL | CLEAN AFTER VIA ETCH | DEVELOP PHOTORESIST | OXIDE ETCH | SHIP LOT"
        elif "Complete the remaining" in prompt:
            c = "<think>backbone</think>\nDEPOSIT METAL 1 | ANNEAL METAL 1 | SHIP LOT"
        else:
            c = "INVALID. RULE_DEP_NO_CLEAN"
        return {"choices": [{"message": {"content": c}}]}

    p = ServedLLMPredictor(chat_fn=fake_chat)
    vi = ValidInput("v1", "MOSFET", 0.6, ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"])
    ns = p.next_step(vi)
    assert ns == ["DEPOSIT BARRIER METAL", "CLEAN AFTER VIA ETCH", "DEVELOP PHOTORESIST", "OXIDE ETCH", "SHIP LOT"], ns
    cp = p.complete(vi)
    assert cp == ["DEPOSIT METAL 1", "ANNEAL METAL 1", "SHIP LOT"], cp
    iv, score, rule = p.anomaly(AnomalyInput("a1", "MOSFET", ["RECEIVE WAFER LOT", "SHIP LOT"]))
    assert iv == 0 and rule == "RULE_DEP_NO_CLEAN" and 0.0 <= score <= 1.0, (iv, score, rule)
    print(f"predict_llm.py smoke OK — next_step={len(ns)} complete={len(cp)} anomaly=({iv}, {score}, {rule})")


def _smoke_zeroshot() -> None:  # pragma: no cover - manual
    def fake_chat(messages, **kw):
        user = messages[-1]["content"]
        assert messages[0]["role"] == "system"
        assert "RULE_DEP_NO_CLEAN" in messages[0]["content"]
        if "next process step" in user.lower():
            c = "DEPOSIT BARRIER METAL | CLEAN AFTER VIA ETCH | DEVELOP PHOTORESIST"
        elif "Complete the remaining" in user:
            c = "DEPOSIT METAL 1 | ANNEAL METAL 1 | SHIP LOT"
        else:
            c = "INVALID. RULE_SHIP_BEFORE_TEST"
        return {"choices": [{"message": {"content": c}}]}

    p = RulesContextLLMPredictor(chat_fn=fake_chat)
    vi = ValidInput("v1", "MOSFET", 0.6, ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"])
    assert p.next_step(vi)[0] == "DEPOSIT BARRIER METAL"
    assert "SHIP LOT" in p.complete(vi)
    iv, score, rule = p.anomaly(AnomalyInput("a1", "MOSFET", ["RECEIVE WAFER LOT", "SHIP LOT"]))
    assert iv == 0 and rule == "RULE_SHIP_BEFORE_TEST"
    print("RulesContextLLMPredictor smoke OK")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
    _smoke_zeroshot()
