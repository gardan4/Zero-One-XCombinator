"""LLM-backed predictors behind the shared ``Predictor`` interface.

- ``ServedLLMPredictor`` queries an OpenAI-compatible endpoint (vLLM via ``just serve``) using
  ``zo_common.llm.chat`` and pipes output through the ``predict`` normalizer. This is the path
  used to score a fine-tuned checkpoint (Streams 1/2). A ``chat_fn`` can be injected for testing.
- ``HFGeneratePredictor`` runs ``transformers.generate`` locally (lazy import) — for laptop/CPU
  smoke or environments where vLLM won't install.

Prompts mirror the training framing (``datagen.SEP`` = ``" | "``) so eval matches training; the
normalizer maps free text back onto the exact vocab and the ``"|"`` submission separator.

Anomaly SCORE: derived from VALID/INVALID token logprobs when the server returns them, else a
verdict-based fallback — **never None** (the ROC-AUC scorer drops the metric if any row lacks a score).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from zo_common.llm import chat as _default_chat
from zo_common.llm import content, message_text, token_logprobs
from zo_train.datagen import anomaly_example, completion_example, nextstep_example

from zo_eval.predict import extract_answer, parse_anomaly, parse_pipe_list, vocab
from zo_eval.submission import AnomalyInput, ValidInput

if TYPE_CHECKING:
    from zo_common.featherless import HFModelRef


# Prompts MUST match the instruction-tuning framing byte-for-byte (datagen.*_example), or a
# small instruct model won't follow them — divergent prompts collapsed next-step/anomaly to chance.
def _ns_prompt(item: ValidInput) -> str:
    return nextstep_example(item.family, list(item.partial_sequence), "")["prompt"]


def _cp_prompt(item: ValidInput) -> str:
    return completion_example(item.family, list(item.partial_sequence), [])["prompt"]


def _an_prompt(item: AnomalyInput) -> str:
    return anomaly_example(item.family, list(item.sequence), True)["prompt"]


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

    def __init__(self, model: str = "default", base_url: str | None = None, temperature: float = 0.0, chat_fn=None):
        self.model = model
        self.base_url = base_url
        self.temp = temperature
        self._chat = chat_fn or _default_chat
        self.vocab = vocab()

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
        ranked = parse_pipe_list(content(self._ask(_ns_prompt(item), 24)), self.vocab, strict=True)
        out: list[str] = []
        for s in ranked:  # dedupe, preserve rank, cap 5
            if s not in out:
                out.append(s)
        return out[:5]

    def complete(self, item: ValidInput) -> list[str]:
        txt = extract_answer(content(self._ask(_cp_prompt(item), 1024)))
        return parse_pipe_list(txt, self.vocab, strict=True)

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        resp = self._ask(_an_prompt(item), 64, logprobs=True, top_logprobs=5)
        is_valid, rule = parse_anomaly(content(resp))
        score = _valid_prob(resp)
        if score is None:
            score = 0.9 if is_valid else 0.1  # fallback — never None
        return (is_valid, score, rule)


class HFGeneratePredictor:
    """Local ``transformers.generate`` predictor (lazy import; for CPU/GPU smoke without a server)."""

    name = "hf"

    def __init__(self, model: str, device: str | None = None, max_new_tokens: int = 256):
        from zo_common.hub_inference import HubInferenceClient

        self._client = HubInferenceClient(model, device=device, max_new_tokens=max_new_tokens)
        self.vocab = vocab()

    def _gen(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return self._client.generate(prompt, max_new_tokens=max_new_tokens or self._client.max_new_tokens)

    def next_step(self, item: ValidInput) -> list[str]:
        # Training framing emits ONE next step → that's rank-1 (top-1 is the headline metric).
        seen: list[str] = []
        for s in parse_pipe_list(self._gen(_ns_prompt(item), 24), self.vocab, strict=True):
            if s not in seen:
                seen.append(s)
        return seen[:5]

    def complete(self, item: ValidInput) -> list[str]:
        return parse_pipe_list(extract_answer(self._gen(_cp_prompt(item), 1024)), self.vocab, strict=True)

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        is_valid, rule = parse_anomaly(self._gen(_an_prompt(item), 64))
        return (is_valid, 0.9 if is_valid else 0.1, rule)


class FeatherlessPredictor:
    """Track predictor backed by Featherless serverless inference (OpenAI-compatible).

    Pass ``model`` as an HF repo id (full checkpoint) or ``base:lora[:merged]`` segments:
    ``Qwen/Qwen2.5-1.5B-Instruct`` (full/base), or
    ``Qwen/Qwen2.5-1.5B-Instruct:XCombinator/sft-cot`` (LoRA — needs merged weights on HF).
    """

    name = "featherless"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        temperature: float = 0.0,
        route: str | None = None,
        api_key: str | None = None,
    ):
        from zo_common.featherless import (
            FeatherlessClient,
            featherless_chat_fn,
            resolve_featherless_model,
        )

        ref = _parse_featherless_model(model)
        self.model_id = resolve_featherless_model(ref)
        self.client = FeatherlessClient(
            self.model_id,
            route=route,  # type: ignore[arg-type]
            api_key=api_key,
        )
        self.temp = temperature
        self._chat = featherless_chat_fn(self.client)
        self.vocab = vocab()

    def _ask(self, prompt: str, max_tokens: int, **kw) -> dict:
        return self._chat(
            [{"role": "user", "content": prompt}],
            temperature=self.temp,
            max_tokens=max_tokens,
            **kw,
        )

    def _text(self, resp: dict) -> str:
        return message_text(resp)

    def next_step(self, item: ValidInput) -> list[str]:
        ranked = parse_pipe_list(self._text(self._ask(_ns_prompt(item), 24)), self.vocab, strict=True)
        out: list[str] = []
        for s in ranked:
            if s not in out:
                out.append(s)
        return out[:5]

    def complete(self, item: ValidInput) -> list[str]:
        txt = extract_answer(self._text(self._ask(_cp_prompt(item), 1024)))
        return parse_pipe_list(txt, self.vocab, strict=True)

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        resp = self._ask(_an_prompt(item), 64, logprobs=True, top_logprobs=5)
        is_valid, rule = parse_anomaly(self._text(resp))
        score = _valid_prob(resp)
        if score is None:
            score = 0.9 if is_valid else 0.1
        return (is_valid, score, rule)


def _parse_featherless_model(spec: str) -> HFModelRef:
    from zo_common.featherless import HFModelRef

    parts = [p.strip() for p in spec.split(":") if p.strip()]
    if len(parts) == 1:
        return HFModelRef(full=parts[0])
    if len(parts) == 2:
        base, second = parts
        return HFModelRef(base=base, lora=second)
    base, lora, merged = parts[0], parts[1], parts[2]
    return HFModelRef(base=base, lora=lora, merged=merged)


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


if __name__ == "__main__":  # pragma: no cover
    _smoke()
