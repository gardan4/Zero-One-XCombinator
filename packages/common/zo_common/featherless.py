"""Featherless serverless inference for Hugging Face models.

Featherless exposes an OpenAI-compatible API (``https://api.featherless.ai/v1``) and is also
available via the Hugging Face inference router (``https://router.huggingface.co/featherless-ai``).
Our fine-tunes land on the ``XCombinator`` org; point ``HFModelRef.full`` or ``.merged`` at the
published repo once weights are pushed.

**LoRA note:** Featherless loads full safetensors checkpoints only — not adapter-only repos.
For a LoRA run, merge the adapter into the base weights and push the merged checkpoint to HF
(see ``resolve_featherless_model``). Set ``HFModelRef.merged`` to that repo, or push merged
weights into the same repo as the adapter so resolution can detect full weights automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from zo_common.llm import chat as _chat
from zo_common.llm import message_text as _message_text

# OpenAI-compatible bases (chat() appends ``/chat/completions``).
FEATHERLESS_DIRECT_URL = "https://api.featherless.ai/v1"
FEATHERLESS_HF_ROUTER_URL = "https://router.huggingface.co/featherless-ai"

Route = Literal["direct", "hf-router"]
_DEFAULT_ROUTE: Route = "direct"


def featherless_api_key() -> str:
    """Featherless API key, or HF token when routing through the HF inference router."""
    return (
        os.environ.get("FEATHERLESS_API_KEY")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    )


def featherless_base_url(route: Route | None = None) -> str:
    route = route or os.environ.get("FEATHERLESS_ROUTE", _DEFAULT_ROUTE)  # type: ignore[assignment]
    if route == "hf-router":
        return os.environ.get("FEATHERLESS_HF_ROUTER_URL", FEATHERLESS_HF_ROUTER_URL)
    return os.environ.get("FEATHERLESS_BASE_URL", FEATHERLESS_DIRECT_URL)


@dataclass(frozen=True)
class HFModelRef:
    """Which HF repo(s) to run on Featherless.

    - ``full``: a merged full checkpoint (Stream 1 full fine-tunes).
    - ``base`` + ``lora``: LoRA training metadata; inference uses ``merged`` or a lora repo that
      already contains full weights.
    - ``merged``: explicit merged checkpoint repo (preferred for LoRA eval).
    """

    full: str | None = None
    base: str | None = None
    lora: str | None = None
    merged: str | None = None

    @classmethod
    def from_training(
        cls,
        *,
        base_model: str,
        lora: bool,
        hub_repo: str | None = None,
        merged_repo: str | None = None,
    ) -> HFModelRef:
        """Build a ref from a training config + optional HF push targets."""
        if hub_repo and not lora:
            return cls(full=hub_repo)
        if merged_repo:
            return cls(base=base_model, lora=hub_repo, merged=merged_repo)
        if hub_repo and lora:
            return cls(base=base_model, lora=hub_repo)
        return cls(full=base_model if not lora else None, base=base_model if lora else None)


def _hf_list_files(repo_id: str, token: str | None = None) -> list[str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    resp = httpx.get(url, headers=headers, timeout=30.0)
    if resp.status_code == 404:
        url = f"https://huggingface.co/api/models/{repo_id}/tree/master"
        resp = httpx.get(url, headers=headers, timeout=30.0)
    resp.raise_for_status()
    return [item["path"] for item in resp.json()]


def hub_has_full_weights(repo_id: str, token: str | None = None) -> bool:
    """True when the HF repo contains a full model (not adapter-only)."""
    try:
        files = _hf_list_files(repo_id, token=token)
    except httpx.HTTPError:
        return False
    names = {f.rsplit("/", 1)[-1] for f in files}
    if "adapter_config.json" in names and "adapter_model.safetensors" in names:
        if "model.safetensors" in names or any(n.startswith("model-") and n.endswith(".safetensors") for n in names):
            return True
        return False
    return bool(
        "model.safetensors" in names
        or "pytorch_model.bin" in names
        or any(n.startswith("model-") and n.endswith(".safetensors") for n in names)
    )


def resolve_featherless_model(ref: HFModelRef, hf_token: str | None = None) -> str:
    """Return the HF model id to pass to Featherless chat/completions."""
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if ref.full:
        return ref.full
    if ref.merged:
        return ref.merged
    if ref.lora and hub_has_full_weights(ref.lora, token=token):
        return ref.lora
    if ref.lora:
        raise ValueError(
            f"LoRA repo {ref.lora!r} has adapter weights only. Featherless requires a merged "
            f"full checkpoint (safetensors). Merge the adapter onto {ref.base!r}, push to HF, "
            "then set HFModelRef.merged=<repo> or push merged weights into the lora repo."
        )
    if ref.base:
        return ref.base
    raise ValueError("HFModelRef needs full=, merged=, or a lora repo with full weights.")


def message_text(resp: dict[str, Any], *, include_reasoning: bool = True) -> str:
    """Extract assistant text from a chat completion (handles reasoning models)."""
    return _message_text(resp, include_reasoning=include_reasoning)


class FeatherlessClient:
    """OpenAI-compatible client for Featherless-hosted HF models."""

    def __init__(
        self,
        model: str | HFModelRef,
        *,
        route: Route | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.model_id = resolve_featherless_model(model) if isinstance(model, HFModelRef) else model
        self.route = route
        self.base_url = base_url or featherless_base_url(route)
        self.api_key = api_key or featherless_api_key() or None
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _chat(
            messages,
            model=self.model_id,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            timeout=self.timeout,
            **kwargs,
        )

    def complete(
        self,
        prompt: str,
        *,
        include_reasoning: bool = True,
        **kwargs: Any,
    ) -> str:
        resp = self.chat([{"role": "user", "content": prompt}], **kwargs)
        return message_text(resp, include_reasoning=include_reasoning)

    def completion_prompt(self, user_prompt: str) -> str:
        """Fab track: next-step / completion task prompt."""
        return user_prompt

    def reasoning_prompt(self, user_prompt: str) -> str:
        """Fab track: anomaly / CoT task prompt."""
        return user_prompt


def featherless_chat_fn(client: FeatherlessClient):
    """Return a ``chat(messages, **kw) -> dict`` suitable for ``ServedLLMPredictor(chat_fn=...)``."""

    def _fn(messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        return client.chat(messages, **kwargs)

    return _fn


def _integration_smoke() -> None:  # pragma: no cover - manual / CI with FEATHERLESS_API_KEY
    from zo_train.datagen import SEP

    key = featherless_api_key()
    if not key:
        raise SystemExit("Set FEATHERLESS_API_KEY (or HF_TOKEN for hf-router) to run the smoke.")

    client = FeatherlessClient("Qwen/Qwen2.5-1.5B-Instruct")
    vi_prompt = (
        f"Product family: MOSFET\nProcess so far: {SEP.join(['RECEIVE WAFER LOT', 'LOT IDENTIFICATION'])}\n\n"
        "List the 5 most likely next process steps, best first, pipe-separated."
    )
    completion = client.complete(vi_prompt, max_tokens=128)
    print(f"[full/base] next-step completion ({len(completion)} chars): {completion[:200]!r}", flush=True)

    reason_client = FeatherlessClient("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    anomaly_prompt = (
        f"Product family: MOSFET\nProcess sequence: {SEP.join(['RECEIVE WAFER LOT', 'SHIP LOT'])}\n\n"
        "Is this a valid process sequence? Answer VALID or INVALID; if INVALID, name the rule id."
    )
    reasoning = reason_client.complete(anomaly_prompt, max_tokens=512)
    safe = reasoning[:300].encode("ascii", errors="replace").decode("ascii")
    print(f"[reasoning] anomaly ({len(reasoning)} chars): {safe!r}", flush=True)

    # LoRA path: base + adapter-only repo should fail resolution with a clear message.
    try:
        resolve_featherless_model(HFModelRef(base="Qwen/Qwen2.5-1.5B-Instruct", lora="trl-lib/lora-gpt2"))
    except ValueError as exc:
        print(f"[lora-only] resolve correctly rejected: {exc}")
    else:
        raise AssertionError("expected ValueError for adapter-only repo")

    # LoRA path stand-in: a public full finetune of the same base (merged weights on HF).
    merged_ref = HFModelRef(
        base="Qwen/Qwen2.5-1.5B-Instruct",
        lora="Qwen/Qwen2.5-1.5B-Instruct",  # same repo has full weights
    )
    merged_id = resolve_featherless_model(merged_ref)
    merged_client = FeatherlessClient(merged_id)
    merged_out = merged_client.complete(vi_prompt, max_tokens=64)
    print(f"[lora/merged stand-in] model={merged_id}: {merged_out[:120]!r}")
    print("featherless smoke OK")


if __name__ == "__main__":  # pragma: no cover
    _integration_smoke()
