"""Local Hugging Face checkpoint inference (full weights or LoRA adapter).

Use this for **XCombinator** (and any other HF) fine-tunes. Loads weights via
``transformers`` (+ ``peft`` for LoRA). Works for private org repos when ``HF_TOKEN`` is set.

**Minimal setup (no uv, no full monorepo):**

.. code-block:: bash

   python -m pip install -r requirements-inference.txt
   python scripts/hub_infer.py --prompt "Say hello"

Set ``HF_TOKEN`` in the environment or in ``.env`` at the repo root for private models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zo_common.hf_hub_util import hub_has_full_weights

_INFERENCE_DEPS: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("huggingface_hub", "huggingface_hub"),
)


def ensure_inference_deps() -> None:
    """Raise ``SystemExit`` with a pip one-liner when ML deps are missing."""
    missing = [pip for mod, pip in _INFERENCE_DEPS if not _can_import(mod)]
    if not missing:
        return
    root = Path(__file__).resolve().parents[3]
    req = root / "requirements-inference.txt"
    hint = f"python -m pip install -r {req}" if req.is_file() else f"python -m pip install {' '.join(missing)}"
    raise SystemExit(
        "Missing inference packages: "
        + ", ".join(missing)
        + f"\nInstall from the repo root:\n  {hint}\n"
        "Then run: python scripts/hub_infer.py"
    )


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _load_dotenv(path: Path | None = None) -> None:
    env_file = path or Path(__file__).resolve().parents[3] / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, val.strip().strip('"').strip("'"))

def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


@dataclass(frozen=True)
class HubModelSpec:
    """HF repo and/or local path; ``base_model`` required for adapter-only repos."""

    repo_id: str | None = None
    local_path: str | None = None
    base_model: str | None = None

    @classmethod
    def parse(cls, model: str, *, base_model: str | None = None) -> HubModelSpec:
        """Parse ``org/name``, a local path, or ``base:adapter`` (colon-separated HF ids)."""
        if os.path.isdir(model) or model.startswith(("/", ".", "~")) or (len(model) > 1 and model[1] == ":"):
            return cls(local_path=model, base_model=base_model)
        if model.count(":") == 1 and "/" in model:
            base, adapter = model.split(":", 1)
            if "/" in base and "/" in adapter:
                return cls(repo_id=adapter.strip(), base_model=base.strip())
        return cls(repo_id=model, base_model=base_model)


def _normalize_base_model(base: str | None) -> str | None:
    """Map Leonardo scratch paths in adapter_config to public HF hub ids."""
    if not base:
        return base
    if not base.startswith(("/","\\")) and os.path.isdir(base):
        return base
    if base.startswith(("/", "\\")) or base.startswith("leonardo"):
        from pathlib import Path as _Path

        leaf = _Path(base.rstrip("/\\")).name
        override = os.environ.get("ZO_INFER_BASE_MODEL")
        if override:
            return override
        # Common bases used in our Leonardo configs
        known = {
            "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        }
        return known.get(leaf, f"Qwen/{leaf}")
    return base


def resolve_hub_spec(spec: HubModelSpec, token: str | None = None) -> HubModelSpec:
    """Fill in ``base_model`` from ``adapter_config.json`` when missing."""
    token = token or hf_token()
    path = spec.local_path or spec.repo_id
    if not path:
        raise ValueError("HubModelSpec needs repo_id or local_path")
    base = _normalize_base_model(spec.base_model)
    if base:
        return HubModelSpec(repo_id=spec.repo_id, local_path=spec.local_path, base_model=base)
    if hub_has_full_weights(path, token=token):
        return spec
    # LoRA adapter — read base from adapter_config
    base = _normalize_base_model(_adapter_base_model(path, token=token))
    if not base:
        raise ValueError(
            f"{path!r} is adapter-only; pass base_model= (e.g. Qwen/Qwen2.5-0.5B-Instruct)."
        )
    return HubModelSpec(repo_id=spec.repo_id, local_path=spec.local_path, base_model=base)


def _adapter_base_model(repo_or_path: str, token: str | None = None) -> str | None:
    import json

    if os.path.isdir(repo_or_path):
        cfg = os.path.join(repo_or_path, "adapter_config.json")
        if os.path.isfile(cfg):
            return json.loads(open(cfg, encoding="utf-8").read()).get("base_model_name_or_path")
        return None
    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo_or_path, "adapter_config.json", token=token)
        return json.loads(open(cfg_path, encoding="utf-8").read()).get("base_model_name_or_path")
    except Exception:
        return None


class HubInferenceClient:
    """Generate completions from a HF full checkpoint or LoRA adapter."""

    def __init__(
        self,
        model: str | HubModelSpec,
        *,
        base_model: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
        token: str | None = None,
    ):
        self.token = token or hf_token()
        self.spec = resolve_hub_spec(
            model if isinstance(model, HubModelSpec) else HubModelSpec.parse(model, base_model=base_model),
            token=self.token,
        )
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tok = None

    @property
    def model_id(self) -> str:
        return self.spec.local_path or self.spec.repo_id or ""

    def _load(self) -> None:
        if self._model is not None:
            return
        ensure_inference_deps()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device or _default_device()
        self._device = device
        tok_src = self.spec.base_model or self.spec.local_path or self.spec.repo_id
        adapter = None
        if self.spec.base_model and (self.spec.repo_id or self.spec.local_path):
            adapter = self.spec.local_path or self.spec.repo_id
            model_src = self.spec.base_model
        else:
            model_src = self.spec.local_path or self.spec.repo_id

        kwargs: dict[str, Any] = {"token": self.token}
        if os.path.isdir(str(model_src)):
            kwargs["local_files_only"] = True

        self._tok = AutoTokenizer.from_pretrained(tok_src or model_src, **kwargs)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        if adapter:
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(model_src, **kwargs)
            peft_kw = dict(kwargs)
            if os.path.isdir(str(adapter)):
                peft_kw["local_files_only"] = True
            self._model = PeftModel.from_pretrained(base, adapter, **peft_kw)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_src, **kwargs)

        self._model.to(device).eval()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        import torch

        self._load()
        messages = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self._tok(text, return_tensors="pt").to(self._device)
        gen_kw: dict[str, Any] = {"max_new_tokens": max_new_tokens or self.max_new_tokens}
        if temperature <= 0:
            gen_kw["do_sample"] = False
        else:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
        with torch.no_grad():
            out = self._model.generate(**ids, **gen_kw)
        return self._tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        import torch

        self._load()
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self._tok(text, return_tensors="pt").to(self._device)
        gen_kw: dict[str, Any] = {"max_new_tokens": kwargs.pop("max_new_tokens", self.max_new_tokens)}
        temperature = kwargs.pop("temperature", 0.0)
        if temperature <= 0:
            gen_kw["do_sample"] = False
        else:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
        with torch.no_grad():
            out = self._model.generate(**ids, **gen_kw)
        return self._tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)


def hub_chat_fn(client: HubInferenceClient):
    """OpenAI-shaped ``chat(messages, **kw) -> dict`` for ``ServedLLMPredictor(chat_fn=...)``."""

    def _fn(messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        max_tokens = kwargs.pop("max_tokens", client.max_new_tokens)
        text = client.chat(messages, max_new_tokens=max_tokens, **kwargs)
        return {"choices": [{"message": {"content": text}}]}

    return _fn


def default_xcombinator_model() -> str:
    return os.environ.get("ZO_INFER_MODEL") or "XCombinator/leonardo-smoke-qwen-0.5b-lora"


def _smoke() -> None:  # pragma: no cover
    _load_dotenv()
    ensure_inference_deps()
    model = default_xcombinator_model()
    print(f"HubInferenceClient smoke — model={model!r}")
    client = HubInferenceClient(model)
    print(f"  resolved base={client.spec.base_model!r} adapter={client.spec.repo_id!r}")
    out = client.complete("Say hello in one short sentence.", max_new_tokens=32)
    print(f"  completion ({len(out)} chars): {out!r}")
    print("hub_inference smoke OK")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
