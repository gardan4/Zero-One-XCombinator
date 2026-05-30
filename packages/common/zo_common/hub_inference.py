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
import re
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


_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]", re.IGNORECASE)

# Conservative generation batch sizes for bf16 on a 64 GB A100 (short ≤128, medium ≤512, long >512).
_BATCH_BY_PARAMS: dict[float, dict[str, int]] = {
    0.5: {"short": 32, "medium": 16, "long": 8},
    1.5: {"short": 16, "medium": 8, "long": 4},
    3.0: {"short": 8, "medium": 4, "long": 2},
    7.0: {"short": 4, "medium": 2, "long": 1},
    13.0: {"short": 2, "medium": 1, "long": 1},
}


def parse_model_param_b(model: str | None) -> float | None:
    """Best-effort parameter count from a HF id or local path (e.g. ``Qwen2.5-1.5B-Instruct`` → 1.5)."""
    if not model:
        return None
    for part in Path(model.replace("\\", "/")).parts[::-1]:
        m = _PARAM_RE.search(part)
        if m:
            return float(m.group(1))
    m = _PARAM_RE.search(model)
    return float(m.group(1)) if m else None


def _token_tier(max_new_tokens: int) -> str:
    if max_new_tokens <= 128:
        return "short"
    if max_new_tokens <= 512:
        return "medium"
    return "long"


def _snap_param_b(params_b: float) -> float:
    known = sorted(_BATCH_BY_PARAMS)
    return min(known, key=lambda k: abs(k - params_b))


def cuda_vram_gb(device: str | None = None) -> float | None:
    """Return total VRAM (GiB) for the active CUDA device, if available."""
    try:
        import torch
    except ImportError:
        return None
    dev = device or ("cuda" if torch.cuda.is_available() else None)
    if not dev or not str(dev).startswith("cuda"):
        return None
    idx = 0 if dev == "cuda" else int(str(dev).split(":")[-1])
    if idx >= torch.cuda.device_count():
        return None
    return float(torch.cuda.get_device_properties(idx).total_memory) / (1024**3)


def default_infer_batch_size(
    model: str | None,
    *,
    max_new_tokens: int = 256,
    device: str | None = None,
    vram_gb: float | None = None,
    params_b: float | None = None,
) -> int:
    """Pick a conservative HF generation batch size from model size, output length, and VRAM."""
    dev = device or _default_device()
    if dev == "cpu" and vram_gb is None:
        return 1
    params = params_b or parse_model_param_b(model) or 1.5
    tier = _token_tier(max_new_tokens)
    base = _BATCH_BY_PARAMS[_snap_param_b(params)][tier]
    vram = vram_gb if vram_gb is not None else cuda_vram_gb(dev if dev.startswith("cuda") else None)
    if vram is None:
        return base
    scaled = max(1, int(base * vram / 64.0))
    return min(base, scaled) if vram < 64 else base


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
        batch_size: int | None = None,
        token: str | None = None,
    ):
        self.token = token or hf_token()
        self.spec = resolve_hub_spec(
            model if isinstance(model, HubModelSpec) else HubModelSpec.parse(model, base_model=base_model),
            token=self.token,
        )
        self.device = device
        self.max_new_tokens = max_new_tokens
        explicit = batch_size if batch_size is not None else os.environ.get("ZO_TRACK_BATCH_SIZE")
        self._batch_size_override = int(explicit) if explicit not in (None, "", "auto") else None
        self._model = None
        self._tok = None
        self._last_batch_log: tuple[int, int] | None = None

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
        if device == "cuda":
            dtype = os.environ.get("ZO_INFER_TORCH_DTYPE", "bfloat16").lower()
            if dtype in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
                kwargs["torch_dtype"] = torch.bfloat16
            elif dtype in {"fp16", "float16", "half"}:
                kwargs["torch_dtype"] = torch.float16
        if os.path.isdir(str(model_src)):
            kwargs["local_files_only"] = True

        self._tok = AutoTokenizer.from_pretrained(tok_src or model_src, **kwargs)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"

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

    def _generate_texts(
        self,
        texts: list[str],
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        batch_size: int | None = None,
    ) -> list[str]:
        import torch

        self._load()
        if not texts:
            return []
        gen_kw: dict[str, Any] = {"max_new_tokens": max_new_tokens or self.max_new_tokens}
        if temperature <= 0:
            gen_kw["do_sample"] = False
        else:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
        outputs: list[str] = []
        ntok = max_new_tokens or self.max_new_tokens
        bs = self._resolve_batch_size(max_new_tokens=ntok, override=batch_size)
        log_key = (bs, ntok)
        if self._last_batch_log != log_key:
            print(
                f"[hub_inference] batch_size={bs} model={self.model_id!r} max_new_tokens={ntok}",
                flush=True,
            )
            self._last_batch_log = log_key
        for start in range(0, len(texts), bs):
            chunk = texts[start : start + bs]
            ids = self._tok(chunk, return_tensors="pt", padding=True).to(self._device)
            prompt_width = ids["input_ids"].shape[1]
            with torch.inference_mode():
                out = self._model.generate(**ids, **gen_kw)
            for row in out:
                outputs.append(self._tok.decode(row[prompt_width:], skip_special_tokens=True))
        return outputs

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

    def chat_batch(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        batch_size: int | None = None,
        **_: Any,
    ) -> list[str]:
        self._load()
        texts = [
            self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        return self._generate_texts(
            texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
        )

    def _resolve_batch_size(self, *, max_new_tokens: int, override: int | None = None) -> int:
        if override is not None:
            return max(1, override)
        if self._batch_size_override is not None:
            return max(1, self._batch_size_override)
        base = self.spec.base_model or self.model_id
        device = getattr(self, "_device", None) or self.device
        return default_infer_batch_size(base, max_new_tokens=max_new_tokens, device=device)


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
