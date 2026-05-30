#!/usr/bin/env python3
"""Local OpenAI-compatible inference server for the Fab Copilot demo (Apple Silicon / MPS).

Why this exists: vLLM (`just serve`) needs CUDA and won't run on a Mac. The copilot only
does SHORT next-step generations (max_tokens ~24), which transformers on MPS handles fast.
This server loads MULTIPLE fab models and routes `POST /v1/chat/completions` by the request's
`model` field, so the copilot UI can let you SELECT which model runs — all inference local.

Run:
    uv run python scripts/serve_copilot_mac.py            # serves on http://localhost:8001
Models (name -> local path or HF id) come from ZO_COPILOT_MODELS (JSON) or the defaults below.
Each model is LAZY-loaded on first request, so startup is instant and unused models cost nothing.

The copilot points at it with:
    VITE_MODEL_BASE_URL=http://localhost:8001/v1
and sends `model: <name>` from the dropdown. GET /v1/models lists what's available.
"""

from __future__ import annotations

import json
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

# name -> local checkpoint dir OR Hugging Face repo id. Override with ZO_COPILOT_MODELS=<json>.
DEFAULT_MODELS: dict[str, str] = {
    "base-qwen": "Qwen/Qwen2.5-1.5B-Instruct",  # frozen base — the "before"
    "sft-fab-all": os.path.expanduser("~/zo-models/sft-fab-all"),  # fine-tuned — the "best"
}
MODELS: dict[str, str] = json.loads(os.environ.get("ZO_COPILOT_MODELS") or "null") or DEFAULT_MODELS

# ALL models use the fab COMPLETION prompt ("Process sequence: a | b | c | ") so the output is a
# clean pipe-separated step list, not chatty prose. The base (instruct) model then emits a clean
# *wrong* step instead of a rambling sentence that gets truncated at max_tokens — a far better
# base-vs-best contrast for the demo, and snapToVocab always has a real step to match.
def _completion_mode(name: str) -> bool:
    return True


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32
_loaded: dict[str, tuple] = {}


def _load(name: str):
    if name not in _loaded:
        src = MODELS.get(name)
        if src is None:
            raise KeyError(name)
        print(f"[serve] loading {name} <- {src} on {DEVICE} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=DTYPE).to(DEVICE).eval()
        _loaded[name] = (tok, model)
        print(f"[serve] {name} ready", flush=True)
    return _loaded[name]


def _build_prompt(tok, name: str, messages: list[dict]) -> str:
    user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if _completion_mode(name):
        # Reframe the copilot's chat prompt into the SFT training format and let the model continue.
        family = "MOSFET"
        steps = ""
        for line in user.splitlines():
            if line.lower().startswith("product family:"):
                family = line.split(":", 1)[1].strip()
            if line.lower().startswith(("process so far:", "process sequence:")):
                steps = line.split(":", 1)[1].strip()
        return f"Product family: {family}\nProcess sequence: {steps} | "
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


app = FastAPI(title="Fab Copilot local server")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": n, "object": "model"} for n in MODELS]}


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "models": list(MODELS), "loaded": list(_loaded)}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    name = body.get("model") or next(iter(MODELS))
    if name not in MODELS:
        name = next(iter(MODELS))
    messages = body.get("messages", [])
    max_new = int(body.get("max_tokens") or 24)
    tok, model = _load(name)
    prompt = _build_prompt(tok, name, messages)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": name,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }


if __name__ == "__main__":
    print(f"[serve] device={DEVICE} models={list(MODELS)}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("ZO_COPILOT_PORT", "8001")))
