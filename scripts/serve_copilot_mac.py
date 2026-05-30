#!/usr/bin/env python3
"""Local OpenAI-compatible inference server for the Fab Copilot demo (Apple Silicon / MPS).

Why this exists: vLLM (`just serve`) needs CUDA and won't run on a Mac. The copilot does SHORT
next-step generations, which transformers on MPS handles fast. This server loads MULTIPLE fab models
and routes `POST /v1/chat/completions` by the request's `model` field, so the demo can SELECT which
model runs (base vs best) — all inference local.

The models are trained on the UNIFIED JSON format (numbered input → `{"reasoning":…, "steps":[…]}`),
so this server reframes the copilot's plain prompt into the trained system+user messages
(`zo_train.prompts.build_messages`) and parses the model's JSON to return a clean next step the
copilot's snapToVocab can match.

Run:
    uv run python scripts/serve_copilot_mac.py            # serves on http://localhost:8001
Models (name -> local path or HF id) from ZO_COPILOT_MODELS (JSON) or the defaults below; each is
LAZY-loaded on first request. The copilot points at it with VITE_MODEL_BASE_URL=http://localhost:8001/v1
and sends `model: <name>`. GET /v1/models lists what's available.
"""

from __future__ import annotations

import json
import math
import os
import re
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

from zo_train.prompts import PromptItem, build_messages

# name -> local checkpoint dir OR Hugging Face repo id. Override with ZO_COPILOT_MODELS=<json>.
DEFAULT_MODELS: dict[str, str] = {
    "base-qwen": "Qwen/Qwen2.5-1.5B-Instruct",  # frozen base — the "before"
    "sft-best": os.path.expanduser("~/zo-models/sft-instruct-all"),  # fine-tuned — the "after"
}
MODELS: dict[str, str] = json.loads(os.environ.get("ZO_COPILOT_MODELS") or "null") or DEFAULT_MODELS

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


def _generation_confidence(new_ids, scores) -> float:
    """Geometric mean of the greedy token probabilities over the generation — the model's own
    certainty about what it produced. Clamped to [0.05, 0.99]."""
    if not scores:
        return 0.6
    logp, n = 0.0, 0
    for i, tok_id in enumerate(new_ids.tolist()):
        if i >= len(scores):
            break
        probs = torch.softmax(scores[i][0].float(), dim=-1)
        logp += math.log(float(probs[tok_id].clamp_min(1e-9)))
        n += 1
    return 0.6 if n == 0 else max(0.05, min(0.99, math.exp(logp / n)))


def _parse_user(messages: list[dict]) -> tuple[str, list[str]]:
    """Pull product family + the steps-so-far out of the copilot's plain prompt."""
    user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    family, steps_str = "MOSFET", ""
    for line in user.splitlines():
        low = line.lower()
        if low.startswith("product family:"):
            family = line.split(":", 1)[1].strip() or family
        if low.startswith(("process so far:", "process sequence:", "partial sequence:")):
            steps_str = line.split(":", 1)[1].strip()
    steps = [s.strip() for s in steps_str.split("|") if s.strip()]
    return family, steps


def _extract_steps(text: str) -> list[str] | None:
    """Parse the model's JSON answer → list of step strings (or None if unparseable)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(t[s : e + 1])
        except Exception:
            return None
        steps = obj.get("steps")
        if isinstance(steps, list) and steps:
            return [str(x) for x in steps if str(x).strip()]
        if "valid" in obj:  # anomaly-style answer
            return ["VALID" if obj.get("valid") else f"INVALID {obj.get('rule') or ''}".strip()]
    return None


app = FastAPI(title="Fab Copilot local server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": n, "object": "model"} for n in MODELS]}


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "models": list(MODELS), "loaded": list(_loaded)}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    name = body.get("model") if body.get("model") in MODELS else next(iter(MODELS))
    family, steps = _parse_user(body.get("messages", []))
    # JSON answers need more room than the copilot's tiny default; reasoning is empty for next-step.
    max_new = max(64, int(body.get("max_tokens") or 0))
    tok, model = _load(name)

    messages = build_messages("nextstep", PromptItem(family, partial_sequence=steps))
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    new_ids = gen.sequences[0][inputs["input_ids"].shape[1] :]
    raw = tok.decode(new_ids, skip_special_tokens=True)
    parsed = _extract_steps(raw)
    # Return the single next step (steps[0]) as clean content so the copilot's snapToVocab matches it;
    # fall back to the raw text if the model didn't emit valid JSON.
    content = parsed[0] if parsed else raw.strip()
    try:
        confidence = _generation_confidence(new_ids, gen.scores)
    except Exception:
        confidence = 0.75

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": name,
        "confidence": round(confidence, 4),  # non-standard: copilot reads this for the confidence bar
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


if __name__ == "__main__":
    print(f"[serve] device={DEVICE} models={list(MODELS)}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("ZO_COPILOT_PORT", "8001")))
