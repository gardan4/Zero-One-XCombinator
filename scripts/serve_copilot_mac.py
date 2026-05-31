#!/usr/bin/env python3
"""Local OpenAI-compatible inference server for the Fab Copilot demo.

Routes `POST /v1/chat/completions` by the request's `model` field across THREE fab models, so the
demo can SELECT which one runs and compare them side by side:

  • deepseek-v4-flash  — hosted DeepSeek-V4-Flash via Featherless.ai (the model we evaluated zero-shot)
  • qwen-base          — frozen Qwen2.5-1.5B-Instruct, local on Apple Silicon / MPS (the "before")
  • sft-best           — our best fine-tuned checkpoint, local on MPS (the "after")

Why local for two of them: vLLM (`just serve`) needs CUDA and won't run on a Mac. The copilot does
SHORT generations, which transformers on MPS handles fast. DeepSeek-V4-Flash is too big to run on the
laptop, so that one is proxied to Featherless (credentials from `.env`: FEATHERLESS_API_KEY, or
FEATHERLESS_EMAIL/PASSWORD to log in and fetch one).

All three are prompted with the UNIFIED JSON format (numbered input → `{"reasoning":…, "steps":[…]}`),
so the server reframes the copilot's plain prompt into the trained system+user messages
(`zo_train.prompts.build_messages`) and parses the model's JSON. The response returns the next step
PLUS the model's own `reasoning` and ranked `alternates`, so the copilot can show the model's thoughts
next to the prediction.

Run:
    uv run python scripts/serve_copilot_mac.py            # serves on http://localhost:8001
The copilot points at it with VITE_MODEL_BASE_URL=http://localhost:8001/v1 and sends `model: <name>`.
GET /v1/models lists what's available; override the roster with ZO_COPILOT_MODELS=<json>.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForCausalLM, AutoTokenizer
from zo_common.env import load_dotenv
from zo_train.prompts import PromptItem, build_messages

load_dotenv()  # FEATHERLESS_* + any model-path overrides from the repo-root .env

# --- Model roster ----------------------------------------------------------
# Local models map name -> local checkpoint dir OR HF repo id (lazy-loaded on MPS).
LOCAL_MODELS: dict[str, str] = {
    "qwen-base": "Qwen/Qwen2.5-1.5B-Instruct",  # frozen base — the "before"
    "sft-best": os.path.expanduser("~/zo-models/sft-instruct-all"),  # fine-tuned — the "after"
}
# Remote models map name -> Featherless model id (proxied, never loaded locally).
REMOTE_MODELS: dict[str, str] = {
    "deepseek-v4-flash": os.environ.get("FEATHERLESS_MODEL") or "deepseek-ai/DeepSeek-V4-Flash",
}
# Display order in the copilot's picker (sft-best is the safe default — local, instant, no network).
MODEL_ORDER = ["deepseek-v4-flash", "qwen-base", "sft-best"]

_override = json.loads(os.environ.get("ZO_COPILOT_MODELS") or "null")
if isinstance(_override, dict):
    LOCAL_MODELS, REMOTE_MODELS = _override, {}
    MODEL_ORDER = list(_override)

MODELS: dict[str, str] = {**REMOTE_MODELS, **LOCAL_MODELS}
ORDERED = [n for n in MODEL_ORDER if n in MODELS] + [n for n in MODELS if n not in MODEL_ORDER]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32
_loaded: dict[str, tuple] = {}


def _load(name: str):
    if name not in _loaded:
        src = LOCAL_MODELS.get(name)
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


def _parse_json_answer(text: str) -> dict | None:
    """Parse the model's JSON answer → {steps, reasoning, valid, rule} (or None if unparseable)."""
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
        out: dict = {"reasoning": str(obj.get("reasoning") or "").strip()}
        steps = obj.get("steps")
        if isinstance(steps, list) and steps:
            out["steps"] = [str(x) for x in steps if str(x).strip()]
        if "valid" in obj:  # anomaly-style answer
            out["valid"] = bool(obj.get("valid"))
            out["rule"] = obj.get("rule")
            out.setdefault("steps", ["VALID" if obj.get("valid") else f"INVALID {obj.get('rule') or ''}".strip()])
        return out if out.get("steps") else None
    return None


# Featherless's free tier allows only ~1 concurrent request, so serialize all remote calls behind a
# lock — the Live-compare fan-out (and React-dev double-renders) would otherwise self-collide into 429s.
_remote_lock = threading.Lock()


def _remote_chat(name: str, messages: list[dict], max_new: int) -> str:
    """Proxy a chat to Featherless and return the raw assistant text (with the duplicate-`content`
    JSON repair). Serialized + fast-fail: one quick retry on a 429 instead of the minute-long backoff
    in zo_common.llm — a live demo card must never hang on a transient 429."""
    import httpx

    from zo_common.llm import _parse_chat_response, message_text
    from zo_eval.featherless import base_url, resolve_api_key

    url = base_url().rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {resolve_api_key()}"}
    payload = {
        "model": REMOTE_MODELS[name],
        "messages": messages,
        "temperature": 0,
        "max_tokens": max(256, max_new),
    }
    with _remote_lock:
        last = None
        attempts = 2  # 1 retry only — Featherless free tier has a hard quota; extra retries just burn it
        for attempt in range(attempts):
            last = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            if last.status_code != 429 or attempt == attempts - 1:
                break
            time.sleep(2.0)  # brief settle for a transient concurrency 429
        last.raise_for_status()
        return message_text(_parse_chat_response(last))


app = FastAPI(title="Fab Copilot local server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": n, "object": "model"} for n in ORDERED]}


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "models": ORDERED, "loaded": list(_loaded)}


# MPS / torch isn't safe for concurrent model LOADS or generate() calls from worker threads (the
# Live-compare fan-out hits two local models at once) — serialize all local GPU work behind one lock.
# This is separate from _remote_lock, so a local model and DeepSeek still run concurrently.
_local_lock = threading.Lock()


def _predict(name: str, family: str, steps: list[str], max_new: int) -> tuple[str, float | None]:
    """Run one model and return (raw_text, confidence|None). Blocking — call via run_in_threadpool."""
    messages = build_messages("nextstep", PromptItem(family, partial_sequence=steps))
    if name in REMOTE_MODELS:
        return _remote_chat(name, messages, max_new), None
    with _local_lock:
        tok, model = _load(name)
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
        try:
            confidence = _generation_confidence(new_ids, gen.scores)
        except Exception:
            confidence = None
    return raw, confidence


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    name = body.get("model") if body.get("model") in MODELS else ORDERED[0]
    family, steps = _parse_user(body.get("messages", []))
    # JSON answers need more room than the copilot's tiny default; reasoning can be a sentence or two.
    max_new = max(192, int(body.get("max_tokens") or 0))

    # Offload the blocking work (torch generate / sync remote call) to a worker thread so the event
    # loop stays free — the Live-compare page fans out to all 3 models at once and they run concurrently
    # instead of serializing behind one another.
    try:
        raw, confidence = await run_in_threadpool(_predict, name, family, steps, max_new)
    except Exception as exc:  # creds missing / network / load error — surface so the copilot falls back
        print(f"[serve] {name} failed: {type(exc).__name__}: {exc}", flush=True)
        return {"error": {"message": f"model {name} failed: {exc}"}}

    parsed = _parse_json_answer(raw)
    steps_out = parsed["steps"] if parsed else [raw.strip()]
    # content = the single next step (steps[0]) so the copilot's snapToVocab matches it; alternates
    # are the remaining ranked candidates; reasoning is the model's own chain-of-thought (often empty
    # for the fine-tuned model, rich for DeepSeek which reasons zero-shot).
    content = steps_out[0] if steps_out else raw.strip()
    reasoning = (parsed or {}).get("reasoning", "")

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": name,
        # non-standard fields the copilot reads for the confidence bar + the "model thoughts" panel:
        "confidence": round(confidence, 4) if confidence is not None else None,
        "reasoning": reasoning,
        "alternates": steps_out[1:5],
        "raw": raw.strip()[:1200],
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


if __name__ == "__main__":
    print(f"[serve] device={DEVICE} models={ORDERED}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("ZO_COPILOT_PORT", "8001")))
