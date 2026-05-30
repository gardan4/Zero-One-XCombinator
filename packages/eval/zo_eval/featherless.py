"""Featherless.ai hosted inference for track eval (zero-shot rules-in-context).

Uses the same ``RulesContextLLMPredictor`` + ``run_track`` path as Leonardo judge eval,
but calls Featherless's OpenAI-compatible API instead of local vLLM/transformers.

Env (see ``.env.example``):
  FEATHERLESS_API_KEY   — from https://featherless.ai/account/api-keys (or auto-fetched via login)
  FEATHERLESS_BASE_URL  — default https://api.featherless.ai/v1
  FEATHERLESS_MODEL     — default deepseek-ai/DeepSeek-V4-Flash
  FEATHERLESS_EMAIL / FEATHERLESS_PASSWORD — optional; used to fetch API key when unset
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
LOGIN_URL = "https://featherless.ai/api/auth/login"
KEYS_URL = "https://featherless.ai/api/api-keys"


class FeatherlessConfigError(RuntimeError):
    """Missing or invalid Featherless credentials."""


def base_url() -> str:
    return (os.environ.get("FEATHERLESS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def default_model() -> str:
    return os.environ.get("FEATHERLESS_MODEL") or DEFAULT_MODEL


def _login(email: str, password: str) -> httpx.Client:
    client = httpx.Client(follow_redirects=True, timeout=30.0)
    resp = client.post(LOGIN_URL, json={"email": email, "password": password})
    resp.raise_for_status()
    return client


def fetch_api_key_from_login(email: str | None = None, password: str | None = None) -> str:
    """Return the first API key for the account (creates none — use dashboard for new keys)."""
    email = (email or os.environ.get("FEATHERLESS_EMAIL") or "").strip()
    password = (password or os.environ.get("FEATHERLESS_PASSWORD") or "").strip()
    if not email or not password:
        raise FeatherlessConfigError(
            "Set FEATHERLESS_API_KEY, or FEATHERLESS_EMAIL + FEATHERLESS_PASSWORD to fetch one."
        )
    with _login(email, password) as client:
        resp = client.get(KEYS_URL)
        resp.raise_for_status()
        keys = resp.json()
    if not keys:
        raise FeatherlessConfigError(
            "No API keys on account — create one at https://featherless.ai/account/api-keys"
        )
    return str(keys[0]["key"])


def resolve_api_key() -> str:
    key = (os.environ.get("FEATHERLESS_API_KEY") or "").strip()
    if key:
        return key
    key = fetch_api_key_from_login()
    os.environ["FEATHERLESS_API_KEY"] = key
    return key


def configure_llm_env(*, api_key: str | None = None, url: str | None = None) -> tuple[str, str]:
    """Point ``zo_common.llm`` at Featherless for the current process."""
    key = api_key or resolve_api_key()
    url = (url or base_url()).rstrip("/")
    os.environ["ZO_MODEL_BASE_URL"] = url
    os.environ["ZO_MODEL_API_KEY"] = key
    os.environ.setdefault("FEATHERLESS_BASE_URL", url)
    os.environ.setdefault("FEATHERLESS_API_KEY", key)
    return url, key


def probe(
    model: str | None = None,
    *,
    api_key: str | None = None,
    url: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Smoke-test chat/completions; returns the parsed JSON response."""
    from zo_common.llm import chat as llm_chat, message_text

    url, key = configure_llm_env(api_key=api_key, url=url)
    model = model or default_model()
    resp = llm_chat(
        [
            {"role": "system", "content": "Reply with exactly: OK"},
            {"role": "user", "content": "ping"},
        ],
        model=model,
        base_url=url,
        api_key=key,
        max_tokens=16,
        temperature=0,
        timeout=timeout,
    )
    if not message_text(resp).strip():
        raise RuntimeError(f"Featherless probe returned empty content for {model!r}")
    return resp


def run_featherless_eval(
    *,
    model: str | None = None,
    version: str,
    eval_dir: str | Path = "extras/eval_local",
    tasks: tuple[str, ...] = ("nextstep", "completion", "anomaly"),
    tags: list[str] | None = None,
    out_dir: str | None = None,
    self_check: bool = True,
    wandb_log: bool = True,
    promote: str | None = None,
    note: str | None = None,
    timeout: float = 300.0,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """Run rules-in-context zero-shot track eval against Featherless."""
    from zo_eval.predict_llm import RulesContextLLMPredictor
    from zo_eval.track import run_track

    url, _key = configure_llm_env()
    model = model or default_model()
    eval_dir = Path(eval_dir)
    valid = eval_dir / "eval_input_valid.csv"
    anomaly = eval_dir / "eval_input_anomaly.csv"
    gold = eval_dir / "gold.json"
    for path in (valid, anomaly, gold):
        if not path.is_file():
            raise FileNotFoundError(f"eval input missing: {path}")

    tag_list = list(tags or [])
    for t in (
        "split:id",
        "role:baseline",
        "method:rules-in-context",
        "baseline:zeroshot",
        "provider:featherless",
        "reportable",
    ):
        if t not in tag_list:
            tag_list.append(t)

    os.environ["ZO_TRACK_USE_WANDB"] = "1" if wandb_log else "0"
    if concurrency is not None:
        os.environ["ZO_LLM_CONCURRENCY"] = str(concurrency)

    def _chat(messages, **kw):
        from zo_common.llm import chat as llm_chat

        kw.pop("base_url", None)  # RulesContextLLMPredictor also passes base_url
        return llm_chat(
            messages,
            base_url=url,
            api_key=_key,
            timeout=timeout,
            **kw,
        )

    predictor = RulesContextLLMPredictor(
        model=model,
        base_url=url,
        chat_fn=_chat,
        backend="served",
        concurrency=concurrency,
    )
    res = run_track(
        predictor,
        valid_csv=str(valid),
        anomaly_csv=str(anomaly),
        gold=str(gold),
        tasks=tasks,
        out_dir=out_dir,
        tags=tag_list,
        version=version,
        model_ref=model,
        eval_set="local",
        notes=note or f"Featherless zero-shot eval ({model})",
        run_proxy=False,
        self_check=self_check,
        promote=promote,
    )
    if predictor.last_batch_stats is not None:
        res["concurrent_stats"] = {
            "summary": predictor.last_batch_stats.summary(),
            "errors": dict(predictor.last_batch_stats.errors),
        }
    return res


COMPARE_MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-7B-Instruct", "7B"),
    ("deepseek-ai/DeepSeek-V4-Flash", "DeepSeek-V4-Flash"),
]


def _metrics_from_run(res: dict[str, Any]) -> dict[str, float]:
    import json
    from pathlib import Path

    report = Path(res["out_dir"]) / "metrics_report.json"
    if not report.is_file():
        return {}
    data = json.loads(report.read_text(encoding="utf-8"))
    flat: dict[str, float] = {}
    for task, sections in (data.get("tasks") or {}).items():
        overall = (sections or {}).get("by_family", {}).get("overall") or {}
        for k, v in overall.items():
            if isinstance(v, (int, float)):
                flat[f"{task}_{k}"] = float(v)
    return flat


def run_featherless_compare(
    *,
    eval_dir: str | Path = "extras/eval_local_smoke",
    wandb_log: bool = False,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run zero-shot eval for 1.5B / 7B / DeepSeek and return side-by-side metrics."""
    rows: list[dict[str, Any]] = []
    for model, label in COMPARE_MODELS:
        version = f"zeroshot-featherless-compare-{label.lower().replace('.', '')}-v1"
        res = run_featherless_eval(
            model=model,
            version=version,
            eval_dir=eval_dir,
            tags=[f"size:{label}", "suite:featherless-compare"],
            self_check=True,
            wandb_log=wandb_log,
            timeout=timeout,
        )
        metrics = _metrics_from_run(res)
        rows.append({"label": label, "model": model, "run_id": res["run_id"], **metrics})

    summary = {"eval_dir": str(eval_dir), "rows": rows}
    keys = ["nextstep_top1", "nextstep_mrr", "completion_token_acc", "completion_norm_edit_dist", "anomaly_binary_acc"]
    print("\n=== Featherless zero-shot model comparison ===")
    print(f"eval_dir: {eval_dir}\n")
    header = f"{'model':<22}" + "".join(f"{k:>22}" for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:<22}"
            + "".join(f"{row.get(k, float('nan')):>22.4f}" for k in keys)
        )
    return summary
