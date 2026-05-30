"""Shared predictor factory for CLI, API, and dashboard inference jobs."""

from __future__ import annotations

import os

PREDICTOR_KINDS = (
    "ngram",
    "freq",
    "oracle",
    "llm",
    "hf",
    "base",
    "base-hf",
    "likelihood-ngram",
    "classifier",
)

BASELINE_PREDICTORS = frozenset({"ngram", "freq", "oracle"})

# Default base (un-fine-tuned) LLM for the `base` / `base-hf` predictors, so the dashboard's
# "default" model maps to a real repo. Overridable via env. 7B follows the rich-context prompt
# well; on a CPU/MPS laptop a smaller Qwen (0.5B/1.5B) generates faster.
BASE_LLM_DEFAULT = os.environ.get("ZO_BASE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


class PredictorBuildError(ValueError):
    """Raised when predictor kind/model configuration is invalid."""


def build_predictor(
    kind: str,
    *,
    train_families: str | None = None,
    order: int = 3,
    model: str = "default",
    base_url: str | None = None,
):
    """Construct a track ``Predictor`` implementation (lazy-imports heavy deps)."""
    kind = kind.strip().lower()
    tf = [f.strip().upper() for f in train_families.split(",")] if train_families else None

    if kind == "ngram":
        from zo_eval.baselines import NGramPredictor

        return NGramPredictor(train_families=tf, order=order)
    if kind == "oracle":
        from zo_eval.baselines import OraclePredictor

        return OraclePredictor()
    if kind == "freq":
        from zo_eval.baselines import FreqPredictor

        return FreqPredictor(train_families=tf)
    if kind in ("llm", "hf"):
        from zo_eval.predict_llm import HFGeneratePredictor, ServedLLMPredictor

        if kind == "llm":
            return ServedLLMPredictor(model=model, base_url=base_url)
        return HFGeneratePredictor(model=model)
    if kind in ("base", "base-hf"):
        # Base (un-fine-tuned) LLM with the rich-context next-step prompt. `base` = served
        # (OpenAI endpoint, like `llm`); `base-hf` = local transformers.generate (like `hf`).
        from zo_eval.predict_llm import HFGeneratePredictor, ServedLLMPredictor

        m = model if (model and model != "default") else BASE_LLM_DEFAULT
        if kind == "base":
            return ServedLLMPredictor(model=m, base_url=base_url, style="base")
        return HFGeneratePredictor(model=m, style="base")
    if kind == "likelihood-ngram":
        from zo_eval.anomaly_detect import LikelihoodDetector
        from zo_eval.baselines import NGramPredictor

        ng = NGramPredictor(train_families=tf, order=order)

        def _score(item):
            return ng.pooled.mean_logprob(item.sequence)

        return LikelihoodDetector(_score, name="likelihood-ngram")
    if kind == "classifier":
        from zo_eval.anomaly_detect import ClassifierDetector

        if model == "default":
            raise PredictorBuildError("classifier predictor needs a model (served name or HF id)")
        return ClassifierDetector(model=model, base_url=base_url)
    raise PredictorBuildError(f"unknown predictor {kind!r}; use: {', '.join(PREDICTOR_KINDS)}")
