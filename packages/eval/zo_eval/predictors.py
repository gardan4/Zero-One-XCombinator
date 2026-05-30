"""Shared predictor factory for CLI, API, and dashboard inference jobs."""

from __future__ import annotations

PREDICTOR_KINDS = (
    "ngram",
    "freq",
    "oracle",
    "llm",
    "hf",
    "llm-zeroshot",
    "likelihood-ngram",
    "classifier",
)

BASELINE_PREDICTORS = frozenset({"ngram", "freq", "oracle"})


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
    if kind == "llm-zeroshot":
        from zo_eval.predict_llm import RulesContextLLMPredictor

        if base_url or model == "default":
            return RulesContextLLMPredictor(model=model, base_url=base_url, backend="served")
        return RulesContextLLMPredictor(model=model, backend="hf")
    if kind in ("llm", "hf"):
        from zo_eval.predict_llm import HFGeneratePredictor, ServedLLMPredictor

        if kind == "llm":
            return ServedLLMPredictor(model=model, base_url=base_url)
        return HFGeneratePredictor(model=model)
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
