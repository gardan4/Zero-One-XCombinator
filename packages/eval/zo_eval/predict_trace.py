"""Optional trace side-channel for predictors without breaking the ``Predictor`` protocol."""

from __future__ import annotations

from typing import Any

from zo_eval.submission import AnomalyInput, ValidInput


class TracingPredictor:
    """Wraps a predictor and records last-call trace metadata per example id."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "wrapped")
        self._last: dict[str, dict[str, Any]] = {}

    def pop_trace(self, example_id: str, task: str) -> dict[str, Any]:
        return self._last.pop(f"{task}:{example_id}", {})

    def _set(self, example_id: str, task: str, trace: dict[str, Any]) -> None:
        self._last[f"{task}:{example_id}"] = trace

    def next_step(self, item: ValidInput) -> list[str]:
        if hasattr(self._inner, "next_step_with_trace"):
            ranks, trace = self._inner.next_step_with_trace(item)
            self._set(item.example_id, "nextstep", trace)
            return ranks
        out = self._inner.next_step(item)
        self._set(
            item.example_id,
            "nextstep",
            {"source": getattr(self._inner, "name", "?"), "prediction": out},
        )
        return out

    def complete(self, item: ValidInput) -> list[str]:
        if hasattr(self._inner, "complete_with_trace"):
            steps, trace = self._inner.complete_with_trace(item)
            self._set(item.example_id, "completion", trace)
            return steps
        out = self._inner.complete(item)
        self._set(
            item.example_id,
            "completion",
            {"source": getattr(self._inner, "name", "?"), "prediction": out},
        )
        return out

    def anomaly(self, item: AnomalyInput) -> tuple[int, float, str | None]:
        if hasattr(self._inner, "anomaly_with_trace"):
            result, trace = self._inner.anomaly_with_trace(item)
            self._set(item.example_id, "anomaly", trace)
            return result
        out = self._inner.anomaly(item)
        iv, sc, rule = out
        self._set(
            item.example_id,
            "anomaly",
            {
                "source": getattr(self._inner, "name", "?"),
                "is_valid": iv,
                "score": sc,
                "rule": rule,
            },
        )
        return out


def wrap_with_tracing(predictor):
    """Return ``TracingPredictor`` unless already wrapped."""
    if isinstance(predictor, TracingPredictor):
        return predictor
    return TracingPredictor(predictor)
