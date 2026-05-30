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

    def next_step_batch(self, items: list[ValidInput]) -> list[list[str]]:
        if hasattr(self._inner, "next_step_batch"):
            raw = self._inner.next_step_batch(items)
            out: list[list[str]] = []
            for item, result in zip(items, raw, strict=False):
                if isinstance(result, tuple) and len(result) == 2:
                    ranks, trace = result
                else:
                    ranks, trace = result, {"source": getattr(self._inner, "name", "?"), "prediction": result}
                self._set(item.example_id, "nextstep", trace)
                out.append(ranks)
            return out
        return [self.next_step(item) for item in items]

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

    def complete_batch(self, items: list[ValidInput]) -> list[list[str]]:
        if hasattr(self._inner, "complete_batch"):
            raw = self._inner.complete_batch(items)
            out: list[list[str]] = []
            for item, result in zip(items, raw, strict=False):
                if isinstance(result, tuple) and len(result) == 2:
                    steps, trace = result
                else:
                    steps, trace = result, {"source": getattr(self._inner, "name", "?"), "prediction": result}
                self._set(item.example_id, "completion", trace)
                out.append(steps)
            return out
        return [self.complete(item) for item in items]

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

    def anomaly_batch(self, items: list[AnomalyInput]) -> list[tuple[int, float, str | None]]:
        if hasattr(self._inner, "anomaly_batch"):
            raw = self._inner.anomaly_batch(items)
            out: list[tuple[int, float, str | None]] = []
            for item, result in zip(items, raw, strict=False):
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                    pred, trace = result
                else:
                    pred = result
                    iv, sc, rule = pred
                    trace = {
                        "source": getattr(self._inner, "name", "?"),
                        "is_valid": iv,
                        "score": sc,
                        "rule": rule,
                    }
                self._set(item.example_id, "anomaly", trace)
                out.append(pred)
            return out
        return [self.anomaly(item) for item in items]


def wrap_with_tracing(predictor):
    """Return ``TracingPredictor`` unless already wrapped."""
    if isinstance(predictor, TracingPredictor):
        return predictor
    return TracingPredictor(predictor)
