"""Thread-pool concurrency for served (HTTP) LLM predictors."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ConcurrentStats:
    total: int = 0
    ok: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def record_error(self, exc: BaseException) -> str:
        key = classify_error(exc)
        self.errors[key] = self.errors.get(key, 0) + 1
        return key

    def summary(self) -> str:
        if not self.errors:
            return f"ok={self.ok}/{self.total}"
        err = ", ".join(f"{k}={v}" for k, v in sorted(self.errors.items()))
        return f"ok={self.ok}/{self.total} errors: {err}"


def classify_error(exc: BaseException) -> str:
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            return f"http_{exc.response.status_code}"
    except ImportError:
        pass
    return type(exc).__name__


def map_concurrent(
    items: list[T],
    fn: Callable[[T], R],
    *,
    concurrency: int,
    on_error: Callable[[T, BaseException], R] | None = None,
) -> tuple[list[R], ConcurrentStats]:
    """Run ``fn`` over ``items`` with up to ``concurrency`` parallel workers (order preserved)."""
    stats = ConcurrentStats(total=len(items))
    if not items:
        return [], stats
    if concurrency <= 1:
        out: list[R] = []
        for item in items:
            try:
                out.append(fn(item))
                stats.ok += 1
            except Exception as exc:
                stats.record_error(exc)
                if on_error is None:
                    raise
                out.append(on_error(item, exc))
        return out, stats

    results: list[R | None] = [None] * len(items)

    def _run(idx_item: tuple[int, T]) -> tuple[int, R | BaseException]:
        idx, item = idx_item
        try:
            return idx, fn(item)
        except Exception as exc:
            return idx, exc

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for idx, result in pool.map(_run, list(enumerate(items))):
            if isinstance(result, BaseException):
                stats.record_error(result)
                if on_error is None:
                    raise result
                results[idx] = on_error(items[idx], result)
            else:
                stats.ok += 1
                results[idx] = result

    return results, stats  # type: ignore[return-value]
