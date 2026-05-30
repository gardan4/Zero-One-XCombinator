"""Tests for served LLM concurrent map."""

from __future__ import annotations

import httpx

from zo_eval.concurrent_served import ConcurrentStats, classify_error, map_concurrent


def test_classify_http_error():
    req = httpx.Request("POST", "https://example.com")
    resp = httpx.Response(429, request=req)
    err = httpx.HTTPStatusError("rate limited", request=req, response=resp)
    assert classify_error(err) == "http_429"


def test_map_concurrent_preserves_order():
    def double(x: int) -> int:
        return x * 2

    out, stats = map_concurrent([1, 2, 3], double, concurrency=2)
    assert out == [2, 4, 6]
    assert stats.ok == 3
    assert stats.errors == {}


def test_map_concurrent_on_error():
    def maybe(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    out, stats = map_concurrent(
        [1, 2, 3],
        maybe,
        concurrency=3,
        on_error=lambda _x, _e: -1,
    )
    assert out == [1, -1, 3]
    assert stats.ok == 2
    assert stats.errors == {"ValueError": 1}


def test_concurrent_stats_summary():
    s = ConcurrentStats(total=10, ok=8, errors={"http_429": 2})
    assert "http_429=2" in s.summary()
