from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    prompt: str
    answer: str | None = None


class TaskSpec(BaseModel):
    name: str
    metric: str = "exact_match"  # exact_match | contains | numeric | regex
    system: str | None = None
    pattern: str | None = None  # capture group for the `regex` metric
    items: list[EvalItem] = Field(default_factory=list)
    dataset: str | None = None  # optional .jsonl of {"prompt", "answer"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskSpec:
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))

    def load_items(self) -> list[EvalItem]:
        if self.dataset and Path(self.dataset).exists():
            rows = [
                json.loads(line)
                for line in Path(self.dataset).read_text().splitlines()
                if line.strip()
            ]
            return [EvalItem(**r) for r in rows]
        return self.items


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _num(s: str | None) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    return float(m.group()) if m else None


def score(metric: str, prediction: str, answer: str | None, pattern: str | None = None) -> float:
    if answer is None:
        return 0.0
    if metric == "exact_match":
        return float(_norm(prediction) == _norm(answer))
    if metric == "contains":
        return float(_norm(answer) in _norm(prediction))
    if metric == "numeric":
        p, a = _num(prediction), _num(answer)
        return float(p is not None and a is not None and abs(p - a) < 1e-6)
    if metric == "regex":
        m = re.search(pattern or "", prediction or "")
        got = (m.group(1) if (m and m.groups()) else (m.group() if m else "")) if m else ""
        return float(_norm(got) == _norm(answer))
    return 0.0
