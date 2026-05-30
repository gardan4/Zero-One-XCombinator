"""Per-example trace artifacts and cross-run disagreement comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zo_eval import track_metrics as M
from zo_eval.predict import extract_answer, extract_reasoning
from zo_eval.submission import AnomalyInput, ValidInput


def _seq_list(steps: list[str]) -> list[str]:
    return list(steps)


class TraceCollector:
    """Accumulates per-example rows during ``run_track``; written to ``examples.jsonl``."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        *,
        example_id: str,
        task: str,
        family: str,
        input_payload: dict[str, Any],
        prediction: Any,
        gold: Any = None,
        trace: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "example_id": example_id,
            "task": task,
            "family": family,
            "input": input_payload,
            "prediction": prediction,
            "gold": gold,
            "correct": None,
            "trace": trace or {},
        }
        if extra:
            row.update(extra)
        self._score_row(row)
        self.rows.append(row)

    def _score_row(self, row: dict[str, Any]) -> None:
        task = row["task"]
        gold = row.get("gold")
        pred = row.get("prediction")
        if gold is None:
            return
        if task == "nextstep" and isinstance(pred, list) and isinstance(gold, str):
            pos = pred.index(gold) + 1 if gold in pred else 0
            row["correct"] = pos == 1
            row["rank"] = pos if pos else None
            row["mrr_contrib"] = 1.0 / pos if pos else 0.0
        elif task == "completion" and isinstance(pred, list) and isinstance(gold, list):
            row["correct"] = pred == gold
            row["token_acc"] = M.token_accuracy(pred, gold)
            row["norm_edit_dist"] = M._levenshtein(pred, gold) / (max(len(pred), len(gold)) or 1)
        elif task == "anomaly" and isinstance(pred, dict) and isinstance(gold, dict):
            g_inv = gold.get("is_valid") == 0
            p_inv = pred.get("is_valid", 1) == 0
            row["correct"] = g_inv == p_inv
            row["rule_match"] = (pred.get("rule") or None) == (gold.get("rule") or None) if g_inv and p_inv else None

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path


def load_examples(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """``{task: {example_id: row}}`` from ``examples.jsonl``."""
    path = Path(path)
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    if not path.exists():
        return by_task
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row.get("task", "?")
        ex = row.get("example_id", "?")
        by_task.setdefault(task, {})[ex] = row
    return by_task


def compare_example_traces(
    path_a: str | Path,
    path_b: str | Path,
    *,
    task: str,
    mode: str = "different",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Align two ``examples.jsonl`` files and return comparison records."""
    a = load_examples(path_a).get(task, {})
    b = load_examples(path_b).get(task, {})
    common = sorted(set(a) & set(b))
    out: list[dict[str, Any]] = []

    for ex in common:
        ra, rb = a[ex], b[ex]
        ca, cb = ra.get("correct"), rb.get("correct")
        pa, pb = ra.get("prediction"), rb.get("prediction")
        same_pred = pa == pb
        record = {
            "example_id": ex,
            "family": ra.get("family") or rb.get("family"),
            "input": ra.get("input") or rb.get("input"),
            "gold": ra.get("gold") or rb.get("gold"),
            "a": {"prediction": pa, "correct": ca, "trace": ra.get("trace")},
            "b": {"prediction": pb, "correct": cb, "trace": rb.get("trace")},
        }
        if mode == "different" and (same_pred and ca == cb):
            continue
        if mode == "both_wrong" and not (ca is False and cb is False):
            continue
        if mode == "only_a_correct" and ca is not True:
            continue
        if mode == "only_b_correct" and cb is not True:
            continue
        if mode == "largest_score_gap" and task != "anomaly":
            continue
        if mode == "largest_score_gap" and task == "anomaly":
            sa = (pa or {}).get("score")
            sb = (pb or {}).get("score")
            if sa is None or sb is None:
                continue
            record["score_gap"] = abs(float(sa) - float(sb))
        out.append(record)

    if mode == "largest_score_gap":
        out.sort(key=lambda r: r.get("score_gap", 0), reverse=True)
    return out[:limit]


def valid_input_payload(item: ValidInput) -> dict[str, Any]:
    return {
        "partial_sequence": _seq_list(item.partial_sequence),
        "completion_fraction": item.completion_fraction,
    }


def anomaly_input_payload(item: AnomalyInput) -> dict[str, Any]:
    return {"sequence": _seq_list(item.sequence)}


def trace_from_llm_response(raw: str) -> dict[str, Any]:
    return {
        "raw_response": raw,
        "reasoning": extract_reasoning(raw),
        "extracted_answer": extract_answer(raw),
    }
