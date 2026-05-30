"""Stage dashboard inference inputs and run ``run_track`` as a registry job."""

from __future__ import annotations

import csv
import json
import os
import threading
from pathlib import Path
from typing import Any

from zo_common.registry import get_run, run_dir, update_run

from zo_eval import submission as sub
from zo_eval.official_metrics import KICKOFF_ANOMALY, KICKOFF_VALID
from zo_eval.predictors import BASELINE_PREDICTORS, build_predictor
from zo_eval.track import run_track

MAX_VALID_ROWS = int(os.environ.get("ZO_INFERENCE_MAX_VALID_ROWS", "500"))
MAX_ANOMALY_ROWS = int(os.environ.get("ZO_INFERENCE_MAX_ANOMALY_ROWS", "500"))
STEP_SEP = sub.STEP_SEP


def parse_steps_text(text: str) -> list[str]:
    """Parse pipe- or newline-separated fab steps."""
    return [s.strip() for s in text.replace("\r", "").replace("\n", "|").split("|") if s.strip()]


def resolve_eval_paths(eval_set: str, valid: str | None, anomaly: str | None) -> tuple[str | None, str | None]:
    if eval_set == "kickoff":
        valid = valid or str(KICKOFF_VALID)
        anomaly = anomaly or str(KICKOFF_ANOMALY)
    return valid, anomaly


def predictor_allowed(predictor: str) -> bool:
    if predictor in BASELINE_PREDICTORS:
        return True
    return bool(os.environ.get("ZO_ALLOW_DASHBOARD_INFERENCE"))


def write_manual_valid_csv(
    path: Path,
    *,
    example_id: str,
    family: str,
    completion_fraction: float,
    partial_sequence: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"])
        w.writerow([example_id, family, completion_fraction, STEP_SEP.join(partial_sequence)])


def write_manual_anomaly_csv(
    path: Path,
    *,
    example_id: str,
    family: str,
    sequence: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["EXAMPLE_ID", "FAMILY", "SEQUENCE"])
        w.writerow([example_id, family, STEP_SEP.join(sequence)])


_VALID_HEADERS = frozenset({"EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"})
_ANOMALY_HEADERS = frozenset({"EXAMPLE_ID", "FAMILY", "SEQUENCE"})


def _check_csv_headers(path: Path, required: frozenset[str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            raw = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc
    headers = {(h or "").lstrip("\ufeff").strip().strip('"') for h in raw}
    missing = required - headers
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")


def stage_uploaded_csv(src: Path, dest: Path, *, kind: str) -> dict[str, Any]:
    """Validate and copy uploaded CSV; return row counts."""
    if kind == "valid":
        _check_csv_headers(src, _VALID_HEADERS)
        rows = sub.read_valid_inputs(src)
        if len(rows) > MAX_VALID_ROWS:
            raise ValueError(f"valid CSV exceeds max {MAX_VALID_ROWS} rows ({len(rows)})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return {"valid_rows": len(rows), "anomaly_rows": 0}
    _check_csv_headers(src, _ANOMALY_HEADERS)
    rows = sub.read_anomaly_inputs(src)
    if len(rows) > MAX_ANOMALY_ROWS:
        raise ValueError(f"anomaly CSV exceeds max {MAX_ANOMALY_ROWS} rows ({len(rows)})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return {"valid_rows": 0, "anomaly_rows": len(rows)}


def stage_manual_example(
    inputs_dir: Path,
    *,
    task: str,
    family: str,
    completion_fraction: float,
    partial_sequence: list[str] | None,
    sequence: list[str] | None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Write one-row organizer CSVs for a manual example."""
    inputs_dir.mkdir(parents=True, exist_ok=True)
    valid_path: str | None = None
    anomaly_path: str | None = None
    summary: dict[str, Any] = {"valid_rows": 0, "anomaly_rows": 0, "task": task}

    if task in ("nextstep", "completion"):
        if not partial_sequence:
            raise ValueError("partial_sequence required for nextstep/completion")
        valid_path = str(inputs_dir / "eval_input_valid.csv")
        write_manual_valid_csv(
            Path(valid_path),
            example_id="manual_valid_0001",
            family=family,
            completion_fraction=completion_fraction,
            partial_sequence=partial_sequence,
        )
        summary["valid_rows"] = 1
    elif task == "anomaly":
        if not sequence:
            raise ValueError("sequence required for anomaly")
        anomaly_path = str(inputs_dir / "eval_input_anomaly.csv")
        write_manual_anomaly_csv(
            Path(anomaly_path),
            example_id="manual_anomaly_0001",
            family=family,
            sequence=sequence,
        )
        summary["anomaly_rows"] = 1
    else:
        raise ValueError(f"unknown task {task!r}")
    return valid_path, anomaly_path, summary


def tasks_for_inputs(valid_csv: str | None, anomaly_csv: str | None, requested: list[str]) -> tuple[str, ...]:
    allowed = set(requested)
    out: list[str] = []
    if valid_csv and "nextstep" in allowed:
        out.append("nextstep")
    if valid_csv and "completion" in allowed:
        out.append("completion")
    if anomaly_csv and "anomaly" in allowed:
        out.append("anomaly")
    if not out:
        raise ValueError("no tasks to run — provide matching CSVs or adjust tasks list")
    return tuple(out)


def run_inference_job(
    run_id: str,
    *,
    predictor_kind: str,
    model: str,
    model_ref: str | None,
    base_url: str | None,
    order: int,
    train_families: str | None,
    valid_csv: str | None,
    anomaly_csv: str | None,
    tasks: tuple[str, ...],
    version: str,
    eval_set: str,
    tags: list[str],
    name: str | None,
    notes: str | None,
    gold_path: str | None = None,
    run_proxy: bool = True,
) -> None:
    """Execute ``run_track`` for an existing registry run (intended for background thread)."""
    try:
        update_run(run_id, status="running")
        pred = build_predictor(
            predictor_kind,
            train_families=train_families,
            order=order,
            model=model,
            base_url=base_url,
        )
        gold = None
        if gold_path and Path(gold_path).exists():
            gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
        run_track(
            pred,
            valid_csv=valid_csv,
            anomaly_csv=anomaly_csv,
            gold=gold,
            gold_path=gold_path,
            tasks=tasks,
            run_id=run_id,
            tags=tags,
            version=version,
            model_ref=model_ref,
            eval_set=eval_set,
            name=name,
            notes=notes,
            run_proxy=run_proxy and gold is None,
            write_examples=True,
        )
    except Exception as exc:
        meta = get_run(run_id)
        prior = (meta.notes if meta else "") or ""
        msg = f"inference failed: {exc}"
        update_run(run_id, status="failed", notes=f"{prior}\n{msg}".strip())


def start_inference_job_thread(**kwargs) -> None:
    threading.Thread(target=run_inference_job, kwargs=kwargs, daemon=True).start()


def input_dir_for_run(run_id: str) -> Path:
    return run_dir(run_id) / "inputs"
