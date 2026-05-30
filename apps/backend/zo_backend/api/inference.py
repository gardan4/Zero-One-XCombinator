"""Dashboard inference jobs: upload organizer CSVs or manual examples → ``run_track``."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from zo_common.registry import get_run, new_run
from zo_eval.inference_jobs import (
    input_dir_for_run,
    parse_steps_text,
    predictor_allowed,
    resolve_eval_paths,
    stage_manual_example,
    stage_uploaded_csv,
    start_inference_job_thread,
    tasks_for_inputs,
)
from zo_eval.predictors import PREDICTOR_KINDS, PredictorBuildError, build_predictor

router = APIRouter(tags=["inference"])

MAX_UPLOAD_BYTES = int(os.environ.get("ZO_INFERENCE_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))


def _parse_tags(tags: str | None, extra: list[str] | None = None) -> list[str]:
    out = [t.strip() for t in (tags or "").split(",") if t.strip()]
    for t in extra or []:
        if t and t not in out:
            out.append(t)
    return out


def _parse_tasks(tasks: str) -> list[str]:
    return [t.strip() for t in tasks.split(",") if t.strip()]


def _job_links(run_id: str) -> dict[str, str]:
    return {
        "run": f"/api/runs/{run_id}",
        "examples": f"/api/runs/{run_id}/examples",
        "compare_report": "/api/compare/report",
    }


def _model_ref(model: str, model_ref: str | None) -> str | None:
    if model_ref:
        return model_ref
    if "/" in model and not model.startswith("default"):
        return model
    return None


async def _read_upload(upload: UploadFile | None) -> bytes | None:
    if upload is None or not upload.filename:
        return None
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"upload {upload.filename!r} exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    return data


def _validate_predictor(predictor: str, model: str, base_url: str | None, order: int, train_families: str | None):
    pred = predictor.strip().lower()
    if pred not in PREDICTOR_KINDS:
        raise HTTPException(status_code=400, detail=f"unknown predictor {pred!r}; use: {', '.join(PREDICTOR_KINDS)}")
    if not predictor_allowed(pred):
        raise HTTPException(
            status_code=403,
            detail=(
                f"predictor {pred!r} requires ZO_ALLOW_DASHBOARD_INFERENCE=1 "
                "(baselines ngram/freq/oracle allowed by default)"
            ),
        )
    try:
        build_predictor(pred, train_families=train_families, order=order, model=model, base_url=base_url)
    except PredictorBuildError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_manual(manual_json: str | None) -> dict[str, Any] | None:
    if not manual_json or not manual_json.strip():
        return None
    try:
        data = json.loads(manual_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"manual must be JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="manual must be a JSON object")
    return data


def _sequence_from_manual(data: dict[str, Any], key: str) -> list[str] | None:
    raw = data.get(key)
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return parse_steps_text(str(raw))


def _stage_inputs_to(
    inputs_dir: Path,
    *,
    valid_data: bytes | None,
    anomaly_data: bytes | None,
    manual: dict[str, Any] | None,
    eval_set: str,
) -> tuple[str | None, str | None, dict[str, Any]]:
    summary: dict[str, Any] = {"valid_rows": 0, "anomaly_rows": 0, "eval_set": eval_set}
    valid_path: str | None = None
    anomaly_path: str | None = None

    if manual:
        task = str(manual.get("task", "nextstep")).strip().lower()
        family = str(manual.get("family", "MOSFET")).strip().upper()
        try:
            cf = float(manual.get("completion_fraction", 0.6))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="completion_fraction must be a number") from exc
        partial = _sequence_from_manual(manual, "partial_sequence")
        sequence = _sequence_from_manual(manual, "sequence")
        try:
            valid_path, anomaly_path, manual_sum = stage_manual_example(
                inputs_dir,
                task=task,
                family=family,
                completion_fraction=cf,
                partial_sequence=partial,
                sequence=sequence,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        summary.update(manual_sum)
        summary["input_mode"] = "manual"
        return valid_path, anomaly_path, summary

    if not valid_data and not anomaly_data:
        if eval_set == "kickoff":
            valid_path, anomaly_path = resolve_eval_paths(eval_set, None, None)
            summary["input_mode"] = "kickoff"
            return valid_path, anomaly_path, summary
        raise HTTPException(
            status_code=400,
            detail="provide valid_csv and/or anomaly_csv uploads, manual JSON, or eval_set=kickoff",
        )

    summary["input_mode"] = "upload"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if valid_data:
            src = tmp_path / "upload_valid.csv"
            src.write_bytes(valid_data)
            dest = inputs_dir / "eval_input_valid.csv"
            try:
                part = stage_uploaded_csv(src, dest, kind="valid")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            summary.update(part)
            valid_path = str(dest)
        if anomaly_data:
            src = tmp_path / "upload_anomaly.csv"
            src.write_bytes(anomaly_data)
            dest = inputs_dir / "eval_input_anomaly.csv"
            try:
                part = stage_uploaded_csv(src, dest, kind="anomaly")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            summary.update(part)
            anomaly_path = str(dest)
    return valid_path, anomaly_path, summary


def _stage_inputs(
    run_id: str,
    *,
    valid_data: bytes | None,
    anomaly_data: bytes | None,
    manual: dict[str, Any] | None,
    eval_set: str,
) -> tuple[str | None, str | None, dict[str, Any]]:
    return _stage_inputs_to(
        input_dir_for_run(run_id),
        valid_data=valid_data,
        anomaly_data=anomaly_data,
        manual=manual,
        eval_set=eval_set,
    )


@router.post("/inference/preview")
async def preview_inference(
    predictor: Annotated[str, Form()] = "ngram",
    model: Annotated[str, Form()] = "default",
    base_url: Annotated[str | None, Form()] = None,
    order: Annotated[int, Form()] = 3,
    train_families: Annotated[str | None, Form()] = None,
    tasks: Annotated[str, Form()] = "nextstep,completion,anomaly",
    eval_set: Annotated[str, Form()] = "dashboard",
    manual_json: Annotated[str | None, Form()] = None,
    valid_csv: Annotated[UploadFile | None, File()] = None,
    anomaly_csv: Annotated[UploadFile | None, File()] = None,
) -> dict[str, Any]:
    """Validate predictor settings and inputs without starting a job."""
    _validate_predictor(predictor, model, base_url, order, train_families)
    manual = _parse_manual(manual_json)
    valid_data = await _read_upload(valid_csv)
    anomaly_data = await _read_upload(anomaly_csv)
    with tempfile.TemporaryDirectory() as tmp:
        valid_path, anomaly_path, summary = _stage_inputs_to(
            Path(tmp) / "inputs",
            valid_data=valid_data,
            anomaly_data=anomaly_data,
            manual=manual,
            eval_set=eval_set,
        )
    task_list = _parse_tasks(tasks)
    try:
        resolved_tasks = tasks_for_inputs(valid_path, anomaly_path, task_list)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "predictor": predictor.strip().lower(),
        "tasks": list(resolved_tasks),
        "input_summary": summary,
        "valid_csv": valid_path,
        "anomaly_csv": anomaly_path,
    }


@router.post("/inference/jobs")
async def create_inference_job(
    predictor: Annotated[str, Form()] = "ngram",
    model: Annotated[str, Form()] = "default",
    model_ref: Annotated[str | None, Form()] = None,
    base_url: Annotated[str | None, Form()] = None,
    order: Annotated[int, Form()] = 3,
    train_families: Annotated[str | None, Form()] = None,
    tasks: Annotated[str, Form()] = "nextstep,completion,anomaly",
    version: Annotated[str, Form()] = "dashboard-v1",
    name: Annotated[str | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
    tags: Annotated[str, Form()] = "",
    eval_set: Annotated[str, Form()] = "dashboard",
    manual_json: Annotated[str | None, Form()] = None,
    valid_csv: Annotated[UploadFile | None, File()] = None,
    anomaly_csv: Annotated[UploadFile | None, File()] = None,
) -> dict[str, Any]:
    """Start an async inference job (registry run + background ``run_track``)."""
    _validate_predictor(predictor, model, base_url, order, train_families)
    manual = _parse_manual(manual_json)
    valid_data = await _read_upload(valid_csv)
    anomaly_data = await _read_upload(anomaly_csv)

    pred_kind = predictor.strip().lower()
    tag_list = _parse_tags(tags, ["source:dashboard-inference"])
    run_name = name or f"inference:{pred_kind}:{version}"
    meta = new_run(
        run_name,
        "eval",
        config={
            "source": "dashboard-inference",
            "predictor": pred_kind,
            "version": version,
            "model": model,
            "eval_set": eval_set,
            "tasks": _parse_tasks(tasks),
        },
        tags=tag_list,
    )
    from zo_common.registry import update_run

    update_run(meta.id, status="queued", notes=notes or "")

    try:
        valid_path, anomaly_path, summary = _stage_inputs(
            meta.id,
            valid_data=valid_data,
            anomaly_data=anomaly_data,
            manual=manual,
            eval_set=eval_set,
        )
        resolved_tasks = tasks_for_inputs(valid_path, anomaly_path, _parse_tasks(tasks))
    except HTTPException:
        update_run(meta.id, status="failed", notes="input staging failed")
        raise
    except ValueError as exc:
        update_run(meta.id, status="failed", notes=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mref = _model_ref(model, model_ref)
    start_inference_job_thread(
        run_id=meta.id,
        predictor_kind=pred_kind,
        model=model,
        model_ref=mref,
        base_url=base_url,
        order=order,
        train_families=train_families,
        valid_csv=valid_path,
        anomaly_csv=anomaly_path,
        tasks=resolved_tasks,
        version=version,
        eval_set=eval_set,
        tags=tag_list,
        name=run_name,
        notes=notes,
    )

    return {
        "run_id": meta.id,
        "status": "queued",
        "input_summary": {**summary, "tasks": list(resolved_tasks)},
        "links": _job_links(meta.id),
    }


@router.get("/inference/jobs/{run_id}")
def inference_job_status(run_id: str) -> dict[str, Any]:
    """Thin alias over run detail for dashboard polling."""
    meta = get_run(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="no such run")
    return {
        "run_id": meta.id,
        "status": meta.status,
        "name": meta.name,
        "tags": meta.tags,
        "metrics": meta.metrics,
        "links": _job_links(run_id),
    }
