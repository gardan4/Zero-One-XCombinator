"""Dashboard inference API: manual examples, CSV upload, gating, failures."""

from __future__ import annotations

import json
import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from zo_backend.main import app
from zo_common.registry import get_run, run_dir


@pytest.fixture
def experiments_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ZO_EXPERIMENTS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client():
    return TestClient(app)


def _wait_run(run_id: str, *, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    status = "queued"
    while time.time() < deadline:
        meta = get_run(run_id)
        if meta:
            status = meta.status
            if status in ("completed", "failed"):
                return status
        time.sleep(0.2)
    return status


def _mini_valid_csv() -> bytes:
    return (
        b"EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE\n"
        b"t_valid_001,MOSFET,0.6,SPIN COAT PHOTORESIST|EXPOSE LITHO LEVEL 1\n"
    )


def _mini_anomaly_csv() -> bytes:
    return (
        b"EXAMPLE_ID,FAMILY,SEQUENCE\n"
        b"t_anom_001,MOSFET,SPIN COAT PHOTORESIST|EXPOSE LITHO LEVEL 1|DEPOSIT POLYSILICON\n"
    )


def test_manual_nextstep_job_completes(client, experiments_tmp):
    manual = json.dumps(
        {
            "task": "nextstep",
            "family": "MOSFET",
            "completion_fraction": 0.6,
            "partial_sequence": ["SPIN COAT PHOTORESIST", "EXPOSE LITHO LEVEL 1"],
        }
    )
    r = client.post(
        "/api/inference/jobs",
        data={
            "predictor": "ngram",
            "version": "test-manual-ns",
            "tasks": "nextstep",
            "manual_json": manual,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    run_id = body["run_id"]
    assert body["status"] == "queued"

    status = _wait_run(run_id)
    assert status == "completed", get_run(run_id).notes if get_run(run_id) else ""
    assert (run_dir(run_id) / "results" / "nextstep.csv").exists()


def test_manual_anomaly_job_writes_anomaly_csv(client, experiments_tmp):
    manual = json.dumps(
        {
            "task": "anomaly",
            "family": "MOSFET",
            "sequence": [
                "SPIN COAT PHOTORESIST",
                "EXPOSE LITHO LEVEL 1",
                "DEPOSIT POLYSILICON",
            ],
        }
    )
    r = client.post(
        "/api/inference/jobs",
        data={
            "predictor": "ngram",
            "version": "test-manual-anom",
            "tasks": "anomaly",
            "manual_json": manual,
        },
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert _wait_run(run_id) == "completed"
    assert (run_dir(run_id) / "results" / "anomaly.csv").exists()


def test_csv_upload_valid_only(client, experiments_tmp):
    r = client.post(
        "/api/inference/jobs",
        data={"predictor": "ngram", "version": "test-csv-valid", "tasks": "nextstep,completion"},
        files={"valid_csv": ("eval_input_valid.csv", BytesIO(_mini_valid_csv()), "text/csv")},
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert _wait_run(run_id) == "completed"
    results = run_dir(run_id) / "results"
    assert (results / "nextstep.csv").exists()
    assert (results / "completion.csv").exists()


def test_bad_csv_headers_return_400(client, experiments_tmp):
    bad = b"EXAMPLE_ID,WRONG\nx,y\n"
    r = client.post(
        "/api/inference/jobs",
        data={"predictor": "ngram", "version": "test-bad-csv"},
        files={"valid_csv": ("bad.csv", BytesIO(bad), "text/csv")},
    )
    assert r.status_code == 400


def test_gated_predictor_hf_without_env_returns_403(client, experiments_tmp, monkeypatch):
    monkeypatch.delenv("ZO_ALLOW_DASHBOARD_INFERENCE", raising=False)
    manual = json.dumps(
        {
            "task": "nextstep",
            "family": "MOSFET",
            "completion_fraction": 0.6,
            "partial_sequence": "SPIN COAT PHOTORESIST",
        }
    )
    r = client.post(
        "/api/inference/jobs",
        data={"predictor": "hf", "model": "gpt2", "manual_json": manual},
    )
    assert r.status_code == 403


def test_background_failure_marks_run_failed(client, experiments_tmp, monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("injected failure")

    monkeypatch.setattr("zo_eval.inference_jobs.run_track", _boom)
    manual = json.dumps(
        {
            "task": "nextstep",
            "family": "MOSFET",
            "completion_fraction": 0.6,
            "partial_sequence": "SPIN COAT PHOTORESIST",
        }
    )
    r = client.post(
        "/api/inference/jobs",
        data={"predictor": "ngram", "version": "test-fail", "tasks": "nextstep", "manual_json": manual},
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert _wait_run(run_id) == "failed"
    meta = get_run(run_id)
    assert meta and "injected failure" in (meta.notes or "")


def test_preview_ok(client, experiments_tmp):
    manual = json.dumps(
        {
            "task": "nextstep",
            "family": "MOSFET",
            "completion_fraction": 0.6,
            "partial_sequence": "SPIN COAT PHOTORESIST",
        }
    )
    r = client.post(
        "/api/inference/preview",
        data={"predictor": "ngram", "tasks": "nextstep", "manual_json": manual},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "nextstep" in r.json()["tasks"]
