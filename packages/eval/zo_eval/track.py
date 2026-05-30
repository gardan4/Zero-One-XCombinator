"""Predict → write the 3 submission CSVs → score → log a tagged registry run. Model-agnostic.

Outputs per run (under ``experiments/<run_id>/results/`` unless ``out_dir`` set):

- ``nextstep.csv``, ``completion.csv``, ``anomaly.csv`` — organizer submission formats
- ``metrics_report.json`` / ``metrics_report.md`` — labeled metrics (requires ``gold``)
- ``proxy_report.json`` — kickoff / unlabeled sanity proxies (grammar, vocab, oracle agreement)
- ``official_scores.txt`` — vendored ``eval_metrics.py`` transcript (``self_check=True`` + gold)
- ``manifest.json`` — repro metadata for promote / compare
- Optional promote → ``extras/results/<slug>/`` + ``extras/results/INDEX.json``
"""

from __future__ import annotations

import json
from pathlib import Path

from zo_common import append_metric, new_run, update_run
from zo_common.registry import get_run, run_dir

from zo_eval import submission as sub
from zo_eval import track_metrics as M
from zo_eval.examples_trace import (
    TraceCollector,
    anomaly_input_payload,
    valid_input_payload,
)
from zo_eval.predict_trace import wrap_with_tracing
from zo_eval.proxy_metrics import build_proxy_report
from zo_eval.results_io import load_predictions_from_dir, promote_results, write_manifest
from zo_eval.self_check import run_self_check

TASKS = ("nextstep", "completion", "anomaly")


def build_run_tags(
    *,
    version: str,
    predictor: str,
    model_ref: str | None = None,
    eval_set: str | None = None,
    train_run_id: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    tags = [f"version:{version}", f"predictor:{predictor}"]
    if model_ref:
        tags.append(f"model-ref:{model_ref.replace('/', '--')}")
    if eval_set:
        tags.append(f"eval-set:{eval_set}")
    if train_run_id:
        tags.append(f"train-run:{train_run_id}")
    if extra:
        for t in extra:
            t = t.strip()
            if t and t not in tags:
                tags.append(t)
    return tags


def rescore_results(
    results_dir: str | Path,
    gold: dict | str | Path,
    *,
    run_id: str | None = None,
    version: str = "rescore",
    predictor: str = "rescore",
    model_ref: str | None = None,
    eval_set: str | None = None,
    tags: list[str] | None = None,
    self_check: bool = False,
    valid_csv: str | None = None,
    anomaly_csv: str | None = None,
) -> dict:
    """Score existing CSVs against ``gold`` without re-running inference."""
    results_dir = Path(results_dir)
    gold_path = gold if isinstance(gold, (str, Path)) else None
    if isinstance(gold, (str, Path)):
        gold = json.loads(Path(gold).read_text())
    preds = load_predictions_from_dir(results_dir)
    fam_of = gold.get("family_of", {})
    cut_of = gold.get("cut_fraction_of", {})

    tag_list = build_run_tags(
        version=version,
        predictor=predictor,
        model_ref=model_ref,
        eval_set=eval_set,
        extra=tags,
    )
    run = get_run(run_id) if run_id else None
    if run is None:
        run = new_run(
            f"track:rescore:{version}",
            "eval",
            config={"mode": "rescore", "results_dir": str(results_dir), "version": version},
            tags=tag_list,
        )
    update_run(run.id, status="running", tags=tag_list)

    metrics: dict[str, float] = {}
    report_tasks: dict = {}

    if gold.get("next") and preds["nextstep"]:
        fam_br = M.per_family(M.score_nextstep, preds["nextstep"], gold["next"], fam_of)
        metrics.update(M.flatten_breakdown(fam_br, M._FLAT_NEXT))
        entry: dict = {"by_family": fam_br}
        if cut_of:
            cut_br = M.per_cut_fraction(M.score_nextstep, preds["nextstep"], gold["next"], cut_of)
            metrics.update(M.flatten_breakdown(cut_br, M._FLAT_NEXT))
            entry["by_cut"] = cut_br
        report_tasks["nextstep"] = entry

    if gold.get("completion") and preds["completion"]:
        fam_br = M.per_family(M.score_completion, preds["completion"], gold["completion"], fam_of)
        metrics.update(M.flatten_breakdown(fam_br, M._FLAT_COMPL))
        entry = {"by_family": fam_br}
        if cut_of:
            cut_br = M.per_cut_fraction(M.score_completion, preds["completion"], gold["completion"], cut_of)
            metrics.update(M.flatten_breakdown(cut_br, M._FLAT_COMPL))
            entry["by_cut"] = cut_br
        report_tasks["completion"] = entry

    if gold.get("anomaly") and preds["anomaly"]:
        anom_preds = {ex: {"is_valid": v["is_valid"], "score": v["score"], "rule": v["rule"]} for ex, v in preds["anomaly"].items()}
        fam_br = M.per_family(M.score_anomaly, anom_preds, gold["anomaly"], fam_of)
        report_tasks["anomaly"] = {"by_family": fam_br}
        metrics.update(M.flatten_breakdown(fam_br, M._FLAT_ANOM, include_confusion=True))

    report = M.build_metrics_report(
        version=version,
        predictor=predictor,
        model_ref=model_ref,
        eval_set=eval_set,
        tags=tag_list,
        nextstep=report_tasks.get("nextstep"),
        completion=report_tasks.get("completion"),
        anomaly=report_tasks.get("anomaly"),
    )
    (results_dir / "metrics_report.json").write_text(json.dumps(report, indent=2))
    (results_dir / "metrics_report.md").write_text(M.format_report_markdown(report))

    if self_check and valid_csv and anomaly_csv:
        run_self_check(results_dir, gold=gold, valid_csv=valid_csv, anomaly_csv=anomaly_csv)

    gold_manifest = gold_path if gold_path else (gold if isinstance(gold, str) else None)
    artifacts = [p.name for p in results_dir.iterdir() if p.is_file()]
    write_manifest(
        results_dir,
        run_id=run.id,
        version=version,
        predictor=predictor,
        model_ref=model_ref,
        eval_set=eval_set,
        tags=tag_list,
        valid_csv=valid_csv,
        anomaly_csv=anomaly_csv,
        gold=gold_manifest,
        artifacts=artifacts,
    )

    if metrics:
        append_metric(run.id, step=0, **metrics)
    update_run(run.id, status="completed", metrics=metrics or {"note": "rescore — no matching preds"})
    return {"run_id": run.id, "out_dir": str(results_dir), "version": version, **metrics}


def run_track(
    predictor,
    valid_csv: str | None = None,
    anomaly_csv: str | None = None,
    gold: dict | str | None = None,
    tasks: tuple[str, ...] = TASKS,
    out_dir: str | None = None,
    run_id: str | None = None,
    tags: list[str] | None = None,
    name: str | None = None,
    version: str = "unspecified",
    model_ref: str | None = None,
    eval_set: str | None = None,
    gold_path: str | None = None,
    train_run_id: str | None = None,
    notes: str | None = None,
    *,
    run_proxy: bool = True,
    self_check: bool = False,
    promote: str | None = None,
    write_examples: bool = True,
) -> dict:
    """Run a predictor over the requested tasks; write CSVs + score + log a run."""
    if gold_path and gold is None:
        gold = gold_path
    if isinstance(gold, str):
        gold_path = gold_path or gold
        gold = json.loads(Path(gold).read_text())
    fam_of = (gold or {}).get("family_of", {})
    cut_of = (gold or {}).get("cut_fraction_of", {})

    traced = wrap_with_tracing(predictor)
    pred_name = getattr(predictor, "name", "model")
    tag_list = build_run_tags(
        version=version,
        predictor=pred_name,
        model_ref=model_ref,
        eval_set=eval_set,
        train_run_id=train_run_id,
        extra=tags,
    )

    train_meta = get_run(train_run_id) if train_run_id else None

    run = get_run(run_id) if run_id else None
    if run is None:
        run = new_run(
            name or f"track:{pred_name}:{version}",
            "eval",
            config={
                "predictor": pred_name,
                "version": version,
                "model_ref": model_ref,
                "eval_set": eval_set,
                "tasks": list(tasks),
                "valid_csv": valid_csv,
                "anomaly_csv": anomaly_csv,
                "train_run_id": train_run_id,
            },
            tags=tag_list,
        )
    else:
        update_run(
            run.id,
            tags=tag_list,
            config={
                **(run.config or {}),
                "predictor": pred_name,
                "version": version,
                "model_ref": model_ref,
                "eval_set": eval_set,
                "train_run_id": train_run_id,
            },
        )
    if notes:
        update_run(run.id, notes=notes)
    update_run(run.id, status="running")
    out = Path(out_dir) if out_dir else (run_dir(run.id) / "results")
    out.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, float] = {}
    report_tasks: dict = {}
    nextstep_preds: dict[str, list[str]] = {}
    completion_preds: dict[str, list[str]] = {}
    anomaly_preds: dict[str, dict] = {}
    valid_inputs: list[sub.ValidInput] = []
    anomaly_inputs: list[sub.AnomalyInput] = []
    trace_col = TraceCollector() if write_examples else None

    def _fallback(fn, item, default):
        try:
            return fn(item)
        except Exception:
            return default

    if {"nextstep", "completion"} & set(tasks) and valid_csv:
        valid_inputs = sub.read_valid_inputs(valid_csv)
        if "nextstep" in tasks:
            nextstep_preds = {}
            for it in valid_inputs:
                ranks = _fallback(traced.next_step, it, [])
                nextstep_preds[it.example_id] = ranks
                if trace_col is not None:
                    trace_col.add(
                        example_id=it.example_id,
                        task="nextstep",
                        family=it.family,
                        input_payload=valid_input_payload(it),
                        prediction=ranks,
                        gold=gold.get("next", {}).get(it.example_id) if gold else None,
                        trace=traced.pop_trace(it.example_id, "nextstep"),
                    )
            sub.write_nextstep(list(nextstep_preds.items()), out / "nextstep.csv")
            if gold and gold.get("next"):
                fam_br = M.per_family(M.score_nextstep, nextstep_preds, gold["next"], fam_of)
                metrics.update(M.flatten_breakdown(fam_br, M._FLAT_NEXT))
                entry: dict = {"by_family": fam_br}
                if cut_of:
                    cut_br = M.per_cut_fraction(M.score_nextstep, nextstep_preds, gold["next"], cut_of)
                    metrics.update(M.flatten_breakdown(cut_br, M._FLAT_NEXT))
                    entry["by_cut"] = cut_br
                report_tasks["nextstep"] = entry
        if "completion" in tasks:
            completion_preds = {}
            for it in valid_inputs:
                steps = _fallback(traced.complete, it, [])
                completion_preds[it.example_id] = steps
                if trace_col is not None:
                    trace_col.add(
                        example_id=it.example_id,
                        task="completion",
                        family=it.family,
                        input_payload=valid_input_payload(it),
                        prediction=steps,
                        gold=gold.get("completion", {}).get(it.example_id) if gold else None,
                        trace=traced.pop_trace(it.example_id, "completion"),
                    )
            sub.write_completion(list(completion_preds.items()), out / "completion.csv")
            if gold and gold.get("completion"):
                fam_br = M.per_family(M.score_completion, completion_preds, gold["completion"], fam_of)
                metrics.update(M.flatten_breakdown(fam_br, M._FLAT_COMPL))
                entry = {"by_family": fam_br}
                if cut_of:
                    cut_br = M.per_cut_fraction(
                        M.score_completion, completion_preds, gold["completion"], cut_of
                    )
                    metrics.update(M.flatten_breakdown(cut_br, M._FLAT_COMPL))
                    entry["by_cut"] = cut_br
                report_tasks["completion"] = entry

    if "anomaly" in tasks and anomaly_csv:
        anomaly_inputs = sub.read_anomaly_inputs(anomaly_csv)
        rows = []
        for it in anomaly_inputs:
            iv, sc, rule = _fallback(traced.anomaly, it, (1, 0.5, None))
            rows.append((it.example_id, iv, sc, rule))
            pred_dict = {"is_valid": iv, "score": sc, "rule": rule}
            anomaly_preds[it.example_id] = pred_dict
            if trace_col is not None:
                trace_col.add(
                    example_id=it.example_id,
                    task="anomaly",
                    family=it.family,
                    input_payload=anomaly_input_payload(it),
                    prediction=pred_dict,
                    gold=gold.get("anomaly", {}).get(it.example_id) if gold else None,
                    trace=traced.pop_trace(it.example_id, "anomaly"),
                )
        sub.write_anomaly(rows, out / "anomaly.csv")
        if gold and gold.get("anomaly"):
            anom_for_score = {
                ex: {"is_valid": v["is_valid"], "score": v["score"], "rule": v["rule"]}
                for ex, v in anomaly_preds.items()
            }
            fam_br = M.per_family(M.score_anomaly, anom_for_score, gold["anomaly"], fam_of)
            report_tasks["anomaly"] = {"by_family": fam_br}
            metrics.update(M.flatten_breakdown(fam_br, M._FLAT_ANOM, include_confusion=True))

    if run_proxy and not gold:
        proxy = build_proxy_report(
            valid_inputs=valid_inputs or None,
            anomaly_inputs=anomaly_inputs or None,
            nextstep_preds=nextstep_preds or None,
            completion_preds=completion_preds or None,
            anomaly_preds=anomaly_preds or None,
        )
        (out / "proxy_report.json").write_text(json.dumps(proxy, indent=2))
        for section, scores in proxy.items():
            if section == "completion_grammar":
                metrics["proxy_grammar_valid"] = round(float(scores.get("grammar_valid_rate", 0)), 4)
            elif section == "nextstep_vocab":
                metrics["proxy_rank1_vocab"] = round(float(scores.get("rank1_in_vocab", 0)), 4)
            elif section == "anomaly_vs_oracle":
                metrics["proxy_oracle_acc"] = round(float(scores.get("oracle_agreement_acc", 0)), 4)

    report = M.build_metrics_report(
        version=version,
        predictor=pred_name,
        model_ref=model_ref,
        eval_set=eval_set,
        tags=tag_list,
        nextstep=report_tasks.get("nextstep"),
        completion=report_tasks.get("completion"),
        anomaly=report_tasks.get("anomaly"),
    )
    if gold:
        (out / "metrics_report.json").write_text(json.dumps(report, indent=2))
        (out / "metrics_report.md").write_text(M.format_report_markdown(report))

    if self_check and gold and valid_csv and anomaly_csv:
        run_self_check(out, gold=gold, valid_csv=valid_csv, anomaly_csv=anomaly_csv)

    if trace_col is not None and trace_col.rows:
        trace_col.write(out / "examples.jsonl")

    artifacts = sorted(p.name for p in out.iterdir() if p.is_file())
    if (out / "ground_truth").is_dir():
        artifacts.append("ground_truth/")
    manifest_gold = gold_path if gold_path else None
    train_snapshot = None
    if train_meta:
        train_snapshot = {
            "id": train_meta.id,
            "name": train_meta.name,
            "tags": train_meta.tags,
            "config": train_meta.config,
        }
    write_manifest(
        out,
        run_id=run.id,
        version=version,
        predictor=pred_name,
        model_ref=model_ref,
        eval_set=eval_set,
        tags=tag_list,
        valid_csv=valid_csv,
        anomaly_csv=anomaly_csv,
        gold=manifest_gold,
        artifacts=artifacts,
        train_run_id=train_run_id,
        notes=notes,
        train_run=train_snapshot,
    )

    if metrics:
        append_metric(run.id, step=0, **metrics)
    update_run(run.id, status="completed", metrics=metrics or {"note": "no gold — CSVs + proxy only"})

    result = {"run_id": run.id, "out_dir": str(out), "version": version, **metrics}
    if promote:
        dest = promote_results(out, promote)
        result["promoted_to"] = str(dest)
    return result
