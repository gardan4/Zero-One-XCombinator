"""Predict → write the 3 submission CSVs → score → log a tagged registry run. Model-agnostic.

This is the spine every stream shares: pick a ``Predictor`` (ngram / oracle / llm / hf / …), point it
at organizer-format eval inputs, and get submission CSVs in ``extras/results/`` + scored, tagged runs
the dashboard can compare. CSV formats match ``submission.py``; metrics match ``track_metrics.py``
(documented in ``generation_rules.md`` §5 — organizers score submissions with their own tooling).

Metric/tag convention (the dashboard's contract — keep flat, the frontend reads `Record<str,num>`):
  scalars: top1/top3/top5/mrr · em/ned/token_acc/block_acc ·
           anomaly_acc/anomaly_p/anomaly_r/anomaly_f1/anomaly_auc/rule_attr_acc · cm_tp/fp/tn/fn
  per family: _MOSFET|_IGBT|_IC · per cut: _frac60|_frac80
  repro tags (required): version:<label> · model-ref:<hf-id> · eval-set:local|kickoff ·
    split:id|ood · family:* · predictor:*
"""

from __future__ import annotations

import json
from pathlib import Path

from zo_common import append_metric, new_run, update_run
from zo_common.registry import get_run, run_dir

from zo_eval import submission as sub
from zo_eval import track_metrics as M

TASKS = ("nextstep", "completion", "anomaly")


def build_run_tags(
    *,
    version: str,
    predictor: str,
    model_ref: str | None = None,
    eval_set: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Standard tags so baseline vs finetuned runs compare cleanly in the dashboard."""
    tags = [f"version:{version}", f"predictor:{predictor}"]
    if model_ref:
        tags.append(f"model-ref:{model_ref.replace('/', '--')}")
    if eval_set:
        tags.append(f"eval-set:{eval_set}")
    if extra:
        for t in extra:
            t = t.strip()
            if t and t not in tags:
                tags.append(t)
    return tags


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
) -> dict:
    """Run a predictor over the requested tasks; write CSVs + score (if gold) + log a run."""
    if isinstance(gold, str):
        gold = json.loads(Path(gold).read_text())
    fam_of = (gold or {}).get("family_of", {})
    cut_of = (gold or {}).get("cut_fraction_of", {})

    pred_name = getattr(predictor, "name", "model")
    tag_list = build_run_tags(
        version=version,
        predictor=pred_name,
        model_ref=model_ref,
        eval_set=eval_set,
        extra=tags,
    )

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
            },
            tags=tag_list,
        )
    else:
        update_run(run.id, tags=tag_list, config={
            **(run.config or {}),
            "predictor": pred_name,
            "version": version,
            "model_ref": model_ref,
            "eval_set": eval_set,
        })
    update_run(run.id, status="running")
    out = Path(out_dir) if out_dir else (run_dir(run.id) / "results")
    out.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, float] = {}
    report_tasks: dict = {}

    def _fallback(fn, item, default):
        try:
            return fn(item)
        except Exception:
            return default

    if {"nextstep", "completion"} & set(tasks) and valid_csv:
        valid = sub.read_valid_inputs(valid_csv)
        if "nextstep" in tasks:
            preds = {it.example_id: _fallback(predictor.next_step, it, []) for it in valid}
            sub.write_nextstep(list(preds.items()), out / "nextstep.csv")
            if gold and gold.get("next"):
                fam_br = M.per_family(M.score_nextstep, preds, gold["next"], fam_of)
                metrics.update(M.flatten_breakdown(fam_br, M._FLAT_NEXT))
                entry: dict = {"by_family": fam_br}
                if cut_of:
                    cut_br = M.per_cut_fraction(M.score_nextstep, preds, gold["next"], cut_of)
                    metrics.update(M.flatten_breakdown(cut_br, M._FLAT_NEXT))
                    entry["by_cut"] = cut_br
                report_tasks["nextstep"] = entry
        if "completion" in tasks:
            preds = {it.example_id: _fallback(predictor.complete, it, []) for it in valid}
            sub.write_completion(list(preds.items()), out / "completion.csv")
            if gold and gold.get("completion"):
                fam_br = M.per_family(M.score_completion, preds, gold["completion"], fam_of)
                metrics.update(M.flatten_breakdown(fam_br, M._FLAT_COMPL))
                entry = {"by_family": fam_br}
                if cut_of:
                    cut_br = M.per_cut_fraction(
                        M.score_completion, preds, gold["completion"], cut_of
                    )
                    metrics.update(M.flatten_breakdown(cut_br, M._FLAT_COMPL))
                    entry["by_cut"] = cut_br
                report_tasks["completion"] = entry

    if "anomaly" in tasks and anomaly_csv:
        anom = sub.read_anomaly_inputs(anomaly_csv)
        rows, preds = [], {}
        for it in anom:
            iv, sc, rule = _fallback(predictor.anomaly, it, (1, 0.5, None))
            rows.append((it.example_id, iv, sc, rule))
            preds[it.example_id] = {"is_valid": iv, "score": sc, "rule": rule}
        sub.write_anomaly(rows, out / "anomaly.csv")
        if gold and gold.get("anomaly"):
            fam_br = M.per_family(M.score_anomaly, preds, gold["anomaly"], fam_of)
            report_tasks["anomaly"] = {"by_family": fam_br}
            metrics.update(M.flatten_breakdown(fam_br, M._FLAT_ANOM, include_confusion=True))

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
    (out / "metrics_report.json").write_text(json.dumps(report, indent=2))
    (out / "metrics_report.md").write_text(M.format_report_markdown(report))

    if metrics:
        append_metric(run.id, step=0, **metrics)
    update_run(run.id, status="completed", metrics=metrics or {"note": "no gold — CSVs only"})
    return {"run_id": run.id, "out_dir": str(out), "version": version, **metrics}
