"""Predict → write the 3 submission CSVs → score → log a tagged registry run. Model-agnostic.

This is the spine every stream shares: pick a ``Predictor`` (ngram / oracle / llm), point it at
organizer-format eval inputs, and get submission CSVs in ``extras/results/`` + scored, tagged runs
the dashboard can compare. Mirrors the organizers' ``eval_metrics.py`` so it's drop-in once theirs
lands at kickoff (the CSV formats already match ``submission.py``).

Metric/tag convention (the dashboard's contract — keep flat, the frontend reads `Record<str,num>`):
  scalars: top1/top3/top5/mrr · em/ned/token_acc/block_acc ·
           anomaly_acc/anomaly_p/anomaly_r/anomaly_f1/anomaly_auc/rule_attr_acc · cm_tp/fp/tn/fn
  per family: the same keys with a `_MOSFET|_IGBT|_IC` suffix.
  ID vs OOD is carried by run TAGS (split:id|ood), not metric names.
  tags: eval:nextstep|completion|anomaly · split:id|ood · family:* · predictor:ngram|oracle|llm|...
"""

from __future__ import annotations

import json
from pathlib import Path

from zo_common import append_metric, new_run, update_run
from zo_common.registry import get_run, run_dir

from zo_eval import submission as sub
from zo_eval import track_metrics as M

TASKS = ("nextstep", "completion", "anomaly")

_NEXT_MAP = {"top1": "top1", "top3": "top3", "top5": "top5", "mrr": "mrr"}
_COMPL_MAP = {"exact_match": "em", "norm_edit_dist": "ned", "token_acc": "token_acc", "block_acc": "block_acc"}
_ANOM_MAP = {
    "binary_acc": "anomaly_acc", "precision": "anomaly_p", "recall": "anomaly_r",
    "f1": "anomaly_f1", "roc_auc": "anomaly_auc", "rule_attribution_acc": "rule_attr_acc",
}


def _flat(per_family_result: dict, name_map: dict) -> dict:
    """per_family output {overall, MOSFET, ...} → flat scalars (overall unsuffixed, others _FAM)."""
    out: dict = {}
    for fam, scores in per_family_result.items():
        suffix = "" if fam == "overall" else f"_{fam}"
        for raw, flat in name_map.items():
            v = scores.get(raw)
            if isinstance(v, (int, float)):
                out[f"{flat}{suffix}"] = round(float(v), 4)
        conf = scores.get("confusion")
        if conf and fam == "overall":
            out.update({f"cm_{k}": conf[k] for k in ("tp", "fp", "tn", "fn")})
    return out


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
) -> dict:
    """Run a predictor over the requested tasks; write CSVs + score (if gold) + log a run."""
    if isinstance(gold, str):
        gold = json.loads(Path(gold).read_text())
    fam_of = (gold or {}).get("family_of", {})

    run = get_run(run_id) if run_id else None
    if run is None:
        run = new_run(
            name or f"track:{getattr(predictor, 'name', 'model')}",
            "eval",
            config={"predictor": getattr(predictor, "name", "?"), "tasks": list(tasks)},
            tags=tags or [],
        )
    update_run(run.id, status="running")
    out = Path(out_dir) if out_dir else (run_dir(run.id) / "results")  # namespaced per run
    out.mkdir(parents=True, exist_ok=True)

    metrics: dict = {}

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
                metrics.update(_flat(M.per_family(M.score_nextstep, preds, gold["next"], fam_of), _NEXT_MAP))
        if "completion" in tasks:
            preds = {it.example_id: _fallback(predictor.complete, it, []) for it in valid}
            sub.write_completion(list(preds.items()), out / "completion.csv")
            if gold and gold.get("completion"):
                metrics.update(_flat(M.per_family(M.score_completion, preds, gold["completion"], fam_of), _COMPL_MAP))

    if "anomaly" in tasks and anomaly_csv:
        anom = sub.read_anomaly_inputs(anomaly_csv)
        rows, preds = [], {}
        for it in anom:
            iv, sc, rule = _fallback(predictor.anomaly, it, (1, 0.5, None))
            rows.append((it.example_id, iv, sc, rule))
            preds[it.example_id] = {"is_valid": iv, "score": sc, "rule": rule}
        sub.write_anomaly(rows, out / "anomaly.csv")
        if gold and gold.get("anomaly"):
            metrics.update(_flat(M.per_family(M.score_anomaly, preds, gold["anomaly"], fam_of), _ANOM_MAP))

    if metrics:
        append_metric(run.id, step=0, **metrics)
    update_run(run.id, status="completed", metrics=metrics or {"note": "no gold — CSVs only"})
    return {"run_id": run.id, "out_dir": str(out), **metrics}
