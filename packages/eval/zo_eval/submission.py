"""Eval-input + submission CSV I/O for the Industrial AI track — EXACT organizer formats.

Getting the columns right matters most: a perfect model with the wrong header scores zero.
These match generation_rules.md §5.3 byte-for-byte. Organizers score our CSVs with their script;
``track_metrics.py`` implements the documented metrics for self-eval on data we label locally.

Eval inputs (organizers distribute at kickoff; we also synthesize our own for local OOD eval):
  eval_input_valid.csv   : EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE   (Tasks 1 & 2)
  eval_input_anomaly.csv : EXAMPLE_ID, FAMILY, SEQUENCE                                 (Task 3)

Submissions (we write; → extras/results/):
  nextstep.csv   : EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5
  completion.csv : EXAMPLE_ID, PREDICTED_SEQUENCE          (pipe-joined; steps AFTER the cut only)
  anomaly.csv    : EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

STEP_SEP = "|"  # pipe, no surrounding spaces — matches the eval files


# --------------------------------------------------------------------------- read eval inputs


@dataclass
class ValidInput:
    example_id: str
    family: str
    completion_fraction: float
    partial_sequence: list[str]


@dataclass
class AnomalyInput:
    example_id: str
    family: str
    sequence: list[str]


def _norm_rows(path: str | Path):
    """Yield dict rows with BOM/quote/space-stripped header keys (robust to CSV quirks)."""
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        keymap = {k: (k or "").lstrip("﻿").strip().strip('"').strip() for k in (reader.fieldnames or [])}
        for row in reader:
            yield {keymap[k]: (v or "").strip() for k, v in row.items() if k in keymap}


def _split(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(STEP_SEP) if x.strip()]


def read_valid_inputs(path: str | Path) -> list[ValidInput]:
    return [
        ValidInput(
            r["EXAMPLE_ID"],
            r.get("FAMILY", ""),
            float(r["COMPLETION_FRACTION"]) if r.get("COMPLETION_FRACTION") else 0.0,
            _split(r.get("PARTIAL_SEQUENCE", "")),
        )
        for r in _norm_rows(path)
    ]


def read_anomaly_inputs(path: str | Path) -> list[AnomalyInput]:
    return [
        AnomalyInput(r["EXAMPLE_ID"], r.get("FAMILY", ""), _split(r.get("SEQUENCE", "")))
        for r in _norm_rows(path)
    ]


# --------------------------------------------------------------------------- write submissions


def write_nextstep(rows: list[tuple[str, list[str]]], path: str | Path) -> Path:
    """``rows`` = (example_id, [ranked step names]); padded/truncated to 5 ranks."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for ex, ranks in rows:
            r = list(ranks)[:5] + [""] * (5 - len(ranks))
            w.writerow([ex, *r])
    return path


def write_completion(rows: list[tuple[str, list[str]]], path: str | Path) -> Path:
    """``rows`` = (example_id, [steps AFTER the cut]). Do NOT include the given partial."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for ex, steps in rows:
            w.writerow([ex, STEP_SEP.join(steps)])
    return path


def write_anomaly(rows: list[tuple[str, int, float | None, str | None]], path: str | Path) -> Path:
    """``rows`` = (example_id, is_valid 1/0, score in [0,1] or None, predicted_rule or None)."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        for ex, is_valid, score, rule in rows:
            w.writerow([ex, int(is_valid), "" if score is None else f"{float(score):.4f}", rule or ""])
    return path


# --------------------------------------------------------------------------- local eval set


def make_local_eval_set(
    valid_seqs: list[tuple[str, list[str]]],
    negatives: list[dict],
    out_dir: str | Path,
    fractions: tuple[float, ...] = (0.6, 0.8),
) -> dict:
    """Synthesize organizer-format eval inputs + gold from held-out data, so the full
    input→predict→write→score path runs *today* (and unchanged when kickoff CSVs land).

    ``valid_seqs`` = (family, steps) held-out valid sequences (e.g. a LOFO test family);
    ``negatives`` = datagen records {steps, rules, ...} for the anomaly mix.
    Writes ``eval_input_valid.csv``, ``eval_input_anomaly.csv``, and ``gold.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_rows, gold_next, gold_compl, fam_of, cut_of = [], {}, {}, {}, {}
    i = 0
    for fam, steps in valid_seqs:
        for frac in fractions:
            cut = max(1, int(len(steps) * frac))
            if cut >= len(steps):
                continue
            ex = f"valid_{i:04d}"
            i += 1
            valid_rows.append([ex, fam, frac, STEP_SEP.join(steps[:cut])])
            gold_next[ex] = steps[cut]
            gold_compl[ex] = steps[cut:]
            fam_of[ex] = fam
            cut_of[ex] = frac
    with (out_dir / "eval_input_valid.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"])
        w.writerows(valid_rows)

    anom_rows, gold_anom = [], {}
    items = [("valid", fam, steps, None) for fam, steps in valid_seqs]
    items += [("forbidden", n.get("family", "?"), n["steps"], (n["rules"][0] if n["rules"] else None)) for n in negatives]
    for j, (kind, fam, steps, rule) in enumerate(items):
        ex = f"{kind}_{j:04d}"
        anom_rows.append([ex, fam, STEP_SEP.join(steps)])
        gold_anom[ex] = {"is_valid": 1 if kind == "valid" else 0, "rule": rule}
        fam_of[ex] = fam
    with (out_dir / "eval_input_anomaly.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["EXAMPLE_ID", "FAMILY", "SEQUENCE"])
        w.writerows(anom_rows)

    gold = {
        "next": gold_next,
        "completion": gold_compl,
        "anomaly": gold_anom,
        "family_of": fam_of,
        "cut_fraction_of": cut_of,
    }
    (out_dir / "gold.json").write_text(json.dumps(gold, indent=2))
    return gold


# --------------------------------------------------------------------------- round-trip smoke


def smoke() -> None:  # pragma: no cover - manual
    """Prove the path: synth eval inputs from held-out data → perfect & degraded predictions →
    write submission CSVs → read back → score. Asserts perfect preds give perfect scores."""
    import random
    import tempfile

    from zo_train.datagen import make_negative, make_splits
    from zo_train.fab import read_sequences

    from zo_eval import track_metrics as M

    rng = random.Random(0)
    sp = make_splits()
    held = "MOSFET"  # pretend this is the OOD/held-out family
    seqs = read_sequences(held)
    test = [seqs[i] for i in sp["per_family"][held]["test"][:40]]
    valid_seqs = [(held, s) for s in test]
    negs = []
    for s in test[:20]:
        neg = make_negative(s, rng)
        if neg:
            neg["family"] = held
            negs.append(neg)

    out = Path(tempfile.mkdtemp())
    gold = make_local_eval_set(valid_seqs, negs, out)
    print(f"local eval set: {len(gold['next'])} next/compl items, {len(gold['anomaly'])} anomaly items")

    # --- PERFECT predictions (sanity: scores should be maxed) ---
    write_nextstep([(ex, [g]) for ex, g in gold["next"].items()], out / "nextstep.csv")
    write_completion(list(gold["completion"].items()), out / "completion.csv")
    write_anomaly(
        [(ex, g["is_valid"], 1.0 if g["is_valid"] else 0.0, g["rule"]) for ex, g in gold["anomaly"].items()],
        out / "anomaly.csv",
    )
    # read back the nextstep CSV to prove the writer/columns round-trip
    ns_back = {r["EXAMPLE_ID"]: [r[f"RANK_{k}"] for k in range(1, 6) if r.get(f"RANK_{k}")] for r in _norm_rows(out / "nextstep.csv")}
    ns = M.score_nextstep(ns_back, gold["next"])
    cp = M.score_completion(dict(gold["completion"]), gold["completion"])
    an = M.score_anomaly(
        {ex: {"is_valid": g["is_valid"], "score": 1.0 if g["is_valid"] else 0.0, "rule": g["rule"]} for ex, g in gold["anomaly"].items()},
        gold["anomaly"],
    )
    print("PERFECT  next:", {k: round(v, 3) for k, v in ns.items() if k != "n"})
    print("PERFECT  compl:", {k: round(v, 3) for k, v in cp.items() if k != "n"})
    print("PERFECT  anom:", {k: v for k, v in an.items() if k in ("binary_acc", "f1", "roc_auc", "rule_attribution_acc")})
    assert ns["top1"] == 1.0 and cp["exact_match"] == 1.0 and an["f1"] == 1.0, "perfect preds must score perfectly"

    # --- DEGRADED anomaly (predict everything valid): recall→0, but TN keep some acc ---
    an2 = M.score_anomaly({ex: {"is_valid": 1, "score": 0.9, "rule": None} for ex in gold["anomaly"]}, gold["anomaly"])
    print("ALL-VALID anom:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in an2.items() if k in ("binary_acc", "recall", "f1")})
    assert an2["recall"] == 0.0, "predicting all-valid must give zero recall on anomalies"
    print("OK — submission round-trip + scorer verified.")


if __name__ == "__main__":  # pragma: no cover
    smoke()
