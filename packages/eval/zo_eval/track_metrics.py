"""Self-eval scorer for the 3 Industrial AI track tasks.

Organizers score submitted ``nextstep.csv`` / ``completion.csv`` / ``anomaly.csv`` with the vendored
``data/industrial-infineon/eval/eval_metrics.py`` (requires their ground-truth CSVs). This module
implements the **same** metric definitions for local hold-out / ``gold.json`` scoring via
``zo-track predict``.

> Submission **column names** are authoritative — wrong format → zero on their side. For exact parity
> with organizer reports when you have labels, use ``zo-track score-official`` (wraps ``eval_metrics.py``).

Conventions:
- **Anomaly = positive class is INVALID** (we are detecting violations). So precision/recall/F1
  are computed with "predicted invalid" vs "truly invalid".
- **ROC-AUC** uses ``SCORE`` = P(valid) against the ``IS_VALID`` label (higher score ⇒ valid);
  this equals the invalid-class AUC under ``1-SCORE``. Blank scores are imputed from ``IS_VALID``,
  matching the organizer script. ``None`` if only one class is present.
- **Rule-Attribution Acc** is measured *among detected violations* (truly-invalid AND predicted
  invalid), per the spec.
"""

from __future__ import annotations

from collections.abc import Sequence

# --------------------------------------------------------------------------- helpers


def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance over token lists (not characters)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """AUC via the Mann–Whitney rank statistic (tie-aware). ``labels`` are 1/0; higher
    ``score`` should rank label==1 higher. Returns ``None`` if a class is absent."""
    pairs = sorted(zip(scores, labels, strict=False), key=lambda x: x[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(1 for _, lbl in pairs if lbl == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sum_pos = sum(r for r, (_, lbl) in zip(ranks, pairs, strict=False) if lbl == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


# --------------------------------------------------------------------------- Task 1


def score_nextstep(preds: dict[str, list[str]], gold: dict[str, str]) -> dict:
    """``preds[ex]`` = up to 5 ranked step names; ``gold[ex]`` = the true next step.
    Returns Top-1/3/5 accuracy + MRR (over the examples present in ``gold``)."""
    n = len(gold)
    top1 = top3 = top5 = 0
    rr = 0.0
    for ex, true in gold.items():
        ranks = preds.get(ex, [])
        pos = ranks.index(true) + 1 if true in ranks else 0
        if pos == 1:
            top1 += 1
        if 1 <= pos <= 3:
            top3 += 1
        if 1 <= pos <= 5:
            top5 += 1
        rr += 1.0 / pos if pos else 0.0
    return {
        "n": n,
        "top1": _safe_div(top1, n),
        "top3": _safe_div(top3, n),
        "top5": _safe_div(top5, n),
        "mrr": _safe_div(rr, n),
    }


# --------------------------------------------------------------------------- Task 2 (matches eval_metrics.py)


def _major_block(step: str) -> str:
    """Map a process step to a coarse major process block (organizer block-level metric)."""
    s = step.upper()
    if "LITHO" in s or s.startswith("SPIN COAT PHOTORESIST") or "MASK LEVEL" in s:
        return "LITHO"
    if "ETCH" in s or s.startswith("OPEN PAD WINDOW"):
        return "ETCH"
    if "IMPLANT" in s or "ANNEAL" in s or "DIFFUSION" in s:
        return "DOPING_THERMAL"
    if s.startswith("DEPOSIT") or "OXIDATION" in s or "GROWTH" in s:
        return "DEPOSITION"
    if s.startswith("CMP") or "PLANAR" in s:
        return "PLANARIZATION"
    if "VIA" in s:
        return "VIA"
    if "PASSIVATION" in s:
        return "PASSIVATION"
    if "BACKSIDE" in s or "GRIND" in s:
        return "BACKSIDE"
    if "TEST" in s or "MEASURE" in s or "INSPECT" in s or "ANALYSIS" in s:
        return "METROLOGY_TEST"
    if "LOT" in s or "RELEASE" in s or "SHIP" in s:
        return "LOGISTICS"
    return "OTHER"


def _block_signature(seq: list[str]) -> list[str]:
    """Collapse a token sequence to de-duplicated major-process blocks."""
    sig: list[str] = []
    prev: str | None = None
    for step in seq:
        b = _major_block(step)
        if b != prev:
            sig.append(b)
            prev = b
    return sig


def token_accuracy(pred: list[str], ref: list[str]) -> float:
    """Position-aligned token matches up to min length (organizer definition)."""
    n = min(len(pred), len(ref))
    if n == 0:
        return 0.0
    return sum(p == r for p, r in zip(pred, ref, strict=False)) / n


def block_level_accuracy(pred: list[str], ref: list[str]) -> float:
    """Accuracy over collapsed major-process block signatures."""
    return token_accuracy(_block_signature(pred), _block_signature(ref))


def score_completion(preds: dict[str, list[str]], gold: dict[str, list[str]]) -> dict:
    """``preds[ex]``/``gold[ex]`` = the predicted/true steps AFTER the cut."""
    n = len(gold)
    em = 0
    ned_sum = tok_sum = blk_sum = 0.0
    for ex, g in gold.items():
        p = preds.get(ex, [])
        if p == g:
            em += 1
        denom = max(len(p), len(g)) or 1
        ned_sum += _levenshtein(p, g) / denom
        tok_sum += token_accuracy(p, g)
        blk_sum += block_level_accuracy(p, g)
    return {
        "n": n,
        "exact_match": _safe_div(em, n),
        "norm_edit_dist": _safe_div(ned_sum, n),
        "token_acc": _safe_div(tok_sum, n),
        "block_acc": _safe_div(blk_sum, n),
    }


# --------------------------------------------------------------------------- Task 3


def score_anomaly(
    preds: dict[str, dict], gold: dict[str, dict]
) -> dict:
    """``preds[ex]`` = {is_valid:0/1, score:float|None, rule:str|None};
    ``gold[ex]`` = {is_valid:0/1, rule:str|None}. Positive class = INVALID."""
    tp = fp = tn = fn = 0
    attr_ok = attr_total = 0
    scores: list[float] = []
    labels: list[int] = []
    for ex, g in gold.items():
        p = preds.get(ex, {"is_valid": 1, "score": None, "rule": None})
        g_invalid = g["is_valid"] == 0
        p_invalid = p.get("is_valid", 1) == 0
        if g_invalid and p_invalid:
            tp += 1
        elif (not g_invalid) and p_invalid:
            fp += 1
        elif (not g_invalid) and (not p_invalid):
            tn += 1
        else:
            fn += 1
        if g_invalid and p_invalid:  # rule attribution among detected violations
            attr_total += 1
            if (p.get("rule") or None) == (g.get("rule") or None):
                attr_ok += 1
        s = p.get("score")
        if s is None:
            s = float(p.get("is_valid", 1))
        scores.append(float(s))
        labels.append(int(g["is_valid"]))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    per_rule: dict[str, dict] = {}
    for ex, g in gold.items():
        if g.get("is_valid") != 0:
            continue
        rule = g.get("rule") or "UNKNOWN"
        p = preds.get(ex, {})
        detected = p.get("is_valid", 1) == 0
        bucket = per_rule.setdefault(rule, {"detected": 0, "total": 0})
        bucket["total"] += 1
        if detected:
            bucket["detected"] += 1
    per_rule_rate = {
        rule: _safe_div(v["detected"], v["total"]) for rule, v in sorted(per_rule.items())
    }
    return {
        "n": len(gold),
        "binary_acc": _safe_div(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "roc_auc": roc_auc(scores, labels) if scores else None,
        "rule_attribution_acc": (_safe_div(attr_ok, attr_total) if attr_total else None),
        "per_rule_detection": per_rule_rate,
    }


# --------------------------------------------------------------------------- breakdown


def per_family(score_fn, preds: dict, gold: dict, family_of: dict[str, str]) -> dict:
    """Run ``score_fn`` overall + split by family (the per-family breakdown the report needs)."""
    return per_group(score_fn, preds, gold, family_of)


def _frac_key(frac: float) -> str:
    """Stable suffix for completion-fraction breakdowns (0.6 → ``frac60``)."""
    return f"frac{int(round(float(frac) * 100))}"


def per_cut_fraction(
    score_fn, preds: dict, gold: dict, cut_of: dict[str, float]
) -> dict:
    """Run ``score_fn`` overall + split by ``COMPLETION_FRACTION`` (60% vs 80% cuts)."""
    return per_group(
        score_fn,
        preds,
        gold,
        {ex: _frac_key(cut_of[ex]) for ex in gold if ex in cut_of},
    )


def per_group(score_fn, preds: dict, gold: dict, group_of: dict[str, str]) -> dict:
    """Run ``score_fn`` overall + split by an arbitrary example-id → group label map."""
    out = {"overall": score_fn(preds, gold)}
    groups = sorted(set(group_of.get(ex, "?") for ex in gold))
    for grp in groups:
        g = {ex: v for ex, v in gold.items() if group_of.get(ex) == grp}
        p = {ex: preds[ex] for ex in g if ex in preds}
        out[grp] = score_fn(p, g)
    return out


# --------------------------------------------------------------------------- reporting

_TASK_METRIC_KEYS = {
    "nextstep": ("top1", "top3", "top5", "mrr"),
    "completion": ("exact_match", "norm_edit_dist", "token_acc", "block_acc"),
    "anomaly": (
        "binary_acc",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "rule_attribution_acc",
    ),
}

_FLAT_NEXT = {"top1": "top1", "top3": "top3", "top5": "top5", "mrr": "mrr"}
_FLAT_COMPL = {
    "exact_match": "em",
    "norm_edit_dist": "ned",
    "token_acc": "token_acc",
    "block_acc": "block_acc",
}
_FLAT_ANOM = {
    "binary_acc": "anomaly_acc",
    "precision": "anomaly_p",
    "recall": "anomaly_r",
    "f1": "anomaly_f1",
    "roc_auc": "anomaly_auc",
    "rule_attribution_acc": "rule_attr_acc",
}


def flatten_breakdown(
    per_result: dict,
    name_map: dict[str, str],
    *,
    include_confusion: bool = False,
) -> dict[str, float]:
    """``per_family`` / ``per_cut_fraction`` output → flat registry scalars."""
    out: dict[str, float] = {}
    for grp, scores in per_result.items():
        suffix = "" if grp == "overall" else f"_{grp}"
        for raw, flat in name_map.items():
            v = scores.get(raw)
            if isinstance(v, (int, float)):
                out[f"{flat}{suffix}"] = round(float(v), 4)
        if include_confusion and grp == "overall":
            conf = scores.get("confusion")
            if conf:
                out.update({f"cm_{k}": conf[k] for k in ("tp", "fp", "tn", "fn")})
    return out


def build_metrics_report(
    *,
    version: str,
    predictor: str,
    model_ref: str | None = None,
    eval_set: str | None = None,
    tags: list[str] | None = None,
    nextstep: dict | None = None,
    completion: dict | None = None,
    anomaly: dict | None = None,
) -> dict:
    """Structured report for ``metrics_report.json`` / REPORT.md (all documented metrics)."""
    return {
        "version": version,
        "predictor": predictor,
        "model_ref": model_ref,
        "eval_set": eval_set,
        "tags": tags or [],
        "tasks": {
            k: v
            for k, v in (
                ("nextstep", nextstep),
                ("completion", completion),
                ("anomaly", anomaly),
            )
            if v
        },
        "metric_keys": _TASK_METRIC_KEYS,
    }


def format_report_markdown(report: dict) -> str:
    """Human-readable summary for ``metrics_report.md`` and REPORT.md paste-in."""
    lines = [
        f"# Track eval — {report.get('version', '?')}",
        "",
        f"- **Predictor:** `{report.get('predictor', '?')}`",
    ]
    if report.get("model_ref"):
        lines.append(f"- **Model:** `{report['model_ref']}`")
    if report.get("eval_set"):
        lines.append(f"- **Eval set:** `{report['eval_set']}`")
    if report.get("tags"):
        lines.append(f"- **Tags:** `{', '.join(report['tags'])}`")
    lines.append("")

    def _section_lines(task_name: str, label: str, scores: dict) -> list[str]:
        keys = _TASK_METRIC_KEYS.get(task_name, ())
        parts = [
            f"{k}={round(scores[k], 4)}"
            for k in keys
            if k in scores and isinstance(scores.get(k), (int, float))
        ]
        if scores.get("confusion"):
            c = scores["confusion"]
            parts.append(f"confusion=tp{c['tp']}/fp{c['fp']}/tn{c['tn']}/fn{c['fn']}")
        pr = scores.get("per_rule_detection")
        if pr:
            top = ", ".join(f"{k}={round(v, 3)}" for k, v in list(pr.items())[:5])
            parts.append(f"per_rule=[{top}]")
        return [f"- **{label}:** " + ", ".join(parts)] if parts else []

    for task, data in (report.get("tasks") or {}).items():
        lines.append(f"## {task}")
        sections: list[tuple[str, dict]] = []
        if "by_family" in data or "by_cut" in data:
            for key in ("by_family", "by_cut"):
                block = data.get(key) or {}
                for grp, scores in sorted(block.items()):
                    prefix = f"{key}/{grp}" if grp != "overall" else f"{key}/overall"
                    sections.append((prefix, scores))
        else:
            for grp, scores in sorted(data.items()):
                sections.append((grp, scores))
        for label, scores in sections:
            lines.extend(_section_lines(task, label, scores))
        lines.append("")
    return "\n".join(lines).strip() + "\n"
