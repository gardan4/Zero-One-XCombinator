"""Local stand-in scorer for the 3 Industrial AI track tasks.

Matches the **documented metric set** (generation_rules.md §5.2) so every stream can measure
locally *before* the organizers' authoritative ``eval_metrics.py`` arrives at kickoff. Pure
stdlib; operates on in-memory structures (the CSV I/O lives in ``submission.py``).

> Honesty note: the submission **format** (column names) is authoritative — get it right or score
> zero. These **numbers** are a faithful guide, but the exact definitions of Normalized Edit
> Distance (normalization), Token Accuracy (alignment), and "Block-level Accuracy" are the
> organizers' to fix; ours are sensible, documented choices to reconcile with ``eval_metrics.py``
> at kickoff. Top-k / MRR / binary-acc / P-R-F1 / ROC-AUC / rule-attribution are unambiguous.

Conventions:
- **Anomaly = positive class is INVALID** (we are detecting violations). So precision/recall/F1
  are computed with "predicted invalid" vs "truly invalid".
- **ROC-AUC** uses ``SCORE`` = P(valid) against the ``IS_VALID`` label (higher score ⇒ valid);
  this equals the invalid-class AUC under ``1-SCORE``. ``None`` if only one class is present.
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


# --------------------------------------------------------------------------- Task 2


def score_completion(
    preds: dict[str, list[str]], gold: dict[str, list[str]], block: int = 5
) -> dict:
    """``preds[ex]``/``gold[ex]`` = the predicted/true steps AFTER the cut.
    Exact-Match, Normalized Edit Distance (lower better), Token Acc, Block Acc (stand-in defs)."""
    n = len(gold)
    em = 0
    ned_sum = tok_sum = blk_sum = 0.0
    for ex, g in gold.items():
        p = preds.get(ex, [])
        if p == g:
            em += 1
        denom = max(len(p), len(g)) or 1
        ned_sum += _levenshtein(p, g) / denom
        # Token accuracy: position-aligned matches / reference length.
        tok_sum += _safe_div(sum(1 for i in range(min(len(p), len(g))) if p[i] == g[i]), len(g) or 1)
        # Block accuracy: fraction of fixed-size reference blocks reproduced exactly in place.
        nblocks = (len(g) + block - 1) // block or 1
        ok = sum(1 for s in range(0, len(g), block) if p[s : s + block] == g[s : s + block])
        blk_sum += ok / nblocks
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
        if s is not None:
            scores.append(float(s))
            labels.append(int(g["is_valid"]))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "n": len(gold),
        "binary_acc": _safe_div(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "roc_auc": roc_auc(scores, labels) if len(scores) == len(gold) and scores else None,
        "rule_attribution_acc": (_safe_div(attr_ok, attr_total) if attr_total else None),
    }


# --------------------------------------------------------------------------- breakdown


def per_family(score_fn, preds: dict, gold: dict, family_of: dict[str, str]) -> dict:
    """Run ``score_fn`` overall + split by family (the per-family breakdown the report needs)."""
    out = {"overall": score_fn(preds, gold)}
    fams = sorted(set(family_of.get(ex, "?") for ex in gold))
    for fam in fams:
        g = {ex: v for ex, v in gold.items() if family_of.get(ex) == fam}
        p = {ex: preds[ex] for ex in g if ex in preds}
        out[fam] = score_fn(p, g)
    return out
