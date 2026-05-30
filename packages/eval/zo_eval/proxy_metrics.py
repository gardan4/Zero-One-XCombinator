"""Development proxies when kickoff labels are unavailable (organizers hold GT).

These do **not** replace organizer scoring; they sanity-check predictions before submit.
"""

from __future__ import annotations

from zo_train.grammar import validate_sequence

from zo_eval import submission as sub
from zo_eval.predict import vocab


def score_completion_grammar(
    valid_inputs: list[sub.ValidInput],
    completion_preds: dict[str, list[str]],
) -> dict:
    """Fraction of (partial + predicted) sequences that pass ``validate_sequence``."""
    matched = 0
    valid_n = 0
    rule_hits: dict[str, int] = {}
    for it in valid_inputs:
        pred = completion_preds.get(it.example_id)
        if pred is None:
            continue
        matched += 1
        full = it.partial_sequence + pred
        viol = validate_sequence(full)
        if not viol:
            valid_n += 1
        else:
            for v in viol:
                rid = getattr(v, "rule", str(v))
                rule_hits[rid] = rule_hits.get(rid, 0) + 1
    rate = valid_n / matched if matched else 0.0
    return {
        "n": matched,
        "grammar_valid_rate": rate,
        "top_violation_rules": dict(sorted(rule_hits.items(), key=lambda x: -x[1])[:10]),
    }


def score_nextstep_vocab(
    nextstep_preds: dict[str, list[str]],
    *,
    known_vocab: set[str] | None = None,
) -> dict:
    """Fraction of rank-1 predictions that are known fab steps (format sanity)."""
    vocab_set = known_vocab or vocab()
    n = len(nextstep_preds)
    if not n:
        return {"n": 0, "rank1_in_vocab": 0.0, "any_rank_in_vocab": 0.0}
    rank1_ok = any_ok = 0
    for ranks in nextstep_preds.values():
        if ranks and ranks[0] in vocab_set:
            rank1_ok += 1
        if any(r in vocab_set for r in ranks):
            any_ok += 1
    return {
        "n": n,
        "rank1_in_vocab": rank1_ok / n,
        "any_rank_in_vocab": any_ok / n,
    }


def score_anomaly_oracle(
    anomaly_inputs: list[sub.AnomalyInput],
    anomaly_preds: dict[str, dict],
) -> dict:
    """Compare model anomaly verdicts to the free ``validate_sequence`` oracle."""
    tp = fp = tn = fn = 0
    for it in anomaly_inputs:
        oracle_invalid = validate_sequence(it.sequence) != []
        p = anomaly_preds.get(it.example_id, {})
        pred_invalid = p.get("is_valid", 1) == 0
        if oracle_invalid and pred_invalid:
            tp += 1
        elif (not oracle_invalid) and pred_invalid:
            fp += 1
        elif (not oracle_invalid) and (not pred_invalid):
            tn += 1
        else:
            fn += 1
    n = len(anomaly_inputs)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": n,
        "oracle_agreement_acc": acc,
        "oracle_agreement_precision": prec,
        "oracle_agreement_recall": rec,
        "confusion_vs_oracle": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def build_proxy_report(
    *,
    valid_inputs: list[sub.ValidInput] | None = None,
    anomaly_inputs: list[sub.AnomalyInput] | None = None,
    nextstep_preds: dict[str, list[str]] | None = None,
    completion_preds: dict[str, list[str]] | None = None,
    anomaly_preds: dict[str, dict] | None = None,
) -> dict:
    """Aggregate proxy metrics for kickoff / unlabeled runs."""
    out: dict = {}
    if valid_inputs and completion_preds:
        out["completion_grammar"] = score_completion_grammar(valid_inputs, completion_preds)
    if nextstep_preds:
        out["nextstep_vocab"] = score_nextstep_vocab(nextstep_preds)
    if anomaly_inputs and anomaly_preds:
        out["anomaly_vs_oracle"] = score_anomaly_oracle(anomaly_inputs, anomaly_preds)
    return out
