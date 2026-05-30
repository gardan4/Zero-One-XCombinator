"""Tests for eval final plumbing: gold export, proxy, rescore, promote."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from zo_eval import submission as sub
from zo_eval import track_metrics as M
from zo_eval.gold_export import export_all_ground_truth
from zo_eval.official_metrics import run_official_capture
from zo_eval.proxy_metrics import score_completion_grammar, score_nextstep_vocab
from zo_eval.results_io import load_predictions_from_dir, promote_results
from zo_eval.self_check import run_self_check
from zo_eval.track import rescore_results


def _mini_gold(tmp: Path):
    from zo_eval.submission import make_local_eval_set
    from zo_train.fab import read_sequences

    steps = read_sequences("MOSFET")[0][:25]
    return make_local_eval_set([("MOSFET", steps)], [], tmp, fractions=(0.6,))


def test_gold_export_and_rescore_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gold = _mini_gold(tmp)
        valid_csv = tmp / "eval_input_valid.csv"
        anomaly_csv = tmp / "eval_input_anomaly.csv"
        gt_dir = tmp / "gt"
        paths = export_all_ground_truth(gold, valid_csv, anomaly_csv, gt_dir)
        assert paths["valid"].exists()
        assert paths["anomaly_forbidden"].exists()

        results = tmp / "results"
        results.mkdir()
        sub.write_nextstep([(ex, [g]) for ex, g in gold["next"].items()], results / "nextstep.csv")
        sub.write_completion(list(gold["completion"].items()), results / "completion.csv")
        sub.write_anomaly(
            [(ex, g["is_valid"], 1.0, g.get("rule")) for ex, g in gold["anomaly"].items()],
            results / "anomaly.csv",
        )

        loaded = load_predictions_from_dir(results)
        assert loaded["nextstep"]
        assert loaded["completion"]

        res = rescore_results(results, gold, version="test-rescore", self_check=False)
        assert res.get("top1") == 1.0
        assert (results / "metrics_report.json").exists()


def test_self_check_captures_official_scores():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gold = _mini_gold(tmp)
        results = tmp / "results"
        results.mkdir()
        sub.write_nextstep([(ex, [g]) for ex, g in gold["next"].items()], results / "nextstep.csv")
        sub.write_completion(list(gold["completion"].items()), results / "completion.csv")
        sub.write_anomaly(
            [(ex, g["is_valid"], 1.0 if g["is_valid"] else 0.0, g.get("rule")) for ex, g in gold["anomaly"].items()],
            results / "anomaly.csv",
        )

        transcript, codes = run_self_check(
            results,
            gold=gold,
            valid_csv=tmp / "eval_input_valid.csv",
            anomaly_csv=tmp / "eval_input_anomaly.csv",
        )

        text = transcript.read_text(encoding="utf-8")
        assert codes == {"next-step": "ok", "completion": "ok", "anomaly": "ok"}
        assert "EVAL RESULTS" in text
        assert "Top-1 Accuracy" in text
        assert "Mean Normalized Edit Distance" in text
        assert "Binary Accuracy" in text


def test_track_metrics_match_official_scorer_on_labeled_fixture():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        valid_gt = tmp / "gt_valid.csv"
        valid_gt.write_text(
            "\n".join(
                [
                    "EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE,NEXT_STEP,FULL_SEQUENCE",
                    "valid_0001,MOSFET,0.6,A|B,C,A|B|C|D",
                    "valid_0002,IC,0.8,A,X,A|X|Y",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        forbidden_gt = tmp / "gt_anomaly_forbidden.csv"
        forbidden_gt.write_text(
            "EXAMPLE_ID,FAMILY,SEQUENCE,VIOLATION_RULE\nforbidden_0001,MOSFET,A|BAD,RULE_A\n",
            encoding="utf-8",
        )
        valid_sup = tmp / "gt_anomaly_valid_supplement.csv"
        valid_sup.write_text("EXAMPLE_ID,FAMILY,SEQUENCE\nvalid_anom_0001,MOSFET,A|B\n", encoding="utf-8")

        next_preds = {"valid_0001": ["C", "Z"], "valid_0002": ["Z", "Y", "X"]}
        completion_preds = {"valid_0001": ["C", "D"], "valid_0002": ["X", "Z"]}
        anomaly_preds = {
            "forbidden_0001": {"is_valid": 0, "score": 0.1, "rule": "RULE_A"},
            "valid_anom_0001": {"is_valid": 0, "score": 0.2, "rule": "RULE_A"},
        }

        sub.write_nextstep(list(next_preds.items()), tmp / "nextstep.csv")
        sub.write_completion(list(completion_preds.items()), tmp / "completion.csv")
        sub.write_anomaly(
            [(ex, row["is_valid"], row["score"], row["rule"]) for ex, row in anomaly_preds.items()],
            tmp / "anomaly.csv",
        )

        local_next = M.score_nextstep(next_preds, {"valid_0001": "C", "valid_0002": "X"})
        official_next = run_official_capture("next-step", valid_gt, tmp / "nextstep.csv")
        assert official_next.returncode == 0
        assert _official_value(official_next.stdout, "Top-1 Accuracy") == round(local_next["top1"], 4)
        assert _official_value(official_next.stdout, "Top-3 Accuracy") == round(local_next["top3"], 4)
        assert _official_value(official_next.stdout, "Top-5 Accuracy") == round(local_next["top5"], 4)
        assert _official_value(official_next.stdout, "MRR") == round(local_next["mrr"], 4)

        local_completion = M.score_completion(
            completion_preds,
            {"valid_0001": ["C", "D"], "valid_0002": ["X", "Y"]},
        )
        official_completion = run_official_capture("completion", valid_gt, tmp / "completion.csv")
        assert official_completion.returncode == 0
        assert _official_value(official_completion.stdout, "Mean Normalized Edit Distance") == round(
            local_completion["norm_edit_dist"], 4
        )
        assert _official_value(official_completion.stdout, "Exact Match Rate") == round(
            local_completion["exact_match"], 4
        )
        assert _official_value(official_completion.stdout, "Mean Token Accuracy") == round(
            local_completion["token_acc"], 4
        )
        assert _official_value(official_completion.stdout, "Mean Block-level Accuracy") == round(
            local_completion["block_acc"], 4
        )

        local_anomaly = M.score_anomaly(
            anomaly_preds,
            {
                "forbidden_0001": {"is_valid": 0, "rule": "RULE_A"},
                "valid_anom_0001": {"is_valid": 1, "rule": None},
            },
        )
        official_anomaly = run_official_capture(
            "anomaly",
            forbidden_gt,
            tmp / "anomaly.csv",
            valid_supplement=valid_sup,
        )
        assert official_anomaly.returncode == 0
        assert _official_value(official_anomaly.stdout, "Binary Accuracy") == round(local_anomaly["binary_acc"], 4)
        assert _official_value(official_anomaly.stdout, "Precision \\(invalid class\\)") == round(
            local_anomaly["precision"], 4
        )
        assert _official_value(official_anomaly.stdout, "Recall \\(invalid class\\)") == round(local_anomaly["recall"], 4)
        assert _official_value(official_anomaly.stdout, "F1 \\(invalid class\\)") == round(local_anomaly["f1"], 4)
        assert _official_value(official_anomaly.stdout, "ROC-AUC") == round(local_anomaly["roc_auc"], 4)
        assert _official_value(official_anomaly.stdout, "Rule Attribution Accuracy") == round(
            local_anomaly["rule_attribution_acc"], 4
        )


def test_anomaly_blank_score_matches_official_imputation():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        forbidden_gt = tmp / "gt_anomaly_forbidden.csv"
        forbidden_gt.write_text(
            "EXAMPLE_ID,FAMILY,SEQUENCE,VIOLATION_RULE\nforbidden_0001,MOSFET,A|BAD,RULE_A\n",
            encoding="utf-8",
        )
        valid_sup = tmp / "gt_anomaly_valid_supplement.csv"
        valid_sup.write_text("EXAMPLE_ID,FAMILY,SEQUENCE\nvalid_anom_0001,MOSFET,A|B\n", encoding="utf-8")
        preds = {
            "forbidden_0001": {"is_valid": 0, "score": None, "rule": "RULE_A"},
            "valid_anom_0001": {"is_valid": 1, "score": None, "rule": None},
        }
        sub.write_anomaly(
            [(ex, row["is_valid"], row["score"], row["rule"]) for ex, row in preds.items()],
            tmp / "anomaly.csv",
        )

        local = M.score_anomaly(
            preds,
            {
                "forbidden_0001": {"is_valid": 0, "rule": "RULE_A"},
                "valid_anom_0001": {"is_valid": 1, "rule": None},
            },
        )
        official = run_official_capture(
            "anomaly",
            forbidden_gt,
            tmp / "anomaly.csv",
            valid_supplement=valid_sup,
        )

        assert official.returncode == 0
        assert _official_value(official.stdout, "ROC-AUC") == round(local["roc_auc"], 4)


def test_proxy_grammar_and_promote():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gold = _mini_gold(tmp)
        valid_inputs = sub.read_valid_inputs(tmp / "eval_input_valid.csv")
        preds = dict(gold["completion"])
        report = score_completion_grammar(valid_inputs, preds)
        assert report["grammar_valid_rate"] == 1.0

        vocab_report = score_nextstep_vocab({ex: [g] for ex, g in gold["next"].items()})
        assert vocab_report["rank1_in_vocab"] == 1.0

        results = tmp / "results"
        results.mkdir()
        (results / "manifest.json").write_text(json.dumps({"version": "v1"}))
        (results / "nextstep.csv").write_text("EXAMPLE_ID,RANK_1\n")
        dest = promote_results(results, "test-slug", extras_root=tmp / "extras")
        assert (dest / "nextstep.csv").exists()
        index = json.loads((tmp / "extras" / "INDEX.json").read_text())
        assert "test-slug" in index


def _official_value(text: str, label_regex: str) -> float:
    match = re.search(rf"{label_regex}\s*:\s*([0-9.]+)", text)
    assert match, text
    return float(match.group(1))
