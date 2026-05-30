"""Tests for eval final plumbing: gold export, proxy, rescore, promote."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from zo_eval import submission as sub
from zo_eval.gold_export import export_all_ground_truth
from zo_eval.proxy_metrics import score_completion_grammar, score_nextstep_vocab
from zo_eval.results_io import load_predictions_from_dir, promote_results
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
