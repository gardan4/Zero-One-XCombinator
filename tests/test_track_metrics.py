"""Track metrics + reporting (Industrial AI self-eval; organizers score our CSVs separately)."""

from zo_eval import track_metrics as M


def test_per_cut_fraction_breakdown():
    gold = {"a": "X", "b": "Y", "c": "Z"}
    preds = {"a": ["X"], "b": ["Y"], "c": ["W"]}
    cut_of = {"a": 0.6, "b": 0.6, "c": 0.8}

    def score(p, g):
        return {"top1": sum(1 for ex, t in g.items() if p.get(ex, [None])[0] == t) / len(g)}

    out = M.per_cut_fraction(score, preds, gold, cut_of)
    assert out["overall"]["top1"] == 2 / 3
    assert "frac60" in out and "frac80" in out
    assert out["frac60"]["top1"] == 1.0
    assert out["frac80"]["top1"] == 0.0


def test_flatten_breakdown_per_family_and_cut():
    per = {"overall": {"top1": 0.5}, "MOSFET": {"top1": 0.7}, "frac60": {"top1": 0.9}}
    flat = M.flatten_breakdown(per, M._FLAT_NEXT)
    assert flat["top1"] == 0.5
    assert flat["top1_MOSFET"] == 0.7
    assert flat["top1_frac60"] == 0.9


def test_build_and_format_report():
    report = M.build_metrics_report(
        version="test-v1",
        predictor="ngram",
        model_ref=None,
        eval_set="local",
        tags=["version:test-v1", "split:id"],
        nextstep={
            "by_family": {"overall": {"top1": 0.8, "top3": 0.9, "top5": 1.0, "mrr": 0.85}},
            "by_cut": {"frac60": {"top1": 0.7, "top3": 0.8, "top5": 0.9, "mrr": 0.75}},
        },
    )
    md = M.format_report_markdown(report)
    assert "test-v1" in md
    assert "top1=0.8" in md
    assert "by_cut/frac60" in md


def test_gold_includes_cut_fraction_of():
    import json
    import tempfile
    from pathlib import Path

    from zo_eval.submission import make_local_eval_set

    steps = ["RECEIVE WAFER LOT", "A", "B", "C", "D", "SHIP LOT"]
    with tempfile.TemporaryDirectory() as tmp:
        gold = make_local_eval_set([("MOSFET", steps)], [], tmp, fractions=(0.5,))
        assert "cut_fraction_of" in gold
        assert len(gold["cut_fraction_of"]) >= 1
        loaded = json.loads((Path(tmp) / "gold.json").read_text())
        assert loaded["cut_fraction_of"]
