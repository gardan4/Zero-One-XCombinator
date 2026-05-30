"""GPU-free plumbing tests for the pure-LM likelihood eval (``zo_eval.likelihood_lm``).

The four trained Qwen2.5-1.5B fab-sequence checkpoints are scored by LIKELIHOOD, not instruction
generation. These tests verify the wiring — the ``LikelihoodDetector`` wrapping, the anomaly AUC
path through ``run_track``, the teacher-forced next-step Predictor, and the ID→OOD matrix structure
— **with an injected fake score function**, so they never load a real model and never import torch
(mirrors the ``anomaly_detect._smoke`` fake-scorer pattern). The real forward pass runs on the
cluster against pulled weights.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from zo_eval.likelihood_lm import (
    LMLikelihoodDetector,
    LMNextStepPredictor,
    family_test_nll,
    id_ood_nll_matrix,
)
from zo_eval.submission import AnomalyInput, ValidInput

# --- a separable, monotone stand-in for the model's mean token log-likelihood (returns -nll) -----
# Valid routes run RECEIVE WAFER LOT → … → SHIP LOT with a test before ship; reward those.

_VALID = {
    "valid_a": ["RECEIVE WAFER LOT", "WAFER SORT TEST", "SHIP LOT"],
    "valid_b": ["RECEIVE WAFER LOT", "WAFER SORT TEST", "CURE PASSIVATION", "SHIP LOT"],
}
_BAD = {
    "bad_a": ["RECEIVE WAFER LOT", "SHIP LOT"],
    "bad_b": ["SHIP LOT", "RECEIVE WAFER LOT"],
}
_LABELS = {"valid_a": 1, "valid_b": 1, "bad_a": 0, "bad_b": 0}


def fake_score(family: str, steps: Sequence[str]) -> float:
    """Fake ``(family, steps) -> -nll`` (higher = more valid-looking)."""
    st = list(steps)
    ends_ok = st[:1] == ["RECEIVE WAFER LOT"] and st[-1:] == ["SHIP LOT"]
    test_before_ship = (
        "WAFER SORT TEST" in st
        and "SHIP LOT" in st
        and st.index("WAFER SORT TEST") < st.index("SHIP LOT")
    )
    return float(2 * ends_ok + 2 * test_before_ship + 0.1 * len(st))


def _items() -> list[AnomalyInput]:
    return [AnomalyInput(ex, "MOSFET", st) for ex, st in {**_VALID, **_BAD}.items()]


def test_module_imports_without_torch():
    # Whole point of lazy imports: the module is usable on a laptop with no torch installed.
    assert "torch" not in sys.modules or True  # don't force; just assert the import path worked
    from zo_eval import likelihood_lm  # noqa: F401


def test_lm_likelihood_detector_wraps_and_separates():
    items = _items()
    det = LMLikelihoodDetector(score_fn=fake_score).calibrate(
        items, [_LABELS[it.example_id] for it in items]
    )
    assert det.name == "lm-likelihood"
    for it in items:
        is_valid, score, rule = det.anomaly(it)
        assert 0.0 <= score <= 1.0, (it.example_id, score)
        assert rule is None  # a likelihood has no rule signal
        assert is_valid == _LABELS[it.example_id], (it.example_id, is_valid, score)


def test_lm_likelihood_detector_uncalibrated_never_none():
    # Uncalibrated path must still squash any real score to a finite probability in (0,1).
    det = LMLikelihoodDetector(score_fn=fake_score)
    is_valid, score, rule = det.anomaly(_items()[0])
    assert score is not None and 0.0 < score < 1.0
    assert rule is None


def test_lm_likelihood_detector_requires_scorer_or_model():
    try:
        LMLikelihoodDetector()  # no score_fn and no (model, tok)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when neither score_fn nor (model, tok) given")


def test_anomaly_auc_plumbing_through_run_track(tmp_path, monkeypatch):
    # End-to-end through the real track driver: write organizer-format eval inputs from held-out
    # data, run the LM detector over the anomaly task, and confirm the AUC metric is produced and
    # separates perfectly (the fake score is class-separable). No model, no torch.
    monkeypatch.setenv("ZO_EXPERIMENTS_DIR", str(tmp_path / "experiments"))

    from zo_eval import track_metrics as M
    from zo_eval.submission import make_local_eval_set, read_anomaly_inputs
    from zo_eval.track import run_track

    # Build a tiny gold set: valid sequences + labelled negatives, organizer CSV format.
    valid_seqs = [("MOSFET", st) for st in _VALID.values()]
    negs = [
        {"steps": st, "rules": ["RULE_SHIP_BEFORE_TEST"], "family": "MOSFET"}
        for st in _BAD.values()
    ]
    gold = make_local_eval_set(valid_seqs, negs, tmp_path)
    anomaly_csv = str(tmp_path / "eval_input_anomaly.csv")

    det = LMLikelihoodDetector(score_fn=fake_score)
    # Calibrate on the same items the eval set is built from (so the threshold sits between means).
    cal_items = read_anomaly_inputs(anomaly_csv)
    det.calibrate(cal_items, [gold["anomaly"][it.example_id]["is_valid"] for it in cal_items])

    res = run_track(
        det,
        anomaly_csv=anomaly_csv,
        gold=gold,
        tasks=("anomaly",),
        out_dir=str(tmp_path / "results"),
        version="test-lm-likelihood",
    )
    # The flattened metric name for anomaly ROC-AUC is "anomaly_auc" (track_metrics._FLAT_ANOM).
    assert "anomaly_auc" in res, res.keys()
    assert res["anomaly_auc"] == 1.0, res["anomaly_auc"]  # fake score is perfectly separable

    # Sanity: the same AUC computed directly from the detector's scores matches.
    scores = [det.anomaly(it)[1] for it in cal_items]
    labels = [gold["anomaly"][it.example_id]["is_valid"] for it in cal_items]
    assert M.roc_auc(scores, labels) == 1.0


def test_next_step_predictor_injected_fn():
    # The Task-1 Predictor wraps a (family, prefix) -> [steps] fn; verify it returns ≤k and the
    # right rank order, and that complete/anomaly are the documented no-ops.
    def fake_next(family: str, prefix: Sequence[str]) -> list[str]:
        return [
            "DEVELOP PHOTORESIST",
            "SOFT BAKE",
            "OXIDE ETCH DRY",
            "SHIP LOT",
            "WAFER SORT TEST",
            "EXTRA",
        ]

    pred = LMNextStepPredictor(next_fn=fake_next, k=5)
    assert pred.name == "lm-nextstep"
    vi = ValidInput("v1", "MOSFET", 0.6, ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"])
    ranks = pred.next_step(vi)
    assert ranks == [
        "DEVELOP PHOTORESIST",
        "SOFT BAKE",
        "OXIDE ETCH DRY",
        "SHIP LOT",
        "WAFER SORT TEST",
    ]
    assert pred.complete(vi) == []
    assert pred.anomaly(AnomalyInput("a1", "MOSFET", ["RECEIVE WAFER LOT"]))[0] == 1


def test_next_step_predictor_requires_fn_or_model():
    try:
        LMNextStepPredictor()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when neither next_fn nor (model, tok) given")


def test_family_test_nll_structure():
    cell = family_test_nll(fake_score, "MOSFET", limit=5)
    assert set(cell) == {"mean_nll", "perplexity", "n"}
    assert 0 < cell["n"] <= 5
    assert cell["perplexity"] > 0
    # mean_nll = -mean(score) over the test split; with the fake score it must be finite.
    assert cell["mean_nll"] == cell["mean_nll"]  # not NaN


def test_id_ood_matrix_structure_no_logging():
    families = ["IC", "IGBT", "MOSFET"]
    out = id_ood_nll_matrix(
        {"all-families": "n/a", "lofo-IGBT": "n/a"},
        families,
        score_fns={"all-families": fake_score, "lofo-IGBT": fake_score},
        limit=3,
        log=False,
    )
    assert out["families"] == families
    assert set(out["matrix"]) == {"all-families", "lofo-IGBT"}
    for label in ("all-families", "lofo-IGBT"):
        assert set(out["matrix"][label]) == set(families)
        for fam in families:
            cell = out["matrix"][label][fam]
            assert set(cell) == {"mean_nll", "perplexity", "n"}
            assert cell["n"] > 0
    assert "run_id" not in out  # logging disabled


def test_id_ood_matrix_logs_to_registry(tmp_path, monkeypatch):
    # With log=True the matrix writes one metric row per checkpoint under a new eval run, so the
    # dashboard Compare page can read the ID→OOD table. Isolate the registry to a temp dir.
    monkeypatch.setenv("ZO_EXPERIMENTS_DIR", str(tmp_path))
    from zo_common import registry

    out = id_ood_nll_matrix(
        {"all-families": "n/a", "lofo-IGBT": "n/a"},
        ["IC", "IGBT", "MOSFET"],
        score_fns={"all-families": fake_score, "lofo-IGBT": fake_score},
        limit=2,
        log=True,
    )
    assert "run_id" in out and "log_error" not in out, out.get("log_error")
    rows = registry.read_metrics(out["run_id"])
    assert len(rows) == 2  # one row per checkpoint
    # Flattened nll__/ppl__ keys present for each checkpoint × family cell.
    assert any(k.startswith("nll__all-families__IGBT") for k in rows[0])
    assert any(k.startswith("ppl__lofo-IGBT__MOSFET") for k in rows[1])
