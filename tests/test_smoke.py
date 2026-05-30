"""Light smoke tests — no torch / GPU stack required.

Exercise the shared run-registry contract and config round-trip so a broken
laptop install is caught before anyone burns cluster time. Run: `just test`.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ZO_EXPERIMENTS_DIR", str(tmp_path))
    from zo_common import registry

    run = registry.new_run(name="smoke test", kind="sft")
    # id format: <YYYYMMDD_HHMMSS>_<kind>_<slug>_<rand6> (trailing random suffix avoids
    # same-second collisions on a shared store), so assert the kind+slug is present.
    assert "_sft_smoke-test" in run.id

    registry.append_metric(run.id, step=0, loss=1.0)
    registry.append_metric(run.id, step=1, loss=0.4)

    rows = registry.read_metrics(run.id)
    assert len(rows) == 2
    assert rows[-1]["step"] == 1
    assert rows[-1]["loss"] == 0.4

    got = registry.get_run(run.id)
    assert got is not None
    assert got.metrics.get("loss") == 0.4  # latest summary kept on meta.json

    assert any(r.id == run.id for r in registry.list_runs())


def test_config_roundtrip(tmp_path):
    from zo_common.config import ExperimentConfig

    cfg = ExperimentConfig(name="t", kind="grpo", extra={"num_generations": 4})
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)

    loaded = ExperimentConfig.from_yaml(path)
    assert loaded.name == "t"
    assert loaded.kind == "grpo"
    assert loaded.extra["num_generations"] == 4


def test_example_configs_parse():
    from zo_common.config import ExperimentConfig

    configs = sorted((REPO_ROOT / "packages" / "training" / "configs").glob("*.yaml"))
    assert configs, "expected example training configs to exist"
    for path in configs:
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.name


# ----------------------------------------------------------------------- Stream 4: anomaly detect


def test_likelihood_detector_injected_scorer():
    """LikelihoodDetector turns an injected (fake) score fn into P(valid) + a verdict, GPU-free.

    Asserts the load-bearing contracts: score is always a float in [0,1] (never None — the ROC-AUC
    scorer drops the metric otherwise), monotone in the raw score, and calibration separates a
    valid/invalid class so the verdict flips at the right place.
    """
    from zo_eval.anomaly_detect import LikelihoodDetector
    from zo_eval.submission import AnomalyInput

    # Fake "likelihood": higher = more valid-looking. Two valid, two invalid, cleanly separable.
    raw = {"v1": 5.0, "v2": 4.0, "b1": -3.0, "b2": -4.0}
    labels = {"v1": 1, "v2": 1, "b1": 0, "b2": 0}
    items = [AnomalyInput(ex, "MOSFET", ["RECEIVE WAFER LOT", "SHIP LOT"]) for ex in raw]
    scorer = lambda it: raw[it.example_id]  # noqa: E731 — terse fake for the test

    # Uncalibrated: sigmoid → every row a finite probability in (0,1), never None, and monotone.
    det = LikelihoodDetector(scorer)
    scores = {}
    for it in items:
        iv, sc, rule = det.anomaly(it)
        assert sc is not None and 0.0 <= sc <= 1.0, (it.example_id, sc)
        assert rule is None  # a likelihood detector has no rule signal
        scores[it.example_id] = sc
    assert scores["v1"] > scores["v2"] > scores["b1"] > scores["b2"]  # monotone in raw score

    # Calibrated on the labels: verdicts now match the ground-truth class for every item.
    det.calibrate(items, [labels[it.example_id] for it in items])
    for it in items:
        iv, sc, _ = det.anomaly(it)
        assert iv == labels[it.example_id], (it.example_id, iv, sc)
        assert 0.0 <= sc <= 1.0


def test_classifier_detector_injected_chat():
    """ClassifierDetector parses VALID/INVALID + rule from a fake chat and derives a [0,1] score."""
    from zo_eval.anomaly_detect import ClassifierDetector
    from zo_eval.submission import AnomalyInput

    def fake_chat(messages, **kw):
        # Pretend the model flags any sequence that does not start with RECEIVE WAFER LOT.
        seq_line = messages[-1]["content"].split("Process sequence:")[1].strip()
        invalid = not seq_line.startswith("RECEIVE WAFER LOT")
        verdict = "INVALID. RULE_SHIP_BEFORE_TEST" if invalid else "VALID."
        first = "INVALID" if invalid else "VALID"
        logprobs = {
            "content": [
                {
                    "token": first,
                    "logprob": -0.3,
                    "top_logprobs": [
                        {"token": "VALID", "logprob": -3.0 if invalid else -0.1},
                        {"token": "INVALID", "logprob": -0.1 if invalid else -3.0},
                    ],
                }
            ]
        }
        return {"choices": [{"message": {"content": verdict}, "logprobs": logprobs}]}

    clf = ClassifierDetector(chat_fn=fake_chat)
    iv_ok, sc_ok, rule_ok = clf.anomaly(AnomalyInput("g", "MOSFET", ["RECEIVE WAFER LOT", "WAFER SORT TEST", "SHIP LOT"]))
    assert iv_ok == 1 and rule_ok is None and 0.5 < sc_ok <= 1.0, (iv_ok, sc_ok, rule_ok)
    iv_bad, sc_bad, rule_bad = clf.anomaly(AnomalyInput("b", "MOSFET", ["SHIP LOT", "RECEIVE WAFER LOT"]))
    assert iv_bad == 0 and rule_bad == "RULE_SHIP_BEFORE_TEST" and 0.0 <= sc_bad < 0.5, (iv_bad, sc_bad, rule_bad)


def test_likelihood_detector_drops_into_run_track(tmp_path, monkeypatch):
    """The detector + the n-gram's mean-logprob as a score fn flow through run_track and produce
    a scored anomaly run with a float AUC — the actual Stream-4 headline path (GPU-free)."""
    import random

    monkeypatch.setenv("ZO_EXPERIMENTS_DIR", str(tmp_path))
    from zo_eval.anomaly_detect import LikelihoodDetector
    from zo_eval.baselines import NGramPredictor
    from zo_eval.submission import make_local_eval_set
    from zo_eval.track import run_track
    from zo_train.datagen import make_negative, make_splits
    from zo_train.fab import read_sequences

    rng = random.Random(0)
    sp = make_splits()
    seqs = read_sequences("MOSFET")
    test = [seqs[i] for i in sp["per_family"]["MOSFET"]["test"][:12]]
    negs = []
    for s in test[:6]:
        n = make_negative(s, rng)
        if n:
            n["family"] = "MOSFET"
            negs.append(n)
    gold = make_local_eval_set([("MOSFET", s) for s in test], negs, tmp_path / "ev")

    ng = NGramPredictor()  # in-distribution (all families) → near-perfect AUC
    det = LikelihoodDetector(lambda it: ng.pooled.mean_logprob(it.sequence))
    res = run_track(
        det,
        anomaly_csv=str(tmp_path / "ev" / "eval_input_anomaly.csv"),
        gold=dict(gold),
        tasks=("anomaly",),
        tags=["split:id", "family:MOSFET", "eval:anomaly", "stream:4"],
    )
    assert isinstance(res["anomaly_auc"], (int, float))  # a real number, never None
    assert res["anomaly_auc"] > 0.9  # ID likelihood ranks valids above invalids
