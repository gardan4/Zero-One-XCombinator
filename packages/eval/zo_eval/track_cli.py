"""`zo-track` — CLI for the track-eval driver (predict → submission CSVs → score → tagged run).

Examples:
  zo-track make-local-eval --family MOSFET            # synth organizer-format inputs + gold.json
  zo-track predict -p ngram  --valid extras/eval_local/eval_input_valid.csv \\
                   --anomaly extras/eval_local/eval_input_anomaly.csv \\
                   --gold extras/eval_local/gold.json --tags split:id,family:MOSFET
  zo-track predict -p ngram  --train-families IGBT,IC ... --tags split:ood,family:MOSFET   # OOD
  zo-track predict -p oracle --anomaly ... --gold ... --tasks anomaly                       # oracle
"""

from __future__ import annotations

import random

import typer

from zo_common.env import load_dotenv

load_dotenv()

app = typer.Typer(no_args_is_help=True, help="Track eval: predict → submission CSVs → score → tagged run.")


def _build_predictor(kind: str, train_families: str | None, order: int, model: str, base_url: str | None):
    tf = [f.strip().upper() for f in train_families.split(",")] if train_families else None
    if kind == "ngram":
        from zo_eval.baselines import NGramPredictor

        return NGramPredictor(train_families=tf, order=order)
    if kind == "oracle":
        from zo_eval.baselines import OraclePredictor

        return OraclePredictor()
    if kind == "freq":
        from zo_eval.baselines import FreqPredictor

        return FreqPredictor(train_families=tf)
    if kind in ("llm", "hf"):
        from zo_eval.predict_llm import HFGeneratePredictor, ServedLLMPredictor

        return ServedLLMPredictor(model=model, base_url=base_url) if kind == "llm" else HFGeneratePredictor(model=model)
    if kind == "featherless":
        from zo_eval.predict_llm import FeatherlessPredictor

        return FeatherlessPredictor(model=model)
    raise typer.BadParameter(f"unknown predictor {kind!r} (ngram|oracle|freq|llm|hf|featherless)")


@app.command()
def predict(
    predictor: str = typer.Option(..., "--predictor", "-p", help="ngram|oracle|freq|llm|hf|featherless"),
    valid: str = typer.Option(None, help="eval_input_valid.csv (Tasks 1 & 2)"),
    anomaly: str = typer.Option(None, help="eval_input_anomaly.csv (Task 3)"),
    gold: str = typer.Option(None, help="gold.json — enables scoring"),
    tasks: str = typer.Option("nextstep,completion,anomaly", help="subset to run"),
    out: str = typer.Option(None, help="output dir (default experiments/<run>/results)"),
    tags: str = typer.Option("", help="comma-separated run tags, e.g. split:id,family:MOSFET"),
    model: str = typer.Option("default", help="served model name (llm/hf) or HF repo (featherless)"),
    base_url: str = typer.Option(None, help="OpenAI-compatible base url (llm)"),
    order: int = typer.Option(3, help="n-gram order"),
    train_families: str = typer.Option(None, help="restrict baseline to these families (for OOD)"),
    run_id: str = typer.Option(None, "--run-id", help="Attach to an existing registry run (cluster jobs)."),
):
    from zo_eval.track import run_track

    pred = _build_predictor(predictor, train_families, order, model, base_url)
    res = run_track(
        pred,
        valid_csv=valid,
        anomaly_csv=anomaly,
        gold=gold,
        tasks=tuple(t.strip() for t in tasks.split(",") if t.strip()),
        out_dir=out,
        run_id=run_id,
        tags=[t.strip() for t in tags.split(",") if t.strip()] + [f"predictor:{predictor}"],
    )
    typer.echo(f"run {res['run_id']} → {res['out_dir']}")
    for k in sorted(res):
        if k not in ("run_id", "out_dir"):
            typer.echo(f"  {k}: {res[k]}")


@app.command("make-local-eval")
def make_local_eval(
    family: str = typer.Option(..., help="family to use as the eval set"),
    lofo: bool = typer.Option(False, help="label only — eval set is this held-out family"),
    out: str = typer.Option("extras/eval_local", help="output dir for inputs + gold.json"),
    seed: int = typer.Option(42),
    n: int = typer.Option(0, help="cap #sequences (0 = all of the test split)"),
):
    from zo_train.datagen import make_negative, make_splits
    from zo_train.fab import read_sequences

    from zo_eval.submission import make_local_eval_set

    fam = family.upper()
    sp = make_splits(seed=seed)
    seqs = read_sequences(fam)
    test = [seqs[i] for i in sp["per_family"][fam]["test"]]
    if n:
        test = test[:n]
    valid_seqs = [(fam, s) for s in test]
    rng = random.Random(seed)
    negs = []
    for s in test:
        neg = make_negative(s, rng)
        if neg:
            neg["family"] = fam
            negs.append(neg)
    gold = make_local_eval_set(valid_seqs, negs, out)
    typer.echo(
        f"wrote eval inputs + gold.json → {out}  "
        f"({len(gold['next'])} next/compl items, {len(gold['anomaly'])} anomaly items)"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
