"""`zo-track` — CLI for the track-eval driver (predict → submission CSVs → score → tagged run).

Examples:
  zo-track make-local-eval --family MOSFET
  zo-track predict -p ngram --version ngram-v1 --valid extras/eval_local/eval_input_valid.csv \\
      --anomaly extras/eval_local/eval_input_anomaly.csv --gold extras/eval_local/gold.json \\
      --tags split:id,family:MOSFET --eval-set local
  zo-track predict -p hf --model XCombinator/sft-fab-lofo-mosfet --version sft-lofo-mosfet-v1 \\
      --model-ref XCombinator/sft-fab-lofo-mosfet --train-families IGBT,IC \\
      --tags split:ood,family:MOSFET --eval-set local
  zo-track predict -p oracle --version oracle-v1 --tasks anomaly --gold ... --anomaly ...
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer

from zo_common.env import load_dotenv

load_dotenv()

app = typer.Typer(no_args_is_help=True, help="Track eval: predict → submission CSVs → score → tagged run.")

PREDICTOR_HELP = (
    "ngram|freq|oracle|llm|hf|featherless|likelihood-ngram|classifier "
    "(baselines, finetuned, learned anomaly)"
)


def _build_predictor(
    kind: str,
    train_families: str | None,
    order: int,
    model: str,
    base_url: str | None,
):
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
    if kind == "likelihood-ngram":
        from zo_eval.anomaly_detect import LikelihoodDetector
        from zo_eval.baselines import NGramPredictor

        ng = NGramPredictor(train_families=tf, order=order)

        def _score(item):
            return ng.pooled.mean_logprob(item.sequence)

        det = LikelihoodDetector(_score, name="likelihood-ngram")
        if tf:
            # Calibrate on pooled train sequences (valid vs synthetic invalids optional).
            pass
        return det
    if kind == "classifier":
        from zo_eval.anomaly_detect import ClassifierDetector

        if model == "default":
            raise typer.BadParameter("classifier predictor needs --model (served name or HF id)")
        return ClassifierDetector(model=model, base_url=base_url)
    raise typer.BadParameter(f"unknown predictor {kind!r}; use: {PREDICTOR_HELP}")


def _parse_tags(tags: str, extra: list[str]) -> list[str]:
    out = [t.strip() for t in tags.split(",") if t.strip()]
    for t in extra:
        if t and t not in out:
            out.append(t)
    return out


@app.command()
def predict(
    predictor: str = typer.Option(..., "--predictor", "-p", help=PREDICTOR_HELP),
    version: str = typer.Option(..., "--version", "-V", help="Repro label (required), e.g. ngram-v1"),
    valid: str = typer.Option(None, help="eval_input_valid.csv (Tasks 1 & 2)"),
    anomaly: str = typer.Option(None, help="eval_input_anomaly.csv (Task 3)"),
    gold: str = typer.Option(None, help="gold.json — enables scoring"),
    tasks: str = typer.Option("nextstep,completion,anomaly", help="subset to run"),
    out: str = typer.Option(None, help="output dir (default experiments/<run>/results)"),
    tags: str = typer.Option("", help="extra tags: split:id|ood,family:MOSFET,..."),
    model: str = typer.Option("default", help="HF repo / served model name"),
    model_ref: str = typer.Option(None, help="HF repo id for tags (default: --model if looks like org/name)"),
    base_url: str = typer.Option(None, help="OpenAI-compatible base url (llm/classifier)"),
    order: int = typer.Option(3, help="n-gram order"),
    train_families: str = typer.Option(None, help="restrict baseline to these families (OOD)"),
    eval_set: str = typer.Option("local", help="eval-set tag: local|kickoff"),
    run_id: str = typer.Option(None, "--run-id", help="Attach to an existing registry run"),
):
    from zo_eval.track import run_track

    mref = model_ref or (model if "/" in model and not model.startswith("default") else None)
    pred = _build_predictor(predictor, train_families, order, model, base_url)
    res = run_track(
        pred,
        valid_csv=valid,
        anomaly_csv=anomaly,
        gold=gold,
        tasks=tuple(t.strip() for t in tasks.split(",") if t.strip()),
        out_dir=out,
        run_id=run_id,
        tags=_parse_tags(tags, []),
        version=version,
        model_ref=mref,
        eval_set=eval_set,
    )
    typer.echo(f"run {res['run_id']} → {res['out_dir']}")
    report_md = Path(res["out_dir"]) / "metrics_report.md"
    if report_md.exists():
        typer.echo(f"report: {report_md}")
    for k in sorted(res):
        if k not in ("run_id", "out_dir", "version"):
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


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Registry run id to rebuild metrics_report from meta"),
):
    """Re-emit metrics_report.md from the run's last metrics (no re-inference)."""
    from zo_common.registry import get_run, run_dir

    meta = get_run(run_id)
    if meta is None:
        raise typer.BadParameter(f"unknown run {run_id}")
    out = run_dir(run_id) / "results"
    json_path = out / "metrics_report.json"
    if not json_path.exists():
        typer.secho(f"No {json_path}; run predict with --gold first.", fg="red")
        raise typer.Exit(1)
    from zo_eval import track_metrics as M

    data = json.loads(json_path.read_text())
    (out / "metrics_report.md").write_text(M.format_report_markdown(data))
    typer.echo(f"updated {out / 'metrics_report.md'}")


if __name__ == "__main__":  # pragma: no cover
    app()
