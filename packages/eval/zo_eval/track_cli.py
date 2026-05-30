"""`zo-track` — CLI for the track-eval driver (predict → submission CSVs → score → tagged run)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from zo_common.env import load_dotenv

from zo_eval.inference_jobs import resolve_eval_paths as _resolve_eval_paths_impl
from zo_eval.official_metrics import KICKOFF_ANOMALY, KICKOFF_VALID
from zo_eval.predictors import PredictorBuildError, build_predictor

load_dotenv()

app = typer.Typer(no_args_is_help=True, help="Track eval: predict → submission CSVs → score → tagged run.")

PREDICTOR_HELP = (
    "ngram|freq|oracle|llm|hf|likelihood-ngram|classifier "
    "(baselines, finetuned, learned anomaly)"
)


def _build_predictor(
    kind: str,
    train_families: str | None,
    order: int,
    model: str,
    base_url: str | None,
):
    try:
        return build_predictor(
            kind,
            train_families=train_families,
            order=order,
            model=model,
            base_url=base_url,
        )
    except PredictorBuildError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _parse_tags(tags: str, extra: list[str]) -> list[str]:
    out = [t.strip() for t in tags.split(",") if t.strip()]
    for t in extra:
        if t and t not in out:
            out.append(t)
    return out


def _resolve_eval_paths(eval_set: str, valid: str | None, anomaly: str | None) -> tuple[str | None, str | None]:
    return _resolve_eval_paths_impl(eval_set, valid, anomaly)


@app.command()
def predict(
    predictor: str = typer.Option(..., "--predictor", "-p", help=PREDICTOR_HELP),
    version: str = typer.Option(..., "--version", "-V", help="Repro label (required), e.g. ngram-v1"),
    valid: str = typer.Option(None, help="eval_input_valid.csv (Tasks 1 & 2)"),
    anomaly: str = typer.Option(None, help="eval_input_anomaly.csv (Task 3)"),
    gold: str = typer.Option(None, help="gold.json — enables labeled scoring"),
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
    train_run: str = typer.Option(None, "--train-run", help="Link to a training run id (adds tag + manifest snapshot)"),
    note: str = typer.Option(None, "--note", help="Free-form note on this eval run (meta.json notes field)"),
    self_check: bool = typer.Option(False, "--self-check", help="Run vendored eval_metrics.py (needs --gold)"),
    no_proxy: bool = typer.Option(False, "--no-proxy", help="Skip kickoff proxy metrics when no gold"),
    promote: str = typer.Option(None, "--promote", help="Copy results to extras/results/<slug>/"),
):
    from zo_eval.track import run_track

    valid, anomaly = _resolve_eval_paths(eval_set, valid, anomaly)
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
        run_proxy=not no_proxy,
        self_check=self_check,
        promote=promote,
        train_run_id=train_run,
        notes=note,
    )
    _echo_run_result(res)


@app.command()
def rescore(
    results: str = typer.Option(..., "--results", "-r", help="Dir with submission CSVs"),
    gold: str = typer.Option(..., help="gold.json"),
    valid: str = typer.Option(None, help="eval_input_valid.csv (for self-check export)"),
    anomaly: str = typer.Option(None, help="eval_input_anomaly.csv (for self-check export)"),
    version: str = typer.Option("rescore-v1", "-V"),
    tags: str = typer.Option("", help="extra registry tags"),
    self_check: bool = typer.Option(True, "--self-check/--no-self-check"),
    eval_set: str = typer.Option("local", help="eval-set tag"),
):
    """Re-score existing CSVs against gold (no re-inference)."""
    from zo_eval.track import rescore_results

    valid, anomaly = _resolve_eval_paths(eval_set, valid, anomaly)
    if not gold and eval_set != "kickoff":
        pass
    res = rescore_results(
        results,
        gold,
        version=version,
        tags=_parse_tags(tags, []),
        self_check=self_check,
        valid_csv=valid,
        anomaly_csv=anomaly,
        eval_set=eval_set,
    )
    _echo_run_result(res)


@app.command()
def promote(
    run_id: str = typer.Argument(None, help="Registry run id (uses experiments/<id>/results/)"),
    results: str = typer.Option(None, "--results", "-r", help="Results dir (alternative to run_id)"),
    slug: str = typer.Option(..., "--slug", "-s", help="Name under extras/results/"),
):
    """Copy a run's results to ``extras/results/<slug>/`` for submission / REPORT."""
    from zo_common.registry import run_dir

    from zo_eval.results_io import promote_results

    src = Path(results) if results else (run_dir(run_id) / "results")
    if not src.is_dir():
        typer.secho(f"Results dir not found: {src}", fg="red")
        raise typer.Exit(1)
    dest = promote_results(src, slug)
    typer.echo(f"promoted -> {dest}")
    typer.echo("index: extras/results/INDEX.json")


@app.command("export-gt")
def export_gt(
    gold: str = typer.Option(..., help="gold.json"),
    valid: str = typer.Option(..., help="eval_input_valid.csv"),
    anomaly: str = typer.Option(..., help="eval_input_anomaly.csv"),
    out: str = typer.Option("extras/results/ground_truth", help="output directory"),
):
    """Export organizer-format ground-truth CSVs from gold.json."""
    from zo_eval.gold_export import export_all_ground_truth

    paths = export_all_ground_truth(gold, valid, anomaly, out)
    for k, p in paths.items():
        typer.echo(f"  {k}: {p}")


@app.command("score-official")
def score_official(
    task: str = typer.Option(..., "--task", help="next-step|completion|anomaly"),
    ground_truth: str = typer.Option(..., "--ground-truth", help="Organizer ground-truth CSV"),
    predictions: str = typer.Option(..., "--predictions", help="Your submission CSV"),
    valid_supplement: str = typer.Option(None, "--valid-supplement"),
    save: str = typer.Option(None, "--save", help="Append transcript to this file"),
):
    """Run vendored ``eval_metrics.py`` (optionally save output)."""
    from zo_eval.official_metrics import run_official_capture

    proc = run_official_capture(task, ground_truth, predictions, valid_supplement=valid_supplement)
    text = proc.stdout
    if proc.stderr.strip():
        text += "\nSTDERR:\n" + proc.stderr
    typer.echo(text, nl=False)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        with Path(save).open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== {task} ===\n{text}\n")
        typer.echo(f"\n(saved to {save})")
    raise typer.Exit(proc.returncode)


@app.command("self-check")
def self_check_cmd(
    results: str = typer.Option(..., "--results", "-r"),
    gold: str = typer.Option(...),
    valid: str = typer.Option(...),
    anomaly: str = typer.Option(...),
):
    """Export GT + run official scorer on all 3 tasks; write ``official_scores.txt``."""
    from zo_eval.self_check import run_self_check_cli

    path = run_self_check_cli(results, gold, valid, anomaly)
    typer.echo(f"official scores -> {path}")
    typer.echo(Path(path).read_text(encoding="utf-8"))


@app.command()
def suite(
    config: str = typer.Argument(..., help="YAML suite spec (see packages/eval/eval_suites/)"),
    model: str = typer.Option(None, help="Override HF model for all hf runs"),
    model_ref: str = typer.Option(None, help="Override model-ref tag"),
    tags: str = typer.Option("", help="Extra tags for every run in the suite"),
    stop_on_error: bool = typer.Option(False, "--stop-on-error"),
):
    """Run a tagged eval matrix from YAML (baselines + finetuned comparisons)."""
    from zo_eval.suite import run_suite

    def _builder(kind, train_families, order, run_model, base_url):
        m = model or run_model
        return _build_predictor(kind, train_families, order, m, base_url)

    summary = run_suite(
        config,
        build_predictor=_builder,
        suite_tags=_parse_tags(tags, []),
        model_override=model,
        model_ref_override=model_ref,
        stop_on_error=stop_on_error,
    )
    typer.echo(f"suite summary -> {summary['summary_path']}")
    for run in summary["runs"]:
        status = run.get("status")
        name = run.get("name")
        if status == "ok":
            typer.echo(f"  OK  {name} -> {run.get('run_id')}  promoted={run.get('promoted_to', '-')}")
        else:
            typer.secho(f"  ERR {name}: {run.get('error')}", fg="red")


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
    extra_tags = f"split:ood,family:{fam}" if lofo else "split:id"
    typer.echo(
        f"wrote eval inputs + gold.json -> {out}  "
        f"({len(gold['next'])} next/compl, {len(gold['anomaly'])} anomaly)  "
        f"suggested tags: {extra_tags}"
    )


@app.command()
def validate(
    valid: str = typer.Option(str(KICKOFF_VALID), help="eval_input_valid.csv"),
    completion: str = typer.Option(..., help="completion.csv"),
    save: str = typer.Option(None, "--save", help="Write JSON report to this path"),
):
    """Grammar-check completions via ``validate_sequence`` (kickoff dev proxy)."""
    from zo_eval import submission as sub
    from zo_eval.proxy_metrics import score_completion_grammar

    valid_inputs = sub.read_valid_inputs(valid)
    compl_preds = {
        row["EXAMPLE_ID"]: sub._split(row.get("PREDICTED_SEQUENCE", ""))
        for row in sub._norm_rows(completion)
    }
    report = score_completion_grammar(valid_inputs, compl_preds)
    typer.echo(
        f"Grammar-valid completions: {report['grammar_valid_rate']:.1%} "
        f"({report['n']} examples)"
    )
    if report.get("top_violation_rules"):
        typer.echo("Top violation rules:")
        for rule, cnt in list(report["top_violation_rules"].items())[:5]:
            typer.echo(f"  {rule}: {cnt}")
    if save:
        Path(save).write_text(json.dumps(report, indent=2))
        typer.echo(f"saved -> {save}")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Registry run id to rebuild metrics_report from meta"),
):
    from zo_common.registry import get_run, run_dir

    from zo_eval import track_metrics as M

    meta = get_run(run_id)
    if meta is None:
        raise typer.BadParameter(f"unknown run {run_id}")
    out = run_dir(run_id) / "results"
    json_path = out / "metrics_report.json"
    if not json_path.exists():
        typer.secho(f"No {json_path}; run predict/rescore with --gold first.", fg="red")
        raise typer.Exit(1)
    data = json.loads(json_path.read_text())
    (out / "metrics_report.md").write_text(M.format_report_markdown(data))
    typer.echo(f"updated {out / 'metrics_report.md'}")


def _echo_run_result(res: dict) -> None:
    typer.echo(f"run {res['run_id']} -> {res['out_dir']}")
    if res.get("promoted_to"):
        typer.echo(f"promoted -> {res['promoted_to']}")
    report_md = Path(res["out_dir"]) / "metrics_report.md"
    if report_md.exists():
        typer.echo(f"report: {report_md}")
    proxy = Path(res["out_dir"]) / "proxy_report.json"
    if proxy.exists():
        typer.echo(f"proxy: {proxy}")
    official = Path(res["out_dir"]) / "official_scores.txt"
    if official.exists():
        typer.echo(f"official: {official}")
    for k in sorted(res):
        if k not in ("run_id", "out_dir", "version", "promoted_to"):
            typer.echo(f"  {k}: {res[k]}")


if __name__ == "__main__":  # pragma: no cover
    app()
