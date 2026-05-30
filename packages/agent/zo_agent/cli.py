from __future__ import annotations

import typer

from zo_agent.harness import ScenarioSuite, run_suite

app = typer.Typer(help="Run agent scenario suites and score task success.", no_args_is_help=True)


@app.command()
def run(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Path to a scenario-suite YAML."),
    model: str = typer.Option("default", "--model", "-m"),
    base_url: str = typer.Option(None, "--base-url", help="Overrides ZO_MODEL_BASE_URL."),
    max_steps: int = typer.Option(6, "--max-steps"),
    scripted: bool = typer.Option(
        False,
        "--scripted",
        help="Use the deterministic validate→explain→repair→re-validate loop "
        "(the reliable demo path; doesn't rely on the model emitting tool_calls).",
    ),
) -> None:
    suite = ScenarioSuite.from_yaml(scenario)
    res = run_suite(suite, model, base_url=base_url, max_steps=max_steps, scripted=scripted)
    typer.secho(
        f"{suite.name}: success_rate={res['success_rate']} n={res['n']} "
        f"avg_steps={res['avg_steps']} avg_tool_calls={res['avg_tool_calls']}  (run {res['run_id']})",
        fg="green",
    )


if __name__ == "__main__":
    app()
