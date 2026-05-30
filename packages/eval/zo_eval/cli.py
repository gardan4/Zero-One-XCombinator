from __future__ import annotations

import typer

from zo_common.env import load_dotenv
from zo_eval.harness import run_eval
from zo_eval.tasks import TaskSpec

load_dotenv()

app = typer.Typer(help="Evaluate models against task specs.", no_args_is_help=True)


@app.command()
def run(
    task: str = typer.Option(..., "--task", "-t", help="Path to a task YAML."),
    model: str = typer.Option("default", "--model", "-m"),
    base_url: str = typer.Option(None, "--base-url", help="Overrides ZO_MODEL_BASE_URL."),
    limit: int = typer.Option(None, "--limit"),
) -> None:
    spec = TaskSpec.from_yaml(task)
    res = run_eval(spec, model, base_url=base_url, limit=limit)
    typer.secho(
        f"{spec.name}: accuracy={res['accuracy']} over n={res['n']}  (run {res['run_id']})",
        fg="green",
    )


if __name__ == "__main__":
    app()
