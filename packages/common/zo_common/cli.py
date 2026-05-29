from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from zo_common.registry import get_run, list_runs, read_metrics

app = typer.Typer(help="Inspect Zero One experiment runs.", no_args_is_help=True)
console = Console()


@app.command("ls")
def ls() -> None:
    """List all runs, newest first."""
    runs = list_runs()
    if not runs:
        console.print(
            "[dim]No runs yet. Start one with `just train <config>` or `zo-train ...`.[/dim]"
        )
        raise typer.Exit()
    table = Table(show_header=True, header_style="bold")
    for col in ("id", "kind", "status", "name", "branch", "metrics"):
        table.add_column(col, overflow="fold")
    for r in runs:
        metrics = ", ".join(f"{k}={v}" for k, v in list(r.metrics.items())[:3])
        table.add_row(r.id, r.kind, r.status, r.name, r.git_branch or "-", metrics or "-")
    console.print(table)


@app.command("show")
def show(run_id: str) -> None:
    """Show a run's metadata + a tail of its metrics."""
    meta = get_run(run_id)
    if meta is None:
        console.print(f"[red]No run named {run_id}[/red]")
        raise typer.Exit(1)
    console.print_json(meta.model_dump_json())
    rows = read_metrics(run_id)
    console.print(f"[dim]{len(rows)} metric rows; last: {rows[-1] if rows else '—'}[/dim]")


if __name__ == "__main__":
    app()
