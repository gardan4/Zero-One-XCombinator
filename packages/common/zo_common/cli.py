from __future__ import annotations

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

from zo_common.registry import get_run, list_runs, read_metrics

app = typer.Typer(help="Inspect Zero One experiment runs.", no_args_is_help=True)
console = Console()


def _runs_matching_tag(runs, tag: str | None):
    if not tag:
        return runs
    needle = tag.strip().lower()
    out = []
    for r in runs:
        tags = [t.lower() for t in (r.tags or [])]
        if any(needle in t for t in tags):
            out.append(r)
    return out


@app.command("ls")
def ls(
    tag: str = typer.Option(None, "--tag", "-t", help="Filter: tag substring match (any tag)"),
    kind: str = typer.Option(None, "--kind", "-k", help="Filter by run kind (train, eval, …)"),
) -> None:
    """List runs, newest first. Tags are free-form strings stored on each run."""
    runs = list_runs()
    if kind:
        runs = [r for r in runs if r.kind == kind]
    runs = _runs_matching_tag(runs, tag)
    if not runs:
        console.print("[dim]No matching runs.[/dim]")
        raise typer.Exit()
    table = Table(show_header=True, header_style="bold")
    for col in ("id", "kind", "status", "name", "tags", "metrics"):
        table.add_column(col, overflow="fold")
    for r in runs:
        metrics = ", ".join(f"{k}={v}" for k, v in list(r.metrics.items())[:3])
        tags = ", ".join(r.tags[:6]) if r.tags else "-"
        if r.tags and len(r.tags) > 6:
            tags += ", …"
        table.add_row(r.id, r.kind, r.status, r.name, tags, metrics or "-")
    console.print(table)


@app.command("tag")
def tag_lookup(
    query: str = typer.Argument(..., help="Substring to match in any tag"),
) -> None:
    """Find runs whose tags contain QUERY (same filter as `ls --tag`)."""
    ls(tag=query)


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


@app.command("hub-manifest")
def hub_manifest(
    run_id: str = typer.Argument(..., help="Training run id"),
    artifact_dir: str = typer.Option(None, "--artifact-dir", "-a", help="Override artifacts path"),
    hub_model_id: str = typer.Option(None, "--hub-model-id", help="HF repo id for the model card title"),
    note: str = typer.Option(None, "--note", help="Extra note on the HF model card"),
) -> None:
    """Write training_manifest.json + README.md for a run (local or pre-upload)."""
    from zo_common.hub_metadata import write_hub_artifact_metadata
    from zo_common.registry import run_dir

    out = Path(artifact_dir) if artifact_dir else (run_dir(run_id) / "artifacts")
    path = write_hub_artifact_metadata(
        out,
        run_id,
        hub_model_id=hub_model_id,
        notes=note,
    )
    console.print(f"[green]Wrote[/green] {path}")
    console.print(f"[green]Wrote[/green] {out / 'README.md'}")


if __name__ == "__main__":
    app()
