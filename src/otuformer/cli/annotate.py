"""annotate command stub with full options."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Apply expert corrections to cluster assignments.")


@app.callback(invoke_without_command=True)
def annotate(
    ctx: typer.Context,
    assignments: Path = typer.Option(
        ..., "--assignments", help="Partition assignment CSV."
    ),
    corrections: Path = typer.Option(..., "--corrections", help="Corrections CSV."),
    out_dir: Path = typer.Option(
        Path("runs/annotate"), "--out-dir", help="Output directory."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.delineation.annotate import (
        apply_corrections,
        build_annotation_summary,
    )
    from otuformer.utils.io import read_csv, write_csv, write_json

    assignments_df = read_csv(assignments)
    corrections_df = read_csv(corrections)
    result_df = apply_corrections(assignments_df, corrections_df)
    summary = build_annotation_summary(assignments_df, result_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = assignments.stem
    out_csv = out_dir / f"{stem}_annotated.csv"
    write_csv(result_df, out_csv)
    write_json(summary, out_dir / "annotation_summary.json")
    typer.echo(f"Corrections applied: {summary['n_corrections']}")
    typer.echo(f"Clusters affected: {summary['n_clusters_affected']}")
    typer.echo(f"Output: {out_csv}")
