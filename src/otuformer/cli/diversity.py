"""diversity command stub with full options."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Compute diversity indices from OTU assignments.")


@app.callback(invoke_without_command=True)
def diversity(
    ctx: typer.Context,
    assignments: Path = typer.Option(
        ..., "--assignments", help="Partition assignment CSV."
    ),
    out_dir: Path = typer.Option(
        Path("runs/diversity"), "--out-dir", help="Output directory."
    ),
    prefix: str = typer.Option("OTU", "--prefix", help="Prefix for reporting."),
    min_abundance: str = typer.Option(
        "0,2,5", "--min-abundance", help="Comma-separated min abundance thresholds."
    ),
    phylo: bool = typer.Option(
        False, "--phylo/--no-phylo", help="Compute MPD from tree when provided."
    ),
    tree: Path | None = typer.Option(None, "--tree", help="Optional Newick tree path."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.delineation.diversity import diversity_table
    from otuformer.utils.io import read_csv

    min_values = [int(x.strip()) for x in min_abundance.split(",") if x.strip()]
    assignments_df = read_csv(assignments)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = diversity_table(
        assignments_df, min_values, tree_newick_path=tree if phylo else None
    )
    out_csv = out_dir / "diversity_indices.csv"
    table.reset_index().rename(columns={"index": "metric"}).to_csv(out_csv, index=False)
    typer.echo(f"Diversity table: {out_csv}")
