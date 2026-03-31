"""diversity command stub with full options."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import click
import typer

app = typer.Typer(help="Compute diversity indices from OTU assignments.")


def _format_user_command(ctx: typer.Context, params: dict[str, object]) -> str:
    parts = ["otuformer", "diversity"]
    for key, value in params.items():
        source = ctx.get_parameter_source(key)
        if source is not click.core.ParameterSource.COMMANDLINE:
            continue
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            parts.extend([option, str(value).lower()])
            continue
        if value in (None, ""):
            continue
        parts.extend([option, str(value)])
    return " ".join(parts)


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

    out_dir.mkdir(parents=True, exist_ok=True)
    from otuformer.utils.logging import TeeLogger

    tee = TeeLogger(out_dir / "logs" / "diversity.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        params = {
            "assignments": str(assignments),
            "out_dir": str(out_dir),
            "prefix": prefix,
            "min_abundance": min_abundance,
            "phylo": phylo,
            "tree": str(tree) if tree is not None else "",
        }
        print(f"Command: {_format_user_command(ctx, params)}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        min_values = [int(x.strip()) for x in min_abundance.split(",") if x.strip()]
        assignments_df = read_csv(assignments)
        table = diversity_table(
            assignments_df, min_values, tree_newick_path=tree if phylo else None
        )
        out_csv = out_dir / "diversity_indices.csv"
        table.reset_index().rename(columns={"index": "metric"}).to_csv(
            out_csv, index=False
        )
        typer.echo(f"Diversity table: {out_csv}")
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
