"""annotate command stub with full options."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import click
import typer

app = typer.Typer(help="Apply expert corrections to cluster assignments.")


def _format_user_command(ctx: typer.Context, params: dict[str, object]) -> str:
    parts = ["otuformer", "annotate"]
    for key, value in params.items():
        source = ctx.get_parameter_source(key)
        if source is not click.core.ParameterSource.COMMANDLINE:
            continue
        option = f"--{key.replace('_', '-')}"
        if value in (None, ""):
            continue
        parts.extend([option, str(value)])
    return " ".join(parts)


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

    out_dir.mkdir(parents=True, exist_ok=True)
    from otuformer.utils.logging import TeeLogger

    tee = TeeLogger(out_dir / "logs" / "annotate.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        params = {
            "assignments": str(assignments),
            "corrections": str(corrections),
            "out_dir": str(out_dir),
        }
        print(f"Command: {_format_user_command(ctx, params)}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        assignments_df = read_csv(assignments)
        corrections_df = read_csv(corrections)
        result_df = apply_corrections(assignments_df, corrections_df)
        summary = build_annotation_summary(assignments_df, result_df)

        stem = assignments.stem
        out_csv = out_dir / f"{stem}_annotated.csv"
        write_csv(result_df, out_csv)
        write_json(summary, out_dir / "annotation_summary.json")
        typer.echo(f"Corrections applied: {summary['n_corrections']}")
        typer.echo(f"Clusters affected: {summary['n_clusters_affected']}")
        typer.echo(f"Output: {out_csv}")
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
