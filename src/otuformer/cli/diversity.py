"""diversity command: compute diversity indices from OTU assignments or OTU tables."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import click
import typer

app = typer.Typer(
    help=(
        "Compute diversity indices from OTU assignments or OTU tables.\n\n"
        "Input modes (one required, mutually exclusive):\n\n"
        "  --assignments: CSV with id/image, cluster, optional sample columns.\n"
        "  --otu-table-csv: CSV where column 1 is sample, columns 2..N are OTU IDs "
        "with abundances.\n\n"
        "Outputs:\n\n"
        "  diversity_indices.csv - global (all-samples) diversity table.\n"
        "  per-sample/ - one file per sample when sample labels are valid.\n\n"
        "Metrics:\n\n"
        "  Richness: number of unique OTUs.\n"
        "  Chao1: estimated richness (accounts for rare OTUs).\n"
        "  ACE: abundance-based coverage estimator.\n"
        "  Shannon: entropy-based diversity (higher = more diverse).\n"
        "  Simpson: probability two individuals differ (higher = more diverse).\n"
        "  Hill_q0/q1/q2: Hill numbers (richness/evenness/diversity at orders 0,1,2).\n"
        "  Pielou_J: evenness (Shannon / log(richness)).\n"
        "  MPD: mean pairwise phylogenetic distance across OTUs "
        "(--phylo + --tree required).\n\n"
        "Note: --no-phylo has been removed; omit --phylo to disable MPD."
    )
)


def validate_input_sources(
    assignments: Path | None, otu_table_csv: Path | None
) -> None:
    if (assignments is None) == (otu_table_csv is None):
        raise ValueError("Provide exactly one of --assignments or --otu-table-csv")


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
    assignments: Path | None = typer.Option(
        None,
        "--assignments",
        help="Partition assignment CSV (id/image, cluster, optional sample).",
    ),
    otu_table_csv: Path | None = typer.Option(
        None, "--otu-table-csv", help="OTU table CSV (sample column + OTU ID headers)."
    ),
    otu_table_has_header: bool = typer.Option(
        False,
        "--otu-table-has-header",
        help="Force first row of OTU table as header (required if OTU IDs are numeric).",
    ),
    out_dir: Path = typer.Option(
        Path("runs/diversity"), "--out-dir", help="Output directory."
    ),
    min_abundance: str = typer.Option(
        "0,2,5", "--min-abundance", help="Comma-separated min abundance thresholds."
    ),
    phylo: bool = typer.Option(
        False, "--phylo", help="Compute MPD from tree when provided."
    ),
    tree: Path | None = typer.Option(None, "--tree", help="Optional Newick tree path."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.delineation.diversity import (
        build_diversity_tables_from_otu_table,
        build_per_sample_paths,
        diversity_table,
        has_valid_samples,
        normalize_assignments,
        parse_otu_table,
        split_assignments_by_sample,
    )
    from otuformer.utils.io import read_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    from otuformer.utils.logging import TeeLogger

    tee = TeeLogger(out_dir / "logs" / "diversity.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        validate_input_sources(assignments, otu_table_csv)

        params = {
            "assignments": str(assignments) if assignments else "",
            "otu_table_csv": str(otu_table_csv) if otu_table_csv else "",
            "otu_table_has_header": otu_table_has_header,
            "out_dir": str(out_dir),
            "min_abundance": min_abundance,
            "phylo": phylo,
            "tree": str(tree) if tree is not None else "",
        }
        print(f"Command: {_format_user_command(ctx, params)}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        min_values = [int(x.strip()) for x in min_abundance.split(",") if x.strip()]
        tree_path = tree if phylo else None

        if phylo and tree is None:
            typer.echo("Warning: --phylo set but no --tree provided; MPD skipped.")
        if tree is not None and not phylo:
            typer.echo("Warning: --tree provided without --phylo; tree ignored.")

        if assignments is not None:
            assignments_df = normalize_assignments(
                read_csv(assignments, encoding="utf-8-sig")
            )
            table = diversity_table(
                assignments_df, min_values, tree_newick_path=tree_path
            )
            out_csv = out_dir / "diversity_indices.csv"
            table.reset_index().rename(columns={"index": "metric"}).to_csv(
                out_csv, index=False
            )
            typer.echo(f"Diversity table: {out_csv}")

            if has_valid_samples(assignments_df):
                per_sample_dir = out_dir / "per-sample"
                per_sample_dir.mkdir(parents=True, exist_ok=True)
                parts = split_assignments_by_sample(assignments_df)
                paths = build_per_sample_paths(list(parts.keys()), per_sample_dir)
                for sample, subset in parts.items():
                    sub_table = diversity_table(
                        subset, min_values, tree_newick_path=tree_path
                    )
                    sub_table.reset_index().rename(columns={"index": "metric"}).to_csv(
                        paths[sample], index=False
                    )
                typer.echo(f"Per-sample outputs: {per_sample_dir}")
            else:
                typer.echo(
                    "Warning: sample column missing or empty; per-sample outputs skipped."
                )
        else:
            otu_raw = read_csv(otu_table_csv, encoding="utf-8-sig", header=None)
            otu_table = parse_otu_table(otu_raw, has_header=otu_table_has_header)
            global_table, per_sample = build_diversity_tables_from_otu_table(
                otu_table, min_values, tree_path
            )
            out_csv = out_dir / "diversity_indices.csv"
            global_table.reset_index().rename(columns={"index": "metric"}).to_csv(
                out_csv, index=False
            )
            typer.echo(f"Diversity table: {out_csv}")

            per_sample_dir = out_dir / "per-sample"
            per_sample_dir.mkdir(parents=True, exist_ok=True)
            paths = build_per_sample_paths(list(per_sample.keys()), per_sample_dir)
            for sample, sub_table in per_sample.items():
                sub_table.reset_index().rename(columns={"index": "metric"}).to_csv(
                    paths[sample], index=False
                )
            typer.echo(f"Per-sample outputs: {per_sample_dir}")
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
