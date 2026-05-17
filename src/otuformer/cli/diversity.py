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
        "  Faith's PD (MPD): morphological phylogenetic diversity, computed as "
        "the sum of branch lengths of the minimal spanning subtree via "
        "scikit-bio alpha_diversity('faith_pd').\n"
        "  MPD_w: abundance-weighted rooted PD (rPD_w), branches weighted by "
        "relative abundance of descending taxa.\n"
        "  PD_richness_norm: Faith's PD divided by species richness (PD per species).\n\n"
        "Quick example:\n\n"
        "  otuformer diversity --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv\n"
        "  otuformer diversity --assignments partition_0.30_assignments.csv --phylo --tree runs/cluster/UPGMA/UPGMA_Cosine.nwk\n"
        "  otuformer diversity --otu-table-csv otu_table.csv\n\n"
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
        False,
        "--phylo",
        help=(
            "Compute phylogenetic diversity (Faith's PD). "
            "Provide --embeddings (recommended, NJ tree built from OTU centroids) "
            "or --tree (legacy UPGMA Newick)."
        ),
    ),
    embeddings: Path | None = typer.Option(
        None,
        "--embeddings",
        help=(
            "embeddings.csv from the extract step. When provided with --phylo, "
            "an OTU-centroid NJ tree is built automatically for Faith's PD. "
            "Recommended over --tree."
        ),
    ),
    tree: Path | None = typer.Option(
        None,
        "--tree",
        help="Legacy: Newick tree path for Faith's PD (used when --embeddings is absent).",
    ),
    save_nj_tree: bool = typer.Option(
        False,
        "--save-nj-tree",
        help="Save the OTU-centroid NJ Newick to <out-dir>/NJ_OTU.nwk.",
    ),
    nj_bootstrap: int = typer.Option(
        0,
        "--nj-bootstrap",
        help=(
            "Number of bootstrap replicates for the NJ tree (0 = disabled). "
            "Writes annotated consensus Newick to <out-dir>/NJ_OTU_bootstrap.nwk."
        ),
    ),
    nj_bootstrap_mode: str = typer.Option(
        "subsample",
        "--nj-bootstrap-mode",
        help="Bootstrap mode: 'subsample' (default) or 'bootstrap'.",
    ),
    nj_subsample_ratio: float = typer.Option(
        0.8,
        "--nj-subsample-ratio",
        help="Fraction of embedding dims per bootstrap replicate (subsample mode).",
    ),
    cpus: int = typer.Option(
        1,
        "--cpus",
        help="Parallel workers for NJ bootstrap.",
    ),
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
            "embeddings": str(embeddings) if embeddings is not None else "",
            "tree": str(tree) if tree is not None else "",
            "save_nj_tree": save_nj_tree,
            "nj_bootstrap": nj_bootstrap,
            "nj_bootstrap_mode": nj_bootstrap_mode,
            "nj_subsample_ratio": nj_subsample_ratio,
            "cpus": cpus,
        }
        print(f"Command: {_format_user_command(ctx, params)}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        min_values = [int(x.strip()) for x in min_abundance.split(",") if x.strip()]

        # Resolve phylo inputs
        embeddings_df = None
        tree_path = None
        if phylo:
            if embeddings is not None:
                import pandas as pd
                embeddings_df = pd.read_csv(embeddings)
            elif tree is not None:
                tree_path = tree
            else:
                typer.echo("Warning: --phylo set but neither --embeddings nor --tree provided; MPD skipped.")
        if embeddings is not None and not phylo:
            typer.echo("Warning: --embeddings provided without --phylo; embeddings ignored.")
        if tree is not None and not phylo:
            typer.echo("Warning: --tree provided without --phylo; tree ignored.")

        # NJ tree / bootstrap output paths (global only)
        nj_tree_out = (out_dir / "NJ_OTU.nwk") if (save_nj_tree and embeddings_df is not None) else None
        nj_boot_out = (out_dir / "NJ_OTU_bootstrap.nwk") if (nj_bootstrap > 0 and embeddings_df is not None) else None

        if assignments is not None:
            assignments_df = normalize_assignments(
                read_csv(assignments, encoding="utf-8-sig")
            )
            table = diversity_table(
                assignments_df, min_values,
                embeddings=embeddings_df,
                tree_newick_path=tree_path,
                nj_tree_path=nj_tree_out,
                nj_bootstrap_replicates=nj_bootstrap,
                nj_bootstrap_path=nj_boot_out,
                nj_bootstrap_support_mode=nj_bootstrap_mode,
                nj_bootstrap_subsample_ratio=nj_subsample_ratio,
                nj_jobs=cpus,
            )
            out_csv = out_dir / "diversity_indices.csv"
            table.reset_index().rename(columns={"index": "metric"}).to_csv(
                out_csv, index=False
            )
            typer.echo(f"Diversity table: {out_csv}")
            if nj_tree_out is not None and nj_tree_out.exists():
                typer.echo(f"NJ tree: {nj_tree_out}")
            if nj_boot_out is not None and nj_boot_out.exists():
                typer.echo(f"NJ bootstrap tree: {nj_boot_out}")

            if has_valid_samples(assignments_df):
                per_sample_dir = out_dir / "per-sample"
                per_sample_dir.mkdir(parents=True, exist_ok=True)
                parts = split_assignments_by_sample(assignments_df)
                paths = build_per_sample_paths(list(parts.keys()), per_sample_dir)
                for sample, subset in parts.items():
                    sub_table = diversity_table(
                        subset, min_values,
                        embeddings=embeddings_df,
                        tree_newick_path=tree_path,
                        # no tree output for per-sample tables
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
