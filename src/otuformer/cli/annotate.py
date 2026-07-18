"""Annotate command: strict correction and downstream artifact refresh."""

from __future__ import annotations

import json
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

import click
import typer

app = typer.Typer(
    help=(
        "Apply expert corrections to cluster assignments.\n\n"
        "Takes raw partition assignments from the cluster command and a corrections CSV\n"
        "to produce refined OTU assignments with updated intra-class distance summaries\n"
        "and annotated UPGMA tree visualizations.\n\n"
        "Quick example:\n\n"
        "  otuformer annotate --raw-assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv --corrections corrections.csv\n"
        "  otuformer annotate --raw-assignments partition_0.30_assignments.csv --corrections corrections.csv --embeddings runs/extract/embeddings.csv --show-annotation-bar\n"
    )
)


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
    raw_assignments: Path = typer.Option(
        ...,
        "--raw-assignments",
        help=(
            "Raw partition assignment CSV from cluster output: "
            ".../UPGMA/partitions/tables/partition_<cutoff>_assignments.csv"
        ),
    ),
    corrections: Path = typer.Option(
        ...,
        "--corrections",
        help=(
            "Corrections CSV. Recommended: edit from raw assignments. "
            "Minimum required columns: id (or image), cluster."
        ),
    ),
    embeddings: Path | None = typer.Option(
        None,
        "--embeddings",
        help=(
            "Optional embeddings CSV for distance recomputation. "
            "If omitted, both pairwise_distance_summary_intra-class.csv and "
            "UPGMA_tree_partitions_annotated.pdf are skipped."
        ),
    ),
    support_display_cutoff: float = typer.Option(
        50.0,
        "--support-display-cutoff",
        help=(
            "Only display bootstrap support labels >= this value in annotated UPGMA PDF."
        ),
    ),
    figure_width: float | None = typer.Option(
        None,
        "--figure-width",
        help=(
            "Optional width (inches) for annotated UPGMA PDF. "
            "Height is still driven by tip count."
        ),
    ),
    annotate_bar_width: float = typer.Option(
        0.08,
        "--annotate-bar-width",
        help="Relative width of corrected OTU color bars in annotated UPGMA PDF.",
    ),
    show_annotation_bar: bool = typer.Option(
        False,
        "--show-annotation-bar",
        help="Show corrected OTU annotation bars in annotated UPGMA PDF.",
    ),
    show_partitioning_bars: bool = typer.Option(
        False,
        "--show-partitioning-bars",
        help="Show partitioning bars in annotated UPGMA PDF.",
    ),
    out_dir: Path = typer.Option(
        Path("runs/annotate"), "--out-dir", help="Output directory."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Clear an existing non-empty output directory."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.delineation.annotate import (
        apply_corrections,
        build_changed_only_table,
        build_annotation_summary,
        build_otu_table,
        compute_distance_matrix_for_ids,
        infer_cluster_context,
        load_partitions_from_tables,
        parse_bootstrap_support_from_newick,
        parse_cluster_params,
        render_annotated_partitions_pdf,
        resolve_distance_mode,
        summarize_intra_class_distances,
        validate_raw_assignments,
    )
    from otuformer.delineation.tree import build_upgma
    from otuformer.utils.io import prepare_output_dir, read_csv, write_csv, write_json

    prepare_output_dir(out_dir, overwrite=overwrite)
    from otuformer.utils.logging import TeeLogger

    tee = TeeLogger(out_dir / "logs" / "annotate.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        params = {
            "raw_assignments": str(raw_assignments),
            "corrections": str(corrections),
            "embeddings": str(embeddings) if embeddings is not None else None,
            "support_display_cutoff": support_display_cutoff,
            "figure_width": figure_width,
            "annotate_bar_width": annotate_bar_width,
            "show_annotation_bar": show_annotation_bar,
            "show_partitioning_bars": show_partitioning_bars,
            "out_dir": str(out_dir),
            "overwrite": overwrite,
        }
        print(f"Command: {_format_user_command(ctx, params)}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        context = infer_cluster_context(raw_assignments)
        cluster_run_dir = Path(context["cluster_run_dir"])
        cluster_log = Path(context["cluster_log"])
        cutoff_tag = str(context["cutoff_tag"])
        tables_dir = Path(context["tables_dir"])
        typer.echo(f"Cluster run: {cluster_run_dir}")
        typer.echo(f"Detected partition cutoff: {cutoff_tag}")

        assignments_df = validate_raw_assignments(read_csv(raw_assignments))
        corrections_df = read_csv(corrections)
        result_df = apply_corrections(assignments_df, corrections_df)
        summary = build_annotation_summary(assignments_df, result_df)
        changed_df = build_changed_only_table(assignments_df, result_df)

        params_from_log = parse_cluster_params(cluster_log)
        embeddings_path = embeddings.resolve() if embeddings is not None else None
        if embeddings_path is not None:
            distance_mode = resolve_distance_mode(params_from_log)
            typer.echo(f"Distance mode: {distance_mode}")
            typer.echo(f"Embeddings path: {embeddings_path}")
        else:
            typer.echo(
                "Embeddings path: <not provided>; intra-class distance summary and "
                "annotated UPGMA PDF will be skipped"
            )

        out_csv = out_dir / f"partition_{cutoff_tag}_assignments.csv"
        out_changed = out_dir / f"partition_{cutoff_tag}_assignments_changed_only.csv"
        out_otu = out_dir / "otu_table.csv"
        out_intra = out_dir / "pairwise_distance_summary_intra-class.csv"
        out_pdf = out_dir / "UPGMA_tree_partitions_annotated.pdf"

        write_csv(result_df, out_csv)
        write_csv(changed_df, out_changed)

        otu_df = build_otu_table(result_df)
        write_csv(otu_df, out_otu)

        ids = result_df["id"].astype(str).tolist()
        id_to_cluster = dict(
            zip(result_df["id"].astype(str), result_df["cluster"].astype(str))
        )
        dist = None
        if embeddings_path is not None:
            distance_matrix_path = (
                cluster_run_dir / "distance_statistics" / "distance_matrix.csv"
            )
            if distance_matrix_path.exists():
                dist_df = read_csv(distance_matrix_path, index_col=0)
                dist_df.index = dist_df.index.astype(str)
                dist_df.columns = dist_df.columns.astype(str)
                missing = [rid for rid in ids if rid not in dist_df.index]
                if missing:
                    raise ValueError(
                        "distance_matrix.csv is missing assignment IDs, cannot update "
                        "intra-class summary."
                    )
                dist = dist_df.loc[ids, ids].to_numpy(dtype=float)
            else:
                dist = compute_distance_matrix_for_ids(
                    ids=ids,
                    embeddings_path=embeddings_path,
                    distance_mode=distance_mode,
                    pca_whitening=bool(params_from_log.get("pca_whitening", False)),
                    pca_components=int(params_from_log.get("pca_components", 256)),
                    local_scaling=bool(params_from_log.get("local_scaling", False)),
                    local_k=int(params_from_log.get("local_k", 0)),
                    local_k_strategy=str(
                        params_from_log.get("local_k_strategy", "adaptive")
                    ),
                )
            intra_df = summarize_intra_class_distances(dist, ids, id_to_cluster)
            write_csv(intra_df, out_intra)

            z = build_upgma(dist)
            partitions = load_partitions_from_tables(tables_dir, ids)
            if not isinstance(partitions, OrderedDict) or len(partitions) == 0:
                raise ValueError(
                    "No partition assignment tables found under detected cluster tables dir."
                )

            support_path = (
                cluster_run_dir
                / "UPGMA"
                / f"UPGMA_{distance_mode.capitalize()}_bootstrap.nwk"
            )
            support_dict = None
            if support_path.exists():
                support_dict = parse_bootstrap_support_from_newick(support_path, ids)

            render_annotated_partitions_pdf(
                z=z,
                ids=ids,
                partitions=partitions,
                corrected_labels=id_to_cluster if show_annotation_bar else None,
                support_dict=support_dict,
                bootstrap_cutoff=support_display_cutoff,
                corrected_bar_width=annotate_bar_width,
                show_partitioning_bars=show_partitioning_bars,
                figure_width=figure_width,
                out_path=out_pdf,
            )
        else:
            typer.echo(
                "Skipped pairwise_distance_summary_intra-class.csv and "
                "UPGMA_tree_partitions_annotated.pdf (no --embeddings provided)."
            )

        write_json(summary, out_dir / "annotation_summary.json")
        typer.echo(f"Corrections applied: {summary['n_corrections']}")
        typer.echo(f"Clusters affected: {summary['n_clusters_affected']}")
        typer.echo(f"Assignments: {out_csv}")
        typer.echo(f"Changed rows: {out_changed}")
        typer.echo(f"OTU table: {out_otu}")
        if embeddings_path is not None:
            typer.echo(f"Intra-class distances: {out_intra}")
            typer.echo(f"Annotated UPGMA PDF: {out_pdf}")
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
