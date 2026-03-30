"""cluster command stub with full options."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Cluster embeddings into OTUs.")


@app.callback(invoke_without_command=True)
def cluster(
    ctx: typer.Context,
    embeddings: Path = typer.Option(..., "--embeddings", help="Embeddings CSV."),
    out_dir: Path = typer.Option(
        Path("runs/cluster"), "--out-dir", help="Output directory."
    ),
    distance: str = typer.Option(
        "cosine", "--distance", help="Distance metric: cosine | euclidean."
    ),
    prefix: str = typer.Option("OTU", "--prefix", help="OTU prefix."),
    pca_whitening: bool = typer.Option(
        False, "--pca-whitening/--no-pca-whitening", help="Enable PCA whitening."
    ),
    pca_components: int = typer.Option(256, "--pca-components", help="PCA components."),
    local_scaling: bool = typer.Option(
        False, "--local-scaling/--no-local-scaling", help="Enable local scaling."
    ),
    local_k: int = typer.Option(
        0, "--local-k", help="Fixed k for local scaling (0=auto)."
    ),
    local_k_strategy: str = typer.Option(
        "adaptive", "--local-k-strategy", help="Auto-k strategy."
    ),
    cutoff_min: float = typer.Option(0.05, "--cutoff-min", help="Minimum cutoff."),
    cutoff_max: float = typer.Option(1.0, "--cutoff-max", help="Maximum cutoff."),
    cutoff_step: float = typer.Option(
        0.05, "--cutoff-step", help="Coarse cutoff step."
    ),
    custom_cutoffs: str | None = typer.Option(
        None, "--custom-cutoffs", help="Comma-separated custom cutoffs."
    ),
    num_bootstraps: int = typer.Option(
        0, "--num-bootstraps", help="Bootstrap replicates."
    ),
    bootstrap_subsample_ratio: float = typer.Option(
        0.8, "--bootstrap-subsample-ratio", help="Bootstrap subsample ratio."
    ),
    bootstrap_display_cutoff: float = typer.Option(
        50.0, "--bootstrap-display-cutoff", help="Display bootstrap support cutoff."
    ),
    save_distances: bool = typer.Option(
        False, "--save-distances/--no-save-distances", help="Save distance matrix."
    ),
    max_distance_pairs: int = typer.Option(
        1000000, "--max-distance-pairs", help="Max pairs to store."
    ),
    labels: Path | None = typer.Option(
        None, "--labels", help="Optional labels CSV for metric scan."
    ),
    metrics_sample_size: int = typer.Option(
        10000, "--metrics-sample-size", help="Max samples for metrics."
    ),
    cpus: int = typer.Option(8, "--cpus", help="CPU threads."),
    random_state: int = typer.Option(42, "--random-state", help="Random seed."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    import numpy as np

    from otuformer.delineation.distance import (
        apply_local_scaling,
        apply_pca_whitening,
        auto_select_k,
        compute_cosine_distances,
        compute_euclidean_distances,
    )
    import pandas as pd

    from otuformer.delineation.partition import (
        build_partitions_from_linkage,
        compute_partition_metrics,
        export_partition_tables,
        two_stage_threshold_scan,
    )
    from otuformer.delineation.tree import build_upgma, upgma_to_newick
    from otuformer.utils.io import read_csv, write_json

    out_dir.mkdir(parents=True, exist_ok=True)
    emb_df = read_csv(embeddings)
    ids = emb_df["id"].astype(str).tolist()
    x = emb_df[[c for c in emb_df.columns if c.startswith("dim_")]].to_numpy()

    n = len(ids)
    n_pairs = n * (n - 1) // 2
    if n_pairs > max_distance_pairs:
        raise typer.BadParameter(
            f"Pair count {n_pairs} exceeds --max-distance-pairs={max_distance_pairs}. "
            "Increase --max-distance-pairs or reduce input size."
        )

    if pca_whitening:
        n_comp = min(pca_components, x.shape[1], x.shape[0])
        x, _ = apply_pca_whitening(x, n_components=n_comp)

    if distance == "euclidean":
        d = compute_euclidean_distances(x)
    else:
        d = compute_cosine_distances(x)

    if local_scaling:
        k = (
            local_k
            if local_k > 0
            else auto_select_k(len(ids), strategy=local_k_strategy)
        )
        d, _ = apply_local_scaling(d, k=k)

    if save_distances:
        pd.DataFrame(d, index=ids, columns=ids).to_csv(out_dir / "distance_matrix.csv")

    z = build_upgma(d)
    upgma_to_newick(z, ids, out_dir / "tree.nwk")

    # Resolve true labels if provided (needed for two-stage scan & metrics)
    true_labels_arr = None
    x_for_metrics = None
    if labels is not None:
        lbl_df = read_csv(labels)
        merged = emb_df[["id"]].merge(lbl_df[["id", "label"]], on="id", how="inner")
        if len(merged) > 0:
            # re-order x and ids to match merged subset
            id_to_row = {v: i for i, v in enumerate(ids)}
            merged_idx = [id_to_row[i] for i in merged["id"].astype(str)]
            true_labels_arr = merged["label"].to_numpy()
            x_for_metrics = x[merged_idx]
            # rebuild z on the subset if sizes differ
            if len(merged_idx) < len(ids):
                d_sub = d[np.ix_(merged_idx, merged_idx)]
                from scipy.cluster import hierarchy
                from scipy.spatial.distance import squareform

                z_metrics = hierarchy.linkage(squareform(d_sub), method="average")
                ids_metrics = merged["id"].astype(str).tolist()
            else:
                z_metrics = z
                ids_metrics = ids
        else:
            typer.echo("[Warning] No overlapping IDs between embeddings and labels.")

    if custom_cutoffs:
        cutoffs = [float(v.strip()) for v in custom_cutoffs.split(",") if v.strip()]
    else:
        # Two-stage scan: use true labels when available (matches ref description.txt)
        coarse, fine = two_stage_threshold_scan(
            z,
            labels_true=true_labels_arr,
            coarse_min=cutoff_min,
            coarse_max=cutoff_max,
            coarse_step=cutoff_step,
        )
        cutoffs = sorted(set(np.concatenate([coarse, fine]).tolist()))

    partitions = build_partitions_from_linkage(z, cutoffs)
    export_partition_tables(ids, partitions, out_dir, prefix=prefix)

    if true_labels_arr is not None:
        z_for_metrics = z_metrics if "z_metrics" in dir() else z
        parts_for_metrics = build_partitions_from_linkage(z_for_metrics, cutoffs)
        metrics = compute_partition_metrics(
            parts_for_metrics, true_labels_arr, x=x_for_metrics
        )
        # Save per-cutoff metrics as CSV
        metrics_rows = [{"cutoff": th, **vals} for th, vals in metrics.items()]
        pd.DataFrame(metrics_rows).sort_values("cutoff").to_csv(
            out_dir / "partition_metrics.csv", index=False
        )
        write_json(
            {str(k): v for k, v in metrics.items()},
            out_dir / "partition_metrics.json",
        )
        typer.echo(f"Partition metrics saved to: {out_dir / 'partition_metrics.csv'}")

    typer.echo(f"Cluster outputs written to: {out_dir}")
