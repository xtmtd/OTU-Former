"""evaluate command stub with full options."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Evaluate embedding quality.")


@app.callback(invoke_without_command=True)
def evaluate(
    ctx: typer.Context,
    embeddings: Path = typer.Option(..., "--embeddings", help="Embeddings CSV."),
    labels: Path = typer.Option(..., "--labels", help="Labels CSV."),
    out_dir: Path = typer.Option(
        Path("runs/evaluate"), "--out-dir", help="Output directory."
    ),
    umap_dims: int = typer.Option(2, "--umap-dims", help="UMAP output dimensions."),
    umap_n_neighbors: int = typer.Option(
        15, "--umap-n-neighbors", help="UMAP n_neighbors."
    ),
    umap_min_dist: float = typer.Option(0.1, "--umap-min-dist", help="UMAP min_dist."),
    umap_metric: str = typer.Option("cosine", "--umap-metric", help="UMAP metric."),
    visualize_class_number: int = typer.Option(
        20, "--visualize-class-number", help="Max classes to show in UMAP."
    ),
    knn_k: str = typer.Option(
        "1,5,10", "--knn-k", help="Comma-separated k values for kNN/Recall@K."
    ),
    metrics_sample_size: int = typer.Option(
        10000, "--metrics-sample-size", help="Max samples for metrics."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.embedding.evaluator import (
        compute_clustering_metrics,
        compute_knn_accuracy,
        compute_linear_probing,
        compute_map,
        compute_metric_learning_diagnostics,
        compute_recall_at_k,
        run_umap,
    )
    from otuformer.utils.io import read_csv, write_json

    out_dir.mkdir(parents=True, exist_ok=True)

    emb_df = read_csv(embeddings)
    lbl_df = read_csv(labels)
    merged = emb_df.merge(lbl_df, on="id", how="inner")
    if len(merged) == 0:
        raise typer.BadParameter("No overlapping ids between embeddings and labels")

    feature_cols = [c for c in merged.columns if c.startswith("dim_")]
    x = merged[feature_cols].to_numpy()
    y = merged["label"].to_numpy()

    k_values = [int(v.strip()) for v in knn_k.split(",") if v.strip()]
    metrics = {}
    metrics.update(compute_knn_accuracy(x, y, k_values=k_values))
    metrics.update(compute_recall_at_k(x, y, k_values=k_values))
    metrics["mAP"] = compute_map(x, y)
    if len(x) >= 20:
        metrics["LinearProbe"] = compute_linear_probing(x, y)
    else:
        metrics["LinearProbe"] = float("nan")
    metrics.update(compute_clustering_metrics(x, y))
    metrics.update(compute_metric_learning_diagnostics(x, y))

    write_json(metrics, out_dir / "metrics.json")
    import pandas as pd

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    if len(x) >= 10:
        run_umap(
            x,
            y,
            out_dir / "umap.pdf",
            n_components=umap_dims,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            max_classes=visualize_class_number,
        )
    typer.echo(f"Evaluation outputs written to: {out_dir}")
