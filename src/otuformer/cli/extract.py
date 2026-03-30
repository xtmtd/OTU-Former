"""extract command - embeddings extraction."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import click
import typer

app = typer.Typer(help="Extract embeddings from images.")


def _format_user_command(ctx: typer.Context, params: dict[str, object]) -> str:
    parts = ["otuformer", "extract"]
    for key, value in params.items():
        source = ctx.get_parameter_source(key)
        if source is not click.core.ParameterSource.COMMANDLINE:
            continue
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(option)
            continue
        if value in (None, ""):
            continue
        parts.extend([option, str(value)])
    return " ".join(parts)


@app.callback(invoke_without_command=True)
def extract(
    ctx: typer.Context,
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", help="Path to pretrain or finetune checkpoint."
    ),
    input_images_dir: Path = typer.Option(
        ..., "--input-images-dir", help="Image directory or parent directory."
    ),
    out_dir: Path = typer.Option(
        Path("runs/extract"), "--out-dir", help="Output directory."
    ),
    model_name: str = typer.Option(
        "vit_tiny_patch16_224", "--model-name", help="timm backbone."
    ),
    extract_size: int = typer.Option(
        224, "--extract-size", help="Resize/crop size for extraction."
    ),
    use_projector_output: bool = typer.Option(
        False,
        "--use-projector-output",
        help="Use projector output instead of CLS token.",
    ),
    use_student: bool = typer.Option(
        False,
        "--use-student",
        help=(
            "Load student weights instead of teacher (EMA model). "
            "Default loads teacher, which gives better embeddings for downstream "
            "analysis (DINO/iBOT convention). No effect on finetune checkpoints."
        ),
    ),
    token_mode: str = typer.Option(
        "cls", "--token-mode", help="Token mode: cls | patch-topk | attention-pool."
    ),
    topk_patches: int = typer.Option(
        20, "--topk-patches", help="Top-K patches for patch-topk mode."
    ),
    attention_pooling_type: str = typer.Option(
        "lightweight",
        "--attention-pooling-type",
        help="Attention pooling type.",
    ),
    attention_pooling_epochs: int = typer.Option(
        20, "--attention-pooling-epochs", help="Epochs for attention query."
    ),
    label_csv: Path | None = typer.Option(
        None,
        "--label-csv",
        help=(
            "CSV with 'image' and 'label' columns for evaluating embedding quality "
            "and generating UMAP. If omitted, only extraction is performed."
        ),
    ),
    metrics_sample_size: int = typer.Option(
        10000, "--metrics-sample-size", help="Max samples for metrics evaluation."
    ),
    # UMAP options (mirrors pretrain/finetune)
    umap_n_neighbors: int = typer.Option(
        15, "--umap-n-neighbors", help="UMAP n_neighbors."
    ),
    umap_min_dist: float = typer.Option(0.1, "--umap-min-dist", help="UMAP min_dist."),
    umap_metric: str = typer.Option(
        "cosine",
        "--umap-metric",
        help="Distance metric for UMAP projection. Common choices: cosine, euclidean.",
    ),
    visualize_class_number: int = typer.Option(
        20,
        "--visualize-class-number",
        help="Max classes to show in UMAP plot.",
    ),
    disable_umap: bool = typer.Option(
        False,
        "--disable-umap",
        help="Skip UMAP generation even when --label-csv is provided.",
    ),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader workers."),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto | cpu | cuda | mps."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.embedding.extractor import extract_embeddings
    from otuformer.utils.io import write_csv

    out_dir.mkdir(parents=True, exist_ok=True)

    from otuformer.utils.logging import TeeLogger

    tee = TeeLogger(out_dir / "logs" / "extract.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        params = dict(
            checkpoint=str(checkpoint),
            input_images_dir=str(input_images_dir),
            out_dir=str(out_dir),
            model_name=model_name,
            extract_size=extract_size,
            use_projector_output=use_projector_output,
            use_student=use_student,
            token_mode=token_mode,
            topk_patches=topk_patches,
            attention_pooling_type=attention_pooling_type,
            attention_pooling_epochs=attention_pooling_epochs,
            label_csv=str(label_csv) if label_csv is not None else "",
            metrics_sample_size=metrics_sample_size,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            visualize_class_number=visualize_class_number,
            disable_umap=disable_umap,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=seed,
        )
        cli_command = _format_user_command(ctx, params)
        print(f"Command: {cli_command}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        # Enable MPS fallback for ops not yet supported on MPS
        if device in {"mps", "auto"}:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        df = extract_embeddings(
            checkpoint_path=checkpoint,
            images_dir=input_images_dir,
            model_name=model_name,
            extract_size=extract_size,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            use_projector_output=use_projector_output,
            use_student=use_student,
        )
        out_path = out_dir / "embeddings.csv"
        write_csv(df, out_path)
        print(f"Embeddings saved to: {out_path} ({len(df)} samples)")

        if label_csv is not None:
            import numpy as np
            import pandas as pd

            from otuformer.embedding.evaluator import (
                compute_clustering_metrics,
                compute_knn_accuracy,
                compute_linear_probing,
                compute_map,
                compute_recall_at_k,
                run_umap,
            )

            label_df = pd.read_csv(label_csv)
            if "image" not in label_df.columns or "label" not in label_df.columns:
                print(
                    "[Warning] label-csv must have 'image' and 'label' columns; skipping metrics."
                )
            else:
                # Align labels to embedding rows by filename (id column)
                emb_ids = df["id"].tolist()
                emb_id_set = set(emb_ids)
                label_df = label_df[label_df["image"].isin(emb_id_set)].copy()
                label_indexed = label_df.set_index("image")
                # Keep only rows whose id appears in label_csv, preserving emb order
                valid_ids = [x for x in emb_ids if x in label_indexed.index]
                emb_mask = df["id"].isin(set(valid_ids))
                dim_cols = [c for c in df.columns if c.startswith("dim_")]
                embeddings = df.loc[emb_mask, dim_cols].values
                labels = label_indexed.loc[
                    df.loc[emb_mask, "id"].tolist(), "label"
                ].values

                # subsample
                n = len(embeddings)
                if metrics_sample_size > 0 and n > metrics_sample_size:
                    rng = np.random.default_rng(seed)
                    idx = np.sort(
                        rng.choice(n, size=metrics_sample_size, replace=False)
                    )
                    embeddings = embeddings[idx]
                    labels = labels[idx]
                    print(
                        f"[Info] Metrics subsample: {len(idx)}/{n} samples (seed={seed})"
                    )

                print(f"[Metrics] Computing metrics on {len(embeddings)} samples ...")
                metrics: dict[str, object] = {}

                try:
                    metrics.update(compute_recall_at_k(embeddings, labels))
                except Exception as e:
                    print(f"[Warning] Recall@K failed: {e}")

                try:
                    metrics.update(compute_knn_accuracy(embeddings, labels))
                except Exception as e:
                    print(f"[Warning] kNN accuracy failed: {e}")

                try:
                    metrics["Linear_Probing_Acc"] = compute_linear_probing(
                        embeddings, labels
                    )
                except Exception as e:
                    print(f"[Warning] Linear probing failed: {e}")

                try:
                    metrics["mAP"] = compute_map(embeddings, labels)
                except Exception as e:
                    print(f"[Warning] mAP failed: {e}")

                try:
                    metrics.update(compute_clustering_metrics(embeddings, labels))
                except Exception as e:
                    print(f"[Warning] Clustering metrics failed: {e}")

                # Save metrics as CSV
                metrics_path = out_dir / "metrics.csv"
                metrics_rows = [{"metric": k, "value": v} for k, v in metrics.items()]
                pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
                print(f"[Metrics] Results saved to: {metrics_path}")
                for k, v in metrics.items():
                    if v != "" and v is not None:
                        try:
                            print(f"  {k}: {float(v):.4f}")
                        except (TypeError, ValueError):
                            print(f"  {k}: {v}")

                # UMAP
                if not disable_umap and len(embeddings) >= 10:
                    umap_path = out_dir / "umap.pdf"
                    print(f"[UMAP] Generating UMAP projection -> {umap_path}")
                    try:
                        run_umap(
                            embeddings,
                            labels,
                            umap_path,
                            n_neighbors=umap_n_neighbors,
                            min_dist=umap_min_dist,
                            metric=umap_metric,
                            max_classes=visualize_class_number,
                        )
                        print(f"[UMAP] Saved to: {umap_path}")
                    except Exception as e:
                        print(f"[Warning] UMAP failed: {e}")

    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
