"""finetune command - ArcFace metric learning fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import click
import typer

app = typer.Typer(
    help=(
        "ArcFace metric learning fine-tuning.\n\n"
        "Fine-tunes a pretrained backbone with ArcFace loss to produce discriminative\n"
        "embeddings for OTU clustering. Requires a pretrain checkpoint and labeled data.\n\n"
        "Quick example:\n\n"
        "  otuformer finetune --checkpoint runs/pretrain/best.pt --train-data labels.csv --input-images-dir ./images\n"
        "  otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth --train-data labels.csv --input-images-dir ./images --finetune-epochs 50\n"
    )
)


def _format_user_command(ctx: typer.Context, params: dict[str, object]) -> str:
    parts = ["otuformer", "finetune"]
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
def finetune(
    ctx: typer.Context,
    checkpoint: str = typer.Option(
        "",
        "--checkpoint",
        help=(
            "Pretrained checkpoint path (typically runs/pretrain/SSL_latest.pth). "
            "Used as initialization when --resume is not set."
        ),
    ),
    train_data: Path = typer.Option(
        ..., "--train-data", help="CSV with 'image' and 'label' columns."
    ),
    input_images_dir: Path = typer.Option(
        ..., "--input-images-dir", help="Root image directory."
    ),
    out_dir: Path = typer.Option(
        Path("runs/finetune"), "--out-dir", help="Output directory."
    ),
    model_name: str = typer.Option(
        "vit_tiny_patch16_224",
        "--model-name",
        help="timm backbone name for metric-learning encoder.",
    ),
    metric_embed_dim: int = typer.Option(
        256,
        "--metric-embed-dim",
        help=(
            "Projector embedding dimension used by ArcFace (not raw CLS dimension). "
            "If checkpoint provides config.out_dim, that value is used by default."
        ),
    ),
    finetune_epochs: int = typer.Option(
        20, "--finetune-epochs", help="Total ArcFace fine-tuning epochs."
    ),
    finetune_lr: float = typer.Option(
        1e-4,
        "--finetune-lr",
        help="Learning rate for ArcFace fine-tuning optimizer.",
    ),
    freeze_ratio: float = typer.Option(
        0.7,
        "--freeze-ratio",
        help="Fraction of backbone blocks to freeze (0.0=none, 1.0=all).",
    ),
    loss: str = typer.Option(
        "arcface",
        "--loss",
        help="Metric-learning loss name from LOSS_REGISTRY (default: arcface).",
    ),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader workers."),
    cpus: int = typer.Option(12, "--cpus", help="CPU threads for PyTorch/MKL."),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto | cpu | cuda | mps."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    log_every_n_steps: int = typer.Option(
        50, "--log-every-n-steps", help="Log metrics every N iterations."
    ),
    save_every_epochs: int = typer.Option(
        10, "--save-every-epochs", help="Save checkpoint every N epochs."
    ),
    keep_last_checkpoints: int = typer.Option(
        10, "--keep-last-checkpoints", help="Keep only last N checkpoints."
    ),
    # Visualisation / embedding metrics (mirror pretrain)
    visualize_data: Path | None = typer.Option(
        None,
        "--visualize-data",
        help=(
            "CSV used for periodic embedding metrics + UMAP. "
            "If omitted, --train-data is reused."
        ),
    ),
    extract_size: int = typer.Option(
        0,
        "--extract-size",
        help=(
            "Image size for periodic metrics/UMAP embedding extraction. "
            "<=0 means auto from backbone input size."
        ),
    ),
    metrics_sample_size: int = typer.Option(
        10000,
        "--metrics-sample-size",
        help="Max samples for expensive periodic metrics (<=0 means no cap).",
    ),
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
    disable_embedding_metrics: bool = typer.Option(
        False,
        "--disable-embedding-metrics",
        help=(
            "Disable periodic embedding metrics + UMAP generation during fine-tuning "
            "to reduce runtime overhead."
        ),
    ),
    resume: str = typer.Option(
        "",
        "--resume",
        help="Fine-tune checkpoint path to resume from (restores model/optimizer state).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Clear an existing non-empty output directory."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.utils.logging import TeeLogger
    from otuformer.utils.io import prepare_output_dir

    if resume and overwrite:
        raise typer.BadParameter("--resume and --overwrite cannot be used together.")
    if resume and not Path(resume).is_file():
        raise typer.BadParameter(f"Resume checkpoint not found: {resume}")
    prepare_output_dir(out_dir, overwrite=overwrite, allow_existing=bool(resume))
    tee = TeeLogger(
        out_dir / "logs" / "finetune.log",
        append=bool(resume),
    )
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        ns = argparse.Namespace(
            checkpoint=checkpoint,
            resume=resume,
            train_data=str(train_data),
            input_images_dir=str(input_images_dir),
            out_dir=str(out_dir),
            overwrite=overwrite,
            model_name=model_name,
            metric_embed_dim=metric_embed_dim,
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
            freeze_ratio=freeze_ratio,
            loss=loss,
            batch_size=batch_size,
            num_workers=num_workers,
            cpus=cpus,
            device=device,
            seed=seed,
            log_every_n_steps=log_every_n_steps,
            save_every_epochs=save_every_epochs,
            keep_last_checkpoints=keep_last_checkpoints,
            visualize_data=str(visualize_data) if visualize_data is not None else "",
            extract_size=extract_size,
            metrics_sample_size=metrics_sample_size,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            visualize_class_number=visualize_class_number,
            compute_embedding_metrics=not disable_embedding_metrics,
        )
        params = vars(ns)
        cli_command = _format_user_command(ctx, params)
        print(f"Command: {cli_command}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        if device in {"mps", "auto"}:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        from otuformer.training.trainer import run_finetune

        run_finetune(ns)
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
