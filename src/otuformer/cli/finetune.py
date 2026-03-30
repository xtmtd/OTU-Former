"""finetune command - ArcFace metric learning fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import typer

app = typer.Typer(help="ArcFace metric learning fine-tuning.")


@app.callback(invoke_without_command=True)
def finetune(
    ctx: typer.Context,
    checkpoint: str = typer.Option(
        "", "--checkpoint", help="Pretrained checkpoint path."
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
        "vit_tiny_patch16_224", "--model-name", help="timm backbone."
    ),
    metric_embed_dim: int = typer.Option(
        256, "--metric-embed-dim", help="Embedding dimension."
    ),
    finetune_epochs: int = typer.Option(
        20, "--finetune-epochs", help="Fine-tuning epochs."
    ),
    finetune_lr: float = typer.Option(1e-4, "--finetune-lr", help="Learning rate."),
    freeze_ratio: float = typer.Option(
        0.7, "--freeze-ratio", help="Fraction of blocks to freeze."
    ),
    loss: str = typer.Option("arcface", "--loss", help="Loss function."),
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
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.utils.logging import TeeLogger

    out_dir.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(out_dir / "logs" / "finetune.log")
    sys.stdout = tee
    try:
        from otuformer.training.trainer import run_finetune

        ns = argparse.Namespace(
            checkpoint=checkpoint,
            train_data=str(train_data),
            input_images_dir=str(input_images_dir),
            out_dir=str(out_dir),
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
        )
        run_finetune(ns)
    finally:
        sys.stdout = tee.terminal
        tee.close()
