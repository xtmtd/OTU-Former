"""pretrain command - SSL self-supervised pre-training."""

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
        "SSL self-supervised pre-training (DINO/iBOT style).\n\n"
        "Trains a student-teacher ViT backbone using global/local crops and masked-token\n"
        "consistency. Outputs checkpoints that can be used for fine-tuning or direct\n"
        "embedding extraction.\n\n"
        "Quick example:\n\n"
        "  otuformer pretrain --train-data images.csv --input-images-dir ./images\n"
        "  otuformer pretrain --input-images-dir ./images --model-name vit_small_patch16_224 --max-epochs 100\n"
    )
)


def _format_user_command(ctx: typer.Context, params: dict[str, object]) -> str:
    parts = ["otuformer", "pretrain"]
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
def pretrain(
    ctx: typer.Context,
    train_data: Path | None = typer.Option(
        None,
        "--train-data",
        help=(
            "Optional CSV with an 'image' column (paths relative to --input-images-dir). "
            "If omitted, all images under --input-images-dir are used recursively."
        ),
    ),
    input_images_dir: Path = typer.Option(
        ..., "--input-images-dir", help="Root image directory."
    ),
    out_dir: Path = typer.Option(
        Path("runs/pretrain"), "--out-dir", help="Output directory."
    ),
    model_name: str = typer.Option(
        "vit_tiny_patch16_224",
        "--model-name",
        help="timm backbone name used for student/teacher encoders.",
    ),
    out_dim: int = typer.Option(
        256, "--out-dim", help="SSL projector output dimension."
    ),
    max_epochs: int = typer.Option(
        50, "--max-epochs", help="Total SSL pretraining epochs."
    ),
    lr: float = typer.Option(
        5e-4,
        "--lr",
        help="Base learning rate before warmup/cosine scheduling.",
    ),
    weight_decay: float = typer.Option(
        0.05, "--weight-decay", help="AdamW weight decay coefficient."
    ),
    warmup_epochs: int = typer.Option(
        3,
        "--warmup-epochs",
        help="Warmup epochs before cosine LR decay.",
    ),
    global_crop_size: int = typer.Option(
        224, "--global-crop-size", help="Global crop resolution."
    ),
    local_crop_size: int = typer.Option(
        96, "--local-crop-size", help="Local crop resolution."
    ),
    local_crops: int = typer.Option(6, "--local-crops", help="Number of local crops."),
    mask_ratio: float = typer.Option(
        0.5,
        "--mask-ratio",
        help="Fraction of patch tokens masked for masked-token consistency loss.",
    ),
    lambda_local: float = typer.Option(
        1.5, "--lambda-local", help="Weight for local-crop SSL loss term."
    ),
    lambda_mask: float = typer.Option(
        1.0, "--lambda-mask", help="Weight for masked-token loss term."
    ),
    teacher_momentum: float = typer.Option(
        0.995, "--teacher-momentum", help="Initial EMA momentum."
    ),
    teacher_momentum_end: float = typer.Option(
        0.999, "--teacher-momentum-end", help="Final EMA momentum."
    ),
    student_temp: float = typer.Option(
        0.1, "--student-temp", help="Student temperature."
    ),
    teacher_temp_start: float = typer.Option(
        0.04, "--teacher-temp-start", help="Initial teacher temperature."
    ),
    teacher_temp_end: float = typer.Option(
        0.07, "--teacher-temp-end", help="Final teacher temperature."
    ),
    disable_cross_view_loss: bool = typer.Option(
        False,
        "--disable-cross-view-loss",
        help=(
            "Disable cross-view global loss pairing. "
            "Default keeps full cross-view matching between global crops."
        ),
    ),
    log_every_n_steps: int = typer.Option(
        50, "--log-every-n-steps", help="Log metrics every N iterations."
    ),
    save_every_epochs: int = typer.Option(
        10, "--save-every-epochs", help="Save checkpoint every N epochs."
    ),
    keep_last_checkpoints: int = typer.Option(
        10, "--keep-last-checkpoints", help="Keep only last N checkpoints."
    ),
    visualize_data: Path | None = typer.Option(
        None,
        "--visualize-data",
        help=(
            "CSV with 'image' and optional 'label' for periodic embedding metrics/UMAP. "
            "If omitted, --train-data is reused when available."
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
        help=(
            "Distance metric for UMAP projection. Common choices: cosine, euclidean."
        ),
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
            "Disable periodic embedding metrics + UMAP generation during pretraining "
            "to reduce runtime overhead."
        ),
    ),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader workers."),
    cpus: int = typer.Option(12, "--cpus", help="CPU threads for PyTorch/MKL."),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto | cpu | cuda | mps."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    resume: str = typer.Option(
        "",
        "--resume",
        help="Checkpoint path for resuming interrupted pretraining.",
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
        out_dir / "logs" / "pretrain.log",
        append=bool(resume),
    )
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        ns = argparse.Namespace(
            train_data=str(train_data) if train_data is not None else "",
            input_images_dir=str(input_images_dir),
            out_dir=str(out_dir),
            overwrite=overwrite,
            model_name=model_name,
            out_dim=out_dim,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            local_crops=local_crops,
            mask_ratio=mask_ratio,
            lambda_local=lambda_local,
            lambda_mask=lambda_mask,
            teacher_momentum=teacher_momentum,
            teacher_momentum_end=teacher_momentum_end,
            student_temp=student_temp,
            teacher_temp_start=teacher_temp_start,
            teacher_temp_end=teacher_temp_end,
            disable_cross_view_loss=disable_cross_view_loss,
            resume=resume,
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
            batch_size=batch_size,
            num_workers=num_workers,
            cpus=cpus,
            device=device,
            seed=seed,
        )
        params = vars(ns)
        cli_command = _format_user_command(ctx, params)
        print(f"Command: {cli_command}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        if device in {"mps", "auto"}:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        from otuformer.training.trainer import run_pretrain

        run_pretrain(ns)
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
