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

from otuformer.cli import (
    SIZE_EXAMPLES,
    _parse_size,
    _validate_augmentation,
    _validate_orientation_policy,
    orientation_policy_choices,
    pretrain_augmentation_choices,
)

app = typer.Typer(
    help=(
        "SSL self-supervised pre-training (DINO/iBOT style).\n\n"
        "Trains a student-teacher ViT backbone using global/local crops and a\n"
        "patch-level objective (none | consistency | masked-feature | ibot).\n"
        "Outputs checkpoints that can be used for fine-tuning or direct\n"
        "embedding extraction. See the pretrain section of the README for the\n"
        "full patch-loss and masking contract.\n\n"
        "Quick example:\n\n"
        "  otuformer pretrain --train-data images.csv --input-images-dir ./images\n"
        "  otuformer pretrain --input-images-dir ./images --model-name vit_small_patch16_224 --max-epochs 100\n"
        "\nAugmentation contract:\n"
        "\n"
        "  --augmentation: global-barcode, color-robust, or legacy.\n"
        "  default for a new run: global-barcode.\n"
        "  --orientation-policy: invariant or sensitive.\n"
        "  default for a new run: sensitive.\n"
        "  invariant (opt-in): full rotation and horizontal reflection.\n"
        "\n"
        "  Dorsal, ventral, and lateral images are distinct markers, as are\n"
        "  anatomical-part views; arbitrary in-plane orientation is supported.\n"
        "  color-robust can suppress diagnostic color, pattern, or metallic sheen.\n"
        "  legacy reproduces 0.2.1 and records either policy without changing its\n"
        "  historical transforms. An omitted policy can still be inherited by a new\n"
        "  fine-tuning conservative run from that checkpoint.\n"
        "  sensitive is not inferred automatically from the image or taxon.\n"
        "  Orientation-policy=sensitive keeps every new pretraining view and\n"
        "  fine-tuning conservative view free of horizontal flip, with only -15 to\n"
        "  15 degrees of rotation.\n"
        "  All global and local pretraining views use the selected policy, so\n"
        "  local-to-global training cannot silently reintroduce broad rotation or\n"
        "  reflection.\n"
        "  Augmentation encourages but does not guarantee invariance.\n"
        "  Omitted values inherit on resume; conflicting explicit values fail, and\n"
        "  parameter changes require a new run.\n"
    )
)


def _parse_mask_ratio(value: object) -> object:
    """Parse ``auto`` or a float in ``(0, 1)``; return ``"auto"`` or the float."""
    if value is None or value == "auto":
        return "auto"
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise typer.BadParameter(
            f"--mask-ratio must be 'auto' or a float in (0, 1), got '{value}'"
        ) from None
    if not 0.0 < ratio < 1.0:
        raise typer.BadParameter(
            f"--mask-ratio must be 'auto' or a float in (0, 1), got '{value}'"
        )
    return ratio


PATCH_LOSS_CHOICES = ("none", "consistency", "masked-feature", "ibot")
MASKING_STRATEGY_CHOICES = ("random", "blockwise", "hybrid")
IBOT_PROTOTYPES_MIN = 2


def _validate_patch_options(
    patch_loss: str, masking_strategy: str, ibot_prototypes: int
) -> None:
    """Reject unknown enum values before any output directory is touched."""
    if patch_loss not in PATCH_LOSS_CHOICES:
        raise typer.BadParameter(
            f"--patch-loss must be one of {', '.join(PATCH_LOSS_CHOICES)}, "
            f"got '{patch_loss}'"
        )
    if masking_strategy not in MASKING_STRATEGY_CHOICES:
        raise typer.BadParameter(
            f"--masking-strategy must be one of {', '.join(MASKING_STRATEGY_CHOICES)}, "
            f"got '{masking_strategy}'"
        )
    if ibot_prototypes < IBOT_PROTOTYPES_MIN:
        raise typer.BadParameter(
            f"--ibot-prototypes must be an integer >= {IBOT_PROTOTYPES_MIN}, "
            f"got '{ibot_prototypes}'"
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
    augmentation: str | None = typer.Option(
        None,
        "--augmentation",
        autocompletion=pretrain_augmentation_choices,
        help=(
            "Augmentation profile: global-barcode, color-robust, or legacy. "
            "Default for a new run: global-barcode. Omit to inherit the saved "
            "profile on --resume. See 'Augmentation contract' above."
        ),
    ),
    orientation_policy: str | None = typer.Option(
        None,
        "--orientation-policy",
        autocompletion=orientation_policy_choices,
        help=(
            "Orientation policy for every global and local view: invariant or "
            "sensitive. Default for a new run: sensitive. Omit to inherit the "
            "saved policy on --resume. See 'Augmentation contract' above."
        ),
    ),
    global_crop_size: str = typer.Option(
        "auto",
        "--global-crop-size",
        help=(
            "Global crop resolution. 'auto' resolves to the backbone's native "
            "input size on a new run, or the checkpoint's recorded size on "
            f"--resume. {SIZE_EXAMPLES}"
        ),
    ),
    local_crop_size: int | None = typer.Option(
        None,
        "--local-crop-size",
        help=(
            "Local crop resolution. Omit to use 96 for a new run or inherit "
            "the checkpoint value on --resume. Must be divisible by the "
            "backbone patch size (e.g. 98 or 112 for patch-14 models)."
        ),
    ),
    local_crops: int | None = typer.Option(
        None,
        "--local-crops",
        help=(
            "Number of local crops. Omit to use 6 for a new run or inherit "
            "the checkpoint value on --resume."
        ),
    ),
    patch_loss: str = typer.Option(
        "consistency",
        "--patch-loss",
        help=(
            "Patch-level objective. 'none', 'consistency' (default; selects "
            "visible same-position patches for normalized cosine regression, "
            "i.e. masked-position consistency, never input masking), "
            "'masked-feature' (true continuous masked feature prediction "
            "of the teacher's final-four-block patch target) or 'ibot' "
            "(experimental prototype-distribution prediction)."
        ),
    ),
    masking_strategy: str = typer.Option(
        "random",
        "--masking-strategy",
        help=(
            "Masking geometry for masked-feature/ibot only. 'random' samples "
            "independent patch positions; 'blockwise' merges bounded rectangular "
            "regions; 'hybrid' takes half blockwise and half random positions. "
            "Not applicable to none/consistency; the exact geometry limits are "
            "documented in the README."
        ),
    ),
    mask_ratio: str = typer.Option(
        "auto",
        "--mask-ratio",
        help=(
            "'auto' or a float in (0, 1). Fraction of patch positions used by "
            "the selected patch objective. A new run resolves 'auto' to 0.30 "
            "(v0.6.x used a 0.50 default). Used as the visible same-position "
            "share for consistency and as the masked student-input fraction "
            "for masked-feature/ibot."
        ),
    ),
    ibot_prototypes: int = typer.Option(
        512,
        "--ibot-prototypes",
        help=(
            "Prototype dictionary size for the experimental ibot mode: any "
            "integer >= 2, default 512. Larger dictionaries suit larger "
            "datasets (for example 4096 or 16384); powers of two are a "
            "convenient habit, not a requirement. Not applicable to other "
            "patch modes."
        ),
    ),
    lambda_local: float = typer.Option(
        1.5, "--lambda-local", help="Weight for local-crop SSL loss term."
    ),
    lambda_mask: float = typer.Option(
        1.0,
        "--lambda-mask",
        help=(
            "Weight for the patch-loss term. ibot's cross-entropy is on a "
            "different scale from the cosine patch losses, so lower it (for "
            "example 0.25-0.5) when using --patch-loss ibot."
        ),
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
            "CSV with an 'image' column and optional 'label' for periodic embedding metrics/UMAP. "
            "Without labels, only UMAP is generated. If omitted, --train-data is reused when available."
        ),
    ),
    extract_size: str = typer.Option(
        "auto",
        "--extract-size",
        help=(
            "Image size for periodic metrics/UMAP embedding extraction. "
            "'auto' uses the model's img_size. "
            f"{SIZE_EXAMPLES}"
        ),
    ),
    metrics_sample_size: int = typer.Option(
        10000,
        "--metrics-sample-size",
        help="Max samples for periodic metrics and UMAP (<=0 means no cap).",
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

    if device in {"mps", "auto"}:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    from otuformer.utils.logging import TeeLogger
    from otuformer.utils.io import prepare_output_dir

    if resume and overwrite:
        raise typer.BadParameter("--resume and --overwrite cannot be used together.")
    if resume and not Path(resume).is_file():
        raise typer.BadParameter(f"Resume checkpoint not found: {resume}")
    _validate_augmentation(augmentation, stage="pretrain")
    _validate_orientation_policy(orientation_policy)
    _validate_patch_options(patch_loss, masking_strategy, ibot_prototypes)
    requested_mask_ratio = _parse_mask_ratio(mask_ratio)
    explicit_sources = {
        name: ctx.get_parameter_source(name) is click.core.ParameterSource.COMMANDLINE
        for name in (
            "patch_loss",
            "masking_strategy",
            "mask_ratio",
            "ibot_prototypes",
            "lambda_mask",
        )
    }
    if not resume:
        # Reject semantic patch-option conflicts before --overwrite can clear an
        # existing output directory. Resume never clears it, and its saved mode
        # is only known once the checkpoint is loaded in run_pretrain().
        from otuformer.training.trainer import _resolve_patch_config

        _resolve_patch_config(
            argparse.Namespace(
                patch_loss=patch_loss,
                masking_strategy=masking_strategy,
                mask_ratio=requested_mask_ratio,
                ibot_prototypes=ibot_prototypes,
                lambda_mask=lambda_mask,
                patch_loss_explicit=explicit_sources["patch_loss"],
                masking_strategy_explicit=explicit_sources["masking_strategy"],
                mask_ratio_explicit=explicit_sources["mask_ratio"],
                ibot_prototypes_explicit=explicit_sources["ibot_prototypes"],
                lambda_mask_explicit=explicit_sources["lambda_mask"],
            ),
            None,
        )
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
            augmentation=augmentation,
            orientation_policy=orientation_policy,
            global_crop_size=_parse_size(global_crop_size, stage="--global-crop-size"),
            local_crop_size=local_crop_size,
            local_crops=local_crops,
            mask_ratio=requested_mask_ratio,
            lambda_local=lambda_local,
            lambda_mask=lambda_mask,
            patch_loss=patch_loss,
            patch_loss_explicit=explicit_sources["patch_loss"],
            masking_strategy=masking_strategy,
            masking_strategy_explicit=explicit_sources["masking_strategy"],
            mask_ratio_explicit=explicit_sources["mask_ratio"],
            ibot_prototypes=ibot_prototypes,
            ibot_prototypes_explicit=explicit_sources["ibot_prototypes"],
            lambda_mask_explicit=explicit_sources["lambda_mask"],
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
            extract_size=_parse_size(extract_size, stage="--extract-size"),
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

        from otuformer.training.trainer import run_pretrain

        run_pretrain(ns)
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
