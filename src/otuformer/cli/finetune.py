"""finetune command - supervised metric-learning fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import typer

from otuformer.cli import (
    SIZE_EXAMPLES,
    _parse_size,
    _validate_augmentation,
    _validate_orientation_policy,
    finetune_augmentation_choices,
    format_user_command,
    orientation_policy_choices,
)
from otuformer.constants import MAX_SUBCENTERS, MIN_SUBCENTERS

app = typer.Typer(
    help=(
        "Supervised metric-learning fine-tuning.\n\n"
        "Fine-tunes a pretrained backbone with a selectable supervised objective to\n"
        "produce discriminative embeddings for OTU clustering. --loss chooses arcface\n"
        "(default), supcon, subcenter-arcface, or subcenter-arcface-compact. Requires a\n"
        "pretrain checkpoint and labeled data.\n\n"
        "Quick example:\n\n"
        "  otuformer finetune --checkpoint runs/pretrain/best.pt --train-data labels.csv --input-images-dir ./images\n"
        "  otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth --train-data labels.csv --input-images-dir ./images --finetune-epochs 50\n"
        "\nAugmentation contract:\n"
        "\n"
        "  --augmentation: none or conservative.\n"
        "  default for a new run: none.\n"
        "  conservative is experimental.\n"
        "  --orientation-policy: invariant or sensitive.\n"
        "  default for a new run: sensitive.\n"
        "  invariant (opt-in): full rotation and horizontal reflection.\n"
        "\n"
        "  Dorsal, ventral, and lateral images are distinct markers, as are\n"
        "  anatomical-part views; arbitrary in-plane orientation is supported.\n"
        "  conservative uses the selected policy. Orientation-policy=sensitive\n"
        "  keeps every conservative view free of horizontal flip, with only -15 to\n"
        "  15 degrees of rotation.\n"
        "  orientation-policy is a no-op for augmentation=none; an omitted\n"
        "  initialization inherits the pretraining checkpoint policy.\n"
        "  Augmentation encourages but does not guarantee invariance.\n"
        "  sensitive is not inferred automatically from the image or taxon.\n"
        "  Omitted values inherit on resume; conflicting explicit values fail, and\n"
        "  parameter changes require a new run.\n"
    )
)


@app.callback(invoke_without_command=True)
def finetune(
    ctx: typer.Context,
    checkpoint: str = typer.Option(
        "",
        "--checkpoint",
        help=(
            "Initialization checkpoint used when --resume is not set. An SSL "
            "pretrain checkpoint installs a fresh embedding head; a "
            "fine-tune checkpoint with a matching head keeps its trained "
            "projector."
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
    metric_embed_dim: int | None = typer.Option(
        None,
        "--metric-embed-dim",
        help=(
            "Fine-tune embedding dimension (the metric-embedding head output, not "
            "the raw "
            "CLS dimension). Default: the checkpoint's recorded metric "
            "dimension, else the pretrained projector dimension."
        ),
    ),
    finetune_epochs: int = typer.Option(
        20, "--finetune-epochs", help="Total fine-tuning epochs."
    ),
    finetune_lr: float = typer.Option(
        1e-4,
        "--finetune-lr",
        help="Learning rate for the fine-tuned backbone.",
    ),
    metric_head_lr: float | None = typer.Option(
        None,
        "--metric-head-lr",
        help=(
            "Learning rate for the embedding head and, for prototype losses, the "
            "classifier; defaults to --finetune-lr."
        ),
    ),
    weight_decay: float = typer.Option(
        1e-4,
        "--weight-decay",
        help="AdamW weight decay for fine-tuning; pass 0.05 for the legacy script's setting.",
    ),
    freeze_ratio: float = typer.Option(
        0.7,
        "--freeze-ratio",
        help=(
            "Fraction of backbone blocks to freeze (0.0=none, 1.0=all). Must "
            "match the saved value on --resume."
        ),
    ),
    loss: str = typer.Option(
        "arcface",
        "--loss",
        help=(
            "Metric-learning loss: arcface (default), supcon, subcenter-arcface, "
            "or subcenter-arcface-compact. Rejected before any output when the "
            "name is unknown. On --resume an omitted --loss inherits the "
            "recorded mode; an explicit conflicting value fails."
        ),
    ),
    subcenters: int = typer.Option(
        2,
        "--subcenters",
        help=(
            "Centers per class (K) for --loss subcenter-arcface or "
            f"subcenter-arcface-compact: an integer from {MIN_SUBCENTERS} to "
            f"{MAX_SUBCENTERS}. arcface and supcon reject this flag."
        ),
    ),
    compact_weight: float = typer.Option(
        0.1,
        "--compact-weight",
        help=(
            "Same-class center-distance hinge weight for --loss "
            "subcenter-arcface-compact (cap is fixed at 0.5). Rejected for other "
            "loss modes."
        ),
    ),
    supcon_temperature: float = typer.Option(
        0.07,
        "--supcon-temperature",
        help=(
            "Temperature for --loss supcon; must be > 0. Rejected for other loss "
            "modes."
        ),
    ),
    trace_batch_ids: bool = typer.Option(
        False,
        "--trace-batch-ids",
        help=(
            "Opt in to logs/batch_ids.finetune.jsonl, recording the ordered "
            "image IDs of every training batch. Can be large; morphology is "
            "never read."
        ),
    ),
    augmentation: str | None = typer.Option(
        None,
        "--augmentation",
        autocompletion=finetune_augmentation_choices,
        help=(
            "Augmentation profile: none or conservative. Default for a new "
            "run: none. Omit to inherit the saved profile on --resume. See "
            "'Augmentation contract' above."
        ),
    ),
    orientation_policy: str | None = typer.Option(
        None,
        "--orientation-policy",
        autocompletion=orientation_policy_choices,
        help=(
            "Orientation policy: invariant or sensitive. Default for a new "
            "run: sensitive. Affects conservative; a no-op for none. Omit on "
            "initialization to inherit the checkpoint policy. See "
            "'Augmentation contract' above."
        ),
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
            "CSV with an 'image' column and optional 'label' for periodic metrics + UMAP. "
            "Without labels, only UMAP is generated. If omitted, --train-data is reused."
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
        help=(
            "Fine-tune checkpoint path to resume from (restores model/optimizer "
            "state; saved optimizer settings override the LR and weight-decay "
            "options)."
        ),
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
    _validate_augmentation(augmentation, stage="finetune")
    _validate_orientation_policy(orientation_policy)
    # v0.8.0 loss settings are validated read-only before any output directory
    # or log file exists, so a rejected run leaves no trace on disk.
    explicit_options = frozenset(
        key
        for key in ("loss", "subcenters", "compact_weight", "supcon_temperature")
        if getattr(ctx.get_parameter_source(key), "name", None) == "COMMANDLINE"
    )
    from otuformer.training.trainer import (
        _classify_finetune_source,
        _resolve_finetune_loss_config,
        _validate_finetune_resume_source,
    )
    from otuformer.utils.checkpoint import load_checkpoint

    loss_preflight = argparse.Namespace(
        loss=loss,
        subcenters=subcenters,
        compact_weight=compact_weight,
        supcon_temperature=supcon_temperature,
    )
    # Validate the initialization or resume source before any output exists.
    # An empty --checkpoint is left to run_finetune, which reports it in place.
    source_path = Path(resume) if resume else (Path(checkpoint) if checkpoint else None)
    try:
        source_checkpoint = (
            load_checkpoint(source_path) if source_path is not None else None
        )
        resolved_loss = _resolve_finetune_loss_config(
            loss_preflight,
            source_checkpoint if resume else None,
            explicit_options=explicit_options,
        )
        if source_checkpoint is not None:
            if resume:
                _validate_finetune_resume_source(
                    source_checkpoint, str(resolved_loss["loss"])
                )
            else:
                _classify_finetune_source(source_checkpoint)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
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
            metric_head_lr=metric_head_lr,
            weight_decay=weight_decay,
            freeze_ratio=freeze_ratio,
            loss=loss,
            subcenters=subcenters,
            compact_weight=compact_weight,
            supcon_temperature=supcon_temperature,
            trace_batch_ids=trace_batch_ids,
            augmentation=augmentation,
            orientation_policy=orientation_policy,
            batch_size=batch_size,
            num_workers=num_workers,
            cpus=cpus,
            device=device,
            seed=seed,
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
        )
        params = vars(ns)
        cli_command = format_user_command(ctx, params, "finetune")
        # Echo the effective loss settings so an inapplicable flag reads null
        # instead of its CLI default, matching the recorded checkpoint config.
        displayed = {
            **params,
            "loss": resolved_loss["loss"],
            "subcenters": resolved_loss["subcenters"],
            "compact_weight": resolved_loss["compact_weight"],
            "supcon_temperature": resolved_loss["supcon_temperature"],
        }
        print(f"Command: {cli_command}")
        print("Parameters:")
        print(json.dumps(displayed, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        from otuformer.training.trainer import run_finetune

        # Hand over the checkpoint the preflight already read; passing it only
        # when present keeps direct callers (and their mocks) unchanged.
        extra = (
            {"source_checkpoint": source_checkpoint}
            if source_checkpoint is not None
            else {}
        )
        run_finetune(ns, explicit_options=explicit_options, **extra)
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
