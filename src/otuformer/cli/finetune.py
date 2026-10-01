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
    docs_url,
    finetune_augmentation_choices,
    format_user_command,
    orientation_policy_choices,
)
from otuformer.cli.constraints import validate_pseudo_options
from otuformer.constants import ARCFACE_FAMILY_LOSSES, MAX_SUBCENTERS, MIN_SUBCENTERS

# Every finetune CLI parameter is classified so that pseudo-mode inheritance and
# cross-run identity cannot silently drift as options are added. A test fails
# when a parameter has no classification here.
PARAM_CLASSIFICATION: dict[str, str] = {
    # experiment: part of the resolved cross-run identity
    "model_name": "experiment",
    "metric_embed_dim": "experiment",
    "finetune_epochs": "experiment",
    "finetune_lr": "experiment",
    "metric_head_lr": "experiment",
    "weight_decay": "experiment",
    "freeze_ratio": "experiment",
    "loss": "experiment",
    "subcenters": "experiment",
    "compact_weight": "experiment",
    "supcon_temperature": "experiment",
    "augmentation": "experiment",
    "orientation_policy": "experiment",
    "batch_size": "experiment",
    "seed": "experiment",
    "long_tail": "experiment",
    # operational: no cross-run identity key
    "out_dir": "operational",
    "device": "operational",
    "cpus": "operational",
    "num_workers": "operational",
    "trace_batch_ids": "operational",
    "log_every_n_steps": "operational",
    "save_every_epochs": "operational",
    "keep_last_checkpoints": "operational",
    "visualize_data": "operational",
    "extract_size": "operational",
    "metrics_sample_size": "operational",
    "umap_n_neighbors": "operational",
    "umap_min_dist": "operational",
    "umap_metric": "operational",
    "visualize_class_number": "operational",
    "disable_embedding_metrics": "operational",
    "overwrite": "operational",
    # control: validated and recorded by its own flow
    "train_data": "control",
    "input_images_dir": "control",
    "checkpoint": "control",
    "resume": "control",
    "pseudo_label_from": "control",
    "pseudo_similarity_floor": "control",
    "pseudo_min_gap": "control",
    "pseudo_neighbors": "control",
    "pseudo_cap_multiplier": "control",
    "pseudo_absolute_cap": "control",
}

EXPERIMENT_IDENTITY_KEYS: dict[str, tuple[str, ...]] = {
    "model_name": ("model_name",),
    "metric_embed_dim": ("metric_embed_dim", "embedding_head"),
    "finetune_epochs": ("finetune_epochs",),
    "finetune_lr": ("finetune_lr",),
    "metric_head_lr": ("effective_metric_head_lr",),
    "weight_decay": ("weight_decay",),
    "freeze_ratio": ("freeze_ratio",),
    "loss": ("loss",),
    "subcenters": ("subcenters",),
    "compact_weight": ("compact_weight",),
    "supcon_temperature": ("supcon_temperature",),
    "augmentation": ("augmentation_profile", "augmentation_config"),
    "orientation_policy": ("orientation_policy",),
    "batch_size": ("batch_size",),
    "seed": ("seed",),
    "long_tail": ("long_tail",),
}

# Derived identity keys without a direct CLI parameter; the fixed values are
# recorded so cross-run comparison never depends on an unrecorded constant.
DERIVED_IDENTITY_KEYS: tuple[str, ...] = (
    "image_size",
    "class_labels",
    "optimizer_name",
    "optimizer_groups",
    "arcface_scale",
    "arcface_margin",
    "compact_cap",
    # The patch set of the initialized backbone. Two runs can agree on every
    # other identity value and still train on different tokens (zero versus four
    # registers, or a CLS-free backbone), which would make finetune#1's pseudo
    # labels meaningless for finetune#2. For every supported model the prefix
    # count follows from ``(model_name, register_tokens)``, so this one key pins
    # the patch set without a second field. A pre-v0.10.0 source records no such
    # key and is therefore not compared.
    "register_tokens",
)

FIXED_IDENTITY_VALUES: dict[str, object] = {
    "optimizer_name": "adamw",
    "arcface_scale": 64.0,
    "arcface_margin": 0.5,
    "compact_cap": 0.5,
}

PSEUDO_RULE_OPTIONS: tuple[str, ...] = (
    "pseudo_similarity_floor",
    "pseudo_min_gap",
    "pseudo_neighbors",
    "pseudo_cap_multiplier",
    "pseudo_absolute_cap",
)


def _pseudo_round_summary_lines(data: dict) -> list[str]:
    """Concise per-class expert/pseudo counts for the run log.

    The full per-candidate diagnostics stay in ``pseudo_labels.csv`` and
    ``pseudo_summary.json``; dumping them into the log buries the training
    output, so the log only reports the per-class totals.
    """
    rows = data.get("rows") or []
    order: list[str] = []
    seeds: dict[str, int] = {}
    added: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label"))
        if label not in order:
            order.append(label)
        if row.get("source") == "expert":
            seeds[label] = seeds.get(label, 0) + 1
        elif row.get("source") == "known-pseudo":
            added[label] = added.get(label, 0) + 1
    summary = data.get("summary") or {}
    total_seeds = sum(seeds.values())
    total_added = sum(added.values())
    lines = [
        "[Info] Pseudo round: "
        f"{summary.get('accepted_count', total_added)} accepted from "
        f"{summary.get('candidate_count', 0)} candidates "
        f"({float(summary.get('acceptance_rate', 0.0)) * 100:.1f}%)"
    ]
    lines.append(f"[Info]   {'class':<34}{'expert':>7}{'+pseudo':>9}{'total':>8}")
    for label in order:
        expert_count = seeds.get(label, 0)
        added_count = added.get(label, 0)
        lines.append(
            f"[Info]   {label:<34}{expert_count:>7}{added_count:>9}"
            f"{expert_count + added_count:>8}"
        )
    lines.append(
        f"[Info]   {'TOTAL':<34}{total_seeds:>7}{total_added:>9}"
        f"{total_seeds + total_added:>8}"
    )
    reasons = summary.get("rejection_reason_counts") or {}
    if reasons:
        lines.append(
            "[Info]   rejections: "
            + ", ".join(f"{key}={value}" for key, value in sorted(reasons.items()))
        )
    lines.append(
        "[Info]   full per-candidate diagnostics: pseudo_labels.csv; "
        "summary: pseudo_summary.json"
    )
    return lines


def _validate_pseudo_options(
    *,
    long_tail: str,
    loss: str,
    pseudo_label_from: str,
    resume: str,
    similarity_floor: float,
    min_gap: float,
    neighbors: int,
    cap_multiplier: int,
    absolute_cap: int,
    explicit_options: frozenset[str],
) -> None:
    """Reject invalid pseudo/long-tail settings before any output exists."""
    try:
        validate_pseudo_options(
            long_tail=long_tail,
            loss=loss,
            pseudo_label_from=pseudo_label_from,
            similarity_floor=similarity_floor,
            min_gap=min_gap,
            neighbors=neighbors,
            cap_multiplier=cap_multiplier,
            absolute_cap=absolute_cap,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    supplied = set(explicit_options) & set(PSEUDO_RULE_OPTIONS)
    if supplied and not pseudo_label_from and not resume:
        raise typer.BadParameter(
            "Pseudo rule options require --pseudo-label-from or a finetune#2 "
            f"--resume: {', '.join(sorted(supplied))}."
        )
    if supplied and resume:
        raise typer.BadParameter(
            "Resume takes the pseudo rule constants from its checkpoint; do "
            f"not resupply: {', '.join(sorted(supplied))}."
        )


app = typer.Typer(
    help=(
        "Supervised metric-learning fine-tuning.\n\n"
        "Fine-tunes a pretrained backbone with a selectable supervised objective to\n"
        "produce discriminative embeddings for OTU clustering. --loss chooses arcface\n"
        "(default), supcon, subcenter-arcface, or subcenter-arcface-compact. Requires a\n"
        "pretrain checkpoint and labeled data.\n\n"
        "Quick examples:\n\n"
        "  # ordinary fine-tune\n"
        "  otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth \\\n"
        "      --train-data labels.csv --input-images-dir DATA_ROOT/dorsal \\\n"
        "      --out-dir RUN_ROOT/finetune1 --finetune-epochs 50\n"
        "\n"
        "  # two-round sparse-label feedback: round 1 is the pseudo source, round 2\n"
        "  # adds one automatic known-class pseudo-label round and still starts from\n"
        "  # the same original SSL checkpoint. Candidates are the images under\n"
        "  # --input-images-dir that are not in labels.csv.\n"
        "  otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth \\\n"
        "      --train-data labels.csv --input-images-dir DATA_ROOT/dorsal \\\n"
        "      --out-dir RUN_ROOT/finetune1 --finetune-epochs 50\n"
        "  otuformer finetune --train-data labels.csv --input-images-dir DATA_ROOT/dorsal \\\n"
        "      --pseudo-label-from RUN_ROOT/finetune1/finetune_latest.pth \\\n"
        "      --out-dir RUN_ROOT/finetune2 --finetune-epochs 50\n"
        "\nDocs:\n"
        f"  {docs_url('finetune')}\n"
        f"  {docs_url('training-augmentation')}\n"
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
    long_tail: str = typer.Option(
        "none",
        "--long-tail",
        help=(
            "Long-tail strategy: none (default) or cb-drw (Class-Balanced Deferred "
            "Reweighting over the ArcFace-family cross-entropy term). Rejected for "
            "--loss supcon, and must match between pseudo rounds."
        ),
    ),
    pseudo_label_from: str = typer.Option(
        "",
        "--pseudo-label-from",
        help=(
            "Completed finetune#1 ArcFace-family checkpoint used to generate one "
            "automatic known-class pseudo-label round. Finetune#2 still initializes "
            "from the same original SSL checkpoint; in that mode --checkpoint only "
            "locates that SSL file if it moved. Omitted experiment options inherit "
            "finetune#1 resolved values instead of the new-run defaults; explicit "
            f"conflicts fail. See {docs_url('finetune')}."
        ),
    ),
    pseudo_similarity_floor: float = typer.Option(
        0.75,
        "--pseudo-similarity-floor",
        help=(
            "Uncalibrated winning top-three-mean raw-CLS cosine class score floor "
            "in [-1, 1]. This is not a probability. Higher rejects more; lower "
            "accepts more. [default: 0.75]"
        ),
    ),
    pseudo_min_gap: float = typer.Option(
        0.10,
        "--pseudo-min-gap",
        help=(
            "Minimum winning-score minus runner-up-score gap in [0, 2]. Higher "
            "rejects more ambiguous candidates; lower accepts more. [default: 0.10]"
        ),
    ),
    pseudo_neighbors: int = typer.Option(
        15,
        "--pseudo-neighbors",
        help=(
            "Neighbor count k for the asymmetric own-class-excluded mutual-kNN "
            "check. Smaller is usually more local and stricter; larger is usually "
            "broader and more permissive (data-dependent). [default: 15]"
        ),
    ),
    pseudo_cap_multiplier: int = typer.Option(
        3,
        "--pseudo-cap-multiplier",
        help=(
            "Per-class pseudo-label safety cap multiplier: a class accepts at "
            "most min(multiplier * expert_seed_count, absolute_cap) rows after the "
            "floor/gap/mutual-kNN rules. Must be >= 1. [default: 3]"
        ),
    ),
    pseudo_absolute_cap: int = typer.Option(
        50,
        "--pseudo-absolute-cap",
        help=(
            "Absolute per-class pseudo-label cap, applied after the multiplier. "
            "Must be >= 1. [default: 50]"
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
            "run: none. `conservative` is experimental and not proven superior "
            "to `none`. Omit to inherit the saved profile on --resume. See "
            f"{docs_url('training-augmentation')}."
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
            f"{docs_url('training-augmentation')}."
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
        for key in (
            "loss",
            "subcenters",
            "compact_weight",
            "supcon_temperature",
            "long_tail",
            *PSEUDO_RULE_OPTIONS,
        )
        if getattr(ctx.get_parameter_source(key), "name", None) == "COMMANDLINE"
    )
    _validate_pseudo_options(
        long_tail=long_tail,
        loss=loss,
        pseudo_label_from=pseudo_label_from,
        resume=resume,
        similarity_floor=pseudo_similarity_floor,
        min_gap=pseudo_min_gap,
        neighbors=pseudo_neighbors,
        cap_multiplier=pseudo_cap_multiplier,
        absolute_cap=pseudo_absolute_cap,
        explicit_options=explicit_options,
    )
    from otuformer.training.trainer import (
        _assert_pseudo_source_identity,
        _classify_finetune_source,
        _resolve_finetune_loss_config,
        _sha256_file,
        _validate_finetune_resume_source,
        preflight_finetune_source,
        prepare_pseudo_round,
        pseudo_cli_identity,
    )
    from otuformer.utils.checkpoint import load_checkpoint

    loss_preflight = argparse.Namespace(
        loss=loss,
        subcenters=subcenters,
        compact_weight=compact_weight,
        supcon_temperature=supcon_temperature,
    )
    # Pseudo mode resolves the whole round in memory before any output exists;
    # an empty or invalid round leaves no directory or file behind.
    pseudo_round_data = None
    progress_lines: list[str] = []
    inherit_cfg: dict | None = None
    if pseudo_label_from and not resume:
        try:
            pseudo_source_cfg = (
                load_checkpoint(Path(pseudo_label_from)).get("config") or {}
            )
        except OSError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if "long_tail" not in explicit_options:
            long_tail = str(pseudo_source_cfg.get("long_tail", "none"))
        if "loss" not in explicit_options:
            loss_preflight.loss = pseudo_source_cfg.get("loss", loss_preflight.loss)
        if "subcenters" not in explicit_options:
            loss_preflight.subcenters = pseudo_source_cfg.get("subcenters")
        if "compact_weight" not in explicit_options:
            loss_preflight.compact_weight = pseudo_source_cfg.get("compact_weight")
        if "supcon_temperature" not in explicit_options:
            loss_preflight.supcon_temperature = pseudo_source_cfg.get(
                "supcon_temperature"
            )
        # An inherited loss parameter is a deliberate choice, so mark the
        # applicable non-null ones explicit; otherwise the resolver would fall
        # back to the new-run default instead of the source's value.
        explicit_options = explicit_options | {
            key
            for key in ("subcenters", "compact_weight", "supcon_temperature")
            if key not in explicit_options
            and getattr(loss_preflight, key, None) is not None
        }
        inherit_cfg = pseudo_source_cfg
    elif resume:
        # Resume takes accepted rows and settings from its own checkpoint;
        # --pseudo-label-from only validates the recorded source SHA-256.
        try:
            resume_cfg = load_checkpoint(Path(resume)).get("config") or {}
        except OSError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if pseudo_label_from:
            recorded = resume_cfg.get("pseudo_source_checkpoint_sha256")
            if not recorded:
                raise typer.BadParameter(
                    "resumed checkpoint records no pseudo-source SHA-256, so "
                    "--pseudo-label-from cannot be validated"
                )
            if recorded != _sha256_file(Path(pseudo_label_from)):
                raise typer.BadParameter(
                    "--pseudo-label-from file SHA-256 does not match the resumed "
                    "checkpoint"
                )
        if int(resume_cfg.get("pseudo_round", 0)) == 1:
            # A finetune#2 resume fixes every experiment key; omitted options
            # inherit the checkpoint's resolved values rather than new-run
            # defaults, so the identity gate sees no spurious difference.
            inherit_cfg = resume_cfg
            if "long_tail" not in explicit_options:
                long_tail = str(resume_cfg.get("long_tail", long_tail))

    # Omitted experiment options inherit the relevant source's resolved values
    # rather than the ordinary new-run defaults; explicit conflicts are then
    # caught by the identity check before any output directory is created.
    explicit_experiment = {
        key
        for key in EXPERIMENT_IDENTITY_KEYS
        if getattr(ctx.get_parameter_source(key), "name", None) == "COMMANDLINE"
    }
    if inherit_cfg is not None:

        def _inherit(name: str, current, key: str | None = None):
            if name in explicit_experiment:
                return current
            value = inherit_cfg.get(key or name)
            return current if value is None else value

        model_name = _inherit("model_name", model_name)
        metric_embed_dim = _inherit("metric_embed_dim", metric_embed_dim)
        finetune_epochs = _inherit("finetune_epochs", finetune_epochs)
        finetune_lr = _inherit("finetune_lr", finetune_lr)
        metric_head_lr = _inherit(
            "metric_head_lr", metric_head_lr, key="effective_metric_head_lr"
        )
        weight_decay = _inherit("weight_decay", weight_decay)
        freeze_ratio = _inherit("freeze_ratio", freeze_ratio)
        augmentation = _inherit(
            "augmentation", augmentation, key="augmentation_profile"
        )
        orientation_policy = _inherit("orientation_policy", orientation_policy)
        batch_size = _inherit("batch_size", batch_size)
        seed = _inherit("seed", seed)

    if pseudo_label_from and not resume:
        def _progress(message: str) -> None:
            # Live console feedback: the round runs before the log exists.
            progress_lines.append(message)
            print(message, flush=True)

        try:
            pseudo_round_data = prepare_pseudo_round(
                pseudo_label_from=Path(pseudo_label_from),
                train_data=Path(train_data),
                input_images_dir=Path(input_images_dir),
                out_dir=Path(out_dir),
                similarity_floor=pseudo_similarity_floor,
                min_gap=pseudo_min_gap,
                neighbors=pseudo_neighbors,
                cap_multiplier=pseudo_cap_multiplier,
                absolute_cap=pseudo_absolute_cap,
                long_tail=long_tail,
                loss=loss if "loss" in explicit_options else None,
                checkpoint=Path(checkpoint) if checkpoint else None,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                visualize_data=(
                    Path(visualize_data) if visualize_data is not None else None
                ),
                progress=_progress,
            )
        except (ValueError, OSError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    # Validate the initialization or resume source before any output exists.
    # An empty --checkpoint is left to run_finetune, which reports it in place.
    if pseudo_round_data is not None:
        source_path = Path(pseudo_round_data["ssl_checkpoint"])
    else:
        source_path = (
            Path(resume) if resume else (Path(checkpoint) if checkpoint else None)
        )
    try:
        source_checkpoint = (
            load_checkpoint(source_path) if source_path is not None else None
        )
        resolved_loss = _resolve_finetune_loss_config(
            loss_preflight,
            source_checkpoint if resume else None,
            explicit_options=explicit_options,
        )
        # The training namespace must carry the same resolved loss settings the
        # preflight used, otherwise pseudo-mode inheritance is silently lost.
        loss = str(resolved_loss["loss"])
        subcenters = resolved_loss["subcenters"]
        compact_weight = resolved_loss["compact_weight"]
        supcon_temperature = resolved_loss["supcon_temperature"]
        # Full experiment-identity check before any output directory is created
        # or cleared, so an explicit conflict cannot destroy a prior run first.
        if pseudo_round_data is not None:
            _assert_pseudo_source_identity(
                pseudo_round_data["source_config"],
                pseudo_cli_identity(
                    ssl_checkpoint=source_checkpoint,
                    source_cfg=pseudo_round_data["source_config"],
                    resolved_loss=resolved_loss,
                    model_name=model_name,
                    metric_embed_dim=metric_embed_dim,
                    finetune_epochs=finetune_epochs,
                    finetune_lr=finetune_lr,
                    metric_head_lr=metric_head_lr,
                    weight_decay=weight_decay,
                    freeze_ratio=freeze_ratio,
                    augmentation=augmentation,
                    orientation_policy=orientation_policy,
                    batch_size=batch_size,
                    seed=seed,
                    long_tail=long_tail,
                    source_class_labels=pseudo_round_data.get(
                        "source_class_labels"
                    ),
                ),
            )
        if source_checkpoint is not None:
            if resume:
                _validate_finetune_resume_source(
                    source_checkpoint, str(resolved_loss["loss"])
                )
            else:
                _classify_finetune_source(source_checkpoint)
            # Register-layout preflight before --overwrite can clear an existing
            # output directory.
            preflight_finetune_source(source_checkpoint, model_name)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    prepare_output_dir(out_dir, overwrite=overwrite, allow_existing=bool(resume))
    tee = TeeLogger(
        out_dir / "logs" / "finetune.log",
        append=bool(resume),
    )
    if progress_lines:
        # The pseudo round ran before this log existed; record its progress so
        # the log explains what happened before the header block.
        for line in progress_lines:
            tee.log.write(TeeLogger._timestamp_prefix() + line + "\n")
        tee.log.flush()
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
            long_tail=long_tail,
            pseudo_label_from=pseudo_label_from,
            pseudo_similarity_floor=pseudo_similarity_floor,
            pseudo_min_gap=pseudo_min_gap,
            pseudo_neighbors=pseudo_neighbors,
            pseudo_cap_multiplier=pseudo_cap_multiplier,
            pseudo_absolute_cap=pseudo_absolute_cap,
            pseudo_round_data=pseudo_round_data,
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
        # The pseudo diagnostics payload can be thousands of rows; it is
        # summarized below and written to the output files instead of the log.
        displayed.pop("pseudo_round_data", None)
        print(f"Command: {cli_command}")
        print("Parameters:")
        print(json.dumps(displayed, ensure_ascii=False, indent=2, sort_keys=True))
        if pseudo_round_data is not None:
            for line in _pseudo_round_summary_lines(pseudo_round_data):
                print(line)
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
