"""Finetune orchestration: the supervised finetune step sequence."""

from __future__ import annotations

import os

from otuformer.utils.checkpoint import (
    ARCFACE_EMBEDDING_HEAD,
    PROJECTION_EMBEDDING_HEAD,
    load_checkpoint,
    recorded_register_count,
    resolve_checkpoint,
    resolve_checkpoint_embedding_head,
    resolve_projector_out_dim,
    save_checkpoint,
)
from typing import Any, Callable, NamedTuple
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
from otuformer.utils.size import (
    _positive_int,
    resolve_backbone_native_size,
    resolve_training_image_size,
    validate_input_size,
)
import shutil
import torch
from tqdm import tqdm
from .augmentation import (
    _finetune_register_plan,
    _select_augmentation_profile,
    _select_orientation_policy,
    _validate_augmentation_config,
)
from .dataset import (
    IndexedDataset,
    MetricDataset,
    build_finetune_augmentation_config,
)
from .pretrain_loop import _freeze_backbone_blocks
from .finetune_support import (
    BatchIdTraceLogger,
    CB_DRW_BETA,
    CB_DRW_MAX_CLASS_WEIGHT,
    CB_DRW_RAMP_FRACTION,
    CB_DRW_START_FRACTION,
    LossDiagnosticsLogger,
    _accumulate_loss_diagnostics,
    _assert_pseudo_source_identity,
    _build_finetune_loss,
    _build_finetune_optimizer_for_layout,
    _cb_drw_class_weights,
    _cb_drw_r_bar,
    _center_direction_cosine,
    _class_row_counts,
    _classify_finetune_source,
    _compact_hinge_stats,
    _effective_number_class_weights,
    _empty_loss_diagnostic_counts,
    _expert_path_identity,
    _expert_rows_from_csv,
    _finetune_optimizer_layout,
    _prototype_centers,
    _prototype_k,
    _pseudo_source_status,
    _resolve_finetune_loss_config,
    _resolve_finetune_ssl_source,
    _resolved_finetune_identity,
    _rows_sha256,
    _select_finetune_embedding_head,
    _sha256_file,
    _train_manifest_sha256,
    _validate_finetune_embedding_dim,
    _validate_finetune_freeze_ratio,
    _validate_finetune_resume,
    _validate_finetune_resume_source,
    _warn_pseudo_resume_drift,
    _weighted_ce,
    _write_pseudo_outputs,
    preflight_finetune_source,
    resolve_pseudo_ssl_source,
)
from .model import (
    ArcFaceEmbeddingHead,
    OTUFormerEncoder,
)
from .reporting import (
    EnhancedMetricsLogger,
    InstantMetricsLogger,
    _compute_and_log_all_metrics,
    _compute_grad_norm,
    _compute_instant_metrics,
)
from .runtime import (
    _phase_done,
    _phase_start,
    _resolve_device,
    _set_cpus,
    _set_seed,
    warn_if_mps_fallback_unarmed,
)

def run_finetune(
    args: argparse.Namespace,
    *,
    explicit_options: frozenset[str] | set[str] = frozenset(),
    source_checkpoint: dict[str, Any] | None = None,
) -> None:
    _set_seed(args.seed)
    _set_cpus(args.cpus)
    device = _resolve_device(args.device)
    warn_if_mps_fallback_unarmed(device)
    print(f"Resolved device: {device}")

    resume_path = (
        Path(getattr(args, "resume", "")) if getattr(args, "resume", "") else None
    )
    trace_batch_ids = bool(getattr(args, "trace_batch_ids", False))
    pseudo_data = getattr(args, "pseudo_round_data", None)
    if resume_path is not None:
        ckpt_path = resume_path
    elif pseudo_data is not None:
        ckpt_path = Path(pseudo_data["ssl_checkpoint"])
    else:
        ckpt_path = Path(args.checkpoint)
    if source_checkpoint is not None:
        # The CLI preflight already read this checkpoint; reusing it keeps the
        # validation read and the training read from diverging.
        ckpt = source_checkpoint
    else:
        ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt.get("config", {})
    source_kind = _classify_finetune_source(ckpt)
    previous_centers: torch.Tensor | None = None
    if resume_path is not None:
        _validate_finetune_resume_source(ckpt)
    # Identity/lineage validation happens before the output directory can be
    # created or cleared, for direct callers as well as the CLI.
    preflight_finetune_source(ckpt, str(cfg.get("model_name") or args.model_name))
    # Resume preserves the pair recorded in its own config; a new run derives it
    # from the loaded SSL/metric payload. Neither path reopens the SSL file.
    ssl_source_identity = _resolve_finetune_ssl_source(ckpt)
    if pseudo_data is not None and resume_path is None:
        # A new pseudo round cross-checks finetune#1's optional pair against the
        # original SSL payload this run actually initializes from.
        ssl_source_identity = resolve_pseudo_ssl_source(
            ckpt, pseudo_data.get("source_config") or {}
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg.get("model_name", args.model_name)
    # ``encoder_out_dim`` sizes the pretrained projector; the fine-tune
    # embedding width (``out_dim``) is resolved separately below so
    # ``--metric-embed-dim`` can resize the ArcFace head independently.
    encoder_out_dim = int(
        cfg.get("out_dim") or cfg.get("metric_embed_dim") or args.metric_embed_dim or 256
    )
    out_dim = getattr(args, "metric_embed_dim", None)
    if out_dim is None:
        out_dim = cfg.get("metric_embed_dim") or cfg.get("out_dim") or encoder_out_dim
    out_dim = int(out_dim)
    if resume_path is not None:
        _validate_finetune_embedding_dim(cfg, out_dim)
    finetune_image_size = resolve_training_image_size(ckpt)

    requested_profile = getattr(args, "augmentation", None)
    requested_policy = getattr(args, "orientation_policy", None)
    # ``--checkpoint`` starts a new run: the pretraining augmentation profile is
    # never inherited, but an omitted orientation policy still inherits the
    # pretraining checkpoint's saved policy.
    resume_ckpt = ckpt if resume_path is not None else None
    profile = _select_augmentation_profile(
        requested_profile, resume_ckpt, stage="finetune"
    )
    policy = _select_orientation_policy(
        requested_policy,
        ckpt,
        stage="finetune",
        resume=resume_path is not None,
    )
    augmentation_config = build_finetune_augmentation_config(
        profile, finetune_image_size, orientation_policy=policy
    )
    augmentation_config = _validate_augmentation_config(
        profile, augmentation_config, resume_ckpt, stage="finetune"
    )
    args.augmentation = profile
    args.orientation_policy = policy
    print(f"Augmentation profile: {profile}")
    print(
        "Augmentation config: "
        + json.dumps(augmentation_config, indent=2, sort_keys=True)
    )

    finetune_register_mode, fine_tune_register_tokens = _finetune_register_plan(
        ckpt, model_name, finetune_image_size, encoder_out_dim
    )
    model_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "out_dim": encoder_out_dim,
        "img_size": finetune_image_size,
    }
    if finetune_register_mode == "added":
        model_kwargs.update(pretrained=False, reg_tokens=4)
    elif finetune_register_mode == "native":
        model_kwargs.update(pretrained=False)
    phase = _phase_start("model construction and device placement")
    model = OTUFormerEncoder(**model_kwargs).to(device)
    _phase_done("model construction and device placement", phase)
    validate_input_size(finetune_image_size, model, model_name)
    embedding_head = _select_finetune_embedding_head(ckpt)
    used_arcface_head = embedding_head == ARCFACE_EMBEDDING_HEAD
    if "model_state_dict" not in ckpt:
        raise ValueError(
            "Fine-tune checkpoint must contain 'model_state_dict'; "
            "ref-script checkpoints are readable by extract/export/cam only."
        )
    state_dict = ckpt["model_state_dict"]
    # ``center`` is SSL-only training state and is never read by fine-tuning;
    # dropping it keeps a resized embedding head loadable.
    state_dict = {key: value for key, value in state_dict.items() if key != "center"}
    if used_arcface_head:
        model.projector = ArcFaceEmbeddingHead(
            model.backbone.num_features, out_dim
        ).to(device)
        # ``center`` is SSL-only state and is never read by fine-tuning, but it
        # is saved with every checkpoint; size it to the metadata so readers
        # can rebuild the model.
        model.center = torch.zeros(1, out_dim, device=device)
    elif embedding_head == PROJECTION_EMBEDDING_HEAD:
        # A historical ProjectionHead feeds the loss directly, so the fine-tune
        # embedding width is fixed by the pretrained projector.
        if out_dim != encoder_out_dim:
            raise ValueError(
                f"--metric-embed-dim {out_dim} cannot be applied to a "
                f"ProjectionHead checkpoint whose projector outputs "
                f"{encoder_out_dim} dims; omit the flag to inherit it."
            )
    else:
        raise ValueError(f"Unsupported embedding head: {embedding_head}")

    # Reuse the saved projector only when it matches the head being trained.
    source_head = resolve_checkpoint_embedding_head(ckpt, state_dict)
    source_dim = resolve_projector_out_dim(state_dict)
    projector_replaced = source_head != embedding_head or (
        source_dim is not None and source_dim != out_dim
    )
    if projector_replaced:
        state_dict = {
            key: value for key, value in state_dict.items() if not key.startswith("projector.")
        }
    load_result = model.load_state_dict(state_dict, strict=False)
    # Only a deliberate head replacement (a new run whose source head or width
    # does not match) may leave the projector uninitialized; every other case,
    # and every resume, must load the trained projector instead of silently
    # re-initializing it underneath a restored optimizer state.
    needs_projector = resume_path is not None or not projector_replaced
    missing_encoder = [
        key
        for key in load_result.missing_keys
        if key.startswith("backbone.")
        or (needs_projector and key.startswith("projector."))
    ]
    if missing_encoder:
        raise ValueError(
            "Checkpoint does not provide complete encoder weights: "
            f"{len(missing_encoder)} parameter(s) are missing (for example "
            f"'{missing_encoder[0]}'). Refusing to train from a partially "
            "loaded model; the trained projector is required unless the head "
            "is deliberately replaced."
        )

    _freeze_backbone_blocks(model, args.freeze_ratio)

    pseudo_rows = None
    pseudo_class_labels = None
    if pseudo_data is not None:
        pseudo_rows = pseudo_data["rows"]
        pseudo_class_labels = pseudo_data["class_labels"]
    elif resume_path is not None and int(cfg.get("pseudo_round", 0)) == 1:
        pseudo_rows = _expert_rows_from_csv(Path(args.train_data)) + [
            {"image": row["image"], "label": row["label"], "source": "known-pseudo"}
            for row in cfg.get("accepted_pseudo_rows", [])
        ]
        # ``class_order`` (actual row order) is preferred over the historical
        # sorted-string ``class_labels``; both are top-level checkpoint fields.
        pseudo_class_labels = [
            str(label)
            for label in (
                ckpt.get("class_order") or ckpt.get("class_labels") or []
            )
        ]
    phase = _phase_start("dataset indexing and loader")
    # CSV/path resolution happens inside the dataset constructor.
    if pseudo_rows is not None:
        ds = MetricDataset(
            images_dir=Path(args.input_images_dir),
            image_size=finetune_image_size,
            augmentation_profile=profile,
            orientation_policy=policy,
            rows=pseudo_rows,
            class_labels=pseudo_class_labels,
        )
    else:
        ds = MetricDataset(
            csv_path=Path(args.train_data),
            images_dir=Path(args.input_images_dir),
            image_size=finetune_image_size,
            augmentation_profile=profile,
            orientation_policy=policy,
        )
    expert_label_indices = (
        [
            ds.class_to_idx[row["label"]]
            for row in pseudo_rows
            if row.get("source") == "expert"
        ]
        if pseudo_rows is not None
        else None
    )
    # Pseudo rounds save the class table in its actual insertion order so the
    # numeric-label order (2 before 10) survives save -> resume; ordinary
    # finetune keeps the historical sorted-string behaviour.
    saved_class_labels = (
        [str(label) for label in ds.class_to_idx]
        if pseudo_rows is not None
        else sorted(str(label) for label in ds.class_to_idx)
    )
    pseudo_config: dict[str, object] = {}
    if pseudo_data is not None:
        accepted_rows = [
            {"image": row["image"], "label": row["label"]}
            for row in pseudo_data["rows"]
            if row.get("source") == "known-pseudo"
        ]
        pseudo_config = {
            "pseudo_round": 1,
            "pseudo_source_checkpoint_sha256": _sha256_file(
                Path(pseudo_data["source_checkpoint"])
            ),
            "accepted_pseudo_rows": accepted_rows,
            "accepted_pseudo_rows_sha256": _rows_sha256(accepted_rows),
            "pseudo_rule": pseudo_data["summary"].get("thresholds"),
            "pseudo_candidate_pool_sha256": pseudo_data["summary"].get(
                "candidate_pool_sha256"
            ),
            "preprocessing": {
                "eval_transform": "center-crop",
                "image_size": int(finetune_image_size),
            },
        }
        _write_pseudo_outputs(
            out_dir, pseudo_data["diagnostics"], pseudo_data["summary"]
        )
        pseudo_config["pseudo_labels_sha256"] = _sha256_file(
            Path(out_dir) / "pseudo_labels.csv"
        )
    elif resume_path is not None and int(cfg.get("pseudo_round", 0)) == 1:
        pseudo_config = {
            "pseudo_round": 1,
            **{
                key: cfg[key]
                for key in (
                    "pseudo_source_checkpoint_sha256",
                    "accepted_pseudo_rows",
                    "accepted_pseudo_rows_sha256",
                    "pseudo_rule",
                    "pseudo_candidate_pool_sha256",
                    "pseudo_labels_sha256",
                    "preprocessing",
                )
                if key in cfg
            },
        }
    loader = DataLoader(
        IndexedDataset(ds) if trace_batch_ids else ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    _phase_done("dataset indexing and loader", phase)
    initialization_sha = _sha256_file(ckpt_path)
    if resume_path is None:
        # State the initialization source explicitly: pseudo mode starts from
        # the original SSL checkpoint, never from finetune#1.
        origin = (
            "the original SSL checkpoint"
            if pseudo_data is not None
            else "checkpoint"
        )
        print(
            f"[Info] Initializing from {origin}: "
            f"{Path(ckpt_path).resolve()} (sha256 {initialization_sha[:12]})"
        )
    ssl_initialization_sha = (
        initialization_sha
        if source_kind == "ssl"
        else cfg.get("ssl_initialization_checkpoint_sha256")
    )
    manifest_sha = _train_manifest_sha256(
        ds.image_refs, ds.label_names, Path(args.input_images_dir)
    )
    if resume_path is not None and int(cfg.get("pseudo_round", 0)) == 1:
        # Finetune#2 resume gates: the expert+accepted training rows must be the
        # recorded ones, and the checkpointed accepted rows must be unaltered.
        if cfg.get("train_manifest_sha256") != manifest_sha:
            raise ValueError(
                "Cannot resume finetune#2: expert/accepted training rows changed "
                "(training manifest hash mismatch)."
            )
        recorded_hash = cfg.get("accepted_pseudo_rows_sha256")
        if not recorded_hash:
            raise ValueError(
                "Cannot resume finetune#2: checkpoint lacks the accepted "
                "pseudo-row hash."
            )
        if _rows_sha256(cfg.get("accepted_pseudo_rows", [])) != recorded_hash:
            raise ValueError(
                "Cannot resume finetune#2: checkpointed accepted pseudo rows were "
                "altered."
            )
        _warn_pseudo_resume_drift(cfg, args, out_dir)

    n_classes = len(ds.class_to_idx)
    loss_config = _resolve_finetune_loss_config(
        args,
        ckpt if resume_path is not None else None,
        explicit_options=explicit_options,
    )
    # Building the loss allocates prototypes/centers on the device; it is part
    # of the preparation block, so its cost must not land outside every phase.
    phase = _phase_start("loss, optimizer and recovery")
    loss_fn = _build_finetune_loss(loss_config, out_dim, n_classes).to(device)
    print(
        "[Info] Effective loss config: "
        + json.dumps(loss_config, sort_keys=True)
    )
    optimizer_layout = _finetune_optimizer_layout(
        ckpt, resume_path is not None, str(loss_config["loss"])
    )

    # Pseudo-source provenance: resolved identity plus an expert-only path
    # check. Ordinary manifest validation already ran (and keeps its hard
    # failures); this only records whether the checkpoint can seed one round.
    requested_long_tail = str(getattr(args, "long_tail", "none"))
    if resume_path is not None and "long_tail" in cfg:
        if (
            "long_tail" in set(explicit_options)
            and requested_long_tail != str(cfg["long_tail"])
        ):
            raise ValueError(
                "Cannot resume: --long-tail differs from the recorded strategy "
                f"{cfg['long_tail']!r}."
            )
        long_tail = str(cfg["long_tail"])
    else:
        long_tail = requested_long_tail
    if long_tail == "cb-drw" and str(loss_config["loss"]) == "supcon":
        raise ValueError(
            "--long-tail cb-drw is rejected for SupCon: it has no class-level "
            "cross-entropy term."
        )
    identity_extras = _resolved_finetune_identity(
        optimizer_layout=optimizer_layout,
        finetune_lr=float(args.finetune_lr),
        metric_head_lr=getattr(args, "metric_head_lr", None),
        weight_decay=float(getattr(args, "weight_decay", 1e-4)),
        orientation_policy=policy,
        batch_size=int(args.batch_size),
        finetune_epochs=int(args.finetune_epochs),
        long_tail=long_tail,
    )
    if resume_path is not None:
        # A resume must not rewrite recorded identity with inert CLI values; old
        # checkpoints that lack a key stay incomplete rather than looking fresh.
        identity_extras = {
            key: value for key, value in identity_extras.items() if key in cfg
        }
    expert_ok, expert_reason = _expert_path_identity(
        ds.image_refs, Path(args.input_images_dir)
    )
    pseudo_round = int(cfg.get("pseudo_round", 0)) if resume_path is not None else 0
    eligible, ineligible_reason = _pseudo_source_status(
        loss=str(loss_config["loss"]),
        ssl_initialization_sha256=ssl_initialization_sha,
        expert_reason=expert_reason,
    )
    if resume_path is not None and "pseudo_source_eligible" in cfg:
        eligible = bool(cfg["pseudo_source_eligible"])
        ineligible_reason = cfg.get("pseudo_source_ineligible_reason")
    ssl_source_path = (
        str(Path(ckpt_path).resolve())
        if source_kind == "ssl"
        else str(cfg.get("ssl_initialization_checkpoint_path") or "")
    )
    print(
        "[Info] Pseudo-source eligibility: "
        + ("eligible" if eligible else f"ineligible ({ineligible_reason})")
    )
    pseudo_identity_cfg = None
    if pseudo_data is not None:
        pseudo_identity_cfg = pseudo_data.get("source_config") or {}
    elif resume_path is not None and int(cfg.get("pseudo_round", 0)) == 1:
        # A finetune#2 resume must also fix every experiment key, not only the
        # training rows; operational keys stay free.
        pseudo_identity_cfg = cfg
    if pseudo_identity_cfg is not None:
        _assert_pseudo_source_identity(
            pseudo_identity_cfg,
            {
                "model_name": model_name,
                "metric_embed_dim": int(out_dim),
                "embedding_head": embedding_head,
                "loss": str(loss_config["loss"]),
                "subcenters": loss_config["subcenters"],
                "compact_weight": loss_config["compact_weight"],
                "compact_cap": loss_config["compact_cap"],
                "supcon_temperature": loss_config["supcon_temperature"],
                "freeze_ratio": float(args.freeze_ratio),
                "weight_decay": float(getattr(args, "weight_decay", 1e-4)),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "finetune_epochs": int(args.finetune_epochs),
                "image_size": int(finetune_image_size),
                "augmentation_profile": profile,
                "augmentation_config": augmentation_config,
                "orientation_policy": policy,
                "long_tail": long_tail,
                "class_labels": saved_class_labels,
                "register_tokens": int(fine_tune_register_tokens),
                **identity_extras,
                # These are genuinely CLI-driven on resume, so compare the actual
                # values rather than the preserved checkpoint copies.
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
            },
            # Resume may extend the epoch budget; _validate_finetune_resume
            # owns that rule, so it is not an identity mismatch here.
            ignore=(
                frozenset({"finetune_epochs"}) if pseudo_data is None else frozenset()
            ),
        )

    optimizer = _build_finetune_optimizer_for_layout(
        optimizer_layout,
        model,
        loss_fn,
        args.finetune_lr,
        getattr(args, "metric_head_lr", None),
        getattr(args, "weight_decay", 1e-4),
    )

    use_cb_drw = long_tail == "cb-drw"
    cb_drw_record: dict[str, object] | None = None
    if use_cb_drw:
        training_counts = torch.bincount(
            torch.as_tensor(ds.labels, dtype=torch.long), minlength=n_classes
        ).double()
        # Class weights come from expert rows only; pseudo rows still change the
        # manifest-wide denominator (r_bar) through training_counts.
        expert_counts = (
            torch.bincount(
                torch.as_tensor(expert_label_indices, dtype=torch.long),
                minlength=n_classes,
            ).double()
            if expert_label_indices is not None
            else training_counts
        )
        # Keep the target weights on CPU in float64 (device-agnostic); the
        # per-step loss reduction casts the small (C,) vector to the batch
        # device/dtype, so MPS never sees float64.
        target_weights = _effective_number_class_weights(expert_counts)
        recorded = (cfg.get("cb_drw") or {}) if resume_path is not None else {}
        if recorded.get("total_steps"):
            # Resume restores the original horizon T (design: restore t and T).
            # Extra epochs continue at the target weights, since lambda is
            # clamped to 1 past the original ramp end.
            total_steps = int(recorded["total_steps"])
        else:
            total_steps = max(1, len(loader) * int(args.finetune_epochs))
        # start/ramp must come from the final T, so the recorded schedule and
        # the schedule the training loop actually applies cannot diverge.
        start_step = int(total_steps * CB_DRW_START_FRACTION)
        ramp_steps = int(total_steps * CB_DRW_RAMP_FRACTION)
        r_bar_points = {
            name: _cb_drw_r_bar(
                _cb_drw_class_weights(target_weights, step, total_steps),
                training_counts,
            )
            for name, step in (
                ("r_bar_step0", 0),
                ("r_bar_ramp_start", start_step),
                ("r_bar_ramp_end", start_step + ramp_steps),
            )
        }
        cb_drw_record = {
            "beta": CB_DRW_BETA,
            "max_class_weight": CB_DRW_MAX_CLASS_WEIGHT,
            "start_step": start_step,
            "ramp_steps": ramp_steps,
            "total_steps": total_steps,
            "expert_counts": [int(v) for v in expert_counts.tolist()],
            "target_weights": [round(float(v), 6) for v in target_weights.tolist()],
            **{name: round(value, 6) for name, value in r_bar_points.items()},
            "effective_max_step0": round(
                float(target_weights.max())
                / max(r_bar_points["r_bar_step0"], 1e-12),
                6,
            ),
        }
        print("[Info] CB-DRW: " + json.dumps(cb_drw_record, sort_keys=True))
    start_epoch = 0
    if resume_path is not None:
        _validate_finetune_resume(ckpt, saved_class_labels, args.finetune_epochs)
        _validate_finetune_freeze_ratio(ckpt, args.freeze_ratio)
        saved_loss = ckpt.get("loss_state_dict")
        if isinstance(saved_loss, dict) and "head.weight" in saved_loss:
            saved_classes = saved_loss["head.weight"].shape[0]
            if saved_classes != n_classes:
                raise ValueError(
                    "Cannot resume: checkpoint class count differs from training data."
                )
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if isinstance(saved_loss, dict):
            loss_fn.load_state_dict(saved_loss, strict=False)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if start_epoch > 0:
            # Continue the center-direction chain from the checkpoint's
            # post-epoch centers instead of restarting it.
            previous_centers = _prototype_centers(loss_fn)
        print(
            f"[Info] Resume from {resume_path} at epoch {start_epoch}, iteration {int(ckpt.get('iteration', 0))}"
        )

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    instant_logger = InstantMetricsLogger(
        logs_dir / "instant_metrics.finetune.csv", mode="finetune"
    )
    diagnostics_logger = LossDiagnosticsLogger(
        logs_dir / "loss_diagnostics.finetune.csv"
    )
    trace_logger = (
        BatchIdTraceLogger(logs_dir / "batch_ids.finetune.jsonl")
        if trace_batch_ids
        else None
    )

    compute_embedding_metrics = bool(getattr(args, "compute_embedding_metrics", True))
    epoch_logger = (
        EnhancedMetricsLogger(logs_dir / "metrics.finetune.csv", mode="pretrain")
        if compute_embedding_metrics
        else None
    )

    eval_image_size = (
        int(args.extract_size)
        if getattr(args, "extract_size", None) is not None
        else finetune_image_size
    )
    validate_input_size(eval_image_size, model, model_name)
    if args.extract_size is None:
        print(f"[Info] Auto eval crop size from model img_size: {eval_image_size}")

    # Global iteration counter (mirrors ref finetune_arcface `it` variable)
    global_step = int(ckpt.get("iteration", 0))
    _phase_done("loss, optimizer and recovery", phase)

    for epoch in range(start_epoch, args.finetune_epochs):
        model.train()
        loss_fn.train()
        running = 0.0
        batches = 0
        usable = 0
        skipped = 0
        valid_anchors = 0
        counts = _empty_loss_diagnostic_counts(n_classes, _prototype_k(loss_fn))

        pbar = tqdm(
            loader,
            desc=f"Finetune Epoch {epoch + 1}/{args.finetune_epochs}",
            ncols=120,
        )
        for step, batch in enumerate(pbar):
            batches += 1
            if trace_logger is not None:
                imgs, labels, dataset_indices = batch
                indices = [int(index) for index in dataset_indices.tolist()]
                trace_logger.log(
                    epoch=epoch,
                    step=step,
                    indices=indices,
                    refs=[ds.image_refs[index] for index in indices],
                )
            else:
                imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            emb = model(imgs)
            if use_cb_drw:
                class_weights_t = _cb_drw_class_weights(
                    target_weights, global_step, total_steps
                )
                r_bar = _cb_drw_r_bar(class_weights_t, training_counts)
                per_row = loss_fn.unreduced(emb, labels)
                loss = _weighted_ce(
                    per_row,
                    labels,
                    class_weights_t.to(device=per_row.device, dtype=per_row.dtype),
                    r_bar,
                )
                if hasattr(loss_fn, "compact_penalty"):
                    loss = loss + loss_fn.compact_penalty()
            else:
                loss = loss_fn(emb, labels)
            if loss is None:
                # SupCon: no valid positive anchor, or no different-species
                # pair. Nothing to backpropagate; count it and move on.
                skipped += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            grad_norm = _compute_grad_norm(model)
            optimizer.step()

            usable += 1
            running += float(loss.item())
            valid_anchors += int(getattr(loss_fn, "last_valid_anchors", 0))
            _accumulate_loss_diagnostics(counts, loss_fn, emb, labels)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # Log instant metrics every N steps (mirrors ref per-iteration logging)
            if global_step % max(1, args.log_every_n_steps) == 0:
                with torch.no_grad():
                    instant_metrics = _compute_instant_metrics(model, imgs, device)
                instant_logger.log(
                    iteration=global_step,
                    epoch=epoch,
                    step=step,
                    loss=float(loss.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    grad_norm=float(grad_norm),
                    **instant_metrics,
                )

            global_step += 1

        if usable == 0:
            raise ValueError(
                f"No usable batches in epoch {epoch + 1} for loss "
                f"'{loss_config['loss']}': every batch had no valid positive "
                "anchor or no different-species pair. Refusing to save a "
                "checkpoint trained on an empty signal."
            )
        avg_loss = running / usable
        print(
            f"[Finetune] Epoch {epoch + 1}/{args.finetune_epochs} - Avg Loss: "
            f"{avg_loss:.4f}"
        )
        if loss_config["loss"] == "supcon":
            # Only SupCon can skip a batch; every other mode has usable == batches
            # with skipped/valid_anchors always 0, so this line would be noise.
            print(
                f"[Finetune] Epoch {epoch + 1} batches: {batches} "
                f"usable={usable} skipped={skipped} valid_anchors={valid_anchors}"
            )
        # Snapshot the prototypes at the post-epoch boundary, after every
        # optimizer step, so this row describes the epoch that just finished.
        centers = _prototype_centers(loss_fn)
        compact_fraction, compact_penalty = _compact_hinge_stats(loss_fn)
        center_cosines = _center_direction_cosine(previous_centers, centers)
        diagnostics_logger.log(
            epoch=epoch,
            mode=loss_config["loss"],
            usable_batches=usable,
            skipped_batches=skipped,
            valid_anchors=valid_anchors,
            margin_satisfied_fraction=(
                counts["margin_hits"] / counts["samples"]
                if n_classes >= 2 and counts["samples"]
                else ""
            ),
            compact_hinge_fraction=(
                "" if compact_fraction is None else compact_fraction
            ),
            compact_mean_penalty=(
                "" if compact_penalty is None else compact_penalty
            ),
            local_assignments=(
                json.dumps(counts["local"]) if centers is not None else ""
            ),
            argmax_hits=json.dumps(counts["global"]) if centers is not None else "",
            center_direction_cosine=(
                json.dumps(center_cosines) if center_cosines else ""
            ),
        )
        previous_centers = centers

        # Save checkpoint per --save-every-epochs (mirrors ref arcface_epoch_XXXX.pth)
        should_save = (epoch + 1) % max(1, args.save_every_epochs) == 0 or (
            epoch + 1
        ) == args.finetune_epochs
        if should_save:
            ckpt_payload = {
                "epoch": epoch,
                "iteration": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_state_dict": loss_fn.state_dict(),
                "config": {
                    "model_name": model_name,
                    "metric_embed_dim": out_dim,
                    "out_dim": out_dim,
                    "embedding_head": embedding_head,
                    "register_tokens": fine_tune_register_tokens,
                    # Recorded optimizer policy: required to tell a v0.8.0
                    # two-group SupCon checkpoint from a legacy two-group one.
                    "loss": loss_config["loss"],
                    "subcenters": loss_config["subcenters"],
                    "compact_weight": loss_config["compact_weight"],
                    "compact_cap": loss_config["compact_cap"],
                    "supcon_temperature": loss_config["supcon_temperature"],
                    "seed": int(args.seed),
                    "initialization_checkpoint_sha256": initialization_sha,
                    "train_manifest_sha256": manifest_sha,
                    "optimizer_groups": optimizer_layout,
                    "prototype_weight_decay": (
                        0.0 if optimizer_layout == "v080_prototype" else None
                    ),
                    **(
                        {
                            "ssl_initialization_checkpoint_sha256": ssl_initialization_sha
                        }
                        if ssl_initialization_sha
                        else {}
                    ),
                    # Source lineage only: never active ssl_method/ssl_head, and
                    # never a pair without its SHA anchor.
                    **(
                        {
                            "ssl_source_method": ssl_source_identity[0],
                            "ssl_source_head": ssl_source_identity[1],
                        }
                        if ssl_source_identity is not None
                        and ssl_initialization_sha
                        else {}
                    ),
                    "freeze_ratio": args.freeze_ratio,
                    "image_size": finetune_image_size,
                    "input_images_dir": str(Path(args.input_images_dir).resolve()),
                    "augmentation_profile": profile,
                    "augmentation_config": augmentation_config,
                    "pseudo_round": pseudo_round,
                    **pseudo_config,
                    **(
                        {
                            "pseudo_training_summary": {
                                "epochs": int(args.finetune_epochs),
                                "training_rows": len(ds),
                                "batches_per_epoch": len(loader),
                                "completed_optimizer_steps": int(global_step),
                                "expert_counts_by_class": _class_row_counts(
                                    pseudo_rows, ds.class_to_idx, "expert"
                                ),
                                "pseudo_counts_by_class": _class_row_counts(
                                    pseudo_rows, ds.class_to_idx, "known-pseudo"
                                ),
                            }
                        }
                        if pseudo_rows is not None
                        else {}
                    ),
                    "pseudo_source_eligible": bool(eligible),
                    "pseudo_source_ineligible_reason": ineligible_reason,
                    "ssl_initialization_checkpoint_path": ssl_source_path,
                    **({"cb_drw": cb_drw_record} if cb_drw_record else {}),
                    **identity_extras,
                },
                "class_labels": saved_class_labels,
                # The actual classifier row order used by this run. ``class_labels``
                # keeps its historical sorted-string value; for numeric labels the
                # two differ, and pseudo rounds must inherit the real order.
                "class_order": [str(label) for label in ds.class_to_idx],
            }
            save_path = out_dir / f"finetune_epoch_{epoch + 1:04d}.pth"
            save_checkpoint(ckpt_payload, save_path)
            shutil.copy(str(save_path), out_dir / "finetune_latest.pth")
            print(f"[Info] Saved checkpoint {save_path}")

            keep = max(0, args.keep_last_checkpoints)
            if keep > 0:
                ckpts = sorted(out_dir.glob("finetune_epoch_*.pth"))
                if len(ckpts) > keep:
                    for old_ckpt in ckpts[:-keep]:
                        old_ckpt.unlink(missing_ok=True)
                        print(f"[Info] Deleted old checkpoint {old_ckpt.name}")

        # Compute embedding metrics at save epochs (mirrors ref compute_and_log_all_metrics)
        if should_save and compute_embedding_metrics and epoch_logger is not None:
            _compute_and_log_all_metrics(
                args=args,
                model=model,
                device=device,
                epoch=epoch,
                logs_dir=logs_dir,
                metrics_logger=epoch_logger,
                eval_image_size=eval_image_size,
                force_linear_probe=True,
            )

    instant_logger.plot(
        logs_dir, epoch_logger.path if epoch_logger is not None else None
    )
    print(f"[Info] ArcFace fine-tuning complete. Logs: {logs_dir}")
