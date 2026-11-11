"""Pretrain orchestration: the pretrain step and epoch sequence."""

from __future__ import annotations

import os

from typing import Any, Callable, NamedTuple
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from otuformer.constants import (
    ARCFACE_FAMILY_LOSSES,
    DINO_CENTERING_MODES,
    DINO_PROTOTYPES_MIN,
    IBOT_PROTOTYPES_MIN,
    LEGACY_RELATIONAL_PAIRING,
    MASKING_STRATEGIES,
    MAX_SUBCENTERS,
    MIN_SUBCENTERS,
    PATCH_LOSS_MODES,
    RELATIONAL_CENTER_SOURCE_CHOICES,
    RELATIONAL_PAIRING_CHOICES,
    RELATIONAL_TEACHER_VIEW_CHOICES,
    SSL_HEAD_BY_METHOD,
    SSL_METHODS,
)
import argparse
import json
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
import math
import torch.nn as nn
import shutil
import torch
from tqdm import tqdm
from otuformer.utils.size import (
    _positive_int,
    resolve_backbone_native_size,
    resolve_training_image_size,
    validate_input_size,
)
from .augmentation import (
    _check_register_option_against_resume,
    _resume_register_plan,
)
from .dataset import (
    MultiCropDataset,
)
from .dino import (
    DINOObjective,
    validate_dino_prototype_layer,
)
from .loss import (
    ibot_patch_loss,
    masked_patch_cosine_loss,
)
from .model import (
    OTUFormerEncoder,
    PatchObjective,
    native_register_count,
)
from .objectives import (
    _compute_dino_global_loss,
    _compute_dino_local_loss,
    _compute_global_loss,
    _compute_koleo_raw,
    _compute_local_loss,
    _dino_cls_diagnostics,
    _koleo_instant_metrics,
    _make_relational_step_targets,
    _update_patch_center,
    _update_teacher_center,
)
from .relational_queue import (
    RelationalQueue,
    queue_config_bundle,
)
from .patch_objective import (
    _sample_patch_masks,
    _validate_blockwise_grid,
)
from .pretrain_recipe import (
    _build_pretrain_schedules,
    _effective_pretrain_ssl_config,
    take_pretrain_handoff,
    _initialize_dino_student_projector,
    _pretrain_decay_partition_empty,
    _pretrain_optimizer_groups,
    _report_restored_weight_decay,
    _resolve_pretrain_recipe,
    _ResolvedPretrainRecipe,
    _ssl_display_config,
    _synchronize_dino_teacher,
    _TRANSIENT_ARGS,
    _UNSAVED_ARGS,
    _validate_pretrain_optimizer_layout,
    KOLEO_STORED_KEYS,
    preflight_pretrain_run,
    prepare_pretrain_relational_queue,
    RELATIONAL_QUEUE_CONFIG_KEY,
    RELATIONAL_QUEUE_STATE_KEY,
    RELATIONAL_TARGET_CONFIG_KEY,
    resolve_pretrain_settings,
    resume_payload_for,
    verify_relational_queue_preparation,
)
from .reporting import (
    EnhancedMetricsLogger,
    InstantMetricsLogger,
    _compute_and_log_all_metrics,
    _compute_grad_norm_parameters,
    _compute_instant_metrics,
    _ibot_diagnostics,
    _mask_ratio_diagnostics,
    _relational_queue_metrics,
)
from .runtime import (
    _capture_rng_state,
    _phase_done,
    _phase_start,
    _resolve_device,
    _restore_rng_state,
    _set_cpus,
    _set_seed,
    warn_if_mps_fallback_unarmed,
    update_teacher,
)

def _fifo_cast_state(
    module: nn.Module, saved_state: object
) -> dict[str, object]:
    """Cast saved tensors to this module's dtype before loading them.

    ``load_state_dict`` moves a float64 source tensor to the target device as
    float64, which backends without float64 (MPS) reject. Casting on the CPU
    first keeps the restore expression identical to what the enabled preflight
    validates.
    """
    if not isinstance(saved_state, dict):
        return {}
    target = module.state_dict()
    cast: dict[str, object] = {}
    for key, value in saved_state.items():
        reference = target.get(key)
        if (
            isinstance(value, torch.Tensor)
            and isinstance(reference, torch.Tensor)
            and value.is_floating_point()
            and reference.is_floating_point()
        ):
            cast[key] = value.to(dtype=reference.dtype)
        else:
            cast[key] = value
    return cast


def _restore_relational_center(
    center: torch.Tensor, reference: torch.Tensor, device: torch.device
) -> None:
    """Restore the saved center into the live float32 buffer.

    The cast happens before the device move: ``.to(device)`` alone would carry a
    float64 payload onto a backend that does not support float64.
    """
    reference.copy_(center.to(dtype=reference.dtype).to(device=device))


def _unpack_pretrain_batch(
    batch: object, *, queue_enabled: bool
) -> tuple[list[torch.Tensor], tuple[str, ...] | None]:
    """Split one loader batch into its views and (enabled only) image keys.

    A queue-off run keeps the historical list-of-views contract; an enabled run
    requires the keyed mapping and never falls back to a silent list.
    """
    if not queue_enabled:
        if isinstance(batch, dict) or not isinstance(batch, (list, tuple)):
            raise ValueError(
                "The pretraining loader produced an unexpected batch envelope; a "
                "queue-disabled run reads a list of view tensors."
            )
        return list(batch), None
    if not isinstance(batch, dict):
        raise ValueError(
            "An enabled relational queue requires the keyed batch envelope from "
            "MultiCropDataset(prepared_relational_data=...); the loader produced "
            f"{type(batch).__name__} instead."
        )
    if set(batch) != {"views", "image_key"}:
        raise ValueError(
            "An enabled relational queue requires exactly the batch keys "
            f"'views' and 'image_key', got {sorted(batch)}."
        )
    views = list(batch["views"])
    keys = batch["image_key"]
    if isinstance(keys, (str, bytes)):
        raise ValueError("The enabled batch envelope must carry image-key strings.")
    resolved_keys = tuple(str(key) for key in keys)
    for view in views:
        if int(view.shape[0]) != len(resolved_keys):
            raise ValueError(
                "The enabled batch envelope must align one image key per row: "
                f"{len(resolved_keys)} key(s) for {int(view.shape[0])} view "
                "rows."
            )
    return views, resolved_keys


def run_pretrain(
    args: argparse.Namespace,
    *,
    pretrain_recipe: _ResolvedPretrainRecipe | None = None,
) -> None:
    # Take the CLI's handoff before any check can raise, so a rejected call
    # never leaves an answer behind for a retry with a different request.
    handoff = take_pretrain_handoff(args)
    register_option = getattr(args, "register_tokens", None)
    if register_option not in (None, 0, 4):
        raise ValueError(
            f"--register-tokens must be 0 or 4 (got {register_option!r})."
        )
    if hasattr(args, "disable_cross_view_loss"):
        raise ValueError(
            "--disable-cross-view-loss was removed; use --relational-pairing "
            "{cross,full} instead."
        )
    # Resolve and validate before any output file exists, so a doomed run never
    # clears or pollutes an existing output directory. The CLI already did both
    # and hands the result over.
    # Bound to the *current* resume path: changing args.resume must not keep
    # serving a previously loaded checkpoint.
    resume_ckpt = resume_payload_for(args, handoff)
    # The queue preparation is model-free and runs before every probe: enabled
    # data identity, the batch/steps-per-epoch locks and the strict recovery
    # checks are all settled before a probe encoder exists. The CLI usually
    # hands it over; a direct-core caller resolves it here.
    preparation = verify_relational_queue_preparation(
        handoff.relational_queue_preparation, args, resume_ckpt
    )
    settings = handoff.settings
    if pretrain_recipe is not None and settings is None:
        raise ValueError(
            "pretrain_recipe requires the matching pretrain_settings handoff; "
            "pass both or neither."
        )
    if settings is None:
        settings = resolve_pretrain_settings(args, resume_ckpt)
        preflight_pretrain_run(args, settings)
    if pretrain_recipe is None:
        pretrain_recipe = _resolve_pretrain_recipe(
            args,
            resume_ckpt,
            settings=settings,
            relational_queue_preparation=preparation,
        )
    elif pretrain_recipe.settings is not settings:
        raise ValueError(
            "pretrain_recipe was resolved from a different PretrainSettings "
            "object; resolve the recipe once and pass both."
        )
    elif (
        preparation.active
        and pretrain_recipe.relational_queue_preparation is not preparation
    ):
        raise ValueError(
            "The supplied pretrain_recipe was resolved from a different "
            "relational queue preparation; resolve the recipe and the "
            "preparation from the same request."
        )
    # Effective versus requested: the raw request stays in the printed
    # ``Parameters:`` record while the run and the saved args read the effective
    # values. Queue-off runs keep their existing batch handling untouched.
    effective_batch_size = preparation.effective_batch_size
    if preparation.ssl_method == "relational":
        args.relational_queue_size = preparation.capacity
        args.requested_relational_queue_size = preparation.requested_capacity
        args.relational_queue_size_explicit = (
            preparation.capacity_source == "explicit"
        )
    if preparation.active:
        args.batch_size = effective_batch_size
        args.requested_batch_size = preparation.requested_batch_size
        args.batch_size_explicit = preparation.batch_size_source == "explicit"
    _set_seed(args.seed)
    _set_cpus(args.cpus)
    device = (
        preparation.resolved_device
        if preparation.resolved_device is not None
        else _resolve_device(args.device)
    )
    warn_if_mps_fallback_unarmed(device)
    print(f"Resolved device: {device}")

    # Effective SSL config must be resolved before any output mutation; the CLI
    # usually supplies it, and a direct caller resolves it here exactly once.
    # It validates the payload this invocation already loaded, never a re-read.
    ssl_config = _effective_pretrain_ssl_config(
        args,
        handoff,
        resume_ckpt,
        relational_queue_preparation=preparation,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path: Path | None = None
    resume_register_mode = "new"
    actual_register_tokens = 0
    start_epoch = 0
    global_step = 0
    model_name = settings.model_name
    global_crop_size = settings.global_crop_size
    args.global_crop_size = global_crop_size
    if getattr(args, "resume", ""):
        resume_path = Path(args.resume)
        resume_register_mode, actual_register_tokens = _resume_register_plan(
            resume_ckpt,
            model_name,
            global_crop_size,
            resolve_checkpoint(resume_ckpt, model_name).embedding_dim,
        )
        _check_register_option_against_resume(
            resume_register_mode, actual_register_tokens, register_option
        )

    patch_config = settings.patch_config
    patch_mode = settings.patch_mode
    patch_strategy = settings.masking_strategy
    patch_ratio = patch_config["mask_ratio_resolved"]
    print("Patch config: " + json.dumps(patch_config, sort_keys=True))
    print("SSL config: " + json.dumps(_ssl_display_config(ssl_config), sort_keys=True))

    # One source of truth: the recipe already resolved, validated and (on a
    # resume) drift-checked the ordinary augmentation config.
    augmentation_config = pretrain_recipe.augmentation_config
    profile = str(augmentation_config["profile"])
    policy = str(augmentation_config["orientation_policy"])
    args.local_crop_size = settings.local_crop_size
    args.local_crops = settings.local_crops
    args.augmentation = profile
    args.orientation_policy = policy
    print(f"Augmentation profile: {profile}")
    print(
        "Augmentation config: "
        + json.dumps(augmentation_config, indent=2, sort_keys=True)
    )

    relational_views = str(
        pretrain_recipe.relational_target_config["relational_teacher_views"]
    )
    relational_center_source = str(
        pretrain_recipe.relational_target_config["relational_center_source"]
    )
    relational_pairing = str(
        pretrain_recipe.relational_target_config["relational_pairing"]
    )
    for warning in pretrain_recipe.warnings:
        print(f"[Warning] {warning}")
    print(
        "Relational target config: "
        + json.dumps(pretrain_recipe.relational_target_config, sort_keys=True)
    )
    # Display-only: requested, effective and inactive values stay separate from
    # the raw ``Parameters:`` request record.
    print(
        "Relational queue config: "
        + json.dumps(preparation.resolved_log(), sort_keys=True)
    )
    print(
        "Teacher temperature schedule: "
        + json.dumps(
            {
                "start": pretrain_recipe.relational_target_config[
                    "teacher_temp_start"
                ],
                "end": pretrain_recipe.relational_target_config["teacher_temp_end"],
                "warmup_ratio": pretrain_recipe.teacher_temp_warmup_ratio,
            },
            sort_keys=True,
        )
    )
    # The schedules and the saved checkpoint read effective values, while the
    # CLI already logged the raw requests.
    args.relational_teacher_views = relational_views
    args.relational_pairing = relational_pairing
    args.relational_center_source = relational_center_source
    args.teacher_temp_start = pretrain_recipe.relational_target_config[
        "teacher_temp_start"
    ]
    args.teacher_temp_end = pretrain_recipe.relational_target_config[
        "teacher_temp_end"
    ]
    args.teacher_temp_warmup_ratio = pretrain_recipe.teacher_temp_warmup_ratio
    if relational_views not in RELATIONAL_TEACHER_VIEW_CHOICES:
        raise ValueError(
            "Resolved relational_teacher_views must be one of: "
            f"{', '.join(RELATIONAL_TEACHER_VIEW_CHOICES)}; got "
            f"{relational_views!r}."
        )
    # The weak teacher views are relational-only: DINO keeps two global views.
    include_weak_teacher_views = (
        ssl_config["ssl_method"] == "relational" and relational_views == "weak"
    )
    phase = _phase_start("dataset indexing and loader")
    # Path/CSV resolution happens inside the dataset constructor, so the phase
    # must start before it or its cost is attributed to nothing.
    ds = MultiCropDataset(
        csv_path=Path(args.train_data) if getattr(args, "train_data", "") else None,
        images_dir=Path(args.input_images_dir),
        global_crop_size=global_crop_size,
        local_crop_size=args.local_crop_size,
        local_crops=args.local_crops,
        augmentation_profile=profile,
        orientation_policy=policy,
        include_weak_teacher_views=include_weak_teacher_views,
        augmentation_config=pretrain_recipe.augmentation_config,
        weak_teacher_augmentation_config=(
            pretrain_recipe.weak_teacher_augmentation_config
            if include_weak_teacher_views
            else None
        ),
        prepared_relational_data=(
            preparation.data if preparation.active else None
        ),
    )
    loader = DataLoader(
        ds,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    _phase_done("dataset indexing and loader", phase)

    ssl_method = str(ssl_config["ssl_method"])
    effective_out_dim = int(ssl_config["out_dim"])
    if effective_out_dim != int(args.out_dim):
        print(
            f"[Info] Effective SSL out_dim: {effective_out_dim} "
            f"(requested {args.out_dim} is inherited from the checkpoint)."
        )
    student_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "out_dim": effective_out_dim,
        "return_patch_tokens": True,
        "img_size": global_crop_size,
    }
    if resume_register_mode == "added":
        # Registered resume rebuilds without a pretrained download and loads the
        # saved registers instead of migrating from timm weights.
        student_kwargs.update(pretrained=False, reg_tokens=4)
    elif resume_register_mode == "native":
        student_kwargs.update(pretrained=False)
    elif register_option is not None:
        # ``0`` must stay distinct from omission: only the explicit option
        # rejects a CLS-free backbone.
        student_kwargs["reg_tokens"] = register_option

    phase = _phase_start("student construction")
    student = OTUFormerEncoder(**student_kwargs).to(device)
    _phase_done("student construction", phase)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    _set_seed(args.seed + 1)
    phase = _phase_start("teacher construction")
    teacher = OTUFormerEncoder(**student_kwargs).to(device)
    _phase_done("teacher construction", phase)
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    if resume_ckpt is None:
        # A fresh run starts the EMA pair from the same encoder state. Resumes
        # keep their saved, deliberately distinct student/teacher pair.
        with torch.no_grad():
            teacher.load_state_dict(student.state_dict(), strict=True)
    if resume_ckpt is None:
        actual_register_tokens = native_register_count(student.backbone)

    for p in teacher.parameters():
        p.requires_grad = False

    # The objective belongs to the same preparation block as the optimizer and
    # recovery restore; time it there instead of leaving it unattributed.
    phase = _phase_start("objective, optimizer and recovery")
    objective = PatchObjective(
        patch_mode,
        hidden_dim=student.backbone.num_features,
        ibot_prototypes=patch_config["ibot_prototypes"],
    ).to(device)
    dino_objective: DINOObjective | None = None
    # An independent regularizer for both methods, not a DINO feature.
    koleo_enabled = float(ssl_config["koleo_weight"]) > 0.0
    if ssl_method == "dino":
        dino_objective = DINOObjective(
            effective_out_dim,
            int(ssl_config["dino_prototypes"]),
            str(ssl_config["dino_centering"]),
        ).to(device)
        # The resolver already probed before output setup; also check the real
        # device layers before restoring state or entering the training loop.
        validate_dino_prototype_layer(dino_objective.student_prototypes)
        validate_dino_prototype_layer(dino_objective.teacher_prototypes)
        if resume_ckpt is None:
            # Fresh DINO only: reference head init, then teacher synchronization.
            _initialize_dino_student_projector(student)
            _synchronize_dino_teacher(student, teacher, dino_objective)
        if int(args.max_epochs) <= 1 and resume_ckpt is None:
            print(
                "[Warning] --max-epochs 1 keeps the DINO prototype direction "
                "frozen for the whole run (the freeze covers absolute epoch 0 "
                "only). This is valid for a smoke run, not for real training."
            )
    # The training loop always uses forward_pretrain (even for patch_loss=none),
    # so every mode validates the backbone before the loop starts.
    require_intermediates = patch_mode == "masked-feature"
    student.validate_pretrain_backbone(require_intermediates=require_intermediates)
    teacher.validate_pretrain_backbone(require_intermediates=require_intermediates)
    grid_size = student._patch_grid_size(
        torch.empty(1, 3, global_crop_size, global_crop_size)
    )
    patch_count = grid_size[0] * grid_size[1]
    if patch_mode != "none":
        if patch_strategy in ("blockwise", "hybrid"):
            _validate_blockwise_grid(grid_size)
        if patch_count < 2:
            raise ValueError(
                f"patch grid {grid_size} has fewer than two patch tokens; the "
                "selected patch objective needs at least one masked and one "
                "visible position. Increase the input size or use "
                "--patch-loss none."
            )

    wd_policy = bool(ssl_config["norm_bias_weight_decay"])
    has_decay_partition = wd_policy or not _pretrain_decay_partition_empty(
        student, objective, dino_objective
    )
    optimizer = torch.optim.AdamW(
        _pretrain_optimizer_groups(
            student,
            objective,
            dino_objective,
            weight_decay=args.weight_decay,
            norm_bias_weight_decay=wd_policy,
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    instant_logger = InstantMetricsLogger(
        logs_dir / "instant_metrics.pretrain.csv", mode="pretrain"
    )
    compute_embedding_metrics = bool(getattr(args, "compute_embedding_metrics", True))
    epoch_logger = (
        EnhancedMetricsLogger(logs_dir / "metrics.pretrain.csv", mode="pretrain")
        if compute_embedding_metrics
        else None
    )

    if resume_ckpt is not None:
        if preparation.active:
            # Enabled recovery loads the states the preparation already validated,
            # cast on the CPU so a float64 payload cannot reach the device.
            student.load_state_dict(
                _fifo_cast_state(student, resume_ckpt["student"]), strict=True
            )
            teacher.load_state_dict(
                _fifo_cast_state(teacher, resume_ckpt["teacher"]), strict=True
            )
        elif dino_objective is not None:
            # DINO never leaves random state behind: the preflight already
            # validated complete encoder states, so this load is strict.
            student.load_state_dict(resume_ckpt["student"], strict=True)
            teacher.load_state_dict(resume_ckpt["teacher"], strict=True)
        elif "student" in resume_ckpt:
            student.load_state_dict(resume_ckpt["student"], strict=False)
        elif "model_state_dict" in resume_ckpt:
            student.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        if dino_objective is None and "teacher" in resume_ckpt:
            teacher.load_state_dict(resume_ckpt["teacher"], strict=False)
        saved_patch_state = resume_ckpt.get("patch_objective")
        if patch_mode in ("masked-feature", "ibot") and (
            not isinstance(saved_patch_state, dict) or not saved_patch_state
        ):
            raise ValueError(
                "Cannot resume: this checkpoint has no patch_objective state for "
                f"patch_loss={patch_mode!r}. Start a new run to use real masking."
            )
        if isinstance(saved_patch_state, dict) and saved_patch_state:
            try:
                objective.load_state_dict(
                    _fifo_cast_state(objective, saved_patch_state), strict=True
                )
            except RuntimeError as exc:
                raise ValueError(
                    "Cannot resume: patch_objective state is malformed or "
                    f"incompatible with patch_loss={patch_mode!r}: {exc}"
                ) from exc
        if dino_objective is not None:
            try:
                dino_objective.load_state_dict(
                    resume_ckpt["dino_objective"], strict=True
                )
            except (KeyError, RuntimeError) as exc:
                raise ValueError(
                    "Cannot resume: dino_objective state is missing or "
                    f"incompatible with the resolved DINO config: {exc}"
                ) from exc
        if "optimizer" in resume_ckpt:
            _validate_pretrain_optimizer_layout(
                optimizer,
                resume_ckpt["optimizer"],
                norm_bias_weight_decay=wd_policy,
                decay_partition_empty=not has_decay_partition,
                label="pretrain resume",
            )
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            _report_restored_weight_decay(
                optimizer,
                requested_weight_decay=float(args.weight_decay),
                norm_bias_weight_decay=wd_policy,
                has_decay_partition=has_decay_partition,
                label="pretrain resume",
            )
        if "center" in resume_ckpt:
            _restore_relational_center(
                resume_ckpt["center"], teacher.center, device
            )
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        global_step = int(resume_ckpt.get("iteration", start_epoch * max(1, len(loader))))
        print(
            f"[Info] Resume from {resume_path} at epoch {start_epoch}, iteration {global_step}"
        )
    queue: RelationalQueue | None = None
    if preparation.active:
        data = preparation.data
        assert data is not None  # an enabled preparation always prepares data
        if resume_ckpt is None:
            queue = RelationalQueue(
                int(preparation.capacity), effective_out_dim, device=device
            )
        else:
            # Reconstruct exactly once, on the resolved device, from the state
            # the preflight already validated against this prepared selection.
            queue = RelationalQueue.from_state_dict(
                preparation.resume_queue_state,
                capacity=int(preparation.capacity),
                out_dim=effective_out_dim,
                completed_steps=int(resume_ckpt["iteration"]),
                allowed_keys=frozenset(data.image_keys),
                data_fingerprint=data.data_fingerprint,
                data_image_count=len(data.image_keys),
                data_size_bytes=data.data_size_bytes,
                device=device,
            )
            if queue.completed_steps != global_step:
                raise ValueError(
                    "The restored relational queue counts "
                    f"{queue.completed_steps} completed step(s) but the "
                    f"checkpoint resumes at iteration {global_step}."
                )
    _phase_done("objective, optimizer and recovery", phase)

    niter_per_epoch = max(1, len(loader))
    lr_schedule, momentum_schedule, temp_student_schedule, teacher_temp_schedule, schedule_state = _build_pretrain_schedules(
        args,
        resume_ckpt.get("schedule") if resume_ckpt is not None else None,
        steps_per_epoch=niter_per_epoch,
        global_step=global_step,
        completed_epochs=start_epoch,
        original_max_epochs=(
            int(resume_ckpt["args"]["max_epochs"])
            if resume_ckpt is not None
            and isinstance(resume_ckpt.get("args"), dict)
            and "max_epochs" in resume_ckpt["args"]
            else None
        ),
        teacher_temp_warmup_ratio=float(
            ssl_config["teacher_temp_warmup_ratio"]
        ),
    )
    total_steps = len(lr_schedule)

    cosine_sim = 0.0
    eval_image_size = (
        int(args.extract_size)
        if getattr(args, "extract_size", None) is not None
        else global_crop_size
    )
    validate_input_size(eval_image_size, teacher, model_name)
    if args.extract_size is None:
        print(f"[Info] Auto eval crop size from model img_size: {eval_image_size}")

    with torch.no_grad():
        phase = _phase_start("first batch and initial forwards")
        sample_batch = next(iter(loader))
        # The startup sample never populates the queue: it is a compatibility
        # probe, not a successful optimization step.
        sample_views, _sample_keys = _unpack_pretrain_batch(
            sample_batch, queue_enabled=queue is not None
        )
        sample_views = [v.to(device) for v in sample_views[:2]]
        s_out, _ = student(sample_views[0])
        t_out, _ = teacher(sample_views[0])
        _phase_done("first batch and initial forwards", phase)
        init_cosine = F.cosine_similarity(
            s_out.mean(dim=0, keepdim=True),
            t_out.mean(dim=0, keepdim=True),
        ).item()
        print(
            "[Info] Initial ordinary-global cosine similarity: "
            f"{init_cosine:.4f}"
        )
        print(
            f"[Info] Starting/Resuming with learning rate: {float(lr_schedule[min(global_step, total_steps - 1)]):.6f}"
        )

    if queue is not None:
        # Never bypass the validated copy: restoring the raw payload again could
        # apply streams the target-aware validation explicitly excluded.
        pending_rng_restore = preparation.rng_state_to_restore
    else:
        pending_rng_restore = (
            resume_ckpt.get("rng_state") if resume_ckpt is not None else None
        )
    if queue is None and resume_ckpt is not None and not pending_rng_restore:
        print(
            "[Warning] Legacy checkpoint has no RNG state; the random sequence "
            "cannot continue exactly. Mask sampling and the main-process shuffle "
            "resume best-effort."
        )

    for epoch in range(start_epoch, args.max_epochs):
        # Restore the resumed RNG exactly once. Restoring every epoch would
        # restart every non-saving epoch from the last checkpoint's state and
        # repeat the mask and shuffle sequence.
        if pending_rng_restore is not None:
            _restore_rng_state(pending_rng_restore)
            pending_rng_restore = None
        cosine_sim = 0.0
        student.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.max_epochs}", ncols=150)
        for step, batch in enumerate(pbar):
            views, batch_keys = _unpack_pretrain_batch(
                batch, queue_enabled=queue is not None
            )
            views = [v.to(device) for v in views]
            # One immutable candidate snapshot per step, taken before any model
            # work: every global/local term of this iteration shares it.
            queue_occupancy_before = 0 if queue is None else len(queue)
            queue_snapshot = (
                queue.snapshot(batch_keys, step=global_step)
                if queue is not None
                else None
            )
            # DINO keeps the two global views only; the weak relational views
            # are relational-only.
            expected_views = (
                (4 if include_weak_teacher_views else 2) + int(args.local_crops)
            )
            if len(views) != expected_views:
                raise ValueError(
                    f"the loader produced {len(views)} views but the resolved "
                    f"recipe needs {expected_views} "
                    f"(relational_teacher_views={relational_views!r}, "
                    f"local_crops={args.local_crops})."
                )
            global_views = views[:2]
            local_views = views[2 : 2 + int(args.local_crops)]
            weak_views = views[2 + int(args.local_crops) : 4 + int(args.local_crops)]

            iter_idx = global_step
            for group in optimizer.param_groups:
                group["lr"] = float(lr_schedule[iter_idx])

            student_global: list[torch.Tensor] = []
            student_tokens: list[torch.Tensor] = []
            for gv in global_views:
                s_out = student.forward_pretrain(gv)
                student_global.append(s_out.cls)
                student_tokens.append(s_out.tokens)

            teacher_tokens: list[torch.Tensor] = []
            teacher_final_four: list[list[torch.Tensor] | None] = []
            teacher_logits: list[torch.Tensor] = []
            prefix = student.backbone.num_prefix_tokens
            center_before = teacher.center
            # Detached pre-center A/B teacher observations, cached for a
            # success-only queue commit; nothing is stored before the step wins.
            observation_pair: torch.Tensor | None = None
            with torch.no_grad():
                # Ordinary forwards serve the patch objective, the ordinary
                # center population and the ordinary-A diagnostic cosine.
                need_ordinary_both = (
                    dino_objective is not None
                    or patch_mode != "none"
                    or relational_center_source == "ordinary"
                )
                ordinary_outputs = [
                    teacher.forward_pretrain(
                        gv,
                        return_intermediates=patch_mode == "masked-feature",
                    )
                    for gv in global_views[: 2 if need_ordinary_both else 1]
                ]
                if dino_objective is None:
                    if relational_views == "weak":
                        relational_outputs = [
                            teacher.forward_pretrain(wv) for wv in weak_views
                        ]
                    else:
                        relational_outputs = ordinary_outputs

                    def _centered(
                        outputs: list[Any],
                    ) -> list[torch.Tensor]:
                        return [
                            F.normalize(out.cls - center_before, dim=-1)
                            for out in outputs
                        ]

                    relational_teacher = _centered(relational_outputs)
                    ordinary_teacher = _centered(ordinary_outputs)
                    center_inputs = (
                        ordinary_teacher
                        if relational_center_source == "ordinary"
                        else relational_teacher
                    )
                    teacher.center = _update_teacher_center(
                        center_before, torch.cat(center_inputs, dim=0)
                    )
                    teacher_global = relational_teacher
                    if queue is not None:
                        observation_pair = torch.stack(
                            [out.cls.detach() for out in relational_outputs], dim=1
                        )
                else:
                    teacher_global = []
                for t_out in ordinary_outputs:
                    if dino_objective is not None:
                        # DINO sends the unmodified normalized bottleneck to its
                        # prototype layer and bypasses the legacy center path.
                        teacher_global.append(t_out.cls)
                    teacher_tokens.append(t_out.tokens)
                    teacher_final_four.append(t_out.final_four)
                    if patch_mode == "ibot":
                        teacher_logits.append(
                            objective.teacher_ibot_head(t_out.tokens[:, prefix:])
                        )

            temp_student = float(temp_student_schedule[iter_idx])
            temp_teacher = float(teacher_temp_schedule[iter_idx])

            # Rectangular current-plus-history candidates and their teacher
            # targets are built once per step from the same snapshot and the
            # *pre-step* center. Both diagnostic families reduce only on log
            # steps, so an ordinary step pays no extra statistics.
            diagnostics_step = global_step % max(1, args.log_every_n_steps) == 0
            relational_targets = None
            if queue_snapshot is not None:
                relational_targets = _make_relational_step_targets(
                    teacher_global,
                    queue_snapshot.features,
                    center_before,
                    temp_teacher,
                    collect_diagnostics=diagnostics_step,
                )
                if observation_pair is None:  # pragma: no cover - defensive
                    raise ValueError(
                        "The enabled relational queue needs the current teacher "
                        "observations to commit after a successful step."
                    )

            dino_teacher_logits: list[torch.Tensor] = []
            dino_teacher_targets: list[torch.Tensor] = []
            student_stats: list[dict[str, torch.Tensor]] | None = (
                [] if relational_targets is not None and diagnostics_step else None
            )
            if dino_objective is None:
                loss_global = _compute_global_loss(
                    student_global,
                    teacher_global,
                    temp_student,
                    temp_teacher,
                    relational_pairing=relational_pairing,
                    relational_targets=relational_targets,
                    student_stats=student_stats,
                )
            else:
                with torch.no_grad():
                    dino_teacher_logits = [
                        dino_objective.teacher_logits(t) for t in teacher_global
                    ]
                    # Built once from the cached pre-EMA logits and reused for
                    # every global and local pair in this iteration.
                    dino_teacher_targets = dino_objective.teacher_targets(
                        dino_teacher_logits, temp_teacher
                    )
                loss_global = _compute_dino_global_loss(
                    [dino_objective(z) for z in student_global],
                    dino_teacher_targets,
                    temp_student,
                )

            patch_loss_raw = torch.zeros((), device=device)
            center_logits: list[torch.Tensor] = []
            step_masks: list[torch.Tensor] = []
            ibot_probs: list[torch.Tensor] = []
            if patch_mode == "consistency":
                view_losses = []
                for s_tok, t_tok in zip(student_tokens, teacher_tokens):
                    mask = _sample_patch_masks(
                        s_tok.shape[0], grid_size, patch_ratio, "random", device
                    )
                    step_masks.append(mask)
                    view_losses.append(
                        masked_patch_cosine_loss(
                            s_tok[:, prefix:], t_tok[:, prefix:], mask
                        )
                    )
                patch_loss_raw = torch.stack(view_losses).mean()
            elif patch_mode == "masked-feature":
                final_norm = teacher.pretrain_final_norm
                view_losses = []
                for index, gv in enumerate(global_views):
                    mask = _sample_patch_masks(
                        gv.shape[0], grid_size, patch_ratio, patch_strategy, device
                    )
                    step_masks.append(mask)
                    with torch.no_grad():
                        layers = [
                            final_norm(block)[:, prefix:]
                            for block in teacher_final_four[index]
                        ]
                        target = F.normalize(
                            torch.stack(layers).mean(dim=0), dim=-1
                        )
                    masked_out = student.forward_pretrain(
                        gv, mask=mask, mask_token=objective.mask_token
                    )
                    prediction = objective.predict_masked_features(
                        masked_out.tokens[:, prefix:]
                    )
                    view_losses.append(
                        masked_patch_cosine_loss(prediction, target, mask)
                    )
                patch_loss_raw = torch.stack(view_losses).mean()
            elif patch_mode == "ibot":
                view_losses = []
                for index, gv in enumerate(global_views):
                    mask = _sample_patch_masks(
                        gv.shape[0], grid_size, patch_ratio, patch_strategy, device
                    )
                    step_masks.append(mask)
                    masked_out = student.forward_pretrain(
                        gv, mask=mask, mask_token=objective.mask_token
                    )
                    view_loss, view_probs = ibot_patch_loss(
                        objective.student_ibot_head(
                            masked_out.tokens[:, prefix:]
                        ),
                        teacher_logits[index],
                        mask,
                        objective.patch_center,
                        temp_student,
                        temp_teacher,
                    )
                    view_losses.append(view_loss)
                    ibot_probs.append(view_probs)
                    center_logits.append(teacher_logits[index][mask].detach())
                patch_loss_raw = torch.stack(view_losses).mean()

            loss_mask = patch_loss_raw
            patch_loss_weighted = args.lambda_mask * patch_loss_raw

            local_student_projs: list[torch.Tensor] = []
            for lv in local_views:
                l_proj, _ = student(lv)
                local_student_projs.append(l_proj)
            if dino_objective is None:
                loss_local = _compute_local_loss(
                    local_student_projs,
                    teacher_global,
                    temp_student,
                    temp_teacher,
                    relational_targets=relational_targets,
                    student_stats=student_stats,
                )
            else:
                loss_local = _compute_dino_local_loss(
                    [dino_objective(z) for z in local_student_projs],
                    dino_teacher_targets,
                    temp_student,
                )

            koleo_raw = torch.zeros((), device=device)
            koleo_pool = [0, 0]
            if koleo_enabled:
                koleo_raw, koleo_pool = _compute_koleo_raw(student_tokens, ssl_config)

            total_loss = (
                loss_global
                + args.lambda_local * loss_local
                + patch_loss_weighted
            )
            if koleo_enabled:
                total_loss = (
                    total_loss + float(ssl_config["koleo_weight"]) * koleo_raw
                )

            if not torch.isfinite(total_loss):
                raise ValueError(
                    "The pretraining loss became non-finite at iteration "
                    f"{global_step}; refusing to update the parameters. Check the "
                    "temperatures, learning rate and inputs."
                )

            optimizer.zero_grad()
            total_loss.backward()
            trainable_params = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            if dino_objective is not None and epoch == 0:
                # Absolute zero-based epoch 0 freezes the prototype direction.
                # Zero gradients are not enough: AdamW would still decay it.
                for parameter in dino_objective.trainable_parameters():
                    parameter.grad = None
            grad_norm = _compute_grad_norm_parameters(trainable_params)
            if not math.isfinite(grad_norm):
                raise ValueError(
                    "The pretraining gradient norm became non-finite at iteration "
                    f"{global_step}; refusing to update the parameters. A finite "
                    "loss is not enough: the backward pass can still overflow."
                )
            # ``error_if_nonfinite`` repeats the reduction inside the clip, which
            # is not the reduction checked above: a total that overflows there
            # would otherwise zero every gradient and still advance the step.
            try:
                nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=3.0, error_if_nonfinite=True
                )
            except RuntimeError as exc:
                raise ValueError(
                    "The pretraining gradient norm became non-finite at iteration "
                    f"{global_step}; refusing to update the parameters. The "
                    "clipping reduction can overflow where the per-parameter "
                    "check does not."
                ) from exc
            optimizer.step()
            if queue is not None and not bool(torch.stack([
                torch.isfinite(parameter).all() for parameter in trainable_params
            ]).all()):
                raise ValueError(
                    "The pretraining optimizer update became non-finite at "
                    f"iteration {global_step}; refusing to advance the FIFO or "
                    "save a checkpoint. Check the optimizer coefficients."
                )

            m = float(momentum_schedule[iter_idx])
            update_teacher(student, teacher, m)
            if dino_objective is not None:
                # Name-matched EMA keeps both prototype layers aligned; the
                # fixed scales stay one and are absent from the optimizer.
                update_teacher(
                    dino_objective.student_prototypes,
                    dino_objective.teacher_prototypes,
                    m,
                )
                # EMA centering only, once per completed step, from the cached
                # pre-EMA logits used to build this step's targets.
                dino_objective.update_center(dino_teacher_logits)
            if patch_mode == "ibot":
                update_teacher(
                    objective.student_ibot_head, objective.teacher_ibot_head, m
                )
                objective.patch_center = _update_patch_center(
                    objective.patch_center,
                    torch.cat(center_logits, dim=0),
                )

            queue_commit: dict[str, int] | None = None
            if queue is not None:
                # Success boundary: every finite check, the backward pass,
                # clipping, the optimizer step and the EMA/patch-center updates
                # have already succeeded.
                queue_commit = queue.commit(batch_keys, observation_pair, step=global_step)

            if dino_objective is None:
                with torch.no_grad():
                    cosine_sim = F.cosine_similarity(
                        student_global[0].mean(dim=0, keepdim=True),
                        ordinary_teacher[0].mean(dim=0, keepdim=True),
                    ).item()
            else:
                with torch.no_grad():
                    cosine_sim = F.cosine_similarity(
                        student_global[0].mean(dim=0, keepdim=True),
                        teacher_global[0].mean(dim=0, keepdim=True),
                    ).item()

            if global_step % max(1, args.log_every_n_steps) == 0:
                with torch.no_grad():
                    instant_metrics = _compute_instant_metrics(
                        student, global_views[0], device
                    )
                total_value = float(total_loss.item())
                weighted_value = float(patch_loss_weighted.item())
                ssl_metrics: dict[str, object] = {
                    "ssl_method": ssl_method,
                    "dino_centering": (
                        str(ssl_config["dino_centering"])
                        if dino_objective is not None
                        else None
                    ),
                    # Shared setting: recorded numerically for both methods,
                    # including zero, so a default relational row states the
                    # regularizer is off instead of leaving it unrecorded.
                    "koleo_weight": float(ssl_config["koleo_weight"]),
                    "global_loss_weighted": float(loss_global.item()),
                    "local_loss_weighted": float(
                        args.lambda_local * loss_local.item()
                    ),
                }
                # KoLeo diagnostics are method-independent; only the DINO
                # prototype/center fields are relational's N/A.
                ssl_metrics.update(
                    _koleo_instant_metrics(
                        ssl_config, koleo_enabled, koleo_raw, koleo_pool
                    )
                )
                if dino_objective is None:
                    # Relational has no K-dimensional CLS center.
                    ssl_metrics["cls_center_norm"] = None
                else:
                    center = dino_objective.cls_center
                    ssl_metrics["cls_center_norm"] = (
                        float(center.norm().item()) if center is not None else None
                    )
                    ssl_metrics.update(
                        _dino_cls_diagnostics(
                            dino_teacher_targets, dino_objective.prototypes
                        )
                    )
                patch_metrics: dict[str, object] = {
                    "patch_loss_raw": float(patch_loss_raw.item()),
                    "patch_loss_weighted": weighted_value,
                    # A signed KoLeo contribution has no ordinary positive-loss
                    # fraction, so the column is retired for the whole enabled run
                    # (including batches that skip it).
                    "patch_loss_fraction_of_total": (
                        None
                        if koleo_enabled
                        else (
                            weighted_value / total_value
                            if total_value != 0.0
                            else 0.0
                        )
                    ),
                    "requested_mask_ratio": patch_config["mask_ratio_requested"],
                }
                if patch_mode == "consistency" and step_masks:
                    patch_metrics["selected_patch_ratio"] = float(
                        torch.stack(
                            [mask.float().mean() for mask in step_masks]
                        ).mean()
                    )
                if patch_mode in ("masked-feature", "ibot") and step_masks:
                    patch_metrics.update(
                        _mask_ratio_diagnostics(step_masks, patch_count)
                    )
                if patch_mode == "ibot" and ibot_probs:
                    patch_metrics.update(
                        _ibot_diagnostics(
                            torch.cat(ibot_probs, dim=0), objective.patch_center
                        )
                    )
                # Coverage/age describe the pre-step snapshot; the refresh and
                # eviction counts are this step's successful commit. Queue-off
                # and DINO rows leave every cell blank.
                queue_metrics: dict[str, object] = {}
                if queue is not None:
                    snapshot_keys = (
                        () if queue_snapshot is None else queue_snapshot.keys
                    )
                    queue_metrics = _relational_queue_metrics(
                        capacity=queue.capacity,
                        occupancy=queue_occupancy_before,
                        eligible=len(snapshot_keys),
                        candidate_count=len(snapshot_keys) + len(batch_keys or ()),
                        ages=(
                            ()
                            if queue_snapshot is None
                            else queue_snapshot.ages_at(global_step)
                        ),
                        refreshed=(
                            None if queue_commit is None else queue_commit["refreshed"]
                        ),
                        evicted=(
                            None if queue_commit is None else queue_commit["evicted"]
                        ),
                        teacher_stats=(
                            None
                            if relational_targets is None
                            else relational_targets.teacher_stats
                        ),
                        student_stats=student_stats,
                        has_local=bool(local_views),
                    )
                instant_logger.log(
                    iteration=global_step,
                    epoch=epoch,
                    step=step,
                    loss=float(total_loss.item()),
                    global_loss=float(loss_global.item()),
                    local_loss=float(loss_local.item()),
                    mask_loss=float(loss_mask.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    teacher_temp=float(temp_teacher),
                    grad_norm=float(grad_norm),
                    teacher_center_norm=(
                        None
                        if dino_objective is not None
                        else float(teacher.center.norm().item())
                    ),
                    cosine_similarity=(
                        None if dino_objective is not None else float(cosine_sim)
                    ),
                    **instant_metrics,
                    **patch_metrics,
                    **ssl_metrics,
                    **queue_metrics,
                )

            if step % 10 == 0 or step == len(loader) - 1:
                cos_display = (
                    "N/A"
                    if dino_objective is not None
                    else (
                        f"{cosine_sim:.3e}"
                        if abs(cosine_sim) < 1e-3
                        else f"{cosine_sim:.4f}"
                    )
                )
                pbar.set_postfix(
                    L=f"{total_loss.item():.2e}",
                    GL=f"{loss_global.item():.2e}",
                    LocL=f"{loss_local.item():.2e}",
                    ML=f"{loss_mask.item():.2e}",
                    COS=cos_display,
                    LR=f"{optimizer.param_groups[0]['lr']:.2e}",
                    MOM=f"{m:.4f}",
                )

            global_step += 1

        should_save = (epoch + 1) % max(1, args.save_every_epochs) == 0 or (
            epoch + 1
        ) == args.max_epochs
        if should_save and compute_embedding_metrics and epoch_logger is not None:
            # Epoch-end evaluation and UMAP must run before the RNG snapshot so
            # the saved state is the boundary after all logging.
            _compute_and_log_all_metrics(
                args=args,
                model=teacher,
                device=device,
                epoch=epoch,
                logs_dir=logs_dir,
                metrics_logger=epoch_logger,
                eval_image_size=eval_image_size,
                force_linear_probe=False,
            )
        if should_save:
            # Relational's canonical zero-weight encoding is absence: a disabled
            # regularizer must not add the four previously DINO-only keys, so a
            # zero-weight relational checkpoint stays readable by older code.
            # DINO keeps recording them at any weight.
            saved_ssl_config = dict(ssl_config)
            if ssl_method != "dino" and not koleo_enabled:
                for name in KOLEO_STORED_KEYS:
                    saved_ssl_config.pop(name, None)
            ckpt = {
                "epoch": epoch,
                "iteration": global_step,
                "model_state_dict": teacher.state_dict(),
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "patch_objective": objective.state_dict(),
                "rng_state": _capture_rng_state(),
                "args": {
                    key: value
                    for key, value in vars(args).items()
                    if key not in _UNSAVED_ARGS
                    and key not in _TRANSIENT_ARGS
                },
                "config": {
                    # Effective SSL recipe: relational records its complete
                    # identity pair and omits every DINO-exclusive field;
                    # ``args`` keeps the requested invocation values.
                    **saved_ssl_config,
                    "model_name": model_name,
                    "image_size": global_crop_size,
                    "register_tokens": actual_register_tokens,
                    "augmentation_profile": profile,
                    "augmentation_config": augmentation_config,
                    **patch_config,
                    RELATIONAL_TARGET_CONFIG_KEY: (
                        pretrain_recipe.relational_target_config
                    ),
                },
                "schedule": {
                    **schedule_state,
                    "last_lr": float(optimizer.param_groups[0]["lr"]),
                },
            }
            if dino_objective is None:
                # Relational keeps its historical top-level encoder center and
                # owns no prototype state.
                ckpt["center"] = teacher.center.detach().cpu()
            else:
                # DINO owns the K-dimensional center inside dino_objective and
                # writes no top-level center.
                ckpt["dino_objective"] = dino_objective.state_dict()
            if queue is not None:
                # Enabled runs persist the closed config bundle and the logical
                # queue state at the existing epoch boundary. A disabled run
                # omits both rather than writing a fake enabled marker.
                data = preparation.data
                assert data is not None  # guaranteed by an active preparation
                ckpt["config"][RELATIONAL_QUEUE_CONFIG_KEY] = queue_config_bundle(
                    queue.capacity
                )
                ckpt[RELATIONAL_QUEUE_STATE_KEY] = queue.state_dict(
                    data_fingerprint=data.data_fingerprint,
                    data_image_count=len(data.image_keys),
                    data_size_bytes=data.data_size_bytes,
                )
            ckpt_path = out_dir / f"SSL_epoch_{epoch + 1:04d}.pth"
            save_checkpoint(ckpt, ckpt_path)
            shutil.copy(str(ckpt_path), out_dir / "SSL_latest.pth")
            print(f"[Info] Saved checkpoint {ckpt_path}")

            keep = max(0, args.keep_last_checkpoints)
            if keep > 0:
                ckpts = sorted(out_dir.glob("SSL_epoch_*.pth"))
                if len(ckpts) > keep:
                    for old_ckpt in ckpts[:-keep]:
                        old_ckpt.unlink(missing_ok=True)
                        print(f"[Info] Deleted old checkpoint {old_ckpt.name}")

    instant_logger.plot(
        logs_dir, epoch_logger.path if epoch_logger is not None else None
    )
    print(f"[Info] SSL pretraining complete. Logs: {logs_dir}")

def _freeze_backbone_blocks(model: OTUFormerEncoder, freeze_ratio: float) -> None:
    blocks = getattr(model.backbone, "blocks", None)
    if blocks is None:
        return
    n_blocks = len(blocks)
    n_freeze = int(math.floor(n_blocks * max(0.0, min(1.0, freeze_ratio))))
    for i, block in enumerate(blocks):
        requires_grad = i >= n_freeze
        for p in block.parameters():
            p.requires_grad = requires_grad
