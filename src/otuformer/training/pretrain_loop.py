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
from otuformer.training.dataset import collate_pretrain_geometry
from otuformer.training.dense_recipe import (
    CONFIG_KEY,
    STATE_KEY,
    dense_config_bundle,
    dense_state_bundle,
    new_dense_runtime,
)
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
    ResolvedTrainingImages,
)
from .dino import (
    DINOObjective,
    validate_dino_prototype_layer,
)
from .loss import (
    ibot_selected_patch_loss,
    masked_patch_cosine_loss,
    selected_patch_cosine_loss,
    _validate_patch_mask,
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
    _verify_restorable_rng_state,
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
    prepare_evaluation_manifest,
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


def _to_feature_grid(
    tokens: torch.Tensor, prefix: int, grid_size: tuple[int, int]
) -> torch.Tensor:
    """Reshape final-normalized patch tokens to BHWC using verified metadata."""
    if prefix < 0:
        raise ValueError(f"prefix must be non-negative, got {prefix}.")
    patches = tokens[:, int(prefix) :, :]
    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    if int(patches.shape[1]) != grid_h * grid_w:
        raise ValueError(
            f"{int(patches.shape[1])} patch token(s) do not fill the resolved grid "
            f"{grid_size}; refusing to infer a grid from the token count."
        )
    return patches.reshape(
        int(patches.shape[0]), grid_h, grid_w, int(patches.shape[-1])
    )


def _dense_status(weight: float, raw: Any) -> str:
    """Distinguish a switched-off objective from an enabled one with no signal.

    Both used to be logged as ``skipped``, so a Coordinate-only run showed a
    cumulative "ranking skipped" series for an objective it never ran.
    """
    if weight <= 0.0:
        return "disabled"
    return "skipped" if raw is None else "observed"


def _dense_stat_or_blank(value: Any) -> float | str:
    """Render one dense statistic; ``None`` means unavailable, not measured zero."""
    return "" if value is None else float(value)


def _dense_terms(
    settings: Any,
    runtime: Any,
    ranking_objective: Any,
    student_grids: dict[int, torch.Tensor],
    teacher_grids: dict[int, torch.Tensor],
    batch_geometry: Any,
    grid_hws: tuple[tuple[int, int], ...],
    *,
    patch_hw: tuple[int, int],
) -> tuple[
    torch.Tensor | None, dict[str, Any], torch.Tensor | None, dict[str, Any]
]:
    """Build the shared pairs once, then evaluate each enabled objective.

    Coordinate and ranking share coordinates but keep separate feature spaces,
    denominators and availability rules.
    """
    from otuformer.training import dense_geometry as _dense_geometry
    from otuformer.training import dense_objective as _dense_objective

    empty = {"contributing_pairs": 0, "selected_points": 0}
    if not student_grids or len(teacher_grids) < 2:
        return None, dict(empty), None, dict(empty)
    pairs = _dense_geometry.build_dense_pairs(
        batch_geometry,
        grid_hws=grid_hws,
        patch_hw=patch_hw,
        local_crops=len(student_grids) - 2,
        query_cap=int(settings.query_cap),
        generator=runtime.query_generator,
    )

    coordinate_raw = None
    coordinate_stats: dict[str, Any] = dict(empty)
    if settings.coordinate_weight > 0.0:
        result = _dense_objective.coordinate_loss(
            student_grids, teacher_grids, pairs, patch_hw=patch_hw
        )
        coordinate_raw = result.raw
        coordinate_stats = dict(result.statistics)
        coordinate_stats["raw"] = (
            None if result.raw is None else float(result.raw.detach())
        )

    ranking_raw = None
    ranking_stats: dict[str, Any] = dict(empty)
    if ranking_objective is not None and settings.ranking_enabled:
        ranking_result = ranking_objective.compute(
            student_grids,
            teacher_grids,
            pairs,
            patch_hw=patch_hw,
            reference_cap=int(settings.reference_cap),
            generator=runtime.reference_generator,
        )
        ranking_raw = ranking_result.raw
        ranking_stats = dict(ranking_result.statistics)
        ranking_stats["raw"] = (
            None if ranking_result.raw is None else float(ranking_result.raw.detach())
        )
        ranking_stats["ranking_contributed"] = ranking_result.raw is not None
    return coordinate_raw, coordinate_stats, ranking_raw, ranking_stats


def _unpack_pretrain_batch(
    batch: object,
    *,
    queue_enabled: bool,
    dense_enabled: bool = False,
    local_crops: int | None = None,
    include_weak_teacher_views: bool | None = None,
) -> tuple[list[torch.Tensor], tuple[str, ...] | None, object | None]:
    """Split one loader batch into views, (enabled only) keys and geometry.

    A queue-off, dense-off run keeps the historical list-of-views contract. Each
    enabled feature adds exactly one documented key and never falls back to a
    silent alternative envelope.
    """
    if dense_enabled:
        return _unpack_dense_batch(
            batch,
            queue_enabled=queue_enabled,
            local_crops=local_crops,
            include_weak_teacher_views=include_weak_teacher_views,
        )
    if not queue_enabled:
        if isinstance(batch, dict) or not isinstance(batch, (list, tuple)):
            raise ValueError(
                "The pretraining loader produced an unexpected batch envelope; a "
                "queue-disabled run reads a list of view tensors."
            )
        return list(batch), None, None
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
    return views, resolved_keys, None


def _unpack_dense_batch(
    batch: object,
    *,
    queue_enabled: bool,
    local_crops: int | None,
    include_weak_teacher_views: bool | None,
) -> tuple[list[torch.Tensor], tuple[str, ...] | None, object | None]:
    """Unpack an enabled dense batch and validate its exact view layout.

    The two resolved layout values are required rather than inferred from
    ``None`` records: a trace with a well-formed but wrongly placed weak slot
    must fail instead of silently shifting every later view index.
    """
    from otuformer.training import dense_geometry as _dense_geometry

    if local_crops is None or include_weak_teacher_views is None:
        raise ValueError(
            "An enabled dense batch requires the resolved local_crops and "
            "include_weak_teacher_views values; weak slots are never inferred "
            "from empty records."
        )
    if not isinstance(batch, dict):
        raise ValueError(
            "An enabled dense objective requires the geometry mapping envelope "
            f"from MultiCropDataset(return_geometry=True); got {type(batch).__name__}."
        )
    expected = {"views", "geometry"}
    if queue_enabled:
        expected.add("image_key")
    if set(batch) != expected:
        raise ValueError(
            f"An enabled dense batch requires exactly {sorted(expected)}, got "
            f"{sorted(batch)}."
        )

    views = list(batch["views"])
    ordinary = 2 + int(local_crops)
    expected_views = ordinary + (2 if include_weak_teacher_views else 0)
    if len(views) != expected_views:
        raise ValueError(
            f"The enabled dense batch has {len(views)} view(s); the resolved layout "
            f"requires {expected_views} (2 globals + {local_crops} locals"
            f"{' + 2 weak' if include_weak_teacher_views else ''})."
        )

    geometry = tuple(batch["geometry"])
    if len(geometry) != len(views):
        raise ValueError(
            f"geometry has {len(geometry)} slot(s) for {len(views)} view(s)."
        )
    decoded_geometry = []
    for index, slot in enumerate(geometry):
        if index < ordinary:
            if slot is None or len(slot) != len(views[index]):
                raise ValueError(
                    f"geometry slot {index} must carry one trace per batch row."
                )
            decoded_slot = []
            for row, trace in enumerate(slot):
                if isinstance(trace, dict):
                    try:
                        trace = _dense_geometry._geometry.GeometryTrace.from_json(trace)
                    except (KeyError, TypeError, ValueError, OverflowError) as exc:
                        raise ValueError(
                            f"geometry slot {index} row {row} has an invalid serialized trace: {exc}"
                        ) from exc
                if not isinstance(trace, _dense_geometry._geometry.GeometryTrace):
                    raise ValueError(
                        f"geometry slot {index} row {row} carries a non-trace record."
                    )
                decoded_slot.append(trace)
            decoded_geometry.append(tuple(decoded_slot))
        else:
            # Accept old collator envelopes only at this boundary; downstream is canonical.
            if slot is not None and (
                len(slot) != len(views[index]) or any(item is not None for item in slot)
            ):
                raise ValueError(
                    f"geometry slot {index} is a weak-teacher view and must be None "
                    "for every row; no support or trace is computed for weak views."
                )
            decoded_geometry.append(None)
    geometry = tuple(decoded_geometry)

    rows = int(views[0].shape[0])
    for view in views:
        if int(view.shape[0]) != rows:
            raise ValueError(
                "The enabled dense batch must align one row across every view."
            )
    for index, slot in enumerate(geometry):
        if slot is None or index >= ordinary:
            continue
        for row, trace in enumerate(slot):
            view = views[index][row]
            if (int(view.shape[-1]), int(view.shape[-2])) != tuple(trace.view_wh):
                raise ValueError(
                    f"view {index} row {row} is {tuple(view.shape[-2:])} but its "
                    f"trace says {trace.view_wh}."
                )

    if queue_enabled:
        keys = batch["image_key"]
        if isinstance(keys, (str, bytes)):
            raise ValueError("The enabled batch envelope must carry image-key strings.")
        resolved_keys = tuple(str(key) for key in keys)
        if len(resolved_keys) != rows:
            raise ValueError(
                "The enabled batch envelope must align one image key per row: "
                f"{len(resolved_keys)} key(s) for {rows} row(s)."
            )
    else:
        resolved_keys = None
    return views, resolved_keys, geometry


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
    # Initialization is a new-run entry point: reject a both-at-once request
    # before the resume path can re-validate the source as a resume target.
    init_requested = str(getattr(args, "init_checkpoint", "") or "")
    if init_requested and str(getattr(args, "resume", "") or ""):
        raise ValueError(
            "--init-checkpoint and --resume are mutually exclusive: initialization "
            "starts a new run with fresh heads and progress."
        )
    if bool(getattr(args, "init_source_explicit", False)) and not init_requested:
        raise ValueError(
            "--init-source requires --init-checkpoint; omit it for a fresh run."
        )
    # Prepared once, before the settings (which inherit the source architecture,
    # size and register layout) and before any output directory is cleared. The
    # CLI validates the same source before --overwrite can touch anything and
    # hands the prepared snapshot over.
    prepared_initialization = getattr(args, "pretrain_initialization_prepared", None)
    if init_requested and prepared_initialization is None:
        from otuformer.utils.checkpoint import prepare_pretrain_initialization

        prepared_initialization = prepare_pretrain_initialization(
            Path(init_requested),
            source_key=str(getattr(args, "init_source", None) or "teacher"),
            default_model_name=str(getattr(args, "model_name", "") or ""),
        )
    resume_ckpt = resume_payload_for(args, handoff)
    # The queue preparation is model-free and runs before every probe: enabled
    # data identity, the batch/steps-per-epoch locks and the strict recovery
    # checks are all settled before a probe encoder exists. The CLI usually
    # hands it over; a direct-core caller resolves it here.
    preparation = verify_relational_queue_preparation(
        handoff.relational_queue_preparation, args, resume_ckpt
    )
    # Resolve once per invocation, before any output mutation. An omitted value
    # inherits the saved recipe on resume; an explicit conflict is rejected.
    from otuformer.training.dense_recipe import (
        validate_dense_resume as _validate_dense_resume,
    )
    from otuformer.training.dense_recipe import (
        resolve_dense_settings as _resolve_dense_settings,
    )

    # The CLI resolves and validates the dense recipe before it can clear an
    # output directory; a direct caller resolves it here exactly once.
    dense_settings = getattr(args, "pretrain_dense_settings", None)
    if dense_settings is None:
        dense_settings = _resolve_dense_settings(args, resume_ckpt)
    settings = handoff.settings
    if pretrain_recipe is not None and settings is None:
        raise ValueError(
            "pretrain_recipe requires the matching pretrain_settings handoff; "
            "pass both or neither."
        )
    if settings is None:
        settings = resolve_pretrain_settings(
            args, resume_ckpt, initialization=prepared_initialization
        )
        preflight_pretrain_run(
            args, settings, initialization=prepared_initialization
        )
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
    # Keep the target-filtered copy for the eventual restore, not the raw payload.
    if preparation.active:
        pending_rng_restore = preparation.rng_state_to_restore
    elif resume_ckpt is not None and (
        dense_settings.enabled or ssl_config["ssl_method"] == "dino"
    ):
        pending_rng_restore = _verify_restorable_rng_state(
            resume_ckpt, label="pretrain resume", target_device=device
        )
    else:
        pending_rng_restore = (
            resume_ckpt.get("rng_state") if resume_ckpt is not None else None
        )
    # Backbone-only initialization transfers into the fresh encoder below.
    init_source = getattr(args, "init_source", None) or "teacher"
    if prepared_initialization is not None and prepared_initialization.source_key != init_source:
        raise ValueError(
            "The prepared initialization source "
            f"({prepared_initialization.source_key}) does not match the requested "
            f"--init-source {init_source}."
        )
    # The optional sorter is probed before any output exists: a missing or
    # unqualified package must not clear or create the run directory.
    if dense_settings is not None and dense_settings.ranking_enabled:
        ranking_dependency = getattr(args, "pretrain_dense_dependency", None)
        if ranking_dependency is None:
            from otuformer.training.neighbor_ranking import probe_ranking_dependency

            ranking_dependency = probe_ranking_dependency(torch.device(device))
    else:
        ranking_dependency = None
    if dense_settings.enabled:
        from otuformer.training.dense_recipe import _requested

        resolved_dense = {}
        for name, effective, applicable in (
            ("coordinate_weight", dense_settings.coordinate_weight, True),
            ("ranking_weight", dense_settings.ranking_weight, True),
            ("dense_query_points", dense_settings.query_cap, True),
            ("ranking_reference_points", dense_settings.reference_cap,
             dense_settings.ranking_enabled),
        ):
            requested = _requested(args, name)
            resolved_dense[name] = {
                "requested": requested,
                "effective": effective,
                "applicable": applicable,
                "source": (
                    "not_applicable" if not applicable else
                    "requested" if requested is not None else
                    "inherited" if resume_ckpt is not None else "default"
                ),
            }
        print("Resolved dense config: " + json.dumps(resolved_dense, sort_keys=True))
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
        prepared_images=getattr(args, "pretrain_resolved_images", None),
        return_geometry=bool(dense_settings is not None and dense_settings.enabled),
    )
    loader = DataLoader(
        ds,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=(
            collate_pretrain_geometry
            if dense_settings is not None and dense_settings.enabled
            else None
        ),
    )
    _phase_done("dataset indexing and loader", phase)
    if resume_ckpt is not None:
        # Independent re-check with the real loader length: the CLI predicted it
        # before --overwrite, and this confirms the prediction against the object
        # that will actually run, still before any student/optimizer mutation.
        _validate_dense_resume(
            (resume_ckpt.get("config") or {}).get(CONFIG_KEY),
            resume_ckpt.get(STATE_KEY),
            iteration=int(resume_ckpt.get("iteration", -1)),
            batch_size=int(effective_batch_size),
            steps_per_epoch=len(loader),
            dependency=ranking_dependency,
        )

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
    elif prepared_initialization is not None:
        # The source supplies the complete backbone, so a timm download would be
        # wasted work (and could silently differ from the transferred weights).
        # The register layout is inherited from the same resolved source.
        student_kwargs.update(pretrained=False)
        requested_register = getattr(args, "register_tokens", None)
        if prepared_initialization.architecture.added_register_tokens:
            if requested_register not in (None, 4):
                raise ValueError(
                    "the initialization source carries four added register "
                    f"tokens, but --register-tokens {requested_register} was "
                    "requested; the layout cannot be reduced."
                )
            student_kwargs["reg_tokens"] = 4
        elif requested_register is not None:
            student_kwargs["reg_tokens"] = requested_register
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

    if prepared_initialization is not None:
        # Backbone-only transfer: heads, masks, centers, optimizer, schedules
        # and relational history are all rebuilt fresh below.
        from otuformer.utils.checkpoint import initialize_pretrain_backbone

        initialize_pretrain_backbone(student, prepared_initialization)
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
    patch_size = int(student.backbone.patch_embed.patch_size[0]) if isinstance(
        student.backbone.patch_embed.patch_size, (tuple, list)
    ) else int(student.backbone.patch_embed.patch_size)
    patch_count = grid_size[0] * grid_size[1]
    # Enabled-only dense state: one runtime, one grid-size tuple. The all-off
    # path must not create generators, geometry requests or a head.
    dense_grid_hws: tuple[tuple[int, int], ...] = ()
    dense_runtime = None
    if dense_settings is not None and dense_settings.enabled:
        dense_runtime = new_dense_runtime(
            dense_settings, seed=int(getattr(args, "seed", 0))
        )
        if resume_ckpt is not None:
            from otuformer.training.dense_recipe import (
                restore_dense_runtime as _restore_dense_runtime,
            )

            saved_state = resume_ckpt.get(STATE_KEY)
            if saved_state is None:
                raise ValueError(
                    "The checkpoint enables a dense objective but carries no "
                    "dense_training state bundle."
                )
            dense_runtime = _restore_dense_runtime(dense_settings, saved_state)
        local_grid = student._patch_grid_size(
            torch.empty(1, 3, int(args.local_crop_size), int(args.local_crop_size))
        )
        dense_grid_hws = (grid_size, grid_size) + (
            (local_grid,) * int(args.local_crops)
        )
        if include_weak_teacher_views:
            # Weak slots keep a grid entry so the slot/view contract is one
            # shape; they are never consumed for pairing.
            dense_grid_hws = dense_grid_hws + (grid_size, grid_size)
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
    ranking_objective = None
    if dense_settings is not None and dense_settings.ranking_enabled:
        from otuformer.training.neighbor_ranking import NeighborRankingObjective

        hidden = int(student.backbone.embed_dim)
        ranking_objective = NeighborRankingObjective(hidden).to(device)
        if resume_ckpt is not None:
            saved = resume_ckpt.get(STATE_KEY) or {}
            ranking_state = saved.get("ranking") or {}
            ranking_objective.student_head.load_state_dict(
                ranking_state["student_head"], strict=True
            )
            ranking_objective.teacher_head.load_state_dict(
                ranking_state["teacher_head"], strict=True
            )
    has_decay_partition = wd_policy or not _pretrain_decay_partition_empty(
        student, objective, dino_objective, ranking_objective
    )
    optimizer = torch.optim.AdamW(
        _pretrain_optimizer_groups(
            student,
            objective,
            dino_objective,
            weight_decay=args.weight_decay,
            norm_bias_weight_decay=wd_policy,
            ranking_objective=ranking_objective,
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    dense_enabled = dense_settings is not None and dense_settings.enabled
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    instant_logger = InstantMetricsLogger(
        logs_dir / "instant_metrics.pretrain.csv",
        mode="pretrain",
        dense_enabled=dense_settings is not None and dense_settings.enabled,
    )
    compute_embedding_metrics = bool(getattr(args, "compute_embedding_metrics", True))
    epoch_logger = (
        EnhancedMetricsLogger(logs_dir / "metrics.pretrain.csv", mode="pretrain")
        if compute_embedding_metrics
        else None
    )
    # One evaluation snapshot per invocation, taken after the tee is active and
    # before any save can decode images. A manifest that cannot be prepared
    # disables periodic evaluation for this run without stopping training, and
    # the epoch logger and its curves input stay exactly as before.
    evaluation_manifest = handoff.evaluation_manifest
    manifest_unavailable = False
    evaluation_csv = getattr(args, "visualize_data", "") or getattr(
        args, "train_data", ""
    )
    if compute_embedding_metrics and evaluation_manifest is None and evaluation_csv:
        phase = _phase_start("evaluation manifest")
        try:
            resolved_training = getattr(args, "pretrain_resolved_images", None)
            reuse = (
                evaluation_csv == str(getattr(args, "train_data", "") or "")
                and isinstance(resolved_training, ResolvedTrainingImages)
                and resolved_training.csv_path == evaluation_csv
                and resolved_training.images_dir == str(Path(args.input_images_dir))
            )
            evaluation_manifest = prepare_evaluation_manifest(
                Path(evaluation_csv),
                Path(args.input_images_dir),
                max_samples=int(getattr(args, "metrics_sample_size", 10000)),
                seed=int(args.seed),
                image_paths=resolved_training.paths if reuse else None,
            )
        except (ValueError, OSError, KeyError) as exc:
            manifest_unavailable = True
            print(
                "[Warning] Periodic embedding evaluation is disabled for this "
                f"invocation: {exc}"
            )
        _phase_done("evaluation manifest", phase)

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
        elif dense_enabled:
            # An enabled dense resume was validated as complete in preflight, so
            # a permissive load here would silently keep random weights for any
            # tensor the payload lacks.
            student.load_state_dict(resume_ckpt["student"], strict=True)
            teacher.load_state_dict(resume_ckpt["teacher"], strict=True)
        elif "student" in resume_ckpt:
            student.load_state_dict(resume_ckpt["student"], strict=False)
        elif "model_state_dict" in resume_ckpt:
            student.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        if dino_objective is None and "teacher" in resume_ckpt:
            teacher.load_state_dict(
                resume_ckpt["teacher"], strict=dense_enabled
            )
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
        if "optimizer" not in resume_ckpt and dense_enabled:
            # The preflight already rejects this; keep the loop honest for direct
            # callers that bypass it.
            raise ValueError(
                "Cannot resume: an enabled dense run requires its saved optimizer "
                "state; a fresh optimizer would silently change the trajectory."
            )
        if "optimizer" in resume_ckpt:
            _validate_pretrain_optimizer_layout(
                optimizer,
                resume_ckpt["optimizer"],
                norm_bias_weight_decay=wd_policy,
                decay_partition_empty=not has_decay_partition,
                label="pretrain resume",
            )
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            if ranking_objective is not None:
                # Method-agnostic: the restored head state must match the saved
                # contribution count, whichever SSL recipe produced it.
                from otuformer.training.dense_recipe import (
                    validate_ranking_head_optimizer_state as _validate_head_state,
                )

                _validate_head_state(
                    optimizer,
                    ranking_objective.trainable_parameters(),
                    contributed_steps=dense_runtime.ranking_contributed_steps,
                )
            _report_restored_weight_decay(
                optimizer,
                requested_weight_decay=float(args.weight_decay),
                norm_bias_weight_decay=wd_policy,
                has_decay_partition=has_decay_partition,
                label="pretrain resume",
            )
        if "center" not in resume_ckpt and dense_enabled and dino_objective is None:
            raise ValueError(
                "Cannot resume: an enabled dense relational run requires its saved "
                "center; a fresh center would change the teacher targets."
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
        sample_views, _sample_keys, _sample_geometry = _unpack_pretrain_batch(
            sample_batch,
            queue_enabled=queue is not None,
            dense_enabled=dense_settings is not None and dense_settings.enabled,
            local_crops=int(args.local_crops),
            include_weak_teacher_views=(
                include_weak_teacher_views
                if dense_settings is not None and dense_settings.enabled
                else None
            ),
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
            views, batch_keys, batch_geometry = _unpack_pretrain_batch(
                batch,
                queue_enabled=queue is not None,
                dense_enabled=dense_settings is not None and dense_settings.enabled,
                local_crops=int(args.local_crops),
                include_weak_teacher_views=(
                    include_weak_teacher_views
                    if dense_settings is not None and dense_settings.enabled
                    else None
                ),
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

            prefix = student.backbone.num_prefix_tokens
            student_global: list[torch.Tensor] = []
            student_tokens: list[torch.Tensor] = []
            # Dense objectives read ordinary unmasked patch grids. Globals come
            # from forward_pretrain (prefix still attached); locals reuse the
            # encoder's already prefix-free patch output.
            student_grids: dict[int, torch.Tensor] = {}
            teacher_grids: dict[int, torch.Tensor] = {}
            for view_index, gv in enumerate(global_views):
                s_out = student.forward_pretrain(gv)
                student_global.append(s_out.cls)
                student_tokens.append(s_out.tokens)
                if dense_enabled:
                    student_grids[view_index] = _to_feature_grid(
                        s_out.tokens, prefix, s_out.grid_size
                    )

            teacher_tokens: list[torch.Tensor] = []
            teacher_final_four: list[list[torch.Tensor] | None] = []
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
                    # Both ordinary teacher globals are the dense feature source.
                    or dense_enabled
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
                for teacher_index, t_out in enumerate(ordinary_outputs):
                    if dense_enabled:
                        teacher_grids[teacher_index] = _to_feature_grid(
                            t_out.tokens.detach(), prefix, t_out.grid_size
                        )
                    if dino_objective is not None:
                        # DINO sends the unmodified normalized bottleneck to its
                        # prototype layer and bypasses the legacy center path.
                        teacher_global.append(t_out.cls)
                    teacher_tokens.append(t_out.tokens)
                    teacher_final_four.append(t_out.final_four)

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
            # Non-FIFO teacher targets are per-slot constants within a step: one
            # build each, shared by every global and local pair.
            teacher_cache: dict[int, torch.Tensor] = {}
            if dino_objective is None:
                loss_global = _compute_global_loss(
                    student_global,
                    teacher_global,
                    temp_student,
                    temp_teacher,
                    relational_pairing=relational_pairing,
                    relational_targets=relational_targets,
                    student_stats=student_stats,
                    teacher_cache=teacher_cache,
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
                        # Gather each pointwise block to the selected rows first:
                        # averaging and normalizing commute with the selection.
                        layers = [
                            final_norm(block)[:, prefix:][mask]
                            for block in teacher_final_four[index]
                        ]
                        target = F.normalize(
                            torch.stack(layers).mean(dim=0), dim=-1
                        )
                    masked_out = student.forward_pretrain(
                        gv, mask=mask, mask_token=objective.mask_token
                    )
                    student_patches = masked_out.tokens[:, prefix:]
                    _validate_patch_mask(mask, student_patches)
                    prediction = objective.predict_masked_features(
                        student_patches[mask]
                    )
                    view_losses.append(
                        selected_patch_cosine_loss(prediction, target)
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
                    student_patches = masked_out.tokens[:, prefix:]
                    teacher_patches = teacher_tokens[index][:, prefix:]
                    _validate_patch_mask(mask, student_patches)
                    with torch.no_grad():
                        # The teacher head is pointwise, so gathering the mask
                        # before it skips the unselected tokenwise work without
                        # touching either encoder forward or any RNG draw.
                        teacher_selected = objective.teacher_ibot_head(
                            teacher_patches[mask]
                        )
                    student_selected = objective.student_ibot_head(
                        student_patches[mask]
                    )
                    view_loss, view_probs = ibot_selected_patch_loss(
                        student_selected,
                        teacher_selected,
                        objective.patch_center,
                        temp_student,
                        temp_teacher,
                    )
                    view_losses.append(view_loss)
                    ibot_probs.append(view_probs)
                    center_logits.append(teacher_selected.detach())
                patch_loss_raw = torch.stack(view_losses).mean()

            loss_mask = patch_loss_raw
            patch_loss_weighted = args.lambda_mask * patch_loss_raw

            local_student_projs: list[torch.Tensor] = []
            for local_index, lv in enumerate(local_views):
                l_proj, l_tokens = student(lv)
                local_student_projs.append(l_proj)
                if dense_enabled:
                    # ``OTUFormerEncoder.forward`` has already stripped the
                    # CLS/register prefix, so this output must not be sliced.
                    local_grid = student._patch_grid_size(lv)
                    student_grids[2 + local_index] = _to_feature_grid(
                        l_tokens, 0, local_grid
                    )
            if dino_objective is None:
                loss_local = _compute_local_loss(
                    local_student_projs,
                    teacher_global,
                    temp_student,
                    temp_teacher,
                    relational_targets=relational_targets,
                    student_stats=student_stats,
                    teacher_cache=teacher_cache,
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

            dense_step_stats: dict[str, Any] = {}
            dense_metrics: dict[str, Any] = {}
            if dense_settings is not None and dense_settings.enabled:
                dense_coordinate, dense_stats, dense_ranking, ranking_stats = (
                    _dense_terms(
                        dense_settings,
                        dense_runtime,
                        ranking_objective,
                        student_grids,
                        teacher_grids,
                        batch_geometry,
                        dense_grid_hws,
                        patch_hw=(patch_size, patch_size),
                    )
                )
                dense_step_stats = {**dense_stats, **ranking_stats}
                coordinate_raw = dense_stats.get("raw")
                dense_metrics = {
                    "dense_coordinate_loss_raw": (
                        "" if coordinate_raw is None else float(coordinate_raw)
                    ),
                    "dense_coordinate_loss_weighted": (
                        ""
                        if coordinate_raw is None
                        else float(coordinate_raw) * float(dense_settings.coordinate_weight)
                    ),
                    "dense_coordinate_status": _dense_status(
                        float(dense_settings.coordinate_weight), coordinate_raw
                    ),
                    "dense_coordinate_points": int(
                        dense_stats.get("selected_points", 0)
                    ),
                    "dense_coordinate_samples": int(
                        dense_stats.get("contributing_samples", 0)
                    ),
                    "dense_coordinate_pairs": int(
                        dense_stats.get("contributing_pairs", 0)
                    ),
                    # ``None`` populations stay empty: a missing measurement is
                    # not an observed 0 norm or a perfect rho of 1.
                    "dense_min_norm": _dense_stat_or_blank(dense_stats.get("min_norm")),
                    "dense_min_rho": _dense_stat_or_blank(dense_stats.get("min_rho")),
                    "dense_coordinate_eligible": int(
                        dense_stats.get("eligible_points", 0)
                    ),
                }
                ranking_raw = ranking_stats.get("raw")
                dense_metrics.update(
                    {
                        "dense_ranking_loss_raw": (
                            "" if ranking_raw is None else float(ranking_raw)
                        ),
                        "dense_ranking_loss_weighted": (
                            ""
                            if ranking_raw is None
                            else float(ranking_raw) * float(dense_settings.ranking_weight)
                        ),
                        "dense_ranking_status": _dense_status(
                            float(dense_settings.ranking_weight), ranking_raw
                        ),
                        "dense_ranking_points": int(
                            ranking_stats.get("selected_points", 0)
                        ),
                        "dense_ranking_samples": int(
                            ranking_stats.get("contributing_samples", 0)
                        ),
                        "dense_ranking_pairs": int(
                            ranking_stats.get("contributing_pairs", 0)
                        ),
                        "dense_ranking_references": _dense_stat_or_blank(
                            ranking_stats.get("reference_population")
                        ),
                        "dense_ranking_reference_count": _dense_stat_or_blank(
                            ranking_stats.get("reference_count")
                        ),
                        "dense_ranking_shortage_samples": int(
                            ranking_stats.get("reference_shortage_samples", 0)
                        ),
                        "dense_ranking_min_probability": _dense_stat_or_blank(
                            ranking_stats.get("min_probability")
                        ),
                        # This objective's own projected-space diagnostics; the
                        # Coordinate minima describe a different feature space.
                        "dense_ranking_min_norm": _dense_stat_or_blank(
                            ranking_stats.get("teacher_min_norm")
                        ),
                        "dense_ranking_min_rho": _dense_stat_or_blank(
                            ranking_stats.get("teacher_min_rho")
                        ),
                        # Raw projection norm *before* per-patch L2: the value the
                        # projection guard actually tests.
                        "dense_ranking_projection_min_norm": _dense_stat_or_blank(
                            ranking_stats.get("teacher_projection_min_norm")
                        ),
                        "dense_ranking_reference_min": _dense_stat_or_blank(
                            ranking_stats.get("reference_count_min")
                        ),
                        "dense_ranking_reference_population_min": _dense_stat_or_blank(
                            ranking_stats.get("reference_population_min")
                        ),
                        "dense_ranking_unavailable_pairs": int(
                            ranking_stats.get("reference_unavailable_pairs", 0)
                        ),
                    }
                )
                from .dense_objective import BRANCH_COUNT_FIELDS

                for objective_name, stats, enabled in (
                    ("coordinate", dense_stats, dense_settings.coordinate_weight > 0),
                    ("ranking", ranking_stats, dense_settings.ranking_enabled),
                ):
                    for branch in ("global", "local"):
                        for field in BRANCH_COUNT_FIELDS:
                            key = f"{branch}_{field}"
                            dense_metrics[f"dense_{objective_name}_{key}"] = (
                                int(stats.get(key, 0)) if enabled else ""
                            )
                if dense_coordinate is not None:
                    total_loss = (
                        total_loss
                        + float(dense_settings.coordinate_weight) * dense_coordinate
                    )
                if dense_ranking is not None:
                    total_loss = (
                        total_loss
                        + float(dense_settings.ranking_weight) * dense_ranking
                    )

            if not torch.isfinite(total_loss):
                raise ValueError(
                    "The pretraining loss became non-finite at iteration "
                    f"{global_step}; refusing to update the parameters. Check the "
                    "temperatures, learning rate and inputs."
                )

            if dense_enabled:
                # Explicit set_to_none=True: a ranking head that becomes
                # unavailable must not keep a stale gradient that would create or
                # decay AdamW moments for a step it did not contribute to.
                optimizer.zero_grad(set_to_none=True)
            else:
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
            logged_step = global_step % max(1, args.log_every_n_steps) == 0
            grad_norm = _compute_grad_norm_parameters(
                trainable_params, numeric=logged_step
            )
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
            if (queue is not None or dense_enabled) and not bool(torch.stack([
                torch.isfinite(parameter).all() for parameter in trainable_params
            ]).all()):
                raise ValueError(
                    "The pretraining optimizer update became non-finite at "
                    f"iteration {global_step}; refusing to advance the EMA, FIFO or "
                    "save a checkpoint. Check the optimizer coefficients."
                )

            m = float(momentum_schedule[iter_idx])
            update_teacher(student, teacher, m)
            if ranking_objective is not None:
                # Ranking teacher head EMA and the contributed-step counter both
                # follow a successful optimizer step, never a skipped one.
                ranking_objective.update_teacher(m)
                if dense_step_stats.get("ranking_contributed"):
                    dense_runtime.ranking_contributed_steps += 1
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

            needs_display = step % 10 == 0 or step == len(loader) - 1
            # One host read per scalar for every consumer of this step: the CSV
            # row and the progress postfix. Steps that consume neither stay
            # unsynchronized.
            step_values: dict[str, float] = {}
            if logged_step or needs_display:
                step_values = {
                    "total": float(total_loss.item()),
                    "global": float(loss_global.item()),
                    "local": float(loss_local.item()),
                    "mask": float(loss_mask.item()),
                }
            cosine_sim = 0.0
            if dino_objective is None and (logged_step or needs_display):
                with torch.no_grad():
                    cosine_sim = F.cosine_similarity(
                        student_global[0].mean(dim=0, keepdim=True),
                        ordinary_teacher[0].mean(dim=0, keepdim=True),
                    ).item()

            if logged_step:
                with torch.no_grad():
                    instant_metrics = _compute_instant_metrics(
                        student, global_views[0], device
                    )
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
                    "global_loss_weighted": step_values["global"],
                    "local_loss_weighted": float(
                        args.lambda_local * step_values["local"]
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
                            weighted_value / step_values["total"]
                            if step_values["total"] != 0.0
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
                    loss=step_values["total"],
                    global_loss=step_values["global"],
                    local_loss=step_values["local"],
                    mask_loss=step_values["mask"],
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
                    **dense_metrics,
                )

            if needs_display:
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
                    L=f"{step_values['total']:.2e}",
                    GL=f"{step_values['global']:.2e}",
                    LocL=f"{step_values['local']:.2e}",
                    ML=f"{step_values['mask']:.2e}",
                    COS=cos_display,
                    LR=f"{optimizer.param_groups[0]['lr']:.2e}",
                    MOM=f"{m:.4f}",
                )

            global_step += 1

        should_save = (epoch + 1) % max(1, args.save_every_epochs) == 0 or (
            epoch + 1
        ) == args.max_epochs
        if (
            should_save
            and compute_embedding_metrics
            and epoch_logger is not None
            and not manifest_unavailable
        ):
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
                manifest=evaluation_manifest,
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
            # A resumed run keeps its recorded immediate source without
            # reopening the file; a new run records the transfer it performed.
            existing_initialization = (
                (resume_ckpt.get("config") or {}).get("pretrain_initialization")
                if resume_ckpt is not None
                else None
            )
            if existing_initialization is not None:
                from otuformer.utils.checkpoint import (
                    validate_pretrain_initialization_bundle,
                )

                validate_pretrain_initialization_bundle(existing_initialization)
            dense_config = (
                dense_config_bundle(
                    dense_settings,
                    batch_size=int(effective_batch_size),
                    steps_per_epoch=int(len(loader)),
                    dependency=ranking_dependency,
                )
                if dense_settings is not None and dense_settings.enabled
                else None
            )
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
                    **(
                        {CONFIG_KEY: dense_config}
                        if dense_config is not None
                        else {}
                    ),
                    **(
                        {
                            "pretrain_initialization": (
                                prepared_initialization.config_bundle()
                                if prepared_initialization is not None
                                else existing_initialization
                            )
                        }
                        if (
                            prepared_initialization is not None
                            or existing_initialization is not None
                        )
                        else {}
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
            if dense_runtime is not None:
                # Enabled runs persist the closed dense bundles at the existing
                # epoch boundary; the all-off path writes neither key.
                ckpt[STATE_KEY] = dense_state_bundle(
                    dense_runtime,
                    completed_steps=int(global_step),
                    ranking=ranking_objective,
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
