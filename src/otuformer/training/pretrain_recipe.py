"""Pretrain settings, SSL recipe resolution, recovery and preflight."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple
import numpy as np
import torch
import torch.nn as nn
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
from otuformer.training.dataset import (
    FINETUNE_AUGMENTATIONS,
    ORIENTATION_POLICIES,
    PRETRAIN_AUGMENTATIONS,
    IndexedDataset,
    MetricDataset,
    MultiCropDataset,
    PreparedRelationalData,
    _build_recursive_index,
    _resolve_image_path,
    _supports_recursive_lookup,
    build_finetune_augmentation_config,
    build_pretrain_augmentation_config,
    build_weak_teacher_augmentation_config,
    center_crop_eval_transform,
    prepare_relational_data,
)
from otuformer.training.dino import (
    DINOObjective,
    check_dino_weight_norm_api_record,
    validate_dino_prototype_layer,
)
from otuformer.training.model import (
    PATCH_TARGET_LAYERS,
    ArcFaceEmbeddingHead,
    OTUFormerEncoder,
    PatchObjective,
    native_register_count,
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
from otuformer.utils.size import (
    _positive_int,
    resolve_backbone_native_size,
    resolve_training_image_size,
    validate_input_size,
)
from .augmentation import (
    _REGISTER_RESUME_KEYS,
    _assert_encoder_state_matches,
    _check_register_option_against_resume,
    _resolve_pretrain_local_views,
    _resume_register_plan,
    _select_augmentation_profile,
    _select_orientation_policy,
    _validate_augmentation_config,
)
from .patch_objective import (
    _resolve_patch_config,
    _validate_blockwise_grid,
)
from .runtime import (
    _restore_rng_state,
)

def _student_optimizer_parameters(
    student: nn.Module,
    patch_objective: nn.Module | None,
    dino_objective: nn.Module | None = None,
) -> list[nn.Parameter]:
    """Student-side parameters plus each objective's trainable state.

    One stable order: encoder parameters, patch parameters, then the DINO
    student prototype direction. This is the ordered source for the optimizer
    groups; gradient-norm statistics and clipping always cover the joined list,
    independent of how many weight-decay groups it is split into.
    """
    params = [p for p in student.parameters() if p.requires_grad]
    if patch_objective is not None:
        params.extend(patch_objective.trainable_parameters())
    if dino_objective is not None:
        params.extend(dino_objective.trainable_parameters())
    return params


def _norm_affine_types() -> tuple[type[nn.Module], ...]:
    """Normalization modules whose direct affine parameters skip weight decay."""
    types: list[type[nn.Module]] = [
        nn.LayerNorm,
        nn.GroupNorm,
        nn.modules.batchnorm._BatchNorm,
        nn.modules.instancenorm._InstanceNorm,
    ]
    rms_norm = getattr(nn, "RMSNorm", None)
    if isinstance(rms_norm, type) and rms_norm not in types:
        types.append(rms_norm)
    return tuple(types)


def _no_decay_parameter_ids(
    student: nn.Module,
    patch_objective: nn.Module | None,
    dino_objective: nn.Module | None,
) -> set[int]:
    """Identity set of parameters that skip weight decay under the opt-in policy.

    Membership is module-based: every directly named ``bias`` and the affine
    ``weight`` of a recognized normalization module. Tokens, generic scales and
    weight-normed ``weight_v`` are deliberately not exempt.
    """
    types = _norm_affine_types()
    roots: list[nn.Module] = [student]
    for objective in (patch_objective, dino_objective):
        if objective is not None:
            roots.append(objective)
    ids: set[int] = set()
    for root in roots:
        for module in root.modules():
            bias = getattr(module, "bias", None)
            if isinstance(bias, nn.Parameter):
                ids.add(id(bias))
            if isinstance(module, types):
                weight = getattr(module, "weight", None)
                if isinstance(weight, nn.Parameter):
                    ids.add(id(weight))
    return ids


def _pretrain_optimizer_groups(
    student: nn.Module,
    patch_objective: nn.Module | None,
    dino_objective: nn.Module | None,
    *,
    weight_decay: float,
    norm_bias_weight_decay: bool,
) -> list[dict[str, Any]]:
    """Build the pretrain optimizer groups for the selected WD policy.

    ``True`` returns the historical single group unchanged: no explicit
    ``weight_decay`` key, so ``AdamW`` supplies the same default it always did.
    ``False`` splits the same ordered parameter list into a decay group followed
    by a no-decay group, each keeping source-relative order and empty groups
    omitted.
    """
    params = _student_optimizer_parameters(student, patch_objective, dino_objective)
    if norm_bias_weight_decay:
        return [{"params": params}]
    no_decay_ids = _no_decay_parameter_ids(
        student, patch_objective, dino_objective
    )
    seen: set[int] = set()
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in params:
        if id(parameter) in seen:
            # Alias registration can expose one Norm parameter twice; the
            # optimizer must never see it in both groups.
            continue
        seen.add(id(parameter))
        if id(parameter) in no_decay_ids:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _pretrain_decay_partition_empty(
    student: nn.Module,
    patch_objective: nn.Module | None,
    dino_objective: nn.Module | None,
) -> bool:
    """True when no trainable parameter keeps weight decay under the opt-in policy."""
    params = _student_optimizer_parameters(student, patch_objective, dino_objective)
    no_decay_ids = _no_decay_parameter_ids(
        student, patch_objective, dino_objective
    )
    return all(id(parameter) in no_decay_ids for parameter in params)


def _validate_saved_optimizer_structure(
    saved_optimizer: Any,
    *,
    norm_bias_weight_decay: bool,
    label: str,
) -> list[dict[str, Any]]:
    """Validate the saved optimizer envelope without rebuilding the model.

    Covers the group envelope, the group counts the recorded policy can produce,
    the canonical parameter-id order of a saved optimizer and the two-group
    no-decay coefficient. Group sizes and the meaning of a single group need the
    rebuilt parameter partition, so ``_validate_pretrain_optimizer_layout`` owns
    those.
    """
    saved_groups = (
        saved_optimizer.get("param_groups")
        if isinstance(saved_optimizer, dict)
        else None
    )
    if not isinstance(saved_groups, list) or not saved_groups:
        raise ValueError(
            f"Cannot resume: the {label} optimizer state records no parameter "
            "groups."
        )
    # True is always one all-parameter group; False is one or two.
    allowed = (1,) if norm_bias_weight_decay else (1, 2)
    if len(saved_groups) not in allowed:
        expected = "one" if len(allowed) == 1 else "one or two"
        raise ValueError(
            f"Cannot resume: norm_bias_weight_decay={norm_bias_weight_decay} "
            f"rebuilds {expected} optimizer parameter group(s), but the "
            f"checkpoint records {len(saved_groups)}. A saved decay/no-decay "
            "partition is never converted on resume; start a new run to change "
            "the policy."
        )
    offset = 0
    for index, saved in enumerate(saved_groups):
        _validate_adamw_param_group(saved, label)
        saved_params = saved.get("params")
        if not isinstance(saved_params, list) or not saved_params:
            raise ValueError(
                f"Cannot resume: optimizer group {index} lists no parameters."
            )
        # A saved optimizer numbers its parameters 0..N-1 in group order. Any
        # other order would attach saved moments to a different parameter of the
        # same shape, which load_state_dict() maps by position and cannot detect.
        if list(saved_params) != list(range(offset, offset + len(saved_params))):
            raise ValueError(
                f"Cannot resume: optimizer group {index} records parameter ids "
                "in an order a saved optimizer never produces, so the stored "
                "moments could belong to other parameters. Start a new run "
                "instead."
            )
        offset += len(saved_params)
    if len(saved_groups) == 2 and float(saved_groups[1]["weight_decay"]) != 0.0:
        raise ValueError(
            "Cannot resume: the no-decay optimizer group records "
            f"weight_decay={saved_groups[1]['weight_decay']!r}; it must be "
            "exactly 0.0."
        )
    return saved_groups


def _validate_pretrain_optimizer_layout(
    optimizer: torch.optim.Optimizer,
    saved_optimizer: Any,
    *,
    norm_bias_weight_decay: bool,
    label: str,
    decay_partition_empty: bool = False,
) -> None:
    """Reject a saved optimizer layout this run cannot map exactly.

    The rebuilt partition supplies per-group size and the meaning of a single
    group on top of ``_validate_saved_optimizer_structure``; a historical
    one-group checkpoint is never converted into a two-group one. The saved
    decay coefficient itself stays authoritative and is not compared with the
    invocation.
    """
    saved_groups = _validate_saved_optimizer_structure(
        saved_optimizer,
        norm_bias_weight_decay=norm_bias_weight_decay,
        label=label,
    )
    built = optimizer.param_groups
    if len(saved_groups) != len(built):
        raise ValueError(
            f"Cannot resume: the checkpoint records {len(saved_groups)} optimizer "
            f"parameter group(s), but norm_bias_weight_decay="
            f"{norm_bias_weight_decay} rebuilds {len(built)} group(s) from this "
            "model. A saved decay/no-decay partition is never converted on "
            "resume; start a new run to change the policy."
        )
    for index, (saved, current) in enumerate(zip(saved_groups, built)):
        saved_count = len(saved["params"])
        if saved_count != len(current["params"]):
            raise ValueError(
                f"Cannot resume: optimizer group {index} records {saved_count} "
                f"parameter(s) but this run rebuilds {len(current['params'])} "
                "for the same partition. The saved moments cannot be mapped "
                "by shape; start a new run instead."
            )
    # A single rebuilt group is the no-decay partition only when the decay
    # partition is empty; the group count alone cannot decide its meaning.
    if (
        not norm_bias_weight_decay
        and decay_partition_empty
        and float(saved_groups[0]["weight_decay"]) != 0.0
    ):
        raise ValueError(
            "Cannot resume: the no-decay optimizer group records "
            f"weight_decay={saved_groups[0]['weight_decay']!r}; it must be "
            "exactly 0.0."
        )


def _report_restored_weight_decay(
    optimizer: torch.optim.Optimizer,
    *,
    requested_weight_decay: float,
    norm_bias_weight_decay: bool,
    has_decay_partition: bool,
    label: str,
) -> None:
    """Print the authoritative resumed WD instead of the invocation value.

    Called after a successful ``load_state_dict`` so the emitted numbers are the
    loaded ones, never the requested ones.
    """
    if not has_decay_partition:
        print(
            f"[Info] {label}: the rebuilt layout has no decay group, so the "
            f"invocation value --weight-decay {requested_weight_decay:g} does "
            "not apply to it. The no-decay group keeps weight decay 0.0."
        )
        return
    restored = float(optimizer.param_groups[0]["weight_decay"])
    if math.isclose(restored, requested_weight_decay, rel_tol=0.0, abs_tol=1e-12):
        return
    print(
        f"[Info] {label}: resumed weight decay is {restored:g} from the "
        f"checkpoint. The invocation value {requested_weight_decay:g} "
        "(requested or default) is not applied to a resumed run."
    )


def _initialize_dino_student_projector(student: OTUFormerEncoder) -> None:
    """Reference DINO head init for the student's projection MLP only.

    Truncated-normal weights and zero biases, applied on a fresh DINO run only:
    backbone parameters, relational projectors and resumed DINO projectors keep
    their own initialization.
    """
    for module in student.projector.net:
        if not isinstance(module, nn.Linear):
            continue
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _synchronize_dino_teacher(
    student: OTUFormerEncoder,
    teacher: OTUFormerEncoder,
    dino_objective: nn.Module,
) -> None:
    """Copy the student encoder (including registers) and prototype layers."""
    with torch.no_grad():
        teacher.load_state_dict(student.state_dict(), strict=True)
        dino_objective.teacher_prototypes.load_state_dict(
            dino_objective.student_prototypes.state_dict()
        )


def _cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: float = 0.0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    total_iters = max(1, epochs * niter_per_ep)
    warmup_iters = int(max(0.0, warmup_epochs) * niter_per_ep)
    warmup_iters = min(warmup_iters, total_iters)
    schedule = np.ones(total_iters, dtype=np.float32) * float(final_value)

    if warmup_iters > 0:
        warmup_schedule = np.linspace(
            start_warmup_value,
            base_value,
            warmup_iters,
            dtype=np.float32,
        )
        schedule[:warmup_iters] = warmup_schedule

    iters = np.arange(total_iters, dtype=np.float32)
    after = iters[warmup_iters:]
    if len(after) > 0:
        schedule[warmup_iters:] = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * (after - warmup_iters) / max(1, len(after)))
        )
    return schedule


def _validate_pretrain_epoch_target(
    requested_epochs: int | None,
    completed_epochs: int,
    schedule_state: dict[str, Any] | None,
    recorded_epochs: int | None,
) -> None:
    """Reject targets that cannot continue the saved schedule before output."""
    if requested_epochs is None:
        return
    if completed_epochs > 0 and requested_epochs <= completed_epochs:
        raise ValueError("--max-epochs must exceed the completed checkpoint epoch.")
    if schedule_state is None:
        if completed_epochs > 0 and (
            recorded_epochs is None or requested_epochs != int(recorded_epochs)
        ):
            raise ValueError(
                "Cannot resume a legacy checkpoint without schedule metadata."
            )
        return
    if not isinstance(schedule_state, dict):
        return
    original_epochs = schedule_state.get("original_max_epochs")
    if isinstance(original_epochs, bool) or not isinstance(original_epochs, int):
        return
    if requested_epochs < original_epochs:
        raise ValueError("Cannot decrease the original pretrain epoch plan.")
    if (
        requested_epochs == original_epochs
        and isinstance(recorded_epochs, int)
        and recorded_epochs > original_epochs
    ):
        raise ValueError(
            "Cannot return an extended pretrain run to its original epoch plan; "
            f"use --max-epochs > {original_epochs} to continue the extension."
        )


def _build_pretrain_schedules(
    args: argparse.Namespace,
    schedule_state: dict[str, Any] | None,
    *,
    steps_per_epoch: int,
    global_step: int,
    completed_epochs: int,
    original_max_epochs: int | None = None,
    teacher_temp_warmup_ratio: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build schedules without restarting LR after a resumed pretrain run."""
    _validate_pretrain_epoch_target(
        args.max_epochs, completed_epochs, schedule_state, original_max_epochs
    )
    final_lr = float(args.lr) * 0.01
    if schedule_state is None:
        total_steps = max(1, args.max_epochs * steps_per_epoch)
        warmup_steps = int(max(0, args.warmup_epochs) * steps_per_epoch)
        metadata = {
            "original_max_epochs": args.max_epochs,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "warmup_steps": warmup_steps,
            "last_lr": float(args.lr),
            "final_lr": final_lr,
        }
        lr = _cosine_scheduler(
            args.lr, final_lr, args.max_epochs, steps_per_epoch,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr * 0.1,
        )
        momentum = _cosine_scheduler(
            args.teacher_momentum, args.teacher_momentum_end,
            args.max_epochs, steps_per_epoch,
        )
        teacher_temp = _build_teacher_temp_schedule(
            total_steps,
            args.teacher_temp_start,
            args.teacher_temp_end,
            _teacher_temp_warmup_ratio(args),
        )
    else:
        original_epochs = int(schedule_state["original_max_epochs"])
        original_steps = int(schedule_state["total_steps"])
        original_spe = int(schedule_state["steps_per_epoch"])
        if args.max_epochs == original_epochs:
            if steps_per_epoch != original_spe:
                raise ValueError("same-plan resume requires the original DataLoader length.")
            lr = _cosine_scheduler(
                args.lr, float(schedule_state["final_lr"]), original_epochs,
                original_spe, warmup_epochs=float(schedule_state["warmup_steps"]) / original_spe,
                start_warmup_value=args.lr * 0.1,
            )
            momentum = _cosine_scheduler(
                args.teacher_momentum, args.teacher_momentum_end, original_epochs, original_spe
            )
            teacher_temp = _build_teacher_temp_schedule(
                original_steps,
                args.teacher_temp_start,
                args.teacher_temp_end,
                _teacher_temp_warmup_ratio(args),
            )
            metadata = dict(schedule_state)
        else:
            extension_steps = (args.max_epochs - completed_epochs) * steps_per_epoch
            extension_lr = _cosine_scheduler(
                float(schedule_state["last_lr"]), float(schedule_state["final_lr"]),
                1, extension_steps,
            )
            lr = np.concatenate([np.full(global_step, float(schedule_state["last_lr"])), extension_lr])
            momentum = np.full(len(lr), args.teacher_momentum_end, dtype=np.float32)
            teacher_temp = np.full(len(lr), args.teacher_temp_end, dtype=np.float32)
            metadata = {
                **schedule_state,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": len(lr),
                "last_lr": float(schedule_state["last_lr"]),
            }
    if global_step >= len(lr):
        raise ValueError("Checkpoint iteration exceeds the available pretrain schedule.")
    student_temp = np.full(len(lr), float(args.student_temp), dtype=np.float32)
    return lr, momentum, student_temp, teacher_temp, metadata


def _teacher_temp_warmup_ratio(args: argparse.Namespace) -> float:
    """The effective ramp fraction.

    Only a missing or ``None`` attribute falls back to the historical constant;
    an explicit ``0.0`` is a legal "no ramp at all" and must reach the schedule.
    """
    value = getattr(args, "teacher_temp_warmup_ratio", None)
    if value is None:
        return HISTORICAL_TEACHER_TEMP_WARMUP_RATIO
    return float(value)


def _build_teacher_temp_schedule(
    total_iters: int,
    teacher_temp_start: float,
    teacher_temp_end: float,
    teacher_temp_warmup_ratio: float = 0.7,
) -> np.ndarray:
    warmup_iters = int(total_iters * float(teacher_temp_warmup_ratio))
    if warmup_iters > 0:
        return np.concatenate(
            [
                np.linspace(
                    teacher_temp_start,
                    teacher_temp_end,
                    warmup_iters,
                    dtype=np.float32,
                ),
                np.ones(total_iters - warmup_iters, dtype=np.float32)
                * float(teacher_temp_end),
            ]
        )
    return np.ones(total_iters, dtype=np.float32) * float(teacher_temp_end)


_SSL_DINO_ONLY = ("dino_prototypes", "dino_centering")


# KoLeo is one shared pretrain regularizer, not a DINO feature: both methods read
# these constants and neither module owns a second copy. DINO still records all
# four keys at any weight; relational records them only when the weight is
# positive, so a zero-weight relational checkpoint keeps its historical shape.
KOLEO_WEIGHT_KEY = "koleo_weight"
KOLEO_EPSILON_KEYS = (
    "koleo_normalize_eps",
    "koleo_distance_eps",
    "koleo_log_eps",
)
KOLEO_STORED_KEYS = (KOLEO_WEIGHT_KEY, *KOLEO_EPSILON_KEYS)
KOLEO_FIXED_CONFIG = {
    "koleo_normalize_eps": 1e-8,
    "koleo_distance_eps": 1e-8,
    "koleo_log_eps": 1e-8,
}
KOLEO_DEFAULT_WEIGHT = 0.0


_SSL_RELATIONAL_ONLY = (
    "relational_teacher_views",
    "relational_center_source",
    "relational_pairing",
)


@dataclass
class _SavedSSLState:
    """What a resume payload records about its SSL recipe."""

    identity: tuple[str, str] | None
    config: dict[str, Any]
    saved_args: dict[str, Any]
    out_dim: int | None
    weight_out_dim: int | None


def _saved_ssl_state(resume_ckpt: dict[str, Any] | None) -> _SavedSSLState | None:
    from otuformer.utils.checkpoint import (
        resolve_checkpoint_ssl_identity,
        resolve_checkpoint_state_dict,
    )

    if resume_ckpt is None:
        return None
    from otuformer.utils.checkpoint import resolve_projector_out_dim

    state, _ = resolve_checkpoint_state_dict(resume_ckpt)
    cfg = resume_ckpt.get("config")
    cfg = cfg if isinstance(cfg, dict) else {}
    saved_args = resume_ckpt.get("args")
    saved_args = saved_args if isinstance(saved_args, dict) else {}
    return _SavedSSLState(
        identity=resolve_checkpoint_ssl_identity(resume_ckpt, state),
        config=cfg,
        saved_args=saved_args,
        out_dim=_positive_int(cfg.get("out_dim")),
        # A historical payload may record no width while its weights have one.
        weight_out_dim=resolve_projector_out_dim(state),
    )


def _ssl_option(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, default)
    return default if value is None else value


def _ssl_explicit(args: argparse.Namespace, name: str) -> bool:
    return bool(getattr(args, f"{name}_explicit", False))


def _resolve_inherited_ssl_value(
    args: argparse.Namespace,
    name: str,
    default: Any,
    saved: _SavedSSLState | None,
) -> Any:
    """Omitted values inherit a DINO checkpoint; explicit conflicts fail."""
    requested = _ssl_option(args, name, default)
    recorded = saved.config.get(name) if saved is not None else None
    if _ssl_explicit(args, name):
        if recorded is not None and requested != recorded:
            raise ValueError(
                f"Cannot change {name.replace('_', '-')} on resume: the checkpoint "
                f"records {name}={recorded!r}, got {requested!r}."
            )
        return requested
    if recorded is not None:
        return recorded
    return requested


RECIPE_TEACHER_TEMP_WARMUP_RATIO = 0.7


RECIPE_NORM_BIAS_WEIGHT_DECAY = True


_COMMON_RECIPE_OPTIONS = (
    "teacher_temp_warmup_ratio",
    "norm_bias_weight_decay",
)


def _validate_recipe_ratio(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"teacher_temp_warmup_ratio must be a number, got {value!r}."
        )
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(
            "teacher_temp_warmup_ratio must be a finite value in [0, 1], got "
            f"{value!r}."
        )
    return ratio


def _validate_recipe_wd_flag(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"norm_bias_weight_decay must be a boolean, got {value!r}."
        )
    return value


def _resolve_common_recipe(
    args: argparse.Namespace, saved: _SavedSSLState | None
) -> dict[str, Any]:
    """Resolve the two shared pretrain recipe fields for both SSL methods.

    Recorded values inherit like DINO settings. A payload that records neither
    field predates this option, so the historical 0.7/True pair is assumed with
    a warning, and an explicit nondefault is rejected instead of being dressed
    up as a recorded conflict.
    """
    recorded = dict(saved.config) if saved is not None else {}
    # ``0.12.0`` recorded the ratio in ``args``; the v2 delivery moved it into
    # ``config`` next to the weight-decay policy. A payload taken before the move
    # records it exactly once, there, and that record is authoritative: treating
    # it as absent rejected a matching request and re-saved a ``config`` that
    # contradicted the ``args`` it had just resumed from.
    if saved is not None and "teacher_temp_warmup_ratio" not in recorded:
        legacy_ratio = saved.saved_args.get("teacher_temp_warmup_ratio")
        if legacy_ratio is not None:
            recorded["teacher_temp_warmup_ratio"] = legacy_ratio
    # A build between ``0.12.0`` and this one read only one location and could
    # write ``config`` from the assumed default while ``args`` kept the resumed
    # value. Two resolvers would then train at one ratio and record the other, so
    # the contradiction is refused instead of being carried forward again.
    if saved is not None:
        recorded_config_ratio = saved.config.get("teacher_temp_warmup_ratio")
        recorded_args_ratio = saved.saved_args.get("teacher_temp_warmup_ratio")
        if (
            recorded_config_ratio is not None
            and recorded_args_ratio is not None
            # Exact, not approximate: the schedule is ``int(total * ratio)``, so
            # 0.3 and 0.29999999999999993 can be different numbers of warmup
            # steps. Two records that differ at all cannot both be trusted.
            and float(recorded_config_ratio) != float(recorded_args_ratio)
        ):
            raise ValueError(
                "Cannot resume: the checkpoint records contradictory teacher "
                f"temperatures (config.teacher_temp_warmup_ratio="
                f"{recorded_config_ratio!r}, args.teacher_temp_warmup_ratio="
                f"{recorded_args_ratio!r}). One of them did not drive training, "
                "so neither can be trusted; start a new run."
            )
    # Presence is decided by the key, never by whether its value is null: a
    # recorded-but-null field is damaged metadata, not a historical payload.
    missing = [name for name in _COMMON_RECIPE_OPTIONS if name not in recorded]
    # A ``config`` carrying exactly one of the two keys is damaged v2 metadata.
    # One carrying neither predates both fields and is handled below.
    if saved is not None and (
        len([name for name in _COMMON_RECIPE_OPTIONS if name in saved.config]) == 1
    ):
        raise ValueError(
            "Cannot resume: the checkpoint records only part of the shared "
            "pretrain recipe (missing "
            + next(name for name in _COMMON_RECIPE_OPTIONS if name not in saved.config)
            + "). Partial recipe metadata is not resumable; start a new run."
        )
    # The command line spells the weight-decay control negatively, like
    # --disable-cross-view-loss; the persisted key stays positive.
    no_decay_requested = bool(
        _ssl_option(args, "no_norm_bias_weight_decay", False)
    )
    no_decay_explicit = _ssl_explicit(args, "no_norm_bias_weight_decay")
    if saved is not None and len(missing) == len(_COMMON_RECIPE_OPTIONS):
        ratio = _validate_recipe_ratio(
            _ssl_option(
                args, "teacher_temp_warmup_ratio", RECIPE_TEACHER_TEMP_WARMUP_RATIO
            )
        )
        if _ssl_explicit(args, "teacher_temp_warmup_ratio") and (
            ratio != RECIPE_TEACHER_TEMP_WARMUP_RATIO
        ):
            raise ValueError(
                "The checkpoint does not record teacher_temp_warmup_ratio; "
                "historical behavior is assumed to be "
                f"{RECIPE_TEACHER_TEMP_WARMUP_RATIO}. Explicit nondefault values "
                "are conservatively rejected to avoid changing an unrecorded "
                "temperature schedule. Start a new run to use a different ratio."
            )
        if no_decay_explicit and no_decay_requested:
            raise ValueError(
                "The checkpoint does not record norm_bias_weight_decay; "
                "historical behavior is assumed to be True. Changing this "
                "policy would require an unsupported optimizer-partition "
                "conversion. Start a new run to change Norm/bias decay."
            )
        print(
            "[Warning] The checkpoint records neither teacher_temp_warmup_ratio "
            "nor norm_bias_weight_decay; assuming the historical 0.7/True pair "
            "and its single all-parameter optimizer group. A run from the "
            "temporary hard-coded 0.3 temperature experiment cannot be "
            "continued exactly."
        )
        return {
            "teacher_temp_warmup_ratio": RECIPE_TEACHER_TEMP_WARMUP_RATIO,
            "norm_bias_weight_decay": RECIPE_NORM_BIAS_WEIGHT_DECAY,
        }
    # The recorded values are validated as stored before any comparison with the
    # request, so a null or boolean could never be replaced by a requested value
    # that merely compares equal (True == 1.0, False == 0.0).
    recorded_ratio = (
        _validate_recipe_ratio(recorded["teacher_temp_warmup_ratio"])
        if "teacher_temp_warmup_ratio" in recorded
        else None
    )
    recorded_wd = (
        _validate_recipe_wd_flag(recorded["norm_bias_weight_decay"])
        if "norm_bias_weight_decay" in recorded
        else None
    )
    # The shared helper is not used here: it reads the recorded value with
    # ``.get()`` (a recorded null would look absent) and compares the request
    # numerically (True == 1.0). The recorded ratio is already validated above,
    # so this decision only compares strict, non-null values.
    requested_ratio = _validate_recipe_ratio(
        _ssl_option(
            args, "teacher_temp_warmup_ratio", RECIPE_TEACHER_TEMP_WARMUP_RATIO
        )
    )
    if not _ssl_explicit(args, "teacher_temp_warmup_ratio"):
        ratio = requested_ratio if recorded_ratio is None else recorded_ratio
    else:
        if recorded_ratio is not None and requested_ratio != recorded_ratio:
            raise ValueError(
                "Cannot change teacher-temp-warmup-ratio on resume: the "
                f"checkpoint records teacher_temp_warmup_ratio={recorded_ratio!r}, "
                f"got {requested_ratio!r}."
            )
        ratio = requested_ratio
    if no_decay_explicit:
        requested_wd = not no_decay_requested
        if (
            saved is not None
            and recorded_wd is None
            and requested_wd != RECIPE_NORM_BIAS_WEIGHT_DECAY
        ):
            # A payload without the key predates the policy, so its single
            # all-parameter optimizer group is the only partition it can have.
            raise ValueError(
                "The checkpoint does not record norm_bias_weight_decay; "
                "historical behavior is assumed to be True. Changing this "
                "policy would require an unsupported optimizer-partition "
                "conversion. Start a new run to change Norm/bias decay."
            )
        if recorded_wd is not None and requested_wd != recorded_wd:
            raise ValueError(
                "Cannot change --no-norm-bias-weight-decay on resume: the "
                f"checkpoint records norm_bias_weight_decay={recorded_wd!r}. "
                "Switching the policy would require converting the saved "
                "optimizer partition; start a new run instead."
            )
        wd = requested_wd
    elif recorded_wd is not None:
        wd = recorded_wd
    else:
        wd = not no_decay_requested
    return {
        "teacher_temp_warmup_ratio": ratio,
        "norm_bias_weight_decay": wd,
    }


def _validate_koleo_weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"koleo_weight must be a number, got {value!r}.")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"koleo_weight must be a finite value >= 0, got {value!r}."
        )
    return weight


def _resolve_relational_ssl_config(
    args: argparse.Namespace, saved: _SavedSSLState | None
) -> dict[str, Any]:
    for name in _SSL_DINO_ONLY:
        if _ssl_explicit(args, name):
            raise ValueError(
                f"--{name.replace('_', '-')} is not applicable to "
                "--ssl-method relational; omit it or use --ssl-method dino."
            )
    out_dim = _positive_int(_ssl_option(args, "out_dim", 256))
    if out_dim is None:
        raise ValueError(f"--out-dim must be a positive integer, got {args.out_dim!r}.")
    if saved is not None:
        if (
            saved.out_dim is not None
            and saved.weight_out_dim is not None
            and saved.out_dim != saved.weight_out_dim
        ):
            raise ValueError(
                "Cannot resume: the checkpoint config records out_dim="
                f"{saved.out_dim} but its stored projector is "
                f"{saved.weight_out_dim} wide. The metadata and the weights "
                "disagree, so neither width can be trusted."
            )
        saved_width = saved.out_dim if saved.out_dim is not None else saved.weight_out_dim
        if saved_width is not None and saved_width != out_dim:
            recorded = (
                f"records out_dim={saved.out_dim}"
                if saved.out_dim is not None
                else f"stores a {saved.weight_out_dim}-wide projector"
            )
            raise ValueError(
                f"Cannot resume: the checkpoint {recorded} but this invocation "
                f"requests --out-dim {out_dim}. Relational resume keeps the "
                "requested width instead of inheriting the saved one; pass "
                f"--out-dim {saved_width} to resume it."
            )
    return {
        "ssl_method": "relational",
        "ssl_head": SSL_HEAD_BY_METHOD["relational"],
        "out_dim": out_dim,
        KOLEO_WEIGHT_KEY: _resolve_relational_koleo_weight(args, saved),
        **KOLEO_FIXED_CONFIG,
        **_resolve_common_recipe(args, saved),
    }


def _saved_koleo_bundle(config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Read the four KoLeo keys as one bundle from a checkpoint config.

    Relational's canonical zero-weight encoding is *absence* of all four keys,
    which is also what every pre-extension relational checkpoint records. A
    partial bundle is damaged metadata, never a value to fill in from the
    invocation or from current defaults.

    Presence is decided by the key, not by the value: an explicit ``None`` under
    a present key is damaged metadata, not the "absent" encoding.
    """
    present = [name for name in KOLEO_STORED_KEYS if name in config]
    if not present:
        return False, {}
    if len(present) != len(KOLEO_STORED_KEYS):
        missing = [name for name in KOLEO_STORED_KEYS if name not in config]
        raise ValueError(
            "Cannot resume: the checkpoint records a partial KoLeo bundle "
            f"(missing {', '.join(missing)}). Record all four keys or none; a "
            "partial bundle cannot be trusted."
        )
    return True, {name: config[name] for name in KOLEO_STORED_KEYS}


def _validate_relational_koleo_bundle(bundle: dict[str, Any]) -> float:
    """Validate a complete relational bundle and return its weight.

    A recorded relational bundle describes an *enabled* regularizer, so the
    weight must be strictly positive; zero is encoded by absence instead.
    """
    weight = _validate_koleo_weight(bundle[KOLEO_WEIGHT_KEY])
    if weight <= 0.0:
        raise ValueError(
            "Cannot resume: the checkpoint records a complete KoLeo bundle with "
            f"koleo_weight={bundle[KOLEO_WEIGHT_KEY]!r}. A relational checkpoint "
            "encodes a disabled regularizer by recording no KoLeo keys at all, "
            "so this metadata is inconsistent."
        )
    for name, value in KOLEO_FIXED_CONFIG.items():
        recorded = bundle[name]
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
            raise ValueError(
                f"Cannot resume: the checkpoint records {name}={recorded!r}, "
                "which is not a number."
            )
        if not math.isfinite(float(recorded)) or float(recorded) <= 0.0:
            raise ValueError(
                f"Cannot resume: the checkpoint records {name}={recorded!r}, "
                "which is not a supported positive epsilon."
            )
        if recorded != value:
            raise ValueError(
                f"Cannot resume: the checkpoint records {name}={recorded!r} "
                f"but this version supports {value!r}. The KoLeo formula cannot "
                "change silently on resume."
            )
    return weight


def _resolve_relational_koleo_weight(
    args: argparse.Namespace, saved: _SavedSSLState | None
) -> float:
    """KoLeo weight for relational: absent bundle means disabled, forever.

    Relational's zero-weight encoding is an absent four-key bundle, so a
    checkpoint that carries no bundle cannot be upgraded to an enabled run; a
    positive weight requires a new run instead. This holds for direct trainer
    entry points too, not only for an explicit ``--koleo-weight`` flag: the
    recorded state, not the explicitness marker, decides.
    """
    requested = getattr(args, KOLEO_WEIGHT_KEY, None)
    explicit = _ssl_explicit(args, KOLEO_WEIGHT_KEY)
    if saved is None:
        return _validate_koleo_weight(
            KOLEO_DEFAULT_WEIGHT if requested is None else requested
        )
    enabled, bundle = _saved_koleo_bundle(saved.config)
    if not enabled:
        # Omission resolves to zero; only an actual request may be refused. A
        # caller that passes 0.1 without the explicitness marker still asked
        # for an enabled regularizer, so it must not fall through.
        wanted = KOLEO_DEFAULT_WEIGHT if requested is None else _validate_koleo_weight(requested)
        if wanted > 0.0:
            raise ValueError(
                "Cannot enable --koleo-weight on resume: this relational "
                "checkpoint records no KoLeo recipe. Start a new run to use it."
            )
        return wanted
    recorded = _validate_relational_koleo_bundle(bundle)
    if not explicit:
        return recorded
    resolved = _validate_koleo_weight(requested)
    if resolved != recorded:
        raise ValueError(
            "Cannot change koleo-weight on resume: the checkpoint records "
            f"{recorded!r}, got {resolved!r}. A changed weight requires a new run."
        )
    return resolved


def _resolve_dino_ssl_config(
    args: argparse.Namespace,
    saved: _SavedSSLState | None,
    resume_ckpt: dict[str, Any] | None,
    model_name: str,
) -> dict[str, Any]:
    from otuformer.training.dino import (
        DINO_DEFAULTS,
        DINO_FIXED_CONFIG,
        preflight_dino_prototypes,
    )

    for name in _SSL_RELATIONAL_ONLY:
        if _ssl_explicit(args, name):
            raise ValueError(
                f"--{name.replace('_', '-')} is not applicable to "
                "--ssl-method dino; omit it or use --ssl-method relational."
            )
    out_dim = _positive_int(
        _resolve_inherited_ssl_value(args, "out_dim", 256, saved)
    )
    if out_dim is None:
        raise ValueError("--out-dim must be a positive integer.")
    values = {
        name: _resolve_inherited_ssl_value(args, name, default, saved)
        for name, default in DINO_DEFAULTS.items()
    }
    # KoLeo belongs to the shared pretrain owner, not to ``dino.py``.
    values[KOLEO_WEIGHT_KEY] = _resolve_inherited_ssl_value(
        args, KOLEO_WEIGHT_KEY, KOLEO_DEFAULT_WEIGHT, saved
    )
    prototypes = _positive_int(values["dino_prototypes"])
    if prototypes is None or prototypes < DINO_PROTOTYPES_MIN:
        raise ValueError(
            f"--dino-prototypes must be an integer >= {DINO_PROTOTYPES_MIN}, "
            f"got {values['dino_prototypes']!r}."
        )
    if values["dino_centering"] not in DINO_CENTERING_MODES:
        raise ValueError(
            f"--dino-centering must be one of {', '.join(DINO_CENTERING_MODES)}, "
            f"got {values['dino_centering']!r}."
        )
    koleo_weight = _validate_koleo_weight(values[KOLEO_WEIGHT_KEY])

    effective: dict[str, Any] = {
        "ssl_method": "dino",
        "ssl_head": SSL_HEAD_BY_METHOD["dino"],
        "out_dim": out_dim,
        "dino_prototypes": prototypes,
        "dino_centering": values["dino_centering"],
        KOLEO_WEIGHT_KEY: koleo_weight,
        **_resolve_common_recipe(args, saved),
    }
    if effective["dino_centering"] == "ema":
        effective["dino_center_momentum"] = 0.9
    else:
        effective["dino_sinkhorn_iterations"] = 3
    effective.update(DINO_FIXED_CONFIG)
    effective.update(KOLEO_FIXED_CONFIG)

    if saved is not None and saved.identity is not None and saved.identity[0] == "dino":
        # A DINO payload records its whole recipe; a missing field is damaged
        # metadata, never a reason to fall back to this invocation or a default.
        mode_field = (
            "dino_center_momentum"
            if saved.config.get("dino_centering") == "ema"
            else "dino_sinkhorn_iterations"
        )
        required = (
            "out_dim",
            "dino_prototypes",
            "dino_centering",
            KOLEO_WEIGHT_KEY,
            mode_field,
            *DINO_FIXED_CONFIG,
            # The four KoLeo keys moved to the shared owner above; the strict
            # DINO required-field check must still demand them.
            *KOLEO_STORED_KEYS,
        )
        missing = [name for name in required if saved.config.get(name) is None]
        if missing:
            raise ValueError(
                "Cannot resume DINO: the checkpoint records no "
                + ", ".join(missing)
                + ". A DINO payload must record its effective recipe; values are "
                "never filled in from this invocation or from current defaults."
            )

    if saved is not None:
        # The mode-specific constants are fixed settings too: a checkpoint that
        # records a different momentum/iteration count cannot be resumed at this
        # version's value without silently changing its recipe.
        mode_fixed = {"dino_center_momentum": 0.9, "dino_sinkhorn_iterations": 3}
        for name, value in {
            **DINO_FIXED_CONFIG,
            **KOLEO_FIXED_CONFIG,
            **mode_fixed,
        }.items():
            recorded = saved.config.get(name)
            if recorded is None:
                continue
            # ``dino_weight_norm_api`` names a parameterisation, not an
            # implementation: releases that recorded ``legacy`` used the same
            # weight_g/weight_v layout this version builds directly.
            matches = (
                check_dino_weight_norm_api_record(recorded)
                if name == "dino_weight_norm_api"
                else recorded == value
            )
            if not matches:
                raise ValueError(
                    f"Checkpoint records {name}={recorded!r}, but this version "
                    f"implements only {value!r}; fixed DINO settings are not "
                    "migrated between versions."
                )
        if saved.identity is None:
            raise ValueError(
                "Cannot resume as DINO: the checkpoint records no recognizable "
                "SSL identity, so its method and prototype state cannot be "
                "validated. Start a new run instead."
            )

    preflight_dino_prototypes(out_dim, prototypes, str(effective["dino_centering"]))

    if saved is not None and saved.identity is not None and saved.identity[0] == "dino":
        # The checkpoint's recorded architecture wins over the caller default,
        # so the validation probe cannot be built from the wrong backbone.
        resume_model = str(saved.config.get("model_name") or model_name)
        _validate_dino_resume_payload(
            resume_ckpt,
            effective,
            resume_model,
            getattr(args, "max_epochs", None),
        )
    return effective


def _assert_complete_ssl_projector(
    state: dict[str, Any], model: OTUFormerEncoder, out_dim: int, label: str
) -> None:
    expected = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith("projector.")
    }
    missing = sorted(set(expected) - set(state))
    if missing:
        raise ValueError(
            f"DINO resume state {label!r} is missing projector parameter(s) "
            f"{missing[:2]}; a complete D-dimensional MLP is required."
        )
    unexpected = sorted(
        key for key in state if key.startswith("projector.") and key not in expected
    )
    if unexpected:
        raise ValueError(
            f"DINO resume state {label!r} stores unexpected projector parameter(s) "
            f"{unexpected[:2]}; the prototype layer belongs to dino_objective."
        )
    for key, value in expected.items():
        if tuple(state[key].shape) != tuple(value.shape):
            raise ValueError(
                f"DINO resume state {label!r} has {key} with shape "
                f"{tuple(state[key].shape)} instead of {tuple(value.shape)}; the "
                f"recorded out_dim={out_dim} does not fit the saved weights."
            )


def _validate_dino_objective_state(
    state: dict[str, Any], out_dim: int, prototypes: int, centering: str
) -> None:
    expected = {
        "student_prototypes.weight_g": (prototypes, 1),
        "student_prototypes.weight_v": (prototypes, out_dim),
        "teacher_prototypes.weight_g": (prototypes, 1),
        "teacher_prototypes.weight_v": (prototypes, out_dim),
    }
    missing = sorted(set(expected) - set(state))
    if missing:
        raise ValueError(
            "DINO objective state is incomplete: missing " + ", ".join(missing) + "."
        )
    for key, shape in expected.items():
        if tuple(state[key].shape) != shape:
            raise ValueError(
                f"DINO objective state {key} has shape "
                f"{tuple(state[key].shape)} instead of {shape}."
            )
    for key in ("student_prototypes.weight_g", "teacher_prototypes.weight_g"):
        if not torch.equal(state[key], torch.ones_like(state[key])):
            raise ValueError(
                f"DINO objective state {key} must be exactly one; the fixed "
                "prototype scale is never inherited or migrated."
            )
    has_center = "cls_center" in state
    if centering == "ema":
        if not has_center:
            raise ValueError(
                "DINO EMA centering requires the 'cls_center' objective buffer."
            )
        if tuple(state["cls_center"].shape) != (1, prototypes):
            raise ValueError(
                "DINO objective 'cls_center' has shape "
                f"{tuple(state['cls_center'].shape)} instead of (1, {prototypes})."
            )
    elif has_center:
        raise ValueError(
            "Sinkhorn-Knopp DINO checkpoints must not store an EMA 'cls_center'."
        )
    unexpected = sorted(set(state) - set(expected) - {"cls_center"})
    if unexpected:
        raise ValueError(
            "DINO objective state stores unexpected entry " + unexpected[0] + "."
        )


def _validate_adamw_param_group(group: dict, label: str) -> None:
    """Require the AdamW hyperparameter fields a resumed step() reads.

    ``Optimizer.load_state_dict()`` accepts a parameter group without them and
    training then raises ``KeyError`` on the first ``step()``.
    """
    if not isinstance(group, dict):
        raise ValueError(f"{label} requires an optimizer parameter group mapping.")
    # lr, betas, eps and weight_decay are read by AdamW.step(); the remaining
    # flags fall back to the optimizer defaults, so only these four are required.
    missing = [
        name for name in ("lr", "betas", "eps", "weight_decay") if name not in group
    ]
    if missing:
        raise ValueError(
            f"{label} requires a complete AdamW parameter group; missing "
            + ", ".join(missing)
            + "."
        )

    def finite(name: str, minimum: float, strict: bool) -> float:
        value = group[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{label} requires a numeric AdamW {name}, got {value!r}."
            )
        value = float(value)
        lower_ok = value > minimum if strict else value >= minimum
        if not math.isfinite(value) or not lower_ok:
            raise ValueError(
                f"{label} requires a finite AdamW {name} "
                f"{'>' if strict else '>='} {minimum}, got {value!r}."
            )
        return value

    finite("lr", 0.0, False)
    finite("eps", 0.0, True)
    finite("weight_decay", 0.0, False)
    betas = group["betas"]
    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) < 1.0
            for value in betas
        )
    ):
        raise ValueError(
            f"{label} requires AdamW betas as two finite values in [0, 1), "
            f"got {betas!r}."
        )


def _continues_recorded_plan(epoch: int, recorded_epochs: int | None) -> bool:
    """True when the checkpoint was not produced by an extended run.

    Only an unextended payload describes a rectangular ``original_max_epochs *
    steps_per_epoch`` plan. ``epoch < original_max_epochs - 1`` proves that: an
    extended checkpoint is saved at the original plan's final epoch, so a later
    epoch means the loader length already changed. The unknown-``args`` case is
    treated as a same-plan resume, matching ``_build_pretrain_schedules``.
    """
    if recorded_epochs is None:
        return True
    return epoch < int(recorded_epochs) - 1


def _validate_saved_progress(
    resume_ckpt: dict[str, Any], requested_max_epochs: int | None, label: str
) -> None:
    """Reject progress that no schedule can continue from, before any output.

    Uses only values already recorded in the payload, so it can never depend on
    the current DataLoader length. Extensions are legitimate for every
    ``step >= current`` target: ``--max-epochs`` only chooses the target, so it
    can neither create nor remove remaining steps.
    """
    epoch = resume_ckpt.get("epoch")
    iteration = resume_ckpt.get("iteration")
    schedule = resume_ckpt.get("schedule")
    for name, value in (("epoch", epoch), ("iteration", iteration)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{label} requires {name} to be an integer >= 0, got {value!r}."
            )
    if not isinstance(epoch, int) or not isinstance(iteration, int):
        return
    if iteration < epoch + 1:
        raise ValueError(
            f"{label} requires iteration to cover the completed epochs; got "
            f"epoch={epoch} and iteration={iteration}."
        )
    if not isinstance(schedule, dict):
        return
    counts = {}
    for name in ("original_max_epochs", "total_steps", "steps_per_epoch"):
        value = schedule.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            # A method-specific preflight reports malformed metadata in detail.
            return
        counts[name] = value
    original_epochs = counts["original_max_epochs"]
    total_steps = counts["total_steps"]
    steps_per_epoch = counts["steps_per_epoch"]
    saved_args = resume_ckpt.get("args")
    recorded_epochs = (
        saved_args.get("max_epochs") if isinstance(saved_args, dict) else None
    )
    if isinstance(recorded_epochs, bool) or not isinstance(recorded_epochs, int):
        recorded_epochs = None
    if iteration > total_steps:
        raise ValueError(
            f"{label} found iteration={iteration} beyond the recorded plan of "
            f"{total_steps} steps; the checkpoint cannot be resumed."
        )
    # A completed epoch only proves `steps_per_epoch` steps when the payload still
    # describes the original rectangle. An extension keeps its earlier, possibly
    # shorter DataLoader, so iteration counts a mixed number of steps and only
    # the at-least-one-step-per-epoch bound above applies.
    rectangular = (
        recorded_epochs == original_epochs
        and total_steps == original_epochs * steps_per_epoch
    )
    if rectangular and iteration < (epoch + 1) * steps_per_epoch:
        raise ValueError(
            f"{label} found iteration={iteration}, which does not cover the "
            f"{epoch + 1} completed epochs of {steps_per_epoch} recorded steps "
            "each."
        )
    # ``_build_pretrain_schedules`` has exactly two branches, so the progress
    # check mirrors them instead of inventing a third rule.
    if requested_max_epochs is None:
        # No target chosen; only the plan-independent bounds above apply.
        return
    if requested_max_epochs < original_epochs:
        # Already rejected by the schedule builder, but checked here so the
        # failure happens before any output exists.
        raise ValueError(
            f"{label} cannot decrease the original plan: --max-epochs "
            f"{requested_max_epochs} < recorded {original_epochs}."
        )
    if requested_max_epochs == original_epochs:
        capacity = (epoch + 1) * steps_per_epoch
        if iteration > capacity:
            # The rebuilt schedule is the original rectangle, so it cannot cover
            # both the recorded progress and the epochs that still have to train.
            raise ValueError(
                f"{label} found iteration={iteration} at epoch={epoch}, but a "
                f"same-plan resume leaves only {capacity} steps for the epochs "
                f"already completed. Use --max-epochs > {original_epochs} to "
                "extend a run whose DataLoader length changed."
            )
        # Same-plan resume: the schedule is the original rectangle, so progress
        # must sit exactly on a recorded epoch boundary. Any other value leaves
        # too few schedule entries for the epochs that still have to train.
        expected = (epoch + 1) * steps_per_epoch
        if rectangular and iteration != expected:
            raise ValueError(
                f"{label} found iteration={iteration} at epoch={epoch}, but the "
                f"recorded {original_epochs} x {steps_per_epoch} plan places that "
                f"boundary at {expected}. Use --max-epochs > {original_epochs} to "
                "extend a run whose loader length changed."
            )
        return
    # Extension: the schedule is rebuilt from `iteration` onward, so any recorded
    # iteration is resumable as long as a further epoch is requested.
    if requested_max_epochs <= epoch + 1:
        raise ValueError(
            f"{label} cannot continue: the checkpoint completed epoch {epoch}, "
            f"so --max-epochs must exceed {epoch + 1}."
        )


def _validate_dino_resume_payload(
    resume_ckpt: dict[str, Any] | None,
    effective: dict[str, Any],
    model_name: str,
    requested_max_epochs: int | None,
) -> None:
    """Strictly validate a DINO resume before any output is created."""
    from otuformer.utils.checkpoint import preserved_rng, resolve_register_layout

    if resume_ckpt is None:
        return
    states: dict[str, dict[str, Any]] = {}
    for key in _REGISTER_RESUME_KEYS:
        state = resume_ckpt.get(key)
        if not isinstance(state, dict) or not state:
            raise ValueError(
                f"DINO resume requires a complete {key!r} encoder state; the "
                "inference alias cannot substitute for a missing encoder."
            )
        states[key] = state
    objective_state = resume_ckpt.get("dino_objective")
    if not isinstance(objective_state, dict) or not objective_state:
        raise ValueError(
            "DINO resume requires the training-only 'dino_objective' state."
        )
    # Optimizer/schedule/progress must be present and structurally usable, so an
    # incomplete payload cannot reach the training loop or append to a log.
    for key in ("optimizer", "schedule", "epoch", "iteration", "rng_state"):
        if resume_ckpt.get(key) is None:
            raise ValueError(
                f"DINO resume requires the saved {key!r} state; an incomplete "
                "DINO payload is not resumable."
            )
    rng_state = resume_ckpt["rng_state"]
    if not isinstance(rng_state, dict) or not {"python", "numpy", "torch"} <= set(
        rng_state
    ):
        raise ValueError(
            "DINO resume requires a complete rng_state (python, numpy, torch); "
            "resuming without it would restart the mask and shuffle sequence."
        )
    # Reuse the actual restore rule inside an RNG-preserving guard: a null or
    # malformed stream fails here instead of mid-training.
    from otuformer.utils.checkpoint import preserved_rng

    try:
        with preserved_rng():
            _restore_rng_state(rng_state)
    except Exception as exc:
        raise ValueError(
            f"DINO resume requires a restorable rng_state: {exc}"
        ) from exc

    def integer(name: str, value: Any, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"DINO resume requires {name} to be an integer >= {minimum}, "
                f"got {value!r}."
            )
        return value

    epoch = integer("epoch", resume_ckpt["epoch"], 0)
    iteration = integer("iteration", resume_ckpt["iteration"], 0)
    if iteration < epoch + 1:
        raise ValueError(
            "DINO resume requires iteration to cover the completed epochs; "
            f"got epoch={epoch} and iteration={iteration}."
        )

    optimizer_state = resume_ckpt["optimizer"]
    if (
        not isinstance(optimizer_state, dict)
        or not optimizer_state.get("param_groups")
        or not isinstance(optimizer_state.get("state"), dict)
        or not optimizer_state["state"]
    ):
        raise ValueError(
            "DINO resume requires a complete optimizer state with parameter "
            "groups and non-empty parameter state."
        )
    group_params: list[Any] = []
    for index, group in enumerate(optimizer_state["param_groups"]):
        # The group count, envelope and ordered ids are validated against the
        # rebuilt partition below.
        _validate_adamw_param_group(group, "DINO resume")
        saved_params = group.get("params")
        if (
            not isinstance(saved_params, list)
            or not saved_params
            or len(set(map(str, saved_params))) != len(saved_params)
        ):
            raise ValueError(
                f"DINO resume requires optimizer group {index} to list "
                f"non-empty, non-duplicate parameter ids, got {saved_params!r}."
            )
        group_params.extend(saved_params)
    orphaned = sorted(
        str(key)
        for key in optimizer_state["state"]
        if str(key) not in {str(item) for item in group_params}
    )
    if orphaned:
        # load_state_dict() would silently drop these; the run would then train
        # with a different parameter group than the one recorded.
        raise ValueError(
            "DINO resume found optimizer state for parameters outside the saved "
            f"parameter group: {', '.join(orphaned[:3])}."
        )

    schedule_state = resume_ckpt["schedule"]
    required_schedule = (
        "original_max_epochs",
        "total_steps",
        "steps_per_epoch",
        "warmup_steps",
        "last_lr",
        "final_lr",
    )
    missing_schedule = [
        key
        for key in required_schedule
        if not isinstance(schedule_state, dict) or key not in schedule_state
    ]
    if missing_schedule:
        raise ValueError(
            "DINO resume requires complete schedule metadata; missing "
            + ", ".join(missing_schedule)
            + "."
        )

    def schedule_count(name: str, minimum: int) -> int:
        value = schedule_state[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"DINO resume requires schedule.{name} to be an integer, got "
                f"{value!r}."
            )
        if value < minimum:
            raise ValueError(
                f"DINO resume requires schedule.{name} >= {minimum}, got {value!r}."
            )
        return value

    def schedule_lr(name: str) -> float:
        value = schedule_state[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"DINO resume requires a numeric schedule.{name}, got {value!r}."
            )
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"DINO resume requires schedule.{name} to be finite and >= 0, "
                f"got {value!r}."
            )
        return float(value)

    original_epochs = schedule_count("original_max_epochs", 1)
    total_steps = schedule_count("total_steps", 1)
    steps_per_epoch = schedule_count("steps_per_epoch", 1)
    warmup_steps = schedule_count("warmup_steps", 0)
    schedule_lr("last_lr")
    schedule_lr("final_lr")
    # Cross-field consistency: the recorded plan must be able to contain the
    # completed epochs and the warmup it claims.
    if total_steps < original_epochs or total_steps < steps_per_epoch:
        raise ValueError(
            "DINO resume requires schedule.total_steps to cover the recorded "
            f"plan; got total_steps={total_steps}, "
            f"original_max_epochs={original_epochs}, "
            f"steps_per_epoch={steps_per_epoch}."
        )
    saved_args = resume_ckpt.get("args")
    recorded_epochs = (
        saved_args.get("max_epochs", original_epochs)
        if isinstance(saved_args, dict) else original_epochs
    )
    # Extensions retain original_max_epochs but can change DataLoader length;
    # only an unextended plan has the original rectangular step count.
    if recorded_epochs == original_epochs and total_steps != original_epochs * steps_per_epoch:
        raise ValueError(
            "DINO resume requires schedule.total_steps to equal "
            "original_max_epochs * steps_per_epoch for the original plan."
        )
    # warmup_steps describes the original rectangle; an extension keeps it while
    # total_steps covers the new, possibly shorter plan.
    if recorded_epochs == original_epochs and warmup_steps > total_steps:
        raise ValueError(
            "DINO resume requires schedule.warmup_steps <= total_steps; got "
            f"{warmup_steps} > {total_steps}."
        )
    _validate_saved_progress(
        resume_ckpt, requested_max_epochs, "DINO resume"
    )
    if "center" in resume_ckpt:
        raise ValueError(
            "DINO resume must not carry a top-level 'center'; the K-dimensional "
            "center belongs to dino_objective."
        )

    out_dim = int(effective["out_dim"])
    image_size = resolve_training_image_size(resume_ckpt)
    with preserved_rng():
        probe = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
        )
    layouts = {
        key: resolve_register_layout(resume_ckpt, state, probe.backbone)
        for key, state in states.items()
    }
    if len(set(layouts.values())) != 1:
        raise ValueError(
            "DINO resume encoder states disagree on the register layout: "
            + json.dumps({key: list(value) for key, value in layouts.items()})
            + "."
        )
    added = layouts["model_state_dict"][1]
    with preserved_rng():
        expected = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
            **({"reg_tokens": 4} if added else {}),
        )
    # The restore is strict, so the envelope must match exactly: every expected
    # tensor present, and no unknown tensor beyond the training-only normalization
    # that masked-feature checkpoints add.
    expected_state = expected.state_dict()
    norm_state = {
        f"_pretrain_final_norm.{name}": value
        for name, value in expected.backbone.norm.state_dict().items()
    }
    for key, state in states.items():
        restore_state = dict(expected_state)
        # ``validate_pretrain_backbone()`` registers the alias, so the strict
        # restore always expects it. Accepting a partial or absent alias here
        # would defer a guaranteed KeyError/missing-keys failure to mid-training.
        restore_state.update(norm_state)
        missing = sorted(set(restore_state) - set(state))
        unexpected = sorted(set(state) - set(restore_state))
        mismatched = sorted(
            name for name in set(restore_state) & set(state)
            if not isinstance(state[name], torch.Tensor)
            or state[name].shape != restore_state[name].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                f"DINO resume state {key!r} does not match a strict restore: "
                f"{len(missing)} missing (e.g. {missing[:2]}), "
                f"{len(unexpected)} unexpected (e.g. {unexpected[:2]}) and "
                f"{len(mismatched)} shape mismatches (e.g. {mismatched[:2]})."
            )
        _assert_encoder_state_matches(state, expected, key)
        _assert_complete_ssl_projector(state, expected, out_dim, key)
        center = state.get("center")
        if isinstance(center, torch.Tensor) and int(torch.count_nonzero(center)) != 0:
            raise ValueError(
                f"DINO resume requires a zero encoder center in {key!r}; the "
                "K-dimensional center lives in dino_objective."
            )
    _validate_dino_objective_state(
        objective_state,
        out_dim,
        int(effective["dino_prototypes"]),
        str(effective["dino_centering"]),
    )

    # AdamW restores state by parameter-group position, not by shape. Rebuild
    # the same encoder/patch/prototype order used by the training optimizer.
    from otuformer.training.dino import DINOObjective as _ProbeObjective

    patch_config = _resolve_patch_config(argparse.Namespace(), resume_ckpt)
    with preserved_rng():
        probe_patch = PatchObjective(
            str(patch_config["patch_loss"]),
            expected.backbone.num_features,
            patch_config["ibot_prototypes"],
        )
        probe_objective = _ProbeObjective(
            out_dim,
            int(effective["dino_prototypes"]),
            str(effective["dino_centering"]),
        )
    if probe_patch.trainable_parameters():
        try:
            probe_patch.load_state_dict(resume_ckpt.get("patch_objective", {}), strict=True)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"DINO resume has invalid patch_objective state: {exc}") from exc
    probe_groups = _pretrain_optimizer_groups(
        expected,
        probe_patch,
        probe_objective,
        weight_decay=0.05,
        norm_bias_weight_decay=bool(effective["norm_bias_weight_decay"]),
    )
    # The same layout validation the trainer uses for the real optimizer runs
    # here, so a payload it would reject never reaches output creation.
    with preserved_rng():
        probe_optimizer = torch.optim.AdamW(probe_groups, lr=0.0, weight_decay=0.0)
    _validate_pretrain_optimizer_layout(
        probe_optimizer,
        optimizer_state,
        norm_bias_weight_decay=bool(effective["norm_bias_weight_decay"]),
        decay_partition_empty=_pretrain_decay_partition_empty(
            expected, probe_patch, probe_objective
        ),
        label="DINO resume",
    )
    # AdamW indexes state across the concatenation of the built groups, so the
    # no-decay policy changes the global order the saved ids refer to.
    probe_parameters = [
        parameter for group in probe_groups for parameter in group["params"]
    ]
    if len(group_params) != len(probe_parameters):
        raise ValueError(
            f"DINO resume expects {len(probe_parameters)} optimizer parameters, "
            f"the checkpoint records {len(group_params)}."
        )
    frozen = probe_objective.trainable_parameters()[0]
    for state_key, parameter in zip(group_params, probe_parameters):
        if isinstance(state_key, bool) or not isinstance(state_key, int):
            raise ValueError("DINO resume optimizer parameter ids must be integers.")
        if state_key not in optimizer_state["state"]:
            if epoch == 0 and parameter is frozen:
                continue
            raise ValueError(
                f"DINO resume is missing AdamW state for optimizer parameter "
                f"{state_key} of shape {tuple(parameter.shape)}."
            )
        entry = optimizer_state["state"][state_key]
        if not isinstance(entry, dict):
            raise ValueError(
                f"DINO resume optimizer state {state_key!r} is not a parameter "
                "state mapping."
            )
        missing_fields = [
            field for field in ("step", "exp_avg", "exp_avg_sq") if field not in entry
        ]
        if missing_fields:
            raise ValueError(
                f"DINO resume optimizer state {state_key!r} is missing "
                + ", ".join(missing_fields) + "."
            )
        step = entry["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.ndim != 0
            or not torch.isfinite(step)
            or step.item() < 1
            or step.item() != int(step.item())
        ):
            raise ValueError(f"DINO resume optimizer state {state_key} has invalid AdamW step.")
        for name in ("exp_avg", "exp_avg_sq"):
            value = entry[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(
                    f"DINO resume optimizer state {state_key} has {name} with "
                    f"the wrong shape; expected {tuple(parameter.shape)}."
                )


def _validate_relational_optimizer_preflight(
    resume_ckpt: dict[str, Any],
    model_name: str,
    effective: dict[str, Any],
    *,
    strict_fifo_state: bool = False,
    target_device: torch.device | None = None,
) -> None:
    """Prove the saved optimizer groups map onto the rebuilt relational layout.

    The trainer runs the same check before restoring, so doing it here only
    moves an existing rejection in front of output creation.
    """
    from otuformer.utils.checkpoint import (
        preserved_rng,
        resolve_checkpoint_state_dict,
        resolve_register_layout,
    )

    state, _ = resolve_checkpoint_state_dict(resume_ckpt)
    image_size = resolve_training_image_size(resume_ckpt)
    out_dim = int(effective["out_dim"])
    policy = bool(effective["norm_bias_weight_decay"])
    # The payload's own model name wins over an omitted invocation default.
    resolved_name = resolve_checkpoint(resume_ckpt, model_name).model_name
    with preserved_rng():
        probe = OTUFormerEncoder(
            model_name=resolved_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
        )
    _, added = resolve_register_layout(resume_ckpt, state, probe.backbone)
    with preserved_rng():
        expected = OTUFormerEncoder(
            model_name=resolved_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
            **({"reg_tokens": 4} if added else {}),
        )
        patch_config = _resolve_patch_config(argparse.Namespace(), resume_ckpt)
        probe_patch = PatchObjective(
            str(patch_config["patch_loss"]),
            expected.backbone.num_features,
            patch_config["ibot_prototypes"],
        )
    probe_groups = _pretrain_optimizer_groups(
        expected,
        probe_patch,
        None,
        weight_decay=0.05,
        norm_bias_weight_decay=policy,
    )
    with preserved_rng():
        probe_optimizer = torch.optim.AdamW(probe_groups, lr=0.0, weight_decay=0.0)
    _validate_pretrain_optimizer_layout(
        probe_optimizer,
        resume_ckpt["optimizer"],
        norm_bias_weight_decay=policy,
        decay_partition_empty=_pretrain_decay_partition_empty(
            expected, probe_patch, None
        ),
        label="pretrain resume",
    )
    if not strict_fifo_state:
        return
    # Enabled recovery needs the usable state behind the layout: moments for the
    # parameters that already participated, the independent patch tensors, both
    # complete encoders and the exact relational center, all before any output
    # exists.
    completed_steps = resume_ckpt.get("iteration")
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
        completed_steps = 0
    _validate_fifo_optimizer_execution_flags(
        resume_ckpt.get("optimizer"), target_device=target_device
    )
    _validate_fifo_adamw_state(
        probe_optimizer,
        resume_ckpt.get("optimizer"),
        completed_steps=completed_steps,
        label="pretrain resume",
        target_device=target_device,
    )
    _validate_fifo_patch_state(
        probe_patch,
        resume_ckpt.get("patch_objective"),
        str(patch_config["patch_loss"]),
    )
    expected.validate_pretrain_backbone(
        require_intermediates=str(patch_config["patch_loss"]) == "masked-feature"
    )
    _validate_fifo_encoder_state("student", resume_ckpt.get("student"), expected)
    _validate_fifo_encoder_state("teacher", resume_ckpt.get("teacher"), expected)
    _validate_fifo_center(resume_ckpt.get("center"), out_dim)


def resolve_pretrain_ssl_config(
    args: argparse.Namespace,
    resume_ckpt: dict[str, Any] | None,
    *,
    relational_queue_preparation: "_RelationalQueuePreparation | None" = None,
) -> dict[str, Any]:
    """Resolve the effective SSL recipe before any output is created.

    Returns a fresh effective-config dictionary and never mutates the requested
    namespace. A recognized checkpoint supplies the method, and DINO resumes
    inherit the operator-configurable DINO values they omit.
    """
    requested_method = _ssl_option(args, "ssl_method", "relational")
    if requested_method not in SSL_METHODS:
        raise ValueError(
            f"--ssl-method must be one of {', '.join(SSL_METHODS)}, "
            f"got {requested_method!r}."
        )
    # Shared by both methods and checked before any output is cleared: a zero or
    # non-finite student temperature reaches a division, not a validation error.
    requested_student_temp = getattr(args, "student_temp", None)
    candidate_bound = (
        relational_queue_preparation.candidate_bound
        if relational_queue_preparation is not None
        else None
    )
    admission_length = (
        int(candidate_bound)
        if candidate_bound is not None
        else int(getattr(args, "batch_size", 2) or 2)
    )
    if requested_student_temp is not None:
        student_temp_value = _admissible_temperature(
            _requested_float(requested_student_temp, "--student-temp"),
            admission_length,
            "--student-temp",
        )
        if student_temp_value <= 0.0:
            raise ValueError(
                "--student-temp must be finite and greater than 0, got "
                f"{requested_student_temp!r}."
            )
    saved = _saved_ssl_state(resume_ckpt)
    saved_method = saved.identity[0] if saved is not None and saved.identity else None

    method = requested_method
    if saved_method is not None:
        if _ssl_explicit(args, "ssl_method") and requested_method != saved_method:
            raise ValueError(
                "Cannot change SSL method on resume: the checkpoint records "
                f"ssl_method={saved_method!r}, got --ssl-method "
                f"{requested_method!r}. Start a new run instead."
            )
        method = requested_method if _ssl_explicit(args, "ssl_method") else saved_method

    if resume_ckpt is not None:
        epoch = resume_ckpt.get("epoch")
        saved_args = resume_ckpt.get("args")
        _validate_pretrain_epoch_target(
            getattr(args, "max_epochs", None),
            epoch + 1 if isinstance(epoch, int) and not isinstance(epoch, bool) else 0,
            resume_ckpt.get("schedule"),
            saved_args.get("max_epochs") if isinstance(saved_args, dict) else None,
        )

    model_name = str(_ssl_option(args, "model_name", ""))
    if method == "relational":
        effective = _resolve_relational_ssl_config(args, saved)
        if resume_ckpt is not None and "optimizer" in resume_ckpt:
            # The relational path never builds the training models here, so the
            # saved groups are proven against a rebuilt parameter partition
            # before any output exists.
            _validate_relational_optimizer_preflight(
                resume_ckpt,
                model_name,
                effective,
                strict_fifo_state=bool(
                    relational_queue_preparation is not None
                    and relational_queue_preparation.active
                ),
                target_device=(
                    relational_queue_preparation.resolved_device
                    if relational_queue_preparation is not None
                    else None
                ),
            )
    else:
        effective = _resolve_dino_ssl_config(args, saved, resume_ckpt, model_name)
    return effective


def _ssl_display_config(effective: dict[str, Any]) -> dict[str, Any]:
    """Display projection: inactive DINO fields become null, never omitted."""
    display = {
        "ssl_method": effective["ssl_method"],
        "ssl_head": effective["ssl_head"],
        "out_dim": effective["out_dim"],
        "teacher_temp_warmup_ratio": effective["teacher_temp_warmup_ratio"],
        "norm_bias_weight_decay": effective["norm_bias_weight_decay"],
    }
    dino_fields = (
        "dino_prototypes",
        "dino_centering",
        "dino_hidden_dim",
        "dino_mlp_init_std",
        "dino_mlp_init_bias",
        "dino_weight_norm_api",
        "dino_prototype_scale",
        "dino_freeze_last_layer_epochs",
        "dino_target_dtype",
        "dino_center_momentum",
        "dino_sinkhorn_iterations",
    )
    active = effective["ssl_method"] == "dino"
    for name in dino_fields:
        display[name] = effective.get(name) if active else None
    # KoLeo is shared, so it is always shown with its effective value (zero
    # weight included). Display is not persistence: relational still omits the
    # four keys from the saved config when the weight is zero.
    for name in KOLEO_STORED_KEYS:
        display[name] = effective.get(name)
    return display


# Both handoffs below are one-shot inputs from this invocation's preflight,
# never caches. A caller that reuses its Namespace after changing any request
# must be resolved again, because a persisted answer would silently ignore the
# new request and skip its validation. Each is taken (and cleared) at the start
# of the public entry, so an early failure cannot leave it for a later call.
def take_pretrain_handoff(args: argparse.Namespace) -> "_PretrainHandoff":
    """Take this invocation's preflight handoff and clear the slots.

    Must run before any validation that can raise: a slot left behind after a
    rejected call would be reused by a retry with a different request.
    """
    supplied = getattr(args, "ssl_config_effective", None)
    resume_state = getattr(args, "resume_state", None)
    resume_state_path = getattr(args, "resume_state_path", None)
    preparation = vars(args).pop("relational_queue_preparation", None)
    settings = vars(args).pop("pretrain_settings", None)
    for name in ("ssl_config_effective", "resume_state", "resume_state_path"):
        setattr(args, name, None)
    return _PretrainHandoff(
        ssl_config=supplied if isinstance(supplied, dict) and supplied else None,
        resume_state=resume_state,
        resume_state_path=resume_state_path,
        settings=settings,
        relational_queue_preparation=(
            preparation
            if isinstance(preparation, _RelationalQueuePreparation)
            else None
        ),
    )


@dataclass(frozen=True)
class _PretrainHandoff:
    ssl_config: dict[str, Any] | None
    resume_state: dict[str, Any] | None
    resume_state_path: str | None
    settings: "PretrainSettings | None"
    relational_queue_preparation: "_RelationalQueuePreparation | None" = None


def resume_payload_for(
    args: argparse.Namespace, handoff: _PretrainHandoff | None = None
) -> dict[str, Any] | None:
    """Return the resume payload for the *current* ``args.resume``.

    A handed-over payload is used only when it came from the path still
    requested: the same path may hold a newer checkpoint, so it is read again
    on the next invocation rather than cached for the process lifetime.
    """
    requested = str(getattr(args, "resume", "") or "")
    if not requested:
        return None
    if handoff is not None and handoff.resume_state is not None:
        if handoff.resume_state_path == requested:
            return handoff.resume_state
    return load_checkpoint(Path(requested))


#: Sentinel: no payload was supplied by the caller.
_UNLOADED = object()


def _effective_pretrain_ssl_config(
    args: argparse.Namespace,
    handoff: _PretrainHandoff | None = None,
    resume_ckpt: Any = _UNLOADED,
    *,
    relational_queue_preparation: "_RelationalQueuePreparation | None" = None,
) -> dict[str, Any]:
    """Use this invocation's handed-over recipe, else resolve before output.

    ``resume_ckpt`` is the payload the caller already loaded for *this*
    invocation. Passing it keeps validation and the later restore on one
    checkpoint: reloading here could validate a different file than the one the
    run restores from.
    """
    if handoff is None:
        handoff = take_pretrain_handoff(args)
    if handoff.ssl_config is not None:
        return handoff.ssl_config
    if relational_queue_preparation is None:
        relational_queue_preparation = handoff.relational_queue_preparation
    if resume_ckpt is _UNLOADED:
        resume_ckpt = resume_payload_for(args, handoff)
    return resolve_pretrain_ssl_config(
        args,
        resume_ckpt,
        relational_queue_preparation=relational_queue_preparation,
    )


_UNSAVED_ARGS = frozenset(
    {
        "resume_state",
        "resume_state_path",
        "pretrain_settings",
        "ssl_config_effective",
        "relational_queue_preparation",
    }
)


_TRANSIENT_ARGS = frozenset(
    {
        "pretrain_recipe",
        "include_weak_teacher_views",
        "weak_relational_teacher_views",
        "weak_teacher_augmentation_config",
    }
)


@dataclass(frozen=True)
class PretrainSettings:
    """One resolution of the backbone, sizes and patch configuration.

    ``run_pretrain`` and the CLI preflight read the same object, so preflight
    never validates against a model or a size the trainer will not use.
    """

    model_name: str
    global_crop_size: int
    local_crop_size: int
    local_crops: int
    patch_config: dict[str, object]

    @property
    def patch_mode(self) -> str:
        return str(self.patch_config["patch_loss"])

    @property
    def masking_strategy(self) -> str | None:
        strategy = self.patch_config["masking_strategy"]
        return None if strategy is None else str(strategy)


def resolve_pretrain_settings(
    args: argparse.Namespace, checkpoint: dict[str, Any] | None = None
) -> PretrainSettings:
    """Resolve the backbone, sizes and patch configuration exactly once.

    Pure: the checkpoint is read, never mutated, and the ``auto`` native-size
    lookup is RNG-isolated so preflight cannot shift the student init a given
    ``--seed`` produces. Resume uses the checkpoint-resolved backbone and size.
    """
    from otuformer.utils.checkpoint import preserved_rng

    requested_size = getattr(args, "global_crop_size", None)
    if checkpoint is not None:
        model_name = resolve_checkpoint(checkpoint, args.model_name).model_name
        recorded_size = resolve_training_image_size(checkpoint)
        if requested_size is not None and int(requested_size) != recorded_size:
            raise ValueError(
                "resume cannot change the input size: checkpoint records "
                f"{recorded_size}, got --global-crop-size {requested_size}. "
                "Start a new run for a different input size."
            )
        global_crop_size = recorded_size
    else:
        model_name = args.model_name
        if requested_size is not None:
            global_crop_size = int(requested_size)
        else:
            with preserved_rng():
                global_crop_size = resolve_backbone_native_size(model_name)
    local_crop_size, local_crops = _resolve_pretrain_local_views(
        getattr(args, "local_crop_size", None),
        getattr(args, "local_crops", None),
        checkpoint,
    )
    return PretrainSettings(
        model_name=model_name,
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        local_crops=local_crops,
        patch_config=_resolve_patch_config(args, checkpoint),
    )


def preflight_pretrain_run(
    args: argparse.Namespace, settings: PretrainSettings
) -> None:
    """Reject an impossible pretraining configuration before output setup.

    Builds one RNG-isolated ``pretrained=False`` probe of the resolved model at
    the resolved native size and uses it for the divisibility, backbone
    capability, blockwise-grid and patch-count checks; no pretrained weights are
    downloaded, and ``--overwrite`` cannot clear an existing output directory
    for a configuration that is rejected anyway.
    """
    from otuformer.utils.checkpoint import preserved_rng

    size = settings.global_crop_size
    register_tokens = (
        None
        if getattr(args, "resume", "")
        else getattr(args, "register_tokens", None)
    )
    with preserved_rng():
        probe = OTUFormerEncoder(
            model_name=settings.model_name,
            out_dim=256,
            return_patch_tokens=True,
            img_size=size,
            pretrained=False,
            reg_tokens=register_tokens,
        )
    validate_input_size(size, probe, settings.model_name)
    if settings.local_crops > 0:
        validate_input_size(settings.local_crop_size, probe, settings.model_name)
    probe.validate_pretrain_backbone(
        require_intermediates=settings.patch_mode == "masked-feature"
    )
    if settings.patch_mode == "none":
        return
    grid_size = probe._patch_grid_size(torch.empty(1, 3, size, size))
    if settings.masking_strategy in ("blockwise", "hybrid"):
        _validate_blockwise_grid(grid_size)
    if grid_size[0] * grid_size[1] < 2:
        raise ValueError(
            f"patch grid {grid_size} has fewer than two patch tokens; the "
            "selected patch objective needs at least one masked and one "
            "visible position. Increase the input size or use "
            "--patch-loss none."
        )


def preflight_pretrain_resume(
    resume_ckpt: dict[str, Any],
    model_name: str,
    register_tokens: int | None,
    model_name_explicit: bool,
) -> tuple[str, int]:
    """Validate a pretraining resume before the CLI creates any output file.

    Returns ``(mode, actual_count)``. Raises ``ValueError``/``KeyError`` so the
    caller can map it onto a clean CLI error.
    """
    architecture = resolve_checkpoint(resume_ckpt, model_name)
    if (
        architecture.model_name_from_metadata
        and model_name_explicit
        and architecture.model_name != model_name
    ):
        raise ValueError(
            "The resume checkpoint records model_name="
            f"{architecture.model_name!r}, which conflicts with the explicit "
            f"--model-name {model_name!r}."
        )
    _validate_saved_progress_basics(resume_ckpt, "Resume")
    image_size = resolve_training_image_size(resume_ckpt)
    mode, actual = _resume_register_plan(
        resume_ckpt, architecture.model_name, image_size, architecture.embedding_dim
    )
    _check_register_option_against_resume(mode, actual, register_tokens)
    return mode, actual


RELATIONAL_TARGET_CONFIG_KEY = "relational_target_config"


RELATIONAL_BUNDLE_KEYS = (
    "version",
    "relational_teacher_views",
    "relational_pairing",
    "relational_center_source",
    "teacher_temp_start",
    "teacher_temp_end",
    "weak_teacher_augmentation_config",
)


RELATIONAL_ARGS_KEYS = (
    "relational_teacher_views",
    "relational_pairing",
    "relational_center_source",
    "teacher_temp_start",
    "teacher_temp_end",
)


RELATIONAL_MARKER_KEYS = (
    "relational_teacher_views",
    "relational_pairing",
    "relational_center_source",
    "teacher_center_source",
    "weak_teacher_augmentation_config",
    "weak_relational_teacher_views",
)


NEW_RUN_TEACHER_TEMP_START = 0.08


NEW_RUN_TEACHER_TEMP_END = 0.11


NEW_RUN_TEACHER_TEMP_BAND = (0.001, 100.0)


HISTORICAL_TEACHER_TEMP_START = 0.04


HISTORICAL_TEACHER_TEMP_END = 0.07


HISTORICAL_TEACHER_TEMP_WARMUP_RATIO = 0.7


def _strict_equal(left: object, right: object) -> bool:
    """Type-strict structural equality: ``True`` never equals ``1``."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_strict_equal(a, b) for a, b in zip(left, right))
    if type(left) is not type(right):
        return False
    return left == right


def _requested_float(value: object, option: str) -> float:
    """Normalize an explicit numeric request; booleans and junk are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{option} expects a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{option} must be finite, got {value!r}.")
    return number


def _admissible_temperature(value: float, batch_size: int, option: str) -> float:
    """Reject a temperature whose float32 reduction bound overflows."""
    f32 = np.finfo(np.float32)
    tau32 = float(np.float32(value))
    if not (math.isfinite(tau32) and tau32 > 0.0):
        # A value that underflows to 0.0 in float32 must fail as a parameter
        # error, not as a ZeroDivisionError from the bound below.
        raise ValueError(
            f"{option} {value!r} is not a usable float32 temperature: it becomes "
            f"{tau32!r} at schedule precision."
        )
    limit = float(f32.max) * (1.0 - 4.0 * float(f32.eps))
    bound = float(max(int(batch_size), 2)) / tau32
    if not math.isfinite(bound):
        raise ValueError(
            f"{option} {value!r} is not a usable float32 temperature."
        )
    if bound > limit:
        raise ValueError(
            f"{option} {value!r} overflows the float32 teacher relation "
            f"reduction at batch size {batch_size}; increase the temperature."
        )
    return value


def _requested_choice(
    value: object, choices: tuple[str, ...], option: str
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"{option} must be one of: {', '.join(choices)}; got {value!r}."
        )
    return value


def _validate_saved_progress_basics(resume_ckpt: dict[str, Any], label: str) -> None:
    """Shape checks that no saved training plan can make valid.

    Runs for every method before any output exists. Checks that depend on the
    recorded schedule plan live in the method-specific preflight, so an unread
    or legacy payload is not rejected here.
    """
    for key in ("optimizer", "schedule", "epoch", "iteration"):
        value = resume_ckpt.get(key)
        if value is None:
            continue
        if key == "optimizer":
            if not isinstance(value, dict) or not value.get("param_groups"):
                raise ValueError(f"{label} found a malformed optimizer state.")
            for group in value["param_groups"]:
                _validate_adamw_param_group(group, label)
        elif key == "schedule":
            if not isinstance(value, dict):
                raise ValueError(f"{label} found a malformed schedule state.")
        else:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{label} requires {key} to be an integer >= 0, got {value!r}."
                )
    epoch = resume_ckpt.get("epoch")
    iteration = resume_ckpt.get("iteration")
    schedule = resume_ckpt.get("schedule")
    if not isinstance(epoch, int) or not isinstance(iteration, int):
        return
    if iteration < epoch + 1:
        raise ValueError(
            f"{label} requires iteration to cover the completed epochs; got "
            f"epoch={epoch} and iteration={iteration}."
        )
    if not isinstance(schedule, dict):
        return
    if int(schedule.get("total_steps", iteration)) < iteration:
        raise ValueError(
            f"{label} found iteration={iteration} beyond the recorded plan of "
            f"{schedule.get('total_steps')} steps."
        )


def _new_register_position_tensor(model: OTUFormerEncoder) -> torch.Tensor | None:
    """The four registered prefix positional entries, or ``None`` if patch-only."""
    backbone = model.backbone
    prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    pos = getattr(backbone, "pos_embed", None)
    if pos is None or pos.shape[1] <= prefix:
        return None
    if pos.shape[1] - prefix != int(getattr(backbone.patch_embed, "num_patches", -1)):
        return None
    return pos[:, 1:prefix]


def _validate_relational_combination(views: str, center: str) -> None:
    if views not in RELATIONAL_TEACHER_VIEW_CHOICES:
        raise ValueError(
            "relational_teacher_views must be one of: "
            f"{', '.join(RELATIONAL_TEACHER_VIEW_CHOICES)}; got {views!r}."
        )
    if center not in RELATIONAL_CENTER_SOURCE_CHOICES:
        raise ValueError(
            "relational_center_source must be one of: "
            f"{', '.join(RELATIONAL_CENTER_SOURCE_CHOICES)}; got {center!r}."
        )
    if views == "ordinary" and center != "ordinary":
        raise ValueError(
            "relational_teacher_views='ordinary' requires "
            "relational_center_source='ordinary': ordinary teacher views have no "
            "weak features to populate the center."
        )


def _validated_bundle(
    payload: object, saved_args: dict[str, Any]
) -> dict[str, Any]:
    """Validate a complete version-1 canonical bundle."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"config.{RELATIONAL_TARGET_CONFIG_KEY} must be a mapping, got "
            f"{type(payload).__name__}."
        )
    missing = [key for key in RELATIONAL_BUNDLE_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"config.{RELATIONAL_TARGET_CONFIG_KEY} is incomplete; missing: "
            + ", ".join(missing)
        )
    extra = sorted(set(payload) - set(RELATIONAL_BUNDLE_KEYS))
    if extra:
        raise ValueError(
            f"config.{RELATIONAL_TARGET_CONFIG_KEY} has unknown keys: "
            + ", ".join(extra)
        )
    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError(
            f"config.{RELATIONAL_TARGET_CONFIG_KEY}.version must be the integer "
            f"1, got {version!r}."
        )
    views = payload["relational_teacher_views"]
    center = payload["relational_center_source"]
    pairing = payload["relational_pairing"]
    _validate_relational_combination(views, center)
    if pairing == LEGACY_RELATIONAL_PAIRING:
        if views != "ordinary":
            raise ValueError(
                "A relational target bundle records the historical 'same' pairing "
                "together with weak teacher views; those are contradictory."
            )
    elif pairing not in RELATIONAL_PAIRING_CHOICES:
        raise ValueError(
            "relational_pairing must be one of: "
            f"{', '.join(RELATIONAL_PAIRING_CHOICES + (LEGACY_RELATIONAL_PAIRING,))}"
            f"; got {pairing!r}."
        )
    for key in ("teacher_temp_start", "teacher_temp_end"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"config.{RELATIONAL_TARGET_CONFIG_KEY}.{key} must be a number, "
                f"got {value!r}."
            )
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(
                f"config.{RELATIONAL_TARGET_CONFIG_KEY}.{key} must be finite and "
                f"positive, got {value!r}."
            )
    weak = payload["weak_teacher_augmentation_config"]
    if views == "ordinary":
        if weak is not None:
            raise ValueError(
                "A relational target bundle records ordinary teacher views with a "
                "weak augmentation config; those are contradictory."
            )
    elif not isinstance(weak, dict):
        # Weak views must attest the transform they used; null would silently
        # fall back to a freshly derived record and bypass the drift lock.
        raise ValueError(
            f"config.{RELATIONAL_TARGET_CONFIG_KEY}.weak_teacher_augmentation_config "
            f"must be a mapping for weak teacher views, got {type(weak).__name__}."
        )
    for key in RELATIONAL_ARGS_KEYS:
        if key in saved_args and not _strict_equal(saved_args[key], payload[key]):
            raise ValueError(
                f"Cannot resume: saved args.{key}={saved_args[key]!r} disagrees "
                f"with config.{RELATIONAL_TARGET_CONFIG_KEY}.{key}="
                f"{payload[key]!r}."
            )
    return dict(payload)


def _resolve_saved_warmup_ratio(
    saved_args: dict[str, Any],
    ratio_request: float,
    ratio_explicit: bool,
    warnings: list[str],
) -> float:
    """Resolve the inherited warmup ratio by key presence, not truthiness.

    An absent key is the only case that may assume the historical constant; a
    present ``null`` is damaged provenance and must fail instead of being
    rewritten to the default on the next save.
    """
    if "teacher_temp_warmup_ratio" not in saved_args:
        warnings.append(
            "teacher_temp_warmup_ratio is not recorded; assuming the historical "
            f"{HISTORICAL_TEACHER_TEMP_WARMUP_RATIO}."
        )
        if ratio_explicit and ratio_request != HISTORICAL_TEACHER_TEMP_WARMUP_RATIO:
            raise ValueError(
                "Cannot resume with --teacher-temp-warmup-ratio "
                f"{ratio_request!r}: the checkpoint records no warmup ratio, so "
                f"only the historical {HISTORICAL_TEACHER_TEMP_WARMUP_RATIO} can be "
                "assumed. Start a new run to change it."
            )
        return HISTORICAL_TEACHER_TEMP_WARMUP_RATIO
    saved_ratio = saved_args["teacher_temp_warmup_ratio"]
    ratio = _requested_float(saved_ratio, "args.teacher_temp_warmup_ratio")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            "The checkpoint records an out-of-range "
            f"teacher_temp_warmup_ratio {saved_ratio!r}."
        )
    if ratio_explicit and ratio_request != ratio:
        raise ValueError(
            "Cannot resume with --teacher-temp-warmup-ratio "
            f"{ratio_request!r}: the checkpoint records {ratio!r}."
        )
    return ratio


def _resolve_relational_target_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
    *,
    candidate_count: int | None = None,
) -> tuple[dict[str, Any], float, tuple[str, ...]]:
    """Resolve the relational target bundle, the warmup ratio and warnings.

    Pure: no logging, no transform derivation, no mutation of ``args`` or the
    loaded checkpoint.
    """
    saved_args: dict[str, Any] = {}
    config: dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        raw_args = checkpoint.get("args")
        saved_args = raw_args if isinstance(raw_args, dict) else {}
        raw_config = checkpoint.get("config")
        config = raw_config if isinstance(raw_config, dict) else {}

    warnings: list[str] = []
    bundle_present = RELATIONAL_TARGET_CONFIG_KEY in config
    saved_bundle = (
        _validated_bundle(config[RELATIONAL_TARGET_CONFIG_KEY], saved_args)
        if bundle_present
        else None
    )
    if not bundle_present:
        markers = [
            key
            for key in RELATIONAL_MARKER_KEYS
            if key in saved_args or key in config
        ]
        if markers:
            raise ValueError(
                "Cannot resume: the checkpoint carries partial relational target "
                "metadata ("
                + ", ".join(sorted(markers))
                + ") without a complete config."
                + RELATIONAL_TARGET_CONFIG_KEY
                + ". Start a new run or resume from an intact checkpoint."
            )

    views_request = _requested_choice(
        getattr(args, "relational_teacher_views", None),
        RELATIONAL_TEACHER_VIEW_CHOICES,
        "--relational-teacher-views",
    )
    center_request = _requested_choice(
        getattr(args, "relational_center_source", None),
        RELATIONAL_CENTER_SOURCE_CHOICES,
        "--relational-center-source",
    )
    pairing_request = _requested_choice(
        getattr(args, "relational_pairing", None),
        RELATIONAL_PAIRING_CHOICES,
        "--relational-pairing",
    )
    start_request = getattr(args, "teacher_temp_start", None)
    end_request = getattr(args, "teacher_temp_end", None)
    start_request = (
        None
        if start_request is None
        else _requested_float(start_request, "--teacher-temp-start")
    )
    end_request = (
        None
        if end_request is None
        else _requested_float(end_request, "--teacher-temp-end")
    )
    raw_ratio = getattr(args, "teacher_temp_warmup_ratio", None)
    if raw_ratio is None:
        raw_ratio = HISTORICAL_TEACHER_TEMP_WARMUP_RATIO
    ratio_request = _requested_float(raw_ratio, "--teacher-temp-warmup-ratio")
    if not 0.0 <= ratio_request <= 1.0:
        raise ValueError(
            "--teacher-temp-warmup-ratio must be in the closed range [0, 1], got "
            f"{ratio_request!r}."
        )
    ratio_explicit = bool(
        getattr(args, "teacher_temp_warmup_ratio_explicit", False)
    ) or ratio_request != HISTORICAL_TEACHER_TEMP_WARMUP_RATIO
    # Temperature admission covers the reduction length this run really uses:
    # the effective batch, or the conservative ``B + min(K, N - B)`` bound when
    # the queue is active. The parser default is never used for an enabled run.
    batch_size = (
        int(candidate_count)
        if candidate_count is not None
        else int(getattr(args, "batch_size", 1) or 1)
    )

    if checkpoint is None:
        views = views_request or "weak"
        center = center_request or views
        pairing = pairing_request or "cross"
        start = NEW_RUN_TEACHER_TEMP_START if start_request is None else start_request
        end = NEW_RUN_TEACHER_TEMP_END if end_request is None else end_request
        low, high = NEW_RUN_TEACHER_TEMP_BAND
        for option, value in (
            ("--teacher-temp-start", start),
            ("--teacher-temp-end", end),
        ):
            if not low <= value <= high:
                raise ValueError(
                    f"{option} must be inside the new-run policy band "
                    f"[{low}, {high}], got {value!r}. A checkpoint may record a "
                    "value outside this band; a fresh request may not."
                )
        _validate_relational_combination(views, center)
        start = _admissible_temperature(start, batch_size, "--teacher-temp-start")
        end = _admissible_temperature(end, batch_size, "--teacher-temp-end")
        weak_config: dict[str, Any] | None = None
        ratio = ratio_request
    elif saved_bundle is not None:
        views = str(saved_bundle["relational_teacher_views"])
        center = str(saved_bundle["relational_center_source"])
        pairing = str(saved_bundle["relational_pairing"])
        start = float(saved_bundle["teacher_temp_start"])
        end = float(saved_bundle["teacher_temp_end"])
        for option, requested, effective in (
            ("--relational-teacher-views", views_request, views),
            ("--relational-center-source", center_request, center),
            ("--relational-pairing", pairing_request, pairing),
            ("--teacher-temp-start", start_request, start),
            ("--teacher-temp-end", end_request, end),
        ):
            if requested is not None and not _strict_equal(requested, effective):
                raise ValueError(
                    f"Cannot resume: {option}={requested!r} conflicts with the "
                    f"recorded recipe value {effective!r}. A changed training "
                    "recipe requires a new run."
                )
        _validate_relational_combination(views, center)
        start = _admissible_temperature(start, batch_size, "teacher_temp_start")
        end = _admissible_temperature(end, batch_size, "teacher_temp_end")
        ratio = _resolve_saved_warmup_ratio(
            saved_args, ratio_request, ratio_explicit, warnings
        )
        weak_config = saved_bundle["weak_teacher_augmentation_config"]
    else:
        views = "ordinary"
        center = "ordinary"
        if "disable_cross_view_loss" not in saved_args:
            warnings.append(
                "disable_cross_view_loss is not recorded; assuming the historical "
                "False, which selects the 'full' pairing."
            )
            pairing = "full"
        else:
            raw_disable = saved_args["disable_cross_view_loss"]
            if not isinstance(raw_disable, bool):
                raise ValueError(
                    "The checkpoint records a non-boolean disable_cross_view_loss "
                    f"{raw_disable!r}; the legacy pairing cannot be decoded."
                )
            pairing = LEGACY_RELATIONAL_PAIRING if raw_disable else "full"
        for option, requested, effective in (
            ("--relational-teacher-views", views_request, views),
            ("--relational-center-source", center_request, center),
            ("--relational-pairing", pairing_request, pairing),
        ):
            if requested is not None and not _strict_equal(requested, effective):
                raise ValueError(
                    f"Cannot resume: {option}={requested!r} conflicts with the "
                    f"legacy recipe value {effective!r}. Start a new run to change "
                    "the recipe."
                )
        start, end = _resolve_legacy_temperatures(
            saved_args, start_request, end_request, warnings
        )
        ratio = _resolve_saved_warmup_ratio(
            saved_args, ratio_request, ratio_explicit, warnings
        )
        start = _admissible_temperature(start, batch_size, "teacher_temp_start")
        end = _admissible_temperature(end, batch_size, "teacher_temp_end")
        weak_config = None

    bundle: dict[str, Any] = {
        "version": 1,
        "relational_teacher_views": views,
        "relational_pairing": pairing,
        "relational_center_source": center,
        "teacher_temp_start": start,
        "teacher_temp_end": end,
        "weak_teacher_augmentation_config": weak_config,
    }
    return bundle, ratio, tuple(warnings)


def _resolve_legacy_temperatures(
    saved_args: dict[str, Any],
    start_request: float | None,
    end_request: float | None,
    warnings: list[str],
) -> tuple[float, float]:
    """Resolve legacy teacher temperatures, warning about assumed fields."""
    resolved: list[float] = []
    for key, requested, assumed in (
        ("teacher_temp_start", start_request, HISTORICAL_TEACHER_TEMP_START),
        ("teacher_temp_end", end_request, HISTORICAL_TEACHER_TEMP_END),
    ):
        if key not in saved_args:
            warnings.append(
                f"{key} is not recorded; assuming the historical {assumed}."
            )
            if requested is not None and requested != assumed:
                raise ValueError(
                    f"Cannot resume with --{key.replace('_', '-')}={requested!r}: "
                    f"the checkpoint records no value, so only the historical "
                    f"{assumed} can be assumed. Start a new run to change it."
                )
            resolved.append(assumed)
            continue
        saved = saved_args[key]
        value = _requested_float(saved, f"args.{key}")
        if value <= 0.0:
            raise ValueError(
                f"The checkpoint records a non-positive {key} {saved!r}."
            )
        if requested is not None and requested != value:
            raise ValueError(
                f"Cannot resume: --{key.replace('_', '-')}={requested!r} conflicts "
                f"with the recorded value {value!r}. A changed training recipe "
                "requires a new run."
            )
        resolved.append(value)
    return resolved[0], resolved[1]


RELATIONAL_QUEUE_CONFIG_KEY = "relational_queue_config"
RELATIONAL_QUEUE_STATE_KEY = "relational_queue"

#: Saved ``args`` keys that record the queue request and its effective value.
QUEUE_SIZE_ARGS_KEY = "relational_queue_size"
QUEUE_REQUESTED_SIZE_ARGS_KEY = "requested_relational_queue_size"
BATCH_SIZE_ARGS_KEY = "batch_size"
QUEUE_REQUESTED_BATCH_ARGS_KEY = "requested_batch_size"

#: Saved schedule fields the enabled resume rebuild reads directly.
_QUEUE_REQUIRED_SCHEDULE_INTS = (
    "original_max_epochs",
    "total_steps",
    "steps_per_epoch",
    "warmup_steps",
)
_QUEUE_REQUIRED_SCHEDULE_FLOATS = ("last_lr", "final_lr")

QUEUE_CAPACITY_SOURCES = ("explicit", "default", "inherited", "inactive")
QUEUE_BATCH_SOURCES = ("explicit", "default", "inherited")


@dataclass(frozen=True)
class _RelationalQueuePreparation:
    """One invocation's resolved queue applicability, data and recovery inputs.

    Transient: it carries no mutable queue instance and never enters saved
    ``args``. ``capacity`` is ``None`` only for an inactive DINO run; relational
    with the option omitted resolves to the effective zero. Only a positive
    capacity prepares data, resolves the device and validates saved RNG.
    """

    ssl_method: str
    capacity: int | None
    requested_capacity: int | None
    capacity_source: str
    effective_batch_size: int
    requested_batch_size: int | None
    batch_size_source: str
    data: "PreparedRelationalData | None"
    resume_queue_state: dict[str, Any] | None
    resume_payload: dict[str, Any] | None
    request_binding: tuple[object, ...]
    resolved_device: torch.device | None
    rng_state_to_restore: dict[str, object] | None

    @property
    def active(self) -> bool:
        return bool(self.capacity)

    @property
    def image_count(self) -> int:
        return 0 if self.data is None else len(self.data.image_keys)

    @property
    def candidate_bound(self) -> int | None:
        """Conservative reduction length ``B + min(K, N - B)`` when enabled."""
        if not self.active or self.data is None:
            return None
        batch = self.effective_batch_size
        return batch + min(int(self.capacity), self.image_count - batch)

    def resolved_log(self) -> dict[str, Any]:
        """The display-only record the trainer emits once after resolution."""
        if self.ssl_method == "dino":
            applicability = "inactive"
        else:
            applicability = "active" if self.active else "disabled"
        return {
            "applicability": applicability,
            "requested_capacity": self.requested_capacity,
            "effective_capacity": self.capacity,
            "capacity_source": self.capacity_source,
            "requested_batch_size": self.requested_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "batch_size_source": self.batch_size_source,
        }


def _queue_capacity_request(args: argparse.Namespace) -> tuple[int | None, bool]:
    """The raw ``--relational-queue-size`` request and its explicitness.

    A missing attribute means omission. A numeric value without an explicitness
    marker comes from a direct-core caller and counts as an explicit override.
    """
    raw = getattr(args, "relational_queue_size", None)
    if raw is None:
        return None, False
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            "--relational-queue-size must be a non-negative integer, got "
            f"{raw!r}."
        )
    if raw < 0:
        raise ValueError(
            f"--relational-queue-size must be >= 0, got {raw!r}."
        )
    marker = getattr(args, "relational_queue_size_explicit", None)
    explicit = True if marker is None else bool(marker)
    return int(raw), explicit


def _queue_batch_request(args: argparse.Namespace) -> tuple[int | None, bool]:
    """The raw batch-size request and its explicitness for enabled recovery."""
    raw = getattr(args, "batch_size", None)
    if raw is None:
        return None, False
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"--batch-size must be an integer, got {raw!r}.")
    if raw < 1:
        raise ValueError(f"--batch-size must be >= 1, got {raw!r}.")
    marker = getattr(args, "batch_size_explicit", None)
    explicit = True if marker is None else bool(marker)
    return int(raw), explicit


def _queue_enabled_request(args: argparse.Namespace) -> bool:
    """True when this invocation explicitly asks for a positive capacity."""
    requested, _explicit = _queue_capacity_request(args)
    return bool(requested)


def _queue_ssl_method(
    args: argparse.Namespace, saved: "_SavedSSLState | None"
) -> tuple[str, bool]:
    """Resolve the effective method exactly as the SSL resolver will."""
    requested = _ssl_option(args, "ssl_method", "relational")
    if requested not in SSL_METHODS:
        raise ValueError(
            f"--ssl-method must be one of {', '.join(SSL_METHODS)}, got "
            f"{requested!r}."
        )
    explicit = _ssl_explicit(args, "ssl_method")
    saved_method = saved.identity[0] if saved is not None and saved.identity else None
    if saved_method is None:
        return str(requested), explicit
    if explicit and requested != saved_method:
        raise ValueError(
            "Cannot change SSL method on resume: the checkpoint records "
            f"ssl_method={saved_method!r}, got --ssl-method {requested!r}. Start "
            "a new run instead."
        )
    return (str(requested) if explicit else str(saved_method)), explicit


def _tolerant_saved_ssl_state(
    checkpoint: dict[str, Any] | None,
) -> tuple["_SavedSSLState | None", Exception | None]:
    """Resolve the saved SSL state, keeping the failure for a deferred raise.

    A payload without model weights cannot supply an identity; that is decided by
    the model-constructing resume preflight. The queue layer only needs the
    identity when the queue request or the saved bundles make it relevant.
    """
    if checkpoint is None:
        return None, None
    try:
        return _saved_ssl_state(checkpoint), None
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        return None, exc


def _checkpoint_queue_bundles(
    checkpoint: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, int | None]:
    """Return the saved queue state and its capacity, or ``(None, None)``.

    A one-sided bundle is contradictory, never silently repaired: queue metadata
    only ever exists as the enabled version-1 config *and* its logical state.
    """
    from .relational_queue import validate_queue_config_bundle

    if not isinstance(checkpoint, dict) or not checkpoint:
        return None, None
    config = checkpoint.get("config")
    config = config if isinstance(config, dict) else {}
    state = checkpoint.get(RELATIONAL_QUEUE_STATE_KEY)
    has_config = RELATIONAL_QUEUE_CONFIG_KEY in config
    if has_config != (state is not None):
        present = RELATIONAL_QUEUE_CONFIG_KEY if has_config else RELATIONAL_QUEUE_STATE_KEY
        missing = (
            RELATIONAL_QUEUE_STATE_KEY
            if has_config
            else f"config.{RELATIONAL_QUEUE_CONFIG_KEY}"
        )
        raise ValueError(
            "Cannot resume: the checkpoint carries "
            f"{present} without {missing}. An enabled queue run persists both "
            "bundles together; start a new run or resume from an intact "
            "checkpoint."
        )
    if not has_config:
        return None, None
    capacity = validate_queue_config_bundle(config[RELATIONAL_QUEUE_CONFIG_KEY])
    if not isinstance(state, dict):
        raise ValueError(
            "Cannot resume: relational_queue must be a mapping, got "
            f"{type(state).__name__}."
        )
    return state, capacity


def _queue_binding(
    args: argparse.Namespace,
    *,
    method_requested: str,
    method_explicit: bool,
    requested_capacity: int | None,
    capacity_explicit: bool,
    requested_batch: int | None,
    batch_explicit: bool,
    checkpoint: dict[str, Any] | None,
) -> tuple[object, ...]:
    """Identity of the request a prepared handoff may be reused for."""
    return (
        str(getattr(args, "resume", "") or ""),
        str(getattr(args, "input_images_dir", "") or ""),
        str(getattr(args, "train_data", "") or ""),
        str(method_requested),
        bool(method_explicit),
        requested_capacity,
        bool(capacity_explicit),
        requested_batch,
        bool(batch_explicit),
        str(getattr(args, "device", "") or ""),
        # The schedule-coverage rule depends on the epoch target, so a reused
        # preparation must not serve a differently targeted request.
        None if getattr(args, "max_epochs", None) is None else int(args.max_epochs),
        None if checkpoint is None else id(checkpoint),
    )


def prepare_pretrain_relational_queue(
    args: argparse.Namespace, checkpoint: dict[str, Any] | None
) -> _RelationalQueuePreparation:
    """Resolve queue applicability, capacity, batch lock and enabled data once.

    Model-free and RNG-free: it runs before any architecture, register-layout or
    optimizer preflight, so a conflicting or unreadable request is rejected
    before output setup and before a probe encoder exists.
    """
    from otuformer.training.runtime import _resolve_device

    requested_capacity, capacity_explicit = _queue_capacity_request(args)
    requested_batch, batch_explicit = _queue_batch_request(args)
    saved, saved_error = _tolerant_saved_ssl_state(checkpoint)
    method, method_explicit = _queue_ssl_method(args, saved)
    queue_state, saved_capacity = _checkpoint_queue_bundles(checkpoint)
    if checkpoint is not None and saved is None and (
        queue_state is not None or requested_capacity is not None
    ):
        raise ValueError(
            "Cannot resolve the saved SSL identity this queue request must be "
            f"checked against: {saved_error}"
        )
    saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    saved_args = saved_args if isinstance(saved_args, dict) else {}
    binding = _queue_binding(
        args,
        method_requested=method,
        method_explicit=method_explicit,
        requested_capacity=requested_capacity,
        capacity_explicit=capacity_explicit,
        requested_batch=requested_batch,
        batch_explicit=batch_explicit,
        checkpoint=checkpoint,
    )

    if method == "dino":
        if requested_capacity is not None:
            raise ValueError(
                "--relational-queue-size is not applicable to --ssl-method dino "
                f"(got {requested_capacity!r}); it is a relational-only option. "
                "Omit it or use --ssl-method relational."
            )
        if queue_state is not None:
            raise ValueError(
                "The resume checkpoint carries relational queue metadata on a "
                "DINO SSL identity; those bundles belong to relational runs only."
            )
        claimed = saved_args.get(QUEUE_SIZE_ARGS_KEY)
        if claimed not in (None, 0):
            raise ValueError(
                "The resume checkpoint records an active relational queue on a "
                f"DINO SSL identity (args.{QUEUE_SIZE_ARGS_KEY}={claimed!r})."
            )
        return _RelationalQueuePreparation(
            ssl_method="dino",
            capacity=None,
            requested_capacity=None,
            capacity_source="inactive",
            effective_batch_size=requested_batch or 32,
            requested_batch_size=requested_batch,
            batch_size_source="explicit" if batch_explicit else "default",
            data=None,
            resume_queue_state=None,
            resume_payload=checkpoint,
            request_binding=binding,
            resolved_device=None,
            rng_state_to_restore=None,
        )

    if checkpoint is None:
        capacity = 0 if requested_capacity is None else int(requested_capacity)
        capacity_source = "default" if requested_capacity is None else "explicit"
    elif saved_capacity is not None:
        if (
            capacity_explicit
            and requested_capacity is not None
            and int(requested_capacity) != int(saved_capacity)
        ):
            raise ValueError(
                "Cannot change --relational-queue-size on resume: the checkpoint "
                f"records capacity {saved_capacity}, got {requested_capacity}. A "
                "changed queue recipe requires a new run."
            )
        capacity = int(saved_capacity)
        capacity_source = (
            "explicit"
            if capacity_explicit and requested_capacity is not None
            else "inherited"
        )
    else:
        # No queue bundle means the run was disabled; that stays true forever.
        claimed = saved_args.get(QUEUE_SIZE_ARGS_KEY)
        if claimed not in (None, 0):
            raise ValueError(
                "Cannot resume: "
                f"args.{QUEUE_SIZE_ARGS_KEY}={claimed!r} claims an enabled queue "
                "without the queue config and state bundles. Start a new run."
            )
        if requested_capacity:
            raise ValueError(
                "Cannot enable the relational queue on resume: this checkpoint "
                "records no queue state, so history cannot be enabled later. "
                "Start a new run with --relational-queue-size to use it."
            )
        capacity = 0
        capacity_source = (
            "explicit" if requested_capacity is not None else "inherited"
        )

    active = capacity > 0
    device = None
    if active:
        # The requested learning rate enters the float32 schedule on a fresh run
        # and on a same-plan resume alike, so validate it whenever the queue is
        # active — before any output changes.
        _validate_queue_lr_precision(getattr(args, "lr", None))
        device = _resolve_device(str(getattr(args, "device", "") or "auto"))
    if active and checkpoint is not None:
        effective_batch, batch_source = _enabled_resume_batch(
            saved_args,
            requested_batch=requested_batch,
            batch_explicit=batch_explicit,
        )
    else:
        effective_batch = requested_batch or 32
        batch_source = "explicit" if batch_explicit else "default"

    data: PreparedRelationalData | None = None
    rng_state: dict[str, object] | None = None
    if active:
        train_data = getattr(args, "train_data", "")
        data = prepare_relational_data(
            Path(train_data) if train_data else None,
            Path(args.input_images_dir),
            batch_size=effective_batch,
        )
        if checkpoint is not None:
            _validate_saved_progress(
                checkpoint,
                getattr(args, "max_epochs", None),
                "Queue-enabled resume",
            )
            _validate_queue_schedule_coverage(
                checkpoint, requested_max_epochs=getattr(args, "max_epochs", None)
            )

    preparation = _RelationalQueuePreparation(
        ssl_method="relational",
        capacity=int(capacity),
        requested_capacity=requested_capacity,
        capacity_source=capacity_source,
        effective_batch_size=effective_batch,
        requested_batch_size=requested_batch,
        batch_size_source=batch_source,
        data=data,
        resume_queue_state=queue_state,
        resume_payload=checkpoint,
        request_binding=binding,
        resolved_device=device,
        rng_state_to_restore=None,
    )
    if active and checkpoint is not None:
        _validate_relational_queue_resume(checkpoint, preparation)
        rng_state = _validate_fifo_rng_state(
            checkpoint.get("rng_state"), target_device=device
        )
    if rng_state is not None:
        from dataclasses import replace

        preparation = replace(preparation, rng_state_to_restore=rng_state)
    if active:
        # G1 hashes and decodes the complete selection before execution probes.
        _validate_queue_optimizer_precision(
            checkpoint,
            requested_lr=getattr(args, "lr", None),
            requested_weight_decay=getattr(args, "weight_decay", None),
            target_device=device,
            requested_max_epochs=getattr(args, "max_epochs", None),
            warmup_epochs=getattr(args, "warmup_epochs", 0),
        )
    return preparation


def queue_request_binding(
    args: argparse.Namespace, checkpoint: dict[str, Any] | None
) -> tuple[object, ...]:
    """Recompute the request identity a prepared handoff was bound to."""
    requested_capacity, capacity_explicit = _queue_capacity_request(args)
    requested_batch, batch_explicit = _queue_batch_request(args)
    saved, _saved_error = _tolerant_saved_ssl_state(checkpoint)
    method, method_explicit = _queue_ssl_method(args, saved)
    return _queue_binding(
        args,
        method_requested=method,
        method_explicit=method_explicit,
        requested_capacity=requested_capacity,
        capacity_explicit=capacity_explicit,
        requested_batch=requested_batch,
        batch_explicit=batch_explicit,
        checkpoint=checkpoint,
    )


def verify_relational_queue_preparation(
    preparation: _RelationalQueuePreparation | None,
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> _RelationalQueuePreparation:
    """Prove a supplied preparation still describes this invocation's request."""
    if preparation is None:
        return prepare_pretrain_relational_queue(args, checkpoint)
    if preparation.resume_payload is not checkpoint:
        raise ValueError(
            "The supplied relational queue preparation was resolved from a "
            "different resume payload; resolve it again for this invocation."
        )
    if preparation.request_binding != queue_request_binding(args, checkpoint):
        raise ValueError(
            "The supplied relational queue preparation was resolved for a "
            "different request (resume path, image root, data, method, queue "
            "size, batch size, epoch target or device changed); resolve it "
            "again before use."
        )
    if preparation.active:
        # Re-run the checks that depend on this invocation's request values: the
        # binding already rejects a changed --max-epochs, and the learning rate
        # is not part of the binding because a valid reuse may change it.
        _validate_queue_lr_precision(getattr(args, "lr", None))
        _validate_queue_optimizer_precision(
            checkpoint,
            requested_lr=getattr(args, "lr", None),
            requested_weight_decay=getattr(args, "weight_decay", None),
            target_device=preparation.resolved_device,
            requested_max_epochs=getattr(args, "max_epochs", None),
            warmup_epochs=getattr(args, "warmup_epochs", 0),
        )
        if checkpoint is not None:
            _validate_queue_schedule_coverage(
                checkpoint, requested_max_epochs=getattr(args, "max_epochs", None)
            )
    return preparation


def _validate_queue_schedule_endpoint(value: object, name: str) -> None:
    """A saved learning-rate endpoint must be usable at schedule precision.

    ``_build_pretrain_schedules`` builds float32 arrays, so a Python-finite
    ``1e100`` becomes ``Inf``/``NaN`` there and produces non-finite parameters
    while the loss and gradient checks still pass. Zero is legitimate: ``--lr 0``
    is accepted by the CLI and by AdamW, and the cosine schedule then yields a
    finite all-zero array.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "Queue-enabled resume requires the saved schedule "
            f"{name} to be a number, got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            "Queue-enabled resume requires the saved schedule "
            f"{name} to be finite, got {value!r}."
        )
    if number < 0.0:
        raise ValueError(
            "Queue-enabled resume requires the saved schedule "
            f"{name} to be non-negative, got {value!r}."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        at_schedule_precision = np.float32(number)
    if not bool(np.isfinite(at_schedule_precision)):
        raise ValueError(
            "Queue-enabled resume requires the saved schedule "
            f"{name} to survive the float32 schedule precision; {value!r} "
            f"becomes {at_schedule_precision!r}, which would drive the rebuilt "
            "learning rate non-finite."
        )


def _validate_queue_lr_precision(requested_lr: object) -> None:
    """The invocation's learning rate enters the same float32 schedule.

    The saved endpoints are validated with the schedule state itself; this
    covers the value this invocation adds to the cosine base.
    """
    if requested_lr is None:
        return
    _validate_queue_schedule_endpoint(requested_lr, "--lr")


def _validate_queue_schedule_coverage(
    checkpoint: dict[str, Any], *, requested_max_epochs: object
) -> None:
    """Prove the saved schedule covers the steps the rebuild will index.

    A same-plan resume rebuilds the learning-rate array from the recorded
    rectangle (``original_max_epochs * steps_per_epoch``) but the teacher
    temperature array from ``schedule.total_steps``. Those two must describe the
    same plan: a payload recording a shorter ``total_steps`` survives every other
    preflight and then indexes past the shorter array once training starts, i.e.
    after the logger has already created its CSV. An extension rebuild ignores
    the recorded total and fills every array to the same length, so only the
    recorded progress bound applies there.
    """
    if not isinstance(requested_max_epochs, int) or isinstance(
        requested_max_epochs, bool
    ):
        return
    schedule = checkpoint.get("schedule")
    if not isinstance(schedule, dict):
        return
    fields = {}
    for name in ("original_max_epochs", "total_steps", "steps_per_epoch"):
        value = schedule.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            return
        fields[name] = value
    if requested_max_epochs != fields["original_max_epochs"]:
        return
    rectangle = fields["original_max_epochs"] * fields["steps_per_epoch"]
    if fields["total_steps"] != rectangle:
        raise ValueError(
            "Queue-enabled resume found an inconsistent saved schedule: "
            f"total_steps={fields['total_steps']} does not describe the recorded "
            f"{fields['original_max_epochs']} epoch(s) of "
            f"{fields['steps_per_epoch']} step(s) ({rectangle}). The rebuilt "
            "learning-rate and teacher-temperature schedules would have "
            "different lengths."
        )


def _enabled_resume_batch(
    saved_args: dict[str, Any],
    *,
    requested_batch: int | None,
    batch_explicit: bool,
) -> tuple[int, str]:
    """Inherit the saved batch on an enabled resume, or reject a conflict."""
    saved_batch = saved_args.get(BATCH_SIZE_ARGS_KEY)
    if (
        isinstance(saved_batch, bool)
        or not isinstance(saved_batch, int)
        or saved_batch < 1
    ):
        raise ValueError(
            "Cannot resume an enabled queue run: the checkpoint records no usable "
            f"args.{BATCH_SIZE_ARGS_KEY} ({saved_batch!r}). Queue-enabled recovery "
            "requires the saved batch size; start a new run."
        )
    if batch_explicit and requested_batch is not None:
        if int(requested_batch) != int(saved_batch):
            raise ValueError(
                "Cannot change --batch-size on an enabled queue resume: the "
                f"checkpoint records {saved_batch}, got {requested_batch}. The "
                "queue, its steps-per-epoch and the loss candidates all depend on "
                "the saved batch size; start a new run instead."
            )
        return int(requested_batch), "explicit"
    return int(saved_batch), "inherited"


def _required_queue_records(preparation: _RelationalQueuePreparation) -> int:
    return min(int(preparation.capacity), preparation.effective_batch_size)


def _validate_relational_queue_resume(
    checkpoint: dict[str, Any], preparation: _RelationalQueuePreparation
) -> None:
    """Strict enabled-resume checks for the queue bundle, state and locks.

    Runs before any output setup: the config/state pair, the codec against the
    prepared selection, the exact batch/steps-per-epoch and schedule locks, the
    latest-commit tail and the complete-epoch occupancy bound.
    """
    from .relational_queue import RelationalQueue, validate_queue_config_bundle

    if not preparation.active or preparation.resume_payload is None:
        return
    config = checkpoint.get("config")
    config = config if isinstance(config, dict) else {}
    if RELATIONAL_QUEUE_CONFIG_KEY not in config:
        raise ValueError(
            "Cannot resume an enabled queue run: "
            f"config.{RELATIONAL_QUEUE_CONFIG_KEY} is missing."
        )
    capacity = validate_queue_config_bundle(config[RELATIONAL_QUEUE_CONFIG_KEY])
    if int(capacity) != int(preparation.capacity):
        raise ValueError(
            "Queue config/args conflict: "
            f"config.{RELATIONAL_QUEUE_CONFIG_KEY}.capacity={capacity} but the "
            f"resolved capacity is {preparation.capacity}."
        )
    saved_args = checkpoint.get("args")
    saved_args = saved_args if isinstance(saved_args, dict) else {}
    recorded = saved_args.get(QUEUE_SIZE_ARGS_KEY)
    if recorded is not None:
        if isinstance(recorded, bool) or not isinstance(recorded, int):
            raise ValueError(
                "The checkpoint records a non-integer "
                f"args.{QUEUE_SIZE_ARGS_KEY}={recorded!r}; the effective capacity "
                "is an integer count of image records."
            )
        if recorded != capacity:
            raise ValueError(
                "Queue config/args conflict: "
                f"args.{QUEUE_SIZE_ARGS_KEY}={recorded!r} but the queue config "
                f"records capacity {capacity}."
            )
    requested = saved_args.get(QUEUE_REQUESTED_SIZE_ARGS_KEY)
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int):
            raise ValueError(
                "The checkpoint records a malformed "
                f"args.{QUEUE_REQUESTED_SIZE_ARGS_KEY}={requested!r}; a recorded "
                "request is an integer count."
            )
        if requested < 0:
            raise ValueError(
                "The checkpoint records a negative "
                f"args.{QUEUE_REQUESTED_SIZE_ARGS_KEY}={requested!r}."
            )
        if requested != capacity:
            raise ValueError(
                "Queue request/config conflict: the checkpoint records "
                f"args.{QUEUE_REQUESTED_SIZE_ARGS_KEY}={requested!r} but the "
                f"effective capacity is {capacity}. An enabled bundle must match "
                "the recorded request."
            )

    state = checkpoint.get(RELATIONAL_QUEUE_STATE_KEY)
    if not isinstance(state, dict):
        raise ValueError(
            "Cannot resume an enabled queue run: relational_queue is missing."
        )
    iteration = checkpoint.get("iteration")
    epoch = checkpoint.get("epoch")
    schedule = checkpoint.get("schedule")
    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration < 0
    ):
        raise ValueError(
            "Queue-enabled resume requires an integer iteration >= 0, got "
            f"{iteration!r}."
        )
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(
            f"Queue-enabled resume requires an integer epoch >= 0, got {epoch!r}."
        )
    if not isinstance(schedule, dict):
        raise ValueError(
            "Queue-enabled resume requires the saved schedule state."
        )
    # Every field the schedule rebuild reads must be present, integral and
    # finite here: a missing ``final_lr`` or a NaN endpoint survives the rest of
    # the preflight and then fails (or produces a non-finite learning rate) only
    # after the logger has already opened its CSV.
    for name in _QUEUE_REQUIRED_SCHEDULE_INTS:
        if name not in schedule:
            raise ValueError(
                "Queue-enabled resume requires the saved "
                f"schedule.{name}; the checkpoint's schedule state is incomplete."
            )
    for name in _QUEUE_REQUIRED_SCHEDULE_FLOATS:
        if name not in schedule:
            raise ValueError(
                "Queue-enabled resume requires the saved "
                f"schedule.{name}; the checkpoint's schedule state is incomplete."
            )
    steps_per_epoch = schedule["steps_per_epoch"]
    if (
        isinstance(steps_per_epoch, bool)
        or not isinstance(steps_per_epoch, int)
        or steps_per_epoch < 1
    ):
        raise ValueError(
            "Queue-enabled resume requires the saved schedule.steps_per_epoch as "
            f"a positive integer, got {steps_per_epoch!r}."
        )
    for name in ("total_steps", "warmup_steps", "original_max_epochs"):
        value = schedule[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "Queue-enabled resume requires a valid saved schedule "
                f"{name}, got {value!r}."
            )
    for name in _QUEUE_REQUIRED_SCHEDULE_FLOATS:
        _validate_queue_schedule_endpoint(schedule[name], name)
    if int(schedule["original_max_epochs"]) < 1:
        raise ValueError(
            "Queue-enabled resume requires a positive saved "
            f"schedule.original_max_epochs, got "
            f"{schedule['original_max_epochs']!r}."
        )
    if int(schedule["total_steps"]) < 1 or int(schedule["total_steps"]) < int(
        iteration
    ):
        raise ValueError(
            "Queue-enabled resume found iteration="
            f"{iteration} beyond the recorded schedule of "
            f"{schedule['total_steps']} step(s)."
        )
    if iteration != (epoch + 1) * steps_per_epoch:
        raise ValueError(
            "Queue-enabled resume requires the checkpoint to sit exactly on an "
            f"epoch boundary: epoch={epoch}, iteration={iteration}, "
            f"steps_per_epoch={steps_per_epoch}."
        )
    prepared_count = preparation.image_count
    expected_steps = prepared_count // preparation.effective_batch_size
    if steps_per_epoch != expected_steps:
        raise ValueError(
            "Cannot resume an enabled queue run with a different loader: the "
            f"checkpoint records {steps_per_epoch} step(s) per epoch, but "
            f"{prepared_count} selected images at batch_size="
            f"{preparation.effective_batch_size} produce {expected_steps}. Equal "
            "loader lengths do not make a changed batch size valid."
        )
    if "optimizer" not in checkpoint:
        raise ValueError(
            "Queue-enabled resume requires the saved optimizer state."
        )
    for role in ("student", "teacher"):
        if not isinstance(checkpoint.get(role), dict):
            raise ValueError(
                f"Queue-enabled resume requires the saved {role} encoder state."
            )
    if "center" not in checkpoint:
        raise ValueError(
            "Queue-enabled resume requires the saved relational center."
        )

    data = preparation.data
    assert data is not None  # active preparation always prepares data
    from otuformer.utils.checkpoint import resolve_checkpoint_state_dict

    state_dict, _ = resolve_checkpoint_state_dict(checkpoint)
    out_dim = resolve_projector_out_dim(state_dict)
    if out_dim is None:
        raise ValueError(
            "Queue-enabled resume requires a projector width in the saved "
            "weights; none was found."
        )
    queue = RelationalQueue.from_state_dict(
        state,
        capacity=int(capacity),
        out_dim=int(out_dim),
        completed_steps=int(iteration),
        allowed_keys=frozenset(data.image_keys),
        data_fingerprint=data.data_fingerprint,
        data_image_count=len(data.image_keys),
        data_size_bytes=data.data_size_bytes,
        device=torch.device("cpu"),
    )
    observed = queue.observed_steps
    if iteration > 0:
        required = _required_queue_records(preparation)
        if len(observed) < required:
            raise ValueError(
                "Queue-enabled resume found truncated history: the latest "
                f"successful batch must leave {required} record(s), got "
                f"{len(observed)}."
            )
        tail = observed[-required:]
        if any(step != iteration - 1 for step in tail):
            raise ValueError(
                "Queue-enabled resume found history that does not end with the "
                f"latest successful batch at step {iteration - 1}."
            )
        earlier = observed[: len(observed) - required]
        if any(step == iteration - 1 for step in earlier):
            raise ValueError(
                "Queue-enabled resume found duplicated latest-commit timestamps "
                f"at step {iteration - 1}; the newest records must be exactly one "
                "successful batch."
            )
    per_epoch = (prepared_count // preparation.effective_batch_size) * (
        preparation.effective_batch_size
    )
    occupancy_bound = min(int(capacity), per_epoch)
    if len(queue) < occupancy_bound:
        raise ValueError(
            "Queue-enabled resume found truncated history: after a complete epoch "
            f"of {per_epoch} distinct image(s), at least {occupancy_bound} "
            f"record(s) must remain, got {len(queue)}."
        )


def _validate_fifo_rng_state(
    state: object, *, target_device: torch.device | None
) -> dict[str, object]:
    """Validate the saved RNG streams for the actual recovery target.

    Missing, ``None`` or malformed streams are a hard error: an enabled queue
    resume never falls back to a best-effort random sequence, and never seeds a
    substitute CUDA stream. A CUDA-shaped payload recovered on a non-CUDA target
    is structurally checked, omitted from the restore copy and reported.
    """
    from otuformer.utils.checkpoint import preserved_rng

    if not isinstance(state, dict) or not state:
        raise ValueError(
            "Queue-enabled resume requires the saved rng_state; a legacy "
            "checkpoint without one cannot continue an enabled queue run."
        )
    for name in ("python", "numpy", "torch"):
        if name not in state or state[name] is None:
            raise ValueError(
                f"Queue-enabled resume requires rng_state.{name}; the saved "
                "streams must be complete."
            )
    restore_copy = dict(state)
    cuda = state.get("cuda")
    cuda_states: list[torch.Tensor] | None = None
    if cuda is not None:
        if isinstance(cuda, (list, tuple)):
            entries = list(cuda)
        else:
            raise ValueError(
                "Queue-enabled resume found a malformed rng_state.cuda payload; "
                "it must be a sequence of CPU uint8 tensors."
            )
        if not entries:
            raise ValueError(
                "Queue-enabled resume found an empty rng_state.cuda sequence; "
                "an absent CUDA stream is recorded by omitting the key."
            )
        for entry in entries:
            if (
                not isinstance(entry, torch.Tensor)
                or entry.dtype != torch.uint8
                or entry.dim() != 1
                or entry.numel() == 0
                or entry.device.type != "cpu"
            ):
                raise ValueError(
                    "Queue-enabled resume found a malformed rng_state.cuda "
                    "stream; each entry must be a nonempty 1-D CPU uint8 tensor."
                )
        cuda_states = entries

    is_cuda_target = target_device is not None and target_device.type == "cuda"
    if is_cuda_target:
        if cuda_states is None:
            raise ValueError(
                "Queue-enabled resume targets CUDA but the checkpoint records no "
                "CUDA RNG stream; a CPU-generated checkpoint cannot continue an "
                "enabled run on CUDA."
            )
        expected = torch.cuda.device_count()
        if len(cuda_states) != expected:
            raise ValueError(
                "Queue-enabled resume found "
                f"{len(cuda_states)} saved CUDA RNG stream(s) but this host "
                f"exposes {expected}; the streams cannot be restored exactly."
            )
        with preserved_rng():
            _restore_rng_state(restore_copy)
        return restore_copy

    if cuda_states is not None:
        restore_copy.pop("cuda", None)
        print(
            "[Info] Queue-enabled resume: the checkpoint records CUDA RNG "
            "streams, but this target is not CUDA. Their structure was "
            "validated and they are not applied; this is a non-CUDA recovery, "
            "not a CUDA continuation."
        )
    with preserved_rng():
        _restore_rng_state(restore_copy)
    return restore_copy


def _validate_adamw_moment(
    entry: dict[str, Any], name: str, parameter: torch.Tensor, *, label: str
) -> None:
    value = entry.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"Cannot resume: optimizer state for {label} has no {name} tensor."
        )
    if tuple(value.shape) != tuple(parameter.shape):
        raise ValueError(
            f"Cannot resume: optimizer state {name} for {label} has shape "
            f"{tuple(value.shape)} but the parameter is "
            f"{tuple(parameter.shape)}."
        )
    if not value.is_floating_point():
        raise ValueError(
            f"Cannot resume: optimizer state {name} for {label} is not a "
            "floating-point tensor."
        )
    # Finiteness at the recorded precision is not enough: a float64 moment just
    # above the float32 range becomes Inf when the training parameter loads it.
    try:
        cast = value.to(dtype=parameter.dtype)
    except (RuntimeError, TypeError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Cannot resume: optimizer state {name} for {label} cannot be "
            f"loaded as {parameter.dtype}: {exc}"
        ) from exc
    if not bool(torch.isfinite(cast).all()):
        raise ValueError(
            f"Cannot resume: optimizer state {name} for {label} is not finite at "
            f"{parameter.dtype}; a value that is finite at its recorded precision "
            "can still overflow the training dtype."
        )
    if name in ("exp_avg_sq", "max_exp_avg_sq") and bool((cast < 0).any()):
        raise ValueError(
            f"Cannot resume: optimizer state {name} for {label} has negative "
            "values; a second moment is never negative."
        )


def _validate_fifo_step_format(
    step: object,
    *,
    flags: dict[str, bool],
    target_device: torch.device | None,
    label: str,
) -> float:
    """Validate one saved optimizer step against its group's execution branch.

    The format the branch can consume is not uniform, measured on the trainer's
    own path (default construct, load, step) on CPU and MPS:

    * ``differentiable=True`` needs a scalar (0-dim) step; a ``(1,)`` tensor only
      survives on CPU and a ``(1, 1)`` tensor broadcast-fails, while
      ``requires_grad`` is rejected by the differentiable branch only.
    * ``foreach=True`` (without ``fused``) accepts float32/float64 tensor steps,
      plus bfloat16 on CPU. Float16 and integer tensors fail its dtype check.
    * The default and ``fused`` paths accept every scalar form measured.
    """
    if isinstance(step, torch.Tensor):
        if step.requires_grad and flags["differentiable"]:
            raise ValueError(
                f"Cannot resume: {label} sets requires_grad on the optimizer "
                "step, which the differentiable AdamW path rejects."
            )
        if flags["differentiable"] and step.dim() != 0:
            device_type = None if target_device is None else target_device.type
            cpu_only = device_type == "cpu" and step.dim() == 1
            if not cpu_only:
                raise ValueError(
                    f"Cannot resume: {label} has shape {tuple(step.shape)} but "
                    "the differentiable path runs a shaped step only on CPU, "
                    "with a single element; MPS rejects it and a (1, 1) step "
                    "broadcast-fails everywhere."
                )
        if flags["foreach"] and not flags["fused"]:
            cpu_bfloat16 = (
                target_device is not None and target_device.type == "cpu"
                and step.dtype == torch.bfloat16
            )
            if step.dtype not in (torch.float32, torch.float64) and not cpu_bfloat16:
                raise ValueError(
                    f"Cannot resume: {label} records a {step.dtype} step, which "
                    "the foreach path rejects on this device (float32/float64, "
                    "or bfloat16 on CPU, are supported)."
                )
        if step.numel() != 1:
            raise ValueError(
                f"Cannot resume: {label} holds {step.numel()} values; an "
                "optimizer step is a single counter."
            )
        value = step.reshape(()).item()
    else:
        value = step
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Cannot resume: {label} is not an integer ({step!r})."
        )
    if not math.isfinite(float(value)) or float(value) != int(value):
        raise ValueError(
            f"Cannot resume: {label} is not a finite integer ({step!r})."
        )
    return float(value)


def _validate_fifo_adamw_state(
    optimizer: torch.optim.Optimizer,
    saved_optimizer: object,
    *,
    completed_steps: int,
    label: str,
    target_device: torch.device | None = None,
) -> None:
    """Prove the usable per-parameter AdamW state, not only the group layout.

    Walks the saved groups alongside the rebuilt ones so every check can use the
    *group's own* flags (AMSGrad, execution mode) instead of a cross-group
    summary: a mixed payload with one AMSGrad group and one plain group keeps
    working, and the step format is judged against the branch that will run.
    """
    if not isinstance(saved_optimizer, dict):
        raise ValueError(f"Cannot resume: the {label} optimizer state is missing.")
    state = saved_optimizer.get("state")
    if not isinstance(state, dict):
        raise ValueError(
            f"Cannot resume: the {label} optimizer state records no tensor state."
        )
    saved_groups = saved_optimizer.get("param_groups")
    saved_groups = saved_groups if isinstance(saved_groups, list) else []
    index = 0
    for group_index, (saved_group, rebuilt_group) in enumerate(
        zip(saved_groups, optimizer.param_groups)
    ):
        if not isinstance(saved_group, dict):  # pragma: no cover - guarded earlier
            continue
        flags = {
            name: bool(saved_group.get(name, False))
            for name in ("capturable", "fused", "foreach", "differentiable")
        }
        amsgrad = bool(saved_group.get("amsgrad", False))
        for parameter in rebuilt_group["params"]:
            entry = state.get(index)
            if entry is None:
                if parameter.requires_grad:
                    raise ValueError(
                        f"Cannot resume: the saved {label} optimizer state has no "
                        f"moments for participating parameter {index}; missing "
                        "state is not proof of legitimate inactivity."
                    )
                index += 1
                continue
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Cannot resume: optimizer state for parameter {index} is "
                    "malformed."
                )
            step_value = _validate_fifo_step_format(
                entry.get("step"),
                flags=flags,
                target_device=target_device,
                label=f"optimizer step for parameter {index}",
            )
            if not 1 <= int(step_value) <= int(max(completed_steps, 1)):
                raise ValueError(
                    f"Cannot resume: optimizer step for parameter {index} is "
                    f"{step_value}, outside [1, {completed_steps}]."
                )
            _validate_adamw_moment(
                entry, "exp_avg", parameter, label=f"parameter {index}"
            )
            _validate_adamw_moment(
                entry, "exp_avg_sq", parameter, label=f"parameter {index}"
            )
            if amsgrad:
                _validate_adamw_moment(
                    entry, "max_exp_avg_sq", parameter, label=f"parameter {index}"
                )
            index += 1


def _validate_queue_optimizer_coefficients(
    group: object, *, group_index: int, lr_bound: float,
    target_device: torch.device | None = None,
    lr_min: float | None = None, step: object = None,
) -> None:
    """Check the real restored AdamW branch on isolated float32 parameters.

    Python-scalar bias correction, tensor bias correction and device subnormal
    handling differ. A tiny restore/update probe avoids treating them as one
    float32 cast. It consumes no RNG and never touches the saved moments.
    """
    if not isinstance(group, dict):
        return
    device = target_device if target_device is not None else torch.device("cpu")
    label = f"optimizer group {group_index}"
    coefficients = {
        "lr": lr_bound, "eps": 1e-8, "betas": (0.9, 0.999),
        "weight_decay": 0.0,
        **{name: group[name] for name in (
            "eps", "betas", "weight_decay", "amsgrad", "maximize", "foreach",
            "fused", "capturable", "differentiable", "decoupled_weight_decay",
        ) if name in group},
    }
    _validate_adamw_param_group(coefficients, label)
    _validate_fifo_optimizer_execution_flags(
        {"param_groups": [coefficients]}, target_device=device
    )
    flags = {name: bool(coefficients.get(name, False)) for name in (
        "capturable", "fused", "foreach", "differentiable"
    )}
    saved_step = torch.tensor(1.0) if step is None else step
    _validate_fifo_step_format(
        saved_step, flags=flags, target_device=device, label=f"{label} step"
    )
    rates = {float(lr_bound), float(lr_bound if lr_min is None else lr_min)}
    for rate in rates:
        for gradient in (0.0, 1.0):
            parameter = nn.Parameter(torch.ones(4, device=device))
            optimizer = torch.optim.AdamW([parameter], lr=rate)
            saved = optimizer.state_dict()
            saved["param_groups"][0].update({**coefficients, "lr": rate})
            counter = saved_step.detach().clone() if isinstance(saved_step, torch.Tensor) else saved_step
            if isinstance(counter, torch.Tensor):
                counter.requires_grad_(saved_step.requires_grad)
            entry = {
                "step": counter, "exp_avg": torch.zeros(4),
                "exp_avg_sq": torch.zeros(4),
            }
            if coefficients.get("amsgrad", False):
                entry["max_exp_avg_sq"] = torch.zeros(4)
            saved["state"][0] = entry
            try:
                optimizer.load_state_dict(saved)
                parameter.grad = torch.full_like(parameter, gradient)
                optimizer.step()
                finite = bool(torch.isfinite(parameter).all()) and all(
                    bool(torch.isfinite(value).all())
                    for value in optimizer.state[parameter].values()
                    if isinstance(value, torch.Tensor)
                )
                if not finite:
                    raise ValueError("the restored update produced non-finite values")
            except (ValueError, RuntimeError, TypeError, AssertionError) as exc:
                raise ValueError(
                    f"Cannot resume: {label} optimizer update is unusable on "
                    f"{device} at lr={rate:g}, gradient={gradient:g}: {exc}."
                ) from exc


def _validate_queue_optimizer_precision(
    checkpoint: dict[str, Any] | None,
    *,
    requested_lr: object,
    requested_weight_decay: object,
    target_device: torch.device | None = None,
    requested_max_epochs: int | None = None,
    warmup_epochs: float = 0,
) -> None:
    """Validate effective coefficients before outputs, including handoff reuse."""
    base = float(requested_lr) if requested_lr is not None else 0.001
    endpoints = [base, base * 0.01]
    schedule = checkpoint.get("schedule") if isinstance(checkpoint, dict) else None
    if isinstance(schedule, dict):
        extension = requested_max_epochs != schedule.get("original_max_epochs")
        names = _QUEUE_REQUIRED_SCHEDULE_FLOATS if extension else ("final_lr",)
        endpoints = [] if extension else [base]
        for name in _QUEUE_REQUIRED_SCHEDULE_FLOATS:
            if name in schedule:
                _validate_queue_schedule_endpoint(schedule[name], name)
        for name in names:
            if name in schedule:
                endpoints.append(float(np.float32(schedule[name])))
        if not extension and schedule.get("warmup_steps", 0) > 0:
            endpoints.append(base * 0.1)
    elif warmup_epochs > 0:
        endpoints.append(base * 0.1)
    rates = [float(np.float32(value)) for value in endpoints] or [base]
    if isinstance(schedule, dict) and isinstance(requested_max_epochs, int):
        fields = [schedule.get(name) for name in (
            "original_max_epochs", "steps_per_epoch", "warmup_steps"
        )]
        iteration = checkpoint.get("iteration")
        epoch = checkpoint.get("epoch")
        if (
            all(isinstance(value, int) and not isinstance(value, bool) for value in fields)
            and fields[0] > 0 and fields[1] > 0 and fields[2] >= 0
            and isinstance(iteration, int) and not isinstance(iteration, bool)
            and isinstance(epoch, int) and not isinstance(epoch, bool)
            and iteration >= 0 and requested_max_epochs > epoch + 1
            and all(name in schedule for name in _QUEUE_REQUIRED_SCHEDULE_FLOATS)
        ):
            if requested_max_epochs == fields[0]:
                remaining = _cosine_scheduler(
                    base, float(schedule["final_lr"]), fields[0], fields[1],
                    warmup_epochs=fields[2] / fields[1],
                    start_warmup_value=base * 0.1,
                )[iteration:]
            else:
                remaining = _cosine_scheduler(
                    float(schedule["last_lr"]), float(schedule["final_lr"]),
                    1, (requested_max_epochs - epoch - 1) * fields[1],
                )
            if remaining.size:
                rates = [float(remaining.min()), float(remaining.max())]
    groups = []
    state = {}
    if isinstance(checkpoint, dict):
        optimizer = checkpoint.get("optimizer")
        if isinstance(optimizer, dict):
            groups = optimizer.get("param_groups") or []
            state = optimizer.get("state") or {}
    elif requested_weight_decay is not None:
        groups = [{"weight_decay": requested_weight_decay}]
    for index, group in enumerate(groups):
        # Probe each distinct saved counter format/value once, not each weight.
        counters = {}
        if isinstance(group, dict) and isinstance(state, dict):
            for parameter_id in group.get("params", []):
                entry = state.get(parameter_id)
                if not isinstance(entry, dict) or "step" not in entry:
                    continue
                step = entry["step"]
                if isinstance(step, torch.Tensor):
                    signature = (str(step.dtype), tuple(step.shape), step.requires_grad,
                                 step.item() if step.numel() == 1 else None)
                else:
                    signature = (type(step).__name__, repr(step))
                counters.setdefault(signature, step)
        for step in counters.values() if counters else (None,):
            _validate_queue_optimizer_coefficients(
                group, group_index=index, lr_bound=max(rates), lr_min=min(rates),
                target_device=target_device, step=step,
            )


#: AdamW flag pairs the restored optimizer rejects on every device. Measured on
#: the trainer's own path (default construct -> load_state_dict -> step):
#: ``differentiable`` fails with ``fused`` and with ``foreach``, everywhere.
_QUEUE_DIFFERENTIABLE_CONFLICTS = (
    ("fused", "`fused=True` does not support `differentiable=True`"),
    (
        "foreach",
        "`_foreach` ops do not support autograd (`foreach=True` with "
        "`differentiable=True`)",
    ),
)


def _validate_fifo_optimizer_execution_flags(
    saved_optimizer: object, *, target_device: torch.device | None
) -> None:
    """Reject saved AdamW execution flags the recovery target cannot run.

    ``load_state_dict`` restores these flags, so the failure would only appear on
    the first ``optimizer.step()`` — after the logger opened its CSV. The rules
    follow the trainer's real path (default construct, load, backward, step),
    measured on CPU and MPS: standalone ``differentiable``, standalone
    ``foreach``/``fused`` and ``fused`` with ``foreach`` all run, so they are
    accepted; ``differentiable`` with ``fused`` or ``foreach`` fails everywhere;
    and ``capturable`` without ``fused`` needs a CUDA-class device because the
    capturable path is the one that runs then. Accepting a flag for a CUDA target
    is a rule, not evidence that a CUDA step was exercised.
    """
    if not isinstance(saved_optimizer, dict):
        return
    groups = saved_optimizer.get("param_groups")
    if not isinstance(groups, list):
        return
    target = "unknown" if target_device is None else target_device.type
    on_cuda = target_device is not None and target_device.type == "cuda"
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        flags = {
            name: bool(group.get(name, False))
            for name in ("capturable", "fused", "foreach", "differentiable")
        }
        if flags["differentiable"]:
            for name, reason in _QUEUE_DIFFERENTIABLE_CONFLICTS:
                if flags[name]:
                    raise ValueError(
                        f"Queue-enabled resume cannot restore optimizer group "
                        f"{index} with {name}=True and differentiable=True: "
                        f"{reason}. The saved execution mode is unusable on every "
                        "device."
                    )
        if on_cuda:
            continue
        if flags["capturable"] and not flags["fused"]:
            raise ValueError(
                f"Queue-enabled resume cannot restore optimizer group {index} "
                "with capturable=True (without fused=True): that execution path "
                f"needs a CUDA-class device, but the recovery target is "
                f"{target}. Start a new run on a compatible device."
            )


def _alias_groups(model: nn.Module) -> list[list[str]]:
    """State-dict keys that name one live tensor (registered training aliases)."""
    by_identity: dict[int, list[str]] = {}
    for key, value in model.state_dict(keep_vars=True).items():
        by_identity.setdefault(id(value), []).append(key)
    return [
        sorted(keys) for keys in by_identity.values() if len(keys) > 1
    ]


def _validate_fifo_encoder_state(
    role: str,
    saved_state: object,
    expected: OTUFormerEncoder,
) -> None:
    """Require an independently complete, shaped, finite encoder state.

    Backbone-only or projector-layout checks are not a complete restore check:
    the saved keys must match this rebuilt encoder exactly, including the
    ``_pretrain_final_norm`` aliases the training path registers.
    """
    if not isinstance(saved_state, dict) or not saved_state:
        raise ValueError(
            f"Queue-enabled resume requires the saved {role} encoder state."
        )
    reference = expected.state_dict()
    missing = sorted(set(reference) - set(saved_state))
    extra = sorted(set(saved_state) - set(reference))
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing[:4]))
        if extra:
            detail.append("unexpected " + ", ".join(extra[:4]))
        raise ValueError(
            f"Cannot resume: the saved {role} state does not match the rebuilt "
            f"encoder ({'; '.join(detail)})."
        )
    for key, tensor in saved_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Cannot resume: saved {role}.{key} is not a tensor."
            )
        if tuple(tensor.shape) != tuple(reference[key].shape):
            raise ValueError(
                f"Cannot resume: saved {role}.{key} has shape "
                f"{tuple(tensor.shape)}; the rebuilt encoder needs "
                f"{tuple(reference[key].shape)}."
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"Cannot resume: saved {role}.{key} contains non-finite values."
            )
    for group in _alias_groups(expected):
        first = saved_state[group[0]]
        for alias in group[1:]:
            if not torch.equal(first, saved_state[alias]):
                raise ValueError(
                    f"Cannot resume: the saved {role} state records conflicting "
                    f"values for one live tensor ({group[0]} vs {alias})."
                )
    try:
        expected.load_state_dict(saved_state, strict=True)
    except RuntimeError as exc:  # pragma: no cover - guarded above
        raise ValueError(
            f"Cannot resume: the saved {role} state cannot be loaded strictly: "
            f"{exc}"
        ) from exc
    # Check the *loaded* tensors: a float64 parameter that is finite in its
    # recorded precision can still overflow the float32 training parameter.
    for key, tensor in expected.state_dict().items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"Cannot resume: saved {role}.{key} is not finite at "
                f"{tensor.dtype} after loading; a value that is finite at its "
                "recorded precision can still overflow the training dtype."
            )


def _validate_fifo_center(center: object, out_dim: int) -> None:
    if not isinstance(center, torch.Tensor):
        raise ValueError(
            "Queue-enabled resume requires the saved relational center tensor, "
            f"got {type(center).__name__}."
        )
    if tuple(center.shape) != (1, int(out_dim)):
        raise ValueError(
            "Cannot resume: the relational center must have exact shape "
            f"[1, {out_dim}], got {tuple(center.shape)}. A broadcastable shape is "
            "not accepted."
        )
    if not center.is_floating_point():
        raise ValueError(
            "Cannot resume: the relational center must be a floating-point "
            f"tensor, got {center.dtype}."
        )
    # The restore casts on the CPU and then moves to the device (a float64
    # payload cannot be moved to a float64-free backend like MPS), so the cast
    # validated here is the one the trainer performs.
    if not bool(torch.isfinite(center.to(dtype=torch.float32)).all()):
        raise ValueError(
            "Cannot resume: the relational center is not finite at float32; a "
            "value that is finite at its recorded precision can still overflow "
            "the restored buffer."
        )


def _validate_fifo_patch_state(
    probe_patch: PatchObjective, saved_patch_state: object, patch_mode: str
) -> None:
    """Strictly load the independent patch state and require finite tensors."""
    if patch_mode in ("masked-feature", "ibot"):
        if not isinstance(saved_patch_state, dict) or not saved_patch_state:
            raise ValueError(
                "Cannot resume: this checkpoint has no patch_objective state for "
                f"patch_loss={patch_mode!r}. Start a new run to use real masking."
            )
    if not isinstance(saved_patch_state, dict) or not saved_patch_state:
        return
    try:
        probe_patch.load_state_dict(saved_patch_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Cannot resume: patch_objective state is malformed or incompatible "
            f"with patch_loss={patch_mode!r}: {exc}"
        ) from exc
    for key, value in probe_patch.state_dict().items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"Cannot resume: patch_objective.{key} contains non-finite "
                    "values."
                )


class _ResolvedPretrainRecipe(NamedTuple):
    settings: PretrainSettings
    relational_target_config: dict[str, Any]
    teacher_temp_warmup_ratio: float
    augmentation_config: dict[str, object]
    warnings: tuple[str, ...]
    relational_queue_preparation: "_RelationalQueuePreparation | None" = None

    @property
    def weak_teacher_augmentation_config(self) -> dict[str, object] | None:
        return self.relational_target_config["weak_teacher_augmentation_config"]


def _resolve_pretrain_recipe(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
    *,
    settings: PretrainSettings,
    relational_queue_preparation: "_RelationalQueuePreparation | None" = None,
) -> _ResolvedPretrainRecipe:
    """Resolve and validate the whole pretraining recipe without building a model.

    ``relational_queue_preparation`` is the invocation's model-free queue
    resolution. An enabled request or enabled checkpoint without it is an error:
    an off recipe is never enabled by default and the finalizer never re-hashes.
    """
    preparation = relational_queue_preparation
    if preparation is None:
        if _queue_enabled_request(args) or _checkpoint_queue_bundles(checkpoint)[1]:
            raise ValueError(
                "The relational queue is enabled for this invocation but no "
                "prepared selection was supplied. Resolve "
                "prepare_pretrain_relational_queue() first and pass it to the "
                "recipe."
            )
    elif preparation.active and preparation.data is None:
        raise ValueError(
            "An enabled relational queue preparation must carry its prepared "
            "selection; none was supplied."
        )
    candidate_count = (
        preparation.candidate_bound if preparation is not None else None
    )
    bundle, ratio, warnings = _resolve_relational_target_config(
        args, checkpoint, candidate_count=candidate_count
    )
    profile = _select_augmentation_profile(
        getattr(args, "augmentation", None), checkpoint, stage="pretrain"
    )
    policy = _select_orientation_policy(
        getattr(args, "orientation_policy", None),
        checkpoint,
        stage="pretrain",
        resume=checkpoint is not None,
    )
    augmentation_config = build_pretrain_augmentation_config(
        profile,
        settings.global_crop_size,
        settings.local_crop_size,
        settings.local_crops,
        orientation_policy=policy,
    )
    augmentation_config = _validate_augmentation_config(
        profile, augmentation_config, checkpoint, stage="pretrain"
    )
    saved_weak = bundle["weak_teacher_augmentation_config"]
    if bundle["relational_teacher_views"] == "weak":
        derived_weak = build_weak_teacher_augmentation_config(augmentation_config)
        if saved_weak is not None and not _strict_equal(derived_weak, saved_weak):
            raise ValueError(
                "Cannot resume: the recorded weak teacher augmentation config no "
                "longer matches the config this code derives. Resume with a code "
                "version that reproduces the saved transform, or start a new run."
            )
        bundle["weak_teacher_augmentation_config"] = derived_weak
    elif saved_weak is not None:
        raise ValueError(
            "Cannot resume: ordinary teacher views cannot carry a weak "
            "augmentation config."
        )
    return _ResolvedPretrainRecipe(
        settings=settings,
        relational_target_config=bundle,
        teacher_temp_warmup_ratio=ratio,
        augmentation_config=augmentation_config,
        warnings=tuple(warnings),
        relational_queue_preparation=preparation,
    )
