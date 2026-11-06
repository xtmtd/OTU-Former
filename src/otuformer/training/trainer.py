"""Training entry points for pretrain and finetune."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from otuformer.embedding.evaluator import (
    compute_clustering_metrics,
    compute_knn_accuracy,
    compute_linear_probing_metrics,
    compute_map,
    compute_recall_at_k,
    run_umap,
)
from otuformer.training.dataset import (
    FINETUNE_AUGMENTATIONS,
    ORIENTATION_POLICIES,
    PRETRAIN_AUGMENTATIONS,
    IndexedDataset,
    MetricDataset,
    MultiCropDataset,
    _build_recursive_index,
    _resolve_image_path,
    _supports_recursive_lookup,
    build_finetune_augmentation_config,
    build_pretrain_augmentation_config,
    build_weak_teacher_augmentation_config,
    center_crop_eval_transform,
)
from otuformer.training.dino import (
    DINOObjective,
    check_dino_weight_norm_api_record,
    validate_dino_prototype_layer,
)
from otuformer.training.loss import (
    ArcFaceLoss,
    LOSS_REGISTRY,
    SubCenterArcFaceLoss,
    SupConLoss,
    dino_distillation_loss,
    ibot_patch_loss,
    koleo_loss,
    masked_patch_cosine_loss,
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
from otuformer.utils.paths import canonical_image_ref, path_identity
from otuformer.utils.size import (
    _positive_int,
    resolve_backbone_native_size,
    resolve_training_image_size,
    validate_input_size,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpus(cpus: int) -> None:
    if cpus > 0:
        torch.set_num_threads(cpus)


def _capture_rng_state() -> dict[str, object]:
    """Snapshot Python, NumPy, PyTorch CPU, and CUDA RNG states."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object] | None) -> None:
    """Restore a snapshot produced by :func:`_capture_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)


def _resolve_device(device: str) -> torch.device:
    requested = (device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """Name-matched EMA update.

    Positional ``zip`` is unsafe once the student owns parameters the teacher
    does not (mask token, A+ predictor), so parameters are matched by name and a
    missing teacher counterpart is a hard error.
    """
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        if name not in student_params:
            raise ValueError(
                f"Cannot EMA-update '{name}': the student module has no parameter "
                "with that name. Student-only parameters must not appear on the "
                "teacher module."
            )
        teacher_param.mul_(momentum).add_(student_params[name], alpha=1.0 - momentum)


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


_RELATIONAL_PAIR_SETS: dict[str, tuple[tuple[int, int], ...]] = {
    "cross": ((0, 1), (1, 0)),
    "full": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "same": ((0, 0), (1, 1)),
}


def _compute_global_loss(
    student_global: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    temp_student: float,
    temp_teacher: float,
    *,
    relational_pairing: str,
) -> torch.Tensor:
    """Average the selected global student/teacher view pairs.

    ``cross`` averages A-to-B and B-to-A, ``full`` averages the whole Cartesian
    product, and the internal ``same`` pairing keeps the historical
    matching-index terms for legacy checkpoints.
    """
    pairs = _RELATIONAL_PAIR_SETS.get(relational_pairing)
    if pairs is None:
        raise ValueError(
            f"Unsupported relational_pairing {relational_pairing!r}; expected one "
            f"of: {', '.join(_RELATIONAL_PAIR_SETS)}."
        )
    if len(student_global) != 2 or len(teacher_global) != 2:
        raise ValueError(
            "global relational loss needs exactly two student and two teacher "
            f"features, got {len(student_global)} and {len(teacher_global)}."
        )
    loss = torch.tensor(0.0, device=student_global[0].device)
    for student_index, teacher_index in pairs:
        loss = loss + _ssl_loss(
            student_global[student_index],
            teacher_global[teacher_index],
            temp_student,
            temp_teacher,
        )
    return loss / len(pairs)


def _compute_local_loss(
    local_views_student: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    temp_student: float,
    temp_teacher: float,
) -> torch.Tensor:
    """Compute local-to-global distillation loss following ref/ibot20260115.py.

    Averages over all local_student x teacher_global pairs.
    """
    if not local_views_student:
        return torch.tensor(0.0, device=teacher_global[0].device)
    device = local_views_student[0].device
    loss = torch.tensor(0.0, device=device)
    for lv in local_views_student:
        for t in teacher_global:
            loss += _ssl_loss(lv, t, temp_student, temp_teacher)
    loss /= max(1, len(local_views_student) * len(teacher_global))
    return loss


def _compute_dino_global_loss(
    student_logits: list[torch.Tensor],
    teacher_targets: list[torch.Tensor],
    temp_student: float,
) -> torch.Tensor:
    """Average the two cross-view pairs: student A -> teacher B, B -> A."""
    losses = [
        dino_distillation_loss(student_logits[0], teacher_targets[1], temp_student),
        dino_distillation_loss(student_logits[1], teacher_targets[0], temp_student),
    ]
    return torch.stack(losses).mean()


def _compute_dino_local_loss(
    local_logits: list[torch.Tensor],
    teacher_targets: list[torch.Tensor],
    temp_student: float,
) -> torch.Tensor:
    """Average every student-local -> teacher-global pair; zero without locals."""
    if not local_logits:
        return torch.zeros((), device=teacher_targets[0].device)
    losses = [
        dino_distillation_loss(logits, target, temp_student)
        for logits in local_logits
        for target in teacher_targets
    ]
    return torch.stack(losses).mean()


def _compute_koleo_raw(
    student_tokens: list[torch.Tensor],
    koleo_config: dict[str, Any],
) -> tuple[torch.Tensor, list[int]]:
    """Sum the per-global-crop KoLeo losses over the unmasked backbone CLS.

    Uses the existing unmasked global forwards, never locals and never a pooled
    cross-crop neighbor set, so two augmentations of one image cannot repel each
    other. Crops with fewer than two samples are skipped and counted.
    """
    device = student_tokens[0].device
    total = torch.zeros((), device=device)
    sizes: list[int] = []
    for tokens in student_tokens:
        backbone_cls = tokens[:, 0]
        count = int(backbone_cls.shape[0])
        sizes.append(count)
        if count < 2:
            continue
        total = total + koleo_loss(
            backbone_cls,
            normalize_eps=float(koleo_config["koleo_normalize_eps"]),
            distance_eps=float(koleo_config["koleo_distance_eps"]),
            log_eps=float(koleo_config["koleo_log_eps"]),
        )
    return total, sizes


def _dino_cls_diagnostics(
    teacher_targets: list[torch.Tensor], prototypes: int
) -> dict[str, float]:
    """CLS-target diagnostics over the cached assignments of both globals.

    Entropy and perplexity use finite-safe arithmetic: zero probabilities
    contribute zero entropy instead of NaN. These describe the teacher targets
    only; they are not morphology or collapse detection.
    """
    probabilities = torch.cat([t.detach().float() for t in teacher_targets], dim=0)
    positive = probabilities > 0
    log_probabilities = torch.where(
        positive, probabilities.log(), torch.zeros_like(probabilities)
    )
    entropy = float(-(probabilities * log_probabilities).sum(dim=-1).mean())

    marginal = probabilities.mean(dim=0)
    total = marginal.sum().clamp_min(1e-12)
    marginal = marginal / total
    marginal_positive = marginal > 0
    marginal_log = torch.where(
        marginal_positive, marginal.log(), torch.zeros_like(marginal)
    )
    perplexity = float(torch.exp(-(marginal * marginal_log).sum()))

    counts = torch.bincount(
        probabilities.argmax(dim=-1), minlength=prototypes
    ).float()
    active_ratio = float((counts > 0).float().mean())
    occupancy = float(counts.max() / counts.sum().clamp_min(1.0))
    return {
        "cls_assignment_entropy": entropy,
        "cls_prototype_perplexity": perplexity,
        "cls_active_prototype_ratio": active_ratio,
        "cls_max_prototype_occupancy": occupancy,
    }


def _koleo_instant_metrics(
    ssl_config: dict[str, Any],
    enabled: bool,
    koleo_raw: torch.Tensor,
    sizes: list[int],
) -> dict[str, Any]:
    """Status and counts for the optional KoLeo term (blank when N/A)."""
    weight = float(ssl_config["koleo_weight"])
    if not enabled:
        return {
            "koleo_status": "disabled",
            "koleo_loss_raw": None,
            "koleo_loss_weighted": None,
        }
    eligible = sum(size for size in sizes if size >= 2)
    skipped = sum(size for size in sizes if size < 2)
    if eligible == 0:
        # A skipped batch contributes exactly zero, still numeric.
        return {
            "koleo_status": "skipped",
            "koleo_loss_raw": 0.0,
            "koleo_loss_weighted": 0.0,
            "koleo_eligible_samples": 0,
            "koleo_skipped_samples": skipped,
        }
    raw = float(koleo_raw.item())
    return {
        "koleo_status": "active",
        "koleo_loss_raw": raw,
        "koleo_loss_weighted": weight * raw,
        "koleo_eligible_samples": eligible,
        "koleo_skipped_samples": skipped,
    }


def _update_teacher_center(
    center: torch.Tensor,
    teacher_global: torch.Tensor,
) -> torch.Tensor:
    """Update teacher center buffer with EMA of current-iteration teacher global outputs.

    Uses fixed 0.9/0.1 EMA coefficients matching ref/ibot20260115.py.
    """
    batch_mean = teacher_global.mean(dim=0, keepdim=True)
    return center * 0.9 + batch_mean * 0.1


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


def _ssl_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    temp_student: float,
    temp_teacher: float,
) -> torch.Tensor:
    student_sim = student_out @ teacher_out.T / temp_student
    with torch.no_grad():
        teacher_sim = teacher_out @ teacher_out.T / temp_teacher
        teacher_sim = teacher_sim - teacher_sim.mean(dim=1, keepdim=True)
        teacher_probs = F.softmax(teacher_sim, dim=1)
    student_log_probs = F.log_softmax(student_sim, dim=1)
    return -(teacher_probs * student_log_probs).sum(dim=1).mean()


PATCH_CENTER_MOMENTUM = 0.9
_DEFAULT_PATCH_RATIOS = {
    "consistency": 0.30,
    "random": 0.30,
    "blockwise": 0.20,
    "hybrid": 0.30,
}


def _parse_requested_mask_ratio(value: object) -> object:
    """Return ``"auto"`` or a validated float in ``(0, 1)``."""
    if value is None or value == "auto":
        return "auto"
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"mask_ratio must be 'auto' or a float in (0, 1), got {value!r}."
        ) from None
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"mask_ratio must be 'auto' or a float in (0, 1), got {value!r}."
        )
    return ratio


def _resolve_patch_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> dict[str, object]:
    """Resolve the effective patch-loss configuration for a run or resume.

    One helper owns resolution so startup validation, checkpoint comparison,
    logging and tests all read the same effective values.
    """
    if checkpoint is not None:
        return _resolve_resumed_patch_config(args, checkpoint)
    return _resolve_new_patch_config(args)


_PATCH_CONFIG_KEYS = (
    "patch_loss",
    "masking_strategy",
    "mask_ratio_requested",
    "mask_ratio_resolved",
    "ibot_prototypes",
    "patch_target_layers",
    "patch_center_momentum",
)
_PATCH_RESUME_COMPARISON_KEYS = (
    "patch_loss",
    "masking_strategy",
    "mask_ratio_resolved",
    "ibot_prototypes",
    "patch_target_layers",
    "patch_center_momentum",
)


def _validate_patch_resume(current: dict, saved: dict) -> None:
    """Reject any effective patch setting that differs from the checkpoint."""
    for key in _PATCH_RESUME_COMPARISON_KEYS:
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"Cannot resume: patch setting '{key}' is {saved.get(key)!r} in the "
                f"checkpoint but {current.get(key)!r} for this invocation. Start a "
                "new run to change patch settings."
            )


def _resolve_resumed_patch_config(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> dict[str, object]:
    cfg = checkpoint.get("config")
    if not isinstance(cfg, dict):
        cfg = {}
    if "patch_loss" not in cfg:
        return _resolve_legacy_patch_config(args, checkpoint)

    saved = {key: cfg.get(key) for key in _PATCH_CONFIG_KEYS}
    merged = argparse.Namespace(
        patch_loss=(
            getattr(args, "patch_loss", "consistency")
            if getattr(args, "patch_loss_explicit", False)
            else saved["patch_loss"]
        ),
        masking_strategy=(
            getattr(args, "masking_strategy", "random")
            if getattr(args, "masking_strategy_explicit", False)
            else (saved["masking_strategy"] or "random")
        ),
        mask_ratio=(
            getattr(args, "mask_ratio", "auto")
            if getattr(args, "mask_ratio_explicit", False)
            else saved["mask_ratio_requested"]
        ),
        ibot_prototypes=(
            getattr(args, "ibot_prototypes", 512)
            if getattr(args, "ibot_prototypes_explicit", False)
            else (saved["ibot_prototypes"] or 512)
        ),
        lambda_mask=getattr(args, "lambda_mask", 1.0),
        mask_ratio_explicit=bool(getattr(args, "mask_ratio_explicit", False)),
        patch_loss_explicit=bool(getattr(args, "patch_loss_explicit", False)),
        masking_strategy_explicit=bool(
            getattr(args, "masking_strategy_explicit", False)
        ),
        ibot_prototypes_explicit=bool(
            getattr(args, "ibot_prototypes_explicit", False)
        ),
        lambda_mask_explicit=bool(getattr(args, "lambda_mask_explicit", False)),
    )
    current = _resolve_new_patch_config(merged)
    _validate_patch_resume(current, saved)
    return current


def _resolve_legacy_patch_config(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> dict[str, object]:
    """Resolve a pre-v0.7 checkpoint to consistency with the historical ratio."""
    requested_mode = getattr(args, "patch_loss", "consistency")
    if getattr(args, "patch_loss_explicit", False) and requested_mode != "consistency":
        raise ValueError(
            f"Cannot resume a legacy checkpoint as patch_loss={requested_mode!r}: it "
            "has no mask token, predictor, or iBOT state. Start a new run for "
            "masked-feature or ibot."
        )
    saved_args = checkpoint.get("args")
    raw_ratio = saved_args.get("mask_ratio") if isinstance(saved_args, dict) else None
    if raw_ratio is None:
        saved_ratio = 0.50
    else:
        try:
            saved_ratio = float(raw_ratio)
        except (TypeError, ValueError):
            raise ValueError(
                "Cannot resume: legacy checkpoint records an invalid mask_ratio "
                f"{raw_ratio!r}."
            ) from None
    if not 0.0 < saved_ratio < 1.0:
        raise ValueError(
            "Cannot resume: legacy checkpoint records mask_ratio "
            f"{saved_ratio!r}, which is outside (0, 1)."
        )
    if getattr(args, "mask_ratio_explicit", False):
        requested = _parse_requested_mask_ratio(getattr(args, "mask_ratio", "auto"))
        if requested != "auto" and abs(float(requested) - saved_ratio) > 1e-9:
            raise ValueError(
                f"Cannot resume: --mask-ratio {requested} differs from the legacy "
                f"checkpoint value {saved_ratio}. Omit --mask-ratio to inherit it."
            )
    if getattr(args, "masking_strategy_explicit", False):
        raise ValueError(
            "masking_strategy is not applicable to a legacy consistency checkpoint; "
            "omit it or start a new run."
        )
    if getattr(args, "ibot_prototypes_explicit", False):
        raise ValueError(
            "ibot_prototypes is not applicable to a legacy consistency checkpoint; "
            "omit it or start a new run."
        )
    return {
        "patch_loss": "consistency",
        "masking_strategy": None,
        "mask_ratio_requested": saved_ratio,
        "mask_ratio_resolved": saved_ratio,
        "ibot_prototypes": None,
        "patch_target_layers": None,
        "patch_center_momentum": None,
    }


def _resolve_new_patch_config(args: argparse.Namespace) -> dict[str, object]:
    mode = getattr(args, "patch_loss", "consistency")
    if mode not in PATCH_LOSS_MODES:
        raise ValueError(
            f"Unsupported patch_loss {mode!r}; choose from: "
            f"{', '.join(PATCH_LOSS_MODES)}."
        )
    strategy = getattr(args, "masking_strategy", "random")
    requested = _parse_requested_mask_ratio(getattr(args, "mask_ratio", "auto"))
    ratio_explicit = bool(getattr(args, "mask_ratio_explicit", False)) or (
        requested != "auto"
    )
    strategy_explicit = bool(getattr(args, "masking_strategy_explicit", False))
    prototypes_explicit = bool(getattr(args, "ibot_prototypes_explicit", False))
    lambda_mask = getattr(args, "lambda_mask", 1.0)
    lambda_explicit = bool(getattr(args, "lambda_mask_explicit", False))
    prototypes = getattr(args, "ibot_prototypes", 512)

    if mode == "none":
        if ratio_explicit:
            raise ValueError(
                "mask_ratio is not applicable to patch_loss='none'; omit it "
                "or choose a masking patch loss."
            )
        if strategy_explicit:
            raise ValueError(
                "masking_strategy is not applicable to patch_loss='none'; omit it "
                "or choose masked-feature/ibot."
            )
        if prototypes_explicit:
            raise ValueError(
                "ibot_prototypes is not applicable to patch_loss='none'; omit it "
                "or choose ibot."
            )
        if lambda_explicit and float(lambda_mask) != 1.0:
            raise ValueError(
                "lambda_mask is not applicable to patch_loss='none'; omit it or "
                "reset it to 1.0."
            )
        strategy_effective: str | None = None
        resolved: float | None = None
        prototypes_effective: int | None = None
        requested_effective: object = None
    elif mode == "consistency":
        if strategy_explicit:
            raise ValueError(
                "masking_strategy is not applicable to patch_loss='consistency'; "
                "consistency selects visible positions and never masks input."
            )
        if prototypes_explicit:
            raise ValueError(
                "ibot_prototypes is not applicable to patch_loss='consistency'; "
                "omit it or choose ibot."
            )
        strategy_effective = None
        resolved = (
            float(requested)
            if requested != "auto"
            else _DEFAULT_PATCH_RATIOS["consistency"]
        )
        prototypes_effective = None
        requested_effective = requested
    else:
        if strategy not in MASKING_STRATEGIES:
            raise ValueError(
                f"Unsupported masking_strategy {strategy!r}; choose from: "
                f"{', '.join(MASKING_STRATEGIES)}."
            )
        if mode == "masked-feature" and prototypes_explicit:
            raise ValueError(
                "ibot_prototypes is not applicable to patch_loss='masked-feature'; "
                "omit it or choose ibot."
            )
        if mode == "ibot":
            if (
                isinstance(prototypes, bool)
                or not isinstance(prototypes, int)
                or prototypes < IBOT_PROTOTYPES_MIN
            ):
                raise ValueError(
                    f"Unsupported ibot_prototypes {prototypes!r}; must be an integer "
                    f">= {IBOT_PROTOTYPES_MIN}. Larger dictionaries suit larger "
                    "datasets (for example 4096 or 16384)."
                )
            prototypes_effective = int(prototypes)
        else:
            prototypes_effective = None
        strategy_effective = strategy
        resolved = (
            float(requested)
            if requested != "auto"
            else _DEFAULT_PATCH_RATIOS[strategy]
        )
        requested_effective = requested

    return {
        "patch_loss": mode,
        "masking_strategy": strategy_effective,
        "mask_ratio_requested": requested_effective,
        "mask_ratio_resolved": resolved,
        "ibot_prototypes": prototypes_effective,
        "patch_target_layers": (
            PATCH_TARGET_LAYERS if mode == "masked-feature" else None
        ),
        "patch_center_momentum": (
            PATCH_CENTER_MOMENTUM if mode == "ibot" else None
        ),
    }


# --- Mask generation --------------------------------------------------------

_MAX_BLOCK_ROUNDS = 16


def _target_patch_count(ratio: float, patch_count: int) -> int:
    """Resolve the number of masked positions, leaving one visible patch."""
    return min(max(round(ratio * patch_count), 1), patch_count - 1)


def _legal_block_shapes(grid_size: tuple[int, int]) -> list[tuple[int, int]]:
    """Enumerate integer block shapes legal for ``(H, W)``.

    A shape needs sides of at least two patches, an aspect ratio within
    ``[0.5, 2.0]``, and an area of at most ``floor(0.20 * H * W)``.
    """
    height, width = int(grid_size[0]), int(grid_size[1])
    if height < 1 or width < 1:
        return []
    max_area = (20 * height * width) // 100
    shapes: list[tuple[int, int]] = []
    for block_h in range(2, height + 1):
        for block_w in range(2, width + 1):
            if block_h * block_w > max_area:
                continue
            if 0.5 <= block_h / block_w <= 2.0:
                shapes.append((block_h, block_w))
    return shapes


def _validate_blockwise_grid(grid_size: tuple[int, int]) -> None:
    """Fail fast when no legal block rectangle fits the actual patch grid."""
    if _legal_block_shapes(grid_size):
        return
    height, width = int(grid_size[0]), int(grid_size[1])
    raise ValueError(
        "blockwise masking is infeasible for patch grid "
        f"({height}, {width}): no integer rectangle with sides >= 2, aspect ratio "
        "in [0.5, 2.0] and area <= floor(0.20 * H * W) fits the grid. Choose a "
        "different input size or use --masking-strategy random."
    )


def _sample_rectangle(
    shapes: list[tuple[int, int]],
    grid_size: tuple[int, int],
    device: torch.device,
) -> torch.BoolTensor:
    """Sample one rectangle and return its flattened ``[H * W]`` position mask."""
    height, width = int(grid_size[0]), int(grid_size[1])
    index = int(torch.randint(len(shapes), (1,), device=device).item())
    block_h, block_w = shapes[index]
    top = int(torch.randint(height - block_h + 1, (1,), device=device).item())
    left = int(torch.randint(width - block_w + 1, (1,), device=device).item())
    mask = torch.zeros(height * width, dtype=torch.bool, device=device)
    mask.view(height, width)[top : top + block_h, left : left + block_w] = True
    return mask


def _random_positions_mask(
    count: int,
    total: int,
    device: torch.device,
    available: torch.Tensor | None = None,
) -> torch.BoolTensor:
    """Sample ``count`` unique positions, optionally excluding ``available``."""
    mask = torch.zeros(total, dtype=torch.bool, device=device)
    if count <= 0:
        return mask
    if available is None:
        candidates = torch.arange(total, device=device)
    else:
        candidates = torch.nonzero(~available, as_tuple=False).flatten()
    count = min(count, int(candidates.numel()))
    if count <= 0:
        return mask
    picked = candidates[torch.randperm(candidates.numel(), device=device)[:count]]
    mask[picked] = True
    return mask


def _sample_blockwise_positions(
    target: int, grid_size: tuple[int, int], device: torch.device
) -> torch.BoolTensor:
    """Merge overlapping rectangle proposals until ``target`` is reached."""
    height, width = int(grid_size[0]), int(grid_size[1])
    total = height * width
    shapes = _legal_block_shapes((height, width))
    selected = torch.zeros(total, dtype=torch.bool, device=device)
    rounds = 0
    while int(selected.sum()) < target and rounds < _MAX_BLOCK_ROUNDS:
        proposals = int(torch.randint(2, 5, (1,), device=device).item())
        for _ in range(proposals):
            selected |= _sample_rectangle(shapes, (height, width), device)
        rounds += 1
    return selected


def _exact_mask(
    selected: torch.BoolTensor,
    target: int,
    total: int,
    device: torch.device,
) -> torch.BoolTensor:
    """Trim or random-fill ``selected`` so it holds exactly ``target`` positions."""
    count = int(selected.sum())
    if count == target:
        return selected
    if count > target:
        index = torch.nonzero(selected, as_tuple=False).flatten()
        keep = index[torch.randperm(index.numel(), device=device)[:target]]
        trimmed = torch.zeros(total, dtype=torch.bool, device=device)
        trimmed[keep] = True
        return trimmed
    return selected | _random_positions_mask(
        target - count, total, device, available=selected
    )


def _sample_patch_masks(
    batch_size: int,
    grid_size: tuple[int, int],
    ratio: float,
    strategy: str,
    device: torch.device,
) -> torch.BoolTensor:
    """Sample independent masks for ``batch_size`` samples and one global view."""
    if strategy not in MASKING_STRATEGIES:
        raise ValueError(
            f"Unsupported masking_strategy {strategy!r}; choose from: "
            f"{', '.join(MASKING_STRATEGIES)}."
        )
    height, width = int(grid_size[0]), int(grid_size[1])
    total = height * width
    target = _target_patch_count(ratio, total)
    if strategy in ("blockwise", "hybrid"):
        _validate_blockwise_grid((height, width))

    rows: list[torch.BoolTensor] = []
    for _ in range(batch_size):
        if strategy == "random":
            row = _random_positions_mask(target, total, device)
        elif strategy == "blockwise":
            blocks = _sample_blockwise_positions(target, (height, width), device)
            row = _exact_mask(blocks, target, total, device)
        else:
            block_share = target // 2
            blocks = _sample_blockwise_positions(
                block_share, (height, width), device
            )
            blocks = _exact_mask(blocks, block_share, total, device)
            row = _exact_mask(blocks, target, total, device)
        rows.append(row)
    return torch.stack(rows, dim=0)


_PRETRAIN_BASE_FIELDS = ["iteration", "epoch", "step"]
_PRETRAIN_LEGACY_METRIC_FIELDS = [
    "loss",
    "global_loss",
    "local_loss",
    "mask_loss",
    "lr",
    "teacher_temp",
    "grad_norm",
    "feature_std",
    "embedding_norm_mean",
    "cls_token_norm_mean",
    "teacher_center_norm",
    "cosine_similarity",
]
_PRETRAIN_PATCH_FIELDS = [
    "patch_loss_raw",
    "patch_loss_weighted",
    "patch_loss_fraction_of_total",
    "requested_mask_ratio",
    "selected_patch_ratio",
    "actual_mask_ratio_mean",
    "actual_mask_ratio_min",
    "actual_mask_ratio_max",
    "prototype_perplexity",
    "active_prototype_ratio",
    "assignment_entropy",
    "max_prototype_occupancy",
    "patch_center_norm",
]
_PRETRAIN_LEGACY_FIELDS = _PRETRAIN_BASE_FIELDS + _PRETRAIN_LEGACY_METRIC_FIELDS
# Frozen v0.7.0 header: the extended schema is derived from it, so "current" can
# never silently drop a column the previous release wrote.
_PRETRAIN_V070_FIELDS = _PRETRAIN_LEGACY_FIELDS + _PRETRAIN_PATCH_FIELDS
_PRETRAIN_SSL_SETTING_FIELDS = ["ssl_method", "dino_centering", "koleo_weight"]
_PRETRAIN_SSL_METRIC_FIELDS = [
    "cls_center_norm",
    "global_loss_weighted",
    "local_loss_weighted",
    "cls_assignment_entropy",
    "cls_prototype_perplexity",
    "cls_active_prototype_ratio",
    "cls_max_prototype_occupancy",
    "koleo_loss_raw",
    "koleo_loss_weighted",
    "koleo_status",
    "koleo_eligible_samples",
    "koleo_skipped_samples",
]
_PRETRAIN_CURRENT_FIELDS = (
    _PRETRAIN_V070_FIELDS
    + _PRETRAIN_SSL_SETTING_FIELDS
    + _PRETRAIN_SSL_METRIC_FIELDS
)
# Numeric instant-metric columns: blank/N-A cells become missing values in plots.
_PRETRAIN_PLOT_NUMERIC_FIELDS = (
    _PRETRAIN_LEGACY_METRIC_FIELDS
    + _PRETRAIN_PATCH_FIELDS
    + _PRETRAIN_SSL_METRIC_FIELDS
)
_FINETUNE_FIELDS = [
    "loss",
    "lr",
    "grad_norm",
    "feature_std",
    "embedding_norm_mean",
    "cls_token_norm_mean",
]


@dataclass
class InstantMetricsLogger:
    path: Path
    mode: str = "pretrain"

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "pretrain":
            self.fieldnames = list(_PRETRAIN_CURRENT_FIELDS)
        else:
            self.fieldnames = list(_PRETRAIN_BASE_FIELDS) + list(_FINETUNE_FIELDS)
        if self.path.exists() and self.path.stat().st_size > 0:
            self._validate_or_migrate_existing()
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def _read_existing_csv(self) -> tuple[list[str], list[list[str]]]:
        try:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError(
                        f"instant-metrics schema in {self.path} is empty; refusing "
                        "to append."
                    ) from None
                rows: list[list[str]] = []
                for line_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"instant-metrics schema in {self.path} is malformed: "
                            f"row {line_number} has {len(row)} fields but the header "
                            f"has {len(header)}."
                        )
                    rows.append(row)
        except csv.Error as exc:
            raise ValueError(
                f"instant-metrics schema in {self.path} is malformed CSV: {exc}"
            ) from exc
        return header, rows

    def _validate_or_migrate_existing(self) -> None:
        header, rows = self._read_existing_csv()
        if len(set(header)) != len(header):
            raise ValueError(
                f"instant-metrics schema in {self.path} has duplicate columns; "
                "refusing to append."
            )
        if header == self.fieldnames:
            return
        if self.mode == "pretrain" and header in (
            _PRETRAIN_LEGACY_FIELDS,
            _PRETRAIN_V070_FIELDS,
        ):
            self._migrate_pretrain_schema(header, rows)
            return
        raise ValueError(
            f"Unrecognized instant-metrics schema in {self.path}: {header}. "
            "Refusing to append. Move or remove the file to start a new log."
        )

    def _migrate_pretrain_schema(
        self, header: list[str], rows: list[list[str]]
    ) -> None:
        """Atomically rewrite a recognized prior header to the current schema."""
        import os
        import tempfile

        index = {name: position for position, name in enumerate(header)}
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                writer = csv.writer(handle)
                writer.writerow(self.fieldnames)
                for row in rows:
                    writer.writerow(
                        [
                            row[index[name]] if name in index else ""
                            for name in self.fieldnames
                        ]
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            for leftover in self.path.parent.glob(self.path.name + ".*.tmp"):
                leftover.unlink(missing_ok=True)

    def log(self, **kwargs: Any) -> None:
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    @staticmethod
    def _plottable_series(df_instant, name: str):
        """One numeric column with blanks as missing, or ``None`` if absent.

        An all-missing column returns ``None`` so callers skip it instead of
        drawing an empty legend entry with no data.
        """
        if name not in df_instant.columns:
            return None
        values = df_instant[name]
        return None if values.isna().all() else values

    def plot(self, out_dir: Path, metrics_path: Path | None = None) -> None:
        if not self.path.exists():
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            print("[Warning] matplotlib/pandas missing -> skip training curves.")
            return

        df_instant = pd.read_csv(self.path)
        if df_instant.empty:
            return
        # Optional metric cells can be blank (historical rows, N/A under the
        # other method). They become missing values, never fabricated zeros, and
        # an all-missing series is skipped instead of drawn.
        for name in _PRETRAIN_PLOT_NUMERIC_FIELDS:
            if name in df_instant.columns:
                df_instant[name] = pd.to_numeric(df_instant[name], errors="coerce")

        def series(name: str):
            return self._plottable_series(df_instant, name)

        df_metrics = None
        if metrics_path is not None and metrics_path.exists():
            df_metrics = pd.read_csv(metrics_path)
            if "epoch" in df_metrics.columns:
                df_metrics = df_metrics[df_metrics["epoch"].notna()]

        # The panel count grew with the DINO/KoLeo diagnostics, so the grid is
        # sized for them; unused cells are switched off at the end.
        fig, axes = plt.subplots(4, 4, figsize=(26, 20))
        axes = axes.flatten()
        plot_idx = 0

        # Mixed history: rows written before the patch schema have no
        # ``patch_loss_raw`` and only ``mask_loss``. Fall back per row so earlier
        # observations are not dropped once a later row fills the new column, and
        # keep working from a pure legacy file that has no raw column at all.
        patch_series = self._plottable_series(df_instant, "patch_loss_raw")
        legacy_mask = self._plottable_series(df_instant, "mask_loss")
        if patch_series is not None:
            patch_series = (
                patch_series.fillna(legacy_mask)
                if legacy_mask is not None
                else patch_series
            )
        else:
            patch_series = legacy_mask
        if patch_series is not None and patch_series.isna().all():
            patch_series = None
        ax = axes[plot_idx]
        loss_lines = {
            "Total loss": self._plottable_series(df_instant, "loss"),
            "Global loss": self._plottable_series(df_instant, "global_loss"),
            "Local loss": self._plottable_series(df_instant, "local_loss"),
            "Patch loss (raw)": patch_series,
        }
        for label, values in loss_lines.items():
            if values is None:
                continue
            ax.plot(
                df_instant["iteration"],
                values,
                label=label,
                linewidth=1.5 if label == "Total loss" else 1.0,
                alpha=1.0 if label == "Total loss" else 0.8,
            )
        # KoLeo is signed, so a total loss can legitimately reach non-positive
        # values. A log axis cannot show those; keep it only when every drawn
        # value is strictly positive.
        drawn = [
            value
            for values in loss_lines.values()
            if values is not None
            for value in values.dropna()
        ]
        if drawn and all(value > 0 for value in drawn):
            ax.set_yscale("log")
            ax.set_ylabel("Loss (log scale)")
        elif drawn:
            # symlog keeps small positive values readable without clipping or
            # hiding the sign of a negative contribution.
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylabel("Loss (symlog scale; signed values kept)")
        ax.set_xlabel("Iteration")
        ax.set_title("Training Losses")
        if drawn:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plot_idx += 1

        ax = axes[plot_idx]
        has_cos = series("cosine_similarity") is not None
        has_grad = series("grad_norm") is not None
        second_panel_has_data = False
        if has_cos and has_grad:
            ax_twin = ax.twinx()
            ax.plot(
                df_instant["iteration"],
                df_instant["cosine_similarity"],
                label="Cosine Similarity",
                color="tab:blue",
                linewidth=1.5,
            )
            ax.set_ylabel("Cosine Similarity", color="tab:blue")
            ax.tick_params(axis="y", labelcolor="tab:blue")
            ax_twin.plot(
                df_instant["iteration"],
                df_instant["grad_norm"],
                label="Grad Norm",
                color="tab:orange",
                linewidth=1.5,
                alpha=0.8,
            )
            ax_twin.set_ylabel("Gradient Norm", color="tab:orange")
            ax_twin.tick_params(axis="y", labelcolor="tab:orange")
            ax.set_title("Cosine Similarity & Grad Norm")
            second_panel_has_data = True
        elif has_cos:
            ax.plot(
                df_instant["iteration"],
                df_instant["cosine_similarity"],
                label="Cosine Similarity",
                color="tab:blue",
            )
            ax.set_title("Cosine Similarity")
            second_panel_has_data = True
        elif has_grad:
            ax.plot(
                df_instant["iteration"],
                df_instant["grad_norm"],
                label="Grad Norm",
                color="tab:orange",
            )
            ax.set_title("Gradient Norm")
            second_panel_has_data = True
        if second_panel_has_data:
            ax.set_xlabel("Iteration")
            if has_cos and has_grad:
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r,
                    labels_l + labels_r,
                    fontsize=8,
                    loc="best",
                )
            else:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        else:
            ax.axis("off")

        ax = axes[plot_idx]
        has_left = any(
            series(c) is not None for c in ["feature_std", "embedding_norm_mean"]
        )
        has_cls = series("cls_token_norm_mean") is not None
        if has_left:
            if series("feature_std") is not None:
                ax.plot(
                    df_instant["iteration"],
                    df_instant["feature_std"],
                    label="Feature Std",
                    color="tab:green",
                )
            if series("embedding_norm_mean") is not None:
                ax.plot(
                    df_instant["iteration"],
                    df_instant["embedding_norm_mean"],
                    label="Embedding Norm",
                    color="tab:cyan",
                )
        if has_cls:
            ax_twin = ax.twinx()
            ax_twin.plot(
                df_instant["iteration"],
                df_instant["cls_token_norm_mean"],
                label="CLS Token Norm",
                color="tab:purple",
                alpha=0.8,
            )
            ax_twin.set_ylabel("CLS Token Norm", color="tab:purple")
            ax_twin.tick_params(axis="y", labelcolor="tab:purple")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Feature Std / Embedding Norm")
        ax.set_title("Feature Statistics")
        if has_cls or has_left:
            lines_l, labels_l = ax.get_legend_handles_labels()
            lines_r, labels_r = (
                ax_twin.get_legend_handles_labels() if has_cls else ([], [])
            )
            if labels_l or labels_r:
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=8, loc="best"
                )
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        ax = axes[plot_idx]
        fourth_panel_has_data = False
        has_teacher_center = series("teacher_center_norm") is not None
        has_cls_center = series("cls_center_norm") is not None
        if has_teacher_center:
            ax.plot(
                df_instant["iteration"],
                df_instant["teacher_center_norm"],
                label="Teacher center (D, relational)",
            )
            fourth_panel_has_data = True
        if has_cls_center:
            ax.plot(
                df_instant["iteration"],
                df_instant["cls_center_norm"],
                label="CLS center (K, DINO)",
                color="tab:green",
            )
            fourth_panel_has_data = True
        has_lr = series("lr") is not None
        if has_lr:
            ax_twin = ax.twinx()
            ax_twin.plot(
                df_instant["iteration"],
                df_instant["lr"],
                label="Learning Rate",
                color="tab:red",
            )
            # An all-zero LR (for example a frozen-optimizer test run) has no
            # positive values, and matplotlib warns when log-scaling it.
            if (df_instant["lr"] > 0).any():
                ax_twin.set_yscale("log")
            ax_twin.set_ylabel("Learning Rate", color="tab:red")
            ax_twin.tick_params(axis="y", labelcolor="tab:red")
            fourth_panel_has_data = True
        if has_teacher_center or has_cls_center:
            ax.set_ylabel("Center Norm")
        if fourth_panel_has_data:
            ax.set_xlabel("Iteration")
            if has_lr and (has_teacher_center or has_cls_center):
                ax.set_title("Center Norms & Learning Rate")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r,
                    labels_l + labels_r,
                    fontsize=8,
                    loc="best",
                )
            elif has_teacher_center or has_cls_center:
                ax.set_title("Center Norms")
                ax.legend(fontsize=8)
            else:
                ax.set_title("Learning Rate")
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(lines_r, labels_r, fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        else:
            ax.axis("off")

        # Teacher temperature, weighted contributions and the assignment
        # diagnostics that reveal target sharpness. Without these the DINO runs
        # have no per-iteration view of what the CLS objective is actually doing.
        weighted = [
            name
            for name in (
                "global_loss_weighted",
                "local_loss_weighted",
                "patch_loss_weighted",
                "koleo_loss_weighted",
            )
            if series(name) is not None
        ]
        teacher_temp_series = series("teacher_temp")
        if teacher_temp_series is not None or weighted:
            ax = axes[plot_idx]
            left_drawn = False
            if teacher_temp_series is not None:
                ax.plot(
                    df_instant["iteration"],
                    teacher_temp_series,
                    label="Teacher temperature",
                    color="tab:red",
                )
                left_drawn = True
                ax.set_ylabel("Teacher temperature", color="tab:red")
                ax.tick_params(axis="y", labelcolor="tab:red")
            if weighted:
                ax_twin = ax.twinx()
                for name in weighted:
                    ax_twin.plot(
                        df_instant["iteration"],
                        series(name),
                        label=name,
                        alpha=0.8,
                    )
                ax_twin.set_ylabel("Weighted contribution")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best"
                )
            elif left_drawn:
                ax.legend(fontsize=8)
            ax.set_xlabel("Iteration")
            ax.set_title("Teacher Temperature & Weighted Losses")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # DINO CLS assignment health. The statistics pool is the cached teacher
        # targets of both global crops (2 x batch), not the whole dataset, and
        # the active-prototype ratio divides by K.
        cls_entropy = series("cls_assignment_entropy")
        cls_occupancy = series("cls_active_prototype_ratio")
        if cls_entropy is not None or cls_occupancy is not None:
            ax = axes[plot_idx]
            drawn = False
            if cls_entropy is not None:
                ax.plot(
                    df_instant["iteration"],
                    cls_entropy,
                    label="Target entropy (nats)",
                    color="tab:blue",
                )
                drawn = True
            cls_perplexity = series("cls_prototype_perplexity")
            if cls_perplexity is not None:
                ax.plot(
                    df_instant["iteration"],
                    cls_perplexity,
                    label="Marginal perplexity",
                    color="tab:cyan",
                    alpha=0.8,
                )
                drawn = True
            ax.set_ylabel("Entropy (nats) / Perplexity", color="tab:blue")
            if cls_occupancy is not None:
                ax_twin = ax.twinx()
                ax_twin.plot(
                    df_instant["iteration"],
                    cls_occupancy,
                    label="Active prototype ratio (K denominator)",
                    color="tab:green",
                    alpha=0.8,
                )
                cls_occupancy_max = series("cls_max_prototype_occupancy")
                if cls_occupancy_max is not None:
                    ax_twin.plot(
                        df_instant["iteration"],
                        cls_occupancy_max,
                        label="Max occupancy (batch)",
                        color="tab:olive",
                        alpha=0.8,
                    )
                ax_twin.set_ylabel("Ratio", color="tab:green")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best"
                )
            elif drawn:
                ax.legend(fontsize=7)
            ax.set_xlabel("Iteration")
            ax.set_title("DINO CLS Target Health (2 x batch pool)")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # iBOT patch diagnostics use a different statistics pool, so they get
        # their own panel instead of sharing the CLS numbers.
        patch_entropy = series("assignment_entropy")
        patch_occupancy = series("active_prototype_ratio")
        if patch_entropy is not None or patch_occupancy is not None:
            ax = axes[plot_idx]
            drawn = False
            if patch_entropy is not None:
                ax.plot(
                    df_instant["iteration"],
                    patch_entropy,
                    label="Patch target entropy (nats)",
                    color="tab:blue",
                )
                drawn = True
            patch_perplexity = series("prototype_perplexity")
            if patch_perplexity is not None:
                ax.plot(
                    df_instant["iteration"],
                    patch_perplexity,
                    label="Patch marginal perplexity",
                    color="tab:cyan",
                    alpha=0.8,
                )
                drawn = True
            ax.set_ylabel("Entropy (nats) / Perplexity", color="tab:blue")
            if patch_occupancy is not None:
                ax_twin = ax.twinx()
                ax_twin.plot(
                    df_instant["iteration"],
                    patch_occupancy,
                    label="Active prototype ratio",
                    color="tab:green",
                    alpha=0.8,
                )
                patch_occupancy_max = series("max_prototype_occupancy")
                if patch_occupancy_max is not None:
                    ax_twin.plot(
                        df_instant["iteration"],
                        patch_occupancy_max,
                        label="Max occupancy",
                        color="tab:olive",
                        alpha=0.8,
                    )
                ax_twin.set_ylabel("Ratio", color="tab:green")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best"
                )
            elif drawn:
                ax.legend(fontsize=7)
            ax.set_xlabel("Iteration")
            ax.set_title("iBOT Patch Target Health")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Signed KoLeo contribution; absent when the regularizer is disabled.
        koleo_raw = series("koleo_loss_raw")
        koleo_weighted = series("koleo_loss_weighted")
        if koleo_raw is not None or koleo_weighted is not None:
            ax = axes[plot_idx]
            if koleo_raw is not None:
                ax.plot(
                    df_instant["iteration"],
                    koleo_raw,
                    label="KoLeo loss (raw, signed)",
                    color="tab:purple",
                )
            if koleo_weighted is not None:
                ax.plot(
                    df_instant["iteration"],
                    koleo_weighted,
                    label="KoLeo loss (weighted, signed)",
                    color="tab:pink",
                    alpha=0.8,
                )
            skips = series("koleo_skipped_samples")
            if skips is not None and (skips > 0).any():
                # A skipped batch has no observed loss, so its count is drawn on
                # a second axis rather than read as a loss value.
                ax_twin = ax.twinx()
                ax_twin.plot(
                    df_instant["iteration"],
                    skips,
                    label="Skipped samples",
                    color="tab:gray",
                    linestyle="--",
                )
                ax_twin.set_ylabel("Skipped samples")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best"
                )
            else:
                ax.legend(fontsize=7)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("KoLeo loss (signed)")
            ax.set_title("KoLeo Contribution")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        if df_metrics is not None and not df_metrics.empty:
            for cols, title in [
                (["Recall@1", "Recall@5", "Recall@10"], "Recall@K"),
                (["kNN_Acc_k1", "kNN_Acc_k5", "kNN_Acc_k20"], "kNN Accuracy"),
                (["NMI", "ARI"], "Clustering"),
                (["mAP", "Linear_Probing_Acc", "Linear_Probing_Balanced_Acc"], "Retrieval/Probe"),
                (["Silhouette_Score", "Purity"], "Structure Quality"),
            ]:
                if plot_idx >= len(axes):
                    break
                ax = axes[plot_idx]
                use_twin = title in {"Structure Quality", "Clustering"}
                ax_right = ax.twinx() if use_twin else None
                # Detect which splits are actually present to avoid empty split labels
                splits_present = [
                    s
                    for s in ["train", "test"]
                    if not df_metrics[df_metrics.get("split", "") == s].empty
                ]
                for split, marker in [("train", "o"), ("test", "s")]:
                    if split not in splits_present:
                        continue
                    sub = df_metrics[df_metrics.get("split", "") == split]
                    multi_split = len(splits_present) > 1
                    for col in cols:
                        if col in sub.columns:
                            numeric = pd.to_numeric(sub[col], errors="coerce")
                            if numeric.notna().any():
                                if use_twin and title == "Structure Quality":
                                    target_ax = (
                                        ax_right if col == "Silhouette_Score" else ax
                                    )
                                elif use_twin and title == "Clustering":
                                    target_ax = ax_right if col == "ARI" else ax
                                else:
                                    target_ax = ax
                                # Only append split suffix when multiple splits are present
                                lbl = f"{col} ({split})" if multi_split else col
                                color = None
                                if use_twin and title == "Structure Quality":
                                    color = (
                                        "tab:red"
                                        if col == "Silhouette_Score"
                                        else "tab:blue"
                                    )
                                elif use_twin and title == "Clustering":
                                    color = "tab:red" if col == "ARI" else "tab:blue"
                                target_ax.plot(
                                    sub["epoch"],
                                    numeric,
                                    marker=marker,
                                    label=lbl,
                                    color=color,
                                )
                ax.set_xlabel("Epoch")
                ax.set_title(title)
                if use_twin and title == "Structure Quality":
                    ax.set_ylabel("Purity", color="tab:blue")
                    ax.tick_params(axis="y", labelcolor="tab:blue")
                    ax_right.set_ylabel("Silhouette Score", color="tab:red")
                    ax_right.tick_params(axis="y", labelcolor="tab:red")
                    lines_l, labels_l = ax.get_legend_handles_labels()
                    lines_r, labels_r = ax_right.get_legend_handles_labels()
                    ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7)
                elif use_twin and title == "Clustering":
                    ax.set_ylabel("NMI", color="tab:blue")
                    ax.tick_params(axis="y", labelcolor="tab:blue")
                    ax_right.set_ylabel("ARI", color="tab:red")
                    ax_right.tick_params(axis="y", labelcolor="tab:red")
                    lines_l, labels_l = ax.get_legend_handles_labels()
                    lines_r, labels_r = ax_right.get_legend_handles_labels()
                    ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7)
                else:
                    ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()
        out_path = out_dir / f"training_curves_{self.mode}.pdf"
        fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[Info] Saved training curves to {out_path}")


_ENHANCED_BASE_FIELDS = ["epoch", "split"]
_ENHANCED_V070_METRIC_FIELDS = [
    "NMI",
    "ARI",
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "kNN_Acc_k1",
    "kNN_Acc_k5",
    "kNN_Acc_k20",
    "Linear_Probing_Acc",
    "mAP",
    "Silhouette_Score",
    "Purity",
]
_ENHANCED_V070_FIELDS = _ENHANCED_BASE_FIELDS + _ENHANCED_V070_METRIC_FIELDS
_ENHANCED_METRIC_FIELDS = [
    "NMI",
    "ARI",
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "kNN_Acc_k1",
    "kNN_Acc_k5",
    "kNN_Acc_k20",
    "Linear_Probing_Acc",
    "Linear_Probing_Balanced_Acc",
    "mAP",
    "Silhouette_Score",
    "Purity",
]


@dataclass
class EnhancedMetricsLogger:
    """Append-only embedding-metric log with a recognized v0.7.0 migration."""

    path: Path
    mode: str = "pretrain"

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = _ENHANCED_BASE_FIELDS + _ENHANCED_METRIC_FIELDS
        if self.path.exists() and self.path.stat().st_size > 0:
            self._validate_or_migrate_existing()
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def _read_existing_csv(self) -> tuple[list[str], list[list[str]]]:
        try:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError(
                        f"enhanced-metrics schema in {self.path} is empty; refusing "
                        "to append."
                    ) from None
                rows: list[list[str]] = []
                for line_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"enhanced-metrics schema in {self.path} is malformed: "
                            f"row {line_number} has {len(row)} fields but the header "
                            f"has {len(header)}."
                        )
                    rows.append(row)
        except csv.Error as exc:
            raise ValueError(
                f"enhanced-metrics schema in {self.path} is malformed CSV: {exc}"
            ) from exc
        return header, rows

    def _validate_or_migrate_existing(self) -> None:
        header, rows = self._read_existing_csv()
        if len(set(header)) != len(header):
            raise ValueError(
                f"enhanced-metrics schema in {self.path} has duplicate columns; "
                "refusing to append."
            )
        if header == self.fieldnames:
            return
        if header == _ENHANCED_V070_FIELDS:
            self._migrate_v070_schema(rows)
            return
        raise ValueError(
            f"Unrecognized enhanced-metrics schema in {self.path}: {header}. "
            "Refusing to append. Move or remove the file to start a new log."
        )

    def _migrate_v070_schema(self, rows: list[list[str]]) -> None:
        """Atomically add an empty balanced-probe column to a v0.7.0 log."""
        import os
        import tempfile

        index = {name: position for position, name in enumerate(_ENHANCED_V070_FIELDS)}
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                writer = csv.writer(handle)
                writer.writerow(self.fieldnames)
                for row in rows:
                    writer.writerow(
                        [
                            row[index[name]] if name in index else ""
                            for name in self.fieldnames
                        ]
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            for leftover in self.path.parent.glob(self.path.name + ".*.tmp"):
                leftover.unlink(missing_ok=True)

    def log(self, epoch: int, split: str, metrics: dict[str, Any]) -> None:
        row = {"epoch": epoch, "split": split}
        for k in self.fieldnames[2:]:
            row[k] = metrics.get(k, "")
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def _extract_tokens(model: OTUFormerEncoder, images: torch.Tensor) -> torch.Tensor:
    feats = model.backbone.forward_features(images)
    if isinstance(feats, dict):
        return feats["x"]
    return feats


def _compute_grad_norm(model: nn.Module) -> float:
    return _compute_grad_norm_parameters(model.parameters())


def _compute_grad_norm_parameters(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            n = p.grad.data.norm(2)
            total += float(n.item() ** 2)
    return float(total**0.5)


def _update_patch_center(
    patch_center: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """EMA of the iBOT patch center from current pre-EMA teacher logits."""
    return patch_center * PATCH_CENTER_MOMENTUM + teacher_logits.mean(
        dim=0, keepdim=True
    ) * (1.0 - PATCH_CENTER_MOMENTUM)


@torch.no_grad()
def _ibot_diagnostics(
    teacher_probs: torch.Tensor, patch_center: torch.Tensor
) -> dict[str, float]:
    """Per-step iBOT stability diagnostics from teacher probabilities."""
    mean_assignment = teacher_probs.mean(dim=0)
    mean_entropy = -(
        mean_assignment * torch.log(mean_assignment.clamp_min(1e-12))
    ).sum()
    hard = teacher_probs.argmax(dim=-1)
    counts = torch.bincount(
        hard, minlength=teacher_probs.shape[1]
    ).float()
    per_position_entropy = -(
        teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))
    ).sum(dim=-1).mean()
    return {
        "prototype_perplexity": float(torch.exp(mean_entropy)),
        "active_prototype_ratio": float((counts > 0).sum())
        / teacher_probs.shape[1],
        "assignment_entropy": float(per_position_entropy),
        "max_prototype_occupancy": float(counts.max())
        / teacher_probs.shape[0],
        "patch_center_norm": float(patch_center.norm()),
    }


def _mask_ratio_diagnostics(
    masks: list[torch.Tensor], patch_count: int
) -> dict[str, float]:
    """Per-sample actual masked fraction across the current step's views."""
    fractions = torch.cat(
        [mask.float().sum(dim=1) / patch_count for mask in masks]
    )
    return {
        "actual_mask_ratio_mean": float(fractions.mean()),
        "actual_mask_ratio_min": float(fractions.min()),
        "actual_mask_ratio_max": float(fractions.max()),
    }


@torch.no_grad()
def _compute_instant_metrics(
    model: OTUFormerEncoder,
    sample_view: torch.Tensor,
    device: torch.device,
) -> dict[str, float | None]:
    tokens = _extract_tokens(model, sample_view.to(device))
    cls_token_raw = tokens[:, 0]
    proj = model.projector(cls_token_raw)
    # A batch standard deviation needs at least two samples; a singleton batch is
    # a legitimate input, and reporting NaN as an observed value would be wrong.
    feature_std = (
        float(cls_token_raw.std(dim=0).mean().item())
        if cls_token_raw.shape[0] >= 2
        else None
    )
    return {
        "feature_std": feature_std,
        "embedding_norm_mean": float(proj.norm(dim=1).mean().item()),
        "cls_token_norm_mean": float(cls_token_raw.norm(dim=1).mean().item()),
    }


@torch.no_grad()
def _compute_embeddings_from_csv(
    model: OTUFormerEncoder,
    csv_path: Path,
    root_dir: Path,
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    import pandas as pd
    from PIL import Image

    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        raise ValueError(f"CSV {csv_path} missing required 'image' column")

    refs = [str(v) for v in df["image"]]
    missing_direct = [
        ref
        for ref in refs
        if _supports_recursive_lookup(ref) and not (root_dir / Path(ref)).exists()
    ]
    by_relative, by_name = ({}, {})
    if missing_direct:
        by_relative, by_name = _build_recursive_index(root_dir)
    image_paths = [
        _resolve_image_path(root_dir, ref, by_relative, by_name) for ref in refs
    ]
    labels = df["label"].astype(str).to_numpy() if "label" in df.columns else None

    tf = center_crop_eval_transform(image_size)
    all_embs: list[np.ndarray] = []
    model.eval()
    _ = num_workers
    for start in range(0, len(image_paths), max(1, batch_size)):
        batch_paths = image_paths[start : start + max(1, batch_size)]
        imgs = []
        for p in batch_paths:
            with Image.open(p) as im:
                img = im.convert("RGB")
            imgs.append(tf(img))
        batch = torch.stack(imgs, dim=0).to(device)
        # Use raw CLS token (backbone output) for evaluation, matching ref default
        # (ref uses --use_projector_output flag which defaults to False)
        tokens = _extract_tokens(model, batch)
        emb = tokens[:, 0].cpu().numpy()
        all_embs.append(emb)
    embs = (
        np.concatenate(all_embs, axis=0)
        if all_embs
        else np.zeros((0, 1), dtype=np.float32)
    )
    return embs, labels


def _maybe_subsample_for_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray | None,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_samples <= 0 or len(embeddings) <= max_samples:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(embeddings), size=max_samples, replace=False))
    if labels is None:
        return embeddings[idx], None
    return embeddings[idx], labels[idx]


def _compute_all_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray | None,
    compute_linear_probe: bool,
) -> dict[str, Any]:
    fields = {
        "NMI": "",
        "ARI": "",
        "Recall@1": "",
        "Recall@5": "",
        "Recall@10": "",
        "kNN_Acc_k1": "",
        "kNN_Acc_k5": "",
        "kNN_Acc_k20": "",
        "Linear_Probing_Acc": "",
        "Linear_Probing_Balanced_Acc": "",
        "mAP": "",
        "Silhouette_Score": "",
        "Purity": "",
    }
    if (
        labels is None
        or len(embeddings) == 0
        or len(np.unique(labels)) < 2
    ):
        return fields

    try:
        fields.update(compute_recall_at_k(embeddings, labels, k_values=[1, 5, 10]))
        fields.update(compute_knn_accuracy(embeddings, labels, k_values=[1, 5, 20]))
        fields["mAP"] = compute_map(embeddings, labels)
        clustering = compute_clustering_metrics(embeddings, labels)
        fields["NMI"] = clustering.get("NMI", "")
        fields["ARI"] = clustering.get("ARI", "")
        fields["Silhouette_Score"] = clustering.get(
            "Silhouette_Score", clustering.get("Silhouette", "")
        )
        fields["Purity"] = clustering.get("Purity", "")
        if compute_linear_probe:
            probing = compute_linear_probing_metrics(embeddings, labels)
            fields["Linear_Probing_Acc"] = probing["Linear_Probing_Acc"]
            fields["Linear_Probing_Balanced_Acc"] = probing[
                "Linear_Probing_Balanced_Acc"
            ]
    except Exception as exc:
        print(f"[Warning] Error computing metrics: {exc}")
    return {key: "" if value is None else value for key, value in fields.items()}


def _compute_and_log_all_metrics(
    args: argparse.Namespace,
    model: OTUFormerEncoder,
    device: torch.device,
    epoch: int,
    logs_dir: Path,
    metrics_logger: EnhancedMetricsLogger,
    eval_image_size: int,
    force_linear_probe: bool = False,
) -> None:
    visualize_csv = getattr(args, "visualize_data", "") or getattr(
        args, "train_data", ""
    )
    compute_lp = force_linear_probe or (((epoch + 1) % 10) == 0)

    if visualize_csv:
        try:
            feats, labels = _compute_embeddings_from_csv(
                model=model,
                csv_path=Path(visualize_csv),
                root_dir=Path(args.input_images_dir),
                image_size=eval_image_size,
                device=device,
                batch_size=max(1, args.batch_size),
                num_workers=max(0, args.num_workers),
            )
            feats_eval, labels_eval = _maybe_subsample_for_metrics(
                feats,
                labels,
                max_samples=args.metrics_sample_size,
                seed=args.seed,
            )
            metrics = _compute_all_metrics(
                feats_eval, labels_eval, compute_linear_probe=compute_lp
            )
            metrics_logger.log(epoch + 1, "train", metrics)
            print(f"[Metrics] Epoch {epoch + 1}:")
            if labels_eval is None:
                print("  [Info] Skipping supervised embedding metrics: no labels provided.")
            elif len(np.unique(labels_eval)) < 2:
                print(
                    "  [Info] Skipping supervised embedding metrics: "
                    "fewer than two label classes after sampling."
                )
            for k, v in metrics.items():
                if v != "":
                    print(f"  {k}: {float(v):.4f}")
            if len(feats_eval) >= 10:
                out_path = logs_dir / f"umap.train.epoch_{epoch + 1:04d}.pdf"
                run_umap(
                    feats_eval,
                    labels_eval,
                    out_path,
                    n_components=2,
                    n_neighbors=args.umap_n_neighbors,
                    min_dist=args.umap_min_dist,
                    metric=args.umap_metric,
                    max_classes=args.visualize_class_number,
                    title=f"UMAP Train - Epoch {epoch + 1}",
                )
                print(f"[Info] Saved UMAP plot to {out_path}")
            else:
                print("[Info] Skipping UMAP: fewer than 10 samples after sampling.")
        except Exception as exc:
            print(
                f"[Warning] Failed to compute embedding metrics at epoch {epoch + 1}: {exc}"
            )


_AUGMENTATION_STAGES = ("pretrain", "finetune")


def _validate_stage(stage: str) -> None:
    if stage not in _AUGMENTATION_STAGES:
        raise ValueError(
            f"Unsupported augmentation stage '{stage}'; "
            f"choose from: {', '.join(_AUGMENTATION_STAGES)}"
        )


def _valid_local_crops(value: Any) -> int | None:
    """Return ``value`` as a non-negative non-boolean int, or ``None`` if invalid."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _checkpoint_augmentation_metadata(
    checkpoint: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """Return ``(profile, expanded_config)`` or ``None`` for absent/old metadata.

    A checkpoint is old only when both augmentation keys are absent from its
    ``config`` dict. Any other malformed combination raises ``ValueError``: the
    top-level ``augmentation_profile`` must match the self-contained
    ``augmentation_config['profile']``, and the saved orientation policy must be
    a known policy.
    """
    if checkpoint is None:
        return None
    cfg = checkpoint.get("config")
    if not isinstance(cfg, dict):
        return None
    has_profile = "augmentation_profile" in cfg
    has_config = "augmentation_config" in cfg
    if not has_profile and not has_config:
        return None
    if not has_profile or not has_config:
        raise ValueError(
            "malformed augmentation metadata: checkpoint must contain both "
            "'augmentation_profile' and 'augmentation_config'."
        )
    profile = cfg["augmentation_profile"]
    config = cfg["augmentation_config"]
    if not isinstance(profile, str):
        raise ValueError(
            "malformed augmentation metadata: 'augmentation_profile' must be a string."
        )
    if not isinstance(config, dict):
        raise ValueError(
            "malformed augmentation metadata: 'augmentation_config' must be a dict."
        )
    nested_profile = config.get("profile")
    if not isinstance(nested_profile, str):
        raise ValueError(
            "malformed augmentation metadata: 'augmentation_config.profile' must "
            "be a string."
        )
    if nested_profile != profile:
        raise ValueError(
            "malformed augmentation metadata: 'augmentation_profile' "
            f"({profile!r}) does not match 'augmentation_config.profile' "
            f"({nested_profile!r})."
        )
    if config.get("orientation_policy") not in ORIENTATION_POLICIES:
        raise ValueError(
            "malformed augmentation metadata: "
            "'augmentation_config.orientation_policy' must be one of: "
            f"{', '.join(ORIENTATION_POLICIES)}."
        )
    return profile, config


def _select_augmentation_profile(
    requested_profile: str | None,
    checkpoint: dict[str, Any] | None,
    *,
    stage: str,
) -> str:
    """Resolve the effective augmentation profile for ``stage``.

    Omitted values default per stage, inherit the saved profile on resume, and
    never inherit the pretraining profile when starting a new fine-tuning run
    (the caller passes ``checkpoint=None`` for that case).
    """
    _validate_stage(stage)
    default_profile = "global-barcode" if stage == "pretrain" else "none"
    metadata = _checkpoint_augmentation_metadata(checkpoint)
    if metadata is None:
        if checkpoint is None:
            if requested_profile is None:
                return default_profile
            return requested_profile
        compatible = "legacy" if stage == "pretrain" else "none"
        if requested_profile is None or requested_profile == compatible:
            return compatible
        raise ValueError(
            f"Cannot resume: checkpoint predates augmentation metadata; the only "
            f"compatible profile is '{compatible}'. Start a new run to change "
            "augmentation settings."
        )
    saved_profile, _ = metadata
    allowed_profiles = (
        PRETRAIN_AUGMENTATIONS if stage == "pretrain" else FINETUNE_AUGMENTATIONS
    )
    if saved_profile not in allowed_profiles:
        raise ValueError(
            f"Checkpoint augmentation profile '{saved_profile}' is not valid for "
            f"{stage}; choose from: {', '.join(allowed_profiles)}."
        )
    if requested_profile is None or requested_profile == saved_profile:
        return saved_profile
    raise ValueError(
        f"Cannot resume: augmentation profile '{requested_profile}' differs from "
        f"checkpoint profile '{saved_profile}'. Start a new run to change "
        "augmentation settings."
    )


def _select_orientation_policy(
    requested_policy: str | None,
    checkpoint: dict[str, Any] | None,
    *,
    stage: str,
    resume: bool = False,
) -> str:
    """Resolve the effective orientation policy for ``stage``.

    ``resume=True`` means the checkpoint is being continued and must match the
    requested policy exactly. ``resume=False`` with a checkpoint means a new
    fine-tuning run initialized via ``--checkpoint``: the saved policy is
    inherited when omitted and an explicit policy is honored.
    """
    _validate_stage(stage)
    if requested_policy is not None and requested_policy not in ORIENTATION_POLICIES:
        raise ValueError(
            f"Unknown orientation policy '{requested_policy}'; "
            f"choose from: {', '.join(ORIENTATION_POLICIES)}"
        )
    metadata = _checkpoint_augmentation_metadata(checkpoint)
    if metadata is None:
        if checkpoint is None:
            return requested_policy if requested_policy is not None else "sensitive"
        if stage == "pretrain" or resume:
            if requested_policy in (None, "invariant"):
                return "invariant"
            raise ValueError(
                "Cannot resume: checkpoint predates augmentation metadata and has no "
                "saved orientation policy. Use invariant or start a new run."
            )
        # New fine-tuning run initialized from an old pretraining checkpoint:
        # an explicit policy is honored, otherwise fall back to sensitive.
        return requested_policy if requested_policy is not None else "sensitive"
    _, saved_config = metadata
    saved_policy = saved_config["orientation_policy"]
    if stage == "pretrain" or resume:
        if requested_policy is None or requested_policy == saved_policy:
            return saved_policy
        raise ValueError(
            f"Cannot resume: orientation policy '{requested_policy}' differs from "
            f"checkpoint policy '{saved_policy}'. Start a new run to change it."
        )
    if requested_policy is None:
        return saved_policy
    return requested_policy


def _resolve_pretrain_local_views(
    requested_local_crop_size: int | None,
    requested_local_crops: int | None,
    checkpoint: dict[str, Any] | None,
) -> tuple[int, int]:
    """Resolve pretraining local-view size/count for a new run or resume.

    A new run uses the requested values or the historical defaults ``96``/``6``.
    A resume inherits the saved expanded config (new-style checkpoint) or saved
    ``args`` (old checkpoint), falling back to ``96``/``6`` only for missing or
    invalid old fields. Explicit values must match the resolved saved values.
    """
    if requested_local_crop_size is not None and _positive_int(
        requested_local_crop_size
    ) is None:
        raise ValueError(
            "local_crop_size must be a non-boolean positive integer, got "
            f"{requested_local_crop_size!r}."
        )
    if requested_local_crops is not None and _valid_local_crops(
        requested_local_crops
    ) is None:
        raise ValueError(
            "local_crops must be a non-boolean non-negative integer, got "
            f"{requested_local_crops!r}."
        )
    if checkpoint is None:
        return (
            int(requested_local_crop_size)
            if requested_local_crop_size is not None
            else 96,
            int(requested_local_crops) if requested_local_crops is not None else 6,
        )

    metadata = _checkpoint_augmentation_metadata(checkpoint)
    if metadata is not None:
        _, saved_config = metadata
        local_crop = saved_config.get("local_crop")
        if not isinstance(local_crop, dict):
            raise ValueError(
                "malformed augmentation metadata: 'augmentation_config.local_crop' "
                "must be a dict."
            )
        saved_size = _positive_int(local_crop.get("size"))
        saved_crops = _valid_local_crops(saved_config.get("local_crops"))
        if saved_size is None or saved_crops is None:
            raise ValueError(
                "malformed augmentation metadata: checkpoint records invalid "
                "local-view settings."
            )
    else:
        saved_args = checkpoint.get("args")
        if not isinstance(saved_args, dict):
            saved_args = {}
        saved_size = _positive_int(saved_args.get("local_crop_size")) or 96
        saved_crops = _valid_local_crops(saved_args.get("local_crops"))
        if saved_crops is None:
            saved_crops = 6

    if requested_local_crop_size is not None and (
        int(requested_local_crop_size) != saved_size
    ):
        raise ValueError(
            f"Cannot resume: local crop size {requested_local_crop_size} differs from "
            f"checkpoint value {saved_size}. Start a new run to change local-view "
            "settings."
        )
    if requested_local_crops is not None and int(requested_local_crops) != saved_crops:
        raise ValueError(
            f"Cannot resume: local crops {requested_local_crops} differs from "
            f"checkpoint value {saved_crops}. Start a new run to change local-view "
            "settings."
        )
    return saved_size, saved_crops


def _validate_augmentation_config(
    profile: str,
    current_config: dict[str, object],
    checkpoint: dict[str, Any] | None,
    *,
    stage: str,
) -> dict[str, object]:
    """Validate the freshly built config against checkpoint metadata.

    Returns ``current_config`` unchanged when no comparable metadata exists or
    when it matches exactly; raises on malformed metadata or any difference.
    """
    _validate_stage(stage)
    if not isinstance(current_config, dict):
        raise ValueError(
            "malformed augmentation metadata: current augmentation config must "
            "be a dict."
        )
    metadata = _checkpoint_augmentation_metadata(checkpoint)
    if metadata is None:
        return current_config
    _, saved_config = metadata
    if current_config != saved_config:
        raise ValueError(
            f"Augmentation configuration differs from checkpoint for profile "
            f"'{profile}'. Start a new run or resume with the exact saved "
            "configuration."
        )
    return current_config


_REGISTER_RESUME_KEYS = ("student", "teacher", "model_state_dict")


def _assert_encoder_state_matches(
    state_dict: dict, model: OTUFormerEncoder, label: str
) -> None:
    """Require every ``backbone.*`` key and shape of a resume state to fit.

    Only the backbone is compared, so ``center`` and the ``_pretrain_final_norm.*``
    alias remain allowed inside an encoder state dict.
    """
    expected = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith("backbone.")
    }
    provided = {
        key: value for key, value in state_dict.items() if key.startswith("backbone.")
    }
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(provided)
        if expected[key].shape != provided[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Resume checkpoint '{label}' does not fit '{model.model_name}': "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} shape-mismatched backbone tensor(s) "
            f"(e.g. {(missing + unexpected + mismatched)[0]})."
        )


def _resume_register_plan(
    resume_ckpt: dict[str, Any], model_name: str, image_size: int, out_dim: int
) -> tuple[str, int]:
    """Resolve ``(mode, actual_count)`` for a pretraining resume.

    ``mode`` is ``"legacy"`` (no registers), ``"added"`` (four added registers)
    or ``"native"`` (the model's own registers). A registered resume requires
    explicit ``student`` and ``teacher`` states that agree on the layout, and
    every present encoder state is checked against the rebuilt backbone before
    any training encoder exists. Probes are RNG-isolated and never download
    pretrained weights.
    """
    from otuformer.utils.checkpoint import (
        preserved_rng,
        reject_noncanonical_register_keys,
        resolve_register_layout,
    )

    present = {
        key: resume_ckpt[key]
        for key in _REGISTER_RESUME_KEYS
        if isinstance(resume_ckpt.get(key), dict)
    }
    for state in present.values():
        reject_noncanonical_register_keys(state)

    with preserved_rng():
        probe = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
        )
    native = native_register_count(probe.backbone)
    recorded = recorded_register_count(resume_ckpt)
    registered = (
        (recorded is not None and recorded > 0)
        or native > 0
        or any("backbone.reg_token" in state for state in present.values())
    )
    if not registered:
        # Legacy zero-register resume keeps its historical permissive partial
        # load: the design explicitly excludes this path from the registered
        # complete-backbone requirement.
        return "legacy", 0

    missing = [key for key in ("student", "teacher") if key not in present]
    if missing:
        raise ValueError(
            "Registered resume requires explicit "
            + " and ".join(f"'{key}'" for key in missing)
            + " encoder state(s): the legacy model_state_dict fallback is a "
            "teacher copy, not a student replacement."
        )
    layouts = {
        key: resolve_register_layout(resume_ckpt, state, probe.backbone)
        for key, state in present.items()
    }
    if len(set(layouts.values())) != 1:
        raise ValueError(
            "Registered resume encoder states disagree on the register layout: "
            + json.dumps({key: list(value) for key, value in layouts.items()})
            + "."
        )
    actual, added = layouts["student"]
    mode = "added" if added == 4 else "native"

    with preserved_rng():
        expected = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
            **({"reg_tokens": 4} if mode == "added" else {}),
        )
    for key, state in present.items():
        _assert_encoder_state_matches(state, expected, key)
    return mode, actual


def _check_register_option_against_resume(
    mode: str, actual: int, register_tokens: int | None
) -> None:
    """Reject an explicit option that conflicts with the saved architecture."""
    if register_tokens is None:
        return
    if mode == "native":
        raise ValueError(
            "Cannot resume with --register-tokens: the checkpoint natively has "
            f"{actual} register tokens, which the option cannot add or replace."
        )
    if mode == "added" and register_tokens != 4:
        raise ValueError(
            "Cannot resume a four-register checkpoint with --register-tokens "
            f"{register_tokens}; resume with 4 or omit the option."
        )
    if mode == "legacy" and register_tokens != 0:
        raise ValueError(
            "Cannot add registers while resuming a zero-register checkpoint; "
            "start a new run with --register-tokens 4."
        )


def _finetune_register_plan(
    checkpoint: dict[str, Any], model_name: str, image_size: int, out_dim: int
) -> tuple[str, int]:
    """Resolve ``(mode, actual_count)`` for a fine-tune source.

    Fine-tuning always reads the ``model_state_dict`` source. A legacy
    zero-register checkpoint keeps its existing ``pretrained=True``
    construction; a registered source rebuilds without a download and must
    match every ``backbone.*`` key and shape, including under the head
    replacement's ``strict=False`` load.
    """
    from otuformer.utils.checkpoint import (
        preserved_rng,
        reject_noncanonical_register_keys,
        resolve_register_layout,
    )

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        # The trainer reports the missing-source error in its own words.
        return "legacy", 0
    reject_noncanonical_register_keys(state)
    with preserved_rng():
        probe = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
        )
    native = native_register_count(probe.backbone)
    recorded = recorded_register_count(checkpoint)
    registered = (
        (recorded is not None and recorded > 0)
        or native > 0
        or "backbone.reg_token" in state
    )
    if not registered:
        return "legacy", 0
    actual, added = resolve_register_layout(checkpoint, state, probe.backbone)
    mode = "added" if added == 4 else "native"
    with preserved_rng():
        expected = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
            **({"reg_tokens": 4} if mode == "added" else {}),
        )
    _assert_encoder_state_matches(state, expected, "model_state_dict")
    return mode, actual


def _resolve_finetune_ssl_source(
    checkpoint: dict[str, Any],
) -> tuple[str, str] | None:
    """SSL identity of a fine-tune source.

    An active SSL payload yields its own pair (including a historical relational
    inference); a metric-head descendant yields only its validated, SHA-anchored
    source pair. A hash or path alone never produces a pair.
    """
    from otuformer.utils.checkpoint import (
        resolve_checkpoint_ssl_identity,
        resolve_checkpoint_ssl_source,
    )

    state = checkpoint.get("model_state_dict")
    if isinstance(state, dict) and state:
        identity = resolve_checkpoint_ssl_identity(checkpoint, state)
        if identity is not None:
            return identity
    return resolve_checkpoint_ssl_source(checkpoint)


def preflight_finetune_source(
    checkpoint: dict[str, Any], model_name: str
) -> tuple[str, int]:
    """Validate a fine-tune source before the CLI creates or clears output.

    Active/source SSL identity is validated against the payload's actual
    ``model_state_dict``, so a valid teacher selection, a zero-register fallback
    or a later metric-head replacement cannot hide a conflicting alias.
    """
    from otuformer.utils.checkpoint import (
        preserved_rng,
        resolve_checkpoint_ssl_identity,
        resolve_checkpoint_ssl_source,
        resolve_projector_out_dim,
    )

    cfg = checkpoint.get("config") or {}
    resolved = cfg.get("model_name") or model_name
    image_size = resolve_training_image_size(checkpoint)
    state = checkpoint.get("model_state_dict")
    declared_dim = cfg.get("out_dim") or cfg.get("metric_embed_dim")
    state_dim = resolve_projector_out_dim(state) if isinstance(state, dict) else None
    out_dim = int(declared_dim) if declared_dim else (state_dim or 256)
    if isinstance(state, dict) and state:
        identity = resolve_checkpoint_ssl_identity(checkpoint, state)
        # A recorded source pair is provenance only, but it must be anchored and
        # consistent even on a source that is not itself an SSL payload.
        resolve_checkpoint_ssl_source(checkpoint)
        if identity is not None:
            with preserved_rng():
                probe = OTUFormerEncoder(
                    model_name=resolved,
                    out_dim=out_dim,
                    return_patch_tokens=True,
                    img_size=image_size,
                    pretrained=False,
                )
            _assert_complete_ssl_projector(
                state, probe, out_dim, "model_state_dict"
            )
        else:
            from otuformer.utils.checkpoint import (
                assert_complete_projector,
                resolve_checkpoint_embedding_head,
            )

            head = resolve_checkpoint_embedding_head(checkpoint, state)
            with preserved_rng():
                probe = OTUFormerEncoder(
                    model_name=resolved, out_dim=out_dim, img_size=image_size,
                    pretrained=False,
                )
                if head == ARCFACE_EMBEDDING_HEAD:
                    probe.projector = ArcFaceEmbeddingHead(probe.backbone.num_features, out_dim)
            assert_complete_projector(state, probe.projector)
    return _finetune_register_plan(checkpoint, resolved, image_size, out_dim)


_SSL_DINO_ONLY = ("dino_prototypes", "dino_centering", "koleo_weight")
# The v0.12.0 relational selectors only steer the relational target bundle.
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
            and not math.isclose(
                float(recorded_config_ratio),
                float(recorded_args_ratio),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
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
        if _ssl_explicit(args, "teacher_temp_warmup_ratio") and not math.isclose(
            ratio, RECIPE_TEACHER_TEMP_WARMUP_RATIO, rel_tol=0.0, abs_tol=1e-12
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
        **_resolve_common_recipe(args, saved),
    }


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
    koleo_weight = _validate_koleo_weight(values["koleo_weight"])

    effective: dict[str, Any] = {
        "ssl_method": "dino",
        "ssl_head": SSL_HEAD_BY_METHOD["dino"],
        "out_dim": out_dim,
        "dino_prototypes": prototypes,
        "dino_centering": values["dino_centering"],
        "koleo_weight": koleo_weight,
        **_resolve_common_recipe(args, saved),
    }
    if effective["dino_centering"] == "ema":
        effective["dino_center_momentum"] = 0.9
    else:
        effective["dino_sinkhorn_iterations"] = 3
    effective.update(DINO_FIXED_CONFIG)

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
            "koleo_weight",
            mode_field,
            *DINO_FIXED_CONFIG,
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
        for name, value in {**DINO_FIXED_CONFIG, **mode_fixed}.items():
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
    resume_ckpt: dict[str, Any], model_name: str, effective: dict[str, Any]
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


def resolve_pretrain_ssl_config(
    args: argparse.Namespace, resume_ckpt: dict[str, Any] | None
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
    if requested_student_temp is not None:
        student_temp_value = _admissible_temperature(
            _requested_float(requested_student_temp, "--student-temp"),
            int(getattr(args, "batch_size", 2) or 2),
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
                resume_ckpt, model_name, effective
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
        "koleo_weight",
        "dino_hidden_dim",
        "dino_mlp_init_std",
        "dino_mlp_init_bias",
        "dino_weight_norm_api",
        "dino_prototype_scale",
        "dino_freeze_last_layer_epochs",
        "dino_target_dtype",
        "dino_center_momentum",
        "dino_sinkhorn_iterations",
        "koleo_normalize_eps",
        "koleo_distance_eps",
        "koleo_log_eps",
    )
    active = effective["ssl_method"] == "dino"
    for name in dino_fields:
        display[name] = effective.get(name) if active else None
    return display


def _effective_pretrain_ssl_config(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse a supplied effective slot, else resolve once before output setup."""
    supplied = getattr(args, "ssl_config_effective", None)
    if isinstance(supplied, dict) and supplied:
        return supplied
    resume_ckpt = getattr(args, "resume_state", None)
    if resume_ckpt is None and getattr(args, "resume", ""):
        resume_ckpt = load_checkpoint(Path(args.resume))
        # Direct callers: hand the loaded payload on so it is read only once.
        args.resume_state = resume_ckpt
    effective = resolve_pretrain_ssl_config(args, resume_ckpt)
    args.ssl_config_effective = effective
    return effective


# In-memory handoff attributes; never serialized into a checkpoint's ``args``.
_UNSAVED_ARGS = frozenset(
    {"resume_state", "pretrain_settings", "ssl_config_effective"}
)
# Names that must never reach a saved ``args`` block.
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


# --- relational target recipe ------------------------------------------------

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
# Any of these outside a complete bundle is a partial new-style record.
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
# Project policy band for fresh requests only; no part of numerical safety.
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
    args: argparse.Namespace, checkpoint: dict[str, Any] | None
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
    batch_size = int(getattr(args, "batch_size", 1) or 1)

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


class _ResolvedPretrainRecipe(NamedTuple):
    settings: PretrainSettings
    relational_target_config: dict[str, Any]
    teacher_temp_warmup_ratio: float
    augmentation_config: dict[str, object]
    warnings: tuple[str, ...]

    @property
    def weak_teacher_augmentation_config(self) -> dict[str, object] | None:
        return self.relational_target_config["weak_teacher_augmentation_config"]


def _resolve_pretrain_recipe(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
    *,
    settings: PretrainSettings,
) -> _ResolvedPretrainRecipe:
    """Resolve and validate the whole pretraining recipe without building a model."""
    bundle, ratio, warnings = _resolve_relational_target_config(args, checkpoint)
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
    )


def run_pretrain(
    args: argparse.Namespace,
    *,
    pretrain_recipe: _ResolvedPretrainRecipe | None = None,
) -> None:
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
    resume_ckpt: dict[str, Any] | None = getattr(args, "resume_state", None)
    if getattr(args, "resume", "") and resume_ckpt is None:
        resume_ckpt = load_checkpoint(Path(args.resume))
    # The handoff is a one-shot input, never a cache: a caller that reuses its
    # Namespace after changing a request must be resolved again.
    settings = vars(args).pop("pretrain_settings", None)
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
            args, resume_ckpt, settings=settings
        )
    elif pretrain_recipe.settings is not settings:
        raise ValueError(
            "pretrain_recipe was resolved from a different PretrainSettings "
            "object; resolve the recipe once and pass both."
        )
    _set_seed(args.seed)
    _set_cpus(args.cpus)
    device = _resolve_device(args.device)
    print(f"Resolved device: {device}")

    # Effective SSL config must be resolved before any output mutation; the CLI
    # usually supplies it, and a direct caller resolves it here exactly once.
    ssl_config = _effective_pretrain_ssl_config(args)
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
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
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
    elif register_option is not None:
        # ``0`` must stay distinct from omission: only the explicit option
        # rejects a CLS-free backbone.
        student_kwargs["reg_tokens"] = register_option

    student = OTUFormerEncoder(**student_kwargs).to(device)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    _set_seed(args.seed + 1)
    teacher = OTUFormerEncoder(**student_kwargs).to(device)
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

    objective = PatchObjective(
        patch_mode,
        hidden_dim=student.backbone.num_features,
        ibot_prototypes=patch_config["ibot_prototypes"],
    ).to(device)
    dino_objective: DINOObjective | None = None
    koleo_enabled = False
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
        koleo_enabled = float(ssl_config["koleo_weight"]) > 0.0
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
        if dino_objective is not None:
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
                objective.load_state_dict(saved_patch_state, strict=True)
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
            teacher.center.copy_(resume_ckpt["center"].to(device))
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        global_step = int(resume_ckpt.get("iteration", start_epoch * max(1, len(loader))))
        print(
            f"[Info] Resume from {resume_path} at epoch {start_epoch}, iteration {global_step}"
        )

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
        sample_batch = next(iter(loader))
        sample_views = [v.to(device) for v in sample_batch[:2]]
        s_out, _ = student(sample_views[0])
        t_out, _ = teacher(sample_views[0])
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

    pending_rng_restore = (
        resume_ckpt.get("rng_state") if resume_ckpt is not None else None
    )
    if resume_ckpt is not None and not pending_rng_restore:
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
        for step, views in enumerate(pbar):
            views = [v.to(device) for v in views]
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

            dino_teacher_logits: list[torch.Tensor] = []
            dino_teacher_targets: list[torch.Tensor] = []
            if dino_objective is None:
                loss_global = _compute_global_loss(
                    student_global,
                    teacher_global,
                    temp_student,
                    temp_teacher,
                    relational_pairing=relational_pairing,
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
                    local_student_projs, teacher_global, temp_student, temp_teacher
                )
            else:
                loss_local = _compute_dino_local_loss(
                    [dino_objective(z) for z in local_student_projs],
                    dino_teacher_targets,
                    temp_student,
                )

            koleo_raw = torch.zeros((), device=device)
            koleo_pool = [0, 0]
            if dino_objective is not None and koleo_enabled:
                koleo_raw, koleo_pool = _compute_koleo_raw(student_tokens, ssl_config)

            total_loss = (
                loss_global
                + args.lambda_local * loss_local
                + patch_loss_weighted
            )
            if dino_objective is not None and koleo_enabled:
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
            nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
            optimizer.step()

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
                    "koleo_weight": (
                        float(ssl_config["koleo_weight"])
                        if dino_objective is not None
                        else None
                    ),
                    "global_loss_weighted": float(loss_global.item()),
                    "local_loss_weighted": float(
                        args.lambda_local * loss_local.item()
                    ),
                }
                if dino_objective is None:
                    # Relational has no K-dimensional CLS center and no KoLeo.
                    ssl_metrics["cls_center_norm"] = None
                    ssl_metrics["koleo_status"] = "not-applicable"
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
                    ssl_metrics.update(
                        _koleo_instant_metrics(
                            ssl_config, koleo_enabled, koleo_raw, koleo_pool
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
                    **ssl_config,
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


def _validate_finetune_resume(
    checkpoint: dict[str, Any], current_class_labels: list[str], finetune_epochs: int
) -> None:
    completed_epochs = int(checkpoint.get("epoch", -1)) + 1
    if finetune_epochs <= completed_epochs:
        raise ValueError("--finetune-epochs must exceed the completed checkpoint epoch.")
    saved_labels = checkpoint.get("class_labels")
    if saved_labels is not None and list(saved_labels) != current_class_labels:
        raise ValueError("Cannot resume: training class labels differ from checkpoint.")
    if saved_labels is None:
        print("[Warning] Resume checkpoint lacks class labels; validating class count only.")


_MISSING = object()
_PROJECTOR_FIRST_LINEAR_KEYS = ("projector.net.0.weight", "projector.0.weight")
_ENCODER_STATE_PREFIXES = ("backbone.", "projector.")


def _has_encoder_weights(state: object) -> bool:
    """True when ``state`` carries at least one recognizable encoder parameter."""
    return isinstance(state, dict) and any(
        str(key).startswith(_ENCODER_STATE_PREFIXES) for key in state
    )

_LOSS_DIAGNOSTIC_FIELDS = (
    "epoch",
    "mode",
    "usable_batches",
    "skipped_batches",
    "valid_anchors",
    "margin_satisfied_fraction",
    "compact_hinge_fraction",
    "compact_mean_penalty",
    "local_assignments",
    "argmax_hits",
    "center_direction_cosine",
)


class LossDiagnosticsLogger:
    """Append-only per-epoch v0.8.0 loss diagnostics with a fixed schema."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), None)
            if header != list(_LOSS_DIAGNOSTIC_FIELDS):
                raise ValueError(
                    f"Unrecognized loss-diagnostics schema in {self.path}: "
                    f"{header}. Refusing to append. Move or remove the file to "
                    "start a new log."
                )
            return
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=_LOSS_DIAGNOSTIC_FIELDS).writeheader()

    def log(self, **fields: object) -> None:
        row = {name: fields.get(name, "") for name in _LOSS_DIAGNOSTIC_FIELDS}
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=_LOSS_DIAGNOSTIC_FIELDS).writerow(row)


class BatchIdTraceLogger:
    """Opt-in JSON-lines trace of each training batch's indices and image refs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            # Appending on resume matches LossDiagnosticsLogger: an existing
            # trace is extended, never truncated.
            self.path.write_text("", encoding="utf-8")

    def log(self, *, epoch: int, step: int, indices, refs) -> None:
        record = {
            "epoch": int(epoch),
            "step": int(step),
            "indices": [int(index) for index in indices],
            "refs": [str(ref) for ref in refs],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prototype_centers(loss_fn: nn.Module) -> torch.Tensor | None:
    """Detached L2-normalized ``(C, K, D)`` prototypes, or ``None`` if none.

    Ordinary ArcFace stores one center per class, so its ``(C, D)`` weight is
    returned as ``(C, 1, D)`` to keep every consumer on one shape.
    """
    weight = getattr(getattr(loss_fn, "head", None), "weight", None)
    if weight is None:
        return None
    centers = F.normalize(weight.detach(), dim=-1).clone()
    return centers.unsqueeze(1) if centers.ndim == 2 else centers


def _prototype_k(loss_fn: nn.Module) -> int:
    """Number of centers per class, for sizing the diagnostic counters."""
    centers = _prototype_centers(loss_fn)
    return int(centers.shape[1]) if centers is not None else 1


def _empty_loss_diagnostic_counts(num_classes: int, k: int) -> dict[str, object]:
    return {
        "local": [[0] * k for _ in range(num_classes)],
        "global": [[0] * k for _ in range(num_classes)],
        "margin_hits": 0,
        "samples": 0,
    }


@torch.no_grad()
def _accumulate_loss_diagnostics(
    counts: dict[str, object],
    loss_fn: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Accumulate detached assignment and margin counts for one batch.

    A local winner is the closest center of the sample's own class; a global
    winner is the closest center of any class. The two counts are separate
    states, and the margin fraction compares the margin-adjusted target logit
    against the best rival logit rather than the ArcFace threshold branch.
    """
    centers = _prototype_centers(loss_fn)
    if centers is None:
        return
    num_classes, k, _ = centers.shape
    features = F.normalize(embeddings.detach(), dim=-1)
    cosine = (features @ centers.reshape(-1, centers.shape[-1]).T).reshape(
        features.shape[0], num_classes, k
    )
    rows = torch.arange(features.shape[0], device=features.device)
    global_winners = cosine.reshape(features.shape[0], -1).argmax(dim=1)
    local_winners = cosine[rows, labels].argmax(dim=1)
    for row in range(features.shape[0]):
        winner = int(global_winners[row])
        counts["global"][winner // k][winner % k] += 1
        counts["local"][int(labels[row])][int(local_winners[row])] += 1
    if num_classes >= 2:
        logits = loss_fn.head(embeddings.detach(), labels)
        target = logits.gather(1, labels[:, None]).squeeze(1)
        one_hot = F.one_hot(labels, num_classes).bool()
        rival = logits.masked_fill(one_hot, float("-inf")).max(dim=1).values
        counts["margin_hits"] += int((target > rival).sum())
    counts["samples"] += int(features.shape[0])


@torch.no_grad()
def _compact_hinge_stats(loss_fn: nn.Module) -> tuple[float | None, float | None]:
    """Same-class center-pair hinge activation and mean weighted penalty."""
    centers = _prototype_centers(loss_fn)
    weight = float(getattr(loss_fn, "compact_weight", 0.0))
    cap = getattr(loss_fn, "cap", None)
    if centers is None or weight == 0.0 or cap is None or centers.shape[1] < 2:
        return (None, None)
    penalties: list[float] = []
    for i in range(centers.shape[1]):
        for j in range(i + 1, centers.shape[1]):
            distance = 1.0 - (centers[:, i] * centers[:, j]).sum(dim=-1)
            penalties.extend(F.relu(distance - float(cap)).tolist())
    if not penalties:
        return (None, None)
    fraction = sum(1 for value in penalties if value > 0.0) / len(penalties)
    return (fraction, weight * (sum(penalties) / len(penalties)))


def _center_direction_cosine(
    previous: torch.Tensor | None, current: torch.Tensor | None
) -> list[list[float]]:
    """Per-center cosine between two post-epoch normalized center snapshots."""
    if previous is None or current is None:
        return []
    cosine = (previous * current).sum(dim=-1).clamp(-1.0, 1.0)
    return [[float(value) for value in row] for row in cosine]


def _sha256_file(path: Path) -> str:
    """SHA-256 of a checkpoint's bytes, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_image_ref(ref: object, images_root: Path) -> str:
    """Canonical POSIX reference relative to ``input_images_dir``.

    Shared with pseudo-label discovery so both use one definition; see
    :func:`otuformer.utils.paths.canonical_image_ref`.
    """
    return canonical_image_ref(ref, images_root)


def _train_manifest_sha256(
    image_refs: object, label_names: object, images_root: Path
) -> str:
    """SHA-256 of the canonical ``[image_ref, label]`` training manifest.

    Duplicate rows are preserved and rows are sorted, so the hash covers which
    images and labels were listed rather than their CSV order.
    """
    rows = sorted(
        [_canonical_image_ref(ref, images_root), str(label)]
        for ref, label in zip(image_refs, label_names)
    )
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _path_identity(resolved: Path) -> object:
    """Per-scan file identity; see :func:`otuformer.utils.paths.path_identity`."""
    return path_identity(resolved)


def _expert_path_identity(
    image_refs: object, images_root: Path
) -> tuple[bool, str | None]:
    """Expert-only path-identity check used for pseudo-source eligibility.

    Ordinary manifest validation runs first and keeps its hard failures; this
    check only answers whether the expert rows can seed one pseudo round. A
    duplicate resolved identity (alias) or a reference escaping the image root
    makes the checkpoint ineligible rather than failing ordinary training.

    References are resolved with the same rule :class:`MetricDataset` uses:
    direct path first, then the recursive relative/name index. Otherwise a CSV
    that names bare filenames for a species-subdirectory tree reads fine during
    training yet would be reported missing here.
    """
    from otuformer.training.dataset import (
        _build_recursive_index,
        _resolve_image_path,
        _supports_recursive_lookup,
    )

    root = Path(images_root).resolve()
    refs = [str(ref) for ref in image_refs]
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    if any(
        _supports_recursive_lookup(ref) and not (root / Path(ref)).exists()
        for ref in refs
    ):
        by_relative, by_name = _build_recursive_index(root)
    seen: set[object] = set()
    for text in refs:
        try:
            target = _resolve_image_path(root, text, by_relative, by_name)
        except FileNotFoundError as exc:
            # Several files share the bare name, so no single identity exists.
            return False, f"ambiguous expert reference {text!r}: {exc}"
        try:
            resolved = target.resolve()
        except OSError as exc:
            return False, f"unresolvable expert reference {text!r}: {exc}"
        if resolved != root and root not in resolved.parents:
            return False, f"expert reference escapes the image root: {text!r}"
        try:
            identity = _path_identity(resolved)
        except OSError as exc:
            return False, f"missing expert image {text!r}: {exc}"
        if identity in seen:
            return False, f"duplicate expert path identity: {text!r}"
        seen.add(identity)
    return True, None


def _pseudo_source_status(
    *, loss: str, ssl_initialization_sha256: str | None, expert_reason: str | None
) -> tuple[bool, str | None]:
    """Decide whether a completed finetune#1 checkpoint may seed a pseudo round."""
    if loss not in ARCFACE_FAMILY_LOSSES:
        return False, f"loss {loss!r} is not ArcFace-family"
    if not ssl_initialization_sha256:
        return False, "missing SSL initialization file SHA-256"
    if expert_reason:
        return False, expert_reason
    return True, None


def _resolved_finetune_identity(
    *,
    optimizer_layout: str,
    finetune_lr: float,
    metric_head_lr: float | None,
    weight_decay: float,
    orientation_policy: str,
    batch_size: int,
    finetune_epochs: int,
    long_tail: str,
) -> dict[str, object]:
    """Resolved experiment-identity keys the existing config block lacks.

    The other identity keys (loss settings, dimensions, freeze ratio, seed,
    augmentation, image size) are already written by the caller; this returns
    only the additive ones so the checkpoint keeps a single identity record.
    """
    return {
        "finetune_lr": float(finetune_lr),
        "effective_metric_head_lr": float(
            finetune_lr if metric_head_lr is None else metric_head_lr
        ),
        "weight_decay": float(weight_decay),
        "orientation_policy": orientation_policy,
        "batch_size": int(batch_size),
        "finetune_epochs": int(finetune_epochs),
        "optimizer_name": "adamw",
        "optimizer_groups": optimizer_layout,
        "arcface_scale": 64.0,
        "arcface_margin": 0.5,
        "long_tail": long_tail,
    }


CB_DRW_BETA = 0.99
CB_DRW_MAX_CLASS_WEIGHT = 3.0
CB_DRW_START_FRACTION = 0.5
CB_DRW_RAMP_FRACTION = 0.1


def _effective_number_class_weights(
    expert_counts, *, beta: float = CB_DRW_BETA, cap: float = CB_DRW_MAX_CLASS_WEIGHT
) -> torch.Tensor:
    """Class-Balanced effective-number weights, sample-mean normalized then capped.

    Normalization uses expert training rows, and the cap applies to the target
    class weight before the manifest-wide denominator, so the final effective
    coefficient can exceed ``cap``.
    """
    # Compute in CPU float64: MPS has no float64, and the tiny class-weight
    # vector never needs to live on the accelerator.
    counts = torch.as_tensor(expert_counts).cpu().to(torch.float64)
    raw = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    row_mean = float((raw * counts).sum() / counts.sum())
    return torch.clamp(raw / row_mean, max=cap)


def _drw_lambda(completed_steps: int, total_steps: int) -> float:
    """Deferred-reweighting interpolation: 0 before start, 1 after the ramp."""
    start = int(total_steps * CB_DRW_START_FRACTION)
    ramp = int(total_steps * CB_DRW_RAMP_FRACTION)
    if ramp <= 0:
        return 1.0 if completed_steps >= start else 0.0
    return min(max((completed_steps - start) / ramp, 0.0), 1.0)


def _cb_drw_class_weights(
    target: torch.Tensor, completed_steps: int, total_steps: int
) -> torch.Tensor:
    lam = _drw_lambda(completed_steps, total_steps)
    return 1.0 + lam * (target - 1.0)


def _cb_drw_r_bar(class_weights: torch.Tensor, training_counts) -> float:
    """Manifest-wide mean row weight; never depends on the current batch."""
    counts = torch.as_tensor(
        training_counts, dtype=class_weights.dtype, device=class_weights.device
    )
    return float((class_weights * counts).sum() / counts.sum())


def _weighted_ce(
    per_row: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor, r_bar: float
) -> torch.Tensor:
    """``sum_i(w[label_i] * CE_i) / (actual_batch_size * r_bar)``."""
    return (class_weights[labels] * per_row).sum() / (per_row.numel() * r_bar)


# Every resolved experiment setting a v0.9.0 finetune#1 must carry before it
# can seed a pseudo round. A checkpoint missing any of these cannot be a pseudo
# source (design sections 3.2 and 7): the design forbids guessing defaults for
# them, so an old or hand-crafted "eligible" checkpoint fails loudly instead of
# silently inheriting finetune#2 CLI defaults.
REQUIRED_PSEUDO_SOURCE_KEYS = (
    "model_name",
    "metric_embed_dim",
    "embedding_head",
    "loss",
    "subcenters",
    "compact_weight",
    "compact_cap",
    "supcon_temperature",
    "freeze_ratio",
    "finetune_lr",
    "effective_metric_head_lr",
    "weight_decay",
    "batch_size",
    "seed",
    "finetune_epochs",
    "image_size",
    "augmentation_profile",
    "augmentation_config",
    "orientation_policy",
    "long_tail",
    "optimizer_name",
    "optimizer_groups",
    "arcface_scale",
    "arcface_margin",
    "train_manifest_sha256",
    "ssl_initialization_checkpoint_sha256",
)


def _assert_pseudo_source_identity(
    source_cfg: dict, resolved: dict, *, ignore: frozenset[str] = frozenset()
) -> None:
    """Reject any resolved experiment difference from the finetune#1 source.

    Omitted options are inherited before this check, so a difference here is a
    genuine mismatch (or an explicit conflict), never a default. ``ignore``
    covers keys governed by another check (for example ``--finetune-epochs``,
    which legitimately grows on resume).
    """
    for key, value in resolved.items():
        if key in ignore:
            continue
        if key in source_cfg and source_cfg[key] != value:
            raise ValueError(
                "pseudo mode requires the finetune#2 experiment identity to "
                f"match finetune#1: {key} is {value!r}, source has "
                f"{source_cfg[key]!r}."
            )


def _rows_sha256(rows: list[dict]) -> str:
    """Canonical hash of ``[image, label]`` rows, independent of row order."""
    payload = json.dumps(
        sorted([[str(row["image"]), str(row["label"])] for row in rows]),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expert_rows_from_csv(path: Path) -> list[dict]:
    from otuformer.utils.io import read_csv

    frame = read_csv(path)
    return [
        {"image": str(row["image"]), "label": str(row["label"]), "source": "expert"}
        for _index, row in frame.iterrows()
    ]


def _class_row_counts(
    rows: list[dict] | None, class_to_idx: dict, source: str
) -> dict[str, int]:
    """Per-class training-row counts for one row source (record-keeping only)."""
    counts = {str(label): 0 for label in class_to_idx}
    for row in rows or []:
        if row.get("source") != source:
            continue
        label = str(row.get("label"))
        if label in counts:
            counts[label] += 1
    return counts


def _warn_pseudo_resume_drift(
    cfg: dict[str, Any], args: Any, out_dir: Path
) -> None:
    """Warning-only finetune#2 resume checks (design section 7).

    None of these block: the checkpointed accepted rows are authoritative. A
    moved root, a changed non-accepted candidate pool, or a missing/edited
    ``pseudo_labels.csv`` only warns and logs the current state.
    """
    recorded_root = cfg.get("input_images_dir")
    current_root = str(Path(args.input_images_dir).resolve())
    if recorded_root and str(recorded_root) != current_root:
        print(
            "[Warning] Resuming finetune#2 with a different image root: "
            f"recorded {recorded_root}, current {current_root}."
        )

    from otuformer.embedding.pseudo_label import discover_candidates

    try:
        expert_rows = _expert_rows_from_csv(Path(args.train_data))
        _candidates, _rejections, discovery = discover_candidates(
            expert_rows, Path(args.input_images_dir)
        )
    except Exception as exc:  # noqa: BLE001 - warning only, never blocking
        print(f"[Warning] Cannot rescan the pseudo candidate pool: {exc}")
    else:
        recorded_pool = cfg.get("pseudo_candidate_pool_sha256")
        current_pool = discovery.get("candidate_pool_sha256")
        if recorded_pool and current_pool and recorded_pool != current_pool:
            print(
                "[Warning] Pseudo candidate pool changed since finetune#2 was "
                "checkpointed; checkpointed accepted rows still apply."
            )

    labels_path = Path(out_dir) / "pseudo_labels.csv"
    if not labels_path.is_file():
        print(
            "[Warning] pseudo_labels.csv is missing; checkpointed accepted rows "
            "remain authoritative."
        )
        return
    current_hash = _sha256_file(labels_path)
    recorded_csv = cfg.get("pseudo_labels_sha256")
    if recorded_csv and recorded_csv != current_hash:
        print(
            "[Warning] pseudo_labels.csv changed since checkpointing; current "
            f"sha256 {current_hash[:12]}."
        )
    else:
        print(f"[Info] pseudo_labels.csv sha256 {current_hash[:12]}")


def _write_pseudo_outputs(out_dir: Path, diagnostics: list[dict], summary: dict) -> None:
    import pandas as pd

    from otuformer.utils.io import write_json

    logs_dir = Path(out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(
        Path(out_dir) / "pseudo_labels.csv", index=False
    )
    write_json(summary, Path(out_dir) / "pseudo_summary.json")


def _assert_pseudo_paths(out_dir: Path, image_root: Path, inputs: list[Path]) -> None:
    """Reject output/root overlap and any file-valued input inside ``out_dir``."""
    out = Path(out_dir).resolve()
    root = Path(image_root).resolve()
    if out == root or out in root.parents or root in out.parents:
        raise ValueError(
            "pseudo out_dir must not overlap the biological image root: "
            f"out_dir={out} root={root}"
        )
    for path in inputs:
        resolved = Path(path).resolve()
        if resolved == out or out in resolved.parents:
            raise ValueError(f"input file is inside out_dir: {path}")


def _resolve_pseudo_ssl_path(cfg: dict, checkpoint: Path | None) -> Path:
    """Locate the recorded SSL initialization, verifying its file SHA-256."""
    recorded_sha = cfg.get("ssl_initialization_checkpoint_sha256")
    if not recorded_sha:
        raise ValueError("pseudo source records no SSL initialization file SHA-256")
    if checkpoint is not None:
        if not Path(checkpoint).is_file():
            raise ValueError(f"relocated SSL checkpoint not found: {checkpoint}")
        if _sha256_file(checkpoint) != recorded_sha:
            raise ValueError("relocated SSL checkpoint file SHA-256 does not match")
        return Path(checkpoint)
    recorded = cfg.get("ssl_initialization_checkpoint_path")
    if not recorded:
        raise ValueError(
            "pseudo source records no SSL checkpoint path; pass --checkpoint to locate it"
        )
    path = Path(recorded)
    if not path.is_file():
        raise ValueError(f"recorded SSL checkpoint is missing: {path}; pass --checkpoint")
    if _sha256_file(path) != recorded_sha:
        raise ValueError("recorded SSL checkpoint file SHA-256 does not match")
    return path


def resolve_pseudo_ssl_source(
    ssl_checkpoint: dict[str, Any], source_config: dict[str, Any]
) -> tuple[str, str] | None:
    """Validate the original SSL identity for a NEW pseudo round.

    The original payload's actual ``model_state_dict`` decides its identity.
    Fine-tune#1 may record a pair, but it is provenance only: it must match the
    original and be SHA-anchored. An unknown original identity with no recorded
    pair stays valid legacy behaviour; a recorded pair that cannot be verified
    is rejected instead of being inherited.
    """
    from otuformer.utils.checkpoint import (
        resolve_checkpoint_ssl_identity,
        resolve_checkpoint_ssl_source,
    )

    state = ssl_checkpoint.get("model_state_dict")
    original = None
    if isinstance(state, dict) and state:
        original = resolve_checkpoint_ssl_identity(ssl_checkpoint, state)
    recorded = resolve_checkpoint_ssl_source({"config": source_config})
    if recorded is None:
        return original
    if original is None:
        raise ValueError(
            "the pseudo source records an SSL source pair, but the original SSL "
            "checkpoint's identity is unknown; a SHA-256 alone cannot verify it."
        )
    if recorded != original:
        raise ValueError(
            f"the pseudo source records SSL source identity {recorded}, which "
            f"contradicts the original SSL checkpoint identity {original}."
        )
    return original


def pseudo_cli_identity(
    *,
    ssl_checkpoint: dict,
    source_cfg: dict,
    resolved_loss: dict,
    model_name: str,
    metric_embed_dim: int | None,
    finetune_epochs: int,
    finetune_lr: float,
    metric_head_lr: float | None,
    weight_decay: float,
    freeze_ratio: float,
    augmentation: str | None,
    orientation_policy: str | None,
    batch_size: int,
    seed: int,
    long_tail: str,
    source_class_labels: list[str] | None = None,
) -> dict:
    """Resolve the finetune#2 experiment identity from CLI inputs alone.

    Used by the CLI preflight so an identity conflict is reported before any
    output directory is created or cleared; ``run_finetune`` re-checks the same
    keys against its fully resolved values as a backstop.
    """
    profile = _select_augmentation_profile(augmentation, None, stage="finetune")
    policy = _select_orientation_policy(
        orientation_policy, ssl_checkpoint, stage="finetune", resume=False
    )
    image_size = resolve_training_image_size(ssl_checkpoint)
    ssl_cfg = ssl_checkpoint.get("config") or {}
    out_dim = metric_embed_dim
    if out_dim is None:
        out_dim = ssl_cfg.get("metric_embed_dim") or ssl_cfg.get("out_dim") or 256
    # The patch set of the initialized backbone is part of the experiment
    # identity: the same model name can train on different token sets (zero
    # versus four registers, or a CLS-free backbone).
    _register_mode, register_tokens = _finetune_register_plan(
        ssl_checkpoint, model_name, image_size, int(out_dim)
    )
    return {
        "model_name": model_name,
        "metric_embed_dim": int(out_dim),
        "embedding_head": _select_finetune_embedding_head(ssl_checkpoint),
        "loss": str(resolved_loss["loss"]),
        "subcenters": resolved_loss["subcenters"],
        "compact_weight": resolved_loss["compact_weight"],
        "compact_cap": resolved_loss["compact_cap"],
        "supcon_temperature": resolved_loss["supcon_temperature"],
        "freeze_ratio": float(freeze_ratio),
        "finetune_lr": float(finetune_lr),
        "effective_metric_head_lr": float(
            finetune_lr if metric_head_lr is None else metric_head_lr
        ),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "finetune_epochs": int(finetune_epochs),
        "image_size": int(image_size),
        "augmentation_profile": profile,
        "augmentation_config": build_finetune_augmentation_config(
            profile, image_size, orientation_policy=policy
        ),
        "orientation_policy": policy,
        "long_tail": long_tail,
        "class_labels": [
            str(label)
            for label in (
                source_class_labels
                if source_class_labels is not None
                else source_cfg.get("class_labels", [])
            )
        ],
        "optimizer_name": "adamw",
        "optimizer_groups": _finetune_optimizer_layout(
            ssl_checkpoint, False, str(resolved_loss["loss"])
        ),
        "arcface_scale": 64.0,
        "arcface_margin": 0.5,
        "register_tokens": int(register_tokens),
    }


def _load_pseudo_source(path: Path) -> dict[str, Any]:
    """Load a pseudo-source checkpoint as one lightweight error type."""
    from otuformer.embedding.pseudo_label import PseudoLabelError

    try:
        return load_checkpoint(Path(path))
    except PseudoLabelError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as one lightweight error
        raise PseudoLabelError(
            f"cannot read pseudo source checkpoint: {exc}"
        ) from exc


def prepare_pseudo_round(
    *,
    pseudo_label_from: Path,
    train_data: Path,
    input_images_dir: Path,
    out_dir: Path,
    similarity_floor: float,
    min_gap: float,
    neighbors: int,
    long_tail: str,
    loss: str | None,
    checkpoint: Path | None,
    device: str,
    batch_size: int,
    num_workers: int,
    cap_multiplier: int | None = None,
    absolute_cap: int | None = None,
    visualize_data: Path | None = None,
    progress=None,
) -> dict:
    """Validate the pseudo source, generate one round in memory, and return it.

    Runs entirely before ``prepare_output_dir``: an empty or invalid round
    leaves no output directory or file behind.
    """
    from otuformer.embedding.pseudo_label import (
        ABSOLUTE_CAP,
        CAP_MULTIPLIER,
        PseudoLabelError,
        generate_pseudo_rows,
    )
    from otuformer.utils.io import read_csv

    if cap_multiplier is None:
        cap_multiplier = CAP_MULTIPLIER
    if absolute_cap is None:
        absolute_cap = ABSOLUTE_CAP

    source = _load_pseudo_source(pseudo_label_from)
    cfg = source.get("config") or {}
    if int(cfg.get("pseudo_round", 0)) != 0:
        raise ValueError("--pseudo-label-from cannot be a finetune#2 checkpoint")
    if not cfg.get("pseudo_source_eligible"):
        raise ValueError(
            "pseudo source is ineligible: "
            f"{cfg.get('pseudo_source_ineligible_reason') or 'unknown reason'}"
        )
    missing = [key for key in REQUIRED_PSEUDO_SOURCE_KEYS if key not in cfg]
    if missing:
        raise ValueError(
            "pseudo source checkpoint is missing required provenance/resolved "
            f"settings: {', '.join(sorted(missing))}"
        )
    if str(cfg.get("loss")) not in ARCFACE_FAMILY_LOSSES:
        raise ValueError("pseudo feedback supports only ArcFace-family losses")
    if loss is not None and str(cfg.get("loss")) != loss:
        raise ValueError(
            f"--loss must match the pseudo source ({cfg.get('loss')!r})"
        )
    if cfg.get("long_tail") != long_tail:
        raise ValueError(
            f"--long-tail must match the pseudo source ({cfg.get('long_tail')!r})"
        )

    frame = read_csv(train_data)
    expert_rows = [
        {"image": str(row["image"]), "label": str(row["label"])}
        for _index, row in frame.iterrows()
    ]
    expert_hash = _train_manifest_sha256(
        [row["image"] for row in expert_rows],
        [row["label"] for row in expert_rows],
        Path(input_images_dir),
    )
    if cfg.get("train_manifest_sha256") != expert_hash:
        raise ValueError(
            "--train-data does not match the pseudo source expert manifest"
        )

    # ``class_order`` is the actual classifier row order of the source run;
    # ``class_labels`` is the historical sorted-string metadata and can differ
    # for numeric labels. A missing table is rejected, never guessed, and these
    # checks run before the expensive candidate generation.
    class_labels = [str(label) for label in (source.get("class_order") or [])]
    if not class_labels:
        raise ValueError("pseudo source checkpoint has no recorded class order")
    recorded_labels = [str(label) for label in (source.get("class_labels") or [])]
    if not recorded_labels:
        raise ValueError("pseudo source checkpoint has no recorded class labels")
    if set(recorded_labels) != set(class_labels):
        raise ValueError(
            "pseudo source class order and historical class labels disagree"
        )
    if set(class_labels) != {row["label"] for row in expert_rows}:
        raise ValueError(
            "expert CSV labels do not match the pseudo source class table"
        )
    planned_epochs = int(cfg.get("finetune_epochs") or 0)
    completed_epochs = int(source.get("epoch", -1)) + 1
    if planned_epochs and completed_epochs < planned_epochs:
        raise ValueError(
            "pseudo source run is incomplete: "
            f"epoch {completed_epochs} of {planned_epochs}"
        )

    ssl_path = _resolve_pseudo_ssl_path(cfg, checkpoint)
    protected = [Path(pseudo_label_from), ssl_path, Path(train_data)]
    if visualize_data is not None:
        protected.append(Path(visualize_data))
    if checkpoint is not None:
        protected.append(Path(checkpoint))
    _assert_pseudo_paths(Path(out_dir), Path(input_images_dir), protected)

    accepted, diagnostics, summary = generate_pseudo_rows(
        expert_rows=expert_rows,
        image_root=Path(input_images_dir),
        pseudo_checkpoint=Path(pseudo_label_from),
        similarity_floor=similarity_floor,
        min_gap=min_gap,
        neighbors=neighbors,
        cap_multiplier=cap_multiplier,
        absolute_cap=absolute_cap,
        model_name=str(cfg.get("model_name") or "vit_tiny_patch16_224"),
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        progress=progress,
    )
    if not accepted:
        raise PseudoLabelError(
            "no candidate accepted; candidate count: "
            + str(summary.get("candidate_count"))
            + "; accepted count: "
            + str(summary.get("accepted_count"))
            + "; acceptance rate: "
            + json.dumps(summary.get("acceptance_rate"))
            + "; rejection counts: "
            + json.dumps(summary.get("rejection_reason_counts", {}), sort_keys=True)
            + "; score quantiles: "
            + json.dumps(summary.get("score_quantiles", {}), sort_keys=True)
            + "; counterfactual pass counts: "
            + json.dumps(
                summary.get("counterfactual_pass_counts", {}), sort_keys=True
            )
            + "; seed-count bins: "
            + json.dumps(summary.get("seed_count_bins", {}), sort_keys=True)
        )

    rows = [
        {"image": row["image"], "label": row["label"], "source": "expert"}
        for row in expert_rows
    ] + [
        {"image": row["image"], "label": row["label"], "source": "known-pseudo"}
        for row in accepted
    ]
    return {
        "rows": rows,
        "accepted": accepted,
        "diagnostics": diagnostics,
        "summary": summary,
        "class_labels": class_labels,
        "source_class_labels": class_labels,
        "ssl_checkpoint": str(ssl_path),
        "source_checkpoint": str(pseudo_label_from),
        "source_config": cfg,
    }


def _classify_finetune_source(checkpoint: dict[str, Any]) -> str:
    """Classify a finetune source per the v0.8.0 checkpoint decision table.

    Loss metadata and embedding-head metadata are independent: ``config.loss``
    selects the loss, ``config.embedding_head`` selects the head, and a
    present-but-empty ``loss_state_dict`` is never the same as an absent one.
    An SSL source additionally needs real encoder weights, so a checkpoint with
    neither loss metadata nor a non-empty ``model_state_dict`` is rejected
    instead of being mistaken for SSL initialization.
    Returns ``ssl``, ``v080``, ``legacy_arcface`` or ``head_only``; ambiguous
    or ref-script sources raise, because guessing a head or a loss from them is
    exactly what the v0.8.0 contract forbids.
    """
    if (
        "loss_func" in checkpoint
        or "model" in checkpoint
        or ("args" in checkpoint and "config" not in checkpoint)
    ):
        raise ValueError(
            "Unsupported legacy fine-tune checkpoint format: expected "
            "'model_state_dict' and 'loss_state_dict'. Ref-script checkpoints "
            "('model'/'loss_func', or 'args' without 'config') can be read by "
            "extract/export/cam but not resumed by finetune."
        )
    config = checkpoint.get("config") or {}
    if not _has_encoder_weights(checkpoint.get("model_state_dict")):
        raise ValueError(
            "Cannot use this checkpoint for finetune: it has no valid encoder "
            "weights in 'model_state_dict', so the encoder could only be "
            "re-initialized. Refusing an empty or unrecognized state dict."
        )
    declared_head = config.get("embedding_head")
    state = checkpoint.get("loss_state_dict", _MISSING)
    classifier = isinstance(state, dict) and "head.weight" in state
    if "loss" in config:
        if not declared_head:
            raise ValueError(
                "Malformed v0.8.0 fine-tune checkpoint: config.loss without "
                "config.embedding_head."
            )
        _validate_v080_loss_state(config, state)
        return "v080"
    if classifier:
        return "legacy_arcface"
    if declared_head:
        if state is not _MISSING:
            raise ValueError(
                "Ambiguous fine-tune checkpoint: it declares an embedding head "
                "but has a present loss_state_dict without a classifier, so it "
                "is neither a head-only initialization nor a fine-tune "
                "checkpoint."
            )
        # A declared head and no loss state at all: usable for a new run, but
        # there is nothing for --resume to restore.
        return "head_only"
    if state is _MISSING and "class_labels" not in checkpoint:
        return "ssl"
    raise ValueError(
        "Ambiguous fine-tune checkpoint: it declares neither an embedding head "
        "nor a classifier state, so the loss and head cannot be recovered; "
        "refusing to guess."
    )


def _validate_v080_loss_state(config: dict[str, Any], state: object) -> None:
    """Reject a v0.8.0 loss state that does not match its recorded mode.

    The recorded K and embedding width must agree with the classifier tensor,
    so a corrupt resume is rejected before any output directory is created.
    """
    loss = str(config["loss"])
    if loss not in LOSS_REGISTRY:
        raise ValueError(f"Malformed v0.8.0 checkpoint: unknown loss '{loss}'.")
    if loss == "supcon":
        if state != {}:
            raise ValueError(
                "Malformed v0.8.0 checkpoint: loss 'supcon' has no classifier "
                "and requires an empty loss_state_dict."
            )
        return
    weight = state.get("head.weight") if isinstance(state, dict) else None
    expected_ndim = 2 if loss == "arcface" else 3
    if not isinstance(weight, torch.Tensor) or weight.ndim != expected_ndim:
        raise ValueError(
            f"Malformed v0.8.0 checkpoint: loss '{loss}' requires a "
            f"{expected_ndim}-D loss_state_dict['head.weight']."
        )
    expected_k = config.get("subcenters")
    if expected_k is None and loss == "subcenter-arcface-compact":
        # Compact checkpoints written before K was recorded defaulted to 2.
        expected_k = 2
    if expected_ndim == 3 and expected_k is not None and int(weight.shape[1]) != int(
        expected_k
    ):
        raise ValueError(
            f"Malformed v0.8.0 checkpoint: loss '{loss}' records K={expected_k} "
            f"but head.weight has {int(weight.shape[1])} centers."
        )
    recorded_embed = config.get("metric_embed_dim") or config.get("out_dim")
    if recorded_embed is not None and int(weight.shape[-1]) != int(recorded_embed):
        raise ValueError(
            f"Malformed v0.8.0 checkpoint: recorded embedding dimension "
            f"{recorded_embed} but head.weight is {int(weight.shape[-1])}-wide."
        )


def _select_finetune_embedding_head(checkpoint: dict[str, Any]) -> str:
    kind = _classify_finetune_source(checkpoint)
    declared_head = (checkpoint.get("config") or {}).get("embedding_head")
    if declared_head:
        return declared_head
    if kind == "ssl":
        # SSL initialization installs a fresh fine-tune head.
        return ARCFACE_EMBEDDING_HEAD
    state_dict = checkpoint.get("model_state_dict") or {}
    if not any(key in state_dict for key in _PROJECTOR_FIRST_LINEAR_KEYS):
        raise ValueError(
            "Cannot determine the embedding head: the checkpoint has no "
            "config.embedding_head and no projector weights to infer it from."
        )
    return resolve_checkpoint_embedding_head(checkpoint, state_dict)


def _validate_finetune_embedding_dim(cfg: dict[str, Any], out_dim: int) -> None:
    """Reject a resume that changes the fine-tune embedding width.

    The saved classifier and optimizer state are shaped for the recorded
    dimension, so resizing is only valid for a new ``--checkpoint`` run.
    """
    saved = cfg.get("metric_embed_dim") or cfg.get("out_dim")
    if saved is None or int(saved) == int(out_dim):
        return
    raise ValueError(
        f"Cannot resume: checkpoint embedding dimension is {saved} but "
        f"--metric-embed-dim is {out_dim}; resize only when starting a new "
        "--checkpoint run."
    )


def _validate_finetune_freeze_ratio(checkpoint: dict[str, Any], freeze_ratio: float) -> None:
    """Reject a resume whose ``--freeze-ratio`` differs from the saved run.

    The optimizer only holds ``requires_grad`` backbone parameters, so a
    different freeze ratio changes a group's parameter count and
    ``optimizer.load_state_dict`` would fail with an opaque message.
    """
    saved = (checkpoint.get("config") or {}).get("freeze_ratio")
    if saved is None:
        return
    if not math.isclose(float(saved), float(freeze_ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Cannot resume: checkpoint was trained with freeze_ratio={saved} "
            f"but --freeze-ratio is {freeze_ratio}; pass --freeze-ratio {saved}."
        )


_OPTIMIZER_LAYOUT_GROUPS = {
    "legacy_single": 1,
    "legacy_split": 2,
    "v080_supcon": 2,
    "v080_prototype": 3,
}



def _finetune_optimizer_layout(
    checkpoint: dict[str, Any] | None,
    resume: bool,
    loss: str,
) -> str:
    """Decide which optimizer layout this run must build and load into.

    Group count alone cannot separate a v0.8.0 two-group SupCon checkpoint from
    a legacy two-group ArcFace one. The recorded ``config.optimizer_groups`` is
    the authority when present (a legacy resume keeps its historical layout
    through later resumes); otherwise the recorded ``config.loss`` policy
    decides, and an unmarked unexpected group count is rejected rather than
    guessed.
    """
    if not resume:
        return "v080_supcon" if loss == "supcon" else "v080_prototype"
    config = (checkpoint or {}).get("config") or {}
    groups = len(((checkpoint or {}).get("optimizer") or {}).get("param_groups") or [])
    recorded = config.get("optimizer_groups")
    if recorded is not None:
        if recorded not in _OPTIMIZER_LAYOUT_GROUPS:
            raise ValueError(
                f"Cannot resume: unknown recorded optimizer layout '{recorded}'."
            )
        expected = _OPTIMIZER_LAYOUT_GROUPS[recorded]
        if groups != expected:
            raise ValueError(
                f"Cannot resume: recorded optimizer layout '{recorded}' expects "
                f"{expected} optimizer parameter groups but the checkpoint has "
                f"{groups}."
            )
        return recorded
    saved_loss = config.get("loss")
    if saved_loss is not None:
        layout = "v080_supcon" if saved_loss == "supcon" else "v080_prototype"
        expected = _OPTIMIZER_LAYOUT_GROUPS[layout]
        if groups != expected:
            raise ValueError(
                f"Cannot resume: recorded loss '{saved_loss}' expects {expected} "
                f"optimizer parameter groups but the checkpoint has {groups}."
            )
        return layout
    if groups == 1:
        return "legacy_single"
    if groups == 2:
        return "legacy_split"
    raise ValueError(
        f"Cannot resume: unexpected optimizer parameter-group count {groups}."
    )


def _validate_finetune_resume_source(
    checkpoint: dict[str, Any], loss: str | None = None
) -> None:
    """Reject a resume whose source cannot restore a fine-tune run.

    The CLI preflight passes the resolved loss so the optimizer layout is also
    checked before any output exists; ``run_finetune`` uses it for the source
    check and relies on its own layout decision later.
    """
    kind = _classify_finetune_source(checkpoint)
    if kind in ("ssl", "head_only"):
        raise ValueError(
            "Cannot resume: the checkpoint is an SSL/initialization checkpoint "
            "with no fine-tune loss state; pass it as --checkpoint to start a "
            "new run."
        )
    if loss is not None:
        _finetune_optimizer_layout(checkpoint, True, loss)


def _build_finetune_optimizer_for_layout(
    layout: str,
    model: OTUFormerEncoder,
    loss_fn: nn.Module,
    backbone_lr: float,
    metric_head_lr: float | None,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build the optimizer for a resolved layout, by Parameter identity.

    Legacy layouts reproduce their historical grouping and decay exactly;
    only ``v080_*`` layouts use the zero-decay prototype group.
    """
    head_lr = backbone_lr if metric_head_lr is None else metric_head_lr
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    projector_params = list(model.projector.parameters())
    prototype_params = list(loss_fn.parameters())
    if layout == "v080_prototype":
        return _build_finetune_optimizer(
            model, loss_fn, backbone_lr, metric_head_lr, weight_decay
        )
    if layout == "legacy_single":
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad] + prototype_params,
            lr=backbone_lr,
            weight_decay=weight_decay,
        )
    group_specs = {
        "legacy_split": [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {
                "params": projector_params + prototype_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
        ],
        "v080_supcon": [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": projector_params, "lr": head_lr, "weight_decay": weight_decay},
        ],
    }
    if layout not in group_specs:
        raise ValueError(f"Unsupported optimizer layout: {layout}")
    return torch.optim.AdamW(group_specs[layout])


def _resolve_finetune_loss_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
    *,
    explicit_options: frozenset[str] | set[str] = frozenset(),
) -> dict[str, object]:
    """Resolve the effective loss mode and only its applicable settings.

    ``checkpoint`` is the saved fine-tune checkpoint on ``--resume`` and
    ``None`` for a new run, so an omitted ``--loss`` is ArcFace on a new run
    but inherits the recorded mode on resume. Only options the caller marked
    explicit may conflict with saved settings or be rejected as inapplicable;
    a CLI default value is not an explicit choice. Direct Python callers that
    omit ``explicit_options`` therefore get new-run defaults, not validation
    errors.
    """
    explicit = set(explicit_options)
    saved = (checkpoint or {}).get("config") or {}
    saved_loss = saved.get("loss")
    requested = getattr(args, "loss", None) or "arcface"

    if saved_loss is None:
        if checkpoint is not None and "loss" in explicit and requested != "arcface":
            raise ValueError(
                f"Cannot resume: legacy checkpoint has no recorded loss, so only "
                f"--loss arcface can be resumed, not '{requested}'."
            )
        loss = requested
    else:
        if "loss" in explicit and requested != saved_loss:
            raise ValueError(
                f"Cannot resume: checkpoint loss is '{saved_loss}' but --loss "
                f"is '{requested}'."
            )
        loss = saved_loss
    if loss not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown --loss '{loss}'; choose from {sorted(LOSS_REGISTRY)}."
        )

    # A compact checkpoint written before K was recorded stored
    # ``subcenters: None`` while training with the then-fixed K=2. Treat that as
    # a recorded 2 so an explicit conflicting --subcenters is rejected here
    # instead of later by a head-size mismatch. Copy rather than mutate the
    # checkpoint's own config dict.
    if (
        checkpoint is not None
        and loss == "subcenter-arcface-compact"
        and saved.get("subcenters") is None
    ):
        saved = {**saved, "subcenters": 2}

    def _effective(
        name: str,
        saved_key: str,
        default: object,
        applicable: bool,
        validate: Callable[[object], None],
    ) -> object:
        option = f"--{name.replace('_', '-')}"
        if name in explicit:
            if not applicable:
                raise ValueError(f"{option} does not apply to --loss {loss}.")
            value = getattr(args, name, None)
            validate(value)
            if saved.get(saved_key) is not None and saved[saved_key] != value:
                raise ValueError(
                    f"Cannot resume: checkpoint {name} is {saved[saved_key]} but "
                    f"{option} is {value}."
                )
            return value
        # A recorded ``None`` means the setting did not apply to the saved mode.
        if saved.get(saved_key) is not None:
            validate(saved[saved_key])
            return saved[saved_key]
        return default if applicable else None

    def _validate_subcenters(value: object) -> None:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"--subcenters must be an integer, got {value}.")
        if not MIN_SUBCENTERS <= int(number) <= MAX_SUBCENTERS:
            raise ValueError(
                f"--subcenters must be between {MIN_SUBCENTERS} and "
                f"{MAX_SUBCENTERS}, got {value}."
            )

    def _validate_weight(value: object) -> None:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(
                f"--compact-weight must be a finite value >= 0, got {value}."
            )

    def _validate_temperature(value: object) -> None:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(
                f"--supcon-temperature must be a finite value > 0, got {value}."
            )

    is_compact = loss == "subcenter-arcface-compact"
    # Both Sub-center modes share the K setting; compact no longer fixes K=2.
    subcenters = _effective(
        "subcenters",
        "subcenters",
        2,
        loss == "subcenter-arcface" or is_compact,
        _validate_subcenters,
    )

    return {
        "loss": loss,
        "subcenters": subcenters,
        "compact_weight": _effective(
            "compact_weight", "compact_weight", 0.1, is_compact, _validate_weight
        ),
        "compact_cap": 0.5 if is_compact else None,
        "supcon_temperature": _effective(
            "supcon_temperature",
            "supcon_temperature",
            0.07,
            loss == "supcon",
            _validate_temperature,
        ),
    }


def _build_finetune_loss(
    loss_config: dict[str, object], embed_dim: int, num_classes: int
) -> nn.Module:
    """Instantiate the resolved loss by explicit mode dispatch.

    The four modes do not share a constructor signature, so each is built with
    only its applicable settings.
    """
    mode = loss_config["loss"]
    if mode == "arcface":
        return ArcFaceLoss(embed_dim, num_classes)
    if mode == "supcon":
        return SupConLoss(temperature=loss_config["supcon_temperature"])
    if mode == "subcenter-arcface":
        return SubCenterArcFaceLoss(
            embed_dim,
            num_classes,
            k=loss_config["subcenters"],
            compact_weight=0.0,
        )
    if mode == "subcenter-arcface-compact":
        return SubCenterArcFaceLoss(
            embed_dim,
            num_classes,
            k=loss_config["subcenters"],
            compact_weight=loss_config["compact_weight"],
            cap=loss_config["compact_cap"],
        )
    raise ValueError(f"Unsupported loss mode: {mode}")


def _build_finetune_optimizer(
    model: OTUFormerEncoder,
    loss_fn: nn.Module,
    backbone_lr: float,
    metric_head_lr: float | None,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """v0.8.0 prototype layout: backbone, projector, zero-decay prototype.

    Only normalized class prototypes drop weight decay; every other parameter
    keeps the requested policy.
    """
    head_lr = backbone_lr if metric_head_lr is None else metric_head_lr
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": list(model.projector.parameters()),
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": list(loss_fn.parameters()),
                "lr": head_lr,
                "weight_decay": 0.0,
            },
        ]
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
    model = OTUFormerEncoder(**model_kwargs).to(device)
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
