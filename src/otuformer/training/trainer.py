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
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from otuformer.constants import MAX_SUBCENTERS, MIN_SUBCENTERS
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
    center_crop_eval_transform,
)
from otuformer.training.loss import (
    ArcFaceLoss,
    LOSS_REGISTRY,
    SubCenterArcFaceLoss,
    SupConLoss,
    ibot_patch_loss,
    masked_patch_cosine_loss,
)
from otuformer.training.model import (
    PATCH_TARGET_LAYERS,
    ArcFaceEmbeddingHead,
    OTUFormerEncoder,
    PatchObjective,
)
from otuformer.utils.checkpoint import (
    ARCFACE_EMBEDDING_HEAD,
    PROJECTION_EMBEDDING_HEAD,
    load_checkpoint,
    resolve_checkpoint_embedding_head,
    resolve_projector_out_dim,
    save_checkpoint,
)
from otuformer.utils.size import (
    _positive_int,
    resolve_backbone_native_size,
    resolve_training_image_size,
    validate_input_size,
    validate_model_name_size,
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
    student: nn.Module, patch_objective: nn.Module | None
) -> list[nn.Parameter]:
    """Student-side parameters plus the selected objective's trainable state."""
    params = [p for p in student.parameters() if p.requires_grad]
    if patch_objective is not None:
        params.extend(patch_objective.trainable_parameters())
    return params


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


def _build_pretrain_schedules(
    args: argparse.Namespace,
    schedule_state: dict[str, Any] | None,
    *,
    steps_per_epoch: int,
    global_step: int,
    completed_epochs: int,
    original_max_epochs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build schedules without restarting LR after a resumed pretrain run."""
    final_lr = float(args.lr) * 0.01
    if schedule_state is None:
        if completed_epochs > 0 and original_max_epochs != args.max_epochs:
            raise ValueError(
                "Cannot resume a legacy checkpoint without schedule metadata."
            )
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
            total_steps, args.teacher_temp_start, args.teacher_temp_end
        )
    else:
        original_epochs = int(schedule_state["original_max_epochs"])
        original_steps = int(schedule_state["total_steps"])
        original_spe = int(schedule_state["steps_per_epoch"])
        if args.max_epochs <= completed_epochs:
            raise ValueError("--max-epochs must exceed the completed checkpoint epoch.")
        if args.max_epochs < original_epochs:
            raise ValueError("Cannot decrease the original pretrain epoch plan.")
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
                original_steps, args.teacher_temp_start, args.teacher_temp_end
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


def _compute_global_loss(
    student_global: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    temp_student: float,
    temp_teacher: float,
    disable_cross_view_loss: bool = False,
) -> torch.Tensor:
    """Compute global distillation loss following ref/ibot20260115.py.

    When disable_cross_view_loss=False (default): full Cartesian product.
    When disable_cross_view_loss=True: matching-pair zip only.
    """
    device = student_global[0].device
    loss = torch.tensor(0.0, device=device)
    if disable_cross_view_loss:
        for s, t in zip(student_global, teacher_global):
            loss += _ssl_loss(s, t, temp_student, temp_teacher)
        loss /= max(1, len(student_global))
    else:
        for s in student_global:
            for t in teacher_global:
                loss += _ssl_loss(s, t, temp_student, temp_teacher)
        loss /= max(1, len(student_global) * len(teacher_global))
    return loss


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


def _update_teacher_center(
    center: torch.Tensor,
    teacher_global: torch.Tensor,
) -> torch.Tensor:
    """Update teacher center buffer with EMA of current-iteration teacher global outputs.

    Uses fixed 0.9/0.1 EMA coefficients matching ref/ibot20260115.py.
    """
    batch_mean = teacher_global.mean(dim=0, keepdim=True)
    return center * 0.9 + batch_mean * 0.1


def _build_teacher_temp_schedule(
    total_iters: int,
    teacher_temp_start: float,
    teacher_temp_end: float,
) -> np.ndarray:
    warmup_iters = int(total_iters * 0.7)
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


PATCH_LOSS_MODES = ("none", "consistency", "masked-feature", "ibot")
MASKING_STRATEGIES = ("random", "blockwise", "hybrid")
IBOT_PROTOTYPES_MIN = 2
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
_PRETRAIN_CURRENT_FIELDS = _PRETRAIN_LEGACY_FIELDS + _PRETRAIN_PATCH_FIELDS
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
        if self.mode == "pretrain" and header == _PRETRAIN_LEGACY_FIELDS:
            self._migrate_pretrain_schema(header, rows)
            return
        raise ValueError(
            f"Unrecognized instant-metrics schema in {self.path}: {header}. "
            "Refusing to append. Move or remove the file to start a new log."
        )

    def _migrate_pretrain_schema(
        self, header: list[str], rows: list[list[str]]
    ) -> None:
        """Atomically rewrite a recognized legacy header to the v0.7.0 schema."""
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

        df_metrics = None
        if metrics_path is not None and metrics_path.exists():
            df_metrics = pd.read_csv(metrics_path)
            if "epoch" in df_metrics.columns:
                df_metrics = df_metrics[df_metrics["epoch"].notna()]

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        plot_idx = 0

        ax = axes[plot_idx]
        if "loss" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                df_instant["loss"],
                label="Total loss",
                linewidth=1.5,
            )
        if "global_loss" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                df_instant["global_loss"],
                label="Global loss",
                alpha=0.8,
            )
        if "local_loss" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                df_instant["local_loss"],
                label="Local loss",
                alpha=0.8,
            )
        if "patch_loss_raw" in df_instant.columns:
            patch_series = pd.to_numeric(
                df_instant["patch_loss_raw"], errors="coerce"
            )
        else:
            patch_series = None
        if patch_series is not None and patch_series.notna().any():
            ax.plot(
                df_instant["iteration"],
                patch_series,
                label="Patch loss (raw)",
                alpha=0.8,
            )
        elif "mask_loss" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                pd.to_numeric(df_instant["mask_loss"], errors="coerce"),
                label="Mask loss",
                alpha=0.8,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Training Losses")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        ax = axes[plot_idx]
        has_cos = "cosine_similarity" in df_instant.columns
        has_grad = "grad_norm" in df_instant.columns
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
            c in df_instant.columns for c in ["feature_std", "embedding_norm_mean"]
        )
        has_cls = "cls_token_norm_mean" in df_instant.columns
        if has_left:
            if "feature_std" in df_instant.columns:
                ax.plot(
                    df_instant["iteration"],
                    df_instant["feature_std"],
                    label="Feature Std",
                    color="tab:green",
                )
            if "embedding_norm_mean" in df_instant.columns:
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
        if has_cls:
            lines_l, labels_l = ax.get_legend_handles_labels()
            lines_r, labels_r = ax_twin.get_legend_handles_labels()
            ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=8, loc="best")
        else:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        ax = axes[plot_idx]
        fourth_panel_has_data = False
        if "teacher_center_norm" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                df_instant["teacher_center_norm"],
                label="Teacher center",
            )
            fourth_panel_has_data = True
        if "lr" in df_instant.columns:
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
        if "teacher_center_norm" in df_instant.columns:
            ax.set_ylabel("Teacher Center Norm", color="tab:purple")
            ax.tick_params(axis="y", labelcolor="tab:purple")
        if fourth_panel_has_data:
            ax.set_xlabel("Iteration")
            if (
                "lr" in df_instant.columns
                and "teacher_center_norm" in df_instant.columns
            ):
                ax.set_title("Teacher Center & Learning Rate")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r,
                    labels_l + labels_r,
                    fontsize=8,
                    loc="best",
                )
            elif "teacher_center_norm" in df_instant.columns:
                ax.set_title("Teacher Center Norm")
                ax.legend(fontsize=8)
            else:
                ax.set_title("Learning Rate")
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(lines_r, labels_r, fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        else:
            ax.axis("off")

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
) -> dict[str, float]:
    tokens = _extract_tokens(model, sample_view.to(device))
    cls_token_raw = tokens[:, 0]
    proj = model.projector(cls_token_raw)
    return {
        "feature_std": float(cls_token_raw.std(dim=0).mean().item()),
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


def run_pretrain(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    _set_cpus(args.cpus)
    device = _resolve_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the global crop size before dataset/model construction.
    # ``args.global_crop_size`` is None (auto) or an explicit resolved int.
    resume_ckpt: dict[str, Any] | None = None
    start_epoch = 0
    global_step = 0
    if getattr(args, "resume", ""):
        resume_path = Path(args.resume)
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        checkpoint_size = resolve_training_image_size(resume_ckpt)
        if (
            args.global_crop_size is not None
            and args.global_crop_size != checkpoint_size
        ):
            raise ValueError(
                "resume cannot change the input size: checkpoint records "
                f"{checkpoint_size}, got --global-crop-size {args.global_crop_size}. "
                "Start a new run for a different input size."
            )
        global_crop_size = checkpoint_size
    else:
        global_crop_size = (
            args.global_crop_size
            if args.global_crop_size is not None
            else resolve_backbone_native_size(args.model_name)
        )
    args.global_crop_size = global_crop_size
    validate_model_name_size(args.model_name, global_crop_size)

    patch_config = _resolve_patch_config(args, resume_ckpt)
    patch_mode = str(patch_config["patch_loss"])
    patch_strategy = patch_config["masking_strategy"]
    patch_ratio = patch_config["mask_ratio_resolved"]
    print("Patch config: " + json.dumps(patch_config, sort_keys=True))

    requested_profile = getattr(args, "augmentation", None)
    requested_policy = getattr(args, "orientation_policy", None)
    profile = _select_augmentation_profile(
        requested_profile, resume_ckpt, stage="pretrain"
    )
    policy = _select_orientation_policy(
        requested_policy,
        resume_ckpt,
        stage="pretrain",
        resume=resume_ckpt is not None,
    )
    local_crop_size, local_crops = _resolve_pretrain_local_views(
        getattr(args, "local_crop_size", None),
        getattr(args, "local_crops", None),
        resume_ckpt,
    )
    args.local_crop_size = local_crop_size
    args.local_crops = local_crops
    augmentation_config = build_pretrain_augmentation_config(
        profile,
        global_crop_size,
        args.local_crop_size,
        args.local_crops,
        orientation_policy=policy,
    )
    augmentation_config = _validate_augmentation_config(
        profile, augmentation_config, resume_ckpt, stage="pretrain"
    )
    args.augmentation = profile
    args.orientation_policy = policy
    print(f"Augmentation profile: {profile}")
    print(
        "Augmentation config: "
        + json.dumps(augmentation_config, indent=2, sort_keys=True)
    )

    # local crops are processed by the same patch embed: when used, they must
    # satisfy the same divisibility constraint (the default 96 is invalid for
    # patch-14). With --local-crops 0 the local size is unused, so skip it.
    if args.local_crops > 0:
        validate_model_name_size(args.model_name, args.local_crop_size)

    ds = MultiCropDataset(
        csv_path=Path(args.train_data) if getattr(args, "train_data", "") else None,
        images_dir=Path(args.input_images_dir),
        global_crop_size=global_crop_size,
        local_crop_size=args.local_crop_size,
        local_crops=args.local_crops,
        augmentation_profile=profile,
        orientation_policy=policy,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    student = OTUFormerEncoder(
        model_name=args.model_name,
        out_dim=args.out_dim,
        return_patch_tokens=True,
        img_size=global_crop_size,
    ).to(device)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    _set_seed(args.seed + 1)
    teacher = OTUFormerEncoder(
        model_name=args.model_name,
        out_dim=args.out_dim,
        return_patch_tokens=True,
        img_size=global_crop_size,
    ).to(device)
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    for p in teacher.parameters():
        p.requires_grad = False

    objective = PatchObjective(
        patch_mode,
        hidden_dim=student.backbone.num_features,
        ibot_prototypes=patch_config["ibot_prototypes"],
    ).to(device)
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

    optimizer = torch.optim.AdamW(
        _student_optimizer_parameters(student, objective),
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
        if "student" in resume_ckpt:
            student.load_state_dict(resume_ckpt["student"], strict=False)
        elif "model_state_dict" in resume_ckpt:
            student.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        if "teacher" in resume_ckpt:
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
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "center" in resume_ckpt:
            teacher.center.copy_(resume_ckpt["center"].to(device))
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        global_step = int(resume_ckpt.get("iteration", start_epoch * max(1, len(loader))))
        legacy_max_epochs = resume_ckpt.get("args", {}).get("max_epochs")
        if (
            resume_ckpt.get("schedule") is None
            and legacy_max_epochs is not None
            and args.max_epochs != int(legacy_max_epochs)
        ):
            raise ValueError(
                "Cannot extend a legacy checkpoint without schedule metadata."
            )
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
    )
    total_steps = len(lr_schedule)

    cosine_sim = 0.0
    eval_image_size = (
        int(args.extract_size)
        if getattr(args, "extract_size", None) is not None
        else global_crop_size
    )
    validate_input_size(eval_image_size, teacher, args.model_name)
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
        print(f"[Info] Initial cosine similarity: {init_cosine:.4f}")
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
            global_views = views[:2]
            local_views = views[2:]

            iter_idx = global_step
            for group in optimizer.param_groups:
                group["lr"] = float(lr_schedule[iter_idx])

            student_global: list[torch.Tensor] = []
            student_tokens: list[torch.Tensor] = []
            for gv in global_views:
                s_out = student.forward_pretrain(gv)
                student_global.append(s_out.cls)
                student_tokens.append(s_out.tokens)

            teacher_global: list[torch.Tensor] = []
            teacher_tokens: list[torch.Tensor] = []
            teacher_final_four: list[list[torch.Tensor] | None] = []
            teacher_logits: list[torch.Tensor] = []
            prefix = student.backbone.num_prefix_tokens
            with torch.no_grad():
                for gv in global_views:
                    t_out = teacher.forward_pretrain(
                        gv,
                        return_intermediates=patch_mode == "masked-feature",
                    )
                    t_proj = t_out.cls - teacher.center
                    t_proj = F.normalize(t_proj, dim=-1)
                    teacher_global.append(t_proj)
                    teacher_tokens.append(t_out.tokens)
                    teacher_final_four.append(t_out.final_four)
                    if patch_mode == "ibot":
                        teacher_logits.append(
                            objective.teacher_ibot_head(t_out.tokens[:, prefix:])
                        )
                all_teacher = torch.cat(teacher_global, dim=0)
                teacher.center = _update_teacher_center(teacher.center, all_teacher)

            temp_student = float(temp_student_schedule[iter_idx])
            temp_teacher = float(teacher_temp_schedule[iter_idx])

            loss_global = _compute_global_loss(
                student_global,
                teacher_global,
                temp_student,
                temp_teacher,
                disable_cross_view_loss=args.disable_cross_view_loss,
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
            loss_local = _compute_local_loss(
                local_student_projs, teacher_global, temp_student, temp_teacher
            )

            total_loss = (
                loss_global
                + args.lambda_local * loss_local
                + patch_loss_weighted
            )

            optimizer.zero_grad()
            total_loss.backward()
            trainable_params = optimizer.param_groups[0]["params"]
            grad_norm = _compute_grad_norm_parameters(trainable_params)
            nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
            optimizer.step()

            m = float(momentum_schedule[iter_idx])
            update_teacher(student, teacher, m)
            if patch_mode == "ibot":
                update_teacher(
                    objective.student_ibot_head, objective.teacher_ibot_head, m
                )
                objective.patch_center = _update_patch_center(
                    objective.patch_center,
                    torch.cat(center_logits, dim=0),
                )

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
                patch_metrics: dict[str, object] = {
                    "patch_loss_raw": float(patch_loss_raw.item()),
                    "patch_loss_weighted": weighted_value,
                    "patch_loss_fraction_of_total": (
                        weighted_value / total_value if total_value != 0.0 else 0.0
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
                    teacher_center_norm=float(teacher.center.norm().item()),
                    cosine_similarity=float(cosine_sim),
                    **instant_metrics,
                    **patch_metrics,
                )

            if step % 10 == 0 or step == len(loader) - 1:
                cos_display = (
                    f"{cosine_sim:.3e}"
                    if abs(cosine_sim) < 1e-3
                    else f"{cosine_sim:.4f}"
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
                "center": teacher.center.detach().cpu(),
                "patch_objective": objective.state_dict(),
                "rng_state": _capture_rng_state(),
                "args": vars(args),
                "config": {
                    "model_name": args.model_name,
                    "out_dim": args.out_dim,
                    "image_size": global_crop_size,
                    "augmentation_profile": profile,
                    "augmentation_config": augmentation_config,
                    **patch_config,
                },
                "schedule": {
                    **schedule_state,
                    "last_lr": float(optimizer.param_groups[0]["lr"]),
                },
            }
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

    Relative references are normalized lexically and never resolved through the
    filesystem, so a recursively discovered image path does not change the
    manifest hash. An absolute reference must stay inside the image root.
    """
    text = str(ref)
    path = Path(text)
    if path.is_absolute():
        root = images_root.resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"Manifest image reference escapes --input-images-dir: {text}"
            )
        return resolved.relative_to(root).as_posix()
    return Path(os.path.normpath(text)).as_posix()


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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path = (
        Path(getattr(args, "resume", "")) if getattr(args, "resume", "") else None
    )
    trace_batch_ids = bool(getattr(args, "trace_batch_ids", False))
    if resume_path is not None:
        ckpt_path = resume_path
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

    model = OTUFormerEncoder(
        model_name=model_name,
        out_dim=encoder_out_dim,
        img_size=finetune_image_size,
    ).to(device)
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

    ds = MetricDataset(
        csv_path=Path(args.train_data),
        images_dir=Path(args.input_images_dir),
        image_size=finetune_image_size,
        augmentation_profile=profile,
        orientation_policy=policy,
    )
    loader = DataLoader(
        IndexedDataset(ds) if trace_batch_ids else ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    initialization_sha = _sha256_file(ckpt_path)
    ssl_initialization_sha = (
        initialization_sha
        if source_kind == "ssl"
        else cfg.get("ssl_initialization_checkpoint_sha256")
    )
    manifest_sha = _train_manifest_sha256(
        ds.image_refs, ds.label_names, Path(args.input_images_dir)
    )

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

    optimizer = _build_finetune_optimizer_for_layout(
        optimizer_layout,
        model,
        loss_fn,
        args.finetune_lr,
        getattr(args, "metric_head_lr", None),
        getattr(args, "weight_decay", 1e-4),
    )
    start_epoch = 0
    if resume_path is not None:
        class_labels = sorted(str(label) for label in ds.class_to_idx)
        _validate_finetune_resume(ckpt, class_labels, args.finetune_epochs)
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
                    "freeze_ratio": args.freeze_ratio,
                    "image_size": finetune_image_size,
                    "augmentation_profile": profile,
                    "augmentation_config": augmentation_config,
                },
                "class_labels": sorted(str(label) for label in ds.class_to_idx),
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
