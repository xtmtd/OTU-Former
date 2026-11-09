"""Patch masking strategies and their resolved configuration."""

from __future__ import annotations

import argparse
from typing import Any, Callable, NamedTuple
import torch
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
from otuformer.training.model import (
    PATCH_TARGET_LAYERS,
    ArcFaceEmbeddingHead,
    OTUFormerEncoder,
    PatchObjective,
    native_register_count,
)

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
