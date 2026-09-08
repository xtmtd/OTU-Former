"""Shared input-size resolution and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm


def _positive_int(value: Any) -> int | None:
    """Return ``value`` as a positive int, or ``None`` if invalid.

    Rejects booleans, negatives, and non-integers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return int(value)
    return None


def resolve_training_image_size(checkpoint: dict | None) -> int:
    """Resolve the recorded training size from checkpoint metadata.

    Precedence:
      1. fine-tune  ``config.augmentation_config.image_size``
      2. pretrain   ``config.augmentation_config.global_crop.size``
      3. this plan's fine-tune ``config.image_size``
      4. legacy     ``args.global_crop_size``
      5. ``224``

    Levels 1-2 are forward-compatible no-ops until the augmentation plan
    writes them. Invalid (non-positive-int) values are treated as absent and
    resolution falls through to the next level.
    """
    if checkpoint is None:
        return 224
    cfg = checkpoint.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    aug = cfg.get("augmentation_config") or {}
    if isinstance(aug, dict):
        size = _positive_int(aug.get("image_size"))
        if size is not None:
            return size
        global_crop = aug.get("global_crop") or {}
        if isinstance(global_crop, dict):
            size = _positive_int(global_crop.get("size"))
            if size is not None:
                return size
    size = _positive_int(cfg.get("image_size"))
    if size is not None:
        return size
    args = checkpoint.get("args") or {}
    if isinstance(args, dict):
        size = _positive_int(args.get("global_crop_size"))
        if size is not None:
            return size
    return 224


def resolve_backbone_native_size(model_name: str) -> int:
    """Read the backbone's native ``default_cfg["input_size"]``.

    Uses a ``pretrained=False`` timm instantiation so no weights are
    downloaded.
    """
    try:
        model = timm.create_model(model_name, pretrained=False)
    except Exception:
        return 224
    cfg = getattr(model, "default_cfg", None) or {}
    input_size = cfg.get("input_size")
    if isinstance(input_size, (tuple, list)) and len(input_size) >= 3:
        return int(input_size[1])
    return 224


def resolve_backbone_patch_size(model) -> int | None:
    """Read the patch size from ``model.backbone.patch_embed``.

    ``model`` may be an ``OTUFormerEncoder`` or its backbone. Returns ``None``
    for CNN backbones (no patch embedding, no divisibility constraint).
    """
    backbone = getattr(model, "backbone", model)
    patch_embed = getattr(backbone, "patch_embed", None)
    if patch_embed is None:
        return None
    patch_size = getattr(patch_embed, "patch_size", None)
    if isinstance(patch_size, (tuple, list)) and len(patch_size) >= 1:
        return int(patch_size[0])
    if isinstance(patch_size, int):
        return int(patch_size)
    return None


def validate_input_size(size: int, model, model_name: str) -> None:
    """Raise a clear error when ``size`` is not divisible by the patch size.

    CNN backbones (no ``patch_embed``) skip the check.
    """
    patch_size = resolve_backbone_patch_size(model)
    if patch_size is None:
        return
    if size % patch_size != 0:
        nearest = (size // patch_size) * patch_size
        raise ValueError(
            f"input size {size} is not divisible by patch size {patch_size} for "
            f"{model_name}; nearest valid: {nearest}"
        )


def validate_model_name_size(model_name: str, size: int) -> None:
    """Validate ``size`` against the named backbone via a ``pretrained=False``
    probe (no weight download). Lazy-probes only what the model name fixes.
    """
    probe = timm.create_model(model_name, pretrained=False)
    validate_input_size(size, probe, model_name)


def infer_backbone_image_size(model, fallback: int = 224) -> int:
    """Resolve the model's constructed image size (pos-embed resolution)."""
    backbone = getattr(model, "backbone", model)
    patch_embed = getattr(backbone, "patch_embed", None)
    if patch_embed is not None:
        img_size = getattr(patch_embed, "img_size", None)
        if isinstance(img_size, (tuple, list)) and len(img_size) >= 1:
            return int(img_size[0])
        if isinstance(img_size, int):
            return int(img_size)
    default_cfg = getattr(backbone, "default_cfg", None)
    if isinstance(default_cfg, dict):
        input_size = default_cfg.get("input_size")
        if isinstance(input_size, (tuple, list)) and len(input_size) >= 3:
            return int(input_size[1])
    return int(fallback)


def resolve_onnx_input_size(onnx_path: Path) -> int:
    """Validate the ONNX graph input and return its static H (== W).

    Rejects dynamic, non-NCHW, or non-square inputs.
    """
    import onnx

    proto = onnx.load(str(onnx_path))
    graph = proto.graph
    if not graph.input:
        raise ValueError("ONNX graph has no inputs")
    value_info = graph.input[0]
    dims: list[int | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    if len(dims) != 4:
        raise ValueError(f"ONNX input must be 4D NCHW, got shape {dims}")
    _, channels, height, width = dims
    if any(v is None for v in (channels, height, width)):
        raise ValueError("ONNX input must have static (non-dynamic) spatial dims")
    if channels != 3:
        raise ValueError(f"ONNX input must be NCHW with 3 channels, got {channels}")
    if height != width:
        raise ValueError(f"ONNX input must be square (H == W), got {height}x{width}")
    if height <= 0:
        raise ValueError("ONNX input spatial dims must be positive integers")
    return int(height)
