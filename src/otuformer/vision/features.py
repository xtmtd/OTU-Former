"""Verified complete-block ViT feature grids.

This module owns the narrow, importable feature-reading contract: module
lookup/listing, the resolved :class:`FeatureContract`, one tensor-returning
forward that yields the requested complete block outputs, and bilinear sampling
of a grid at view-edge coordinates.

It never imports ``vision.cam``, the CLI or the trainer, and it never detaches,
enters ``inference_mode`` or changes the model's mode/RNG state: CAM needs its
own gradient path and callers own the execution context.

Tap semantics: for block ``i`` the recorded grid is the tensor returned by
``backbone.blocks[i](tokens)`` — after that complete module's residual and
internal normalizations, before any next block or the final ``backbone.norm``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..constants import FEATURE_NORM_CHOICES

#: Fields that identify the resolved backbone final norm module.
NORM_MODULE_FIELDS = frozenset({"path", "qualname"})

__all__ = [
    "FeatureContract",
    "LayerSpec",
    "SpatialFeatureForward",
    "describe_feature_layers",
    "get_module_by_name",
    "resolve_feature_contract",
    "resolve_feature_layers",
    "sample_grid_features",
]


@dataclass(frozen=True)
class LayerSpec:
    """One verified complete-block output."""

    path: str
    block_index: int
    grid_hw: tuple[int, int]
    channels: int

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "block_index": self.block_index,
            "grid_hw": [int(self.grid_hw[0]), int(self.grid_hw[1])],
            "channels": int(self.channels),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "LayerSpec":
        _require_keys(payload, {"path", "block_index", "grid_hw", "channels"}, "layer")
        grid = _positive_pair(payload["grid_hw"], "layer grid_hw")
        path = payload["path"]
        if not isinstance(path, str) or not path:
            raise ValueError(f"layer path must be a non-empty string, got {path!r}")
        block_index = _exact_int(payload["block_index"], "layer block_index")
        # The stored pair is what a reader uses to locate the grid; a mismatch
        # between the path and its index would name a layer the product does not
        # hold, so it is checked instead of trusted.
        if path != f"blocks.{block_index}":
            raise ValueError(
                f"layer path {path!r} does not match block_index {block_index}"
            )
        channels = _exact_int(payload["channels"], "layer channels")
        if channels <= 0:
            raise ValueError(f"layer channels must be positive, got {channels}")
        return cls(
            path=path,
            block_index=block_index,
            grid_hw=grid,
            channels=channels,
        )


@dataclass(frozen=True)
class FeatureContract:
    """Everything a stored grid needs to be interpreted without the model.

    ``norm_module`` records the resolved backbone-relative module path and the
    fully qualified Python class when ``norm`` is ``backbone``; it is ``None``
    when norm is ``none``. Round-tripping that identity is what lets a reader
    refuse a graph or artifact whose normalization is not the stored one.
    """

    layers: tuple[LayerSpec, ...]
    input_hw: tuple[int, int]
    patch_hw: tuple[int, int]
    prefix_tokens: int
    norm: str
    norm_module: dict[str, str] | None
    l2: bool
    epsilon: float
    normalization: dict[str, Any]
    model_identity: dict[str, Any]

    @property
    def layer_paths(self) -> tuple[str, ...]:
        return tuple(layer.path for layer in self.layers)

    def to_json(self) -> dict[str, Any]:
        return {
            "layers": [layer.to_json() for layer in self.layers],
            "input_hw": [int(self.input_hw[0]), int(self.input_hw[1])],
            "patch_hw": [int(self.patch_hw[0]), int(self.patch_hw[1])],
            "prefix_tokens": int(self.prefix_tokens),
            "norm": self.norm,
            "norm_module": None if self.norm_module is None else dict(self.norm_module),
            "l2": bool(self.l2),
            "epsilon": float(self.epsilon),
            "normalization": dict(self.normalization),
            "model_identity": dict(self.model_identity),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "FeatureContract":
        _require_keys(
            payload,
            {
                "layers",
                "input_hw",
                "patch_hw",
                "prefix_tokens",
                "norm",
                "norm_module",
                "l2",
                "epsilon",
                "normalization",
                "model_identity",
            },
            "feature contract",
        )
        input_hw = _positive_pair(payload["input_hw"], "feature contract input_hw")
        patch_hw = _positive_pair(payload["patch_hw"], "feature contract patch_hw")
        if input_hw[0] % patch_hw[0] or input_hw[1] % patch_hw[1]:
            raise ValueError(
                f"patch {patch_hw} does not tile the input canvas {input_hw}"
            )
        grid_hw = (input_hw[0] // patch_hw[0], input_hw[1] // patch_hw[1])
        prefix_tokens = _exact_int(
            payload["prefix_tokens"], "feature contract prefix_tokens"
        )
        if prefix_tokens < 0:
            raise ValueError(
                f"feature contract prefix_tokens must be non-negative, got "
                f"{prefix_tokens}"
            )
        norm = payload["norm"]
        if norm not in FEATURE_NORM_CHOICES:
            raise ValueError(
                f"feature contract norm must be one of {FEATURE_NORM_CHOICES}, "
                f"got {norm!r}"
            )
        if not isinstance(payload["l2"], bool):
            # ``bool("false")`` is True, so a string must not decide L2.
            raise ValueError(
                f"feature contract l2 must be a Boolean, got {payload['l2']!r}"
            )
        epsilon = payload["epsilon"]
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError(
                f"feature contract epsilon must be a number, got {epsilon!r}"
            )
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError(
                f"feature contract epsilon must be finite and positive, got {epsilon!r}"
            )
        norm_module = payload["norm_module"]
        if norm_module is not None:
            if not isinstance(norm_module, Mapping):
                raise ValueError("feature contract norm_module must be a mapping")
            if set(norm_module) != NORM_MODULE_FIELDS:
                raise ValueError(
                    "feature contract norm_module must hold exactly "
                    f"{sorted(NORM_MODULE_FIELDS)}, got {sorted(norm_module)}"
                )
            for field, value in norm_module.items():
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"feature contract norm_module {field!r} must be a "
                        f"non-empty string, got {value!r}"
                    )
            norm_module = dict(norm_module)
            if norm == "none":
                # ``none`` means no module is applied, so recording one contradicts
                # the representation it describes.
                raise ValueError(
                    "feature contract norm 'none' must not record a norm_module"
                )
        elif norm == "backbone":
            # ``norm="backbone"`` means "apply this resolved module"; without its
            # identity the stored representation cannot be reproduced.
            raise ValueError(
                "feature contract norm 'backbone' needs the resolved norm_module"
            )
        normalization = payload["normalization"]
        model_identity = payload["model_identity"]
        for name, value in (
            ("normalization", normalization),
            ("model_identity", model_identity),
        ):
            if not isinstance(value, Mapping):
                raise ValueError(f"feature contract {name} must be a mapping")
        layers = tuple(LayerSpec.from_json(item) for item in payload["layers"])
        if not layers:
            raise ValueError("feature contract must declare at least one layer")
        if len({layer.path for layer in layers}) != len(layers):
            raise ValueError("feature contract repeats a layer path")
        for layer in layers:
            if layer.grid_hw != grid_hw:
                raise ValueError(
                    f"layer {layer.path!r} grid {layer.grid_hw} does not match the "
                    f"input/patch lattice {grid_hw}"
                )
        return cls(
            layers=layers,
            input_hw=input_hw,
            patch_hw=patch_hw,
            prefix_tokens=prefix_tokens,
            norm=str(norm),
            norm_module=norm_module,
            l2=bool(payload["l2"]),
            epsilon=float(epsilon),
            normalization=dict(normalization),
            model_identity=dict(model_identity),
        )


def _exact_int(value: Any, what: str) -> int:
    """Accept only a real integer, never a float or a string that coerces."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return int(value)


def _positive_pair(value: Any, what: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{what} must have two entries, got {value!r}")
    first, second = (_exact_int(item, what) for item in value)
    if first <= 0 or second <= 0:
        raise ValueError(f"{what} must be positive, got {value!r}")
    return (first, second)


def _require_keys(payload: Mapping[str, Any], required: set[str], what: str) -> None:
    """Reject unexpected or missing fields instead of guessing a meaning."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{what} must be a mapping")
    missing = required - set(payload)
    unexpected = set(payload) - required
    if missing:
        raise ValueError(f"{what} is missing field(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(
            f"{what} has unexpected field(s): {', '.join(sorted(unexpected))}"
        )


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Resolve a dotted backbone-relative module path.

    Numeric segments index sequences, so ``blocks.5`` addresses a block. The
    lookup is shared: :mod:`otuformer.vision.cam` delegates here.
    """
    module = model
    for attr in name.split("."):
        if attr.isdigit():
            index = int(attr)
            try:
                module = module[index]
            except Exception as exc:  # noqa: BLE001 - resolved as one lookup error
                raise AttributeError(
                    f"module {type(module).__name__!r} has no index {attr!r}"
                ) from exc
            continue
        if isinstance(module, torch.nn.ModuleDict) and attr in module:
            module = module[attr]
            continue
        if not hasattr(module, attr):
            raise AttributeError(
                f"module {type(module).__name__!r} has no attribute {attr!r}"
            )
        module = getattr(module, attr)
    return module


def _backbone(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "backbone", model)


def _blocks(backbone: torch.nn.Module) -> Sequence[torch.nn.Module]:
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or not hasattr(blocks, "__len__") or len(blocks) == 0:
        raise ValueError(
            "spatial feature reading needs a standard ViT backbone with a "
            "non-empty 'blocks' sequence"
        )
    return blocks


def _block_index(backbone: torch.nn.Module, module: torch.nn.Module) -> int | None:
    for index, block in enumerate(_blocks(backbone)):
        if block is module:
            return index
    return None


def describe_feature_layers(backbone: torch.nn.Module) -> list[dict[str, Any]]:
    """List every module path and mark the spatial-eligible complete blocks.

    Mirrors CAM's dump convention (dotted backbone-relative paths) so
    ``--dump-model-structure`` can reuse one report. Only direct children of
    ``backbone.blocks`` are eligible; internal submodules are listed but marked
    ineligible because a spatial grid is a complete block output.
    """
    backbone = _backbone(backbone)
    rows: list[dict[str, Any]] = []
    for name, module in backbone.named_modules():
        index = _block_index(backbone, module) if name.count(".") == 1 else None
        eligible = index is not None and name == f"blocks.{index}"
        rows.append(
            {
                "path": name or "<root>",
                "module_class": f"{type(module).__module__}.{type(module).__qualname__}",
                "parameter_count": sum(
                    parameter.numel() for parameter in module.parameters()
                ),
                "spatial_eligible": bool(eligible),
                "embedding_eligible": bool(eligible and getattr(backbone, "cls_token", None) is not None),
                "block_index": index if eligible else None,
            }
        )
    return rows


def write_model_layers(path: Path, backbone: torch.nn.Module) -> Path:
    """Write checkpoint discovery for embedding and spatial extraction."""
    lines = [
        "# Named modules for --feature-layers",
        "# Format: <name>\\t<class>\\t<spatial-eligible>\\t<embedding-cls-eligible>",
    ]
    for row in describe_feature_layers(backbone):
        lines.append(
            f"{row['path']}\\t{row['module_class'].rsplit('.', 1)[-1]}"
            f"\\t{'yes' if row['spatial_eligible'] else ''}"
            f"\\t{'yes' if row['embedding_eligible'] else ''}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def embedding_block_tokens(
    backbone: torch.nn.Module, images: torch.Tensor, path: str
) -> torch.Tensor:
    """Read one full block through the native forward, retaining its CLS token."""
    captured: list[torch.Tensor] = []
    block = get_module_by_name(backbone, path)
    handle = block.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        backbone.forward_features(images)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError(f"feature layer {path!r} must run exactly once")
    norm, _ = _norm_module(backbone, "backbone")
    return norm(captured[0]) if norm is not None else captured[0]


def resolve_feature_layers(
    backbone: torch.nn.Module, text: str | None
) -> tuple[str, ...]:
    """Resolve the requested complete-block paths, preserving request order.

    Omitted (``None`` or an empty string) means the final complete block. Empty
    entries, duplicates, malformed lists, unknown paths and non-block modules
    fail before any forward.
    """
    backbone = _backbone(backbone)
    _blocks(backbone)
    if text is None or (isinstance(text, str) and not text.strip()):
        last = len(_blocks(backbone)) - 1
        return (f"blocks.{last}",)

    if not isinstance(text, str):
        raise ValueError(f"feature layer list must be text, got {type(text).__name__}")
    entries = [entry.strip() for entry in text.split(",")]
    if any(entry == "" for entry in entries):
        raise ValueError(
            f"feature layer list has an empty entry: {text!r}; "
            "use a comma-separated list such as 'blocks.5,blocks.11'"
        )
    resolved: list[str] = []
    for entry in entries:
        if entry in resolved:
            raise ValueError(f"feature layer {entry!r} is requested twice")
        try:
            module = get_module_by_name(backbone, entry)
        except AttributeError as exc:
            raise ValueError(f"unknown feature layer {entry!r}: {exc}") from exc
        if _block_index(backbone, module) is None:
            raise ValueError(
                f"feature layer {entry!r} is not a complete transformer block; "
                "spatial v1 accepts only 'blocks.<index>' paths"
            )
        resolved.append(entry)
    return tuple(resolved)


def _patch_size(backbone: torch.nn.Module) -> tuple[int, int]:
    patch_embed = getattr(backbone, "patch_embed", None)
    if patch_embed is None:
        raise ValueError("spatial feature reading needs a ViT patch embedding")
    if bool(getattr(patch_embed, "dynamic_img_pad", False)):
        raise ValueError(
            "dynamic patch padding is not supported: the patch lattice would "
            "depend on the input size and hidden padding would become pixels"
        )
    size = getattr(patch_embed, "patch_size", None)
    if size is None:
        raise ValueError("patch embedding does not report patch_size")
    if isinstance(size, int):
        return (int(size), int(size))
    values = tuple(int(v) for v in size)
    if len(values) != 2:
        raise ValueError(f"unsupported patch_size {size!r}")
    return values


def _prefix_count(backbone: torch.nn.Module) -> int:
    count = getattr(backbone, "num_prefix_tokens", None)
    if not isinstance(count, int) or count < 0:
        raise ValueError(
            "backbone does not report an integer num_prefix_tokens; the real "
            "prefix/register layout cannot be verified"
        )
    return int(count)


def _norm_module(
    backbone: torch.nn.Module, norm: str
) -> tuple[torch.nn.Module | None, dict[str, str] | None]:
    if norm not in ("backbone", "none"):
        raise ValueError(f"unknown feature norm {norm!r}; expected 'backbone' or 'none'")
    if norm == "none":
        return None, None
    module = getattr(backbone, "norm", None)
    if module is None or not isinstance(module, torch.nn.Module):
        raise ValueError(
            "feature_norm='backbone' requires the backbone's final norm module, "
            "which this backbone does not declare"
        )
    path = None
    for name, candidate in backbone.named_modules():
        if candidate is module:
            path = name
            break
    if path is None:  # pragma: no cover - a declared module is always named
        raise ValueError("the backbone's final norm is not a named submodule")
    if not _normalizes_last_dim(module):
        raise ValueError(
            f"the backbone's final norm ({type(module).__qualname__}) is not a "
            "per-token normalization; spatial v1 will not treat it as one"
        )
    return module, {
        "path": path,
        "qualname": f"{type(module).__module__}.{type(module).__qualname__}",
    }


def _normalizes_last_dim(module: torch.nn.Module) -> bool:
    """True when the module normalizes channels of a ``(B, N, C)`` tensor.

    LayerNorm (timm's fast variant included) normalizes the trailing dimension.
    ``nn.Identity`` and unknown modules are rejected rather than silently
    treated as an identity normalization.
    """
    if isinstance(module, torch.nn.Identity):
        return False
    normalized_shape = getattr(module, "normalized_shape", None)
    if normalized_shape is not None:
        return True
    # timm's fast LayerNorm exposes ``weight`` but not ``normalized_shape``.
    return hasattr(module, "weight") and hasattr(module, "eps")


def resolve_feature_contract(
    encoder: torch.nn.Module,
    image_size: int,
    layers: Sequence[str],
    *,
    norm: str = "backbone",
    l2: bool = False,
    model_identity: Mapping[str, Any] | None = None,
    normalization: Mapping[str, Any] | None = None,
    epsilon: float = 1e-12,
) -> FeatureContract:
    """Verify the backbone capability and freeze one :class:`FeatureContract`.

    Everything here is a resolved fact: block sequence, positional-embedding
    path, prefix/register count, patch lattice, final norm identity and channel
    width. Unsupported architectures, duplicate/non-block/missing/empty paths
    and hidden patch padding fail before any forward.
    """
    backbone = _backbone(encoder)
    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError(f"image size must be positive, got {image_size}")
    patch_hw = _patch_size(backbone)
    grid_hw = (image_size // patch_hw[0], image_size // patch_hw[1])
    if grid_hw[0] <= 0 or grid_hw[1] <= 0:
        raise ValueError(
            f"image size {image_size} is smaller than the patch size {patch_hw}"
        )
    if image_size % patch_hw[0] or image_size % patch_hw[1]:
        raise ValueError(
            f"image size {image_size} is not a whole multiple of the patch size "
            f"{patch_hw}; the patch lattice would need padding"
        )
    prefix = _prefix_count(backbone)
    _verify_positional_embedding(backbone, grid_hw, patch_hw, prefix)
    norm_module, norm_module_meta = _norm_module(backbone, norm)

    channels = _channels(backbone, patch_hw)
    requested = resolve_feature_layers(
        backbone, ",".join(layers) if layers else None
    )
    specs = tuple(
        LayerSpec(
            path=path,
            block_index=int(_block_index(backbone, get_module_by_name(backbone, path))),
            grid_hw=grid_hw,
            channels=channels,
        )
        for path in requested
    )
    return FeatureContract(
        layers=specs,
        input_hw=(image_size, image_size),
        patch_hw=patch_hw,
        prefix_tokens=prefix,
        norm=norm,
        norm_module=norm_module_meta,
        l2=bool(l2),
        epsilon=float(epsilon),
        normalization=dict(
            normalization
            if normalization is not None
            else {"mean": [], "std": [], "space": "unspecified"}
        ),
        model_identity=dict(model_identity or {}),
    )


def _verify_positional_embedding(
    backbone: torch.nn.Module,
    grid_hw: tuple[int, int],
    patch_hw: tuple[int, int],
    prefix: int,
) -> None:
    """Reject a lattice that would resample or pad positional embeddings."""
    pos_embed = getattr(backbone, "pos_embed", None)
    pos_embed_fn = getattr(backbone, "_pos_embed", None)
    if pos_embed_fn is None or not callable(pos_embed_fn):
        raise ValueError(
            "backbone does not expose the standard '_pos_embed' path; the "
            "prefix/positional layout cannot be verified"
        )
    patch_embed = getattr(backbone, "patch_embed", None)
    declared = getattr(patch_embed, "img_size", None)
    if declared is not None:
        declared_hw = (
            (int(declared), int(declared))
            if isinstance(declared, int)
            else tuple(int(v) for v in declared)
        )
        expected_grid = (declared_hw[0] // patch_hw[0], declared_hw[1] // patch_hw[1])
        if expected_grid != grid_hw:
            raise ValueError(
                f"requested grid {grid_hw} differs from the backbone's declared "
                f"patch grid {expected_grid}; spatial v1 requires the fixed "
                "declared resolution rather than positional-embedding resampling"
            )
    if pos_embed is None or not hasattr(pos_embed, "shape"):
        raise ValueError("backbone does not declare a positional embedding tensor")
    tokens = int(pos_embed.shape[1])
    expected = grid_hw[0] * grid_hw[1] + prefix
    if tokens != expected:
        raise ValueError(
            f"positional embedding holds {tokens} tokens but the resolved "
            f"{grid_hw} grid with {prefix} prefix token(s) needs {expected}; the "
            "model would resample or pad the lattice"
        )


def _channels(backbone: torch.nn.Module, patch_hw: tuple[int, int]) -> int:
    embed_dim = getattr(backbone, "embed_dim", None)
    if isinstance(embed_dim, int) and embed_dim > 0:
        return int(embed_dim)
    patch_embed = getattr(backbone, "patch_embed", None)
    projection = getattr(patch_embed, "proj", None)
    out_channels = getattr(projection, "out_channels", None)
    if isinstance(out_channels, int) and out_channels > 0:
        return int(out_channels)
    raise ValueError("backbone does not report a channel width for its grids")


class SpatialFeatureForward:
    """One forward obtains every requested complete-block grid.

    The callable returns a tuple of BHWC tensors in requested order. It performs
    exactly the timm ViT token path — patch embedding, positional/prefix
    embedding, patch drop, pre-normalization, blocks — and applies the requested
    final norm and channel L2 after removing prefix tokens on the real grid.
    It does not fuse layers, does not detach and never enters inference mode.
    """

    def __init__(self, encoder: torch.nn.Module, contract: FeatureContract) -> None:
        self.encoder = encoder
        self.contract = contract
        backbone = _backbone(encoder)
        self._backbone = backbone
        self._blocks = _blocks(backbone)
        self._indices = tuple(
            int(_block_index(backbone, get_module_by_name(backbone, path)))
            for path in contract.layer_paths
        )
        self._norm_module, _ = _norm_module(backbone, contract.norm)
        for layer in contract.layers:
            if tuple(layer.grid_hw) != tuple(contract.input_hw[i] // contract.patch_hw[i] for i in (0, 1)):
                raise ValueError(
                    f"layer {layer.path!r} grid {layer.grid_hw} does not match the "
                    f"contract input/patch sizes {contract.input_hw}/{contract.patch_hw}"
                )
            if int(layer.channels) != _channels(backbone, contract.patch_hw):
                raise ValueError(
                    f"layer {layer.path!r} channel width {layer.channels} does not "
                    "match the backbone"
                )

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        backbone = self._backbone
        if images.ndim != 4:
            raise ValueError(f"expected (B, 3, H, W) input, got {tuple(images.shape)}")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        if (height, width) != tuple(self.contract.input_hw):
            raise ValueError(
                f"input is {width}x{height} but the feature contract fixes "
                f"{self.contract.input_hw[0]}x{self.contract.input_hw[1]}"
            )
        grid_h, grid_w = (
            height // self.contract.patch_hw[0],
            width // self.contract.patch_hw[1],
        )

        patches = backbone.patch_embed(images)
        if patches.dim() == 4:
            # timm's dynamic-size ViT keeps the NHWC grid for ``_pos_embed``.
            batch, grid_h_patch, grid_w_patch, _ = patches.shape
            if (int(grid_h_patch), int(grid_w_patch)) != (grid_h, grid_w):
                raise ValueError(
                    f"patch embedding produced grid {(grid_h_patch, grid_w_patch)} "
                    f"but the contract expects {(grid_h, grid_w)}"
                )
        tokens = backbone._pos_embed(patches)
        if tokens.dim() == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        tokens = backbone.patch_drop(tokens)
        tokens = backbone.norm_pre(tokens)
        prefix = int(self.contract.prefix_tokens)
        expected_tokens = grid_h * grid_w + prefix
        if int(tokens.shape[1]) != expected_tokens:
            raise ValueError(
                f"token path produced {int(tokens.shape[1])} tokens for a "
                f"{grid_h}x{grid_w} grid and {prefix} prefix token(s)"
            )

        wanted = {index: order for order, index in enumerate(self._indices)}
        captured: list[torch.Tensor | None] = [None] * len(self._indices)
        for index, block in enumerate(self._blocks):
            tokens = block(tokens)
            order = wanted.get(index)
            if order is not None:
                captured[order] = tokens

        outputs: list[torch.Tensor] = []
        for order, tensor in enumerate(captured):
            if tensor is None:  # pragma: no cover - guarded by construction
                raise RuntimeError(f"feature layer {order} was never executed")
            values = self._prepare(tensor)
            # ``-1`` keeps the batch axis symbolic: a traced constant batch would
            # bake a fixed reshape into an exported graph and break batch 2.
            channels = int(values.shape[-1])
            outputs.append(values.reshape(-1, grid_h, grid_w, channels))
        return tuple(outputs)

    def _prepare(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._norm_module is not None:
            tokens = self._norm_module(tokens)
        tokens = tokens[:, int(self.contract.prefix_tokens) :, :]
        if self.contract.l2:
            norms = torch.linalg.vector_norm(tokens, dim=-1, keepdim=True)
            tokens = tokens / norms.clamp_min(float(self.contract.epsilon))
        return tokens


def sample_grid_features(
    grid: torch.Tensor,
    xy: torch.Tensor,
    *,
    patch_hw: tuple[int, int],
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear-sample ``grid`` at view-edge coordinates ``xy``.

    ``grid`` is BHWC, ``xy`` is ``[B,Q,2]`` in view-edge coordinates (x right,
    y down) and ``valid`` is the Boolean ``[B,Hg,Wg]`` support of the grid.
    Returns channels ``[B,Q,C]`` and a complete-support Boolean ``[B,Q]``.

    Mapping: a point ``(x,y)`` becomes zero-based feature index coordinates
    ``u=x/patch_w-0.5`` and ``v=y/patch_h-0.5``; with ``align_corners=False``
    the normalized coordinates are ``gx=2*(u+0.5)/Gw-1`` and ``gy=2*(v+0.5)/Gh-1``.
    Every contributor with nonzero interpolation weight must be inside the grid
    and valid: out-of-bounds or invalid support is unavailable, never zero-filled
    evidence. Zero-weight neighbours outside the grid do not invalidate a sample.
    A non-finite coordinate has no support at all, so it is reported unusable
    instead of inheriting the neighbouring cells' validity.
    """
    if grid.ndim != 4:
        raise ValueError(f"grid must be BHWC, got {tuple(grid.shape)}")
    if xy.ndim != 3 or int(xy.shape[-1]) != 2:
        raise ValueError(f"xy must be [B,Q,2], got {tuple(xy.shape)}")
    if valid.dtype is not torch.bool:
        raise ValueError(f"valid must be Boolean, got {valid.dtype}")
    batch, grid_h, grid_w, channels = grid.shape
    if int(valid.shape[0]) != batch or tuple(valid.shape[-2:]) != (grid_h, grid_w):
        raise ValueError(
            f"valid shape {tuple(valid.shape)} does not match grid {tuple(grid.shape)}"
        )
    if int(xy.shape[0]) != batch:
        raise ValueError("grid and xy batch sizes differ")

    patch_w = int(patch_hw[1])
    patch_h = int(patch_hw[0])
    if patch_w <= 0 or patch_h <= 0:
        raise ValueError(f"patch_hw must be positive, got {patch_hw}")

    finite = torch.isfinite(xy).all(dim=-1)
    points = torch.where(finite.unsqueeze(-1), xy, torch.zeros_like(xy)).to(
        dtype=grid.dtype
    )
    # A finite coordinate can still overflow the grid dtype (1e100 in a float32
    # grid), so representability is part of validity, not only the input value.
    finite = finite & torch.isfinite(points).all(dim=-1)
    u = points[..., 0] / patch_w - 0.5
    v = points[..., 1] / patch_h - 0.5

    # grid_sample needs the explicit BHWC -> BCHW adaption; callers must not
    # pre-permute a second time.
    gx = 2.0 * (u + 0.5) / float(grid_w) - 1.0
    gy = 2.0 * (v + 0.5) / float(grid_h) - 1.0
    sampling = torch.stack([gx, gy], dim=-1).unsqueeze(1)
    sampled = F.grid_sample(
        grid.permute(0, 3, 1, 2),
        sampling,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    values = sampled[:, :, 0, :].permute(0, 2, 1)

    usable = _complete_support(u, v, valid, grid_h=grid_h, grid_w=grid_w)
    return values, usable & finite


def _complete_support(
    u: torch.Tensor,
    v: torch.Tensor,
    valid: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """True when every nonzero-weight contributor is inside and valid."""
    batch = int(u.shape[0])
    col0 = torch.floor(u)
    row0 = torch.floor(v)
    weight_col1 = (u - col0).clamp(0.0, 1.0)
    weight_row1 = (v - row0).clamp(0.0, 1.0)
    col0 = col0.to(torch.int64)
    row0 = row0.to(torch.int64)
    col1 = col0 + 1
    row1 = row0 + 1

    usable = torch.ones_like(u, dtype=torch.bool)
    for rows, row_weights in ((row0, 1.0 - weight_row1), (row1, weight_row1)):
        for cols, col_weights in ((col0, 1.0 - weight_col1), (col1, weight_col1)):
            weight = row_weights * col_weights
            contributing = weight > 0
            inside = (
                (rows >= 0)
                & (rows < grid_h)
                & (cols >= 0)
                & (cols < grid_w)
            )
            clamped_rows = rows.clamp(0, grid_h - 1)
            clamped_cols = cols.clamp(0, grid_w - 1)
            index = clamped_rows * grid_w + clamped_cols
            flat_valid = valid.reshape(batch, grid_h * grid_w)
            neighbour_valid = torch.gather(flat_valid, 1, index)
            usable &= ~contributing | (inside & neighbour_valid)
    return usable
