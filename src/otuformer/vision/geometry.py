"""Realized image geometry, traces and annotation mapping.

The module executes the deterministic Pillow/torchvision operations this project
already uses, records the parameters that were *actually* applied, and derives
one checked affine map plus conservative support records from them.

Conventions (design §5.1): continuous image-edge coordinates, ``[0,W] x [0,H]``,
pixel ``(column i, row j)`` centered at ``(i+0.5, j+0.5)``; arrays stay row-major
``(y, x)``; coordinate pairs and matrices are ``(x, y)`` with **x right and y
down**. Matrices map column vectors, ``p_view = M p_source``; resamplers use the
inverse. Operations compose by left multiplication in execution order.

``p_B = M_B inverse(M_A) p_A`` maps between two views of the same source. A
matrix is invertible only where content survived: crop/rotation clipping at an
intermediate canvas cannot be undone, which is why every intermediate keeps its
own realized canvas and clipping history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from . import geometry_support as support

from ..constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SPATIAL_COORDINATE_ADAPTER,
    SPATIAL_SUPPORT_VERSION,
)

GEOMETRY_OP_KINDS = ("crop", "resize", "pad", "rotate", "affine", "hflip", "vflip")

#: Per-channel fill of the whole-specimen padding protocol when the image cannot
#: be read (matches ``training.dataset``'s documented fallback).
BACKGROUND_FILL_FALLBACK = (124, 116, 104)


@dataclass(frozen=True)
class GeometryOp:
    """One realized image operation and the parameters needed to replay it."""

    kind: str
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in GEOMETRY_OP_KINDS:
            raise ValueError(
                f"unknown geometry operation {self.kind!r}; expected one of "
                f"{GEOMETRY_OP_KINDS}"
            )
        if not isinstance(self.params, dict):
            raise ValueError("geometry operation params must be a dict")

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "GeometryOp":
        if set(payload) != {"kind", "params"}:
            raise ValueError(
                "geometry operation must carry exactly 'kind' and 'params'"
            )
        return cls(kind=str(payload["kind"]), params=dict(payload["params"]))


@dataclass(frozen=True)
class GeometryTrace:
    """Source/view sizes, ordered operations and the checked affine maps."""

    source_wh: tuple[int, int]
    view_wh: tuple[int, int]
    operations: tuple[GeometryOp, ...]
    source_to_view: np.ndarray
    view_to_source: np.ndarray

    def __post_init__(self) -> None:
        source_to_view = np.asarray(self.source_to_view, dtype=np.float64)
        view_to_source = np.asarray(self.view_to_source, dtype=np.float64)
        if source_to_view.shape != (3, 3) or view_to_source.shape != (3, 3):
            raise ValueError("geometry maps must be 3x3")
        if not support.affine_is_invertible(source_to_view):
            raise ValueError("source-to-view map is singular or non-finite")
        if not support.affine_is_invertible(view_to_source):
            raise ValueError("view-to-source map is singular or non-finite")
        product = view_to_source @ source_to_view
        if not np.allclose(product, np.eye(3), atol=1e-9):
            raise ValueError("view-to-source is not the inverse of source-to-view")
        for width, height in (self.source_wh, self.view_wh):
            if int(width) <= 0 or int(height) <= 0:
                raise ValueError("geometry sizes must be positive width/height pairs")

    def to_json(self) -> dict[str, Any]:
        return {
            "source_wh": [int(self.source_wh[0]), int(self.source_wh[1])],
            "view_wh": [int(self.view_wh[0]), int(self.view_wh[1])],
            "operations": [op.to_json() for op in self.operations],
            "source_to_view": self.source_to_view.tolist(),
            "view_to_source": self.view_to_source.tolist(),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "GeometryTrace":
        required = {
            "source_wh",
            "view_wh",
            "operations",
            "source_to_view",
            "view_to_source",
        }
        missing = required - set(payload)
        unexpected = set(payload) - required
        if missing or unexpected:
            raise ValueError(
                "geometry trace fields do not match the v1 record "
                f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})"
            )
        source_wh = tuple(int(v) for v in payload["source_wh"])
        view_wh = tuple(int(v) for v in payload["view_wh"])
        if len(source_wh) != 2 or len(view_wh) != 2:
            raise ValueError("geometry sizes must have two entries")
        return cls(
            source_wh=(source_wh[0], source_wh[1]),
            view_wh=(view_wh[0], view_wh[1]),
            operations=tuple(
                GeometryOp.from_json(item) for item in payload["operations"]
            ),
            source_to_view=np.asarray(payload["source_to_view"], dtype=np.float64),
            view_to_source=np.asarray(payload["view_to_source"], dtype=np.float64),
        )

    @property
    def support_version(self) -> str:
        return SPATIAL_SUPPORT_VERSION

    def canvas_before(self, index: int) -> tuple[int, int]:
        """Canvas size entering operation ``index``."""
        if index == 0:
            return self.source_wh
        previous = self.operations[index - 1]
        params = previous.params
        for key in ("canvas_wh", "to_wh"):
            if key in params:
                return tuple(int(v) for v in params[key])  # type: ignore[return-value]
        if previous.kind == "crop":
            return (int(params["width"]), int(params["height"]))
        raise ValueError(  # pragma: no cover - every op records its canvas
            f"operation {index - 1} does not record its realized canvas"
        )


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


def _rotation(angle_degrees: float) -> np.ndarray:
    """Canonical centered rotation: positive is visually counter-clockwise."""
    radians = math.radians(float(angle_degrees))
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array(
        [[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _affine_rotation(angle_degrees: float) -> np.ndarray:
    """``torchvision.transforms.functional.affine``'s opposite angle convention."""
    radians = math.radians(float(angle_degrees))
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _about(center: Sequence[float], linear: np.ndarray) -> np.ndarray:
    cx, cy = float(center[0]), float(center[1])
    return _translation(cx, cy) @ linear @ _translation(-cx, -cy)


def matrix_for(op: GeometryOp) -> np.ndarray:
    """Forward content map of one realized operation."""
    params = op.params
    kind = op.kind
    if kind == "crop":
        return _translation(-float(params["left"]), -float(params["top"]))
    if kind == "resize":
        from_w, from_h = (int(v) for v in params["from_wh"])
        to_w, to_h = (int(v) for v in params["to_wh"])
        return _scale(to_w / from_w, to_h / from_h)
    if kind == "pad":
        offset = params["offset"]
        return _translation(float(offset[0]), float(offset[1]))
    if kind == "rotate":
        canvas = [int(v) for v in params["canvas_wh"]]
        source = [int(v) for v in params["from_wh"]]
        linear = _about(params["center"], _rotation(float(params["angle"])))
        offset = (
            (canvas[0] - source[0]) / 2.0,
            (canvas[1] - source[1]) / 2.0,
        )
        return _translation(*offset) @ linear
    if kind == "affine":
        shear = [float(v) for v in params.get("shear", (0.0, 0.0))]
        if any(abs(value) > 1e-12 for value in shear):
            raise ValueError(
                "shear is not part of spatial v1: no delivered transform applies "
                "it, and copying a library sign convention without a fixture "
                "would be a guess"
            )
        linear = _scale(float(params["scale"]), float(params["scale"]))
        matrix = _about(
            params["center"], _affine_rotation(float(params["angle"])) @ linear
        )
        translate = params["translate"]
        return _translation(float(translate[0]), float(translate[1])) @ matrix
    if kind == "hflip":
        width = float(params["from_wh"][0])
        return np.array([[-1.0, 0.0, width], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    if kind == "vflip":
        height = float(params["from_wh"][1])
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height], [0.0, 0.0, 1.0]])
    raise ValueError(f"unhandled geometry operation {kind!r}")  # pragma: no cover


def _interpolation(name: str) -> InterpolationMode:
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"unsupported interpolation {name!r}")
    return mapping[key]


def _fill(params: Mapping[str, Any]) -> tuple[int, int, int] | None:
    value = params.get("fill")
    if value is None:
        return None
    return tuple(int(v) for v in value)  # type: ignore[return-value]


def _execute(image: Image.Image, op: GeometryOp) -> Image.Image:
    params = op.params
    kind = op.kind
    if kind == "crop":
        return TF.crop(
            image,
            int(params["top"]),
            int(params["left"]),
            int(params["height"]),
            int(params["width"]),
        )
    if kind == "resize":
        to_w, to_h = (int(v) for v in params["to_wh"])
        return TF.resize(
            image,
            [to_h, to_w],
            interpolation=_interpolation(params["interpolation"]),
            antialias=bool(params.get("antialias", True)),
        )
    if kind == "pad":
        canvas = [int(v) for v in params["canvas_wh"]]
        offset = [int(v) for v in params["offset"]]
        fill = _fill(params) or (0, 0, 0)
        output = Image.new(image.mode, (canvas[0], canvas[1]), color=fill)
        output.paste(image, (offset[0], offset[1]))
        return output
    if kind == "rotate":
        return TF.rotate(
            image,
            float(params["angle"]),
            interpolation=_interpolation(params["interpolation"]),
            expand=bool(params["expand"]),
            center=[float(v) for v in params["center"]],
            fill=_fill(params),
        )
    if kind == "affine":
        return TF.affine(
            image,
            float(params["angle"]),
            [float(v) for v in params["translate"]],
            float(params["scale"]),
            [float(v) for v in params["shear"]],
            interpolation=_interpolation(params["interpolation"]),
            fill=_fill(params),
        )
    if kind == "hflip":
        return TF.hflip(image)
    if kind == "vflip":
        return TF.vflip(image)
    raise ValueError(f"unhandled geometry operation {kind!r}")  # pragma: no cover


def _realized_canvas(op: GeometryOp, image: Image.Image) -> tuple[int, int]:
    """The canvas an operation must report, checked against the executed image."""
    params = op.params
    expected = params.get("canvas_wh") or params.get("to_wh")
    if expected is not None:
        expected_hw = (int(expected[0]), int(expected[1]))
        if (int(image.size[0]), int(image.size[1])) != expected_hw:
            raise ValueError(
                f"{op.kind} op records canvas {expected_hw} but the operation "
                f"produced {image.size}"
            )
    return (int(image.size[0]), int(image.size[1]))


def trace_operations(
    source_wh: tuple[int, int], operations: Sequence[GeometryOp]
) -> GeometryTrace:
    """Compose recorded operations into one checked trace."""
    matrix = np.eye(3, dtype=np.float64)
    canvas = (int(source_wh[0]), int(source_wh[1]))
    for op in operations:
        op_matrix = matrix_for(op)
        matrix = op_matrix @ matrix
        canvas = _trace_canvas(op, canvas)
    return GeometryTrace(
        source_wh=(int(source_wh[0]), int(source_wh[1])),
        view_wh=canvas,
        operations=tuple(operations),
        source_to_view=matrix,
        view_to_source=support.invert(matrix),
    )


def _trace_canvas(op: GeometryOp, canvas: tuple[int, int]) -> tuple[int, int]:
    recorded = op.params.get("canvas_wh") or op.params.get("to_wh")
    if recorded is None and op.kind == "crop":
        recorded = [int(op.params["width"]), int(op.params["height"])]
    if recorded is not None:
        target = (int(recorded[0]), int(recorded[1]))
        declared_from = op.params.get("from_wh")
        if declared_from is not None and (int(declared_from[0]), int(declared_from[1])) != canvas:
            raise ValueError(
                f"{op.kind} op records previous canvas {tuple(declared_from)} but "
                f"the trace is at {canvas}"
            )
        return target
    if op.kind == "hflip" or op.kind == "vflip":
        declared = op.params.get("from_wh")
        if declared is not None and tuple(int(v) for v in declared) != canvas:
            raise ValueError(
                f"{op.kind} op records previous canvas {tuple(declared)} but the "
                f"trace is at {canvas}"
            )
        return canvas
    raise ValueError(  # pragma: no cover - every op records its canvas
        f"{op.kind} op does not record its realized canvas"
    )


def prepare_operation(op: GeometryOp, before_wh: tuple[int, int]) -> GeometryOp:
    """Fill the facts known before execution: entering canvas and rotation centre.

    A centred rotation's pivot is the canvas it is applied to, so the operation
    records that resolution instead of every caller repeating ``width / 2``.
    """
    params = dict(op.params)
    params.setdefault("from_wh", [int(before_wh[0]), int(before_wh[1])])
    if op.kind in ("rotate", "affine") and "center" not in params:
        params["center"] = [int(before_wh[0]) / 2.0, int(before_wh[1]) / 2.0]
    return GeometryOp(op.kind, params)


def complete_operation(
    op: GeometryOp, before_wh: tuple[int, int], after_wh: tuple[int, int]
) -> GeometryOp:
    """Fill in the facts of one executed operation.

    ``from_wh`` is always the canvas entering the operation. ``canvas_wh`` is
    derived from the executed image for the operations whose output size the
    library decides (an expanding rotation), so a trace records what actually
    happened instead of a re-derivation of it.
    """
    params = dict(op.params)
    params.setdefault("from_wh", [int(before_wh[0]), int(before_wh[1])])
    if op.kind in ("rotate", "affine", "pad", "resize", "hflip", "vflip"):
        params.setdefault("canvas_wh", [int(after_wh[0]), int(after_wh[1])])
    return GeometryOp(op.kind, params)


def apply_geometry(
    image: Image.Image,
    operations: Sequence[GeometryOp],
    *,
    track: bool = False,
    interleave: Mapping[int, Any] | None = None,
) -> tuple[Image.Image, GeometryTrace | None]:
    """Execute one realized operation sequence, optionally tracing it.

    ``interleave`` maps an operation index to a callable applied to the image
    immediately *before* that operation. A pixel step that must stay between two
    geometry operations (a colour jitter before the final resize) belongs here
    instead of being reordered: the execution order is the preserved one, and no
    check that only exists in the caller is needed.
    """
    current = image
    matrix = np.eye(3, dtype=np.float64)
    canvas = (int(image.size[0]), int(image.size[1]))
    completed: list[GeometryOp] = []
    for index, op in enumerate(operations):
        hook = interleave.get(index) if interleave else None
        if hook is not None:
            current = hook(current)
        before = canvas
        op = prepare_operation(op, before)
        current = _execute(current, op)
        realized = _realized_canvas(op, current)
        op = complete_operation(op, before, realized)
        completed.append(op)
        canvas = realized
        matrix = matrix_for(op) @ matrix
    if not track:
        return current, None
    return current, GeometryTrace(
        source_wh=(int(image.size[0]), int(image.size[1])),
        view_wh=canvas,
        operations=tuple(completed),
        source_to_view=matrix,
        view_to_source=support.invert(matrix),
    )


def estimate_background_color(image: Image.Image) -> tuple[int, int, int]:
    """Per-channel median of the four outer edges (documented fallback on error)."""
    try:
        array = np.asarray(image.convert("RGB"))
        if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] < 2 or array.shape[1] < 2:
            return BACKGROUND_FILL_FALLBACK
        edges = np.concatenate(
            [array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]], axis=0
        )
        return tuple(int(v) for v in np.median(edges, axis=0))  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 - documented fallback
        return BACKGROUND_FILL_FALLBACK


def resize_output_size(size: int, width: int, height: int) -> tuple[int, int]:
    """Output size of ``transforms.Resize(size)`` on a ``width x height`` image.

    Mirrors the installed torchvision rule: the shorter edge becomes ``size`` and
    the longer edge is ``int(size * long / short)`` (truncation, not rounding).
    """
    target = int(size)
    short, long = (width, height) if width <= height else (height, width)
    new_short, new_long = target, int(target * long / short)
    new_w, new_h = (new_short, new_long) if width <= height else (new_long, new_short)
    return int(new_w), int(new_h)


def _center_crop_ops_for_size(width: int, height: int, size: int) -> list[GeometryOp]:
    new_w, new_h = resize_output_size(size, width, height)
    ops = [
        GeometryOp(
            "resize",
            {
                "from_wh": [width, height],
                "to_wh": [new_w, new_h],
                "interpolation": "bicubic",
                "antialias": True,
            },
        )
    ]
    crop_w = min(int(size), new_w)
    crop_h = min(int(size), new_h)
    left = int(round((new_w - crop_w) / 2.0))
    top = int(round((new_h - crop_h) / 2.0))
    ops.append(
        GeometryOp(
            "crop",
            {
                "from_wh": [new_w, new_h],
                "left": left,
                "top": top,
                "width": crop_w,
                "height": crop_h,
            },
        )
    )
    return ops


def _whole_specimen_ops_for_size(
    width: int, height: int, size: int, fill: tuple[int, int, int]
) -> list[GeometryOp]:
    side = max(width, height)
    offset = (
        (0, (side - height) // 2) if width > height else ((side - width) // 2, 0)
    )
    ops: list[GeometryOp] = []
    if width != height:
        ops.append(
            GeometryOp(
                "pad",
                {
                    "from_wh": [width, height],
                    "canvas_wh": [side, side],
                    "offset": [offset[0], offset[1]],
                    "fill": list(fill),
                },
            )
        )
    canvas = (side, side)
    ops.append(
        GeometryOp(
            "resize",
            {
                "from_wh": [canvas[0], canvas[1]],
                "to_wh": [int(size), int(size)],
                "interpolation": "bicubic",
                "antialias": True,
            },
        )
    )
    return ops


def evaluation_view_operations(
    source_wh: tuple[int, int],
    *,
    name: str,
    size: int,
    fill: tuple[int, int, int] | None = None,
) -> list[GeometryOp]:
    """Operations of one named deterministic eval protocol for a source size.

    Size-based so a consumer that only knows the stored dimensions (CAM's
    original-coordinate mapping) derives the crop bounds and padding offsets from
    the same owner as extraction, instead of recomputing them.
    """
    width, height = int(source_wh[0]), int(source_wh[1])
    if name == "center-crop":
        return _center_crop_ops_for_size(width, height, size)
    if name == "whole-specimen-pad":
        return _whole_specimen_ops_for_size(
            width, height, size, fill if fill is not None else BACKGROUND_FILL_FALLBACK
        )
    raise ValueError(
        f"unknown eval transform {name!r}; expected one of "
        "'center-crop', 'whole-specimen-pad'"
    )


def evaluation_view_ops(image: Image.Image, *, name: str, size: int) -> list[GeometryOp]:
    """The recorded operation list of one named deterministic eval protocol."""
    fill = (
        estimate_background_color(image) if name == "whole-specimen-pad" else None
    )
    return evaluation_view_operations(image.size, name=name, size=size, fill=fill)


def normalize_tensor(image: Image.Image):
    """Tensorize and apply the shared ImageNet normalization exactly once."""
    import torch
    from torchvision.transforms import Normalize, ToTensor

    tensor = ToTensor()(image.convert("RGB"))
    return Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))(tensor)


def build_eval_view(
    image: Image.Image, *, name: str, size: int
):
    """Build one deterministic eval view plus its geometry trace."""
    ops = evaluation_view_ops(image, name=name, size=size)
    viewed, trace = apply_geometry(image, ops, track=True)
    assert trace is not None
    return normalize_tensor(viewed), trace


def pixel_support(trace: GeometryTrace) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-level geometric coverage (float32) and conservative validity (Boolean)."""
    polygon = content_support(trace)
    valid = content_validity(trace)
    coverage = support.pixel_coverage(polygon, trace.view_wh)
    if valid.shape != (trace.view_wh[1], trace.view_wh[0]):
        raise ValueError(
            f"validity is {valid.shape} but the view is {trace.view_wh}"
        )
    return coverage.astype(np.float32), valid


def content_support(trace: GeometryTrace) -> np.ndarray:
    """View-space polygon of source content that survived the whole sequence.

    Every operation keeps its own clipping history: the current support is
    clipped to the crop window it enters, transformed by the realized map, and
    clipped again to the canvas the operation produced. An invertible matrix
    alone cannot express content an intermediate crop, rotation or canvas
    truncation discarded, so this polygon - not the final map - is the support
    domain of the stored view. Everything outside it holds fill.
    """
    polygon = support.rect_polygon(*trace.source_wh)
    for index, op in enumerate(trace.operations):
        polygon = _transform_support_polygon(polygon, op)
        if len(polygon) < 3:
            return np.zeros((0, 2), dtype=np.float64)
        canvas = trace.canvas_before(index + 1)
        polygon = support.clip_polygon_to_rect(
            polygon, float(canvas[0]), float(canvas[1])
        )
    return polygon


def content_validity(trace: GeometryTrace) -> np.ndarray:
    """Boolean "this view pixel still holds source content" mask.

    The image-level counterpart of :func:`content_support`: exact for crop, pad
    and flips, conservative (kernel footprint) for the resamplers that mix
    neighbouring samples. Used for categorical rasters, whose support is not a
    polygon.
    """
    valid = np.ones((trace.source_wh[1], trace.source_wh[0]), dtype=bool)
    for op in trace.operations:
        valid = _propagate_valid(valid, op)
    return valid


def _transform_support_polygon(polygon: np.ndarray, op: GeometryOp) -> np.ndarray:
    """Clip to the operation's own window, then map into the produced canvas."""
    if op.kind == "crop":
        # The crop window is an offset rectangle in the entering canvas; clipping
        # to [0,left+width] x [0,top+height] would keep discarded content.
        polygon = support.clip_polygon_to_window(
            polygon,
            float(op.params["left"]),
            float(op.params["top"]),
            float(op.params["width"]),
            float(op.params["height"]),
        )
        if len(polygon) < 3:
            return np.zeros((0, 2), dtype=np.float64)
    transformed = support.transform_polygon(matrix_for(op), polygon)
    return transformed


def _propagate_valid(valid: np.ndarray, op: GeometryOp) -> np.ndarray:
    params = op.params
    kind = op.kind
    if kind == "crop":
        top = int(params["top"])
        left = int(params["left"])
        return valid[top : top + int(params["height"]), left : left + int(params["width"])]
    if kind == "pad":
        canvas = [int(v) for v in params["canvas_wh"]]
        offset = [int(v) for v in params["offset"]]
        padded = np.zeros((canvas[1], canvas[0]), dtype=bool)
        padded[
            offset[1] : offset[1] + valid.shape[0],
            offset[0] : offset[0] + valid.shape[1],
        ] = valid
        return padded
    if kind == "hflip":
        return valid[:, ::-1]
    if kind == "vflip":
        return valid[::-1, :]
    if kind == "resize":
        to_wh = (int(params["to_wh"][0]), int(params["to_wh"][1]))
        from_wh = (int(params["from_wh"][0]), int(params["from_wh"][1]))
        radius = support.interpolation_radius(params["interpolation"])
        if bool(params.get("antialias", True)):
            # An antialiased resampler widens its kernel by the downscale factor.
            radius *= max(
                1.0, from_wh[0] / to_wh[0], from_wh[1] / to_wh[1]
            )
        return support.footprint_valid(
            valid,
            support.invert(matrix_for(op)),
            to_wh,
            radius=radius,
            clamp=True,
        )
    if kind in ("rotate", "affine"):
        canvas = [int(v) for v in params["canvas_wh"]]
        # The kernel radius is the resampler's source-pixel support. Upscaling
        # does not shrink it: torchvision keeps kernel width 2.0 source pixels
        # for bicubic and widens it only when downscaling, so dividing by the
        # affine scale would mark fill-dependent pixels as valid.
        radius = support.interpolation_radius(params["interpolation"])
        return support.footprint_valid(
            valid,
            support.invert(matrix_for(op)),
            (canvas[0], canvas[1]),
            radius=radius,
            clamp=False,
        )
    raise ValueError(f"unhandled geometry operation {kind!r}")  # pragma: no cover


def patch_support(
    trace: GeometryTrace,
    *,
    pixel_sampling_valid: np.ndarray,
    grid_hw: tuple[int, int],
    patch_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Patch-level coverage/validity from the pixel arrays."""
    coverage, _ = pixel_support(trace)
    return support.patch_reduce(
        coverage,
        pixel_sampling_valid,
        grid_hw=grid_hw,
        patch_hw=patch_hw,
    )


def map_points(
    trace: GeometryTrace, xy: np.ndarray, *, inverse: bool = False
) -> np.ndarray:
    """Map ``(N, 2)`` view-edge points between source and view coordinates."""
    matrix = trace.view_to_source if inverse else trace.source_to_view
    return support.transform_points(matrix, xy)


def transform_annotations(
    annotations: Sequence[Mapping[str, Any]],
    categories: Mapping[Any, Mapping[str, Any]],
    trace: GeometryTrace,
    *,
    role: str,
) -> list[dict[str, Any]]:
    """Map source annotations through one realized geometry.

    Source records, ids, category schema, visibility and crowd flags are kept
    verbatim. Derived view geometry is additive: a clipped ``polygon_xy`` with its
    own retained area, a separately exposed axis-aligned enclosure, mapped
    keypoints in source (pixel-index) convention, and a nearest-neighbour
    categorical raster with its own support.

    Everything derived is clipped to the same surviving-content domain as the
    stored image view (:func:`content_support`), so an annotation whose content an
    intermediate crop or canvas truncation discarded is out of view even when the
    final invertible map would place it inside the canvas.
    """
    if role not in ("instances", "keypoints", "panoptic"):
        raise ValueError(f"unknown annotation role {role!r}")
    view_width, view_height = trace.view_wh
    matrix = trace.source_to_view
    support_polygon = content_support(trace)
    support_mask = content_validity(trace)
    results: list[dict[str, Any]] = []
    for annotation in annotations:
        category = _category_for(categories, annotation.get("category_id"), role)
        record: dict[str, Any] = {
            "annotation_id": annotation.get("id"),
            "image_id": annotation.get("image_id"),
            "category_id": annotation.get("category_id"),
            "role": role,
            "category": category,
            "visibility": annotation.get("visibility"),
            "iscrowd": annotation.get("iscrowd"),
            "source": dict(annotation),
            "coordinate_adapter": SPATIAL_COORDINATE_ADAPTER,
            "annotation_in_view": None,
            "polygon_xy": None,
            "polygons_xy": None,
            "bbox_xyxy": None,
            "bbox_xy": None,
            "area_source": None,
            "area_retained": None,
            "keypoints": None,
            "keypoints_in_view": None,
            "mask": None,
            "mask_support": None,
            "mask_unavailable": None,
        }

        segmentation = annotation.get("segmentation")
        polygons = support_polygon_parts(segmentation)
        # A source raster already describes the object; a polygon then does not
        # need the bbox fallback either.
        polygons_absent = not polygons and annotation.get("raster") is None
        if polygons:
            transformed: list[list[list[float]]] = []
            retained = 0.0
            source_area = 0.0
            for part in polygons:
                points = np.asarray(part, dtype=np.float64).reshape(-1, 2)
                source_area += support.polygon_area(points)
                moved = support.transform_polygon(matrix, points)
                clipped = _clip_to_view(
                    moved, support_polygon, view_width, view_height
                )
                retained += support.polygon_area(clipped)
                if len(clipped) >= 3:
                    transformed.append(
                        [[float(x), float(y)] for x, y in clipped]
                    )
            record["area_source"] = float(source_area)
            record["area_retained"] = float(retained)
            if transformed:
                primary = max(
                    transformed,
                    key=lambda part: support.polygon_area(
                        np.asarray(part, dtype=np.float64)
                    ),
                )
                record["polygon_xy"] = primary
                record["polygons_xy"] = transformed
                rectangle = support.series_bbox(
                    np.asarray(primary, dtype=np.float64)
                )
                record["bbox_xyxy"] = [float(v) for v in rectangle]
                record["bbox_xy"] = [
                    float(rectangle[0]),
                    float(rectangle[1]),
                    float(rectangle[2] - rectangle[0]),
                    float(rectangle[3] - rectangle[1]),
                ]
            record["annotation_in_view"] = retained > 0.0

        bbox = annotation.get("bbox")
        # The bbox is a fallback for an annotation that *carries no usable
        # polygon*, not a second chance for one whose segmentation was clipped
        # away: two parts on either side of a centre crop are genuinely out of
        # view, and their enclosure would otherwise reappear as a visible object.
        if (
            polygons_absent
            and isinstance(bbox, (list, tuple))
            and len(bbox) == 4
        ):
            x, y, box_w, box_h = (float(v) for v in bbox)
            corners = np.array(
                [[x, y], [x + box_w, y], [x + box_w, y + box_h], [x, y + box_h]],
                dtype=np.float64,
            )
            record["area_source"] = float(box_w * box_h)
            moved = support.transform_polygon(matrix, corners)
            clipped = _clip_to_view(moved, support_polygon, view_width, view_height)
            retained = support.polygon_area(clipped)
            record["area_retained"] = float(retained)
            record["annotation_in_view"] = bool(retained > 0.0)
            if len(clipped) >= 3:
                record["polygon_xy"] = [[float(a), float(b)] for a, b in clipped]
                record["polygons_xy"] = [record["polygon_xy"]]
                rectangle = support.series_bbox(clipped)
                record["bbox_xyxy"] = [float(v) for v in rectangle]
                record["bbox_xy"] = [
                    float(rectangle[0]),
                    float(rectangle[1]),
                    float(rectangle[2] - rectangle[0]),
                    float(rectangle[3] - rectangle[1]),
                ]

        if role == "keypoints" or "keypoints" in annotation:
            record["keypoints"], record["keypoints_in_view"] = _map_keypoints(
                annotation.get("keypoints") or [], trace, support_polygon
            )

        raster = annotation.get("raster")
        if raster is not None and role in ("panoptic", "instances"):
            warped, warped_support = warp_categorical(np.asarray(raster), trace)
            record["mask"] = np.where(warped_support, warped, 0).astype(warped.dtype)
            record["mask_support"] = warped_support
        elif role in ("panoptic", "instances") and isinstance(segmentation, dict):
            record["mask_unavailable"] = "rle_mask_requires_explicit_decode"

        results.append(record)
    return results


def _category_for(
    categories: Mapping[Any, Mapping[str, Any]], category_id: Any, role: str
) -> dict[str, Any]:
    """One annotation's category within its source's namespace.

    A product stores categories per source role (``instances:1``,
    ``keypoints:1``), because a COCO category id is source-local. A flat table, or
    a product written before that change, still resolves by bare id - in either key
    form, because a JSON round trip turns the integer key into a string and an
    integer lookup would silently lose the whole category schema.
    """
    if category_id is None:
        return {}
    try:
        numeric = int(category_id)
    except (TypeError, ValueError):
        return {}
    for key in (f"{role}:{numeric}", numeric, str(numeric), str(category_id)):
        if key in categories:
            return dict(categories[key] or {})
    return {}


def _execute_categorical(image: Image.Image, op: GeometryOp) -> Image.Image:
    """Execute one recorded operation on a categorical raster.

    The mask must move exactly as the image did, so the same library call is used
    with nearest-neighbour interpolation instead of a re-derived sampling rule:
    every resampler has its own rounding and centre convention (Pillow's ``COORD``
    truncation, resize's ``xmin/xmax`` rounding, rotation's pivot arithmetic), and
    reimplementing them produced categories one pixel off. Categories are integers,
    so the working image is mode ``I`` and every fill is the identity value 0.
    """
    params = op.params
    kind = op.kind
    if kind == "crop":
        return TF.crop(
            image,
            int(params["top"]),
            int(params["left"]),
            int(params["height"]),
            int(params["width"]),
        )
    if kind == "resize":
        to_w, to_h = (int(v) for v in params["to_wh"])
        return TF.resize(
            image, [to_h, to_w], interpolation=InterpolationMode.NEAREST
        )
    if kind == "pad":
        canvas = [int(v) for v in params["canvas_wh"]]
        offset = [int(v) for v in params["offset"]]
        output = Image.new(image.mode, (canvas[0], canvas[1]), color=0)
        output.paste(image, (offset[0], offset[1]))
        return output
    if kind == "rotate":
        return TF.rotate(
            image,
            float(params["angle"]),
            interpolation=InterpolationMode.NEAREST,
            expand=bool(params["expand"]),
            center=[float(v) for v in params["center"]],
            fill=0,
        )
    if kind == "affine":
        return TF.affine(
            image,
            float(params["angle"]),
            [float(v) for v in params["translate"]],
            float(params["scale"]),
            [float(v) for v in params["shear"]],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
    if kind == "hflip":
        return TF.hflip(image)
    if kind == "vflip":
        return TF.vflip(image)
    raise ValueError(f"unhandled geometry operation {kind!r}")  # pragma: no cover


def warp_categorical(raster: np.ndarray, trace: GeometryTrace) -> tuple[np.ndarray, np.ndarray]:
    """Replay a categorical raster through the stored operations, one at a time.

    Composing two nearest-neighbour resamples into the final matrix is not
    equivalent to resampling twice: nearest sampling is not linear, so sampling
    once over the composed map moves categories the real sequence never moved.
    Each operation is therefore applied to the raster with the *same* library call
    the image used, which keeps the backend's own rounding and centre rules.

    A known-value mask travels through the identical sequence: where an operation
    wrote its fill the mask is 0, so the result is the mask's own
    ``categorical-nearest-neighbour-v1`` support rather than the image resampler's
    wider bicubic footprint. A real category is therefore never cleared because an
    unrelated image kernel reached a neighbouring cell, and a real category 0 (void)
    is never confused with fill.
    """
    values = np.asarray(raster)
    if values.ndim != 2:
        raise ValueError(
            f"a categorical raster must be 2-D, got shape {values.shape}"
        )
    if values.dtype.kind not in ("u", "i"):
        raise ValueError(
            f"a categorical raster must hold integer ids, got dtype {values.dtype}"
        )
    # The backend works in int32, so a 64-bit id would be silently truncated
    # (`4294967297` becomes `1`). Rather than shrink the accepted range, relabel
    # the ids to a dense index, replay, then map the index back: every id the
    # caller supplied is recovered exactly, whatever its width.
    ids, inverse = np.unique(values, return_inverse=True)
    index = (inverse.reshape(values.shape) + 1).astype(np.int32)  # 0 stays fill
    warped = Image.fromarray(index, mode="I")
    known = Image.fromarray(np.ones(values.shape, dtype=np.int32), mode="I")
    for op in trace.operations:
        warped = _execute_categorical(warped, op)
        known = _execute_categorical(known, op)
    positions = np.asarray(warped, dtype=np.int64)
    surviving = np.asarray(known) > 0
    restored = np.zeros(positions.shape, dtype=values.dtype)
    restored[surviving] = ids[positions[surviving] - 1]
    return restored, surviving


def _clip_to_view(
    polygon: np.ndarray,
    support_polygon: np.ndarray,
    view_width: int,
    view_height: int,
) -> np.ndarray:
    """Clip a mapped annotation to the canvas *and* the surviving content."""
    clipped = support.clip_polygon_to_rect(
        polygon, float(view_width), float(view_height)
    )
    if len(clipped) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    clipped = support.clip_polygon_to_convex(clipped, support_polygon)
    if len(clipped) < 3 or support.polygon_area(clipped) <= 1e-12:
        # Touching the support boundary is not retained content.
        return np.zeros((0, 2), dtype=np.float64)
    return clipped


def support_polygon_parts(segmentation: Any) -> list[list[float]]:
    if not isinstance(segmentation, list):
        return []
    return [list(part) for part in segmentation if len(part) >= 6]


def _map_keypoints(
    values: Sequence[float], trace: GeometryTrace, support_polygon: np.ndarray
) -> tuple[list[list[float]], list[bool]]:
    """Map keypoints through the chosen ``coco-pixel-index-to-edge-v1`` adapter.

    Labeled points (``v`` 1 or 2) are shifted ``+0.5`` into internal edge
    coordinates, mapped, then shifted back for source-format output. The
    unlabeled ``v=0, x=y=0`` sentinel is passed through unchanged: it is not a
    location, so shifting it would invent one.

    ``keypoints_in_view`` additionally requires the mapped point to lie in the
    surviving-content domain, so a point whose content an intermediate crop
    discarded is out of view even though the final map keeps it on the canvas.
    """
    width, height = trace.view_wh
    mapped: list[list[float]] = []
    inside: list[bool] = []
    points = []
    indices = []
    for index in range(0, len(values) - 2, 3):
        x, y, visibility = values[index], values[index + 1], values[index + 2]
        if visibility in (1, 2):
            points.append([float(x) + 0.5, float(y) + 0.5])
            indices.append(index)
    if points:
        transformed = support.transform_points(
            trace.source_to_view, np.asarray(points, dtype=np.float64)
        )
    else:
        transformed = np.zeros((0, 2), dtype=np.float64)
    table = {index: position for position, index in enumerate(indices)}
    for index in range(0, len(values) - 2, 3):
        x, y, visibility = values[index], values[index + 1], values[index + 2]
        if visibility in (1, 2):
            point = transformed[table[index]]
            mapped.append([float(point[0] - 0.5), float(point[1] - 0.5), float(visibility)])
            on_canvas = bool(
                0.0 <= point[0] <= width and 0.0 <= point[1] <= height
            )
            inside.append(
                bool(
                    on_canvas
                    and support.point_in_polygon(support_polygon, point)
                )
            )
        else:
            mapped.append([float(x), float(y), float(visibility)])
            inside.append(False)
    return mapped, inside
