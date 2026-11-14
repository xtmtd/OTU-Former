"""Frozen same-image correspondence protocol (`spatial-geometry-v1`).

The protocol is fixed: one reference view plus eleven geometric transforms of the
same decoded source, a deterministic image/query ranking, and geometric
eligibility decided from support only. No model value influences which images or
queries are used, no appearance perturbation is applied, and nothing here trains
or changes a checkpoint.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..constants import (
    SPATIAL_PROTOCOL,
    SPATIAL_QUERY_LIMIT,
    SPATIAL_SCHEMA_VERSION,
    SPATIAL_SUPPORT_VERSION,
)
from ..data.provenance import canonical_digest
from ..vision.features import sample_grid_features
from ..vision.geometry import (
    GeometryOp,
    GeometryTrace,
    apply_geometry,
    evaluation_view_ops,
    normalize_tensor,
)
from ..vision.geometry_support import transform_points
from .records import SpatialRecordError, SpatialView

#: Ordered view ids. ``reference`` is the deterministic eval view of the source.
REFERENCE_VIEW = "reference"
#: The reference may alias the stored base view instead of being saved twice.
REFERENCE_ALIASES = ("reference", "base")
PROTOCOL_VIEW_IDS = (
    "reference",
    "crop075",
    "tx10",
    "ty10",
    "scale090",
    "scale110",
    "rot-15",
    "rot15",
    "rot90",
    "hflip",
    "vflip",
    "combo",
)
DIRECTIONS = ("reference_to_target", "target_to_reference")


def _half_even(value: float) -> int:
    """Integer round-to-nearest with half-to-even ties (the protocol's ``round``)."""
    return int(round(value))


def protocol_operations(size: int, view_id: str) -> list[GeometryOp]:
    """Operations of one protocol view, relative to the reference canvas.

    ``size`` is the resolved square model input; the caller has already built the
    reference view, so every operation starts from an ``size x size`` canvas.
    """
    size = int(size)
    if size <= 0:
        raise ValueError("protocol size must be positive")
    shift = _half_even(0.10 * size)
    fixed_canvas = [size, size]

    def affine(*, translate=(0.0, 0.0), scale=1.0) -> GeometryOp:
        return GeometryOp(
            "affine",
            {
                "from_wh": fixed_canvas,
                "angle": 0.0,
                "translate": [float(translate[0]), float(translate[1])],
                "scale": float(scale),
                "shear": [0.0, 0.0],
                "interpolation": "bicubic",
                "canvas_wh": fixed_canvas,
            },
        )

    def rotate(angle: float) -> GeometryOp:
        return GeometryOp(
            "rotate",
            {
                "from_wh": fixed_canvas,
                "angle": float(angle),
                "expand": False,
                "interpolation": "bicubic",
            },
        )

    def crop075() -> list[GeometryOp]:
        side = int(math.floor(0.75 * size))
        side = max(1, side)
        offset = (size - side) // 2
        return [
            GeometryOp(
                "crop",
                {
                    "from_wh": fixed_canvas,
                    "left": offset,
                    "top": offset,
                    "width": side,
                    "height": side,
                },
            ),
            GeometryOp(
                "resize",
                {
                    "from_wh": [side, side],
                    "to_wh": fixed_canvas,
                    "interpolation": "bicubic",
                    "antialias": True,
                },
            ),
        ]

    if view_id == "reference":
        return []
    if view_id == "crop075":
        return crop075()
    if view_id == "tx10":
        return [affine(translate=(shift, 0.0))]
    if view_id == "ty10":
        return [affine(translate=(0.0, shift))]
    if view_id == "scale090":
        return [affine(scale=0.9)]
    if view_id == "scale110":
        return [affine(scale=1.1)]
    if view_id == "rot-15":
        return [rotate(-15.0)]
    if view_id == "rot15":
        return [rotate(15.0)]
    if view_id == "rot90":
        return [rotate(90.0)]
    if view_id == "hflip":
        return [GeometryOp("hflip", {"from_wh": fixed_canvas})]
    if view_id == "vflip":
        return [GeometryOp("vflip", {"from_wh": fixed_canvas})]
    if view_id == "combo":
        # crop075, then +15 degrees, then a positive x translation: that order.
        return [*crop075(), rotate(15.0), affine(translate=(shift, 0.0))]
    raise SpatialRecordError(
        f"unknown protocol view {view_id!r}; expected one of {PROTOCOL_VIEW_IDS}"
    )


def build_evaluation_views(
    image, *, image_key: str, eval_transform: str, size: int
) -> tuple[tuple[str, Any, GeometryTrace], ...]:
    """Build the twelve protocol views in the table's exact order.

    Each view is executed from the same decoded source: the deterministic eval
    construction followed by that view's own operations, so all twelve traces share
    one ``source_wh`` and image normalization is applied exactly once at the end.
    """
    base = evaluation_view_ops(image, name=eval_transform, size=size)
    views: list[tuple[str, Any, GeometryTrace]] = []
    for view_id in PROTOCOL_VIEW_IDS:
        operations = [*base, *protocol_operations(size, view_id)]
        viewed, trace = apply_geometry(image, operations, track=True)
        assert trace is not None
        if tuple(trace.view_wh) != (int(size), int(size)):
            raise SpatialRecordError(
                f"protocol view {view_id!r} produced canvas {trace.view_wh}, expected "
                f"{(int(size), int(size))}"
            )
        views.append((view_id, normalize_tensor(viewed), trace))
    return tuple(views)


def select_evaluation_keys(
    keys: Sequence[str], *, seed: int, limit: int = 100
) -> tuple[str, ...]:
    """Deterministic image selection: SHA-256 ranking, original order restored.

    Ranking never advances the training/process RNG, and the result is independent
    of the caller's iteration order once the original index order is restored.
    """
    ordered = list(keys)
    if len(set(ordered)) != len(ordered):
        raise SpatialRecordError("evaluation keys must be unique")
    if limit <= 0:
        raise SpatialRecordError("the evaluation sample limit must be positive")
    if len(ordered) <= limit:
        return tuple(ordered)
    ranked = sorted(
        ordered,
        key=lambda key: (
            canonical_digest([SPATIAL_PROTOCOL, int(seed), "image", key]),
            key,
        ),
    )
    chosen = set(ranked[: int(limit)])
    return tuple(key for key in ordered if key in chosen)


def grid_center_point(row: int, column: int, patch_hw: tuple[int, int]) -> tuple[float, float]:
    """View-edge coordinates of one patch center."""
    patch_h, patch_w = int(patch_hw[0]), int(patch_hw[1])
    return ((column + 0.5) * patch_w, (row + 0.5) * patch_h)


def _bilinear_contributors(
    xy: Sequence[float], shape: tuple[int, int]
) -> list[tuple[int, int, float]]:
    """Pixel-index contributors and weights of one view-edge point."""
    height, width = int(shape[0]), int(shape[1])
    x, y = float(xy[0]) - 0.5, float(xy[1]) - 0.5
    col0, row0 = math.floor(x), math.floor(y)
    weight_x, weight_y = x - col0, y - row0
    contributors: list[tuple[int, int, float]] = []
    for row, row_weight in ((row0, 1.0 - weight_y), (row0 + 1, weight_y)):
        for column, column_weight in ((col0, 1.0 - weight_x), (col0 + 1, weight_x)):
            weight = row_weight * column_weight
            if weight <= 0.0:
                continue
            inside = 0 <= row < height and 0 <= column < width
            contributors.append((int(row), int(column), float(weight) if inside else -1.0))
    return contributors


def pixel_point_support(valid: np.ndarray, xy: Sequence[float]) -> bool:
    """True when every nonzero-weight pixel contributor of ``xy`` is valid."""
    for row, column, weight in _bilinear_contributors(xy, valid.shape):
        if weight < 0.0:
            return False
        if not bool(valid[row, column]):
            return False
    return True


def grid_point_support(
    patch_sampling_valid: np.ndarray, xy: Sequence[float], patch_hw: tuple[int, int]
) -> bool:
    """True when a feature interpolation at ``xy`` has complete valid support.

    Uses the shared sampler with a value-free grid: only the support rule matters
    here, which is why no feature value can influence geometric eligibility.
    """
    import torch

    valid = np.asarray(patch_sampling_valid, dtype=bool)
    grid = torch.zeros(
        (1, int(valid.shape[0]), int(valid.shape[1]), 1), dtype=torch.float32
    )
    points = torch.tensor([[[float(xy[0]), float(xy[1])]]], dtype=torch.float32)
    _, usable = sample_grid_features(
        grid,
        points,
        patch_hw=(int(patch_hw[0]), int(patch_hw[1])),
        valid=torch.as_tensor(valid)[None, ...],
    )
    return bool(usable[0, 0])


def geometry_signature(view: SpatialView) -> dict[str, Any]:
    """Stored geometry/support identity of one view, without any feature value."""
    patch_hw = (
        int(view.trace.view_wh[1]) // int(view.patch_sampling_valid.shape[0]),
        int(view.trace.view_wh[0]) // int(view.patch_sampling_valid.shape[1]),
    )
    return {
        "view_id": view.view_id,
        "role": view.role,
        "source_wh": [int(v) for v in view.trace.source_wh],
        "view_wh": [int(v) for v in view.trace.view_wh],
        "patch_hw": [patch_hw[0], patch_hw[1]],
        "grid_hw": [
            int(view.patch_sampling_valid.shape[0]),
            int(view.patch_sampling_valid.shape[1]),
        ],
        "source_to_view": [[float(v) for v in row] for row in view.trace.source_to_view],
        "support_version": SPATIAL_SUPPORT_VERSION,
    }


def _eligible_queries(
    source_view: SpatialView, target_view: SpatialView, patch_hw: tuple[int, int]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Geometric eligibility of every source grid center against one target.

    Vectorized: a plan check recomputes eligibility for every stored branch, and a
    per-query Python loop made that check cost seconds per product. Both support
    tests are the same rules the scalar helpers apply (complete bilinear feature
    support for the target grid, complete pixel support for the target view).
    """
    import torch

    from ..vision.features import sample_grid_features
    from ..vision.geometry_support import transform_points

    source_valid = np.asarray(source_view.patch_sampling_valid, dtype=bool)
    target_patch_valid = np.asarray(target_view.patch_sampling_valid, dtype=bool)
    target_pixel_valid = np.asarray(target_view.pixel_sampling_valid, dtype=bool)
    rows, columns = np.nonzero(source_valid)
    # ``p_target = M_target inverse(M_source) p_source``: map the source view's
    # point back to the shared source frame first, then forward into the target.
    pairwise = target_view.trace.source_to_view @ source_view.trace.view_to_source
    centers = (
        np.stack(
            [
                (columns + 0.5) * float(patch_hw[1]),
                (rows + 0.5) * float(patch_hw[0]),
            ],
            axis=1,
        )
        if rows.size
        else np.zeros((0, 2), dtype=np.float64)
    )
    mapped = (
        transform_points(pairwise, centers)
        if rows.size
        else np.zeros((0, 2), dtype=np.float64)
    )
    keep = np.ones(rows.size, dtype=bool)
    if rows.size:
        keep &= np.array(
            [pixel_point_support(target_pixel_valid, point) for point in mapped],
            dtype=bool,
        )
        grid = torch.zeros(
            (1, int(target_patch_valid.shape[0]), int(target_patch_valid.shape[1]), 1),
            dtype=torch.float32,
        )
        _values, usable = sample_grid_features(
            grid,
            torch.as_tensor(mapped, dtype=torch.float32)[None, ...],
            patch_hw=(int(patch_hw[0]), int(patch_hw[1])),
            valid=torch.as_tensor(target_patch_valid)[None, ...],
        )
        keep &= usable[0].to(torch.bool).numpy()
    queries = [
        {
            "row": int(row),
            "column": int(column),
            "xy_source": [float(point[0]), float(point[1])],
            "xy_target": [float(target_point[0]), float(target_point[1])],
        }
        for row, column, point, target_point in zip(
            rows[keep].tolist(),
            columns[keep].tolist(),
            centers[keep].tolist(),
            mapped[keep].tolist(),
        )
    ]
    counts = {
        "source_valid_count": int(rows.size),
        "eligible_count": len(queries),
    }
    return queries, counts


def _rank_queries(
    queries: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    image_key: str,
    pair_id: str,
    direction: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Seeded SHA-256 ranking with row-major tie-breaking, order restored."""
    if limit <= 0:
        raise SpatialRecordError("the query limit must be positive")
    if len(queries) <= limit:
        return [dict(query) for query in queries]
    ranked = sorted(
        queries,
        key=lambda query: (
            canonical_digest(
                [
                    SPATIAL_PROTOCOL,
                    int(seed),
                    "query",
                    image_key,
                    pair_id,
                    direction,
                    int(query["row"]),
                    int(query["column"]),
                ]
            ),
            int(query["row"]),
            int(query["column"]),
        ),
    )
    chosen = {
        (int(query["row"]), int(query["column"])) for query in ranked[: int(limit)]
    }
    return [
        dict(query)
        for query in queries
        if (int(query["row"]), int(query["column"])) in chosen
    ]


def build_query_plan(
    image_key: str,
    reference: SpatialView,
    target: SpatialView,
    *,
    pair_id: str,
    seed: int,
    limit: int = SPATIAL_QUERY_LIMIT,
) -> dict[str, Any]:
    """Frozen plan for one ordered pair, in both directions.

    Eligibility uses support only: no feature value, label or true location shrinks
    the candidate set. ``pair_id`` is the target view id.
    """
    if reference.view_id not in REFERENCE_ALIASES:
        raise SpatialRecordError(
            f"the reference view must be one of {REFERENCE_ALIASES}, got "
            f"{reference.view_id!r}"
        )
    if target.view_id != pair_id:
        raise SpatialRecordError(
            f"pair_id {pair_id!r} must equal the target view id {target.view_id!r}"
        )
    if image_key != reference.image_key or image_key != target.image_key:
        raise SpatialRecordError("a query plan pairs views of one image")
    patch_hw = (
        int(reference.trace.view_wh[1]) // int(reference.patch_sampling_valid.shape[0]),
        int(reference.trace.view_wh[0]) // int(reference.patch_sampling_valid.shape[1]),
    )
    directions: dict[str, Any] = {}
    pairs = (
        ("reference_to_target", reference, target),
        ("target_to_reference", target, reference),
    )
    for direction, source, destination in pairs:
        queries, counts = _eligible_queries(source, destination, patch_hw)
        sampled = _rank_queries(
            queries,
            seed=seed,
            image_key=image_key,
            pair_id=pair_id,
            direction=direction,
            limit=limit,
        )
        directions[direction] = {
            "source_view": source.view_id,
            "target_view": destination.view_id,
            "source_valid_count": counts["source_valid_count"],
            "eligible_count": counts["eligible_count"],
            "sampled_count": len(sampled),
            "unavailable": counts["eligible_count"] == 0,
            "unavailable_reason": (
                "no_overlap"
                if counts["source_valid_count"] > 0
                else "no_source_valid_center"
            )
            if counts["eligible_count"] == 0
            else None,
            "queries": sampled,
        }
    return {
        "protocol": SPATIAL_PROTOCOL,
        "schema_version": SPATIAL_SCHEMA_VERSION,
        "support_version": SPATIAL_SUPPORT_VERSION,
        "seed": int(seed),
        "limit": int(limit),
        "image_key": image_key,
        "pair_id": pair_id,
        "reference_view": reference.view_id,
        "target_view": target.view_id,
        "patch_hw": [patch_hw[0], patch_hw[1]],
        "geometry": {
            "reference": geometry_signature(reference),
            "target": geometry_signature(target),
        },
        "directions": directions,
    }


def build_evaluation_plan(
    views_by_key: Mapping[str, Sequence[SpatialView]],
    *,
    seed: int,
    image_limit: int = 100,
    query_limit: int = SPATIAL_QUERY_LIMIT,
) -> dict[str, Any]:
    """Store the frozen plan for the sampled keys, in selected order."""
    keys = select_evaluation_keys(list(views_by_key), seed=seed, limit=image_limit)
    pairs: list[dict[str, Any]] = []
    for key in keys:
        views = {view.view_id: view for view in views_by_key[key]}
        reference = next(
            (views[alias] for alias in REFERENCE_ALIASES if alias in views), None
        )
        if reference is None:
            raise SpatialRecordError(
                f"{key!r} has no reference view (expected one of {REFERENCE_ALIASES})"
            )
        missing = [
            view_id
            for view_id in PROTOCOL_VIEW_IDS
            if view_id not in views and view_id not in REFERENCE_ALIASES
        ]
        if missing:
            raise SpatialRecordError(
                f"{key!r} is missing protocol view(s): {', '.join(missing)}"
            )
        for view_id in PROTOCOL_VIEW_IDS:
            if view_id == REFERENCE_VIEW:
                continue
            pairs.append(
                build_query_plan(
                    key,
                    reference,
                    views[view_id],
                    pair_id=view_id,
                    seed=seed,
                    limit=query_limit,
                )
            )
    return {
        "protocol": SPATIAL_PROTOCOL,
        "schema_version": SPATIAL_SCHEMA_VERSION,
        "support_version": SPATIAL_SUPPORT_VERSION,
        "seed": int(seed),
        "image_limit": int(image_limit),
        "query_limit": int(query_limit),
        "view_ids": list(PROTOCOL_VIEW_IDS),
        "selected_keys": list(keys),
        "pairs": pairs,
    }


def plan_pairs_for_key(plan: Mapping[str, Any], image_key: str) -> list[dict[str, Any]]:
    return [
        dict(pair) for pair in plan.get("pairs", []) if pair.get("image_key") == image_key
    ]


def iter_plan_pairs(plan: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for pair in plan.get("pairs", []):
        yield dict(pair)
