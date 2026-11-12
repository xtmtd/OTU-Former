"""Numeric support math for the realized-geometry contract.

This module owns only array math: convex-polygon clipping, exact per-cell
coverage integration, conservative resampling footprints and patch reductions.
:mod:`otuformer.vision.geometry` decides which operations happen; nothing here
imports torch, torchvision, the CLI or the trainer, so a reader can reproduce
support from a stored trace without the image stack.

Naming follows design §5.3: pixel-level ``pixel_geometric_coverage`` (fraction)
and ``pixel_sampling_valid`` (Boolean support), patch-level
``geometric_coverage`` and ``patch_sampling_valid``. Coverage is a fraction;
validity is conservative support. Neither is a foreground or anatomy mask.

Support rule version: ``conservative-resampling-v1``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def rect_polygon(width: float, height: float) -> np.ndarray:
    """The whole-canvas support polygon in edge coordinates."""
    return np.array(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float64,
    )


def homogeneous(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.concatenate([points, ones], axis=1)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 column-vector map to ``(N, 2)`` points."""
    plain = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    mapped = homogeneous(plain) @ np.asarray(matrix, dtype=np.float64).T
    if np.any(np.abs(mapped[:, 2]) <= 1e-12):
        raise ValueError("degenerate geometry map: homogeneous weight is zero")
    return mapped[:, :2] / mapped[:, 2:3]


def transform_polygon(matrix: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    return transform_points(matrix, polygon)


def _signed_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(polygon: np.ndarray) -> float:
    return abs(_signed_area(np.asarray(polygon, dtype=np.float64)))


def clip_polygon_to_rect(polygon: np.ndarray, width: float, height: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon to ``[0,w] x [0,h]``."""
    points = [tuple(point) for point in np.asarray(polygon, dtype=np.float64)]
    for axis, limit, keep_greater in (
        (0, 0.0, True),
        (0, float(width), False),
        (1, 0.0, True),
        (1, float(height), False),
    ):
        if not points:
            break
        clipped: list[tuple[float, float]] = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            current_inside = (
                current[axis] >= limit if keep_greater else current[axis] <= limit
            )
            previous_inside = (
                previous[axis] >= limit if keep_greater else previous[axis] <= limit
            )
            if current_inside != previous_inside:
                denominator = current[axis] - previous[axis]
                if abs(denominator) > 1e-15:
                    ratio = (limit - previous[axis]) / denominator
                    clipped.append(
                        (
                            previous[0] + ratio * (current[0] - previous[0]),
                            previous[1] + ratio * (current[1] - previous[1]),
                        )
                    )
            if current_inside:
                clipped.append(current)
        points = clipped
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(points, dtype=np.float64)


def _extent_at(polygon: np.ndarray, y: float) -> tuple[float, float] | None:
    """Left/right x of a convex polygon at one scanline, or ``None``."""
    xs: list[float] = []
    count = len(polygon)
    for index in range(count):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % count]
        if y0 == y1:
            if y0 == y:
                xs.extend([float(x0), float(x1)])
            continue
        low, high = (y0, y1) if y0 < y1 else (y1, y0)
        if low <= y <= high:
            ratio = (y - y0) / (y1 - y0)
            xs.append(float(x0 + ratio * (x1 - x0)))
    if not xs:
        return None
    return min(xs), max(xs)


def pixel_coverage(polygon: np.ndarray, view_wh: tuple[int, int]) -> np.ndarray:
    """Exact geometric coverage of every view pixel cell.

    For each view row the polygon is intersected with the unit-high strip; the
    strip is split at every vertex and every integer-x crossing, so each
    sub-interval keeps ``x_left``/``x_right`` inside one column. The overlap with
    each touched cell is then integrated exactly from the linear edge functions.
    """
    width, height = int(view_wh[0]), int(view_wh[1])
    coverage = np.zeros((height, width), dtype=np.float64)
    points = np.asarray(polygon, dtype=np.float64)
    if len(points) < 3:
        return coverage.astype(np.float32)
    points = clip_polygon_to_rect(points, width, height)
    if len(points) < 3:
        return coverage.astype(np.float32)

    integer_lines = set(range(0, width + 1))
    for row in range(height):
        y0, y1 = float(row), float(row + 1.0)
        breaks = {y0, y1}
        for index in range(len(points)):
            x_a, y_a = points[index]
            x_b, y_b = points[(index + 1) % len(points)]
            for value in (y_a, y_b):
                if y0 < value < y1:
                    breaks.add(float(value))
            if abs(y_b - y_a) <= 1e-15:
                continue
            low_x, high_x = (x_a, x_b) if x_a <= x_b else (x_b, x_a)
            for line in integer_lines:
                if low_x < line < high_x:
                    ratio = (line - x_a) / (x_b - x_a)
                    crossing = y_a + ratio * (y_b - y_a)
                    if y0 < crossing < y1:
                        breaks.add(float(crossing))
        ordered = sorted(breaks)
        for lower, upper in zip(ordered[:-1], ordered[1:]):
            span = upper - lower
            if span <= 0:
                continue
            middle = 0.5 * (lower + upper)
            extent_middle = _extent_at(points, middle)
            if extent_middle is None:
                continue
            left_middle = extent_middle[0]
            right_middle = extent_middle[1]
            if right_middle <= left_middle:
                continue
            extent_lower = _extent_at(points, lower) or extent_middle
            extent_upper = _extent_at(points, upper) or extent_middle
            mean_left = 0.5 * (extent_lower[0] + extent_upper[0])
            mean_right = 0.5 * (extent_lower[1] + extent_upper[1])
            first = int(math.floor(left_middle))
            last = int(math.floor(right_middle))
            if last < 0 or first > width - 1:
                continue
            first = max(first, 0)
            last = min(last, width - 1)
            if last < first:  # pragma: no cover - guarded by the crossing split
                continue
            if first == last:
                coverage[row, first] += span * (mean_right - mean_left)
                continue
            coverage[row, first] += span * ((first + 1) - mean_left)
            coverage[row, last] += span * (mean_right - last)
            if last > first + 1:
                coverage[row, first + 1 : last] += span
    return np.clip(coverage, 0.0, 1.0).astype(np.float32)


def _integral(mask: np.ndarray) -> np.ndarray:
    """Zero-padded summed-area table of a Boolean/0-1 array."""
    padded = np.zeros(
        (mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64
    )
    padded[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    return padded


def _box_sum(integral: np.ndarray, rows0, rows1, cols0, cols1) -> np.ndarray:
    return (
        integral[rows1 + 1, cols1 + 1]
        - integral[rows0, cols1 + 1]
        - integral[rows1 + 1, cols0]
        + integral[rows0, cols0]
    )


def footprint_valid(
    valid: np.ndarray,
    inverse_matrix: np.ndarray,
    out_wh: tuple[int, int],
    *,
    radius: float,
    clamp: bool,
) -> np.ndarray:
    """Conservative support for one resampling operation.

    For every output pixel the inverse map gives its source coordinate; an
    isotropic ``radius`` (in source pixels) covers the interpolation kernel's
    contributing indices. Coverage of an invalid source index makes the output
    pixel invalid.

    ``clamp=True`` models kernels whose out-of-bounds taps replicate the nearest
    in-bounds sample (resize). ``clamp=False`` models kernels that take the fill
    value for out-of-bounds taps (rotate/affine), so a footprint leaving the
    source is invalid rather than clamped.
    """
    out_h, out_w = int(out_wh[1]), int(out_wh[0])
    source_h, source_w = valid.shape
    rows, cols = np.indices((out_h, out_w), dtype=np.float64)
    points = np.stack([cols + 0.5, rows + 0.5], axis=-1)
    coordinates = transform_points(inverse_matrix, points.reshape(-1, 2))
    source_x = coordinates[:, 0].reshape(out_h, out_w)
    source_y = coordinates[:, 1].reshape(out_h, out_w)

    col0 = np.floor(source_x - radius)
    col1 = np.floor(source_x + radius)
    row0 = np.floor(source_y - radius)
    row1 = np.floor(source_y + radius)

    inside = (col0 >= 0) & (row0 >= 0) & (col1 <= source_w - 1) & (row1 <= source_h - 1)
    col0 = np.clip(col0, 0, source_w - 1)
    col1 = np.clip(col1, 0, source_w - 1)
    row0 = np.clip(row0, 0, source_h - 1)
    row1 = np.clip(row1, 0, source_h - 1)
    if clamp:
        inside = np.ones_like(inside, dtype=bool)
    integral = _integral(~np.asarray(valid, dtype=bool))
    invalid = _box_sum(
        integral,
        row0.astype(np.int64),
        row1.astype(np.int64),
        col0.astype(np.int64),
        col1.astype(np.int64),
    )
    return inside & (invalid == 0)


def interpolation_radius(interpolation: str) -> float:
    """Conservative kernel radius in source pixels for one resampler."""
    name = str(interpolation).lower()
    if "nearest" in name:
        return 0.5
    if "bicubic" in name:
        return 2.0
    if "bilinear" in name or "linear" in name:
        return 1.0
    raise ValueError(f"unsupported interpolation {interpolation!r}")


def patch_reduce(
    pixel_values: np.ndarray,
    pixel_valid: np.ndarray,
    *,
    grid_hw: tuple[int, int],
    patch_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce pixel coverage/validity onto the feature patch lattice.

    Patch coverage is the mean of the pixel coverage inside the patch; patch
    validity requires every pixel of the patch footprint to be valid.
    """
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    patch_h, patch_w = int(patch_hw[0]), int(patch_hw[1])
    height, width = pixel_values.shape
    if (height, width) != (grid_h * patch_h, grid_w * patch_w):
        raise ValueError(
            f"pixel arrays are {height}x{width} but grid {grid_hw} and patch "
            f"{patch_hw} require {grid_h * patch_h}x{grid_w * patch_w}"
        )
    coverage = np.asarray(pixel_values, dtype=np.float64).reshape(
        grid_h, patch_h, grid_w, patch_w
    ).mean(axis=(1, 3))
    valid = np.asarray(pixel_valid, dtype=bool).reshape(
        grid_h, patch_h, grid_w, patch_w
    ).all(axis=(1, 3))
    return coverage.astype(np.float32), valid


def affine_is_invertible(matrix: np.ndarray) -> bool:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return False
    determinant = float(np.linalg.det(matrix))
    return bool(np.isfinite(determinant) and abs(determinant) > 1e-12)


def invert(matrix: np.ndarray) -> np.ndarray:
    if not affine_is_invertible(matrix):
        raise ValueError("geometry map is singular or non-finite")
    return np.linalg.inv(np.asarray(matrix, dtype=np.float64))


def series_bbox(polygon: np.ndarray) -> np.ndarray:
    """Axis-aligned enclosure of a polygon as ``[x0, y0, x1, y1]``."""
    points = np.asarray(polygon, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(4, dtype=np.float64)
    return np.array(
        [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()],
        dtype=np.float64,
    )


def nearest_warp(
    raster: np.ndarray,
    forward_matrix: np.ndarray,
    out_wh: tuple[int, int],
    *,
    fill: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Category-safe nearest-neighbour warp of a categorical raster.

    Returns the warped raster and the Boolean support of output cells whose
    nearest source sample lay inside the source raster. Fill is written where it
    did not, and the support tells a consumer not to read that fill as a real
    category.
    """
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    inverse = invert(forward_matrix)
    rows, cols = np.indices((out_h, out_w), dtype=np.float64)
    points = np.stack([cols + 0.5, rows + 0.5], axis=-1)
    coordinates = transform_points(inverse, points.reshape(-1, 2))
    source_cols = np.floor(coordinates[:, 0]).astype(np.int64).reshape(out_h, out_w)
    source_rows = np.floor(coordinates[:, 1]).astype(np.int64).reshape(out_h, out_w)
    source_h, source_w = raster.shape
    support = (
        (source_cols >= 0)
        & (source_cols < source_w)
        & (source_rows >= 0)
        & (source_rows < source_h)
    )
    warped = np.full((out_h, out_w), fill, dtype=raster.dtype)
    warped[support] = raster[source_rows[support], source_cols[support]]
    return warped, support


def coverage_summary(pixel_coverage_array: np.ndarray) -> dict[str, float]:
    """Deterministic coverage statistics for diagnostics."""
    values = np.asarray(pixel_coverage_array, dtype=np.float64)
    if values.size == 0:  # pragma: no cover - guarded by callers
        return {"min": 0.0, "mean": 0.0, "full_valid_fraction": 0.0}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "full_valid_fraction": float(np.count_nonzero(values >= 1.0) / values.size),
    }


def iter_patch_blocks(
    grid_hw: tuple[int, int], patch_hw: tuple[int, int]
) -> Iterable[tuple[int, int, int, int]]:
    for row in range(int(grid_hw[0])):
        for column in range(int(grid_hw[1])):
            yield (
                row * int(patch_hw[0]),
                (row + 1) * int(patch_hw[0]),
                column * int(patch_hw[1]),
                (column + 1) * int(patch_hw[1]),
            )


def as_float32(array: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)
