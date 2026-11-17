"""Enabled-only dense batch geometry: correspondence pairs and query sampling.

This module answers "which positions can be compared" (design §4) and nothing
else: it builds ordered view pairs, filters them by conservative support in both
views, and samples query points without replacement. It owns no optimizer, no
CLI and no loss.

Coordinate conventions follow :mod:`otuformer.vision.geometry`: continuous
image-edge ``(x, y)`` coordinates, row-major ``(row, col)`` array indices, and
pixel centers at ``(column + 0.5, row + 0.5)``. Student queries are exact patch
centers; teacher coordinates are mapped through ``M_t @ inv(M_s)`` and stay
continuous because they generally do not land on a patch center.

Trace identity: both views describe the same source image, so every trace in a
row must agree on ``source_wh``. Only the realized trace is used; no augmentation
is ever resampled to recover geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from otuformer.vision import geometry as _geometry

#: One slot per tensor view in batch order; ``None`` only for weak-only views.
DenseBatchGeometry = tuple[tuple[_geometry.GeometryTrace, ...] | None, ...]


@dataclass(frozen=True)
class DenseSampleQueries:
    """One image's selected correspondence points for one ordered view pair."""

    student_indices: torch.Tensor
    teacher_xy: torch.Tensor
    eligible_count: int


@dataclass(frozen=True)
class DensePairQueries:
    """One ordered (teacher global, student view) pair across the batch."""

    teacher_view: int
    student_view: int
    samples: tuple[DenseSampleQueries, ...]
    student_valid: torch.Tensor
    teacher_valid: torch.Tensor


def _patch_valid(trace: _geometry.GeometryTrace, grid_hw, patch_hw) -> np.ndarray:
    """Conservative patch support for one view's feature grid."""
    pixel_valid = _geometry.content_validity(trace)
    grid_h, grid_w = map(int, grid_hw)
    patch_h, patch_w = map(int, patch_hw)
    if pixel_valid.shape != (grid_h * patch_h, grid_w * patch_w):
        raise ValueError(
            f"pixel shape {pixel_valid.shape} does not fit grid {grid_hw} "
            f"and patch {patch_hw}."
        )
    return pixel_valid.reshape(grid_h, patch_h, grid_w, patch_w).all(axis=(1, 3))


def _validate_layout(
    geometry: DenseBatchGeometry,
    grid_hws: Sequence[tuple[int, int]],
    *,
    local_crops: int,
) -> None:
    """Reject a wrongly shaped geometry tuple before any pairing work."""
    ordinary = 2 + int(local_crops)
    if len(grid_hws) != len(geometry):
        raise ValueError(
            f"geometry has {len(geometry)} view slots but {len(grid_hws)} grid "
            "sizes were supplied."
        )
    if len(geometry) < ordinary:
        raise ValueError(
            f"geometry needs at least {ordinary} ordinary view slots "
            f"(2 globals + {local_crops} locals), got {len(geometry)}."
        )
    for index in range(ordinary):
        if geometry[index] is None:
            raise ValueError(
                f"geometry slot {index} is an ordinary view and must carry a trace."
            )
    for index in range(ordinary, len(geometry)):
        slot = geometry[index]
        if slot is None:
            continue
        raise ValueError(
            f"geometry slot {index} is a weak-only slot and must be None."
        )

    rows = {
        len(slot)
        for slot in geometry
        if slot is not None and any(item is not None for item in slot)
    }
    if len(rows) != 1:
        raise ValueError(
            f"geometry slots disagree on the batch row count: {sorted(rows)}."
        )
    for index, slot in enumerate(geometry):
        if slot is None or index >= ordinary:
            continue
        for row, trace in enumerate(slot):
            grid_hw = (int(grid_hws[index][0]), int(grid_hws[index][1]))
            patch_hw_expected = (
                trace.view_wh[1] // grid_hw[0] if grid_hw[0] else 0,
                trace.view_wh[0] // grid_hw[1] if grid_hw[1] else 0,
            )
            if min(patch_hw_expected) <= 0:
                raise ValueError(
                    f"grid {grid_hw} does not divide view {trace.view_wh} at slot "
                    f"{index} row {row}."
                )
    # Same image per row: every ordinary slot must agree on the source size.
    for row in range(next(iter(rows))):
        source_sizes = {geometry[index][row].source_wh for index in range(ordinary)}
        if len(source_sizes) != 1:
            raise ValueError(
                f"row {row} mixes source sizes {sorted(source_sizes)}; correspondence "
                "requires one source image per batch row."
            )


def _batch_mask(masks: Sequence[np.ndarray]) -> torch.Tensor:
    """Stack one view's per-row Boolean support masks into ``[B,Hg,Wg]``.

    Augmentation is realized per image, so two rows of the same view do not
    share a support region. Repeating row 0's mask over the batch would mark
    padded or cropped regions as valid (or exclude valid queries).
    """
    if not masks:
        raise ValueError("A view needs at least one batch row to build a mask.")
    stacked = np.stack([np.asarray(mask, dtype=bool) for mask in masks], axis=0)
    return torch.from_numpy(np.ascontiguousarray(stacked))


def _grid_centers(grid_hw: tuple[int, int], patch_hw: tuple[int, int]) -> np.ndarray:
    """Exact patch centers as ``(N, 2)`` edge coordinates in row-major order."""
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    patch_h, patch_w = int(patch_hw[0]), int(patch_hw[1])
    rows, cols = np.meshgrid(
        np.arange(grid_h, dtype=np.float64),
        np.arange(grid_w, dtype=np.float64),
        indexing="ij",
    )
    x = (cols + 0.5) * float(patch_w)
    y = (rows + 0.5) * float(patch_h)
    return np.stack([x.reshape(-1), y.reshape(-1)], axis=1)


def _support_mask(
    xy: np.ndarray, valid: np.ndarray, grid_hw, patch_hw
) -> np.ndarray:
    """Conservative bilinear support under the *sampler's* float32 arithmetic.

    Mirrors :func:`otuformer.vision.features.sample_grid_features`: every
    contributor with a nonzero interpolation weight must be inside the grid and
    valid. Zero-weight neighbours outside the grid are fine.

    Eligibility must use the same float32 representation as the sampler that
    runs later: a float64 boundary such as ``7.9999999999999964`` rounds to
    exactly ``8.0`` in float32, so a double-precision test can reject (or
    accept) a different contributor set than the real sample. The float64
    mapping metadata is preserved; only this decision is narrowed.
    """
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    patch_h, patch_w = int(patch_hw[0]), int(patch_hw[1])
    xy32 = np.asarray(xy, dtype=np.float32)
    finite = np.isfinite(xy32).all(axis=1)
    # Same narrowing order as the sampler: cast the coordinates, then do the
    # patch-index arithmetic in float32.
    u = (xy32[:, 0] / np.float32(patch_w) - np.float32(0.5)).astype(np.float32)
    v = (xy32[:, 1] / np.float32(patch_h) - np.float32(0.5)).astype(np.float32)
    finite = finite & np.isfinite(u) & np.isfinite(v)
    usable = finite.copy()

    col0 = np.floor(np.where(finite, u, np.float32(0.0)))
    row0 = np.floor(np.where(finite, v, np.float32(0.0)))
    w_col1 = np.clip(
        np.where(finite, u, np.float32(0.0)) - col0, np.float32(0.0), np.float32(1.0)
    )
    w_row1 = np.clip(
        np.where(finite, v, np.float32(0.0)) - row0, np.float32(0.0), np.float32(1.0)
    )
    col0 = col0.astype(np.int64)
    row0 = row0.astype(np.int64)

    for rows, row_weights in ((row0, np.float32(1.0) - w_row1), (row0 + 1, w_row1)):
        for cols, col_weights in ((col0, np.float32(1.0) - w_col1), (col0 + 1, w_col1)):
            weight = (row_weights * col_weights).astype(np.float32)
            contributing = weight > np.float32(0.0)
            inside = (
                (rows >= 0) & (rows < grid_h) & (cols >= 0) & (cols < grid_w)
            )
            neighbour_valid = np.zeros_like(inside)
            safe_rows = np.clip(rows, 0, grid_h - 1)
            safe_cols = np.clip(cols, 0, grid_w - 1)
            neighbour_valid[inside] = valid[safe_rows[inside], safe_cols[inside]]
            usable &= (~contributing) | (inside & neighbour_valid)
    return usable


def _map_to_view(
    trace_from: _geometry.GeometryTrace,
    trace_to: _geometry.GeometryTrace,
    xy: np.ndarray,
) -> np.ndarray:
    """``p_to = M_to @ inv(M_from) @ p_from`` in view-edge coordinates."""
    combined = np.asarray(trace_to.source_to_view, dtype=np.float64) @ np.asarray(
        np.linalg.inv(trace_from.source_to_view), dtype=np.float64
    )
    return _geometry.support.transform_points(combined, xy)


def _sample_indices(
    eligible: np.ndarray, query_cap: int, generator: torch.Generator
) -> np.ndarray:
    """Uniform draw without replacement, then restore row-major output order.

    The draw is over candidate *positions*. Passing the patch indices themselves
    as ``multinomial`` weights would bias selection toward the end of the grid
    and give index 0 zero probability.
    """
    count = int(eligible.shape[0])
    if count <= int(query_cap):
        return np.arange(count, dtype=np.int64)
    weights = torch.ones(count, dtype=torch.float64)
    drawn = torch.multinomial(
        weights, int(query_cap), replacement=False, generator=generator
    )
    return np.sort(drawn.numpy().astype(np.int64))


def build_dense_pairs(
    geometry: DenseBatchGeometry,
    *,
    grid_hws: tuple[tuple[int, int], ...],
    patch_hw: tuple[int, int],
    local_crops: int,
    query_cap: int,
    generator: torch.Generator,
) -> tuple[DensePairQueries, ...]:
    """Enumerate ordered pairs and sample correspondence queries per image.

    Pairs are ``teacher global -> other global`` and ``teacher global -> every
    local``; a view never pairs with itself, and weak-only slots never pair.
    """
    _validate_layout(geometry, grid_hws, local_crops=local_crops)
    patch_hw = (int(patch_hw[0]), int(patch_hw[1]))
    ordinary = 2 + int(local_crops)
    rows = len(geometry[0])

    # Support is realized per image: each batch row traces its own augmentation,
    # so the mask is computed once per (view, row) and reused across pairs.
    row_masks: dict[int, list[np.ndarray]] = {
        view: [
            _patch_valid(geometry[view][row], grid_hws[view], patch_hw)
            for row in range(rows)
        ]
        for view in range(ordinary)
    }

    masks = {view: _batch_mask(row_masks[view]) for view in range(ordinary)}
    centers = {view: _grid_centers(grid_hws[view], patch_hw) for view in range(ordinary)}
    pairs: list[DensePairQueries] = []
    for teacher_view in range(2):
        for student_view in range(ordinary):
            if student_view == teacher_view:
                continue
            samples: list[DenseSampleQueries] = []
            for row in range(rows):
                student_xy = centers[student_view]
                teacher_xy = _map_to_view(
                    geometry[student_view][row],
                    geometry[teacher_view][row],
                    student_xy,
                )
                student_ok = row_masks[student_view][row].reshape(-1)
                teacher_ok = _support_mask(
                    teacher_xy,
                    row_masks[teacher_view][row],
                    grid_hws[teacher_view],
                    patch_hw,
                )
                eligible = np.flatnonzero(student_ok & teacher_ok)
                chosen = _sample_indices(eligible, query_cap, generator)
                samples.append(
                    DenseSampleQueries(
                        student_indices=torch.from_numpy(
                            eligible[chosen].astype(np.int64)
                        ),
                        teacher_xy=torch.from_numpy(
                            np.ascontiguousarray(teacher_xy[eligible[chosen]])
                        ),
                        eligible_count=int(eligible.shape[0]),
                    )
                )
            pairs.append(
                DensePairQueries(
                    teacher_view=teacher_view,
                    student_view=student_view,
                    samples=tuple(samples),
                    # The frozen contract is one [B,Hg,Wg] mask per side, built
                    # from each row's own realized augmentation.
                    student_valid=masks[student_view],
                    teacher_valid=masks[teacher_view],
                )
            )
    return tuple(pairs)
