"""Coordinate objective and the shared dense numerical contract.

Two responsibilities, one owner:

* :func:`sample_dense_features` wraps the existing complete-support sampler with
  the numeric guards (absolute feature norm and relative interpolation
  cancellation). It does not define a second sampling convention.
* :func:`coordinate_loss` and :func:`reduce_dense_losses` own the exact
  ``1 - cos`` objective and the three-level point/sample/pair reduction.

Developer contract: the teacher side is detached, the student gradient is the
only one that flows, reductions stay float32, and unavailable geometry is
``None`` rather than a measured zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from otuformer.training.dense_geometry import DensePairQueries
from otuformer.vision.features import sample_grid_features

#: Provisional author thresholds (design §5). Changing any of them changes the
#: recipe identity; qualification precedes freezing, never widens a guard.
FEATURE_NORM_FLOOR = 1e-6
CANCELLATION_FLOOR = 1e-6
LOW_RHO_REVIEW_THRESHOLD = 0.1


class DenseNumericalError(RuntimeError):
    """A dense feature violates the frozen numeric contract."""


@dataclass
class DenseLossResult:
    """One objective's contribution plus the statistics that justify it."""

    raw: torch.Tensor | None
    statistics: dict[str, float | int | None] = field(default_factory=dict)


def _bilinear_contributors(
    xy: torch.Tensor, patch_hw: tuple[int, int], grid_hw: tuple[int, int]
):
    """Return the four contributor indices and weights for each query point."""
    patch_h, patch_w = int(patch_hw[0]), int(patch_hw[1])
    u = xy[..., 0] / float(patch_w) - 0.5
    v = xy[..., 1] / float(patch_h) - 0.5
    col0 = torch.floor(u)
    row0 = torch.floor(v)
    w_col1 = (u - col0).clamp(0.0, 1.0)
    w_row1 = (v - row0).clamp(0.0, 1.0)
    return row0.to(torch.int64), w_row1, col0.to(torch.int64), w_col1


def _cancellation_ratio(
    grid: torch.Tensor,
    xy: torch.Tensor,
    *,
    patch_hw: tuple[int, int],
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cancellation ratio and participating contributor norm minimum.

    ``rho = |sum_i w_i f_i| / sum_i w_i |f_i|`` for the contributing set.

    ``rho`` is 1 for an exact-centre sample and approaches 0 when the linear
    mixture cancels. It is a numerical diagnostic only: it is not a
    representation-quality or anti-collapse test.
    """
    batch, grid_h, grid_w, _ = grid.shape
    rows0, w_rows1, cols0, w_cols1 = _bilinear_contributors(
        xy, patch_hw, (grid_h, grid_w)
    )
    channels = int(grid.shape[-1])
    numerator = torch.zeros(
        batch, xy.shape[1], channels, dtype=grid.dtype, device=grid.device
    )
    denominator = torch.zeros(batch, xy.shape[1], dtype=grid.dtype, device=grid.device)
    contributor_minima = []
    for rows, row_weights in ((rows0, 1.0 - w_rows1), (rows0 + 1, w_rows1)):
        for cols, col_weights in ((cols0, 1.0 - w_cols1), (cols0 + 1, w_cols1)):
            weight = (row_weights * col_weights).to(grid.dtype)
            inside = (rows >= 0) & (rows < grid_h) & (cols >= 0) & (cols < grid_w)
            safe_rows = rows.clamp(0, grid_h - 1)
            safe_cols = cols.clamp(0, grid_w - 1)
            neighbour = grid[
                torch.arange(batch, device=grid.device).unsqueeze(1),
                safe_rows,
                safe_cols,
            ]
            neighbour = torch.where(
                inside.unsqueeze(-1), neighbour, torch.zeros_like(neighbour)
            )
            numerator += weight.unsqueeze(-1) * neighbour
            neighbour_norms = neighbour.norm(dim=-1)
            denominator += weight * neighbour_norms
            contributor_minima.append(torch.where(
                inside & (weight > 0), neighbour_norms, torch.inf
            ).min())
    rho = numerator.norm(dim=-1) / denominator.clamp_min(torch.finfo(grid.dtype).tiny)
    return rho, torch.stack(contributor_minima).min()


def sample_dense_features(
    grid: torch.Tensor,
    xy: torch.Tensor,
    *,
    patch_hw: tuple[int, int],
    valid: torch.Tensor,
    context: str = "",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Complete-support sample plus norm/rho guards, then channel L2.

    ``grid`` is BHWC, ``xy`` is ``[B, Q, 2]`` and ``valid`` is the Boolean patch
    support of that grid. Both guards raise instead of dropping a point: a
    numerical failure is not missing geometry, and silently shrinking the
    denominator would make the loss incomparable between steps.
    """
    if grid.dtype is not torch.float32:
        grid = grid.float()
    if xy.dtype is not torch.float32:
        if xy.device.type != "cpu":
            raise DenseNumericalError(
                "query coordinates must be built in float64 on CPU and converted "
                "to float32 there before transfer"
                f"{' (' + context + ')' if context else ''}; this device does not "
                "support float64 metadata transfer."
            )
        xy = xy.float()
    suffix = f" ({context})" if context else ""

    sampled, usable = sample_grid_features(grid, xy, patch_hw=patch_hw, valid=valid)
    # Keep the real normalization in autograd; diagnostics need no graph.
    norms = sampled.norm(dim=-1)
    with torch.no_grad():
        rho, contributor_min = _cancellation_ratio(grid, xy, patch_hw=patch_hw, valid=valid)
        checks = torch.stack((
            usable.all(), torch.isfinite(sampled).all(), torch.isfinite(xy).all(),
            torch.isfinite(norms).all(), norms.min(),
            torch.isfinite(rho).all(), rho.min(), contributor_min,
        )).cpu().tolist()
    supported, finite, finite_xy, finite_norm, min_norm, finite_rho, min_rho, min_contributor = checks
    if not supported:
        raise DenseNumericalError(
            f"{int((~usable).sum())} query point(s) lack complete support{suffix}."
        )
    if not finite:
        raise DenseNumericalError(f"sampled features are not finite{suffix}.")
    if not finite_xy:
        raise DenseNumericalError(f"query coordinates are not finite{suffix}.")
    if not np.isfinite(min_contributor) or min_contributor <= FEATURE_NORM_FLOOR:
        raise DenseNumericalError(
            f"teacher contributor feature norm {min_contributor:.3e} does not exceed "
            f"{FEATURE_NORM_FLOOR:.1e}{suffix}."
        )
    if not finite_norm or min_norm <= FEATURE_NORM_FLOOR:
        raise DenseNumericalError(
            f"sampled feature norm {min_norm:.3e} does not exceed "
            f"{FEATURE_NORM_FLOOR:.1e}{suffix}."
        )
    if not finite_rho or min_rho <= CANCELLATION_FLOOR:
        raise DenseNumericalError(
            f"interpolation cancellation rho {min_rho:.3e} does not exceed "
            f"{CANCELLATION_FLOOR:.1e}{suffix}; the interpolated direction is not a "
            "meaningful target."
        )
    return sampled / norms.unsqueeze(-1), {
        "min_norm": min(min_norm, min_contributor), "min_rho": min_rho,
        "queries": int(xy.shape[1]),
    }

def _gather_centers(grid: torch.Tensor, indices: torch.Tensor, grid_hw) -> torch.Tensor:
    """Read exact patch centers from a BHWC grid for one image's flat indices."""
    if indices.numel() == 0:
        return grid.new_zeros((0, grid.shape[-1]))
    grid_w = int(grid_hw[1])
    rows = torch.div(indices, grid_w, rounding_mode="floor")
    cols = indices - rows * grid_w
    return grid[0, rows, cols]


def reduce_dense_losses(per_pair: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor | None:
    """Mean of per-sample point means, then mean over contributing pairs."""
    pair_means: list[torch.Tensor] = []
    for samples in per_pair:
        available = [s for s in samples if s is not None]
        if not available:
            continue
        pair_means.append(torch.stack(available).mean())
    if not pair_means:
        return None
    return torch.stack(pair_means).mean()


def _pair_points(
    student_grid: torch.Tensor,
    teacher_grid: torch.Tensor,
    pair: DensePairQueries,
    *,
    patch_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, list[int], dict[str, float]] | None:
    """Gather student rows and sample normalized teacher targets once per pair."""
    if len(pair.samples) != int(student_grid.shape[0]):
        raise ValueError(
            f"pair {pair.teacher_view}->{pair.student_view} has "
            f"{len(pair.samples)} sample record(s) for "
            f"{int(student_grid.shape[0])} batch row(s)."
        )
    rows = [
        row for row, sample in enumerate(pair.samples)
        if sample.student_indices.numel()
    ]
    lengths = [int(pair.samples[row].student_indices.numel()) for row in rows]
    if not rows:
        return None
    context = f"pair {pair.teacher_view}->{pair.student_view}"
    row_index = torch.tensor(rows)
    indices = torch.cat([pair.samples[row].student_indices for row in rows]).to(
        student_grid.device
    )
    point_rows = torch.repeat_interleave(row_index, torch.tensor(lengths)).to(
        student_grid.device
    )
    points = student_grid.reshape(
        student_grid.shape[0], -1, student_grid.shape[-1]
    )[point_rows, indices]

    # Pad with an existing real query, not an artificial zero/unsupported point.
    # It preserves guard minima; only real queries enter pools and reductions.
    width = max(lengths)
    coordinates = []
    for row, length in zip(rows, lengths):
        xy = pair.samples[row].teacher_xy.float()
        coordinates.append(torch.cat((xy, xy[:1].expand(width - length, -1))))
    xy = torch.stack(coordinates).to(teacher_grid.device)
    teacher_rows = row_index.to(teacher_grid.device)
    try:
        target, teacher_stats = sample_dense_features(
            teacher_grid.detach().index_select(0, teacher_rows), xy, patch_hw=patch_hw,
            valid=pair.teacher_valid.to(teacher_grid.device).index_select(0, teacher_rows),
            context=context,
        )
    except DenseNumericalError:
        # The exceptional path identifies the original row without serializing healthy batches.
        for row in rows:
            sample_dense_features(
                teacher_grid[row:row + 1].detach(),
                pair.samples[row].teacher_xy.float().unsqueeze(0).to(teacher_grid.device),
                patch_hw=patch_hw,
                valid=pair.teacher_valid[row:row + 1].to(teacher_grid.device),
                context=f"{context} row {row}",
            )
        raise
    selected = torch.cat([
        torch.arange(length) + index * width
        for index, length in enumerate(lengths)
    ]).to(target.device)
    target = target.reshape(-1, target.shape[-1]).index_select(0, selected)
    return points, target.detach(), lengths, teacher_stats


BRANCH_COUNT_FIELDS = (
    "eligible_points", "selected_points", "attempted_samples", "contributing_samples",
    "skipped_samples", "attempted_pairs", "contributing_pairs", "skipped_pairs",
)


def _branch_counts(pairs) -> dict[str, int]:
    counts = {f"{branch}_{field}": 0 for branch in ("global", "local")
              for field in BRANCH_COUNT_FIELDS}
    for pair in pairs:
        branch = "global" if pair.student_view < 2 else "local"
        counts[f"{branch}_eligible_points"] += sum(s.eligible_count for s in pair.samples)
        counts[f"{branch}_attempted_samples"] += len(pair.samples)
        counts[f"{branch}_attempted_pairs"] += 1
    return counts


def _contribute_branch(counts: dict[str, int], pair, lengths) -> None:
    branch = "global" if pair.student_view < 2 else "local"
    counts[f"{branch}_selected_points"] += sum(lengths)
    counts[f"{branch}_contributing_samples"] += len(lengths)
    counts[f"{branch}_contributing_pairs"] += 1


def _finish_branch_counts(counts: dict[str, int]) -> dict[str, int]:
    for branch in ("global", "local"):
        for unit in ("samples", "pairs"):
            counts[f"{branch}_skipped_{unit}"] = (
                counts[f"{branch}_attempted_{unit}"] - counts[f"{branch}_contributing_{unit}"]
            )
    return counts


def coordinate_loss(
    student_grids: dict[int, torch.Tensor],
    teacher_grids: dict[int, torch.Tensor],
    pairs: tuple[DensePairQueries, ...],
    *,
    patch_hw: tuple[int, int],
) -> DenseLossResult:
    """``mean(1 - cos)`` over correspondences, reduced point/sample/pair.

    Student features are read at exact patch centers; teacher features are
    bilinearly sampled at the mapped points and normalized afterwards, so
    contributor magnitudes shape the target. The teacher is detached.
    """
    per_pair: list[list[torch.Tensor]] = []
    branch_counts = _branch_counts(pairs)
    min_norm: float | None = None
    min_rho: float | None = None
    selected = 0
    eligible = 0
    contributing_samples = 0
    contributing_pairs = 0

    for pair in pairs:
        student_grid = student_grids[pair.student_view]
        teacher_grid = teacher_grids[pair.teacher_view]
        points = _pair_points(student_grid, teacher_grid, pair, patch_hw=patch_hw)
        eligible += sum(int(sample.eligible_count) for sample in pair.samples)
        if points is None:
            per_pair.append([])
            continue
        student_points, teacher_unit, lengths, stats = points
        context = f"pair {pair.teacher_view}->{pair.student_view}"
        try:
            student_unit, student_stats = _unit_rows(student_points, context=context)
        except DenseNumericalError:
            for row, sample in enumerate(pair.samples):
                if sample.student_indices.numel():
                    _unit_rows(
                        _gather_centers(
                            student_grid[row:row + 1], sample.student_indices.to(student_grid.device),
                            student_grid.shape[1:3],
                        ), context=f"{context} row {row}",
                    )
            raise
        value = min(stats["min_norm"], student_stats["min_norm"])
        min_norm = value if min_norm is None else min(min_norm, value)
        value = stats["min_rho"]
        min_rho = value if min_rho is None else min(min_rho, value)
        losses = 1.0 - (student_unit * teacher_unit).sum(-1)
        per_pair.append([row.mean() for row in losses.split(lengths)])
        selected += sum(lengths)
        contributing_samples += len(lengths)
        contributing_pairs += 1
        _contribute_branch(branch_counts, pair, lengths)

    raw = reduce_dense_losses(per_pair)
    return DenseLossResult(
        raw=raw,
        statistics={
            "selected_points": selected,
            "eligible_points": eligible,
            "contributing_samples": contributing_samples,
            "contributing_pairs": contributing_pairs,
            **_finish_branch_counts(branch_counts),
            # ``None`` means the population is unavailable, not a measured 0/1.
            "min_norm": min_norm,
            "min_rho": min_rho,
        },
    )


def _unit_rows(
    points: torch.Tensor, *, context: str
) -> tuple[torch.Tensor, dict[str, float]]:
    """Channel-L2 normalize exact-centre student rows with the same guards.

    The norm stays attached to the autograd graph so the loss is the designed
    ``1 - cos``; a detached denominator would scale the gradient by ``1/||x||``
    and drop the radial term.
    """
    norms = points.norm(dim=-1)
    diagnostic_norms = norms.detach()
    finite, minimum = torch.stack((
        torch.isfinite(diagnostic_norms).all(), diagnostic_norms.min(),
    )).cpu().tolist()
    if not finite or minimum <= FEATURE_NORM_FLOOR:
        raise DenseNumericalError(
            f"student feature norm {minimum:.3e} does not "
            f"exceed {FEATURE_NORM_FLOOR:.1e} ({context})."
        )
    return points / norms.unsqueeze(-1), {"min_norm": minimum}


def as_numpy_points(xy: torch.Tensor) -> np.ndarray:
    """CPU float64 view of query coordinates, for geometry logging only."""
    return np.asarray(xy.detach().to("cpu", torch.float64))
