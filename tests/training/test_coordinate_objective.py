"""Coordinate objective: shared numerical guards, reduction and the cosine term."""

from __future__ import annotations

import pytest
import torch

from otuformer.training import dense_geometry as geo
from otuformer.training import dense_objective as objective
from otuformer.vision.features import sample_grid_features


def _pair(student_view, teacher_view, samples, *, grid=(2, 2), batch=1):
    """One ordered pair record with all-valid supports."""
    return geo.DensePairQueries(
        teacher_view=teacher_view,
        student_view=student_view,
        samples=tuple(samples),
        student_valid=torch.ones(batch, *grid, dtype=torch.bool),
        teacher_valid=torch.ones(batch, *grid, dtype=torch.bool),
    )


def _sample(indices, xy, eligible=None):
    return geo.DenseSampleQueries(
        student_indices=torch.tensor(indices, dtype=torch.int64),
        teacher_xy=torch.tensor(xy, dtype=torch.float64).reshape(-1, 2),
        eligible_count=len(indices) if eligible is None else eligible,
    )


def _field(values, *, height=1, width=None):
    """BHWC float32 field from a list of per-cell vectors."""
    width = len(values) if width is None else width
    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, height, width, -1)
    return tensor


def test_coordinate_raw_sample_then_l2():
    """Interpolate the raw vectors first, then normalize; unit-first differs."""
    grid = _field([[2.0, 0.0], [0.0, 1.0]])
    # x = 16 sits exactly between the two horizontal patch centres (width 16).
    xy = torch.tensor([[[16.0, 8.0]]], dtype=torch.float32)
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    sampled, stats = objective.sample_dense_features(
        grid, xy, patch_hw=(16, 16), valid=valid
    )
    expected = torch.tensor([2.0, 1.0]) / 5.0**0.5
    torch.testing.assert_close(sampled[0, 0], expected, rtol=1e-6, atol=1e-6)
    # Unequal magnitudes make this mixture's rho < 1: the interpolation keeps
    # the larger contributor's direction but the raw path still dominates.
    assert stats["min_rho"] == pytest.approx(0.745356, abs=1e-5)


def test_unit_first_interpolation_would_differ():
    """Guard the recipe: unit-first interpolation is a different target."""
    raw = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    unit = raw / raw.norm(dim=-1, keepdim=True)
    raw_mean = 0.5 * raw[0] + 0.5 * raw[1]
    unit_mean = 0.5 * unit[0] + 0.5 * unit[1]
    assert not torch.allclose(
        raw_mean / raw_mean.norm(), unit_mean / unit_mean.norm(), atol=1e-3
    )


def test_coordinate_point_sample_pair_means():
    """Three-level reduction: points, then samples, then ordered pairs."""
    grid_student = {0: torch.ones(1, 2, 2, 1), 1: torch.ones(1, 2, 2, 1)}
    grid_teacher = {0: torch.ones(1, 2, 2, 1), 1: torch.ones(1, 2, 2, 1)}
    # Pair 1: one image with two points.  Pair 2: one image with one point.
    # The two pairs therefore have unequal point counts, and the reduction is
    # a mean of pair means rather than a point-weighted average.
    pairs = (
        _pair(1, 0, (_sample([0, 1], [[8, 8], [24, 8]]),)),
        _pair(0, 1, (_sample([0], [[8, 8]]),)),
    )
    result = objective.coordinate_loss(
        grid_student, grid_teacher, pairs, patch_hw=(16, 16)
    )
    assert result.raw is not None
    # Identical unit features -> zero loss, with the denominators recorded.
    assert float(result.raw) == pytest.approx(0.0, abs=1e-6)
    assert result.statistics["contributing_pairs"] == 2
    assert result.statistics["contributing_samples"] == 2
    assert result.statistics["selected_points"] == 3


def test_reduction_is_not_point_weighted():
    """A pair with many points must not outweigh a pair with one point."""
    per_pair = [[torch.tensor(1.0)], [torch.tensor(0.0), torch.tensor(0.0)]]
    assert float(objective.reduce_dense_losses(per_pair)) == pytest.approx(0.5)

    empty = objective.reduce_dense_losses([[], [torch.tensor(2.0)]])
    assert float(empty) == pytest.approx(2.0)
    assert objective.reduce_dense_losses([[], []]) is None


def test_exact_opposition_is_a_norm_failure():
    """Exact cancellation is caught: the sampled vector has no direction."""
    grid = _field([[1.0, 0.0], [-1.0, 0.0]])
    xy = torch.tensor([[[16.0, 8.0]]], dtype=torch.float32)
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    with pytest.raises(objective.DenseNumericalError, match="norm"):
        objective.sample_dense_features(grid, xy, patch_hw=(16, 16), valid=valid)


def test_relative_cancellation_fires_before_the_norm_floor():
    """Nearly opposed contributors of large magnitude keep a small rho."""
    grid = _field([[1000.0, 0.0], [-1000.0, 0.001]])
    xy = torch.tensor([[[16.0, 8.0]]], dtype=torch.float32)
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    with pytest.raises(objective.DenseNumericalError, match="rho"):
        objective.sample_dense_features(grid, xy, patch_hw=(16, 16), valid=valid)


def test_zero_vector_is_a_norm_failure():
    grid = torch.zeros(1, 1, 1, 2)
    xy = torch.tensor([[[8.0, 8.0]]], dtype=torch.float32)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    with pytest.raises(objective.DenseNumericalError, match="norm"):
        objective.sample_dense_features(grid, xy, patch_hw=(16, 16), valid=valid)


def test_each_positive_weight_teacher_contributor_has_a_valid_norm():
    grid = torch.ones(1, 2, 2, 4)
    grid[0, 0, 1] = 0
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    # The mixture is healthy, but one actual contributor has no direction.
    with pytest.raises(objective.DenseNumericalError, match="contributor.*norm"):
        objective.sample_dense_features(grid, torch.tensor([[[16., 8.]]]),
                              patch_hw=(16, 16), valid=valid)
    # At a patch centre that neighbour has exactly zero weight and is irrelevant.
    result, _ = objective.sample_dense_features(grid, torch.tensor([[[8., 8.]]]),
                                      patch_hw=(16, 16), valid=valid)
    assert torch.isfinite(result).all()


def test_exact_center_has_unit_rho():
    torch.manual_seed(0)
    grid = torch.rand(1, 2, 2, 4)
    xy = torch.tensor([[[8.0, 8.0], [24.0, 24.0]]], dtype=torch.float32)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    sampled, stats = objective.sample_dense_features(
        grid, xy, patch_hw=(16, 16), valid=valid
    )
    assert stats["min_rho"] == pytest.approx(1.0, abs=1e-6)
    raw = grid[0, [0, 1], [0, 1]]
    torch.testing.assert_close(
        sampled[0], raw / raw.norm(dim=-1, keepdim=True), rtol=1e-6, atol=1e-6
    )


def test_incomplete_support_is_unavailable_not_zero_filled():
    grid = torch.ones(1, 2, 2, 3)
    xy = torch.tensor([[[64.0, 8.0]]], dtype=torch.float32)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    with pytest.raises(objective.DenseNumericalError, match="support"):
        objective.sample_dense_features(grid, xy, patch_hw=(16, 16), valid=valid)


def test_coordinate_gradient_and_unavailable():
    """Analytic 1-cos, student-only gradient, and None when nothing overlaps."""
    torch.manual_seed(0)
    student = torch.rand(1, 2, 2, 4, requires_grad=True)
    teacher = torch.rand(1, 2, 2, 4)
    pairs = (_pair(1, 0, (_sample([0, 3], [[8, 8], [24, 24]]),)),)
    result = objective.coordinate_loss(
        {1: student}, {0: teacher}, pairs, patch_hw=(16, 16)
    )
    assert result.raw is not None and result.raw.requires_grad
    result.raw.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None

    no_overlap = (_pair(1, 0, (_sample([], []),)),)
    empty = objective.coordinate_loss(
        {1: student}, {0: teacher}, no_overlap, patch_hw=(16, 16)
    )
    assert empty.raw is None
    assert empty.statistics["contributing_pairs"] == 0


def test_student_normalisation_keeps_the_true_cosine_gradient():
    """The L2 denominator must stay differentiable: 1-cos, not 1-cos-with-|x|."""

    def _gradient(normalise):
        points = torch.tensor([[3.0, 4.0]], requires_grad=True)
        target = torch.tensor([1.0, 0.0])
        unit, _ = normalise(points)
        (1.0 - (unit[0] * target).sum()).backward()
        return points.grad[0]

    got = _gradient(lambda points: objective._unit_rows(points, context="fixture"))
    # d/dx (1 - <x,t>/(|x||t|)) at x=[3,4], t=[1,0]: -t/5 + (3/125)*x.
    torch.testing.assert_close(got, torch.tensor([-0.128, 0.096]), rtol=1e-6, atol=1e-7)


def test_sampled_teacher_normalisation_keeps_the_gradient_too():
    """The sampled-feature path uses the same differentiable denominator."""
    grid = torch.tensor([[[[3.0, 4.0]]]], requires_grad=True)
    xy = torch.tensor([[[8.0, 8.0]]], dtype=torch.float64)
    unit, stats = objective.sample_dense_features(
        grid,
        xy,
        patch_hw=(16, 16),
        valid=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    assert stats["min_rho"] == pytest.approx(1.0)
    (1.0 - (unit[0, 0] * torch.tensor([1.0, 0.0])).sum()).backward()
    torch.testing.assert_close(
        grid.grad[0, 0, 0], torch.tensor([-0.128, 0.096]), rtol=1e-6, atol=1e-7
    )


def test_coordinate_loss_converts_float64_metadata_before_transfer(monkeypatch):
    """float64 geometry metadata must be narrowed on CPU, never on the device."""
    seen: dict[str, object] = {}
    real = objective.sample_dense_features

    def spy(grid, xy, *, patch_hw, valid, context=""):
        seen["dtype"] = xy.dtype
        seen["device_type"] = xy.device.type
        return real(grid, xy, patch_hw=patch_hw, valid=valid, context=context)

    monkeypatch.setattr(objective, "sample_dense_features", spy)
    torch.manual_seed(3)
    student = torch.rand(1, 2, 2, 4, requires_grad=True)
    teacher = torch.rand(1, 2, 2, 4)
    pairs = (_pair(1, 0, (_sample([0], [[16.0, 8.0]]),)),)
    objective.coordinate_loss({1: student}, {0: teacher}, pairs, patch_hw=(16, 16))
    assert seen["dtype"] is torch.float32
    assert seen["device_type"] == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="no MPS device to prove float64 metadata never reaches it",
)
def test_coordinate_loss_runs_on_a_device_that_rejects_float64():
    """Regression: the CPU float64 coordinates must be narrowed before transfer."""
    torch.manual_seed(5)
    student = (torch.rand(1, 2, 2, 4) + 0.5).to("mps").requires_grad_(True)
    teacher = (torch.rand(1, 2, 2, 4) + 0.5).to("mps")
    pairs = (_pair(1, 0, (_sample([0], [[16.0, 8.0]]),)),)
    result = objective.coordinate_loss(
        {1: student}, {0: teacher}, pairs, patch_hw=(16, 16)
    )
    assert result.raw is not None and torch.isfinite(result.raw).all()
    result.raw.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_one_point_is_usable():
    student = torch.ones(1, 2, 2, 3, requires_grad=True)
    teacher = torch.ones(1, 2, 2, 3)
    pairs = (_pair(1, 0, (_sample([2], [[24, 8]]),)),)
    result = objective.coordinate_loss(
        {1: student}, {0: teacher}, pairs, patch_hw=(16, 16)
    )
    assert result.raw is not None
    assert result.statistics["selected_points"] == 1


def test_sampling_matches_existing_sampler():
    """The guard wrapper must not fork the coordinate convention."""
    torch.manual_seed(2)
    grid = torch.rand(1, 2, 2, 5)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    xy = torch.tensor([[[8.0, 8.0], [24.0, 24.0], [24.0, 8.0]]], dtype=torch.float32)
    got, _ = objective.sample_dense_features(grid, xy, patch_hw=(16, 16), valid=valid)
    reference, usable = sample_grid_features(grid, xy, patch_hw=(16, 16), valid=valid)
    assert bool(usable.all())
    unit = reference / reference.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(got, unit, rtol=0, atol=1e-6)


def test_sampling_device_matches_cpu():
    """Supported devices reproduce CPU sampling within float tolerance."""
    if not torch.backends.mps.is_available():
        pytest.skip("no supported accelerator on this host")
    torch.manual_seed(4)
    grid = torch.rand(1, 2, 2, 4)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    xy = torch.tensor([[[8.0, 8.0], [24.0, 24.0], [16.0, 8.0]]], dtype=torch.float32)
    cpu, _ = objective.sample_dense_features(grid, xy, patch_hw=(16, 16), valid=valid)
    moved, _ = objective.sample_dense_features(
        grid.to("mps"), xy.to("mps"), patch_hw=(16, 16), valid=valid.to("mps")
    )
    torch.testing.assert_close(moved.cpu(), cpu, rtol=1e-5, atol=1e-6)


def test_numeric_error_carries_context():
    """A failure names the pair and row so the run can be diagnosed."""
    torch.manual_seed(0)
    student = torch.rand(1, 2, 2, 4, requires_grad=True)
    teacher = torch.zeros(1, 2, 2, 4)  # zero teacher -> norm failure
    pairs = (_pair(1, 0, (_sample([0], [[8, 8]]),)),)
    with pytest.raises(objective.DenseNumericalError, match="pair 0->1 row 0"):
        objective.coordinate_loss(
            {1: student}, {0: teacher}, pairs, patch_hw=(16, 16)
        )
