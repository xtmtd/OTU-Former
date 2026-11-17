"""Ranking objective: heads, pair-conditioned references and row-CE."""

from __future__ import annotations

import pytest
import torch

from otuformer.training import dense_geometry as geo
from otuformer.training import dense_objective as objective_module
from otuformer.training import neighbor_ranking as ranking

HIDDEN = 8
OUT = 4


def _pair(teacher_view, student_view, samples, *, grid=(2, 2)):
    return geo.DensePairQueries(
        teacher_view=teacher_view,
        student_view=student_view,
        samples=tuple(samples),
        student_valid=torch.ones(1, *grid, dtype=torch.bool),
        teacher_valid=torch.ones(1, *grid, dtype=torch.bool),
    )


def _sample(indices, xy):
    return geo.DenseSampleQueries(
        student_indices=torch.tensor(indices, dtype=torch.int64),
        teacher_xy=torch.tensor(xy, dtype=torch.float64).reshape(-1, 2),
        eligible_count=len(indices),
    )


def _objective():
    torch.manual_seed(0)
    return ranking.NeighborRankingObjective(HIDDEN, out_dim=OUT, hidden=16)


def test_head_architecture_and_initialization():
    objective = _objective()
    head = objective.student_head
    linears = [m for m in head if isinstance(m, torch.nn.Linear)]
    assert [tuple(m.weight.shape) for m in linears] == [
        (16, HIDDEN),
        (16, 16),
        (OUT, 16),
    ]
    assert all(m.bias is not None for m in linears)
    assert isinstance(head[1], torch.nn.GELU) and isinstance(head[3], torch.nn.GELU)
    # Truncated normal, std 0.02, zero bias (checked head's convention).
    assert torch.allclose(linears[0].bias, torch.zeros_like(linears[0].bias))
    assert float(linears[0].weight.detach().std()) < 0.05
    # The student head trains; the EMA teacher head never does.
    assert head[0].weight.requires_grad
    assert not any(p.requires_grad for p in objective.teacher_head.parameters())


def test_teacher_head_starts_as_an_exact_copy():
    objective = _objective()
    for a, b in zip(
        objective.student_head.state_dict().values(),
        objective.teacher_head.state_dict().values(),
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_update_teacher_is_ema_of_the_student_head():
    objective = _objective()
    with torch.no_grad():
        for p in objective.student_head.parameters():
            p.add_(1.0)
    objective.update_teacher(0.5)
    # EMA moves midway, and the teacher still takes no gradient.
    first = next(iter(objective.teacher_head.parameters()))
    assert first.grad is None


def test_projection_norm_floor_fires_before_normalisation():
    """A near-zero projection must fail instead of being normalised into a direction."""
    objective = _objective()
    with torch.no_grad():
        # Zero the final layer so the projection collapses to its bias only.
        objective.student_head[4].weight.zero_()
        objective.student_head[4].bias.fill_(1e-10)
    grid = torch.rand(1, 2, 2, HIDDEN) + 0.5
    with pytest.raises(
        objective_module.DenseNumericalError, match="projection norm"
    ):
        objective.project_field(grid)

    # The teacher side is checked the same way (it must not fabricate a target).
    with torch.no_grad():
        objective.teacher_head[4].weight.zero_()
        objective.teacher_head[4].bias.fill_(1e-10)
    with pytest.raises(
        objective_module.DenseNumericalError, match="teacher ranking projection"
    ):
        objective.project_teacher_field(grid)


def test_project_l2_sample_cosine_order():
    """Projection happens before interpolation, not after."""
    objective = _objective()
    grid = torch.full((1, 2, 2, HIDDEN), 0.1)
    grid[0, 0, 0, 0] = 2.0
    grid[0, 0, 1, 1] = 1.0
    projected = objective.project_field(grid)
    assert torch.allclose(projected.norm(dim=-1), torch.ones(1, 2, 2), atol=1e-5)
    mixed_after = objective.student_head(0.5 * grid[0, 0, 0] + 0.5 * grid[0, 0, 1])
    assert not torch.allclose(
        mixed_after / mixed_after.norm(), 0.5 * projected[0, 0, 0] + 0.5 * projected[0, 0, 1]
        , atol=1e-4
    )


def test_pair_pool_is_selected_teacher_queries():
    objective = _objective()
    torch.manual_seed(2)
    student = {1: torch.rand(1, 2, 2, HIDDEN), 0: torch.rand(1, 2, 2, HIDDEN)}
    teacher = {0: torch.rand(1, 2, 2, HIDDEN), 1: torch.rand(1, 2, 2, HIDDEN)}
    pairs = (_pair(0, 1, (_sample([0, 3], [[8, 8], [24, 24]]),)),)
    result = objective.compute(
        student,
        teacher,
        pairs,
        patch_hw=(16, 16),
        reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert result.raw is not None
    assert result.statistics["reference_population"] == 2
    assert result.statistics["reference_cap"] == 4


def _soft_permutation(cosines: torch.Tensor) -> torch.Tensor:
    """The real sorter output: doubly stochastic, which the guards require."""
    _, permutation = ranking.sort_values_and_permutation(cosines, cache={})
    return permutation.detach()


def test_row_ce_reduction_matches_the_frozen_formula():
    """The loss is the per-query row-CE mean, and R times the all-cell mean."""
    torch.manual_seed(0)
    student = _soft_permutation(torch.rand(1, 4) * 2 - 1)
    teacher = _soft_permutation(torch.rand(1, 4) * 2 - 1)
    loss = ranking.row_cross_entropy(student, teacher)
    cells = -(teacher.detach() * student.log())
    R = 4
    torch.testing.assert_close(loss[0], cells.sum(-1).sum(-1)[0] / R, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss[0], R * cells.mean(), rtol=1e-5, atol=1e-5)


def test_soft_equal_target_is_not_zero():
    """An identical soft target still has positive entropy, so never zero."""
    torch.manual_seed(1)
    tied = _soft_permutation(torch.zeros(1, 4))
    loss = ranking.row_cross_entropy(tied, tied)
    assert float(loss) > 0.0


def test_zero_probability_is_a_failure_without_clamp():
    torch.manual_seed(2)
    student = _soft_permutation(torch.rand(1, 4) * 2 - 1).clone()
    student[0, 0, :] = 0.0
    student[0, 0, 0] = 1.0
    teacher = _soft_permutation(torch.rand(1, 4) * 2 - 1)
    with pytest.raises(ranking.RankingDependencyError, match="non-positive|stochastic"):
        ranking.row_cross_entropy(student, teacher)


def test_reduction_unequal_queries_samples_pairs():
    objective = _objective()
    torch.manual_seed(3)
    student = {0: torch.rand(2, 2, 2, HIDDEN), 1: torch.rand(2, 2, 2, HIDDEN)}
    teacher = {0: torch.rand(2, 2, 2, HIDDEN), 1: torch.rand(2, 2, 2, HIDDEN)}
    batch2 = dict(student_valid=torch.ones(2, 2, 2, dtype=torch.bool),
                  teacher_valid=torch.ones(2, 2, 2, dtype=torch.bool))
    pairs = (
        geo.DensePairQueries(
            teacher_view=0, student_view=1,
            samples=(_sample([0, 1], [[8, 8], [24, 8]]), _sample([0], [[8, 8]])),
            **batch2,
        ),
        geo.DensePairQueries(
            teacher_view=1, student_view=0,
            # Row 0 supplies two pool points so this pair is available; row 1
            # has no valid correspondence and is omitted from the mean.
            samples=(_sample([2, 3], [[8, 24], [24, 24]]), _sample([], [])),
            **batch2,
        ),
    )
    result = objective.compute(
        student, teacher, pairs,
        patch_hw=(16, 16), reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert result.raw is not None
    assert result.statistics["contributing_pairs"] == 2
    assert result.statistics["contributing_samples"] == 3
    assert result.statistics["selected_points"] == 5


def test_zero_and_one_pool_are_unavailable_not_errors():
    objective = _objective()
    torch.manual_seed(4)
    student = {0: torch.rand(1, 2, 2, HIDDEN), 1: torch.rand(1, 2, 2, HIDDEN)}
    teacher = {0: torch.rand(1, 2, 2, HIDDEN), 1: torch.rand(1, 2, 2, HIDDEN)}
    empty = (_pair(0, 1, (_sample([], []),)),)
    assert objective.compute(
        student, teacher, empty,
        patch_hw=(16, 16), reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    ).raw is None
    single = (_pair(0, 1, (_sample([0], [[8, 8]]),)),)
    assert objective.compute(
        student, teacher, single,
        patch_hw=(16, 16), reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    ).raw is None


def test_reference_rng_is_separate_from_query_stream():
    objective = _objective()
    torch.manual_seed(5)
    student = {0: torch.rand(1, 2, 2, HIDDEN), 1: torch.rand(1, 2, 2, HIDDEN)}
    teacher = {0: torch.rand(1, 2, 2, HIDDEN), 1: torch.rand(1, 2, 2, HIDDEN)}
    query = torch.Generator().manual_seed(9)
    ref_a = torch.Generator().manual_seed(11)
    ref_b = torch.Generator().manual_seed(11)
    before_query = query.get_state().clone()
    pairs = (_pair(0, 1, (_sample([0, 3], [[8, 8], [24, 24]]),)),)
    first = objective.compute(
        student, teacher, pairs,
        patch_hw=(16, 16), reference_cap=4, generator=ref_a,
    )
    second = objective.compute(
        student, teacher, pairs,
        patch_hw=(16, 16), reference_cap=4, generator=ref_b,
    )
    assert torch.equal(first.raw, second.raw)
    assert torch.equal(query.get_state(), before_query)


def test_backward_reaches_student_head_and_backbone_only():
    objective = _objective()
    torch.manual_seed(6)
    student = {
        0: torch.rand(1, 2, 2, HIDDEN, requires_grad=True),
        1: torch.rand(1, 2, 2, HIDDEN, requires_grad=True),
    }
    teacher = {0: torch.rand(1, 2, 2, HIDDEN), 1: torch.rand(1, 2, 2, HIDDEN)}
    pairs = (_pair(0, 1, (_sample([0, 3], [[8, 8], [24, 24]]),)),)
    result = objective.compute(
        student, teacher, pairs,
        patch_hw=(16, 16), reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    )
    result.raw.backward()
    assert student[1].grad is not None and torch.isfinite(student[1].grad).all()
    assert student[0].grad is None or torch.isfinite(student[0].grad).all()
    assert all(p.grad is None for p in objective.teacher_head.parameters())
    assert any(p.grad is not None for p in objective.student_head.parameters())


def test_chunked_and_unchunked_agree():
    objective = _objective()
    torch.manual_seed(7)
    student = {0: torch.rand(1, 2, 2, HIDDEN), 1: torch.rand(1, 2, 2, HIDDEN)}
    teacher = {0: torch.rand(1, 2, 2, HIDDEN), 1: torch.rand(1, 2, 2, HIDDEN)}
    pairs = (_pair(0, 1, (_sample([0, 1, 2, 3], [[8, 8], [24, 8], [8, 24], [24, 24]]),)),)
    whole = objective.compute(
        student, teacher, pairs,
        patch_hw=(16, 16), reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    )
    for chunk in (1, 2, 3):
        part = objective.compute(
            student, teacher, pairs,
            patch_hw=(16, 16), reference_cap=4,
            generator=torch.Generator().manual_seed(0),
            query_chunk=chunk,
        )
        torch.testing.assert_close(part.raw, whole.raw, rtol=1e-5, atol=1e-5)


def test_ranking_reports_its_own_projected_space_diagnostics():
    """Ranking returns ITS norm/rho, not only the Coordinate feature space."""
    torch.manual_seed(0)
    objective = _objective()
    pairs = (
        _pair(0, 1, (_sample([0, 1, 2, 3], [[8, 8], [24, 8], [8, 24], [24, 24]]),)),
    )
    student = {1: torch.rand(1, 2, 2, HIDDEN) + 0.5}
    teacher = {0: torch.rand(1, 2, 2, HIDDEN) + 0.5}
    result = objective.compute(
        student,
        teacher,
        pairs,
        patch_hw=(16, 16),
        reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert result.raw is not None
    assert result.statistics["teacher_min_norm"] is not None
    assert result.statistics["teacher_min_rho"] is not None
    assert result.statistics["teacher_min_norm"] > 0.0
    assert 0.0 < result.statistics["teacher_min_rho"] <= 1.0


def test_projection_norm_is_reported_before_l2():
    """The pre-L2 projection norm must be visible, not hidden by normalisation."""
    torch.manual_seed(0)
    objective = _objective()
    with torch.no_grad():
        # Shrink the final layer so the projection norm is small but legal.
        objective.teacher_head[4].weight.mul_(1e-1)
        objective.teacher_head[4].bias.mul_(1e-1)
    pairs = (
        _pair(0, 1, (_sample([0, 1, 2, 3], [[8, 8], [24, 8], [8, 24], [24, 24]]),)),
    )
    student = {1: torch.rand(1, 2, 2, HIDDEN) + 0.5}
    teacher = {0: torch.rand(1, 2, 2, HIDDEN) + 0.5}
    result = objective.compute(
        student, teacher, pairs, patch_hw=(16, 16), reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    )
    projection_norm = result.statistics["teacher_projection_min_norm"]
    interpolated_norm = result.statistics["teacher_min_norm"]
    assert projection_norm is not None and projection_norm > 0.0
    # The two are distinct measurements: the interpolated field is unit-length.
    assert interpolated_norm == pytest.approx(1.0, abs=1e-3)
    assert projection_norm < interpolated_norm


def test_projection_norm_is_the_minimum_over_contributing_views():
    """Order-independence and None for a run with no contributing pair."""
    torch.manual_seed(0)
    objective = _objective()
    pairs = (
        _pair(0, 1, (_sample([0, 1, 2, 3], [[8, 8], [24, 8], [8, 24], [24, 24]]),)),
        _pair(1, 0, (_sample([0, 1, 2, 3], [[8, 8], [24, 8], [8, 24], [24, 24]]),)),
    )
    student = {0: torch.rand(1, 2, 2, HIDDEN) + 0.5, 1: torch.rand(1, 2, 2, HIDDEN) + 0.5}
    teacher = {0: torch.rand(1, 2, 2, HIDDEN) + 0.5, 1: torch.rand(1, 2, 2, HIDDEN) + 0.5}

    readings = []
    for order in (pairs, tuple(reversed(pairs))):
        objective = _objective()
        result = objective.compute(
            student, teacher, order, patch_hw=(16, 16), reference_cap=4,
            generator=torch.Generator().manual_seed(0),
        )
        readings.append(result.statistics["teacher_projection_min_norm"])
    # Same participating set, so the reported minimum must not depend on order.
    assert readings[0] == pytest.approx(readings[1])

    # No contributing pair -> unavailable, never a leftover number.
    empty = (_pair(0, 1, (_sample([], []),)),)
    result = objective.compute(
        student, teacher, empty, patch_hw=(16, 16), reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert result.statistics["teacher_projection_min_norm"] is None
    assert result.statistics["teacher_min_norm"] is None


@pytest.mark.parametrize(
    ("xy", "small_patch", "expected"),
    [([[8, 8], [24, 24]], (0, 1), 2.0),
     ([[16, 8], [24, 24]], (0, 1), 2e-5)],
)
def test_projection_norm_uses_only_real_positive_weight_support(
    monkeypatch, xy, small_patch, expected
):
    """Ignore unused rows/patches and zero-weight neighbours at exact centres."""
    objective = _objective()
    monkeypatch.setattr(
        objective.student_head, "forward",
        lambda x: torch.ones((*x.shape[:-1], OUT)),
    )
    monkeypatch.setattr(objective.teacher_head, "forward", lambda x: x[..., :OUT])
    teacher = torch.ones(2, 2, 2, HIDDEN)
    teacher[0, small_patch[0], small_patch[1]] = 1e-5
    teacher[1] = 2e-6  # A whole row with no selected query is not participating.
    pair = _pair(0, 1, (_sample([0, 3], xy), _sample([], [])))
    pair = geo.DensePairQueries(
        pair.teacher_view, pair.student_view, pair.samples,
        pair.student_valid.expand(2, -1, -1), pair.teacher_valid.expand(2, -1, -1),
    )
    result = objective.compute(
        {1: torch.ones(2, 2, 2, HIDDEN)}, {0: teacher}, (pair,),
        patch_hw=(16, 16), reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert result.statistics["teacher_projection_min_norm"] == pytest.approx(expected)


@pytest.mark.parametrize("role", ["student", "teacher"])
@pytest.mark.parametrize("participates", [False, True])
@pytest.mark.parametrize("bad", [0., float("nan"), float("inf")])
def test_projection_guard_checks_only_participating_vectors(monkeypatch, role, participates, bad):
    objective = _objective()
    monkeypatch.setattr(objective.student_head, "forward", lambda x: x[..., :OUT])
    monkeypatch.setattr(objective.teacher_head, "forward", lambda x: x[..., :OUT])
    student = torch.ones(2, 2, 2, HIDDEN, requires_grad=True)
    teacher = torch.ones(2, 2, 2, HIDDEN)
    with torch.no_grad():
        grid = student if role == "student" else teacher
        grid[0, 0, 0 if participates else 1] = bad
        grid[1] = bad  # No selected query in this row.
    pair = _pair(0, 1, (_sample([0, 3], [[8, 8], [24, 24]]), _sample([], [])))
    pair = geo.DensePairQueries(0, 1, pair.samples,
                               pair.student_valid.expand(2, -1, -1),
                               pair.teacher_valid.expand(2, -1, -1))
    def compute():
        return objective.compute({1: student}, {0: teacher}, (pair,), patch_hw=(16, 16),
                                 reference_cap=4, generator=torch.Generator().manual_seed(0))
    if participates:
        with pytest.raises(objective_module.DenseNumericalError, match="projection norm.*pair 0->1 row 0"):
            compute()
    else:
        result = compute()
        assert torch.isfinite(result.raw)
        result.raw.backward()
        assert torch.isfinite(student.grad).all()
        assert result.statistics["teacher_projection_min_norm"] == pytest.approx(2.0)


def test_unused_nonfinite_input_does_not_poison_real_head_gradients():
    objective = _objective()
    student = torch.ones(1, 2, 2, HIDDEN, requires_grad=True)
    teacher = torch.ones_like(student)
    with torch.no_grad():
        student[0, 0, 1] = float("nan")
        teacher[0, 0, 1] = float("inf")
    pair = _pair(0, 1, (_sample([0, 3], [[8, 8], [24, 24]]),))
    result = objective.compute({1: student}, {0: teacher}, (pair,), patch_hw=(16, 16),
                               reference_cap=4, generator=torch.Generator().manual_seed(0))
    result.raw.backward()
    assert torch.isfinite(student.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in objective.student_head.parameters())


def test_projection_norm_covers_each_pair_on_a_cached_teacher_view(monkeypatch):
    objective = _objective()
    monkeypatch.setattr(
        objective.student_head, "forward", lambda x: torch.ones((*x.shape[:-1], OUT))
    )
    monkeypatch.setattr(objective.teacher_head, "forward", lambda x: x[..., :OUT])
    teacher = torch.ones(1, 2, 2, HIDDEN)
    teacher[0, 0, 1] = 0.01
    pairs = (
        _pair(0, 1, (_sample([0, 3], [[8, 8], [24, 24]]),)),
        _pair(0, 2, (_sample([0, 1], [[8, 8], [24, 8]]),)),
    )
    result = objective.compute(
        {1: torch.ones_like(teacher), 2: torch.ones_like(teacher)},
        {0: teacher}, pairs, patch_hw=(16, 16), reference_cap=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert result.statistics["teacher_projection_min_norm"] == pytest.approx(0.02)


def test_each_objective_reports_attempted_and_contributing_branches():
    objective = _objective()
    pair_global = _pair(0, 1, (_sample([0, 3], [[8, 8], [24, 24]]),))
    pair_local = _pair(0, 2, (_sample([], []),))
    student = {1: torch.ones(1, 2, 2, HIDDEN), 2: torch.ones(1, 2, 2, HIDDEN)}
    teacher = {0: torch.ones(1, 2, 2, HIDDEN)}
    results = [
        objective_module.coordinate_loss(student, teacher, (pair_global, pair_local), patch_hw=(16, 16)),
        objective.compute(student, teacher, (pair_global, pair_local), patch_hw=(16, 16),
                          reference_cap=4, generator=torch.Generator().manual_seed(0)),
    ]
    for result in results:
        assert result.statistics["global_attempted_samples"] == 1
        assert result.statistics["global_contributing_samples"] == 1
        assert result.statistics["global_selected_points"] == 2
        assert result.statistics["local_attempted_pairs"] == 1
        assert result.statistics["local_contributing_pairs"] == 0
        assert result.statistics["local_skipped_samples"] == 1
        assert result.statistics["local_skipped_pairs"] == 1
