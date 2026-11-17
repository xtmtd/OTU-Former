"""Dense batching preserves the rowwise objective, RNG and guard contract."""
from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from otuformer.training import dense_geometry as geometry
from otuformer.training import dense_objective as dense
from otuformer.training import neighbor_ranking as ranking
from otuformer.vision import geometry as image_geometry


def _fixture():
    generator = torch.Generator().manual_seed(19)
    student = {i: (torch.rand(3, 2, 2, 8, generator=generator) + 0.2).requires_grad_() for i in (0, 1, 2)}
    teacher = {i: (torch.rand(3, 2, 2, 8, generator=generator) + 0.2).requires_grad_() for i in (0, 1)}
    def sample(indices, xy):
        return geometry.DenseSampleQueries(torch.tensor(indices, dtype=torch.int64), torch.tensor(xy, dtype=torch.float64).reshape(-1, 2), len(indices) + 2)
    samples = (sample([0, 1, 3], [[8, 8], [16, 8], [20, 20]]), sample([], []), sample([2], [[12, 16]]))
    other = (sample([1], [[16, 12]]), sample([0, 2], [[8, 8], [16, 16]]), sample([], []))
    pairs = tuple(geometry.DensePairQueries(t, s, queries, torch.ones(3, 2, 2, dtype=torch.bool), torch.ones(3, 2, 2, dtype=torch.bool)) for t, s, queries in ((0, 1, samples), (0, 2, other), (1, 2, samples)))
    return student, teacher, pairs


def _rowwise_coordinate(student, teacher, pairs):
    losses = []
    for pair in pairs:
        rows = []
        for row, sample in enumerate(pair.samples):
            if not sample.student_indices.numel():
                continue
            points = dense._gather_centers(student[pair.student_view][row:row+1], sample.student_indices, (2, 2))
            a, _ = dense._unit_rows(points, context='reference')
            b, _ = dense.sample_dense_features(teacher[pair.teacher_view][row:row+1], sample.teacher_xy.float().unsqueeze(0), patch_hw=(16, 16), valid=pair.teacher_valid[row:row+1])
            rows.append((1 - (a * b[0].detach()).sum(-1)).mean())
        losses.append(rows)
    return dense.reduce_dense_losses(losses)


def _rowwise_ranking(head, student, teacher, pairs, generator, cap):
    losses = []
    for pair in pairs:
        a = head.project_field(student[pair.student_view])
        b = head.project_teacher_field(teacher[pair.teacher_view])
        rows = []
        for row, sample in enumerate(pair.samples):
            if sample.student_indices.numel():
                target, _ = dense.sample_dense_features(b[row:row+1], sample.teacher_xy.float().unsqueeze(0), patch_hw=(16, 16), valid=pair.teacher_valid[row:row+1])
                rows.append((row, sample, target[0]))
        pool = torch.cat([x[2] for x in rows]) if rows else b.new_empty(0, b.shape[-1])
        if len(pool) < 2:
            losses.append([])
            continue
        count = min(cap, len(pool))
        sample_losses = []
        for row, sample, target in rows:
            refs = pool if count == len(pool) else pool[torch.multinomial(torch.ones(len(pool), dtype=torch.float64), count, replacement=False, generator=generator)]
            points = dense._gather_centers(a[row:row+1], sample.student_indices, (2, 2))
            points, _ = dense._unit_rows(points, context='reference')
            _, ps = ranking.sort_values_and_permutation(points @ refs.T)
            with torch.no_grad():
                _, pt = ranking.sort_values_and_permutation(target @ refs.T)
            sample_losses.append(ranking.row_cross_entropy(ps, pt).mean())
        losses.append(sample_losses)
    return dense.reduce_dense_losses(losses)


def test_dense_validity_never_computes_unused_polygon_coverage(monkeypatch):
    from PIL import Image
    from otuformer.vision.geometry import GeometryOp, apply_geometry
    _, trace = apply_geometry(Image.new('RGB', (32, 32)), [GeometryOp('resize', {'from_wh': [32, 32], 'to_wh': [32, 32], 'interpolation': 'bicubic', 'antialias': True})], track=True)
    expected = image_geometry.content_validity(trace).reshape(2, 16, 2, 16).all(axis=(1, 3))
    def forbidden(*args, **kwargs):
        raise AssertionError('dense training must not compute polygon coverage')
    monkeypatch.setattr(image_geometry, 'pixel_support', forbidden)
    np.testing.assert_array_equal(geometry._patch_valid(trace, (2, 2), (16, 16)), expected)
    with pytest.raises(ValueError, match='grid|shape|pixel'):
        geometry._patch_valid(trace, (3, 2), (16, 16))


def test_coordinate_batches_ragged_rows_with_same_loss_and_gradients():
    student, teacher, pairs = _fixture()
    expected = _rowwise_coordinate(student, teacher, pairs)
    grads = torch.autograd.grad(expected, tuple(student.values()), allow_unused=True)
    with patch.object(dense, 'sample_dense_features', wraps=dense.sample_dense_features) as sampler:
        got = dense.coordinate_loss(student, teacher, pairs, patch_hw=(16, 16))
    assert sampler.call_count == len(pairs)
    torch.testing.assert_close(got.raw, expected)
    for actual, reference in zip(torch.autograd.grad(got.raw, tuple(student.values()), allow_unused=True), grads):
        if reference is not None:
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
    assert all(value.grad is None for value in teacher.values())
    assert got.statistics['selected_points'] == 11
    assert got.statistics['contributing_samples'] == 6


@pytest.mark.parametrize('cap', [2, 4, 16, 49])
def test_ranking_batches_with_rowwise_loss_gradients_rng_and_stats(cap):
    pytest.importorskip('diffsort')
    student, teacher, pairs = _fixture()
    torch.manual_seed(7)
    head = ranking.NeighborRankingObjective(8, hidden=16, out_dim=4)
    reference_head = copy.deepcopy(head)
    expected_rng = torch.Generator().manual_seed(17)
    actual_rng = torch.Generator().manual_seed(17)
    expected = _rowwise_ranking(reference_head, student, teacher, pairs, expected_rng, cap)
    refs = torch.autograd.grad(expected, tuple(student.values()) + tuple(reference_head.student_head.parameters()), allow_unused=True)
    with patch.object(head, 'project_field', wraps=head.project_field) as project, patch.object(head, 'project_teacher_field', wraps=head.project_teacher_field) as project_teacher, patch.object(dense, 'sample_dense_features', wraps=dense.sample_dense_features) as sample, patch.object(ranking, 'sort_values_and_permutation', wraps=ranking.sort_values_and_permutation) as sorter:
        got = head.compute(student, teacher, pairs, patch_hw=(16, 16), reference_cap=cap, generator=actual_rng)
    assert project.call_count == 2
    assert project_teacher.call_count == 2
    assert sample.call_count == len(pairs)
    assert sorter.call_count == 2 * len(pairs)
    torch.testing.assert_close(got.raw, expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(actual_rng.get_state(), expected_rng.get_state())
    for actual, reference in zip(torch.autograd.grad(got.raw, tuple(student.values()) + tuple(head.student_head.parameters()), allow_unused=True), refs):
        if reference is not None:
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
    assert all(value.grad is None for value in teacher.values())
    assert all(value.grad is None for value in head.teacher_head.parameters())
    assert got.statistics['selected_points'] == 11
    assert got.statistics['contributing_samples'] == 6
    assert got.statistics['reference_population_min'] == 3
    assert got.statistics['reference_shortage_samples'] == (0 if cap == 2 else 2 if cap == 4 else 6)


@pytest.mark.parametrize('role', ['student', 'teacher'])
def test_batched_coordinate_error_identifies_the_original_row(role):
    student, teacher, pairs = _fixture()
    with torch.no_grad():
        (student[1] if role == 'student' else teacher[0])[2].zero_()
    with pytest.raises(dense.DenseNumericalError, match='pair 0->1 row 2'):
        dense.coordinate_loss(student, teacher, pairs, patch_hw=(16, 16))


@pytest.mark.parametrize('kind', ['support', 'nonfinite', 'cancellation'])
def test_bad_nonfirst_teacher_row_still_aborts(kind):
    student, teacher, pairs = _fixture()
    pair = pairs[0]
    with torch.no_grad():
        if kind == 'support':
            pair.teacher_valid[2].fill_(False)
        elif kind == 'nonfinite':
            teacher[0][2].fill_(float('nan'))
        else:
            teacher[0][2, :, 0] = 1
            teacher[0][2, :, 1] = -3
    with pytest.raises(dense.DenseNumericalError, match='row 2'):
        dense.coordinate_loss(student, teacher, (pair,), patch_hw=(16, 16))


def test_all_valid_resize_preserves_support_without_footprint_raster(monkeypatch):
    from otuformer.vision.geometry import GeometryOp
    op = GeometryOp('resize', {'from_wh': [40, 32], 'to_wh': [16, 16], 'interpolation': 'bicubic', 'antialias': True})
    valid = np.ones((32, 40), dtype=bool)
    def forbidden(*args, **kwargs):
        raise AssertionError('all-valid clamped resize needs no footprint raster')
    monkeypatch.setattr(image_geometry.support, 'footprint_valid', forbidden)
    np.testing.assert_array_equal(image_geometry._propagate_valid(valid, op), np.ones((16, 16), dtype=bool))


def test_cancellation_diagnostics_are_not_recorded_by_autograd():
    field = torch.ones(1, 2, 2, 4, requires_grad=True)
    recorded = []
    original = dense._cancellation_ratio
    def inspect(*args, **kwargs):
        recorded.append(torch.is_grad_enabled())
        return original(*args, **kwargs)
    with patch.object(dense, '_cancellation_ratio', side_effect=inspect):
        unit, _ = dense.sample_dense_features(field, torch.tensor([[[16., 16.]]]), patch_hw=(16, 16), valid=torch.ones(1, 2, 2, dtype=torch.bool))
    assert recorded == [False]
    unit.sum().backward()
    assert field.grad is not None


@pytest.mark.parametrize('kind', ['nonfinite', 'negative', 'row', 'column', 'student_zero'])
def test_permutation_guards_reject_corrupt_nonfirst_query(kind):
    teacher = torch.full((2, 2, 2), 0.5)
    student = teacher.clone().requires_grad_()
    if kind == 'student_zero':
        with torch.no_grad():
            student[1] = torch.eye(2)
    elif kind == 'nonfinite':
        teacher[1, 0, 0] = float('nan')
    elif kind == 'negative':
        teacher[1] = torch.tensor([[-0.1, 1.1], [1.1, -0.1]])
    elif kind == 'row':
        teacher[1] *= 2
    else:
        teacher[1] = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
    with pytest.raises(ranking.RankingDependencyError):
        ranking.row_cross_entropy(student, teacher)


def test_projection_cache_expires_after_optimizer_and_ema_update():
    pytest.importorskip('diffsort')
    student, teacher, pairs = _fixture()
    head = ranking.NeighborRankingObjective(8, hidden=16, out_dim=4)
    optimizer = torch.optim.AdamW(head.student_head.parameters(), lr=1e-4)
    first = head.compute(student, teacher, pairs, patch_hw=(16, 16), reference_cap=2, generator=torch.Generator().manual_seed(9))
    first.raw.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    head.update_teacher(.9)
    with patch.object(head, 'project_field', wraps=head.project_field) as project:
        second = head.compute(student, teacher, pairs, patch_hw=(16, 16), reference_cap=2, generator=torch.Generator().manual_seed(9))
    assert project.call_count == 2
    second.raw.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.student_head.parameters())


def test_ranking_query_chunk_bounds_sorter_calls():
    pytest.importorskip('diffsort')
    student, teacher, pairs = _fixture()
    head = ranking.NeighborRankingObjective(8, hidden=16, out_dim=4)
    with patch.object(ranking, 'sort_values_and_permutation', wraps=ranking.sort_values_and_permutation) as sorter:
        head.compute(student, teacher, pairs, patch_hw=(16, 16), reference_cap=2, generator=torch.Generator().manual_seed(1), query_chunk=2)
    assert max(call.args[0].shape[0] for call in sorter.call_args_list) <= 2
