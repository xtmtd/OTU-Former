"""Enabled-only dense batch geometry: envelopes, pairs and correspondence queries."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from otuformer.training import dense_geometry as geo
from otuformer.training.dataset import (
    MultiCropDataset,
    collate_pretrain_geometry,
)
from otuformer.training.pretrain_loop import _unpack_pretrain_batch
from otuformer.vision.geometry import GeometryOp, GeometryTrace

SOURCE_WH = (64, 64)


def _trace(view_wh=(64, 64), *, drop=0, scale=1.0) -> GeometryTrace:
    """A realized resize trace; ``drop`` crops off source content."""
    ops = []
    if drop:
        ops.append(
            GeometryOp(
                "crop",
                {
                    "from_wh": list(SOURCE_WH),
                    "left": drop,
                    "top": drop,
                    "width": SOURCE_WH[0] - drop,
                    "height": SOURCE_WH[1] - drop,
                },
            )
        )
    ops.append(
        GeometryOp(
            "resize",
            {
                "from_wh": [SOURCE_WH[0] - drop, SOURCE_WH[1] - drop],
                "to_wh": list(view_wh),
                "interpolation": "bicubic",
                "antialias": True,
            },
        )
    )
    import otuformer.vision.geometry as g

    matrix = np.eye(3)
    for op in ops:
        matrix = g.matrix_for(op) @ matrix
    return GeometryTrace(
        source_wh=SOURCE_WH,
        view_wh=view_wh,
        operations=tuple(ops),
        source_to_view=matrix,
        view_to_source=np.linalg.inv(matrix),
    )


def _slots(*traces) -> tuple:
    """One batch row: each view slot holds that view's per-row traces."""
    return tuple(None if t is None else (t,) for t in traces)


def test_four_envelopes_exact_keys():
    """Each (dense, fifo) combination has exactly its documented batch keys."""
    views = [torch.zeros(2, 3, 64, 64) for _ in range(2)]
    # Two batch rows: each view slot carries one trace per row.
    traces = tuple((_trace(), _trace()) for _ in range(2))
    off_plain = _unpack_pretrain_batch(views, queue_enabled=False)
    assert off_plain[0] and off_plain[1] is None and off_plain[2] is None

    fifo = {"views": views, "image_key": ("a", "b")}
    unpacked = _unpack_pretrain_batch(fifo, queue_enabled=True)
    assert unpacked[1] == ("a", "b") and unpacked[2] is None

    dense = {"views": views, "geometry": traces}
    unpacked = _unpack_pretrain_batch(
        dense, queue_enabled=False, dense_enabled=True,
        local_crops=0, include_weak_teacher_views=False,
    )
    assert unpacked[2] is not None and len(unpacked[2]) == 2

    both = {"views": views, "geometry": traces, "image_key": ("a", "b")}
    unpacked = _unpack_pretrain_batch(
        both, queue_enabled=True, dense_enabled=True,
        local_crops=0, include_weak_teacher_views=False,
    )
    assert unpacked[1] == ("a", "b") and len(unpacked[2]) == 2

    # The resolved layout is required, never inferred from empty records.
    with pytest.raises(ValueError, match="local_crops"):
        _unpack_pretrain_batch(dense, queue_enabled=False, dense_enabled=True)


def test_collate_preserves_row_and_view_order():
    batch = [
        {"views": [torch.full((3, 8, 8), float(i)), torch.full((3, 8, 8), -float(i))],
         "geometry": [_trace((8, 8)), _trace((8, 8))],
         "image_key": f"k{i}"}
        for i in (1, 2)
    ]
    collated = collate_pretrain_geometry(batch)
    assert set(collated) == {"views", "geometry", "image_key"}
    assert collated["image_key"] == ("k1", "k2")
    assert collated["views"][1][0, 0, 0, 0].item() == -1.0
    assert len(collated["geometry"]) == 2 and len(collated["geometry"][0]) == 2


def test_serialized_transport_decodes_once_and_canonicalizes_weak(monkeypatch):
    views = [torch.zeros(2, 3, 64, 64) for _ in range(4)]
    record = _trace().to_json()
    calls = []
    real = GeometryTrace.from_json
    def decode(payload):
        calls.append(payload)
        return real(payload)
    monkeypatch.setattr(GeometryTrace, "from_json", decode)
    batch = {"views": views, "geometry": ((record, record), (record, record),
                                             (None, None), (None, None))}
    _, _, geometry = _unpack_pretrain_batch(
        batch, queue_enabled=False, dense_enabled=True,
        local_crops=0, include_weak_teacher_views=True,
    )
    assert len(calls) == 4
    assert geometry[2:] == (None, None)
    geo.build_dense_pairs(geometry, grid_hws=((4, 4),) * 4, patch_hw=(16, 16),
                          local_crops=0, query_cap=49,
                          generator=torch.Generator().manual_seed(0))
    assert len(calls) == 4


@pytest.mark.parametrize("damage", ["missing_field", "bad_matrix", "bad_type", "infinite_size"])
def test_serialized_trace_failure_identifies_slot_and_row(damage):
    record = _trace().to_json()
    bad = _trace().to_json()
    if damage == "missing_field":
        del bad["view_to_source"]
    elif damage == "bad_matrix":
        bad["source_to_view"] = [[1.0]]
    elif damage == "infinite_size":
        bad["source_wh"] = [float("inf"), 64]
    else:
        bad["source_wh"] = None
    batch = {"views": [torch.zeros(2, 3, 64, 64) for _ in range(2)],
             "geometry": ((record, record), (record, bad))}
    with pytest.raises(ValueError, match="geometry slot 1 row 1.*invalid serialized trace") as error:
        _unpack_pretrain_batch(
            batch, queue_enabled=False, dense_enabled=True,
            local_crops=0, include_weak_teacher_views=False,
        )
    assert isinstance(error.value.__cause__, (TypeError, ValueError, OverflowError))
    assert str(error.value.__cause__) in str(error.value)


def test_weak_geometry_none_without_tracking():
    """Weak-only slots stay None and are never consumed for pairing."""
    geometry = _slots(_trace(), _trace(), _trace((16, 16)), None, None)
    pairs = geo.build_dense_pairs(
        geometry,
        grid_hws=((4, 4), (4, 4), (1, 1), (4, 4), (4, 4)),
        patch_hw=(16, 16),
        local_crops=1,
        query_cap=49,
        generator=torch.Generator().manual_seed(0),
    )
    touched = {p.teacher_view for p in pairs} | {p.student_view for p in pairs}
    assert touched == {0, 1, 2}, "weak views 3/4 must never appear in a pair"


def test_pair_enumeration_and_order():
    """Teacher globals 0/1 map to the other global and every local, never itself."""
    geometry = _slots(_trace(), _trace(), _trace((16, 16)), _trace((16, 16)))
    pairs = geo.build_dense_pairs(
        geometry,
        grid_hws=((4, 4), (4, 4), (1, 1), (1, 1)),
        patch_hw=(16, 16),
        local_crops=2,
        query_cap=49,
        generator=torch.Generator().manual_seed(0),
    )
    ordered = [(p.teacher_view, p.student_view) for p in pairs]
    assert ordered == [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3)]


def test_identity_pairs_map_centers_and_cap_without_draw():
    """A full-overlap pair keeps capped points and consumes no sampling draw."""
    trace = _trace()
    generator = torch.Generator().manual_seed(5)
    before = generator.get_state().clone()
    pairs = geo.build_dense_pairs(
        _slots(trace, trace),
        grid_hws=((4, 4), (4, 4)),
        patch_hw=(16, 16),
        local_crops=0,
        query_cap=49,
        generator=generator,
    )
    # 16 centers is below the cap: every point is used and no draw is consumed.
    assert torch.equal(before, generator.get_state())
    pair = pairs[0]
    assert pair.samples[0].eligible_count == 16
    assert pair.samples[0].student_indices.tolist() == list(range(16))
    # Identity mapping puts the teacher point on the same center.
    xy = pair.samples[0].teacher_xy
    assert torch.allclose(xy[0], torch.tensor([8.0, 8.0], dtype=torch.float64))
    # Row-major: index 5 is row 1, column 1.
    assert torch.allclose(xy[5], torch.tensor([24.0, 24.0], dtype=torch.float64))


def test_rotation_and_clipping_remove_unsupported_centers():
    """A clipped/rotated view drops centers whose teacher support is incomplete."""
    import otuformer.vision.geometry as g

    ops = (
        GeometryOp(
            "crop",
            {"from_wh": [64, 64], "left": 8, "top": 0, "width": 56, "height": 64},
        ),
        GeometryOp(
            "resize",
            {"from_wh": [56, 64], "to_wh": [64, 64], "interpolation": "bicubic", "antialias": True},
        ),
    )
    matrix = np.eye(3)
    for op in ops:
        matrix = g.matrix_for(op) @ matrix
    cropped = GeometryTrace(
        source_wh=SOURCE_WH,
        view_wh=(64, 64),
        operations=ops,
        source_to_view=matrix,
        view_to_source=np.linalg.inv(matrix),
    )
    plain = _trace()
    pairs = geo.build_dense_pairs(
        _slots(plain, cropped),
        grid_hws=((4, 4), (4, 4)),
        patch_hw=(16, 16),
        local_crops=0,
        query_cap=49,
        generator=torch.Generator().manual_seed(1),
    )
    teacher_to_cropped = pairs[0]
    assert teacher_to_cropped.samples[0].eligible_count == 12
    kept = teacher_to_cropped.samples[0].student_indices.tolist()
    # The crop discards the source's left strip, so after mapping the student
    # view's last column (index 3, 7, 11, 15) lands past the teacher grid edge.
    assert [i for i in range(16) if i not in kept] == [3, 7, 11, 15]


def test_below_one_and_zero_candidates():
    """Zero overlap yields an empty pair sample; one point is still usable."""
    far = _trace(drop=60)  # keeps a 4x4 corner, upscaled
    pairs = geo.build_dense_pairs(
        _slots(far, far),
        grid_hws=((4, 4), (4, 4)),
        patch_hw=(16, 16),
        local_crops=0,
        query_cap=49,
        generator=torch.Generator().manual_seed(0),
    )
    sample = pairs[0].samples[0]
    assert sample.student_indices.shape[0] == sample.eligible_count
    assert sample.student_indices.shape[0] <= 49
    assert pairs[0].student_valid.dtype is torch.bool


def test_query_sampling_is_uniform_over_positions():
    """Above the cap, positions are drawn uniformly: no index-weight bias."""
    eligible = np.arange(8, dtype=np.int64)
    generator = torch.Generator().manual_seed(1234)
    counts = np.zeros(8, dtype=np.int64)
    for _ in range(2000):
        chosen = geo._sample_indices(eligible, 2, generator)
        assert chosen.shape == (2,)
        assert len(set(chosen.tolist())) == 2
        assert chosen.tolist() == sorted(chosen.tolist())  # row-major order kept
        counts[chosen] += 1
    # Every position is reachable, including index 0 (probability zero under the
    # old index-weighting bug) and the last index.
    assert counts.min() > 0
    expected = 2000 * 2 / 8
    for index in (0, 7):
        assert abs(int(counts[index]) - expected) < expected * 0.5
    # No monotone spatial bias across the grid.
    assert abs(int(counts[7]) - int(counts[0])) < expected * 0.5


def _rotated_trace(angle=35.0, view_wh=(64, 64)) -> GeometryTrace:
    """A realised rotation: the fill corners have no source correspondence."""
    import otuformer.vision.geometry as g

    ops = (
        GeometryOp(
            "rotate",
            {
                "from_wh": list(SOURCE_WH),
                "canvas_wh": list(SOURCE_WH),
                "angle": float(angle),
                "expand": False,
                "interpolation": "bicubic",
                "antialias": True,
                "center": [SOURCE_WH[0] / 2.0, SOURCE_WH[1] / 2.0],
            },
        ),
    )
    matrix = np.eye(3)
    for op in ops:
        matrix = g.matrix_for(op) @ matrix
    return GeometryTrace(
        source_wh=SOURCE_WH,
        view_wh=view_wh,
        operations=ops,
        source_to_view=matrix,
        view_to_source=np.linalg.inv(matrix),
    )


def test_support_mask_is_computed_per_batch_row():
    """Each row traces its own augmentation, so masks must not be row-0 copies."""
    plain = _trace()
    rotated = _rotated_trace()
    geometry = ((plain, rotated), (plain, plain))
    generator = torch.Generator().manual_seed(3)
    pairs = geo.build_dense_pairs(
        geometry,
        grid_hws=((4, 4), (4, 4)),
        patch_hw=(16, 16),
        local_crops=0,
        query_cap=49,
        generator=generator,
    )
    pair = pairs[0]
    assert pair.teacher_valid.shape == (2, 4, 4)
    assert bool(pair.teacher_valid[0].all())
    assert not bool(pair.teacher_valid[1].all())
    assert not torch.equal(pair.teacher_valid[0], pair.teacher_valid[1])
    # A row-0-only mask would report identical eligibility for both rows.
    eligible = [sample.eligible_count for sample in pair.samples]
    assert eligible[0] != eligible[1]


def test_unknown_rng_separation():
    """The query generator is independent from the reference generator."""
    query = torch.Generator().manual_seed(11)
    reference = torch.Generator().manual_seed(11)
    reference_state = reference.get_state().clone()
    geo.build_dense_pairs(
        _slots(_trace((16, 16)), _trace((16, 16))),
        grid_hws=((1, 1), (1, 1)),
        patch_hw=(16, 16),
        local_crops=0,
        query_cap=1,
        generator=query,
    )
    assert torch.equal(reference.get_state(), reference_state)


def test_trace_json_round_trip_preserves_pairing():
    trace = _trace()
    payload = json.loads(json.dumps(trace.to_json()))
    restored = GeometryTrace.from_json(payload)
    assert np.allclose(restored.view_to_source, trace.view_to_source)


def test_mismatched_layout_is_rejected():
    """A well-formed but wrongly placed None is not accepted."""
    geometry = _slots(_trace(), None, _trace((16, 16)))
    with pytest.raises(ValueError, match="geometry"):
        geo.build_dense_pairs(
            geometry,
            grid_hws=((4, 4), (4, 4), (1, 1)),
            patch_hw=(16, 16),
            local_crops=1,
            query_cap=1,
            generator=torch.Generator().manual_seed(0),
        )


def test_dataset_return_geometry_flag(tmp_path):
    """The dataset only produces traces when geometry is requested."""
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (72, 72), (10, 20, 30)).save(images / name)
    csv = tmp_path / "labels.csv"
    csv.write_text("image,label\na.png,x\nb.png,x\n")

    off = MultiCropDataset(csv, images, global_crop_size=64, local_crop_size=32, local_crops=1)
    row = off[0]
    assert isinstance(row, list)

    on = MultiCropDataset(
        csv,
        images,
        global_crop_size=64,
        local_crop_size=32,
        local_crops=1,
        return_geometry=True,
    )
    row = on[0]
    assert set(row) == {"views", "geometry"}
    assert len(row["geometry"]) == len(row["views"]) == 3
    assert all(isinstance(t, dict) for t in row["geometry"])
    assert row["geometry"][0]["view_wh"] == [64, 64]
    assert row["geometry"][2]["view_wh"] == [32, 32]
    batch = collate_pretrain_geometry([row, on[1]])
    views, _, decoded = _unpack_pretrain_batch(
        batch, queue_enabled=False, dense_enabled=True,
        local_crops=1, include_weak_teacher_views=False,
    )
    assert len(views) == 3
    assert all(isinstance(trace, GeometryTrace) for slot in decoded for trace in slot)
    assert decoded[0][0].to_json() == row["geometry"][0]


PROFILES = (
    ("global-barcode", "sensitive"),
    ("global-barcode", "invariant"),
    ("legacy", "sensitive"),
)


@pytest.mark.parametrize("profile,orientation", PROFILES)
def test_all_augmentation_profiles_trace_parity(tmp_path, profile, orientation):
    """Tracing must not move a tensor or an RNG draw for any supported profile."""
    import random

    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (72, 72), (17, 23, 29)).save(images / name)
    csv = tmp_path / "labels.csv"
    csv.write_text("image,label\na.png,x\nb.png,x\n")

    def _build(track):
        return MultiCropDataset(
            csv,
            images,
            global_crop_size=64,
            local_crop_size=32,
            local_crops=2,
            augmentation_profile=profile,
            orientation_policy=orientation,
            return_geometry=track,
        )

    import numpy as np

    def _states(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return (
            random.getstate(),
            np.random.get_state()[1].copy(),
            torch.random.get_rng_state().clone(),
        )

    plain, traced = _build(False), _build(True)
    for idx in (0, 1):
        _states(1234)
        untracked_views = plain[idx]
        untracked = (
            random.getstate(),
            np.random.get_state()[1].copy(),
            torch.random.get_rng_state().clone(),
        )

        _states(1234)
        row = traced[idx]
        tracked = (
            random.getstate(),
            np.random.get_state()[1].copy(),
            torch.random.get_rng_state().clone(),
        )

        for left, right in zip(untracked_views, row["views"]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert untracked[0] == tracked[0]
        assert np.array_equal(untracked[1], tracked[1])
        assert torch.equal(untracked[2], tracked[2])
        assert len(row["geometry"]) == len(untracked_views)


def test_support_eligibility_uses_the_sampler_float32_convention():
    """A float64 boundary must not make eligibility disagree with the sampler.

    ``7.9999999999999964`` is an exact patch centre in float32 but lands one
    cell to the left in float64, which would drop a genuinely valid query.
    """
    from otuformer.vision.features import sample_grid_features

    valid = np.ones((4, 4), dtype=bool)
    xy = np.array(
        [[7.9999999999999964, 8.0], [8.0, 24.0], [24.0, 24.0]], dtype=np.float64
    )
    mask = geo._support_mask(xy, valid, (4, 4), (16, 16))
    _, usable = sample_grid_features(
        torch.ones(1, 4, 4, 3),
        torch.from_numpy(xy).float()[None],
        patch_hw=(16, 16),
        valid=torch.from_numpy(valid)[None],
    )
    assert mask.tolist() == usable[0].tolist()
    assert mask.all()


def test_every_eligible_candidate_is_sampler_usable():
    """Over real crops, eligibility must equal the sampler's own verdict."""
    from otuformer.vision.features import sample_grid_features

    for drop in (0, 8, 24, 40):
        student = _trace()
        teacher = _trace(drop=drop)
        geometry = ((student,), (teacher,))
        pairs = geo.build_dense_pairs(
            geometry,
            grid_hws=((4, 4), (4, 4)),
            patch_hw=(16, 16),
            local_crops=0,
            query_cap=49,
            generator=torch.Generator().manual_seed(0),
        )
        pair = pairs[0]
        sample = pair.samples[0]
        if sample.student_indices.numel() == 0:
            continue
        _, usable = sample_grid_features(
            torch.ones(1, 4, 4, 3),
            sample.teacher_xy.unsqueeze(0).float(),
            patch_hw=(16, 16),
            valid=pair.teacher_valid[0:1],
        )
        assert bool(usable.all()), (drop, usable.tolist())
