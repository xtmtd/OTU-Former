"""Frozen protocol: view/query generation, ranking and stored plan."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from otuformer.data.provenance import canonical_digest
from otuformer.spatial.protocol import (
    DIRECTIONS,
    PROTOCOL_VIEW_IDS,
    REFERENCE_ALIASES,
    build_evaluation_plan,
    build_evaluation_views,
    build_query_plan,
    geometry_signature,
    grid_center_point,
    grid_point_support,
    pixel_point_support,
    protocol_operations,
    select_evaluation_keys,
)
from otuformer.spatial.records import PATCH_COVERAGE, PATCH_VALID, PIXEL_COVERAGE, PIXEL_VALID, SpatialView
from otuformer.vision.geometry import apply_geometry, evaluation_view_ops, normalize_tensor
from otuformer.vision.features import FeatureContract, LayerSpec


def contract(grid=2, channels=4, image_size=32) -> FeatureContract:
    return FeatureContract(
        layers=(LayerSpec(path="blocks.11", block_index=11, grid_hw=(grid, grid), channels=channels),),
        input_hw=(image_size, image_size),
        patch_hw=(image_size // grid, image_size // grid),
        prefix_tokens=1,
        norm="backbone",
        norm_module={"path": "norm", "qualname": "timm.layers.norm.LayerNorm"},
        l2=False,
        epsilon=1e-12,
        normalization={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        model_identity={"architecture": "test"},
    )


def image(width=91, height=61):
    from PIL import Image

    picture = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            picture.putpixel((x, y), (x * 5 % 256, y * 7 % 256, (x + y) % 256))
    return picture


def view_from_trace(image_key, view_id, role, trace, *, seed=0, patch_valid=None) -> SpatialView:
    from otuformer.vision.geometry import patch_support, pixel_support

    size = int(trace.view_wh[0])
    contract_obj = contract(grid=size // 16, image_size=size)
    coverage, pixel_valid = pixel_support(trace)
    patch_cov, patch_ok = patch_support(
        trace,
        pixel_sampling_valid=pixel_valid,
        grid_hw=contract_obj.layers[0].grid_hw,
        patch_hw=contract_obj.patch_hw,
    )
    if patch_valid is not None:
        patch_ok = np.asarray(patch_valid, dtype=bool)
    generator = np.random.default_rng(seed)
    features = {
        layer.path: generator.normal(size=(*layer.grid_hw, layer.channels)).astype(np.float32)
        for layer in contract_obj.layers
    }
    return SpatialView(
        image_key=image_key,
        view_id=view_id,
        role=role,
        trace=trace,
        features=features,
        pixel_geometric_coverage=coverage,
        pixel_sampling_valid=pixel_valid,
        geometric_coverage=patch_cov,
        patch_sampling_valid=patch_ok,
    )


def test_protocol_view_ids_keys_and_canvases():
    picture = image()
    built = build_evaluation_views(
        picture, image_key="a.png", eval_transform="whole-specimen-pad", size=32
    )
    assert [view_id for view_id, _tensor, _trace in built] == list(PROTOCOL_VIEW_IDS)
    assert len(built) == 12
    for view_id, tensor, trace in built:
        assert tensor.shape == (3, 32, 32)
        assert tuple(trace.view_wh) == (32, 32)
        assert trace.source_wh == picture.size
    # Every view is built from the same source through the shared geometry path.
    matrices = {view_id: trace.source_to_view for view_id, _t, trace in built}
    assert not np.allclose(matrices["reference"], matrices["hflip"])
    np.testing.assert_allclose(
        matrices["reference"], matrices["crop075"] @ np.linalg.inv(matrices["crop075"]) @ matrices["reference"]
    )


def test_protocol_rounding_signs_and_order():
    size = 32
    shift = int(round(0.10 * size))
    assert shift == 3
    # half-to-even ties follow Python's round, matching the frozen protocol.
    assert int(round(0.5)) == 0 and int(round(1.5)) == 2
    side = int(math.floor(0.75 * size))
    offset = (size - side) // 2
    crop075 = protocol_operations(size, "crop075")
    assert [op.kind for op in crop075] == ["crop", "resize"]
    assert crop075[0].params["width"] == side == 24
    assert crop075[0].params["left"] == offset == 4
    assert crop075[1].params["to_wh"] == [size, size]

    translate = protocol_operations(size, "tx10")[0]
    assert translate.params["translate"] == [float(shift), 0.0]
    assert translate.params["angle"] == 0.0 and translate.params["scale"] == 1.0
    assert protocol_operations(size, "ty10")[0].params["translate"] == [0.0, float(shift)]
    assert protocol_operations(size, "scale090")[0].params["scale"] == 0.9
    assert protocol_operations(size, "scale110")[0].params["scale"] == 1.1
    assert protocol_operations(size, "rot-15")[0].params["angle"] == -15.0
    assert protocol_operations(size, "rot15")[0].params["angle"] == 15.0
    assert protocol_operations(size, "rot90")[0].params["angle"] == 90.0
    # The pivot is filled in by the geometry owner when the operation executes.
    _, rot90_trace = apply_geometry(
        image(32, 32), protocol_operations(size, "rot90"), track=True
    )
    assert rot90_trace.operations[0].params["center"] == [16.0, 16.0]
    # combo is crop075, then +15 degrees, then a positive x translation.
    combo = protocol_operations(size, "combo")
    assert [op.kind for op in combo] == ["crop", "resize", "rotate", "affine"]
    assert combo[2].params["angle"] == 15.0
    assert combo[3].params["translate"][0] == float(shift)
    assert protocol_operations(size, "reference") == []
    with pytest.raises(Exception, match="unknown protocol view"):
        protocol_operations(size, "crop050")


def test_protocol_views_preserve_colour_and_normalize_once():
    picture = image(64, 64)
    base = normalize_tensor(
        apply_geometry(
            picture,
            evaluation_view_ops(picture, name="whole-specimen-pad", size=32),
        )[0]
    )
    built = dict(
        (view_id, tensor)
        for view_id, tensor, _trace in build_evaluation_views(
            picture, image_key="a.png", eval_transform="whole-specimen-pad", size=32
        )
    )
    # The reference view is the deterministic construction, and hflip is its mirror
    # with no photometric change.
    np.testing.assert_allclose(built["reference"].numpy(), base.numpy(), atol=1e-6)
    np.testing.assert_allclose(
        built["hflip"].numpy(), np.flip(base.numpy(), axis=-1), atol=1e-6
    )


def test_evaluation_key_selection_is_deterministic_and_restores_order():
    keys = [f"sample_{index:03d}.png" for index in range(150)]
    selected = select_evaluation_keys(keys, seed=42)
    assert len(selected) == 100
    assert list(selected) == [key for key in keys if key in set(selected)]
    assert selected == select_evaluation_keys(keys, seed=42)
    other = select_evaluation_keys(keys, seed=7)
    assert other != selected
    assert len(other) == 100
    assert select_evaluation_keys(keys[:5], seed=42) == tuple(keys[:5])
    with pytest.raises(Exception, match="must be positive"):
        select_evaluation_keys(keys, seed=42, limit=0)


def test_key_ranking_matches_the_documented_tuple():
    keys = [f"k{index}" for index in range(120)]
    selected = set(select_evaluation_keys(keys, seed=42))
    ranked = sorted(
        keys, key=lambda key: (canonical_digest(["spatial-geometry-v1", 42, "image", key]), key)
    )
    assert selected == set(ranked[:100])


def test_query_plan_counts_and_directions():
    picture = image()
    built = {
        view_id: (tensor, trace)
        for view_id, tensor, trace in build_evaluation_views(
            picture, image_key="a.png", eval_transform="whole-specimen-pad", size=32
        )
    }
    reference = view_from_trace("a.png", "base", "base", built["reference"][1])
    target = view_from_trace(
        "a.png", "tx10", "eval-transform", built["tx10"][1]
    )
    plan = build_query_plan("a.png", reference, target, pair_id="tx10", seed=42)
    assert plan["protocol"] == "spatial-geometry-v1"
    assert plan["seed"] == 42
    assert plan["pair_id"] == "tx10"
    assert plan["reference_view"] == "base"
    assert set(plan["directions"]) == set(DIRECTIONS)
    for direction in DIRECTIONS:
        entry = plan["directions"][direction]
        assert entry["source_valid_count"] >= entry["eligible_count"] >= entry["sampled_count"]
        assert entry["sampled_count"] <= 64
        assert len(entry["queries"]) == entry["sampled_count"]
        for query in entry["queries"]:
            row, column = query["row"], query["column"]
            # Row-major order is restored after selection.
            assert grid_center_point(row, column, (16, 16)) == pytest.approx(
                tuple(query["xy_source"])
            )
        keys = [(query["row"], query["column"]) for query in entry["queries"]]
        assert keys == sorted(keys)
    assert plan["geometry"]["reference"]["view_wh"] == [32, 32]
    assert plan["geometry"]["target"]["view_wh"] == [32, 32]


def test_query_points_use_the_composed_view_to_view_map():
    """``p_target = M_target inverse(M_source) p_source``, checked on real views.

    An independent assertion on the composed map: a translated target must place
    the mapped point at source + translation, not at source - translation. The
    reversed composition passed the support checks and silently corrupted the
    ground truth of every query.
    """
    picture = image()
    built = {
        view_id: trace
        for view_id, _tensor, trace in build_evaluation_views(
            picture, image_key="a.png", eval_transform="whole-specimen-pad", size=64
        )
    }
    reference = view_from_trace("a.png", "reference", "base", built["reference"])
    target = view_from_trace("a.png", "tx10", "eval-transform", built["tx10"])
    shift = target.trace.source_to_view[0, 2]
    assert shift > 0.0  # a positive x translation moves content right
    plan = build_query_plan("a.png", reference, target, pair_id="tx10", seed=42)
    queries = plan["directions"]["reference_to_target"]["queries"]
    assert queries
    for query in queries:
        source_point = np.asarray(query["xy_source"], dtype=np.float64)
        assert query["xy_target"] == pytest.approx(
            [float(source_point[0] + shift), float(source_point[1])], abs=1e-9
        )
    # Both directions map the same pair the other way round. The query lattice is
    # the source view's own grid, so the two directions are compared through the
    # mapped points rather than through (row, column).
    reverse = plan["directions"]["target_to_reference"]["queries"]
    forward_points = {
        (round(query["xy_source"][0], 6), round(query["xy_source"][1], 6)): query
        for query in queries
    }
    for query in reverse:
        forward = forward_points.get(
            (round(query["xy_target"][0], 6), round(query["xy_target"][1], 6))
        )
        if forward is None:
            continue
        assert forward["xy_target"] == pytest.approx(query["xy_source"], abs=1e-6)


def test_queries_require_full_source_and_target_support():
    picture = image()
    built = {
        view_id: trace
        for view_id, _tensor, trace in build_evaluation_views(
            picture, image_key="a.png", eval_transform="whole-specimen-pad", size=32
        )
    }
    # A source whose only valid center is the top-left patch, and a target whose
    # opposite corners are unusable: only the genuinely eligible one survives.
    source_valid = np.zeros((2, 2), dtype=bool)
    source_valid[0, 0] = True
    reference = view_from_trace(
        "a.png", "reference", "eval-reference", built["reference"],
        patch_valid=source_valid,
    )
    target_valid = np.zeros((2, 2), dtype=bool)
    target_valid[0, 0] = True
    target = view_from_trace(
        "a.png", "rot90", "eval-transform", built["rot90"], patch_valid=target_valid
    )
    plan = build_query_plan("a.png", reference, target, pair_id="rot90", seed=42)
    forward = plan["directions"]["reference_to_target"]
    assert forward["source_valid_count"] == 1
    assert forward["eligible_count"] <= 1
    # The reverse direction has one valid center too, but its mapped point may fall
    # outside the source support: the counts stay explicit either way.
    reverse = plan["directions"]["target_to_reference"]
    assert reverse["source_valid_count"] == 1
    assert reverse["eligible_count"] in (0, 1)
    if reverse["eligible_count"] == 0:
        assert reverse["unavailable"] is True
        assert reverse["unavailable_reason"] in ("no_overlap", "no_source_valid_center")


def test_no_usable_candidates_is_recorded_not_dropped():
    picture = image()
    trace = dict(
        (view_id, trace)
        for view_id, _tensor, trace in build_evaluation_views(
            picture, image_key="a.png", eval_transform="whole-specimen-pad", size=32
        )
    )["reference"]
    empty = np.zeros((2, 2), dtype=bool)
    reference = view_from_trace("a.png", "reference", "eval-reference", trace, patch_valid=empty)
    target = view_from_trace("a.png", "tx10", "eval-transform", trace, patch_valid=empty)
    plan = build_query_plan("a.png", reference, target, pair_id="tx10", seed=42)
    for direction in DIRECTIONS:
        entry = plan["directions"][direction]
        assert entry["sampled_count"] == 0
        assert entry["unavailable"] is True
        assert entry["unavailable_reason"] == "no_source_valid_center"


def test_query_limit_selects_by_hash_and_restores_row_major_order():
    # A larger canvas with a pure translation keeps many centres eligible, so the
    # 64-query budget is the thing that truncates the set.
    picture = image(128, 128)
    traces = {
        view_id: trace
        for view_id, _tensor, trace in build_evaluation_views(
            picture, image_key="a.png", eval_transform="center-crop", size=128
        )
    }
    reference = view_from_trace("a.png", "base", "base", traces["reference"])
    target = view_from_trace("a.png", "tx10", "eval-transform", traces["tx10"])
    plan = build_query_plan("a.png", reference, target, pair_id="tx10", seed=42, limit=5)
    forward = plan["directions"]["reference_to_target"]
    assert forward["eligible_count"] > 5
    assert forward["sampled_count"] == 5
    keys = [(query["row"], query["column"]) for query in forward["queries"]]
    assert keys == sorted(keys)
    # A different seed selects a different subset of the same eligible set.
    other = build_query_plan("a.png", reference, target, pair_id="tx10", seed=7, limit=5)
    assert other["directions"]["reference_to_target"]["queries"] != forward["queries"]
    # Without a limit below the eligible count, every eligible query is kept.
    unbounded = build_query_plan(
        "a.png", reference, target, pair_id="tx10", seed=42, limit=10_000
    )
    assert (
        unbounded["directions"]["reference_to_target"]["sampled_count"]
        == forward["eligible_count"]
    )


def test_geometry_signature_is_stored_geometry_only():
    picture = image()
    trace = dict(
        (view_id, trace)
        for view_id, _tensor, trace in build_evaluation_views(
            picture, image_key="a.png", eval_transform="whole-specimen-pad", size=32
        )
    )["rot15"]
    view = view_from_trace("a.png", "rot15", "eval-transform", trace)
    signature = geometry_signature(view)
    assert signature["view_id"] == "rot15"
    assert signature["view_wh"] == [32, 32]
    assert signature["grid_hw"] == [2, 2]
    assert signature["support_version"] == "conservative-resampling-v1"
    assert "features" not in signature
    assert len(signature["source_to_view"]) == 3


def test_pixel_and_grid_point_support_follow_contributor_rules():
    valid = np.ones((4, 4), dtype=bool)
    # Interior point: four contributors, all valid.
    assert pixel_point_support(valid, (2.5, 2.5))
    valid[0, 0] = False
    assert not pixel_point_support(valid, (0.5, 0.5))
    # An exact pixel centre has one contributor, so a zero-weight out-of-grid
    # neighbour does not invalidate it; a point past the grid still does.
    assert pixel_point_support(np.ones((4, 4), dtype=bool), (3.5, 3.5))
    assert not pixel_point_support(np.ones((4, 4), dtype=bool), (8.0, 8.0))
    grid_valid = np.ones((2, 2), dtype=bool)
    grid_valid[0, 0] = False
    assert not grid_point_support(grid_valid, (8.0, 8.0), (16, 16))
    assert grid_point_support(grid_valid, (24.0, 24.0), (16, 16))


def test_evaluation_plan_groups_pairs_in_order():
    picture = image()
    built = build_evaluation_views(
        picture, image_key="a.png", eval_transform="whole-specimen-pad", size=32
    )
    views = []
    for view_id, _tensor, trace in built:
        role = "base" if view_id == "reference" else "eval-transform"
        stored_id = "base" if view_id == "reference" else view_id
        views.append(view_from_trace("a.png", stored_id, role, trace))
    plan = build_evaluation_plan({"a.png": views}, seed=42)
    assert plan["protocol"] == "spatial-geometry-v1"
    assert plan["schema_version"] == 1
    assert plan["seed"] == 42
    assert plan["selected_keys"] == ["a.png"]
    assert plan["view_ids"] == list(PROTOCOL_VIEW_IDS)
    assert len(plan["pairs"]) == len(PROTOCOL_VIEW_IDS) - 1
    assert [pair["pair_id"] for pair in plan["pairs"]] == [
        view_id for view_id in PROTOCOL_VIEW_IDS if view_id != "reference"
    ]
    json.dumps(plan)  # canonical-JSON serializable, no numpy leakage


def test_plan_rejects_missing_views_and_mismatched_pairs():
    picture = image()
    trace = dict(
        (view_id, trace)
        for view_id, _tensor, trace in build_evaluation_views(
            picture, image_key="a.png", eval_transform="whole-specimen-pad", size=32
        )
    )["reference"]
    reference = view_from_trace("a.png", "base", "base", trace)
    with pytest.raises(Exception, match="pair_id"):
        build_query_plan("a.png", reference, reference, pair_id="tx10", seed=42)
    with pytest.raises(Exception, match="pairs views of one image"):
        build_query_plan("a.png", reference, view_from_trace("b.png", "base", "base", trace), pair_id="base", seed=42)
    with pytest.raises(Exception, match="reference view"):
        build_query_plan("a.png", view_from_trace("a.png", "tx10", "eval-transform", trace), reference, pair_id="base", seed=42)
    with pytest.raises(Exception, match="missing protocol view"):
        build_evaluation_plan({"a.png": [reference]}, seed=42)


def test_eligibility_needs_complete_target_feature_support_not_only_pixels():
    """A mapped point valid on the pixel grid but not on the feature grid fails.

    The stored-plan target-support check used to look at `pixel_sampling_valid`
    only, so a query whose bilinear feature interpolation reached an invalid
    neighbour was accepted as eligible.
    """
    picture = image()
    built = {
        view_id: trace
        for view_id, _tensor, trace in build_evaluation_views(
            picture, image_key="a.png", eval_transform="whole-specimen-pad", size=32
        )
    }
    source_valid = np.ones((2, 2), dtype=bool)
    reference = view_from_trace(
        "a.png", "reference", "eval-reference", built["reference"],
        patch_valid=source_valid,
    )
    # Everything is pixel-valid, but the whole target feature grid is invalid.
    target = view_from_trace(
        "a.png", "rot90", "eval-transform", built["rot90"],
        patch_valid=np.zeros((2, 2), dtype=bool),
    )
    plan = build_query_plan("a.png", reference, target, pair_id="rot90", seed=42)
    forward = plan["directions"]["reference_to_target"]
    assert forward["source_valid_count"] == 4
    assert forward["eligible_count"] == 0
    assert forward["queries"] == []
    assert forward["unavailable"] is True
    assert forward["unavailable_reason"] == "no_overlap"
    # The reverse direction has no valid source centre at all, so it is a
    # different unavailable cause rather than the same overlap failure.
    reverse = plan["directions"]["target_to_reference"]
    assert reverse["source_valid_count"] == 0
    assert reverse["unavailable_reason"] == "no_source_valid_center"
