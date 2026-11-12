"""Realized geometry traces, coverage and conservative sampling support."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from otuformer.training.dataset import (
    center_crop_eval_transform,
    whole_specimen_pad_transform,
)
from otuformer.vision.geometry import (
    GeometryOp,
    GeometryTrace,
    apply_geometry,
    build_eval_view,
    evaluation_view_ops,
    map_points,
    patch_support,
    pixel_support,
    resize_output_size,
    trace_operations,
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def asymmetric_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), (x * 7 % 256, y * 11 % 256, (x * 3 + y * 5) % 256))
    return image


def old_center_crop(size: int):
    """Independent test-local copy of the historical composition."""
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def old_whole_specimen_pad(size: int):
    def pad_to_square(image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        array = np.asarray(image.convert("RGB"))
        edges = np.concatenate(
            [array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]], axis=0
        )
        fill = tuple(int(v) for v in np.median(edges, axis=0))
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), color=fill)
        offset = (0, (side - height) // 2) if width > height else ((side - width) // 2, 0)
        canvas.paste(image, offset)
        return canvas

    return transforms.Compose(
        [
            transforms.Lambda(pad_to_square),
            transforms.Resize((size, size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


@pytest.mark.parametrize("name", ["center-crop", "whole-specimen-pad"])
@pytest.mark.parametrize("size", [(37, 91), (90, 37), (64, 64), (63, 65)])
def test_eval_views_match_existing_builders(name, size):
    image = asymmetric_image(*size)
    reference = (
        old_center_crop(48) if name == "center-crop" else old_whole_specimen_pad(48)
    )
    tensor, trace = build_eval_view(image, name=name, size=48)
    torch.testing.assert_close(tensor, reference(image))
    assert tensor.shape == (3, 48, 48)
    assert (trace.source_wh[0], trace.source_wh[1]) == size
    assert tuple(trace.view_wh) == (48, 48)

    retained = (
        center_crop_eval_transform(48) if name == "center-crop"
        else whole_specimen_pad_transform(48)
    )
    torch.testing.assert_close(tensor, retained(image))


def test_center_crop_records_realized_bounds_and_resize_ratio():
    image = asymmetric_image(91, 37)  # width 91, height 37: short side is height
    ops = evaluation_view_ops(image, name="center-crop", size=48)
    assert [op.kind for op in ops] == ["resize", "crop"]
    new_w, new_h = resize_output_size(48, 91, 37)
    assert (new_w, new_h) == (int(48 * 91 / 37), 48) == (118, 48)
    assert ops[0].params["to_wh"] == [new_w, new_h]
    resize_ratio = ops[0].params["to_wh"][0] / 91
    assert math.isclose(resize_ratio, 118 / 91)
    # Center crop of a 118x48 canvas to 48x48 keeps integer bounds.
    assert ops[1].params["left"] == int(round((118 - 48) / 2.0))
    assert ops[1].params["top"] == 0
    trace = trace_operations((91, 37), ops)
    # The crop offset appears in the composed map: source x=0 -> view x=-35.
    assert map_points(trace, np.array([[0.0, 0.0]]))[0][0] == pytest.approx(-35.0)
    # Round trip is exact for a point that survived the crop.
    point = np.array([[45.0, 20.0]])
    round_trip = map_points(trace, map_points(trace, point), inverse=True)
    np.testing.assert_allclose(round_trip, point, atol=1e-9)


def test_whole_specimen_pad_records_offset_and_fill():
    image = asymmetric_image(20, 8)
    ops = evaluation_view_ops(image, name="whole-specimen-pad", size=32)
    assert [op.kind for op in ops] == ["pad", "resize"]
    pad = ops[0].params
    assert pad["canvas_wh"] == [20, 20]
    assert pad["offset"] == [0, (20 - 8) // 2] == [0, 6]
    array = np.asarray(image.convert("RGB"))
    edges = np.concatenate(
        [array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]], axis=0
    )
    assert pad["fill"] == [int(v) for v in np.median(edges, axis=0)]
    assert ops[1].params["to_wh"] == [32, 32]


def test_signed_marker_rotation_and_translation():
    canvas = Image.new("RGB", (100, 100), (0, 0, 0))
    canvas.putpixel((70, 50), (255, 0, 0))
    rotate = GeometryOp(
        "rotate",
        {
            "from_wh": [100, 100],
            "angle": 90.0,
            "center": [50.0, 50.0],
            "expand": False,
            "interpolation": "nearest",
            "fill": [0, 0, 0],
            "canvas_wh": [100, 100],
        },
    )
    _, trace = apply_geometry(canvas, [rotate], track=True)
    assert trace is not None
    moved = map_points(trace, np.array([[70.0, 50.0]]))[0]
    np.testing.assert_allclose(moved, [50.0, 30.0], atol=1e-9)

    translate = GeometryOp(
        "affine",
        {
            "from_wh": [100, 100],
            "center": [50.0, 50.0],
            "angle": 0.0,
            "translate": [10.0, 0.0],
            "scale": 1.0,
            "shear": [0.0, 0.0],
            "interpolation": "nearest",
            "fill": [0, 0, 0],
            "canvas_wh": [100, 100],
        },
    )
    _, combined = apply_geometry(canvas, [rotate, translate], track=True)
    assert combined is not None
    moved = map_points(combined, np.array([[70.0, 50.0]]))[0]
    np.testing.assert_allclose(moved, [60.0, 30.0], atol=1e-9)


def test_rotation_centre_convention_matches_the_real_resampler():
    """The canonical map must place a marker where Pillow/vision actually does."""
    canvas = Image.new("RGB", (100, 100), (0, 0, 0))
    canvas.putpixel((70, 50), (255, 0, 0))
    rotate = GeometryOp(
        "rotate",
        {
            "from_wh": [100, 100],
            "angle": 90.0,
            "center": [50.0, 50.0],
            "expand": False,
            "interpolation": "nearest",
            "fill": [0, 0, 0],
            "canvas_wh": [100, 100],
        },
    )
    rotated, trace = apply_geometry(canvas, [rotate], track=True)
    assert trace is not None
    array = np.asarray(rotated).sum(axis=-1)
    rows, cols = np.nonzero(array > 0)
    assert (int(cols[0]), int(rows[0])) == (50, 29)
    # The marker's pixel centre maps to the centre of the output pixel.
    mapped = map_points(trace, np.array([[70.5, 50.5]]))[0]
    np.testing.assert_allclose(mapped, [50.5, 29.5], atol=1e-9)


def test_expanded_rotation_translation_matches_real_output():
    canvas = Image.new("RGB", (100, 60), (0, 0, 0))
    canvas.putpixel((25, 15), (255, 0, 0))
    ops = [
        GeometryOp(
            "rotate",
            {
                "from_wh": [100, 60],
                "angle": 30.0,
                "center": [50.0, 30.0],
                "expand": True,
                "interpolation": "nearest",
                "fill": [0, 0, 0],
                "canvas_wh": [118, 102],
            },
        )
    ]
    rotated, trace = apply_geometry(canvas, ops, track=True)
    assert trace is not None
    assert tuple(trace.view_wh) == rotated.size
    mapped = map_points(trace, np.array([[25.5, 15.5]]))[0]
    rows, cols = np.nonzero(np.asarray(rotated).sum(axis=-1) > 0)
    assert (int(cols[0]), int(rows[0])) == (int(math.floor(mapped[0])), int(math.floor(mapped[1])))


def test_crop_inverse_cannot_recover_discarded_content():
    image = asymmetric_image(40, 40)
    crop = GeometryOp(
        "crop",
        {"from_wh": [40, 40], "left": 10, "top": 10, "width": 20, "height": 20},
    )
    _, trace = apply_geometry(image, [crop], track=True)
    assert trace is not None
    coverage, valid = pixel_support(trace)
    assert coverage.shape == (20, 20) and coverage.dtype == np.float32
    assert valid.dtype == np.bool_ and valid.all()
    np.testing.assert_allclose(coverage, 1.0)
    # A point outside the crop is outside the view even though the map inverts.
    outside = np.array([[2.0, 2.0]])
    mapped = map_points(trace, outside)
    assert mapped[0][0] < 0.0
    np.testing.assert_allclose(
        map_points(trace, mapped, inverse=True), outside, atol=1e-9
    )


def test_pixel_and_patch_support_for_padded_views():
    image = asymmetric_image(20, 8)
    tensor, trace = build_eval_view(image, name="whole-specimen-pad", size=32)
    coverage, valid = pixel_support(trace)
    assert coverage.shape == valid.shape == (32, 32)
    assert coverage.dtype == np.float32 and valid.dtype == np.bool_
    assert 0.0 <= float(coverage.min()) and float(coverage.max()) <= 1.0
    # The top/bottom padding rows carry no source content and are invalid.
    assert coverage[0].max() == 0.0
    assert not valid[0].any()
    # Padding is geometry support, never foreground: the resized specimen rows
    # are fully covered and valid.
    assert coverage[16].min() == pytest.approx(1.0)
    assert valid[16].all()

    patch_cov, patch_valid = patch_support(
        trace, pixel_sampling_valid=valid, grid_hw=(2, 2), patch_hw=(16, 16)
    )
    assert patch_cov.shape == patch_valid.shape == (2, 2)
    assert patch_cov.dtype == np.float32 and patch_valid.dtype == np.bool_
    # Both patch rows cross the padding, so neither is fully usable; the patch
    # coverage stays a fraction instead of being clamped to a Boolean.
    assert patch_cov[0, 0] < 1.0
    assert not patch_valid[0, 0]
    assert not patch_valid[1, 1]
    assert 0.0 < float(patch_cov[1, 1]) < 1.0

    # A cropped, unpadded view keeps fully valid patches.
    crop = GeometryOp(
        "crop",
        {"from_wh": [32, 32], "left": 0, "top": 0, "width": 32, "height": 32},
    )
    _, crop_trace = apply_geometry(asymmetric_image(32, 32), [crop], track=True)
    assert crop_trace is not None
    crop_coverage, crop_valid = pixel_support(crop_trace)
    np.testing.assert_allclose(crop_coverage, 1.0)
    patch_cov, patch_valid = patch_support(
        crop_trace, pixel_sampling_valid=crop_valid, grid_hw=(2, 2), patch_hw=(16, 16)
    )
    assert patch_valid.all()
    np.testing.assert_allclose(patch_cov, 1.0)


def test_fractional_rotation_edges_keep_bounds_and_clipping():
    image = asymmetric_image(64, 64)
    rotate = GeometryOp(
        "rotate",
        {
            "from_wh": [64, 64],
            "angle": 15.0,
            "center": [32.0, 32.0],
            "expand": False,
            "interpolation": "bicubic",
            "fill": [0, 0, 0],
            "canvas_wh": [64, 64],
        },
    )
    _, trace = apply_geometry(image, [rotate], track=True)
    assert trace is not None
    coverage, valid = pixel_support(trace)
    assert coverage.shape == (64, 64)
    assert float(coverage.min()) >= 0.0 and float(coverage.max()) <= 1.0
    # Rotating a square in place leaves the corners uncovered.
    assert coverage[0, 0] < 0.5
    assert coverage[32, 32] == pytest.approx(1.0)
    assert not valid[0, 0] and valid[32, 32]


def test_resampler_invalid_pixel_inside_a_covered_patch():
    """An invalid source pixel makes its footprint patch unusable, not partial."""
    image = asymmetric_image(32, 32)
    resize = GeometryOp(
        "resize",
        {
            "from_wh": [32, 32],
            "to_wh": [16, 16],
            "interpolation": "bicubic",
            "antialias": True,
        },
    )
    _, trace = apply_geometry(image, [resize], track=True)
    assert trace is not None
    coverage, valid = pixel_support(trace)
    assert valid.all()  # nothing invalid at the source
    assert float(coverage.min()) == pytest.approx(1.0)

    # Now propagate an explicit invalid source pixel through the same resampler.
    from otuformer.vision.geometry_support import footprint_valid, invert

    source_valid = np.ones((32, 32), dtype=bool)
    source_valid[8, 8] = False
    # The op maps source -> view with scale 16/32, so its inverse scales by 2.
    propagated = footprint_valid(
        source_valid,
        np.array(
            [[32 / 16, 0.0, 0.0], [0.0, 32 / 16, 0.0], [0.0, 0.0, 1.0]]
        ),
        (16, 16),
        radius=2.0,
        clamp=True,
    )
    assert not propagated.all()

    patch_cov, patch_valid = patch_support(
        trace, pixel_sampling_valid=propagated, grid_hw=(8, 8), patch_hw=(2, 2)
    )
    assert patch_cov.shape == (8, 8)
    # Source pixel (8, 8) resampled by 2 lands in output pixel (4, 4): patch (2, 2).
    assert not propagated[4, 4]
    assert not patch_valid[2, 2]
    assert patch_valid[0, 0]
    # Geometric coverage of that patch is still complete: coverage and validity
    # are separate records.
    assert float(patch_cov[2, 2]) == pytest.approx(1.0)


def test_support_agrees_with_the_real_resampler_footprint():
    """Conservative validity must cover every pixel the resampler actually moves.

    The oracle is the real operation itself: pad a small specimen with two
    different fill colours and observe which output pixels change. Every pixel
    that depends on the (invalid) fill must be marked unavailable.
    """
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, size=(12, 20, 3), dtype=np.uint8)
    image = Image.fromarray(base, mode="RGB")

    pad_black = GeometryOp(
        "pad",
        {
            "from_wh": [20, 12],
            "canvas_wh": [20, 20],
            "offset": [0, 4],
            "fill": [0, 0, 0],
        },
    )
    pad_red = GeometryOp(
        "pad",
        {
            "from_wh": [20, 12],
            "canvas_wh": [20, 20],
            "offset": [0, 4],
            "fill": [255, 0, 0],
        },
    )
    followers = (
        GeometryOp(
            "resize",
            {
                "from_wh": [20, 20],
                "to_wh": [9, 9],
                "interpolation": "bicubic",
                "antialias": True,
            },
        ),
        GeometryOp(
            "rotate",
            {
                "from_wh": [20, 20],
                "angle": 21.0,
                "center": [10.0, 10.0],
                "expand": False,
                "interpolation": "bilinear",
                "fill": [0, 0, 0],
                "canvas_wh": [20, 20],
            },
        ),
    )
    for follower in followers:
        _, trace = apply_geometry(image, [pad_black, follower], track=True)
        assert trace is not None
        _, valid = pixel_support(trace)
        assert valid.shape == (trace.view_wh[1], trace.view_wh[0])
        assert not valid.all()
        with_black, _ = apply_geometry(image, [pad_black, follower], track=False)
        with_red, _ = apply_geometry(image, [pad_red, follower], track=False)
        difference = np.abs(
            np.asarray(with_black).astype(np.float64)
            - np.asarray(with_red).astype(np.float64)
        ).sum(axis=-1)
        affected = difference > 1e-9
        assert affected.any()
        # Soundness: every pixel that depends on the invalid fill is unavailable.
        assert not valid[affected].any()
        # And the interior stays usable, so validity is not vacuously False.
        assert valid.any()


def test_trace_round_trips_and_rejects_inconsistent_records():
    image = asymmetric_image(40, 30)
    ops = evaluation_view_ops(image, name="whole-specimen-pad", size=32)
    trace = trace_operations((40, 30), ops)
    payload = trace.to_json()
    restored = GeometryTrace.from_json(payload)
    assert restored.source_wh == trace.source_wh
    assert restored.view_wh == trace.view_wh
    np.testing.assert_allclose(restored.source_to_view, trace.source_to_view)
    np.testing.assert_allclose(restored.view_to_source, trace.view_to_source)
    assert restored.support_version == "conservative-resampling-v1"

    broken = dict(payload)
    broken["view_to_source"] = np.eye(3).tolist()
    with pytest.raises(ValueError, match="not the inverse"):
        GeometryTrace.from_json(broken)

    singular = dict(payload)
    singular["source_to_view"] = np.zeros((3, 3)).tolist()
    with pytest.raises(ValueError, match="singular"):
        GeometryTrace.from_json(singular)

    extra = dict(payload)
    extra["unexpected"] = 1
    with pytest.raises(ValueError, match="do not match the v1 record"):
        GeometryTrace.from_json(extra)

    with pytest.raises(ValueError, match="realized canvas|records previous canvas"):
        trace_operations(
            (40, 30),
            [
                GeometryOp(
                    "crop",
                    {"from_wh": [40, 30], "left": 0, "top": 0, "width": 10, "height": 10},
                ),
                GeometryOp(
                    "resize",
                    {
                        "from_wh": [11, 11],
                        "to_wh": [5, 5],
                        "interpolation": "bicubic",
                        "antialias": True,
                    },
                ),
            ],
        )


def test_shear_is_rejected_rather_than_guessed():
    image = asymmetric_image(16, 16)
    op = GeometryOp(
        "affine",
        {
            "from_wh": [16, 16],
            "center": [8.0, 8.0],
            "angle": 0.0,
            "translate": [0.0, 0.0],
            "scale": 1.0,
            "shear": [0.2, 0.0],
            "interpolation": "bilinear",
            "fill": [0, 0, 0],
            "canvas_wh": [16, 16],
        },
    )
    with pytest.raises(ValueError, match="shear is not part of spatial v1"):
        apply_geometry(image, [op], track=True)


def test_unknown_operation_and_eval_name_are_rejected():
    image = asymmetric_image(8, 8)
    with pytest.raises(ValueError, match="unknown geometry operation"):
        GeometryOp("stretch", {})
    with pytest.raises(ValueError, match="unknown eval transform"):
        build_eval_view(image, name="random-crop", size=16)
