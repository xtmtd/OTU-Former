"""Tracked variants of the existing random view transforms.

Tracking must not change a single pixel or a single RNG draw: the same samplers
run in the same order, and only the executed operations are recorded.
"""

from __future__ import annotations

import pickle
import random

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from otuformer.training.dataset import (
    _ConservativeTransform,
    _PretrainViewTransform,
    build_finetune_augmentation_config,
    build_pretrain_augmentation_config,
)
from otuformer.vision.geometry import (
    GeometryOp,
    apply_geometry,
    map_points,
    patch_support,
    pixel_support,
    trace_operations,
)

BICUBIC = InterpolationMode.BICUBIC


def asymmetric_image(width: int, height: int, *, baseline: int = 0) -> Image.Image:
    image = Image.new("RGB", (width, height), (baseline, baseline, baseline))
    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), (x * 7 % 256, y * 11 % 256, (x * 3 + y * 5) % 256))
    return image


def marker_image(width: int = 100, height: int = 100, color=(255, 0, 0)) -> Image.Image:
    image = Image.new("RGB", (width, height), (0, 0, 0))
    image.putpixel((70, 50), color)
    return image


# --------------------------------------------------------------- references --
def old_pretrain_view(
    transform: _PretrainViewTransform, image: Image.Image, fill: tuple[int, int, int]
) -> torch.Tensor:
    """Test-local copy of the historical ``__call__`` composition.

    It drives torchvision's own transform objects, so a rewrite that only looks
    right through our helper cannot pass this comparison.
    """
    image = transform._crop(image)
    angle = transforms.RandomRotation.get_params(transform._rotation_degrees)
    image = TF.rotate(
        image,
        angle,
        interpolation=BICUBIC,
        expand=transform._rotation_expand,
        fill=tuple(fill) if transform._uses_edge_fill else transform._fixed_fill,
    )
    if transform._post_rotation_resize:
        image = TF.resize(
            image, [transform._crop_size, transform._crop_size],
            interpolation=BICUBIC, antialias=True,
        )
    for flip in (transform._horizontal_flip, transform._vertical_flip):
        if flip is not None:
            image = flip(image)
    for appearance in (transform._color, transform._grayscale, transform._blur):
        if appearance is not None:
            image = appearance(image)
    return transform._normalize(transform._to_tensor(image))


def old_conservative_view(
    transform: _ConservativeTransform, image: Image.Image, fill: tuple[int, int, int]
) -> torch.Tensor:
    """Test-local copy of the historical conservative ``__call__``."""
    from otuformer.training.dataset import _pad_to_safe_square

    canvas = _pad_to_safe_square(
        image,
        fill,
        max_scale=transform._safe_canvas_max_scale,
        max_translation_fraction=transform._safe_canvas_max_translation_fraction,
    )
    angle, translate, scale, shear = transforms.RandomAffine.get_params(
        degrees=list(transform._rotation_degrees),
        translate=list(transform._translate),
        scale_ranges=list(transform._scale),
        shears=None,
        img_size=canvas.size,
    )
    image = TF.affine(
        canvas, angle=angle, translate=translate, scale=scale, shear=shear,
        interpolation=BICUBIC, fill=list(fill),
    )
    if transform._horizontal_flip is not None:
        image = transform._horizontal_flip(image)
    if transform._color is not None:
        image = transform._color(image)
    image = transform._resize(image)
    return transform._normalize(transform._to_tensor(image))


def seeded(seed: int, call):
    """Run ``call`` from one fixed RNG point and return value plus final states."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    value = call()
    return value, torch.random.get_rng_state().clone(), random.getstate()


def pretrain_transform(**overrides) -> _PretrainViewTransform:
    settings = dict(
        crop_size=24,
        scale=(0.8, 1.0),
        crop_ratio=(0.9, 1.1),
        rotation_degrees=(-15.0, 15.0),
        rotation_expand=False,
        rotation_fill="edge-median-rgb",
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.5,
        color_jitter={
            "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1,
            "probability": 0.8,
        },
        grayscale_probability=0.2,
        blur_probability=0.5,
        blur_kernel=3,
        blur_sigma=(0.1, 2.0),
        post_rotation_resize=True,
    )
    settings.update(overrides)
    return _PretrainViewTransform(**settings)


@pytest.mark.parametrize("profile", ["legacy", "global-barcode", "color-robust"])
@pytest.mark.parametrize("orientation", ["sensitive", "invariant"])
def test_profile_view_tracking_preserves_tensor_and_rng(profile, orientation):
    config = build_pretrain_augmentation_config(
        profile, 32, 16, 2, orientation_policy=orientation
    )
    blur = config["blur"]["probabilities"][0]
    transform = _PretrainViewTransform(
        crop_size=int(config["global_crop"]["size"]),
        scale=tuple(float(v) for v in config["global_crop"]["scale"]),
        crop_ratio=tuple(float(v) for v in config["crop_ratio"]),
        rotation_degrees=tuple(float(v) for v in config["rotation"]["degrees"]),
        rotation_expand=bool(config["rotation"]["expand"]),
        rotation_fill=config["rotation"]["fill"],
        horizontal_flip_probability=float(config["horizontal_flip_probability"]),
        vertical_flip_probability=float(config["vertical_flip_probability"]),
        color_jitter=config["color_jitter"],
        grayscale_probability=float(config["grayscale_probability"]),
        blur_probability=float(blur),
        blur_kernel=int(config["blur"]["global_kernel"]),
        blur_sigma=tuple(float(v) for v in config["blur"]["sigma"]),
        post_rotation_resize="resize" in config["transform_order"],
    )
    image = asymmetric_image(48, 36)
    fill = (12, 34, 56)

    reference, ref_state, ref_python_state = seeded(
        1234, lambda: old_pretrain_view(transform, image, fill)
    )
    plain, plain_state, plain_python_state = seeded(
        1234, lambda: transform(image, fill)
    )
    tracked, tracked_state, tracked_python_state = seeded(
        1234, lambda: transform.transform_with_geometry(image, fill)[0]
    )

    torch.testing.assert_close(plain, reference)
    torch.testing.assert_close(tracked, reference)
    assert torch.equal(plain_state, ref_state)
    assert torch.equal(tracked_state, ref_state)
    assert plain_python_state == ref_python_state == tracked_python_state


@pytest.mark.parametrize("params", [(), ("hflip",), ("vflip",), ("hflip", "vflip")])
@pytest.mark.parametrize("expand", [False, True])
def test_crop_rotate_resize_flip_sequence_matches_reference(params, expand):
    transform = pretrain_transform(
        rotation_degrees=(-20.0, 20.0),
        rotation_expand=expand,
        horizontal_flip_probability=1.0 if "hflip" in params else 0.0,
        vertical_flip_probability=1.0 if "vflip" in params else 0.0,
        color_jitter={**pretrain_transform()._color.__dict__.get("transform", {}), }
        if False
        else {
            "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0,
            "probability": 0.0,
        },
        grayscale_probability=0.0,
        blur_probability=0.0,
    )
    image = asymmetric_image(57, 41)
    fill = (5, 6, 7)
    (tensor, trace), _, _ = seeded(
        99, lambda: transform.transform_with_geometry(image, fill)
    )
    reference, ref_state, _ = seeded(99, lambda: old_pretrain_view(transform, image, fill))
    torch.testing.assert_close(tensor, reference)
    assert tensor.shape == (3, 24, 24)
    assert tuple(trace.view_wh) == (24, 24)
    kinds = [op.kind for op in trace.operations]
    assert kinds[0] == "crop" and kinds[1] == "resize" and "rotate" in kinds
    assert "hflip" in kinds if "hflip" in params else True
    assert ("hflip" in kinds) == ("hflip" in params)
    assert ("vflip" in kinds) == ("vflip" in params)


def test_conservative_tracking_preserves_tensor_and_rng():
    config = build_finetune_augmentation_config("conservative", image_size=32)
    transform = _ConservativeTransform(config)
    image = asymmetric_image(48, 30)
    fill = (40, 30, 20)

    reference, ref_state, ref_python = seeded(7, lambda: old_conservative_view(transform, image, fill))
    plain, plain_state, plain_python = seeded(7, lambda: transform(image, fill))
    (tracked, trace), tracked_state, tracked_python = seeded(
        7, lambda: transform.transform_with_geometry(image, fill)
    )
    torch.testing.assert_close(plain, reference)
    torch.testing.assert_close(tracked, reference)
    assert torch.equal(plain_state, ref_state)
    assert torch.equal(tracked_state, ref_state)
    assert plain_python == ref_python == tracked_python

    kinds = [op.kind for op in trace.operations]
    assert kinds[0] == "pad" and "affine" in kinds and kinds[-1] == "resize"
    assert tuple(trace.view_wh) == (32, 32)
    affine = next(op for op in trace.operations if op.kind == "affine")
    # Raw library parameters are recorded, including the zero shear this profile uses.
    assert set(affine.params) >= {"angle", "translate", "scale", "shear", "center"}
    assert affine.params["shear"] == [0.0, 0.0]
    assert transform.rotation_degrees == tuple(
        float(v) for v in config["rotation"]["degrees"]
    )


def test_tracked_transform_pickles_and_still_matches():
    transform = pretrain_transform(color_jitter={
        "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1, "probability": 1.0,
    })
    clone = pickle.loads(pickle.dumps(transform))
    image = asymmetric_image(40, 40)
    fill = (1, 2, 3)
    left, _, _ = seeded(11, lambda: transform(image, fill))
    right, _, _ = seeded(11, lambda: clone(image, fill))
    torch.testing.assert_close(left, right)
    (tracked, trace), _, _ = seeded(11, lambda: clone.transform_with_geometry(image, fill))
    torch.testing.assert_close(tracked, right)
    assert tuple(trace.view_wh) == (24, 24)

    conservative = _ConservativeTransform(
        build_finetune_augmentation_config("conservative", image_size=32)
    )
    clone = pickle.loads(pickle.dumps(conservative))
    left, _, _ = seeded(3, lambda: conservative(image, fill))
    right, _, _ = seeded(3, lambda: clone(image, fill))
    torch.testing.assert_close(left, right)


# ------------------------------------------------------------------ geometry --
def test_signed_marker_rotation_then_translation():
    rotate = GeometryOp(
        "rotate",
        {
            "from_wh": [100, 100],
            "angle": 90.0,
            "center": [50.0, 50.0],
            "expand": False,
            "interpolation": "bicubic",
            "fill": [0, 0, 0],
        },
    )
    translate = GeometryOp(
        "affine",
        {
            "from_wh": [100, 100],
            "center": [50.0, 50.0],
            "angle": 0.0,
            "translate": [10.0, 0.0],
            "scale": 1.0,
            "shear": [0.0, 0.0],
            "interpolation": "bicubic",
            "fill": [0, 0, 0],
            "canvas_wh": [100, 100],
        },
    )
    _, rotated = apply_geometry(marker_image(), [rotate], track=True)
    np.testing.assert_allclose(
        map_points(rotated, np.array([[70.0, 50.0]]))[0], [50.0, 30.0], atol=1e-9
    )
    _, combined = apply_geometry(marker_image(), [rotate, translate], track=True)
    np.testing.assert_allclose(
        map_points(combined, np.array([[70.0, 50.0]]))[0], [60.0, 30.0], atol=1e-9
    )


@pytest.mark.parametrize(
    "operations, expected_view",
    [
        (
            [
                GeometryOp(
                    "crop",
                    {"from_wh": [61, 47], "left": 5, "top": 3, "width": 40, "height": 30},
                )
            ],
            (40, 30),
        ),
        (
            [
                GeometryOp(
                    "crop",
                    {"from_wh": [61, 47], "left": 5, "top": 3, "width": 40, "height": 30},
                ),
                GeometryOp(
                    "rotate",
                    {
                        "angle": -15.0,
                        "center": [20.0, 15.0],
                        "expand": True,
                        "interpolation": "bicubic",
                        "fill": [0, 0, 0],
                    },
                ),
            ],
            None,  # an expanding rotation decides its own canvas
        ),
    ],
)
def test_odd_canvases_and_expansion_are_recorded(operations, expected_view):
    _, trace = apply_geometry(asymmetric_image(61, 47), operations, track=True)
    assert trace is not None
    if expected_view is not None:
        assert tuple(trace.view_wh) == expected_view
    else:
        # The realized canvas is recorded instead of re-derived, and the recorded
        # operation list reproduces the same view geometry.
        replay = trace_operations(trace.source_wh, trace.operations)
        assert tuple(replay.view_wh) == tuple(trace.view_wh)
        np.testing.assert_allclose(replay.source_to_view, trace.source_to_view)


def test_intermediate_clipping_never_resurrects_content():
    operations = [
        GeometryOp(
            "crop",
            {"from_wh": [64, 64], "left": 16, "top": 16, "width": 32, "height": 32},
        ),
        GeometryOp(
            "rotate",
            {
                "angle": 10.0,
                "center": [16.0, 16.0],
                "expand": False,
                "interpolation": "bicubic",
                "fill": [0, 0, 0],
            },
        ),
    ]
    _, trace = apply_geometry(asymmetric_image(64, 64), operations, track=True)
    coverage, valid = pixel_support(trace)
    # Everything outside the crop is gone: the composed map still inverts, but
    # support there is zero and no patch can be sampled from it.
    outside = map_points(trace, np.array([[2.0, 2.0]]))[0]
    assert outside[0] < 0.0 or outside[1] < 0.0
    assert float(coverage.min()) == pytest.approx(0.0)
    assert not valid[0, 0]
    assert valid[16, 16]
    patch_coverage, patch_valid = patch_support(
        trace, pixel_sampling_valid=valid, grid_hw=(2, 2), patch_hw=(16, 16)
    )
    assert patch_coverage.shape == (2, 2) and patch_valid.dtype == np.bool_


def test_scaled_and_translated_canvas_can_lose_all_overlap():
    operations = [
        GeometryOp(
            "affine",
            {
                "from_wh": [64, 64],
                "center": [32.0, 32.0],
                "angle": 0.0,
                "translate": [200.0, 0.0],
                "scale": 0.9,
                "shear": [0.0, 0.0],
                "interpolation": "bicubic",
                "fill": [0, 0, 0],
                "canvas_wh": [64, 64],
            },
        )
    ]
    _, trace = apply_geometry(asymmetric_image(64, 64), operations, track=True)
    coverage, valid = pixel_support(trace)
    np.testing.assert_allclose(coverage, 0.0)
    assert not valid.any()


@pytest.mark.parametrize("scale", [0.9, 1.1])
def test_scaled_view_support_is_a_fraction_not_a_mask(scale):
    operations = [
        GeometryOp(
            "affine",
            {
                "from_wh": [64, 64],
                "center": [32.0, 32.0],
                "angle": 0.0,
                "translate": [0.0, 0.0],
                "scale": scale,
                "shear": [0.0, 0.0],
                "interpolation": "bicubic",
                "fill": [0.0, 0.0, 0.0],
                "canvas_wh": [64, 64],
            },
        )
    ]
    _, trace = apply_geometry(asymmetric_image(64, 64), operations, track=True)
    coverage, _ = pixel_support(trace)
    # The centre is always fully covered.
    assert coverage[32, 32] == pytest.approx(1.0)
    if scale > 1.0:
        # Enlarged content covers the whole canvas.
        np.testing.assert_allclose(coverage, 1.0)
        return
    # Shrinking leaves an uncovered border and a fractional boundary ring, and
    # the covered area follows scale^2 rather than a Boolean mask.
    assert float(coverage[0, 0]) == 0.0
    fractional = [
        float(value) for value in coverage.ravel() if 0.0 < float(value) < 1.0
    ]
    assert fractional
    expected = scale**2 * 64 * 64
    assert float(coverage.sum()) == pytest.approx(expected, rel=0.02)


def test_tracked_view_support_matches_the_real_resampler():
    """A tracked training view's validity must cover the pixels the fill touches."""
    transform = _ConservativeTransform(
        build_finetune_augmentation_config("conservative", image_size=32)
    )
    image = asymmetric_image(48, 30, baseline=0)
    black_fill = (0, 0, 0)
    red_fill = (255, 0, 0)

    torch.manual_seed(5)
    _, trace = transform.transform_with_geometry(image, black_fill)
    _, valid = pixel_support(trace)
    assert valid.shape == (32, 32)
    assert not valid.all()

    # Oracle: change only the fill colour and observe which output pixels change.
    probe = asymmetric_image(48, 30, baseline=0)
    reference, _ = apply_geometry(probe, trace.operations, track=False)
    recolored, _ = apply_geometry(
        probe,
        tuple(
            GeometryOp(
                op.kind,
                {**op.params, "fill": list(red_fill)} if "fill" in op.params else dict(op.params),
            )
            for op in trace.operations
        ),
        track=False,
    )
    difference = np.abs(
        np.asarray(reference).astype(np.float64) - np.asarray(recolored).astype(np.float64)
    ).sum(axis=-1)
    affected = difference > 1e-9
    assert affected.any()
    # Soundness: every pixel that depends on the (invalid) fill is unavailable.
    assert not valid[affected].any()
    assert valid.any()
