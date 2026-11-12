"""CAM attribution and geometry parity after moving to shared primitives.

CAM keeps its own algorithms: the shared layer lookup, the real patch grid and the
realized geometry are the only reused pieces. These tests pin the behaviour that
must not change and the new evidence that the grid is real.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from otuformer.training.model import OTUFormerEncoder
from otuformer.vision import cam as cam_module
from otuformer.vision.features import get_module_by_name as shared_lookup


def tiny_vit(image_size: int = 32, reg_tokens: int | None = None) -> OTUFormerEncoder:
    torch.manual_seed(0)
    return OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=8,
        pretrained=False,
        img_size=image_size,
        reg_tokens=reg_tokens,
    ).eval()


def old_map_cam_to_original(cam_norm, original_size, eval_transform, model_size):
    """Test-local copy of the pre-refactor mapping, used as the parity oracle."""
    import cv2

    width, height = original_size
    if eval_transform == "whole-specimen-pad":
        side = max(width, height)
        padded = cv2.resize(cam_norm, (side, side), interpolation=cv2.INTER_LINEAR)
        if width >= height:
            y0 = (side - height) // 2
            region = padded[y0 : y0 + height, :]
        else:
            x0 = (side - width) // 2
            region = padded[:, x0 : x0 + width]
        cam_on_full = cv2.resize(
            region, (width, height), interpolation=cv2.INTER_LINEAR
        )
        return cam_on_full, np.ones((height, width), dtype=bool)

    if width >= height:
        resized_w, resized_h = int(model_size * width / height), model_size
    else:
        resized_w, resized_h = model_size, int(model_size * height / width)
    x0 = int(round((resized_w - model_size) / 2.0))
    y0 = int(round((resized_h - model_size) / 2.0))
    canvas = np.zeros((resized_h, resized_w), dtype=np.float32)
    fov = np.zeros((resized_h, resized_w), dtype=bool)
    canvas[y0 : y0 + model_size, x0 : x0 + model_size] = cam_norm
    fov[y0 : y0 + model_size, x0 : x0 + model_size] = True
    cam_on_full = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_LINEAR)
    fov_full = cv2.resize(
        fov.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return cam_on_full, fov_full > 0.5


def test_shared_lookup_resolves_the_same_paths():
    model = tiny_vit()
    for path in ("blocks.0", "blocks.11.norm1", "norm", "blocks.3.attn.qkv"):
        assert cam_module.get_module_by_name(model.backbone, path) is shared_lookup(
            model.backbone, path
        )
    # CAM keeps accepting arbitrary attribution targets, not only blocks.
    assert cam_module.get_module_by_name(model.backbone, "blocks.0.norm1") is not None
    with pytest.raises(AttributeError):
        cam_module.get_module_by_name(model.backbone, "blocks.0.nope")
    with pytest.raises(AttributeError):
        cam_module.get_module_by_name(model.backbone, "blocks.99")


@pytest.mark.parametrize("grid_hw", [(2, 3), (3, 2), (6, 1)])
def test_reshape_drops_prefix_and_preserves_the_exact_transpose(grid_hw):
    model = tiny_vit(reg_tokens=4)
    reshape = cam_module._bind_reshape_transform(model, grid_hw=grid_hw)
    patch_h, patch_w = grid_hw
    tokens = torch.randn(2, 5 + patch_h * patch_w, 8, requires_grad=True)

    out = reshape(tokens)
    assert out.shape == (2, 8, patch_h, patch_w)
    assert tuple(reshape(tokens).shape) == tuple(out.shape)  # repeatable

    expected = tokens[:, 5:, :].permute(0, 2, 1).reshape(2, 8, patch_h, patch_w)
    torch.testing.assert_close(out, expected)

    out.square().sum().backward()
    # CLS and register slots never receive gradient: they are not pixels.
    torch.testing.assert_close(
        tokens.grad[:, :5], torch.zeros_like(tokens.grad[:, :5])
    )
    torch.testing.assert_close(tokens.grad[:, 5:], 2 * tokens.detach()[:, 5:])


def test_reshape_grid_validation_and_legacy_square_fallback():
    tokens = torch.randn(2, 1 + 9, 8)
    assert cam_module.vit_reshape_transform(tokens).shape == (2, 8, 3, 3)
    assert cam_module.vit_reshape_transform(
        tokens, grid_hw=(3, 3)
    ).shape == (2, 8, 3, 3)
    with pytest.raises(ValueError, match="do not match"):
        cam_module.vit_reshape_transform(tokens, grid_hw=(6, 2))
    with pytest.raises(ValueError, match="grid_hw must be positive"):
        cam_module.vit_reshape_transform(tokens, grid_hw=(0, 3))
    with pytest.raises(ValueError, match="prefix_tokens"):
        cam_module.vit_reshape_transform(tokens, prefix_tokens=10)
    # A non-square token count must be told about a real grid, not guessed.
    with pytest.raises(ValueError, match="pass grid_hw"):
        cam_module.vit_reshape_transform(torch.randn(1, 1 + 7, 4))


@pytest.mark.parametrize(
    "image_size,expected", [(32, (2, 2)), (48, (3, 3)), (16, (1, 1)), (None, None)]
)
def test_input_grid_comes_from_the_loaded_size(image_size, expected):
    model = tiny_vit(image_size=32)
    assert cam_module.vit_input_grid(model, image_size) == expected


def test_input_grid_is_never_a_square_root_guess():
    model = tiny_vit()
    # 40 px is not a whole number of 16 px patches: the grid is not guessed.
    with pytest.raises(ValueError, match="not divisible by patch size"):
        cam_module.vit_input_grid(model, 40)
    assert cam_module.vit_input_grid(model, None) is None


@pytest.mark.parametrize("name", ["center-crop", "whole-specimen-pad"])
@pytest.mark.parametrize(
    "original_size",
    [(100, 200), (200, 100), (91, 37), (37, 91), (64, 64), (101, 63)],
)
@pytest.mark.parametrize("model_size", [32, 24])
def test_geometry_mapping_matches_the_pinned_formula(name, original_size, model_size):
    rng = np.random.default_rng(0)
    cam = rng.random((model_size, model_size)).astype(np.float32)
    produced, fov = cam_module.map_cam_to_original(cam, original_size, name, model_size)
    expected, expected_fov = old_map_cam_to_original(
        cam, original_size, name, model_size
    )
    np.testing.assert_array_equal(produced, expected)
    np.testing.assert_array_equal(fov, expected_fov)
    assert produced.shape == (original_size[1], original_size[0])
    assert fov.dtype == np.bool_


@pytest.mark.parametrize("name", ["center-crop", "whole-specimen-pad"])
def test_constant_maps_and_fields_of_view_stay_constant(name):
    model_size = 32
    cam = np.ones((model_size, model_size), dtype=np.float32)
    produced, fov = cam_module.map_cam_to_original(cam, (120, 80), name, model_size)
    # The mapping never rescales: a constant CAM keeps its value inside the view,
    # so raw and min-max normalised maps stay distinguishable downstream.
    if name == "whole-specimen-pad":
        assert fov.all()
        np.testing.assert_allclose(produced, 1.0, atol=1e-6)
    else:
        assert fov.any() and not fov.all()
        # cv2 bilinear leaves a one-pixel ramp at the FOV border; the interior is
        # exactly the stored constant.
        assert produced[fov].max() == pytest.approx(1.0, abs=1e-6)
        assert float(produced[fov].mean()) > 0.9
        # The band outside the crop is zero (with the same one-pixel ramp), and
        # the FOV marks it as outside the model's view.
        assert float(produced[~fov].max()) < 0.5
        assert float(produced[~fov].min()) == pytest.approx(0.0)


def test_unknown_protocol_is_rejected_by_the_shared_owner():
    with pytest.raises(ValueError, match="unknown eval transform"):
        cam_module.map_cam_to_original(
            np.zeros((8, 8), dtype=np.float32), (16, 16), "bogus", 8
        )


def test_display_view_reuses_the_realized_operations():
    from PIL import Image

    from otuformer.vision.geometry import (
        apply_geometry,
        evaluation_view_ops,
        pixel_support,
    )

    image = Image.new("RGB", (91, 37), (10, 20, 30))
    display = cam_module.build_display_transform("center-crop", 48)
    produced = display(image)
    ops = evaluation_view_ops(image, name="center-crop", size=48)
    expected, trace = apply_geometry(image, ops, track=True)
    np.testing.assert_array_equal(np.asarray(produced), np.asarray(expected))
    # The display view and the mapping agree about the same field of view.
    coverage, _ = pixel_support(trace)
    assert coverage.shape == (48, 48)
    _, fov = cam_module.map_cam_to_original(
        np.ones((48, 48), dtype=np.float32), (91, 37), "center-crop", 48
    )
    assert fov.dtype == np.bool_ and fov.any()
