import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from otuformer.training.dataset import (
    EVAL_TRANSFORMS,
    FINETUNE_AUGMENTATIONS,
    ORIENTATION_POLICIES,
    PRETRAIN_AUGMENTATIONS,
    MetricDataset,
    MultiCropDataset,
    _PretrainViewTransform,
    _estimate_background_color,
    build_eval_transform,
    build_finetune_augmentation_config,
    build_pretrain_augmentation_config,
    center_crop_eval_transform,
    whole_specimen_pad_transform,
)


def make_dummy_images(tmp_path: Path, n: int = 4) -> Path:
    records = []
    for i in range(n):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 50, 0, 0)).save(img_path)
        records.append({"image": f"img_{i}.jpg"})
    csv_path = tmp_path / "images.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


def make_dummy_labeled_images(tmp_path: Path, n: int = 4) -> Path:
    records = []
    for i in range(n):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 50, 0, 0)).save(img_path)
        records.append({"image": f"img_{i}.jpg", "label": f"class_{i % 2}"})
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


def test_multicrop_dataset_returns_list(tmp_path: Path):
    csv_path = make_dummy_images(tmp_path)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    views = ds[0]
    assert isinstance(views, list)
    assert len(views) == 4


def test_multicrop_dataset_view_shapes(tmp_path: Path):
    csv_path = make_dummy_images(tmp_path)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    views = ds[0]
    for v in views[:2]:
        assert v.shape == (3, 32, 32)
    for v in views[2:]:
        assert v.shape == (3, 16, 16)


def test_metric_dataset_returns_image_label(tmp_path: Path):
    csv_path = make_dummy_labeled_images(tmp_path)
    ds = MetricDataset(csv_path=csv_path, images_dir=tmp_path, image_size=32)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)


def test_multicrop_dataset_resolves_image_in_subdir_by_filename(tmp_path: Path):
    subdir = tmp_path / "nested"
    subdir.mkdir(parents=True, exist_ok=True)
    img_path = subdir / "sample.jpg"
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(img_path)

    csv_path = tmp_path / "images.csv"
    pd.DataFrame([{"image": "sample.jpg"}]).to_csv(csv_path, index=False)

    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    views = ds[0]
    assert len(views) == 4


def test_metric_dataset_resolves_image_in_subdir_by_filename(tmp_path: Path):
    subdir = tmp_path / "nested"
    subdir.mkdir(parents=True, exist_ok=True)
    img_path = subdir / "sample.jpg"
    Image.new("RGB", (64, 64), color=(0, 255, 0)).save(img_path)

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([{"image": "sample.jpg", "label": "class_a"}]).to_csv(
        csv_path, index=False
    )

    ds = MetricDataset(csv_path=csv_path, images_dir=tmp_path, image_size=32)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert label == 0


def test_multicrop_dataset_uses_all_images_when_csv_missing(tmp_path: Path):
    root_img = tmp_path / "a.jpg"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    nested_img = nested_dir / "b.jpg"

    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(root_img)
    Image.new("RGB", (64, 64), color=(4, 5, 6)).save(nested_img)

    ds = MultiCropDataset(
        csv_path=None,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    assert len(ds) == 2


def test_center_crop_eval_transform_equals_old_composition():
    tf = center_crop_eval_transform(64)
    assert [type(t) for t in tf.transforms] == [
        transforms.Resize,
        transforms.CenterCrop,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert tf.transforms[0].size == 64
    assert tf.transforms[1].size == (64, 64)


def test_whole_specimen_pad_transform_preserves_aspect_ratio():
    img = Image.new("RGB", (8, 4))
    for x in range(4):
        for y in range(4):
            img.putpixel((x, y), (0, 128, 0))
    for x in range(4, 8):
        for y in range(4):
            img.putpixel((x, y), (0, 0, 255))

    tf = whole_specimen_pad_transform(32)
    out = tf(img)
    assert out.shape == (3, 32, 32)
    # green specimen half stays on the left, blue on the right: no distortion
    assert out[1, :, 2].mean() > out[2, :, 2].mean()  # col 2 is green-dominant
    assert out[2, :, 30].mean() > out[1, :, 30].mean()  # col 30 is blue-dominant


def test_estimate_background_color_returns_edge_median():
    img = Image.new("RGB", (6, 6), (255, 0, 0))
    for x in range(1, 5):
        for y in range(1, 5):
            img.putpixel((x, y), (0, 255, 0))
    assert _estimate_background_color(img) == (255, 0, 0)


def test_estimate_background_color_fallback_on_failure():
    assert _estimate_background_color(123) == (124, 116, 104)
    assert _estimate_background_color(Image.new("RGB", (1, 1))) == (124, 116, 104)


def test_build_eval_transform_maps_names_and_rejects_unknown():
    assert set(EVAL_TRANSFORMS) == {"center-crop", "whole-specimen-pad"}

    def spec(tf):
        return [(type(t).__name__, getattr(t, "size", None)) for t in tf.transforms]

    assert spec(build_eval_transform("center-crop", 64)) == spec(
        center_crop_eval_transform(64)
    )
    assert spec(build_eval_transform("whole-specimen-pad", 64)) == spec(
        whole_specimen_pad_transform(64)
    )
    with pytest.raises(ValueError, match="Unknown eval transform 'bogus'"):
        build_eval_transform("bogus", 64)


def test_pad_to_square_uses_edge_median_fill():
    from otuformer.training.dataset import pad_to_square

    img = Image.new("RGB", (6, 4), (10, 20, 30))
    padded = pad_to_square(img)
    assert padded.size == (6, 6)
    assert padded.getpixel((0, 0)) == (10, 20, 30)


def test_augmentation_profile_names_are_fixed():
    assert PRETRAIN_AUGMENTATIONS == ("global-barcode", "color-robust", "legacy")
    assert FINETUNE_AUGMENTATIONS == ("none", "conservative")
    assert ORIENTATION_POLICIES == ("invariant", "sensitive")


def test_builders_default_to_sensitive_orientation_policy():
    assert (
        build_pretrain_augmentation_config("global-barcode", 224, 96, 6)[
            "orientation_policy"
        ]
        == "sensitive"
    )
    assert (
        build_pretrain_augmentation_config("legacy", 224, 96, 6)[
            "orientation_policy"
        ]
        == "sensitive"
    )
    assert (
        build_finetune_augmentation_config("conservative", 224)["orientation_policy"]
        == "sensitive"
    )
    assert (
        build_finetune_augmentation_config("none", 224)["orientation_policy"]
        == "sensitive"
    )


def test_pretrain_profile_configs_are_fully_expanded():
    barcode = build_pretrain_augmentation_config(
        "global-barcode", 224, 96, 6, orientation_policy="invariant"
    )
    robust = build_pretrain_augmentation_config(
        "color-robust", 224, 96, 6, orientation_policy="invariant"
    )
    sensitive = build_pretrain_augmentation_config(
        "global-barcode", 224, 96, 6, orientation_policy="sensitive"
    )
    sensitive_robust = build_pretrain_augmentation_config(
        "color-robust", 224, 96, 6, orientation_policy="sensitive"
    )
    legacy = build_pretrain_augmentation_config("legacy", 224, 96, 6)
    sensitive_legacy = build_pretrain_augmentation_config(
        "legacy", 224, 96, 6, orientation_policy="sensitive"
    )

    assert barcode["global_crop"]["scale"] == [0.4, 1.0]
    assert barcode["local_crop"]["scale"] == [0.05, 0.4]
    assert barcode["crop_ratio"] == [0.75, 4 / 3]
    assert barcode["local_crops"] == 6
    assert barcode["rotation"] == {
        "degrees": [-180.0, 180.0],
        "interpolation": "bicubic",
        "expand": True,
        "fill": "edge-median-rgb",
    }
    assert barcode["profile"] == "global-barcode"
    assert barcode["orientation_policy"] == "invariant"
    assert barcode["horizontal_flip_probability"] == 0.5
    assert barcode["vertical_flip_probability"] == 0.0
    assert sensitive["orientation_policy"] == "sensitive"
    assert sensitive["rotation"]["degrees"] == [-15.0, 15.0]
    assert sensitive["horizontal_flip_probability"] == 0.0
    assert sensitive_robust["orientation_policy"] == "sensitive"
    assert sensitive_robust["rotation"]["degrees"] == [-15.0, 15.0]
    assert sensitive_robust["horizontal_flip_probability"] == 0.0
    assert sensitive_legacy["orientation_policy"] == "sensitive"
    assert sensitive_legacy["rotation"] == legacy["rotation"]
    assert sensitive_legacy["horizontal_flip_probability"] == 0.5
    assert barcode["color_jitter"] == {
        "probability": 0.8,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.1,
        "hue": 0.02,
    }
    assert barcode["grayscale_probability"] == 0.0
    assert barcode["blur"]["probabilities"] == [1.0, 0.1, 0.5]
    assert robust["color_jitter"]["brightness"] == 0.4
    assert robust["color_jitter"]["saturation"] == 0.2
    assert robust["grayscale_probability"] == 0.2
    assert legacy["rotation"]["expand"] is False
    assert legacy["rotation"]["fill"] == [0, 0, 0]
    assert legacy["vertical_flip_probability"] == 0.5
    assert legacy["blur"]["probabilities"] == [1.0, 1.0, 1.0]

    none = build_finetune_augmentation_config("none", image_size=32)
    conservative = build_finetune_augmentation_config(
        "conservative", image_size=32
    )
    sensitive_conservative = build_finetune_augmentation_config(
        "conservative", image_size=32, orientation_policy="sensitive"
    )
    assert none["image_size"] == 32
    assert conservative["image_size"] == 32
    assert sensitive_conservative["orientation_policy"] == "sensitive"
    assert sensitive_conservative["rotation"]["degrees"] == [-15.0, 15.0]
    assert sensitive_conservative["horizontal_flip_probability"] == 0.0
    assert conservative["transform_order"][-3:] == [
        "resize",
        "to_tensor",
        "normalize",
    ]


def test_unknown_augmentation_profiles_fail():
    with pytest.raises(ValueError, match="Unknown pretrain augmentation profile"):
        build_pretrain_augmentation_config("unknown", 224, 96, 6)
    with pytest.raises(ValueError, match="Unknown orientation policy"):
        build_pretrain_augmentation_config(
            "global-barcode", 224, 96, 6, orientation_policy="unknown"
        )
    with pytest.raises(ValueError, match="Unknown finetune augmentation profile"):
        build_finetune_augmentation_config("unknown", 224)


def test_expanded_augmentation_configs_are_json_serializable():
    for profile in PRETRAIN_AUGMENTATIONS:
        for policy in ORIENTATION_POLICIES:
            config = build_pretrain_augmentation_config(profile, 224, 96, 6, policy)
            json.loads(json.dumps(config))
    for profile in FINETUNE_AUGMENTATIONS:
        for policy in ORIENTATION_POLICIES:
            config = build_finetune_augmentation_config(profile, 224, policy)
            json.loads(json.dumps(config))


def test_builders_return_fresh_configs():
    first = build_pretrain_augmentation_config(
        "global-barcode", 224, 96, 6, orientation_policy="invariant"
    )
    first["global_crop"]["scale"].append(9.9)
    first["blur"]["probabilities"].append(9.9)
    first["rotation"]["degrees"].append(9.9)
    first["color_jitter"]["brightness"] = 9.9
    second = build_pretrain_augmentation_config(
        "global-barcode", 224, 96, 6, orientation_policy="invariant"
    )
    assert second["global_crop"]["scale"] == [0.4, 1.0]
    assert second["blur"]["probabilities"] == [1.0, 0.1, 0.5]
    assert second["rotation"]["degrees"] == [-180.0, 180.0]
    assert second["color_jitter"]["brightness"] == 0.2

    none_first = build_finetune_augmentation_config("none", 32)
    none_first["transform_order"].append("bogus")
    assert build_finetune_augmentation_config("none", 32)["transform_order"] == [
        "resize",
        "center_crop",
        "to_tensor",
        "normalize",
    ]


def test_new_profile_blur_kernels_resolve_to_odd_values_within_crop_size():
    config = build_pretrain_augmentation_config("global-barcode", 20, 6, 2)
    assert config["blur"]["global_kernel"] == 19
    assert config["blur"]["local_kernel"] == 5


def test_new_profiles_reject_crop_sizes_below_three():
    with pytest.raises(ValueError, match="crop size"):
        build_pretrain_augmentation_config("global-barcode", 2, 96, 6)
    with pytest.raises(ValueError, match="crop size"):
        build_pretrain_augmentation_config("global-barcode", 224, 2, 6)


def test_local_crops_zero_skips_local_size_validation():
    config = build_pretrain_augmentation_config("global-barcode", 224, 2, 0)
    assert config["local_crops"] == 0
    assert config["local_crop"]["size"] == 2


def test_multicrop_dataset_local_crops_zero_returns_two_views(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=2,  # small and unused when local_crops == 0
        local_crops=0,
    )
    views = ds[0]
    assert [tuple(view.shape) for view in views] == [(3, 32, 32), (3, 32, 32)]


@pytest.mark.parametrize("profile", PRETRAIN_AUGMENTATIONS)
def test_pretrain_profiles_return_fixed_shapes_for_rectangular_images(
    tmp_path, profile
):
    csv_path = make_dummy_images(tmp_path, n=1)
    Image.new("RGB", (91, 47), color=(240, 240, 240)).save(tmp_path / "img_0.jpg")
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
        augmentation_profile=profile,
    )
    views = ds[0]
    assert [tuple(view.shape) for view in views] == [
        (3, 32, 32),
        (3, 32, 32),
        (3, 16, 16),
        (3, 16, 16),
    ]


def test_multicrop_default_still_returns_six_local_views(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    views = MultiCropDataset(csv_path=csv_path, images_dir=tmp_path)[0]
    assert len(views) == 8


@pytest.mark.parametrize("profile", FINETUNE_AUGMENTATIONS)
def test_finetune_profiles_return_fixed_shape_for_rectangular_images(
    tmp_path, profile
):
    csv_path = make_dummy_labeled_images(tmp_path, n=1)
    Image.new("RGB", (91, 47), color=(240, 240, 240)).save(tmp_path / "img_0.jpg")
    image, label = MetricDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        image_size=32,
        augmentation_profile=profile,
    )[0]
    assert image.shape == (3, 32, 32)
    assert label == 0


@pytest.mark.parametrize("profile", ["global-barcode", "color-robust"])
def test_pretrain_view_transforms_are_separate_with_expected_blur_probabilities(
    tmp_path, profile
):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
        augmentation_profile=profile,
    )
    assert ds.global_tf1 is not ds.global_tf2
    assert ds.global_tf1 is not ds.local_tf
    assert ds.global_tf2 is not ds.local_tf
    assert ds.global_tf1.blur_probability == 1.0
    assert ds.global_tf2.blur_probability == 0.1
    assert ds.local_tf.blur_probability == 0.5
    assert ds.global_tf1.uses_edge_fill is True
    assert ds.global_tf1.post_rotation_resize is True
    assert ds.global_tf1.crop_size == 32
    assert ds.local_tf.crop_size == 16


def test_legacy_profile_keeps_historical_transform_structure(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
        augmentation_profile="legacy",
    )
    config = ds.augmentation_config
    assert config["rotation"]["expand"] is False
    assert config["rotation"]["fill"] == [0, 0, 0]
    assert config["vertical_flip_probability"] == 0.5
    assert config["horizontal_flip_probability"] == 0.5
    assert config["grayscale_probability"] == 0.2
    assert config["blur"]["probabilities"] == [1.0, 1.0, 1.0]
    assert "resize" not in config["transform_order"]
    for view in (ds.global_tf1, ds.global_tf2, ds.local_tf):
        assert view.uses_edge_fill is False
        assert view.rotation_expand is False
        assert view.post_rotation_resize is False
        assert view.vertical_flip_probability == 0.5
        assert view.grayscale_probability == 0.2
        assert view.rotation_fill == (0, 0, 0)
    assert ds.global_tf1.blur_kernel == 23
    assert ds.local_tf.blur_kernel == 7


def test_finetune_none_profile_keeps_deterministic_compose(tmp_path):
    csv_path = make_dummy_labeled_images(tmp_path, n=1)
    ds = MetricDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        image_size=32,
        augmentation_profile="none",
    )
    assert isinstance(ds.transform, transforms.Compose)
    assert [type(t) for t in ds.transform.transforms] == [
        transforms.Resize,
        transforms.CenterCrop,
        transforms.ToTensor,
        transforms.Normalize,
    ]


def test_conservative_config_has_no_removed_transforms():
    config = build_finetune_augmentation_config("conservative", image_size=32)
    assert "random_resized_crop" not in config["transform_order"]
    assert "blur" not in config
    assert config["grayscale_probability"] == 0.0
    assert config["solarization_probability"] == 0.0
    assert config["translation"] == [0.05, 0.05]
    assert config["scale"] == [0.9, 1.1]
    assert config["transform_order"][-3:] == ["resize", "to_tensor", "normalize"]


@pytest.mark.parametrize("policy", ORIENTATION_POLICIES)
def test_legacy_records_orientation_policy_without_changing_transforms(
    tmp_path, policy
):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
        augmentation_profile="legacy",
        orientation_policy=policy,
    )
    assert ds.orientation_policy == policy
    assert ds.augmentation_config["orientation_policy"] == policy
    assert ds.augmentation_config["rotation"]["degrees"] == [-180.0, 180.0]
    assert ds.augmentation_config["horizontal_flip_probability"] == 0.5
    assert ds.augmentation_config["vertical_flip_probability"] == 0.5
    for view in (ds.global_tf1, ds.global_tf2, ds.local_tf):
        assert view.rotation_degrees == (-180.0, 180.0)
        assert view.horizontal_flip_probability == 0.5


@pytest.mark.parametrize("profile", ["global-barcode", "color-robust", "legacy"])
def test_sensitive_orientation_policy_reaches_every_pretrain_view(tmp_path, profile):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
        augmentation_profile=profile,
        orientation_policy="sensitive",
    )
    assert ds.augmentation_config["orientation_policy"] == "sensitive"
    if profile == "legacy":
        expected_degrees = (-180.0, 180.0)
        expected_flip = 0.5
    else:
        expected_degrees = (-15.0, 15.0)
        expected_flip = 0.0
    assert ds.augmentation_config["rotation"]["degrees"] == list(expected_degrees)
    assert ds.augmentation_config["horizontal_flip_probability"] == expected_flip
    for view in (ds.global_tf1, ds.global_tf2, ds.local_tf):
        assert view.rotation_degrees == expected_degrees
        assert view.horizontal_flip_probability == expected_flip


def test_sensitive_orientation_policy_reaches_conservative_finetune(tmp_path):
    csv_path = make_dummy_labeled_images(tmp_path, n=1)
    ds = MetricDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        image_size=32,
        augmentation_profile="conservative",
        orientation_policy="sensitive",
    )
    config = ds.augmentation_config
    assert config["orientation_policy"] == "sensitive"
    assert config["rotation"]["degrees"] == [-15.0, 15.0]
    assert config["horizontal_flip_probability"] == 0.0
    assert ds.transform.rotation_degrees == (-15.0, 15.0)
    assert ds.transform.horizontal_flip_probability == 0.0


def test_build_finetune_conservative_sensitive_config():
    config = build_finetune_augmentation_config(
        "conservative", image_size=32, orientation_policy="sensitive"
    )
    assert config["rotation"]["degrees"] == [-15.0, 15.0]
    assert config["horizontal_flip_probability"] == 0.0


def _asymmetric_half_image(size: int = 32) -> Image.Image:
    image = Image.new("RGB", (size, size), (255, 0, 0))
    for x in range(size // 2, size):
        for y in range(size):
            image.putpixel((x, y), (0, 0, 255))
    return image


def _controlled_view(profile: str, policy: str) -> _PretrainViewTransform:
    """A policy-driven view transform with identity spatial geometry.

    Cropping, rotation, and resize are disabled so the only source of output
    variation is the profile's flip probabilities.
    """
    config = build_pretrain_augmentation_config(
        profile, 32, 16, 0, orientation_policy=policy
    )
    return _PretrainViewTransform(
        crop_size=32,
        scale=(1.0, 1.0),
        crop_ratio=(1.0, 1.0),
        rotation_degrees=(0.0, 0.0),
        rotation_expand=False,
        rotation_fill=(0, 0, 0),
        horizontal_flip_probability=config["horizontal_flip_probability"],
        vertical_flip_probability=config["vertical_flip_probability"],
        color_jitter={**config["color_jitter"], "probability": 0.0},
        grayscale_probability=0.0,
        blur_probability=0.0,
        blur_kernel=1,
        blur_sigma=tuple(config["blur"]["sigma"]),
        post_rotation_resize=False,
    )


def test_sensitive_policy_never_horizontally_mirrors():
    transform = _controlled_view("global-barcode", "sensitive")
    assert transform.horizontal_flip_probability == 0.0
    image = _asymmetric_half_image()
    reference = transform(image, (0, 0, 0))
    torch.manual_seed(1234)
    for _ in range(25):
        assert torch.allclose(transform(image, (0, 0, 0)), reference)


def test_legacy_policy_keeps_historical_flips():
    transform = _controlled_view("legacy", "sensitive")
    assert transform.horizontal_flip_probability == 0.5
    assert transform.vertical_flip_probability == 0.5
    image = _asymmetric_half_image()
    reference = transform(image, (0, 0, 0))
    torch.manual_seed(0)
    outputs = [transform(image, (0, 0, 0)) for _ in range(50)]
    assert any(not torch.allclose(out, reference) for out in outputs)


def test_finetune_none_skips_background_estimation(tmp_path, monkeypatch):
    csv_path = make_dummy_labeled_images(tmp_path, n=1)
    ds = MetricDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        image_size=32,
        augmentation_profile="none",
    )

    def fail_estimation(image):
        raise AssertionError("profile none must skip background estimation")

    monkeypatch.setattr(
        "otuformer.training.dataset._estimate_background_color", fail_estimation
    )
    image, label = ds[0]
    assert image.shape == (3, 32, 32)
    assert label == 0


def test_pretrain_view_shape_mismatch_raises_value_error(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    ds.global_tf2 = lambda image, fill: torch.zeros(3, 8, 8)
    with pytest.raises(ValueError) as excinfo:
        ds[0]
    message = str(excinfo.value)
    assert "(3, 32, 32)" in message
    assert "(3, 8, 8)" in message


def test_pretrain_non_tensor_view_raises_value_error(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    ds.global_tf2 = lambda image, fill: [1, 2, 3]
    with pytest.raises(ValueError) as excinfo:
        ds[0]
    message = str(excinfo.value)
    assert "(3, 32, 32)" in message
    assert "None" in message


def test_finetune_conservative_shape_mismatch_raises_value_error(tmp_path):
    csv_path = make_dummy_labeled_images(tmp_path, n=1)
    ds = MetricDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        image_size=32,
        augmentation_profile="conservative",
    )
    ds.transform = lambda image, fill: torch.zeros(3, 8, 8)
    with pytest.raises(ValueError) as excinfo:
        ds[0]
    message = str(excinfo.value)
    assert "(3, 32, 32)" in message
    assert "(3, 8, 8)" in message


def test_datasets_reject_unknown_profiles_and_policies(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    with pytest.raises(ValueError, match="Unknown pretrain augmentation profile"):
        MultiCropDataset(
            csv_path=csv_path, images_dir=tmp_path, augmentation_profile="unknown"
        )
    with pytest.raises(ValueError, match="Unknown orientation policy"):
        MultiCropDataset(
            csv_path=csv_path, images_dir=tmp_path, orientation_policy="unknown"
        )
    labeled_csv = make_dummy_labeled_images(tmp_path, n=1)
    with pytest.raises(ValueError, match="Unknown finetune augmentation profile"):
        MetricDataset(
            csv_path=labeled_csv, images_dir=tmp_path, augmentation_profile="unknown"
        )
    with pytest.raises(ValueError, match="Unknown orientation policy"):
        MetricDataset(
            csv_path=labeled_csv, images_dir=tmp_path, orientation_policy="unknown"
        )


def test_all_pretrain_views_share_one_background_estimate(tmp_path, monkeypatch):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    estimates = []

    def fake_estimate(image):
        estimates.append(image)
        return (11, 22, 33)

    monkeypatch.setattr(
        "otuformer.training.dataset._estimate_background_color", fake_estimate
    )
    recorded = []

    def recorder(size):
        def view(image, fill):
            recorded.append(fill)
            return torch.zeros(3, size, size)

        return view

    ds.global_tf1 = recorder(32)
    ds.global_tf2 = recorder(32)
    ds.local_tf = recorder(16)
    views = ds[0]
    assert [tuple(view.shape) for view in views] == [
        (3, 32, 32),
        (3, 32, 32),
        (3, 16, 16),
        (3, 16, 16),
    ]
    assert recorded == [(11, 22, 33)] * 4
    assert len(estimates) == 1


def test_legacy_views_skip_background_estimation(tmp_path, monkeypatch):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
        augmentation_profile="legacy",
    )
    estimates = []

    def fake_estimate(image):
        estimates.append(image)
        return (11, 22, 33)

    monkeypatch.setattr(
        "otuformer.training.dataset._estimate_background_color", fake_estimate
    )
    views = ds[0]
    assert [tuple(view.shape) for view in views] == [
        (3, 32, 32),
        (3, 32, 32),
        (3, 16, 16),
        (3, 16, 16),
    ]
    assert estimates == []


def test_pretrain_view_transform_uses_supplied_edge_fill(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    view = ds.global_tf1
    image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    torch.manual_seed(0)
    first = view(image, (11, 22, 33))
    torch.manual_seed(0)
    second = view(image, (200, 200, 200))
    assert not torch.equal(first, second)


def test_legacy_view_transform_ignores_supplied_edge_fill(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
        augmentation_profile="legacy",
    )
    view = ds.global_tf1
    image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    torch.manual_seed(0)
    first = view(image, (11, 22, 33))
    torch.manual_seed(0)
    second = view(image, (200, 200, 200))
    assert torch.equal(first, second)


def test_pretrain_view_transform_is_picklable(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    for view in (ds.global_tf1, ds.global_tf2, ds.local_tf):
        pickle.loads(pickle.dumps(view))
