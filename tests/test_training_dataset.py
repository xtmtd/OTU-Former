from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torchvision import transforms

from otuformer.training.dataset import (
    EVAL_TRANSFORMS,
    MetricDataset,
    MultiCropDataset,
    _estimate_background_color,
    build_eval_transform,
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
