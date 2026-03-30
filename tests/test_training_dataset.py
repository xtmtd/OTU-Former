from pathlib import Path

import pandas as pd
from PIL import Image

from otuformer.training.dataset import MetricDataset, MultiCropDataset


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
