"""Datasets for SSL pretraining and metric learning fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

# 8-bit ImageNet mean, used as the background fill fallback when edge-median
# estimation fails (e.g. degenerate or empty images).
_BACKGROUND_FILL_FALLBACK = (124, 116, 104)


def _supports_recursive_lookup(image_ref: str) -> bool:
    path = Path(image_ref)
    return not path.is_absolute()


def _build_recursive_index(
    images_dir: Path,
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    for path in images_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(images_dir).as_posix()
        by_relative[rel] = path
        by_name.setdefault(path.name, []).append(path)
    return by_relative, by_name


def _resolve_image_path(
    images_dir: Path,
    image_ref: str,
    by_relative: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> Path:
    ref = Path(image_ref)
    if ref.is_absolute() and ref.exists():
        return ref

    direct = images_dir / ref
    if direct.exists():
        return direct

    if not _supports_recursive_lookup(image_ref):
        return direct

    normalized_ref = ref.as_posix().lstrip("./")
    if normalized_ref in by_relative:
        return by_relative[normalized_ref]

    matches = by_name.get(ref.name, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matched = ", ".join(str(m.relative_to(images_dir)) for m in matches[:3])
        raise FileNotFoundError(
            f"Ambiguous image reference '{image_ref}' under {images_dir}: {matched}"
        )
    return direct


def _make_global_transform(crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.4, 1.0),
                interpolation=Image.BICUBIC,
            ),
            transforms.RandomRotation(
                degrees=180,
                interpolation=Image.BICUBIC,
                expand=False,
                fill=0,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _make_local_transform(crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.05, 0.4),
                interpolation=Image.BICUBIC,
            ),
            transforms.RandomRotation(
                degrees=180,
                interpolation=Image.BICUBIC,
                expand=False,
                fill=0,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def center_crop_eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic center-crop evaluation protocol.

    Resize the shorter side to ``image_size`` (bicubic), center-crop to a
    square of ``image_size``, then tensorize and apply ImageNet normalization.
    This is the shared deterministic preprocessing used by extraction, periodic
    evaluation, fine-tuning ``none``, and CAM.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# Backward-compatible alias during the transition to the named protocol.
_make_eval_transform = center_crop_eval_transform


def _estimate_background_color(image: Image.Image) -> tuple[int, int, int]:
    """Per-channel median of all four outer edges of the image.

    Returns the documented fallback ``(124, 116, 104)`` on any conversion or
    shape failure.
    """
    try:
        arr = np.asarray(image.convert("RGB"))
        if (
            arr.ndim != 3
            or arr.shape[2] != 3
            or arr.shape[0] < 2
            or arr.shape[1] < 2
        ):
            return _BACKGROUND_FILL_FALLBACK
        edges = np.concatenate(
            [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
        )
        return tuple(int(v) for v in np.median(edges, axis=0))
    except Exception:
        return _BACKGROUND_FILL_FALLBACK


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad the image to a square canvas with the edge-median fill.

    Preserves aspect ratio and introduces no distortion; the specimen is
    centered on the square canvas.
    """
    width, height = image.size
    if width == height:
        return image
    fill = _estimate_background_color(image)
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), color=fill)
    if width > height:
        offset = (0, (side - height) // 2)
    else:
        offset = ((side - width) // 2, 0)
    canvas.paste(image, offset)
    return canvas


def whole_specimen_pad_transform(image_size: int) -> transforms.Compose:
    """Whole-specimen padding evaluation protocol (not the default).

    Pads the source image to a square canvas with the edge-median background
    fill (preserving aspect ratio, no distortion), resizes to a fixed square,
    then tensorizes and normalizes. Reserved for later promotion once it wins
    on downstream metrics.
    """
    return transforms.Compose(
        [
            transforms.Lambda(pad_to_square),
            transforms.Resize(
                (image_size, image_size), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


EVAL_TRANSFORMS = {
    "center-crop": center_crop_eval_transform,
    "whole-specimen-pad": whole_specimen_pad_transform,
}


def build_eval_transform(name: str, size: int) -> transforms.Compose:
    """Return the named evaluation transform at the given resolution."""
    builder = EVAL_TRANSFORMS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown eval transform '{name}'; choose from: {', '.join(EVAL_TRANSFORMS)}"
        )
    return builder(size)


class MultiCropDataset(Dataset):
    """SSL pretraining dataset with multi-crop augmentation."""

    def __init__(
        self,
        csv_path: Path | None,
        images_dir: Path,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        local_crops: int = 6,
    ) -> None:
        images_root = Path(images_dir)
        if csv_path is None:
            self.image_paths = sorted(
                path
                for path in images_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            df = pd.read_csv(csv_path)
            refs = [str(row) for row in df["image"]]
            missing_direct = [
                ref
                for ref in refs
                if _supports_recursive_lookup(ref)
                and not (images_root / Path(ref)).exists()
            ]
            by_relative, by_name = ({}, {})
            if missing_direct:
                by_relative, by_name = _build_recursive_index(images_root)
            self.image_paths = [
                _resolve_image_path(images_root, ref, by_relative, by_name)
                for ref in refs
            ]
        self.global_tf = _make_global_transform(global_crop_size)
        self.local_tf = _make_local_transform(local_crop_size)
        self.local_crops = local_crops

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        with Image.open(self.image_paths[idx]) as im:
            img = im.convert("RGB")
        views = [self.global_tf(img), self.global_tf(img)]
        for _ in range(self.local_crops):
            views.append(self.local_tf(img))
        return views


class MetricDataset(Dataset):
    """Fine-tuning dataset for metric learning."""

    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        image_size: int = 224,
    ) -> None:
        df = pd.read_csv(csv_path)
        images_root = Path(images_dir)
        refs = [str(row) for row in df["image"]]
        missing_direct = [
            ref
            for ref in refs
            if _supports_recursive_lookup(ref)
            and not (images_root / Path(ref)).exists()
        ]
        by_relative, by_name = ({}, {})
        if missing_direct:
            by_relative, by_name = _build_recursive_index(images_root)
        self.image_paths = [
            _resolve_image_path(images_root, ref, by_relative, by_name) for ref in refs
        ]
        classes = sorted(df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.class_to_idx[l] for l in df["label"]]
        self.transform = _make_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[idx]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.labels[idx]
