"""Datasets for SSL pretraining and metric learning fine-tuning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

# 8-bit ImageNet mean, used as the background fill fallback when edge-median
# estimation fails (e.g. degenerate or empty images).
_BACKGROUND_FILL_FALLBACK = (124, 116, 104)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

PRETRAIN_AUGMENTATIONS: tuple[str, ...] = ("global-barcode", "color-robust", "legacy")
FINETUNE_AUGMENTATIONS: tuple[str, ...] = ("none", "conservative")
ORIENTATION_POLICIES: tuple[str, ...] = ("invariant", "sensitive")

_PRETRAIN_GLOBAL_SCALE = [0.4, 1.0]
_PRETRAIN_LOCAL_SCALE = [0.05, 0.4]
_PRETRAIN_CROP_RATIO = [0.75, 4 / 3]
_PRETRAIN_GLOBAL_BLUR_KERNEL = 23
_PRETRAIN_LOCAL_BLUR_KERNEL = 7
_BLUR_SIGMA = [0.1, 2.0]
_INVARIANT_ROTATION_DEGREES = [-180.0, 180.0]
_SENSITIVE_ROTATION_DEGREES = [-15.0, 15.0]
_PRETRAIN_TRANSFORM_ORDER = [
    "random_resized_crop",
    "rotation",
    "resize",
    "horizontal_flip",
    "color",
    "blur",
    "to_tensor",
    "normalize",
]
# 0.2.1 history: no post-rotation resize, and vertical reflection is retained.
_LEGACY_TRANSFORM_ORDER = [
    "random_resized_crop",
    "rotation",
    "horizontal_flip",
    "vertical_flip",
    "color",
    "blur",
    "to_tensor",
    "normalize",
]
_FINETUNE_NONE_TRANSFORM_ORDER = ["resize", "center_crop", "to_tensor", "normalize"]
_FINETUNE_CONSERVATIVE_TRANSFORM_ORDER = [
    "pad_to_safe_square",
    "affine",
    "horizontal_flip",
    "color",
    "resize",
    "to_tensor",
    "normalize",
]
_SAFE_CANVAS_MAX_SCALE = 1.1
_SAFE_CANVAS_MAX_TRANSLATION_FRACTION = 0.05

_PRETRAIN_PROFILE_SETTINGS: dict[str, dict[str, object]] = {
    "global-barcode": {
        "color_jitter": {
            "probability": 0.8,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.1,
            "hue": 0.02,
        },
        "grayscale_probability": 0.0,
        "blur_probabilities": [1.0, 0.1, 0.5],
    },
    "color-robust": {
        "color_jitter": {
            "probability": 0.8,
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.2,
            "hue": 0.1,
        },
        "grayscale_probability": 0.2,
        "blur_probabilities": [1.0, 0.1, 0.5],
    },
    "legacy": {
        "color_jitter": {
            "probability": 1.0,
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.1,
        },
        "grayscale_probability": 0.2,
        "blur_probabilities": [1.0, 1.0, 1.0],
    },
}

_FINETUNE_CONSERVATIVE_COLOR_JITTER: dict[str, float] = {
    "probability": 0.8,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.1,
    "hue": 0.02,
}


def _resolve_blur_kernel(standard_kernel: int, crop_size: int, *, required: bool) -> int:
    """Largest odd kernel no greater than both the standard kernel and crop size.

    ``required`` is False when the view is never generated (``local_crops`` is
    zero), so an unused crop size is recorded without rejection.
    """
    if crop_size < 3:
        if required:
            raise ValueError(
                f"Augmentation profile requires crop size >= 3, got {crop_size}"
            )
        return 1
    kernel = min(standard_kernel, crop_size)
    if kernel % 2 == 0:
        kernel -= 1
    return kernel


def build_pretrain_augmentation_config(
    profile: str,
    global_crop_size: int,
    local_crop_size: int,
    local_crops: int,
    orientation_policy: str = "invariant",
) -> dict[str, object]:
    """Return the fully expanded, JSON-serializable pretraining augmentation config."""
    if profile not in PRETRAIN_AUGMENTATIONS:
        raise ValueError(
            f"Unknown pretrain augmentation profile '{profile}'; "
            f"choose from: {', '.join(PRETRAIN_AUGMENTATIONS)}"
        )
    if orientation_policy not in ORIENTATION_POLICIES:
        raise ValueError(
            f"Unknown orientation policy '{orientation_policy}'; "
            f"choose from: {', '.join(ORIENTATION_POLICIES)}"
        )
    settings = _PRETRAIN_PROFILE_SETTINGS[profile]

    if profile == "legacy":
        # Historical compatibility: the recorded policy must not rewrite these
        # fixed 0.2.1 transform settings.
        rotation_degrees = list(_INVARIANT_ROTATION_DEGREES)
        horizontal_flip_probability = 0.5
        vertical_flip_probability = 0.5
        rotation_expand = False
        rotation_fill: object = [0, 0, 0]
        transform_order = list(_LEGACY_TRANSFORM_ORDER)
        global_kernel = _PRETRAIN_GLOBAL_BLUR_KERNEL
        local_kernel = _PRETRAIN_LOCAL_BLUR_KERNEL
    else:
        if orientation_policy == "sensitive":
            rotation_degrees = list(_SENSITIVE_ROTATION_DEGREES)
            horizontal_flip_probability = 0.0
        else:
            rotation_degrees = list(_INVARIANT_ROTATION_DEGREES)
            horizontal_flip_probability = 0.5
        vertical_flip_probability = 0.0
        rotation_expand = True
        rotation_fill = "edge-median-rgb"
        transform_order = list(_PRETRAIN_TRANSFORM_ORDER)
        global_kernel = _resolve_blur_kernel(
            _PRETRAIN_GLOBAL_BLUR_KERNEL, global_crop_size, required=True
        )
        local_kernel = _resolve_blur_kernel(
            _PRETRAIN_LOCAL_BLUR_KERNEL, local_crop_size, required=local_crops > 0
        )

    return {
        "profile": profile,
        "orientation_policy": orientation_policy,
        "global_crop": {
            "size": int(global_crop_size),
            "scale": list(_PRETRAIN_GLOBAL_SCALE),
        },
        "local_crop": {
            "size": int(local_crop_size),
            "scale": list(_PRETRAIN_LOCAL_SCALE),
        },
        "crop_ratio": list(_PRETRAIN_CROP_RATIO),
        "local_crops": int(local_crops),
        "transform_order": transform_order,
        "rotation": {
            "degrees": rotation_degrees,
            "interpolation": "bicubic",
            "expand": rotation_expand,
            "fill": rotation_fill,
        },
        "fallback_fill": list(_BACKGROUND_FILL_FALLBACK),
        "horizontal_flip_probability": horizontal_flip_probability,
        "vertical_flip_probability": vertical_flip_probability,
        "color_jitter": dict(settings["color_jitter"]),
        "grayscale_probability": settings["grayscale_probability"],
        "blur": {
            "probabilities": list(settings["blur_probabilities"]),
            "global_kernel": global_kernel,
            "local_kernel": local_kernel,
            "sigma": list(_BLUR_SIGMA),
        },
        "solarization_probability": 0.0,
    }


def build_finetune_augmentation_config(
    profile: str,
    image_size: int,
    orientation_policy: str = "invariant",
) -> dict[str, object]:
    """Return the fully expanded, JSON-serializable fine-tuning augmentation config.

    ``image_size`` is the already resolved training size supplied by the caller;
    this builder does not apply a second size-resolution rule.
    """
    if profile not in FINETUNE_AUGMENTATIONS:
        raise ValueError(
            f"Unknown finetune augmentation profile '{profile}'; "
            f"choose from: {', '.join(FINETUNE_AUGMENTATIONS)}"
        )
    if orientation_policy not in ORIENTATION_POLICIES:
        raise ValueError(
            f"Unknown orientation policy '{orientation_policy}'; "
            f"choose from: {', '.join(ORIENTATION_POLICIES)}"
        )

    if profile == "none":
        return {
            "profile": "none",
            "orientation_policy": orientation_policy,
            "image_size": int(image_size),
            "transform_order": list(_FINETUNE_NONE_TRANSFORM_ORDER),
            "horizontal_flip_probability": 0.0,
            "vertical_flip_probability": 0.0,
            "grayscale_probability": 0.0,
            "solarization_probability": 0.0,
        }

    if orientation_policy == "sensitive":
        rotation_degrees = list(_SENSITIVE_ROTATION_DEGREES)
        horizontal_flip_probability = 0.0
    else:
        rotation_degrees = list(_INVARIANT_ROTATION_DEGREES)
        horizontal_flip_probability = 0.5

    return {
        "profile": "conservative",
        "orientation_policy": orientation_policy,
        "image_size": int(image_size),
        "transform_order": list(_FINETUNE_CONSERVATIVE_TRANSFORM_ORDER),
        "safe_canvas": {
            "max_scale": _SAFE_CANVAS_MAX_SCALE,
            "max_translation_fraction": _SAFE_CANVAS_MAX_TRANSLATION_FRACTION,
            "formula": (
                "ceil(ceil(hypot(width, height)) * max_scale / "
                "(1 - 2 * max_translation_fraction))"
            ),
        },
        "rotation": {
            "degrees": rotation_degrees,
            "interpolation": "bicubic",
            "fill": "edge-median-rgb",
        },
        "translation": [
            _SAFE_CANVAS_MAX_TRANSLATION_FRACTION,
            _SAFE_CANVAS_MAX_TRANSLATION_FRACTION,
        ],
        "scale": [0.9, 1.1],
        "horizontal_flip_probability": horizontal_flip_probability,
        "vertical_flip_probability": 0.0,
        "color_jitter": dict(_FINETUNE_CONSERVATIVE_COLOR_JITTER),
        "grayscale_probability": 0.0,
        "solarization_probability": 0.0,
        "fallback_fill": list(_BACKGROUND_FILL_FALLBACK),
    }


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


def _random_apply(transform: Any, probability: float) -> Any:
    """Wrap ``transform`` in RandomApply, or return it directly when always applied.

    Returning the transform directly when ``probability`` is 1.0 keeps the legacy
    profile's random-number-consumption identical to 0.2.1.
    """
    if probability <= 0.0:
        return None
    if probability >= 1.0:
        return transform
    return transforms.RandomApply([transform], p=probability)


class _PretrainViewTransform:
    """Fill-aware transform for one pretraining view (global 1, global 2, or local).

    Module-level and picklable so DataLoader workers using the spawn start
    method can serialize the dataset. ``rotation_degrees`` and
    ``horizontal_flip_probability`` are read-only views of the resolved
    augmentation configuration.
    """

    def __init__(
        self,
        *,
        crop_size: int,
        scale: tuple[float, float],
        rotation_degrees: tuple[float, float],
        rotation_expand: bool,
        rotation_fill: str | tuple[int, int, int] | list[int],
        horizontal_flip_probability: float,
        vertical_flip_probability: float,
        color_jitter: dict[str, float],
        grayscale_probability: float,
        blur_probability: float,
        blur_kernel: int,
        blur_sigma: tuple[float, float],
        post_rotation_resize: bool,
    ) -> None:
        self._crop_size = int(crop_size)
        self._scale = tuple(float(v) for v in scale)
        self._rotation_degrees = tuple(float(v) for v in rotation_degrees)
        self._rotation_expand = bool(rotation_expand)
        self._uses_edge_fill = rotation_fill == "edge-median-rgb"
        self._fixed_fill = (
            None if self._uses_edge_fill else tuple(int(v) for v in rotation_fill)
        )
        self._horizontal_flip_probability = float(horizontal_flip_probability)
        self._vertical_flip_probability = float(vertical_flip_probability)
        self._grayscale_probability = float(grayscale_probability)
        self._blur_probability = float(blur_probability)
        self._blur_kernel = int(blur_kernel)
        self._post_rotation_resize = bool(post_rotation_resize)

        self._crop = transforms.RandomResizedCrop(
            self._crop_size, scale=self._scale, interpolation=Image.BICUBIC
        )
        self._horizontal_flip = (
            transforms.RandomHorizontalFlip(p=self._horizontal_flip_probability)
            if self._horizontal_flip_probability > 0.0
            else None
        )
        self._vertical_flip = (
            transforms.RandomVerticalFlip(p=self._vertical_flip_probability)
            if self._vertical_flip_probability > 0.0
            else None
        )
        self._color = _random_apply(
            transforms.ColorJitter(
                brightness=color_jitter["brightness"],
                contrast=color_jitter["contrast"],
                saturation=color_jitter["saturation"],
                hue=color_jitter["hue"],
            ),
            float(color_jitter["probability"]),
        )
        self._grayscale = (
            transforms.RandomGrayscale(p=self._grayscale_probability)
            if self._grayscale_probability > 0.0
            else None
        )
        self._blur = _random_apply(
            transforms.GaussianBlur(
                kernel_size=self._blur_kernel,
                sigma=tuple(float(v) for v in blur_sigma),
            ),
            self._blur_probability,
        )
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(
            mean=_IMAGENET_MEAN, std=_IMAGENET_STD
        )

    @property
    def crop_size(self) -> int:
        return self._crop_size

    @property
    def rotation_degrees(self) -> tuple[float, float]:
        return self._rotation_degrees

    @property
    def rotation_expand(self) -> bool:
        return self._rotation_expand

    @property
    def rotation_fill(self) -> str | tuple[int, int, int]:
        if self._uses_edge_fill:
            return "edge-median-rgb"
        return self._fixed_fill

    @property
    def horizontal_flip_probability(self) -> float:
        return self._horizontal_flip_probability

    @property
    def vertical_flip_probability(self) -> float:
        return self._vertical_flip_probability

    @property
    def grayscale_probability(self) -> float:
        return self._grayscale_probability

    @property
    def blur_probability(self) -> float:
        return self._blur_probability

    @property
    def blur_kernel(self) -> int:
        return self._blur_kernel

    @property
    def post_rotation_resize(self) -> bool:
        return self._post_rotation_resize

    @property
    def uses_edge_fill(self) -> bool:
        return self._uses_edge_fill

    def __call__(
        self, image: Image.Image, fill: tuple[int, int, int]
    ) -> torch.Tensor:
        image = self._crop(image)
        angle = transforms.RandomRotation.get_params(self._rotation_degrees)
        image = TF.rotate(
            image,
            angle,
            interpolation=InterpolationMode.BICUBIC,
            expand=self._rotation_expand,
            fill=tuple(fill) if self._uses_edge_fill else self._fixed_fill,
        )
        if self._post_rotation_resize:
            image = TF.resize(
                image,
                [self._crop_size, self._crop_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        if self._horizontal_flip is not None:
            image = self._horizontal_flip(image)
        if self._vertical_flip is not None:
            image = self._vertical_flip(image)
        if self._color is not None:
            image = self._color(image)
        if self._grayscale is not None:
            image = self._grayscale(image)
        if self._blur is not None:
            image = self._blur(image)
        return self._normalize(self._to_tensor(image))


def _build_pretrain_view_transform(
    config: dict[str, object], view: str, blur_probability: float
) -> _PretrainViewTransform:
    crop = config[f"{view}_crop"]
    rotation = config["rotation"]
    blur = config["blur"]
    return _PretrainViewTransform(
        crop_size=int(crop["size"]),
        scale=tuple(float(v) for v in crop["scale"]),
        rotation_degrees=tuple(float(v) for v in rotation["degrees"]),
        rotation_expand=bool(rotation["expand"]),
        rotation_fill=rotation["fill"],
        horizontal_flip_probability=float(config["horizontal_flip_probability"]),
        vertical_flip_probability=float(config["vertical_flip_probability"]),
        color_jitter=config["color_jitter"],
        grayscale_probability=float(config["grayscale_probability"]),
        blur_probability=blur_probability,
        blur_kernel=int(blur[f"{view}_kernel"]),
        blur_sigma=tuple(float(v) for v in blur["sigma"]),
        post_rotation_resize="resize" in config["transform_order"],
    )


def _pad_to_safe_square(
    image: Image.Image,
    fill: tuple[int, int, int],
    max_scale: float = _SAFE_CANVAS_MAX_SCALE,
    max_translation_fraction: float = _SAFE_CANVAS_MAX_TRANSLATION_FRACTION,
) -> Image.Image:
    """Center the source image on a square canvas that no affine sample can clip.

    The diagonal accommodates every in-plane rotation; the scale and translation
    factors reserve room for the largest configured enlargement and offset.
    """
    diagonal = math.ceil(math.hypot(image.width, image.height))
    side = math.ceil(diagonal * max_scale / (1 - 2 * max_translation_fraction))
    canvas = Image.new("RGB", (side, side), color=fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


class _ConservativeTransform:
    """Fill-aware conservative fine-tuning transform.

    Rotation, translation, and scale are sampled as the parameters of a single
    affine transform applied to a safe square canvas.
    """

    def __init__(self, config: dict[str, object]) -> None:
        self._image_size = int(config["image_size"])
        rotation = config["rotation"]
        self._rotation_degrees = tuple(float(v) for v in rotation["degrees"])
        self._horizontal_flip_probability = float(
            config["horizontal_flip_probability"]
        )
        self._translate = tuple(float(v) for v in config["translation"])
        self._scale = tuple(float(v) for v in config["scale"])
        color_jitter = config["color_jitter"]
        self._color = _random_apply(
            transforms.ColorJitter(
                brightness=color_jitter["brightness"],
                contrast=color_jitter["contrast"],
                saturation=color_jitter["saturation"],
                hue=color_jitter["hue"],
            ),
            float(color_jitter["probability"]),
        )
        self._horizontal_flip = (
            transforms.RandomHorizontalFlip(p=self._horizontal_flip_probability)
            if self._horizontal_flip_probability > 0.0
            else None
        )
        self._resize = transforms.Resize(
            (self._image_size, self._image_size),
            interpolation=InterpolationMode.BICUBIC,
        )
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(
            mean=_IMAGENET_MEAN, std=_IMAGENET_STD
        )

    @property
    def rotation_degrees(self) -> tuple[float, float]:
        return self._rotation_degrees

    @property
    def horizontal_flip_probability(self) -> float:
        return self._horizontal_flip_probability

    def __call__(
        self, image: Image.Image, fill: tuple[int, int, int]
    ) -> torch.Tensor:
        canvas = _pad_to_safe_square(image, fill)
        angle, translate, scale, shear = transforms.RandomAffine.get_params(
            degrees=list(self._rotation_degrees),
            translate=list(self._translate),
            scale_ranges=list(self._scale),
            shears=None,
            img_size=canvas.size,
        )
        image = TF.affine(
            canvas,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BICUBIC,
            fill=list(fill),
        )
        if self._horizontal_flip is not None:
            image = self._horizontal_flip(image)
        if self._color is not None:
            image = self._color(image)
        image = self._resize(image)
        return self._normalize(self._to_tensor(image))


def _validate_view_shape(
    view: torch.Tensor, expected_size: int, view_name: str
) -> None:
    """Reject a view whose tensor shape is not (3, expected, expected)."""
    actual = tuple(view.shape)
    expected = (3, expected_size, expected_size)
    if actual != expected:
        raise ValueError(
            f"{view_name} produced tensor shape {actual}; expected {expected}"
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
        augmentation_profile: str = "global-barcode",
        orientation_policy: str = "invariant",
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
        self.global_crop_size = int(global_crop_size)
        self.local_crop_size = int(local_crop_size)
        self.local_crops = int(local_crops)
        self.augmentation_profile = augmentation_profile
        self.orientation_policy = orientation_policy
        self.augmentation_config = build_pretrain_augmentation_config(
            augmentation_profile,
            self.global_crop_size,
            self.local_crop_size,
            self.local_crops,
            orientation_policy,
        )
        blur_probabilities = self.augmentation_config["blur"]["probabilities"]
        self.global_tf1 = _build_pretrain_view_transform(
            self.augmentation_config, "global", blur_probabilities[0]
        )
        self.global_tf2 = _build_pretrain_view_transform(
            self.augmentation_config, "global", blur_probabilities[1]
        )
        self.local_tf = _build_pretrain_view_transform(
            self.augmentation_config, "local", blur_probabilities[2]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_view(
        self,
        transform: Any,
        image: Image.Image,
        fill: tuple[int, int, int],
        expected_size: int,
        view_name: str,
    ) -> torch.Tensor:
        view = transform(image, fill)
        _validate_view_shape(view, expected_size, view_name)
        return view

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        with Image.open(self.image_paths[idx]) as im:
            img = im.convert("RGB")
        fill = _estimate_background_color(img)
        views = [
            self._apply_view(
                self.global_tf1, img, fill, self.global_crop_size, "global_tf1"
            ),
            self._apply_view(
                self.global_tf2, img, fill, self.global_crop_size, "global_tf2"
            ),
        ]
        for _ in range(self.local_crops):
            views.append(
                self._apply_view(
                    self.local_tf, img, fill, self.local_crop_size, "local_tf"
                )
            )
        return views


class MetricDataset(Dataset):
    """Fine-tuning dataset for metric learning."""

    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        image_size: int = 224,
        augmentation_profile: str = "none",
        orientation_policy: str = "invariant",
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
        self.image_size = int(image_size)
        self.augmentation_profile = augmentation_profile
        self.orientation_policy = orientation_policy
        self.augmentation_config = build_finetune_augmentation_config(
            augmentation_profile, self.image_size, orientation_policy
        )
        if augmentation_profile == "none":
            self.transform = _make_eval_transform(self.image_size)
        else:
            self.transform = _ConservativeTransform(self.augmentation_config)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[idx]) as im:
            img = im.convert("RGB")
        if self.augmentation_profile == "none":
            # Deterministic path: no background estimation and no stochastic ops.
            image = self.transform(img)
        else:
            fill = _estimate_background_color(img)
            image = self.transform(img, fill)
        _validate_view_shape(image, self.image_size, "finetune view")
        return image, self.labels[idx]
