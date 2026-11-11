"""Datasets for SSL pretraining and metric learning fine-tuning."""

from __future__ import annotations

import copy
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF

from otuformer.constants import (
    EVAL_TRANSFORM_CHOICES,
    FINETUNE_AUGMENTATIONS,
    ORIENTATION_POLICIES,
    PRETRAIN_AUGMENTATIONS,
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

# 8-bit ImageNet mean, used as the background fill fallback when edge-median
# estimation fails (e.g. degenerate or empty images).
_BACKGROUND_FILL_FALLBACK = (124, 116, 104)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


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
    orientation_policy: str = "sensitive",
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


def _normalize_record(record: object) -> tuple[list[float], list[float]]:
    """Validate a weak-record ``normalize`` mapping and return its mean/std."""
    if not isinstance(record, dict):
        raise ValueError(
            f"weak teacher augmentation normalize must be a mapping, got {record!r}"
        )
    mean = _normalize_vector(record.get("mean"), "mean")
    std = _normalize_vector(record.get("std"), "std")
    if any(value <= 0.0 for value in std):
        raise ValueError(
            f"weak teacher augmentation normalize.std must be positive, got {std!r}"
        )
    return mean, std


def _normalize_vector(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(
            "weak teacher augmentation normalize."
            f"{name} must be an ordered list of three numbers, got {value!r}"
        )
    resolved: list[float] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ValueError(
                "weak teacher augmentation normalize."
                f"{name} must hold numbers, got {entry!r}"
            )
        number = float(entry)
        if not math.isfinite(number):
            raise ValueError(
                "weak teacher augmentation normalize."
                f"{name} must be finite, got {entry!r}"
            )
        resolved.append(number)
    return resolved


def build_weak_teacher_augmentation_config(
    augmentation_config: dict[str, object],
) -> dict[str, object]:
    """Derive the weak teacher global transform from a resolved ordinary config.

    Mirrors the ordinary geometry and normalization so photometric augmentation
    is the only difference. ``local_crop``, ``local_crops``,
    ``blur.local_kernel`` and ``blur.probabilities[2]`` are provenance only: no
    weak global view consumes them, but they stay in the record because the
    whole record is locked for resume drift detection.
    """
    weak = copy.deepcopy(augmentation_config)
    weak["color_jitter"] = {
        "probability": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
        "hue": 0.0,
    }
    weak["grayscale_probability"] = 0.0
    blur = dict(weak["blur"])
    probabilities = list(blur["probabilities"])
    probabilities[0] = 0.0
    probabilities[1] = 0.0
    blur["probabilities"] = probabilities
    weak["blur"] = blur
    weak["normalize"] = {"mean": list(_IMAGENET_MEAN), "std": list(_IMAGENET_STD)}
    return weak


def build_finetune_augmentation_config(
    profile: str,
    image_size: int,
    orientation_policy: str = "sensitive",
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
        "profile": profile,
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


def resolve_dataset_image_paths(
    csv_path: Path | None, images_dir: Path, *, refs: list[str] | None = None
) -> list[Path]:
    """Resolve exactly the paths the CSV and training datasets open."""
    images_root = Path(images_dir)
    if refs is None:
        if csv_path is None:
            return sorted(
                p for p in images_root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        frame = pd.read_csv(csv_path)
        if "image" not in frame.columns:
            raise ValueError(f"CSV {csv_path} missing required 'image' column")
        refs = [str(value) for value in frame["image"]]
    by_relative, by_name = ({}, {})
    if any(
        _supports_recursive_lookup(ref) and not (images_root / Path(ref)).exists()
        for ref in refs
    ):
        by_relative, by_name = _build_recursive_index(images_root)
    return [_resolve_image_path(images_root, ref, by_relative, by_name) for ref in refs]


@dataclass(frozen=True)
class PreparedRelationalData:
    """One ordered, identity-checked selection for an enabled queue run.

    ``image_keys`` is the row identity carried with each training sample; the
    image count is ``len(image_keys)``. Only the compact digest and the two
    dataset summaries are persisted, never the ordered manifest or the bytes.
    """

    image_paths: tuple[Path, ...]
    image_keys: tuple[str, ...]
    data_fingerprint: str
    data_size_bytes: int


def _selected_identity_key(images_root: Path, path: Path) -> str:
    """Root-relative key for one already-selected path, without re-selecting.

    The written spelling of the selected path is handed to the shared
    ``resolve_reference`` semantics, so lexical folding keeps the key the
    reference reaches (``link/../one.png`` addresses ``a/one.png``) and a
    distinct internal symlink alias stays a distinct key.
    """
    from otuformer.data.paths import normalize_key, resolve_reference

    written = path if path.is_absolute() else Path.cwd() / path
    key, _ = resolve_reference(images_root, str(written), basename_fallback=False)
    return normalize_key(key)


def prepare_relational_data(
    csv_path: Path | None, images_dir: Path, *, batch_size: int
) -> PreparedRelationalData:
    """Resolve, identity-check and hash the exact selected images once.

    Enabled-only: queue-off runs keep the legacy selection untouched. Rejects
    escaped/missing/corrupt images, two rows sharing one normalized key, and a
    selection with no complete batch under ``drop_last=True`` before any model
    is built. The manifest digest covers keys, sizes and content in row order.
    """
    from otuformer.data.paths import ReferenceError
    from otuformer.data.provenance import canonical_digest, snapshot_bytes
    from otuformer.training.runtime import _phase_done, _phase_start

    images_root = Path(images_dir)
    resolved_batch = int(batch_size)
    if resolved_batch < 1:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}."
        )

    paths = resolve_dataset_image_paths(csv_path, images_root)
    if not paths:
        raise ValueError(
            "The relational queue needs at least one selected image; the CSV or "
            f"image root {images_root} selected none."
        )

    keys: list[str] = []
    # Order is identity, but membership must not rescan the growing list: with
    # 100k selected images the list form is quadratic.
    seen: set[str] = set()
    for path in paths:
        try:
            key = _selected_identity_key(images_root, Path(path))
        except ReferenceError as exc:
            if exc.code in ("root_escape", "unsupported_reference"):
                raise ValueError(
                    "Queue-enabled pretraining requires every selected image to "
                    f"stay inside the declared image root; rejected {path}: {exc}"
                ) from exc
            raise ValueError(
                "Queue-enabled pretraining could not read the selected image "
                f"{path}: {exc}"
            ) from exc
        if key in seen:
            raise ValueError(
                "Queue-enabled pretraining requires one CSV row per image "
                f"identity; two selected rows map to the key {key!r}. Remove the "
                "duplicate row, or run with --relational-queue-size 0 to keep "
                "the legacy input contract."
            )
        seen.add(key)
        keys.append(key)

    if len(keys) < resolved_batch:
        raise ValueError(
            f"Queue-enabled pretraining needs a complete batch: {len(keys)} "
            "selected image(s) with drop_last=True and batch_size="
            f"{resolved_batch}."
        )

    started = _phase_start(
        f"relational queue data preparation ({len(keys)} images)"
    )
    rows: list[dict[str, object]] = []
    for key, path in zip(keys, paths):
        try:
            data, snapshot = snapshot_bytes(Path(path))
        except OSError as exc:
            raise ValueError(
                "Queue-enabled pretraining could not read the selected image "
                f"{key!r}: {exc}"
            ) from exc
        try:
            with Image.open(io.BytesIO(data)) as handle:
                # A header check is not enough: a truncated JPEG passes
                # ``verify()`` and then fails the training path's own decode.
                # Decode the same bytes the run will read, all the way to the
                # RGB conversion the dataset performs.
                handle.load()
                handle.convert("RGB")
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Queue-enabled pretraining requires decodable images; "
                f"{key!r} is unreadable: {exc}"
            ) from exc
        rows.append(
            {
                "image_key": key,
                "size_bytes": int(snapshot.size_bytes),
                "sha256": snapshot.sha256,
            }
        )
    fingerprint = canonical_digest(rows)
    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    _phase_done(
        "relational queue data preparation "
        f"({len(keys)} images, {total_bytes} bytes)",
        started,
    )
    return PreparedRelationalData(
        image_paths=tuple(Path(path) for path in paths),
        image_keys=tuple(keys),
        data_fingerprint=fingerprint,
        data_size_bytes=total_bytes,
    )


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
        crop_ratio: tuple[float, float],
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
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
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
            self._crop_size,
            scale=self._scale,
            ratio=tuple(crop_ratio),
            interpolation=Image.BICUBIC,
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
            mean=_IMAGENET_MEAN if normalize_mean is None else normalize_mean,
            std=_IMAGENET_STD if normalize_std is None else normalize_std,
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
    config: dict[str, object],
    view: str,
    blur_probability: float,
    *,
    use_recorded_normalize: bool = False,
) -> _PretrainViewTransform:
    crop = config[f"{view}_crop"]
    rotation = config["rotation"]
    blur = config["blur"]
    normalize_mean: list[float] | None = None
    normalize_std: list[float] | None = None
    if use_recorded_normalize:
        # The weak record's normalization is binding: no module-constant fallback.
        normalize_mean, normalize_std = _normalize_record(config["normalize"])
    return _PretrainViewTransform(
        crop_size=int(crop["size"]),
        scale=tuple(float(v) for v in crop["scale"]),
        crop_ratio=tuple(float(v) for v in config["crop_ratio"]),
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
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
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
        safe_canvas = config["safe_canvas"]
        self._safe_canvas_max_scale = float(safe_canvas["max_scale"])
        self._safe_canvas_max_translation_fraction = float(
            safe_canvas["max_translation_fraction"]
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
        canvas = _pad_to_safe_square(
            image,
            fill,
            max_scale=self._safe_canvas_max_scale,
            max_translation_fraction=self._safe_canvas_max_translation_fraction,
        )
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
    view: object, expected_size: int, view_name: str
) -> None:
    """Reject a view whose tensor shape is not (3, expected, expected)."""
    shape = getattr(view, "shape", None)
    actual = tuple(shape) if shape is not None else None
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
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
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
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
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


_GEOMETRY_KEYS = (
    "crop_ratio",
    "transform_order",
    "rotation",
    "fallback_fill",
    "horizontal_flip_probability",
    "vertical_flip_probability",
)


def _validated_supplied_config(
    config: dict[str, object],
    *,
    profile: str,
    orientation_policy: str,
    global_crop_size: int,
    local_crop_size: int,
    local_crops: int,
) -> dict[str, object]:
    """Reject a supplied config that contradicts the constructor arguments."""
    expected = {
        "profile": profile,
        "orientation_policy": orientation_policy,
        "global_crop.size": global_crop_size,
        "local_crop.size": local_crop_size,
        "local_crops": local_crops,
    }
    actual = {
        "profile": config.get("profile"),
        "orientation_policy": config.get("orientation_policy"),
        "global_crop.size": config["global_crop"]["size"],
        "local_crop.size": config["local_crop"]["size"],
        "local_crops": config.get("local_crops"),
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(
                f"supplied augmentation_config.{name} is {actual[name]!r} but the "
                f"constructor requests {value!r}; supplied configs are not overrides"
            )
    return config


def _validated_weak_config(
    weak: dict[str, object], ordinary: dict[str, object]
) -> dict[str, object]:
    """Reject a supplied weak config that is not a photometric-only derivative."""
    for key in ("global_crop",) + _GEOMETRY_KEYS:
        if weak.get(key) != ordinary.get(key):
            raise ValueError(
                f"weak teacher augmentation {key!r} must mirror the ordinary "
                "config; only photometric settings may differ"
            )
    _normalize_record(weak.get("normalize"))
    jitter = weak.get("color_jitter") or {}
    if float(jitter.get("probability", 0.0)) != 0.0 or any(
        float(jitter[name]) != 0.0
        for name in ("brightness", "contrast", "saturation", "hue")
    ):
        raise ValueError(
            "weak teacher augmentation must disable color jitter, got "
            f"{weak.get('color_jitter')!r}"
        )
    if float(weak.get("grayscale_probability", 0.0)) != 0.0:
        raise ValueError(
            "weak teacher augmentation must disable grayscale, got "
            f"{weak.get('grayscale_probability')!r}"
        )
    probabilities = weak["blur"]["probabilities"]
    if float(probabilities[0]) != 0.0 or float(probabilities[1]) != 0.0:
        raise ValueError(
            "weak teacher augmentation must disable both global blur "
            f"probabilities, got {list(probabilities)!r}"
        )
    return weak


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
        orientation_policy: str = "sensitive",
        *,
        include_weak_teacher_views: bool = False,
        augmentation_config: dict[str, object] | None = None,
        weak_teacher_augmentation_config: dict[str, object] | None = None,
        prepared_relational_data: PreparedRelationalData | None = None,
    ) -> None:
        images_root = Path(images_dir)
        # Prepared state is the enabled-only handoff: when present it is the one
        # resolution of the selection, so the dataset never re-resolves a path
        # the run already identity-checked.
        self.prepared_relational_data = prepared_relational_data
        if prepared_relational_data is None:
            self.image_paths = resolve_dataset_image_paths(csv_path, images_root)
            self.image_keys: tuple[str, ...] | None = None
        else:
            self.image_paths = list(prepared_relational_data.image_paths)
            self.image_keys = tuple(prepared_relational_data.image_keys)
        self.global_crop_size = int(global_crop_size)
        self.local_crop_size = int(local_crop_size)
        self.local_crops = int(local_crops)
        self.augmentation_profile = augmentation_profile
        self.orientation_policy = orientation_policy
        self.augmentation_config = (
            build_pretrain_augmentation_config(
                augmentation_profile,
                self.global_crop_size,
                self.local_crop_size,
                self.local_crops,
                orientation_policy,
            )
            if augmentation_config is None
            else _validated_supplied_config(
                augmentation_config,
                profile=augmentation_profile,
                orientation_policy=orientation_policy,
                global_crop_size=self.global_crop_size,
                local_crop_size=self.local_crop_size,
                local_crops=self.local_crops,
            )
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
        self.weak_teacher_tf1: _PretrainViewTransform | None = None
        self.weak_teacher_tf2: _PretrainViewTransform | None = None
        self.weak_teacher_augmentation_config: dict[str, object] | None = None
        if include_weak_teacher_views:
            weak_config = (
                build_weak_teacher_augmentation_config(self.augmentation_config)
                if weak_teacher_augmentation_config is None
                else _validated_weak_config(
                    weak_teacher_augmentation_config, self.augmentation_config
                )
            )
            self.weak_teacher_augmentation_config = weak_config
            weak_blur = weak_config["blur"]["probabilities"]
            self.weak_teacher_tf1 = _build_pretrain_view_transform(
                weak_config, "global", weak_blur[0], use_recorded_normalize=True
            )
            self.weak_teacher_tf2 = _build_pretrain_view_transform(
                weak_config, "global", weak_blur[1], use_recorded_normalize=True
            )
        elif weak_teacher_augmentation_config is not None:
            raise ValueError(
                "weak_teacher_augmentation_config requires "
                "include_weak_teacher_views=True"
            )
        # Legacy profiles use a fixed fill, so the per-item edge-median estimate
        # is skipped entirely when no view consumes it.
        self._estimate_edge_fill = any(
            view.uses_edge_fill
            for view in (self.global_tf1, self.global_tf2, self.local_tf)
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

    def __getitem__(self, idx: int) -> list[torch.Tensor] | dict[str, object]:
        with Image.open(self.image_paths[idx]) as im:
            img = im.convert("RGB")
        if self._estimate_edge_fill:
            fill = _estimate_background_color(img)
        else:
            # Legacy transforms ignore the fill argument (fixed black fill).
            fill = _BACKGROUND_FILL_FALLBACK
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
        if self.weak_teacher_tf1 is not None and self.weak_teacher_tf2 is not None:
            views.append(
                self._apply_view(
                    self.weak_teacher_tf1,
                    img,
                    fill,
                    self.global_crop_size,
                    "weak_teacher_A",
                )
            )
            views.append(
                self._apply_view(
                    self.weak_teacher_tf2,
                    img,
                    fill,
                    self.global_crop_size,
                    "weak_teacher_B",
                )
            )
        if self.image_keys is not None:
            # Enabled envelope: one key per row, aligned with the view list so
            # native collation keeps keys and views in the same row order.
            return {"views": views, "image_key": self.image_keys[idx]}
        return views


class MetricDataset(Dataset):
    """Fine-tuning dataset for metric learning."""

    def __init__(
        self,
        csv_path: Path | None = None,
        images_dir: Path = Path("."),
        image_size: int = 224,
        augmentation_profile: str = "none",
        orientation_policy: str = "sensitive",
        *,
        rows: list[dict] | None = None,
        class_labels: list[str] | None = None,
    ) -> None:
        # ``rows`` is the opt-in pseudo-mode path: already-validated in-memory
        # expert+accepted rows with an explicit ordered class table. The default
        # CSV path is unchanged and keeps ``csv_path`` required.
        if rows is None:
            if csv_path is None:
                raise ValueError("MetricDataset requires csv_path or rows")
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(
                {
                    "image": [str(row["image"]) for row in rows],
                    "label": [str(row["label"]) for row in rows],
                }
            )
        images_root = Path(images_dir)
        refs = [str(row) for row in df["image"]]
        self.image_paths = resolve_dataset_image_paths(None, images_root, refs=refs)
        if class_labels is None:
            classes = sorted(df["label"].unique())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.labels = [self.class_to_idx[l] for l in df["label"]]
        else:
            classes = list(class_labels)
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.labels = [self.class_to_idx[str(l)] for l in df["label"]]
        # Original CSV references and label names, kept for the training
        # manifest hash and the optional batch-ID trace.
        self.image_refs = refs
        self.label_names = [str(label) for label in df["label"]]
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


class IndexedDataset(Dataset):
    """Training-only wrapper that also returns the dataset index.

    Used by the opt-in batch-ID trace so recorded IDs come from the actual
    shuffled sampling order rather than the DataLoader step position. The
    default ``(image, label)`` contract is unchanged for non-trace callers.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index
