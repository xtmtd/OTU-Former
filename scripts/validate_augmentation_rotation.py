#!/usr/bin/env python3
"""Development-only rotation/reflection consistency validation for checkpoints.

This is an implementation-comparison tool, not a product CLI command: it is
deliberately not registered in ``otuformer.cli.main``. It validates the output
embeddings of two checkpoints under deterministic rotation/reflection views;
it does not validate transform-object composition.

For each checkpoint it:
  1. loads the checkpoint and reads ``config.model_name`` plus
     ``config.out_dim``/``config.metric_embed_dim``;
  2. resolves the checkpoint training size with
     ``otuformer.utils.size.resolve_training_image_size``;
  3. constructs ``OTUFormerEncoder(img_size=training_size, pretrained=False)``
     and loads ``model_state_dict``;
  4. resolves the deterministic evaluation size (``--image-size`` when given,
     otherwise the checkpoint training size) and requires the two checkpoints
     to agree before comparing embeddings;
  5. extracts raw CLS tokens for angle-0, 90, 180, 270 rotation views
     (``expand=True`` with the edge-median fill) and a horizontally reflected
     view, all preprocessed with the shared deterministic
     ``center_crop_eval_transform``;
  6. reports per-view cosine summaries and different-image cosine summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

# Allow ``python scripts/validate_augmentation_rotation.py`` from a source
# checkout without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.is_dir() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from otuformer.training.dataset import (  # noqa: E402
    _build_recursive_index,
    _estimate_background_color,
    _resolve_image_path,
    _supports_recursive_lookup,
    center_crop_eval_transform,
)
from otuformer.training.model import OTUFormerEncoder  # noqa: E402
from otuformer.utils.device import resolve_device  # noqa: E402
from otuformer.utils.size import resolve_training_image_size  # noqa: E402

ROTATION_ANGLES = (90, 180, 270)
VIEW_KEYS = ("rotation_0", "rotation_90", "rotation_180", "rotation_270", "reflection")
DIFFERENT_IMAGE_SEED = 42
DIFFERENT_IMAGE_MAX_PAIRS = 100_000
DEFAULT_BATCH_SIZE = 8


def summarize_similarities(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Similarity values must be non-empty and finite.")
    return {
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "minimum": float(np.min(values)),
    }


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left at zero."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms == 0.0, 1.0, norms)


def paired_cosine_similarities(
    reference: np.ndarray, comparison: np.ndarray
) -> np.ndarray:
    """Per-row cosine similarities between two equally sized embedding sets."""
    reference = l2_normalize(reference)
    comparison = l2_normalize(comparison)
    if reference.shape != comparison.shape:
        raise ValueError(
            f"Embedding shapes differ: {reference.shape} vs {comparison.shape}."
        )
    return np.einsum("ij,ij->i", reference, comparison)


def summarize_paired_similarities(
    reference: np.ndarray, comparison: np.ndarray
) -> dict[str, float]:
    """Median/p10/min summary for one rotation or reflection view."""
    return summarize_similarities(paired_cosine_similarities(reference, comparison))


def different_image_similarity_summary(
    embeddings: np.ndarray,
    *,
    seed: int = DIFFERENT_IMAGE_SEED,
    max_pairs: int = DIFFERENT_IMAGE_MAX_PAIRS,
) -> dict[str, float | int]:
    """Median/p90 cosine summary across distinct-image embedding pairs.

    Uses every upper-triangle pair when the dataset is small enough; otherwise
    a fixed-seed sample capped at ``max_pairs`` unordered pairs.
    """
    embeddings = l2_normalize(embeddings)
    count = embeddings.shape[0]
    if count < 2:
        raise ValueError(
            "At least two images are required for different-image similarities."
        )
    total_pairs = count * (count - 1) // 2
    if total_pairs <= max_pairs:
        left, right = np.triu_indices(count, k=1)
    else:
        rng = np.random.default_rng(seed)
        left = rng.integers(0, count, size=max_pairs)
        right = rng.integers(0, count, size=max_pairs)
        keep = left != right
        left, right = left[keep], right[keep]
        # Canonicalize unordered pairs and drop any sampled duplicates.
        pairs = np.unique(
            np.stack([np.minimum(left, right), np.maximum(left, right)], axis=1),
            axis=0,
        )
        left, right = pairs[:, 0], pairs[:, 1]
    values = np.einsum("ij,ij->i", embeddings[left], embeddings[right])
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "pair_count": int(values.size),
    }


def resolve_evaluation_size(
    checkpoint: dict, image_size_override: int | None = None
) -> int:
    """Resolve the deterministic evaluation size for one checkpoint.

    ``--image-size`` is shared by both checkpoints; without it each checkpoint
    evaluates at its own resolved training size.
    """
    if image_size_override is not None:
        if isinstance(image_size_override, bool) or image_size_override <= 0:
            raise ValueError(
                f"--image-size must be a positive integer, got {image_size_override!r}."
            )
        return int(image_size_override)
    return resolve_training_image_size(checkpoint)


def require_matching_evaluation_sizes(baseline_size: int, candidate_size: int) -> int:
    """Return the shared evaluation size or raise when the two checkpoints differ."""
    if baseline_size != candidate_size:
        raise ValueError(
            f"Effective evaluation image sizes differ: baseline={baseline_size}, "
            f"candidate={candidate_size}. Pass --image-size to evaluate both at one size."
        )
    return baseline_size


def read_orientation_policy(checkpoint: dict) -> str | None:
    """Read the saved ``augmentation_config.orientation_policy``, if any."""
    config = checkpoint.get("config") or {}
    if not isinstance(config, dict):
        return None
    augmentation = config.get("augmentation_config") or {}
    if not isinstance(augmentation, dict):
        return None
    policy = augmentation.get("orientation_policy")
    return policy if isinstance(policy, str) else None


def orientation_policy_note(policy: str | None) -> str:
    """Describe how the saved policy bounds the reflection/rotation contract."""
    if policy == "sensitive":
        return (
            "Saved policy 'sensitive': no horizontal flip and rotation limited to "
            "[-15, 15] degrees. Reflection similarity below is diagnostic only and "
            "is not an invariance target."
        )
    if policy == "invariant":
        return (
            "Saved policy 'invariant': training used full rotation and horizontal "
            "reflection augmentation."
        )
    return (
        "No augmentation_config.orientation_policy recorded (legacy checkpoint); "
        "the historical no-flip/narrow-rotation contract is not available."
    )


def _read_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a dict.")
    return checkpoint


def load_checkpoint_model(
    checkpoint: dict, device: torch.device
) -> tuple[OTUFormerEncoder, dict]:
    """Construct the encoder at its recorded training size and load its weights."""
    config = checkpoint.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dict.")
    model_name = config.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Checkpoint config.model_name is missing.")
    out_dim = config.get("out_dim") or config.get("metric_embed_dim")
    if isinstance(out_dim, bool) or not isinstance(out_dim, int) or out_dim <= 0:
        raise ValueError(
            "Checkpoint config.out_dim/config.metric_embed_dim is invalid: "
            f"{out_dim!r}."
        )
    training_size = resolve_training_image_size(checkpoint)
    model = OTUFormerEncoder(
        model_name=model_name,
        out_dim=out_dim,
        pretrained=False,
        img_size=training_size,
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model, {
        "model_name": model_name,
        "out_dim": int(out_dim),
        "training_image_size": training_size,
        "missing_state_dict_keys": len(missing),
        "unexpected_state_dict_keys": len(unexpected),
    }


def _extract_cls_tokens(model: OTUFormerEncoder, images: torch.Tensor) -> torch.Tensor:
    """Raw backbone CLS tokens, matching ``trainer._extract_tokens`` semantics."""
    features = model.backbone.forward_features(images)
    if isinstance(features, dict):
        features = features["x"]
    return features[:, 0]


def _load_image_paths(csv_path: Path, images_dir: Path) -> list[Path]:
    """Resolve every CSV ``image`` reference via the dataset's recursive rules."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        raise ValueError(f"CSV {csv_path} is missing the required 'image' column.")
    refs = [str(value) for value in df["image"]]
    missing_direct = [
        ref
        for ref in refs
        if _supports_recursive_lookup(ref) and not (images_dir / Path(ref)).exists()
    ]
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    if missing_direct:
        by_relative, by_name = _build_recursive_index(images_dir)
    return [
        _resolve_image_path(images_dir, ref, by_relative, by_name) for ref in refs
    ]


def embed_validation_views(
    model: OTUFormerEncoder,
    transform,
    image_paths: list[Path],
    device: torch.device,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """L2-normalized angle-0/90/180/270 and reflected CLS embeddings."""
    collected: dict[str, list[np.ndarray]] = {key: [] for key in VIEW_KEYS}
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        prepared: dict[str, list[Image.Image]] = {key: [] for key in VIEW_KEYS}
        for path in chunk:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            fill = _estimate_background_color(image)
            # Angle 0 is the unrotated source image (rotate(0) is identity).
            prepared["rotation_0"].append(image)
            for angle in ROTATION_ANGLES:
                prepared[f"rotation_{angle}"].append(
                    image.rotate(
                        angle,
                        resample=Image.BICUBIC,
                        expand=True,
                        fillcolor=fill,
                    )
                )
            # Horizontal reflection maps source pixels to existing destination
            # pixels, so no edge-median fill is introduced.
            prepared["reflection"].append(ImageOps.mirror(image))
        tensors = [
            torch.stack([transform(image) for image in prepared[key]])
            for key in VIEW_KEYS
        ]
        batch = torch.cat(tensors, dim=0).to(device)
        with torch.no_grad():
            tokens = _extract_cls_tokens(model, batch).float().cpu().numpy()
        size = len(chunk)
        for index, key in enumerate(VIEW_KEYS):
            collected[key].append(tokens[index * size : (index + 1) * size])
    return {
        key: l2_normalize(np.concatenate(value, axis=0))
        for key, value in collected.items()
    }


def validate_checkpoint(
    checkpoint: dict,
    checkpoint_path: Path,
    *,
    image_paths: list[Path],
    evaluation_size: int,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Compute the per-checkpoint rotation/reflection/different-image summary."""
    model, model_info = load_checkpoint_model(checkpoint, device)
    transform = center_crop_eval_transform(evaluation_size)
    embeddings = embed_validation_views(
        model, transform, image_paths, device, batch_size=batch_size
    )
    baseline = embeddings["rotation_0"]
    policy = read_orientation_policy(checkpoint)
    return {
        "checkpoint": str(checkpoint_path),
        **model_info,
        "evaluation_image_size": int(evaluation_size),
        "orientation_policy": policy,
        "orientation_policy_note": orientation_policy_note(policy),
        "image_count": len(image_paths),
        "rotation": {
            str(angle): summarize_paired_similarities(
                baseline, embeddings[f"rotation_{angle}"]
            )
            for angle in ROTATION_ANGLES
        },
        "reflection": summarize_paired_similarities(
            baseline, embeddings["reflection"]
        ),
        "different_image": different_image_similarity_summary(baseline),
    }


def run(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    image_paths = _load_image_paths(args.images_csv, args.images_dir)
    if len(image_paths) < 2:
        raise ValueError(
            "At least two images are required for different-image similarities; "
            f"resolved {len(image_paths)} from {args.images_csv}."
        )

    baseline_checkpoint = _read_checkpoint(args.baseline_checkpoint)
    candidate_checkpoint = _read_checkpoint(args.candidate_checkpoint)
    baseline_size = resolve_evaluation_size(baseline_checkpoint, args.image_size)
    candidate_size = resolve_evaluation_size(candidate_checkpoint, args.image_size)
    effective_size = require_matching_evaluation_sizes(baseline_size, candidate_size)

    baseline = validate_checkpoint(
        baseline_checkpoint,
        args.baseline_checkpoint,
        image_paths=image_paths,
        evaluation_size=effective_size,
        device=device,
    )
    print(f"Evaluated baseline checkpoint at size {effective_size}.", flush=True)
    candidate = validate_checkpoint(
        candidate_checkpoint,
        args.candidate_checkpoint,
        image_paths=image_paths,
        evaluation_size=effective_size,
        device=device,
    )
    print(f"Evaluated candidate checkpoint at size {effective_size}.", flush=True)

    return {
        "script": "scripts/validate_augmentation_rotation.py",
        "device": str(device),
        "images_csv": str(args.images_csv),
        "images_dir": str(args.images_dir),
        "image_count": len(image_paths),
        "effective_evaluation_image_size": int(effective_size),
        "baseline": baseline,
        "candidate": candidate,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only rotation/reflection consistency validation for two "
            "checkpoints. Not registered as an otuformer CLI command."
        )
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "Optional deterministic evaluation-preprocessing size shared by both "
            "checkpoints; model construction size always resolves from each "
            "checkpoint."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote rotation/reflection validation to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
