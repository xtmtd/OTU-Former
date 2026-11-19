"""CAM heatmap generation for OTU-Former backbones."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

from otuformer.training.dataset import (
    EVAL_TRANSFORMS,
    build_eval_transform,
    pad_to_square,
)
from otuformer.training.model import OTUFormerEncoder
from otuformer.utils.checkpoint import (
    load_checkpoint,
    load_checkpoint_encoder,
    resolve_checkpoint,
)
from otuformer.utils.device import resolve_device
from otuformer.utils.size import resolve_training_image_size, validate_input_size

try:
    from pytorch_grad_cam import (
        AblationCAM,
        EigenCAM,
        GradCAM,
        GradCAMPlusPlus,
        LayerCAM,
        ScoreCAM,
    )
    from pytorch_grad_cam.ablation_layer import AblationLayerVit
    from pytorch_grad_cam.base_cam import BaseCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    _HAS_CAM = True
except Exception:
    _HAS_CAM = False
    BaseCAM = object
    AblationLayerVit = None
    show_cam_on_image = None
    ClassifierOutputTarget = None


class UnnormalizedCAMMixin:
    """Mirror pytorch-grad-cam 1.5.5 BaseCAM without scale_cam_image().

    ReLU, target-size resize, layer concatenation, and layer averaging remain
    identical to BaseCAM. Only its two min-max normalization calls are skipped.
    Re-read these methods when upgrading grad-cam.
    """

    @staticmethod
    def _resize_batch(cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        import cv2

        return np.stack(
            [
                cv2.resize(
                    np.float32(image), target_size, interpolation=cv2.INTER_LINEAR
                )
                for image in cam
            ]
        )

    def compute_cam_per_layer(self, input_tensor, targets, eigen_smooth):
        if self.detach:
            activations_list = [
                activation.cpu().data.numpy()
                for activation in self.activations_and_grads.activations
            ]
            grads_list = [
                gradient.cpu().data.numpy()
                for gradient in self.activations_and_grads.gradients
            ]
        else:
            activations_list = list(self.activations_and_grads.activations)
            grads_list = list(self.activations_and_grads.gradients)

        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []
        for index, target_layer in enumerate(self.target_layers):
            activations = (
                activations_list[index] if index < len(activations_list) else None
            )
            gradients = grads_list[index] if index < len(grads_list) else None
            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                activations,
                gradients,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            scaled = self._resize_batch(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        return np.mean(cam_per_target_layer, axis=1)


if _HAS_CAM:

    class RawGradCAM(UnnormalizedCAMMixin, GradCAM):
        pass

    class RawGradCAMPlusPlus(UnnormalizedCAMMixin, GradCAMPlusPlus):
        pass

    class RawLayerCAM(UnnormalizedCAMMixin, LayerCAM):
        pass

    class RawScoreCAM(UnnormalizedCAMMixin, ScoreCAM):
        pass

    class RawEigenCAM(UnnormalizedCAMMixin, EigenCAM):
        pass

    class RawAblationCAM(UnnormalizedCAMMixin, AblationCAM):
        pass

    CAM_METHODS = {
        "gradcam": RawGradCAM,
        "gradcampp": RawGradCAMPlusPlus,
        "layercam": RawLayerCAM,
        "ablationcam": RawAblationCAM,
        "scorecam": RawScoreCAM,
        "eigencam": RawEigenCAM,
    }
else:
    CAM_METHODS = {}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SaveMode = Literal["none", "raw", "normalized"]

SAVE_MODES = ("none", "raw", "normalized")


def _validate_save_mode(save_npy: str) -> None:
    """Reject non-string or unknown save modes at the module API boundary.

    The CLI restricts values via its own ``--save-npy`` choices; this keeps direct
    ``process_image()``/``run_cam()`` calls from silently accepting a bool or a
    typo and treating it as ``normalized``.
    """
    if save_npy not in SAVE_MODES:
        raise ValueError(
            f"Unknown save_npy '{save_npy}'; choose from: {', '.join(SAVE_MODES)}"
        )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    device: torch.device,
) -> tuple[torch.nn.Module, int, str]:
    """Load the CAM model and return ``(model, checkpoint_size, resolved_name)``.

    The encoder is constructed with ``img_size=checkpoint_size`` so the
    positional embeddings load exactly; ``checkpoint_size`` is also the
    resolution at which CAM and display transforms are built. The resolved
    model name comes from checkpoint metadata so architecture inference is
    not fooled by the CLI default model name.
    """
    ckpt = load_checkpoint(checkpoint_path)
    architecture = resolve_checkpoint(ckpt, model_name)
    checkpoint_size = resolve_training_image_size(ckpt)
    encoder, architecture = load_checkpoint_encoder(
        ckpt, architecture, image_size=checkpoint_size
    )
    validate_input_size(checkpoint_size, encoder, architecture.model_name)
    backbone = encoder.backbone
    backbone.eval().to(device)

    class _CamWrapper(torch.nn.Module):
        def __init__(self, backbone: torch.nn.Module):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            features = self.backbone.forward_features(x)
            if features.ndim == 3:
                return features[:, 0, :]
            return features

    model = _CamWrapper(backbone)
    model.eval().to(device)
    return model, checkpoint_size, architecture.model_name


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Resolve a dotted module path.

    Delegates to :func:`otuformer.vision.features.get_module_by_name`: spatial
    extraction and CAM look a module up the same way, and CAM keeps accepting
    arbitrary attribution targets (not only complete blocks).
    """
    from otuformer.vision.features import get_module_by_name as _lookup

    return _lookup(model, name)


def find_last_conv_module(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "backbone", model)
    last_conv = None
    for _, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError(
            "No Conv2d layer found. Specify --target-layer-name manually."
        )
    return last_conv


def default_vit_target(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        block = backbone.blocks[-1]
        for candidate in ["norm1", "ln1", "ln"]:
            if hasattr(block, candidate):
                return getattr(block, candidate)
        return block
    raise RuntimeError("Could not find ViT block. Specify --target-layer-name.")


def infer_architecture(base_model_name: str, model: torch.nn.Module) -> str:
    name = (base_model_name or model.__class__.__name__).lower()
    if "vit" in name or "transformer" in name:
        return "vit"
    return "cnn"


def vit_reshape_transform(
    tensor: torch.Tensor,
    prefix_tokens: int = 1,
    *,
    grid_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Reshape ViT tokens into a grid, dropping every prefix token.

    ``prefix_tokens`` is the loaded backbone's ``num_prefix_tokens`` so CLS and
    register tokens never become image pixels; the default keeps the historical
    one-prefix behaviour for direct callers.

    ``grid_hw`` is the real patch grid of the loaded input. Production callbacks
    always pass it, because a square root of the token count is only correct for a
    square input; without it that historical fallback is used and still validated.
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected (B, N, C), got {tensor.shape}")
    prefix = int(prefix_tokens)
    if prefix < 0 or prefix >= int(tensor.shape[1]):
        raise ValueError(
            f"prefix_tokens must leave at least one patch token, got {prefix} for "
            f"{int(tensor.shape[1])} tokens"
        )
    tensor = tensor[:, prefix:, :]
    batch, tokens, channels = tensor.shape
    if grid_hw is not None:
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"grid_hw must be positive, got {grid_hw}")
        if grid_h * grid_w != tokens:
            raise ValueError(
                f"{tokens} patch tokens do not match the {grid_h}x{grid_w} grid"
            )
        return tensor.permute(0, 2, 1).reshape(batch, channels, grid_h, grid_w)
    spatial_dim = int(tokens**0.5)
    if spatial_dim * spatial_dim != tokens:
        raise ValueError(
            "Token count cannot form a square grid; pass grid_hw from the loaded "
            "input and patch layout."
        )
    return tensor.permute(0, 2, 1).reshape(batch, channels, spatial_dim, spatial_dim)


def vit_input_grid(model: torch.nn.Module, image_size: int | None) -> tuple[int, int] | None:
    """The real patch grid for a loaded backbone at ``image_size``."""
    if image_size is None:
        return None
    from otuformer.utils.size import resolve_backbone_patch_size

    patch = resolve_backbone_patch_size(model)
    if patch is None:
        return None
    size = int(image_size)
    if size % patch:
        raise ValueError(
            f"input size {size} is not divisible by patch size {patch}; the CAM "
            "grid cannot be derived from a square-root guess"
        )
    return (size // patch, size // patch)


def _bind_reshape_transform(
    model: torch.nn.Module, *, grid_hw: tuple[int, int] | None = None
):
    """Bind the loaded backbone's prefix-token count and real grid."""
    backbone = getattr(model, "backbone", model)
    prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1))

    def reshape(tensor: torch.Tensor) -> torch.Tensor:
        return vit_reshape_transform(
            tensor, prefix_tokens=prefix_tokens, grid_hw=grid_hw
        )

    return reshape


class _DisplayView:
    """The geometric part of one eval protocol, executed by the shared owner."""

    def __init__(self, name: str, size: int) -> None:
        if name not in EVAL_TRANSFORMS:
            raise ValueError(
                f"Unknown eval transform '{name}'; choose from: "
                f"{', '.join(EVAL_TRANSFORMS)}"
            )
        self.name = name
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        from otuformer.vision.geometry import apply_geometry, evaluation_view_ops

        viewed, _ = apply_geometry(
            image, evaluation_view_ops(image, name=self.name, size=self.size)
        )
        return viewed


def build_display_transform(name: str, size: int) -> transforms.Compose:
    """Return the display transform matching the named eval protocol.

    The display transform mirrors the geometric (resize/pad/crop) part of the
    preprocessing without the tensor/normalization steps. It executes the same
    operation list the extractor and :func:`map_cam_to_original` use, so the
    overlay, the field of view and the model input cannot disagree.
    """
    return transforms.Compose([_DisplayView(name, size)])


def _dim_desaturate(rgb: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Dim and desaturate an RGB image (H, W, 3) in [0, 1]."""
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return np.repeat(gray[..., None], 3, axis=2) * factor


def map_cam_to_original(
    cam_norm: np.ndarray,
    original_size: tuple[int, int],
    eval_transform: str,
    model_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map the model-resolution CAM back to original-image coordinates.

    Returns ``(cam_on_full, fov_mask)`` where ``cam_on_full`` is a (H, W)
    float map in [0, 1] at the original resolution and ``fov_mask`` is a bool
    (H, W) marking pixels inside the model's field of view.

    - ``center-crop``: the model's field of view is the center crop of the
      resized image; the CAM is placed into that rectangle on a canvas sized
      to the resized image, then the canvas is resized back to the original.
    - ``whole-specimen-pad``: the whole specimen is in view (nothing is
      cropped), so the CAM covers the full original and the mask is all-True.
    """
    import cv2

    from otuformer.vision.geometry import evaluation_view_operations

    width, height = original_size
    # The crop bounds and padding offsets come from the same owner that builds the
    # extracted view, so CAM and extraction cannot disagree about the field of view.
    operations = evaluation_view_operations(
        (width, height), name=eval_transform, size=model_size
    )
    if eval_transform == "whole-specimen-pad":
        pad = next((op for op in operations if op.kind == "pad"), None)
        side = int(pad.params["canvas_wh"][0]) if pad is not None else max(width, height)
        offset_x, offset_y = (
            (int(pad.params["offset"][0]), int(pad.params["offset"][1]))
            if pad is not None
            else (0, 0)
        )
        padded = cv2.resize(cam_norm, (side, side), interpolation=cv2.INTER_LINEAR)
        region = padded[offset_y : offset_y + height, offset_x : offset_x + width]
        cam_on_full = cv2.resize(
            region, (width, height), interpolation=cv2.INTER_LINEAR
        )
        return cam_on_full, np.ones((height, width), dtype=bool)

    # center-crop: the recorded resize target and the recorded integer crop bounds.
    resize = next(op for op in operations if op.kind == "resize")
    crop = next(op for op in operations if op.kind == "crop")
    resized_w, resized_h = (int(v) for v in resize.params["to_wh"])
    x0 = int(crop.params["left"])
    y0 = int(crop.params["top"])
    canvas = np.zeros((resized_h, resized_w), dtype=np.float32)
    fov = np.zeros((resized_h, resized_w), dtype=bool)
    canvas[y0 : y0 + model_size, x0 : x0 + model_size] = cam_norm
    fov[y0 : y0 + model_size, x0 : x0 + model_size] = True
    cam_on_full = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_LINEAR)
    fov_full = cv2.resize(
        fov.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return cam_on_full, fov_full > 0.5


def prepare_cam(
    model: torch.nn.Module,
    arch: str,
    target_layer_name: Optional[str],
    cam_name: str,
    cam_batch_size: int,
    *,
    image_size: int | None = None,
):
    if not _HAS_CAM:
        raise ImportError("pytorch-grad-cam is required for CAM generation")
    if target_layer_name:
        backbone = getattr(model, "backbone", model)
        target_layers = [get_module_by_name(backbone, target_layer_name)]
    else:
        target_layers = (
            [find_last_conv_module(model)]
            if arch == "cnn"
            else [default_vit_target(model)]
        )

    reshape_transform = (
        _bind_reshape_transform(model, grid_hw=vit_input_grid(model, image_size))
        if arch == "vit"
        else None
    )
    cam_kwargs = {
        "model": model,
        "target_layers": target_layers,
        "reshape_transform": reshape_transform,
    }
    if cam_name == "ablationcam" and arch == "vit":
        cam_kwargs["ablation_layer"] = AblationLayerVit()
    cam = CAM_METHODS[cam_name](**cam_kwargs)
    if isinstance(cam, BaseCAM):
        cam.batch_size = cam_batch_size
    return cam, target_layers, reshape_transform


def collect_image_rows(images_dir: Path, label_csv: Optional[Path]) -> pd.DataFrame:
    if label_csv is not None:
        df = pd.read_csv(label_csv)
        if "image" not in df.columns:
            raise ValueError("label_csv must include 'image' column")
        if "label" not in df.columns:
            df["label"] = ""
        return df[["image", "label"]]
    images = [
        p.relative_to(images_dir).as_posix()
        for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return pd.DataFrame({"image": images, "label": [""] * len(images)})


def selected_image_paths(
    images_dir: Path, label_csv: Optional[Path], max_images: Optional[int] = None
) -> list[Path]:
    """Keep cleanup protection and CAM image selection identical."""
    rows = collect_image_rows(images_dir, label_csv)
    if max_images is not None:
        rows = rows.head(max(0, max_images))
    image_map = {}
    if label_csv is not None:
        for path in images_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_map[path.name] = path
    return [image_map.get(ref, images_dir / ref) for ref in rows["image"]]


def _resolve_device(device: str) -> torch.device:
    return resolve_device(device)


def process_image(
    img_path: Path,
    label: str,
    model: torch.nn.Module,
    preprocess,
    display_transform,
    cam_extractor,
    device: torch.device,
    fig_dir: Path,
    array_dir: Optional[Path],
    image_weight: float,
    fig_format: str,
    save_npy: SaveMode,
    eval_transform: str = "center-crop",
    model_size: int = 224,
) -> Dict:
    import cv2

    _validate_save_mode(save_npy)
    pil_img = Image.open(img_path).convert("RGB")
    original_img = pil_img.copy()
    rgb_display = np.array(original_img).astype(np.float32) / 255.0

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(input_tensor)
        if logits.ndim == 3:
            logits = logits[:, 0, :]
        pred_idx = int(torch.argmax(logits, dim=1).item())
        probs = torch.softmax(logits, dim=1)
        pred_score = float(probs[0, pred_idx].cpu().item())

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]

    cam_norm = grayscale_cam - grayscale_cam.min()
    if cam_norm.max() > 0:
        cam_norm = cam_norm / cam_norm.max()
    else:
        cam_norm = np.zeros_like(cam_norm)

    cam_on_full, fov_mask = map_cam_to_original(
        cam_norm, original_img.size, eval_transform, model_size
    )

    overlay = show_cam_on_image(
        rgb_display,
        cam_on_full,
        use_rgb=True,
        image_weight=image_weight,
    )
    # show_cam_on_image returns uint8 in [0, 255]; rescale to [0, 1] so the
    # dim/desaturate mix and the *255 uint8 conversion below do not overflow.
    overlay = overlay.astype(np.float32) / 255.0
    if not fov_mask.all():
        overlay = np.where(fov_mask[..., None], overlay, _dim_desaturate(rgb_display))
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    combined = Image.new("RGB", (original_img.width * 2, original_img.height))
    combined.paste(original_img, (0, 0))
    combined.paste(overlay_img, (original_img.width, 0))

    stem = img_path.stem
    fig_path = fig_dir / f"{stem}_cam.{fig_format}"
    combined.save(fig_path)

    cam_array_path = ""
    if save_npy != "none" and array_dir is not None:
        npy_path = array_dir / f"{stem}.npy"
        cam_array = grayscale_cam if save_npy == "raw" else cam_norm
        np.save(npy_path, cam_array.astype(np.float32))
        cam_array_path = str(npy_path)

    return {
        "image": img_path.name,
        "label": label,
        "pred_class": pred_idx,
        "pred_prob": pred_score,
        "figure_path": str(fig_path),
        "cam_array_path": cam_array_path,
    }


def run_cam(
    checkpoint: Path,
    images_dir: Path,
    out_dir: Path,
    model_name: str = "vit_tiny_patch16_224",
    label_csv: Optional[Path] = None,
    cam_method: str = "gradcam",
    arch: Optional[str] = None,
    target_layer_name: Optional[str] = None,
    image_weight: float = 0.5,
    fig_format: str = "png",
    save_npy: SaveMode = "none",
    dump_model_structure: bool = False,
    max_images: Optional[int] = None,
    cam_batch_size: int = 1,
    device: str = "auto",
    eval_transform: str = "center-crop",
    selection: Optional[tuple[pd.DataFrame, list[Path]]] = None,
) -> None:
    """Render CAM heatmaps for one ordered image selection.

    ``selection`` is the CLI's already-prepared ``(rows, paths)`` pair, so output
    protection and execution share one selection; a direct call prepares its own.
    """
    from otuformer.utils.io import write_csv

    _validate_save_mode(save_npy)
    if eval_transform not in EVAL_TRANSFORMS:
        raise ValueError(
            f"Unknown eval transform '{eval_transform}'; "
            f"choose from: {', '.join(EVAL_TRANSFORMS)}"
        )
    dev = _resolve_device(device)
    print(f"Resolved device: {dev}")
    model, checkpoint_size, resolved_name = load_model_from_checkpoint(
        checkpoint, model_name, dev
    )
    preprocess = build_eval_transform(eval_transform, checkpoint_size)
    display_transform = build_display_transform(eval_transform, checkpoint_size)

    resolved_arch = arch or infer_architecture(resolved_name, model)
    cam_extractor, target_layers, reshape_transform = prepare_cam(
        model, resolved_arch, target_layer_name, cam_method, cam_batch_size
    )

    logging.info("Architecture inferred as %s", resolved_arch)
    logging.info(
        "Using target layer(s): %s",
        ", ".join([layer.__class__.__name__ for layer in target_layers]),
    )
    if reshape_transform:
        logging.info("Enabled ViT reshape_transform for CAM.")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    array_dir = (out_dir / "arrays") if save_npy != "none" else None
    if array_dir:
        array_dir.mkdir(parents=True, exist_ok=True)

    if dump_model_structure:
        backbone = getattr(model, "backbone", model)
        lines = ["# Named modules for --target-layer-name", "# Format: <name>\t<class>"]
        for name, module in backbone.named_modules():
            lines.append(f"{name or '<root>'}\t{module.__class__.__name__}")
        (out_dir / "model_layers.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        logging.info("Model layer names written to %s", out_dir / "model_layers.txt")

    if selection is None:
        rows_df = collect_image_rows(images_dir, label_csv)
        if max_images is not None:
            rows_df = rows_df.head(max_images)
        paths = selected_image_paths(images_dir, label_csv, max_images)
    else:
        rows_df, paths = selection
    results = []
    for (_, row), selected_path in zip(rows_df.iterrows(), paths):
        img_path = selected_path.resolve()
        if not img_path.exists():
            logging.error("Image not found: %s", img_path)
            continue
        try:
            result = process_image(
                img_path=img_path,
                label=str(row["label"]),
                model=model,
                preprocess=preprocess,
                display_transform=display_transform,
                cam_extractor=cam_extractor,
                device=dev,
                fig_dir=fig_dir,
                array_dir=array_dir,
                image_weight=image_weight,
                fig_format=fig_format,
                save_npy=save_npy,
                eval_transform=eval_transform,
                model_size=checkpoint_size,
            )
            results.append(result)
        except Exception as exc:
            logging.exception("Failed on %s: %s", img_path, exc)

    if results:
        pd.DataFrame(results).to_csv(out_dir / "cam_summary.csv", index=False)
