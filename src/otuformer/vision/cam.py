"""CAM heatmap generation for OTU-Former backbones."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from otuformer.utils.checkpoint import load_checkpoint
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

CAM_METHODS = (
    {
        "gradcam": GradCAM,
        "gradcampp": GradCAMPlusPlus,
        "layercam": LayerCAM,
        "ablationcam": AblationCAM,
        "scorecam": ScoreCAM,
        "eigencam": EigenCAM,
    }
    if _HAS_CAM
    else {}
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
    cfg = ckpt.get("config", {})
    resolved_name = cfg.get("model_name", model_name)
    out_dim = cfg.get("out_dim") or cfg.get("metric_embed_dim", 256)
    checkpoint_size = resolve_training_image_size(ckpt)
    encoder = OTUFormerEncoder(
        model_name=resolved_name,
        out_dim=out_dim,
        pretrained=False,
        img_size=checkpoint_size,
    )
    validate_input_size(checkpoint_size, encoder, resolved_name)
    encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
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
    return model, checkpoint_size, resolved_name


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    module = model
    for attr in name.split("."):
        if attr.isdigit():
            idx = int(attr)
            try:
                module = module[idx]
            except Exception as exc:
                raise AttributeError(
                    f"Module '{module.__class__.__name__}' has no index '{attr}'"
                ) from exc
            continue
        if isinstance(module, torch.nn.ModuleDict) and attr in module:
            module = module[attr]
            continue
        if not hasattr(module, attr):
            raise AttributeError(
                f"Module '{module.__class__.__name__}' has no attribute '{attr}'"
            )
        module = getattr(module, attr)
    return module


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


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected (B, N, C), got {tensor.shape}")
    tensor = tensor[:, 1:, :]
    batch, tokens, channels = tensor.shape
    spatial_dim = int(tokens**0.5)
    if spatial_dim * spatial_dim != tokens:
        raise ValueError("Token count cannot form a square grid.")
    return tensor.permute(0, 2, 1).reshape(batch, channels, spatial_dim, spatial_dim)


def build_display_transform(name: str, size: int) -> transforms.Compose:
    """Return the display transform matching the named eval protocol.

    The display transform mirrors the geometric (resize/pad/crop) part of the
    preprocessing without the tensor/normalization steps.
    """
    if name == "center-crop":
        return transforms.Compose(
            [
                transforms.Resize(size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(size),
            ]
        )
    if name == "whole-specimen-pad":
        return transforms.Compose(
            [
                transforms.Lambda(pad_to_square),
                transforms.Resize((size, size), interpolation=Image.BICUBIC),
            ]
        )
    raise ValueError(
        f"Unknown eval transform '{name}'; choose from: {', '.join(EVAL_TRANSFORMS)}"
    )


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

    # center-crop: replicate torchvision Resize(shorter side) then CenterCrop.
    # torchvision truncates the long side with int(); CenterCrop offsets use
    # int(round((long - size) / 2)).
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


def prepare_cam(
    model: torch.nn.Module,
    arch: str,
    target_layer_name: Optional[str],
    cam_name: str,
    cam_batch_size: int,
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

    reshape_transform = vit_reshape_transform if arch == "vit" else None
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
    save_npy: bool,
    eval_transform: str = "center-crop",
    model_size: int = 224,
) -> Dict:
    import cv2

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
    if save_npy and array_dir is not None:
        npy_path = array_dir / f"{stem}.npy"
        np.save(npy_path, cam_norm.astype(np.float32))
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
    save_npy: bool = False,
    dump_model_structure: bool = False,
    max_images: Optional[int] = None,
    cam_batch_size: int = 1,
    device: str = "auto",
    eval_transform: str = "center-crop",
) -> None:
    from otuformer.utils.io import write_csv

    if eval_transform not in EVAL_TRANSFORMS:
        raise ValueError(
            f"Unknown eval transform '{eval_transform}'; "
            f"choose from: {', '.join(EVAL_TRANSFORMS)}"
        )
    dev = _resolve_device(device)
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
    array_dir = (out_dir / "arrays") if save_npy else None
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

    rows_df = collect_image_rows(images_dir, label_csv)
    if max_images is not None:
        rows_df = rows_df.head(max_images)

    if label_csv is not None:
        image_map: Dict[str, Path] = {}
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                image_map[p.name] = p
    else:
        image_map = {}

    results = []
    for idx, row in rows_df.iterrows():
        if max_images is not None and idx >= max_images:
            break
        if image_map and row["image"] in image_map:
            img_path = image_map[row["image"]].resolve()
        else:
            img_path = (images_dir / row["image"]).resolve()
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
