# OTU-Former Plan 4: Vision (CAM) + Export

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `vision/cam.py` (CAM heatmap generation) and the `export` command (ONNX export). Wire up `cam` and `export` CLI sub-commands.

**Architecture:** `vision/cam.py` is ported directly from `entomokit/src/classification/cam.py`, with AutoGluon support removed and OTU-Former checkpoint loading substituted. The `export` command uses `torch.onnx.export()` to write encoder + projector as ONNX, reading embedding dimension from checkpoint metadata.

**Tech Stack:** pytorch-grad-cam (`pip install grad-cam`), torch.onnx, onnx, onnxruntime (for validation)

**Prerequisites:** Plan 1 (CLI stubs), Plan 2 (training model — needed for checkpoint loading).

**Reference source:** `entomokit/src/classification/cam.py` — full implementation to port.

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/otuformer/vision/cam.py` | `run_cam()` service function; model loading, target layer selection, CAM generation, overlay saving — ported from entomokit |
| `src/otuformer/vision/export.py` | `run_export()` service function; load checkpoint, trace model, export ONNX, validate |
| `src/otuformer/cli/cam.py` | Replace stub: call `run_cam()` |
| `src/otuformer/cli/export.py` | Replace stub: call `run_export()` |
| `tests/test_cam.py` | Unit tests for model loading and target layer selection |
| `tests/test_export.py` | Unit test for ONNX export output |

---

## Task 1: `vision/cam.py` — CAM heatmap generation

Port from `entomokit/src/classification/cam.py`. Remove AutoGluon path; replace model loading with OTU-Former checkpoint loading (load `OTUFormerEncoder` backbone directly, not the projector — CAM needs the classification-style backbone features).

**Files:**
- Create: `src/otuformer/vision/cam.py`
- Create: `tests/test_cam.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cam.py
import torch
import pytest
from pathlib import Path
from PIL import Image
from otuformer.vision.cam import (
    infer_architecture,
    default_vit_target,
    get_module_by_name,
    vit_reshape_transform,
    CAM_METHODS,
)


def make_tiny_vit():
    import timm
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)


def test_cam_methods_complete():
    expected = {"gradcam", "gradcampp", "scorecam", "layercam", "eigencam", "ablationcam"}
    assert expected == set(CAM_METHODS.keys())


def test_infer_architecture_vit():
    model = make_tiny_vit()
    arch = infer_architecture("vit_tiny_patch16_224", model)
    assert arch == "vit"


def test_infer_architecture_cnn():
    import timm
    model = timm.create_model("convnextv2_femto", pretrained=False)
    arch = infer_architecture("convnextv2_femto", model)
    assert arch == "cnn"


def test_default_vit_target_returns_module():
    model = make_tiny_vit()
    target = default_vit_target(model)
    assert isinstance(target, torch.nn.Module)


def test_vit_reshape_transform_shape():
    # 14x14 = 196 patches + 1 CLS = 197
    tensor = torch.randn(2, 197, 192)
    out = vit_reshape_transform(tensor)
    assert out.shape == (2, 192, 14, 14)


def test_get_module_by_name():
    model = make_tiny_vit()
    mod = get_module_by_name(model, "blocks.0")
    assert isinstance(mod, torch.nn.Module)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cam.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `vision/cam.py`**

Port `entomokit/src/classification/cam.py` wholesale. Key adaptations:

1. Remove all AutoGluon paths (`load_ag`, `MultiModalPredictor`)
2. Model loading: load OTU-Former checkpoint → extract backbone from `OTUFormerEncoder`
3. For CAM, use the backbone (`model.backbone`) not the projector — CAM requires classification-style feature layers
4. Keep all helper functions unchanged: `get_module_by_name`, `find_last_conv_module`, `default_vit_target`, `infer_architecture`, `vit_reshape_transform`, `prepare_cam`, `process_image`, `run_cam`

```python
# src/otuformer/vision/cam.py
"""CAM heatmap generation for OTU-Former ViT backbones.

Ported from entomokit/src/classification/cam.py with AutoGluon support removed
and OTU-Former checkpoint loading substituted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import (
    GradCAM, ScoreCAM, EigenCAM, GradCAMPlusPlus, LayerCAM, AblationCAM,
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm.data import resolve_model_data_config
from torchvision.transforms.functional import InterpolationMode
import cv2

from otuformer.training.model import OTUFormerEncoder
from otuformer.utils.checkpoint import load_checkpoint

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcampp": GradCAMPlusPlus,
    "layercam": LayerCAM,
    "ablationcam": AblationCAM,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load OTU-Former encoder backbone for CAM.

    Returns the backbone (not projector) — CAM needs feature layers.
    """
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt.get("config", {})
    resolved_name = cfg.get("model_name", model_name)
    out_dim = cfg.get("out_dim") or cfg.get("metric_embed_dim", 256)
    encoder = OTUFormerEncoder(model_name=resolved_name, out_dim=out_dim)
    encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = encoder.backbone  # Use backbone for CAM
    model.eval().to(device)
    return model


# --- All helper functions ported unchanged from entomokit/src/classification/cam.py ---

def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Navigate nested module by dotted name string."""
    module = model
    for attr in name.split("."):
        if attr.isdigit():
            module = module[int(attr)]
        elif isinstance(module, torch.nn.ModuleDict) and attr in module:
            module = module[attr]
        else:
            module = getattr(module, attr)
    return module


def find_last_conv_module(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found. Specify --target-layer-name manually.")
    return last_conv


def default_vit_target(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        block = model.blocks[-1]
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
    """Reshape ViT tokens (B, N, C) → feature maps (B, C, H, W), excluding CLS."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected (B, N, C), got {tensor.shape}")
    tensor = tensor[:, 1:, :]
    batch, tokens, channels = tensor.shape
    spatial_dim = int(tokens ** 0.5)
    if spatial_dim * spatial_dim != tokens:
        raise ValueError("Token count cannot form a square grid.")
    return tensor.permute(0, 2, 1).reshape(batch, channels, spatial_dim, spatial_dim)


def build_eval_transforms(model: torch.nn.Module) -> Tuple:
    cfg = resolve_model_data_config(model)
    crop_tuple = cfg.get("test_input_size", cfg.get("input_size"))
    crop_size = crop_tuple[1]
    crop_pct = cfg.get("test_crop_pct", cfg.get("crop_pct", 1.0))
    resize_shorter = int(round(crop_size / crop_pct))
    mean = cfg.get("mean")
    std = cfg.get("std")
    interpolation = getattr(
        InterpolationMode,
        cfg.get("interpolation", "bicubic").upper(),
        InterpolationMode.BICUBIC,
    )
    preprocess = transforms.Compose([
        transforms.Resize(resize_shorter, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    display_transform = transforms.Compose([
        transforms.Resize(resize_shorter, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
    ])
    return preprocess, display_transform


def prepare_cam(
    model: torch.nn.Module,
    arch: str,
    target_layer_name: Optional[str],
    cam_name: str,
    cam_batch_size: int,
):
    target_layers = (
        [get_module_by_name(model, target_layer_name)] if target_layer_name
        else ([find_last_conv_module(model)] if arch == "cnn" else [default_vit_target(model)])
    )
    reshape_transform = vit_reshape_transform if arch == "vit" else None
    cam_kwargs = {"model": model, "target_layers": target_layers, "reshape_transform": reshape_transform}
    if cam_name == "ablationcam" and arch == "vit":
        cam_kwargs["ablation_layer"] = AblationLayerVit()
    cam = CAM_METHODS[cam_name](**cam_kwargs)
    if isinstance(cam, BaseCAM):
        cam.batch_size = cam_batch_size
    return cam, target_layers, reshape_transform


def write_model_structure(model: torch.nn.Module, out_dir: Path) -> Path:
    path = out_dir / "model_layers.txt"
    lines = ["# Named modules for --target-layer-name", "# Format: <name>\t<class>"]
    for name, module in model.named_modules():
        lines.append(f"{name or '<root>'}\t{module.__class__.__name__}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def collect_image_rows(
    images_dir: Path, label_csv: Optional[Path]
) -> pd.DataFrame:
    if label_csv is not None:
        df = pd.read_csv(label_csv)
        if "image" not in df.columns or "label" not in df.columns:
            raise ValueError("label_csv must have 'image' and 'label' columns.")
        return df[["image", "label"]]
    images = [
        p.relative_to(images_dir).as_posix()
        for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return pd.DataFrame({"image": images, "label": [""] * len(images)})


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
) -> Dict:
    """Generate and save CAM overlay for a single image."""
    pil_img = Image.open(img_path).convert("RGB")
    rgb_display = np.array(display_transform(pil_img)).astype(np.float32) / 255.0

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_tensor)

    target_class = out.argmax(dim=1).item() if out.ndim > 1 else None
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]

    rgb_display_resized = cv2.resize(
        rgb_display, (grayscale_cam.shape[1], grayscale_cam.shape[0])
    )
    overlay = show_cam_on_image(rgb_display_resized, grayscale_cam, use_rgb=True, image_weight=image_weight)

    stem = img_path.stem
    fig_path = fig_dir / f"{stem}_cam.{fig_format}"
    Image.fromarray(overlay).save(fig_path)

    if save_npy and array_dir is not None:
        import numpy as np
        np.save(array_dir / f"{stem}_cam.npy", grayscale_cam)

    return {"image": img_path.name, "label": label, "cam_file": str(fig_path)}


def run_cam(
    checkpoint: Path,
    images_dir: Path,
    out_dir: Path,
    model_name: str = "vit_small_patch16_224",
    label_csv: Optional[Path] = None,
    cam_method: str = "gradcam",
    arch: Optional[str] = None,
    target_layer_name: Optional[str] = None,
    image_weight: float = 0.5,
    fig_format: str = "png",
    save_npy: bool = False,
    dump_model_structure: bool = False,
    max_images: Optional[int] = None,
    cam_batch_size: int = 32,
    device: str = "cpu",
) -> None:
    """Generate CAM heatmaps for all images."""
    from tqdm import tqdm
    from otuformer.utils.io import write_csv

    dev = torch.device(device)
    model = load_model_from_checkpoint(checkpoint, model_name, dev)
    preprocess, display_transform = build_eval_transforms(model)

    resolved_arch = arch or infer_architecture(model_name, model)
    cam_extractor, _, _ = prepare_cam(model, resolved_arch, target_layer_name, cam_method, cam_batch_size)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    array_dir = (out_dir / "arrays") if save_npy else None
    if array_dir:
        array_dir.mkdir(parents=True, exist_ok=True)

    if dump_model_structure:
        write_model_structure(model, out_dir)

    rows_df = collect_image_rows(images_dir, label_csv)
    if max_images is not None:
        rows_df = rows_df.head(max_images)

    results = []
    for _, row in tqdm(rows_df.iterrows(), total=len(rows_df), desc="Generating CAMs"):
        img_path = images_dir / row["image"]
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
            )
            results.append(result)
        except Exception as exc:
            logging.warning("Failed to process %s: %s", img_path.name, exc)
            results.append({"image": img_path.name, "label": str(row["label"]), "cam_file": "FAILED"})

    summary_df = pd.DataFrame(results)
    write_csv(summary_df, out_dir / "cam_summary.csv")
    print(f"CAM generation complete: {len(results)} images processed → {out_dir}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cam.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/vision/cam.py tests/test_cam.py
git commit -m "feat: add CAM heatmap generation (ported from entomokit)"
```

---

## Task 2: `vision/export.py` — ONNX export

**Files:**
- Create: `src/otuformer/vision/export.py`
- Create: `tests/test_export.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_export.py
import torch
import pytest
from pathlib import Path
from otuformer.vision.export import export_to_onnx, load_exported_onnx


def make_checkpoint(tmp_path, out_dim=64):
    from otuformer.training.model import OTUFormerEncoder
    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=out_dim)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim},
    }
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, p)
    return p


def test_export_creates_onnx_file(tmp_path):
    ckpt = make_checkpoint(tmp_path)
    out_path = tmp_path / "encoder.onnx"
    report = export_to_onnx(
        checkpoint_path=ckpt,
        out_path=out_path,
        imgsz=224,
        opset=17,
    )
    assert out_path.exists()
    assert report["out_dim"] == 64
    assert report["model_name"] == "vit_tiny_patch16_224"


def test_onnx_output_shape(tmp_path):
    """Exported ONNX produces correct embedding shape."""
    ckpt = make_checkpoint(tmp_path, out_dim=64)
    out_path = tmp_path / "encoder.onnx"
    export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)
    session = load_exported_onnx(out_path)
    import numpy as np
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {"input": dummy})
    assert outputs[0].shape == (1, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_export.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `vision/export.py`**

```python
# src/otuformer/vision/export.py
"""ONNX export for OTU-Former encoder + projector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from otuformer.training.model import OTUFormerEncoder
from otuformer.utils.checkpoint import load_checkpoint
from otuformer.utils.io import write_json


def export_to_onnx(
    checkpoint_path: Path,
    out_path: Path,
    imgsz: int = 224,
    opset: int = 17,
) -> dict[str, Any]:
    """Export encoder + projector to ONNX.

    Embedding dimension is read from checkpoint metadata — not hardcoded.
    Always exports encoder + projector only (no ArcFace head).

    Args:
        checkpoint_path: Path to pretrain or finetune checkpoint.
        out_path: Output path for the .onnx file.
        imgsz: Input image size [default: 224].
        opset: ONNX opset version [default: 17].

    Returns:
        Report dict with model_name, out_dim, input_shape, output_shape.
    """
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name", "vit_small_patch16_224")
    out_dim = cfg.get("out_dim") or cfg.get("metric_embed_dim", 256)

    model = OTUFormerEncoder(model_name=model_name, out_dim=out_dim)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        do_constant_folding=True,
    )

    # Validate with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        out = session.run(None, {"input": dummy_input.numpy()})[0]
        output_shape = list(out.shape)
        validated = True
    except Exception as e:
        output_shape = [1, out_dim]
        validated = False

    report = {
        "model_name": model_name,
        "out_dim": out_dim,
        "input_shape": [1, 3, imgsz, imgsz],
        "output_shape": output_shape,
        "opset": opset,
        "validated": validated,
        "onnx_path": str(out_path),
        "note": "Exports encoder + projector only. ArcFace head excluded.",
    }
    write_json(report, out_path.parent / "export_report.json")
    return report


def load_exported_onnx(onnx_path: Path):
    """Load ONNX model for inference validation."""
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_export.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/vision/export.py tests/test_export.py
git commit -m "feat: add ONNX export for encoder + projector"
```

---

## Task 3: Wire `cli/cam.py` and `cli/export.py`

**Files:**
- Modify: `src/otuformer/cli/cam.py`
- Modify: `src/otuformer/cli/export.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Wire `cli/cam.py`**

Replace stub body:

```python
# replace stub body in cam callback:
import sys
from otuformer.vision.cam import run_cam
from otuformer.utils.logging import TeeLogger

out_dir.mkdir(parents=True, exist_ok=True)
tee = TeeLogger(out_dir / "cam.log")
sys.stdout = tee
try:
    run_cam(
        checkpoint=checkpoint,
        images_dir=images_dir,
        out_dir=out_dir,
        label_csv=label_csv,
        cam_method=cam_method,
        arch=arch,
        target_layer_name=target_layer_name,
        image_weight=image_weight,
        fig_format=fig_format,
        save_npy=save_npy,
        dump_model_structure=dump_model_structure,
        max_images=max_images,
        cam_batch_size=cam_batch_size,
        device=device,
    )
finally:
    sys.stdout = tee.terminal
    tee.close()
typer.echo(f"CAM heatmaps written to: {out_dir / 'figures'}")
typer.echo(f"Summary: {out_dir / 'cam_summary.csv'}")
```

- [ ] **Step 2: Wire `cli/export.py`**

Replace stub body:

```python
# replace stub body in export callback:
from otuformer.vision.export import export_to_onnx

out_dir.mkdir(parents=True, exist_ok=True)
onnx_path = out_dir / "encoder.onnx"
report = export_to_onnx(
    checkpoint_path=checkpoint,
    out_path=onnx_path,
    imgsz=imgsz,
    opset=opset,
)
typer.echo(f"Export complete: {onnx_path}")
typer.echo(f"  Model:      {report['model_name']}")
typer.echo(f"  Embed dim:  {report['out_dim']}")
typer.echo(f"  Validated:  {report['validated']}")
typer.echo(f"Report: {out_dir / 'export_report.json'}")
```

- [ ] **Step 3: Add CLI smoke tests**

```python
# append to tests/test_cli_smoke.py
import torch
from pathlib import Path


def _make_ckpt(tmp_path, out_dim=64):
    from otuformer.training.model import OTUFormerEncoder
    m = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=out_dim)
    ckpt = {"model_state_dict": m.state_dict(),
            "config": {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim}}
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, p)
    return p


def test_export_command(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    result = runner.invoke(app, [
        "export",
        "--checkpoint", str(ckpt),
        "--out-dir", str(tmp_path / "out"),
        "--imgsz", "224",
    ])
    assert result.exit_code == 0
    assert (tmp_path / "out" / "encoder.onnx").exists()


def test_cam_command(tmp_path):
    from PIL import Image
    ckpt = _make_ckpt(tmp_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (224, 224)).save(img_dir / "test.jpg")
    result = runner.invoke(app, [
        "cam",
        "--checkpoint", str(ckpt),
        "--images-dir", str(img_dir),
        "--out-dir", str(tmp_path / "cam_out"),
        "--cam-method", "eigencam",  # fastest method for test
        "--device", "cpu",
    ])
    assert result.exit_code == 0
    assert (tmp_path / "cam_out" / "cam_summary.csv").exists()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli_smoke.py::test_export_command tests/test_cli_smoke.py::test_cam_command -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/cli/cam.py src/otuformer/cli/export.py tests/test_cli_smoke.py
git commit -m "feat: wire cam and export CLI commands"
```

---

## Task 4: Final full-suite check

- [ ] **Step 1: Run complete test suite**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 2: Manual CLI smoke**

```bash
otuformer cam --help
otuformer export --help
otuformer doctor
```

Expected: full help printed, doctor shows all packages.

- [ ] **Step 3: Verify `--install-completion`**

```bash
otuformer --install-completion
```

Expected: shell completion instructions printed.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: plan 4 complete — vision (CAM) and ONNX export"
```
