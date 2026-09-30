# otuformer cam

[English](cam.md) | [中文](cam.cn.md)

## Purpose

Generate CAM heatmaps showing which image regions the model uses. Six algorithms are available (Grad-CAM, Grad-CAM++, LayerCAM, Score-CAM, Eigen-CAM, Ablation-CAM), and the overlay is inverted through the same preprocessing the model saw, so the heatmap lines up with the original image.

## Usage

```bash
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --out-dir runs/cam
```

## Parameters

### `--checkpoint`

**Default:** `Required`

Path to the OTU-Former checkpoint (.ckpt or .pth).

### `--images-dir`

**Default:** `Required`

Directory containing images to generate CAM heatmaps for.

### `--label-csv`

**Default:** `None`

Optional CSV with 'image' and 'label' columns. If omitted, all images in --images-dir are used.

### `--out-dir`

**Default:** `runs/cam`

Directory to write CAM visualizations and artifacts.

### `--cam-method`

**Default:** `gradcam` · **Allowed:** gradcam / gradcampp / layercam / scorecam / eigencam / ablationcam

CAM algorithm: gradcam, gradcampp, layercam, scorecam, eigencam, ablationcam

### `--arch`

**Default:** `None` · **Allowed:** cnn / vit (auto-detected when omitted)

Force architecture type (cnn, vit; auto-detected from model name if not set).

### `--target-layer-name`

**Default:** `None`

Specific model layer for CAM (auto-selected when omitted).

### `--image-weight`

**Default:** `0.5`

Blend weight of original image in CAM overlay (0-1).

### `--fig-format`

**Default:** `png` · **Allowed:** png / jpg / pdf

Output format for CAM figures: png, jpg, pdf.

### `--save-npy`

**Default:** `none` · **Allowed:** none / raw / normalized

Save CAM arrays: none (default), raw (positive unnormalized CAM), or normalized (per-image min-max [0, 1]).

### `--dump-model-structure`

**Default:** `No`

Write model layer names to out-dir/model_layers.txt for --target-layer-name reference.

### `--max-images`

**Default:** `None`

Maximum number of images to process (None = all).

### `--cam-batch-size`

**Default:** `32`

Batch size for CAM inference.

### `--eval-transform`

**Default:** `center-crop` · **Allowed:** center-crop / whole-specimen-pad

Evaluation preprocessing protocol: center-crop (Resize + CenterCrop; heatmap confined to the model's field of view) or whole-specimen-pad (aspect-preserving square padding; heatmap covers the whole specimen). Default: center-crop.

### `--num-workers`

**Default:** `4`

Number of dataloader worker processes (reserved).

### `--model-name`

**Default:** `vit_tiny_patch16_224`

Fallback timm backbone for checkpoints that record no model name (ref-script fine-tune checkpoints store neither config nor args).

### `--device`

**Default:** `auto` · **Allowed:** auto / cpu / cuda / mps

Compute device for CAM generation: auto, cpu, cuda, mps.

### `--overwrite`

**Default:** `No`

Clear an existing non-empty output directory.

## Inputs

`--images-dir` holds the images to visualize. Optional `--label-csv` with `image` and `label` columns restricts the run; without it every image in the directory is used.

## Outputs

- `figures/` — CAM overlay images
- `cam_summary.csv` — Metadata summary
- `arrays/` — CAM arrays, created only for `--save-npy raw` or `--save-npy normalized` (no array is written by default). Saved arrays stay `float32` at model input resolution.
- `model_layers.txt` — Model layer names (with `--dump-model-structure`)

## Examples

```bash
# Basic usage
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --out-dir runs/cam

# Use Grad-CAM++ and save raw arrays
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --cam-method gradcampp \
    --save-npy raw \
    --out-dir runs/cam

# Inspect model layer structure first
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --dump-model-structure \
    --out-dir runs/cam
```

## Notes

Accepts OTU checkpoints and ref-script checkpoints (`ref/ibot20260115.py`); CAM uses the backbone only.

**CAM array semantics**:
- `--save-npy raw` keeps the positive unnormalized CAM magnitude (after ReLU and resize to model resolution). `--save-npy normalized` writes the per-image min-max `[0, 1]` copy, which preserves the previous saved-array semantics. The overlay always uses a separate normalized copy, and pixel-identical PNG output is not a contract.
- Raw magnitudes are comparable only within the same model, target layer, and preprocessing configuration; they are not comparable across backbones, layers, or CAM methods. `eigencam` raw values come from an SVD projection with arbitrary sign and are exported for completeness, but are unsuitable for response-strength statistics.

`cam` and `extract` validate enumerated query parameters natively and reject values outside their choices before any output directory is created.
