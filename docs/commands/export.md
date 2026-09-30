# otuformer export

[English](export.md) | [中文](export.cn.md)

## Purpose

Export the encoder and its projector to ONNX for deployment and CPU-accelerated inference. The exported model takes images and returns embedding vectors directly. When ONNX Runtime is installed, the export validates its own numerics against CPU PyTorch at the export size.

## Usage

```bash
otuformer export \
    --checkpoint runs/finetune/finetune_latest.pth \
    --out-dir runs/export
```

## Parameters

### `--checkpoint`

**Default:** `Required`

Checkpoint path.

### `--out-dir`

**Default:** `runs/export`

Output directory.

### `--imgsz`

**Default:** `auto` · **Allowed:** auto, or an integer (common: 224 / 384 / 448; 518 patch-14 only)

Input image size for ONNX export. 'auto' uses the checkpoint's recorded training size. Common examples: 224, 384, 448 (e.g. 518 for patch-14 models).

### `--opset`

**Default:** `18` · **Allowed:** integer

ONNX opset version.

### `--model-name`

**Default:** `vit_tiny_patch16_224`

Fallback timm backbone for checkpoints that record no model name (ref-script fine-tune checkpoints store neither config nor args).

### `--overwrite`

**Default:** `No`

Clear an existing non-empty output directory.

## Inputs

`--checkpoint` is an OTU checkpoint or a ref-script checkpoint (`ref/ibot20260115.py`). The projector is rebuilt from `config.embedding_head`, falling back to the projector shapes recorded in the weights.

## Outputs

- `encoder.onnx` — ONNX encoder model

## Examples

```bash
# Basic usage
otuformer export \
    --checkpoint runs/finetune/finetune_latest.pth \
    --out-dir runs/export

# Custom image size and opset
otuformer export \
    --checkpoint runs/finetune/finetune_latest.pth \
    --imgsz 224 \
    --opset 17 \
    --out-dir runs/export
```

## Notes

Accepts OTU checkpoints and ref-script checkpoints (`ref/ibot20260115.py`). The projector is rebuilt from `config.embedding_head`, falling back to the projector shapes in the weights.
