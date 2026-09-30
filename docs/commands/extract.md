# otuformer extract

[English](extract.md) | [中文](extract.cn.md)

## Purpose

Extract embeddings from images with a trained checkpoint, producing the `embeddings.csv` that `cluster`, `diversity`, and `annotate` consume. It runs a PyTorch model or an exported ONNX model (2-5x faster on CPU). Three token modes are available — raw CLS, `patch-topk`, and `attention-pool` — and label-based quality metrics plus UMAP are written when labels are available.

## Usage

```bash
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --out-dir runs/extract
```

## Parameters

### `--checkpoint`

**Default:** `None`

Path to pretrain or finetune checkpoint. Required unless --onnx-path is provided.

### `--input-images-dir`

**Default:** `Required`

Image directory or parent directory.

### `--out-dir`

**Default:** `runs/extract`

Output directory.

### `--model-name`

**Default:** `vit_tiny_patch16_224`

timm backbone name

### `--extract-size`

**Default:** `auto` · **Allowed:** auto, or an integer (common: 224 / 384 / 448; 518 patch-14 only)

Resize/crop size for extraction. 'auto' uses the checkpoint's recorded training size. Common examples: 224, 384, 448 (e.g. 518 for patch-14 models).

### `--eval-transform`

**Default:** `center-crop` · **Allowed:** center-crop / whole-specimen-pad

Evaluation preprocessing protocol: center-crop (Resize + CenterCrop) or whole-specimen-pad (aspect-preserving square padding). Default: center-crop.

### `--use-projector-output`

**Default:** `No`

Use SSL projector output for SSL checkpoints, or the ArcFace embedding for fine-tune checkpoints; useful for fine-tuned-class retrieval and closed-set clustering.

### `--use-student`

**Default:** `No`

Load student weights instead of teacher (EMA model). Default loads teacher, which gives better embeddings for downstream analysis (DINO/iBOT convention). No effect on finetune checkpoints.

### `--token-mode`

**Default:** `cls` · **Allowed:** cls / patch-topk / attention-pool

Embedding token mode. cls=raw CLS token (the comparison baseline for unseen classes, cross-dataset transfer, and general morphology); patch-topk=top-K patch tokens + PCA; attention-pool=learned query pooling over patch tokens.

### `--topk-patches`

**Default:** `20`

Top-K patch tokens for patch-topk mode (common: 10/20/30).

### `--attention-pooling-type`

**Default:** `lightweight` · **Allowed:** lightweight / multihead / gated

Attention pooling type. Choices: lightweight / multihead / gated.

### `--attention-pooling-epochs`

**Default:** `20`

Epochs to finetune attention query when no cached pooling checkpoint is found.

### `--label-csv`

**Default:** `None`

CSV with an 'image' column and optional 'label' for selecting images, evaluating embedding quality, and generating UMAP. Without labels, only UMAP is generated. In attention-pool mode this CSV must include labels when cached pooling weights are missing. CSV with `image` and optional `label`; image-only input generates UMAP without supervised metrics; attention-pool training still needs `label`

### `--metrics-sample-size`

**Default:** `10000`

Max samples for metrics and UMAP evaluation (<=0 means no cap).

### `--umap-n-neighbors`

**Default:** `15`

UMAP n_neighbors.

### `--umap-min-dist`

**Default:** `0.1`

UMAP min_dist.

### `--umap-metric`

**Default:** `cosine` · **Allowed:** cosine / euclidean

Distance metric for UMAP projection. Common choices: cosine, euclidean.

### `--visualize-class-number`

**Default:** `20`

Max classes to show in UMAP plot.

### `--disable-umap`

**Default:** `No`

Skip UMAP generation even when --label-csv is provided.

### `--batch-size`

**Default:** `32`

Batch size.

### `--num-workers`

**Default:** `4`

DataLoader workers.

### `--device`

**Default:** `auto` · **Allowed:** auto / cpu / cuda / mps

Device: auto / cpu / cuda / mps.

### `--onnx-path`

**Default:** `None`

Path to exported ONNX model for inference. When provided, uses ONNX Runtime instead of PyTorch. Only supports token-mode 'cls' (default). Overrides --checkpoint, --model-name, --use-student, and --device.

### `--seed`

**Default:** `42`

Random seed.

### `--overwrite`

**Default:** `No`

Clear an existing non-empty output directory.

## Inputs

`--input-images-dir` is an image directory, or a parent directory of per-class subdirectories (batch mode). Optional `--label-csv` needs an `image` column and accepts an optional `label` column.

## Outputs

- `embeddings.csv` — Embedding vectors (`id`, `dim_0`, `dim_1`, ...)
- `metrics.csv` — Quality metrics (when `--label-csv` provides at least two classes)
- `umap.pdf` — UMAP visualization (when `--label-csv` is provided and not disabled)

Embedding-quality metrics use the shared [embedding metrics](embedding-metrics.md) definitions and field names.

## Examples

```bash
# Using PyTorch checkpoint (CLS token mode)
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --out-dir runs/extract

# Using ONNX model (2-5x faster on CPU)
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --onnx-path runs/export/encoder.onnx \
    --out-dir runs/extract

# With label CSV in attention-pool mode
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --label-csv labels.csv \
    --token-mode attention-pool \
    --out-dir runs/extract
```

## Notes

Accepts OTU checkpoints and ref-script checkpoints (`ref/ibot20260115.py`). The embedding head is rebuilt from `config.embedding_head`, falling back to the projector shapes in the weights.

`--label-csv` requires an `image` column and accepts an optional `label` column. Image-only input generates UMAP without supervised metrics; attention-pool query training still requires labels.

Embedding-quality metrics use the shared [embedding metrics](embedding-metrics.md) definitions and field names.

`cam` and `extract` validate enumerated query parameters natively and reject values outside their choices before any output directory is created.
