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

### `--output-kind`

**Default:** `embedding` · **Allowed:** embedding / spatial

Graph product: 'embedding' keeps the existing encoder+projector graph with one 'embedding' output; 'spatial' exports one named output per selected complete backbone block plus a fixed feature contract.

### `--feature-layers`

**Default:** `None` (last complete block)

Spatial only. Comma-separated backbone-relative complete block paths (for example 'blocks.5,blocks.11'). Omitted: the last complete block.

### `--feature-norm`

**Default:** `backbone` · **Allowed:** backbone / none

Spatial only. Apply the backbone's final norm to each exported block ('backbone') or export the raw block output ('none').

### `--feature-l2-normalize`

**Default:** `No`

Spatial only. L2-normalize each token vector after the backbone norm and export that representation.

### `--use-student`

**Default:** `No`

Spatial export only. Export the student weights; the graph records the weight source it actually used. Rejected for embedding export, whose graph has always exported the teacher/default weights.

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

Accepts OTU checkpoints and ref-script checkpoints (`ref/ibot20260115.py`). The projector is rebuilt from `config.embedding_head`, falling back to the projector shapes in the weights. A DINO SSL checkpoint exports the normalized D-dimensional bottleneck: the training-only prototype layers, their target softmax and the K-dimensional CLS center are never part of the graph, so the exported width is D (or the metric width of a fine-tuned descendant), never K.

**Spatial export (`--output-kind spatial`).** The graph holds one named output per
selected complete block (`spatial_000`, `spatial_001`, ...), each BHWC per batch and
HWC per image, plus a canonical feature contract in the graph metadata under
`otuformer.spatial.v1`: schema version, ordered output names, block paths, grid and
channel width, prefix/patch lattice, input size and normalization, feature norm and
L2 with epsilon, and the actual model identity and weight source. The export fixes
the model, layers, normalization and size; a reader compares an explicit request
against that contract instead of guessing from the graph.

Validation is mandatory and covers **every** declared output at batches 1 and 2
against CPU FP32 PyTorch within `atol=1e-4`, `rtol=1e-3`, with per-output max errors
reported. A graph without ONNX Runtime available, without a dynamic batch, or with a
mismatched/missing contract is not a spatial model: the attempt publishes nothing and
the previous graph and report stay in place. Channel width is not the embedding
`out_dim`, and per-image geometry never enters the graph.
