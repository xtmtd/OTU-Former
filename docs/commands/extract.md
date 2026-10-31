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

CSV with a required 'image' column plus optional 'label' and any other metadata. Supplying it selects the shared, strictly validated data path; it is the selector, the metadata source, and the metrics/UMAP label source. In attention-pool mode it must include labels when cached pooling weights are missing.

### `--coco-instances`

**Default:** `None`

COCO instance annotations (boxes, polygons, RLE).

### `--coco-keypoints`

**Default:** `None`

COCO keypoint annotations.

### `--coco-panoptic`

**Default:** `None`

COCO panoptic JSON. Requires --panoptic-masks-dir.

### `--panoptic-masks-dir`

**Default:** `None`

Root for panoptic segment-ID PNGs, resolved from each annotation's file_name as written. Requires --coco-panoptic.

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

Every input this run reads -- the image root, `--label-csv`, checkpoint, ONNX graph
and its external tensor files, reusable pooling checkpoints -- is collected before
anything is cleared, and an output directory that would delete one of them is
refused. An ONNX graph whose external-data pointers are ambiguous (duplicate
`location` entries) or that resolve outside the graph directory is rejected, because
readers disagree on which file such a graph reads. Links are followed the way the kernel follows them, including relays and
`..` after a link, so a path entry that cleanup would delete is refused even when the
file itself lives elsewhere.

The two data paths differ in one place, on purpose:

- **Shared path** (`--label-csv` or any `--coco-*` option): an output directory that
  contains an input, or that sits inside one, is refused outright.
- **Legacy path** (no CSV, no `--coco-*` option): only deletion is refused. Writing
  `--out-dir` *inside* the image root stays supported, because the run only creates
  files there; an output directory equal to, or containing, the image root is
  refused.

**Concurrency boundary.** The check happens before the directory is cleared, so it is
a local single-process contract: it does not defend against another process modifying
the tree in between (the classic check-then-act window). What the run guarantees is
detection -- after clearing, each protected input must still resolve to the same real
path, and if it does not, the run stops immediately naming that input instead of
continuing on a path that is no longer the checked one.

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

**Shared data path (v2).** The options below together select a strictly validated data path that stays compatible with legacy extraction.

Supplying `--label-csv` or any `--coco-*` option selects the shared, strictly
validated data path. Supplying none of them keeps the legacy behaviour described
above, including its pandas CSV parsing.

- **Parsing.** The shared path parses the selector once and passes the same records
  to selection, attention-query training, evaluation and provenance. Literal `NA`,
  `N/A`, `unknown`, `0` and leading-zero identifiers are preserved as strings, and
  only an empty cell is missing. There is no separate manifest option: one CSV is
  the selector, the metadata source and the label source.
- **Identity.** The internal key is the normalized POSIX path relative to
  `--input-images-dir`. CSV-selected rows keep their original `image` text as the
  output `id`; rows selected from COCO only use that key. COCO `file_name` values are
  resolved as explicit root-relative paths without the CSV basename fallback.
- **Annotations.** COCO documents are source-local: `instances`, `keypoints` and
  `panoptic` are fixed roles with at most one file each, and a combined
  instance/keypoint document may serve both roles. Category IDs stay source-local and
  never become species labels. Ingesting boxes, polygons, RLE, keypoints and panoptic
  maps is reading and validation only: no detector, head or task prediction runs here.
- **Panoptic roots.** Each annotation's `file_name` is joined to
  `--panoptic-masks-dir` exactly as written. Nothing is prefixed, stripped or
  searched for: a prefixed name needs the parent directory as the root.
- **Mask dependency.** Polygon and RLE decoding need the optional extra:
  `pip install 'otuformer[annotations]'`. Without it, a source that needs it fails
  with installation guidance instead of skipping validation. Panoptic segment-ID
  PNG decoding and its pixel checks use only Pillow and NumPy and need no extra.
- **Validation scope.** Structural problems (types, IDs, references, numeric
  payloads, RLE run totals) are checked for the whole source. Geometry against the
  actual image, decoding and pixel agreement are checked for selected images only;
  unselected files are recorded as not performed, and a well-formed unselected image
  need not exist locally.
- **Missing annotations are not biological absence.** An image absent from a source,
  an image with no annotations, an empty mask, `v=0` versus `v=1` keypoints and
  panoptic void are distinct states. No negative mask or trusted background is
  invented, and custom `ignore`/`unknown` fields are preserved but not interpreted.

The shared path adds three artifacts to `--out-dir`:

| File | Contents |
|---|---|
| `embeddings.csv` | the same embedding columns as the legacy path |
| `data_records.jsonl` | one row per emitted embedding, in embedding order, with `schema_version`, `embedding_row`, `output_id`, `image_key`, dimensions, image SHA-256, metadata and `(source_id, image_id)` references |
| `data_validation.json` | status, scope, counts, methods/tolerances, complete issues and polygon rasterization diagnostics |
| `run_provenance.json` | run status, requested versus effective settings, source identities and hashes, selection and content digests, label schema, metric protocol and produced artifacts |

The `artifacts` list records every file this run produced, with an output-relative
path, byte count and SHA-256. That includes a pooling checkpoint trained here
(`checkpoints/<backbone>.<policy>-<type>_pooling.pth`): it is the only record of which
reusable weights the run generated, and if the run fails before publishing, the
rollback deletes it rather than leaving it unregistered. A pooling checkpoint reused
from disk is an input, so it appears in `used_files` instead and is never deleted.

A pooling checkpoint trained here is published by exclusive creation: an existing
file is never overwritten, and a run that fails at any point after reserving the name
removes the file it created. The guarantee is about the directory entry, not about
partial visibility -- another process reading that path while this run is still
writing can observe an incomplete file. Treat the cache as readable only after the
producing run has returned.

A run writes `run_provenance.json` with status `running` once the output directory
passes safety checks, then `completed`, `failed` or `interrupted`. Validation failure
publishes `data_validation.json` and a failed record but no embedding/mapping pair; a
hard termination can leave `running`, which never means success. Legacy runs without
the new options do not emit these files.
