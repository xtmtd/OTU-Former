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

**Default:** `Required` (except spatial artifact read mode)

Image directory or parent directory. Required for embedding and spatial image extraction; optional in spatial artifact read mode, where it only supplies images for report overlays.

### `--output-kind`

**Default:** `embedding` · **Allowed:** embedding / spatial

Product to write: 'embedding' keeps the existing vector output; 'spatial' writes complete backbone-block feature grids with geometry and diagnostics. No option switches it implicitly.

### `--spatial-input`

**Default:** `None`

Existing completed spatial product to read. Requires `--output-kind spatial`; no model or images are needed for numeric diagnostics, and the source product is never modified.

### `--feature-layers`

**Default:** `None` (last complete block)

Spatial only. Comma-separated backbone-relative complete block paths (for example 'blocks.5,blocks.11'). Omitted: the last complete block. Requested order is preserved; no ranges, duplicates or averaging.

### `--feature-norm`

**Default:** `backbone` · **Allowed:** backbone / none

Spatial only. Apply the backbone's final norm to each selected block ('backbone') or export the raw block output ('none').

### `--feature-l2-normalize`

**Default:** `No`

Spatial only. L2-normalize each token channel vector after the backbone norm. Independent of CAM heatmap min-max scaling.

### `--spatial-layout`

**Default:** `per-image` · **Allowed:** per-image / sharded

Spatial extraction only. 'per-image' writes one uncompressed NPZ per sample; 'sharded' writes bounded NPY shards. Both share one reader.

### `--spatial-eval`

**Default:** `No`

Enable frozen same-image cross-view correspondence (protocol spatial-geometry-v1). Rejected in this build until the evaluation task ships it; never silently skipped.

### `--spatial-eval-sample-size`

**Default:** `100`

Spatial evaluation only. Maximum number of deterministically selected images for correspondence. Rejected in read mode; use --label-csv there.

### `--dump-model-structure`

**Default:** `No`

Checkpoint spatial only. Write backbone-relative module paths with their spatial eligibility to model_layers.txt, then continue. Explicit use in embedding output is rejected rather than silently ignored.

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

**Default:** `center-crop` for embedding output, `whole-specimen-pad` for spatial output · **Allowed:** center-crop / whole-specimen-pad

Evaluation preprocessing protocol: center-crop (Resize + CenterCrop) or whole-specimen-pad (aspect-preserving square padding). Default: center-crop for embedding output, whole-specimen-pad for spatial output; an explicit value wins in both modes.

### `--use-projector-output`

**Default:** `No`

Use SSL projector output for SSL checkpoints, or the ArcFace embedding for fine-tune checkpoints; useful for fine-tuned-class retrieval and closed-set clustering.

For a DINO SSL checkpoint this option returns the normalized D-dimensional bottleneck; the K-dimensional prototype logits are training-only and are never an extraction output. The default remains the backbone CLS token for both SSL methods, and a fine-tuned checkpoint keeps its metric head width.

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

**Spatial output (`--output-kind spatial`).**

`--output-kind spatial` keeps the embedding behaviour untouched and writes a
separate product instead of `embeddings.csv`. The same `extract` command runs one
of three spatial modes, resolved from the request only:

| Mode | Required inputs | Behaviour |
| --- | --- | --- |
| Spatial checkpoint | `--checkpoint` + images (optional CSV/COCO) | Read the selected complete blocks in one forward, map geometry and annotations, write diagnostics |
| Spatial ONNX | exported spatial graph + images | Read only the graph-declared layers and contract; the graph fixes model, layers, normalization and size |
| Spatial read | `--spatial-input` | Validate and reopen stored samples; diagnostics or compatible re-evaluation, written to a separate run directory |

- **Selection.** With `--label-csv`, exactly its rows in CSV order select images.
  Without it, every supported image under the root is scanned recursively, so
  root-level and nested images are both included. COCO sources attach annotations
  and never expand or narrow the set. This is a spatial-only deviation from the
  legacy CSV > COCO > directory selector, which embedding extraction keeps.
- **No inferred groups.** A `sample` value is kept only when the CSV supplies it
  explicitly. Directory-derived `sample` values are suppressed from the index and
  reports, so no specimen grouping is invented.
- **Storage.** `--spatial-layout per-image` is the default; `--spatial-layout sharded`
  writes NPY shards sealed below a fixed uncompressed payload budget. Layouts never
  change a logical array, its order or any diagnostic row.
- **Support records.** Every stored view carries `pixel_geometric_coverage` /
  `pixel_sampling_valid` at the model-input canvas and `geometric_coverage` /
  `patch_sampling_valid` on the patch grid. Coverage is a geometric fraction;
  validity is conservative resampling support. Neither is a foreground or
  anatomical mask.
- **Spatial ONNX.** `--onnx-path` with a spatial graph reads only the outputs and
  contract the graph declares: layers, feature norm, L2 and input size are fixed at
  export, so an explicit conflicting value fails instead of being re-derived, and
  `--model-name`, `--use-student` and `--dump-model-structure` are rejected because a
  graph cannot be relabelled, re-weighted or listed. Geometry, coverage and
  diagnostics are produced by the same shared preprocessor, so a graph and a
  checkpoint yield the same logical product for the same images.
- **Read mode is explicit.** Nothing is discovered, resumed, repaired or inferred.
  A read run never mutates the source product, needs no model for numeric
  diagnostics, and rejects extraction overrides (checkpoint, layers, norm, layout,
  size, transform, COCO sources, model name, student weights) before writing.
- **Read results are reports, not products.** A read-result directory holds
  diagnostics for the selected samples; it is not a reusable spatial product. If
  `--spatial-input` rejects it as incomplete, point `--spatial-input` at the original
  completed extraction directory (the one holding `spatial_manifest.json` and
  `spatial_index.jsonl`), as identified by the report's `source_product` provenance.
- **Cost (arithmetic estimate, not a measurement).** Evaluation stores one reference
  plus eleven views per sampled image, so a 14x14 grid with C=192 costs about
  172 MiB per layer for 100 images, and the stored pixel support adds about
  287 MiB at 224x224 views across layers. Sharding changes only the physical
  packaging: it caps each shard below 256 MiB of uncompressed payload and never
  changes a logical array, its order or a diagnostic row.
- **L2 is a representation choice.** `--feature-l2-normalize` records what the grid
  contains; the cosine matcher normalizes after feature sampling anyway, so no v1
  score requires a unit-norm export, and raw versus unit exports are distinct
  contracts rather than interchangeable ones.
- **Limitation.** Fully valid patches exclude direct fill contamination, but ViT
  global attention can still propagate influence from invalid tokens.

- **Evaluation (`--spatial-eval`).** With evaluation enabled, the selected images
  (default at most 100, chosen by seeded SHA-256 ranking over the existing `--seed`)
  also store one reference plus eleven geometric views, and the run writes
  `spatial_evaluation_plan.json`, `spatial_matches.csv` and `spatial_evaluation.json`.
  Queries are conservatively valid source patch centres whose mapped point and target
  interpolation support are valid; at most 64 per pair and direction are used, chosen
  by the same seeded ranking, and candidates are **all** valid target centres — no
  label, mask or true-location filter. Scores are cosine matching with PCK at 0.01,
  0.02 and 0.05 of the original-image diagonal, aggregated per image first and then
  macro-averaged per transform, layer and direction with explicit denominators.
  Failed matches (invalid feature, no usable candidate) stay in the PCK denominator
  and are reported with no finite error. In read mode the stored plan is reused: a
  CSV may only restrict it, `--spatial-eval-sample-size` is rejected, and an explicit
  `--seed` must equal the stored plan seed. A product without paired views fails with
  `missing_paired_views` instead of silently skipping the request.

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
