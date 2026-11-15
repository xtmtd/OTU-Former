# otuformer extract

[English](extract.md) | [中文](extract.cn.md)

## Purpose

Extract image-level embedding vectors or spatial patch grids from a checkpoint/ONNX graph, or re-read a completed spatial product without inference. Embedding output writes `embeddings.csv` for `cluster`, `diversity`, and `annotate`; spatial output writes geometry, support and diagnostic records. Labels enable embedding-quality metrics; `--spatial-eval` enables same-image geometric correspondence. These measure different abilities.

## Usage

```bash
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --out-dir runs/extract
```

| Mode | Minimum inputs | Output |
| --- | --- | --- |
| Embedding (default) | images + checkpoint or embedding ONNX | vectors, optional labelled metrics/UMAP |
| Spatial image extraction | `--output-kind spatial`, images + checkpoint or spatial ONNX | grids, diagnostics, optional frozen correspondence |
| Spatial read | `--output-kind spatial --spatial-input PRODUCT` | reports in a separate directory; no model needed |

```bash
otuformer extract --output-kind spatial --checkpoint model.pt --input-images-dir ./images --spatial-eval --out-dir runs/spatial
otuformer extract --output-kind spatial --spatial-input runs/spatial --spatial-eval --out-dir runs/spatial_report
otuformer extract --checkpoint model.pt --input-images-dir ./images --feature-layers blocks.5 --dump-model-structure --out-dir runs/layer5
```

No input option is required in every mode, so help has no universal required `*`; the mode-specific requirements above are enforced before output cleanup.

## Parameters

### `--checkpoint`

**Default:** `None`

Pretrain/finetune checkpoint. Image extraction requires a checkpoint or ONNX; spatial read needs neither.

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

**Default:** `None` (spatial: last complete block; embedding: existing output path)

Complete backbone blocks. Spatial: ordered comma-separated paths, omitted means last block. Checkpoint embedding: exactly one block, final backbone norm then CLS; no projector/pooling. Omitted keeps legacy embedding behavior. No multi-layer embedding fusion. Use `blocks.5,blocks.11` for spatial or one `blocks.<index>` for embedding. Embedding requires `--token-mode cls`, rejects ONNX/patch-topk/attention-pool, and reads token zero before removing any prefixes. `--feature-norm`/`--feature-l2-normalize` remain spatial-only; selection adds no embedding L2.

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

Spatial only: evaluate frozen same-image cross-view correspondence (spatial-geometry-v1). Image mode saves paired views; read mode reuses the saved plan without model inference. A product without paired views cannot be evaluated by inventing missing views.

### `--spatial-eval-sample-size`

**Default:** `100`

Spatial evaluation only. Maximum number of deterministically selected images for correspondence. Rejected in read mode; use --label-csv there.

### `--dump-model-structure`

**Default:** `No`

Checkpoint embedding/spatial: write module paths and block eligibility to model_layers.txt, then continue extraction. Rejected for ONNX/read. Spatial and embedding-CLS eligibility are listed separately. This is not a model-only command: images remain required. The selected embedding layer is recorded in effective provenance for CSV/COCO runs.

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

Exported ONNX graph. Embedding: CLS graph output, legacy checkpoint override. Spatial: fixed multi-layer contract; cannot combine with checkpoint/model-name/use-student/dump overrides. Not used in spatial read. Explicit embedding layer selection/structure export are rejected; spatial contract overrides must exactly match the graph.

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

# Using an exported embedding ONNX graph
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

<a id="understanding-the-output-metrics"></a>
## Notes

### Understanding the output metrics

Embedding and spatial evaluation answer different questions. Neither a high score nor a healthy tensor proves anatomical homology or independent-test generalization.

### Embedding: `metrics.csv`

These are the same metric implementations used by `pretrain` (`logs/metrics.pretrain.csv`) and `finetune` (`logs/metrics.finetune.csv`). Extract uses the selected, optionally subsampled labelled images. Training CSVs also carry epoch/split columns: always compare the same split and evaluation population. `metrics.csv` is written as `metric,value`; a value of `0.8` means 80% for rate metrics.

| Field | Meaning and use | Interpretation |
| --- | --- | --- |
| `Recall@1`, `Recall@5`, `Recall@10` | Fraction of queries with at least one same-label image among the top-k cosine neighbours; the query itself is excluded. Measures retrieval success. | Higher is better, 0–1; singleton-label queries stay in the denominator and cannot succeed. |
| `mAP` | Mean average precision of the full non-self cosine ranking; rewards placing all same-label images early. | Higher is better, 0–1; queries without any other same-label item are excluded. Different denominator from Recall. |
| `kNN_Acc_k1/k5/k20` | Cosine k-nearest-neighbour classification accuracy under stratified cross-validation. | Higher is better, 0–1; tests label neighbourhoods, not an independently held-out backbone. |
| `Linear_Probing_Acc` | Cross-validated logistic-regression accuracy on frozen vectors. | Higher is better, 0–1; tests linear label separation. |
| `Linear_Probing_Balanced_Acc` | Mean per-class recall from the same linear-probe CV pass. | Higher is better, 0–1; more informative with unequal class counts. |
| `NMI` | Normalized mutual information between labels and KMeans assignments. | Higher is better, 0–1. |
| `ARI` / `AMI` | Chance-adjusted pair agreement / mutual information between labels and KMeans assignments. | Higher is better; 1 is perfect, about 0 is chance, negative values are possible. |
| `Purity` | Fraction correct after assigning each KMeans cluster its majority label. | Higher is better, 0–1; depends on the chosen cluster count. |
| `Silhouette_Score` | Cosine cohesion/separation using the true labels, not predicted KMeans labels. | −1 to 1; larger means better separation in this representation, not a universal pass mark. |

KMeans uses L2-normalized vectors and as many clusters as observed label classes. CV uses shuffled stratified folds, seed 42, with `min(5, smallest class count)` folds. Invalid k, singleton classes or degenerate vectors can make a metric unavailable: empty CSV fields are not zero; shared-path provenance distinguishes unavailable from failed. `--metrics-sample-size` changes the scored population; `--disable-umap` disables the plot, not metrics. UMAP is a visualization, not an accuracy score. See [the shared embedding-metric contract](embedding-metrics.md) for availability and version rules. Image-level CV does not guarantee specimen/group-safe holdout; train-split results do not establish test generalization.

### Spatial diagnostics: `spatial_results.csv`

One row is an image × view × layer, including base and stored evaluation views. `spatial_diagnostics.json` holds summaries. These inspect values/support, not matching accuracy.

| Fields | Meaning and use |
| --- | --- |
| `sample_id`, `image_key`, `view_id`, `role`, `layer` | Stable sample identity, view and feature source; group comparisons by these keys. |
| `shape`, `grid_hw`, `dtype` | H×W×C feature shape, H×W patch lattice and stored numeric type. |
| `finite_count`, `nonfinite_count` | Counts of token vectors, not channel scalars. Nonfinite features fail diagnostics rather than becoming a quality score. |
| `zero_count`, `near_zero_norm_count` | Token norms equal to zero / at most 1e−12; flag unusable or nearly unusable vectors. |
| `norm_min`, `norm_mean`, `norm_max` | Token L2 norm statistics. Unit norms are expected when export L2 is enabled and do not prove discriminative features. |
| `channel_variance_mean` | Mean within-token variance over channels; not variance across positions/images and not a collapse test. |
| `coverage_min`, `coverage_mean`, `coverage_full_valid_fraction` | Pixel-level geometric coverage and fraction fully covered. Not foreground occupancy. |
| `pixel_valid_fraction`, `patch_valid_fraction` | Conservative sampling-valid pixels / fully valid patch footprints. Filling/interpolation can reduce support. |
| `support_version` | Version of coverage/validity rules; compare compatible contracts. |

### Spatial correspondence: `spatial_matches.csv` and `spatial_evaluation.json`

`spatial_matches.csv` is one query × direction × transform × layer. `image_key`, `pair_id`, `direction`, `layer`, `query_index`, `row`, `column` identify the query. `candidate_count` counts geometrically valid target centres; `invalid_candidate_count` counts unusable candidate features. `match_status=matched` means a usable candidate was found, not that it was correct. `predicted_row/column` locate that candidate; `true_x/y` and `predicted_x/y` are in original-image coordinates.

- `raw_error`: Euclidean distance in original-image pixels; lower is better.
- `normalized_error`: `raw_error / hypot(original_width, original_height)`; lower is better. The denominator is the image diagonal, not the specimen length or bounding box.
- `quantization_error`: minimum normalized error achievable by any valid target grid centre. It is a geometry-only lower bound, not model error; a coarse grid can make strict thresholds impossible.
- `pck_0p01`, `pck_0p02`, `pck_0p05`: **per-query integer bits**, 1 when normalized error ≤ 1%/2%/5%, otherwise 0. Failed queries remain zero in the geometric denominator. Example: error `0.0246` yields `0,0,1`; averaging bits yields a rate, not another integer.

The JSON publishes per-image rates first, then image macro averages per direction and equal-weight bidirectional `combined` rates when both directions exist. Thus `0.64` means 64%; a global CSV-row mean can differ from the published macro average. Families/layers stay separate: there is no pooled overall score. An unavailable no-overlap branch is not zero accuracy. Examine eligibility counts and quantization alongside PCK and error tails; no universal PCK threshold is good for every dataset. Compare identical images, query plans, feature contracts and baselines. Same-image geometric correspondence is not cross-specimen anatomical correspondence, classification or retrieval performance.

Reports include geometric eligibility and overlay status. Optional `spatial_overlays/` figures use bounded flat SHA-256 key names; the figure title/report identifies the image. The reader verifies a payload once per size/mtime/inode identity while per-array digests remain checked; this assumes a read-only source directory, not hostile concurrent rewriting. Candidate geometry is reused within each direction without changing the frozen protocol.

### Additional spatial contracts

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
