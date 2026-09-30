# otuformer finetune

[English](finetune.md) | [中文](finetune.cn.md)

## Purpose

Supervised metric-learning fine-tuning of a pretrained backbone with labelled data. `--loss` selects the objective (`arcface`, `supcon`, `subcenter-arcface`, or `subcenter-arcface-compact`) and produces embeddings that separate known species for OTU clustering. A pretrain checkpoint and a labelled CSV are required; an optional extra round can add pseudo-labels for unlabelled images of known species.

## Usage

```bash
otuformer finetune \
    --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --out-dir runs/finetune
```

## Parameters

### `--checkpoint`

**Default:** `None`

Initialization checkpoint used when --resume is not set. An SSL pretrain checkpoint installs a fresh embedding head; a fine-tune checkpoint with a matching head keeps its trained projector. Pretrained checkpoint path (typically `runs/pretrain/SSL_latest.pth`); optional in `--pseudo-label-from` mode (where it only relocates the recorded SSL file) and on `--resume`

### `--train-data`

**Default:** `Required`

CSV with 'image' and 'label' columns.

### `--input-images-dir`

**Default:** `Required`

Root image directory.

### `--out-dir`

**Default:** `runs/finetune`

Output directory.

### `--model-name`

**Default:** `vit_tiny_patch16_224`

timm backbone name for metric-learning encoder.

### `--metric-embed-dim`

**Default:** `None`

Fine-tune embedding dimension (the metric-embedding head output, not the raw CLS dimension). Default: the checkpoint's recorded metric dimension, else the pretrained projector dimension. Fine-tune embedding dimension (metric-embedding head output, not raw CLS). An explicit value resizes the head; it is rejected for a historical `ProjectionHead` checkpoint, whose width is fixed by the pretrained projector

### `--finetune-epochs`

**Default:** `20`

Total fine-tuning epochs.

### `--finetune-lr`

**Default:** `0.0001`

Learning rate for the fine-tuned backbone.

### `--metric-head-lr`

**Default:** `None`

Learning rate for the embedding head and, for prototype losses, the classifier; defaults to --finetune-lr.

### `--weight-decay`

**Default:** `0.0001`

AdamW weight decay for fine-tuning; pass 0.05 for the legacy script's setting. Fine-tune AdamW weight decay; default `1e-4` is a conservative supervised choice, while `0.05` reproduces the legacy script's setting

### `--freeze-ratio`

**Default:** `0.7`

Fraction of backbone blocks to freeze (0.0=none, 1.0=all). Must match the saved value on --resume.

### `--loss`

**Default:** `arcface` · **Allowed:** arcface / supcon / subcenter-arcface / subcenter-arcface-compact

Metric-learning loss: arcface (default), supcon, subcenter-arcface, or subcenter-arcface-compact. Rejected before any output when the name is unknown. On --resume an omitted --loss inherits the recorded mode; an explicit conflicting value fails. Metric-learning loss: `arcface` (default), `supcon`, `subcenter-arcface`, or `subcenter-arcface-compact`; see "Metric-loss modes (v0.8.0)"

### `--subcenters`

**Default:** `2`

Centers per class (K) for --loss subcenter-arcface or subcenter-arcface-compact: an integer from 2 to 8. arcface and supcon reject this flag.

### `--compact-weight`

**Default:** `0.1`

Same-class center-distance hinge weight for --loss subcenter-arcface-compact (cap is fixed at 0.5). Rejected for other loss modes.

### `--supcon-temperature`

**Default:** `0.07`

Temperature for --loss supcon; must be > 0. Rejected for other loss modes.

### `--long-tail`

**Default:** `none` · **Allowed:** none / cb-drw

Long-tail strategy: none (default) or cb-drw (Class-Balanced Deferred Reweighting over the ArcFace-family cross-entropy term). Rejected for --loss supcon, and must match between pseudo rounds. Long-tail strategy: `none` or `cb-drw` (ArcFace-family only; fixed beta 0.99, cap 3.0, deferred 50%/10%)

### `--pseudo-label-from`

**Default:** `None`

Completed finetune#1 ArcFace-family checkpoint used to generate one automatic known-class pseudo-label round. Finetune#2 still initializes from the same original SSL checkpoint; in that mode --checkpoint only locates that SSL file if it moved. Omitted experiment options inherit finetune#1 resolved values instead of the new-run defaults; explicit conflicts fail.

### `--pseudo-similarity-floor`

**Default:** `0.75`

Uncalibrated winning top-three-mean raw-CLS cosine class score floor in [-1, 1]. This is not a probability. Higher rejects more; lower accepts more. [default: 0.75]

### `--pseudo-min-gap`

**Default:** `0.1`

Minimum winning-score minus runner-up-score gap in [0, 2]. Higher rejects more ambiguous candidates; lower accepts more. [default: 0.10]

### `--pseudo-neighbors`

**Default:** `15`

Neighbor count k for the asymmetric own-class-excluded mutual-kNN check. Smaller is usually more local and stricter; larger is usually broader and more permissive (data-dependent). [default: 15]

### `--pseudo-cap-multiplier`

**Default:** `3`

Per-class pseudo-label safety cap multiplier: a class accepts at most min(multiplier * expert_seed_count, absolute_cap) rows after the floor/gap/mutual-kNN rules. Must be >= 1. [default: 3]

### `--pseudo-absolute-cap`

**Default:** `50`

Absolute per-class pseudo-label cap, applied after the multiplier. Must be >= 1. [default: 50]

### `--trace-batch-ids`

**Default:** `No`

Opt in to logs/batch_ids.finetune.jsonl, recording the ordered image IDs of every training batch. Can be large; morphology is never read.

### `--augmentation`

**Default:** `None` · **Allowed:** none / conservative

Augmentation profile: none or conservative. Default for a new run: none. `conservative` is experimental and not proven superior to `none`. Omit to inherit the saved profile on --resume.

### `--orientation-policy`

**Default:** `None` · **Allowed:** invariant / sensitive

Orientation policy: invariant or sensitive. Default for a new run: sensitive. Affects conservative; a no-op for none. Omit on initialization to inherit the checkpoint policy. Orientation policy: `sensitive` (new-run default) or `invariant`; affects `conservative` only. Omit on `--checkpoint` initialization to inherit the pretraining checkpoint's saved policy; omit on `--resume` to inherit the saved policy

### `--batch-size`

**Default:** `32`

Batch size.

### `--num-workers`

**Default:** `4`

DataLoader workers.

### `--cpus`

**Default:** `12`

CPU threads for PyTorch/MKL.

### `--device`

**Default:** `auto` · **Allowed:** auto / cpu / cuda / mps

Device: auto / cpu / cuda / mps.

### `--seed`

**Default:** `42`

Random seed.

### `--log-every-n-steps`

**Default:** `50`

Log metrics every N iterations.

### `--save-every-epochs`

**Default:** `10`

Save checkpoint every N epochs.

### `--keep-last-checkpoints`

**Default:** `10`

Keep only last N checkpoints.

### `--visualize-data`

**Default:** `None`

CSV with an 'image' column and optional 'label' for periodic metrics + UMAP. Without labels, only UMAP is generated. If omitted, --train-data is reused.

### `--extract-size`

**Default:** `auto`

Image size for periodic metrics/UMAP embedding extraction. 'auto' uses the model's img_size. Common examples: 224, 384, 448 (e.g. 518 for patch-14 models).

### `--metrics-sample-size`

**Default:** `10000`

Max samples for periodic metrics and UMAP (<=0 means no cap).

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

### `--disable-embedding-metrics`

**Default:** `No`

Disable periodic embedding metrics + UMAP generation during fine-tuning to reduce runtime overhead.

### `--resume`

**Default:** `None`

Fine-tune checkpoint path to resume from (restores model/optimizer state; saved optimizer settings override the LR and weight-decay options).

### `--overwrite`

**Default:** `No`

Clear an existing non-empty output directory.

## Inputs

`--input-images-dir` is the image root; `--train-data` is a CSV with `image` and `label` columns whose paths are relative to that root. `--visualize-data` is an optional separate CSV for periodic metrics and UMAP.

## Outputs

- `logs/finetune.log` — Run log
- `finetune_latest.pth`, `finetune_epoch_*.pth` — Model checkpoints, written to `--out-dir` (there is no `checkpoints/` subdirectory and no `finetune_best.pth`)
- `logs/metrics.finetune.csv` — Periodic embedding metrics
- `logs/instant_metrics.finetune.csv` — Per-iteration training metrics
- `logs/loss_diagnostics.finetune.csv` — Per-epoch v0.8.0 loss diagnostics
- `logs/batch_ids.finetune.jsonl` — Training-batch ID trace (only with `--trace-batch-ids`)
- `logs/training_curves_finetune.pdf` — Training-curve plot
- `logs/umap.train.epoch_<N>.pdf` — Periodic UMAP plots (when not disabled)

Embedding-quality metrics use the shared [embedding metrics](embedding-metrics.md) definitions and field names.

## Examples

```bash
otuformer finetune --train-data labels_union.csv --input-images-dir ./images \
    --out-dir runs/finetune --resume runs/finetune/finetune_latest.pth \
    --finetune-epochs 50
```

```bash
# Basic usage
otuformer finetune \
    --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --out-dir runs/finetune

# Resume training
otuformer finetune \
    --resume runs/finetune/finetune_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --out-dir runs/finetune

# Custom parameters
otuformer finetune \
    --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --model-name vit_small_patch16_224 \
    --augmentation conservative \
    --finetune-epochs 50 \
    --finetune-lr 3e-4 \
    --freeze-ratio 0.5 \
    --loss arcface \
    --out-dir runs/finetune
```

```bash
# finetune#1 records its expert-manifest and SSL identity for later reuse
otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv --input-images-dir DATA_ROOT/dorsal \
    --out-dir RUN_ROOT/finetune1 --finetune-epochs 20

# finetune#2: one automatic pseudo round from the same original SSL checkpoint
otuformer finetune --train-data labels.csv --input-images-dir DATA_ROOT/dorsal \
    --pseudo-label-from RUN_ROOT/finetune1/finetune_latest.pth \
    --out-dir RUN_ROOT/finetune2 --finetune-epochs 20
```

## Notes

A separate image-only `--visualize-data` CSV may be used for UMAP; supervised metrics require at least two label classes.

To add images to known classes, train with a CSV containing the union of old and
new labeled images, keep the same `--out-dir`, resume from the latest fine-tune
checkpoint, and increase `--finetune-epochs`:

The resumed CSV must retain exactly the same label set. New classes require a
new fine-tune initialized with `--checkpoint` rather than `--resume`.

**Sparse-label pseudo-label feedback.** One optional round recovers unlabeled
images of known species without a candidate-list input: candidates are discovered
automatically as supported images under `--input-images-dir` minus the expert CSV
references. Use a dedicated data root separate from the run tree
(`DATA_ROOT/dorsal/` vs `RUN_ROOT/finetune1/`); `--out-dir` must not overlap the
image root, and every file-valued input must stay outside `--out-dir`.

- Pseudo mode is ArcFace-family only and one round only; a finetune#2 checkpoint
  cannot seed another round.
- Omitted experiment options inherit finetune#1 resolved values; explicit
  conflicts fail. Finetune#2 always initializes from the same original SSL
  checkpoint; `--checkpoint` in pseudo mode only locates that SSL file after it
  moved (its file SHA-256 must still match).
- Preprocessing is fixed to `center-crop`. Outputs are `pseudo_labels.csv` (one
  diagnostic row per candidate) and `pseudo_summary.json`.
- `--pseudo-similarity-floor` (default 0.75) is an uncalibrated raw-CLS
  top-three-mean cosine class score, not a probability. Higher floor and gap
  reject more; smaller `--pseudo-neighbors` (default 15) is usually more local
  and stricter. Each class additionally accepts at most
  `min(--pseudo-cap-multiplier * expert_seed_count, --pseudo-absolute-cap)`
  rows (defaults 3 and 50) after the three feature rules.
- The diagnostic CSV is not an approval gate; an empty accepted set fails before
  creating the output directory.
- `--long-tail cb-drw` is an independent conventional long-tail option for
  ArcFace-family losses (fixed beta 0.99, cap 3.0, deferred 50%/50%..10%); it must
  match between pseudo rounds.
- `--input-images-dir` must be a real directory tree (directory symlinks are
  not traversed and are reported); explicit `--visualize-data` still works for
  arbitrary UMAP/metric diagnostics.
- Evaluate RUN1 vs RUN2 on held-out images under a separate `HELDOUT_ROOT`:
  `otuformer extract --input-images-dir HELDOUT_ROOT --label-csv HELDOUT.csv`.
  Held-out images left inside the training root would become pseudo candidates.

**Checkpoint handling**:

- `--checkpoint` starts a new run. An SSL pretrain checkpoint installs a fresh embedding head; a fine-tune checkpoint whose head and width match keeps its trained projector. A historical `ProjectionHead` checkpoint keeps its projector, which fixes the embedding width.
- `--resume` restores the saved optimizer state, so `--finetune-lr`, `--metric-head-lr`, and `--weight-decay` have no effect. `--freeze-ratio` must match the saved value.
- Ref-script checkpoints (`ref/ibot20260115.py`) can be read by `extract`, `export`, and `cam`, but cannot be resumed by `finetune`.

**Metric-loss modes (v0.8.0).** `--loss` accepts four explicit supervised
targets, and rejects an unknown name or a setting that does not apply to the
resolved mode before creating any output. ArcFace stays the default and the
v0.8.0 reference loss.

| Mode | Objective |
|------|-----------|
| `arcface` | One angular-margin classifier center per labelled species (default) |
| `supcon` | Single-view supervised contrastive loss over batch-wise image pairs |
| `subcenter-arcface` | K centers per species; a sample approaches its closest center |
| `subcenter-arcface-compact` | Sub-center ArcFace plus a bounded same-class center-distance penalty |

- Classifier centers are **training-only** state: `extract`, `cluster`, and the
  default downstream workflow never read them, and morphology metadata never
  enters training.
- **Raw CLS stays the comparison target.** The loss acts on the fine-tune head
  while the default `extract` vector is the backbone's raw CLS embedding, so a
  lower training loss does not by itself prove better distances. Hold
  `--freeze-ratio`, the backbone and metric-head learning rates, the epoch
  budget, augmentation, the training manifest, batch size, and seed fixed
  across modes. An **open-set** comparison calibrates each loss on held-out
  known species and applies it without retuning to unseen species.
- First-round exploratory settings are SupCon temperature `0.07`, K `2`,
  cosine-distance cap `0.5`, and compact weight `0.1`. They are starting values,
  not validated optima; there is no first-round parameter grid or benchmark
  runner.
- `supcon` skips a batch with no valid positive anchor or no different-species
  pair, reports the skipped batches, and fails the run when a whole epoch is
  unusable instead of saving a checkpoint trained on an empty signal.
- New v0.8.0 checkpoints record the loss name and its effective settings, the
  seed, the source-checkpoint SHA-256 values, `train_manifest_sha256`, the
  optimizer layout, and the prototype weight-decay rule. New v0.8.0 ArcFace
  runs set zero weight decay on the L2-normalized prototypes where v0.7.x
  applied the requested decay (default `1e-4`). Because the forward pass
  normalizes the prototypes, that decay only rescaled their norm, which the
  loss never observes; at the default `--finetune-lr 1e-4` and
  `--weight-decay 1e-4` it is below float32 resolution, so results are
  numerically identical to v0.7.x. A larger learning-rate x weight-decay
  product does change the raw prototype norm, so cross-version checkpoints are
  not guaranteed to be loss-only controlled comparisons. A legacy `--resume`
  keeps its saved optimizer semantics, and resuming an SSL or classifier-free
  initialization checkpoint is rejected.
- `logs/loss_diagnostics.finetune.csv` holds one row per completed epoch
  (usable and skipped batches, valid anchors, margin-satisfied fraction, compact
  hinge activation and penalty, per-class/per-center assignment counts, and
  center-direction cosine). `logs/batch_ids.finetune.jsonl` is written only
  with `--trace-batch-ids`. The existing `logs/metrics.finetune.csv` and
  `logs/instant_metrics.finetune.csv` schemas are unchanged.

Augmentation and orientation policy follow the shared [training augmentation contract](training-augmentation.md).

Embedding-quality metrics use the shared [embedding metrics](embedding-metrics.md) definitions and field names.

## Version Notes

- **v0.9.0** — `--pseudo-label-from` adds one automatic known-class pseudo-label round; `--long-tail` selects the long-tail strategy.
- **v0.8.0** — `--loss` selects `arcface` (default), `supcon`, `subcenter-arcface`, or `subcenter-arcface-compact`, with per-mode parameters and loss diagnostics.
