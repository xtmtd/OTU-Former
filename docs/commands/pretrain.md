# otuformer pretrain

[English](pretrain.md) | [中文](pretrain.cn.md)

## Purpose

Self-supervised contrastive pre-training using a DINO/iBOT-style teacher-student ViT. It trains a student-teacher backbone on global and local crops with a patch-level objective, and requires no labels. The resulting checkpoints initialize `finetune`, or can be passed to `extract` directly.

## Usage

```bash
otuformer pretrain \
    --input-images-dir ./images \
    --out-dir runs/pretrain
```

## Parameters

### `--train-data`

**Default:** `None`

Optional CSV with an 'image' column (paths relative to --input-images-dir). If omitted, all images under --input-images-dir are used recursively.

### `--input-images-dir`

**Default:** `Required`

Root image directory.

### `--out-dir`

**Default:** `runs/pretrain`

Output directory.

### `--model-name`

**Default:** `vit_tiny_patch16_224`

timm backbone name used for student/teacher encoders.

### `--register-tokens`

**Default:** `none` · **Allowed:** none / 0 / 4

Register tokens for an eligible single-CLS timm VisionTransformer: 'none' (default), '0' or '4'. 'none' keeps the backbone's own architecture (no registers on a plain ViT, or a native-register model's own count) and is recorded as null in the saved args. An explicit '0' is not the same as 'none': it is rejected on a CLS-free backbone. Explicit values are rejected on native-register backbones.

### `--out-dim`

**Default:** `256`

SSL projector output dimension.

### `--max-epochs`

**Default:** `50`

Total SSL pretraining epochs.

### `--lr`

**Default:** `0.0005`

Base learning rate before warmup/cosine scheduling.

### `--weight-decay`

**Default:** `0.05`

AdamW weight decay coefficient.

### `--warmup-epochs`

**Default:** `3`

Warmup epochs before cosine LR decay.

### `--augmentation`

**Default:** `None` · **Allowed:** global-barcode / color-robust / legacy

Augmentation profile: global-barcode, color-robust, or legacy. Default for a new run: global-barcode. Omit to inherit the saved profile on --resume. color-robust may suppress diagnostic color, pattern, or metallic sheen; legacy reproduces 0.2.1 and keeps its historical transforms.

### `--orientation-policy`

**Default:** `None` · **Allowed:** invariant / sensitive

Orientation policy for every global and local view: invariant or sensitive. Default for a new run: sensitive. Omit to inherit the saved policy on --resume. Orientation policy for every global and local view (no transform effect for `legacy`): `sensitive` (new-run default; rotation `[-15°, 15°]`, no horizontal reflection) or `invariant` (opt-in broad rotation and reflection); omit to inherit the saved policy on `--resume`

### `--global-crop-size`

**Default:** `auto` · **Allowed:** auto, or an integer (common: 224 / 384 / 448; 518 patch-14 only)

Global crop resolution. 'auto' resolves to the backbone's native input size on a new run, or the checkpoint's recorded size on --resume. Common examples: 224, 384, 448 (e.g. 518 for patch-14 models).

### `--local-crop-size`

**Default:** `None` · **Allowed:** an integer divisible by the patch size

Local crop resolution. Omit to use 96 for a new run or inherit the checkpoint value on --resume. Must be divisible by the backbone patch size (e.g. 98 or 112 for patch-14 models).

### `--local-crops`

**Default:** `None`

Number of local crops. Omit to use 6 for a new run or inherit the checkpoint value on --resume.

### `--patch-loss`

**Default:** `consistency` · **Allowed:** none / consistency / masked-feature / ibot

Patch-level objective. 'none', 'consistency' (default; selects visible same-position patches for normalized cosine regression, i.e. masked-position consistency, never input masking), 'masked-feature' (true continuous masked feature prediction of the teacher's final-four-block patch target) or 'ibot' (experimental prototype-distribution prediction).

### `--masking-strategy`

**Default:** `random` · **Allowed:** random / blockwise / hybrid

Masking geometry for masked-feature/ibot only. 'random' samples independent patch positions; 'blockwise' merges bounded rectangular regions; 'hybrid' takes half blockwise and half random positions. Not applicable to none/consistency; the exact geometry limits are documented in this document.

### `--mask-ratio`

**Default:** `auto` · **Allowed:** auto, or a float in (0, 1)

'auto' or a float in (0, 1). Fraction of patch positions used by the selected patch objective. A new run resolves 'auto' to 0.30 (v0.6.x used a 0.50 default). Used as the visible same-position share for consistency and as the masked student-input fraction for masked-feature/ibot. `auto` or a float in (0, 1). `auto` resolves to 0.30 for a new run (v0.6.x used 0.50); a legacy resume keeps its recorded value (0.50 when absent)

### `--ibot-prototypes`

**Default:** `512`

Prototype dictionary size for the experimental ibot mode: any integer >= 2, default 512. Larger dictionaries suit larger datasets (for example 4096 or 16384); powers of two are a convenient habit, not a requirement. Not applicable to other patch modes.

### `--lambda-local`

**Default:** `1.5`

Weight for local-crop SSL loss term.

### `--lambda-mask`

**Default:** `1.0`

Weight for the patch-loss term. ibot's cross-entropy is on a different scale from the cosine patch losses, so lower it (for example 0.25-0.5) when using --patch-loss ibot.

### `--teacher-momentum`

**Default:** `0.995`

Initial EMA momentum.

### `--teacher-momentum-end`

**Default:** `0.999`

Final EMA momentum.

### `--student-temp`

**Default:** `0.1`

Student temperature.

### `--teacher-temp-start`

**Default:** `0.04`

Initial teacher temperature.

### `--teacher-temp-end`

**Default:** `0.07`

Final teacher temperature.

### `--disable-cross-view-loss`

**Default:** `No`

Disable cross-view global loss pairing. Default keeps full cross-view matching between global crops.

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

CSV with an 'image' column and optional 'label' for periodic embedding metrics/UMAP. Without labels, only UMAP is generated. If omitted, --train-data is reused when available.

### `--extract-size`

**Default:** `auto` · **Allowed:** auto, or an integer (common: 224 / 384 / 448; 518 patch-14 only)

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

Disable periodic embedding metrics + UMAP generation during pretraining to reduce runtime overhead.

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

### `--resume`

**Default:** `None`

Checkpoint path for resuming interrupted pretraining.

### `--overwrite`

**Default:** `No`

Clear an existing non-empty output directory.

## Inputs

`--input-images-dir` is the image root; optional `--train-data` is a CSV with an `image` column whose paths are relative to that root. Without `--train-data`, every image under the root is used, and images without a label get a synthetic `NA_<index>` label. `--visualize-data` is an optional separate CSV for periodic metrics and UMAP.

## Outputs

- `logs/pretrain.log` — Run log
- `SSL_latest.pth`, `SSL_epoch_*.pth` — Model checkpoints, written to `--out-dir` (there is no `checkpoints/` subdirectory and no `SSL_best.pth`)
- `logs/metrics.pretrain.csv` — Periodic embedding metrics
- `logs/instant_metrics.pretrain.csv` — Per-iteration training metrics
- `logs/training_curves_pretrain.pdf` — Training-curve plot
- `logs/umap.train.epoch_<N>.pdf` — Periodic UMAP plots (when not disabled)

Embedding-quality metrics use the shared [embedding metrics](embedding-metrics.md) definitions and field names.

## Examples

```bash
otuformer pretrain --train-data images_union.csv --input-images-dir ./images \
    --out-dir runs/pretrain --resume runs/pretrain/SSL_latest.pth --max-epochs 100
```

```bash
# Basic usage
otuformer pretrain \
    --augmentation global-barcode \
    --input-images-dir ./images \
    --out-dir runs/pretrain

# Use CSV to specify training subset
otuformer pretrain \
    --train-data images.csv \
    --input-images-dir ./images \
    --out-dir runs/pretrain

# Custom model and training parameters
otuformer pretrain \
    --input-images-dir ./images \
    --model-name vit_small_patch16_224 \
    --max-epochs 100 \
    --lr 1e-3 \
    --batch-size 64 \
    --out-dir runs/pretrain
```

## Notes

**Patch-level objectives (v0.7.0).** `--patch-loss` selects the patch objective:

| `--patch-loss` | Student input | Objective |
|---|---|---|
| `none` | unmasked | no patch objective |
| `consistency` (default) | unmasked | normalized cosine regression at selected visible same-position patches |
| `masked-feature` | masked | continuous masked feature prediction |
| `ibot` (experimental) | masked | prototype-distribution prediction |

`consistency` selects fully visible patch positions, so it is **masked-position
patch consistency**, not masked image modeling: the student still observes every
selected pixel. `masked-feature` (continuous masked feature prediction) and
`ibot` (experimental) replace the selected student patch
embeddings with a learnable mask token before positional encoding, so the
student never sees the selected content while its position is preserved. The
teacher target for masked-feature is the L2-normalized mean of the teacher's
final four block outputs. iBOT predicts the teacher's centered prototype
distribution
(`--ibot-prototypes`, default 512; any integer >= 2, with larger dictionaries
for larger datasets) with moving-average centering.
`masked-feature` and `ibot` are mutually exclusive and are never combined.

`--masking-strategy` chooses the real-masking geometry for `masked-feature` and
`ibot`; it has no effect for `none` or `consistency`:

- `random` (default) — `round(ratio * N)` unique patch positions, sampled
  independently per sample and per global view.
- `blockwise` — two to four rectangle proposals per round, merged on overlap. A
  legal rectangle has sides of at least two patches, an aspect ratio within
  `[0.5, 2.0]`, and an area of at most `floor(0.20 * N)` patches. Overlap is
  counted once, and the result is trimmed or random-filled to exactly
  `round(ratio * N)` positions. A grid where no legal rectangle fits (for
  example `4 x 4`, whose maximum block area is 3 while a `2 x 2` block needs 4)
  fails before the training loop instead of silently degrading to `random`.
- `hybrid` — exactly `floor(target / 2)` blockwise positions, then random fill
  to the target count.

Every strategy clamps the count to at least one masked and at least one visible
patch.

`--lambda-mask` (default `1.0`) weights the patch loss. iBOT's cross-entropy is
on a different scale from the cosine patch losses, so `--patch-loss ibot` often
needs a lower value (for example `0.25`-`0.5`). Loss values are not comparable
across patch modes.

**Migration notice: `--mask-ratio` now defaults to `auto` and a new run resolves
it to 0.30, where v0.6.x used a 0.50 default.** Resuming a legacy checkpoint
keeps its recorded ratio (0.50 when absent) and cannot switch to
`masked-feature` or `ibot`. For `consistency`, `--mask-ratio` is the visible
same-position share; for `masked-feature`/`ibot` it is the masked student-input
fraction. Explicitly passing an option that the selected mode does not use is
rejected rather than ignored.

**Optional register tokens (v0.10.0).** `--register-tokens` accepts `none` (the default), `0` or `4`
for an eligible single-CLS timm `VisionTransformer`. `4` adds four trainable
registers and migrates the one-CLS pretrained backbone: the register token keeps
its native initialization while every CLS/patch/block weight and position is
preserved. `none` keeps the backbone's own architecture, so a native-register
model keeps its own count; an explicit value on a native-register or CLS-free
backbone is rejected, and DINOv3/Eva backbones are not supported by this option.
Registers are never image patches: `patch-topk`, `attention-pool` extraction,
CAM reshape, and the patch objective all skip every prefix token
(`backbone.num_prefix_tokens`). New attention-pool checkpoints record their patch
policy (`num_prefix_tokens`, `register_tokens`, `prefix_excluded`,
`attention_pooling_type`) and are saved to a policy-tagged sibling, so a pool
trained on a different patch set is never silently reused. ONNX export also
compares CPU FP32 PyTorch against CPU ONNX Runtime on the same input
(`atol=1e-4`, `rtol=1e-3`) and records `validation_status`, `max_abs_diff`, and
the register count in `export_report.json`. A fine-tune run's register layout
is part of the v0.9.0 pseudo-label experiment identity, so finetune#2 must
initialize from a source with the same patch set.

Masked-feature and iBOT add one masked student forward per global view, so
student encoder work grows by roughly 1.65x at the default two-global/six-local
crop layout. Extraction, fine-tuning, CAM, and export are unchanged: the default
path is still the raw CLS token, they never load the training-only patch state,
and they never invoke the mask token, predictor, or iBOT heads. Pretraining and
pretraining resume support standard timm ViT backbones only. v0.7.0 resume is
strict: patch mode, resolved ratio, masking strategy, prototype count, and the
fixed target/center settings must match the checkpoint, and masked-feature/iBOT
checkpoints
must carry their `patch_objective` state. RNG state round-trips for v0.7.0
checkpoints; legacy checkpoints resume best-effort with a warning.

These objectives are self-supervised contextual feature prediction. They do not
by themselves establish anatomical part semantics or dense morphology
supervision, and v0.7.0 makes no dense-representation claim.

When `--visualize-data` has no `label` column, periodic supervised metrics are skipped and UMAP is still generated from the visualization embeddings.

To continue pretraining after adding images, create a CSV containing both old and
new images, keep the same `--out-dir`, pass the latest checkpoint to `--resume`,
and increase `--max-epochs`:

An extended run may use a larger union dataset. A same-plan resume requires the
original DataLoader length. Training on only newly added images is possible but
can forget previously learned data.

**Local-view resume inheritance.** On `--resume`, omitted `--local-crop-size` and `--local-crops` inherit the checkpoint's saved values; explicit conflicting values fail, and changing them requires a new run.

Augmentation and orientation policy follow the shared [training augmentation contract](training-augmentation.md).

Embedding-quality metrics use the shared [embedding metrics](embedding-metrics.md) definitions and field names.

## Version Notes

- **v0.10.0** — `--register-tokens` accepts `none`, `0`, or `4`; the resolved count is recorded in checkpoints and must match on resume and fine-tuning.
- **v0.7.0** — `--patch-loss` and `--masking-strategy` select the patch objective and its masking geometry; `--mask-ratio` defaults to `auto` and resolves to `0.30` on a new run (v0.6.x used `0.50`).
