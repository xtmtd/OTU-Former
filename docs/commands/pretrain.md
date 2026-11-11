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

SSL projector output dimension D. In relational it is the projection width; in DINO it is the bottleneck width that `--use-projector-output` extracts, never the prototype count K.

The DINO prototype layer is training-only: `--use-projector-output` returns the normalized D-dimensional bottleneck, and the K-dimensional logits are never an extraction output.

Key, target mechanism and KoLeo weight are recorded effective settings. On `--resume` they are inherited when omitted and must match when supplied explicitly; switching EMA/Sinkhorn-Knopp or the KoLeo weight mid-run is rejected, and relational resume keeps the requested dimension instead of inheriting the saved one.

### `--ssl-method`

**Default:** `relational` · **Allowed:** relational / dino

CLS objective for global-to-global and local-to-global pairs: relational (default; weak teacher targets, cross/full pairing) or dino (prototype distillation). Not a complete training recipe. Select --patch-loss independently; both share the encoder and teacher temperature (ibot). On --resume, omit to inherit; switching methods is rejected. The methods are mutually exclusive: a DINO run keeps the two global views and no relational center update, and the `--relational-teacher-views`, `--relational-center-source` and `--relational-pairing` selectors are rejected under `--ssl-method dino`.

### `--dino-prototypes`

**Default:** `64` · **Allowed:** integer >= 2

Prototype dictionary size K for --ssl-method dino: any integer >= 2, default 64. Independent of --ibot-prototypes (default 512) and never inferred from image, species or OTU counts; prototype indices are not OTUs. DINO-only: rejected under relational. Not applicable to other methods.

K much larger than 2B is valid, but assignment memory grows with K x 2B and prototype storage with K x D; no automatic memory budget exists. A prototype index is not a species, an anatomical part or an OTU. Explicit use under relational is rejected, including an explicit default value, while an omitted value has no effect there.

### `--dino-centering`

**Default:** `ema` · **Allowed:** ema / sinkhorn-knopp

DINO teacher-target mechanism: 'ema' (default; exponential moving average center) or 'sinkhorn-knopp' (three fixed iterations of balanced assignment over both teacher global crops, no EMA CLS center). These are alternatives, never two stacked operations. DINO-only and not switchable on resume.

Sinkhorn-Knopp balances soft prototype usage, not species frequencies, and small or homogeneous batches provide limited assignment evidence: uniform identical predictions can look balanced. This option never changes the independent iBOT patch target.

### `--koleo-weight`

**Default:** `0.0` · **Allowed:** finite value >= 0

Shared relational/DINO KoLeo weight on normalized backbone CLS; finite and >= 0. Default 0 skips neighbor search; batches < 2 skip KoLeo. Experimental: also repels same-species/near-duplicate samples; repeats can dominate. On --resume, omit to inherit; changing the weight requires a new run.

The regularizer applies to both pretrain methods and is chosen independently of `--ssl-method` and `--patch-loss`; it is not a fine-tune feature. It retires the `patch_loss_fraction_of_total` column for the whole run once it is positive, because a signed contribution does not have an ordinary positive loss fraction. A checkpoint records the weight and the three epsilon constants as one four-key bundle: relational writes none of them at weight 0 (absence is its canonical "disabled" encoding, so older readers still accept the file) and all four at a positive weight, while DINO always records all four. Resuming a relational checkpoint that carries no bundle cannot enable KoLeo; start a new run.

### `--max-epochs`

**Default:** `50`

Total SSL pretraining epochs.

### `--lr`

**Default:** `0.0005`

Base learning rate before warmup/cosine scheduling.

### `--weight-decay`

**Default:** `0.05`

AdamW weight decay coefficient.

### `--no-norm-bias-weight-decay`

**Default:** `No`

Skip weight decay for biases and normalization affine parameters. Omitted keeps the historical uniform `--weight-decay` on every trainable parameter. Applies to both SSL methods and keeps one joint gradient-norm clipping. Omit to inherit the saved policy on `--resume`; a different requested `--weight-decay` is not applied, and a changed policy is rejected.

Recognized normalization modules are LayerNorm, GroupNorm, BatchNorm, InstanceNorm and RMSNorm where available. Tokens, generic scales and the DINO prototype direction keep decay; the coefficient, the LR and EMA schedules and the clipping threshold are unchanged.

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

Patch objective: none; consistency (default; visible same-position cosine regression, no input masking); masked-feature (masked input, teacher final-four-block feature targets); or ibot (experimental prototype prediction). Select --ssl-method independently; both share the encoder and teacher temperature (ibot).

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

**Default:** unset (`null`); effective new-run default `0.08`

Initial teacher temperature. Effective default for a new run: 0.08. Omit to inherit the saved value on --resume. Shared by the global and local relational losses and by ibot target sharpening.

### `--teacher-temp-end`

**Default:** unset (`null`); effective new-run default `0.11`

Final teacher temperature. Effective default for a new run: 0.11. Omit to inherit the saved value on --resume.

### `--teacher-temp-warmup-ratio`

**Default:** `0.7`

Fraction of the planned steps spent ramping the teacher temperature from start to end, in the closed range [0, 1]. 0 disables the ramp and 1 leaves no plateau. Omit to inherit the saved value on --resume.

### `--relational-teacher-views`

**Default:** unset (`null`); effective new-run default `weak`

Teacher images used for relational target features: weak or ordinary. Effective default for a new run: weak. Omit to inherit the saved value on --resume.

### `--relational-center-source`

**Default:** unset (`null`); effective new-run default: the teacher-view source

Teacher feature population that updates the running CLS center: weak or ordinary. Omit to inherit the saved value on --resume; a new run defaults to the teacher-view source, and ordinary teacher views require an ordinary center.

### `--relational-pairing`

**Default:** unset (`null`); effective new-run default `cross`

Relational global pairs: cross (A-to-B and B-to-A; new-run default) or full (all four). On --resume, omit to inherit. Replaces the removed disable-cross-view-loss switch.

### `--relational-queue-size`

**Default:** unset (`null`); effective new-run default `0` (disabled)

Optional relational FIFO capacity, a count of image records (minimum 0). Omit for a new run to keep the batch-only loss; a positive integer keeps the current batch candidates and adds that many recent teacher observations. On --resume, omit to inherit the saved capacity (a checkpoint without queue state stays disabled), and pass the recorded value to confirm it; any other value is rejected. Relational-only: rejected with --ssl-method dino. Enabled runs hash the whole selected dataset before any probe, require one CSV row per in-root image identity, and need a queue-capable version for training resume.

The capacity is fixed for the run: no `auto` mode, no dataset-derived default, no resizing, no age cutoff and no confidence weighting. The current batch, including the aligned-image candidate, always stays first, and every relation has `B + E` candidate columns instead of `B`.

The selection is identity-checked with a SHA-256 manifest digest, so renaming, reordering, adding or removing rows, or changing image bytes invalidates an enabled resume while relocating the image root does not. History is kept across epochs, and paired slot storage is about `2 * M * D * 4` bytes at occupancy `M`.

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

On Apple GPUs some antialiased resampling backward kernels are unimplemented in PyTorch. Importing the `otuformer` package sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before PyTorch initialises, which lets those kernels run on the CPU. The switch is latched when the MPS backend starts, so when it takes effect depends on the process: the package can only arm it before that point, and a host that imports `torch` first is too late for the package to help. If such a host exports the variable itself before launching, the fallback does work; the package cannot see that, so it reports the state as unknown rather than claiming protection is missing. A run on MPS warns once when the state is not `ARMED`: `DISABLED` says the variable was switched off explicitly, and `UNKNOWN` says this process cannot confirm that the switch took effect, without claiming the operator failed to set it.

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

### Instant pretrain metrics

`logs/instant_metrics.pretrain.csv` carries the constant-per-run `ssl_method`, `dino_centering` and `koleo_weight` settings plus the new CLS diagnostics, weighted component losses and KoLeo status/counts. This is the only schema that changed in this release; `metrics.pretrain.csv` (epoch metrics) and the fine-tune instant metrics keep their previous columns.

- Blank `ssl_method` on a migrated row means unknown/unrecorded, not "relational": the resumed invocation is never backfilled into historical rows.
- Under a recorded method, a blank inactive setting is N/A (for example `dino_centering` under relational), and the JSON form records `null`, never an observed zero. `koleo_weight` is shared, so every new row records it numerically, including `0`; the KoLeo loss cells stay blank while the regularizer is disabled.
- `teacher_center_norm` and `cosine_similarity` remain relational-only; DINO records `cls_center_norm` (EMA only) and shows `COS=N/A`, while the startup `[Info] Initial cosine similarity` line is still printed for DINO.
- `global_loss_weighted` intentionally equals `global_loss` because there is no global weight, while `local_loss_weighted` includes `lambda_local`.
- `cls_prototype_perplexity` describes the teacher assignment marginals only: Sinkhorn-Knopp partly imposes balance and uniform identical predictions can look balanced, so it is not proof against collapse or evidence of learned morphology.

KoLeo values are signed, so its contribution does not have an ordinary positive loss fraction; once `koleo_weight > 0` the legacy `patch_loss_fraction_of_total` column is recorded as N/A for the whole run, including batches that skip KoLeo as too small. With KoLeo disabled the historical positive-fraction meaning is preserved.

KoLeo statuses are `active`, `disabled` and `skipped` for both methods. A default relational run now writes `koleo_weight=0` with status `disabled` where older rows wrote a blank weight and `not-applicable`; that is an intentional reporting change and historical rows are never rewritten or backfilled. It changes no training numerics and adds no checkpoint keys at weight zero.

A resumed instant-metrics file can contain historical blank-identity rows next to current labeled rows; plots keep both. External merged analysis must retain those unknown groups (for example `dropna=False`) instead of inferring them, and loss magnitudes are not comparable across methods or target choices as if they measured the same objective.

Older OTU-Former versions may ignore DINO identity and objective state and resume a DINO checkpoint as relational through permissive loading. Resume DINO only with a version that explicitly supports this format; the new loader cannot protect a file opened by older software.

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

Compatibility is one-directional: this code resumes older checkpoints, but an older trainer reading a new checkpoint ignores `config.relational_target_config` and therefore does not restore the weak/cross recipe. Accepting the file is not evidence the recipe was preserved.

A same-plan resume after an interrupted extension rebuilds the teacher ramp from the rewritten plan length rather than the original one, so the step-wise schedule can differ even though endpoints and recipe are unchanged. Extending a run keeps the inherited final temperature without a new warmup.

**Relational target recipe.** A new run starts from a synchronized student/teacher pair and, by default, uses two independently sampled weak teacher globals for the relational target features, `cross`-only global pairing, and teacher temperatures `0.08` to `0.11`. Resumes restore the saved student/teacher pair without re-synchronizing and inherit the recorded recipe; a changed recipe requires a new run.

Three independent controls select the teacher images that build the targets (`--relational-teacher-views`), the teacher feature population that updates the running CLS center (`--relational-center-source`), and the global view pairs averaged in the loss (`--relational-pairing`). `--relational-teacher-views ordinary` requires `--relational-center-source ordinary`. Omitted values inherit the checkpoint recipe on `--resume`; explicit values that conflict with it fail.

The patch objective keeps the ordinary global views as inputs: at fixed teacher state, ordinary views and masks, the `--patch-loss consistency` and `--patch-loss masked-feature` targets are unchanged by the weak teacher views. That is input-path isolation, not an invariance guarantee — the extra weak crops consume from the same random stream and the shared backbone still updates, so whole training trajectories and patch losses differ. The teacher temperature is shared, so `--patch-loss ibot` target sharpening and the local relational loss follow it too. With weak views enabled and a patch objective active, a step runs four teacher forwards instead of two and one extra weak crop pair per sample, which raises training time and activation memory; `--relational-teacher-views ordinary` keeps the historical two. The `cosine_similarity` instant metric stays the ordinary student-A versus centered ordinary teacher-A cosine.

Staged evaluation: compare the historical target recipe against the combined new default first. The other four combinations are optional diagnostics, not automatic follow-up work; all six use the same synchronized initial encoder pair and therefore cannot measure the initialization change.

| Arm | Options |
| --- | --- |
| Historical target-recipe baseline | `--relational-teacher-views ordinary --relational-pairing full --teacher-temp-start 0.04 --teacher-temp-end 0.07` |
| Combined new recipe | `--relational-teacher-views weak --relational-pairing cross --teacher-temp-start 0.08 --teacher-temp-end 0.11` |
| Weak targets, ordinary center | `--relational-teacher-views weak --relational-center-source ordinary --relational-pairing full --teacher-temp-start 0.04 --teacher-temp-end 0.07` |
| Weak targets and weak center | `--relational-teacher-views weak --relational-center-source weak --relational-pairing full --teacher-temp-start 0.04 --teacher-temp-end 0.07` |
| Pairing only | `--relational-teacher-views ordinary --relational-pairing cross --teacher-temp-start 0.04 --teacher-temp-end 0.07` |
| Shared temperature only | `--relational-teacher-views ordinary --relational-pairing full --teacher-temp-start 0.08 --teacher-temp-end 0.11` |

The `0.001..100` band is acceptance policy for a fresh request, not a numerical guarantee and not a resume rule: a checkpoint may record a value outside it (for example `0.0009`), and that value is inherited, re-saved unchanged and accepted as an explicit matching request. What always applies is numerical safety — a positive finite endpoint whose float32 relation-reduction bound fits the run's effective batch. That per-run check can only be confirmed at execution.

Augmentation and orientation policy follow the shared [training augmentation contract](training-augmentation.md). The weakly augmented teacher record keeps the ordinary geometry, disables color jitter, grayscale and Gaussian blur, and records its own normalization, so the recorded transform controls pixels.

Embedding-quality metrics use the shared [embedding metrics](embedding-metrics.md) definitions and field names.

## Version Notes

- **v0.12.0** — **Breaking new-run recipe.** A fresh run now copies the complete student encoder state into the independently constructed teacher before the first forward, builds relational targets from two weak teacher global views by default, averages `cross`-only global pairs, and ramps the shared teacher temperature `0.08` to `0.11`. `--disable-cross-view-loss` is removed; use `--relational-pairing {cross,full}`. New options: `--relational-teacher-views`, `--relational-center-source`, `--teacher-temp-warmup-ratio` (closed `[0, 1]`, default `0.7`). The five relational/temperature requests default to `null` and resolve to effective new-run values, so `Parameters:` records the request while the resolved recipe is logged separately. Resumes inherit the recorded recipe and reject an explicit conflict; a legacy checkpoint keeps its historical ordinary/full recipe with its own temperatures. Checkpoints gain `config.relational_target_config`; older trainers ignore it, so compatibility is one-directional. The default costs two extra weak crops per sample and, with a patch objective, four teacher forwards per step instead of two. Resuming across SSL methods is rejected before output setup; a payload recording the same method, including a v2/DINO one, resumes normally. The historical target recipe remains expressible for comparison, but the previous independent teacher-projector initialization is no longer reachable.
- **v0.11.2** — The global/local crop-size divisibility checks, the backbone-capability check and the blockwise-grid/patch-count checks now run against the model and sizes the run will actually use, and before any output directory is cleared or any log file is created; `--overwrite` no longer destroys an existing run for those configurations. The checks listed here are the guarantee: `--extract-size` divisibility and dataset/augmentation errors can still fail after output setup. Isolating those preflight probes changed the student initialization a given `--seed` produces relative to v0.11.1: reproduce bit-for-bit only within one version, and compare arms trained under the same version.
- **v0.10.0** — `--register-tokens` accepts `none`, `0`, or `4`; the resolved count is recorded in checkpoints and must match on resume and fine-tuning.
- **v0.7.0** — `--patch-loss` and `--masking-strategy` select the patch objective and its masking geometry; `--mask-ratio` defaults to `auto` and resolves to `0.30` on a new run (v0.6.x used `0.50`).
