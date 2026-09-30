# otuformer training augmentation

[English](training-augmentation.md) | [中文](training-augmentation.cn.md)

## Purpose

`pretrain` and `finetune` share one data-augmentation contract, expressed as an
`--augmentation` profile plus an `--orientation-policy`. This document defines
what each profile and each policy does, and how the values are inherited.

The flags themselves — their allowed values and their defaults — are defined
once per command, in the `Parameters` table of each affected command document.
This document links to them rather than restating them (Section 4.1 of the
documentation conventions).

## Usage

```bash
# pretrain: whole-specimen barcode profile, direction-sensitive markers
otuformer pretrain --input-images-dir ./images \
    --augmentation global-barcode --orientation-policy sensitive

# finetune: opt-in conservative profile
otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv --input-images-dir ./images \
    --augmentation conservative --orientation-policy sensitive
```

## Affected Commands

| Command | Flags defined there | Document |
|---|---|---|
| `otuformer pretrain` | `--augmentation`, `--orientation-policy` | [pretrain](pretrain.md) |
| `otuformer finetune` | `--augmentation`, `--orientation-policy` | [finetune](finetune.md) |

Orientation policy is a no-op for `finetune --augmentation none`.

## Profiles

| Profile | Stage | Summary |
|---|---|---|
| `global-barcode` | pretrain | Whole-specimen barcode profile; orientation handling follows the selected policy. Modest photometric jitter; color is preserved (no grayscale). |
| `color-robust` | pretrain | Same geometry and blur as `global-barcode` with stronger color jitter and grayscale. **Warning:** it may reduce sensitivity to diagnostic body color, color patterns, or metallic sheen. |
| `legacy` | pretrain | Reproduces OTU-Former 0.2.1 augmentation exactly for old-run continuation and comparison, not for new runs. It records either orientation policy without changing its historical transforms. |
| `none` | finetune | Unchanged deterministic `Resize -> CenterCrop -> ToTensor -> Normalize`. |
| `conservative` | finetune | Experimental opt-in profile, not proven superior to `none`; evaluate it on held-out individuals and held-out species. It adds no crop, grayscale, blur, or solarization. |

## Orientation Policies

`sensitive` is an explicit caller choice (never inferred) for direction-sensitive
markers: every global/local pretraining view and fine-tuning `conservative` uses
rotation `[-15°, 15°]` and no horizontal reflection. It was chosen as the default
because the 50-epoch implementation comparison, which used the then-default
`invariant` and did not evaluate `sensitive`, found no rotation-consistency gain
from broad `invariant` rotation, and ±180° rotation is not a plausible routine
augmentation.

`invariant` is the opt-in broad-rotation/reflection policy for
orientation-insensitive markers and preserves the existing behavior. `legacy`
accepts either policy but keeps its historical flips.

## Resume and Inheritance

Omitted `--augmentation` and `--orientation-policy` inherit on resume; explicit
conflicts fail, and changing a profile's expanded parameters requires a new run.
Old pretrain checkpoints map to `legacy`/`invariant`; old finetune checkpoints map
to `none`/`invariant`.

Starting a new fine-tuning run from `--checkpoint` is initialization, not resume:
it never inherits the pretraining augmentation profile, and an omitted policy
inherits the pretraining checkpoint policy (fallback `sensitive` for old
checkpoints).

## Notes

**Biological contract.** Dorsal, ventral, lateral, whole-body, and anatomical-part
images are distinct markers and must not be mixed as interchangeable views of one
marker; arbitrary in-plane orientation is supported. The complete-marker
requirement applies to the source image, not to every stochastic SSL crop. Global
and local crops are partial SSL observations of one complete source marker, so
augmentation encourages but does not guarantee embedding invariance.

**Input size.** Both fine-tuning profiles use the checkpoint-recorded training
input size, with a `224` fallback for old checkpoints that do not record one.

Checkpoint metadata (`config.augmentation_profile`, `config.augmentation_config`)
provides configuration traceability, not bitwise deterministic replay. Full
transform-level definitions are in
[the training augmentation design](../superpowers/specs/2026-09-07-otuformer-training-augmentation-design.md).

## Version Notes

- **0.2.1** — the `legacy` profile reproduces OTU-Former 0.2.1 augmentation
  exactly, for old-run continuation and comparison.
