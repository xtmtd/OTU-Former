# OTU-Former Training Augmentation Design

Date: 2026-09-07

## 1. Goal

Improve the biological suitability, reproducibility, and user-facing clarity of image augmentation used by OTU-Former pretraining and fine-tuning.

This design is limited to a **global morphological barcode**: one embedding represents one complete specimen marker. A marker is defined by the anatomical object, observation view, preparation, and imaging protocol. Images may have arbitrary in-plane orientation, but dorsal, ventral, lateral, and local anatomical views remain different markers and must not be mixed as interchangeable views of one barcode.

The design provides a small set of named augmentation profiles rather than exposing every transform strength as a CLI parameter. It preserves the current multi-crop structure and training algorithm while allowing future dense-morphology work to add a profile without changing the current `--augmentation` interface. It does not claim to implement dense morphology. The recently completed resolution-consistency work is treated as an existing prerequisite: training size propagation, named deterministic evaluation transforms, `img_size` handling, and `--eval-transform` are not re-designed here.

### Current code baseline

This review targets the code at `b4a4b3c` and its relevant ancestors `a583c74` and `7a7604b`. That baseline already includes checkpoint training-size resolution and propagation, `OTUFormerEncoder(img_size=...)`, `auto` size options, the named `center-crop` and `whole-specimen-pad` evaluation protocols, extract/CAM/ONNX size handling, non-square CAM mapping, and legacy SSL/ArcFace checkpoint compatibility. This design does not reimplement those changes. If those interfaces change later, update the resolution-consistency documentation first and re-audit this design.

## 2. Scope

### 2.1 In scope

- Pretraining augmentation profiles:
  - `global-barcode` (new default)
  - `color-robust`
  - `legacy`
- A high-level training orientation policy:
  - `sensitive` (default; small rotation only, no horizontal reflection)
  - `invariant` (opt-in; existing broad rotation and horizontal reflection)
- Fine-tuning augmentation profiles:
  - `none` (default and current behavior)
  - `conservative` (experimental)
- Background-aware fill for rotations and affine transforms.
- CLI validation and detailed help for the profiles.
- Logging and checkpoint metadata for reproducibility.
- Resume compatibility for checkpoints created before augmentation metadata existed.
- Automated tests, smoke training, and one complete 50-epoch `global-barcode` pretraining validation run, using the current resolution and evaluation interfaces as fixed inputs.
- Documentation of biological assumptions and recommended closed-set/open-set evaluation.

### 2.2 Out of scope

- Changes to student-teacher loss, global-view pairing, masked-token loss, or other training objectives.
- Changes to the number of global views or the default six local crops.
- Changes to extraction, CAM, ONNX export, periodic evaluation, or deterministic evaluation preprocessing. These paths already follow the completed resolution-consistency implementation; this design only requires augmentation changes not to alter them.
- Test-time augmentation or embedding averaging across rotations.
- A new augmentation-evaluation CLI command.
- User overrides for individual rotation, color, grayscale, blur, or crop-scale strengths.
- A marker-specific metadata system or automatic inference of orientation sensitivity. The caller must select the high-level orientation policy for the run; when fine-tuning is initialized from a pretraining checkpoint and the option is omitted, the saved policy is reused.
- Mask-based segmentation, foreground detection, or a learned background model.
- Dense heads, coordinate correspondence, dense losses, dense outputs, or a placeholder `dense-morphology` profile.
- A claim that any profile is universally optimal across insect taxa or imaging collections.

## 3. Biological and Algorithmic Principles

### 3.1 Preserve marker identity

Augmentation may model image acquisition variation only when that variation does not change the marker's biological meaning. The following are valid variations for the `global-barcode` contract under the opt-in `invariant` policy, when the marker is treated as orientation-insensitive:

- arbitrary in-plane specimen orientation;
- modest differences in position and scale;
- reflection for a whole-specimen barcode where incidental left/right differences should not dominate;
- lighting, exposure, contrast, and small white-balance differences;
- moderate focus variation during self-supervised pretraining.

The following are not interchangeable views of one marker:

- dorsal versus ventral view;
- dorsal versus lateral view;
- whole specimen versus an isolated anatomical structure;
- images that omit substantial parts of the specimen because the source specimen or source image is incomplete.

Input data are expected to contain one complete specimen, a consistent observation view, a simple background, and enough margin to avoid accidental loss of the specimen. OTU-Former does not repair incomplete source markers.

The complete-marker contract applies to the source image and to the biological interpretation of the final barcode. It does not require every stochastic SSL training view to contain the whole specimen. Global and local crops are partial observations sampled from that complete source marker; they must not be mistaken for separate marker identities.

### 3.2 Preserve the existing multi-crop algorithm

Pretraining continues to produce two independently sampled global views and `--local-crops` local views (default: six). The existing crop sizes, crop area ranges, and torchvision default aspect-ratio range remain unchanged:

- global crop scale: `(0.4, 1.0)`;
- local crop scale: `(0.05, 0.4)`;
- `RandomResizedCrop` ratio: torchvision default `(3/4, 4/3)`;
- global crop size: controlled by `--global-crop-size`;
- local crop size: controlled by `--local-crop-size`;
- local crop count: controlled by `--local-crops`.

Local crops remain auxiliary SSL views. They are not declared to be independent anatomical markers.

This is an explicit design trade-off: preserving the established multi-crop algorithm and its `0.4` global scale floor takes precedence over requiring every global training view to retain the complete specimen. Raising the floor could better preserve whole-body context but would change the reference training distribution without supporting ablation evidence. The source image must still contain a complete marker, and the resulting embedding remains a global barcode.

### 3.3 Augmentation encourages but does not guarantee invariance

Random rotation and reflection encourage similar representations across those transformations; they do not mathematically guarantee identical embeddings. Validation must therefore measure transformed-image consistency and ensure that improved consistency is not caused by representational collapse.

Color augmentation is also a compromise. Color can contain both taxonomic information and collection artifacts caused by lighting, equipment, preservation, and aging. The default profile retains color information while adding modest photometric robustness. A separate profile permits stronger color variation when users determine that color should carry less weight.

## 4. Architecture

Augmentation construction remains in `src/otuformer/training/dataset.py`, building on the resolution-consistency interfaces already present there. `_estimate_background_color`, `center_crop_eval_transform`, `whole_specimen_pad_transform`, `EVAL_TRANSFORMS`, and `build_eval_transform()` are existing shared infrastructure and are reused rather than reimplemented. No new dependency, YAML configuration, strategy-class hierarchy, factory layer, or generic registry is introduced.

The module will expose or internally maintain two bounded profile mappings:

```text
PRETRAIN_AUGMENTATIONS:
  global-barcode
  color-robust
  legacy

FINETUNE_AUGMENTATIONS:
  none
  conservative
```

`MultiCropDataset` will use separate transforms for the two global views so that their blur probabilities can differ:

```text
global_tf1

global_tf2

local_tf
```

All views generated from one source image share one background-color estimate. The dataset data flow is:

```text
source image
  -> estimate edge background color once
  -> global transform 1
  -> global transform 2
  -> local transform repeated --local-crops times
```

`MetricDataset` accepts a fine-tuning augmentation profile:

```text
none
  -> existing deterministic training transform

conservative
  -> background-aware full-image geometric and photometric augmentation
```

The trainer passes the resolved profile to the appropriate dataset. No loss code or model-forward contract changes.

## 5. Background-Aware Geometry

### 5.1 Background-color estimation

For an RGB source image, concatenate pixels from all four outer edges and compute the median independently for the red, green, and blue channels. The resulting 8-bit RGB tuple is reused by every augmented view of that source image.

The median is preferred over the mean because it is less sensitive to occasional specimen contact, scale bars, shadows, dust, and other edge outliers. If estimation fails, use the 8-bit RGB equivalent of the ImageNet normalization mean as the fallback fill.

This operation does not require a mask and does not assert that the estimated color is a perfect background model. It only avoids introducing a fixed black rotation border that could become an augmentation shortcut.

### 5.2 Pretraining rotation

For `global-barcode` and `color-robust`, each global or local view follows this geometric order:

```text
RandomResizedCrop
  -> continuous random rotation in [-180 degrees, 180 degrees]
     with bicubic interpolation, expand=True, and edge-derived fill
  -> Resize back to the requested global/local crop size
  -> RandomHorizontalFlip(p=0.5)
```

`expand=True` prevents the rotation operation from clipping additional content from the selected crop. It does not undo content removed by `RandomResizedCrop`, and it does not guarantee that every stochastic SSL view contains the complete specimen. Resizing after rotation restores a fixed tensor size for batching. Near 45-degree rotations may make the selected content appear smaller because the expanded canvas is resized; this is an accepted scale variation.

Only one reflection transform is needed for the `invariant` policy. With arbitrary in-plane rotation, a second vertical reflection is geometrically redundant. The reflection encodes the orientation-insensitive barcode assumption that incidental left/right differences should not dominate the whole-specimen embedding; it is not a claim that real specimens are perfectly symmetric.

For a direction-sensitive marker, use `orientation-policy=sensitive`. This policy changes the geometric direction operations for every global and local training view in `global-barcode` and `color-robust`, and for fine-tuning `conservative`: rotation becomes continuous `[-15 degrees, 15 degrees]` and horizontal reflection is disabled. Fine-tuning `none` has no stochastic geometric operations, so the policy has no transform effect there but remains part of the run's biological contract. The small rotation range models ordinary placement error without asserting that the marker is invariant to a change in anatomical direction. If the marker is sensitive to even small orientation changes, the caller should preserve the original orientation and not use this stochastic profile as a substitute for a canonicalization protocol. `sensitive` is not inferred automatically from the image or taxon.

`orientation-policy` is a high-level policy, not a collection of low-level strength overrides. The default `sensitive` policy applies narrow rotation with no horizontal reflection; the opt-in `invariant` policy preserves the existing broad rotation and reflection behavior. `legacy` remains a historical compatibility profile and is not rewritten by this option; it accepts either policy so checkpoint metadata remains uniform, but its old vertical and horizontal flips remain part of its exact compatibility contract. A policy recorded by `legacy` is still inherited under the ordinary fine-tuning initialization rule: for example, a new `conservative` run initialized from a `legacy` checkpoint that records `sensitive` uses the real narrow-rotation/no-flip `conservative` transform. Pretraining and fine-tuning profiles may differ because they serve different optimization objectives, but a fine-tuning run should use the pretraining checkpoint's policy by default when initialized from that checkpoint. An explicit fine-tuning policy may override it for a new run.

Solarization remains disabled because intensity inversion has no defined biological interpretation for these specimen images and can introduce an artificial shortcut. Matching every transform in the generic DINO recipe is less important than preserving the marker's plausible imaging variation.

### 5.3 Conservative fine-tuning geometry

`conservative` starts from the complete source image rather than from `RandomResizedCrop`. It creates a sufficiently padded square canvas using the edge-derived fill, then applies:

```text
continuous random rotation in [-180 degrees, 180 degrees]
random translation up to 5% of the safe square canvas side on each axis
random scale in [0.9, 1.1]
RandomHorizontalFlip(p=0.5)
Resize to the resolved fine-tuning image size
```

Both fine-tuning profiles use the same resolved `image_size`; the augmentation profile itself does not hardcode a resolution. Use the already implemented resolution-consistency contract to resolve the checkpoint's training size, instantiate the backbone with the matching `img_size`, and pass the same value to `MetricDataset`. The timm model name alone is not authoritative because OTU-Former can override its native size through `img_size`. The augmentation implementation must not duplicate or diverge from that size-resolution logic.

For a source image of width `W` and height `H`, let

```text
D = ceil(sqrt(W^2 + H^2))
canvas_side = ceil(D * max_scale / (1 - 2 * max_translation_fraction))
            = ceil(D * 1.1 / 0.9)
```

Center the unscaled source image on this square canvas using the edge-derived fill. `D` accommodates every in-plane rotation; the additional factor accommodates 1.1x enlargement plus 5% translation space on both sides. Apply rotation, translation, and scale as one affine transform on this fixed canvas, then resize the result to the resolved fine-tuning `image_size`. This construction prevents the specified augmentation operations from clipping the source image. This profile deliberately differs from deterministic inference preprocessing, as training augmentation commonly does. The default `none` profile remains unchanged.

## 6. Profile Definitions

### 6.1 Pretrain `global-barcode`

`global-barcode` is the default for new pretraining runs. Its default `orientation-policy=sensitive` applies the narrow rotation/no-flip geometry for both global and local views as described in Section 5.2; crop, color, grayscale, blur, and all other settings remain unchanged. With the opt-in `orientation-policy=invariant`, the broad geometric behavior below applies. The `sensitive` default was selected because the 50-epoch `global-barcode` implementation comparison found no rotation-consistency gain from broad `invariant` rotation, and ±180° rotation is not a plausible routine augmentation; that comparison did not evaluate the `sensitive` default.

Common geometric settings for both global views and all local views under `orientation-policy=invariant`:

```text
crop scales and ratio: unchanged from Section 3.2
rotation: continuous [-180 degrees, 180 degrees]
rotation expansion: enabled
rotation fill: source-image edge median
post-rotation resize: requested crop size
horizontal reflection: p=0.5
vertical reflection: disabled
solarization: disabled
```

Color settings:

```text
RandomApply(p=0.8):
  ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.1,
    hue=0.02,
  )
RandomGrayscale: disabled
```

Blur settings:

```text
global view 1: GaussianBlur(p=1.0, sigma=(0.1, 2.0))
global view 2: GaussianBlur(p=0.1, sigma=(0.1, 2.0))
local views:    GaussianBlur(p=0.5, sigma=(0.1, 2.0))
```

For the standard crop sizes, use the current odd kernels (global: 23, local: 7). For a smaller user-selected crop, choose the largest odd integer no greater than both the standard kernel and crop size; reject crop sizes below 3 for profiles that use blur. Record the resolved kernel in the expanded augmentation configuration. `legacy` retains its exact fixed kernels for behavioral compatibility.

The asymmetric blur probabilities follow the established DINO multi-view pattern while avoiding mandatory blur on every view. At least one global view will usually retain fine texture.

### 6.2 Pretrain `color-robust`

`color-robust` uses the same crop, orientation-policy-dependent rotation/reflection, and blur behavior as `global-barcode`. Only its color settings differ:

```text
RandomApply(p=0.8):
  ColorJitter(
    brightness=0.4,
    contrast=0.4,
    saturation=0.2,
    hue=0.1,
  )
RandomGrayscale(p=0.2)
```

The name means that training applies stronger color variation; it does not guarantee color-invariant embeddings. CLI help and README documentation must warn that this profile may reduce sensitivity to diagnostic body color, color patterns, or metallic sheen. It is appropriate only when users judge stronger color robustness to be more important than retaining all chromatic signal.

### 6.3 Pretrain `legacy`

`legacy` reproduces the OTU-Former 0.2.1 augmentation behavior and is intended for old-run continuation and experimental comparison, not as the recommended profile for new runs.

```text
global RandomResizedCrop scale: (0.4, 1.0)
local RandomResizedCrop scale:  (0.05, 0.4)
ratio: torchvision default
RandomRotation:
  degrees=180
  interpolation=bicubic
  expand=False
  fill=0
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
ColorJitter(0.4, 0.4, 0.4, 0.1), always applied
RandomGrayscale(p=0.2)
GaussianBlur, always applied:
  global kernel=23, sigma=(0.1, 2.0)
  local kernel=7, sigma=(0.1, 2.0)
```

It does not use edge-derived fill, post-rotation resize, or asymmetric blur probabilities.

### 6.4 Finetune `none`

`none` is the default for new fine-tuning runs and preserves current behavior exactly:

```text
Resize
  -> CenterCrop
  -> ToTensor
  -> ImageNet Normalize
```

The current supervised ArcFace workflow performs well on the available training data. No unverified random augmentation is made the new default.

### 6.5 Finetune `conservative`

`conservative` is an experimental opt-in profile. Its geometric direction operations follow the selected `orientation-policy`:

```text
orientation-policy=invariant:
  rotation [-180 degrees, 180 degrees]
  RandomHorizontalFlip(p=0.5)

orientation-policy=sensitive:
  rotation [-15 degrees, 15 degrees]
  no horizontal reflection
```

The rest of the profile is unchanged. Rotation, translation, and scale are parameters of one affine transform, not three sequential geometric operations:

```text
full source image
  -> edge-background estimate and safe square canvas
  -> one affine transform with selected orientation-policy rotation,
     translation up to 5% of the safe square canvas side per axis,
     and scale [0.9, 1.1]
  -> selected orientation-policy reflection
  -> RandomApply(p=0.8):
       ColorJitter(
         brightness=0.2,
         contrast=0.2,
         saturation=0.1,
         hue=0.02,
       )
  -> Resize(image_size, image_size)
  -> ToTensor
  -> ImageNet Normalize
```

It does not use `RandomResizedCrop`, `RandomGrayscale`, Gaussian blur, or solarization. Documentation must state that `conservative` has not been proven superior to `none` and should be evaluated on held-out individuals and held-out species.

## 7. CLI Contract

### 7.1 Pretrain

```bash
otuformer pretrain --augmentation global-barcode
otuformer pretrain --augmentation color-robust
otuformer pretrain --augmentation legacy
```

Allowed values are exactly `global-barcode`, `color-robust`, and `legacy`. The effective default for a new run is `global-barcode`.

Pretraining also accepts `--orientation-policy invariant|sensitive`, defaulting to `sensitive`. `sensitive` applies to all global and local views in `global-barcode` and `color-robust` and means rotation `[-15°, 15°]` with no horizontal reflection. `invariant` is the opt-in broad-rotation/reflection policy for markers where direction or left/right asymmetry is not diagnostic. `legacy` accepts and records either policy but does not alter its historical flips. Neither policy is inferred from the image or taxon; the caller selects it.

### 7.2 Finetune

```bash
otuformer finetune --augmentation none
otuformer finetune --augmentation conservative
```

Allowed values are exactly `none` and `conservative`. The effective default for a new run is `none`. Fine-tuning also accepts `--orientation-policy invariant|sensitive`. For a new run initialized from `--checkpoint`, an omitted policy inherits the pretraining checkpoint's saved policy; otherwise the default is `sensitive`. An explicit policy is used for the new run. For `none`, the policy has no transform effect but remains part of the biological run metadata.

### 7.3 Explicit versus implicit values

Resume logic must distinguish an omitted CLI value from an explicitly supplied value for both `--augmentation` and `--orientation-policy`. The CLI may therefore represent both omitted options internally as `None`, resolve effective values after checkpoint inspection, and show the effective new-run defaults in help text. A fine-tuning initialization from `--checkpoint` may inherit the orientation policy but never inherits the pretraining augmentation profile.

Unknown profile names fail before dataset or training-loop construction. The dataset constructors also validate profile names so that direct Python use cannot silently fall back to another policy.

No new CLI flags are added for individual transform strengths. Existing crop size and local-crop count options remain available.

### 7.4 Help requirements

`pretrain --help` and `finetune --help` must describe:

- each allowed profile and its effective default;
- the orientation policies, their geometric meanings, inheritance behavior, and biological use cases;
- the distinction between marker observation view and arbitrary in-plane orientation;
- the requirement not to mix dorsal, ventral, lateral, and anatomical-part markers;
- the color-information risk of `color-robust`;
- the experimental status of finetune `conservative`;
- that `sensitive` is an explicit caller-selected policy rather than automatic biological inference;
- the purpose of `legacy`, including that it records either orientation policy without changing its historical transforms while that recorded policy can be inherited by a new fine-tuning `conservative` run;
- resume inheritance and conflict behavior;
- that changing a profile's expanded parameters constitutes a new training run rather than a resumable continuation;
- that augmentation encourages, but does not guarantee, embedding invariance.

The README may carry the full explanation; CLI help must remain detailed enough for correct selection without requiring source-code inspection.

## 8. Checkpoint, Logging, and Resume Contract

### 8.1 New checkpoint metadata

New pretrain and finetune checkpoints add JSON-serializable augmentation metadata under `config`:

```python
{
    "augmentation_profile": "global-barcode",
    "augmentation_config": {
        "profile": "global-barcode",
        "orientation_policy": "invariant",
        "global_crop": {"size": 224, "scale": [0.4, 1.0]},
        # Fully expanded settings used by this run, including
        # solarization_probability and the resolved crop/image sizes.
    },
}
```

`augmentation_profile` remains top-level for quick identification; the matching nested `augmentation_config.profile` keeps the expanded configuration self-contained for exact comparison and is not an independent override. `orientation_policy` is stored only inside `augmentation_config` to avoid duplicate sources of truth. Resume reads the policy from that expanded configuration. Old checkpoints without the field use `invariant` only where the compatibility rules below explicitly allow it.

The expanded configuration includes, as applicable:

- the selected `orientation_policy` and its resolved rotation/reflection settings for all applicable global, local, and fine-tuning views;
- the self-contained expanded `profile` name;
- crop scales, aspect-ratio range, crop sizes, and local-crop count, including `global_crop.size` for pretraining and `image_size` for fine-tuning;
- transform order;
- rotation range, interpolation, expansion, and fill strategy;
- reflection types and probabilities;
- color-jitter values and application probability;
- grayscale probability;
- blur probabilities, kernel sizes, and sigma range;
- `solarization_probability`, which is `0.0` for the profiles in this design;
- translation and scale ranges;
- fallback background color.

Training startup logs print the resolved profile and expanded configuration.

This metadata provides **configuration traceability**, not bitwise run-to-run reproducibility. Exact stochastic replay also depends on runtime state such as random-number-generator state, worker scheduling, library versions, device kernels, and deterministic-algorithm settings; this design does not add a bitwise-replay guarantee.

### 8.2 Pretrain resume

For a checkpoint containing augmentation metadata:

- omitted `--augmentation` and `--orientation-policy`: inherit the saved profile and policy from `augmentation_config`; the current expanded definition must exactly match the saved configuration, otherwise fail;
- omitted `--local-crop-size` and `--local-crops` on pretrain resume: inherit the saved `augmentation_config.local_crop.size` and `augmentation_config.local_crops`; explicit local-view values must exactly match or resume fails;
- explicitly matching profile/policy, local-view settings, and expanded configuration: resume;
- a different profile or policy: fail;
- the same profile/policy whose current expanded definition differs from the checkpoint: fail.

For a pre-augmentation checkpoint without augmentation metadata:

- omitted `--augmentation`: resolve to `legacy`;
- explicit `--augmentation legacy`: resume;
- any other explicit profile: fail;
- omitted or explicit `--orientation-policy invariant`: resolve to `invariant` for compatibility;
- explicit `--orientation-policy sensitive`: fail because the old run has no saved policy to prove compatibility;
- for `local_crop_size` and `local_crops`, an omitted CLI value inherits the checkpoint's saved `args` value; an explicit value must exactly match it or resume fails;
- `local_crop_size` is a non-boolean positive integer; `local_crops` is a non-boolean non-negative integer, so `0` remains the valid setting that disables local views;
- if either saved local-view field is absent or invalid, use its historical default (`96` for `local_crop_size`, `6` for `local_crops`) and apply the same explicit-value conflict rule. A saved valid `local_crops=0` must remain `0`.

The error explains that `--resume` continues the same training plan. To change augmentation or local-view settings, users must start a new run rather than resume the old run.

### 8.3 Finetune resume

For a checkpoint containing augmentation metadata, use the same inheritance and exact-expanded-config checks as pretraining. Read the saved orientation policy from `augmentation_config`; it is part of this compatibility check.

For a pre-augmentation finetune checkpoint without augmentation metadata:

- omitted `--augmentation`: resolve to `none`;
- explicit `--augmentation none`: resume;
- explicit `--augmentation conservative`: fail;
- omitted or explicit `--orientation-policy invariant`: resolve to `invariant` for compatibility;
- explicit `--orientation-policy sensitive`: fail because the old run has no saved policy to prove compatibility.

Starting a new fine-tuning run from a pretraining checkpoint via `--checkpoint` is not resume:

- omitted `--augmentation`: use `none`;
- explicit `--augmentation conservative`: use `conservative`;
- omitted `--orientation-policy`: inherit the pretraining checkpoint's saved policy, including a policy recorded by `legacy`; if the old pretraining checkpoint has no augmentation metadata, fall back to `sensitive`. The inherited policy has its normal transform effect if the selected fine-tuning profile is `conservative`;
- explicit `--orientation-policy`: use the requested policy, including `sensitive` for an old checkpoint when starting a new fine-tuning run;
- do not inherit the pretraining profile.

## 9. Error Handling

The implementation must fail clearly when:

- an unknown profile is supplied;
- a resumed run requests a different profile;
- a same-named profile expands differently from checkpoint metadata;
- required checkpoint augmentation metadata is malformed;
- generated views do not have the requested fixed output size;
- a requested orientation policy is incompatible with the selected profile or checkpoint contract.

Background estimation failure is recoverable and uses the documented ImageNet-mean fallback. It must not silently switch the selected profile.

Validation occurs before starting the DataLoader or training loop whenever the required information is already available.

## 10. Verification

### 10.1 Automated tests

Extend existing tests rather than creating a new testing framework.

Dataset-level checks:

- all three pretrain profiles return two global views plus the configured number of local views;
- default local-crop count remains six;
- output tensor dimensions remain fixed after expanded rotation;
- rectangular source images can be batched;
- edge-background estimation uses the per-channel median of all four image edges;
- every view from one source image receives the same estimated fill;
- estimation failure uses the documented fallback;
- `legacy` retains black fill, `expand=False`, both flips, grayscale, always-on color jitter, and always-on blur;
- `none` retains the current deterministic fine-tuning transform;
- `conservative` contains no random resized crop, grayscale, blur, or solarization and returns a fixed-size tensor;
- sensitive-policy propagation is checked for global views, local views, and fine-tuning `conservative`; `legacy` records the policy without changing its historical transform;
- generated-view shape mismatches raise a clear error;
- unknown profiles raise an explicit error.

CLI/checkpoint checks:

- help lists allowed values, effective defaults, orientation policies, and biological warnings;
- unknown profiles fail before training;
- omitted profile/policy values resolve to the correct new-run defaults or checkpoint inheritance rules;
- new checkpoints save the profile, orientation policy, and expanded configuration;
- resume inherits metadata when the CLI option is omitted;
- old pretrain checkpoints map to `legacy`/`invariant` for resume and inherit their saved local-view arguments; `local_crop_size` must be a positive integer, `local_crops` may be zero, explicit local-view conflicts fail, and only missing/invalid historical fields fall back to `96` local crop size and `6` local crops;
- old finetune checkpoints map to `none`/`invariant` for resume;
- old pretraining checkpoints used for new fine-tuning initialization fall back to `sensitive` when policy is omitted, and explicit `conservative` initializes without inheriting the pretraining profile;
- profile and expanded-config conflicts reject resume;
- new fine-tuning from a pretraining checkpoint does not inherit the pretraining profile.

Tests assert configuration, dimensions, invariants, and compatibility behavior. They do not assert exact random pixel values.

### 10.2 Smoke training

Run short pretraining smoke checks for:

- `global-barcode`;
- `color-robust`;
- `legacy`.

Run short fine-tuning smoke checks for:

- `none`;
- `conservative`.

Run one additional `orientation-policy=sensitive` smoke per stage: `global-barcode` for pretraining and `conservative` for fine-tuning. Each smoke check confirms finite loss, successful forward/backward execution, checkpoint writing and resume, and complete augmentation logging. Health-log checks must match standalone `NaN`/`Inf` values or `Traceback`, not the `Inf` prefix in normal `[Info]` log lines.

### 10.3 Extended implementation comparison

Run the existing Epidorcus example for 50 pretraining epochs with `global-barcode`, keeping other baseline training parameters unchanged. This is an implementation comparison and smoke-level health check, not evidence that the profile is biologically optimal across datasets.

Training-health checks:

- completion of all 50 epochs;
- every logged loss is finite;
- the last available finite positive `feature_std` is recorded from `instant_metrics.pretrain.csv` with its epoch and step; this is not a checkpoint-final value;
- different-image cosine similarities are not all numerically equal to one.

Embedding checks:

- read the final available rows for the seven existing comparison metrics from each run's `metrics.pretrain.csv`: Recall@1, kNN accuracy at k=1, k=5, and k=20, linear-probe accuracy, mAP, and Silhouette Score;
- do not add a new evaluator; record a metric as `unavailable` when it is absent or was not computed;
- compare the available implementation metrics against the existing checkpoint baseline using the same evaluation data and settings, without treating this comparison as a broad biological-readiness decision.

Rotation- and reflection-consistency checks:

- use a fixed specimen sample recorded in the validation report;
- compare CLS embeddings for 0, 90, 180, and 270 degrees;
- report the median, 10th percentile, and minimum paired cosine similarity for each non-zero angle;
- compare each source image with a horizontally reflected counterpart and report the same summary statistics;
- for a direction-sensitive run, verify and report the selected `sensitive` no-flip/narrow-rotation contract; reflection similarity is diagnostic only and is not an invariance acceptance target;
- report the median and 90th percentile different-image similarity as a baseline so collapse cannot masquerade as invariance;
- construct each encoder with its own checkpoint training size resolved by `resolve_training_image_size()` before loading weights; an optional evaluation image-size override changes deterministic preprocessing only, never that model-construction size;
- compare against the current checkpoint using the same images and effective deterministic evaluation preprocessing size;
- require the new 90-degree median not to fall below the current-checkpoint median. Report the other angles without claiming universal rotation invariance.

`color-robust` and fine-tuning `conservative` require automated and smoke validation in this change, but not complete training runs. Their scientific value remains dataset-dependent.

## 11. User Evaluation Guidance

Because image sources are often heterogeneous and preservation metadata may be unavailable, users should not evaluate profiles only on the same labeled images used for ArcFace training.

Recommended evaluation separates:

1. **Closed-set individual generalization**: test individuals and photographs are absent from training, while their species occur in training.
2. **Open-set species generalization**: hold out complete species from fine-tuning and examine whether unseen species form biologically plausible clusters.
3. **Acquisition robustness**: stress-test orientation and modest color changes without treating synthetic perturbations as proof of real preservation robustness.

Useful outputs include retrieval metrics, kNN metrics, cluster-quality metrics, nearest-neighbor inspection, and transformed-image cosine similarity. Average scores should be supplemented by per-class inspection because augmentation can help common classes while erasing diagnostic traits in particular taxa.

## 12. Documentation Changes After Spec Approval

The later implementation phase will update:

- `docs/superpowers/specs/2026-03-29-otuformer-design.md` with the final pretrain and finetune CLI contract;
- `README.md`;
- `README.cn.md`;
- pretrain CLI help;
- finetune CLI help.

It will not modify `docs/superpowers/specs/2026-03-30-otuformer-pretrain-alignment-design.md`, whose purpose is historical pretraining-algorithm alignment and whose scope explicitly excludes fine-tuning.

The implementation plan is `docs/superpowers/plans/2026-09-07-otuformer-training-augmentation.md` and must be executed only after this design document is reviewed and approved.

## 13. Future Dense Morphology Compatibility

The current `--augmentation` option and internal profile mapping provide a stable naming surface for future work. No dense profile is added now because augmentation alone cannot implement dense morphology.

A real dense-morphology design is expected to address patch-level outputs, coordinate-aware transforms, corresponding-region objectives, anatomical supervision or anchors, extraction formats, and dense validation metrics. When those components exist, a dense-specific augmentation profile can be added to the same interface. If transforms must return geometry metadata, the dataset return contract can then be expanded deliberately rather than prebuilding an unused abstraction now.

## 14. Risks and Mitigations

### Color information loss

Stronger jitter and grayscale may remove taxonomic signal. Mitigation: `global-barcode` is color-preserving by default; `color-robust` is opt-in and prominently warned.

### Crop removes diagnostic structures

Classic multi-crop can remove specimen parts. Mitigation: retain the established crop behavior for algorithmic continuity, document the limitation, and preserve six local views as auxiliary SSL observations. The complete-marker requirement applies to the source image, not to every stochastic view. `expand=True` only prevents additional clipping during rotation; it cannot restore content already removed by `RandomResizedCrop`. Structure-retention audits and downstream validation must reveal whether this trade-off is unacceptable. This design does not claim every crop is a complete marker.

### Rotation artifacts become shortcuts

Fixed black corners can encode augmentation angle. Mitigation: use one edge-derived background estimate per source image and expanded rotation.

### Fine-tuning overfits known classes

ArcFace may improve training-class separation while harming unseen-species structure. Mitigation: keep `none` as the default, mark `conservative` experimental, and recommend held-out-individual and held-out-species evaluation.

### Resume silently changes the data distribution

A profile definition or CLI default could change across versions. Mitigation: persist both the profile name and fully expanded configuration, compare them on resume, and map old checkpoints conservatively.

### Broader pipeline behavior changes unintentionally

Training augmentation changes could accidentally alter the completed resolution-consistency paths. Mitigation: keep deterministic evaluation, extraction, CAM, and ONNX implementation out of scope; reuse the existing named evaluation-transform and size interfaces; and cover their unchanged contracts in regression tests where appropriate.

The former extraction/CAM preprocessing discrepancy has already been addressed by `docs/superpowers/plans/2026-09-07-otuformer-resolution-consistency.md` and its implementation. That work owns `center_crop_eval_transform`, `whole_specimen_pad_transform`, `--eval-transform`, checkpoint-size propagation, and non-square CAM mapping. The augmentation change must treat those interfaces as fixed dependencies, not recreate or modify them.

## 15. Acceptance Criteria

The implementation is acceptable when:

- new pretraining defaults to `global-barcode` with `orientation-policy=sensitive`;
- new fine-tuning defaults to `none`; when initialized from a pretraining checkpoint, omitted `orientation-policy` inherits the checkpoint policy;
- every profile and supported orientation policy behaves as specified and is fully documented;
- pretraining and fine-tuning profiles may differ, while orientation-policy inheritance and explicit override behavior are unambiguous;
- legacy and resume compatibility tests pass;
- checkpoint and logs contain reproducible augmentation metadata;
- existing loss and global-view pairing behavior is unchanged;
- augmentation does not modify extraction, CAM, ONNX, or deterministic evaluation preprocessing; those paths continue to use the completed resolution-consistency interfaces;
- all automated and smoke checks pass;
- the 50-epoch `global-barcode` validation completes without training collapse or broad unexplained metric regression;
- rotation- and reflection-consistency results and different-image baselines are reported without overstating invariance; direction-sensitive validation verifies the selected `sensitive` policy rather than treating reflection similarity as a required invariant.
- no implementation plan, code change, or commit is performed without its separately required approval;
- the extended implementation comparison does not claim broad biological suitability of `global-barcode`.
