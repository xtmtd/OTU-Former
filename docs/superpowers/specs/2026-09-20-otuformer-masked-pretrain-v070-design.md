# OTU-Former v0.7.0 Masked Pretraining Design

Date: 2026-09-20  
Status: Implemented  
Target release: v0.7.0  
Implementation plan: [2026-09-20-otuformer-masked-pretrain-v070.md](../plans/2026-09-20-otuformer-masked-pretrain-v070.md)

This document records the agreed design for upgrading OTU-Former's patch-level
pretraining objectives and masking strategies. It was reviewed and approved,
then implemented as described above.

## 1. Context

The current pretraining loop combines global and local teacher-student losses
with a function named masked-token loss. The current patch loss does not mask
the student input. Both student and teacher receive the complete global view,
and the loss samples matching patch positions from their final token sequences:

```text
student(full view) -> final patch tokens
teacher(full view) -> final patch tokens
random positions   -> same-position regression
```

The current behavior is useful as local regularization, but its accurate name
is **masked-position patch consistency**, not masked image modeling. The student
can directly observe the visual content at every selected position.

The v0.7.0 design keeps this mature behavior as the default while adding two
opt-in objectives that truly hide student patch content:

- continuous masked feature prediction, inspired by data2vec-style latent
  prediction;
- lightweight iBOT-style prototype distribution prediction, marked
  experimental.

These objectives are self-supervised. `L_masked_feature` does not require part,
segmentation, instance, or keypoint labels: the full-image EMA teacher supplies
the target. The objectives may improve contextual and local representations,
but they do not by themselves establish anatomical part semantics or constitute
dense morphology supervision.

## 2. Goals

1. Preserve the current patch consistency objective as the default.
2. Add true student-side token masking without changing the existing global and
   local SSL path.
3. Add four mutually exclusive patch-loss modes for controlled ablation.
4. Add random, blockwise, and hybrid masking as a separate experiment axis.
5. Keep extraction, fine-tuning, CAM, and export interfaces independent of
   training-only patch heads.
6. Preserve strict resume semantics and explicitly handle legacy checkpoints.
7. Add enough diagnostics to detect masking errors and iBOT prototype collapse.
8. Limit v0.7.0 pretraining support to standard timm ViT backbones.

## 3. Non-goals

v0.7.0 will not add:

- part labels, semantic segmentation, instance segmentation, keypoints, or
  coordinate supervision;
- a foreground heuristic or specimen-mask estimator;
- dynamic mask-ratio schedules;
- simultaneous A+ and iBOT losses;
- Sinkhorn-Knopp assignment;
- DINOv3 Gram anchoring;
- a new pretrain-checkpoint initialization workflow;
- separate learning rates for mask tokens or patch heads;
- architecture adapters for EVA, Swin, CNN, or arbitrary timm models;
- a full dense morphology benchmark.

## 4. Terminology

### 4.1 Patch-loss modes

```text
none             no patch-level objective
consistency      current full-input same-position patch consistency
masked-feature   true masked continuous feature prediction (A+)
ibot             true masked prototype-distribution prediction (experimental)
```

`A+` is an internal shorthand used only in this project. It means true masked
continuous feature prediction using the normalized mean of the teacher's final
four block features as the target. Documentation and plans must define this term
on first use; the CLI name remains the descriptive `masked-feature`.

The modes are mutually exclusive. v0.7.0 will not expose a combined
`masked-feature + ibot` mode.

### 4.2 Masking strategies

```text
random      independently selected, spatially dispersed patch positions
blockwise   multiple bounded rectangular regions on the patch grid
hybrid      `floor(target_count / 2)` blockwise positions, then random fill
```

`masking_strategy` controls real input masking only. It is not used by `none`
or `consistency`.

## 5. High-level Architecture

The existing unmasked global and local SSL path remains unchanged for every
patch-loss mode:

```text
student(unmasked global views) -> existing global SSL
student(unmasked local views)  -> existing local-to-global SSL
teacher(unmasked global views) -> existing global targets
```

Patch behavior is added alongside that path:

```text
consistency:
  student(unmasked global view) + teacher(unmasked same view)
  -> selected same-position final patch tokens
  -> normalized cosine regression

masked-feature:
  teacher(unmasked global view) -> last-four-block mean patch target
  student(masked same view)     -> final masked patch tokens -> predictor
  -> normalized cosine regression at masked positions

ibot:
  teacher(unmasked global view) -> final patch tokens -> teacher patch head
  student(masked same view)     -> final patch tokens -> student patch head
  -> centered prototype cross-entropy at masked positions
```

`masked-feature` and `ibot` therefore add one masked student forward for each of
the two global views. They do not reuse the masked CLS output for the global
loss. This adds compute, but it keeps the global/local training contract stable
and makes patch-loss ablations interpretable.

The existing unmasked teacher pass must be extended to return all targets needed
by the selected objective. For `masked-feature`, the same teacher pass returns
both its normal final outputs and the final four transformer block outputs. This
is a model-forward interface change, not an additional teacher forward. The
implementation must not run the teacher twice for the same global view.

At the default two `224 x 224` global crops and six `96 x 96` local crops, an
input-area proxy estimates that the two extra masked global forwards increase
student encoder work to about `1.65x` the current student work and total
student-plus-teacher forward work to about `1.39x`. Backward computation and
hardware effects prevent this proxy from predicting wall-clock time. Controlled
ablations must therefore report step time, images per second, and peak device
memory alongside representation metrics.

## 6. Component Boundaries

### 6.1 General encoder

`OTUFormerEncoder` continues to own:

- the timm ViT backbone;
- the existing CLS projection head;
- the existing global center;
- normal unmasked inference.

Its training forward contract is extended only as requested by the selected
objective. The normal path still returns the current final outputs. The A+ path
also returns the final four block outputs from the same forward pass; it does
not invoke the backbone again.

The default extraction path remains:

```text
full image -> backbone.forward_features -> raw CLS token
```

`--use-projector-output` continues to use the existing CLS projector.
`patch-topk` and `attention-pool` continue to consume normal final backbone
patch tokens.

### 6.2 Training-only patch objective

A separate training component owns:

- the student-only learnable mask token;
- masking generation;
- the A+ predictor when `patch_loss=masked-feature`;
- the student and teacher iBOT heads when `patch_loss=ibot`;
- the iBOT patch center;
- patch-specific diagnostics.

These states must not be required by normal extraction, fine-tuning, CAM, or
export.

### 6.3 EMA update

The current positional parameter update:

```python
for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
    ...
```

must be replaced by name-matched EMA updates. Adding student-only parameters
would otherwise make positional matching unsafe.

EMA ownership is:

```text
student backbone + CLS projector -> teacher encoder EMA
student iBOT patch head          -> teacher iBOT patch head EMA
mask token                       -> student-only, no EMA
A+ predictor                     -> student-only, no EMA
```

The teacher patch head starts as an exact copy of the student patch head, is
gradient-free, and uses the same momentum schedule as the teacher encoder.

## 7. Mask Injection

The student mask token is a single learnable parameter of shape
`[1, 1, hidden_dim]`, initialized with a truncated normal distribution.

Masking replaces patch content while retaining position information. The
supported adapter must preserve the following semantic order:

```text
image
-> patch embedding
-> replace selected patch embeddings with the learnable mask token
-> add prefix tokens and positional embeddings
-> transformer blocks
```

An implementation may replace tokens after positional embedding only when the
replacement is exactly equivalent to `mask_token + original_position_embedding`.
It must never erase the masked position's spatial encoding.

Only patch tokens are maskable. CLS, distillation, and register/prefix tokens
must not be masked. Tests must verify the prefix-token offset and the patch-grid
mapping at multiple supported input sizes.

## 8. Mask Generation

### 8.1 Shared rules

- Each sample and each global view receives an independently sampled mask.
- The two global views never share a mask.
- Teacher and masked student use the same augmented view; the teacher sees it
  unmasked.
- Real masking is performed on every masked student forward. There is no
  `mask_sample_probability` in v0.7.0.
- Ratios are fixed for a run. There is no dynamic ratio range or schedule.
- Sampling covers the complete patch grid. There is no foreground heuristic.
- The resolved count is clamped to at least one selected patch and at least one
  visible patch.
- The patch loss is normalized by the actual number of selected/masked tokens.

For a grid containing `N` patch tokens:

```text
target_count = clamp(round(mask_ratio * N), 1, N - 1)
```

### 8.2 Random

Sample `target_count` unique patch positions uniformly without replacement.

### 8.3 Blockwise

Generate two to four rectangle proposals using these fixed internal rules:

- aspect ratio in `[0.5, 2.0]`;
- minimum side length of two patches;
- maximum single-block area of `floor(0.20 * N)` patches;
- rectangle positions sampled uniformly over valid grid locations.

Before training, `blockwise` and `hybrid` must verify feasibility against the
actual patch-grid height and width `(H, W)`, not only the total `N = H * W`.
The check enumerates candidate integer rectangle heights and widths and requires
at least one pair whose sides fit within `(H, W)`, whose aspect ratio is in
`[0.5, 2.0]`, whose sides are both at least two, and whose area does not exceed
`floor(0.20 * N)`. In particular, a `4 x 4` grid is invalid because its maximum
block area is three patches while the minimum `2 x 2` block needs four; a narrow
grid can also be invalid even when the area bound alone is at least four. An
infeasible grid fails before the training loop; it must not retry indefinitely
or silently degrade to random masking.

After at least one legal rectangle exists, merge overlapping rectangles and
continue sampling until the target count is reached or exceeded. If under
target, fill from uniformly sampled unmasked positions. If over target, remove
excess positions from the final proposal so the returned mask has exactly
`target_count` positions.

### 8.4 Hybrid

Use the blockwise sampler for exactly `floor(target_count / 2)` positions,
then fill the remainder uniformly at random without replacement. Overlap is
counted once, and the final mask contains exactly `target_count` positions.

## 9. Patch-loss Definitions

### 9.1 `none`

No mask is generated, no patch objective is constructed, and no patch loss is
added to the total loss.

### 9.2 `consistency` (default)

Both student and teacher receive the full global view. Randomly select matching
patch positions and compute normalized cosine regression:

```text
s = normalize(student_final_patch[selected])
t = normalize(stop_gradient(teacher_final_patch[selected]))
L_consistency = mean(2 - 2 * dot(s, t))
```

This mode does not consume `masking_strategy`, does not create a mask token, and
must be documented as masked-position consistency rather than masked image
modeling.

The old EVA-specific MSE branch is removed. Standard supported ViTs use one
consistent loss definition.

### 9.3 `masked-feature` (A+)

The teacher target is constructed from its final four transformer block outputs
at each patch position. Apply the backbone's final token normalization to each
selected block output independently, remove all prefix tokens, average the four
normalized patch representations, then L2-normalize the mean:

```text
per_layer = backbone_final_norm(teacher_block_output)
patch_layers = remove_prefix_tokens(per_layer)
target = l2_normalize(mean(last_4_patch_layers))
target = stop_gradient(target)
```

These intermediate and final outputs come from one teacher backbone execution.

The student uses the final masked-forward patch tokens. Its predictor is:

```text
LayerNorm(hidden_dim)
Linear(hidden_dim, hidden_dim)
GELU
Linear(hidden_dim, hidden_dim)
L2 normalization
```

The loss is computed only at masked positions:

```text
L_masked_feature = mean(2 - 2 * dot(student_prediction, teacher_target))
```

The four-layer target depth is fixed in v0.7.0 and is not exposed as a CLI
parameter. The predictor output dimension is the backbone hidden dimension and
does not change the existing CLS `out_dim` or extraction dimensions.

### 9.4 `ibot` (experimental)

iBOT uses final-layer teacher and student patch tokens, not the A+ four-layer
teacher target. CLS and patch heads share the backbone but are otherwise
decoupled.

The lightweight patch head is:

```text
LayerNorm(hidden_dim)
Linear(hidden_dim, hidden_dim)
GELU
Linear(hidden_dim, ibot_prototypes)
```

The teacher target and student prediction are:

```text
teacher_probs = softmax((teacher_logits - patch_center) / teacher_temp)
student_log_probs = log_softmax(student_logits / student_temp)
L_ibot = mean(-sum(teacher_probs * student_log_probs))
```

The loss is computed only at student-masked positions, using corresponding
unmasked teacher positions. Teacher probabilities are stop-gradient.

The iBOT path reuses the existing student temperature and teacher temperature
schedule. It uses moving-average centering, not Sinkhorn-Knopp. Patch-center
momentum is fixed at `0.9`:

```text
patch_center = 0.9 * patch_center + 0.1 * mean(current_teacher_logits)
```

Only teacher logits at positions used by the current loss update the center.
Both global views are concatenated and the center is updated once per step.

Update order is:

1. compute teacher logits with the current teacher and old patch center;
2. compute losses and update the student;
3. EMA-update the teacher encoder and teacher iBOT head;
4. update the patch center from the teacher logits already computed in step 1.

The implementation must not run another teacher forward to update the center.

## 10. CLI Contract

### 10.1 New options

```text
--patch-loss none|consistency|masked-feature|ibot
  default: consistency

--masking-strategy random|blockwise|hybrid
  default: random
  effective only for masked-feature and ibot

--mask-ratio auto|FLOAT
  default: auto
  explicit FLOAT must satisfy 0 < value < 1

--ibot-prototypes INT
  default: 512
  any integer >= 2; larger dictionaries suit larger datasets
  effective only for ibot
```

The existing `--lambda-mask` remains and defaults to `1.0`.

### 10.2 Resolved defaults for new runs

| Patch loss | Masking strategy | Resolved ratio |
|---|---|---:|
| `none` | not applicable | `null` |
| `consistency` | not applicable | `0.30` |
| `masked-feature` | `random` | `0.30` |
| `masked-feature` | `blockwise` | `0.20` |
| `masked-feature` | `hybrid` | `0.30` |
| `ibot` | `random` | `0.30` |
| `ibot` | `blockwise` | `0.20` |
| `ibot` | `hybrid` | `0.30` |

The same numeric ratio has different semantics across modes. In `consistency`,
it is the fraction of fully visible patch positions included in the loss. In
`masked-feature` and `ibot`, it is the fraction of student patch inputs replaced
by the mask token.

This is an intentional new-run default change. v0.6.x used a float CLI default
of `0.50`; v0.7.0 changes the CLI to `auto|FLOAT` and resolves a new
`consistency` run to `0.30`. Existing users must receive this migration notice
in both READMEs, CLI help/release notes, and the version documentation. Legacy
resume behavior remains the `0.50` rule in Section 14.2.

### 10.3 Invalid combinations

The CLI must detect whether the user explicitly supplied an option rather than
only inspecting its parsed default.

- `none`: explicit mask ratio, masking strategy, iBOT prototypes, or a
  non-default lambda-mask is rejected.
- `consistency`: explicit masking strategy or iBOT prototypes is rejected.
- `masked-feature`: explicit iBOT prototypes is rejected.
- `ibot`: all listed patch options are valid.

Defaults that are inapplicable to the selected mode do not raise errors. They
resolve to `null` in the effective configuration. Explicitly supplied but
inapplicable values are never silently ignored.

## 11. Loss Weighting and Optimizer

All patch losses retain the common default:

```text
--lambda-mask 1.0
```

The iBOT cross-entropy is not divided by `log(K)`. Loss values from different
patch modes are not numerically comparable.

Training-only parameters join the existing student AdamW optimizer:

```text
consistency:
  student encoder

masked-feature:
  student encoder + mask token + A+ predictor

ibot:
  student encoder + mask token + student iBOT head
```

They use the existing learning rate, weight decay, and schedules. v0.7.0 adds no
patch-specific optimizer group or learning-rate option. Teacher parameters and
patch-center buffers never enter the optimizer.

## 12. Backbone Support

v0.7.0 pretraining and pretraining resume support only standard timm ViT
backbones supported by the explicit masked-forward adapter. Training no longer
contains EVA-specific loss behavior and does not add adapters for EVA, Swin,
CNN, or arbitrary token-producing models.

For real masking, startup capability checks must verify at least:

- patch embedding access;
- transformer block access;
- reliable prefix-token count;
- token-sequence outputs;
- position handling compatible with dynamic input sizes;
- access to the final four block outputs for A+.

Unsupported models fail before training starts. There is no silent fallback to
consistency or an unmasked forward.

Read-only consumers may continue loading old EVA checkpoints when the generic
timm path works, but this is best-effort compatibility. v0.7.0 does not promise
EVA-specific support for extract, fine-tune, CAM, or export and does not allow
old EVA pretraining runs to resume.

## 13. Checkpoint Layout

The general encoder remains separately loadable:

```text
student:
  backbone + existing CLS projector + global center

teacher:
  backbone + existing CLS projector + global center
```

Training-only state is stored separately:

```text
patch_objective:
  mask_token                         # masked-feature / ibot
  predictor                          # masked-feature only
  student_ibot_head                  # ibot only
  teacher_ibot_head                  # ibot only
  patch_center                       # ibot only
```

Checkpoint config records:

```text
patch_loss
masking_strategy           # null when inapplicable
mask_ratio_requested       # "auto" or explicit value
mask_ratio_resolved        # effective value or null
ibot_prototypes            # null unless ibot
patch_target_layers        # 4 for masked-feature, otherwise null
patch_center_momentum      # 0.9 for ibot, otherwise null
```

Normal extract, fine-tune, CAM, and export loaders continue loading only the
general encoder and ignore `patch_objective`.

## 14. Resume and Legacy Compatibility

### 14.1 v0.7.0 checkpoints

Resume is strict. Before loading optimizer state, the training process rebuilds
the exact patch objective recorded by the checkpoint. All effective settings
must match:

- patch-loss mode;
- resolved mask ratio;
- masking strategy when applicable;
- iBOT prototype count when applicable;
- fixed target/center settings.

Changing patch strategy through `--resume` is rejected. v0.7.0 does not add an
`--init-checkpoint` workflow for starting A+ or iBOT from an older SSL model.

### 14.2 Legacy checkpoints

A checkpoint with no `patch_loss` metadata is interpreted as `consistency`.

```text
saved args.mask_ratio exists -> inherit that value, including 0.50
saved mask_ratio is absent   -> historical fallback 0.50
```

An explicitly supplied conflicting ratio or patch-loss mode is rejected.
Legacy checkpoints cannot resume as `masked-feature` or `ibot` because they
lack the required model and optimizer state.

Legacy checkpoints do not contain RNG state. Their resume remains best-effort
and must emit a warning that the random sequence cannot continue exactly.

## 15. Randomness and Reproducibility

Mask generation reuses the existing training RNG and `--seed`; there is no
separate `--mask-seed`.

New checkpoints store:

- Python `random` state;
- NumPy RNG state;
- PyTorch CPU RNG state;
- PyTorch CUDA RNG state for every available device.

Resume restores these states after model, patch-objective, optimizer, and
schedule reconstruction, and before the next epoch's DataLoader iterator is
created. The saved RNG snapshot must represent the boundary after all epoch-end
evaluation and logging. The implementation must either write the checkpoint
after those operations or isolate them so they cannot advance the training RNG.
Since checkpoints are written at epoch boundaries, this preserves the mask
sequence and main-process shuffle state for v0.7.0 runs.

Exact replay of multi-worker image augmentation is not a v0.7.0 guarantee.
Solving that fully would require a separate per-sample/per-epoch augmentation
seeding design.

## 16. Logging and Diagnostics

Existing raw-CLS periodic evaluation remains the primary global evaluation:

- kNN accuracy;
- linear probing;
- NMI, ARI, and purity;
- Recall@K and mAP;
- silhouette score.

The instant pretraining log adds or clarifies:

```text
patch_loss_raw
patch_loss_weighted
patch_loss_fraction_of_total
requested_mask_ratio
selected_patch_ratio
actual_mask_ratio_mean
actual_mask_ratio_min
actual_mask_ratio_max
```

Actual masking-ratio fields are populated only for true masking modes. For
`consistency`, all `actual_mask_ratio_*` fields are empty and a separate
`selected_patch_ratio` field records the actual fraction of fully visible patch
positions included in its loss. This must never imply that input masking
occurred.

The iBOT mode additionally logs:

```text
prototype_perplexity
active_prototype_ratio
assignment_entropy
max_prototype_occupancy
patch_center_norm
```

These metrics are stability diagnostics, not proof of anatomical semantics.
They are computed from teacher probabilities at all masked positions in both
global views for the current logged step:

```text
mean_assignment = mean(teacher_probs, over masked positions)
prototype_perplexity = exp(entropy(mean_assignment))
active_prototype_ratio = count(hard-assignment count > 0) / K
assignment_entropy = mean(entropy(teacher_probs per masked position))
max_prototype_occupancy = max(hard-assignment count) / number_of_assignments
```

A hard assignment is `argmax(teacher_probs)`. This definition and its per-step
logging window remain fixed within v0.7.x.

### 16.1 CSV schema migration

The v0.7.0 instant-metrics schema adds the fields in this section. On resume,
the logger must validate the existing CSV header before writing:

- a current v0.7.0 schema is appended normally;
- a recognized legacy pretrain schema is atomically rewritten to the v0.7.0
  header, preserving all historical rows and filling every newly introduced
  field with an empty value. This includes `patch_loss_raw`,
  `patch_loss_weighted`, `patch_loss_fraction_of_total`,
  `requested_mask_ratio`, `selected_patch_ratio`, all three defined
  `actual_mask_ratio_*` fields, and every iBOT-only diagnostic listed above;
- an unknown, duplicate-column, or malformed schema fails with an actionable
  error and is not appended.

Migration writes a temporary file in the same directory and installs it with an
atomic replace only after the complete rewritten CSV has been flushed and
closed. Plotting must accept the empty historical fields produced by this
migration.

## 17. Evaluation Strategy

The minimum controlled comparison is:

```text
patch-loss=none
patch-loss=consistency
patch-loss=masked-feature
patch-loss=ibot
```

Runs must use the same:

- dataset and split;
- backbone and input resolution;
- augmentation and orientation policy;
- optimizer and training-step budget;
- seed set;
- global/local loss settings;
- step time, images per second, and peak device memory.

Masking strategies are a second experiment axis. Loss comparisons should first
use a common real-masking strategy where applicable, then evaluate random,
blockwise, and hybrid masking separately.

Recommended explicit iBOT ablations are:

```text
ibot_prototypes: 512 / 2048 / 8192 (any integer >= 2; scale with dataset size)
lambda_mask:     0.25 / 0.5 / 1.0
```

The default remains `consistency` until repeated controlled tests show that
`masked-feature` improves the relevant global and local outcomes. iBOT remains
experimental regardless of a single successful run.

v0.7.0 does not claim a formal dense-representation improvement because the
repository has no validated dense morphology benchmark. Future evaluation may
add patch correspondence, part retrieval, keypoints, or linear segmentation
probes when reliable labels or protocols exist.

## 18. DINOv2 and DINOv3 Relationship

DINOv2 retains an iBOT-style patch objective: a masked student predicts an
unmasked teacher's prototype distribution at corresponding patch positions.
It supports separate image-level and patch-level heads and uses much larger
prototype dictionaries under large-scale training. Those prototype counts are
not transferred to OTU-Former; v0.7.0 uses a default of 512 for its smaller,
domain-specific datasets.

DINOv3 does not replace iBOT. It adds Gram anchoring to address dense feature
degradation observed during prolonged large-scale training while global metrics
continue to improve. OTU-Former should consider Gram anchoring only after it has
a dense metric and observes that failure mode. It is not part of v0.7.0.

## 19. Error Handling

The training command must fail early with actionable messages for:

- unsupported pretraining backbone;
- unsupported patch-loss or masking strategy;
- invalid `mask_ratio`;
- invalid explicit parameter combinations;
- real masking without a reliable patch grid or prefix-token count;
- resume configuration mismatch;
- missing or malformed `patch_objective` state in a v0.7.0 A+/iBOT checkpoint;
- optimizer state incompatible with the reconstructed training objective;
- attempted EVA/Swin/CNN pretraining or resume.

It must not silently:

- downgrade a masked objective to consistency;
- ignore an explicitly supplied inapplicable option;
- initialize missing patch heads during resume;
- mask CLS or other prefix tokens;
- drop positional information at masked positions.

## 20. Test Strategy

### 20.1 Mask generation

- random masks have exact counts and unique positions;
- blockwise masks obey geometry limits and exact final counts;
- hybrid masks combine blockwise and random positions and have exact counts;
- masks are deterministic under a fixed seed;
- samples and global views receive independently sampled masks;
- ratio bounds leave at least one selected and one visible patch;
- blockwise and hybrid positive tests use actual `(H, W)` grids where at least
  one integer rectangle satisfies every geometry constraint;
- a `4 x 4` grid fails fast because of the area limit, and an independently
  chosen narrow grid fails because no legal rectangle fits its actual height and
  width, instead of looping or degrading to random masking;
- the random-fill fallback is exercised only after at least one legal rectangle
  has been sampled;
- dynamic input sizes produce correct patch-grid dimensions.

### 20.2 Masked forward

- only patch content is replaced;
- CLS/prefix tokens are never masked;
- masked positions retain positional encoding;
- unsupported backbones fail before the training loop;
- A+ exposes the final four teacher block outputs from the same forward that
  produces final teacher tokens;
- each selected block output receives the specified token normalization before
  prefix removal and averaging;
- a spy/counter test proves A+ does not perform a second teacher backbone
  execution for a global view;
- real masking tests demonstrate that masked student input does not contain the
  original selected patch embeddings.

### 20.3 Losses

- consistency matches normalized cosine reference behavior;
- the old EVA MSE branch is absent;
- A+ target aggregation and predictor shape are correct;
- A+ and iBOT losses use masked positions only;
- iBOT probabilities and cross-entropy match a small reference calculation;
- patch-center updates use current pre-EMA teacher logits;
- empty/invalid masks are rejected rather than producing NaN or a silent zero.

### 20.4 EMA and optimizer

- encoder EMA matches parameters by name;
- student-only parameters are excluded from encoder EMA;
- teacher iBOT head follows student iBOT head through EMA;
- each patch mode constructs the expected optimizer parameter set;
- teacher and center state are excluded from the optimizer.

### 20.5 CLI and checkpoint compatibility

- all defaults and `auto` resolutions match this design;
- explicit invalid combinations fail;
- new-run consistency resolves to `0.30`;
- legacy consistency checkpoints inherit `0.50` when recorded or fall back to
  `0.50` when absent;
- v0.7.0 resume restores objective state and rejects mode changes;
- legacy resume cannot switch to A+ or iBOT;
- v0.7.0 RNG state round-trips;
- extract and other read-only consumers ignore patch-objective state;
- raw CLS and CLS-projector output dimensions remain unchanged.

### 20.6 Logging schema

- new runs create the complete v0.7.0 CSV schema;
- legacy resume atomically migrates the recognized old header and preserves
  historical rows with empty new fields;
- malformed or unknown schemas fail without modifying the original CSV;
- plotting accepts migrated historical rows;
- consistency writes `selected_patch_ratio` and leaves
  `actual_mask_ratio_*` empty;
- masked-feature and iBOT write actual masking ratios;
- iBOT-only diagnostics are empty for every other mode;
- version and CLI tests cover the schema and migration-facing help text.

### 20.7 Smoke and performance coverage

Run at least one optimizer step for each patch-loss mode using a small standard
ViT test configuration. A+ and iBOT must cover all three masking strategies
without downloading pretrained weights. Blockwise and hybrid smoke coverage
uses an actual `(H, W)` patch grid that passes the rectangle feasibility check;
separate negative tests cover both area-limited and narrow infeasible grids.

A representative default-shape benchmark records step time, images per second,
and peak device memory for `consistency`, `masked-feature`, and `ibot`. It is a
reported cost comparison, not a fixed performance gate.

## 21. File-level Implementation Direction

The subsequent implementation plan should keep the change narrow. Expected
areas are:

- `src/otuformer/cli/pretrain.py`: CLI options, parameter-source validation,
  help text, and effective config construction;
- `src/otuformer/training/model.py`: a single-pass supported-ViT training
  forward that can return final outputs and the final four block outputs for
  A+, without changing normal inference;
- `src/otuformer/training/trainer.py`: objective orchestration, named EMA,
  checkpoint/resume, diagnostics, and RNG state;
- `src/otuformer/training/loss.py`: one authoritative implementation per patch
  loss and removal of the currently disconnected duplicate patch-regression
  path;
- focused training/model/CLI/checkpoint tests.

No broad training-stack refactor is required. In particular, the new design
must not rewrite the existing global/local SSL objective.

## 22. Documentation and Version Updates After Approval

Implementation of the approved design will also update:

- `docs/superpowers/specs/2026-03-29-otuformer-design.md`;
- the relevant historical pretraining design/plan documents with a status note
  and link to this superseding v0.7.0 design, without rewriting their historical
  decisions;
- `ref/morphotu-research-roadmap.md`, correcting the current loss terminology
  and separating self-supervised masked feature prediction from supervised
  dense morphology;
- `README.md` and `README.cn.md`, including the first-use definition of `A+`
  and a prominent `--mask-ratio` `0.50 -> auto/0.30` migration note;
- package and runtime version declarations to `0.7.0`;
- version and CLI tests.

A new implementation plan will be written only after this design is approved.
Code, documentation updates beyond this draft, version changes, and commits are
outside the current review step.

## 23. Acceptance Criteria

The implementation is complete only when:

1. all four patch-loss modes run with their documented semantics;
2. consistency remains the default and new runs resolve its ratio to `0.30`;
3. all three masking strategies return exact, logged mask counts;
4. A+ and iBOT hide the selected student patch content while retaining position;
5. existing global/local losses use unmasked student views;
6. raw CLS and projector extraction contracts remain unchanged;
7. unsupported pretraining architectures fail before training;
8. v0.7.0 resume is strict and legacy consistency resume follows the documented
   `0.50` compatibility rule;
9. iBOT collapse diagnostics are present;
10. targeted and full test suites pass, including legacy CSV schema migration;
11. the representative benchmark reports step time, throughput, and peak memory
    for the three active patch objectives;
12. README, roadmap, design/plan references, migration notes, and version
    declarations describe the implemented behavior accurately;
13. no claim of dense anatomical understanding is made without a corresponding
    benchmark.

## 24. Primary References

- iBOT: Image BERT Pre-Training with Online Tokenizer:
  https://arxiv.org/abs/2111.07832
- DINOv2: Learning Robust Visual Features without Supervision:
  https://arxiv.org/abs/2304.07193
- DINOv2 official iBOT patch loss:
  https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py
- DINOv2 default SSL configuration:
  https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/ssl_default_config.yaml
- DINOv3:
  https://arxiv.org/abs/2508.10104
