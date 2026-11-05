# OTU-Former Coordinate Consistency — Temporary Design Notes

Date: 2026-10-31
Status: Discussion checkpoint; confirmed boundaries and open decisions, not an implementation-ready design.
Authorization: the operator requested this temporary record. Writing it does not authorize implementation, a commit, push, or release. Resume Coordinate design after the separate CLS migration discussion.

## Intent and confirmed boundaries

Develop local morphological representations by supervising corresponding locations in different augmented views of the same image. Local capability is valuable independently of global barcode or clustering improvements. Cross-specimen matching and anatomical homology are outside this first stage.

- Coordinate is disabled by default and has its own loss weight.
- It may accompany exactly one existing `--patch-loss` mode: `none`, `consistency`, `masked-feature`, or `ibot`. These existing modes remain mutually exclusive; do not combine all of them.
- Do not reuse `mask-ratio` to control Coordinate sampling or weighting.
- The teacher target is detached. The exact student feature source and its interaction with input masking still need a decision.
- Preserve the existing augmentation profiles, defaults, orientation behavior, and transform order. Support the spatial transforms already offered by the relevant pretrain/finetune augmentation paths; a crop/resize-only delivery would not satisfy the agreed coverage. This is a geometry-coverage requirement, not authorization to add Coordinate training to finetuning.
- Coordinate does not automatically enable transforms. Color, grayscale, and blur do not require coordinate changes. No new perspective or elastic augmentation is requested.
- Ranking is deferred. Do not build a general loss-composition framework in anticipation of it.
- Dataset splitting remains external, using entomokit or other preparation tools. Roadmap section 4.2 is not a Coordinate implementation gate.

## Existing implementation to reuse

`src/otuformer/training/model.py` already exposes patch tokens and grid information through `PretrainForward` / `forward_pretrain()`. Extend the existing path rather than replacing the backbone interface.

The existing global/local CLS path uses unmasked student views. In masked-feature/ibot modes, additional masked global forwards serve only patch loss; their CLS outputs are not reused. Any later Coordinate masking decision must preserve or explicitly revise that boundary.

The local student forward currently produces patch tokens that its caller discards. These may be reused if local-to-global Coordinate pairs are selected. This does not establish that masked/unmasked features, loss computation, backward cost, or memory use are free. No teacher-local forward has been requested.

The existing spatial operations include crop/resize/center crop, rotations with expansion or truncation, horizontal/vertical flips, padding, and affine translation/scaling. Correspondence must account for the actual sampled operations and their order, including invalid or padded areas.

## Prior art checked during discussion

| Reference | Spatial mechanism | Learning target | Consequence for this design |
|---|---|---|---|
| NeCo | Record crop boxes, map overlapping regions, ROI-Align both feature maps to an aligned grid. The checked default implementation uses crop/resize and appearance perturbations. | Consistent neighbor rankings relative to reference patches. | A direct reference for dense teacher-global to student-other-global/local pairing, but not evidence that a simpler cosine regression is equivalent. |
| DenseDINO | Sample shared positions in crop overlaps, convert them to view coordinates, explicitly correct horizontal flips, and construct reference tokens. | Teacher/student head-distribution distillation at reference positions. | A direct reference for explicit cross-view spatial correspondence; faithfully reproducing it includes its reference-token mechanism. |
| DINOv2 / iBOT | Masked student and unmasked teacher share the same global crop geometry. | Same-position masked patch distribution prediction. | Does not by itself provide arbitrary cross-crop Coordinate alignment. |
| DINOv3 | Retains iBOT; Gram anchoring uses shared crop geometry and aligns feature-grid resolution when needed. | Patch-to-patch Gram similarity matching. | Gram anchoring is not arbitrary cross-view pointwise feature regression. |

These methods simplify geometry by controlling their view generation. Crop boxes alone cannot recover correspondence after independent view rotations, flips, translations, or padding. Keeping OTU-Former's existing augmentation behavior therefore requires tracking those relative geometric changes.

A proposed implementation direction is composed affine coordinate mappings, standard feature sampling such as `grid_sample`, and valid-region handling. This is a proposal, not an approved coordinate convention or sampling algorithm. Pixel centers, patch centers, interpolation, transform inversion, and boundary rules require explicit design and checks.

Sources:

- NeCo: https://arxiv.org/abs/2408.11054 ; https://github.com/vpariza/NeCo/blob/main/src/neco.py ; https://github.com/vpariza/NeCo/blob/main/src/neco_transforms.py
- DenseDINO: https://arxiv.org/abs/2306.04654 ; https://github.com/yyk-wew/DenseDINO/blob/ref_pos/main_dino.py ; https://github.com/yyk-wew/DenseDINO/blob/ref_pos/vision_transformer.py
- DINOv2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py
- DINOv3: https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/ssl_meta_arch.py ; https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py

## Open decisions — do not treat as approved

1. **Exact objective.** Direct normalized-feature regression, DenseDINO reference-token distillation, and NeCo ranking are different objectives. Prefer mature mechanisms, but no complete method or formula has been selected. NeCo-style ranking remains deferred under the current scope.
2. **View pairs.** Teacher globals against student other-global and local views now have direct dense-method precedent and are the leading proposal. Global-only was an earlier premature recommendation, not a settled requirement. Select the actual pairs and treatment of same-view pairs explicitly.
3. **Sampling and weighting.** Number of points/grid resolution, minimum usable overlap, empty-overlap behavior, valid-area masks, pair normalization, and interpolation are not fixed. Avoid rewarding padding or heavily distorted overlap.
4. **Masked patch interaction.** Coordinate may accompany an existing patch mode, but whether it uses unmasked features, excludes masked positions, or needs another forward remains unresolved.
5. **Geometry contract.** Specify forward/inverse maps and conventions for all existing spatial transforms, without silently changing profile behavior.
6. **Acceptance thresholds and runtime cost.** Define matching metrics and practical memory/time measurements before implementation; no biological benefit is claimed from loss reduction alone.

## Validation direction

Validate geometry and learned representation separately:

- Synthetic images and known transforms should establish correct corresponding positions, round trips, and rejection of padding/out-of-view locations.
- Held-out same-image cross-augmentation retrieval or correspondence should measure whether features themselves identify matching regions. Geometric alignment alone is not proof of learned matching.
- No cross-specimen success criterion is required in this stage. Global barcode/clustering improvement is not a required substitute for local evaluation.
- Existing external COCO part polygons may support part-stratified diagnostics, but they are not point-correspondence ground truth. Missing part annotations must not be interpreted as biological absence.
- Real datasets remain read-only; portable checks use synthetic fixtures. Do not add a duplicate split tool or require completion of unrelated roadmap modules.

## Sequence and later integration

The operator has now prioritized a separate CLS migration: align global cross-view pairing and DINO head-distribution distillation, without requiring an exact reproduction of the entire DINO training recipe. The current CLS objective is batch-relative similarity-distribution distillation, and its default global pairs include same-view pairs. This finding motivates that separate work; it does not select a Coordinate objective.

Resume this document after CLS design is settled. Before Coordinate implementation, resolve the open decisions, review a complete design, obtain approval, and prepare an implementation plan. A later implementation should review the master design links, English/Chinese README and command documentation, shared augmentation documentation, and `skills/otuformer-workflow/` references for affected behavior. None of those integrations is performed by this temporary note.
