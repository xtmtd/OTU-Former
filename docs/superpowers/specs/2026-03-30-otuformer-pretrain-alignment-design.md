# OTU-Former Pretrain Alignment Design

Date: 2026-03-30

## Goal

Align the current `otuformer pretrain` implementation with the reference SSL workflow in `ref/ibot20260115.py`, with scope limited to pretraining behavior, periodic embedding evaluation, and pretraining-related plots. The objective is to reduce behavioral drift from the reference implementation while preserving the current package and CLI structure.

## Scope

In scope:

- `src/otuformer/training/trainer.py`
- `src/otuformer/embedding/evaluator.py`
- `src/otuformer/cli/pretrain.py` if a small interface adjustment is needed
- `src/otuformer/training/model.py` only if needed to align projector or center behavior

Allowed plot/report changes in this pass:

- UMAP title and legend-label semantics for periodic pretrain evaluation
- color selection for the `Structure Quality` panel in the pretrain training-curves PDF if touched incidentally, but this is not a required acceptance item for the algorithm-alignment work

Explicitly excluded even if adjacent code is touched:

- renaming unrelated metrics
- redesigning plot layouts
- cleanup of non-pretrain evaluator/reporting behavior
- adding new evaluation metrics

Out of scope:

- ArcFace finetuning behavior
- OTU clustering, partition scanning, tree building, annotation, and diversity modules
- CAM / heatmap generation
- Broad refactors outside the pretrain path

## Problem Statement

The current package already exposes a working `pretrain` command, but observed outputs differ materially from the reference workflow.

Observed differences from the current investigation:

- Current SSL metrics in `runs/pretrain/logs/metrics.pretrain.csv` underperform the reference log for the same dataset, especially in `kNN_Acc`, `Linear_Probing_Acc`, and `Silhouette_Score`.
- Current plotting code hardcodes presentation semantics in places where the reference behavior is context-dependent.
- UMAP output carries unnecessary split wording in labels in some train-only contexts.
- The `Structure Quality` panel in the training curves uses colors that are too similar for the paired metrics.

This suggests a combination of training-path drift and visualization/output drift, not a single isolated bug.

## Root Cause Hypothesis

The mismatch is most likely caused by three layers of drift.

For this spec, the authoritative reference behavior is defined by:

- algorithm and output semantics in `ref/ibot20260115.py`
- reference runtime artifacts in `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/`
- current comparison target in `runs/pretrain/logs/`

Config parity for comparison means matching at least the following on the example run:

- dataset identity (`examples/Epidorcus/figs.csv` mapped to the same sample set as the reference Epidorcus run)
- backbone family: `vit_tiny_patch16_224`
- output dimension: `256`
- epoch count: `50`
- global/local crop sizes: `224` / `96`
- local crop count: `6`
- student temperature: `0.1`
- teacher temperature endpoints: `0.04 -> 0.07`
- teacher temperature warmup span: first `70%` of total training iterations, then constant
- EMA momentum endpoints: `0.995 -> 0.999`
- `lambda_local=1.5`, `lambda_mask=1.0`, `mask_ratio=0.5`
- batch size: `32`
- save cadence used for comparison artifacts: every `10` epochs

Rationale: if these parameters differ, output differences are expected and cannot be attributed to training-path drift. For this work, parameter consistency is a prerequisite for meaningful comparison, not an optional optimization target.

### 1. Training loop drift

The current SSL loop implements the same major ingredients as the reference workflow, but the code paths are no longer a close behavioral mirror. The likely drift points are:

- how teacher outputs are centered before distillation
- exact placement and update timing of `teacher.center`
- how global, local, and masked-token losses are combined
- how teacher temperature and EMA momentum schedules are consumed during iteration
- whether embeddings and patch tokens are normalized in the same places as the reference

### 2. Evaluation drift

The current package routes periodic metrics through `src/otuformer/embedding/evaluator.py`, while the reference script keeps more of the evaluation and plotting assumptions inline. That separation is desirable, but it means the current evaluator may not be reproducing the same semantics the training loop expects.

### 3. Plot semantics drift

The current plotting functions encode some train/test wording and color choices too rigidly. This is why the outputs look unlike the reference even when the underlying data is valid.

## Design Principles

- Preserve the existing package structure and CLI surface unless a narrow interface change is required.
- Prefer behavioral alignment over mechanical code copying.
- Keep modifications local to the pretrain path.
- Treat visualization fixes as part of evaluation output correctness, not as separate cosmetic work.
- Avoid unrelated cleanup while touching pretraining code.

## Approach Options Considered

### Option 1: Full reference code transplant

Copy the reference pretrain and evaluation logic into the package nearly verbatim.

Pros:

- Maximum short-term fidelity to the reference script.

Cons:

- Reintroduces monolithic script structure into the package.
- Increases maintenance cost and duplicates logic already split across package modules.
- Higher risk of breaking the existing CLI/package boundaries.

### Option 2: Incremental alignment inside the current package structure

Keep `trainer.py` and `evaluator.py` as the package boundaries, then align their behavior point-by-point with the reference workflow.

Pros:

- Smallest correct change.
- Preserves the toolbox architecture already built.
- Makes it easier to verify drift item by item.

Cons:

- Requires careful comparison work to avoid missing a subtle behavioral difference.

### Option 3: Visualization-only cleanup first

Fix UMAP labels and training-curve colors now, defer SSL alignment.

Pros:

- Fastest visible improvement.

Cons:

- Does not address the main problem, which is metric drift from the reference implementation.

## Recommended Approach

Choose Option 2.

This provides the best balance of fidelity and maintainability. The package structure is already largely correct; the issue is that parts of the pretrain path are not reproducing the reference behavior closely enough. The work should therefore focus on point-by-point behavioral alignment, not a rewrite.

## Target Design

### A. Training-path alignment in `trainer.py`

The pretraining loop remains in `src/otuformer/training/trainer.py`, but it should become a closer behavioral mirror of `ref/ibot20260115.py` for SSL-only execution.

Planned alignment points:

- Teacher-temperature schedule must follow the exact reference rule seen in `ref/ibot20260115.py:1688-1693`: build an array of length `total_iters`, linearly interpolate from `teacher_temp_start` to `teacher_temp_end` over `int(total_iters * 0.7)` iterations, then fill the remainder with `teacher_temp_end`. The schedule is indexed by the global iteration counter used inside the training loop.
- EMA momentum schedule must follow the exact reference rule in `ref/ibot20260115.py:1695-1700` and `1781-1782`: cosine schedule from `teacher_momentum` to `teacher_momentum_end`, indexed by the same global iteration counter, applied immediately after `optimizer.step()`.
- Teacher-center behavior must follow `ref/ibot20260115.py:1737-1746`: for each global teacher view, compute `proj, tokens = teacher(view)`, then apply `proj = proj - teacher.center`, then `proj = F.normalize(proj, dim=-1)`. After both teacher global views are processed, concatenate the centered-and-normalized teacher projector outputs and update the center buffer as `center = center * 0.9 + mean(all_teacher_global) * 0.1`.
- Global distillation must follow `ref/ibot20260115.py:1751-1760`: if `disable_cross_view_loss` is false, average SSL loss over the full Cartesian product of `student_global x teacher_global`; otherwise average only matching global-view pairs in zip order.
- Local-to-global distillation must follow `ref/ibot20260115.py:1767-1773`: for every local student crop, compute one student projector output and average SSL loss over the Cartesian product of `local_student x teacher_global`.
- Masked-token regression must follow `ref/ibot20260115.py:1441-1471` and `1762-1765`: for each `(student_tokens, teacher_tokens)` pair from the two global views, sample masked positions independently by sorting per-sample random values and selecting the first `max(1, int(mask_ratio * N))` token indices; gather those positions from both tensors; if `"eva" in model_name.lower()` use `F.mse_loss(student_masked, teacher_masked)`, otherwise L2-normalize both and use `mean(2 - 2 * dot(student_masked, teacher_masked))`.
- Periodic evaluation during pretraining must use the same representation choice as the reference pretrain path: the normalized projector output returned by the SSL model for each image, not raw CLS features, patch-token aggregates, or any finetune-specific head output.

The design intent is to preserve the current module boundary while making the numerical path behave like the reference.

Decision on file changes:

- `trainer.py` is expected to carry almost all behavioral changes.
- `trainer.py` must be able to obtain from the encoder a stable two-value return contract for pretraining: `(normalized_projector_output, patch_tokens)`, where `normalized_projector_output` is the SSL projector output used for distillation and periodic evaluation, and `patch_tokens` are the patch embeddings used by masked-token regression. If the current encoder already guarantees that contract, keep the change in `trainer.py`; if not, tighten `model.py` so that this contract is explicit and shared by all pretrain call sites.
- `cli/pretrain.py` should only change if an already-supported reference behavior cannot be expressed with current CLI arguments. No new user-facing options are planned in this pass.

### B. Evaluation-path alignment in `evaluator.py`

`src/otuformer/embedding/evaluator.py` remains the central evaluation helper module, but its interfaces need to become more explicit so the training loop can drive reference-like behavior.

Planned changes:

- Make UMAP plotting metadata caller-controlled rather than hardcoded.
- Ensure train-only evaluations do not inject unnecessary split suffixes into labels or titles.
- Keep metric computation centralized, but verify metric naming and value mapping used by `trainer.py` against the reference logs.

Concrete evaluator decisions:

- `run_umap()` should accept explicit title/split-display context from the caller.
- When only one split is plotted, legend labels must be raw class labels with no appended split marker.
- When multiple splits are plotted in the future, split suffixes are permitted only when they disambiguate otherwise identical labels.
- No broader evaluator cleanup should be bundled into this pass beyond what is required to support pretrain alignment and the two named plot fixes.

This keeps the modular design while removing hidden display assumptions.

### C. Plot output alignment

Plotting changes remain localized to the existing training-curve generation and UMAP helper.

Planned changes:

- Remove unnecessary `(train)` suffixes when the plot is already unambiguously train-only.
- Use clearly separated colors in the `Structure Quality` panel.
- Keep train/test suffixes only when both splits are actually present in the plotted data.

Concrete plotting decisions:

- The `Structure Quality` panel should use two clearly distinguishable colors from the default matplotlib tableau palette, e.g. `tab:green` and `tab:red` or an equivalently separated pair.
- The training-curves layout, subplot count, and metric grouping remain unchanged.

These are small but important output-correctness fixes because the current visuals make the outputs harder to interpret and diverge from the reference style.

## File-Level Responsibilities

### `src/otuformer/training/trainer.py`

Primary responsibility after this change:

- own the SSL training loop
- own schedule consumption
- own periodic evaluation orchestration
- pass correct plot metadata to evaluator helpers

### `src/otuformer/embedding/evaluator.py`

Primary responsibility after this change:

- compute embedding metrics
- render UMAP from explicit caller-provided context
- avoid hardcoded split/title semantics that belong to the caller

### `src/otuformer/cli/pretrain.py`

Only changed if the current CLI does not expose enough context to support the aligned behavior. Any change here should be minimal and backwards-compatible at the CLI level unless a concrete mismatch requires otherwise.

### `src/otuformer/training/model.py`

Only changed if projector output or center-related behavior cannot be aligned from the trainer alone.

## Verification Plan

Verification should use the existing example workflow already used for local testing.

Primary command:

`otuformer pretrain --train-data examples/Epidorcus/figs.csv --input-images-dir examples/Epidorcus/images --max-epochs 50 --log-every-n-steps 50 --save-every-epochs 10`

Expected parameter values for the verification run are the current CLI defaults recorded in `runs/pretrain/logs/pretrain.log` unless explicitly overridden by the command above. The source of truth for these defaults is `src/otuformer/cli/pretrain.py`.

Reference artifacts for comparison:

- Current run under test: `runs/pretrain/logs/pretrain.log`, `runs/pretrain/logs/metrics.pretrain.csv`, `runs/pretrain/logs/umap.train.epoch_0050.pdf`, `runs/pretrain/logs/training_curves_pretrain.pdf`
- Reference run: `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/log.txt`, `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/metrics.pretrain.csv`, `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/umap.train.epoch_0050.pdf`, `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/training_curves_pretrain.pdf`

Dataset equivalence assumption for this spec: `examples/Epidorcus/figs.csv` is treated as fully equivalent to the sample set used by the reference Epidorcus run. This has been confirmed by the user and does not need further auditing in this pass.

Comparison procedure:

1. Confirm the generated pretrain command parameters still match the intended reference parity fields listed earlier.
2. Compare epoch-50 values in `metrics.pretrain.csv` for `Recall@1`, `kNN_Acc_k1`, `kNN_Acc_k5`, `kNN_Acc_k20`, `Linear_Probing_Acc`, `mAP`, and `Silhouette_Score`.
3. Compare trend direction across epochs 10, 20, 30, 40, 50 for those same metrics, not just the final snapshot.
4. Inspect the epoch-50 UMAP PDF for label text only; confirm no unnecessary `(train)` suffixes remain.
5. Inspect `training_curves_pretrain.pdf` and confirm the two `Structure Quality` lines are visually distinct without zooming.
6. Directly verify implementation claims in code/log behavior:
   - teacher temperature schedule uses `70%` warmup then constant tail
   - momentum schedule is applied after `optimizer.step()`
   - center update occurs after teacher global outputs are concatenated for the current iteration
   - global/local loss averaging follows the Cartesian-product rules above
   - periodic evaluation extracts projector outputs, not raw CLS tokens
7. Separate training-path alignment from evaluator-only changes during verification:
   - first inspect code changes to confirm whether metric computation formulas changed
   - if metric formulas changed, record that explicitly in the verification summary
   - do not attribute improved metrics to training alignment unless the training-path checks in step 6 also pass

Current baseline for directional numeric comparison, taken from `runs/pretrain/logs/metrics.pretrain.csv` before this change:

- epoch 50 `Recall@1 = 0.9565`
- epoch 50 `kNN_Acc_k1 = 0.7870`
- epoch 50 `kNN_Acc_k5 = 0.7957`
- epoch 50 `kNN_Acc_k20 = 0.7739`
- epoch 50 `Linear_Probing_Acc = 0.8348`
- epoch 50 `mAP = 0.6973`
- epoch 50 `Silhouette_Score = 0.1224`

Reference epoch-50 values from `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/metrics.pretrain.csv`:

- epoch 50 `Recall@1 = 0.9826`
- epoch 50 `kNN_Acc_k1 = 0.8826`
- epoch 50 `kNN_Acc_k5 = 0.9043`
- epoch 50 `kNN_Acc_k20 = 0.9304`
- epoch 50 `Linear_Probing_Acc = 0.9652`
- epoch 50 `mAP = 0.7738`
- epoch 50 `Silhouette_Score = 0.2217`

Success signals:

- All five direct implementation checks in step 6 pass by code inspection or log inspection.
- Any evaluator-side metric changes are either absent or explicitly called out in the verification summary so they are not confused with training-path improvement.
- `runs/pretrain/logs/metrics.pretrain.csv` trends move closer to the reference log for the same dataset, with no epoch-50 regression in more than one of the seven tracked numeric metrics above relative to the current baseline.
- At least four of the seven tracked numeric metrics above reduce the absolute gap to the reference at epoch 50 by any positive amount relative to the current baseline.
- `runs/pretrain/logs/umap.train.epoch_*.pdf` no longer include unnecessary label noise.

Acceptance is directional, not exact-bitwise identity, because runtime environment differences may still affect numeric results.

## Risks

- Small numeric differences may remain due to device/runtime differences between the current environment and the original reference environment.
- Aligning only part of the loss path could create the appearance of improvement while leaving hidden drift behind; this is why the work must compare the full SSL path rather than patching isolated symptoms.
- Evaluation changes can alter how metrics are reported even if training is unchanged, so code review and verification must distinguish behavioral improvement from reporting-only changes.

## Testing Strategy

- Re-run the example pretrain workflow after changes.
- Compare generated metrics CSV and log summaries against the reference run.
- Inspect the generated UMAP and training-curves PDFs for the two known plotting issues.
- Confirm that the CLI still works without changes to existing example commands.

## Non-Goals

- Perfect numeric identity with the reference run under all environments.
- Unifying all training and evaluation logic across pretrain and finetune in this pass.
- General visual redesign of reports or figures.

## Implementation Handoff

After spec approval, the implementation plan should break the work into small tasks that:

- isolate training-loop alignment from evaluator/plot changes
- include targeted verification after each change group
- avoid touching OTU/delineation/diversity paths in the same pass
