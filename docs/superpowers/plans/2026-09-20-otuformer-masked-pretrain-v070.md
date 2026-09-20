# OTU-Former v0.7.0 Masked Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not delegate unless the operator explicitly authorizes delegation.

**Goal:** Preserve masked-position patch consistency as the default while adding true masked-feature prediction, experimental lightweight iBOT patch distillation, and random/blockwise/hybrid masks with strict resume and extraction compatibility.

**Architecture:** Keep the existing unmasked global/local SSL path unchanged. Add a standard-timm-ViT-only training forward that can replace patch embeddings before positional encoding and, when requested, return the final four block outputs from the same backbone execution. A separate training-only patch-objective module owns the mask token, masked-feature predictor, iBOT heads, and patch center; checkpoints store that state outside the general encoder so read-only consumers remain unchanged.

**Tech Stack:** Python 3.11, PyTorch, timm, Typer/Click, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-otuformer-masked-pretrain-v070-design.md`

## Global Constraints

- `consistency` remains the default patch loss; `ibot` remains experimental.
- Patch-loss modes are exactly `none`, `consistency`, `masked-feature`, and `ibot`, and are mutually exclusive.
- New-run resolved ratios are `consistency=0.30`, `random=0.30`, `blockwise=0.20`, and `hybrid=0.30`.
- Legacy checkpoints without patch metadata remain `consistency` and inherit a saved ratio or fall back to `0.50`.
- Existing unmasked global/local SSL behavior and extraction/fine-tune/CAM/export interfaces must not change.
- Pretraining and pretraining resume support standard timm ViT backbones only. Remove the EVA-specific patch-loss branch; do not add EVA/Swin/CNN adapters.
- Do not add dynamic mask schedules, foreground heuristics, Sinkhorn-Knopp, combined A+/iBOT mode, or patch-specific optimizer groups.
- Use one teacher backbone execution per global view. A+ must obtain final outputs and final-four block outputs from that same execution.
- Do not modify or track `.codegraph/`.
- Do not create a commit until the user has reviewed the implementation, documentation, test evidence, and benchmark results.

## File Map

**Core implementation**

- Modify: `src/otuformer/cli/pretrain.py`
  - Add patch-loss options, parse `auto|FLOAT`, retain Click parameter-source information, reject explicitly inapplicable combinations, and update help text.
- Modify: `src/otuformer/training/model.py`
  - Add the standard-ViT masked training forward and the training-only patch-objective modules without changing normal `forward()` behavior.
- Modify: `src/otuformer/training/loss.py`
  - Replace the disconnected `MaskedPatchRegressionLoss` with authoritative cosine and iBOT patch losses.
- Modify: `src/otuformer/training/trainer.py`
  - Resolve patch configuration, generate masks, orchestrate objectives, perform name-matched EMA, handle optimizer/checkpoint/RNG state, migrate logging, and emit diagnostics.

**Tests**

- Create: `tests/training/test_masked_pretrain.py`
  - Focused mask generation, masked forward, patch objective, EMA, checkpoint, logging, and per-mode smoke coverage.
- Modify: `tests/test_training_model.py`
  - Preserve normal inference and extraction contracts; cover supported-ViT capability checks and masked-forward token semantics.
- Modify: `tests/test_training_loss.py`
  - Cover authoritative cosine and iBOT reference calculations and invalid masks.
- Modify: `tests/training/test_pretrain_alignment.py`
  - Remove the stale EVA branch expectation and retain global/local alignment regression coverage.
- Modify: `tests/test_cli_smoke.py`
  - Cover CLI defaults, explicit-option validation, help/migration text, and forwarding.
- Modify: `tests/test_checkpoint_utils.py`
  - Prove read-only general-encoder loading ignores `patch_objective` state.
- Modify: `tests/test_version.py`
  - Require `0.7.0` in both version declarations.

**Documentation and release metadata**

- Modify: `README.md`
- Modify: `README.cn.md`
- Modify: `docs/superpowers/specs/2026-03-29-otuformer-design.md`
- Modify: `docs/superpowers/specs/2026-03-30-otuformer-pretrain-alignment-design.md`
- Modify: `docs/superpowers/plans/2026-03-30-pretrain-alignment.md`
- Modify: `docs/superpowers/specs/2026-09-20-otuformer-masked-pretrain-v070-design.md`
- Modify: `ref/morphotu-research-roadmap.md`
- Modify: `pyproject.toml`
- Modify: `src/otuformer/__init__.py`

---

### Task 1: Define and validate the effective patch configuration

**Files:**
- Modify: `src/otuformer/cli/pretrain.py`
- Modify: `src/otuformer/training/trainer.py`
- Modify: `tests/test_cli_smoke.py`
- Create: `tests/training/test_masked_pretrain.py`

**Interfaces:**
- Produces CLI values: `patch_loss`, `masking_strategy`, `mask_ratio_requested`, `ibot_prototypes`, and parameter-source flags.
- Produces trainer helper: `_resolve_patch_config(args, checkpoint) -> dict[str, object]`.
- The returned dict contains the exact checkpoint fields from design Section 13.

- [ ] **Step 1: Write failing CLI default and forwarding tests**

Add tests asserting a new invocation forwards:

```python
assert seen["patch_loss"] == "consistency"
assert seen["masking_strategy"] == "random"
assert seen["mask_ratio"] == "auto"
assert seen["ibot_prototypes"] == 512
```

Also assert the namespace records whether each of `--mask-ratio`, `--masking-strategy`, `--ibot-prototypes`, and `--lambda-mask` came from the command line. Use `ctx.get_parameter_source(...) is click.core.ParameterSource.COMMANDLINE`; do not infer explicitness by comparing values with defaults.

- [ ] **Step 2: Run the new CLI test and verify it fails**

Run:

```bash
pytest tests/test_cli_smoke.py -k "patch_loss or mask_ratio_auto" -q
```

Expected: FAIL because the new options and explicit-source flags do not exist.

- [ ] **Step 3: Add the minimum CLI surface**

Add:

```text
--patch-loss none|consistency|masked-feature|ibot       default consistency
--masking-strategy random|blockwise|hybrid              default random
--mask-ratio auto|FLOAT                                 default auto
--ibot-prototypes INT                                default 512 (any integer >= 2)
```

Keep `--lambda-mask 1.0`. Validate enum values and explicit float syntax before `prepare_output_dir()` so invalid invocations do not create or clear output directories. Pass the original requested ratio (`"auto"` or a float) and explicit-source booleans to `run_pretrain()`.

Update command help to say:

- `consistency` is the default and selects visible same-position patches;
- `masked-feature` is A+, meaning continuous masked feature prediction;
- `ibot` is experimental;
- v0.7.0 changes a new consistency run from `0.50` to `auto`, resolved as `0.30`.

- [ ] **Step 4: Write failing effective-config tests**

Cover every row of the default table and every invalid explicit combination:

```python
@pytest.mark.parametrize(
    ("mode", "strategy", "expected"),
    [
        ("none", "random", None),
        ("consistency", "random", 0.30),
        ("masked-feature", "random", 0.30),
        ("masked-feature", "blockwise", 0.20),
        ("masked-feature", "hybrid", 0.30),
        ("ibot", "random", 0.30),
        ("ibot", "blockwise", 0.20),
        ("ibot", "hybrid", 0.30),
    ],
)
def test_patch_config_resolves_new_run_defaults(mode, strategy, expected):
    ...
```

Reject explicit ratios outside `0 < ratio < 1`, and reject explicitly inapplicable options exactly as specified in design Section 10.3. Defaults that are inapplicable must resolve to `None` without error.

- [ ] **Step 5: Implement `_resolve_patch_config()`**

Keep resolution in one trainer helper so startup, checkpoint comparison, logging, and tests use the same effective values. Return:

```python
{
    "patch_loss": mode,
    "masking_strategy": strategy_or_none,
    "mask_ratio_requested": requested_or_none,
    "mask_ratio_resolved": resolved_or_none,
    "ibot_prototypes": prototypes_or_none,
    "patch_target_layers": 4 if mode == "masked-feature" else None,
    "patch_center_momentum": 0.9 if mode == "ibot" else None,
}
```

Do not construct mask modules in this task.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/test_cli_smoke.py -k "patch_loss or mask_ratio or ibot_prototypes" -q
pytest tests/training/test_masked_pretrain.py -k patch_config -q
```

Expected: PASS.

### Task 2: Implement exact random, blockwise, and hybrid masks

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Modify: `tests/training/test_masked_pretrain.py`

**Interfaces:**
- Produces: `_validate_blockwise_grid(grid_size: tuple[int, int]) -> None`, where `grid_size` is `(H, W)`, never the scalar patch count `N`.
- Produces: `_sample_patch_masks(batch_size, grid_size: tuple[int, int], ratio, strategy, device) -> torch.BoolTensor` with shape `[B, H * W]`.
- Uses the existing PyTorch RNG; no separate generator or mask seed is added.

- [ ] **Step 1: Write failing exact-count and determinism tests**

For all three strategies, assert:

- `target_count = clamp(round(ratio * N), 1, N - 1)`;
- every row has exactly `target_count` true entries;
- no sample contains duplicate positions;
- resetting `torch.manual_seed()` reproduces the mask;
- successive calls for two global views do not reuse one mask object or one sampled pattern;
- both `8 x 8` and a second valid non-square grid map to `H * W` positions.

- [ ] **Step 2: Write failing blockwise geometry tests**

Assert legal proposals have side lengths at least two, aspect ratio in `[0.5, 2.0]`, fit the actual `(H, W)`, and do not exceed `floor(0.20 * N)`.

Add explicit fail-fast tests for:

```text
4 x 4       invalid because floor(0.20 * 16) == 3 < 4, the minimum 2 x 2 block area
1 x 32      invalid because no two-patch-high rectangle fits
```

Use a valid rectangular grid for positive tests. Monkeypatch/capture the proposal sampler as needed to prove random fill occurs only after at least one legal rectangle has been sampled.

- [ ] **Step 3: Run the mask tests and verify they fail**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "mask_generation or blockwise or hybrid" -q
```

Expected: FAIL because the helpers do not exist.

- [ ] **Step 4: Implement the samplers without a new source module**

Keep small private helpers in `trainer.py`:

```python
def _target_patch_count(ratio: float, patch_count: int) -> int:
    return min(max(round(ratio * patch_count), 1), patch_count - 1)
```

Use `torch.randperm()` for random sampling. For blockwise sampling:

1. enumerate legal integer `(height, width)` pairs against the actual grid;
2. fail immediately if the list is empty;
3. sample two to four rectangle proposals and legal top-left positions;
4. merge overlaps;
5. continue until at least the requested count is available;
6. random-fill only after a legal rectangle exists when still under target;
7. randomly trim excess from the final proposal;
8. return exactly the target count.

Hybrid requests exactly `floor(target_count / 2)` positions from the blockwise path, then fills the remainder uniformly from still-unmasked positions. A legal block may be trimmed when the requested blockwise share is smaller than four.

- [ ] **Step 5: Run the focused mask tests**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "mask_generation or blockwise or hybrid" -q
```

Expected: PASS without retry loops or fallback to a different strategy.

### Task 3: Add a single-pass standard-ViT masked training forward

**Files:**
- Modify: `src/otuformer/training/model.py`
- Modify: `tests/test_training_model.py`
- Modify: `tests/training/test_masked_pretrain.py`

**Interfaces:**
- Preserves: `OTUFormerEncoder.forward(x)` and `return_patch_tokens` behavior.
- Produces: `OTUFormerEncoder.validate_pretrain_backbone(require_intermediates: bool) -> None`.
- Produces: a training-only forward returning projected CLS, final tokens, actual grid size, and optionally the last four pre-final-normalization block outputs.

- [ ] **Step 1: Write failing normal-path regression tests**

Capture current contracts before adding the training path:

- plain `forward()` returns the normalized CLS projector output;
- `return_patch_tokens=True` still returns `(projected_cls, patch_tokens)`;
- raw CLS and projector output dimensions are unchanged;
- patch token counts remain correct at two supported input sizes.

- [ ] **Step 2: Write failing capability tests**

A supported timm ViT must expose the required patch embedding, `_pos_embed`, prefix-token count, transformer blocks, pre-normalization/final normalization, and token sequence. A CNN, Swin, EVA-specific path, or a test double missing any required capability must fail with a message naming the model and missing capability before the training loop.

A+ additionally requires at least four transformer blocks.

- [ ] **Step 3: Write failing masked-forward semantic tests**

Using a tiny deterministic ViT/test double, prove:

- only selected patch embeddings are replaced;
- CLS/distillation/register/prefix tokens are never replaced;
- masked positions retain their own positional encoding;
- flattened indices map to the correct `(row, column)` positions;
- original selected patch content cannot reach the transformer input;
- final-four block outputs and final tokens come from one patch-embed/backbone execution;
- each selected teacher block output receives `backbone.norm` independently before prefix removal and averaging.

Use a call counter to assert one teacher execution per global view.

- [ ] **Step 4: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_training_model.py tests/training/test_masked_pretrain.py -k "pretrain_backbone or masked_forward or final_four or inference_contract" -q
```

Expected: FAIL because the training-only forward does not exist.

- [ ] **Step 5: Implement the supported-ViT path**

Follow timm VisionTransformer's native sequence once:

```text
patch_embed
-> replace selected flattened patch embeddings with mask_token
-> _pos_embed (prefix + interpolated position)
-> patch_drop
-> norm_pre
-> blocks (capture final four when requested)
-> final norm
```

Handle timm's dynamic patch embedding output without guessing square grids: preserve `(H, W)` before flattening and return it. Use `backbone.num_prefix_tokens` when slicing final patch tokens. Never hard-code a one-token CLS offset.

Keep the existing `forward()` untouched and call the new method only from pretraining orchestration.

- [ ] **Step 6: Run model and masked-forward tests**

Run:

```bash
pytest tests/test_training_model.py -q
pytest tests/training/test_masked_pretrain.py -k "pretrain_backbone or masked_forward or final_four" -q
```

Expected: PASS.

### Task 4: Replace the duplicate patch loss with authoritative losses

**Files:**
- Modify: `src/otuformer/training/loss.py`
- Modify: `tests/test_training_loss.py`
- Modify: `tests/training/test_pretrain_alignment.py`

**Interfaces:**
- Removes: `MaskedPatchRegressionLoss`.
- Produces: `masked_patch_cosine_loss(student, teacher, mask)`.
- Produces: `ibot_patch_loss(student_logits, teacher_logits, mask, patch_center, student_temp, teacher_temp)` returning the scalar loss and detached probabilities used by diagnostics/center updates.

- [ ] **Step 1: Write failing cosine-reference tests**

Use small hand-computable tensors and assert:

```python
expected = (2 - 2 * (F.normalize(s, dim=-1) * F.normalize(t, dim=-1)).sum(-1)).mean()
```

Only selected positions contribute, the teacher is detached, and empty, wrong-shape, or non-boolean masks raise actionable `ValueError`s. Remove the old EVA/non-EVA branching test and replace it with one mode-independent cosine result.

- [ ] **Step 2: Write failing iBOT-reference tests**

For a small `K`, compare exactly with:

```python
teacher_probs = F.softmax((teacher_logits - center) / teacher_temp, dim=-1)
student_log_probs = F.log_softmax(student_logits / student_temp, dim=-1)
expected = -(teacher_probs * student_log_probs).sum(-1).mean()
```

Assert only masked positions contribute and teacher probabilities are detached. Do not divide by `log(K)`.

- [ ] **Step 3: Run loss tests and verify they fail**

Run:

```bash
pytest tests/test_training_loss.py tests/training/test_pretrain_alignment.py -k "patch or masked_token" -q
```

Expected: FAIL until the new functions replace the old disconnected class and trainer branch.

- [ ] **Step 4: Implement the two functions and remove the duplicate class**

Use one private mask-validation helper in `loss.py`. Keep global/local and ArcFace losses unchanged. Delete the model-name argument and all EVA-specific MSE behavior.

- [ ] **Step 5: Run focused loss tests**

Run:

```bash
pytest tests/test_training_loss.py -q
pytest tests/training/test_pretrain_alignment.py -k "masked_token or global_loss or local_loss" -q
```

Expected: PASS.

### Task 5: Add the training-only patch objective and named EMA

**Files:**
- Modify: `src/otuformer/training/model.py`
- Modify: `src/otuformer/training/trainer.py`
- Modify: `tests/training/test_masked_pretrain.py`

**Interfaces:**
- Produces a training-only module with mode-specific state.
- Replaces positional `zip(student.parameters(), teacher.parameters())` EMA with name-matched EMA.
- Produces optimizer parameters containing only the selected objective's trainable student state.

- [ ] **Step 1: Write failing module-shape and ownership tests**

Assert mode construction exactly matches:

```text
none/consistency  no mask token, predictor, iBOT head, or patch center
masked-feature    [1,1,D] mask token + D->D predictor; no iBOT state
ibot              [1,1,D] mask token + student/teacher D->K heads + [1,K] center
```

The masked-feature predictor is `LayerNorm -> Linear -> GELU -> Linear`. The iBOT head is the same except its final output is `K`, the configured prototype count (any integer >= 2; default 512). Initialize the mask token with truncated normal. Copy the student iBOT head exactly into the gradient-free teacher head.

- [ ] **Step 2: Write failing named-EMA tests**

Create student/teacher modules with deliberately different parameter registration order and a student-only parameter. Assert:

- shared names update correctly;
- a missing required teacher name raises rather than silently shifting parameters;
- student-only mask/predictor parameters are excluded;
- the iBOT teacher head follows its student head under the same momentum.

- [ ] **Step 3: Write failing optimizer-membership tests**

For each patch mode, compare parameter identities:

```text
none/consistency  student encoder only
masked-feature    student encoder + mask token + predictor
ibot              student encoder + mask token + student iBOT head
```

Teacher encoder, teacher iBOT head, and patch center must never enter the optimizer. Use the existing LR and weight decay in one optimizer group; do not add named/split groups.

- [ ] **Step 4: Run tests and verify they fail**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "patch_objective or named_ema or optimizer_membership" -q
```

Expected: FAIL because mode-specific state and named EMA do not exist.

- [ ] **Step 5: Implement the smallest training-only module**

Place the small `nn.Module` definitions beside `OTUFormerEncoder` in `model.py`; do not add a factory framework. Expose only the mode-specific trainable parameters needed by optimizer construction.

Implement EMA by dictionary lookup:

```python
student_params = dict(student.named_parameters())
for name, teacher_param in teacher.named_parameters():
    if name not in student_params:
        raise ValueError(...)
    teacher_param.mul_(momentum).add_(student_params[name], alpha=1 - momentum)
```

Call the same helper separately for encoder EMA and iBOT-head EMA.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "patch_objective or named_ema or optimizer_membership" -q
```

Expected: PASS.

### Task 6: Integrate all four patch modes without changing global/local SSL

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Modify: `tests/training/test_masked_pretrain.py`
- Modify: `tests/training/test_pretrain_alignment.py`

**Interfaces:**
- `none`: no mask generation or patch state.
- `consistency`: existing unmasked forward plus selected-position cosine loss.
- `masked-feature`: extra masked student forward and final-four teacher target.
- `ibot`: extra masked student forward and centered prototype cross-entropy.

- [ ] **Step 1: Write failing orchestration tests with spies**

For two global and at least one local view, assert:

- global and local losses always receive unmasked student outputs;
- `none` performs no patch sampling and no masked forward;
- `consistency` performs no masked forward and selects matching final patch positions;
- A+ and iBOT perform exactly one extra masked student forward per global view;
- the teacher performs exactly one unmasked forward per global view in every mode;
- each global view and each batch item gets an independent mask;
- patch losses are normalized by actual selected token count and averaged over the two global views.

- [ ] **Step 2: Write failing A+ target tests**

Compute the reference target directly using the same final-token normalization module resolved and validated by Task 3; do not independently hard-code `teacher.backbone.norm` or fall back to `fc_norm` here:

```python
final_token_norm = teacher.resolved_pretrain_final_norm
layers = [final_token_norm(x)[:, prefix_count:] for x in final_four]
target = F.normalize(torch.stack(layers).mean(dim=0), dim=-1)
```

`resolved_pretrain_final_norm` is illustrative naming: implementation and tests must consume the single capability result established by Task 3, not perform a second attribute-name lookup.

Assert the predictor receives final masked-student patch tokens and the cosine loss receives only masked positions. A+ must not use final-layer-only teacher tokens.

- [ ] **Step 3: Write failing iBOT order and diagnostic-input tests**

Use call tracing to assert this order:

1. teacher logits computed with old center;
2. student optimizer step;
3. encoder and teacher-head EMA;
4. patch center updated once from the already-computed pre-EMA teacher logits.

Both global views' masked-position teacher logits must be concatenated for the center update. No additional teacher forward is allowed.

- [ ] **Step 4: Run tests and verify they fail**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "orchestration or masked_feature or ibot_order" -q
```

Expected: FAIL because `run_pretrain()` still has one consistency-only path.

- [ ] **Step 5: Integrate mode-specific branches**

Resolve patch config and validate backbone/grid before entering the epoch loop. Keep existing unmasked outputs as the sole inputs to `_compute_global_loss()` and `_compute_local_loss()`.

For consistency, sample a boolean selection mask and call the authoritative cosine function. For A+, request final-four teacher outputs from the existing teacher pass, construct one normalized target, then run the masked student path. For iBOT, run the independent heads and carry detached teacher probabilities/logits forward for diagnostics and center update.

Compute:

```python
patch_loss_weighted = args.lambda_mask * patch_loss_raw
loss = global_loss + args.lambda_local * local_loss + patch_loss_weighted
```

When mode is `none`, use a scalar zero on the active device without constructing a mask objective.

- [ ] **Step 6: Add one-step CPU smoke tests**

Run one optimizer step without pretrained downloads for:

```text
none
consistency
masked-feature + random/blockwise/hybrid
ibot + random/blockwise/hybrid
```

Use a small standard ViT and an actual blockwise-feasible grid. Assert finite loss and expected objective gradients. Keep separate negative tests for the invalid `4 x 4` and narrow grids.

- [ ] **Step 7: Run integration and alignment tests**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "orchestration or smoke or masked_feature or ibot" -q
pytest tests/training/test_pretrain_alignment.py -k "global_loss or local_loss or teacher_center or pretrain_encoder" -q
```

Expected: PASS, with existing global/local tests unchanged except removal of the EVA patch branch.

### Task 7: Add strict patch resume, separate state, and RNG round-tripping

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Modify: `tests/training/test_masked_pretrain.py`
- Modify: `tests/test_checkpoint_utils.py`

**Interfaces:**
- Checkpoint `config` stores all effective patch settings.
- Checkpoint root stores `patch_objective` separately from `student`/`teacher`.
- Checkpoint root stores Python, NumPy, PyTorch CPU, and all CUDA RNG states.

- [ ] **Step 1: Write failing v0.7 strict-resume tests**

For each mode, round-trip the expected `patch_objective` state. Reject:

- patch-loss changes;
- resolved ratio changes;
- applicable strategy changes;
- iBOT prototype changes;
- fixed target-layer or center-momentum changes;
- missing/malformed patch-objective state for A+ or iBOT;
- optimizer state incompatible with the reconstructed objective.

Reconstruct and load the objective before loading optimizer state.

- [ ] **Step 2: Write failing legacy-resume tests**

A checkpoint without `config.patch_loss` must resolve to consistency:

```text
args.mask_ratio present -> exact saved value, including 0.50
args.mask_ratio absent  -> 0.50
```

Reject an explicitly conflicting new ratio, `masked-feature`, or `ibot`. Emit one warning that exact random continuation is unavailable because legacy checkpoints have no RNG state.

- [ ] **Step 3: Write failing RNG tests**

Capture and restore:

- `random.getstate()`;
- `np.random.get_state()`;
- `torch.get_rng_state()`;
- `torch.cuda.get_rng_state_all()` when CUDA is available.

After restoration, assert the next Python, NumPy, CPU torch, mask-generation, and CUDA values match the uninterrupted sequence. Monkeypatch the epoch flow to prove restoration happens after model/objective/optimizer initialization and before the next DataLoader iterator is created.

- [ ] **Step 4: Write failing read-only compatibility test**

Save a general encoder checkpoint with an extra root-level `patch_objective`. Assert `resolve_checkpoint()`/`apply_checkpoint_weights()` load the same encoder tensors and ignore the extra training-only state. Raw CLS/projector dimensions must remain unchanged.

- [ ] **Step 5: Run tests and verify they fail**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "resume or legacy or rng" -q
pytest tests/test_checkpoint_utils.py -k patch_objective -q
```

Expected: FAIL until config/state/RNG handling is added.

- [ ] **Step 6: Implement checkpoint and RNG helpers**

Use small private helpers in `trainer.py`:

```python
_capture_rng_state() -> dict[str, object]
_restore_rng_state(state: dict[str, object]) -> None
_validate_patch_resume(current: dict, saved: dict) -> None
```

Save only mode-relevant entries in `patch_objective.state_dict()`. Keep `model_state_dict`, `student`, and `teacher` free of training-only heads.

Move epoch-end metric/UMAP work before checkpoint RNG capture and write. Restore new-checkpoint RNG state immediately before entering the next epoch iterator; any resume-only startup diagnostic must occur before restoration or preserve/restore RNG around itself.

Do not claim exact multi-worker augmentation replay.

- [ ] **Step 7: Run focused checkpoint tests**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "resume or legacy or rng" -q
pytest tests/test_checkpoint_utils.py -q
```

Expected: PASS.

### Task 8: Migrate instant metrics and add patch diagnostics

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Modify: `tests/training/test_masked_pretrain.py`

**Interfaces:**
- Preserves recognized legacy pretrain rows.
- Adds authoritative patch raw/weighted/fraction, requested/selected/actual ratio, and iBOT stability fields.
- Atomically migrates only the exact recognized legacy schema.

- [ ] **Step 1: Write failing schema and migration tests**

Assert a new pretrain CSV contains the existing fields plus:

```text
patch_loss_raw
patch_loss_weighted
patch_loss_fraction_of_total
requested_mask_ratio
selected_patch_ratio
actual_mask_ratio_mean
actual_mask_ratio_min
actual_mask_ratio_max
prototype_perplexity
active_prototype_ratio
assignment_entropy
max_prototype_occupancy
patch_center_norm
```

Retain the existing `mask_loss` column as the raw patch-loss compatibility alias; the new `patch_loss_*` fields are authoritative.

For the exact old pretrain header, assert migration:

- preserves every row and old value;
- fills every newly introduced field with an empty string: `patch_loss_raw`, `patch_loss_weighted`, `patch_loss_fraction_of_total`, `requested_mask_ratio`, `selected_patch_ratio`, all three defined `actual_mask_ratio_*` fields (`actual_mask_ratio_mean`, `actual_mask_ratio_min`, `actual_mask_ratio_max`), `prototype_perplexity`, `active_prototype_ratio`, `assignment_entropy`, `max_prototype_occupancy`, and `patch_center_norm`;
- writes a temporary file in the same directory;
- uses `os.replace()` only after the complete file is closed;
- leaves no partial temporary file.

- [ ] **Step 2: Write failing malformed-schema tests**

Duplicate columns, unknown headers, truncated rows, and malformed CSV must raise an actionable error without changing the original bytes. Current v0.7 headers append normally.

Plotting a migrated file with empty historical numeric fields must not fail.

- [ ] **Step 3: Write failing mode-specific logging tests**

Assert:

- consistency fills `selected_patch_ratio` and leaves all actual-mask fields empty;
- A+ and iBOT fill mean/min/max actual mask ratio and leave selected-visible ratio empty;
- non-iBOT modes leave all iBOT diagnostic fields empty;
- patch fraction is `weighted_patch_loss / total_loss` with a defined zero-total guard.

- [ ] **Step 4: Write failing iBOT diagnostic reference tests**

From teacher probabilities at all masked positions in both global views, compare:

```text
prototype_perplexity       exp(entropy(mean_assignment))
active_prototype_ratio     used hard assignments / K
assignment_entropy         mean per-position entropy
max_prototype_occupancy    largest hard count / assignments
patch_center_norm          L2 norm of updated center
```

Use per-step values only.

- [ ] **Step 5: Run tests and verify they fail**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "metrics_schema or csv_migration or diagnostics" -q
```

Expected: FAIL because the logger still accepts/creates only the legacy schema.

- [ ] **Step 6: Implement strict schema migration and diagnostics**

Define immutable current/legacy field lists next to `InstantMetricsLogger`. Read and validate the existing header before opening append mode. Migrate via `tempfile.NamedTemporaryFile(..., dir=self.path.parent, delete=False)`, flush/close, then `os.replace()`.

Update plots to prefer `patch_loss_raw` while accepting historical/migrated empty values. Keep plotting tolerant of non-numeric empty cells through `pandas.to_numeric(..., errors="coerce")`.

- [ ] **Step 7: Run logger tests**

Run:

```bash
pytest tests/training/test_masked_pretrain.py -k "metrics_schema or csv_migration or diagnostics" -q
```

Expected: PASS.

### Task 9: Update user documentation, historical status notes, and version

**Files:**
- Modify: `README.md`
- Modify: `README.cn.md`
- Modify: `docs/superpowers/specs/2026-03-29-otuformer-design.md`
- Modify: `docs/superpowers/specs/2026-03-30-otuformer-pretrain-alignment-design.md`
- Modify: `docs/superpowers/plans/2026-03-30-pretrain-alignment.md`
- Modify: `docs/superpowers/specs/2026-09-20-otuformer-masked-pretrain-v070-design.md`
- Modify: `ref/morphotu-research-roadmap.md`
- Modify: `pyproject.toml`
- Modify: `src/otuformer/__init__.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_version.py`

- [ ] **Step 1: Write failing documentation/version assertions**

Update tests to require:

- package/project version `0.7.0`;
- English and Chinese README coverage of all four modes and all three strategies;
- first-use definition of A+ as continuous masked feature prediction;
- `ibot` marked experimental;
- prominent `0.50 -> auto/0.30` new-run migration notice and legacy `0.50` resume rule;
- CLI help distinguishes selected visible patches from true input masking.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_version.py tests/test_cli_smoke.py -k "version or readme or patch_loss or mask_ratio" -q
```

Expected: FAIL while version/docs remain at v0.6.0 terminology.

- [ ] **Step 3: Update current documentation**

Document:

- command examples for `consistency`, `masked-feature`, and experimental `ibot`;
- resolved defaults and invalid explicit combinations;
- compute cost of two additional masked student global forwards;
- extraction remains raw CLS by default and never invokes patch-objective state;
- no dense anatomical-semantic claim;
- standard timm ViT-only pretraining support;
- strict v0.7 resume and best-effort legacy random continuation.

Correct the roadmap's current “masked-token” terminology. Separate self-supervised contextual feature prediction from future supervised dense morphology work.

- [ ] **Step 4: Add historical status notes only**

At the top of the 2026-03-30 pretraining design and plan, add a short status note linking to the v0.7 design and this plan. Do not rewrite their historical decisions.

Keep the v0.7 design status unchanged during implementation and the initial Task 10 handoff. Changing it to implemented is an approval-gated documentation edit handled only by Task 10 Step 7; it remains part of the same uncommitted review batch.

- [ ] **Step 5: Bump version declarations**

Set:

```toml
version = "0.7.0"
```

and:

```python
__version__ = "0.7.0"
```

- [ ] **Step 6: Run documentation/version tests**

Run:

```bash
pytest tests/test_version.py tests/test_cli_smoke.py -k "version or readme or patch_loss or mask_ratio" -q
```

Expected: PASS.

### Task 10: Run complete verification and representative benchmarks

**Files:**
- Verify all files listed above.
- Do not add benchmark infrastructure unless the existing CLI cannot produce the required measurements.

- [ ] **Step 1: Run focused test groups**

Run:

```bash
pytest tests/test_training_model.py tests/test_training_loss.py -q
pytest tests/training/test_masked_pretrain.py -q
pytest tests/training/test_pretrain_alignment.py -q
pytest tests/test_cli_smoke.py tests/test_checkpoint_utils.py tests/test_version.py -q
```

Expected: all PASS.

- [ ] **Step 2: Run the complete suite**

Run:

```bash
pytest -q
```

Expected: PASS with no failures.

- [ ] **Step 3: Run one real CLI smoke command per mode**

Use a tiny local fixture dataset, `--device cpu`, `--num-workers 0`, `--local-crops 0`, one epoch, and a standard timm ViT without downloading weights in the test harness. Verify creation and reload of:

```text
none checkpoint
consistency checkpoint
masked-feature checkpoint
ibot checkpoint
```

Inspect each checkpoint to confirm general encoder and `patch_objective` state are separated and its effective config is complete.

- [ ] **Step 4: Benchmark the three active patch objectives**

On the same representative dataset/backbone/global/local sizes, batch size, seed, optimizer settings, warm-up steps, and measured step window, run:

```text
consistency
masked-feature + random
ibot + random + 512 prototypes
```

Record median step time, images/second, and peak device memory. Reset peak-memory statistics immediately before each measured window; synchronize CUDA around timing when CUDA is used. Report hardware, software versions, exact command lines, warm-up length, sample count, and the three results in the implementation handoff. This is a cost report, not a pass/fail performance threshold.

- [ ] **Step 5: Inspect repository changes**

Run:

```bash
git diff --check
git diff --stat
git status --short
git diff -- src/otuformer/cli/pretrain.py src/otuformer/training/model.py src/otuformer/training/loss.py src/otuformer/training/trainer.py
git diff -- README.md README.cn.md docs ref pyproject.toml src/otuformer/__init__.py
git diff -- tests
```

Expected:

- no whitespace errors;
- no `.codegraph/` changes;
- no unrelated refactor or generated artifact;
- no staged files or commit;
- the design, implementation, tests, docs, and version agree.

- [ ] **Step 6: Present the implementation review handoff**

Report exact test counts/outcomes, CLI smoke results, benchmark table, changed-file summary, legacy limitations, residual risk around multi-worker augmentation replay, and the still-pending design-status transition. Wait for user approval; do not change the design status or create a commit yet.

- [ ] **Step 7: Finalize the design status only after approval**

After the user approves the implementation handoff, change the v0.7 design status from draft to implemented and link this implementation plan. Keep that status change in the same uncommitted diff, rerun the documentation/version tests and `git diff --check`, and present the final diff status. A commit still requires explicit user approval after this final documentation edit.
