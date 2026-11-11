# Relational FIFO Candidate Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (only if separately requested by the operator) or superpowers:executing-plans to implement this plan task-by-task after execution approval. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional image-keyed paired-teacher FIFO to the existing global/local relational distributions, with strict recovery and lightweight diagnostics, while preserving the disabled path.

**Architecture:** A narrow queue state owner supplies one immutable candidate snapshot per step. Enabled-only data preparation precedes architecture probes; the existing recipe, objectives, trainer and reporting owners retain their responsibilities. Current aligned-image candidates and encoder layouts stay unchanged.

**Tech Stack:** Existing Python, PyTorch, Typer/Click, pytest and project CSV/checkpoint helpers; no new dependencies.

**Spec:** [Approved FIFO design](../specs/2026-11-10-relational-fifo-design.md), [master design](../specs/2026-03-29-otuformer-design.md), [development testing policy](../specs/2026-11-07-development-testing-policy.md) and [training ownership design](../specs/2026-11-07-training-modularity-cli-testing-design.md). Read the feature spec and this plan together.

**Date:** 2026-11-11. Retain the 2026-11-10 filename required by the design.
**Status:** Implemented and verified on 2026-11-11 against the approved design (Tasks 1-8). Implementation steps are checked because they have executed evidence; every commit checkpoint stays unticked because no commit permission was given. No research training, dependency change, version change, tag or release happened, and nothing was pushed.

## Global Constraints

- Public interface: `--relational-queue-size INTEGER`, minimum zero. Fresh omission resolves to zero; resume omission inherits saved capacity, with historical absence meaning disabled.
- Keep current batch first, including the aligned-image column. Each slot uses `B + E` columns, not `B + 2E`; preserve `cross`/`full`/inherited `same` pairing and local/global weights.
- Cache detached float32 projector outputs before center subtraction; use the current pre-step center for current/history. History never updates the center.
- Capacity zero takes the original loss/data path. Preserve numerical, gradient and RNG behavior; do not make disabled runs use keyed datasets or compute queue diagnostics.
- G1 is enabled-only: unique selected-reference keys within root, full selected-image hashing before the probe, key/content/order lock; root relocation is allowed. Internal aliases are distinct identities, not file-level deduplication.
- G2 is enabled-only: exact saved batch size and steps-per-epoch, omitted-batch inheritance, valid complete RNG/progress/schedule recovery without legacy fallback. Never silently reset, resize or truncate history.
- Persist only positive-capacity queue bundles; keep the existing closed `relational_target_config` version-1 key set unchanged. Save only at existing scheduled epoch boundaries.
- Preserve center-update timing before finite checks and teacher EMA/patch-center timing after optimizer update. A non-finite step raises the existing exception; no skip/retry, rollback or per-step saver.
- Existing single-process scope, explicit fixed capacity, float32 paired slots, cross-epoch retention. No confidence weighting, fill threshold, age cutoff, new warm-up, patch/token queue, distributed queue, automatic capacity or OOM policy.
- New design, plan and developer documentation are English-only. Existing user-facing README/command bilingual conventions remain. No `.cn.md` counterparts for this spec or plan.
- Follow the testing policy: ordinary tiny real loops use `auto`; exact replay/device-dtype checks use CPU. Keep documented smaller-local interpolation tests on CPU; do not silently enable accelerator fallback or infer CUDA coverage from MPS/CPU.

## Review Focus

These are easy-to-miss user inputs, not a second acceptance checklist; their regression tests are assigned below.

1. A checkpoint declares DINO while parser defaults say relational: reject explicit queue options before hashing or building encoders; shared identity resolution must agree in early and late paths (Task 3).
2. `link/../one.png`, duplicate normalized references, two valid aliases of one file, and a relocated root: preserve actual opened-file semantics without physical deduplication (Task 1).
3. `100 // 30 == 100 // 33`: equal loader length does not make a changed batch size valid; missing explicitness flags in direct-core callers must not disguise a requested override (Task 3).
4. A failure after target-center update but before optimizer success: queue/order/ages stay unchanged, although the center need not roll back; no new epoch checkpoint (Task 5).
5. Resuming a real 43-column CSV and a saved 64/128 batch/capacity with parser defaults 32/None: preserve old cells and distinguish raw requests from inherited effective values, without duplicate CLI records (Tasks 6–7).

---

## File Map and Execution Rules

Line numbers below are navigation anchors at baseline `3d69d1812a1b5d6941c49bf070f1037c1304d83d`, package 0.12.0; search named symbols after earlier tasks move them. No `training/pretrain_settings.py` exists in this baseline: settings and recipe ownership are in `pretrain_recipe.py`.

| Owner | Change |
| --- | --- |
| `src/otuformer/training/dataset.py` | Prepared ordered selection and enabled keyed envelope; preserve legacy resolver and disabled return type. |
| `src/otuformer/training/relational_queue.py` (new) | Ordered paired features, eligibility, refresh, ages and strict version-1 codec. No CLI imports. |
| `src/otuformer/training/pretrain_recipe.py` | Model-free request resolution, data preparation, batch/step/RNG locks, handoff and expanded temperature guard. |
| `src/otuformer/training/objectives.py` | Rectangular candidates, per-slot teacher targets and log-step detached statistics. Existing scalar-return callers remain valid. |
| `src/otuformer/training/pretrain_loop.py` | Consume prepared inputs, one queue snapshot, success-only insertion, epoch persistence and resolved log. |
| `src/otuformer/training/reporting.py` | New fields and recognized 43-column predecessor; no duplicate probability calculation. |
| `src/otuformer/utils/checkpoint.py` | Shared method/queue identity guard; no training/data import from readers. |
| `src/otuformer/cli/pretrain.py`, `cli/constraints.py`, `cli_schema.py` | Optional integer, source-aware explicitness, early preparation, applicability and conditional cards. |
| Tests | Focused pure tests in new queue files; reuse existing dataset, alignment, masked-loop, metrics, CLI/schema/skill and consumer tests. No global model fixture. |
| Documentation and workflow skill | Surfaces listed in Task 8; do not introduce an alternative CLI source of truth. |

Execute Tasks 1–8 in order after plan/execution approval; they are one feature, not independently releasable versions. There is no authorized parallel/subagent execution method yet. Each task has a red test, a minimal implementation, a green check and a task-sized commit checkpoint **only if commits are separately authorized**. Do not add code merely to satisfy an intermediate public CLI that the trainer cannot yet execute.

## Task 1: Prepare the Exact Selected Images and Carry Keys with Views

**Files:** Modify `src/otuformer/training/dataset.py:415–436,947–1086`; test `tests/test_training_dataset.py`. Reuse, do not rewrite, `src/otuformer/data/paths.py` and `data/provenance.py`.

**Interfaces:**

- Add frozen `PreparedRelationalData` in `dataset.py`: `image_paths: tuple[Path, ...]`, `image_keys: tuple[str, ...]`, `data_fingerprint: str`, `data_size_bytes: int`; image count is `len(image_keys)`. Do not retain image payload bytes or a checkpoint manifest.
- Add `prepare_relational_data(csv_path: Path | None, images_dir: Path, *, batch_size: int) -> PreparedRelationalData`. The caller decides whether FIFO is enabled.
- Extend `MultiCropDataset.__init__(..., *, prepared_relational_data: PreparedRelationalData | None = None)` after existing keyword arguments. With preparation, use those selected paths without re-resolving; `__getitem__` returns `{'views': views, 'image_key': image_keys[index]}`. Without it, preserve the exact list-of-views result. Native collation yields a mapping of view tensors and aligned keys.

- [x] **Step 1 — Add failing dataset tests.** Add `test_fifo_preparation_preserves_selected_paths_and_alias_keys`, `test_fifo_preparation_content_order_and_relocation`, `test_fifo_preparation_rejects_invalid_inputs` and `test_fifo_keyed_batches_preserve_view_rng_and_alignment`. Assertions: `a/../one.png` and `one.png` collide; `lnk -> a/x` plus `lnk/../one.png` addresses `a/one.png`, not `one.png`; `link.png -> a/one.png` and `a/one.png` remain two keys. Cover legacy basename matches, outside-root and escaping-link rejection, missing/corrupt images, duplicate keys and `N < B`.
The simple alias case must include this executable assertion core (imports belong in the test module):

```python
def test_fifo_preparation_keeps_two_internal_aliases(tmp_path):
    from PIL import Image
    Image.new('RGB', (8, 8)).save(tmp_path / 'one.png')
    (tmp_path / 'alias.png').symlink_to('one.png')
    csv = tmp_path / 'images.csv'
    csv.write_text('image\none.png\nalias.png\n')
    data = prepare_relational_data(csv, tmp_path, batch_size=2)
    assert data.image_keys == ('one.png', 'alias.png')
    assert len(data.data_fingerprint) == 64
```

- [x] **Step 2 — Run the new tests red.** `python -m pytest tests/test_training_dataset.py -q -k fifo`. Expected: missing preparation/type/constructor support, not an unrelated fixture failure.
- [x] **Step 3 — Implement selection and fingerprint.** Call `resolve_dataset_image_paths()` exactly once; derive identity from each already-selected path using `resolve_reference(root, absolute_written_selected_path, basename_fallback=False)` and `normalize_key`, without `.resolve()` or `normpath` on that spelling first. Require containment without changing the shared helper semantics. Never re-resolve the original CSV reference or replace legacy basename selection.
- [x] **Step 4 — Validate then hash once.** Reject path/key/count conflicts before reading all image bytes. For each selected image, use `snapshot_bytes()` once, validate that the bytes are a decodable image with the existing Pillow stack, append an ordered `{image_key, size_bytes, sha256}` row, and discard its bytes. Use `canonical_digest(rows)`; labels, absolute roots and other CSV columns are excluded. Announce the phase before file reads using existing phase helpers and report image count, byte count and elapsed time. Do not hash an off run. Preparation consumes no augmentation RNG.
- [x] **Step 5 — Verify keyed/off envelopes.** Run `python -m pytest tests/test_training_dataset.py -q`. Require fixed-seed crop tensors, crop-call order and RNG state to equal the old dataset for the same selected files; test worker-zero and a small multi-worker aligned batch without promising exact cross-worker replay. Root relocation keeps the digest; renaming, content or row-order changes do not. Two equal-content keys remain distinct. In `test_fifo_keyed_batches_preserve_view_rng_and_alignment`, assert both `set(sample) == {'views', 'image_key'}` and `set(batch) == {'views', 'image_key'}`, including with a label-bearing input CSV; any label or extra key fails. No labels enter the relation target.
- [ ] **Step 6 — Conditional commit checkpoint.** With explicit commit permission only, stage the two named files and commit `feat(pretrain): prepare FIFO image identities without changing legacy views`.

## Task 2: Implement Paired FIFO State and a Strict Logical Codec

**Files:** Create `src/otuformer/training/relational_queue.py` and `tests/training/test_relational_queue.py`.

**Interfaces:**

- Frozen `RelationalQueueSnapshot`: `keys: tuple[str, ...]`, `features: Tensor` shaped `[E, 2, D]`, `observed_steps: tuple[int, ...]`. A zero-history snapshot has shape `[0, 2, D]`; it is not random padding.
- `RelationalQueue(capacity: int, out_dim: int, *, device: torch.device)` owns an oldest-to-newest `OrderedDict` of key → `(cloned_float32_pair, observed_step)` and `completed_steps: int = 0`. Capacity must be positive.
- `snapshot(batch_keys: Sequence[str], *, step: int) -> RelationalQueueSnapshot`; `step` must equal `completed_steps`. Exclude every current key; reads do not refresh.
- `commit(batch_keys: Sequence[str], precenter_features: Tensor, *, step: int) -> dict[str, int]`, where features are `[B, 2, D]`; return `{'refreshed': int, 'evicted': int}`. Validate all inputs before mutation, clone/detach float32 pairs, reinsert in batch order, evict oldest and increment completed steps exactly once.
- `state_dict(*, data_fingerprint: str, data_image_count: int, data_size_bytes: int) -> dict[str, Any]` emits exactly the spec's eight version-1 fields (`version`, `keys`, `features`, `observed_steps`, `completed_steps`, `data_fingerprint`, `data_image_count`, `data_size_bytes`), CPU feature tensors, logical order, no pointer or manifest.
- `from_state_dict(state: Mapping[str, Any], *, capacity: int, out_dim: int, completed_steps: int, allowed_keys: frozenset[str], data_fingerprint: str, data_image_count: int, data_size_bytes: int, device: torch.device) -> RelationalQueue` validates and restores; no silent repair. The recipe uses CPU validation; the trainer reconstructs on its resolved device.

- [x] **Step 1 — Add red state-machine and codec tests.** Cover empty/partial/full, `K < B`, `K > N`, all-current exclusion, repeated-key refresh, no read refresh, same-step batch order, oldest eviction and age `g - observed_step`. Check e.g. committing `[a,b,c]` at step 0 with `K=2` retains `[b,c]`, both observed at 0, completed count 1; snapshot for `[b]` at step 1 retains only `c`, age 1. Reject duplicate current keys, wrong dimension and non-finite insertion before any mutation.
```python
def test_fifo_snapshot_excludes_current_without_refresh():
    q = RelationalQueue(2, 2, device=torch.device('cpu'))
    q.commit(('a', 'b', 'c'), torch.ones(3, 2, 2), step=0)
    snap = q.snapshot(('b',), step=1)
    assert snap.keys == ('c',) and snap.observed_steps == (0,)
    assert q.snapshot((), step=1).keys == ('b', 'c')
    assert q.completed_steps == 1
```

- [x] **Step 2 — Run red.** `python -m pytest tests/training/test_relational_queue.py -q`; expected missing module/type.
- [x] **Step 3 — Implement the minimum state owner.** No dataset/CLI imports, recentering, model forwards or global queue. Snapshots and inserted pairs must not alias mutable model outputs or each other across a commit; mutating a snapshot in a test must not change stored queue features. Reading at epoch boundaries is ordinary reading. Storage grows only with occupancy; do not allocate `K` empty vectors.
- [x] **Step 4 — Test strict serialization.** Add `test_fifo_codec_rejects_malformed_state` parametrized over unexpected/missing keys, bool integers, bad version/capacity, duplicate/outside keys, wrong shape/dtype, non-finite features, timestamps outside `[0, iteration)`, reversed timestamps, count/digest mismatch, and empty state after positive completed steps. Reject `requires_grad=True` features at the persisted trust boundary. An empty step-zero state and CPU round trip are valid. The latest successful batch's required tail depends on effective B; Task 3 validates it in the recovery layer rather than adding batch metadata to this codec.
- [x] **Step 5 — Run green.** `python -m pytest tests/training/test_relational_queue.py -q`; round trip must preserve keys, features, ages, count and next eviction, with detached float32 CPU serialization and no source tensor aliasing.
- [ ] **Step 6 — Conditional commit checkpoint.** Stage only the new module/test, commit `feat(pretrain): add image-keyed paired relational FIFO state`.

## Task 3: Resolve Enabled Requests, Data and Recovery Before Probes

**Files:** Modify `src/otuformer/training/pretrain_recipe.py:610–641,1846–1910,1951–2041,2496–2774`; create `tests/training/test_relational_queue_recipe.py`; extend `tests/training/test_pretrain_alignment.py`. CLI activation is Task 7.

**Interfaces:**

- Frozen `_RelationalQueuePreparation` in `pretrain_recipe.py`: `ssl_method: str`, `capacity: int | None` (None only for inactive DINO), `requested_capacity: int | None`, `capacity_source: str`, `effective_batch_size: int`, `requested_batch_size: int | None`, `batch_size_source: str`, `data: PreparedRelationalData | None`, `resume_queue_state: dict[str, Any] | None`, `resume_payload: dict[str, Any] | None`, `request_binding: tuple[object, ...]`, `resolved_device: torch.device | None`, `rng_state_to_restore: dict[str, object] | None`. The last two are transient and only populated for active FIFO; the validated restore copy must not mutate the saved payload. Keep the payload by reference only in this transient object. Bind the preparation to `(resume path, written image-root path, train-data path, requested SSL method, method explicitness, raw queue request, queue explicitness, raw batch request, batch explicitness, requested device)` and the loaded payload's object identity; verify before replacing args with effective values. Source values are `explicit`, `default`, `inherited` or `inactive` as applicable. This carries no mutable queue instance and is transient.
- `prepare_pretrain_relational_queue(args: argparse.Namespace, checkpoint: dict[str, Any] | None) -> _RelationalQueuePreparation`. Missing request attribute means omission. Numeric request without an explicitness marker in a direct-core namespace is treated as explicit; CLI always supplies markers. For enabled-resume batch inheritance, direct callers carrying parser defaults must supply `batch_size_explicit=False`; a present batch value with no marker is an override, not silently omitted.
- Extend `_PretrainHandoff` with `relational_queue_preparation: _RelationalQueuePreparation | None`; consume/clear `args.relational_queue_preparation` before any raising validation. Add that attribute to `_UNSAVED_ARGS`. Extend `_ResolvedPretrainRecipe` with the same preparation object.
- `_resolve_pretrain_recipe(..., settings: PretrainSettings, relational_queue_preparation: _RelationalQueuePreparation | None = None)` retains existing off-call compatibility. An enabled request/checkpoint with no matching preparation raises. Never enable an off recipe by default or rehash during finalization.
- `resolve_pretrain_ssl_config(args, resume_ckpt, *, relational_queue_preparation: _RelationalQueuePreparation | None = None)` and `_effective_pretrain_ssl_config` use effective candidate bounds for student-temperature checks; they remain after the architecture probe. `_resolve_relational_target_config(args: argparse.Namespace, checkpoint: dict[str, Any] | None, *, candidate_count: int | None = None)` uses that same bound for teacher endpoints, keeping the off default unchanged. `_effective_pretrain_ssl_config(args, handoff=None, resume_ckpt=_UNLOADED, *, relational_queue_preparation: _RelationalQueuePreparation | None = None)` forwards the same preparation. For active FIFO the bound uses the effective batch, not the parser default: `B_eff + min(K, N - B_eff)`.

- [x] **Step 1 — Add red request-table tests.** `test_fifo_request_matrix` covers fresh omitted/0/positive, resume omitted/matching/conflicting, historical absent bundles, and DINO omitted versus any explicit value including 0. Missing core attributes inherit an enabled checkpoint. Preparation uses capacity None/source inactive for omitted DINO, capacity 0/source default or explicit for fresh relational, and source inherited for omitted relational resume, including historical off. Reject negatives/bools/fractions/strings/non-finite values, mismatched args/config/state and one-sided bundles. Checkpoint effective capacity is `args.relational_queue_size`; preserve the original request separately in `args.requested_relational_queue_size` (None for omission) and its explicitness flag. An omitted marker never excuses missing enabled bundles.
```python
def test_fifo_direct_omission_stays_legacy_off():
    prep = prepare_pretrain_relational_queue(argparse.Namespace(batch_size=32), None)
    assert prep.ssl_method == 'relational' and prep.capacity == 0
    assert prep.data is None and prep.requested_capacity is None
    assert prep.effective_batch_size == 32

def test_fifo_dino_explicit_zero_is_rejected():
    args = argparse.Namespace(ssl_method='dino', batch_size=32,
        relational_queue_size=0, relational_queue_size_explicit=True)
    with pytest.raises(ValueError, match='relational'):
        prepare_pretrain_relational_queue(args, None)
```

- [x] **Step 2 — Run red.** `python -m pytest tests/training/test_relational_queue_recipe.py -q`; expected missing resolver/handoff behavior.
- [x] **Step 3 — Resolve without model construction.** Reuse `_saved_ssl_state()` (or its exact state-dict → shared identity steps). Preserve requested/default/saved-method conflicts; never independently trust `config['ssl_method']`. Check request and bundle applicability before `prepare_relational_data`. Do not call the complete SSL resolver early: its relational optimizer preflight builds two encoders. The CLI also calls `preflight_pretrain_resume()` before creating its full namespace; that calls `augmentation._resume_register_plan()` and already constructs an encoder. For enabled resume, load the checkpoint once, build the request namespace from parsed values early enough for Task 3 preparation, finish model-free applicability/data/fingerprint checks, then call the existing register-layout resume preflight. The enabled sequence is model-free queue request → enabled data preparation/locks → register-layout resume probe (if resuming) → existing settings/architecture preflight → recipe and full SSL validation → output setup. Preserve queue-off's existing resolution/probe order, in addition to skipping data preparation; the added pure applicability check must not construct a model or consume RNG. Complete SSL validation remains once per invocation.
- [x] **Step 4 — Implement G2 and handoff.** For enabled resume require saved `args.batch_size`, `schedule.steps_per_epoch`, valid epoch/iteration/schedule, optimizer, both encoders, existing center state, applicable independent `patch_objective` state and complete `rng_state`. Require `prepared_count // effective_batch_size` to match saved steps; compare batch itself too, including 30 versus 33 for 100 images. Omission inherits; an explicit different batch fails before output setup. For active FIFO, resolve the actual target once using the existing `_resolve_device` after data preparation and before output setup; carry that device to the trainer. Validate RNG through Step 4c, not merely `_restore_rng_state` returning without error. Preserve caller RNG on success and failure; do not duplicate DINO's whole preflight. Use existing schedule/progress validation without a legacy fallback. Add `_validate_relational_queue_resume(checkpoint: dict[str, Any], preparation: _RelationalQueuePreparation) -> None`, called before any output setup: validate exact bundle keys/version/types, saved epoch/iteration integers (no bools), `iteration == (epoch + 1) * steps_per_epoch` at the supported epoch boundary, and required schedule fields/finite values through the existing progress/schedule rules. New target extensions follow the existing recovery path; do not accidentally lock `max_epochs`. Validate both saved encoder states/center after the architecture probe but still before output creation; never rely on a later trainer failure to establish strict recovery. The current `_validate_relational_optimizer_preflight()` proves only group layout: add keyword-only `strict_fifo_state: bool = False`, supplied as true by `resolve_pretrain_ssl_config()` only for positive capacity, and reuse its already-built expected encoder, patch objective and optimizer for the state checks below. Keep the existing layout check and queue-off/DINO behavior unchanged. Add recipe-level cases for iteration/epoch mismatch and non-finite saved LR, verifying rejection before logger migration. Validate queue codec against the prepared keys/digest/counts and saved projector width. Then enforce the batch-aware latest-commit invariant: at g > 0 the final `min(K, effective_batch_size)` records exist and all have observation step g-1, with no earlier record stamped g-1. At a supported checkpoint after at least one complete epoch, also require `M >= min(K, (N // B) * B)` with selected-key count N and effective batch B. A full no-replacement/drop-last epoch necessarily contributes that many unique keys; retaining only the latest batch is insufficient. These are recovery-layer checks, not new codec fields or a shuffle-history reconstruction algorithm. Never attach a persistent mutable queue to `args`.
- [x] **Step 4a — Validate FIFO AdamW state, not just layout.** In the strict-FIFO branch, map saved parameter IDs to the rebuilt optimizer's actual ordered groups after layout validation. Require state for parameters that should already have participated in the saved relational/patch recipe; missing entries are not proof of legitimate inactivity. Validate scalar finite integer step in `[1, completed_steps]`, moment tensor shapes against mapped parameters and finiteness (`exp_avg`, `exp_avg_sq`, plus `max_exp_avg_sq` if the saved AdamW group uses AMSGrad). Require every element of `exp_avg_sq >= 0`, and `max_exp_avg_sq >= 0` when present for AMSGrad; finite negative second moments are invalid even when loading succeeds. Do not require every parameter's step to equal global_step or require state for demonstrably inactive/frozen parameters; document/test any actual exception from the existing recipe, without guessing from absent state. Reuse DINO's positional-validation pattern, not its full preflight or DINO-only freeze exception. Add `test_fifo_resume_rejects_missing_or_malformed_adamw_state` for cleared state, missing participating entries, wrong-shaped exp_avg, invalid step, non-finite moments and finite negative exp_avg_sq/AMSGrad max_exp_avg_sq; reject before any output modification. Zero second moments are valid, exp_avg may be signed, and valid lazy/inactive state must not be rejected.
- [x] **Step 4b — Strictly validate independent patch state before outputs.** In the same strict-FIFO probe branch, use saved resolved patch mode/prototypes and `probe_patch.load_state_dict(saved_patch_state, strict=True)` inside the existing RNG-preserving probe scope. Require complete state for masked-feature/ibot; inspect floating tensors for finiteness, including mask token, predictor, both iBOT heads and patch center. Strict loading catches missing/extra keys and wrong shapes but does not alone reject NaN. Preserve empty/stateless none/consistency semantics and queue-off behavior. Add `test_fifo_resume_rejects_invalid_patch_objective_before_outputs` for missing `predictor.1.weight`, wrong iBOT-head shape, missing/wrong-shaped/NaN patch center; do not wait for the trainer's post-logger strict load to fail.
- [x] **Step 4c — Validate RNG for the actual recovery target.** Add `_validate_fifo_rng_state(state: dict[str, object], *, target_device: torch.device) -> dict[str, object]` in `pretrain_recipe.py`. Require the existing Python/NumPy/CPU-torch streams. If a `cuda` key is present, require a nonempty list/tuple of nonempty one-dimensional CPU uint8 tensors; None/empty or malformed payload is not an absent CPU-only stream. For a CUDA target require this key and exactly `torch.cuda.device_count()` states (capture uses all visible devices, not only the selected index), then restore every stream using the existing helper inside `preserved_rng()` to prove backend usability. Reject missing/None/empty/count-mismatched states before outputs; do not seed substitute CUDA states. For a non-CUDA target allow a CPU-generated checkpoint with no CUDA key. If valid CUDA-shaped payload is present, validate its structure but omit it from a shallow restore copy and report explicitly that CUDA RNG was not applied; do not match its length against unavailable/current non-CUDA hardware. Validate supported CPU streams with the existing restorer under `preserved_rng()` and carry this validated copy to the trainer. Raw `args.device` may be auto and cannot prove the saved backend; no new checkpoint fields, GPU-topology adapter or cross-device exact-replay promise is added. CPU-only → CUDA and CUDA device-count mismatch are rejected for enabled continuation; CUDA → CPU/non-CUDA is allowed only as explicitly reported non-CUDA recovery, not CUDA continuation. Keep queue-off/DINO RNG handling unchanged.
- [x] **Step 4d — Validate both full encoders and relational center.** In the strict-FIFO probe branch derive the full expected state after the same training setup, including `validate_pretrain_backbone(require_intermediates=...)` and only its actually registered `_pretrain_final_norm.*` aliases. Check saved student and teacher independently: exact complete key sets, Tensor values, exact shapes and finite floating parameters/buffers (backbone, projector and encoder center). Existing backbone/projector-layout helpers and zero-register permissive loading are insufficient. Require the top-level relational `center` Tensor to be finite with exact `[1, D]` shape; scalar or `[D]` broadcasting is invalid. Handle training aliases from the actual encoder structure and reject conflicting saved values for aliases of one live tensor. Do not substitute teacher/model_state_dict for missing student state. Add `test_fifo_resume_rejects_incomplete_or_nonfinite_encoders_and_center`: complete teacher with missing student projector, missing teacher weight, wrong projector shape, NaN/Inf backbone/projector/buffer, missing/scalar/`[D]`/non-finite top-level center, and valid training aliases. Cover zero-register and registered layouts; keep queue-off behavior unchanged.
- [x] **Step 5 — Pin preflight and numerical guards.** Add `test_fifo_preparation_rejects_before_any_probe`, `test_fifo_handoff_is_consumed_once_and_retry_is_clean`, `test_fifo_resume_locks_exact_batch_not_only_loader_length`, `test_fifo_rng_validation_does_not_change_caller_rng` and `test_fifo_temperature_admission_uses_expanded_candidates`. Spy on every actual encoder-constructor lookup used by resume/register, architecture and optimizer preflights (`training.augmentation`, `training.pretrain_recipe`, and the relevant trainer lookup), not only `preflight_pretrain_run`. Add the real CLI-resume test `test_fifo_cli_resume_preparation_precedes_register_probe` in Task 7: a fingerprint conflict or saved-DINO explicit queue option must produce no encoder construction at all; a valid enabled resume must finish preparation before the first register-layout probe. CLI/direct handoff is hashed once; changed resume path/request invalidates reused preparation. Add `test_fifo_resume_rejects_impossible_latest_commit_tail` here: B=4, K=8, g=10 with only two records stamped 8 must fail, as must a nonempty queue lacking its four latest records stamped 9; valid K<B, K>N and refreshed/repeated-image trajectories must pass. Add `test_fifo_epoch_resume_rejects_truncated_history_with_valid_tail`: N=100, B=10, K=64, g=10 with only the last ten keys stamped 9 passes the tail test but must fail the epoch occupancy bound; a valid 64-record state passes. Test N not divisible by B and K greater than N to avoid requiring the dropped remainder or artificial full capacity. Task 5/7 output-safety tests also cover the AdamW and patch corruptions above and compare original log/CSV/checkpoint bytes. Test a float32 temperature admitted at B but rejected at `B + min(K, N-B)`, student and teacher endpoints, while normal temperatures remain accepted. For effective admission and existing validators that read `args.batch_size`, use a shallow namespace copy with only its batch size replaced by the preparation's effective value; never mutate the raw Parameters request before logging. On the trainer side, apply effective batch/capacity only after validating the request binding and retaining requested values.
- [x] **Step 6 — Run green.** `python -m pytest tests/training/test_relational_queue_recipe.py tests/training/test_pretrain_alignment.py -q`. Retain existing same-plan, historical-zero, target-bundle, pairing and temperature tests. Detailed saved RNG validation must catch malformed Python/NumPy/torch states, including the target-aware CUDA cases, without consuming caller RNG. Add `test_fifo_rng_target_matrix`: CPU source without cuda → CPU succeeds; missing/None/empty CUDA state → CUDA fails; count mismatch → CUDA fails; valid matching states → CUDA restores all streams; CUDA-shaped payload → non-CUDA is structurally checked, omitted from the restore copy and reported; malformed provided payload fails even on CPU. Use mocked backend/restore spies for control-flow coverage without a GPU, and record real-CUDA restoration as unverified unless actually exercised on CUDA. Do not claim the mocked branch proves GPU RNG usability.
- [ ] **Step 7 — Conditional commit checkpoint.** Stage the recipe and the two named tests; commit `feat(pretrain): resolve FIFO data and strict resume recipe before probes`.

## Task 4: Extend Relational Distributions and Reuse Loss-Side Statistics

**Files:** Modify `src/otuformer/training/objectives.py:29–83,228–240`; create `tests/training/test_relational_queue_objectives.py`; retain alignment tests.

**Interfaces:**

- `_RelationalStepTargets` in `objectives.py`: `candidates: tuple[Tensor, Tensor]` (`[B+E,D]` each), `teacher_probs: tuple[Tensor, Tensor]` (`[B,B+E]` each), `teacher_stats: dict[str, Tensor] | None` (already averaged by slot). All are detached. Teacher-stat keys are `self_mass`, `other_current_mass`, `history_mass`, `entropy`, `nonself_effective`; student-stat keys are the first three. Unavailable values are omitted, not NaN. Task 6 adds prefixes/suffixes only.
- `_make_relational_step_targets(teacher_global: list[Tensor], history_features: Tensor, center_before: Tensor, temp_teacher: float, *, collect_diagnostics: bool) -> _RelationalStepTargets`; teacher_global is already centered/normalized for this step, history is `[E,2,D]` pre-center float32. Recenter/normalize history once, concatenate current first, compute teacher row-mean subtraction and probabilities once per slot, and optionally compute teacher diagnostics from those logits. Do not center current targets twice.
- Add keyword-only `candidates: Tensor | None = None`, `teacher_probs: Tensor | None = None`, `stats: dict[str, Tensor] | None = None` to `_ssl_loss` while retaining its four original positional arguments and Tensor return. Candidates/targets must be provided together; absent means the original body. `stats` collects detached student mass only from the logits used by that loss.
- Add optional keyword-only `relational_targets: _RelationalStepTargets | None = None`, `student_stats: list[dict[str, Tensor]] | None = None` to `_compute_global_loss` and `_compute_local_loss`. Existing calls/returns and off-path call signatures remain intact; enabled pairs select their matching slot. Aggregate student statistics per contributing pair, not per concatenated unweighted view pool.

- [x] **Step 1 — Add red formula tests.** Use small explicit CPU vectors, B=2 and E=1, to compare the rectangular CE and student gradients against a direct hand formula. Assert current diagonal index i, same candidate order on teacher/student sides, A uses only slot A history and B only slot B, `B+E` not `B+2E`, and all pairing/local means. Require `student.grad` finite/nonzero and no teacher/history/center gradients.
```python
def test_fifo_rectangular_loss_keeps_student_gradient():
    s = torch.tensor([[0.6, 0.8], [0.8, 0.6]], requires_grad=True)
    t = torch.eye(2, requires_grad=True)
    c = torch.cat([t.detach(), torch.tensor([[0.8, 0.6]])])
    teacher_logits = t.detach() @ c.T / 0.1
    p = (teacher_logits - teacher_logits.mean(1, keepdim=True)).softmax(1)
    actual = _ssl_loss(s, t, 0.2, 0.1, candidates=c, teacher_probs=p)
    expected = -(p * (s @ c.T / 0.2).log_softmax(1)).sum(1).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0 and t.grad is None
```

- [x] **Step 2 — Run red.** `python -m pytest tests/training/test_relational_queue_objectives.py -q`; expected unsupported interfaces.
- [x] **Step 3 — Implement targets and scalar losses.** Keep original teacher row-mean subtraction, selected global pair sets and local averaging. Normalization and target construction are not extra model forwards. Empty eligible history is the same formula with E=0, not a forced additional branch; capacity zero never constructs targets.
- [x] **Step 4 — Add diagnostic/parity tests.** `test_fifo_empty_history_matches_loss_gradient_and_rng`, `test_fifo_current_history_share_pre_step_center`, `test_fifo_probability_partitions_and_nonself_perplexity` and `test_fifo_diagnostics_are_detached_and_log_step_only`. Per row, self + other-current + history = 1; empty history mass=0. Teacher full entropy uses stable logs; conditional-nonself perplexity is `exp(-sum(r*log(r)))`, average per-row perplexities, with `0log0=0` and nonself logits used when self mass rounds to 1. No nonself candidates → unavailable, not NaN. No locals → unavailable local fields. No diagnostics request → no extra entropy/stat reductions, host conversions or feature dumps. Teacher stats are computed once per slot and reused for both branches.
- [x] **Step 5 — Run green.** `python -m pytest tests/training/test_relational_queue_objectives.py tests/training/test_pretrain_alignment.py -q`. E=0 matches old numerical/gradient/RNG results, and enabled diagnostics on/off leave loss and gradients unchanged.
- [ ] **Step 6 — Conditional commit checkpoint.** Stage objectives and the new test; commit `feat(pretrain): extend relational targets across current and FIFO candidates`.

## Task 5: Integrate Successful-Step State and Epoch Recovery

**Files:** Modify `src/otuformer/training/pretrain_loop.py:127–318,487–640,658–859,981–1065`, `src/otuformer/utils/checkpoint.py:193–234,322–374`; extend `tests/training/test_masked_pretrain.py`, `tests/training/test_dino_pretrain.py`, `tests/test_checkpoint_utils.py`, `tests/test_extractor.py`, `tests/test_cam.py`, `tests/test_export.py`, `tests/test_extract_records.py`, `tests/test_output_safety.py` and the fine-tune checks in `tests/training/test_pretrain_alignment.py`. Reuse `_run_tiny_pretrain` in its existing test file, not by importing another test module.

**Interfaces:**

- Add local `_unpack_pretrain_batch(batch: object, *, queue_enabled: bool) -> tuple[list[Tensor], tuple[str, ...] | None]` in `pretrain_loop.py`, used at both startup sample (`sample_batch = next(iter(loader))`) and iteration consumption. Off list behavior stays unchanged; enabled requires matching keyed mapping, not silent list fallback.
- `run_pretrain(args, pretrain_recipe=None)` remains public shape-compatible. Consume the prepared handoff first; direct-core entry performs Task 3 preparation before probes if CLI did not provide it. An enabled supplied recipe must reference the same preparation/settings/request; mismatches raise before outputs.
- Before DataLoader/schedules/full effective-config use, take effective batch size from preparation, preserving raw request in logs. For enabled saves write `args.batch_size` effective plus `args.requested_batch_size` (None for omission) and explicitness; write effective `args.relational_queue_size`, `args.requested_relational_queue_size` and explicitness. Do not impose new batch inheritance on queue-off.
- For active FIFO, trainer uses Task 3's prepared resolved_device and rng_state_to_restore, including the actual one-time pending-RNG restoration before the resumed epoch; it must not bypass validation by restoring raw checkpoint rng_state again. Queue-off/DINO keep their existing device/RNG path. Trainer reconstructs Task 2 state on the resolved device, once, using the actual output dimension. Enabled resume strictly loads both validated student and teacher after the same training-alias setup; it never uses relational's legacy `strict=False` or a missing-student fallback. The validated top-level `[1,D]` center keeps its existing restore role; preflight must reject broadcasting rather than relying on `copy_()`. Config bundle is exactly `{'version': 1, 'capacity': K}`; top-level queue state uses Task 2 codec. Disabled saves omit both bundles and any fake enabled version marker.

- [x] **Step 1 — Add red tiny-loop tests.** In the existing helper's file add `test_fifo_step_reuses_snapshot_and_commits_pre_ema_features`, `test_fifo_aborted_step_does_not_mutate_queue_or_save_epoch`, `test_fifo_disabled_preserves_legacy_trajectory` and `test_fifo_empty_history_and_cross_epoch_retention`. Spy on snapshots/commit, teacher output storage and saved state rather than adding a production debug callback. Cover K<B and repeated keys.
```python
def test_fifo_epoch_checkpoint_uses_enabled_bundles(tmp_path, monkeypatch):
    obs = _run_tiny_pretrain(tmp_path, monkeypatch,
        relational_queue_size=2, relational_queue_size_explicit=True)
    saved = torch.load(obs['out_dir'] / 'SSL_latest.pth',
        map_location='cpu', weights_only=False)
    assert saved['config']['relational_queue_config'] == {'version': 1, 'capacity': 2}
    state = saved['relational_queue']
    assert state['completed_steps'] == saved['iteration']
    assert state['features'].shape == (2, 2, 8)
```

Existing saves place `SSL_latest.pth` directly under `obs['out_dir']`; no checkpoint subdirectory is added. This smaller-local helper retains its documented CPU exception. Its `batch_size=4`, `device='cpu'` and `log_every_n_steps=1` are currently passed explicitly, so passing the same keys through `**settings` raises a duplicate-key error: make those helper defaults overridable with `settings.pop('batch_size', 4)`, `settings.pop('device', 'cpu')` and `settings.pop('log_every_n_steps', 1)`. Crop sizes and max_epochs already live in settings. Add ordinary variants with `global_crop_size=64`, `local_crop_size=64`, `device='auto'`; the tiny encoder's native image size is 64. This is a local test-helper adjustment, not production default/device changes.

- [x] **Step 2 — Run red.** `python -m pytest tests/training/test_masked_pretrain.py -q -k fifo`; expected absent keyed/snapshot/checkpoint integration.
- [x] **Step 3 — Integrate in existing order.** At pre-step completed count g, take one eligible snapshot. Cache cloned detached A/B pre-center outputs from the already-selected relational teacher forwards; keep center update in target preparation before loss/finite checks. All global/local terms share this snapshot and `center_before`. Commit cached observations only after finite checks, backward, clipping, optimizer and existing EMA/patch-center updates succeed; observation step=g and queue.completed_steps becomes g+1. Keep the loop's global_step increment at the end of the iteration (after instant logging); do not shift schedule/CSV indices. Until that increment, use captured g for age/log selection, never take another snapshot; at the next iteration or epoch save both completed counts equal g+1. Do not insert from startup sample, evaluation, probes or a second teacher forward. Retain queue across epochs; count refresh/eviction at successful commit, not candidate-read time.
- [x] **Step 4 — Add strict controlled resume evidence.** `test_fifo_epoch_resume_matches_uninterrupted_cpu` compares an uninterrupted run with an epoch-saved stop/resume under identical total schedule, CPU and workers=0. Stop only after a real scheduled save, not by shortening max_epochs and changing schedules. Compare student, teacher, optimizer, centers, schedule/progress, queue keys/features/ages and saved RNG; exercise a sparse save interval and two resume hops. Require both `SSL_epoch_*.pth` and `SSL_latest.pth` to carry equivalent state. Resumed teacher must not be overwritten with student weights.
- [x] **Step 5 — Add failed-preflight and consumer tests.** Queue metadata attached to active DINO identity must fail through shared identity resolution; read-only relational consumers must not require dataset/RNG/queue reconstruction. Parameterize the existing injected-relational-bundle tests in `test_extractor.py`, `test_cam.py` and `test_export.py` with a valid enabled queue checkpoint; extend `test_extract_provenance_preserves_ssl_identity` in `test_extract_records.py` and existing fine-tune initialization/save checks in `test_pretrain_alignment.py`. No reader may hash files or allocate a queue just to load weights. Verify fine-tune saves do not copy top-level queue state. The shared checkpoint guard rejects queue config or state on active DINO before encoder construction, without importing training/queue/dataset; relational data/RNG/queue-codec validation remains in the training recipe, not read-only consumers. In `tests/test_output_safety.py`, add enabled malformed-payload/conflict cases using the existing `keep.txt` sentinel convention, including cleared AdamW state, malformed/negative second moments, missing predictor, wrong iBOT-head shape, invalid patch center, impossible latest-commit tail, truncated epoch occupancy, target-incomplete CUDA RNG and all Step 4d encoder/top-level-center corruptions. Assert sentinel and original log, CSV and checkpoint bytes remain unchanged; exercise actual CLI resume rejection and direct-core preflight, without stubbing away the validator under test. Keep queue-state assertions in training tests rather than duplicating a weaker output fixture there. For non-finite loss and gradient injections catch existing ValueError: queue unchanged, no new epoch save; center rollback is not asserted.
- [x] **Step 6 — Run the focused real gates.** `python -m pytest tests/training/test_masked_pretrain.py tests/training/test_dino_pretrain.py tests/test_checkpoint_utils.py tests/test_extractor.py tests/test_cam.py tests/test_export.py tests/test_extract_records.py tests/test_output_safety.py tests/training/test_pretrain_alignment.py -q`. Cover relational patch `none/consistency/masked-feature/ibot`, nonzero shared KoLeo, ordinary/weak teacher with appropriate center populations, local crops and no locals, and all saved pair modes without changing patch inputs or gradient/EMA order. Keep ordinary loops on auto/native-size views; smaller-local patch matrix and exact replay remain explicitly CPU with policy reasons. Record selected device.
- [ ] **Step 7 — Conditional commit checkpoint.** Stage only the named trainer/checkpoint files and affected integration/consumer tests explicitly; commit `feat(pretrain): persist FIFO only after successful updates and restore exact state`.

## Task 6: Add Resolved Logs and Backward-Compatible Instant Metrics

**Files:** Modify `src/otuformer/training/reporting.py:44–115,130–239` and `pretrain_loop.py:234–254,859–944`; extend `tests/training/test_ssl_metrics.py` and `test_masked_pretrain.py`.

**Interfaces and chosen column names:**

- Preserve a literal `_PRETRAIN_PRE_FIFO_FIELDS` snapshot of the current 43-column header; define `_PRETRAIN_RELATIONAL_QUEUE_FIELDS` and append to current pretrain fields. Add the 43-column header to the recognized predecessor whitelist, alongside the existing legacy/v0.7.0 headers. Do not create a generic migrator or alter epoch/finetune headers.
- `relational_queue_capacity` records the resolved fixed recipe capacity K on every enabled row, including inherited capacity (e.g. 128), never raw omitted-request None. Here the design's requested capacity is the manually configured logical limit, not occupancy; the raw invocation request and source stay separate in the resolved log. Queue-off/DINO queue cells stay blank.
- Queue columns: `relational_queue_capacity`, `relational_queue_occupancy`, `relational_queue_eligible`, `relational_candidate_count`, `relational_history_age_mean`, `relational_history_age_max`, `relational_queue_refreshed`, `relational_queue_evicted`. Occupancy/eligibility/age refer to the pre-step loss snapshot; refresh/eviction refer to the successful post-step commit.
- For each suffix `global`/`local`, teacher columns are `relational_teacher_{self_mass,other_current_mass,history_mass,entropy,nonself_effective}_{suffix}`; student columns are `relational_student_{self_mass,other_current_mass,history_mass}_{suffix}`. Expand braces in the field list; do not use them as literal CSV names. Unavailable values are empty cells, not fallback zeros; available empty-history mass is exactly 0.
- At existing log steps, the trainer averages Task 4's detached loss-owned tensors and passes scalar fields to the existing `InstantMetricsLogger.log(**kwargs)`. Reporting never recomputes similarities/losses or runs a model. Determine `collect_diagnostics` for the upcoming successful-step log using existing global-step timing, not an extra logging schedule.
- Trainer emits `Relational queue config:` once next to `Relational target config:` after resolution: JSON fields `applicability` (`active`, `disabled`, `inactive`), `requested_capacity`, `effective_capacity`, `capacity_source`, `requested_batch_size`, `effective_batch_size`, `batch_size_source`. DINO omitted has inactive/None capacity, not a false active zero; relational off has effective 0. Raw `Parameters:` stays the requested invocation. Record is display-only and never a training input.

- [x] **Step 1 — Add red actual-migration tests.** `test_fifo_metrics_resume_current_43_column_log` constructs the old literal header and rows, opens `InstantMetricsLogger(path, mode='pretrain')` on that nonempty file (the actual constructor auto-detects append/migration; there is no `resume` argument), logs a new row and checks all old values preserved, new historical cells blank and only the new row populated. Test every supported legacy header, migration idempotence, plot continuity, and failure preserving original bytes for unknown/duplicate headers and injected replacement failure. A mere header-length assertion is insufficient.
The migration assertions must exercise construction and append, using the existing `_write` helper in this test file:

```python
def test_fifo_metrics_resume_current_43_column_log(tmp_path):
    import csv
    path = tmp_path / 'instant.csv'
    old_header = (
        'iteration epoch step loss global_loss local_loss mask_loss lr teacher_temp '
        'grad_norm feature_std embedding_norm_mean cls_token_norm_mean teacher_center_norm '
        'cosine_similarity patch_loss_raw patch_loss_weighted patch_loss_fraction_of_total '
        'requested_mask_ratio selected_patch_ratio actual_mask_ratio_mean actual_mask_ratio_min '
        'actual_mask_ratio_max prototype_perplexity active_prototype_ratio assignment_entropy '
        'max_prototype_occupancy patch_center_norm ssl_method dino_centering koleo_weight '
        'cls_center_norm global_loss_weighted local_loss_weighted cls_assignment_entropy '
        'cls_prototype_perplexity cls_active_prototype_ratio cls_max_prototype_occupancy '
        'koleo_loss_raw koleo_loss_weighted koleo_status koleo_eligible_samples koleo_skipped_samples'
    ).split()
    assert len(old_header) == 43
    _write(path, old_header, [['1'] * len(old_header)])
    InstantMetricsLogger(path, mode='pretrain').log(iteration=2, relational_queue_capacity=128)
    rows = list(csv.DictReader(path.open(newline='')))
    assert all(rows[0][name] == '1' for name in old_header)
    assert rows[0]['relational_queue_capacity'] == ''
    assert rows[1]['relational_queue_capacity'] == '128'
```

This literal fixture is independent of the new production constant; importing that constant alone could let a simultaneous schema mistake pass. Use the test's existing reporting-module import for the logger.

- [x] **Step 2 — Run red.** `python -m pytest tests/training/test_ssl_metrics.py -q -k fifo`; expected unrecognized old schema or absent columns.
- [x] **Step 3 — Wire logger and log ownership.** Use existing atomic migration and existing loss/metric plots; do not add a chart family. Preserve finetune and enhanced epoch schemas. Blank queue-off/DINO columns and no-local statistics; do not compute queue statistics merely to fill columns. Mean/max age unavailable at E=0, refresh distinct from insertion, resolved fixed recipe capacity distinct from occupancy. Assert inherited enabled rows have `relational_queue_capacity == 128` even when the raw invocation omitted the option.
- [x] **Step 4 — Add four resolved-log integration assertions.** `test_fifo_resolved_log_separates_requested_and_effective` covers fresh enabled, inherited resume (raw batch/capacity 32/None, effective saved 64/128), explicit 0 on a compatible disabled run and omitted DINO inactive. Assert one trainer record and no extra probability/teacher forward. CLI-stub absence is tested in Task 7, not implemented by printing during preflight.
- [x] **Step 5 — Run green.** `python -m pytest tests/training/test_ssl_metrics.py -q` and `python -m pytest tests/training/test_masked_pretrain.py -q -k 'fifo or schema or migrated'`. Check probability groups, perplexity and log-step-only computation against Task 4 tests; old loss plots retain valid history and no required numeric fields are silently missing on enabled log rows.
- [ ] **Step 6 — Conditional commit checkpoint.** Stage the two source/two test files and commit `feat(reporting): log FIFO coverage with safe pre-FIFO CSV migration`.

## Task 7: Expose Source-Aware CLI, Schema and Parameter Cards

**Files:** Modify `src/otuformer/cli/pretrain.py:170–181,519–533,626–665,718–727`, `src/otuformer/cli/constraints.py:75–79,492–530`, `src/otuformer/cli_schema.py:390–425,619–714`; extend `tests/test_cli_smoke.py`, `tests/test_cli_schema.py`, `tests/test_cli_startup.py` and `tests/test_output_safety.py`; rendered-help assertions use `tests/test_docs.py`.

**Interfaces:**

- Add `relational_queue_size: int | None = typer.Option(None, '--relational-queue-size', min=0, ...)` to the existing relational panel; declared None is not the effective fresh zero. Pass request and `relational_queue_size_explicit` into core. Obtain both queue and `batch_size_explicit` with `_source_is_commandline(ctx.get_parameter_source(...))`; default batch_size remains 32, not a fabricated None sentinel.
- Before **any** model-constructing preflight on enabled resume, including the current early `preflight_pretrain_resume()` → `augmentation._resume_register_plan()` and later `preflight_pretrain_run()`, call Task 3 preparation using the already-loaded resume payload. The model-free applicability check must precede that early resume preflight even for rejected requests (including saved DINO plus explicit queue zero); only positive capacity performs data preparation. Move only the enabled resume probe behind preparation, construct the request namespace early enough, and preserve queue-off's original model-probe order. Hand the same preparation/settings/recipe to `run_pretrain`; add preparation after raw Parameters display with the other transient attributes. Enabled `--overwrite` protection uses the prepared selected paths, not a second train-data resolution; separate visualize-data protection and queue-off behavior stay unchanged. The full SSL resolver stays after architecture preflight, receives preparation and validates effective temperatures without rehashing.
- In static validation, register min/applicability in the existing constraint owners. Read-only schema/card never reads a checkpoint or imports torch/training. Omitted queue on resume and omitted batch_size on resume are `omitted-conditional` where saved method/queue state is unknown; explain positive-capacity inheritance, absent-bundle disabled semantics, and enabled-only batch inheritance. Explicit DINO plus any explicit queue value (including 0) is rejected. Preserve requested argv omission; never emit an inferred effective value as a user request. No schema-version bump.

- [x] **Step 1 — Add red parser/card tests.** `test_fifo_cli_option_and_explicit_sources`, `test_fifo_cli_prepares_once_before_probe_and_keeps_outputs`, `test_fifo_schema_conditional_resume_and_argv`, and `test_fifo_cli_stub_does_not_emit_resolved_config` and `test_fifo_cli_resume_preparation_precedes_register_probe`. Cover omitted/0/positive/negative/fractional requests, source-aware omitted batch32 versus explicit32, DINO rejection, unknown checkpoint method static deferral, and a legacy installed schema with no queue field. Assert `Parameters:` remains raw and no trainer-resolved record appears when the trainer is stubbed, mirroring the existing target-config test.
```python
def test_fifo_static_dino_zero_is_not_inactive():
    with pytest.raises(ValueError, match='relational'):
        validate_pretrain_ssl_applicability(
            {'ssl_method': 'dino', 'relational_queue_size': 0},
            explicit_options=frozenset({'ssl_method', 'relational_queue_size'}),
        )
```

Use the existing schema fixture for required input paths and its installed-schema stubs in the conditional-card tests; assert `card['provenance']['batch_size'] == 'omitted-conditional'` and that built argv omits both unrequested queue/batch switches.

- [x] **Step 2 — Run red.** `python -m pytest tests/test_cli_smoke.py tests/test_cli_schema.py -q -k fifo`; expected missing option/metadata.
- [x] **Step 3 — Implement callback, constraints and card provenance.** Preserve named callback callers and panel order. Reject conflicts before `prepare_output_dir` or logger migration; no early whole SSL preflight and no second checkpoint load. Extend `test_training_selected_images_survive_cleanup` in `tests/test_output_safety.py` with an enabled pretrain case whose selected image remains within the declared root but lies under the overwritten output (root=tmp_path and image=tmp_path/out/pixels.png, with a CSV selecting it and `--batch-size 1` so a complete batch exists). It must pass G1 path validation and then fail output protection; an escaping-link/G1 failure alone does not prove prepared-selection protection. Spy that training selection is prepared once and protected without re-resolution; preserve existing queue-off, visualize-data and downstream-input cases. No models, outputs, hashing or downloads during help/schema/card generation.
- [x] **Step 4 — Run green and startup guard.** `python -m pytest tests/test_cli_schema.py tests/test_cli_startup.py -q` and `python -m pytest tests/test_cli_smoke.py -q -k 'fifo or relational'`; also run `python -m pytest tests/test_output_safety.py tests/test_docs.py -q`. Inspect the installed console entrypoint `otuformer pretrain --help` and `otuformer --help` for the option, default/resume distinction and relational grouping; this package has no `otuformer.__main__` module. Automated rendered-help assertions use the existing `tests/test_docs.py` `help_output()` / `compact()` path to tolerate Rich wrapping; do not add a module entrypoint or infer the root cause of earlier environment-specific failures. Actual callback and generated argv must match; subprocess import guards retain absence of torch/pandas/training in static discovery.
- [ ] **Step 5 — Conditional commit checkpoint.** Stage the three source/three test files and commit `feat(cli): expose opt-in FIFO with conditional resume provenance`.

## Task 8: Synchronize Documentation, Review Coverage and Run Acceptance Gates

**Files:** Modify `docs/superpowers/specs/2026-03-29-otuformer-design.md`, `docs/superpowers/specs/2026-11-03-relational-pretrain-targets-design.md`, `docs/superpowers/plans/2026-11-04-relational-pretrain-targets.md`, this plan and its FIFO spec; modify the training-ownership design only where the actual boundary changes. Update `README.md`, `README.cn.md`, `docs/commands/pretrain.md`, `docs/commands/pretrain.cn.md`, `skills/otuformer-workflow/SKILL.md`, `skills/otuformer-workflow/references/workflow.md` and `references/troubleshooting.md`; test `tests/test_docs.py`, `tests/test_workflow_skill.py`. Update other workflow references only if they actually describe affected behavior.

- [x] **Step 1 — Add failing documentation/skill assertions.** Pin discover-installed-schema-first behavior; older installed CLI with no queue option must not receive invented argv. Assert option/default/inheritance and the queue-capable-version resume warning in both command-language versions, with user execution approval unchanged. Keep master design linking to the FIFO spec, not command docs. Use existing doc/skill contract tests, not a second generated CLI schema.
```python
def test_fifo_user_command_docs_state_opt_in_and_recovery():
    from pathlib import Path
    for suffix in ('md', 'cn.md'):
        text = Path(f'docs/commands/pretrain.{suffix}').read_text()
        assert '--relational-queue-size' in text
        assert 'SHA-256' in text
    spec = Path('docs/superpowers/specs/2026-11-10-relational-fifo-design.md').read_text()
    assert 'queue-capable' in spec
```

The existing doc-suite root fixture should resolve these paths, not the caller's arbitrary cwd; add semantic assertions for each language's default/recovery wording using the existing bilingual checks.

- [x] **Step 2 — Run red.** `python -m pytest tests/test_docs.py tests/test_workflow_skill.py -q`; new assertions must fail only on missing FIFO documentation.
- [x] **Step 3 — Update bounded documentation surfaces.** Add dated supersession notes to the old relational no-queue design/plan without rewriting history. Explain pre-center paired slots, current anchor, fixed versus effective capacity, approximate storage `2*M*D*4`, compute growth, selected-reference uniqueness/alias limits, full-data startup hashing, center staleness versus encoder staleness and strict epoch recovery. Clarify that old binaries may ignore enabled metadata and are not promised safe for enabled pretrain resume; read-only embedding compatibility is different. Keep English-only development docs and bilingual user docs. Mark implementation complete only with actual test evidence. If locally maintaining ignored `ref/morphotu-dense_morphology-20261019.md` §5.4, annotate only implemented behavior after it exists, do not stage it or treat it as public CLI documentation.
- [x] **Step 4 — Run focused feature boundary gates.** Execute `python -m pytest tests/training/test_relational_queue.py tests/training/test_relational_queue_recipe.py tests/training/test_relational_queue_objectives.py tests/test_training_dataset.py tests/training/test_pretrain_alignment.py tests/training/test_masked_pretrain.py tests/training/test_dino_pretrain.py tests/training/test_ssl_metrics.py tests/test_checkpoint_utils.py tests/test_extractor.py tests/test_cam.py tests/test_export.py tests/test_extract_records.py tests/test_cli_schema.py tests/test_cli_smoke.py tests/test_cli_startup.py tests/test_output_safety.py tests/test_docs.py tests/test_workflow_skill.py -q --durations=25`. Extend consumer-specific paths where Task 5 identified additional guards; do not claim extract/export/CAM/fine-tune compatibility from only one reader. Retain historical batch-only multi-hop/read-only compatibility tests.
- [x] **Step 5 — Run the required full gate on the final candidate.** `python -m pytest --durations=25`. This feature changes shared step mechanics and checkpoint/recovery identity, so focused passes alone do not accept compatibility. Record interpreter/dependencies/source import path, baseline SHA/dirty state, exact command/exit status, selected device and evidence path under a new ignored `runs/fifo_acceptance_<date>/` directory. Use a valid comparable baseline or collect one necessary baseline, not repeated full runs per local edit. If blocked or failing, keep acceptance open and report it; do not weaken assertions or hide slow tests.
- [x] **Step 6 — Measure enabled data preparation cost without training.** Once plan execution/data access are explicitly authorized, call Task 1 preparation on the actual approved selected dataset without model creation or a training loop. Record selected image count, total bytes (and GB with the divisor stated), elapsed seconds, storage/cache context and exact invocation. No cost estimate counts as measured evidence; use a new run artifact, not the existing diagnostic spike outputs. This is a startup-cost check, not permission for capacity sweeps or paired-seed training.
- [x] **Step 7 — Record correctness separately from research.** Update spec/plan with task and gate results plus limitations; leave any unrun tests/measurements open. Research capacities `0 / 32 / 128`, paired seeds, larger grouped data and dense-morphology evaluation remain a separately authorized follow-up, with the already-fixed teacher temperatures/augmentation/pairing/weights and full favorable/unfavorable metrics. No quality or speedup/default claim follows from correctness tests.
- [ ] **Step 8 — Conditional commit checkpoint.** With explicit commit permission, stage the exact changed documentation/skill/test paths (not `git add -A`, not ignored runs/ref files) and commit `docs(pretrain): document FIFO contracts and verified recovery boundaries`. No automatic push, tag or release.

## Acceptance Map and Handoff

| Spec boundary | Plan owner |
| --- | --- |
| §1 scope/defaults/approved G1/G2; §2 adaptations, not research claims | Global Constraints; Tasks 3, 7–8 |
| §3 paired candidates, self anchor, E=0/zero parity and temperature bounds | Tasks 3–5 |
| §4 unique refresh, eviction/age, success timing, failure and epoch-only persistence | Tasks 2, 5 |
| §5 selected paths, symlinks, hashing/data lock, keyed workers and zero behavior | Tasks 1, 3, 5 |
| §6 public request/applicability, conditional inheritance and one-time preparation | Tasks 3, 7 |
| §7 closed queue metadata, strict recovery, historic absence and downstream identity | Tasks 2–3, 5 |
| §8 loss-owned statistics, trainer-only resolved log, CSV migration | Tasks 4, 6–7 |
| §9 owners, bilingual user docs, English development docs and installed-schema skill | File Map; Tasks 7–8 |
| §10 acceptance and separate research; §11 approval boundaries; §12 evidence | Tasks 1–8; this handoff |

**Plan self-review — 2026-11-11:** Completed by the author, not an independent reviewer. Checked spec §§1–12 against the acceptance map, all five Review Focus cases against named task tests, exact existing helper/constructor/signature seams, type/field consistency and task proportion (about 1.1× the spec by text length). Corrected the eight-field codec count, logger's actual no-resume-argument API, independent literal 43-column fixture, real checkpoint path, direct-core/CLI explicitness, request-bound transient handoff, effective-batch temperature forwarding, prepared overwrite protection and unchanged global_step/log indexing. All eight Python assertion examples parse; their production behavior has not been executed because FIFO does not exist yet. Local links, whitespace and fences pass. The planning-only documentation suite returned **42 passed**. A follow-up plan review corrected the console-script help commands, assigned output-safety cases and gates to their existing test owner, pinned capacity cells to the resolved fixed K, and required an exact keyed-envelope key set. A further source audit covered the earlier register-layout resume probe, FIFO-only AdamW moment and independent patch-state preflight, and the batch-aware latest-commit tail invariant; these are bounded plan/spec clarifications, not implemented assertions. The next audit added nonnegative AdamW second moments, a complete-epoch occupancy lower bound and target-aware CUDA RNG presence/count/restore rules (including explicit non-CUDA recovery limits), without adding checkpoint fields or changing queue-off behavior. The final encoder-state audit specified full independent student/teacher keys, exact shapes and finite tensors, actual training-alias handling, exact top-level relational-center shape, strict enabled loading and CLI/direct-core output-preserving corruption tests. At planning time every implementation task was unchecked and the plan was unapproved for execution; the execution record below supersedes that status.

**Evidence limits:** No FIFO implementation, full production test gate, real-data hashing measurement, accelerator acceptance, independent plan review, research training, commit or release occurred during planning. Task 8 describes future required evidence, not results. If that gate fails or a prerequisite is unavailable, record it as open rather than checking completion.

**Execution handoff:** Ask the user to review this plan and separately authorize execution and choose an execution method. Recommend native/inline execution for these tightly coupled recipe/objective/loop interfaces to avoid multiple writers and context handoff cost; use subagent-driven execution only if requested. A plan review does not imply permission to run research training or publish.

## Execution record — 2026-11-11

**Environment:** Python 3.11.15, torch 2.12.1, timm 1.0.27, otuformer 0.12.0 from `src/otuformer` (editable checkout), baseline `3d69d1812a1b5d6941c49bf070f1037c1304d83d` with this feature as uncommitted work. Selected device for every training test: CPU (documented smaller-local exception); no accelerator claim.

**Task-level results.** Tasks 1-8 are implemented in the owners named by the file map and have focused tests:

| Owner | Evidence |
| --- | --- |
| `training/dataset.py` | `tests/test_training_dataset.py -k fifo` (5 new tests: preparation identity, alias/`link/..`, content/order/relocation, rejections, keyed envelope RNG parity). |
| `training/relational_queue.py` (new) | `tests/training/test_relational_queue.py` (36 pure state/codec tests). |
| `training/pretrain_recipe.py` | `tests/training/test_relational_queue_recipe.py` (16 model-free request/lock/RNG tests) plus the resume-corruption matrix in `tests/training/test_masked_pretrain.py`. |
| `training/objectives.py` | `tests/training/test_relational_queue_objectives.py` (8 formula/parity/diagnostic tests). |
| `training/pretrain_loop.py`, `utils/checkpoint.py` | FIFO loop tests in `tests/training/test_masked_pretrain.py` (bundles, one snapshot per step, aborted step, disabled parity, cross-epoch retention, exact CPU epoch resume, AdamW/patch/encoder/center corruption) and the DINO queue-metadata guard in `tests/test_checkpoint_utils.py`. |
| `training/reporting.py` | `tests/training/test_ssl_metrics.py` (43-column migration, unknown/duplicate rejection, metric helper) and the resolved-log test in `test_masked_pretrain.py`. |
| CLI/schema | `tests/test_cli_smoke.py` (option/explicitness, preparation-before-probe ordering, conflict rejection with zero encoder construction, conditional provenance/argv), `tests/test_cli_schema.py`, `tests/test_output_safety.py`. |
| Docs/skill | `tests/test_docs.py`, `tests/test_workflow_skill.py`. |

**Gates.** Focused feature gate: 1117 passed, 1 skipped. Full gate, `python -m pytest --durations=25`, exit status 0: **1869 passed, 1 skipped in 806.71s**. Evidence: `runs/fifo_acceptance_2026-11-11/` (environment capture, `full_gate.log`, `preparation_cost.json`).

**Enabled data-preparation cost (measured, no training).** `prepare_relational_data(None, "examples/Epidorcus/images", batch_size=32)` on the shipped example set: **230 selected images, 6,789,679 bytes ≈ 0.0068 GB (0.00632 GiB), 0.053 s, ≈4375 images/s**, SHA-256 digest prefix `45b22c104877e58e`. This is a local-checkout startup-cost measurement, not a large-corpus estimate; hashing cost grows linearly with selected bytes.

**Open items.** No commit, tag, push or release was performed. No research training (capacity `0 / 32 / 128`, paired seeds, larger grouped data, dense-morphology evaluation) was run: that remains separately authorized work, and nothing here is a quality, speed or default claim. CUDA RNG restoration is covered only by mocked control-flow tests — real CUDA continuation is unverified on this host.
