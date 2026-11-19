# OTU-Former Workflow Execution Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native inline execution. Use subagent-driven development only if the operator explicitly selects it. Execute tasks in dependency order. Complete and validate one task, then commit that task's explicitly staged changes with a Conventional Commit before starting the next task. This standing authorization covers only those task-sized implementation commits. Any additional/final/consolidation commit after the planned tasks requires separate operator approval. This plan does not authorize implementation, profiling, training, any commits, push, versioning or release until the operator explicitly approves execution.

**Goal:** Reduce demonstrated repeated work across pretrain, finetune and six downstream commands while preserving checkpoint, CLI, logging and artifact contracts.

**Architecture:** Keep command orchestration and compatibility entry points in place, move only reusable calculations into their existing domain owners, and pass invocation-private prepared state through the established handoff patterns. Each optimization has a focused correctness gate and a separate measured-benefit gate; close candidates with no demonstrated payoff. Execute the plan in evidence-priority order, beginning with C4, C3 and A before B and downstream work.

**Tech Stack:** Existing Python, PyTorch, NumPy, pandas, SciPy, Typer and pytest. No new dependency, generic trainer, public task API, or worker implementation.

**Spec:** [Unified workflow execution efficiency design](../specs/2026-11-18-workflow-execution-efficiency-design.md), together with the [master design](../specs/2026-03-29-otuformer-design.md) and [development testing policy](../specs/2026-11-07-development-testing-policy.md).

**Date:** 2026-11-19. **Implementation baseline:** `2d37064bcba08235dbe67637f665835ae0eb36f9`; the unified design is an untracked, operator-edited draft at plan authoring. Pin the exact pre-change commit and record dirty state before each implementation task. Do not use current uncommitted design files as implementation baseline inputs. Existing source and test changes not authored by a task must be preserved.

**Authorization boundary:** This plan was requested after design review. It is not approval to execute it. All steps are unchecked. Wait for explicit plan approval and execution-method selection before code edits, microbenchmarks, profiling, training, or commits. Once approved, each completed task is to be committed immediately using explicitly staged paths; that does not authorize any extra commit after all planned tasks are complete, which requires separate operator approval.

## Global Constraints

- Preserve CLI defaults, `training.trainer` import-and-call compatibility, checkpoint/CSV schemas, artifact meaning, and existing read-only checkpoint resolution.
- Do not alter objectives, masks, sampling policies, batch sizes, pair weighting, temperature, update order, checkpoint layout, or accelerator policy except for the specifically accepted fixed-tree monophyly correction and sampled evaluation decode boundary.
- No generic trainer, fast-mode flag, new dependency, new workers, finetune `persistent_workers`, or `drop_last` change.
- Keep private invocation state out of saved args, `Parameters:`, checkpoints, and CSV artifacts.
- Every task uses a focused red/green test cycle. Do not repeatedly run unrelated suites. Do not default-deselect tests. The full refactor gate applies to cross-command ownership moves as specified by the testing policy.
- Correctness and performance are separate acceptance decisions. A candidate without measurable benefit or with meaningful regression is omitted; do not claim performance on an unavailable or silently-fallback device.
- Preserve prior user data, output-protection checks, error handling, and unrelated worktree changes. No destructive git operations.
- Development documents remain English-only. Update affected bilingual user docs, CLI help, README and workflow skill guidance only in the tasks whose user-visible semantics change.
- The six E6 smoke nodes belong only to standalone Task 0, which repairs test fixtures/monkeypatch signatures and has an independent commit boundary. Task 0 was proposed as Task 3a and is deliberately numbered separately to make that boundary explicit. It must pass before Task 3b or any pretrain handoff/loader-preflight source change; Tasks 4 and 16 only require and verify that gate. No production check may be weakened to make them pass.
- C6 is conditional: if its existing scan is below 1% of representative real step time, close it without code changes. If representative timing is not authorized, it may instead be closed as `not pursued: measurement not authorized`; lack of authorization is not a failed task. C5/C7 should be folded into nearby work only when the relevant code is already touched.
- Every performance gate uses the §8 evidence shape: exact argv, data/source hashes, elapsed `seconds`, instrumentation, requested/resolved device, and runtime/fallback state during the window; use paired windows and report median/spread. Mark unavailable fields `not measured` with reason; isolated microbenchmarks do not satisfy the paired gate.
- After execution approval, each completed task is committed before starting the next, using a Conventional Commit subject (`feat:`, `fix:`, `chore:`, `test:`, or `docs:`) and explicitly staged task paths only. This per-task authorization does not cover extra or post-plan commits; obtain separate operator approval for those. Include both design and plan in a task commit only if they are that task's authorized changes and the operator has approved staging them.
- For every §8 performance gate, if measurement authorization is absent, record `not measured: authorization pending`, keep the performance claim open, and do not block correctness acceptance or task closure on that unavailable measurement. Do not present prior microbenchmarks as acceptance evidence.

## Review Focus

1. Evaluation CSV missing, malformed, ambiguous, edited mid-run, or changed between resume invocations: one startup warning disables periodic evaluation, logger/curve artifacts remain as before, and the next invocation sees edits (Task 3b; docs Task 3c).
2. A sampled non-prefix cohort with partial labels: tree leaf IDs and labels remain aligned; monophyly is measured on the fixed output tree and intentionally differs from the characterization value (Task 11).
3. Resume payload path changes, stale path-bound manifests, explicit `None`, and failed preflight under overwrite: private handoff reloads correctly and old outputs survive (Task 4).
4. Irregular iBOT masks, two teacher global views and head call order: only selected rows reach token heads, center updates retain exactly the selected pre-EMA logits, RNG is unchanged (Task 6).
5. Backend-specific finite/overflow, MPS lack of float64, real-step synchronization share, and unavailable device kernels: safety remains enforced and unsupported optimizations stay on their original path (Tasks 1, 7).

## File Ownership and Interfaces

| Owner | Planned responsibility |
| --- | --- |
| `training/pretrain_loop.py`, `training/reporting.py` | Evaluation handoff consumption, selected-before-decode work, step-local C optimizations, preserve direct-call seams. |
| `training/pretrain_recipe.py`, `cli/pretrain.py`, `training/dataset.py` | Add the evaluation manifest to `_PretrainHandoff`; pass ordinary-run identity-checked training paths through a separate lightweight path-only carrier so the dataset does not re-resolve; keep `PreparedRelationalData` queue-only; preserve output protection, tee logging, phase reporting and resume contracts. |
| `training/finetune_support.py`, `training/finetune_loop.py`, `cli/finetune.py` | Private path-bound preparation handoff, shared loss/lineage decisions, batched prototype diagnostics. |
| `training/objectives.py`, `training/loss.py` | Non-FIFO target reuse and token-row loss interfaces; no objective formula changes. |
| `embedding/` | Generic seeded sorted-row index helper for equivalent evaluation/extraction callers; pure extracted-record evaluation where a real embedding consumer exists. |
| `delineation/` | Cluster cohort/monophyly calculations, within-group summary and abundance-native diversity. |
| `spatial/`, `vision/` | Checkpoint preparation, CAM selection and export input preparation. |
| CLI modules and command docs | Parsing, overwrite safety, presentation and compatibility entry points remain in CLI. |

Do not add a generic workflow state container. E1 adds a private manifest field to the existing frozen `_PretrainHandoff` and uses `take_pretrain_handoff`'s pop-and-clear lifecycle; finetune B uses its equivalent private path-bound handoff. For ordinary training dataset reuse, add only a small path-only carrier with an optional `None` default on `MultiCropDataset`; do not call `prepare_relational_data` or pass `PreparedRelationalData`, whose decoding, duplicate rejection, full-batch and queue root-escape policies remain queue-enabled-only. When the carrier is absent, preserve the current `csv_path` resolution path for direct callers and existing tests. Preserve old direct call signatures or update their known callers in the same task.

## Task 0: Repair Baseline Pretrain Smoke Fixtures (E6)

**Files:** Test-only changes in `tests/test_cli_smoke.py`.

- [ ] Repair the three resume nodes with missing `data.csv` by adding readable CSV fixtures with at least 32 rows and batches matching their current assertions: `test_pretrain_resume_allows_existing_output_and_appends_log`, `test_pretrain_resume_allows_equal_input_size`, and `test_pretrain_resume_inherits_saved_local_views_without_explicit_flags`. The path resolver accepts unresolved relative image references, so these contract tests need not create image files; retain any image fixtures their assertions actually consume.
- [ ] Update the monkeypatch in `test_pretrain_preflight_probes_do_not_consume_the_seed` to accept the existing `initialization=` keyword. In `test_pretrain_resume_trains_the_checkpoint_backbone`, accept that keyword and add local `--batch-size 2` for its four-row fixture; do not change `_make_tiny_pretrain_data`. If changing that shared helper becomes necessary, run all of `tests/test_cli_smoke.py` and record duration and behavior changes.
- [ ] Add `--batch-size 1` to `test_pretrain_cli_hands_over_the_resume_payload_and_its_path` so it tests payload/path handoff rather than zero-step rejection.
- [ ] Run each exact node and then all six together. Expected: six pass with no production-source changes; the zero-step guard remains intact.
- [ ] On execution approval, commit this completed task before starting the next; stage only `tests/test_cli_smoke.py` and use `test: repair pretrain resume smoke fixtures`.

## Task 1: Optimize Pretrain Gradient-Norm Checks (C4)

**Files:** Modify `src/otuformer/training/reporting.py`, `src/otuformer/training/pretrain_loop.py`, finetune norm callsite as justified, and focused `tests/training/test_dino_pretrain.py`, `tests/training/test_ssl_config.py`.

**Interfaces:** Preserve `_compute_grad_norm_parameters(params)` at module scope and exactly one call per effective step. Extend its internal mode only as needed to choose numeric logging versus finite-only checks, retaining existing callers.

- [ ] Add tests for numeric parity with the current Python-float sum, exactly one `_compute_grad_norm_parameters` call per effective step, finite-only non-log behavior, NaN/Inf rejection, per-tensor norm overflow, global float32 clip overflow, and unchanged pre-clip `grad_norm` CSV value. Preserve the existing second-step call-count assertion and do not add debug or duplicate checks.
- [ ] Run `python -m pytest tests/training/test_dino_pretrain.py::test_dino_epoch_zero_freeze_then_joint_clip tests/training/test_dino_pretrain.py::test_dino_ce_only_backward_reaches_backbone_projector_and_prototypes tests/training/test_dino_pretrain.py::test_non_finite_gradient_norm_stops_the_step tests/training/test_dino_pretrain.py::test_clip_reduction_overflow_stops_the_step tests/training/test_ssl_config.py::test_pretrain_joint_clip_includes_no_decay_group -q`; confirm new contracts fail before implementation.
- [ ] Implement backend-supported foreach per-tensor norms and one finite reduction; on log steps transfer norms in one batch and calculate the same Python-float numeric norm. Let the existing `clip_grad_norm_(..., error_if_nonfinite=True)` catch global float32 overflow and retain its existing error mapping. Keep the original implementation as fallback where required.
- [ ] Run the named tests plus `tests/training/test_pretrain_alignment.py`'s norm-related tests. Confirm the four seam-dependent tests remain meaningful, the existing second-step call-count assertion remains exactly one call per effective step, and the CSV value remains pre-clip.
- [ ] In an authorized performance gate, compare actual gradient tensor count and paired real-step time on CPU and available MPS/CUDA, recording §8 evidence. If measurement authorization is absent, use the global `not measured: authorization pending` disposition; this does not block correctness acceptance. The submitted MPS isolated numbers (47.2 ms old, 22.5 ms numeric batch transfer, 7.7 ms finite-only for 156 tensors) are expectations only, not acceptance. Omit the optimization on a backend without repeatable benefit or supported operations.

## Task 2: Optimize Finetune Prototype Diagnostics (C3)

**Files:** Modify `src/otuformer/training/finetune_support.py` and `src/otuformer/training/finetune_loop.py`; tests in `tests/training/test_pretrain_alignment.py` and `tests/training/test_ssl_metrics.py`.

**Interfaces:** Retain `_accumulate_loss_diagnostics` and its direct-call signature. Replace per-sample device scalar reads while preserving its result fields.

- [ ] Add red tests for exact own/global winner counts, first-index ties, margin-hit totals, denominator and epoch CSV semantics across one/multiple centers, uneven batch and one-class cases; retain SupCon no-op.
- [ ] Batch winner-index transfer and count on host (or use a supported device count), preserving updated-center timing, class order and strict margin comparison. Do not reuse pre-update training logits or alter loss/gradient/checkpoint.
- [ ] Run `tests/training/test_pretrain_alignment.py::test_loss_diagnostics_counts_local_and_global_winners_differently`, `tests/training/test_pretrain_alignment.py::test_loss_diagnostics_margin_hits_use_margin_adjusted_target_logit`, and `tests/training/test_ssl_metrics.py::test_epoch_and_finetune_headers_remain_unchanged`, plus new count cases. In an authorized performance gate, record §8 evidence and measure paired real-step results; if not authorized, keep the performance claim open without blocking correctness acceptance. Submitted MPS 10.6 ms versus 0.38 ms is prioritization evidence only.

## Task 3b: Snapshot Evaluation Manifest and Select Before Decode (A, E1)

**Files:** Modify `src/otuformer/training/pretrain_recipe.py`, `src/otuformer/cli/pretrain.py`, `src/otuformer/training/pretrain_loop.py`, `src/otuformer/cli/finetune.py`, `src/otuformer/training/finetune_loop.py`, and `src/otuformer/training/reporting.py`; tests in `tests/training/test_pretrain_alignment.py`, `tests/training/test_startup_phases.py`, `tests/test_cli_smoke.py` and `tests/test_output_safety.py`.

**Interfaces:** Add a private prepared evaluation manifest to `_PretrainHandoff`; `take_pretrain_handoff` pops and clears it with other slots. This task introduces the single private finetune handoff object, path-bound and with a manifest slot. Task 4 extends that same object with finetune decision/payload slots and must not introduce another handoff mechanism. Preserve module-level `_compute_and_log_all_metrics`, `_compute_embeddings_from_csv`, and `_maybe_subsample_for_metrics` names and old direct call forms through Task 3b; Task 9 migrates the known sampling tests/callers and removes `_maybe_subsample_for_metrics`.

- [ ] Add red tests for manifest row/path/label snapshot, capped indices before decode/forward, missing/malformed/ambiguous/non-file entries, warning once and continuing, disable flag bypass, logger/CSV/plot artifact parity, mid-run edits, next-resume refresh, RNG parity, and private handoff not serialized. Exercise both pretrain and finetune save boundaries.
- [ ] Cover the full startup resolution chain: resume step prediction when applicable, overwrite protection, `MultiCropDataset` construction, and evaluation-manifest preparation. When source/root/ordered selection and validation contracts match, assert one identity-checked training selection is passed through a separate ordinary-run path-only carrier and the dataset does not call `resolve_dataset_image_paths` again. Count dataset construction in the resolution total; preserve separate resolution only for genuinely different populations or contracts. The carrier must be optional with a `None` default; when absent, `MultiCropDataset` resolves `csv_path` exactly as today, so direct callers and existing dataset tests remain valid. Do not call `prepare_relational_data` or use `PreparedRelationalData` for ordinary runs: no startup image decode/fingerprint pass, duplicate-row rejection, complete-batch requirement or queue root-escape policy. Extend the `take_pretrain_handoff` retry test in `tests/training/test_relational_queue_recipe.py` so every newly added private slot, including the manifest, is absent/cleared on the second take.
- [ ] Add startup-phase tests: the manifest phase starts after `TeeLogger` is active and before evaluation-manifest snapshot/label attachment and validation; done follows success or warning handling. The complete shared path resolution performed earlier for loader-step preflight/output protection is outside this phase and must be reused when contracts match. Assert both phase lines and any warning appear in terminal and `<out>/logs/pretrain.log` or `finetune.log`; preserve destructive output protection and avoid duplicate resolution.
- [ ] Implement startup-once parse only when metrics are enabled and an evaluation CSV is configured. Snapshot ordered rows, paths and labels. On failure warn once, skip periodic evaluator for this invocation, but keep epoch logger and its curves input; explicit disable continues to omit logger and bypass manifest work. Selected decode errors retain epoch warning behavior. Keep this preparation inside `run_pretrain`/the finetune runner after tee activation; do not move it into the CLI callback before output protection, because Task 0's mocked-runner fixtures intentionally contain unresolved image references.
- [ ] Reuse the exact complete resolution for loader steps, overwrite protection, and dataset construction when selection/validation contracts match. Do not replace ambiguity and existence checks with a row count. Pass identity-checked training paths into `MultiCropDataset` through the separate ordinary-run path-only carrier; do not allow an ordinary non-FIFO run to resolve the same training CSV again. Keep the carrier optional with a `None` default, and preserve the `csv_path` resolution path when absent. Do not call `prepare_relational_data` or pass `PreparedRelationalData` for ordinary runs: its startup decode/fingerprint, duplicate-row, complete-batch and queue root-escape behavior is queue-only. Build the evaluation manifest from the shared resolved selection when source/root/population contracts match, retaining labels and evaluation-specific ordering. Put the private manifest in `_PretrainHandoff` via existing pop-and-clear lifecycle. Keep manifest preparation inside `run_pretrain` after `TeeLogger` is active; the Task 0 missing-image fixtures rely on the mocked runner not performing E1 validation. Implement A's seeded sorted selection locally using identical ordering/seed/cap semantics; keep `_maybe_subsample_for_metrics` as a compatibility seam until Task 9 migrates its direct-call tests.
- [ ] Run exact focused tests and `tests/test_docs.py`. For performance acceptance, use a matched evaluation manifest larger than the effective `--metrics-sample-size` cap (default 10,000), or explicitly lower the cap for the measurement; otherwise decode/forward savings are necessarily zero and this part of A is not measured. Count avoided decoded/forwarded rows, path-resolution calls and recursive-index (`rglob`) walks as primary expected-work evidence; include the `MultiCropDataset` resolution in the count and assert it is eliminated only when the shared prepared-data contract matches. Paired wall time is secondary, especially for small manifests such as the 230-row `examples/Epidorcus/figs.csv`. When measurement is authorized, separately time `prepare`, `decode+forward`, `metrics`, and `UMAP`, and count all resolutions including resume loader-step prediction, overwrite protection, dataset construction and evaluation preparation. If measurement is unauthorized, record `not measured: authorization pending`; if no over-cap workload or cap override is available, record that limitation and leave the decode-saving benefit unqualified. Record measurements in the §8 schema. Keep the `compute_embedding_metrics` failure artifact distinction.



## Task 3c: Update Training Evaluation Documentation (A, E5)

**Files:** Bilingual `docs/commands/pretrain.md` and `.cn.md`, `finetune.md` and `.cn.md`; CLI help and workflow skill only if actual help text or guidance changes.

- [ ] Document sampling-before-decode, startup snapshot, one-warning/skip behavior, logger artifact distinction, explicit disable bypass, and that `--num-workers` affects training loaders while evaluation remains serial.
- [ ] Verify bilingual parity and `python -m pytest -q tests/test_docs.py`; CAM reserved-worker documentation needs no change.
- [ ] On execution approval, commit this completed task before starting the next; stage only affected documentation/help and use `docs: clarify periodic evaluation behavior`.

## Task 4: Prepare Finetune Decisions and Preserve Resume Validation (B, E2, E6)

**Files:** Modify `src/otuformer/cli/finetune.py`, `src/otuformer/training/finetune_support.py`, and `src/otuformer/training/finetune_loop.py`; tests in `tests/training/test_finetune_provenance.py` and `tests/test_pseudo_label.py`.

- [ ] Require standalone Task 0 to have passed before Task 3b begins; Task 4 changes finetune only and makes no pretrain source edits.
- [ ] Add finetune tests in `tests/training/test_finetune_provenance.py` and `tests/test_pseudo_label.py` for one authoritative payload deserialization, path-bound reload when path changes, explicit None versus unloaded, new/ordinary/legacy/pseudo resume decision parity, explicit conflict, source lineage/hash integrity, direct `run_finetune` compatibility, and pre-output failure preserving overwrite sentinel.
- [ ] Extract the minimum common preparation into current training ownership; CLI retains parsing, explicit option provenance, messages, overwrite protection and output mutation boundary. Use a private path-bound handoff; do not build a generic recipe system. Keep different pseudo-source and SSL initialization roles and independent integrity checks.
- [ ] Preserve the pretrain resume gate described in E6. No change to `_pretrain_loader_steps` is planned in this finetune task.
- [ ] Verify Task 0's six smoke nodes remain green; run focused `tests/training/test_finetune_provenance.py`, `tests/test_pseudo_label.py`, and `tests/test_docs.py`. Only after green, accept finetune preparation changes.

## Task 5: Reuse Non-FIFO Relational Targets (C1)

**Files:** Modify `src/otuformer/training/objectives.py`; tests in `tests/training/test_relational_queue_objectives.py` and `tests/training/test_relational_queue.py`.

- [ ] Add reference tests for `cross`, `full`, legacy `same`, zero/multiple local crops and representative ordinary/weak teacher inputs. Assert target build count, target detachment, losses, student gradients, center/EMA order and empty/populated FIFO controls.
- [ ] Implement per-step caching only for the used non-FIFO teacher slots. Keep teacher logits, centering, temperature, candidate order, dtype, reductions and batch lifetime unchanged.
- [ ] Run `tests/training/test_relational_queue_objectives.py` and `tests/training/test_relational_queue.py` focused nodes. When authorized, measure target builds and affected step time with §8 evidence; otherwise record `not measured: authorization pending` and keep performance claims open. Do not include DINO log-softmax work.

## Task 6: Gather Selected Tokens Before iBOT and Masked-Feature Heads (C2)

**Files:** Modify `src/otuformer/training/pretrain_loop.py`, `src/otuformer/training/loss.py`; tests in `tests/training/test_pretrain_alignment.py` and objective tests.

- [ ] Add full-head reference tests in `tests/training/test_masked_pretrain.py` and loss contract tests in `tests/test_training_loss.py`, with irregular masks and both global teacher views, asserting selected row order, losses, gradients, center state and unchanged RNG. Cover mask validation and finite outputs.
- [ ] Gather student iBOT and masked-feature predictor inputs before their tokenwise heads. For iBOT, move teacher head invocation from teacher-forward block to after mask generation and gather selected teacher tokens before the teacher head; preserve center update from the same selected pre-EMA logits. For masked-feature, gather each of four teacher blocks before pointwise final LayerNorm, then average/normalize the same selected rows. Preserve full encoder forwards and RNG draw order.
- [ ] Run `tests/training/test_masked_pretrain.py`, `tests/test_training_loss.py` and focused alignment/restore nodes. When authorized, measure head input rows and representative step time with §8 evidence; otherwise record `not measured: authorization pending` and keep performance claims open. If numerical tolerance or measured payoff fails, leave candidate out.

## Task 7: Gate Cosine and Loss Scalar Reuse; Decide C6 (C5–C7)

**Files:** Modify `src/otuformer/training/pretrain_loop.py`, `src/otuformer/training/finetune_loop.py`; focused tests in pretrain/finetune loop suites.

- [ ] Add C5 tests at log intervals 1, around step 10, final loader step, relational/DINO, asserting cosine is present exactly when consumed by CSV or postfix and absent otherwise.
- [ ] Add C7 tests for displayed values/precision, CSV values and epoch accumulation with scalar materialization counts; do not move required epoch-total sync earlier or change semantics.
- [ ] In a separately authorized performance gate, measure C5 avoided cosine/readback work and C7 scalar reuse on paired representative steps with §8 evidence; omit either cleanup when its gain is not repeatable or does not justify the change.
- [ ] In an authorized performance gate, measure existing C6 scan share in representative real FIFO/dense steps and record §8 evidence. If below 1%, record C6 closed/no change. If measurement is not authorized, record `not measured: authorization pending`, keep performance claims open and close the task without code. If measured at or above 1%, add per-group finite-update tests and only replace the current stacked aggregate with a supported primitive when it improves measured work; prohibit `all(bool(...))` generator form. Preserve no EMA/center/FIFO/checkpoint advance on failure and do not claim AdamW rollback.
- [ ] Run focused loop tests. Group C5/C7 with adjacent edits; no isolated refactor absent real reuse.

## Task 8: Bind Extract and Spatial Checkpoint Work to One Snapshot (D1)

**Files:** Modify `src/otuformer/cli/extract.py`, `src/otuformer/embedding/extractor.py`, `src/otuformer/spatial/extractor.py`; tests `tests/test_extract_records.py`, spatial/export focused tests and output-safety fixtures.

- [ ] Add tests that bind the executed model and provenance digest to exact bytes, preserve drift detection, assert load/build counts for explicit embedding layer and spatial paths, and preserve old publication on invalid-layer overwrite.
- [ ] Use the validated snapshot/model at actual consumers; retain separate path mutation checks at later safety boundaries. Do not weaken read-only resolution, teacher/student selection, ONNX/external data or output contracts.
- [ ] Run focused `tests/test_extract_records.py`, `tests/test_extractor.py` and spatial export/extractor tests plus a small real consumer case. When authorized, record §8 evidence for checkpoint reads/builds and peak host memory; otherwise record `not measured: authorization pending` and keep performance claims open.

## Task 9: Share Seeded Row Indices and Extract Evaluation Calculation (E3/E4)

**Files:** Modify `src/otuformer/training/reporting.py`, `src/otuformer/cli/extract.py`, `src/otuformer/cli/cluster.py`, and add the generic sorted-index helper under the existing `otuformer.embedding` owner; tests in `tests/training/test_pretrain_alignment.py`, `tests/test_extractor.py` and `tests/test_cli_smoke.py`.

- [ ] Add parity tests for all four callsites, documenting population, seed, cap, ordering and duplicate ID/label behavior; preserve excluded complement for visualization. Explicitly test that tree bootstrap/subsample and stratified silhouette sampling are not routed through the helper.
- [ ] Add a shared sorted-index helper to the existing `otuformer.embedding` owner, which owns two evaluation/extraction callsites; `delineation` consumes indices for cluster cohort semantics but does not own cross-domain sampling policy. Migrate the three direct-call/monkeypatch tests in `tests/training/test_pretrain_alignment.py` to test the shared helper or public caller, then delete `_maybe_subsample_for_metrics` and its wrapper in this task. Do not retain dead private code solely as a test seam.
- [ ] Move pure `_evaluate_records` calculation to the narrow verified domain owner; keep CLI provenance, safety, publication and orchestration. List real callers and compatibility needs before moving it.
- [ ] Run focused tests plus the master policy's full-refactor regression gate because this is a cross-command move. Document the exact gate in execution records; don't run unrelated suite repeatedly during iteration.

## Task 10: Share Within-Group Distance Summary (D3)

**Files:** Modify `src/otuformer/cli/cluster.py`, `src/otuformer/delineation/annotate.py`, and `src/otuformer/delineation/distance.py`; modify `src/otuformer/cli/annotate.py` only if its import path changes; tests in `tests/test_cli_smoke.py` and `tests/test_cli_annotate.py`.

- [ ] Characterize each current summary for ordering, missing labels, singleton, empty group and exact columns.
- [ ] Add an actual cluster-output-to-annotate cross-command test asserting same-named summary columns equal and identifier alignment is preserved.
- [ ] Move one exact summary implementation to `delineation`, retaining each command's distinct distance source/selection behavior; do not interchange Newick and linkage.
- [ ] Run focused CLI tests and full refactor gate from the master policy.

## Task 11: Fix Cluster Tree Semantics and Avoid Duplicate Metrics Linkage (D2)

**Files:** Modify `src/otuformer/cli/cluster.py` and `src/otuformer/delineation/partition.py` (the existing owner of `compute_partition_metrics`); add characterization and CLI cases to `tests/test_cli_smoke.py`, and partition mathematics to `tests/test_partition.py`. Do not create `tests/test_cli_cluster.py` unless a separately justified test-organization change is approved.

- [ ] Add a current-implementation characterization test to `tests/test_cli_smoke.py` first and run it green, pinning the non-prefix sample's old wrong numeric value from sampled linkage with unsampled IDs/labels. Do not describe this value as expected behavior.
- [ ] Add tests for fixed output-tree topology with full/partial/no labels, singleton, unlabeled tips, different sample caps/seeds/cutoffs, and a labeled-subset topology differing from output-tree restriction.
- [ ] Implement fixed-output-tree labeled-tip monophyly once, plus one sampled partition linkage from one aligned cohort submatrix. Keep bootstrap support, full tree and partition outputs unchanged.
- [ ] Run focused cluster nodes in `tests/test_cli_smoke.py`, `tests/test_partition.py`, and a real extract → cluster artifact consumer test. Treat duplicate-linkage reuse as a code-simplicity/parity gate under the default `--max-distance-pairs=1,000,000`: assert exactly one metrics linkage, aligned inputs and exact non-monophyly metric values. Do not run a standalone performance gate for the default limit. Only if a matched workload explicitly raises the limit to roughly 3,000 or more embeddings may collect optional wall-time evidence; do not frame the fixed-tree monophyly correction as a speedup.

## Task 12: Keep Diversity OTU Abundances Compact (D4)

**Files:** Modify `src/otuformer/delineation/diversity.py` and its focused tests.

- [ ] Add parity fixtures in `tests/test_diversity.py` against the old expanded-row reference for headers, integer abundance, thresholds including nonpositive, zero counts, empty columns, missing tips and supplied-tree MPD.
- [ ] Operate directly on OTU counts for global/per-sample OTU-table path, retaining alpha formulas and warning values. Do not change assignment tree-tip identity or share filtered centroid NJ trees.
- [ ] Verify identical outputs and bounded memory/work on a large-abundance, small-shape table; run focused diversity tests.
- [ ] Treat optional NJ base-tree copy reuse as a separate candidate within this task; omit absent proof of exact topology, branch lengths, support and PD parity.

## Task 13: Share CAM Selection for Protection and Execution (D5)

**Files:** Modify `src/otuformer/cli/cam.py`, `src/otuformer/vision/cam.py`; tests in `tests/test_cam.py`, `tests/test_output_safety.py` and CAM nodes in `tests/test_cli_smoke.py`.

- [ ] Add selector call-count tests for CSV and directory input, ordered label/path correspondence, max-images, missing files, overwrite protection, predictions, overlays and NPY outputs.
- [ ] Prepare one ordered selection per CLI invocation; preserve direct `run_cam` preparation and image geometry. Do not activate reserved workers, reinterpret CAM batch size, or change hooks/method-specific batching.
- [ ] Run focused `tests/test_cam.py`, `tests/test_output_safety.py` and CAM nodes in `tests/test_cli_smoke.py`; verify prediction, overlay and array alignment. For the avoided duplicate selector work, count selector calls and record paired §8 timing only when performance measurement is authorized.

## Task 14: Prepare Embedding Export Before Overwrite (D6)

**Files:** Modify `src/otuformer/cli/export.py`, `src/otuformer/vision/export.py`; tests `tests/test_spatial_export.py`, embedding export tests and output-safety tests.

- [ ] Add direct contract tests for `_prepare_spatial_export` and embedding preparation plus prior-publication sentinel tests for deterministic invalid inputs.
- [ ] Prepare and reuse embedding model before cleanup, retaining direct-call support, source formats, register/head semantics and publication order. Do not claim protection from later tracing/ORT failures.
- [ ] Run focused `tests/test_spatial_export.py`, `tests/test_output_safety.py` and embedding-export nodes in `tests/test_cli_smoke.py`, including ONNX graph/numerical checks where dependencies are available; record any unqualified ONNX gate.

## Task 15: Bounded Domain-Owner Moves and Consumer Chains (E3)

**Files:** Only exact files enumerated by Tasks 9–14, plus the verified `delineation`/`embedding` owner and compatibility wrappers.

- [ ] Before each move, enumerate symbol definition, every caller, current owner, target owner and compatibility requirement. Do not move whole CLI workflows or refactor by file length.
- [ ] Run focused consumer chains `extract → cluster → annotate → diversity` and `export → extract`; ensure `cam` predictions/overlays/arrays separately align.
- [ ] Run the master full-refactor gate once after all cross-command moves. Preserve the concrete test results and unexercised consumers.

## Task 16: Synchronize Remaining Documentation and Final Focused Acceptance

**Files:** Affected bilingual command docs, `docs/commands/embedding-metrics.md` and `.cn.md`, relevant README examples and workflow skill; English design cross-references/plans only. Modify only actual surfaces affected.

- [ ] Add, rather than merely update, user-facing monophyly documentation. In `docs/commands/cluster.md` and `.cn.md`, add a `metrics.csv` field/definition table stating that `monophyly_proportion` is calculated on the fixed output tree, uses labeled tips only under partial coverage, treats represented singleton species as monophyletic, uses the number of labeled species as denominator, and is independent of `--metrics-sample-size`. In `docs/commands/embedding-metrics.md` and `.cn.md`, add `cluster` to Affected Commands or explicitly explain its distinct tree metric; keep the shared-metric scope accurate. Preserve the existing CSV field name. Training evaluation documentation belongs to Task 3c. Do not edit CAM reserved-worker docs that already state the truth.
- [ ] Add dated references to this design and plan without rewriting historical decisions. No internal speed claim without matched measurements.
- [ ] Run `python -m pytest -q tests/test_docs.py`, all exact changed-area tests, six E6 resume nodes, the authorized full refactor gate, and applicable extract/export consumer chains. This plan ends at feature acceptance; merge/release acceptance is separate and, when authorized, must run the full suite required by the master testing policy.
- [ ] Run `git diff --check`, inspect rendered help/schema/docs where changed, inspect `git status`, and verify no benchmark/training artifacts or user data were overwritten.
- [ ] Record each design unit's final disposition in the dated `## 10. Implementation dispositions` addendum at the end of `docs/superpowers/specs/2026-11-18-workflow-execution-efficiency-design.md` using exactly one of: `implemented — measured benefit`; `implemented — benefit unqualified (<reason>)`; `closed-no-benefit`; or `not pursued`, with evidence or reason including C6. An unmeasured or unavailable result must use `implemented — benefit unqualified`, never `closed-no-benefit`. Keep unsupported performance/device claims explicitly open. Commit each completed task before starting another after plan execution is approved; no extra commit after the planned tasks without separate operator approval. No push, version bump, tag or release under this plan.
