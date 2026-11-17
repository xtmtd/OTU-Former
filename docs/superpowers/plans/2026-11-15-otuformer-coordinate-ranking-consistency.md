# Coordinate and NeCo-Style Ranking Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native inline execution. Use superpowers:subagent-driven-development only if the operator separately requests delegation. Execute only authorized tasks in dependency order; unchecked step boxes record implementation, not plan authoring. Task1 is required for ranking, not for delivering Tasks2–5; optional-package qualification failures must not block independent Coordinate work. Task5 is independently schedulable with no prerequisite implementation task; its placement after Task4 is document grouping, not a dependency.

**Execution status (2026-11-17):** Implemented in the uncommitted working tree; acceptance remains partial. The seven checked items are the narrow implementation steps explicitly reconciled during the preceding review, not a complete inventory of implemented behavior. An unchecked item does not imply its code is absent: strict save/recover (Task4 Step4), one-backward optimizer/state integration (Task7 Steps3/4), and Task3/6 objective tests already have implementation and bounded passing evidence in the dated execution notes. Compound steps can still contain unrun qualification, review or commit requirements; historical red/green claims are not reconstructed from current passing tests. This retained checklist is an execution-history index, not the current acceptance matrix; use the dated notes and ledger for verified scope and open gates. No new commit/push/release or research training is authorized.

**Goal:** Add independently weighted Coordinate and NeCo-style ranking objectives with checked geometry and strict recovery, plus an independently deliverable backbone-only new-run initializer; preserve all-off behavior.

**Architecture:** A narrow geometry component supplies correspondence samples to two separate objective functions. Existing recipe/loop/checkpoint/reporting owners retain orchestration; an optional lazy sorting adapter and ranking head do not create another trainer or SSL method. One plan owns their shared contracts; independently testable stages do not require separate design documents.

**Tech Stack:** Existing Python >=3.11, PyTorch >=2.9, torchvision >=0.24, timm >=1.0, NumPy >=1.26, Typer/Click and pytest. Candidate optional `diffsort==0.2.0`, subject to Task 1 qualification; no Lightning, new core dependency or in-repo sorting reimplementation.

**Spec:** [Unified Coordinate/ranking design](../specs/2026-11-15-otuformer-coordinate-ranking-consistency-design.md), read together with the [master design](../specs/2026-03-29-otuformer-design.md) and mandatory [testing policy](../specs/2026-11-07-development-testing-policy.md). The design owns meanings and approvals; this plan freezes implementation names, fields, tests and dependency order.

**Date:** 2026-11-15. **Source baseline:** `fd745ef`. Existing uncommitted changes at plan authoring are the unified design migration plus master/CLS/spatial design links; preserve them. Re-read actual symbols after edits rather than relying on line offsets.

**Original planning authorization (historical):** The operator requested this plan after the unified design discussion and audit revisions, initially authorizing only the design-to-planning transition; reference default **16** was explicitly approved. At that planning snapshot, execution had not been selected and all steps were unchecked. Later implementation, bounded experiments and repair authorizations are recorded in the dated execution sections; the current status above supersedes this historical snapshot. Long qualification, new dependency installation, commits, pushes, versions, tags and releases still require their applicable authorization.

## Global Constraints

- Preserve `ssl_method=relational|dino`, all four existing `patch_loss` modes, KoLeo and permitted relational FIFO combinations; neither new objective becomes an enum choice or automatically enables the other.
- Fixed independent finite nonnegative weights, defaults `coordinate_weight=ranking_weight=0.0`; query default49 and bounds1..49; approved reference default16 and bounds2..49.
- Ordinary unmasked teacher A/B -> opposite student global and every student local, same-image valid overlap only; no identical-view or teacher-local pair, weak-view substitution or new augmentation.
- Student actual patch centers, actual valid counts without replacement/padding; conservative support and bilinear `align_corners=False`; exact per-point -> sample -> ordered-pair mean.
- Coordinate: raw final-normalized backbone interpolation then L2. Ranking: three biased Linear layers D->2048->2048->256 with GELU, per-patch L2 before interpolation, cosine renormalization afterward, Cauchy/bitonic steepness100 on both sides.
- Preserve single backward/optimizer, joint clipping, existing decay and EMA; default decay0.05 applies to head weights and biases, opt-out exempts biases through the shared policy. No head-only LR/decay/warmup.
- Norm/rho candidates1e-6, row/column residual candidate1e-5 and strict positive student permutation entries; no log clamp, silent numerical skip, hidden fallback or mid-run guard tuning. Qualification precedes frozen recipe acceptance.
- Ranking references are detached teacher **selected correspondence points for the current ordered pair**, across the batch; allow same-image/self entries, independent per-sample draws, no cross-pair mixing or historical patch queue.
- Existing ordinary/weak image tensors and RNG order do not change; weak geometry slots are `None`; all-off keeps historical collation, no tracing/support/sorter/head/generators.
- Enabled resume is same recipe at the existing epoch boundary. New objectives/weights/caps require a new run; source initialization transfers only selected teacher/student backbone and starts fresh heads/state.
- Runtime/schema/cards remain the single option truth. Development docs English-only; user-facing README/commands bilingual. No dataset mutations or scientific benefit claims.
- Every execution task follows its named red/green test cycle and exact scope. Choose `auto` for ordinary integration; keep controlled replay/unsupported ops explicitly on CPU under the central policy. Do not certify skipped prerequisites.
- Initial red may include collection-time `ModuleNotFoundError` for the exact new production module named by that task (or the corresponding new-symbol import failure / absent runner file). Inspect the traceback to establish that cause; unrelated import/dependency/environment failures, syntax errors, missing test files and skips are not the intended red. Do not create empty implementations merely to make tests collect. After the implementation step, imports/collection must succeed and the behavior assertions must actually run. `tests/training/` already exists; pytest requires no added `__init__.py` here.
- Runner-test loading is frozen to the existing path-based convention: `tests/test_dense_ranking_qualification.py` defines `_load_dense_ranking_script()` using `Path(__file__).resolve().parents[1] / 'scripts' / 'check_dense_ranking.py'`, `importlib.util.spec_from_file_location('check_dense_ranking', path)`, `module_from_spec` and `spec.loader.exec_module(module)`, asserting a non-None spec/loader. Every runner test added to `tests/test_dense_ranking_qualification.py` by Tasks1/3/7/8 uses that local helper; other test files import production functions directly, not helpers from another test module. No `scripts/__init__.py`, `sys.path` mutation, package move or new installed command. Script import must not call `main`, parse argv, install dependencies, train, write output, import `diffsort` or instantiate a model; each mode imports its own execution dependencies lazily; explicit CLI execution is `python scripts/check_dense_ranking.py ...`, guarded by `if __name__ == '__main__'`.

## Review Focus

1. Malformed trace shapes/counts/view slots must fail, and collation must preserve tested row/view/key alignment rather than silently reorder it: Task2 malformed-envelope and mixed-view fixtures. This does not claim to detect externally forged, internally consistent traces without an identity binding; no image-content hashing protocol is added.
2. Many small-overlap locals, no-overlap samples, a single-image batch and a one-entry reference pool must not change another objective's denominator or cause accidental AdamW head decay: Tasks 3/6/7.
3. Incompatible added/native register layouts, malformed source head identity, missing selected teacher/student branch or required optimizer moments must fail before output mutation; valid matching register layouts must load: Tasks4/5/7.
4. A query/reference pool crossing its cap inside a batch changes draw consumption; same eligibility/order at an epoch checkpoint must replay later selections, loss and state: Tasks 2/6/7.
5. Missing/changed optional package, non-power-of-two R, extreme soft orders and ordinary forward chunking must not silently change recipe, device or memory guarantees: Tasks 1/6/8.
6. Reporting is a rendered surface, not a column list: a dense-enabled run must append its diagnostics page to `logs/training_curves_pretrain.pdf` while an all-off run keeps exactly one page, and an unavailable statistic must leave the cell empty rather than plot a measured zero: Tasks 4/7/8.
7. Every check that can fail without the loader must run before `--overwrite` may clear a directory, and the same complete rule must run in preflight and at restore, so a payload is never accepted early and rejected late: Tasks 4/5/7.
8. Recovery evidence must restart cold. Reusing the live head/optimizer/optimizer state proves nothing, because a no-op restore would reproduce the numbers: Task 7/8.

## File Ownership and Frozen Interfaces

Do not create a plugin framework or move unrelated training modules. New focused files:

| File | Responsibility |
| --- | --- |
| `src/otuformer/training/dense_geometry.py` | Pair enumeration, checked eligibility, query sampling and geometry-only records; no optimizer/CLI. |
| `src/otuformer/training/dense_objective.py` | Shared norm/cancellation checks, hierarchical reduction and Coordinate result. |
| `src/otuformer/training/neighbor_ranking.py` | Lazy optional sorter/probe/cache, independent heads, pair reference sampling and permutation CE. |
| `src/otuformer/training/dense_recipe.py` | Dense-only immutable effective settings, closed bundle serialization/validation and RNG state; pretrain recipe remains invocation owner. |
| `scripts/check_dense_ranking.py` | Explicit numerical/replay/memory qualification runner, import-safe, synthetic inputs and JSON evidence; no production calibration or automatic dependency install. |

Existing owners: `training/dataset.py` collator; `pretrain_recipe.py` sole preparation/recovery and initialization handoff; `pretrain_loop.py` one live loop; `reporting.py` CSV migration; `utils/checkpoint.py` architecture/source identity and downstream readers; `cli/pretrain.py`, `cli/constraints.py`, `cli_schema.py` executable options/schema. Keep `training/trainer.py` compatibility aliases unchanged.

### Shared signatures (define once, reuse later)

- `DenseBatchGeometry = tuple[tuple[GeometryTrace, ...] | None, ...]` in `dense_geometry.py`: validated in-process representation, one slot per tensor view, batch-row order inside ordinary slots, `None` only for weak-only views. Dataset items serialize ordinary traces with existing `GeometryTrace.to_json()`; the collator transposes those ragged records without default-collating operations and writes each weak slot as `None`; `_unpack_pretrain_batch` decodes them once with `GeometryTrace.from_json()` and validates tensor canvas/row/view layout. Direct in-process callers may supply already validated `GeometryTrace` objects; old per-row weak-None envelopes are normalized at unpack only, and pairing accepts canonical weak `None`. No second serialization format or content-hash identity protocol.
- `DenseSampleQueries(student_indices: Tensor, teacher_xy: Tensor, eligible_count: int)`: CPU int64 flat indices `[Q]` in sorted row-major order, CPU float64 mapped edge coordinates `[Q,2]`; Q may be zero. Keep the pre-cap complete-support count for cached eligible/selected diagnostics, with `0 <= Q <= eligible_count`; do not recompute geometry to recover that count in the logger.
- `DensePairQueries(teacher_view: int, student_view: int, samples: tuple[DenseSampleQueries,...], student_valid: Tensor, teacher_valid: Tensor)`: indices refer to original batch view order; teacher index0/1 only. Validity tensors are CPU bool `[B,Hg,Wg]` on each view's actual grid, shared across records for that view rather than recomputed/copied per pair; they let the objectives call the existing complete-support sampler without reconstructing metadata from coordinates.
- `build_dense_pairs(geometry: DenseBatchGeometry, *, grid_hws: tuple[tuple[int,int] | None,...], patch_hw: tuple[int,int], local_crops: int, query_cap: int, generator: torch.Generator) -> tuple[DensePairQueries,...]`: complete-support filtering before capped uniform sampling. Ordinary slots require checked actual grid dimensions; weak slots are `None` and never inspected. Build each ordinary sample/view's validity once via existing `pixel_support(trace)` then `patch_support(trace, pixel_sampling_valid=pixel_valid, grid_hw=..., patch_hw=...)`; reuse masks across pairs, never infer complete support from the final affine matrix alone. Pair order is teacher0->student1,locals in view order, then teacher1->student0,locals in view order; sample order stays batch order. Mapped-coordinate eligibility must include the same float32 representability and nonzero-contributor rule as the eventual sampler, not a separate double-precision-only boundary decision.
- `DenseLossResult(raw: Tensor | None, statistics: dict[str, float | int | None])`: `None` means unavailable, not measured zero. No detached loss is used for backward. Statistics use named numeric skip counters, not fabricated zero measurements; minima/means with no participating population are `None`. Record global/local attempted/contributing pair/sample counts, pre-cap eligible/selected query counts, geometry skips, reference-pool/reference-count ranges and reference-shortage skips. A real count of0 is valid; unavailable loss/norm/rho is not measured0. Attach teacher/student pair and batch-row context when rethrowing a numerical check failure.
- `coordinate_loss(student_grids: dict[int,Tensor], teacher_grids: dict[int,Tensor], pairs: tuple[DensePairQueries,...], *, patch_hw: tuple[int,int]) -> DenseLossResult`: BHWC final-normalized raw fields, teacher detached. Use each pair's student/teacher validity masks (move to the field's device without changing logical values); student-center `(x,y)` follows its recorded flat indices and actual grid/patch sizes.
- `reduce_dense_losses(per_pair: Sequence[Sequence[Tensor]]) -> Tensor | None`: each inner scalar is an available sample's point mean; empty samples are omitted, then samples/pairs equal-weighted.
- `sample_dense_features(grid: Tensor, xy: Tensor, *, patch_hw: tuple[int,int], valid: Tensor) -> tuple[Tensor,dict[str,float]]` in `dense_objective.py`: wrap existing sampler with complete support and norm/rho guards, not a second grid-sample convention. `xy` is `[B,Q,2]` with Q>0 (objective callers omit Q0 samples before this helper); use a small per-image call for ragged Q. Convert coordinate metadata to float32 on CPU before transfer to the feature device (do not first place float64 tensors on MPS); move the matching Boolean validity mask to that same device, with geometry support already checked under this float32 sampling convention. Return sampled vectors without applying L2 inside the helper; Coordinate and ranking call sites keep their distinct normalize/project order. Any already-selected point that no longer satisfies complete support is a metadata/internal-consistency error, not a silently dropped numerical point.
- `DenseSettings(coordinate_weight: float, ranking_weight: float, query_cap: int, reference_cap: int | None)` in `dense_recipe.py`: `enabled` and `ranking_enabled` derived properties; inactive ref cap is `None`.
- `DenseRuntime(query_generator: torch.Generator, reference_generator: torch.Generator | None, ranking_contributed_steps: int=0)` owns only enabled CPU RNG streams/counter, not models/optimizers. `new_dense_runtime(settings: DenseSettings, *, seed: int) -> DenseRuntime | None` returns None when off; derive each seed from the first8bytes (big-endian, masked to63bits) of SHA256 UTF-8 `f'{seed}:dense-query-v1'` / `f'{seed}:dense-reference-v1'`. Never use process-randomized Python hash or advance global RNG for construction.
- `resolve_dense_settings(args: argparse.Namespace, checkpoint: dict | None) -> DenseSettings`; `dense_config_bundle(settings: DenseSettings, *, batch_size: int, steps_per_epoch: int, dependency: dict | None) -> dict | None`; `dense_state_bundle(runtime: DenseRuntime, *, completed_steps: int, ranking: NeighborRankingObjective | None) -> dict`.
- `NeighborRankingObjective(hidden_dim: int)` owns `student_head`, `teacher_head`, `update_teacher(momentum: float) -> None`, and `compute(student_grids, teacher_grids, pairs, *, patch_hw, reference_cap, generator) -> DenseLossResult`; raw input grids have the same ownership as Coordinate, but ranking projects before sampling.
- `probe_ranking_dependency(device: torch.device) -> dict[str,object]` and `sort_permutations(cosines: Tensor, *, cache: dict) -> Tensor` in `neighbor_ranking.py`: lazy qualified package, bounded four-size LRU, unchanged RNG. Shared sorter validation is finite/nonnegative/stochastic; the ranking loss separately enforces **student P>0** for its log and **teacher P>=0**. Do not accidentally make a legal teacher zero a numerical failure by applying the student-only strict guard to both roles.

These are internal focused component APIs, not a public trainer API. Use existing trace/feature/identity helpers; do not expose half-implemented options. Keep annotations cycle-safe through `TYPE_CHECKING`/forward references where needed: `dense_recipe.py` must not eagerly import the ranking implementation or its optional dependency on Coordinate/off or schema-only paths.

### Closed checkpoint schema — version1

All bundles reject missing/unexpected fields, booleans-as-numbers, wrong versions/types/nonfinite values and contradictions. Constants for bundle keys and new policy IDs belong in the new `dense_recipe.py`; reuse `constants.SPATIAL_SUPPORT_VERSION` for `support_policy`, rather than duplicating that existing identifier. Promote nothing else to a global registry without an actual second owner.

`config.dense_training_config` exists iff either weight is positive. Exact base keys:

```text
version=1; coordinate_weight; ranking_weight; query_cap; batch_size; steps_per_epoch;
objectives=[positive-weight objective IDs in fixed coordinate,ranking order];
rng_policy='sha256-seed-stream-v1';
geometry_policy='realized-affine-complete-support-v1';
support_policy='conservative-resampling-v1';
query_policy='student-centers-without-replacement-v1';
reduction_policy='point-sample-pair-mean-v1';
numeric_policy='float32-norm-rho1e-6-stochastic1e-5-v1';
coordinate_features='final-norm-raw-sample-l2-v1'
```

When ranking is positive, add exactly `ranking_config` with:

```text
version=1; reference_cap; pool_policy='pair-aligned-selected-teacher-v1';
reference_policy='per-sample-without-replacement-self-allowed-v1';
head={'layers':3,'hidden_dim':2048,'output_dim':256,'activation':'gelu','bias':True};
feature_policy='project-l2-sample-cosine-v1';
sort={'network':'bitonic','distribution':'cauchy','steepness_student':100.0,'steepness_teacher':100.0,'direction':'ascending-cosine','permutation':'input-by-rank','loss':'row-ce-unclamped-v1'};
dependency={'name':'diffsort','version':'0.2.0','wheel_sha256':'95bd5541ade48919cd926366f83165e07645b6e62fe798343b7f0d0332b8a08b',
  'source_sha256':{
    'diffsort/diffsort.py':'7dd6d3a72e25f17700d3444fae2ca3c91db0a08713054b95808029653fc18574',
    'diffsort/functional.py':'4c8ff02f74d6fe1e0b53a9685954cc25ac8f3e67ec46d7d52fdfa9f314533676',
    'diffsort/networks.py':'1680cf2a0e90a2d158991b91a99983ab4bb2e150feb773d6d440f4423b3b9c85'}}
```

The dependency record is produced only after Task1 qualifies that exact candidate; do not silently upgrade it. **Resume-enforced runtime identity:** dependency name/version and all three hashes of the actually loaded implementation must match the qualified saved recipe. Modified or unverifiable code fails with guidance instead of passing on a version string alone; an editable installation is judged by the same source checks, not its installation style. **Artifact provenance only:** `wheel_sha256` identifies the audited wheel; it does not assert that an installed package's original wheel can be recovered. Preflight/resume must not require, re-download or compare an installed wheel archive, and must not reject an otherwise matching installed implementation merely because that archive/hash is unavailable. Saved bundles still obey closed-schema validation. The supported ranking extra requires access to the loaded modules' verifiable source bytes: a loader that exposes matching source can pass, including a source-readable zip loader; source-less/bytecode-only, frozen or other distributions that cannot expose those bytes fail with guidance. Do not assume every zip installation is unsupported, and do not weaken identity checks based on packaging style. Record the distribution/source-access limitation with environment evidence during execution; Coordinate/off and ordinary checkpoint readers do not acquire this restriction.

Threshold values are **provisional until qualification**. Combining absolute norm `1e-6`, relative cancellation rho `1e-6` and stochasticity tolerance `1e-5` in one `numeric_policy` ID is an intentional **atomic recipe binding**: changing any one changes that identity and invalidates same-recipe resume. Do not split them into independently negotiable policies. If a threshold or qualified implementation must change, stop, revise design/this schema/version policy and obtain approval before implementing a different recipe.

Top-level `dense_training` exact base keys: `version=1`, `completed_steps`, `query_rng` (valid restorable CPU torch generator ByteTensor). If ranking positive, add exactly `ranking` with `student_head`, `teacher_head`, `reference_rng`, `contributed_steps`; otherwise forbid that key. Head dicts must match every expected parameter shape/dtype/finiteness. `0 <= contributed_steps <= completed_steps`; positive count requires valid per-parameter AdamW head state. Count increments only after successful contributing optimizer steps. For a never-contributed head, no head optimizer entries/moments may have been created; reject fabricated complete or partial moments at count0. `completed_steps` equals checkpoint iteration.

All-off omits both bundles. Existing schedule/RNG/patch/DINO/FIFO state retains its current ownership; compare duplicated step/batch constraints for consistency. Do not serialize a reference pool, sorter tensors or a new mid-epoch cursor.

`config.pretrain_initialization`, if source initialization was used, exact keys: `version=1`, `path` (resolved source), `sha256`, `source_key` (`teacher|student`), `ssl_method`, `ssl_head` (source identity), `model_name`, `image_size`, `register_tokens`, `transfer='backbone-only'`. Current top-level SSL identity remains the new run's. The bundle persists on resume without reopening the source file; it may exist when dense training is off.

## Stage A — Qualification and Shared Coordinate Delivery

### Task 1: Qualify the optional sorter and numerical contract

**Files:** Create `src/otuformer/training/neighbor_ranking.py`, `tests/training/test_dense_sorter.py`; create or extend `scripts/check_dense_ranking.py` and `tests/test_dense_ranking_qualification.py` depending on whether independent Task3 has already delivered their Coordinate mode. Modify `pyproject.toml` only after qualification. The first delivered task creates the runner/import helper; later tasks extend the same files, preserving existing modes/tests.

**Interfaces:** Consumes no task output. Produces lazy `probe_ranking_dependency` / `sort_permutations` signatures above and import-safe qualification runner. Head/loss methods are added in Task6, not stubbed now.

- [ ] **Step 1 — Write failing adapter/runner tests.** `test_sorter_orientation_row_ce` uses `c=[.4,-.7,.9,-.2]`, verifies `sorted=c@P`, hard-limit rank indices `[1,3,0,2]`, and row-CE equals `-(P_t*log(P_s)).sum()/R`, not an all-cell mean. `test_sorter_counts_and_guards` covers every R in2..49 (with extended edge fixtures at2/3/4/7/16/17/31/49), ties/extreme cosine, positive student-fixture P (nonnegative teacher target), residual<=1e-5, finite student-only gradients; `test_probe_is_lazy_and_preserves_rng` checks no eager optional import and unchanged Python/NumPy/torch RNG. `test_sorter_source_identity_and_lru` rejects a same-version modified implementation, accepts matching version/source identity when no installed wheel archive is recoverable, and exercises R2/3/7/16/49/2: cache length never exceeds4, stores no query/reference/autograd graph, and an evicted/rebuilt sorter reproduces outputs and gradients. Runner tests use `_load_dense_ranking_script()` above to assert that import is side-effect-free, then call `main(argv)` explicitly; validate bounded streaming summaries, nonzero exit on invalid cases and temporary-directory JSON output. Test import safety with argv/I/O/training entry points trapped, not by importing a `scripts` package.
  Assertion anchor in `test_sorter_orientation_row_ce` (one query, fixture P):
  ```python
  torch.testing.assert_close(c @ P, sorted_c, rtol=0, atol=1e-6)
  assert P.argmax(dim=0).tolist() == [1, 3, 0, 2]
  assert (P.sum(-1) - 1).abs().max() <= 1e-5
  ```

- [ ] **Step 2 — Verify red.** Run `python -m pytest tests/training/test_dense_sorter.py tests/test_dense_ranking_qualification.py -q`; expect missing adapter/runner, not a dependency skip called success.
- [ ] **Step 3 — Obtain separately authorized isolated dependency qualification.** Candidate wheel `diffsort-0.2.0-py3-none-any.whl`, SHA256 `95bd5541ade48919cd926366f83165e07645b6e62fe798343b7f0d0332b8a08b`; sdist SHA256 `9af05f475cd7c5bd307d8489d3fefe1c3ec26d639f7fd2c0f0ecb0ed48d91f98`. Read-only authoring verified wheel API and its functional/networks sources equal the previously audited source; this is **not installation/runtime qualification**. Inspect packaged license and dependency metadata. With operator approval, install in an isolated environment, not by silently mutating the working interpreter.
- [x] **Step 4 — Implement the minimal adapter/runner.** Explicit Cauchy/bitonic/100, float32, input-by-rank P, shared finite/nonnegative/stochasticity checks, student-log positivity at its loss/test call site, deterministic four-size LRU; no log clamp or loss/objective/device fallback. `main(argv: Sequence[str] | None=None) -> int` in the script adds the working `--mode numerical` sorter probe plus `--device`, `--seed`, `--steps`, `--query-rows`, `--output`; stream counts/minP/maxResidual and failure case, no full tensor log. Import the optional adapter only inside this explicitly selected mode. Preserve Task3's `coordinate` mode if already delivered; expose no replay/memory stubs (Tasks7/8 add them). Never auto-install.
- [ ] **Step 5 — Qualify the sorter, not yet the whole recipe.** First measure a100-step CPU sorter probe to budget the longer sequence; obtain approval for material runtime/environment work. Proposed sorter sequence: three seeds0/1/2, each10000 minibatches of4 synthetic rows, cycling all R2..49 plus constructed order/tie/extreme cases; record exact calls, elapsed time, counts, dtype/device/version/hash and failures. Run supported-device tiny sorting forward/backward as well as CPU. Task3 adds norm/rho fixtures and Task8 completes integrated numerical/replay/memory qualification: **do not mark those gates passed here**. Sorter failure blocks package acceptance; revise through the design gate, not widened guards. Set `ranking = ["diffsort==0.2.0"]` only if the candidate's package/sorter checks pass; otherwise stop for another approved candidate. No scientific or production-memory claim.
- [ ] **Step 6 — Green and checkpoint.** Re-run the named tests in the qualified environment; expect passes with no unreported skips, and run `python -m pytest tests/test_cli_schema.py -q` to check lazy discovery. Record exact evidence paths under an operator-approved runtime directory. If committing is separately authorized, stage only this task's explicit files and use `feat: add qualified optional dense sorter`; do not push/release.

### Task 2: Add enabled-only geometry batches and correspondence sampling

**Files:** Create `src/otuformer/training/dense_geometry.py`, `tests/training/test_dense_geometry.py`; modify `training/dataset.py`, `training/pretrain_loop.py` (`_unpack_pretrain_batch` and its two callers), relevant `tests/test_spatial_random_geometry.py` / queue-envelope assertions only where the return contract changes.

**Interfaces:** Produces `DenseBatchGeometry`, sample/pair records, `build_dense_pairs` defined above. Add `MultiCropDataset(..., return_geometry: bool=False)` and `collate_pretrain_geometry(batch: list[object]) -> dict[str,object]`. Extend `_unpack_pretrain_batch(batch, *, queue_enabled: bool, dense_enabled: bool=False, local_crops: int | None=None, include_weak_teacher_views: bool | None=None) -> tuple[list[Tensor],tuple[str,...] | None,DenseBatchGeometry | None]`; enabled callers must supply both resolved layout values (not infer weak slots from untrusted None records). Validate `2+local_crops+(2 if weak else 0)` views and the exact ordinary/weak trace slots, plus per-view tensor batch/canvas and same-row source-size agreement. Update both actual callers to unpack three values; off callers retain defaults and historical validation.

- [ ] **Step 1 — Write failing tests.** `test_four_envelopes_exact_keys` pins list / `{views,image_key}` / `{views,geometry}` / `{views,geometry,image_key}`. `test_weak_geometry_none_without_tracking` spies ordinary/weak transforms: only ordinary global/local call `transform_with_geometry`. `test_ragged_trace_order_and_rejection` covers variable operations, rows/keys/view order, extra fields and illegal None slots. `test_actual_centers_complete_support_and_pair_order` checks A->B/locals and B->A/locals, no self/weak, known64/16 point(24,40), rotation/clipping/padding boundaries, different grids and zero/one candidate. `test_all_augmentation_profiles_trace_parity` parametrizes every supported profile/orientation plus crop/resize/rotation/flip/pad/affine operations and compares image tensors and Python/NumPy/torch RNG with tracing off/on. Include JSON round-trip and mismatched expected-weak layout (well-formed but wrongly placed None) fixtures.
  Assertion anchor in `test_four_envelopes_exact_keys` / below-cap fixture:
  ```python
  assert set(collated) == {"views", "geometry", "image_key"}
  assert geometry[weak_index] is None
  assert torch.equal(query_rng_before, query_generator.get_state())
  ```

- [ ] **Step 2 — Verify red.** `python -m pytest tests/training/test_dense_geometry.py -q` must fail on the absent enabled path/API.
- [x] **Step 3 — Implement just the handoff.** Use existing realized trace, support and sampler contributors; collate ragged traces, never resample augmentations. No new support/trace on all-off or weak views. At cap use all canonical points/no draw; above cap sample distinct CPU indices, sort output order. Respect query/ref stream separation and retain graph-free coordinate metadata. Loop remains dense-disabled until Task4.
- [ ] **Step 4 — Green and parity.** Run `python -m pytest tests/training/test_dense_geometry.py tests/test_spatial_random_geometry.py tests/test_spatial_features.py -q` plus existing queue batch tests selected by actual node IDs. Compare tensor/RNG states before/after trace and across old list/keyed paths; use existing fixtures rather than a new global conftest model.
- [ ] **Step 5 — Review checkpoint.** If separately authorized, stage exact Task2 source/tests and commit `feat: add geometry-aware dense training batches`. No general dataset rewrite.

### Task 3: Implement Coordinate and numerical checks with unequal denominators

**Files:** Create `src/otuformer/training/dense_objective.py`, `tests/training/test_coordinate_objective.py`; create or extend `scripts/check_dense_ranking.py` and `tests/test_dense_ranking_qualification.py` with independent Coordinate norm/rho qualification and the single path-load helper above. Task1 is not a prerequisite: if absent, create only the working Coordinate mode and preserve that ownership when Task1 is later delivered.

**Interfaces:** Consumes Task2 pair records; produces `DenseLossResult`, `sample_dense_features`, `reduce_dense_losses`, `coordinate_loss` as frozen above. Features remain BHWC; no masked predictor/prototype/CLS features.

- [ ] **Step 1 — Write failing objective tests.** `test_coordinate_raw_sample_then_l2` distinguishes unequal-magnitude `[2,0]`/`[0,1]` mixing from unit-first interpolation. `test_coordinate_point_sample_pair_means` constructs unequal point/sample counts and checks the exact three-level mean and 6:1 all-valid pair weighting at six locals. `test_norm_and_relative_cancellation` checks finite norm>1e-6 and rho>1e-6, almost opposed nonzero vectors, exact-center rho1, strict failure with pair/sample context, no geometric denominator filtering. `test_coordinate_gradient_and_unavailable` checks analytic 1-cos, student-only gradient, zero/one point and raw=None when no pairs. `test_dense_sampling_device_and_boundary` compares supported-device sampling to CPU for finite, boundary and nonrepresentable coordinates, with no float64 accelerator allocation; unsupported kernels remain an explicitly reported device gate. `test_coordinate_runner_import_safe` and `test_coordinate_runner_summary` load the script by the fixed helper, trap `diffsort` import, then call Coordinate mode with a tiny seeded valid/invalid fixture and check norm/rho/zero-population summaries and explicit exit status.
  Assertion anchor in the equal-weight raw-interpolation fixture:
  ```python
  expected = torch.tensor([2.0, 1.0]) / 5.0**0.5
  torch.testing.assert_close(sampled_unit, expected)
  assert teacher.grad is None and student.grad is not None
  assert no_overlap.raw is None
  ```

- [ ] **Step 2 — Verify red.** Run `python -m pytest tests/training/test_coordinate_objective.py -q` and `python -m pytest tests/test_dense_ranking_qualification.py -k coordinate -q`; expect the missing objective/Coordinate mode (not an absent test file or unrelated package error).
- [x] **Step 3 — Implement minimal tensor logic.** Use existing `sample_grid_features` and matching nonnegative bilinear contributors to compute rho; do not fork coordinate conversion. FP32 reductions, no cosine/log clamp or extra factor2. Detached teacher, per-sample point mean before pair mean. Return cached minima/counts only; no extra backward or claimed anti-smoothing guarantee. Add working runner `--mode coordinate` using these guards, the same `main(argv)` signature, `--device/--seed/--steps/--query-rows/--output`, and lazy imports; no sorter or ranking dependency on this path.
- [ ] **Step 4 — Green and qualification update.** Run `python -m pytest tests/training/test_coordinate_objective.py tests/training/test_dense_geometry.py -q` and `python -m pytest tests/test_dense_ranking_qualification.py -k coordinate -q`; named runner tests must execute, not skip because Task1/`diffsort` is absent. After explicit long-run approval, Coordinate mode can independently run the Task8 norm/rho sequence with the same declared seeds/counts and no sorter, including valid/invalid and tail coverage. Freeze only the tested Coordinate numerical policy after that gate; ranking still needs its separate permutation qualification. Later Task8 reuses matching evidence instead of rerunning an unchanged source/configuration.
- [ ] **Step 5 — Review checkpoint.** If authorized, stage only Task3 paths and commit `feat: add cross-view coordinate objective`. A callable alone is not full feature acceptance; Task4 must connect the lifecycle.

### Task 4: Deliver a live Coordinate recipe, strict recovery and reporting

**Files:** Create `src/otuformer/training/dense_recipe.py`, `tests/training/test_dense_recipe.py`, `tests/training/test_coordinate_pretrain.py`, `tests/training/test_dense_pretrain.py`; modify `training/pretrain_recipe.py`, `pretrain_loop.py`, `reporting.py`, `cli/pretrain.py`, `cli/constraints.py`, `cli_schema.py`, `tests/test_output_safety.py` and relevant CLI/schema tests.

**Interfaces:** Consumes Tasks2/3. Produces `DenseSettings`, `DenseRuntime` and exact bundles above. Extend `PretrainSettings` with `dense: DenseSettings` using a default-factory for the historical all-off value (0,0,49,None) so existing direct callers remain off; pass the actual once-resolved instance from recipe preflight into that field and through `_PretrainHandoff`/current recipe, not a second CLI resolution. Raw `args` explicitness stays distinct. Implement `validate_dense_resume(config: dict,state: dict,*,iteration: int,batch_size: int,steps_per_epoch: int) -> None` plus existing complete model/optimizer recovery checks in their current owner.

- [ ] **Step 1 — Write failing resolution/safety tests.** New Coordinate0/all-off, positive/negative/NaN/inf, integer/bool caps, explicit inactive options, absent legacy bundles, unknown keys/versions, omitted inherited weights/caps, explicit conflict and forbidden resume enablement. `test_dense_preflight_preserves_output_on_error` plants an existing sentinel/log/CSV then checks every malformed case fails unchanged before setup. `test_dense_startup_probe_preserves_generators` verifies the initial-forward probe consumes no dense sampling draw/contribution. `test_coordinate_nonfinite_post_step_stops_commit` injects a nonfinite optimizer result with FIFOoff and checks that enabled Coordinate aborts before EMA/center/checkpoint commit. `test_dense_statistics_unavailable_is_none` distinguishes zero real counts from missing loss/norm/rho populations and verifies global/local attempted/contributing and reference-shortage counters. A mismatched saved support-policy identifier must fail strict resume.
  Assertion anchor for enabled/off checkpoint fixtures and failed preflight:
  ```python
  assert coordinate_ckpt["config"]["dense_training_config"]["objectives"] == ["coordinate"]
  assert "dense_training" not in off_ckpt
  assert output_snapshot_after == output_snapshot_before
  ```

- [ ] **Step 2 — Verify red.** Run `python -m pytest tests/training/test_dense_recipe.py tests/training/test_coordinate_pretrain.py tests/training/test_dense_pretrain.py -q`. Missing live options/forward/state/reporting must fail, not only test helper arithmetic.
- [x] **Step 3 — Implement Coordinate end to end.** Expose `--coordinate-weight` and `--dense-query-points` now; do not expose ranking switches until Task6 works. All enabled Coordinate dense-config bundles include ranking_weight0; the all-off path still omits both dense bundles. Require both ordinary teacher globals; retain existing local `student(lv)` patch output **already prefix-free** while slicing global `forward_pretrain().tokens` once. No extra local forward; verify real grid. Build shared queries once per training step, add weighted available Coordinate result before existing single backward, leave DINO/FIFO center/update order unchanged. The existing startup/initial-forward probe validates geometry layout but does not build/sample queries, advance dense RNG or count a training contribution; `test_dense_startup_probe_preserves_generators` pins this boundary.
- [ ] **Step 4 — Save/recover exact enabled state.** Extend existing public preflight helpers and handoff rather than imports from loop internals. Validate complete encoders/objectives/AdamW/schedules/RNG before outputs. Dense resume inherits effective batch size and checks loader steps; keep FIFO stronger locks if active. Restore generators before the next query, never sync resumed teacher/student. Extend the existing queue-only post-optimizer finite-parameter check to enabled Coordinate before EMA/center/checkpoint commit; failure aborts without auto-retry or a rollback claim. Task7 reuses this boundary and includes the new student head. Omit all dense state/modules on off path.
- [x] **Step 5 — Implement reporting and live option docs.** Log resolved applicability once and cached raw/weighted Coordinate/count/minnorm/rho stats. Capture the **actual pre-change current CSV headers** in tests, register only recognized predecessors, migrate old rows with empty new fields and reject unknown/duplicate headers unchanged. Update the live Coordinate subset in both pretrain command docs and runtime schema/card constraints at this task; do not advertise ranking early. Schema must stay model/dependency-free.
- [ ] **Step 6 — Green live lifecycle and off parity.** Run named new tests plus `tests/training/test_pretrain_alignment.py`, `tests/training/test_dino_pretrain.py`, `tests/training/test_masked_pretrain.py`, `tests/training/test_ssl_config.py`, `tests/test_cli_schema.py`, `tests/test_output_safety.py` and `tests/test_docs.py` using explicit paths. Real tiny loops cover both SSL methods and four patch modes; off checks compare tensor/RNG/loss/updates/header/state layout with the baseline path. Retain CPU smaller-local exceptions from policy, not silent accelerator fallbacks.
- [ ] **Step 7 — Review checkpoint.** Mark Coordinate stage complete only after save/resume and actual tiny loop pass, not from a loss unit test. If authorized, stage exact Task4 files and commit `feat: integrate coordinate pretraining and recovery`.

### Task 5: Add strict backbone-only source initialization

**Files:** Modify `src/otuformer/utils/checkpoint.py`, `training/pretrain_recipe.py`, `pretrain_loop.py`, `cli/pretrain.py`, `cli/constraints.py`, `cli_schema.py`; create `tests/training/test_pretrain_initialization.py`; extend `tests/test_checkpoint_utils.py`, `tests/test_output_safety.py` and both pretrain command docs; extend `tests/training/test_dense_pretrain.py` only if Task7 has already delivered it (otherwise Task7 adds the joint initialization/dense case).

**Scheduling:** No prerequisite implementation task. Task5 may be reviewed and implemented before Tasks1–4; its Stage A position is document adjacency only. The initializer works with both dense weights zero; its source-preparation/transfer helpers must not depend on dense/ranking modules or require `diffsort`. This does not forbid recipe orchestration from resolving its ordinary all-off settings. If executed before Task4, add its optional handoff field to the existing `_PretrainHandoff`; Task4 later preserves it rather than replacing that handoff. Before implementation begins, select one actual order in the execution record (`Task5-before-Task4` or `Task4-before-Task5`) and bind the handoff/test snapshots to that sequence and source revision. Do not merge test evidence from alternative histories. Neither order is selected or executed by this plan revision.

**Interfaces:** `PretrainInitialization(path: str, sha256: str, source_key: str, source_identity: tuple[str,str], architecture: CheckpointArchitecture, image_size: int, register_tokens: int, backbone_state: dict[str,Tensor])` belongs in `utils/checkpoint.py`. Include explicit image/register fields because current `CheckpointArchitecture` does not own input size; `register_tokens` is the resolved actual total, while `architecture.added_register_tokens` retains the native-versus-added construction distinction. `default_model_name` supplies the existing caller fallback for recognized historical payloads without model metadata; metadata takes precedence, explicit incompatible requests fail, and strict selected-branch validation must never invoke the shared reader's fallback to a different branch. `prepare_pretrain_initialization(path: Path, *, source_key: str, default_model_name: str) -> PretrainInitialization` performs one stable load/hash and recognized source validation. `initialize_pretrain_backbone(student: nn.Module, prepared: PretrainInitialization) -> None` loads exact backbone keys only; use structural module/state interfaces in the helper to avoid importing trainer orchestration. Extend `_PretrainHandoff` with `initialization: PretrainInitialization | None` and resolve once per invocation. Produce the closed initialization bundle above.

- [ ] **Step 1 — Write failing source/identity tests.** `test_selected_teacher_or_student_backbone_only` saves distinguishable branches and verifies only the selected backbone transfers, fresh CLS/patch heads/center/optimizer/queue, identical new-run student/teacher after ordinary initialization. `test_init_missing_selected_branch_never_falls_back` rejects even if another valid branch exists. `test_init_register_and_identity_validation` covers valid native/added registers and rejects incompatible layouts/model/input-position-grid/shape, malformed source projector identity and nonfinite selected encoder state. Recognized legacy SSL layouts cover both recorded-model precedence and omitted-model metadata with the explicit caller fallback; metric/raw/ONNX inputs and missing source projector state fail explicitly. Write Step4's lifecycle/safety tests now too; Step2 runs the complete file before any initializer implementation.
  Assertion anchor in the selected-branch fixture (source branches deliberately differ):
  ```python
  torch.testing.assert_close(target_state[backbone_key], selected_source[backbone_key], rtol=0, atol=0)
  assert new_iteration == 0 and not optimizer.state
  assert current_identity == requested_identity
  ```

- [ ] **Step 2 — Verify red.** `python -m pytest tests/training/test_pretrain_initialization.py -q` fails before adding the new options/strict loader.
- [x] **Step 3 — Implement prepared source transfer.** Add `--init-checkpoint` / `--init-source teacher|student`, explicit inactive/source/resume exclusivity validation before output mutation, source architecture/global/register inheritance for omitted requests and strict explicit conflicts. Source normalization cannot resize position tensors or fabricate registers. Hash the exact bytes loaded (or stable verified file snapshot), not path text. Never call the permissive source fallback; do not require source optimizer/RNG because initialization is not resume. Avoid a timm download when the complete backbone is supplied.
- [ ] **Step 4 — Pin lifecycle/safety behavior.** `test_init_then_resume_without_source_file` deletes or moves the source after the first new checkpoint and resumes from recorded provenance successfully. `test_init_changed_source_snapshot_rejected` uses a controlled read/hash race and verifies output sentinel unchanged. `test_init_no_source_dense_off_and_method_switch` covers initialization-only, DINO<->relational with fresh current heads, and regular no-source new-run parity. When the dense loop already exists, also add `test_dense_initialized_resume_without_source_file` to `tests/training/test_dense_pretrain.py`; otherwise Task7 owns that joint test. Explicit `init_source` without path and init+resume fail. Old source heads/dense state never become current training state.
- [ ] **Step 5 — Green and docs.** Run the new file plus `tests/test_checkpoint_utils.py`, `tests/training/test_finetune_provenance.py`, `tests/test_output_safety.py`, `tests/test_cli_schema.py`, `tests/test_docs.py`. Schema/card registers source input protection, including its bytes-hash checks across approval/launch; no overwrite may destroy the initialization source. Update bilingual live option docs and record immediate-source vs current SSL identity.
- [ ] **Step 6 — Review checkpoint.** If authorized, stage explicit Task5 paths and commit `feat: initialize pretraining from an OTU backbone`. This task is included by operator decision, not an implied universal initializer.

## Stage B — Independent NeCo and Joint Training

### Task 6: Implement ranking heads, pair references and row-CE

**Files:** Extend `src/otuformer/training/neighbor_ranking.py`, `dense_objective.py` only for shared numerical helpers, and `dense_recipe.py` for ranking serialization already frozen; create `tests/training/test_neighbor_ranking_objective.py`.

**Interfaces:** Consumes Task1's sorter, Tasks2/3's geometry/numerics, and Task4's `DenseSettings`/serialization owner. `NeighborRankingObjective.compute(student_grids: dict[int,Tensor], teacher_grids: dict[int,Tensor], pairs: tuple[DensePairQueries,...], *, patch_hw: tuple[int,int], reference_cap: int, generator: torch.Generator) -> DenseLossResult`; `update_teacher(momentum: float) -> None`. Exact head dict keys are `student_head.0/2/4.{weight,bias}` and `teacher_head.0/2/4.{weight,bias}` in the module; save **separate submodule state_dicts** in the closed bundle (`0/2/4.*` inside each). No prototypes/CLS head reuse.

- [ ] **Step 1 — Write failing feature/reference tests.** `test_project_l2_sample_cosine_order` uses a nonlinear unequal-vector field to reject MLP-after-interpolation; teacher outputs/reference pool have no gradient. `test_pair_pool_is_selected_teacher_queries` tags batch/pair positions, verifies U_v contains exactly teacher samples for this pair, no unrelated view/raw-grid patches; `test_same_image_self_and_unique_reference_entries` permits self, handles duplicate-valued vectors as distinct entries, and never repeats pool indices to fill cap. `test_reference_rng_is_separate` checks below/at cap no draw and above cap per-sample distinct uniform draws. `test_row_ce_upstream_layout_equivalence` asserts at R4/16/49 `loss=-(P_t*log(P_s)).sum()/R`, equals the upstream summed-axis then remaining-axis mean, and differs by factorR from an all-cell mean. `test_ranking_reduction_unequal_queries_samples_pairs` checks three-level equal-pair reduction; `test_soft_equal_target_not_zero` distinguishes teacher entropy from disagreement. `test_ranking_zero_one_pool_and_cancellation` checks None availability vs numerical abort, no loss clamp, ties/nonpower counts, both norms/rho. `test_permutation_role_guards` accepts a nonnegative doubly stochastic teacher target with exact zeros against positive student P, rejects student zero/negative/nonfinite entries before log, and rejects negative/nonfinite/unnormalized teacher entries. `test_chunked_single_backward_parity` compares unchunked versus ordinary forward chunks for loss and student-head/backbone gradients under unequal sample/pair counts; it makes no retained-memory claim. Backward must reach student head/backbone only. **All these tests belong to Step1 and are included in Step2's red run.**
  Assertion anchor in `test_row_ce_upstream_layout_equivalence` (one query):
  ```python
  cells = -(P_t.detach() * P_s.log())
  torch.testing.assert_close(loss, cells.sum() / R)
  torch.testing.assert_close(loss, R * cells.mean())
  assert P_t.grad is None and references.grad is None
  ```

- [ ] **Step 2 — Verify red.** `python -m pytest tests/training/test_neighbor_ranking_objective.py -q` must fail because compute/heads are not present, not because qualified dependency tests were skipped.
- [x] **Step 3 — Implement heads and references.** Three biased Linear layers D->2048->2048->256, GELU between, truncated-normal std.02/zero bias; teacher exact initial copy and nontrainable. Project fields once per view, L2, apply Task3 guarded sampling then cosine L2. For each ordered pair concatenate teacher selected queries in row/query order; each contributing sample draws R=min(cap,N_v) unique entries, shared across its student/teacher queries. Q=0 excludes that sample; N<2 makes ranking unavailable, not Coordinate. No feature-value deduplication, background filter or cross-pair fallback.
- [ ] **Step 4 — Verify exact objective tests pass.** Verification only: run the tests already written in Step1 and observed red in Step2. Run `python -m pytest tests/training/test_neighbor_ranking_objective.py tests/training/test_dense_sorter.py tests/training/test_coordinate_objective.py tests/training/test_dense_geometry.py -q`; expect all selected tests PASS in the qualified environment.
- [ ] **Step 5 — Review checkpoint.** If authorized, stage only Task6 files and commit `feat: add pair-conditioned neighbor ranking objective`. Unit success is not live ranking acceptance.

### Task 7: Wire ranking into the one-step lifecycle and strict joint recovery

**Files:** Modify `training/pretrain_recipe.py`, `pretrain_loop.py`, `dense_recipe.py`, `reporting.py`, `cli/pretrain.py`, `cli/constraints.py`, `cli_schema.py`, `scripts/check_dense_ranking.py`; extend `tests/training/test_dense_pretrain.py` for lifecycle, recovery and reporting; extend `test_dense_recipe.py` and existing FIFO/DINO optimizer-boundary tests where behavior is affected.

**Interfaces:** Consumes Tasks1–4/6; preserves Task5's initialization handoff when delivered, but initialization is not a prerequisite for the dense loop. Extend `_student_optimizer_parameters(..., ranking_objective: NeighborRankingObjective | None=None)` and `_no_decay_parameter_ids` / `_pretrain_optimizer_groups` with the same keyword argument; enumerate **student head only**, deterministic once, never teacher aliases. Production optimizer validation reads those same expected groups. Serialize/restore heads via the exact schema, update `DenseRuntime.ranking_contributed_steps` at the success boundary; serialize it only as `dense_training.ranking.contributed_steps`. Add `--ranking-weight` and `--ranking-reference-points` only when live delivery works.

- [ ] **Step 1 — Write failing live-step tests.** `test_dense_modes_and_patch_matrix` parameterizes2 SSL methods*4 patch modes*4 dense combinations (32 cases) using real tiny models/fixtures. Assert components/forwards, finite gradients and raw/weighted values rather than requiring the total to fall. `test_ordinary_teacher_independent_of_weak_targets` covers weak relational/FIFO and patch-none requiring both ordinary teachers; no teacher-local or extra local backbone forward. `test_prefix_contract_global_and_local_registers` proves global slice exactly once and no local double-slice. `test_koleo_and_fifo_dense_composition` exercises KoLeo on/off for both methods and permitted FIFO modes without expanding the core32-case matrix; disabled features keep their existing state and update behavior. `test_ranking_contributed_then_skipped_step` first populates real AdamW state, then makes ranking unavailable and checks head grads remainNone, student head/moments/steps remain bitwise unchanged, contribution count does not advance, and EMA follows the successful other-objective step. `test_dense_nonfinite_post_step_stops_commit` injects a nonfinite optimizer result with FIFOoff and verifies failure before EMA/center/FIFO/checkpoint commit. `test_dense_initialized_resume_without_source_file` is written here when Task5 is present (or added by Task5 if delivered later).
  Assertion anchor for a fresh all-ranking-skipped real-loop step:
  ```python
  assert all(p.grad is None for p in ranking.student_head.parameters())
  assert not any(p in optimizer.state for p in ranking.student_head.parameters())
  assert runtime.ranking_contributed_steps == 0
  ```

- [ ] **Step 2 — Verify red.** Run `python -m pytest tests/training/test_dense_pretrain.py -q`; missing ranking loop/state integration must fail.
- [ ] **Step 3 — Implement one backward with correct optimizer behavior.** Keep total construction and success order: compute old/new terms -> zero grad (`set_to_none=True` explicitly for enabled dense training) -> total.backward -> joint finite-grad/clipping -> optimizer.step -> existing EMA and ranking teacher EMA -> existing successful queue commit/counter/schedule bookkeeping. Preserve verified relative ordering, do not introduce a queue precommit. Reuse Task4's enabled-dense post-optimizer finite-parameter check, including the ranking student parameters before EMA/center/commit; a failure aborts and does not save or auto-retry (it is not rollback of the optimizer mutation). Ranking None leaves head grad=None after zeroing, so no AdamW head decay/moments are created. EMA head can still track an unchanged student on a successful old-objective step. Test default.05 and nondefault.07 weights/bias coefficients and global opt-out bias0; do not hardcode .05 into a new group.
- [ ] **Step 4 — Implement strict state/preflight.** Both head dicts/reference RNG exist iff ranking positive, even if never contributed. Before output setup validate complete config/state/schedule/batch count/model/patch/DINO/FIFO and usable AdamW state using extended existing helpers. Zero contributed count requires no head optimizer entries; positive count requires matching per-parameter steps/moments, finite tensors and correct group membership. For this always-connected three-Linear head when ranking contributes, every saved AdamW head step equals `ranking_contributed_steps`; it is not the overall loop iteration. A skipped step leaves previously accumulated moments/steps unchanged. Unknown/missing/extra bundles, negative/impossible counts, incomplete RNG, mismatched qualified package and explicit cap/weight changes each have named failure tests. When Task5 is delivered, `test_dense_initialized_resume_without_source_file` must instead **succeed** after the initialization source is removed, preserving recorded provenance without reopening/hashing that source; missing required dense state is a different failure. Historical absent bundles cannot be enabled on resume. All-off serialization stays historical.
- [ ] **Step 5 — Pin controlled replay.** Add runner `--mode replay` using production loop/checkpoint interfaces with synthetic portable data and a lightweight real model, never importing `tests/` fixtures into the script; checkpoints remain at epoch boundaries. `test_dense_generator_cap_crossing_replay` uses query48/49/50/48 and ref15/16/17/15, checkpoints before/after relevant draws, and compares later indices/loss/head/optimizer/RNG. Mixed batches include eligible A5+B48/49/50 and separately selected(5,10)/(5,11)/(5,12); assert canonical pool and reference-stream advancement can affect later samples/batches but not the query/augmentation stream. Same eligibility and order are required; changed data is not claimed identical. Exact replay remains CPU/worker0; ordinary integration uses policy device rules.
- [ ] **Step 6 — Complete reporting, safe schema and green tests.** Expose both new weights, applicable caps/inherited provenance and independent initialization in cards. Keep extra imports lazy; mock absence must leave Coordinate/off/read-only commands valid and positive ranking fail before mutation. Extend recognized-header migration with a real resume of the pre-dense header and the Task4 intermediate header; unknown/duplicate headers remain untouched. **Rendered surface:** `InstantMetricsLogger.plot` must append one dense diagnostics page to `training_curves_pretrain.pdf` when dense columns carry data (both raw losses beside the existing total, both weighted contributions, query population, contributing denominators, reference population/count with shortages and skipped log points, and separate min-norm/min-rho/min-P panels), and keep exactly the historical single page otherwise. A CSV column is not reporting acceptance: tests assert the page count and the panel titles from a real enabled run. Run new tests, `tests/training/test_relational_queue_recipe.py`, `tests/training/test_fifo_optimizer_boundaries.py`, `tests/training/test_dino_resume.py`, `tests/test_cli_schema.py`, `tests/test_output_safety.py`, `tests/test_docs.py`. Distinguish unsupported-device tests from genuine failures/skips.
- [ ] **Step 7 — Review checkpoint.** If authorized, stage exact Task7 paths and commit `feat: integrate ranking objectives and strict recovery`. Do not call ranking accepted until the real-loop/memory/consumer gates below pass.

## Stage C — Cost, Consumers, Documentation and Acceptance

### Task 8: Qualify actual numerical/replay and training-memory budgets

**Files:** Extend `scripts/check_dense_ranking.py`, `tests/test_dense_ranking_qualification.py`; update this plan's execution evidence only when actually run. Keep generated logs/JSON/checkpoints in an operator-approved runtime directory, not in source/test fixtures.

**Interfaces:** Script modes now include coordinate/numerical/replay/memory. Retain working `coordinate`/`numerical`/`replay` modes; add `memory` only with its implementation. Memory mode adds explicit `--batch-size`, `--global-size`, `--local-size`, `--local-crops`, `--patch-size`, `--query-cap`, `--reference-cap`, `--steps`; it must not reinterpret caps, change production defaults or auto-install. JSON result includes exact command/environment/source, effective config, actual per-pair queries/references, stage times and measured/censored memory scope.

- [ ] **Step 1 — Write failing evidence tests.** `test_memory_report_keeps_actual_and_bound_separate` checks B32/global224/local96/patch16/L6 yields **all-valid counting bound16960** and generally `0 <= actual_queries <= configured_bound`; an all-valid fixture may equal the bound. A separate explicitly reduced-overlap fixture must satisfy `0 < reduced_overlap_queries < configured_bound` and report its actual population separately. `test_forward_chunking_does_not_claim_retained_memory_relief` uses saved-storage accounting/recomputation metadata, not a false streaming guarantee. `test_report_distinguishes_external_and_local_measurements` prevents copying reviewer bytes or source-snippet extrapolations into measured installed/GPU fields.
  Assertion anchor with separate general, all-valid and reduced-overlap fixtures:
  ```python
  assert configured_bound == 32 * (2 * 49 + 12 * 36) == 16960
  assert 0 <= actual_queries <= configured_bound
  assert all_valid_queries == configured_bound
  assert 0 < reduced_overlap_queries < configured_bound
  assert report["measured_peak_device"] != "cpu-extrapolation"
  ```

- [ ] **Step 2 — Verify red and implement real probes.** `python -m pytest tests/test_dense_ranking_qualification.py -q` first exposes the missing memory/evidence behavior through Step1's assertions, then implement small isolated checks preserving CPU fixture state. No approximate count can stand in for a real-loop peak. Use existing phase timing conventions; synchronize accelerators only inside explicit benchmark mode, never normal training logs.
- [ ] **Step 3 — Run the authorized staged numerical/replay qualification.** The `replay` mode must restart **cold**: after each epoch-boundary save, discard the head, optimizer and runtime, rebuild them from the file through `restore_dense_runtime` and the closed head/optimizer state, and prove the restore was not a no-op (a fresh head must differ from the saved one before loading). A reused live object is not recovery evidence. Finish Task1's justified long CPU stream **using the qualified installed release and final common guards**, record minP/minnorm/minrho/max stochasticity residual plus failure/seed/tail context, then run selected-device representative probes. Distinguish deliberate invalid cases. A recurring low-rho population near indicative.1 requires review, not automatic threshold tuning. No finite run proves100000-step safety; report measured coverage. Before the first long run, record the100-step measured cost and projected budget for each seed; ask approval if the longer run exceeds that budget. Default planned CPU sequence per relevant mode is3 seeds*10000 iterations*4 query rows=120000 query instances, with coverage reported separately by actual R and valid/constructed case. Coordinate norm/rho-only qualification can run after Tasks2/3 without Task1 or `diffsort`; permutation qualification requires Task1, real ranking replay/costs require Tasks6/7. Reuse already recorded evidence only when source/config/environment still match; no double-counted population. Extend the integrated stream to raw/per-patch-normalized fields, global224/local96 and unequal small-grid overlap fixtures; stream min student/teacherP separately, norm/rho histograms and low-rho tail counts, never treating deliberately invalid fixtures as expected-valid evidence. Run the real epoch-boundary replay mode after the stream. Recipe version1 stays provisional until these gates pass.
- [ ] **Step 4 — Measure real-loop costs with explicit approval.** Begin with a tiny model/batch to bound runtime; then measure intended supported production shape/batch combinations without changing the request to force a pass. Compare baseline, Coordinate-only, ranking-only and joint, with both SSL methods/patch interactions selected from actual footprint. Reference16 is the approved default,49 a separate opt-in stress case. Record startup/first/steady/eval, actual contributing queries/refs, CUDA allocated/reserved peak when applicable and retained storage separately; CPU/MPS limitations are explicit. Current source-snippet estimates3.805/.217GiB are not acceptance thresholds.
- [ ] **Step 5 — Stop on an unfit default; otherwise record evidence.** OOM, unsupported package/device or provisional numerical failure blocks that configuration. Obtain a design revision before shrinking counts, adding checkpointing/repeated backward/device fallbacks or widening guards. Green evidence means tested configurations pass with exact device/package/source recorded, not all hardware certified. Research datasets/scientific efficacy remain outside this benchmark authorization. If authorized, commit only probe/tests/evidence-accounting edits as `test: qualify dense training numerical and memory boundaries`.

### Task 9: Verify downstream readers, complete docs and run milestone regression

**Files:** Modify only where needed: `src/otuformer/utils/checkpoint.py`, `src/otuformer/embedding/extractor.py`, `src/otuformer/spatial/extractor.py`, `src/otuformer/vision/export.py`, `src/otuformer/vision/cam.py`; tests `tests/test_checkpoint_utils.py`, `tests/test_spatial_extract.py`, `tests/test_spatial_export.py`, `tests/test_spatial_cam_parity.py`, `tests/test_extractor.py`, `tests/test_export.py`, `tests/test_cam.py`, `tests/test_spatial_onnx.py`, and `tests/training/test_finetune_provenance.py`. Update `README.md`/`README.cn.md`, `docs/commands/pretrain.md`/`.cn.md`, `training-augmentation.md`/`.cn.md`, `embedding-metrics.md`/`.cn.md` where affected, `skills/otuformer-workflow/SKILL.md` and its workflow/troubleshooting/teaching references, master/affected specs/plans via dated notes. No duplicate `.cn.md` development docs.

**Interfaces:** Existing public extraction/export/CAM/finetune APIs and encoder/head identity remain unchanged. Training-only dense heads/config must not require the optional dependency in read-only consumers. Workflow reads installed schema and never proposes unsupported new flags on old installations.

- [ ] **Step 1 — Add failing consumer/skill assertions.** `test_dense_checkpoint_readers_without_diffsort` loads current teacher and explicit student from supported enabled checkpoints and compares global/spatial/finetune architecture and numeric outputs against the same encoder state without training-only dense bundles. ONNX/CAM tests retain their existing support/device prerequisites. `test_workflow_dense_schema_applicability_and_init_safety` checks current/older schemas, init-vs-resume protection, inactive options, full parameter card and no execution without operator approval.
  Assertion anchor in an isolated reader process with optional import trapped:
  ```python
  torch.testing.assert_close(enabled_checkpoint_embedding, encoder_only_embedding, rtol=0, atol=0)
  assert captured_optional_imports == []
  assert recovered_identity == current_ssl_identity
  ```

- [ ] **Step 2 — Check baseline compatibility, then minimally fix actual gaps.** Run the exact Step1 tests. Existing readers may already pass because training fields are additive; record that pass and leave those owners unchanged rather than manufacturing a red test or refactoring working code. For genuinely missing behavior, verify its specific red before the minimal implementation. Shared identity rejects contradictions while keeping current method/head. Update bilingual user docs/defaults/compatibility warnings only to implemented behavior, including disabled defaults, reference16/cap, memory limits, unavailable logs and old-executable resume warning.
- [ ] **Step 3 — Record design/plan extension history without rewriting it.** Master links only to specs. Add dated notes to affected masked-pretrain/CLS/relational/FIFO/spatial/modularity specs and their earlier plans **only when their boundary changed**; do not mass-edit statuses or imply old gates closed. Workflow references retain runtime-schema-first behavior. This plan and unified design status may say implemented only with commit/source/test evidence, never simply because checkboxes were ticked.
- [ ] **Step 4 — Green impacted paths.** Run `python -m pytest tests/test_docs.py tests/test_cli_schema.py tests/test_workflow_skill.py tests/test_checkpoint_utils.py tests/test_output_safety.py tests/training/test_finetune_provenance.py tests/test_spatial_extract.py tests/test_spatial_export.py tests/test_spatial_cam_parity.py -q`, `python -m pytest tests/test_extractor.py tests/test_export.py tests/test_cam.py tests/test_spatial_onnx.py -q` and every new file introduced by Tasks1–8. Verify rendered `python -m otuformer pretrain --help` against options/schema/cards in the same interpreter. No unrelated real training.
- [ ] **Step 5 — Run the required milestone/full gate.** Shared update mechanics, checkpoint format and optional packaging are full-regression triggers under policy: `python -m pytest --durations=25` with no blanket deselection/default-skip tricks. Run package build/install/floor checks only in separately authorized environments, including ranking-present and ranking-absent discovery. Record exact interpreter/import path/dependency versions/selected devices/failures/skips. Fix locally with focused tests before rerunning the final full candidate. Unavailable prerequisite is not pass.
- [ ] **Step 6 — Final acceptance handoff.** Verify actual changed-file diff, no unrelated source/real-data mutation, all-off compatibility and objective-disabled paths; report remaining environment/scientific limits. If authorized, stage exact relevant docs/consumer/tests and commit `docs: document dense pretraining and qualification`. Do not push/tag/release without explicit permission. Operator acceptance/review still precedes release.

## Plan Self-Review and Execution Gates

Author self-review at plan creation (not execution):

- Design coverage: geometry/normalization/pairs/RNG -> Tasks2/3; configuration/recovery/reporting ->4/7; explicit initialization ->5; ranking/pool/sort ->1/6; numerical/cost qualification ->1/8; consumers/CLI/schema/skill/language ->4/5/7/9. All17 design chapters have owning tasks. Historical audit attribution/approval records remain provenance, not implementation work to repeat.
- Type/name check: shared records/signatures and the closed checkpoint schema above are the only new inter-task contracts; later tasks reuse them rather than re-resolve settings or invent a second state bundle. The initializer passes the actual historical model-name fallback, geometry carries pre-cap counts/validated traces and expected weak layout, and unavailable statistics explicitly allow None. Recipe IDs/numerical defaults cannot drift without design revision.
- Review Focus: five failure classes each map to concrete named failing tests; an arithmetic/unit pass alone never marks live objective, strict recovery, installed package or memory accepted.
- Proportion: steps identify tests/assertions/signatures and exact pinned choices, not full implementation bodies. No universal engine, dependency installer, separate head schedule, geometry rewrite or unused abstraction is planned.
- Authoring verification (2026-11-15): `python -m pytest tests/test_docs.py -q` -> **44 passed**; local link/English-only/task inventory checks, the original seven review-contract checks, the16-item cross-contract audit matrix and the nine Python assertion anchors pass; all52 actual implementation steps remain unchecked. The header now uses plain words rather than an extra checkbox-shaped example, so token and step counts agree. `git diff --check` passes, with the untracked plan also checked explicitly for trailing whitespace. These are plan/document checks, not green implementation tasks, independent review, installed-package qualification or trained-model evidence.

### Review disposition — 2026-11-15

Operator-supplied plan review, not a newly dispatched independent audit:

- P1/P2: clarify independently schedulable initialization and move every Task6 test definition into Step1; Step4 performs verification only. Document order is not a dependency.
- P3: accept the narrowly scoped initial-red clarification, not the premise that pytest requires `tests/training/__init__.py`. The existing directory needs no package conversion. Only the intended absent production module/symbol/runner may explain this red; environmental failures or skipped assertions do not.
- P4: freeze the existing `importlib.util` path-loading convention and one local test helper for the standalone runner; add no scripts package or runtime path hack.
- S1/S2/S3: explicitly bind numerical thresholds atomically, separate runtime version/source-hash enforcement from audited-wheel provenance, and distinguish general non-strict population bounds from a deliberately reduced-overlap fixture.

No package/latest-version claims or reviewer-reported measurements were independently rerun in this plan revision. Document checks do not qualify installation, device behavior, numerical tails or memory fit.

Follow-up E1–E3 are execution notes, not new approval: record the one selected Task4/5 order and source snapshots; record source-access/distribution constraints for ranking; retain operator-supplied-review attribution. The reviewer explicitly clarified that they did not execute steps in this environment. If an independent audit is later commissioned, record its actual reviewer/run, audited source revision, commands and logs before describing it as independently executed; these conversation reports alone are not that evidence.

### Inline full audit — 2026-11-15

Scope: all17 design chapters, all9 tasks, connected dataset/trace/support/sampling/model/recipe/loop/checkpoint/reporting interfaces, real referenced paths and the central testing policy. The codegraph index was absent, so the author used focused AST/source/caller reads; no index was created. This is source/document review, not another executed training plan or independent audit.

| Confirmed gap | Smallest correction / execution evidence required |
| --- | --- |
| Coordinate delivery claimed no sorter dependency but Task3 required Task1's runner | Either Task1 or Task3 can create the one runner; Coordinate mode/guard tests run without optional sorting. Later tasks extend only working modes. |
| Task7 wrongly listed removed initialization source as a resume failure | Restore the intended success case; no source reopening, and the later of Tasks5/7 supplies its joint test. |
| Envelope validation lacked the expected weak layout and serialized-to-in-process boundary | Pass resolved layout values, validate JSON traces once, reuse pixel/patch support and retain its existing version in recipe identity. |
| Numeric-only statistics could not represent unavailable populations | Permit None minima/loss populations, retain numeric zero counts and named skip counters. |
| Sampling metadata could hit unsupported float64 devices or use inconsistent boundary precision | Convert coordinates to float32 before accelerator transfer, use matching complete-support eligibility, test device/boundary behavior. |
| Skipped-head tests covered only never-contributed state; post-update finite check was FIFO-only | Explicit set-to-none, contributed-then-skipped state/EMA tests, and common enabled-dense nonfinite stop before commits, without rollback claims. |
| Source helper lacked existing model fallback; valid registers were described as failing | Pass default_model_name, preserve actual/native/added distinctions and require positive load cases beside incompatible-layout failures. |
| Shared sorter guard risked imposing student positivity on teacher targets; some tests were scheduled after red | Keep nonnegative teacher vs positive student checks distinct, list behavior tests before implementation, leave already-compatible consumers untouched. |

The permutation direction, `/R` loss, independent weights, query49/reference16 caps, optimizer/EMA policy and optional external-package choice are unchanged. The dependency discussion did not select vendoring. No installation, long CPU stream, GPU/device benchmark, independent subagent or task implementation was performed in this audit. Full implementation regression remains a future gate; document tests do not substitute for it.

### Operator experiment input — 2026-11-15

The operator ran the four-case experiment (baseline / Coordinate / ranking / joint, 100
images, 3 epochs / 21 steps, MPS) and supplied its results. Three consequences are
accepted into the documentation, not into the acceptance gates:

- **Weight guidance becomes gradient-pressure-based.** Task 8's Step 4/5 weight advice is
  now calibrated on measured per-objective backbone gradient norms (Coordinate 1.0 ≈
  9.2%, ranking 0.5 ≈ 8.5% median pressure), replacing the loss-ratio heuristic the
  command reference previously used. That diagnostic split still needs its own
  authorized pass and is not part of normal logging.
- **The measured cost is recorded as a finding, not a pass.** 21 steps: baseline ~18 s
  versus 277/424/477 s, dominated by the dense path rather than the sorter. This argues
  for resolving the dense-path bottleneck before any production-length run; it does not
  close Task 8's production-shape cost or memory gates, and no GPU-fit claim is made.
- **The example numbers are bound to their configuration and marked non-reproducible**
  from the repository, because `runs/` is not shipped.

The experiment's scientific boundaries are preserved as reported: effects are small,
transformation dependent, one training seed, same-image synthetic geometry, descriptive
image-bootstrap intervals only, no held-out generalization or efficacy claim, and the
historical references had longer training on older sources. Nothing here upgrades Task 7
or Task 8 to accepted.

### Second audit round — 2026-11-15

Operator-supplied re-review; fixes accepted and applied in this revision. Each item
below has a named regression test that fails on the previous behavior:

| Confirmed gap | Correction applied |
| --- | --- |
| Validation after `--overwrite` had cleared the output (negative dense weight, incomplete initialization backbone) | The CLI resolves dense settings/caps and validates the saved dense pair, and fully validates the initialization backbone, before `prepare_output_dir`; the loop reuses the resolved/probed handoff. |
| DINO resume rejected a legal never-contributed ranking head | The DINO optimizer check applies the same participation rule as relational (`contributed_steps == 0` ⇒ no head moments allowed). |
| Preflight accepted head entries or state fields the restore later rejected | One shared `_validate_ranking_head_entry` proves presence, step value, moment shapes and finiteness; unknown top-level dense state fields fail in preflight; the relational and DINO paths both call it. |
| Omitted `--model-name` was treated as a conflicting request | Inheritance decides on `model_name_explicit`; a non-default source now initializes with the omitted option. |
| Replay reused the live ranking/optimizer objects, so a no-op restore passed | Cold restart with rebuilt objects, rebuilt fixtures, moment-step and head equality checks, and a measured sensitivity control. |
| Eligibility decided in float64 while the sampler runs in float32 | `_support_mask` narrows the coordinates and the contributor arithmetic to float32, matching `sample_grid_features`; float64 mapping metadata is preserved. |
| Dense metrics were in the CSV but never in the PDF | `training_curves_pretrain.pdf` gains an appended diagnostics page for enabled runs; the design and the command reference state explicitly that a CSV row does not create a panel. |

Measured here (CPU, `diffsort` 0.2.0): eligibility matches the sampler including the
`7.9999999999999964` boundary; replay 36/36 comparisons across four cold restarts, and
31/36 with a deliberately no-op `AdamW.load_state_dict`; sorter 3 seeds × 10,176 calls
(120,672 query instances, 40,224 per seed); coordinate 3 seeds × 668 calls with
per-shape rho (centres 1.0, interpolated 0.954, constructed unequal 0.031, small
magnitude 1.0); memory 15,356 geometry-derived queries against the 16,960 counting
bound, with the all-valid and 15%-overlap figures labelled as assumptions. Still open
and explicitly not accepted: accelerator memory, production-shape cost, the long
numerical stream and reference-16 fitness. No commit is made by this revision; the
code remains uncommitted at the operator's instruction until every fix is approved.

### Authorized performance follow-up — 2026-11-16

Scope: implement the approved semantics-preserving performance revision in the unified design. Reuse the existing owners; no competing trainer, dependency replacement, recipe change, new efficacy training, commit or release. Existing unresolved recovery/reporting findings are a separate workstream.

1. Add failing regressions for avoiding polygon coverage, exact geometry/RNG parity, ragged batched Coordinate/reference equivalence, preserved numerical failures and call counts.
2. In `dense_geometry.py`, compute validity without unused coverage and reuse per-view centres and masks. In `dense_objective.py`, share pair batching, normalize gathered student points once and sample teacher rows once per pair; keep real-query-only reductions and guards.
3. Add ranking equivalence tests against a rowwise reference including head/backbone gradients, reference RNG and statistics; assert one projection per participating view, one teacher sample per pair and sorting across samples rather than per row. Preserve shortage/empty-pair behavior and chunking.
4. In `neighbor_ranking.py`, cache projections per call, reuse sampled targets, concatenate the cosine query rows for each pair and split per-query losses back into the original sample means. Aggregate diagnostic reads; preserve strict standalone validation.
5. Run relevant geometry/objective/sorter, dense integration and replay tests with bounded subprocesses. Measure synchronized untraced CPU/MPS forward/backward after warmup at fixed shapes, comparing to the preserved pre-change owner functions. Check numerical tolerance, RNG/statistics and source diffs before claiming speedup.
6. Record commands/configuration, timings, source identity and limitations in this plan and the design. Keep production GPU peak memory, long numerical qualification and scientific efficacy open. No timing assertion belongs in ordinary unit tests.

Execution status: implemented inline, no commit and no independently dispatched review. All six follow-up steps above were completed; the original Tasks7/8 acceptance gates are not upgraded. Production changes are confined to the three dense owners and `vision.geometry`'s exact all-valid resize shortcut; tests add `tests/training/test_dense_performance.py`. The shortcut has red/green coverage and the affected spatial support/features/evaluation suites are included. The six initial regressions failed against the old implementation on duplicate coverage, rowwise calls and CE-only chunking, then pass after implementation. Near-tie `bmm` gradient drift failed the fixed tolerance and was reverted rather than relaxing it.

Reproduction artifacts: `runs/dense_perf_20261116/benchmark.py`, `throughput.py`, `compare_throughput.py`, `baseline_identity.json`, old-owner snapshots, per-case JSON and final `throughput_comparison.json`. These are local runtime evidence, not a competing design or production framework. The shared geometry source before/after SHA256 is also recorded in `verification.json`. The throughput payload's historical `source_sha256` field names files on disk; `_before.py` snapshots, `baseline_identity.json` and `loaded_owner_sha256` (present in final payloads) identify swapped pre-change owners. Never label on-disk new hashes as executed old-owner hashes.

Exact commands, run from the repository with the otuformer Python environment and `PYTHONDONTWRITEBYTECODE=1`:

```bash
python -B runs/dense_perf_20261116/benchmark.py --objective geometry --batch 32 --device mps
python -B runs/dense_perf_20261116/benchmark.py --objective coordinate --device mps --batch 32 --real-features --repeats 2
python -B runs/dense_perf_20261116/benchmark.py --objective ranking --device mps --batch 32 --real-features --repeats 2
python -B runs/dense_perf_20261116/benchmark.py --objective ranking --device mps --batch 8 --real-features --reference 49 --repeats 1
python -B runs/dense_perf_20261116/throughput.py --version before --case coordinate
python -B runs/dense_perf_20261116/throughput.py --version after --case coordinate --suffix _final
python -B runs/dense_perf_20261116/compare_throughput.py
```

The throughput driver refuses existing output directories; choose a new suffix for any repeat. The same `--case` choices baseline/ranking/joint cover the remaining comparisons. Before ranking used `_retry` after a55 s timeout; only the completed retry is counted. Every compared production run has3 updates and4 equal realized input batches; intervals exclude the first logged step but have only2 subsequent observations. Final means (seconds/step) are baseline0.405→0.403, Coordinate9.443→1.150, ranking16.137→1.820 and joint18.713→1.947. Same-input geometry/RNG identities are exact; gradients/updates are tolerance-equivalent, not bitwise except baseline. Maximum3-step student drift5.4e-5 and teacher drift6.0e-7 are recorded, without a scientific-equivalence claim.

Validation uses bounded `python -B -m pytest` subprocesses with `-p no:cacheprovider -p no:randomly -o addopts=` and plain terminal settings. Final disjoint completed groups: algorithms/guards/replay/docs130, affected spatial145, Coordinate trainer6, ranking trainer13, ranking resume/DINO5 and recipe/initialization67: **366 passed total**. No skip or timeout is counted as passing. The combined trainer timeout is excluded; its disjoint completed reruns are the evidence. No full regression was run. The final exact commands, source/test hashes and benchmark provenance are in `runs/dense_perf_20261116/verification.json`.

Long-sequence numerical qualification, CUDA/production peak memory, reference16 fit, standalone outer-trainer recovery and held-out/multiseed efficacy remain open. Validity-on-worker handoff, custom kernels/compilation and sorter recomputation are deferred pending evidence that their added ownership/complexity is warranted.

### Authorized repair and execution audit — 2026-11-17

The operator authorized the three remaining repairs and related execution fixes. This is
inline execution, with no independent dispatch, commit, installation or research run.

1. **Recovery owner:** `_verify_restorable_rng_state` returns the validated copy. The
   trainer retains it before output creation for dense/DINO recovery and restores that
   copy once at the resumed epoch. FIFO and ordinary legacy relational all-off rules stay
   with their existing owners. Three real-loop regressions failed before this fix and
   pass for relational+dense, DINO+dense and DINO without dense.
2. **Replay owner:** snapshot both complete heads as detached CPU tensor dictionaries;
   compare every layer/bias in both query and reference trajectories, including the tail.
   Four final-layer scaling corruption cases failed before the fix and now reject. They
   prevent loss invariance after L2 from concealing a bad restore.
3. **Ranking owner:** reuse the existing teacher projection and its checked norm grid;
   aggregate only positive-weight support for real contributing queries, per compute.
   Tests cover unused rows/patches, exact-centre zero weights, interpolation, reversed
   teacher order, repeated cached teacher views and empty contributions. Existing
   performance-equivalence tests retain their gradient/RNG/call-count tolerances.
4. **Related replay gap:** fixed Q49/R16 uses mixed A5+B48/49/50/48 and selected
   (5,10)/(5,11)/(5,12)/(5,10), comparing cold restores and exact draw consumption. The
   runner's controlled candidate subsets are declared fixtures, not a production
   geometry-distribution claim. Per-row support masks replace the fixture's row-0 reuse.
5. **Related reporting gap:** cache global/local attempted/contributing/skipped
   sample/pair counts and eligible/selected points independently for both objectives.
   Export them in enabled CSV (124 columns) and PDF denominator panels. Freeze the
   actual previous 92-column header alongside 83/86/88; migrate without replacing old
   values and leave new/disabled cells empty. The all-off schema does not grow.
6. **Bounded execution matrix:** the real CPU one-update/save/one-resumed-update matrix
   now covers all 32 SSL/patch/dense combinations with local synthetic data and offline
   backbone construction. It checks finite encoders, progress and dense state presence;
   PDF rendering remains separately tested. A broad slice timeout is excluded; only
   completed disjoint slices count. This is lifecycle coverage, not long-run or exact
   full-trajectory qualification.

Verification uses the project environment, no bytecode/pytest cache, bounded path-scoped
subprocesses and the existing assertions. Red/green checks are described above; final
related suites and source/log integrity are reported with the completed repair. The
historical 366-test performance result above is not a current-source acceptance claim.
The source hashes in its immutable runtime artifacts identify that earlier candidate;
these diagnostic/replay changes require their own future performance evidence before
reusing any throughput or memory conclusion. No full regression, accelerator peak,
10,000-iteration qualification or new scientific experiment is claimed here.

### Operator-supplied follow-up review — 2026-11-17

All nine supplied findings were checked against the current source. Changes stay inside
the authorized related repair scope; no commit or independent dispatch was performed.

| Finding | Disposition and current owner |
| --- | --- |
| Resolved dense output absent | Fixed in `pretrain_loop`: one enabled-only JSON print per invocation with requested/effective/applicability/source; new/resume regression observed red then green. |
| Whole-field ranking projection norm | Fixed in `neighbor_ranking`: selected student centres and real positive-weight teacher support, using the existing per-compute projection cache; error-only row attribution. |
| Live trace transport bypassed codec | Fixed in `dataset` and unpack: `to_json` per item, ragged transpose, `from_json` once per ordinary record; existing codec validates operation/matrix consistency. |
| Weak slots wider than contract | Collator/unpack canonicalize to `None`; only unpack accepts legacy per-row weak `None`, pairing requires canonical slots. |
| Individual raw contributor norm absent | Fixed in `dense_objective` within the existing contributor/rho pass; rejects a zero contributor despite a healthy mixture and excludes exact-zero weights. |
| Three named tests and obsolete file names | Added exact startup RNG, contributed-then-skipped and register-prefix real-loop tests; resume/reporting owners are `test_dense_pretrain.py`. |
| Query-cap documentation applicability | Fixed both command languages: either dense weight may enable it. |
| Stale design status and empty checklist | Status now names uncommitted implementation/partial acceptance; 7 narrow implementation steps checked, compound qualification/commit gates remain open. |
| Five existing Rich assertion failures | Reproduced, then fixed ANSI SGR removal in two test assertions; all 11 associated parameter cases pass with the original colored environment. No CLI cleanup behavior changed. |

Named execution tests now present in `tests/training/test_dense_pretrain.py`:
`test_dense_resolved_config_is_printed_once_with_sources`,
`test_dense_startup_probe_preserves_generators`,
`test_ranking_contributed_then_skipped_step`, and
`test_prefix_contract_global_and_local_registers`. They cover real loop boundaries,
not new training features. Existing-loop coverage that was already correct passed on
first execution; no historical red was invented for those test additions.

The current 32-case lifecycle matrix was rerun after production JSON transport and the
new participating guards, using four completed 8-case slices. Source/test hashes and
final bounded counts are recorded in the execution ledger. The author did not run the
full repository gate, long streams, accelerator memory/cost, multi-worker recovery or
research evaluation. Earlier 590/366-test records remain attributed to their candidates.

**Handoff:** Review this plan together with the unified design. Native inline execution is recommended because batch, recipe, optimizer and recovery interfaces are tightly coupled; no independent reviewer/subagent is automatically authorized. If the operator prefers delegated execution or independent review, obtain that explicit selection and follow the relevant skill. Do not execute, install, commit or launch qualification just because the plan was written.
