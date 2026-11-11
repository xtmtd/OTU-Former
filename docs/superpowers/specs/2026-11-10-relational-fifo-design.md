# Relational FIFO Candidate Extension — Design

Date: 2026-11-10
Revision: 2026-11-11 — source-verified audit corrections; G1/G2 and the complete revised design approved on 2026-11-11.
Status: Approved for execution-plan authoring on 2026-11-11 (§12, Design decision). G1/G2 were separately approved (§12, Gate decision). The execution plan requires separate review and approval; implementation, new research training, commit, push, version change, tag, and release remain unauthorized.
Baseline: branch `OTU-Former-dense`, HEAD `3d69d1812a1b5d6941c49bf070f1037c1304d83d`, package version 0.12.0. The working tree was clean before this document was created. Source inspection and the separately authorized self-relation spike are evidence, not implementation of FIFO.

## 1. Intent and constraints

Improve the relational SSL research capability by extending the current batch-relative reference set with a finite history of teacher observations. Keep the useful aligned-image anchor, the existing student/teacher architecture, and the current batch-only default. The goal is to test better reference coverage and eventual morphology benefits, not to repair a demonstrated global-training failure or claim a faithful ReSSL replication.

The first implementation is FIFO plus lightweight diagnostics. It does not implement confidence weighting, AdaSim, hard nearest-neighbor targets, automatic capacity, age-based eviction, a new temperature schedule, a predictor, a second auxiliary loss, or a patch/token queue. Future dense morphology is a motivation, not an acceptance claim or a reason to add a general graph framework now.

The [master design](2026-03-29-otuformer-design.md), especially §§7–8, governs ownership, configuration, documentation and verification. The existing [relational targets design](2026-11-03-relational-pretrain-targets-design.md) continues to own views, pairing, center population and temperature. This feature extends its no-queue boundary only when explicitly enabled; do not rewrite the historical decision. The [training modularity design](2026-11-07-training-modularity-cli-testing-design.md) and [testing policy](2026-11-07-development-testing-policy.md) remain authoritative.

### Agreed decisions

| Decision | Agreed scope |
| --- | --- |
| Objective | Approach A: one softmax over current batch plus eligible history; no auxiliary history loss. |
| Aligned-image anchor | Keep all current batch candidates, including the current aligned-image candidate. Do not fix its probability mass. |
| First feature scope | FIFO and diagnostics only; confidence weighting and adaptive neighbors are deferred. |
| Stored observation | One record per image identity, with both teacher slots A/B, one observation step and common FIFO order. |
| Candidate views | A relation using teacher A reads current A plus historical A; B reads current B plus historical B. |
| Branches | Extend both global and local relational terms. Local views are student queries, not stored candidates. Patch objectives stay separate. |
| Duplicates | Exclude every current-batch image's historical record during loss construction. After a successful step, replace/reinsert current images at the newest end. |
| Lifecycle | Start empty; consume the valid partial queue immediately; retain across epochs; evict by capacity alone. |
| Default | Disabled at capacity zero; a positive explicit capacity enables it. No `auto`, dataset-derived default, or dynamic resizing. |
| Recovery | Inherit saved capacity on omission; reject conflicts. Old missing queue metadata means disabled. No reset-on-resume. |
| Diagnostics | At existing log intervals: coverage/age, teacher target structure and student allocation, separately by branch. Heavy gradient/drift analysis remains a research script. |

### Author proposals and separate decision gates

These settle implementation-relevant gaps. Decisions **3 (G1: input/data identity)** and **6 (G2: recovery lock)** received separate explicit user approval on **2026-11-11**, recorded in §12 (Gate decision). Those decisions are settled; the complete revised design and remaining proposals were also approved for planning on 2026-11-11 (§11; §12, Design decision). Reopening either gate requires revising its dependent sections before implementation or an execution-ready plan.

1. Cache detached float32 projector outputs before center subtraction; apply the current pre-step center to both current and historical features. History never updates the center.
2. Name the option `--relational-queue-size`; declare omission as `None`, resolve fresh omission to zero, and allow only nonnegative integers. Capacity is a count of image records, not vectors or candidate columns.
3. **G1 — approved input/data-identity contract (2026-11-11):** queue-enabled runs require unique in-root selected-reference keys, whole-selected-dataset SHA-256 preparation, and an ordered key/content fingerprint locked on resume (§5). Reject two selected rows mapping to the same key, not distinct valid internal aliases of the same physical file. This rejects previously valid outside-root/duplicate-key inputs and adds hashing cost. It is not a migration of the queue-off legacy data contract.
4. Store a separate enabled-only `config.relational_queue_config` and top-level queue state; preserve the existing closed version-1 relational target bundle. The disabled encoding remains absence of queue-specific persisted bundles.
5. First implementation is the existing single-process training path. No distributed queue, new CLI scope selector, storage-device option, weighting option, or age cutoff.
6. **G2 — approved recovery lock (2026-11-11):** queue-enabled resume requires unchanged batch size, steps-per-epoch and a complete restorable RNG/progress/schedule state (§7), in addition to G1's data lock. Batch-size conflict rejection and disallowing legacy missing-RNG fallback are new restrictions; they do not establish exact worker/accelerator replay beyond tested boundaries.

## 2. Evidence and alternatives

### Self-relation spike

The authorized runtime experiment is at `runs/self_relation_probe_20261110/`. Its original baseline is `runs/pretrain/logs/pretrain.log`, dated 2026-11-07. Six 50-epoch runs used the Epidorcus 230-image, two-label dataset with paired seeds 42, 43 and 44, batch size 32 and 350 updates per run. Teacher views, cross pairing, weak-source center, six local views, consistency patch objective and temperature schedule were held fixed. The removal arm excluded the aligned-image candidate from both student and teacher distributions in both global and local losses, then renormalized. It is not a queue experiment.

The seed-42 keep arm reproduced the original final teacher/student tensors and both metric CSVs exactly. Original log/checkpoint hashes were unchanged. Successful training used MPS with explicitly enabled CPU fallback for an unsupported bicubic backward. Mathematical diagnostics were checked against autograd.

Three-seed means:

| Measurement | Result |
| --- | --- |
| Teacher aligned-image probability, epochs 2–5 | 0.9308 |
| Teacher aligned-image probability, epochs 36–50 | 0.6159 |
| Conditional-nonself component norm ratio, epochs 2–5 / 36–50 | 0.0330 / 0.5348 |
| Final raw-CLS mAP, keep / remove | 0.8193 / 0.7870 |
| Final KMeans NMI, keep / remove | 0.9360 / 0.0974 |
| Centered raw-CLS effective rank, keep / remove | 18.49 / 6.48 |

The component norm ratio is measured at the normalized student output, not the backbone parameters; cancellation prevents interpreting it as an additive gradient-energy share. Existing NMI/ARI use KMeans with known class count. An oracle-two-cluster UPGMA diagnostic was equally poor in both arms and did not distinguish them. Metrics reuse training images. No held-out, larger-data or dense-morphology benefit was demonstrated.

For teacher self mass `a`, student self mass `b`, and conditional nonself distributions `r_T`, `r_S`, the exact identity is:

`CE(p_T, p_S) = BCE(a, b) + (1 - a) * CE(r_T, r_S)`.

The experiment supports retaining the anchor as a baseline. It does not prove an optimal self mass, that candidate expansion helps, or that every queue-only objective is worse. Capacity and temperature counterfactuals on fixed features showed that mass changes can occur without any improvement in learning. Do not promote FIFO to the default on that basis.

Runtime artifacts are ignored by git. The compact evidence above is durable design context; future verification records must include exact commands, source/environment identity and artifact locations rather than depend on the runtime folder being shipped.

### Literature boundaries

- [Weak Augmentation Guided Relational Self-supervised Learning, v3](https://arxiv.org/html/2203.08717v3) motivates relation matching over a memory buffer, reports queue-size/staleness tradeoffs and studies predictor/warm-up variants. It is not a ReSSL-loss-plus-DINO-loss recipe. Its reported teacher-temperature-warmup ablation is not evidence that this project's existing schedule is optimal or wrong.
- The public [ReSSL multi-crop implementation](https://github.com/KyleZheng1997/ReSSL/blob/main/network/ressl_multi.py) compares multiple student crops and one teacher key against one shared queue, then enqueues the current teacher features after computing logits. It has no local-feature queue. The [augmentation code](https://github.com/KyleZheng1997/ReSSL/blob/main/data/augmentation.py) and [training entry](https://github.com/KyleZheng1997/ReSSL/blob/main/ressl_multi.py) define that released recipe. The repository identifies the NeurIPS 2021 release; do not label it a complete implementation of every v3 extension.
- [DINO's official implementation](https://github.com/facebookresearch/dino/blob/main/main_dino.py) uses global-only teacher views and global/local student views with same-view exclusions, but its output distributions are not relations over a sample FIFO. It supports the cross-scale supervision comparison, not equivalence of objectives.
- [AdaSim](https://openaccess.thecvf.com/content/ICCV2023/html/Lebailly_Adaptive_Similarity_Bootstrapping_for_Self-Distillation_Based_Representation_Learning_ICCV_2023_paper.html) motivates caution with nearest-neighbor self-distillation. Full-data indexed memory and adaptive positive sampling are not implemented by this proposal.

Our paired slots, current-batch anchor, unique-image refresh policy and pre-step recentering are explicit project adaptations. No cited work establishes their OTU or dense-morphology gains.

### Approaches

1. **Batch plus history in one candidate distribution — selected.** Directly extends the current relation formula. History changes the softmax denominator and hence current/self masses; that interaction is measured, not hidden.
2. **Original batch loss plus weighted historical auxiliary loss — deferred.** Preserves the original batch term more explicitly but adds an independent weight and potentially conflicting/redundant targets. Reconsider only if the selected approach exposes a specific limitation.
3. **Queue-only ReSSL-style references — not selected.** Removes the guaranteed current aligned-image column and changes the reference construction more substantially. The spike does not rule it out universally.

## 3. Exact objective

Let a batch contain `B` uniquely identified images. For teacher slot `j in {A, B}`, let `z_j` be the current detached projector outputs **before center subtraction**, and `c_g` the teacher center at the start of optimization step `g`. The projector already normalizes its output; "pre-center" does not mean raw CLS or unnormalized head activations.

Let `Q_g` be the stored oldest-to-newest records before this step, and `H_g` that sequence with **all** current-batch image keys removed. Preserve `H_g` order. Its size is `E`, which can be smaller than capacity `K`.

For each slot:

```
t_j = normalize(z_j - c_g)
h_j = normalize(stored_precenter_features[H_g, j] - c_g)
C_j = concatenate([t_j, h_j], dim=0)     # [B + E, D]
```

Current batch order stays first and unchanged. Both sides use the same detached candidate matrix `C_j` and column order, with their respective temperatures. The student query `s_v` remains differentiable; teacher queries and candidates are detached. For a normalized student query matrix `s_v` aligned row-wise to the batch:

```
student_logits = s_v @ C_j.T / tau_student
teacher_logits = t_j @ C_j.T / tau_teacher
teacher_logits -= teacher_logits.mean(dim=1, keepdim=True)
p_teacher = softmax(teacher_logits, dim=1).detach()
log_p_student = log_softmax(student_logits, dim=1)
L(v, j) = mean_rows(-sum_columns(p_teacher * log_p_student))
```

Keep the existing teacher row-mean subtraction even though it is softmax-invariant; no unrelated numerical cleanup is part of FIFO. The aligned-image column for row `i` remains current column `i`. History cannot add a second column with that image key. There are `B + E` columns, not `B + 2E`.

- Global loss averages the existing `cross`, `full`, or inherited legacy `same` pairs. FIFO does not change pair membership or permit changing pairing on resume.
- Every existing local student view compares against `C_A` and `C_B`, using the teacher global row belonging to the same image. Keep pair averaging and `lambda_local` unchanged.
- Teacher candidates, queries and center are detached. No gradient reaches cached history or the teacher.
- Patch `none`/`consistency`/`masked-feature`/`ibot`, mask generation, KoLeo, optimizer groups, clipping, EMA schedule and embedding architecture remain unchanged.
- Weak and ordinary relational teacher-view modes use their already resolved source. FIFO is not a new transform selector.
- Capacity zero takes the original batch-only loss/data path, without building a queue or normalizing an empty history. This must preserve loss values, gradients and RNG behavior.
- Positive capacity with no eligible history means `E = 0`; the formula must equal the original batch-only calculation. This is a parity requirement, not a requirement to add a separate loss branch. Enabled queue state remains distinct from capacity zero.
- Temperature admissibility checks must cover the expanded reduction length `B + E` (a conservative preflight bound may use `B + min(K, N - B)` for `N` unique selected images), not only the old batch length. Default temperatures and schedules do not change.

Center updates preserve the existing formula, input population and timing: only current selected teacher features update the center; historical features never enter its mean. Both current and historical targets in this step use `c_g`, not the just-updated center. Recentring history removes the center-time mismatch, not encoder staleness.

## 4. Queue lifecycle and state

One narrow method-specific state owner, provisionally `training/relational_queue.py`, owns ordered records, selection, refresh, validation and serialization. Do not introduce a universal memory-bank abstraction. Exact private class/function names and a list versus ordered-map storage are builder decisions; observable ordering and snapshot semantics are not.

Each record contains:

- root-relative image key;
- two detached pre-center float32 vectors of dimension `D`, in fixed A/B order;
- the successful optimization step at which those features were observed.

The two slots are independently sampled augmentation slots, not fixed anatomical poses. Do not average them, expose both as separate candidate images, or store local features.

### One step

1. At completed-step count `g`, freeze the candidate snapshot from `Q_g`, excluding current keys. Reading candidates does not refresh age or order.
2. Compute current teacher outputs before EMA and form targets using the pre-step center `c_g`; retain detached, non-aliased pre-center A/B vectors for eventual insertion. **Keep the relational center update at its current position, during teacher-target preparation and before loss construction/finite checks.** Compute all loss terms with the unchanged candidate snapshot; do not move that center update into the success-only phase.
3. Complete the finite-loss check, backward, finite-gradient check, clipping and optimizer update. Keep the teacher-weight EMA and, where applicable, iBOT head EMA/patch-center update after `optimizer.step()` as today.
4. Only after the step succeeds, remove any old records for each current image and append the newly observed pair in deterministic current batch order. Their observation step is `g`. If `K < B`, the last `K` records in that order survive.
5. Evict oldest records until the size is at most `K`. Advance/persist the completed-step count consistently with existing `global_step = g + 1`.

There is no extra teacher forward after EMA. Enqueuing a pre-update observation after success is an acceptance boundary, not a claim that the feature was generated by the post-update teacher.

A non-finite loss or gradient raises `ValueError` and aborts the run; there is no skip-and-continue path. The abort occurs before queue insertion/refresh/eviction, but the relational center may already have changed in memory, matching current behavior. No checkpoint is written for the aborted step, and earlier checkpoints remain intact. Queue state is saved only with the next scheduled **epoch-boundary** checkpoint (`SSL_epoch_*.pth` and its `SSL_latest.pth` copy), not per step. Tests catch the existing exception and inspect queue nonmutation/no new epoch checkpoint; they must not introduce failure recovery or per-step saving. Existing training writes are not transactional and no center/model rollback is promised.

Retain the queue across epochs and after epoch evaluation. Do not populate it from startup probes, evaluation, validation data, failed steps or resumed warmup forwards. It starts empty only for a new enabled run. No fill threshold, random placeholder vectors, age cutoff, confidence filter or probability floor is added.

For candidate age before step `g`, use `g - observed_step`; the previous successful step's observation has age one. Age is not an epoch count or elapsed wall time.

Use one logical capacity `K` even when it exceeds the selected unique-image count. Report requested capacity separately from actual occupancy; do not silently clamp the saved recipe. Storage allocation may grow with actual occupancy instead of allocating enormous unused buffers. Candidate compute is approximately proportional to the number of loss pairs times `B * (B + E) * D`; slot storage is `2 * M * D * 4` bytes for float32 features at occupancy `M`, excluding metadata. No automatic device placement or OOM recovery policy is promised. Device migration follows the model's selected device; checkpoint tensors serialize detached on CPU.

## 5. Dataset identity and data lock — G1 approved 2026-11-11

### Existing boundary

`MultiCropDataset` currently returns views only and resolves paths through `resolve_dataset_image_paths()`. That legacy resolver can accept absolute images outside the declared root and recursive basename fallback. The shared data reader is not currently the pretrain dataset path. Do not claim otherwise or replace it wholesale in this feature.

### Enabled-only identity preparation

Under **G1, explicitly approved on 2026-11-11**, resolve the **same effective ordered image paths** the training dataset will open, once, and pass that selection forward. Reuse `data.paths`'s `resolve_reference()`/`normalize_key()` identity and containment semantics and `data.provenance` snapshots/digests. Derive keys from the already-selected paths, preserving the actual legacy basename match rather than resolving the original reference again. `a/../one.png` and `one.png` must have the same normalized key when they open the same file. Resolve symlinks as written before lexical folding so a `link/..` path cannot acquire a key for a different target; blindly combining `relative_key()` with `normpath` is not sufficient for that case. Distinct valid internal symlink aliases remain distinct keys as agreed; absolute root location is not identity.

For a queue-enabled run, preflight must reject before output mutation:

- an image outside the declared root, including escaping symlinks;
- duplicate selected image keys, which otherwise make an aligned-image column ambiguous;
- unreadable images or missing files;
- a selected dataset with no complete batch under existing `drop_last=True` semantics.

Duplicate selected rows are not silently dropped or reweighted. The error explains that queue-enabled input requires one row per image identity and that queue-off legacy behavior remains available. This restriction is approved only for queue-enabled runs, not as a global data contract. Keys are not labels, specimen IDs or content hashes. Uniqueness is checked across the selected reference rows: reject two rows mapping to the same normalized key, not two distinct valid internal aliases of the same physical file. Byte-identical files at different valid keys remain different identities; near duplicates and physical aliases are not inferred as one specimen. If both aliases occur in a batch, the same underlying pixels may occupy two current-candidate columns; historical records for both current keys are excluded for that step. The aligned-image column remains well-defined by row/key, but candidate diversity can be lower; both identities still follow normal post-success queue refresh. Existing shared-reader warnings are not converted into a new semantic deduplication algorithm.

During preparation, build an ordered data manifest of `(image_key, size_bytes, sha256)` for all selected images and its canonical digest. The full manifest is transient; checkpoints retain only its digest and image-count/byte-count summaries (§7), not an `O(N)` duplicate in every epoch file. No manifest sidecar is required for resume. Exclude labels, absolute root paths and unrelated CSV metadata from this training-data identity. Preserve row order in the digest because shuffle indices address that order. A root relocation with unchanged keys/bytes/order is allowed; rename, reorder, addition/removal or changed image bytes is rejected on queue-enabled resume. Do not fall back to a changed dataset plus an empty queue.

Hashing is an explicit enabled-only startup cost: log a phase start before reading files and report image count/bytes/time. Add a **queue preparation stage before every model-constructing preflight**, including `preflight_pretrain_resume()` / `_resume_register_plan()` on CLI resume as well as `preflight_pretrain_run()`, in both CLI and direct-core entry paths: resolve settings and queue applicability/capacity through the model-free identity check in §6 using the already-loaded checkpoint, validate/prepare the selection once, then run the existing architecture probe and finalize the recipe using that prepared object. The current `_resolve_pretrain_recipe()` runs after the probe; placing hashing only there would violate this ordering. Check input/identity conflicts and finish the fingerprint before any register-layout or architecture probe or output setup; do not hash again in recipe finalization or the trainer. Preserve the existing relative probe order for queue-off runs. Completing the fingerprint before probing is deliberate: it keeps enabled data validation together and ahead of model construction, at the cost of hashing even when a later model-compatibility check fails. Reuse existing snapshot helpers without retaining image payload bytes. Detailed handoff wiring belongs in the execution plan. The caller must not edit images while a run is active; this design's fingerprint detects fresh/resume boundary changes, not arbitrary concurrent mutation after preflight. Existing path containment checks must not be weakened.

The enabled loader transports image keys with the views through native collation, in matching row order; workers do not own or mutate the queue. Preserve crop count/order, augmentation RNG draws, shuffle order, worker seed setup and drop-last behavior. Startup and evaluation consumers must explicitly understand the enabled batch envelope. With capacity zero, retain the existing dataset return type and data path. No labels enter the relation target.

## 6. CLI, resolution and method applicability

Proposed public interface: `--relational-queue-size INTEGER`, minimum zero, in the existing appropriate pretrain help panel. There is no boolean enable flag or second queue option. No automatic strategy string is accepted.

| Invocation | Effective behavior |
| --- | --- |
| Fresh relational, omitted | Capacity zero. |
| Fresh relational, explicit zero | Capacity zero; preserve explicit provenance. |
| Fresh relational, positive integer | Enabled capacity exactly as requested. |
| Resume, omitted | Inherit the saved positive capacity, or zero when the queue bundle is absent. |
| Resume, explicit matching value | Accept; provenance remains explicit. |
| Resume, explicit conflicting value | Reject before outputs/probes that mutate state; explain that recipe changes need a new run. |
| DINO, omitted | Inactive; no queue state, allocations or queue diagnostics. |
| DINO, any explicit queue value (including zero), or queue-bearing resume state | Reject as relational-only, following the existing explicit-option applicability convention. |

Reject negatives, booleans masquerading as integers, fractional numbers, invalid strings, non-finite values and conflicting checkpoint representations at the CLI/schema/core trust boundaries as appropriate. A direct core caller missing the new request attribute follows omission semantics: disabled on a fresh/historical-off run, inherited on an enabled resume, never silently disabled when the checkpoint is enabled. The public core entry performs the same preparation stage when no CLI handoff exists; an enabled finalized recipe without matching prepared selection is an error, not a fallback. Requested `None`, resolved zero and inactive DINO are distinct; logs/schema must not conflate them.

Resolve queue capacity/identity once **before any model-constructing resume/register-layout or architecture preflight**, and carry it into the subsequent recipe handoff with the same loaded resume payload (§5). For this early model-free check, reuse `_saved_ssl_state()` (or the same `resolve_checkpoint_state_dict()` → `resolve_checkpoint_ssl_identity()` steps) on the already-loaded checkpoint, as the later SSL resolution does; do not independently read `config['ssl_method']`. Preserve existing requested/default method and saved-method/explicit-conflict semantics, and read saved capacity from `config.relational_queue_config` subject to §7 validation. Keep the complete `resolve_pretrain_ssl_config()` at its existing later position: its relational optimizer preflight constructs probe encoders, so moving it forward would violate input-before-probe ordering. Do not reorder the architecture probe merely to hide duplicate resolution. Extend the resolved recipe with prepared state as needed; do not attach persistent mutable queues to CLI globals or re-resolve files inside the trainer. Keep queue-specific objects out of saved `args` via the existing transient/unsaved mechanism. No new exported CLI schema shape/version is required; derive the option and card from runtime declarations, add resume provenance and applicability in their existing owners.

## 7. Checkpoints, recovery and consumers — author proposal

### Enabled encoding

Persist a separate `config.relational_queue_config = {version: 1, capacity: K}` for `K > 0`; version 1 fixes the semantics in §§3–5 rather than exposing every invariant as a configurable key. Keep the exact existing `config.relational_target_config` version-1 key set unchanged. Store effective request provenance in `args` using the existing requested/effective convention, without presenting an omitted inherited value as a fresh default.

Top-level `relational_queue` version 1 stores a logical oldest-to-newest sequence. These listed keys are the complete version-1 field set; reject unexpected keys rather than guessing their semantics:

- `version`: integer 1;
- `keys`: unique ordered image-key strings, length `M <= K`;
- `features`: finite detached float32 CPU tensor of shape `[M, 2, D]`;
- `observed_steps`: integer sequence of length `M`, each in `[0, completed_steps)`;
- `completed_steps`: equal to saved `iteration`;
- `data_fingerprint`: the canonical ordered key/size/hash manifest digest from §5;
- `data_image_count`: positive integer number of selected images;
- `data_size_bytes`: nonnegative integer total selected-image bytes.

Do not persist the full manifest per checkpoint. Recovery recomputes the digest/counts from the invocation's prepared selection, which also supplies the allowed key set; epoch checkpoints need no external sidecar. This keeps data-identity metadata constant-sized apart from the queue's own occupied-key list.

Logical serialization need not persist a physical ring pointer: reconstruction must preserve precisely the same order, eligibility, ages and next eviction. An implementation that does use a pointer must serialize/validate enough state to produce this same logical sequence; no pointer-based alternative checkpoint shape is introduced without revising this draft.

Preflight checks both bundles are present together, versions/key sets/types, positive capacity, all tensor shapes/dtypes/finiteness, unique stored keys belonging to the current prepared selection, valid timestamp order/ranges, and fingerprint/count equality with that selection. For chronological records, observation steps are nondecreasing in FIFO order; equal steps preserve batch order. Reject config/args/state conflicts, an enabled record with a missing payload, an invalid partial queue, state with no enabled config, and unsupported versions. Do not discard malformed rows, resize, truncate or silently repair them. Empty state is valid at completed step zero; after positive completed steps, expected occupancy is nonzero for a nonempty complete-batch dataset. At enabled recovery, with validated effective batch size B and completed count g > 0, exactly the newest `min(K, B)` records must have observation step `g - 1`. Reject nonempty truncated or missing-latest-commit state rather than silently losing recent history; this check belongs in the batch-aware recovery layer and needs no new stored field. At an epoch checkpoint after at least one completed epoch, also require occupancy `M >= min(K, (N // B) * B)` for N selected unique keys: the current no-replacement shuffle and drop-last loader consumes that many distinct keys per complete epoch. This lower bound complements the latest-batch tail; it does not reconstruct shuffle history or require every selected key when N is not divisible by B.

Under **G2, explicitly approved on 2026-11-11**, positive-capacity resume must retain `args.batch_size` and **the already-existing `schedule.steps_per_epoch`**, require a valid restorable existing-format `rng_state`, and use the existing progress/schedule recovery path. With `drop_last=True`, recompute steps-per-epoch as `prepared_image_count // batch_size` and compare it to the saved schedule; no new step-count field is needed. Omitted batch size inherits `args.batch_size` for enabled resume and an explicit conflict is rejected; queue-off behavior is unchanged. This enabled-only inheritance is deliberate; queue-off retains its existing DataLoader-length rejection rather than gaining batch-size inheritance. Because the read-only schema/card cannot inspect the resume file or decide whether its queue is enabled, report omitted `batch_size` as `omitted-conditional` on a resume with an explanation of that condition, not blanket inheritance. Do not accept the batch-only legacy best-effort missing-RNG fallback for an enabled queue. RNG completeness is target-aware: selected CUDA resume requires a nonempty CUDA state sequence matching the current visible CUDA-device count and successful restoration of every stream; missing/None/empty, malformed or count-mismatched CUDA state is rejected before outputs, not newly seeded. A CPU-generated checkpoint without CUDA state remains valid for CPU/non-CUDA recovery, but cannot continue an enabled run on CUDA under this strict policy. Non-CUDA recovery restores its supported CPU streams; any saved CUDA payload is structurally checked and explicitly reported as not applied. This is not CUDA-continuation or cross-device replay evidence. Existing raw `args.device` (including `auto`) is not proof of the source backend; no new backend-provenance field is introduced. Config/args capacity checks accept omitted request markers only when the effective enabled bundle/state are complete; a recorded numeric request must equal the effective saved capacity. Before any output/log/CSV mutation, validate student and teacher independently against the actual rebuilt training encoder: require complete state-dict key sets, exact tensor shapes and finite floating parameters/buffers, including backbone, projector and encoder center. Include only the training aliases actually registered by that encoder; backbone-only or projector-layout checks are not a complete restore check, even for zero-register models. Require the top-level relational `center` tensor to have exact shape `[1, D]` and finite values; broadcasting is not accepted. Use strict encoder loading for enabled recovery, with no missing-student fallback from teacher/model_state_dict; queue-off retains its historical loading behavior. Also validate applicable independent patch-objective state and usable per-parameter AdamW state, not merely optimizer group layout. Require the saved patch tensors/centers to match the applicable strict restore and be finite; require AdamW state for parameters that should have participated, with valid steps and correctly shaped finite moment tensors, nonnegative second moments (`exp_avg_sq`, and `max_exp_avg_sq` when AMSGrad applies), while allowing only explicitly justified inactive/frozen cases. Save and restore the entire queue alongside optimizer, teacher/student, centers and RNG at the supported epoch checkpoint boundary. Do not populate it during reconstruction or overwrite the resumed teacher with student weights. Exact replay tests use the existing controlled CPU/worker-zero setup; multiprocessing and accelerators retain their existing reproducibility limits.

### Disabled and historical encoding

When effective capacity is zero, omit both persisted queue bundles. Historical missing metadata means disabled permanently for that run, like a saved zero; explicit positive capacity cannot enable history on its resume. Re-saving a disabled run does not fabricate queue history or add an enabled version marker. The new requested option may appear in `args`; it is not evidence that a queue was active. Reject positive `args` claims with missing bundles as contradictory.

Old executables are not promised to understand queue-enabled training state: they may ignore unrecognized metadata rather than reject it. Help/docs must say that enabled checkpoints require a queue-capable version for pretrain resume. This proposal does not change weight/head identity to force an older binary to reject, because that would also alter embedding consumers.

### Downstream readers

Architecture and tensor layouts remain unchanged. `extract`, `export`, `cam`, fine-tune initialization and provenance may ignore training-only queue tensors but must still resolve the same encoder/head identity. Their shared identity validator must reject queue metadata attached to an active DINO identity while accepting valid relational queue checkpoints. No finetune queue state is copied into a new finetune training state. Do not advertise old-code training-resume compatibility based on successful read-only loading.

## 8. Diagnostics and reporting

Queue-enabled runs record lightweight measurements at `log_every_n_steps`, from the same pre-step candidate snapshot used by the loss. No extra model forward/backward or full feature dump is added. Never describe post-enqueue occupancy as the size of the candidate pool that produced this step's loss.

| Family | Required measurements |
| --- | --- |
| Coverage | Requested capacity; stored occupancy before the step; eligible history count after current-key exclusion; total candidate count. |
| Age | Mean/max eligible-history age in successful optimization steps. No history means unavailable ages, not invented zero age. |
| Updates | Actual refresh and capacity-eviction counts for the successful step, explicitly post-step values. A new insertion is not a refresh. |
| Teacher distribution | Aligned-image, other-current-batch and history probability mass; full target entropy; conditional-nonself effective candidate count. |
| Student distribution | The same three probability masses for global/local queries; retain existing global/local losses. |

Probability groups partition the columns exactly and sum to one per query. For teacher row `p_T` with aligned-image index `i`, conditional-nonself effective candidates means the perplexity `exp(-sum_j r_j * log(r_j))`, where `r_j = p_T[j] / (1 - p_T[i])` over `j != i`, with `0 * log(0) = 0`; it is not the number of nonzero columns. Derive this stably from nonself logits when total nonself mass would round to zero. Average the per-row perplexities (not the exponent of average entropy) and other statistics over contributing pairs consistently with the loss, separately for global and local branches. Teacher statistics may coincide across branches; do not compute them redundantly merely to fill columns. No locals means local fields are unavailable. Empty history has exactly zero history mass. If there are no nonself candidates, the conditional-nonself statistic is unavailable; do not return an undefined entropy. Use stable log probabilities and finite checks.

Final column names are a builder naming decision, subject to the existing CSV conventions and clear `_global`/`_local` distinction. Queue-off/DINO rows do not trigger new relation matrices for diagnostics; added columns are empty/inapplicable rather than observed zeros. Add a resolved `Relational queue config:` record alongside `Relational target config:`, including applicability, effective capacity and, when G2 applies, effective batch size; distinguish requested/defaulted/inherited sources and inactive values without overwriting the raw `Parameters:` request record. The trainer emits this resolved record once; CLI preflight does not print it. It is display-only and never parsed as a training input. This new resolved record together with existing recipe logs covers requested/effective/inactive configuration; raw requests alone do not describe an inherited run.

`InstantMetricsLogger` currently migrates only explicitly recognized headers; it is not a generic additive-column migrator. When adding FIFO columns, register the **pre-FIFO current 43-column header** as a recognized predecessor alongside the existing legacy/v0.7.0 headers, preserve historical values and add empty FIFO cells. Unknown or duplicate headers must still be rejected unchanged, not rewritten or silently discarded. The epoch metric schema is not changed by FIFO; if implementation changes it, its existing current header must likewise remain recognized. The future plan must include an actual logger resume from the 43-column header, not only a header-literal assertion. No new chart family is required for FIFO; existing loss/metric plots remain valid. Heavy gradient decomposition, weak-view agreement, fixed-probe drift, candidate-source errors and capture of per-sample vectors remain opt-in research scripts under runtime artifacts, not normal training behavior.

## 9. Ownership and document integration

This is a method-specific extension, not a refactor mandate. Planned owners:

| Owner/surface | Responsibility/change |
| --- | --- |
| `training/relational_queue.py` (proposed narrow module) | Record state, eligibility, refresh/eviction, tensor serialization/validation; no CLI imports. |
| `training/objectives.py` | Extended candidate-axis formula and unchanged pair averaging; on log steps reduce/cache diagnostics from the probabilities/logits used by that loss. Reporting reads those detached statistics, not a second diagnostic forward/backward or a replay of the relation loss. Original zero path retained; private result/return-shape wiring belongs in the plan. |
| `training/dataset.py` and existing `data.paths`/`data.provenance` | Enabled-only prepared selection, stable keys/fingerprint and key/view batch envelope; reuse actual helpers, no universal dataset rewrite. |
| `training/pretrain_recipe.py` | Model-free queue applicability/capacity and optional data preparation before architecture preflight; keep complete SSL/optimizer validation later and carry the prepared result into the recipe/recovery handoff without re-hashing. |
| `training/pretrain_loop.py` | Emit the resolved `Relational queue config:` record once (§8), not from CLI preflight; one queue snapshot per step, cached pre-center observations, success-boundary refresh, save/restore coordination. |
| `training/reporting.py` | CSV columns/migration and existing logger integration. |
| `cli/pretrain.py`, `cli/constraints.py`, `cli_schema.py` | One option, explicitness, method applicability, complete help and schema/card inheritance. No new torch dependency in schema discovery. |
| `utils/checkpoint.py` and directly affected consumers | Queue/method identity consistency without changing encoder layout or leaking training state into finetune outputs. |

G1/G2 were separately approved on 2026-11-11 (§12, Gate decision). The complete revised design was approved for planning on 2026-11-11 (§12, Design decision); create the separate execution plan at `docs/superpowers/plans/2026-11-10-relational-fifo.md` using writing-plans and submit it for separate approval. The plan owns dependency order, concrete edits, fixtures and test commands; include a real-dataset hashing-cost measurement (selected image count, total bytes and elapsed seconds), not an assumed timing baseline. This design is not an execution checklist and does not authorize that measurement now.

Implementation must synchronize these surfaces:

- Master design: bounded queue extension and link to this spec; preserve architecture/documentation rules.
- Relational targets design and its `2026-11-04-relational-pretrain-targets.md` plan: dated extension/supersession notes about the previous no-queue scope, keeping original baseline/history readable.
- Training modularity design: only any genuinely changed ownership/recovery/reporting boundary; no unrelated historical status rewrite.
- `README.md`/`README.cn.md`: navigation and concise optional-feature/default/recovery warning.
- `docs/commands/pretrain.md`/`.cn.md`: option, effective versus declared default, examples, applicability, memory/startup/input constraints, fixed versus effective capacity, exact supported recovery.
- `skills/otuformer-workflow/SKILL.md` and relevant workflow/troubleshooting references: discover the installed schema first, do not invent queue support for older installations, describe the parameter card and recovery warning, retain explicit execution approval behavior. Runtime-generated cards/schema remain the source of truth, not hard-coded duplicate defaults.
- Morphology reference §5.4 if maintained locally: record implementation only after it exists, distinguish project adaptations, keep deferred confidence/adaptive/dense research accurately scoped. The ignored `ref/` file is not the public CLI reference.

New design, plan and developer documentation are English-only. Existing user-facing README/command bilingual conventions remain. Do not add `.cn.md` counterparts for local development specs/plans. No dependency changes or version/release claims are bundled; decide them separately if later requested.

## 10. Verification and research acceptance

Follow the existing testing policy; the execution plan supplies exact fixtures and commands. Required acceptance boundaries:

1. **Objective parity:** rectangular candidate formula/gradients, differentiable student queries, detached teacher/history, correct self column and A/B pairing; unchanged local/global weights and zero/empty-history numerical/RNG parity. Include expanded candidate-length temperature validation.
2. **Queue lifecycle:** partial/full/capacity edges, identity exclusion/refresh, deterministic eviction/age, cross-epoch retention and one snapshot for all terms. An existing non-finite exception must abort without queue mutation or a new epoch checkpoint; no skip/retry, rollback or per-step saver.
3. **Enabled input contract (G1):** actual selected paths, lexical normalization without breaking symlink semantics, identity/content/order locks and root relocation, aligned worker batches, preparation before the probe in both entries; queue-off input/RNG behavior unchanged.
4. **Recovery contract (G1/G2):** request/inheritance/malformed-state handling before output mutation; compact fingerprint/count and latest-commit tail checks; existing batch/step/RNG locks and complete applicable AdamW/patch-state checks before output mutation; controlled CPU epoch save/resume matches uninterrupted model/optimizer/center/queue state. No sidecar requirement.
5. **Method/consumer compatibility:** supported patch modes and KoLeo; DINO omitted stays inactive but every explicit queue value, including zero, is rejected. Preserve read-only/fine-tune identity and the disclosed old-version training-resume boundary; follow the testing policy's device rules.
6. **Reporting upgrade:** explicitly resume a pre-FIFO **43-column** instant CSV through the real logger, preserving old values and blank new cells; retain supported legacy schemas and reject unknown/duplicate headers unchanged. Check probability groups, per-row perplexity, unavailable fields and log-step-only loss-derived statistics.
7. **CLI/schema/docs/skill:** parsing, explicitness/applicability/conditional inheritance, help, generated cards/argv and schema-first older-install behavior; bilingual user docs and English-only development docs. Verify trainer-owned resolved log content for enabled, inherited resume, explicit zero and DINO inactive cases, including G2's effective batch size where applicable; keep requested and resolved records separate and require no duplicate CLI-preflight record. Preserve import-light discovery.

Use focused tests while iterating. Because this work crosses shared loss mechanics and checkpoint identity/recovery boundaries, complete the testing policy's required wider/full gate before declaring milestone compatibility accepted. Do not certify CUDA from MPS or CPU results. Default batch-only recovery/numerics are explicit acceptance criteria, not merely an absence of visible errors.

Research validation is distinct from correctness testing. First compare capacities `0 / 32 / 128` on the small Epidorcus recipe with paired seeds and fixed temperature, augmentation, pairing and weights. Report full final/trajectory metrics and the lightweight diagnostics plus research-script gradient/view/drift checks; do not select only favorable metrics or substitute target entropy for downstream benefit. A larger independently grouped dataset is needed to evaluate coverage, duplicate-source bias and stale-history tradeoffs beyond near-full small-data memory. Dense morphology requires its own global/local evaluation protocol. No experiment, default change or new mechanism is pre-approved by this design.

## 11. Approval and remaining builder discretion

Complete-design approval covers the FIFO feature contract and the smaller author proposals; the separate G1/G2 approval requirement was already fulfilled on 2026-11-11 (§12, Gate decision). **G1 was explicitly approved on 2026-11-11**, including the enabled-only input restrictions, whole-selected-dataset hashing before the probe, and key/content/order lock with root relocation allowed. **G2 was separately approved on 2026-11-11**, including exact batch-size/steps-per-epoch locks, omitted-batch inheritance and complete restorable RNG/progress/schedule requirements without fallback. The user approved the complete revised design and entry into plan authoring on **2026-11-11** (§12, Design decision). Write the execution plan and submit it for separate review and approval; this approval authorizes no implementation, research training, commit or publishing. No audit repair here is permission to execute those changes.

The future plan owns concrete preparation/handoff edits, exact key-normalization calls, loss-statistic plumbing, the 43-column migration fixture, failure-injection test mechanics and focused test commands. This design retains only observable contracts and required acceptance boundaries; it does not add per-step saving, skip/retry recovery, a generic CSV migration framework or a new data-reader rollout.

Builder discretion: private symbols, logical ordered storage mechanics, bounded batching of diagnostics, exact CSV names following existing conventions, test-file organization, and allocation strategy that preserves dtype/order/recovery. Changes to objective, default, identities/input restrictions, state format, resume semantics, weighting, capacity policy, dense scope or public option names require a design amendment. Keep research evidence and implementation completion status separate.

## 12. Draft review record

The author performed a builder-oriented self-review without delegating to a fresh reviewer. The review checked candidate/query dimensions, preservation of current self columns, same-snapshot use, successful-step refresh ordering, pre-center versus historical-center handling, opt-in identity restrictions, independent queue metadata, legacy zero encoding, missing-RNG recovery, downstream method identity, and diagnostic null/zero distinctions. This is self-review, not independent acceptance.

### Source-verified audit revision — 2026-11-11

Corrected center timing, abort/epoch-checkpoint semantics, preparation order, the 43-column CSV predecessor, loss-owned diagnostics and perplexity definition, student gradients, DINO explicit-option applicability and expanded temperature bounds. Checkpoints use compact data fingerprints/counts. The separate G1/G2 approvals were subsequently recorded below; concrete wiring and fixtures belong in the plan.

Two audit suggestions required qualification: `schedule.steps_per_epoch` already exists, and lexical key normalization alone is unsafe for `link/..`; the design reuses the actual shared path semantics rather than the suggested one-line shortcut.

### Gate decision — 2026-11-11

The user separately and explicitly accepted **G1** and **G2**. G1 retains the enabled-only selected-key restrictions, whole-selected-dataset hashing before the probe, and key/content/order resume lock with root relocation allowed; G2 accepts exact batch/steps locks, omitted-batch inheritance and complete restorable state without the legacy missing-RNG fallback. See §11 for the approval boundaries. This resolves both gates, not approval of the complete revised design, a plan, implementation or training.

### Design decision — 2026-11-11

After the gate decisions and audit corrections, the user explicitly approved entering the plan phase. The complete revised design, including the remaining author proposals, is approved as the basis for `docs/superpowers/plans/2026-11-10-relational-fifo.md`. This is permission to author and self-review the plan, not to execute it, train, commit or publish. The original plan filename is retained deliberately.

### Verification — 2026-11-11

Through the normal command runner, `python -m pytest tests/test_docs.py -q` returned **42 passed**. Running the same command against a **complete** committed-HEAD archive (source, tests, docs and READMEs) also returned **42 passed**. Environment: Python 3.11, pytest 9.1.1, Rich 15.0.0, Typer 0.25.1 and Click 8.4.2.

The earlier partial source-only archive did not validate the documentation corpus and is not evidence of a repository defect or an implementation blocker. Environment/output evidence is local to `runs/fifo_design_audit_20261111/`; no help/test fix is included.

An in-memory one-column probe confirmed the 43-column migration gap without changing source or the original CSV. Local links and whitespace checks pass. This revision changes only this spec, not the plan or production/public-doc surfaces. No spike rerun, full production suite, or FIFO implementation verification was performed.


### Implementation — 2026-11-11

The complete design contract in §§1-9 is implemented, and the execution record in
[the plan](../plans/2026-11-10-relational-fifo.md) holds the task-level results and
gate evidence. Observable boundaries:

- `--relational-queue-size` is declared as omitted (`None`) and resolves a fresh
  omission to zero; the effective capacity 0 takes the original loss/data path and
  builds no queue, keyed dataset or queue diagnostics.
- Enabled runs prepare the selection once with the G1 identity contract (in-root
  unique keys, ordered key/size/SHA-256 manifest digest) before any probe, keep
  `B + E` current-first candidates, use the pre-step center for both current and
  history, commit only after a successful optimization step, and persist the closed
  version-1 bundle pair at epoch boundaries only.
- Enabled recovery applies G2: unchanged batch size and steps-per-epoch, a complete
  target-aware restorable RNG state, strict student/teacher/patch/center states and
  usable per-parameter AdamW state, all validated before output mutation.
- Reporting adds the FIFO columns, recognizes the pre-FIFO 43-column header, and the
  trainer emits the resolved `Relational queue config:` record once.

Certification scope: full-suite correctness evidence on CPU plus model-free request
tests. No accelerator acceptance, research training or quality claim follows from
it, and CUDA RNG continuation is unverified on this host.
