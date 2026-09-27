# OTU-Former Sparse-Label Fine-Tuning Implementation Plan

> **Status:** Implemented and verified (v0.9.0). The task list below is kept
> as the record of what was executed; every step is checked. Commits remain
> a separate, explicitly requested action.

**Goal:** Add one optional CB-DRW long-tail strategy and one automatic,
precision-oriented pseudo-label feedback round without complicating ordinary
fine-tuning.

**Spec:**
`docs/superpowers/specs/2026-09-26-otuformer-sparse-label-pseudolabel-design.md`

**Tech Stack:** Existing Python, PyTorch, pandas, Typer, NumPy, and pytest stack;
no new dependency.

## Public Interface

Add only these fine-tune options:

```text
--long-tail none|cb-drw
--pseudo-label-from FINETUNE1.pth
--pseudo-similarity-floor FLOAT   # default 0.75
--pseudo-min-gap FLOAT            # default 0.10
--pseudo-neighbors INT            # default 15
--pseudo-cap-multiplier INT       # default 3
--pseudo-absolute-cap INT         # default 50
```

All other existing fine-tune options retain their current meaning. No new
pseudo-label subcommand, calibration command, report command, manifest input,
manual review step, general config-difference switch, feature-cache option, or
pseudo-label weighting option is added.

## Global Constraints

- Without `--pseudo-label-from` and with `--long-tail none`, training behavior,
  dataset shape, losses, resume, and raw-CLS extraction remain unchanged.
- New checkpoint metadata is additive and backward-readable. Existing ordinary
  manifest validation remains unchanged: an out-of-root absolute image reference
  is a hard error. Pseudo-source eligibility is evaluated only after a valid
  ordinary manifest exists; it cannot convert an existing training error into a
  warning.
- Pseudo mode is ArcFace-family only. Finetune#1 and finetune#2 use the same
  loss, resolved experiment configuration, and long-tail strategy; `none` and
  `cb-drw` comparisons need separate matching finetune#1 sources.
- Finetune#2 initializes from the same original SSL checkpoint, never from
  finetune#1. It reuses the recorded resolved path automatically; the existing
  `--checkpoint` is needed only if that file moved, and its file hash must match.
- Candidate images are discovered automatically as supported images under the
  input root minus expert CSV references.
- Finetune#1 and finetune#2 remain two explicit `otuformer finetune` invocations
  so users can check first-round training/downstream quality before selecting
  its checkpoint. Use the completed `finetune_latest.pth` unless external
  held-out metrics select another completed epoch checkpoint; there is no
  automatic best checkpoint. There is no per-pseudo-label approval step.
- In pseudo mode, filesystem real-path/inode checks prevent expert images from
  re-entering through aliases; accepted-row SHA-256 deduplication makes fallback
  modes agree without changing the default ordinary CSV path.
- The input root must contain one comparable biological marker. This is a user
  prerequisite documented in help/README, not a programmatic classifier.
- The initial implementation has no DDP or gradient-accumulation semantics.
- Calibration and downstream experimental reporting remain design guidance,
  not implementation tasks.

## File Structure

- Create `src/otuformer/embedding/pseudo_label.py` for deterministic candidate
  discovery, raw-CLS scoring, mutual-kNN, cap selection, and diagnostics.
- Create `src/otuformer/utils/paths.py` (stdlib only) for the shared
  canonical-reference and per-file identity helpers used by both the training
  provenance code and pseudo-label discovery; `extractor.py` needs no forced
  change because the in-memory raw-CLS helper reuses its model loading.
- Modify `src/otuformer/embedding/extractor.py` only as needed to expose a pure
  in-memory raw-CLS forward helper; the pseudo path must not invoke extract CLI
  outputs, metrics, UMAP, or attention-pooling checkpoint writes.
- Modify `src/otuformer/cli/finetune.py` for the five options and read-only
  preflight.
- Reuse `prepare_output_dir` from `src/otuformer/utils/io.py` only after pseudo
  preflight and nonempty acceptance. Existing output signatures are read from
  `src/otuformer/cli/{pretrain,extract,cam,annotate}.py` and finetune outputs in
  `src/otuformer/training/trainer.py`; ordinary command layouts are unchanged.
- Modify `src/otuformer/training/dataset.py` only to accept validated in-memory
  rows on the opt-in path while preserving `(image, label)`.
- Modify `src/otuformer/training/loss.py` for unreduced ArcFace-family CE and a
  reusable compact penalty.
- Modify `src/otuformer/training/trainer.py` for metadata, CB-DRW, pseudo-row
  integration, checkpointing, and resume.
- Add `tests/test_pseudo_label.py` and
  `tests/training/test_finetune_provenance.py`; extend existing CLI, dataset,
  loss, trainer, and extractor tests.
- Update `README.md` and `README.cn.md` for the delivered commands only.

### Task 1: Add the minimal CLI and source identity

**Files:** Modify `src/otuformer/cli/finetune.py`,
`src/otuformer/training/trainer.py`, `tests/test_cli_smoke.py`; create
`tests/training/test_finetune_provenance.py`.

- [x] **Step 1: Write failing CLI tests.** Assert help shows exactly the five
  new options. Help must say that the similarity floor is an uncalibrated
  top-three mean raw-CLS cosine class score rather than a probability; higher
  floor and gap reject more; and smaller neighbor counts are usually more
  local/strict while larger counts are usually broader/permissive but
  data-dependent. Validate `--long-tail` choices, finite similarity floor in
  `[-1,1]`, finite gap in `[0,2]`, and positive integer neighbors before output
  creation.
  Assert pseudo mode rejects SupCon, a missing `--pseudo-label-from` file, a
  finetune#2 source (`pseudo_round=1`), and an old checkpoint missing required
  provenance. Assert `--checkpoint` is optional in pseudo mode, any pseudo rule
  option without a source/resume is an error, and help notes that omitted
  experiment options inherit source values. Add a coverage test that introspects
  every finetune Typer option and requires membership in exactly one explicit
  `experiment`, `operational`, or `control` mapping. Experiment options map to
  exact resolved keys, including `metric_head_lr -> effective_metric_head_lr`
  and `augmentation -> augmentation_profile, augmentation_config`; derived keys
  are listed separately. Future unclassified options fail. Assert an ordinary
  invocation requires no new option.
- [x] **Step 2: Write failing checkpoint tests.** A new ordinary finetune
  checkpoint attempts to record additive metadata needed by a later pseudo run:
  `pseudo_round=0`, canonical expert image/label hash, SSL initialization file
  resolved path and SHA-256, effective backbone/head learning rates, weight
  decay, batch size, planned epochs, loss settings, freeze ratio,
  augmentation/orientation, image/model/embedding sizes, class-label order, and
  long-tail strategy. Record fixed AdamW, ArcFace scale `64.0`, margin `0.5`,
  compact cap `0.5`, and optimizer layout; there is no fine-tune
  scheduler/warmup/AMP/clipping setting. Existing `config` is the only
  configuration authority. Preserve existing ordinary manifest validation,
  including the hard failure for an absolute reference outside `ROOT`. After a
  valid manifest exists, print one startup Info line for pseudo-source
  eligibility; an ineligible line contains the concrete loss/provenance reason
  and records `pseudo_source_eligible=false` with that reason. A real pseudo
  request treats ineligibility as an error.
- [x] **Step 3: Run red.** Run
  `pytest -q tests/test_cli_smoke.py tests/training/test_finetune_provenance.py -k 'long_tail or pseudo_source or pseudo_metadata'`.
- [x] **Step 4: Implement minimal parsing and metadata.** Reuse Click parameter
  sources to distinguish omitted options from explicit values. Additive metadata
  must not alter ordinary training or legacy resume. Hash checkpoints with
  existing `_sha256_file` semantics: SHA-256 of file bytes. Existing manifest
  validation, including out-of-root escape rejection, remains mandatory for
  ordinary and pseudo runs. The additional pseudo path-identity/alias checks
  determine source eligibility after that manifest succeeds; selection of an
  ineligible checkpoint for pseudo mode is a hard error. Use the exact same
  pseudo identity function at finetune#1 startup and finetune#2 preflight,
  printing an immediate one-line
  eligibility result before long training begins. Classification constants use
  exact CLI/config key names and mappings from the design, not broad labels such
  as "loss settings". Preserve the existing v0.8.0 manifest algorithm and label
  paths: lexical `normpath(...).as_posix()` references, `MetricDataset.label_names`
  for manifest labels, and `MetricDataset.class_to_idx` for checkpoint
  `class_labels`. Pseudo provenance reuses these values rather than independently
  reading/converting labels. Add regression fixtures for string, numeric,
  zero-padded-string, whitespace, pandas float-like, and missing-value label
  columns. Preserve the existing result for each fixture, including any current
  validation failure; for successful inputs assert `ds.labels`, `label_names`,
  `class_to_idx`, checkpoint `class_labels`, manifest SHA-256, and legacy resume
  comparisons are byte-for-byte unchanged. Keep the
  existing out-of-root manifest rejection test. NFC is limited to new pseudo
  real-path identity and never redefines `train_manifest_sha256`.
- [x] **Step 5: Run green.** Re-run Step 3 and verify a legacy/default fine-tune
  smoke test still passes.

### Task 2: Implement automatic candidate discovery and scoring

**Files:** Create `src/otuformer/embedding/pseudo_label.py`,
`tests/test_pseudo_label.py`; reuse `src/otuformer/embedding/extractor.py`.

**Core interface:**

```python
generate_pseudo_rows(
    *,
    expert_rows: list[dict],
    image_root: Path,
    pseudo_checkpoint: Path,
    similarity_floor: float = 0.75,
    min_gap: float = 0.10,
    neighbors: int = 15,
    device: str = "auto",
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[list[dict], list[dict], dict]
```

The return values are accepted training rows, all candidate diagnostic rows,
and summary metadata.

- [x] **Step 1: Write failing candidate-discovery tests.** Use one path-identity
  function for experts and candidates; only the read-only pseudo-source
  eligibility check is expert-only. Recursively enumerate without following
  directory symlinks; subtract experts by nonzero `(st_dev, st_ino)` when
  available. If any shared inode conflicts across different real paths and
  bytes, switch the entire scan to platform `normcase` + NFC real path and
  record `inode_collision_count`; content is never ordinary identity. Record
  `inode` or `normalized-realpath`. Cover `./`, case and Unicode behavior,
  distinct `A.jpg`/`a.jpg` on case-sensitive filesystems, hard links,
  root-internal file symlinks, and explicit CSV references through in-root
  directory symlinks; both preserve their link-path canonical reference.
  Cover escaping file symlinks, directory symlink cycles, duplicate aliases,
  expert escapes, and conflicting identities. Record skipped directory-symlink
  count and ten examples. Ignore non-images; distinct files remain separate
  image units even when their bytes match. Assign every pre-scan rejection to
  the `none` seed-count bin and assert all bins sum to candidate count.
- [x] **Step 2: Write failing feature-rule tests.** Cover eligible top-three
  mean for eligible winners and runner-ups, one/two-seed runner-up maximum,
  strict floor/gap equality, no competitor, invalid seed failure, invalid
  candidate rejection, and `min(k,pool_size)`. Candidate-side neighbors are
  valid seeds; for a winning-class seed, seed-side neighbors are valid
  candidates plus valid seeds from other classes, excluding all seeds from its
  own class. Assert that one identical winning-class seed `s` belongs to the
  candidate-side list and reciprocally contains the candidate in its seed-side
  list; two different seeds satisfying one direction each must fail. Verify
  small classes cannot provide a winning reciprocal seed, other-class seeds can
  block a distant unknown cluster, a genuine candidate slightly looser than at
  least six compressed same-class training seeds can still pass, the pre-cap
  class bound is `k*n_c`, and ties are deterministic.
- [x] **Step 3: Write failing cap/diagnostic tests.** Apply the fixed
  `min(3*n_c,50)` class cap after other rules, ranked by score then reference.
  Every candidate receives `accepted`, score/gap, neighbor result, cap rank,
  `max_seed_similarity`, and complete rejection reasons. When any expert
  similarity exceeds `0.999`, hash the candidate and every expert above that
  threshold: any equal bytes reject as `expert_content_duplicate`; otherwise
  set `possible_duplicate=true`. Record near-threshold candidate count/rate and
  exact-content rejection count so widespread high similarity exposes possible
  feature collapse. Distinct equal-content expert files remain legal and are
  diagnostic only. After the class cap, hash only provisional accepted rows;
  keep the canonical-reference-first row per content hash and reject the rest as
  `accepted_content_duplicate` without refilling cap slots. Assert inode and
  normalized-realpath modes produce the same final accepted set. Summary
  contains rejection counts; explicitly
  optimistic `training_seed_loo_within_top3_*`,
  `training_seed_loo_runner_up_*`, and `training_seed_loo_gap_*` quantiles only
  for classes with at least four seeds, using the same eligible top-three versus
  one/two-seed maximum runner-up rule; candidate top1/gap p50/p90/p99;
  floor-only, gap-only, and mutual-kNN-only counterfactual pass counts;
  acceptance by winning-class seed-count bins `none`, `3`, `4–5`, `6–10`,
  `>10`; and ineligible class/seed counts. Invalid/no-winner candidates use
  `none`. Empty acceptance returns these data; the CLI prints them to stderr and
  fails without creating `out_dir`. The library never prints or creates output.
  Invalid source/identity/extraction/empty-pool/seed-feature conditions raise one
  `PseudoLabelError(ValueError)`. A retry may re-extract features. Rejected rows
  never enter accepted rows.
- [x] **Step 4: Run red.** Run `pytest -q tests/test_pseudo_label.py`.
- [x] **Step 5: Implement.** Expose/reuse a pure in-memory raw-CLS forward helper
  built from existing extractor model loading, center-crop transform, and batch
  logic. It returns `(canonical_ref, vector)` pairs and never prints, creates
  directories, writes CSV/metrics/UMAP/checkpoints, or depends on extraction
  order. Do not call the high-level extract command or require an `extract_csv`
  file; align returned vectors by canonical reference. Test an empty-candidate
  failure leaves no new file anywhere under `ROOT` or requested `out_dir`. Use
  the pseudo checkpoint's resolved image size
  and internal `device`, extraction `batch_size`, and `num_workers` defaults
  (`auto`, `32`, `4`) without new pseudo-rule options. Record a raw pre-L2
  little-endian float32 feature hash in canonical reference order, but never use
  it as an equality/resume gate because device numerics can differ. Use
  fixed-size blockwise exact class aggregation/top-k so a `100,000 x 3,000`
  float32 matrix is not retained; reuse similarities for class scoring and both
  neighbor directions without changing deterministic ordering.
- [x] **Step 6: Run green.** Run `pytest -q tests/test_pseudo_label.py` and an
  existing extractor raw-CLS test.

### Task 3: Implement CB-DRW behind one strategy option

**Files:** Modify `src/otuformer/training/loss.py`,
`src/otuformer/training/trainer.py`, `tests/test_training_loss.py`,
`tests/training/test_pretrain_alignment.py`.

- [x] **Step 1: Write failing loss tests.** Add unreduced CE for ArcFace and
  Sub-center ArcFace. Assert ordinary scalar loss is unchanged. For compact
  Sub-center, explicitly assert the old value equals
  `unreduced(...).mean() + compact_penalty()` at the existing cap `0.5` and
  weight `0.1`, preserving its pair/class normalization.
- [x] **Step 2: Write failing CB-DRW tests.** Use fixed beta `0.99`, cap `3.0`,
  start `floor(.5*T)`, and ramp `floor(.1*T)`. Expert rows alone determine class
  frequencies. `r_bar(t)` is the mean over **training rows only**: expert plus
  accepted known-pseudo rows; diagnostic rejected rows never enter it. Test the
  specified 10/128 values (`r_bar≈0.845792`, tail coefficient `≈3.546971`).
  Set `T=len(loader)*epochs` with the current `drop_last=False`; test `T<10`
  immediate switching without division by zero, `T=1` weighting from the first
  step, partial batches, and resume using the completed-step count before the
  next step.
- [x] **Step 3: Run red.** Run
  `pytest -q tests/test_training_loss.py tests/training/test_pretrain_alignment.py -k 'unreduced or compact_penalty or cb_drw or long_tail'`.
- [x] **Step 4: Implement.** Keep the ordinary scalar path for
  `--long-tail none`. For CB-DRW, reduce weighted CE by actual batch size and
  manifest-wide training-row mean; add compact penalty afterward. Record fixed
  constants, expert counts, target weights, total/completed steps, and selected
  strategy in checkpoints and logs.
- [x] **Step 5: Run green.** Re-run Step 3.

### Task 4: Integrate automatic finetune#2 and resume

**Files:** Modify `src/otuformer/training/dataset.py`,
`src/otuformer/training/trainer.py`, `src/otuformer/cli/finetune.py`,
`tests/test_training_dataset.py`, `tests/test_cli_smoke.py`,
`tests/training/test_finetune_provenance.py`,
`tests/training/test_pretrain_alignment.py`, `tests/test_extractor.py`,
`tests/test_export.py`, `tests/test_cam.py`.

- [x] **Step 1: Write failing preflight tests.** For a new pseudo run, require
  the same expert hash, original SSL initialization file SHA-256,
  ArcFace-family loss/settings, model/head/dimensions, freeze ratio, effective
  learning rates, weight decay, augmentation/orientation, seed, batch size,
  planned epochs, class-label order, optimizer/loss constants and layout, and
  long-tail strategy as finetune#1. `num_workers` is operational and may differ.
  All omitted experiment options inherit from the source; explicit conflicts
  fail.
  Reuse the source's recorded SSL path, or accept the optional existing
  `--checkpoint` only as a relocated file with the same SHA-256. Operational
  settings are not compared. Document/recommend separate data and run trees;
  `ROOT` is a biological data root, not the repository or `runs/` root. Before
  extraction or output creation, resolve paths and reject any
  ancestor/descendant overlap between `out_dir` and `ROOT` and
  every file-valued input inside `out_dir`, including pseudo/SSL checkpoints,
  `train_data`, `visualize_data`, and future explicit CSV/config/weight/ONNX
  inputs. Overwrite input protection remains a final guard; resume retains the
  existing overwrite prohibition. Preflight completes before
  `prepare_output_dir`.
- [x] **Step 2: Write failing end-to-end CPU test.** Train a tiny finetune#1,
  run finetune#2 with `--pseudo-label-from`, discover candidates from the image
  root, accept at least one synthetic candidate, initialize from the original
  SSL checkpoint, train directly, and verify expert+pseudo rows share the
  original ordered class mapping. Assert default periodic metrics still read
  expert `train_data`, while explicitly supplied `--visualize-data` remains the
  selected arbitrary metrics/UMAP CSV. An empty accepted set prints diagnostics
  and fails without creating the requested output directory or any file under
  the image root.
- [x] **Step 3: Write failing output tests.** Test rejection when `out_dir` is
  equal to/inside `ROOT`, is a parent of `ROOT`, or contains any file-valued
  input such as the pseudo source, SSL checkpoint, expert/visualization CSV, or
  future config/weight file; ensure checks precede destructive overwrite.
  Do not add a marker to ordinary command outputs. Candidate discovery skips
  strong finetune (`logs/finetune.log` + `finetune_latest.pth`), pretrain
  (`logs/pretrain.log` + `SSL_latest.pth`), extract (`logs/extract.log` plus
  `embeddings.csv`, `metrics.csv`, or `umap.pdf`), CAM (`logs/cam.log` plus
  `cam_summary.csv` or `figures/`), and annotate (`logs/annotate.log` plus
  `annotation_summary.json`, `otu_table.csv`, or
  `UPGMA_tree_partitions_annotated.pdf`) signatures. Treat this as a safety net,
  not permission to mix run output with biological data. Warn but do not skip a
  directory containing only weak `.pth`/`logs` or unrecognized `runs/` signals;
  supported images there remain candidates. Record signature, count, and
  examples, and verify existing tests that assert ordinary output
  structure remain unchanged. After successful preflight, `pseudo_labels.csv`
  includes every candidate and diagnostic columns; `pseudo_summary.json`
  records constants, preprocessing, score distributions,
  rejection/counterfactual/near-duplicate counts, class eligibility, skipped
  symlinks/output subtrees, and hashes. Checkpoint accepted-row hash and CSV
  accepted rows agree. No user edit or `annotate` step is involved.
- [x] **Step 4: Write failing resume tests.** Finetune#2 checkpoints store
  `pseudo_round=1`, source checkpoint file SHA-256, authoritative accepted rows,
  a record-only candidate-pool hash, rule parameters, and diagnostics hash.
  Assert these are additive payload fields ignored by `extract`, `export`, and
  `cam`, and that all three consumers still load the checkpoint directly. Resume
  uses saved accepted rows and config, revalidates expert and accepted rows, and
  rejects explicit conflicts. Resume needs neither the original finetune#1 file nor
  `--pseudo-label-from`; if supplied, the source hash must match. A moved root
  warns but remains valid when required rows match. Added/removed non-accepted
  candidates warn rather than fail. Resume never clears or rewrites
  `pseudo_labels.csv`; a missing or mismatched file only warns and logs its
  current hash because checkpointed accepted rows are authoritative. Explicitly
  test the design's hard-gate/warning-only/record-only matrix: raw feature,
  candidate-pool, and current CSV hashes never gate resume; expert/accepted rows,
  SSL/experiment/class/preprocessing identity do. A `pseudo_round=1` checkpoint
  cannot start a third round.
- [x] **Step 5: Run red.** Run
  `pytest -q tests/test_training_dataset.py tests/test_cli_smoke.py tests/training/test_finetune_provenance.py tests/training/test_pretrain_alignment.py tests/test_extractor.py tests/test_export.py tests/test_cam.py -k 'pseudo or long_tail'`.
- [x] **Step 6: Implement.** Add an opt-in validated in-memory row path to
  `MetricDataset` without changing its return shape. Generate pseudo decisions
  and failure diagnostics in memory; only after all checks and a nonempty
  accepted set call `prepare_output_dir`, write diagnostics, and train. Preserve
  the existing `--checkpoint`/`--resume` loop, checkpoint naming, and
  `--visualize-data` behavior. By default periodic metrics still reuse expert
  `train_data`; an explicitly supplied visualization CSV remains allowed for
  arbitrary UMAP/metric diagnosis. Log that expert true-label CSVs are the
  recommended RUN1/RUN2 training comparison source, while formal comparison
  belongs on external `HELDOUT_ROOT`.
- [x] **Step 7: Run green.** Re-run Step 5, then
  `pytest -q tests/test_training_loss.py tests/test_training_dataset.py`.

### Task 5: Document and verify the delivered workflow

**Files:** Modify `README.md`, `README.cn.md`, `tests/test_cli_smoke.py`.

- [x] **Step 1: Write failing help/documentation assertions.** The documented
  workflow contains only the five new options and clearly states that
  `--train-data` supplies expert labels, candidates are `ROOT - expert refs`,
  pseudo generation feeds finetune#2 automatically, `--checkpoint` may be
  omitted in pseudo mode, held-out images and `out_dir` must remain outside and
  non-overlapping with `ROOT`, all file-valued inputs must remain outside
  `out_dir`, long-tail strategy must match between rounds, directory symlinks
  are not traversed, explicit `--visualize-data` remains supported, expert rows
  are only the recommended RUN1/RUN2 training-diagnostic source, and `annotate`
  is unrelated.
- [x] **Step 2: Add concise documentation.** Show:
  1. ordinary finetune;
  2. optional `--long-tail cb-drw`;
  3. finetune#1 followed by finetune#2 with `--pseudo-label-from` and optional
     three feature-rule overrides; no candidate-list path or repeated first-run
     settings are required.
  State that one root must contain one comparable marker and should be a
  dedicated data tree separate from `RUN_ROOT`; output-signature skipping is a
  safety net, and supported images in unrecognized run trees remain candidates.
  Pseudo mode is one ArcFace-family feedback round across two explicit finetune commands,
  preprocessing is fixed to center-crop, and diagnostic CSV is not an approval
  gate. Explain that the split allows finetune#1 checkpoint-quality inspection,
  not manual pseudo-label correction. Give a concrete check: place held-out
  images under `HELDOUT_ROOT` outside training `ROOT`, then run existing
  `otuformer extract --input-images-dir HELDOUT_ROOT --label-csv HELDOUT.csv`
  for RUN1 and RUN2 and compare `metrics.csv`; training-set metrics are
  optimistic, and held-out images inside `ROOT` would contaminate finetune#2.
  State that default periodic metrics reuse expert `train_data`, and recommend
  expert true-label rows for comparable RUN1/RUN2 training diagnostics; preserve
  `--visualize-data` for arbitrary explicitly requested CSV metrics/UMAP and do
  not silently ignore it. Formal comparisons remain the separate held-out runs.
  Recommend completed `finetune_latest.pth` unless external held-out metrics
  select another completed epoch checkpoint; do not call it "best". State that
  `ROOT` must be a real directory tree, skipped directory symlinks/output
  subtrees are reported, and new pseudo output must not overlap `ROOT` or contain
  any file-valued input. Explain
  that `none` and `cb-drw` need separate matching finetune#1 sources because
  long-tail strategy cannot change between rounds. Explain that the floor is an
  uncalibrated raw-CLS cosine class score, not probability, and state the
  stricter/looser direction for all three feature-rule options.
- [x] **Step 3: Keep evaluation deferred.** Link to the design's future
  evaluation guidance; do not implement calibration grids, statistical report
  commands, transferred thresholds, SSL/flip agreement, or a generic experiment
  runner. Do not alter `annotate`.
- [x] **Step 4: Verify focused suites.** Run
  `pytest -q tests/test_pseudo_label.py tests/test_training_loss.py tests/test_training_dataset.py tests/test_cli_smoke.py tests/training/test_finetune_provenance.py tests/training/test_pretrain_alignment.py tests/test_extractor.py tests/test_export.py tests/test_cam.py`.
- [x] **Step 5: Verify compatibility.** Run `pytest -q` and
  `git diff --check`. Report environment-limited failures instead of claiming a
  pass.

## Execution Gate

> **Historical note.** This gate was written before implementation and is kept
> only as a record. The implementation it guarded was explicitly authorized and
> is delivered in v0.9.0; see the Status line at the top of this plan. It does
> not constrain future work.
