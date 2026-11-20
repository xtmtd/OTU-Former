# Workflow Execution Efficiency — Execution Record

English-only implementation record for
[the plan](2026-11-19-workflow-execution-efficiency.md) and
[the design](../specs/2026-11-18-workflow-execution-efficiency-design.md).

Baseline: `2d37064` (pinned by the plan). Branch: `OTU-Former-dense`. This record
is the authority for what was run, on which commit, and with what result; the
plan's checkboxes are the pre-execution authorization artifact and are not
rewritten.

## Commits (oldest first)

| Task | Commit | Subject |
| --- | --- | --- |
| — | `bd5a455` | docs(dense): add workflow execution efficiency design and plan |
| 0 | `e688ce0` | test: repair pretrain resume smoke fixtures |
| 1 | `b114759` | feat(pretrain): batch gradient-norm checks and defer the finetune readback |
| 2 | `ab555c9` | feat(finetune): batch prototype assignment counts |
| 3b | `0e6dfce` | feat(pretrain): snapshot evaluation rows and select before decode |
| 3c | `db2de08` | docs: clarify periodic evaluation behavior |
| 4 | `aee701a` | feat(finetune): hand over one path-bound payload and loss decision |
| 5 | `23955c8` | feat(pretrain): reuse non-FIFO relational targets within a step |
| 6 | `44b20b8` | feat(pretrain): gather selected tokens before the iBOT and masked heads |
| 7 | `c362336` | feat(pretrain): gate the cosine value and reuse step loss scalars |
| 8 | `95e454e` | feat(spatial): bind the checkpoint digest and model to one read |
| 9 | `52871c6` | feat(embedding): share the seeded sorted row-selection rule |
| 10 | `f254bcf` | feat(delineation): share the within-group distance summary |
| 11 | `0cb7ff3` | fix(cluster): score monophyly on the fixed output tree and build one metrics linkage |
| 12 | `61322bc` | feat(diversity): compute OTU-table metrics on counts without expansion |
| 13 | `e67714d` | feat(cam): share one ordered image selection per invocation |
| 14 | `7376662` | feat(export): prepare the embedding graph before the overwrite boundary |
| 15/16 | `a46475a` | docs: record fixed-tree monophyly and implementation dispositions |
| review fix | `fda300d` | fix(pretrain): keep the recovery timer and narrow masked teacher norms |
| review fix | `0f2e921` | fix(finetune): keep the recovery timer and reuse the loss readback |
| review fix | `008c888` | fix(extract): deserialize the snapshotted checkpoint bytes |
| review fix | `52cf25b` | docs: record the final-review fix pass |
| audit fix | `fb38344` | fix: close the remaining reuse gaps found in the audit |
| audit fix | `2eaef2c` | docs: record the second audit round |
| audit fix | `73cb992` | fix(pretrain): verify row order before reusing evaluation labels |
| audit fix | `5a32daf` | docs: record the label/path snapshot guard |

Per-task commits were authorized by the plan's standing clause; no commit after
the planned tasks was made without a separate reason recorded in the design's
addenda (review and audit rounds).

## Per-task focused commands and results

Every row was run on the commit named above, after that task's changes.

| Task | Command (abbreviated to the node set) | Result |
| --- | --- | --- |
| 0 | six `tests/test_cli_smoke.py` resume nodes | 6 passed |
| 1 | dino/ssl norm nodes + `test_dino_resume.py` + `test_ssl_metrics.py` header node | 5 + 72 passed |
| 2 | loss-diagnostics nodes, header parity, finetune integration node | 8 passed |
| 3b | `test_pretrain_alignment.py` + `test_startup_phases.py` + `test_relational_queue_recipe.py` + `test_output_safety.py` + `test_docs.py`, then `test_cli_smoke.py` + `test_cli_startup.py` | 366, then 230 passed |
| 3c | `tests/test_docs.py` | 44 passed |
| 4 | provenance + pseudo-label + lineage + docs + alignment + output safety, then cli smoke + alignment | 411, then 430 passed |
| 5 | queue objectives + queue, dino + masked + koleo, alignment | 52, 214, 201 passed |
| 6 | masked + loss + alignment, dino | 445, 72 passed |
| 7 | masked + dino + ssl metrics + alignment | 443 passed |
| 8 | `test_spatial_extract.py`, then spatial export/records/extractor/output-safety/spatial cli | 42, 189 passed |
| 9 | extract data/records + alignment + startup, cluster + cli sampling nodes | 282, 42 passed |
| 10 | `test_distance.py`, `test_cli_annotate.py`, cluster/annotate cli nodes | 8, 5, 24 passed |
| 11 | cluster cli nodes + `test_partition.py` + `test_cluster_plot.py` | 20 passed |
| 12 | `test_diversity.py`, diversity cli nodes | 49, 4 passed |
| 13 | `test_cam.py` + output safety + cam cli nodes | 170 passed |
| 14 | `tests/test_export.py` + `tests/test_spatial_export.py`, export cli nodes, `tests/test_output_safety.py` | 46 passed, 6 passed, 63 passed |
| 15/16 | full suite `python -m pytest -q -p no:randomly` | 2688 passed, 1 skipped (1336 s) |
| review fix | startup phases + masked + alignment + checkpoint utils + extract records/data cli, targeted cli nodes | 538, 10 passed |
| audit fix (2nd round) | cli smoke + startup phases + cam + docs + masked, alignment | 538, 202 passed |
| audit fix (3rd round) | alignment + startup phases + cli smoke; then alignment+diversity+cam+partition+distance+annotate+masked+loss+startup | 444 passed; 595 passed |
| audit fix (4th round) | per-fix commands in the round tables below | recorded per fix |

Every new contract was watched fail first: the changed production file was
stashed, the new node was run, and only a real red run was accepted before the
implementation was restored.

## Final acceptance state

- **Full regression runs, both on older commits, neither on the current HEAD:**
  - before the review fixes, on `a46475a`: 2688 passed, 1 skipped (1336 s);
  - after round 1, on `52cf25b`: 2696 passed, 1 skipped (1168 s).
  Every later commit was verified with the focused suites in the round tables
  below. The master testing policy's full-suite gate for merge/release
  acceptance must be re-run on the acceptance commit.
- Consumer chains actually exercised by existing suites:
  - `extract → cluster → diversity`: `tests/test_examples_e2e.py` (this chain
    contains no `annotate` step).
  - `cluster → annotate` on real products:
    `tests/test_cli_annotate.py::test_annotate_summary_matches_the_cluster_product`.
  - spatial `export → extract` (ONNX handed to extract):
    `tests/test_spatial_onnx.py::test_cli_runs_spatial_onnx_extraction_and_reads_it_back`
    and `::test_onnx_and_checkpoint_share_spatial_product_semantics`;
    `tests/test_export.py` / `tests/test_spatial_export.py` cover graph creation,
    not the consumption chain.
  - CAM prediction/array/overlay alignment: `tests/test_cam.py` plus the CAM
    nodes in `tests/test_cli_smoke.py`.
- `git diff --check` clean; `runs/` untouched; no benchmark or training artifact
  written by the implementation.

## Performance

Not authorized. Every §8 gate records `not measured: authorization pending` in
the design's §10 addendum, including C6. No speed claim is made anywhere in this
record.

## Known partial deliveries

Recorded so acceptance reads them as deliberate, not silent:

- **B**: the ordinary resume path reads its checkpoint once. A new pseudo round
  still opens the pseudo-source for inheritance, pseudo validation and candidate
  feature extraction; those are different consumers of one path, not one read.
- **D1**: the spatial path binds digest, model and protected snapshot to one
  read. The embedding record path now deserializes its snapshot bytes and then
  re-reads once as the drift check, so peak memory no longer holds two byte
  buffers plus the payload, but it still performs one verification read. The
  explicit-layer CLI probe keeps its own CPU construction.
- **D2**: the fixed-tree monophyly correction and the single metrics linkage are
  implemented; the cohort submatrix is sliced once for the final (already
  sampled) index set.
- **E3**: `_evaluate_records` stays in the CLI because it publishes the UMAP
  artifact; the shared calculations it used moved to their owners instead.

## Round tables

Each round lists the fixes, their commits, the focused command and the result on
the commit that contains them.

### Round 1 — final whole-branch review

| Fix | Commit | Command | Result |
| --- | --- | --- | --- |
| Nested manifest timer no longer overwrites the recovery phase; masked-feature teacher blocks gathered before the pointwise norm | `fda300d` | `tests/training/test_startup_phases.py`, `tests/training/test_masked_pretrain.py` (new nodes) | passed; both new nodes fail on the pre-fix source |
| Finetune recovery timer; one loss readback per step | `0f2e921` | `tests/training/test_pretrain_alignment.py::test_finetune_reads_the_loss_once_per_step` | passed; 6 reads before the fix |
| Extract deserializes its snapshotted checkpoint bytes | `008c888` | `tests/test_checkpoint_utils.py::test_load_checkpoint_bytes_deserializes_exactly_those_bytes` | passed |
| Round-wide set | `fda300d..52cf25b` | startup phases + masked + alignment + checkpoint utils + extract records + extract data cli, then targeted cli/masked nodes | 538 passed, then 10 passed |

### Round 2 — follow-up audit of the range

| Fix | Commit | Command | Result |
| --- | --- | --- | --- |
| Fresh ordinary run resolves the training selection once for dataset + manifest | `fb38344` | `tests/training/test_startup_phases.py::test_ordinary_run_resolves_the_training_selection_once` | passed; 2 resolutions before the fix |
| CAM collects the manifest once (directory and CSV) | `fb38344` | `tests/test_cli_smoke.py::test_cam_collects_the_manifest_once_for_directory_and_csv` | passed; 2 collections and 2 traversals before the fix |
| Round-wide set | `fb38344..2eaef2c` | cli smoke + startup phases + cam + docs + masked, alignment | 538 passed, 202 passed |

### Round 3 — label/path snapshot race

| Fix | Commit | Command | Result |
| --- | --- | --- | --- |
| Ordered row references travel with the resolved paths; a same-length reorder is rejected | `73cb992` | `tests/training/test_pretrain_alignment.py::test_evaluation_manifest_rejects_a_reordered_rewrite` | passed; the pre-fix signature has no `refs` |
| Round-wide set | `73cb992..5a32daf` | alignment + startup phases + cli smoke | 444 passed |

### Round 4 — result-level audit

| Fix | Commit | Command | Result |
| --- | --- | --- | --- |
| Finetune handoff bound to the requested path; CLI stops double-passing the payload; runner skips an already-done preflight | `5b75f87` | `tests/training/test_pretrain_alignment.py::test_run_finetune_reloads_a_handoff_bound_to_another_path` | passed; the stale 32 px payload was reused before the fix |
| Cluster drops blank labels, dedupes identical duplicate ids, rejects conflicting ones, skips cohorts below two labeled rows, fills unavailable dashboard columns | `7a5c1e4` | `tests/test_cli_smoke.py::test_cluster_skips_blank_labels_and_one_labeled_row`, `::test_cluster_rejects_conflicting_duplicate_labels_and_dedupes_identical_ones` | passed; both fail on the pre-fix source |
| Diversity rejects negative abundances | `74d620b` | `tests/test_diversity.py::test_otu_table_diversity_rejects_negative_abundance` | passed |
| CAM keeps traversal-order resolution for duplicate names | `a36c5c8` | `tests/test_cam.py::test_selected_image_paths_keeps_the_traversal_order_for_duplicate_names` | passed |
| Queue preparation keeps the row references from its own read; queue and finetune overwrite protection reuse the resolved selection; header-only manifest rejected | `d418469`, this round | `tests/training/test_relational_queue_recipe.py::test_queue_preparation_keeps_the_row_references_it_resolved`, `tests/test_cli_smoke.py::test_queue_overwrite_protection_reuses_the_queue_selection`, `::test_finetune_overwrite_protection_resolves_the_training_manifest_once`, `::test_dataset_ignores_a_carrier_for_different_rows`, `tests/training/test_pretrain_alignment.py::test_evaluation_manifest_rejects_an_empty_csv` | all passed; each fails on the pre-fix source |
| C2 independent full-head gradient reference; D2 independent reimplementation of the old monophyly algorithm; D3 real `cluster → annotate` product comparison | `ae518ed` | the three new nodes | passed |

Round 4 focused sets on the round's commits:

- `tests/test_cli_smoke.py` → 234 passed (before the last two additions), then the
  full file → 238 passed.
- `tests/training/test_pretrain_alignment.py` +
  `tests/training/test_masked_pretrain.py` + `tests/test_training_loss.py` → 453 passed.
- `tests/test_docs.py` + `tests/test_cam.py` + `tests/test_diversity.py` +
  `tests/test_partition.py` + `tests/test_distance.py` +
  `tests/test_cli_annotate.py` + `tests/test_extract_records.py` +
  `tests/test_spatial_extract.py` + `tests/training/test_startup_phases.py` →
  266 passed.

### Round 5 — queue snapshot, duplicate labels, remaining protection branches

| Fix | Commit | Command | Result |
| --- | --- | --- | --- |
| Queue keeps the row references from its own manifest read | `631d30d` | `tests/training/test_relational_queue_recipe.py::test_queue_preparation_keeps_the_row_references_it_resolved` | passed; the pre-fix carrier has no `refs` |
| One deduplicated label table drives both cluster metrics; conflicting duplicates rejected | `e00ea0f` | `tests/test_cli_smoke.py::test_cluster_skips_blank_labels_and_one_labeled_row`, `::test_cluster_rejects_conflicting_duplicate_labels_and_dedupes_identical_ones` | passed; both fail on the pre-fix source |
| Queue and finetune overwrite protection resolve once | `f9ba5e0` | `tests/test_cli_smoke.py::test_queue_overwrite_protection_reuses_the_queue_selection`, `::test_finetune_overwrite_protection_resolves_the_training_manifest_once`, `::test_dataset_ignores_a_carrier_for_different_rows` | passed; all three fail on the pre-fix source |
| Round-wide set | `631d30d..f9ba5e0` | `tests/test_cli_smoke.py`; queue + queue objectives + alignment + pseudo + lineage + provenance; docs + cam + diversity + partition + annotate + distance; queue recipe + startup phases + docs | 238 passed; 360 passed; 177 passed; 108 passed |

### Round 6 — fifth audit (ordering and drift)

| Fix | Commit | Command | Result |
| --- | --- | --- | --- |
| Label table read, blank filtering, dedup and conflict check moved before `prepare_output_dir` | this round | `tests/test_cli_smoke.py::test_cluster_label_conflict_is_rejected_before_overwrite` | passed; the sentinel survives, and the pre-fix source deletes it |
| Duplicate-label dedup proved to reach the partition cohort | this round | `tests/test_cli_smoke.py::test_cluster_deduplicates_identical_rows_before_partition_metrics` | passed; duplicated and deduplicated tables produce identical metric rows |
| A manifest changed after protection is refused instead of re-resolved | this round | `tests/test_cli_smoke.py::test_finetune_refuses_a_manifest_changed_after_protection` | passed; the pre-fix source re-resolves |
| An empty (header-only) CSV is read once, not twice | this round | `tests/test_cli_smoke.py::test_resolve_training_images_reads_an_empty_csv_once` | passed; 2 reads before the fix |
| Round-wide set | this round | `tests/test_cli_smoke.py` (cluster/finetune/pretrain/manifest nodes), `tests/test_output_safety.py` | 154 passed, 63 passed |
