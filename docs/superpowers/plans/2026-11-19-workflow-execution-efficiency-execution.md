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
| 14 | export + spatial export, export cli nodes, output safety | 46, 6, passed |
| 15/16 | full suite `python -m pytest -q -p no:randomly` | 2688 passed, 1 skipped (1336 s) |
| review fix | startup phases + masked + alignment + checkpoint utils + extract records/data cli, targeted cli nodes | 538, 10 passed |
| audit fix (2nd round) | cli smoke + startup phases + cam + docs + masked, alignment | 538, 202 passed |
| audit fix (3rd round) | alignment + startup phases + cli smoke; then alignment+diversity+cam+partition+distance+annotate+masked+loss+startup | 444 passed; 595 passed |
| audit fix (4th round, this record) | see below | recorded per row |

Every new contract was watched fail first: the changed production file was
stashed, the new node was run, and only a real red run was accepted before the
implementation was restored.

## Final acceptance state

- Full regression gate: `python -m pytest -q -p no:randomly` on `a46475a` →
  2688 passed, 1 skipped. Later commits were verified with the focused suites
  listed above plus `tests/test_docs.py`; the master policy's full-suite gate for
  merge/release acceptance is re-run at release time, not here.
- Consumer chains exercised by existing suites: `extract → cluster →
  annotate → diversity` (`tests/test_examples_e2e.py`) and `export → extract`
  (`tests/test_export.py`, `tests/test_spatial_export.py`); CAM prediction,
  array and overlay alignment (`tests/test_cam.py`, cli cam nodes).
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
