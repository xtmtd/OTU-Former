# Training Modularity, CLI Help, and Verification — Design

Date: 2026-11-07
Status: Implemented in this working tree (uncommitted). Acceptance is **not** closed: the 2026-11-08 recovery restored this work after an accidental working-tree reset, and the two explicitly deferred batches (training-test device migration to `auto`, and the shared serialized-fixture consolidation of the 19 corrupt-resume cases) are still outstanding. The checklist below records the original execution; see "Recovery status" at the end of this document for what the current candidate actually contains.
Baseline: branch `OTU-Former-dense`, commit `84f845e`; working tree clean before this design file.
Authorization: the operator approved the written design and authorized implementation through the in-document checklist in §10 (no separate execution-plan file). Commit, push, tag, release, and version changes remain unauthorized and unperformed.

Development-document language: English only for new local design specs, execution plans, and developer guides; no Chinese counterpart is required. This operator-directed exception applies to local development documentation, not a translation or deletion of existing user-facing README/command documents.

## 1. Intent and agreed direction

Reduce the cost of extending training, finding CLI options, and validating small changes without weakening experiment recovery or filesystem safety. The operator selected responsibility-based modularization, a shared pretrain KoLeo regularizer for relational and DINO, complete task-grouped help with individually reviewed shortening, and impact-based verification with full regression at meaningful gates.

The operator reports a pause after `Relational target config:` and before training. Startup visibility and measurement are part of this work; a speedup is not an acceptance promise. Refactoring and the KoLeo extension belong to the same project but have separate behavior-preservation and new-behavior gates.

### Goals

- Give pretrain, finetune, recipe resolution, recovery, objectives, and reporting clear owners.
- Make genuinely shared behavior reusable without forcing two training algorithms into one universal loop.
- Enable opt-in relational + KoLeo while preserving zero-weight defaults and supported historical checkpoints.
- Keep all existing CLI options discoverable in a single `--help`, organized by user task.
- Establish small, reusable architecture/help/testing rules in the master design during implementation.

### Non-goals and anti-goals

No general Engine/callback/plugin framework, new configuration-file system, new dependency, new SSL method, finetune KoLeo, command nesting, flag renaming, default retuning, automatic hyperparameter selection, automatic change-based test selector, parallel-test rollout, broad downstream refactor, checkpoint conversion tool, or claimed model-quality improvement. Do not remove recovery checks, real integration coverage, or safety warnings to make code or tests shorter. Do not rewrite existing datasets, checkpoints, or run logs.

## 2. Evidence and limits

Read-only inspection at the baseline found 8,117 lines in `src/otuformer/training/trainer.py`, with `run_pretrain` at lines 5033–5946 and `run_finetune` at 7308–8117. Model, dataset, loss, and DINO objective modules already exist. CLI imports private trainer helpers, and tests patch trainer-owned symbols; moving definitions alone changes those patch boundaries.

AST inspection found 1,224 `test_` functions before parametrization in 46 test-bearing files out of 47 Python files under `tests/`; the remaining file is `tests/embedding/__init__.py`. This is not a collected test count or proof of excessive coverage. `test_cli_smoke.py` mixes help, validation, models, and training integration. Current pytest configuration has no custom tier markers or documented selection policy. No test suite was run for this design.

Existing local logs on 2026-11-05 show the following timestamp differences:

| Run directory | Commit recorded in log header (line 1) | Explicit `SSL config.ssl_method` | Config to auto-eval-size log | Auto-eval-size to initial cosine | Config to starting-LR log |
| --- | --- | --- | ---: | ---: | ---: |
| `runs/pretrain` | `5df8d3e` | `relational` (line 80) | 23 s | 3 s | 26 s |
| `runs/pretrainV2` | `c28aca9` | Not recorded | 1 s | 4 s | 5 s |
| `runs/pretrain_dino` | `ae6bde5-dirty` | `dino` (line 80) | 40 s | 2 s | 42 s |
| `runs/pretrainV2_dino` | `04c4625` | `dino` (line 80) | 20 s | 4 s | 24 s |

Each source is `<run directory>/logs/pretrain.log`. The commit strings are recorded provenance, not a reconstruction or verification of those old working trees. All four logs contain the literal `Relational target config:` label, including the two explicit DINO runs. Neither that label nor a directory suffix determines the active method; the table uses the explicit SSL config where available and leaves the missing record unknown.

These are different old runs on MPS, not a controlled comparison or a current-HEAD benchmark. After the config record, current code resolves dataset paths, builds a loader, constructs student and teacher, prepares state/optimizer/schedules, and finally reads a first batch and runs initial forwards. Model construction can use pretrained-weight loading; filesystem indexing, construction, device initialization, and first-batch work are candidates, not diagnosed bottlenecks.

Current CLI declares 53 pretrain and 43 finetune options, with no `rich_help_panel`. Current KoLeo is DINO-only in validation, config, the loop, checkpoint identity checks, and metric semantics—not merely in module location.

## 3. Approaches and selection

| Approach | Trade-off | Decision |
| --- | --- | --- |
| Split only by command | Small organizational change, but leaves recovery, metrics, and feature branching mixed inside two large files. | Not selected. |
| Responsibility-based modules with separate command loops | Extract existing responsibilities and shared features while retaining explicit algorithm-specific behavior. | Selected by operator. |
| Universal training engine, callbacks, and plugins | More abstractions and migration risk without a demonstrated requirement for external extension. | Not selected. |

For help, select native task grouping with all options visible. Do not introduce basic/full help modes or `pretrain relational` / `pretrain dino` subcommands. Panel placement follows user decisions, not physical Python filenames.

## 4. Ownership and dependency boundaries

The following are responsibility boundaries, not a requirement to create a file for every row or helper. Before moving code, record the concrete file/owner grouping and import/patch migration map under the execution checklist in §10; choose the smallest acyclic grouping without creating a separate plan. Existing `model.py`, `dataset.py`, `loss.py`, and `dino.py` retain their roles.

| Owner | Responsibility | Must not own |
| --- | --- | --- |
| CLI | Parse options, preserve explicitness/requested values, call public preflight/resolution and run entry points, maintain logging/output safety orchestration. | Tensor losses, optimizer recovery, or a second recipe implementation. |
| Pretrain orchestration | Build resolved run objects, run the pretrain step/epoch sequence, coordinate saving and reporting. | Finetune or pseudo-label behavior. |
| Finetune orchestration | Build finetune state, run supervised losses and existing long-tail/diagnostic behavior. | SSL target construction or DINO recovery. |
| Recipe and recovery owners | Resolve requested/effective/inherited settings and validate checkpoint/state compatibility. Keep method-specific recovery checks explicit. | CLI imports, destructive output setup, or a second configuration truth. |
| CLS method owners | Relational view pairing/center rules; existing DINO prototype/assignment/center behavior. | Patch masking, common output setup, or finetune losses. |
| Shared pretrain components | Existing patch-objective/mask mechanics and KoLeo configuration/calculation/diagnostics, each with its own applicability. | Automatically enabling one feature because another is selected. |
| Reporting and evaluation | CSV schemas/migrations, diagnostic output, plots, periodic embedding evaluation. | Selecting training methods or changing recovery identity. |
| Finetune provenance | Existing pseudo-source identity, manifests, hashes, and round preparation. | General training-loop mechanics. |
| Small runtime utilities | Existing seed/RNG/device primitives that are actually shared. | An unbounded `common.py` dumping ground. |

Dependencies flow from CLI to command orchestration to focused components. Recovery/read-only identity helpers may depend on shared definitions, never back on CLI or orchestrators. Promote truly cross-command identity constants to their existing lightweight owner or `constants.py`, not a trainer import from unrelated downstream consumers. Keep schema/help discovery free of new torch/model imports.

Preserve `otuformer.training.trainer.run_pretrain` and `run_finetune` for import-and-call compatibility as direct aliases to their new orchestration owners (`pretrain_loop`, `finetune_loop`). The alias module exports only those entry points plus `_freeze_backbone_blocks`; it does not re-export the moved private helpers. This preserves ordinary imports/calls, not monkeypatch compatibility at the old module path. Internal CLI callers import the actual run owner and explicit public resolution/preflight entry points; repository tests must patch the lookup used by each caller after the move. Neither an alias nor a wrapper at the old path intercepts a CLI call already bound to a different owner. Update string patch targets and `monkeypatch.setattr` together and assert the intended stub was called; no un-intercepted training may run as a fallback in CLI-only tests. Do not re-export every private symbol or use dynamic forwarding to preserve obsolete patch locations. This does not establish a new supported public Python API for private helpers.

Include source-text dependencies in the caller/import/patch map. In particular, `tests/training/test_pretrain_alignment.py::test_teacher_momentum_updates_after_optimizer_step` (baseline lines 501–511) inspects `trainer.run_pretrain` source text. Replace it with a real tiny-loop event-order assertion that optimizer step precedes encoder EMA and, where applicable, prototype/head EMA; make the check fail when that order is reversed. Preserve method-specific center timing separately: relational's feature-center update currently precedes the optimizer step, while DINO objective and iBOT patch-center updates follow it. Do not move all center updates after the step in the name of a generic test. Do not leave it inspecting a compatibility alias/wrapper or weaken it to checking that functions exist. Also audit source-inspection tests for `_compute_all_metrics` and `_compute_embeddings_from_csv` (baseline lines 245 and 281) when moving reporting; point any retained inspection at the actual owner, preferring behavioral checks where practical.

Pure refactoring preserves optimizer parameter membership/order, loss aggregation, forward count and view provenance, clipping, prototype freeze, EMA/center update order, schedules, save/evaluation/RNG boundaries, checkpoint keys, requested-versus-effective logging, pseudo provenance, and output safety. Existing early validation must still precede directory preparation, overwrite, logger creation, and training; direct trainer entry points keep equivalent preflight.

## 5. Shared pretrain KoLeo contract

### Calculation and applicability

KoLeo becomes an independent opt-in regularizer for both relational and DINO pretraining, not a finetune feature or a new CLS method. Default `--koleo-weight 0` remains disabled and performs no neighbor search. Reject booleans, negative weights, and nonfinite weights at preflight.

Reuse the existing `koleo_loss` implementation: float32 protected normalization/distance, nearest other sample within each unmasked student global crop, self excluded, two per-crop losses summed. Reuse existing student tokens; do not add forwards, use masked CLS, pool global crops, include local crops or teacher weak views, deduplicate records, filter labels, or add a queue. For a crop with fewer than two samples, preserve a zero numerical contribution and explicit skip/eligible counts. The standalone `koleo_loss` returns a differentiable zero, but the existing crop aggregator skips that call and returns an ordinary scalar zero when all crops are ineligible; do not claim that the training aggregator already preserves a gradient-connected zero. Preserve signed raw/weighted losses and the duplicate/near-duplicate warning.

For both methods, total loss adds `effective_koleo_weight * koleo_loss_raw` to the existing CLS/local/patch sum. Existing joint gradient validation/clipping includes this contribution; method-specific state updates remain unchanged.

### Config, identity, and resume

- Move `koleo_weight` out of `dino.py`'s `DINO_DEFAULTS` and the three epsilon keys out of `DINO_FIXED_CONFIG` into one shared pretrain owner; do not retain duplicate values or a second configuration truth. Both methods consume that owner. When reconstructing the DINO required-recipe/fixed-value checks, explicitly include the moved common keys so current strict DINO resume validation remains intact. DINO prototypes/centering remain DINO-only.
- Operator decision after audit: relational weight zero omits all four KoLeo keys from saved `config`, preserving the pre-extension metadata shape and old-reader compatibility with respect to these keys. Positive relational weight writes `koleo_weight`, `koleo_normalize_eps`, `koleo_distance_eps`, and `koleo_log_eps` together. DINO continues to record all four keys even at weight zero, with its existing names/values. No new top-level training state is needed.
- For relational resume, all four keys absent means disabled KoLeo with the fixed formula constants, for both historical and newly written zero-weight runs; absence cannot establish the checkpoint's age and must not trigger a misleading historical-only warning. All four present must describe the enabled relational recipe: finite positive nonboolean weight and supported finite positive nonboolean epsilon constants. Any proper nonempty subset is invalid. Do not infer the recipe from invocation args or mutate the loaded payload. Relational's canonical zero-weight encoding is absence, whereas DINO's existing recorded-zero encoding remains valid.
- Split `_ssl_display_config`'s current `dino_fields` list: show effective KoLeo weight and epsilon values from the common owner for both methods, including zero weight, while only genuine DINO fields become null under relational. Display is not persistence: do not spread this display dictionary into checkpoint config; relational zero-weight storage still omits the four keys. Keep `_ssl_display_config` as the single resolved SSL display projection for the `SSL config:` record (baseline production caller: `trainer.py:5110`), separate from requested-parameter replay, checkpoint serialization, and method inheritance. For relational, that record's `koleo_weight` changes from null to numeric 0.0 when disabled; the shared epsilon fields also show their effective constants instead of inactive nulls. This display delta is separate from the CSV delta below; do not rewrite old logs.
- Omitted weight on resume inherits the recorded value, or relational zero when the bundle is absent. Explicit equal values are allowed; conflicts fail before output mutation. Any absent-bundle relational resume, historical or new, cannot enable KoLeo; start a new run. Its user-facing error must identify `--koleo-weight`, explain that KoLeo was disabled for this checkpoint, and direct the user to start a new run to enable it, consistent with A3. Do not describe this valid all-absent case as an incomplete bundle or expose bundle terminology as the recovery instruction; partial/corrupt metadata remains a distinct validation failure. Preserve omitted `--ssl-method` inheritance and reject method conflicts independently of display formatting. Existing DINO strict recipe validation is not weakened to excuse missing DINO state.
- `utils/checkpoint.py` currently classifies KoLeo weight/epsilon keys as DINO-only metadata. Update identity/contradiction rules, trainer preflight, CLI restrictions, schema conditional inheritance, and tests together. Common KoLeo keys alone do not establish DINO identity. Relational plus DINO prototypes, centering, or `dino_objective` remains contradictory. Metric/legacy payloads do not acquire active SSL identity from common regularizer fields. Finetune output keeps only its existing SSL source provenance; it must not start persisting active KoLeo/SSL recipe or DINO training state merely because helpers are shared.
- Downstream readers validate the existing model/head identity, not the presence of resumable training state. Test new relational checkpoints through extract/export/CAM and finetune initialization; preserve existing read-only versus training-resume distinctions. The compatibility promise is bounded: zero-weight relational checkpoints add none of the four previously DINO-only keys, so this extension must not make those checkpoints unreadable by baseline readers. Positive-weight relational checkpoints do add these keys and are rejected by baseline `84f845e` identity checks; they require the updated reader. New code still reads supported historical artifacts. This does not guarantee compatibility with every older release or unrelated future format changes. Preserve original checkpoint files for rollback; do not overwrite them through a migration.

### Metrics and historical data

Reuse the current CSV columns; do not invent a second regularizer CSV. New relational rows record numeric `koleo_weight` (including zero), and reuse DINO's exact existing statuses `active`, `disabled`, and `skipped`. With zero weight, raw/weighted losses remain blank (N/A), not measured zero, and counts remain unrecorded as they do now. An enabled but ineligible crop/batch records skipped counts and numeric zero contribution; active rows record signed raw/weighted losses and eligible/skipped counts. DINO centering/prototype fields remain N/A under relational. This is an intentional reporting change even for default zero-weight relational runs: older code writes blank weight and `not-applicable`; new rows write numeric 0 and `disabled`. It does not change training numerics or add checkpoint keys at zero weight. Do not describe default compatibility as byte-identical CSV output.

Existing historical rows, including relational rows with blank weight and `not-applicable`, remain unchanged on append/resume: they describe the old implementation, not a new observed zero. Do not backfill method/weight/counts or rewrite completed runs. Retain recognized header migrations, atomic failure preservation, and missing-value plotting behavior.

## 6. Startup visibility and performance boundary

Add concise phase start/completion records with elapsed wall time around meaningful startup boundaries: dataset/path resolution, student construction/device placement, teacher construction/device placement, objective/optimizer/recovery preparation, and first-batch/initial-forward readiness. Include earlier preflight/model probes when measurement shows they matter; their validation remains before output mutation. Use existing logging, with a small shared timing helper only if actually reused. No new CLI debug/timing flag is required.

A phase start must appear before its potentially long operation so silence is not confused with a hang. Preserve `Parameters:`, resolved config records, existing command replay, and machine-consumed records; timing is display-only. Do not log sensitive tensor/data payloads, reread image trees just to count progress, consume extra samples, reseed, or perform extra forwards for timing.

Measure wall-clock startup to readiness separately from steady-state step throughput and epoch-end evaluation. Device work can be asynchronous: do not call unsynchronized wall intervals kernel execution time; use explicit diagnostic synchronization only in an approved benchmark, not on every production step. Record device, model, crop/batch/workers, checkpoint/source, and warm/cold cache context. No network downloads or real-data runs are authorized by this design draft.

Optimize only after a measured baseline identifies cost. Possible follow-up changes to redundant initialization, data loading, or cached work must preserve initialization weights, RNG, register handling, and recovery. Non-equivalent optimization requires a separate design amendment; splitting files alone is not a performance result.

## 7. Complete task-grouped help

Use Typer's existing option-panel facility, not a custom renderer. Preserve command syntax, option names/aliases, types, defaults, explicitness tracking, and all 96 user options. The only applicability change is approved shared pretrain KoLeo. Keep the existing safety order `--out-dir` before `--resume` before `--overwrite`; output/initialization and recovery panels appear in that order. The built-in help control remains available. Exactly-once coverage refers to option declaration/help entries, not mentions of another flag inside explanatory prose.

Native panel order follows the first option declaration assigned to each panel; within-panel order also follows declarations. Adding `rich_help_panel` alone does not produce the table order. The implementation scope therefore explicitly includes reordering the pretrain/finetune callback parameters to match these panel/option inventories while retaining each option's name, type, default, requiredness, explicitness, and keyword bindings. Audit direct callback/signature consumers, including positional callers, before reordering; update internal callers/tests intentionally rather than silently changing their argument binding. Baseline `tests/test_docs.py::cli_option_help` and `cli_flags` (callback/signature anchors at lines 183/185 and 204/210) build name/flag-keyed maps or sets; `tests/test_cli_smoke.py::_finetune_param_names` (line 4152) returns a name set. These checks do not depend on declaration order and need confirmation, not automatic rewrites. The cited `test_cli_smoke.py:608` consumer is the update callback and is outside the pretrain/finetune reorder. Check for positional callback callers; separately retain panel-order, schema/card, binding, and parsing checks. This is a presentation-driven callback reorder, not authorization to change the preserved training run-entry signatures. Keep `--out-dir` before `--resume` before `--overwrite` in the rendered help. A local in-memory probe with Typer 0.25.1 confirmed first-panel-appearance ordering; the complete proposed layout and declared `typer>=0.12` floor remain unverified until the implementation help checks.

### Pretrain panel inventory

| Panel | Options (exactly once) |
| --- | --- |
| Input and output | `--train-data`, `--input-images-dir`, `--out-dir` |
| Model | `--model-name`, `--register-tokens`, `--out-dim` |
| CLS objective | `--ssl-method` |
| Relational targets | `--relational-teacher-views`, `--relational-center-source`, `--relational-pairing` |
| DINO targets | `--dino-prototypes`, `--dino-centering` |
| Shared regularization | `--koleo-weight` |
| Patch objective | `--patch-loss`, `--masking-strategy`, `--mask-ratio`, `--ibot-prototypes`, `--lambda-mask` |
| Views and augmentation | `--augmentation`, `--orientation-policy`, `--global-crop-size`, `--local-crop-size`, `--local-crops`, `--lambda-local` |
| Optimization and teacher | `--max-epochs`, `--lr`, `--weight-decay`, `--no-norm-bias-weight-decay`, `--warmup-epochs`, `--teacher-momentum`, `--teacher-momentum-end`, `--student-temp`, `--teacher-temp-start`, `--teacher-temp-end`, `--teacher-temp-warmup-ratio` |
| Evaluation and reporting | `--log-every-n-steps`, `--save-every-epochs`, `--keep-last-checkpoints`, `--visualize-data`, `--extract-size`, `--metrics-sample-size`, `--umap-n-neighbors`, `--umap-min-dist`, `--umap-metric`, `--visualize-class-number`, `--disable-embedding-metrics` |
| Runtime and reproducibility | `--batch-size`, `--num-workers`, `--cpus`, `--device`, `--seed` |
| Recovery and safety | `--resume`, `--overwrite` |

### Finetune panel inventory

| Panel | Options (exactly once) |
| --- | --- |
| Input, output, and initialization | `--checkpoint`, `--train-data`, `--input-images-dir`, `--out-dir` |
| Model | `--model-name`, `--metric-embed-dim` |
| Metric objective | `--loss`, `--subcenters`, `--compact-weight`, `--supcon-temperature`, `--long-tail` |
| Pseudo-label round | `--pseudo-label-from`, `--pseudo-similarity-floor`, `--pseudo-min-gap`, `--pseudo-neighbors`, `--pseudo-cap-multiplier`, `--pseudo-absolute-cap`, `--trace-batch-ids` |
| Views and augmentation | `--augmentation`, `--orientation-policy` |
| Optimization | `--finetune-epochs`, `--finetune-lr`, `--metric-head-lr`, `--weight-decay`, `--freeze-ratio` |
| Evaluation and reporting | `--log-every-n-steps`, `--save-every-epochs`, `--keep-last-checkpoints`, `--visualize-data`, `--extract-size`, `--metrics-sample-size`, `--umap-n-neighbors`, `--umap-min-dist`, `--umap-metric`, `--visualize-class-number`, `--disable-embedding-metrics` |
| Runtime and reproducibility | `--batch-size`, `--num-workers`, `--cpus`, `--device`, `--seed` |
| Recovery and safety | `--resume`, `--overwrite` |

### Specific wording changes proposed for approval

Appendix A lists current and proposed option help. These six changes are the initial wording-change scope; all other per-option wording stays unchanged unless another explicit before/after proposal is approved. Do not mechanically impose one sentence per flag. Preserve safety, effective defaults, method applicability, and recovery caveats at the point of use. Short help links the stable command-document URLs; full explanations remain in bilingual command documents.

Update affected text-pinned assertions in the same help-change stage, not after an unrelated panel-rendering failure. With the proposed A1/A2 text, `test_pretrain_help_distinguishes_cls_and_patch_objectives` has three direct mismatches at baseline `tests/test_cli_smoke.py:1930,1932,1935`: capitalized `Not a complete training recipe` versus the lowercase assertion, and the two `selected independently` phrases. Keep checks for both global/local scope, incomplete-recipe caveat, independent CLS/patch selection, shared encoder/temperature, defaults and method applicability, but target the approved wording or normalized semantics. The wider `test_pretrain_help_documents_patch_modes_and_mask_migration` assertion at baseline line 1904 (`"selected" in normalized`) still passes through unrelated `--mask-ratio` help; it is not proof of correct CLS/patch wording. Strengthen it to the relevant option's semantics while preserving its masking/default/migration and experimental-warning checks. Appendix A names baseline test anchors; those line numbers are navigation hints, not future assertions.

The runtime option declarations remain the help/parameter source of truth. Keep the existing exported JSON schema version/shape; update existing restriction and conditional-inheritance values for KoLeo without adding a grouping field.

Consumer-visible change in that exported schema, stated so a reader does not have to diff it: on a resume, an omitted `koleo_weight` is now labelled `provenance.koleo_weight = omitted-inherited` instead of `omitted-default`, because the weight is a shared pretrain regularizer rather than a DINO-conditional one. A fresh run keeps `0.0` with `omitted-default`. No field is added, removed or renamed, and no compatibility layer is provided; a consumer that keyed on the old label should read the new one. Parameter cards continue to show every parameter exactly once and use the updated schema help/constraints. Grouped cards are not required: if they need new exported metadata, defer them rather than create a second hardcoded grouping table. Preserve argv serialization and omission semantics.

Validate native help at ordinary narrow and wide terminal widths, without ANSI/box-character golden snapshots. Assert complete coverage, panel ownership/order, warnings and stable links; retain real parsing/validation tests. CLI/help/schema must remain read-only and not newly import training-heavy dependencies.

## 8. Impact-based verification policy

Implementation, review, and acceptance follow the active
[Development Testing Policy](2026-11-07-development-testing-policy.md), required
by [master design §8.5](2026-03-29-otuformer-design.md#8-engineering-rules). It is
the single detailed source for impact-based selection, full-regression triggers,
training-test devices and CPU exceptions, test-helper boundaries, and evidence
requirements. The matrix and common rules live there rather than in a second
copy in this feature design.

For this work, record the actual file/node selections and results in §10. The
modularization, KoLeo, help/parser, and downstream-checkpoint checks in that
checklist remain feature-specific acceptance requirements; the policy does not
replace or waive them. Report unrun devices and environment prerequisites
separately, and claim no wall-clock budget or speedup before measurement.

## 9. Proposed master-design rules and documentation ownership

During implementation, add concise normative rules to the [master design](2026-03-29-otuformer-design.md), not an exhaustive file map:

1. **Ownership:** CLI parses and orchestrates; core modules own behavior. Depend on explicit entry points and lightweight shared definitions; do not import orchestration from downstream consumers.
2. **Reuse:** share demonstrated behavior, not hypothetical extension points. Separate method-specific state from reusable objectives and reporting; avoid giant generic trainers and dumping-ground common modules.
3. **Configuration:** keep requested, effective, inherited, and inactive settings distinct. One authoritative resolution per invocation; validate before output mutation. New flags require a real user decision, documented applicability, and explicit recovery semantics.
4. **Help:** expose all options, group by user task, preserve safety/recovery warnings, and approve specific compressions. CLI owns executable declarations, commands own full prose, README owns navigation; schema/cards derive from runtime truth.
5. **Verification:** require the [Development Testing Policy](2026-11-07-development-testing-policy.md) in the master design as the single detailed source for test selection, regression gates, device boundaries, and evidence. Keep feature-specific checks and results in §10 rather than copying the policy matrix.
6. **Performance:** establish stage evidence before optimization, preserve reproducibility and safety, and distinguish startup, step, and evaluation costs.

Update the existing documentation-conventions design with a bounded amendment for grouped help and the approved wording changes; do not rewrite its history or erase the prior no-blanket-shortening rule. Amend DINO/relational design ownership of KoLeo and link this decision rather than duplicating all flags. Update master-design status only when implementation evidence exists.

Implementation documentation surfaces: `README.md`/`README.cn.md`; `docs/commands/pretrain*.md`, `finetune*.md`, and relevant shared training documents; affected design records; the execution checklist and evidence in §10 (no separate plan); `skills/otuformer-workflow/SKILL.md`, its relevant references and adapter tests; CLI schema/restrictions/cards and documentation/skill regressions. The workflow Skill keeps runtime-schema discovery, complete cards and per-step user approval; it must not invent options or claim architecture changes already exist. The active [Development Testing Policy](2026-11-07-development-testing-policy.md) belongs in one dated English-language specification under `docs/superpowers/specs/`; the master design must require and link it, and §10 must record its path. Other documents use short links rather than copied matrices. New local development specs, plans, and guides are English-only as directed by the operator; record this bounded exception in the master design and documentation-conventions amendment during implementation. Preserve existing user-facing README/command language conventions unless separately changed.

This draft changes none of those existing files. No blanket update of unrelated command docs is required.

## 10. Acceptance and in-document execution checklist

Operator decision: use this section as the single execution checklist; do not create a separate plan file or duplicate the design into a second task specification. Approval of the complete written design, including Appendix A, and explicit implementation authorization are still required before execution; commit approval remains separate. All five stages below were implemented and their evidence recorded at the time; the boxes record that execution, not a current acceptance verdict — see "Recovery status".

Before code moves, record the concrete target files and their responsibilities, caller/import/patch/source-inspection/signature-consumer migrations, and corresponding focused tests here. This is a compact implementation map, not another design document or a transcript of function bodies. Choose filenames within §4's boundaries; return for approval if a choice changes the agreed behavior or scope. For each stage, append the actual changed-file scope, verification commands/results and remaining limits here, linking larger evidence artifacts rather than copying logs. Check a stage only when its deliverable and required verification are complete.

- [x] **Startup visibility and baseline:** establish necessary baseline evidence and the implementation map above; add startup phase records without changing training mechanics. Verify relevant log/preflight/RNG behavior with focused tests, and keep timing overhead and evidence limits explicit.
- [x] **Behavior-preserving modularization:** extract responsibilities in coherent, reviewable blocks. Verify relational and DINO plus finetune/pseudo behavior, real gradients/update order, supported patch modes, checkpoint readers, import/patch ownership, safety and RNG recovery before declaring the refactor equivalent. Apply §8's full-regression trigger at the coherent refactor gate, not after every helper move.
- [x] **Shared relational KoLeo:** add formula/gradient checks; verify weight-zero numerical parity and omission of all four saved keys, positive-weight four-key persistence, unchanged DINO zero-weight persistence, effective/display/CSV-versus-storage distinctions, enabled and B<2 behavior, signed metrics, all existing patch modes, inherited/conflicting/absent-bundle resumes, malformed metadata, historical CSV preservation and downstream compatibility. Add a focused baseline-reader contract check that zero-weight relational is not rejected for KoLeo metadata while enabled relational is incompatible with the baseline; do not claim old-reader integration passed from schema shape alone.
- [x] **Help and documentation synchronization:** apply callback declaration reordering, grouped help, approved wording and the linked help-assertion updates together; update affected master/topic designs, README/commands, schema/Skill and test-selection guidance. Verify full rendered panel order and all 53/43 user option entries exactly once with unchanged names/defaults/bindings and no new heavy schema imports. Include parser behavior, direct callback consumers, real-owner patch interception, and no accidental training in CLI-only tests.
- [x] **Final regression and handoff:** finish affected contract gates and one final full-suite candidate run with timing. Report unrun MPS/CUDA, real-data performance, package installation, or unavailable dependency-floor checks as unverified, not passed. Record evidence and unresolved blockers; no release/version/commit follows automatically.

### Implementation map and recorded evidence

Two orchestration modules own the command loops; the remaining extraction is
grouped by responsibility. `trainer.py` is now a 19-line compatibility surface
exporting only `run_pretrain`, `run_finetune` and `_freeze_backbone_blocks`.

| Module | Responsibility | Lines |
| --- | --- | ---: |
| `runtime.py` | seeding, device selection, student/teacher EMA, startup phase records | 95 |
| `augmentation.py` | augmentation profiles, register plans, their validation | 512 |
| `patch_objective.py` | masking strategies and resolved patch configuration | 470 |
| `objectives.py` | pretraining loss terms and per-step diagnostics | 249 |
| `reporting.py` | CSV schemas, loggers, plots, periodic evaluation | 1234 |
| `pretrain_recipe.py` | pretrain settings, SSL recipe, recovery, preflight, shared KoLeo owner | 2708 |
| `finetune_support.py` | finetune losses, provenance, pseudo rounds, optimizer layout | 1518 |
| `pretrain_loop.py` | pretrain step and epoch sequence | 1074 |
| `finetune_loop.py` | supervised finetune step sequence | 926 |
| `trainer.py` | import-and-call compatibility only | 19 |

`trainer.py` went from 8,117 to 19 lines. Every moved symbol was verified
byte-identical to its baseline source, with no duplicate definitions and no
lost or invented top-level names. No module under `src/` imports from
`trainer` any more; the CLI imports the run entry points from their owners.

**Stage 1.** `[Phase] <name>: start` / `done in N.Ns` records wrap dataset
indexing and loader, student construction, teacher construction,
objective/optimizer/recovery, and first-batch/initial forwards in pretrain; and
model construction, dataset/loader, loss/optimizer/recovery in finetune. Each
start precedes the work it names, including the dataset constructor (path and
CSV resolution) and the objective/loss construction. Wall clock only, no device
synchronization, no extra samples or forwards.

**Stage 2.** Patch ownership followed the call site, not the definition: a
`monkeypatch.setattr` only intercepts a global lookup in the calling module.
Names the loops read keep the loop-module target; helpers called only inside a
moved module follow it. The CLI imports each run entry point from its owner, so
the interceptable binding is the real one rather than an alias.

**Stage 3.** `koleo_weight` and the three epsilons moved out of `dino.py` into
`pretrain_recipe.py` (`KOLEO_STORED_KEYS`, `KOLEO_FIXED_CONFIG`), with no second
copy; DINO's strict required-field check re-includes them explicitly. Relational
zero weight omits all four from the saved `config`; a positive weight records
all four. Presence is decided by key membership, not by value, so an explicit
`null` is damaged metadata rather than the "absent" encoding, and a complete
relational bundle must carry a strictly positive weight plus supported
epsilons. An absent-bundle resume cannot enable KoLeo even through a direct
entry point that omits the explicitness marker. Display always shows the four
values. `checkpoint.py` no longer treats these keys as DINO-only metadata. CSV
records numeric `koleo_weight` and the existing `active`/`disabled`/`skipped`
statuses for both methods.

**Stage 4.** All 53 pretrain and 43 finetune options carry `rich_help_panel` in
the approved inventory order, which also required reordering the callback
parameters because Typer renders panels in first-declaration order. Rendered
order verified against the inventory; `--out-dir` still precedes `--resume`
before `--overwrite`. A1–A6 applied verbatim; the three predicted
`test_cli_smoke.py` mismatches were updated to the approved wording and the
broad `selected` check was tightened. Help assertions are ANSI-independent: an
autouse fixture pins colour off and a `flat_help` helper strips escapes before
borders, so results no longer depend on `FORCE_COLOR`/`NO_COLOR`/`TERM`. The
exactly-once check compares every declared option against the rendered option
column. English command documents and their `.cn.md` pairs were updated
together, README and the workflow skill were synchronized, master design §8 now
links the testing policy, and the DINO/KoLeo topic design carries a dated
supersession note instead of being rewritten.

**Stage 5.** Final candidate: `python -m pytest` → `1750 passed, 1 skipped`,
exit 0. Environment: Python 3.11.15, torch 2.12.1, timm 1.0.27, typer 0.25.1,
click 8.4.2, rich 15.0.0; source import path `src/` via `PYTHONPATH=src`;
baseline commit `84f845e` with the working tree dirty. Slowest entries are
`test_dino_resume.py` resume-equivalence runs (8–12 s each), which is real
tiny-loop train/compare work, not collection cost.

Two source-text tests that duplicated coverage were removed rather than
rewritten: `tests/embedding/test_umap_rendering.py` asserted string positions
inside `run_umap`, and `tests/test_evaluator.py::test_run_umap_preserves_rendered_labels_title_and_neighbor_limit`
already checks the rendered labels, caller-supplied title and neighbour limit
behaviourally. `test_pretrain_alignment.py::test_teacher_momentum_updates_after_optimizer_step`
was replaced by a real event-sequence assertion.

Behavioural checks were mutation-tested rather than trusted: removing the KoLeo
term from `total_loss` fails the gradient test, moving the encoder EMA before
`optimizer.step()` fails the order test, and moving the dataset phase start
after `MultiCropDataset` construction fails the startup test. The previous
source-text ordering check in `test_pretrain_alignment.py` was replaced by a
real event-sequence assertion.

The MPS CPU-fallback switch is armed by the package import
(`otuformer/__init__.py`), the earliest point the library controls; command
functions and training loops no longer set it, because a late `setdefault` is
not a guarantee. The package records a three-way state instead of a boolean,
because "the variable is in the environment" is not proof the MPS backend
latched it: `MPSFallbackState.ARMED` (this import set it before PyTorch
started), `DISABLED` (an explicit operator opt-out) and `UNKNOWN` (PyTorch was
already imported, so the operator may or may not have set it beforehand).
`MPS_FALLBACK_EFFECTIVE` stays `True` only for `ARMED`; the run entry points
warn only for `DISABLED` and, for `UNKNOWN`, say the switch cannot be confirmed
rather than accusing the operator of not setting it. The regression test runs
the actual missing kernel — the antialiased bicubic resample backward,
`_upsample_bicubic2d_aa_backward`, not the plain bicubic variant, which is
implemented on MPS — under both import orders, plus the documented operator
case (variable preset, `torch` imported first): the kernel succeeds and the
state is `UNKNOWN`, not "missing".

Both preflight handoffs are one-shot inputs **taken at the start of the public
entry**, before any check that can raise, and cleared in the same step: a
rejected call cannot leave an answer behind for a retry with a different
request, and every resolution uses this call's local copy. The SSL config and
the resume payload are the same shape. The resume payload is additionally bound
to the `--resume` path it came from *and* to that one invocation, so a replaced
checkpoint at the same path is read again instead of serving the previous
file's model, epoch and optimizer state. Covered by focused regressions —
including "first call rejected, request changed, retry" and "same path, newer
file" — and confirmed by mutation (leaving a slot behind, or caching the
payload, fails them).

Unverified and not claimed as passed: MPS/CUDA equivalence beyond the automated
falls back path these tests exercise, real-data performance, any startup speedup
(instrumentation only, no baseline comparison), package installation/upgrade,
and the declared `typer>=0.12` floor (only 0.25.1 was exercised). No full-suite
run was performed against HEAD without these changes, so the parity claim rests
on the identical test outcomes and the verified byte-identical move. No commit,
push, tag, release, or version change was made.

In a move-only stage, default numerical behavior and serialized contracts must match the baseline; tests of shape alone are insufficient. Where exact device-level reproducibility is unavailable, use deterministic CPU checks and report that limit without claiming hardware parity. Add relational KoLeo as the intentional semantic delta, not as an unexplained difference in the refactor gate.

### Follow-up: suite-wide test simplification (2026-11-07)

The operator approved the consolidation recommendations and requested broader
review coverage. The earlier review covered training and related CLI tests; it
was not a review of every project test. This follow-up inventoried all 49
original test-bearing files (1,255 test functions, before parametrization),
including their assertion/exception contexts, parameter matrices, fixtures,
heavy setup helpers and source-inspection checks. Suspected duplicates were
compared with the behavior they protect. This is suite-wide structural and
contract screening with deeper investigation of concrete candidates, not an
independent mutation audit of every assertion or every production path.

| Review area | Original files screened under `tests/` |
| --- | --- |
| CLI, schema, documentation and startup | `test_cli_schema.py`, `test_cli_smoke.py`, `test_cli_startup.py`, `test_docs.py`, `test_workflow_skill.py`, `test_version.py` |
| Input, annotation and provenance | `test_annotate.py`, `test_cli_annotate.py`, `test_data_coco.py`, `test_data_masks.py`, `test_data_provenance.py`, `test_data_reader.py` |
| Filesystem and cleanup | `test_cli_cleanup_review.py`, `test_extract_cleanup_preflight.py`, `test_output_safety.py`, `test_utils.py` |
| Extraction, export, CAM and workflow | `test_cam.py`, `test_export.py`, `test_extract_data_cli.py`, `test_extract_records.py`, `test_extractor.py`, `test_examples_e2e.py` |
| Delineation, metrics and plots | `test_cluster_plot.py`, `test_distance.py`, `test_diversity.py`, `test_evaluator.py`, `test_partition.py`, `test_pseudo_label.py`, `test_tree.py`, `embedding/test_umap_rendering.py` |
| Checkpoint, device and training primitives | `test_checkpoint_utils.py`, `test_device_resolution.py`, `test_training_dataset.py`, `test_training_loss.py`, `test_training_model.py`, `test_training_scheduler.py` |
| Update command | `test_update.py` |
| Training integration and recovery | `training/test_dino_objective.py`, `training/test_dino_pretrain.py`, `training/test_dino_resume.py`, `training/test_finetune_provenance.py`, `training/test_koleo_gradient_and_order.py`, `training/test_koleo_shared.py`, `training/test_masked_pretrain.py`, `training/test_pretrain_alignment.py`, `training/test_ssl_config.py`, `training/test_ssl_lineage.py`, `training/test_ssl_metrics.py`, `training/test_startup_phases.py` |

Changes and retained contracts:

- Merge the identical positive-weight relational KoLeo training setup for
  checkpoint persistence and CSV reporting. Retain all four saved-key checks
  and all method/status/N/A reporting checks.
- Fold the scalar-only KoLeo check into the enabled/disabled optimizer
  comparison. Check finite, nonzero CLS derivatives for both global crops and
  zero derivatives for patch tokens. Compare omitted KoLeo configuration with
  explicit zero on CPU, and fail if either disabled path computes neighbors.
- Retain all 19 corrupt-resume cases (six incomplete payloads, nine encoder
  mutations, four progress/optimizer/alias gaps). Train one serialized source
  per epoch plan; every case copies independent input/output files and loads
  fresh tensors. Pytest fixture events confirm two source setups, replacing
  19 per-case training setups. Exact save/resume replay tests remain independent.
- A failure-reason probe exposed a pre-existing false positive: the
  `final_norm_alias` case loaded a completed two-epoch checkpoint and failed the
  requested-epoch check before reaching strict alias validation. Load the real
  first-epoch checkpoint from that same plan instead. Require the intended CLI
  diagnostic in all four gap cases; the alias assertion failed before this
  input correction and passed afterward. Preserve log-byte and directory checks.
- Replace metric-name and CLS-selection source-string checks with returned
  metric mapping and actual image/batch extraction checks. Fold the two UMAP
  source checks into the existing evaluator test, inspecting actual axis labels,
  caller/default title and neighbor limit. Remove the vacuous EVA source-string
  assertion while retaining numerical masked-position invariance. The old UMAP
  test file is removed; its historical implementation plan remains a historical
  record, not a current test-selection command.
- Retain numerical/RNG replay, serialization, warning/exception assertions,
  filesystem safety, checkpoint compatibility, real ONNX parity and independent
  method/patch-mode interactions. Identical function bodies with different
  invalid-input decorators are not duplicate behavioral coverage.

Recorded verification (overlapping selections, not additive totals):

- Initial focused gate: KoLeo shared/gradient/order, evaluator, training losses
  and the two rewritten reporting nodes: `107 passed in 19.25s`.
- Six temporary in-memory mutations were rejected by the new assertions:
  omit the optimized KoLeo contribution; calculate disabled KoLeo; swap the NMI
  mapping; select a patch instead of CLS; append a split suffix to UMAP labels;
  ignore the caller/default UMAP title. No production file was modified. The
  CLS probe's initial decorator-handling errors were corrected before counting
  its assertion failure as evidence.
- Six affected modules (`test_koleo_shared.py`,
  `test_koleo_gradient_and_order.py`, `test_dino_resume.py`,
  `test_pretrain_alignment.py`, `test_evaluator.py`, `test_training_loss.py`):
  `PYTHONPATH=src python -m pytest -q <these six files> --durations=12`:
  `361 passed in 377.87s`, exit 0. This run loaded the resume module before the
  final alias-input correction; the changed 19-case selection was rerun afterward.
- Final corrected resume selection:
  `PYTHONPATH=src python -m pytest -q tests/training/test_dino_resume.py -k
  'incomplete_resume or corrupt_encoder or preflight_gaps' --tb=short --setup-show`:
  `19 passed, 49 deselected in 32.53s`, exit 0, with two source fixture setups.
- Final documentation/schema/workflow/version gate:
  `PYTHONPATH=src python -m pytest -q tests/test_docs.py
  tests/test_workflow_skill.py tests/test_cli_schema.py tests/test_version.py`:
  `95 passed, 1 skipped in 9.64s`, exit 0.
- Collection after consolidation and the final input correction:
  `PYTHONPATH=src python -m pytest --collect-only -q`:
  `1748 tests collected in 2.06s`, exit 0. Four fewer test functions become
  three fewer collected cases because the combined UMAP behavior test has two
  cases. Both diff whitespace checks pass; per-file hashes confirm unchanged
  production and unrelated files, and the staging area is unchanged.

No post-change full-suite result or measured speedup is claimed. An independently
started full run reported `1750 passed, 1 skipped in 860.72s`, but it began before
these edits and does not verify this final candidate. Production source,
production device defaults and tolerances are unchanged; CUDA and the declared
dependency floors were not separately exercised. No commit, push or release is
authorized by this follow-up.

### Builder discretion and approval boundaries

The builder may choose concrete filenames, small private function signatures, test helper placement and nonsemantic panel spacing after mapping actual imports. They may not add a framework, change exported schema shape, rename flags, alter defaults, weaken recovery/validation, enable new real-data/environment work, or broaden KoLeo beyond pretrain without an amendment. Existing private test patch locations may change; preserved public run-entry imports must continue to work.

Design self-review is not an independent audit or operator acceptance. Current authorization does not authorize subagent delegation. Commit, push, tag and release each remain outside this draft's authority.

## Appendix A. Current and proposed option help

The current text is taken from baseline CLI declarations, with adjacent Python string literals joined. Proposed text is the exact English runtime help to approve; user-facing command explanations follow their existing language conventions, independently of English-only development specs/plans. Unlisted option text is unchanged.

### A1. pretrain --ssl-method

Current:
> CLS distillation objective: 'relational' (default; the relational target recipe — a synchronized student/teacher pair with weak teacher views and cross/full pairing) or 'dino' (prototype distribution distillation with an explicit prototype layer). Applies to global-to-global and local-to-global pairs, not a complete training recipe. The patch objective is selected independently by --patch-loss; both objectives share the encoder and teacher temperature (used by ibot). DINO is opt-in and is not a resume of a relational run. Omit to inherit the saved method on --resume; a cross-method resume is rejected.

Proposed:
> CLS objective for global-to-global and local-to-global pairs: relational (default; weak teacher targets, cross/full pairing) or dino (prototype distillation). Not a complete training recipe. Select --patch-loss independently; both share the encoder and teacher temperature (ibot). On --resume, omit to inherit; switching methods is rejected.

Affected baseline tests for A1: `tests/test_cli_smoke.py::test_pretrain_help_distinguishes_cls_and_patch_objectives`, especially lines 1929–1933; proposed capitalization changes line 1930's lowercase phrase, and the independent-selection wording changes line 1932. Update wording checks while retaining the same semantic constraints.

### A2. pretrain --patch-loss

Current:
> Patch-level objective. 'none', 'consistency' (default; selects visible same-position patches for normalized cosine regression, i.e. masked-position consistency, never input masking), 'masked-feature' (true continuous masked feature prediction of the teacher's final-four-block patch target) or 'ibot' (experimental prototype-distribution prediction). The CLS distillation objective is selected independently by --ssl-method; both objectives share the encoder and teacher temperature (used by ibot).

Proposed:
> Patch objective: none; consistency (default; visible same-position cosine regression, no input masking); masked-feature (masked input, teacher final-four-block feature targets); or ibot (experimental prototype prediction). Select --ssl-method independently; both share the encoder and teacher temperature (ibot).

Affected baseline tests for A2: `tests/test_cli_smoke.py::test_pretrain_help_distinguishes_cls_and_patch_objectives:1934–1936`, especially the `selected independently` assertion at line 1935; and `test_pretrain_help_documents_patch_modes_and_mask_migration:1888–1916`, whose broad `selected` check at line 1904 needs targeted semantics even though unrelated text currently keeps it green. Preserve visible-position/no-input-masking and experimental prototype distinctions.

### A3. pretrain --koleo-weight

Current:
> Weight for the optional KoLeo regularizer on normalized backbone CLS features. 0 (default) disables it and skips all neighbor search; any finite value > 0 enables it. Experimental: it spreads the CLS space but also repels same-species and near-duplicate specimens, batches smaller than 2 skip it, and repeated records can dominate the total. DINO-only and not changeable on resume.

Proposed:
> Shared relational/DINO KoLeo weight on normalized backbone CLS; finite and >= 0. Default 0 skips neighbor search; batches < 2 skip KoLeo. Experimental: also repels same-species/near-duplicate samples; repeats can dominate. On --resume, omit to inherit; changing the weight requires a new run.

Affected checks for A3: current help uses the general rendered-help checks in `tests/test_cli_smoke.py:95–136`; add an explicit shared-method/disabled/experimental/resume-warning assertion. Update the related applicability, metrics and inheritance expectations in `tests/training/test_ssl_config.py`, `test_ssl_metrics.py`, `test_dino_pretrain.py`, and `tests/test_cli_schema.py::test_ssl_schema_conditional_inheritance:1827` as part of the behavior change, not merely the wording edit. These are test owners, not a claim that an existing direct KoLeo-help assertion exists.

### A4. pretrain --relational-pairing

Current:
> Global student/teacher view pairs averaged in the relational loss: cross (A-to-B and B-to-A) or full (all four). Effective default for a new run: cross. Replaces the removed disable-cross-view-loss switch.

Proposed:
> Relational global pairs: cross (A-to-B and B-to-A; new-run default) or full (all four). On --resume, omit to inherit. Replaces the removed disable-cross-view-loss switch.

Affected checks for A4: general rendered help at `tests/test_cli_smoke.py:95–136`; add a focused check for the cross/full definitions, new-run default and omitted-resume inheritance. No dedicated baseline pairing-help literal assertion was found.

### A5. finetune --loss

Current:
> Metric-learning loss: arcface (default), supcon, subcenter-arcface, or subcenter-arcface-compact. Rejected before any output when the name is unknown. On --resume an omitted --loss inherits the recorded mode; an explicit conflicting value fails.

Proposed:
> Metric loss: arcface (default), supcon, subcenter-arcface, or subcenter-arcface-compact. Unknown names fail before output setup. On --resume, omit to inherit; explicit conflicts fail.

Affected checks for A5: general rendered help and finetune key-option coverage at `tests/test_cli_smoke.py:95–136`; retain parser/default/applicability/resume behavior tests and add a targeted approved-wording check. No dedicated baseline literal assertion for this loss help was found.

### A6. finetune --checkpoint

Current:
> Initialization checkpoint used when --resume is not set. An SSL pretrain checkpoint installs a fresh embedding head; a fine-tune checkpoint with a matching head keeps its trained projector.

Proposed:
> Initialization when not using --resume: SSL installs a fresh embedding head; a matching fine-tune head keeps its projector. See --pseudo-label-from for round 2 initialization.

Affected checks for A6: general rendered help and safety ordering at `tests/test_cli_smoke.py:65–72,95–136`; add a focused cross-reference check rather than duplicating the pseudo-mode contract. The original SHA-256 statement was verified in `trainer.py::_resolve_pseudo_ssl_path:6508–6529` (both relocated and recorded paths); it is omitted from this short help to avoid a second explanation beside `--pseudo-label-from`. The full command documentation retains the hash requirement.

## Appendix B. Reference contracts and review record

- [Master architecture and documentation ownership](2026-03-29-otuformer-design.md).
- [DINO/KoLeo, recovery and historical CSV contracts](2026-10-31-otuformer-cls-distillation-design.md).
- [Relational targets and recovery](2026-11-03-relational-pretrain-targets-design.md).
- [Documentation and help conventions](2026-09-29-otuformer-documentation-conventions-design.md).
- [Masked-pretrain contract](2026-09-20-otuformer-masked-pretrain-v070-design.md).
- [Sparse-label provenance](2026-09-26-otuformer-sparse-label-pseudolabel-design.md).
- Typer official documentation: `https://typer.tiangolo.com/tutorial/options/help/` (native `rich_help_panel`).
- Pytest official documentation: `https://docs.pytest.org/en/stable/how-to/usage.html` (selection and duration reporting), `https://docs.pytest.org/en/stable/how-to/mark.html` (registered markers).

Review status: three external audit rounds were supplied by the operator and
checked against the actual source, tests and logs before each revision.

Round one settled the compatibility model (zero-weight relational keeps the
pre-extension metadata shape) and the implementation-impact map. Round two
added callback-driven panel order, the real patch/source-inspection
dependencies, move-versus-copy ownership, the display/storage/CSV split and the
corrected 46/47 file count. Round three found six defects in the first
implementation; all are fixed here:

1. A direct entry point could still enable KoLeo on a checkpoint with no
   recorded bundle, because only the explicitness marker was checked. The
   recorded state now decides, and the explicitness marker only distinguishes
   "changed" from "inherited".
2. Bundle completeness used value-based presence, so an explicit `null` looked
   like an absent key, and a complete bundle was not required to carry a
   positive weight. Presence is now key-based, and a complete relational bundle
   must be strictly positive with supported numeric epsilons.
3. Startup phases opened after the work they name (dataset construction,
   objective and loss construction). Each start now precedes its first real
   operation, and a test asserts that ordering from inside the constructor.
4. The refactor still owned the loops in `trainer.py` and re-exported 205 names,
   which is what §4 forbids. The loops moved to `pretrain_loop` and
   `finetune_loop`; `trainer.py` is a 19-line alias surface with three exports
   and nothing under `src/` imports it.
5. The gradient and update-order tests did not prove their claims: deleting the
   KoLeo term from the loss left them green, and the EMA order check still read
   source text. Both were replaced by behavioural assertions and confirmed by
   mutation.
6. Two help tests depended on terminal colour environment. Help assertions are
   now ANSI-independent and the exactly-once check covers every declared
   option, verified under four different colour environments.

The seventh note was a documentation contradiction: the DINO/KoLeo topic design
still stated that relational rejects explicit KoLeo. It now keeps its historical
paragraph and carries a dated supersession note linking this design.

The earlier claim that the broad normalized `selected` assertion necessarily
fails was not reproduced; it stayed green through unrelated prose, so it was
tightened rather than rewritten.

Local diagnostics used Typer 0.25.1, Click 8.4.2 and Rich 15.0.0; they confirm
first-appearance panel order and the complete rendered option set, but not the
declared `typer>=0.12` floor. No independently executed audit, benchmark, or
GPU/real-data verification is claimed; the full-suite result and the
mutation-test evidence above are the verification record.

## Recovery status (2026-11-08)

An accidental working-tree reset on 2026-11-08 rewrote every tracked path to a
2026-11-06 stash snapshot. The nine new training modules and the two new design
documents survived because they were untracked; the tracked-file side was
rebuilt from archived scripts, isolated replays, and the test suite as the
oracle. This section records what the current candidate contains, so no earlier
claim in this document is read as a fresh acceptance verdict.

Restored and re-verified after the reset:

- `trainer.py` is a 19-line compatibility surface; the nine responsibility
  modules are present and imported by their real owners.
- The cleanup-safety chain (`extraction_image_paths`, `selected_image_paths`,
  the "enumerate actual inputs, then protect, then clear" order) came back from
  commit `84f845e`, which is also where the CLI, schema and reader sources now
  come from.
- Shared KoLeo: the four keys have a single definition in `pretrain_recipe`
  (`KOLEO_FIXED_CONFIG`, `KOLEO_DEFAULT_WEIGHT`); `dino.py` no longer holds a
  second copy. Static applicability, resume provenance and the new-run zero
  default are covered.
- The CLI hands over the resume payload together with `resume_state_path`, so
  one invocation validates and restores from one checkpoint and reads it once.
- Help coverage is enforced over the whole declaration set (53/43 options,
  exactly once, panel order preserved) under both plain and forced-colour
  environments; `flat_help` and the colour-suppressing fixture are persisted.
- Metric-name, CLS-token, and teacher-EMA checks are behavioural: they read
  returned values, real tensors, and a recorded event order instead of source
  text. The EVA source assertion is gone.

Still outstanding, and therefore excluded from any "all complete" claim:

- **Shared serialized fixtures** are now in place in
  `tests/training/test_dino_resume.py`: a module-scoped fixture trains two
  immutable sources once (one-epoch plan with the final checkpoint for the six
  incomplete and nine encoder cases; two-epoch plan with `SSL_epoch_0001.pth` for
  the four preflight gaps, so a completed-plan check cannot mask the intended
  diagnostic). Each case regenerates its own images, manifest and output
  directory and deserializes the payload itself. The process state is restored
  inside the fixture *before* the paths are handed over — not at module teardown,
  which would both leak the training state to the first consumer and roll back
  whatever later tests advanced. The restore covers Python, NumPy (full state:
  array, position cursor and cached Gaussian value), torch CPU, CUDA and the MPS
  backend, plus the thread count: `torch.manual_seed` seeds MPS as well, so a
  CPU-only training run still moves it. Training and restoring live in one
  function that both the module fixture and the regression call, so the
  regression exercises the fixture's real restore path and fails if that restore
  is removed; the before/after boundary is owned by the test, so it does not
  depend on execution order. Isolation is checked by mutating nested optimizer
  state in place — rebinding the outer dict would not prove that two loads do
  not share the inner one. The exact save/resume and RNG replay tests stay
  independent.

The pinned `tests/training/baseline_readers/checkpoint_84f845e.py` is a
historical reference and must **not** be kept in sync with the current reader:
its SHA-256 check guards against accidental edits, not staleness. When a current
interface change makes it unloadable, adapt the loader or the consumer; replace
the file, commit id and digest only on a deliberate decision to change baseline.

Training-test devices were migrated where the policy allows it: the ordinary
DINO pretrain, DINO resume and lineage integration helpers now resolve through
`device="auto"` (CUDA, then MPS, then CPU). The exact-replay tests
(`test_dino_same_plan_resume_matches_uninterrupted`,
`test_pretrain_resume_at_ratio_03_matches_uninterrupted`) pass `device="cpu"`
explicitly on every call, because their `rtol=0, atol=0` contract is a CPU
guarantee that an accelerator pass cannot replace. Two groups stay on CPU with the
reason recorded next to the setting: runs with smaller local crops, whose
positional interpolation needs the antialiased bicubic backward that is
unimplemented on MPS, and constructor-snapshot comparisons taken before and
after device placement. The remaining CPU call sites are exact replay, RNG,
dtype/device and fixed-input contracts. No accelerator pass is claimed beyond
the local MPS run these selections exercised.
- **Downstream KoLeo compatibility regressions** are now persisted in
  `tests/training/test_koleo_shared.py`: a positive-weight relational checkpoint
  produced by a real run goes through the actual consumers — the extract loader,
  the CAM loader (unpacking its `(model, size, name)` triple), a real ONNX export
  in a subprocess, and a real `run_finetune` whose frozen backbone weights must
  equal the source payload. The positive weight is also trained alongside each of
  the four patch modes. The baseline-reader boundary is checked against a pinned
  copy of `84f845e`'s `checkpoint.py` kept in `tests/training/baseline_readers/`
  with its SHA-256 asserted, so the check needs no Git worktree or checkout.

Documentation: the master design and the two topic designs carry the KoLeo and
MPS wording, and the 2026-09-29 documentation-conventions design now records the
approved bounded help-compression exception (§5.3, amended 2026-11-08).

Evidence limits: `/tmp/restore-run5.log` (`1740 passed, 1 skipped`, 2026-11-08
18:13 UTC+8) predates the fixes recorded above and is historical only. Later
checks were targeted selections per the impact-based policy, not a full-suite
run, and overlapping selections must not be added together as full coverage.
