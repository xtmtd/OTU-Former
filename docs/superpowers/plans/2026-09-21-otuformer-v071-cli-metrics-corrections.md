# OTU-Former v0.7.1 CLI and Metrics Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release v0.7.1 with portable CLI parameter-source detection and reproducible, semantically correct embedding metrics across `pretrain`, `finetune`, and `extract`, while preserving existing metric fields and historical CSV compatibility.

**Architecture:** Keep the existing shared evaluator in `src/otuformer/embedding/evaluator.py` as the single implementation for retrieval, classification, probing, and clustering metrics. Keep `trainer.py` as the adapter for epoch logging and plotting, and keep `extract.py` as the standalone CSV/console adapter. Normalize unsupported metrics to the existing empty-string CSV convention at the logging boundaries; do not introduce an `unavailable` enum or an ID-canonicalization pipeline.

**Tech Stack:** Python 3.11, Typer, Click-compatible parameter sources, NumPy, scikit-learn, pandas, Matplotlib, pytest, `pyproject.toml` package metadata.

**Spec:** This plan implements the reviewed v0.7.1 CLI and metrics correction decision recorded in the current conversation: replace fragile `ParameterSource` identity checks; use explicit shuffled stratified CV; make trainer and extract use the same sorted subsample order; fix mAP self-exclusion; preserve `kNN_Acc_k20`; represent unsupported values as empty CSV fields; reject invalid KMeans fallback results; add balanced linear-probe accuracy without replacing the historical accuracy field.

## Global Constraints

- Release version is exactly `0.7.1` in `pyproject.toml` and `src/otuformer/__init__.py`.
- Do not rewrite historical references that document the v0.7.0 masked-pretraining release; only update current package/version tests and current release-facing text where required.
- Do not add a new dependency.
- Do not add a sample-ID canonicalization pipeline; trainer and extract use the same sorted subsample index order.
- Do not add leave-one-out kNN; retain the existing CV-based `kNN_Acc_k1`, `kNN_Acc_k5`, and `kNN_Acc_k20` fields.
- Remove the unused `compute_linear_probing()` compatibility wrapper; repository search found no caller, and all production paths use the combined probing-metrics function exactly once.
- Do not directly import external `click` from any CLI module. `typer>=0.26` may vendor Click, so pass only Typer/native Python option types and validate CAM/extract enumerations with local `*_CHOICES` constants plus `typer.BadParameter` before output-directory creation.
- Do not add a direct Click dependency or constrain Typer below 0.26; fixing the two direct `click.Choice` uses removes the incompatible boundary instead.
- Do not delete or rename existing metric fields.
- Existing valid numeric metrics remain numeric; metrics that cannot be computed are written as empty CSV fields (`""`) and become `NaN` when plotted.
- `Silhouette_Score` continues to mean cosine silhouette computed against the true labels, matching the existing and reference-script semantics; document this in code comments/docstrings where the evaluator contract is clarified.
- Recall and mAP intentionally retain different singleton-query conventions for historical comparability: Recall uses every query in its denominator, while mAP excludes queries with no non-self relevant item; document this known semantic difference rather than treating the two denominators as interchangeable.
- The duplicate-`image` validation in `extract` is an explicit defensive input-integrity fix discovered during review; it is limited to preventing silent embedding/label misalignment and does not add a general data-validation framework.
- All CLI source code, comments, and tests remain in English.

## Review Focus

- A Typer vendored-Click `ParameterSource` object must be recognized by name, explicit patch options must remain explicit, and no CLI module may pass an external `click.Choice` as `click_type`; test this with a source object that is not the external Click enum class and direct validation tests for the CAM/extract choices.
- A two-class dataset with at least five samples per class must use five shuffled folds rather than being forced to two folds; test the actual splitter passed to scikit-learn.
- A singleton class or an unsupported k must produce an empty metric value rather than numeric zero; test both kNN and linear probing.
- mAP must rank only non-query samples, including when the query is tied with another vector; test against a hand-computed small example.
- Trainer and extract must apply the same sorted subsample order, and KMeans must not fabricate one cluster when there are fewer distinct embeddings than requested clusters; test both boundaries.

---

### Task 1: Add failing evaluator and CLI regression tests

**Files:**
- Modify: `tests/test_evaluator.py`
- Modify: `tests/test_cli_smoke.py`
- Test target: `tests/test_evaluator.py`, `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: current evaluator functions and CLI helper functions.
- Produces: failing tests that define the v0.7.1 behavior before implementation changes.

- [ ] **Step 1: Add evaluator tests for explicit CV behavior**

Add tests that monkeypatch both `sklearn.model_selection.cross_val_score` and `sklearn.model_selection.cross_validate` as appropriate, capture their `cv` arguments, and assert that kNN and the combined linear-probe implementation receive a `StratifiedKFold` with:

```python
n_splits == 5
shuffle is True
random_state == 42
```

Use two classes with ten or more samples each so `n_classes == 2` cannot accidentally make the expected fold count pass. Also assert that a label array with a singleton class returns unavailable values instead of `0.0`.

- [ ] **Step 2: Add evaluator tests for unsupported k and balanced probing**

Use a small dataset whose smallest training fold has fewer than 20 samples. Assert that `kNN_Acc_k20` is unavailable while `kNN_Acc_k1` remains numeric. Add a test asserting that the probing result exposes both ordinary and balanced accuracy values, with keys:

```python
"Linear_Probing_Acc"
"Linear_Probing_Balanced_Acc"
```

- [ ] **Step 3: Add a hand-computed mAP regression test**

Construct a small embedding matrix with one query and two non-query samples, where the query is similar to one same-label sample and dissimilar to a different-label sample. Assert that mAP is computed from the non-query ranking only. Include a case with a singleton label and assert that the singleton query is excluded from the mAP mean rather than treated as having a self-match.

- [ ] **Step 4: Add boundary tests for Recall and clustering**

Assert that one-sample Recall returns an unavailable value, that `Recall@20` is unavailable when fewer than 21 samples exist rather than being silently renamed, and that KMeans metrics return unavailable values when the number of distinct normalized embeddings is less than the requested number of clusters. Keep the normal multi-class clustering test asserting numeric `NMI`, `ARI`, `AMI`, `Silhouette_Score`, and `Purity`.

- [ ] **Step 5: Add CLI source-compatibility and native-choice tests**

Add a small fake context whose `get_parameter_source()` returns an enum-like object with `.name == "COMMANDLINE"` but is not `click.core.ParameterSource`. Assert that the pretrain command formatter includes a supplied option. Add a callback-level test or focused helper assertion that the same source rule marks the patch options as explicit.

Add direct tests for each CAM/extract native validator: valid enumerated values proceed and invalid values raise `typer.BadParameter` before `prepare_output_dir()` is reached. Invoke `otuformer cam --help` with the existing CLI test runner and assert that `--arch` lists `cnn, vit`, `--fig-format` lists `png, jpg, pdf`, and `--device` lists `auto, cpu, cuda, mps`. Assert by source inspection that `src/otuformer/cli/` contains neither `import click` nor `click.Choice`, preventing external Click objects from reaching Typer's vendored Click implementation.

- [ ] **Step 6: Update existing evaluator expectations and run the focused tests**

Update the existing unsupported-k test so `kNN_Acc_k20` expects the evaluator-layer unavailable result (`None`), not `0.0`. Run:

```bash
pytest tests/test_evaluator.py tests/test_cli_smoke.py -q
```

Expected: the new v0.7.1 behavior tests fail against the current implementation, while unrelated existing tests may pass. Do not modify production code in this task.

---

### Task 2: Make evaluator CV and unsupported-value semantics correct

**Files:**
- Modify: `src/otuformer/embedding/evaluator.py:11-176`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: the failing tests from Task 1.
- Produces: evaluator functions that use one explicit, reproducible CV policy and return `None` for unsupported results:
  - `compute_knn_accuracy(...) -> dict[str, float | None]`
  - `compute_linear_probing_metrics(...) -> dict[str, float | None]`

Repository search found no caller for `compute_linear_probing()`, so do not retain a compatibility wrapper; all production paths will call the combined probing helper exactly once.

- [ ] **Step 1: Remove the obsolete CV-count helper and define the fold rule**

Delete `_safe_cv_splits()` rather than keeping two overlapping sources of truth. The sole private splitter helper will compute the smallest class count and return `min(5, min_class_count)` when at least two folds are possible. Do not include the number of classes in the fold-count limit. Return `None` for a singleton class; do not return `1` as a usable CV count.

- [ ] **Step 2: Add one shared explicit splitter builder**

Add a private helper with an exact contract such as:

```python
def _make_stratified_cv(labels: np.ndarray, max_splits: int = 5):
    """Return StratifiedKFold or None when stratified CV is impossible."""
```

It must return:

```python
StratifiedKFold(
    n_splits=min(max_splits, min_class_count),
    shuffle=True,
    random_state=42,
)
```

or `None` for a singleton class. This helper is the sole source of CV configuration for kNN and linear probing.

- [ ] **Step 3: Update kNN to use the explicit splitter**

Pass the splitter object to `cross_val_score`. Determine the smallest training-fold size without iterating over all splits: for `n_splits`, use `len(labels) - math.ceil(len(labels) / n_splits)`. Return `None` for k values larger than that size. Retain the keys `kNN_Acc_k1`, `kNN_Acc_k5`, and `kNN_Acc_k20`. Catch only the existing per-metric scoring failure needed to preserve the evaluator’s resilient behavior, and return `None` rather than `0.0` for an unsupported result.

- [ ] **Step 4: Add ordinary and balanced linear-probe scores in one CV pass**

Implement `compute_linear_probing_metrics()` with `sklearn.model_selection.cross_validate`, using the same splitter object and two scorers:

```python
scoring={
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}
```

Return the means under `Linear_Probing_Acc` and `Linear_Probing_Balanced_Acc`. Do not retain or add a `compute_linear_probing()` wrapper: repository search found no caller, and keeping it would be dead compatibility code. Production callers must invoke this combined helper exactly once.

- [ ] **Step 5: Run evaluator tests**

Run:

```bash
pytest tests/test_evaluator.py -q
```

Expected: all evaluator tests pass, including explicit splitter, singleton, unsupported-k, and balanced-accuracy cases. The existing unsupported-k expectation must assert `None`, not `0.0`.

---

### Task 3: Correct retrieval and clustering edge cases

**Files:**
- Modify: `src/otuformer/embedding/evaluator.py:48-126`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: the CV and return-value contracts from Task 2.
- Produces: correct retrieval and clustering metric values with `None` for unsupported cases.

- [ ] **Step 1: Fix Recall@K query and K bounds**

For each query, exclude the query index by constructing candidate indices from all positions except `i`, rather than relying on a diagonal `-inf` sentinel. For each requested k, independently set only that Recall result key to `None` when `k > n_samples - 1`; continue computing other valid k values. Return `None` for an empty dataset or a one-sample dataset. Preserve the current definition that a query is correct when at least one same-label candidate appears in the top-k set, and preserve the current all-query denominator including singleton labels. This `Recall@20` boundary is a generic evaluator test; trainer and extract production calls currently request only `[1, 5, 10]`.

- [ ] **Step 2: Fix mAP query exclusion**

For each query, build `candidate_indices = np.arange(n) != i`, sort only those candidates by descending cosine similarity, and compute relevance against the remaining labels. Exclude queries with no relevant non-query item from the mean. Return `None` when no query has a relevant non-query item. Do not change the existing AP formula beyond removing the query itself.

- [ ] **Step 3: Remove the fabricated KMeans fallback**

If normalized embeddings contain fewer distinct rows than the number of requested clusters, return unavailable values for the clustering metrics instead of assigning every item to cluster zero. Otherwise retain `KMeans(n_clusters=n_clusters, random_state=42, n_init=10)` and the existing NMI, ARI, AMI, and purity formulas.

- [ ] **Step 4: Clarify and preserve silhouette semantics**

Keep `Silhouette_Score` computed with the true label codes and cosine distance. Return `None` for inputs where silhouette is undefined rather than numeric zero. Keep the current all-class requirement and let the surrounding adapter write unavailable values as empty fields.

- [ ] **Step 5: Run focused retrieval and clustering tests**

Run:

```bash
pytest tests/test_evaluator.py -q
```

Expected: mAP no longer counts the query as a relevant result, Recall bounds are explicit, and normal clustering metrics remain numeric.

---

### Task 4: Unify trainer metric adaptation, sampling order, schema migration, and plots

**Files:**
- Modify: `src/otuformer/training/trainer.py:900-930,1188-1245,1280-1300,1452-1508`
- Modify: `tests/training/test_pretrain_alignment.py`
- Modify: `tests/training/test_masked_pretrain.py` only if schema fixtures require the new field
- Test: `tests/training/test_pretrain_alignment.py`

**Interfaces:**
- Consumes: evaluator return values `float | None` and probing metrics dict from Tasks 2-3.
- Produces: trainer adapters that write valid numbers or empty strings to `metrics.pretrain.csv` and `metrics.finetune.csv`, atomically migrate a recognized v0.7.0 metrics header before a resumed append, use the same sorted subsampling order as extract, and plot the new balanced LP series without breaking old fields.

- [ ] **Step 1: Make trainer subsampling use sorted indices**

Change `_maybe_subsample_for_metrics()` to use:

```python
idx = np.sort(rng.choice(len(embeddings), size=max_samples, replace=False))
```

Apply the same operation to embeddings and labels. Keep the seed supplied by the caller.

- [ ] **Step 2: Add the balanced probing field and a recognized legacy schema migration**

Add `Linear_Probing_Balanced_Acc` next to `Linear_Probing_Acc` in the `EnhancedMetricsLogger` field list. Unlike `InstantMetricsLogger`, `EnhancedMetricsLogger` currently writes a new `DictWriter` row without inspecting a pre-existing header; fix that before logging any resumed epoch.

Implement the minimal local equivalent of the existing `InstantMetricsLogger` schema handling: read the header and rows, reject duplicate/malformed headers, and **return without rewriting** when the header already equals the current `fieldnames`, so repeated v0.7.1 `--resume` runs remain appendable. Otherwise, recognize the exact v0.7.0 enhanced-metrics header, then atomically rewrite it with the current header and empty `Linear_Probing_Balanced_Acc` cells before appending. Reject unknown headers. Preserve every existing field and row. Do not broaden this into a shared CSV migration framework.

- [ ] **Step 3: Normalize evaluator `None` values at the trainer boundary**

In `_compute_all_metrics()`, initialize all metric fields with `""`. Update fields only with numeric values, converting `None` to `""` before the existing print loop; this conversion is mandatory because the trainer currently executes `float(v)` whenever `v != ""`. When `compute_linear_probe` is true, call the combined probing-metrics helper exactly once and populate both `Linear_Probing_Acc` and `Linear_Probing_Balanced_Acc`; do not call a removed compatibility wrapper. Keep linear probing disabled on non-designated pretrain epochs and forced on finetune save epochs as currently configured.

- [ ] **Step 4: Add the balanced probing series to the training plot**

Add `Linear_Probing_Balanced_Acc` to the existing Retrieval/Probe panel alongside `mAP` and `Linear_Probing_Acc`. Continue using `pd.to_numeric(..., errors="coerce")` so empty fields create gaps rather than false zeros.

- [ ] **Step 5: Extend trainer tests**

Assert that trainer subsampling returns sorted row order; `_compute_all_metrics()` preserves empty strings for unavailable metrics; and the logger schema contains `Linear_Probing_Balanced_Acc` without removing `kNN_Acc_k20` or `Linear_Probing_Acc`.

Create valid v0.7.0 enhanced-metrics fixtures for both `metrics.pretrain.csv` and `metrics.finetune.csv`, each with a legacy header and row. Construct `EnhancedMetricsLogger` for each, log another row, and assert the atomic migration leaves all rows aligned with the new header while legacy balanced-probe cells are empty. Reconstruct a logger over the resulting current-schema file and verify a second `--resume` append succeeds without rewrite. Add a malformed/unknown-header test that fails before append, so resume cannot silently corrupt either CSV.

- [ ] **Step 6: Run trainer-focused tests**

Run:

```bash
pytest tests/training/test_pretrain_alignment.py tests/training/test_masked_pretrain.py -q
```

Expected: current pretrain/fine-tune metric schema tests pass with the additional balanced-probe field; no historical field is removed; and v0.7.0 pretrain/finetune enhanced-metrics files migrate before a resumed append.

---

### Task 5: Align extract metric behavior and label handling

**Files:**
- Modify: `src/otuformer/cli/extract.py:304-350`
- Modify: `tests/test_cli_smoke.py` or `tests/test_evaluator.py` for extract-specific behavior

**Interfaces:**
- Consumes: evaluator contracts from Tasks 2-3.
- Produces: `extract` metrics that use the same sorted sample order and expose the same metric names as trainer logs.

- [ ] **Step 1: Keep extract subsample order identical to trainer**

Retain sorted subsample indices in extract and document the shared rule in the local comment. Do not add ID canonicalization or change the embedding extraction order outside the current `id`-based label alignment.

- [ ] **Step 2: Add balanced probing to extract output**

Call the combined probing-metrics helper exactly once, write both `Linear_Probing_Acc` and `Linear_Probing_Balanced_Acc`, and preserve the existing `metrics.csv` rows for all historical fields. Do not call a removed compatibility wrapper or run CV twice. Convert unavailable values to empty CSV cells through the existing pandas serialization path.

- [ ] **Step 3: Preserve label alignment and reject duplicate label keys**

This is an extract-specific defensive input-integrity fix discovered during review, not a new general validation framework. Before `set_index("image")`, validate that the filtered label CSV has unique `image` values. Raise a clear `ValueError` for duplicates instead of allowing `.loc` to duplicate label rows and silently desynchronize embeddings and labels. Keep the current embedding-row order as the alignment order.

- [ ] **Step 4: Add extract-specific tests**

Test that extract requests the same metric keys as trainer, writes the balanced probing key, keeps `kNN_Acc_k20`, and rejects duplicate `image` labels before metric computation. Test that an unavailable evaluator result is serialized as an empty metric value rather than zero.

- [ ] **Step 5: Run extract-focused tests**

Run:

```bash
pytest tests/test_cli_smoke.py tests/test_evaluator.py -q
```

Expected: extract output is schema-compatible with trainer metrics and duplicate label rows fail clearly.

---

### Task 6: Remove external Click coupling and fix all CLI parameter-source comparisons

This is a required cross-environment bug fix, not a speculative macOS-only refactor. The current macOS environment uses Typer 0.25.1 with external Click and therefore does not reproduce it. However, the project declares `typer>=0.12` without an upper bound; Typer versions that vendor Click can return a `ParameterSource` class different from the separately imported `click.core.ParameterSource`, and reject separately imported `click.Choice` instances passed through `click_type`. In that environment, identity comparison silently skips CLI options and marks pretrain patch options as non-explicit, while CAM/extract command construction can fail before option validation. The regression tests must model the foreign enum boundary and verify the native validators without depending on the host installation.

**Files:**
- Modify: `src/otuformer/cli/pretrain.py:12,106-110,377-386`
- Modify: `src/otuformer/cli/finetune.py:12,56-61`
- Modify: `src/otuformer/cli/annotate.py:11,27-32`
- Modify: `src/otuformer/cli/cluster.py:11,784-789`
- Modify: `src/otuformer/cli/export.py:10,28-33`
- Modify: `src/otuformer/cli/cam.py:11,27-32,65-149`
- Modify: `src/otuformer/cli/diversity.py:10,53-58`
- Modify: `src/otuformer/cli/extract.py:11,30-35,73-78,185-190`
- Modify: `src/otuformer/vision/cam.py:153-158` documentation only
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: the fake-source regression tests from Task 1.
- Produces: CLI command construction and validation compatible with both external Click and Typer-vendored Click, with no direct Click import in `src/otuformer/cli/`.

- [ ] **Step 1: Replace the fragile enum identity checks**

At every listed comparison point, replace:

```python
source is click.core.ParameterSource.COMMANDLINE
```

with the name-based check:

```python
getattr(source, "name", None) == "COMMANDLINE"
```

For negative filters, use the corresponding `!=` form. Apply the same rule to `pretrain.py`’s `explicit_sources` dictionary so resume merging and patch-option conflict validation see explicit CLI values. The name check is intentionally chosen to bridge external Click and Typer-vendored Click without importing Typer private modules.

- [ ] **Step 2: Replace external `click.Choice` with native validation**

In `cam.py`, define local immutable choice constants and one `_validate_cam_options()` helper covering `cam_method`, `arch` when present, `fig_format`, `save_npy`, `eval_transform`, and `device`. In `extract.py`, define the evaluation-transform choices and a matching local validator. Remove each `click_type=click.Choice(...)`, invoke validation before `prepare_output_dir()`, and raise `typer.BadParameter` with the option name and allowed values on invalid input.

Preserve CLI help as a user-visible contract: explicitly list choices in the `--arch`, `--fig-format`, and `--device` help strings because those options currently rely on Click's automatic `[... ]` suffix. The existing hand-written choice text for `--cam-method`, `--save-npy`, and `--eval-transform` stays sufficient; remove their now-meaningless `show_choices=False` arguments. Update only the stale `vision/cam.py` docstring that says the CLI uses `click.Choice`; leave its direct library-level validation such as `otuformer.vision.cam._validate_save_mode()` intact.

- [ ] **Step 3: Remove all direct CLI Click imports**

After converting the source checks and choices, remove `import click` from every CLI module. Do not add a dependency, import Typer’s private vendored Click module, or set an upper bound on Typer: the code will use only Typer/public Python interfaces.

- [ ] **Step 4: Run CLI regression tests**

Run:

```bash
pytest tests/test_cli_smoke.py -q
```

Expected: command logging includes explicitly supplied options under a foreign enum-like source object; pretrain explicit-source behavior remains covered; invalid CAM/extract choices fail before output creation; CAM help retains the listed choices for `--arch`, `--fig-format`, and `--device`; and `rg -n "\bclick\b" src/otuformer/cli` has no matches.

---

### Task 7: Update v0.7.1 documentation and historical compatibility notes

**Files:**
- Modify: `README.md`
- Modify: `README.cn.md`
- Create: `docs/superpowers/specs/2026-09-21-otuformer-v071-cli-metrics-corrections.md`
- Modify: `docs/superpowers/specs/2026-09-07-global-barcode-validation.md` with a historical evaluator-version note only
- Modify: `docs/superpowers/specs/2026-03-30-otuformer-pretrain-alignment-design.md` with a historical evaluator-version note only
- Do not modify: historical v0.7.0 design specifications or any recorded experiment values

**Interfaces:**
- Consumes: the final metric and CLI contracts from Tasks 2-6.
- Produces: user-facing v0.7.1 documentation and an explicit current metric specification without rewriting historical experiment values.

- [ ] **Step 1: Document the current metric contract in both READMEs**

Add a concise section near the training/extraction output documentation covering explicit shuffled `StratifiedKFold(shuffle=True, random_state=42)`, fold limits based on the smallest class, the retained `kNN_Acc_k20` field, empty values for unsupported metrics, mAP self-exclusion, the Recall/mAP singleton-query denominator difference, true-label cosine silhouette semantics, and the new `Linear_Probing_Balanced_Acc` field. Correct only the inaccurate `metrics.json` wording to the current `logs/metrics.pretrain.csv`, `logs/metrics.finetune.csv`, and extract `metrics.csv` outputs. Preserve the existing v0.7.0 patch-loss/resume paragraphs because `tests/test_version.py` and CLI smoke documentation checks intentionally require that historical migration documentation.

- [ ] **Step 2: Create the v0.7.1 metric specification**

Create the named spec with the exact public metric keys, CV splitter contract, unavailable-value behavior, sampling-order rule, mAP/Recall definitions, KMeans invalid-input behavior, CLI parameter-source compatibility rule, and version target `0.7.1`. Link the implementation plan from the spec.

- [ ] **Step 3: Mark historical reports without changing their numbers**

Add a short evaluator-vintage note to `2026-09-07-global-barcode-validation.md` and `2026-03-30-otuformer-pretrain-alignment-design.md`, each of which records epoch-50 metrics. State that their reported values use the pre-v0.7.1 evaluator and are not numerically comparable with v0.7.1 outputs. Do not recalculate, rewrite, or remove the historical tables, reference values, directional criteria, or v0.7.0 design specifications.

- [ ] **Step 4: Run documentation checks**

Run:

```bash
pytest tests/test_version.py tests/test_cli_smoke.py -q
rg -n "Linear_Probing_Balanced_Acc|StratifiedKFold|pre-v0\\.7\\.1|metrics\\.pretrain\\.csv|metrics\\.finetune\\.csv" README.md README.cn.md docs/superpowers/specs/2026-09-21-otuformer-v071-cli-metrics-corrections.md
```

Expected: English and Chinese README contracts match, the new spec is discoverable, and historical notes are present without changing historical metric values.

---

### Task 8: Bump package version and update current version tests

**Files:**
- Modify: `pyproject.toml:7`
- Modify: `src/otuformer/__init__.py:3`
- Modify: `tests/test_version.py:10,11`
- Modify: `tests/test_cli_smoke.py` only for current release assertions that must move from `0.7.0` to `0.7.1`
- Do not bulk-edit: historical v0.7.0 design documents and migration references

**Interfaces:**
- Consumes: implementation and documentation changes from Tasks 2-7.
- Produces: a package/project version pair equal to `0.7.1`.

- [ ] **Step 1: Update the two version declarations**

Set:

```toml
version = "0.7.1"
```

and:

```python
__version__ = "0.7.1"
```

- [ ] **Step 2: Update current-version tests**

Change exact version assertions in `tests/test_version.py` and release-facing CLI smoke assertions to `0.7.1`. Leave tests that intentionally assert the presence of historical v0.7.0 masked-pretraining documentation unchanged.

- [ ] **Step 3: Run version tests**

Run:

```bash
pytest tests/test_version.py -q
```

Expected: package metadata and runtime version match exactly at `0.7.1`.

---

### Task 9: Run the complete verification suite and inspect the diff

**Files:**
- Modify: none unless verification exposes a task-owned failure
- Test: full repository test suite

**Interfaces:**
- Consumes: completed Tasks 1-8.
- Produces: verified v0.7.1 implementation with no unrelated changes.

- [ ] **Step 1: Run all tests**

Run:

```bash
pytest -q
```

Expected: all tests pass. Any failure must be fixed in the owning task before declaring the plan complete.

- [ ] **Step 2: Run static searches for regressions**

Run separate static searches so tests may retain their foreign-enum fixture while production CLI code cannot retain any Click coupling:

```bash
rg -n "\bclick\b" src/otuformer/cli
rg -n "pred = np\.zeros|np\.fill_diagonal\(sim, -np\.inf\)|min\(max_cv, n_classes" src/otuformer/embedding tests
```

Expected: the CLI search has no matches, while the evaluator search has no fabricated all-zero KMeans fallback, no mAP implementation relying on diagonal sentinels, and no class-count CV cap in production paths.

- [ ] **Step 3: Inspect the final diff**

Run:

```bash
git diff --check
git diff --stat
git status --short
```

Confirm that only the planned CLI modules, evaluator/trainer/extract adapters, focused tests, and current version declarations changed. Do not revert unrelated user changes.

- [ ] **Step 4: Record compatibility notes**

The release notes or final change summary must state that historical `kNN_Acc_k*`, `Linear_Probing_Acc`, and clustering values are not numerically comparable across the pre-v0.7.1 evaluator because CV fold selection, sample ordering, mAP self-inclusion, and unsupported-value handling changed. Existing CSV field names remain available; the new `Linear_Probing_Balanced_Acc` field may be empty for old rows. Also state that resumed v0.7.0 `metrics.pretrain.csv` and `metrics.finetune.csv` logs are atomically migrated by adding an empty balanced-probe column; unrecognized or malformed headers are rejected before append.

---

## Execution Order

Execute Tasks 1 through 9 in order. Tasks 2 and 3 establish the evaluator contract before trainer and extract adapters consume it. Task 6 is required even though the current macOS dependency combination does not reproduce the failure: the project permits Typer versions that vendor Click, while the current source imports external Click's enum and choices. The regression tests must model that cross-module enum boundary without depending on the host platform. Task 7 documents the finalized contract before Task 8 updates the release version. Task 9 is the release gate.

## Deferred Deliberate Simplifications

- No leave-one-out kNN path: `Recall@1` already supplies the all-library nearest-neighbor retrieval view, while kNN fields preserve their historical CV semantics.
- No ID-based canonical ordering: both trainer and extract sort the same seeded subsample indices, which is sufficient for the current single-process comparison workflow.
- Recall and mAP retain the known singleton-query denominator difference for historical comparability; a future metric-contract revision may unify the query population, but that is outside v0.7.1.
- No removal of `kNN_Acc_k20`: preserve CSV, plot, and downstream compatibility; return an empty value only when the requested k cannot be supported.
- The historical epoch-50 tables in `docs/superpowers/specs/2026-03-30-otuformer-pretrain-alignment-design.md`, `docs/superpowers/specs/2026-09-07-global-barcode-validation.md`, and other pre-v0.7.1 reports are not numerically comparable with v0.7.1 metrics; preserve their recorded values and label their evaluator vintage.
