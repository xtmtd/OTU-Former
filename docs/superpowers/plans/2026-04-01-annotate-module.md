# OTU-Former Annotate Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `otuformer annotate` into a strict, cluster-output-aware correction pipeline that generates corrected assignments, updated intra-class distance summary, OTU table, and annotated UPGMA PDF in a target output directory.

**Architecture:** Keep `cli/annotate.py` as orchestration and logging entrypoint. Move schema validation, strict matching, path inference, distance-summary update, OTU table generation, and tree annotation data prep into `delineation/annotate.py`. Reuse existing cluster distance/tree helpers where practical to avoid duplicate algorithms and keep behavior consistent with `cluster` outputs.

**Tech Stack:** Python 3.11+, Typer, pandas, NumPy, scipy, matplotlib/seaborn, pytest

---

## File Map

- Modify: `src/otuformer/cli/annotate.py`
  - Rename `--assignments` to `--raw-assignments`, improve `--help`, enforce standard path contract, orchestrate all new outputs.
- Modify: `src/otuformer/delineation/annotate.py`
  - Add canonicalization/validation logic, strict mismatch checks, correction application, changed-only export frame, OTU table builder, cluster-run path inference, distance mode parsing, and annotated-tree rendering helpers.
- Modify: `src/otuformer/cli/cluster.py`
  - Minimal extraction/reuse point for partition-tree plotting helper if needed so annotate can render a style-consistent PDF without code duplication.
- Modify: `tests/test_annotate.py`
  - Expand unit coverage for schema aliases, strict mode failures, duplicate conflict handling, and OTU table behavior with/without sample.
- Modify: `tests/test_cli_smoke.py`
  - Update annotate help flag assertions (`--raw-assignments`).
- Create: `tests/test_cli_annotate.py`
  - Add focused CLI integration tests for strict failure and success artifact generation in temp directories.

## Task 1: Rename CLI Contract and Help Text

**Files:**
- Modify: `src/otuformer/cli/annotate.py`
- Test: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write failing help-text tests first**

Add assertions that annotate help includes `--raw-assignments` and does not require `--assignments`.

```python
def test_subcommand_help_includes_key_flags(...):
    ...
    ("annotate", ["--raw-assignments", "--corrections", "--out-dir"])
```

- [ ] **Step 2: Run targeted help test and confirm failure**

Run: `pytest tests/test_cli_smoke.py -k "annotate and help" -v`
Expected: FAIL because current flag is `--assignments`.

- [ ] **Step 3: Implement CLI option rename and detailed help text**

In `annotate.py`:

- rename option to `raw_assignments: Path = typer.Option(..., "--raw-assignments", ...)`
- document standard path requirement explicitly
- document corrections minimum schema (`id|image`, `cluster`)
- keep `--out-dir` default as `runs/annotate`

- [ ] **Step 4: Re-run targeted help test**

Run: `pytest tests/test_cli_smoke.py -k "annotate and help" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/cli/annotate.py tests/test_cli_smoke.py
git commit -m "feat: rename annotate input flag to raw-assignments"
```

## Task 2: Build Strict Input Validation + Canonicalization

**Files:**
- Modify: `src/otuformer/delineation/annotate.py`
- Test: `tests/test_annotate.py`

- [ ] **Step 1: Write failing unit tests for schema canonicalization**

Add tests for:

- corrections using `image` instead of `id`
- raw with optional `sample`
- raw missing `id/cluster` should fail
- corrections missing `cluster` should fail

```python
def test_corrections_accept_image_alias():
    corrections = pd.DataFrame({"image": ["a"], "cluster": ["OTU_2"]})
    ...

def test_raw_requires_id_and_cluster():
    with pytest.raises(ValueError, match="id.*cluster"):
        validate_raw_assignments(pd.DataFrame({"id": ["a"]}))
```

- [ ] **Step 2: Run targeted validation tests and confirm failure**

Run: `pytest tests/test_annotate.py -k "schema or alias or requires" -v`
Expected: FAIL (helpers not implemented yet).

- [ ] **Step 3: Implement canonicalization/validation helpers**

Add minimal focused helpers:

- `validate_raw_assignments(df)`
- `canonicalize_corrections(df)`
- `validate_corrections_nonempty_fields(df)`

Use explicit error messages for CLI wrapping.

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_annotate.py -k "schema or alias or requires" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/annotate.py tests/test_annotate.py
git commit -m "feat: add annotate schema canonicalization and validation"
```

## Task 3: Enforce Strict Corrections Matching and Conflict Detection

**Files:**
- Modify: `src/otuformer/delineation/annotate.py`
- Test: `tests/test_annotate.py`

- [ ] **Step 1: Write failing tests for strict mismatch and duplicate conflict**

```python
def test_strict_mode_fails_on_missing_correction_ids():
    raw = pd.DataFrame({"id": ["a"], "cluster": ["OTU_1"]})
    corr = pd.DataFrame({"id": ["b"], "cluster": ["OTU_2"]})
    with pytest.raises(ValueError, match="not found"):
        validate_corrections_against_raw(raw, corr)

def test_fails_on_conflicting_duplicate_correction_ids():
    corr = pd.DataFrame({"id": ["a", "a"], "cluster": ["OTU_2", "OTU_3"]})
    with pytest.raises(ValueError, match="conflict"):
        validate_duplicate_conflicts(corr)
```

- [ ] **Step 2: Run targeted tests and confirm failure**

Run: `pytest tests/test_annotate.py -k "strict or duplicate or conflict" -v`
Expected: FAIL.

- [ ] **Step 3: Implement strict checks and changed-only helper**

Implement:

- `validate_corrections_against_raw(raw, corr, max_examples=20)`
- `validate_duplicate_conflicts(corr)`
- `build_changed_only_table(original, corrected)` returning id/old_cluster/new_cluster (+sample if present)

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_annotate.py -k "strict or duplicate or conflict" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/annotate.py tests/test_annotate.py
git commit -m "feat: enforce strict correction id matching and conflicts"
```

## Task 4: Infer Cluster Context and Distance Mode

**Files:**
- Modify: `src/otuformer/delineation/annotate.py`
- Modify: `src/otuformer/cli/annotate.py`
- Test: `tests/test_cli_annotate.py`

- [ ] **Step 1: Write failing integration tests for path contract and log parsing**

Add tests that:

- reject non-standard `--raw-assignments` path
- parse cluster log and resolve distance mode (`cosine` or `euclidean`)

```python
def test_annotate_rejects_nonstandard_raw_assignments_path(tmp_path):
    ...
    assert result.exit_code != 0
    assert "standard cluster output" in result.output
```

- [ ] **Step 2: Run targeted integration tests and confirm failure**

Run: `pytest tests/test_cli_annotate.py -k "nonstandard or distance" -v`
Expected: FAIL.

- [ ] **Step 3: Implement cluster-context resolvers**

Implement helper functions:

- `infer_cluster_run_context(raw_assignments_path)`
- `parse_distance_mode_from_cluster_log(log_path)`
- `parse_embeddings_path_from_cluster_log(log_path)`

Wire into CLI orchestration and log all resolved values to console and `logs/annotate.log`.

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_cli_annotate.py -k "nonstandard or distance" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/annotate.py src/otuformer/cli/annotate.py tests/test_cli_annotate.py
git commit -m "feat: infer cluster context and distance mode in annotate"
```

## Task 5: Generate Corrected Assignments, OTU Table, and Intra-Class Distance Summary

**Files:**
- Modify: `src/otuformer/cli/annotate.py`
- Modify: `src/otuformer/delineation/annotate.py`
- Test: `tests/test_annotate.py`
- Test: `tests/test_cli_annotate.py`

- [ ] **Step 1: Write failing tests for output artifacts and filename contract**

Add assertions for outputs in `--out-dir`:

- `partition_<cutoff>_assignments.csv`
- `partition_<cutoff>_assignments_changed_only.csv`
- `otu_table.csv`
- `pairwise_distance_summary_intra-class.csv`

- [ ] **Step 2: Run targeted tests and confirm failure**

Run: `pytest tests/test_cli_annotate.py -k "outputs or otu_table or intra_class" -v`
Expected: FAIL.

- [ ] **Step 3: Implement artifact writing and distance summary update logic**

Implementation rules:

- all outputs write to `--out-dir` only
- OTU table: row=`sample`, col=`cluster`, value=count; fallback row `all_samples` when sample missing
- distance summary:
  - reuse `distance_statistics/distance_matrix.csv` if present
  - else recompute matrix from embeddings with parsed distance mode
  - write output as `pairwise_distance_summary_intra-class.csv`

- [ ] **Step 4: Re-run targeted tests**

Run: `pytest tests/test_annotate.py tests/test_cli_annotate.py -k "otu_table or intra_class or changed_only or outputs" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/cli/annotate.py src/otuformer/delineation/annotate.py tests/test_annotate.py tests/test_cli_annotate.py
git commit -m "feat: write corrected annotate artifacts and intra-class summary"
```

## Task 6: Render Annotated UPGMA PDF

**Files:**
- Modify: `src/otuformer/delineation/annotate.py`
- Modify (if needed for reuse): `src/otuformer/cli/cluster.py`
- Test: `tests/test_cli_annotate.py`

- [ ] **Step 1: Write failing integration test for annotated PDF generation**

```python
def test_annotate_generates_upgma_tree_partitions_annotated_pdf(tmp_path):
    ...
    assert (out_dir / "UPGMA_tree_partitions_annotated.pdf").exists()
```

- [ ] **Step 2: Run targeted test and confirm failure**

Run: `pytest tests/test_cli_annotate.py -k "annotated_pdf" -v`
Expected: FAIL.

- [ ] **Step 3: Implement PDF rendering with corrected cluster strip**

Requirements:

- keep partition bands style aligned with cluster output
- add left corrected-cluster annotation column aligned by leaf order
- keep corrected-cluster bars left-anchored within the corrected panel (x=0 baseline)
- add thin color separators between adjacent corrected clusters, with different neighbor colors
- output filename exactly `UPGMA_tree_partitions_annotated.pdf`

- [ ] **Step 4: Re-run targeted test**

Run: `pytest tests/test_cli_annotate.py -k "annotated_pdf" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/annotate.py src/otuformer/cli/cluster.py tests/test_cli_annotate.py
git commit -m "feat: add corrected-cluster panel to annotate UPGMA PDF"
```

## Task 7: End-to-End Annotate CLI Test with Provided Paths Pattern

**Files:**
- Modify: `tests/test_cli_annotate.py`

- [ ] **Step 1: Add e2e-style fixture that mirrors real directory layout**

Create a temp directory tree mimicking:

- `runs/cluster/UPGMA/partitions/tables/partition_0.2_assignments.csv`
- `runs/cluster/logs/cluster.log`
- linked embeddings and tree assets

- [ ] **Step 2: Add happy-path test asserting full output set**

Assert output existence:

- `partition_0.2_assignments.csv`
- `partition_0.2_assignments_changed_only.csv`
- `otu_table.csv`
- `pairwise_distance_summary_intra-class.csv`
- `UPGMA_tree_partitions_annotated.pdf`
- `annotation_summary.json`

- [ ] **Step 3: Run e2e-style test**

Run: `pytest tests/test_cli_annotate.py -k "happy_path" -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli_annotate.py
git commit -m "test: cover annotate happy path with cluster-like layout"
```

## Task 8: Full Verification and Cleanup

**Files:**
- Modify as needed based on failures from prior tasks

- [ ] **Step 1: Run annotate-focused test suite**

Run: `pytest tests/test_annotate.py tests/test_cli_annotate.py tests/test_cli_smoke.py -v`
Expected: PASS.

- [ ] **Step 2: Run broader quick regression suite**

Run: `pytest tests/test_partition.py tests/test_distance.py tests/test_tree.py -v`
Expected: PASS.

- [ ] **Step 3: Run quality checks if configured**

Run (if available):

- `python -m compileall src/otuformer`

Expected: no syntax errors.

- [ ] **Step 4: Final commit**

```bash
git add src/otuformer/cli/annotate.py src/otuformer/delineation/annotate.py src/otuformer/cli/cluster.py tests/test_annotate.py tests/test_cli_annotate.py tests/test_cli_smoke.py
git commit -m "feat: complete annotate correction workflow with strict validation and outputs"
```

## Notes for Implementers

- Keep implementation DRY by reusing existing distance and UPGMA utilities from `delineation/distance.py` and `delineation/tree.py`.
- Do not silently downgrade strict behavior. Missing correction IDs must remain hard failures.
- Preserve current log style: print key resolved paths, distance mode, correction counts, and output locations to screen and log file.
- Prefer focused helper functions over a monolithic `annotate()` body.
