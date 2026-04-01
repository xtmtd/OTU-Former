# OTU-Former Annotate Module Design

Date: 2026-04-01
Status: Draft (user-approved design summary)
Scope: Upgrade `otuformer annotate` into a cluster-output-aware correction and post-processing pipeline.

## 1. Goals

Build `annotate` as a practical "post-cluster correction" tool that:

1. Requires `--raw-assignments` from standard `cluster` outputs.
2. Applies expert corrections from CSV under strict validation.
3. Produces updated assignment artifacts in a target output folder (not original partition folder).
4. Recomputes or refreshes intra-class pairwise distance summaries using corrected clusters.
5. Exports a metabarcoding-style OTU table for downstream diversity workflows.
6. Generates an annotated UPGMA partitions PDF with an additional left annotation strip for corrected clusters.

## 2. Non-Goals

- No dry-run mode.
- No provenance fields embedded into output CSV/PDF (key metadata stays in logs and console output).
- No in-place overwrite of files under the original `runs/cluster/.../partitions/tables` directory.

## 3. CLI Contract

### 3.1 Command

`otuformer annotate`

### 3.2 Options

- `--raw-assignments` (required)
  - Must point to a standard cluster file:
    - `.../UPGMA/partitions/tables/partition_<cutoff>_assignments.csv`
  - This path is used to auto-locate related cluster artifacts:
    - cluster run root (`runs/<cluster_run>`)
    - `logs/cluster.log`
    - tree and partition outputs under `UPGMA/`
- `--corrections` (required)
  - Correction table with at least two required semantic columns:
    - `id` (or `image` alias)
    - `cluster`
  - Recommended workflow: copy from raw assignments and edit target rows.
  - User-prepared CSV is allowed if schema requirements are met.
- `--out-dir` (optional, default `runs/annotate`)
  - All annotate outputs are written here.

## 4. Input Validation and Strictness Rules

Execution is fail-fast. Any violation raises an error and stops.

### 4.1 Raw assignments validation

- Path pattern must match cluster standard structure.
- Columns must include `id` and `cluster`.
- `sample` is optional and preserved when present.

### 4.2 Corrections validation

- Must resolve to canonical columns: `id`, `cluster`.
- `id` can come from `id` or `image`.
- Empty `id` or `cluster` values are invalid.
- Duplicate `id` rows with conflicting target clusters are invalid.

### 4.3 Cross-file strict matching (required)

- Every correction `id` must exist in `--raw-assignments`.
- Any missing ID triggers error and termination (strict mode).

## 5. Data Flow

1. Parse `--raw-assignments` and infer:
   - partition cutoff tag from filename
   - cluster run root and linked files
2. Read and canonicalize raw assignments and corrections.
3. Validate schema and strict ID matching.
4. Apply cluster updates to corrected IDs.
5. Write corrected assignments and changed-only table to `--out-dir`.
6. Build OTU abundance table (`otu_table.csv`).
7. Update intra-class pairwise distance summary using corrected cluster labels.
8. Render annotated UPGMA partition PDF with extra left corrected-cluster annotation strip.
9. Write summary JSON and log key decisions/results to terminal + log file.

## 6. Distance Update Strategy

Target output file: `pairwise_distance_summary_intra-class.csv`

Distance type selection:

1. Read from cluster log parameters (`distance = cosine|euclidean`) when available.
2. If unavailable or unparseable, fail with explicit error (do not guess silently).

Distance matrix source strategy:

1. If `distance_statistics/distance_matrix.csv` exists in the linked cluster run, reuse it.
2. Otherwise, resolve embeddings path from cluster log and recompute matrix using the same method used by cluster:
   - cosine: cosine distance on embeddings
   - euclidean: euclidean distance on L2-normalized embeddings (consistent with cluster behavior)
3. Recompute intra-class summary grouped by corrected cluster labels.

## 7. OTU Table Export

Target output file: `otu_table.csv`

Format:

- Rows: sample IDs (if `sample` present).
- Columns: OTU cluster names.
- Values: abundance counts.

If `sample` is absent:

- Emit one row with synthetic sample key `all_samples`.
- Columns remain OTU clusters with global counts.

This output is designed to integrate directly with downstream diversity/usearch workflows.

## 8. Annotated UPGMA PDF

Target output file: `UPGMA_tree_partitions_annotated.pdf`

Approach:

- Reuse existing UPGMA partition plotting logic/style.
- Add a new left-side annotation panel for corrected cluster labels aligned one-to-one with leaf images/IDs.
- Add thin colored separators between adjacent corrected clusters; neighboring clusters must have different colors.
- Preserve the existing partitioning schemes visual bands and overall readability.

## 9. Outputs

All written under `--out-dir`:

1. `partition_<cutoff>_assignments.csv`
2. `partition_<cutoff>_assignments_changed_only.csv`
3. `otu_table.csv`
4. `pairwise_distance_summary_intra-class.csv`
5. `UPGMA_tree_partitions_annotated.pdf`
6. `annotation_summary.json`
7. `logs/annotate.log`

## 10. Error Taxonomy (proposed)

- `E001`: raw assignments path is not a standard cluster output path.
- `E002`: raw assignments missing required columns (`id`, `cluster`).
- `E003`: corrections missing required semantic columns (`id|image`, `cluster`).
- `E004`: corrections contain IDs absent from raw assignments (strict mismatch).
- `E005`: corrections contain conflicting duplicate IDs.
- `E006`: cannot resolve distance method or distance inputs required for intra-class summary.

## 11. Help Text Requirements

`-h` must clearly explain:

- Why `--raw-assignments` must come from standard cluster outputs.
- Corrections schema minimum (`id|image`, `cluster`), with recommendation to edit from raw file.
- `sample` optionality in both raw and correction context.
- Strict mismatch behavior (missing correction IDs cause immediate failure).

## 12. Testing Plan

### 12.1 Unit tests

- raw with/without `sample`.
- corrections with `id` vs `image` alias.
- strict mismatch failure.
- conflicting duplicate correction IDs failure.
- changed-only output correctness.

### 12.2 CLI integration tests

Use provided real paths for validation baseline:

- raw: `runs/cluster/UPGMA/partitions/tables/partition_0.2_assignments.csv`
- corrections: `runs/annotate/correction.csv`

Assert expected outputs are produced in annotate out-dir and that distance method is correctly resolved from cluster log.

### 12.3 Manual check

- Open `UPGMA_tree_partitions_annotated.pdf` and verify corrected label strip + color separators.
- Validate `otu_table.csv` structure for downstream diversity usage.

## 13. Suggested Future Enhancements (outside current scope)

- Optional dual strictness modes (`strict` and `warn-skip`) if user demand appears later.
- Optional direct handoff command to diversity module (`annotate -> diversity`) for one-shot workflows.
