# OTU-Former Diversity Module Design

Date: 2026-04-01
Status: Draft (user-approved design summary)
Scope: Extend `otuformer diversity` input sources, per-sample outputs, and phylo handling.

## 1. Goals

1. Support diversity calculation from either assignments CSV or OTU table CSV.
2. Enforce mutual exclusivity between `--assignments` and `--otu-table-csv`.
3. Compute per-sample diversity outputs when sample labels are present and valid.
4. Improve phylo behavior: warn when tree is missing; avoid empty MPD rows.
5. Expand CLI help text to explain inputs and metrics (including MPD).
6. Remove `--no-phylo` in favor of `--phylo` (default false).

## 2. Non-Goals

- No changes to upstream modules beyond necessary data contracts.
- No reformatting of existing output tables unless required by new behavior.
- No automatic tree inference or reconstruction for MPD.

## 3. CLI Contract

### 3.1 Command

`otuformer diversity`

### 3.2 Options (new/changed)

- `--assignments <csv>`
  - Cluster assignments input (id/image, cluster, optional sample).
- `--otu-table-csv <csv>` (new)
  - OTU table input: first column is sample (header can be empty), columns 2..N are OTU IDs.
- `--phylo`
  - Enable phylogenetic metrics (MPD). Requires `--tree` to compute MPD.
- `--tree <nwk>`
  - Newick tree file used for MPD.
- `--otu-table-has-header` (new, optional)
  - Explicitly marks the OTU table as having a header row. Required if OTU IDs are numeric.

### 3.3 Mutual Exclusivity

Exactly one of `--assignments` or `--otu-table-csv` is required. If both are provided, exit with an error.

## 4. Input Validation Rules

### 4.1 Assignments CSV

- Must have columns: `id` (or `image`) and `cluster`.
- Optional `sample` column.
- Header names are matched case-insensitively with leading/trailing whitespace trimmed.
- If `sample` exists and any value is empty:
  - Emit a warning.
  - Only compute and emit the global (all-samples) diversity table.
  - Do not create per-sample outputs.

### 4.2 OTU Table CSV

- First column contains sample IDs (header may be empty).
- Columns 2..N must have OTU IDs.
- Header auto-detection:
  - If the first row contains any non-numeric values from column 2 onward, treat it as a header.
  - If columns 2..N are all numeric on the first row, treat as data unless `--otu-table-has-header` is set.
  - If no header is detected, fail with an error (OTU IDs required).
- OTU IDs may be numeric, but then `--otu-table-has-header` must be provided.
- CSV parsing uses comma delimiter, UTF-8 (BOM tolerated), and standard quoted fields.

## 5. Output Behavior

### 5.1 Global Table

- Existing output location and format preserved.
- Includes all requested `--min-abundance` thresholds.

### 5.2 Per-Sample Outputs

- If sample labels are present and valid:
  - Create `--out-dir/per-sample/`.
  - Write one file per sample.
  - Each file matches the global table format and includes all `--min-abundance` thresholds.
  - File naming: trim whitespace; replace path separators and whitespace with `_`; if duplicates occur, append a numeric suffix.
- If sample labels are missing/invalid: do not create the per-sample directory.

## 6. Phylogenetic Metrics (MPD)

- When `--phylo` is set without `--tree`:
  - Emit a warning.
  - Skip MPD computation.
  - Do not include an empty MPD row in output tables.
- When `--phylo` and `--tree` are both set:
  - Compute MPD and populate output rows.
  - If MPD cannot be computed, warn and omit the row.
  - If the tree is missing OTUs from the table, warn and drop those OTUs for MPD.
  - Extra tree tips not found in the OTU table are ignored with a warning.

## 7. Help Text Requirements

`otuformer diversity -h` must explain:

- `--assignments` vs `--otu-table-csv` usage and mutual exclusivity.
- Expected columns for assignments and OTU table inputs.
- Per-sample output behavior.
- Phylo behavior and tree requirement.
- Metric definitions, including MPD (mean pairwise phylogenetic distance).
- Note removal of `--no-phylo` and instruct to use `--phylo` instead.

## 8. Error Handling

- `--assignments` and `--otu-table-csv` both provided: error and exit.
- OTU table without valid OTU column headers: error and exit.
- Assignments missing required columns: error and exit.
- `--no-phylo` used: error and exit with guidance to use `--phylo`.

## 9. Testing Plan

### 9.1 Assignments Input

- 2-column input (global only).
- 3-column input with valid sample (global + per-sample).
- 3-column input with empty sample values (warning + global only).

### 9.2 OTU Table Input

- Valid header detection with OTU IDs.
- Invalid header (numeric-only row 1) -> error.

### 9.3 Phylo

- `--phylo` without `--tree`: warning, MPD omitted.
- `--phylo` with valid tree: MPD populated.

## 10. Out of Scope

- Additional phylogenetic metrics beyond MPD.
- Any tree construction or re-rooting logic.
