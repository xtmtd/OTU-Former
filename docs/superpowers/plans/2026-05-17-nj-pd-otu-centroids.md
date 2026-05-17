# NJ-Based Faith's PD via OTU Centroids — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace individual-level UPGMA-based Faith's PD with OTU-centroid-level NJ-based PD in the diversity module, without touching cluster or annotate modules.

**Architecture:** After OTU assignments are finalised (including any manual annotation), compute per-OTU centroid embeddings by averaging member embeddings and L2-normalising. Build a small NJ tree (k OTUs) from the centroid cosine distance matrix using skbio. Feed that tree to the existing skbio Faith's PD/phydiv calls. The cluster-step UPGMA tree is no longer involved in PD computation.

**Tech Stack:** Python, numpy, pandas, skbio (`DistanceMatrix`, `nj`, `alpha_diversity`, `phydiv`), existing `distance._safe_l2_normalize`.

---

## Data contracts

| File | Columns |
|---|---|
| `embeddings.csv` | `id`, `sample`, `dim_0`…`dim_191` |
| `partition_*_assignments.csv` | `id`, `cluster`, `sample` |

Join key: `id`. OTU label column: `cluster`.

---

## File Map

| File | Change |
|---|---|
| `src/otuformer/delineation/diversity.py` | Replace `compute_mpd_from_counts` + `compute_mpd` with new centroid-NJ pipeline; keep `diversity_table` signature identical |
| `src/otuformer/delineation/tree.py` | Add `build_nj_tree(dist_matrix, ids) -> skbio.TreeNode`; add `compute_nj_bootstrap_support` via `skbio.tree.majority_rule` |
| `tests/test_diversity.py` | Replace / extend PD tests for new interface |
| `tests/test_tree.py` | Add `test_build_nj_tree_*` unit tests |

**No changes** to `cluster.py`, `annotate`, `distance.py`.

---

## Task 1: Add `build_nj_tree` and `compute_nj_bootstrap_support` to `tree.py`

**Files:**
- Modify: `src/otuformer/delineation/tree.py`
- Test: `tests/test_tree.py`

> **Status: COMPLETE** (commit `ea813e3`)
>
> **Design change from original plan:** `_extract_nj_clades` + manual clade counting were replaced by `skbio.tree.majority_rule`. This eliminates ~60 lines of manual logic. `compute_nj_bootstrap_support` no longer takes a `base_tree` parameter — it generates all replicate trees then calls `majority_rule(rep_trees, cutoff=0.5)` to obtain support counts directly.
>
> **Key finding:** `majority_rule` returns raw replicate counts (not fractions) in `node.support`; divide by `n_replicates * 100` for percentage.

- [x] **Step 1: Write failing tests** — `test_build_nj_tree_returns_tree_node`, `test_build_nj_tree_negative_branch_clipped`
- [x] **Step 2: Run to confirm failure**
- [x] **Step 3: Implement `build_nj_tree`** — symmetrise, zero diagonal, `skbio.tree.nj`, clip negatives
- [x] **Step 4: Run tests — PASS**
- [x] **Step 5: Refactor bootstrap** — remove `_extract_nj_clades`, rewrite `compute_nj_bootstrap_support` with `majority_rule`; fix support scaling bug (raw count ÷ n_replicates × 100)
- [x] **Step 6: All 9 tree tests PASS**
- [x] **Committed:** `ea813e3 feat(tree): add NJ bootstrap via skbio majority_rule, remove manual clade counting`

**Final functions in `tree.py`:**
- `build_nj_tree(dist_matrix, ids) -> skbio.TreeNode`
- `compute_nj_bootstrap_support(centroids, ids, support_mode, n_replicates, subsample_ratio, random_state, save_trees_path, n_jobs) -> dict[frozenset, float]`

---

## Task 2: Add `compute_otu_centroids` to `diversity.py`

> **Status: COMPLETE** (commit `0d13c74`)

- [x] 3 tests written and passing: `test_compute_otu_centroids_shape`, `test_compute_otu_centroids_l2_normalised`, `test_compute_otu_centroids_missing_ids_ignored`
- [x] Committed: `0d13c74 feat(diversity): add compute_otu_centroids`

---

## Task 3: Add `compute_pd` (NJ-centroid pipeline) to `diversity.py`

> **Status: COMPLETE** (commit `39c39dd`)
>
> **Design fix during implementation:** Original plan used `alpha_diversity("faith_pd", ...)` which requires a rooted tree. Changed to `phydiv(..., rooted=True)` + `tree.root_at_midpoint()` for Faith's PD; `phydiv(..., rooted=False)` on unrooted tree for MPD_w.
>
> **Final signature:**
> ```python
> compute_pd(assignments, embeddings,
>     nj_tree_path=None, nj_bootstrap_replicates=0,
>     nj_bootstrap_path=None, nj_bootstrap_support_mode="subsample",
>     nj_bootstrap_subsample_ratio=0.8, nj_jobs=1,
>     nj_centroids_path=None) -> dict[str, float]
> ```
> (output path params added in subsequent tasks)

- [x] 4 tests passing: `test_compute_pd_returns_three_keys`, `test_compute_pd_values_are_finite`, `test_compute_pd_single_otu_returns_nan`, `test_compute_pd_richness_norm_relation`
- [x] Committed: `39c39dd feat(diversity): add compute_pd using OTU-centroid NJ tree with midpoint rooting`

---

## Task 4: Update `diversity_table` to use `compute_pd`

> **Status: COMPLETE** (commit `ea0ddd8`)
>
> **Final signature** (extended beyond original plan to pass through NJ output paths):
> ```python
> diversity_table(assignments, min_abundances,
>     embeddings=None, tree_newick_path=None,
>     nj_tree_path=None, nj_bootstrap_replicates=0,
>     nj_bootstrap_path=None, nj_bootstrap_support_mode="subsample",
>     nj_bootstrap_subsample_ratio=0.8, nj_jobs=1,
>     nj_centroids_path=None) -> pd.DataFrame
> ```
> NJ tree/bootstrap/centroids written only on first `min_abundance` threshold (via `_nj_written` flag) to avoid redundant computation.

- [x] 2 integration tests passing: `test_diversity_table_with_embeddings_contains_pd_metrics`, `test_diversity_table_no_pd_when_no_embeddings`
- [x] All 46 diversity tests PASS
- [x] Committed: `ea0ddd8 feat(diversity): update diversity_table to accept embeddings for NJ PD`

---

## Task 5: Update CLI (`cli/diversity.py`) to pass embeddings + NJ outputs

> **Status: COMPLETE** (commits `cb9672e`, `32f4fa9`, `038f3f0`)
>
> **Extended beyond original plan** — three rounds of CLI work:
> 1. `cb9672e` — `--embeddings`, `--phylo` wiring
> 2. `32f4fa9` — `--save-nj-tree`, `--nj-bootstrap`, `--nj-bootstrap-mode`, `--nj-subsample-ratio`, `--cpus`
> 3. `038f3f0` — `--save-nj-centroids`

**Final CLI options added:**

| Option | Default | Description |
|---|---|---|
| `--embeddings` | None | embeddings.csv from extract step |
| `--phylo` | False | enable Faith's PD computation |
| `--save-nj-tree` | False | write `NJ_OTU.nwk` to out-dir |
| `--nj-bootstrap` | 0 | bootstrap replicates; writes `NJ_OTU_bootstrap.nwk` |
| `--nj-bootstrap-mode` | subsample | `subsample` or `bootstrap` |
| `--nj-subsample-ratio` | 0.8 | fraction of dims per replicate |
| `--cpus` | 1 | parallel workers for bootstrap |
| `--save-nj-centroids` | False | write `NJ_OTU_centroids.csv` to out-dir |

**Output files (in `<out-dir>`):**

| File | Content |
|---|---|
| `NJ_OTU.nwk` | Unrooted NJ Newick with branch lengths |
| `NJ_OTU_bootstrap.nwk` | NJ Newick with bootstrap support labels on internal nodes |
| `NJ_OTU_centroids.csv` | L2-normalised OTU centroid embeddings (`cluster`, `dim_0`…`dim_191`) |

**Example:**
```bash
otuformer diversity \
  --assignments runs/annotate/partition_0.2_assignments.csv \
  --embeddings  runs/extract3/embeddings.csv \
  --phylo --save-nj-tree --nj-bootstrap 100 --save-nj-centroids \
  --out-dir runs/diversity
```

---

## Task 6: Full regression — run all tests

> **Status: COMPLETE**

- [x] `327f085` — fixed pre-existing test failure: `test_cluster_writes_ref_like_structure_and_csv_stats` was asserting root dir has no files; updated to check only subdirectories (cluster intentionally copies partition CSVs to root for legacy compat)
- [x] Final result: **197/197 passed**

---

## Self-Review Checklist

- [x] `build_nj_tree` covers negative branch clipping (Task 1)
- [x] `compute_nj_bootstrap_support` uses `majority_rule` — no manual clade counting (Task 1, revised)
- [x] Bootstrap `node.support` is raw count; divide by `n_replicates` × 100 for % (Task 1, bug fixed)
- [x] `compute_otu_centroids` handles single-member OTUs, missing IDs (Task 2)
- [x] `compute_pd` handles <3 OTUs → NaN (Task 3)
- [x] Faith's PD uses midpoint-rooted tree via `phydiv(rooted=True)`; MPD_w uses unrooted (Task 3, design fix)
- [x] `diversity_table` backward-compat via `tree_newick_path` fallback (Task 4)
- [x] NJ tree/bootstrap/centroids written only on first threshold (`_nj_written` flag) (Task 4)
- [x] CLI: `--save-nj-tree`, `--nj-bootstrap`, `--save-nj-centroids` all wired and tested (Task 5)
- [x] Pre-existing test failure fixed (`327f085`) — 197/197 passing (Task 6)
- [x] All function signatures consistent across tasks
- [x] No placeholders — implementation complete

## Commit Log

| Commit | Description |
|---|---|
| `ea813e3` | feat(tree): add NJ bootstrap via skbio majority_rule, remove manual clade counting |
| `edea5d3` | docs: update plan (Task 1) |
| `0d13c74` | feat(diversity): add compute_otu_centroids |
| `39c39dd` | feat(diversity): add compute_pd using OTU-centroid NJ tree with midpoint rooting |
| `ea0ddd8` | feat(diversity): update diversity_table to accept embeddings for NJ PD |
| `cb9672e` | feat(cli): add --embeddings option to diversity command for NJ PD |
| `327f085` | fix(test): check only subdirs in cluster root dir assertion |
| `32f4fa9` | feat(diversity): add --save-nj-tree and --nj-bootstrap options |
| `038f3f0` | feat(diversity): add --save-nj-centroids to export OTU centroid embeddings CSV |
