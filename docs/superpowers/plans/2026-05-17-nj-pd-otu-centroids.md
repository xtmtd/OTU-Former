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

**Files:**
- Modify: `src/otuformer/delineation/diversity.py`
- Test: `tests/test_diversity.py`

This function joins `assignments` with `embeddings`, groups by OTU, computes mean embedding, and L2-normalises each centroid.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_diversity.py`:

```python
def test_compute_otu_centroids_shape():
    from otuformer.delineation.diversity import compute_otu_centroids
    assignments = pd.DataFrame({
        "id":      ["i1", "i2", "i3", "i4"],
        "cluster": ["OTU_1", "OTU_1", "OTU_2", "OTU_2"],
    })
    embeddings = pd.DataFrame({
        "id":    ["i1", "i2", "i3", "i4"],
        "dim_0": [1.0, 3.0, 0.0, 0.0],
        "dim_1": [0.0, 0.0, 1.0, 3.0],
    })
    centroids, otu_ids = compute_otu_centroids(assignments, embeddings)
    assert centroids.shape == (2, 2)
    assert set(otu_ids) == {"OTU_1", "OTU_2"}


def test_compute_otu_centroids_l2_normalised():
    from otuformer.delineation.diversity import compute_otu_centroids
    assignments = pd.DataFrame({
        "id":      ["i1", "i2"],
        "cluster": ["OTU_A", "OTU_B"],
    })
    embeddings = pd.DataFrame({
        "id":    ["i1", "i2"],
        "dim_0": [3.0, 0.0],
        "dim_1": [4.0, 5.0],
    })
    centroids, _ = compute_otu_centroids(assignments, embeddings)
    norms = np.linalg.norm(centroids, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_compute_otu_centroids_missing_ids_ignored():
    """Individuals in assignments but absent from embeddings are silently dropped."""
    from otuformer.delineation.diversity import compute_otu_centroids
    assignments = pd.DataFrame({
        "id":      ["i1", "i2", "MISSING"],
        "cluster": ["OTU_1", "OTU_1", "OTU_1"],
    })
    embeddings = pd.DataFrame({
        "id":    ["i1", "i2"],
        "dim_0": [1.0, 1.0],
        "dim_1": [0.0, 0.0],
    })
    centroids, otu_ids = compute_otu_centroids(assignments, embeddings)
    assert centroids.shape == (1, 2)
    assert otu_ids == ["OTU_1"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_diversity.py::test_compute_otu_centroids_shape tests/test_diversity.py::test_compute_otu_centroids_l2_normalised tests/test_diversity.py::test_compute_otu_centroids_missing_ids_ignored -v
```

Expected: `ImportError` — `compute_otu_centroids` not defined.

- [ ] **Step 3: Implement `compute_otu_centroids`**

In `src/otuformer/delineation/diversity.py`, add after the imports block (before line 14 where module-level code begins):

```python
def compute_otu_centroids(
    assignments: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Compute L2-normalised centroid embedding for each OTU.

    Parameters
    ----------
    assignments:
        DataFrame with columns ``id`` and ``cluster``.
    embeddings:
        DataFrame with column ``id`` and one column per embedding dimension
        (named ``dim_0``, ``dim_1``, … or any non-id, non-sample column).

    Returns
    -------
    centroids : np.ndarray, shape (n_otus, n_dims)
        Row-wise L2-normalised centroid embeddings, one row per OTU.
    otu_ids : list[str]
        OTU labels in the same row order as ``centroids``.
    """
    # Identify embedding dimension columns (everything except id/sample)
    meta_cols = {"id", "sample"}
    dim_cols = [c for c in embeddings.columns if c not in meta_cols]

    merged = assignments[["id", "cluster"]].merge(
        embeddings[["id"] + dim_cols], on="id", how="inner"
    )
    grouped = merged.groupby("cluster", sort=True)[dim_cols].mean()
    otu_ids: list[str] = list(grouped.index)
    raw = grouped.values.astype(np.float64)

    # L2-normalise each row (sphere embedding)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    centroids = raw / norms
    return centroids, otu_ids
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_diversity.py::test_compute_otu_centroids_shape tests/test_diversity.py::test_compute_otu_centroids_l2_normalised tests/test_diversity.py::test_compute_otu_centroids_missing_ids_ignored -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/diversity.py tests/test_diversity.py
git commit -m "feat(diversity): add compute_otu_centroids"
```

---

## Task 3: Replace `compute_mpd_from_counts` and `compute_mpd` with NJ-centroid pipeline

**Files:**
- Modify: `src/otuformer/delineation/diversity.py`
- Test: `tests/test_diversity.py`

Old interface consumed an UPGMA Newick tree file. New interface consumes `assignments` + `embeddings` DataFrames and builds the NJ tree internally.

- [ ] **Step 1: Write failing tests for new `compute_pd`**

Append to `tests/test_diversity.py`:

```python
# ---------------------------------------------------------------------------
# New NJ-centroid PD tests
# ---------------------------------------------------------------------------

def _make_assignments_and_embeddings(n_otus: int = 4, members_per_otu: int = 3, n_dims: int = 8, seed: int = 0):
    """Helper: synthetic assignments + embeddings with n_otus distinct clusters."""
    rng = np.random.default_rng(seed)
    rows_a, rows_e = [], []
    idx = 0
    for otu_i in range(n_otus):
        center = rng.standard_normal(n_dims)
        for _ in range(members_per_otu):
            iid = f"img_{idx:04d}"
            emb = center + rng.standard_normal(n_dims) * 0.05
            rows_a.append({"id": iid, "cluster": f"OTU_{otu_i + 1}"})
            row_e = {"id": iid}
            row_e.update({f"dim_{d}": float(emb[d]) for d in range(n_dims)})
            rows_e.append(row_e)
            idx += 1
    return pd.DataFrame(rows_a), pd.DataFrame(rows_e)


def test_compute_pd_returns_three_keys():
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_pd
    assignments, embeddings = _make_assignments_and_embeddings()
    result = compute_pd(assignments, embeddings)
    assert set(result.keys()) == {"MPD", "MPD_w", "PD_richness_norm"}


def test_compute_pd_values_are_finite():
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_pd
    assignments, embeddings = _make_assignments_and_embeddings()
    result = compute_pd(assignments, embeddings)
    for k, v in result.items():
        assert np.isfinite(v), f"{k} = {v} is not finite"


def test_compute_pd_single_otu_returns_nan():
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_pd
    assignments = pd.DataFrame({"id": ["i1", "i2"], "cluster": ["OTU_1", "OTU_1"]})
    embeddings = pd.DataFrame({"id": ["i1", "i2"], "dim_0": [1.0, 1.0], "dim_1": [0.0, 0.0]})
    result = compute_pd(assignments, embeddings)
    # NJ requires ≥3 taxa; single OTU → all NaN
    for v in result.values():
        assert np.isnan(v)


def test_compute_pd_pd_richness_norm_relation():
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import compute_pd
    assignments, embeddings = _make_assignments_and_embeddings(n_otus=5)
    result = compute_pd(assignments, embeddings)
    n_otus = assignments["cluster"].nunique()
    expected_norm = result["MPD"] / n_otus
    assert abs(result["PD_richness_norm"] - expected_norm) < 1e-9
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_diversity.py::test_compute_pd_returns_three_keys tests/test_diversity.py::test_compute_pd_values_are_finite tests/test_diversity.py::test_compute_pd_single_otu_returns_nan tests/test_diversity.py::test_compute_pd_pd_richness_norm_relation -v
```

Expected: `ImportError` — `compute_pd` not defined.

- [ ] **Step 3: Implement `compute_pd` in `diversity.py`**

Add this function after `compute_otu_centroids` (before `compute_mpd_from_counts`):

```python
def compute_pd(
    assignments: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> dict[str, float]:
    """Compute Faith's PD and related metrics using an OTU-centroid NJ tree.

    Builds one NJ tree per call from OTU centroid cosine distances.
    Requires ≥ 3 OTUs; returns all-NaN dict otherwise.

    Parameters
    ----------
    assignments:
        DataFrame with columns ``id`` and ``cluster``.
    embeddings:
        DataFrame with column ``id`` plus embedding dimension columns.

    Returns
    -------
    dict with keys ``"MPD"`` (Faith's PD), ``"MPD_w"`` (abundance-weighted
    rooted PD), ``"PD_richness_norm"`` (Faith PD / OTU richness).
    """
    _nan = {"MPD": float("nan"), "MPD_w": float("nan"), "PD_richness_norm": float("nan")}

    try:
        from skbio import DistanceMatrix
        from skbio.diversity import alpha_diversity
        from skbio.diversity.alpha import phydiv
    except ImportError:
        return _nan

    try:
        from otuformer.delineation.tree import build_nj_tree
        from otuformer.delineation.distance import compute_cosine_distances

        centroids, otu_ids = compute_otu_centroids(assignments, embeddings)

        if len(otu_ids) < 3:
            return _nan

        # Cosine distance between OTU centroids (already L2-normalised)
        dist_matrix = compute_cosine_distances(centroids)

        tree = build_nj_tree(dist_matrix, otu_ids)

        # OTU abundances (number of individuals per OTU)
        counts_series = assignments["cluster"].value_counts()
        counts_arr = np.array(
            [counts_series.get(otu, 0) for otu in otu_ids], dtype=int
        )
        counts_matrix = counts_arr.reshape(1, -1)

        richness = len(otu_ids)

        faith_vals = alpha_diversity("faith_pd", counts_matrix, otu_ids, tree=tree)
        faith_pd = float(faith_vals[0])

        mpd_w = float(phydiv(counts_arr, otu_ids, tree, rooted=False, weight=True))

        pd_richness_norm = faith_pd / richness

        return {
            "MPD": faith_pd,
            "MPD_w": mpd_w,
            "PD_richness_norm": pd_richness_norm,
        }

    except Exception:
        return _nan
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_diversity.py::test_compute_pd_returns_three_keys tests/test_diversity.py::test_compute_pd_values_are_finite tests/test_diversity.py::test_compute_pd_single_otu_returns_nan tests/test_diversity.py::test_compute_pd_pd_richness_norm_relation -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/diversity.py tests/test_diversity.py
git commit -m "feat(diversity): add compute_pd using OTU-centroid NJ tree"
```

---

## Task 4: Update `diversity_table` to use `compute_pd`

**Files:**
- Modify: `src/otuformer/delineation/diversity.py`
- Test: `tests/test_diversity.py`

`diversity_table` currently accepts `tree_newick_path: Optional[Path]`. Replace this with `embeddings: Optional[pd.DataFrame]`. The old UPGMA-tree path is removed from this function.

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_diversity.py`:

```python
def test_diversity_table_with_embeddings_contains_pd_metrics():
    pytest.importorskip("skbio")
    from otuformer.delineation.diversity import diversity_table
    assignments, embeddings = _make_assignments_and_embeddings(n_otus=4)
    df = diversity_table(assignments, min_abundances=[1], embeddings=embeddings)
    assert "MPD" in df.index
    assert "MPD_w" in df.index
    assert "PD_richness_norm" in df.index


def test_diversity_table_no_pd_when_no_embeddings():
    from otuformer.delineation.diversity import diversity_table
    assignments, _ = _make_assignments_and_embeddings(n_otus=4)
    df = diversity_table(assignments, min_abundances=[1])
    assert "MPD" not in df.index
    assert "MPD_w" not in df.index
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_diversity.py::test_diversity_table_with_embeddings_contains_pd_metrics tests/test_diversity.py::test_diversity_table_no_pd_when_no_embeddings -v
```

Expected: `TypeError` — `diversity_table` doesn't accept `embeddings` kwarg yet.

- [ ] **Step 3: Update `diversity_table` signature and body**

Locate `diversity_table` in `src/otuformer/delineation/diversity.py` (lines 360–384). Replace the entire function:

```python
def diversity_table(
    assignments: pd.DataFrame,
    min_abundances: list[int],
    embeddings: Optional[pd.DataFrame] = None,
    tree_newick_path: Optional[Path] = None,  # kept for backward compat, ignored when embeddings provided
) -> pd.DataFrame:
    """Compute alpha diversity metrics across abundance thresholds.

    Parameters
    ----------
    assignments:
        DataFrame with columns ``id``, ``cluster``, ``sample``.
    min_abundances:
        List of minimum-abundance thresholds; one result column per value.
    embeddings:
        When provided, OTU-centroid NJ tree is built and Faith's PD metrics
        (``MPD``, ``MPD_w``, ``PD_richness_norm``) are added.
    tree_newick_path:
        Deprecated.  Ignored when ``embeddings`` is provided.  When
        ``embeddings`` is None and this path is given the old UPGMA-tree
        path is used as fallback (legacy behaviour).
    """
    records: list[dict] = []
    for min_ab in min_abundances:
        col_label = f"min_abundance_{min_ab}"
        otu_counts = assignments["cluster"].value_counts()
        keep = otu_counts[otu_counts >= min_ab].index
        filtered = assignments[assignments["cluster"].isin(keep)]

        metrics: dict[str, float] = compute_alpha_diversity(filtered)

        if embeddings is not None:
            try:
                pd_metrics = compute_pd(filtered, embeddings)
                metrics.update(pd_metrics)
            except Exception:
                metrics.update({"MPD": float("nan"), "MPD_w": float("nan"), "PD_richness_norm": float("nan")})
        elif tree_newick_path is not None:
            # Legacy fallback
            try:
                pd_metrics = compute_mpd(filtered, tree_newick_path)
                metrics.update(pd_metrics)
            except Exception:
                metrics.update({"MPD": float("nan"), "MPD_w": float("nan"), "PD_richness_norm": float("nan")})

        records.append({"threshold": col_label, **metrics})

    df = pd.DataFrame(records).set_index("threshold").T
    df.index.name = "index"
    return df
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_diversity.py::test_diversity_table_with_embeddings_contains_pd_metrics tests/test_diversity.py::test_diversity_table_no_pd_when_no_embeddings -v
```

Expected: both PASS.

- [ ] **Step 5: Run full diversity test suite**

```bash
pytest tests/test_diversity.py -v
```

Expected: all tests PASS (legacy tests that use `tree_newick_path` still work via fallback path).

- [ ] **Step 6: Commit**

```bash
git add src/otuformer/delineation/diversity.py tests/test_diversity.py
git commit -m "feat(diversity): update diversity_table to accept embeddings for NJ PD"
```

---

## Task 5: Update CLI (`cli/diversity.py`) to pass embeddings

**Files:**
- Modify: `src/otuformer/cli/diversity.py`
- Test: manual smoke test (no unit test needed for CLI wiring)

- [ ] **Step 1: Read current CLI**

```bash
cat src/otuformer/cli/diversity.py
```

Identify: how `diversity_table` is called, what arguments are passed, whether `embeddings_path` is already an option.

- [ ] **Step 2: Add `--embeddings` option and pass DataFrame**

Find the call site for `diversity_table(...)` and update it.  The exact diff depends on what Step 1 reveals, but the pattern is:

```python
# Existing call (approximate)
df = diversity_table(assignments, min_abundances=[1, 2, 5], tree_newick_path=tree_path)

# New call
embeddings_df = pd.read_csv(embeddings_path) if embeddings_path else None
df = diversity_table(assignments, min_abundances=[1, 2, 5], embeddings=embeddings_df)
```

Add CLI option if not already present:

```python
@click.option("--embeddings", "embeddings_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="embeddings.csv from extract step; enables NJ-based Faith's PD")
```

- [ ] **Step 3: Smoke test with real data**

```bash
python -m otuformer diversity \
  --assignments runs/annotate/partition_0.2_assignments.csv \
  --embeddings  runs/extract3/embeddings.csv \
  --output /tmp/diversity_nj_test.csv
```

Expected: exits 0; `/tmp/diversity_nj_test.csv` contains rows `MPD`, `MPD_w`, `PD_richness_norm` with finite values.

```bash
python -c "
import pandas as pd
df = pd.read_csv('/tmp/diversity_nj_test.csv', index_col=0)
print(df.loc[['MPD','MPD_w','PD_richness_norm']])
"
```

Expected: three rows with non-NaN floats.

- [ ] **Step 4: Commit**

```bash
git add src/otuformer/cli/diversity.py
git commit -m "feat(cli): add --embeddings option to diversity command for NJ PD"
```

---

## Task 6: Full regression — run all tests

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: all previously passing tests still PASS; new tests PASS.

- [ ] **Step 2: If any failures, fix before proceeding**

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address regression from NJ PD integration"
```

---

## Self-Review Checklist

- [x] `build_nj_tree` covers negative branch clipping (Task 1)
- [x] `compute_nj_bootstrap_support` uses `majority_rule` — no manual clade counting (Task 1, revised)
- [x] Bootstrap `node.support` is raw count; divide by `n_replicates` × 100 for % (Task 1, bug fixed)
- [x] `compute_otu_centroids` handles single-member OTUs, missing IDs (Task 2)
- [x] `compute_pd` handles <3 OTUs → NaN (Task 3)
- [x] `compute_pd` uses `rooted=False` for NJ (unrooted) tree in `phydiv` call
- [x] `diversity_table` backward-compat via `tree_newick_path` fallback (Task 4)
- [x] CLI wiring task explicitly reads current CLI before editing (Task 5)
- [x] All function signatures consistent across tasks
- [x] No placeholders — every step has concrete code
