# otuformer cluster

[English](cluster.md) | [中文](cluster.cn.md)

## Purpose

Cluster embeddings into morphological OTUs (morphOTUs) by UPGMA hierarchical clustering. It builds a pairwise distance matrix, constructs the dendrogram, and partitions it across a scan of distance cutoffs, with optional PCA whitening, local scaling, and bootstrap support estimation.

## Usage

```bash
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --out-dir runs/cluster
```

## Parameters

### `--embeddings`

**Default:** `Required`

Embeddings CSV.

### `--out-dir`

**Default:** `runs/cluster`

Output directory.

### `--distance`

**Default:** `cosine` · **Allowed:** cosine / euclidean

Distance metric for pairwise matrix: cosine or euclidean.

### `--prefix`

**Default:** `OTU`

Cluster prefix used in partition table labels.

### `--pca-whitening`

**Default:** `false` · **Allowed:** true / false

Enable PCA whitening before distance computation (true/false).

### `--pca-components`

**Default:** `256`

PCA components.

### `--local-scaling`

**Default:** `false` · **Allowed:** true / false

Enable Mutual-Proximity style local scaling on distance matrix (true/false).

### `--local-k`

**Default:** `0`

Fixed k for local scaling (0=auto).

### `--local-k-strategy`

**Default:** `adaptive` · **Allowed:** adaptive / sqrt / log / fixed

Auto-k strategy when --local-k=0: adaptive / sqrt / log / fixed.

### `--cutoff-min`

**Default:** `0.05`

Minimum cutoff.

### `--cutoff-max`

**Default:** `1.0`

Maximum cutoff (inclusive).

### `--cutoff-step`

**Default:** `0.05`

Cutoff step for linear scan when --custom-cutoffs is not provided.

### `--custom-cutoffs`

**Default:** `None`

Comma-separated cutoffs (overrides min/max/step scan).

### `--support-mode`

**Default:** `subsample` · **Allowed:** subsample / bootstrap

Support estimation mode: subsample (without replacement) or bootstrap (with replacement).

### `--num-replicates`

**Default:** `0`

Number of support estimation replicates (0 disables).

### `--subsample-ratio`

**Default:** `0.8`

Feature fraction for subsample mode (ignored in bootstrap mode).

### `--support-display-cutoff`

**Default:** `50.0`

Display threshold for support labels on tree visualizations.

### `--save-bootstrap-trees`

**Default:** `false` · **Allowed:** true / false

Save all bootstrap replicate trees to UPGMA/bootstrap_trees.nwk (true/false).

### `--save-distances`

**Default:** `false` · **Allowed:** true / false

Save full pairwise distance matrix to distance_statistics/distance_matrix.csv (true/false).

### `--max-distance-pairs`

**Default:** `1000000`

Max pairs to store.

### `--label-csv`, `--labels`

**Default:** `None`

Optional label CSV for partition metrics (supports id/label or image/label). IDs and labels are read as text, so `01` and `1` are distinct. Embedding IDs must be nonempty and unique. Blank labels are ignored; identical label rows are collapsed, and conflicting labels for one ID are rejected before `--overwrite` can clear an existing report.

### `--metrics-sample-size`

**Default:** `10000`

Max samples for metrics.

### `--cpus`

**Default:** `8`

CPU threads.

### `--random-state`

**Default:** `42`

Random seed.

### `--overwrite`

**Default:** `No`

Clear an existing non-empty output directory.

## Inputs

`--embeddings` is an `embeddings.csv` from `extract`. Optional `--label-csv` (alias `--labels`) accepts `id/label` or `image/label` for partition metrics.

## Outputs

- `UPGMA/UPGMA_Cosine.nwk` — Newick format phylogenetic tree
- `UPGMA/partitions/partition_scan.csv` — Threshold scan results
- `UPGMA/partitions/tables/partition_<cutoff>_assignments.csv` — OTU assignments at each cutoff
- `UPGMA/partitions/UPGMA_tree_partitions.pdf` — Tree and partition visualization
- `UPGMA/metrics.csv` — Partition quality metrics (when `--label-csv` is provided)
- `UPGMA/metrics_dashboard.pdf` — Metrics dashboard (when `--label-csv` is provided)
- `distance_statistics/` — Distance statistics and distribution plots

### `metrics.csv` fields

`UPGMA/metrics.csv` holds one row per cutoff. The partition columns describe the
metrics linkage rebuilt from the aligned labeled cohort and its seeded
`--metrics-sample-size` subset. `monophyly_proportion` is not a partition
column: it is computed once on the fixed output UPGMA tree and repeated across
rows.

| Field | Definition |
| --- | --- |
| `monophyly_proportion` | Share of labeled species whose labeled tips are exactly the labeled descendants of one clade of the fixed output tree. Under partial coverage only labeled tips are scored: an unlabeled tip neither joins nor breaks a species, and a represented singleton species counts as monophyletic. The denominator is the number of species represented by a matched label. Independent of `--metrics-sample-size`, the sampling seed and the cutoff. |

This value is a property of the fixed output topology and the labels, not of the
sampled partition tree; see the [workflow execution efficiency design](../superpowers/specs/2026-11-18-workflow-execution-efficiency-design.md)
and its [implementation plan](../superpowers/plans/2026-11-19-workflow-execution-efficiency.md)
(2026-11-19).

## Examples

```bash
# Basic usage
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --out-dir runs/cluster

# With PCA whitening, local scaling, and bootstrap support
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --pca-whitening true \
    --local-scaling true \
    --num-replicates 100 \
    --out-dir runs/cluster

# Custom distance metric and cutoff range
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --distance euclidean \
    --cutoff-min 0.1 \
    --cutoff-max 0.8 \
    --cutoff-step 0.02 \
    --out-dir runs/cluster

# With label CSV for partition quality evaluation
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --label-csv labels.csv \
    --out-dir runs/cluster
```
