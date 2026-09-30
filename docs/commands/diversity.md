# otuformer diversity

[English](diversity.md) | [中文](diversity.cn.md)

## Purpose

Compute community alpha diversity from a partition assignment CSV or an OTU table. Richness, Chao1, ACE, Shannon, Simpson, Hill numbers, and Pielou evenness are written for every abundance threshold. Faith's PD is available through `--phylo`, preferably from OTU-centroid embeddings.

## Usage

```bash
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --out-dir runs/diversity
```

## Parameters

### `--assignments`

**Default:** `None`

Partition assignment CSV (id/image, cluster, optional sample).

### `--otu-table-csv`

**Default:** `None`

OTU table CSV (sample column + OTU ID headers).

### `--otu-table-has-header`

**Default:** `No`

Force first row of OTU table as header (required if OTU IDs are numeric).

### `--out-dir`

**Default:** `runs/diversity`

Output directory.

### `--min-abundance`

**Default:** `0,2,5`

Comma-separated min abundance thresholds.

### `--phylo`

**Default:** `No`

Compute phylogenetic diversity (Faith's PD). Provide --embeddings (recommended, NJ tree built from OTU centroids) or --tree (legacy UPGMA Newick).

### `--embeddings`

**Default:** `None`

embeddings.csv from the extract step. When provided with --phylo, an OTU-centroid NJ tree is built automatically for Faith's PD. Recommended over --tree.

### `--tree`

**Default:** `None`

Legacy: Newick tree path for Faith's PD (used when --embeddings is absent).

### `--save-nj-tree`

**Default:** `No`

Save the OTU-centroid NJ Newick to <out-dir>/NJ_OTU.nwk.

### `--nj-bootstrap`

**Default:** `0`

Number of bootstrap replicates for the NJ tree (0 = disabled). Writes annotated consensus Newick to <out-dir>/NJ_OTU_bootstrap.nwk.

### `--nj-bootstrap-mode`

**Default:** `subsample` · **Allowed:** subsample / bootstrap

Bootstrap mode: 'subsample' (default) or 'bootstrap'.

### `--nj-subsample-ratio`

**Default:** `0.8`

Fraction of embedding dims per bootstrap replicate (subsample mode).

### `--cpus`

**Default:** `1`

Parallel workers for NJ bootstrap.

### `--save-nj-centroids`

**Default:** `No`

Save OTU centroid embeddings to <out-dir>/NJ_OTU_centroids.csv.

### `--overwrite`

**Default:** `No`

Clear an existing non-empty output directory.

## Inputs

Exactly one input mode is required: `--assignments` (partition assignment CSV) or `--otu-table-csv` (sample column plus OTU ID columns). `--phylo` takes optional `--embeddings`; `--tree` is the legacy Faith's PD path used when embeddings are absent.

## Outputs

- `diversity_indices.csv` — Global diversity indices; one row per index and one column per `--min-abundance` threshold.
- `per-sample/` — Per-sample diversity files (when a valid sample column exists).
- `NJ_OTU.nwk` — OTU-centroid NJ tree (with `--embeddings --save-nj-tree`).
- `NJ_OTU_bootstrap.nwk` — NJ tree with bootstrap support labels (with `--nj-bootstrap > 0`).
- `NJ_OTU_centroids.csv` — OTU centroid embeddings (with `--save-nj-centroids`).

The definitions of the computed indices are in **Notes** below.

## Examples

```bash
# From partition assignments
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --out-dir runs/diversity

# With Faith's PD from OTU-centroid NJ (recommended)
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --phylo \
    --embeddings runs/extract/embeddings.csv \
    --save-nj-tree \
    --nj-bootstrap 100 \
    --out-dir runs/diversity

# Legacy Faith's PD path from an existing Newick tree
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --phylo \
    --tree runs/cluster/UPGMA/UPGMA_Cosine.nwk \
    --out-dir runs/diversity

# From OTU table
otuformer diversity \
    --otu-table-csv otu_table.csv \
    --out-dir runs/diversity
```

## Notes

**Metrics.** Every index below is written to `diversity_indices.csv`, and per sample to `per-sample/` when sample labels are valid:

- `Richness` — number of unique OTUs.
- `Chao1` — estimated richness (accounts for rare OTUs).
- `ACE` — abundance-based coverage estimator.
- `Shannon` — entropy-based diversity (higher = more diverse).
- `Simpson` — probability two individuals differ (higher = more diverse).
- `Hill_q0`, `Hill_q1`, `Hill_q2` — Hill numbers (richness/evenness/diversity at orders 0, 1, 2).
- `Pielou_J` — evenness (`Shannon / log(richness)`).
- Faith's PD (MPD) — morphological phylogenetic diversity, computed as the sum of branch lengths of the minimal spanning subtree via `scikit-bio` `alpha_diversity('faith_pd')`.
- `MPD_w` — abundance-weighted rooted PD (rPD_w), branches weighted by relative abundance of descending taxa.
- `PD_richness_norm` — Faith's PD divided by species richness (PD per species).
