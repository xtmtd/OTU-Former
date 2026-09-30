# otuformer annotate

[English](annotate.md) | [中文](annotate.cn.md)

## Purpose

Apply expert corrections to a partition produced by `cluster` and write them back as refined OTU assignments. Rows listed in the corrections CSV override the original cluster; every other row is kept unchanged. With embeddings it also recomputes intra-class distances and draws an annotated UPGMA tree.

## Usage

```bash
otuformer annotate \
    --raw-assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --corrections corrections.csv \
    --out-dir runs/annotate
```

## Parameters

### `--raw-assignments`

**Default:** `Required`

Raw partition assignment CSV from cluster output: .../UPGMA/partitions/tables/partition_<cutoff>_assignments.csv

### `--corrections`

**Default:** `Required`

Corrections CSV. Recommended: edit from raw assignments. Minimum required columns: id (or image), cluster.

### `--embeddings`

**Default:** `None`

Optional embeddings CSV for distance recomputation. If omitted, both pairwise_distance_summary_intra-class.csv and UPGMA_tree_partitions_annotated.pdf are skipped.

### `--support-display-cutoff`

**Default:** `50.0`

Only display bootstrap support labels >= this value in annotated UPGMA PDF.

### `--figure-width`

**Default:** `None`

Optional width (inches) for annotated UPGMA PDF. Height is still driven by tip count.

### `--annotate-bar-width`

**Default:** `0.08`

Relative width of corrected OTU color bars in annotated UPGMA PDF.

### `--show-annotation-bar`

**Default:** `No`

Show corrected OTU annotation bars in annotated UPGMA PDF.

### `--show-partitioning-bars`

**Default:** `No`

Show partitioning bars in annotated UPGMA PDF.

### `--out-dir`

**Default:** `runs/annotate`

Output directory.

### `--overwrite`

**Default:** `No`

Clear an existing non-empty output directory.

## Inputs

`--raw-assignments` is a raw partition assignment CSV from `cluster`. `--corrections` needs at least `id` (or `image`) plus a corrected cluster column; editing a copy of the raw assignments is recommended. Optional `--embeddings` enables intra-class distance recomputation.

## Outputs

- `partition_<cutoff>_assignments.csv` — Corrected assignments
- `partition_<cutoff>_assignments_changed_only.csv` — Changed rows only
- `otu_table.csv` — OTU table
- `pairwise_distance_summary_intra-class.csv` — Intra-class distance summary (with `--embeddings`)
- `UPGMA_tree_partitions_annotated.pdf` — Annotated UPGMA tree (with `--embeddings`)
- `annotation_summary.json` — Correction summary

## Examples

```bash
# Basic usage
otuformer annotate \
    --raw-assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --corrections corrections.csv \
    --out-dir runs/annotate

# With embeddings (recompute intra-class distances and annotated UPGMA tree)
otuformer annotate \
    --raw-assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --corrections corrections.csv \
    --embeddings runs/extract/embeddings.csv \
    --show-annotation-bar \
    --out-dir runs/annotate
```
