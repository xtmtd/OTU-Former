# otuformer annotate

[English](annotate.md) | [中文](annotate.cn.md)

## 目的

把专家校正应用到 `cluster` 产出的划分上，回写为精化后的 OTU 分配。校正 CSV 中列出的行会覆盖原 cluster，其余行保持不变。提供嵌入向量时，还会重新计算类内距离并绘制带标注的 UPGMA 树。

## 用法

```bash
otuformer annotate \
    --raw-assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --corrections corrections.csv \
    --out-dir runs/annotate
```

## 参数

### `--raw-assignments`

**默认值：** `Required`

来自 cluster 输出的原始分区分配 CSV：.../UPGMA/partitions/tables/partition_<cutoff>_assignments.csv

### `--corrections`

**默认值：** `Required`

校正 CSV。推荐从原始分配文件开始编辑。最少需要 id（或 image）与 cluster 列。

### `--embeddings`

**默认值：** `None`

用于重新计算距离的可选嵌入 CSV。省略时会跳过 pairwise_distance_summary_intra-class.csv 与 UPGMA_tree_partitions_annotated.pdf。

### `--support-display-cutoff`

**默认值：** `50.0`

只在带标注的 UPGMA PDF 中显示 >= 该值的自举支持度标签。

### `--figure-width`

**默认值：** `None`

带标注 UPGMA PDF 的可选宽度（英寸）。高度仍由末端数量决定。

### `--annotate-bar-width`

**默认值：** `0.08`

带标注 UPGMA PDF 中修正后 OTU 色条的相对宽度。

### `--show-annotation-bar`

**默认值：** `No`

在带标注的 UPGMA PDF 中显示修正后的 OTU 标注条。

### `--show-partitioning-bars`

**默认值：** `No`

在带标注的 UPGMA PDF 中显示分区条。

### `--out-dir`

**默认值：** `runs/annotate`

输出目录。

### `--overwrite`

**默认值：** `No`

清空已存在的非空输出目录。

## 输入

`--raw-assignments` 是来自 `cluster` 的原始划分分配 CSV。`--corrections` 至少需要 `id`（或 `image`）与修正后的 cluster 列；建议从原始分配副本开始编辑。可选的 `--embeddings` 用于重新计算类内距离。

## 输出

- `partition_<cutoff>_assignments.csv` — 校正后的分配
- `partition_<cutoff>_assignments_changed_only.csv` — 仅变更行
- `otu_table.csv` — OTU 表
- `pairwise_distance_summary_intra-class.csv` — 类内距离摘要（提供 `--embeddings` 时）
- `UPGMA_tree_partitions_annotated.pdf` — 标注 UPGMA 树（提供 `--embeddings` 时）
- `annotation_summary.json` — 校正摘要

## 示例

```bash
# 基本用法
otuformer annotate \
    --raw-assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --corrections corrections.csv \
    --out-dir runs/annotate

# 带嵌入向量（重新计算类内距离与标注 UPGMA 树）
otuformer annotate \
    --raw-assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --corrections corrections.csv \
    --embeddings runs/extract/embeddings.csv \
    --show-annotation-bar \
    --out-dir runs/annotate
```
