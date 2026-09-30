# otuformer diversity

[English](diversity.md) | [中文](diversity.cn.md)

## 目的

从分区分配 CSV 或 OTU 表计算群落 alpha 多样性。每个丰度阈值都会输出 Richness、Chao1、ACE、Shannon、Simpson、Hill 数与 Pielou 均匀度。Faith's PD 通过 `--phylo` 计算，推荐基于 OTU 质心嵌入。

## 用法

```bash
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --out-dir runs/diversity
```

## 参数

### `--assignments`

**默认值：** `None`

分区分配 CSV（id/image、cluster，可选 sample）。

### `--otu-table-csv`

**默认值：** `None`

OTU 表 CSV（样本列 + OTU ID 表头）。

### `--otu-table-has-header`

**默认值：** `No`

强制将 OTU 表首行作为表头（OTU ID 为数字时必需）。

### `--out-dir`

**默认值：** `runs/diversity`

输出目录。

### `--min-abundance`

**默认值：** `0,2,5`

逗号分隔的最小丰度阈值。

### `--phylo`

**默认值：** `No`

计算系统发育多样性（Faith's PD）。提供 --embeddings（推荐，由 OTU 质心构建 NJ 树）或 --tree（旧 UPGMA Newick）。

### `--embeddings`

**默认值：** `None`

来自 extract 步骤的 embeddings.csv。与 --phylo 一起提供时，会自动构建 OTU 质心 NJ 树用于 Faith's PD。比 --tree 更推荐。

### `--tree`

**默认值：** `None`

旧方式：用于 Faith's PD 的 Newick 树路径（无 --embeddings 时使用）。

### `--save-nj-tree`

**默认值：** `No`

将 OTU 质心 NJ Newick 保存到 <out-dir>/NJ_OTU.nwk。

### `--nj-bootstrap`

**默认值：** `0`

NJ 树的自助复制次数（0 = 关闭）。将带标注的共识 Newick 写入 <out-dir>/NJ_OTU_bootstrap.nwk。

### `--nj-bootstrap-mode`

**默认值：** `subsample` · **允许值：** subsample / bootstrap

自助法模式：'subsample'（默认）或 'bootstrap'。

### `--nj-subsample-ratio`

**默认值：** `0.8`

每个自助复制抽样的嵌入维度比例（subsample 模式）。

### `--cpus`

**默认值：** `1`

NJ 自助法的并行工作进程数。

### `--save-nj-centroids`

**默认值：** `No`

将 OTU 质心嵌入保存到 <out-dir>/NJ_OTU_centroids.csv。

### `--overwrite`

**默认值：** `No`

清空已存在的非空输出目录。

## 输入

必须二选一提供输入：`--assignments`（划分分配 CSV）或 `--otu-table-csv`（样本列加 OTU ID 列）。`--phylo` 可搭配 `--embeddings`；`--tree` 是无嵌入时的旧 Faith's PD 路径。

## 输出

- `diversity_indices.csv` — 全局多样性指数；每个指数一行，每个 `--min-abundance` 阈值一列。
- `per-sample/` — 按样本输出的多样性文件（存在有效 sample 列时）。
- `NJ_OTU.nwk` — OTU 质心 NJ 树（配合 `--embeddings --save-nj-tree`）。
- `NJ_OTU_bootstrap.nwk` — 带自举支持度标注的 NJ 树（配合 `--nj-bootstrap > 0`）。
- `NJ_OTU_centroids.csv` — OTU 质心嵌入（配合 `--save-nj-centroids`）。

各指标的定义见下方**说明**。

## 示例

```bash
# 从分区分配计算
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --out-dir runs/diversity

# 启用 Faith's PD（推荐：内部构建 OTU 质心 NJ 树）
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --phylo \
    --embeddings runs/extract/embeddings.csv \
    --save-nj-tree \
    --nj-bootstrap 100 \
    --out-dir runs/diversity

# 旧方式：基于现有 Newick 树计算 Faith's PD
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --phylo \
    --tree runs/cluster/UPGMA/UPGMA_Cosine.nwk \
    --out-dir runs/diversity

# 从 OTU 表计算
otuformer diversity \
    --otu-table-csv otu_table.csv \
    --out-dir runs/diversity
```

## 说明

**指标。** 下列指数写入 `diversity_indices.csv`；当样本标签有效时，还会按样本写入 `per-sample/`：

- `Richness` — 唯一 OTU 数。
- `Chao1` — 估计丰富度（考虑稀有 OTU）。
- `ACE` — 基于丰度的覆盖度估计量。
- `Shannon` — 基于熵的多样性（越高越多样）。
- `Simpson` — 两个个体属于不同 OTU 的概率（越高越多样）。
- `Hill_q0`、`Hill_q1`、`Hill_q2` — Hill 数（0、1、2 阶的丰富度/均匀度/多样性）。
- `Pielou_J` — 均匀度（`Shannon / log(richness)`）。
- Faith's PD (MPD) — 形态学系统发育多样性，通过 `scikit-bio` 的 `alpha_diversity('faith_pd')` 计算最小生成子树的枝长之和。
- `MPD_w` — 丰度加权的有根 PD（rPD_w），枝长按下行类群的相对丰度加权。
- `PD_richness_norm` — Faith's PD 除以物种丰富度（每个物种的 PD）。
