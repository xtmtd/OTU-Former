# otuformer cluster

[English](cluster.md) | [中文](cluster.cn.md)

## 目的

通过 UPGMA 层次聚类把嵌入向量划分为形态学 OTU（morphOTU）。它构建两两距离矩阵、生成树状图，并在距离 cutoff 扫描过程中完成划分，可选启用 PCA 白化、局部缩放与 bootstrap 支持度估计。

## 用法

```bash
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --out-dir runs/cluster
```

## 参数

### `--embeddings`

**默认值：** `Required`

嵌入 CSV。

### `--out-dir`

**默认值：** `runs/cluster`

输出目录。

### `--distance`

**默认值：** `cosine` · **允许值：** cosine / euclidean

两两距离矩阵的距离度量：cosine 或 euclidean。

### `--prefix`

**默认值：** `OTU`

用于分区表标签的 cluster 前缀。

### `--pca-whitening`

**默认值：** `false` · **允许值：** true / false

在计算距离前启用 PCA 白化（true/false）。

### `--pca-components`

**默认值：** `256`

PCA 主成分数。

### `--local-scaling`

**默认值：** `false` · **允许值：** true / false

在距离矩阵上启用 Mutual-Proximity 风格的局部缩放（true/false）。

### `--local-k`

**默认值：** `0`

局部缩放的固定 k（0=自动）。

### `--local-k-strategy`

**默认值：** `adaptive` · **允许值：** adaptive / sqrt / log / fixed

--local-k=0 时的自动 k 策略：adaptive / sqrt / log / fixed。

### `--cutoff-min`

**默认值：** `0.05`

最小 cutoff。

### `--cutoff-max`

**默认值：** `1.0`

最大 cutoff（含）。

### `--cutoff-step`

**默认值：** `0.05`

未提供 --custom-cutoffs 时线性扫描的 cutoff 步长。

### `--custom-cutoffs`

**默认值：** `None`

逗号分隔的 cutoff（覆盖 min/max/step 扫描）。

### `--support-mode`

**默认值：** `subsample` · **允许值：** subsample / bootstrap

支持度估计模式：subsample（不放回）或 bootstrap（放回）。

### `--num-replicates`

**默认值：** `0`

支持度估计的重复次数（0 表示关闭）。

### `--subsample-ratio`

**默认值：** `0.8`

subsample 模式的特征比例（bootstrap 模式下忽略）。

### `--support-display-cutoff`

**默认值：** `50.0`

树可视化中支持度标签的显示阈值。

### `--save-bootstrap-trees`

**默认值：** `false` · **允许值：** true / false

将所有自举复制树保存到 UPGMA/bootstrap_trees.nwk（true/false）。

### `--save-distances`

**默认值：** `false` · **允许值：** true / false

将完整两两距离矩阵保存到 distance_statistics/distance_matrix.csv（true/false）。

### `--max-distance-pairs`

**默认值：** `1000000`

最多存储的距离对数。

### `--label-csv`、`--labels`

**默认值：** `None`

用于划分质量评估的可选标签 CSV（支持 id/label 或 image/label）。

### `--metrics-sample-size`

**默认值：** `10000`

指标计算的最大样本数。

### `--cpus`

**默认值：** `8`

CPU 线程数。

### `--random-state`

**默认值：** `42`

随机种子。

### `--overwrite`

**默认值：** `No`

清空已存在的非空输出目录。

## 输入

`--embeddings` 是 `extract` 产出的 `embeddings.csv`。可选的 `--label-csv`（别名 `--labels`）接受 `id/label` 或 `image/label`，用于划分质量评估。

## 输出

- `UPGMA/UPGMA_Cosine.nwk` — Newick 格式系统发育树
- `UPGMA/partitions/partition_scan.csv` — 阈值扫描结果
- `UPGMA/partitions/tables/partition_<cutoff>_assignments.csv` — 各阈值下的 OTU 分配
- `UPGMA/partitions/UPGMA_tree_partitions.pdf` — 树与分区可视化
- `UPGMA/metrics.csv` — 分区质量指标（提供 `--label-csv` 时）
- `UPGMA/metrics_dashboard.pdf` — 指标面板（提供 `--label-csv` 时）
- `distance_statistics/` — 距离统计与分布图

## 示例

```bash
# 基本用法
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --out-dir runs/cluster

# 启用 PCA 白化、局部缩放与 Bootstrap 支持率
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --pca-whitening true \
    --local-scaling true \
    --num-replicates 100 \
    --out-dir runs/cluster

# 自定义距离度量与 cutoff 范围
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --distance euclidean \
    --cutoff-min 0.1 \
    --cutoff-max 0.8 \
    --cutoff-step 0.02 \
    --out-dir runs/cluster

# 带标签 CSV 用于划分质量评估
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --label-csv labels.csv \
    --out-dir runs/cluster
```
