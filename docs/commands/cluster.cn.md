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

用于划分质量评估的可选标签 CSV（支持 id/label 或 image/label）。空白标签视为未标注并被忽略；同一 ID 的相同标签行会合并为一条分配；同一 ID 出现两个不同标签时，会在 `--overwrite` 清空既有报告之前报错。带标签运行还会拒绝会被数值读取改写的 ID（例如 `01` 被读成 `1`），否则匹配可能把不同标本配成一对；cutoff、抽样种子、`--pca-components` 与 local-scaling 的 `k` 也会在清空任何内容之前完成校验。两张输入表都必须列数规整：字段数与表头不一致的记录会被拒绝，而不是被移位解释；空行与仅含空白字符的行会被忽略。压缩表的校验和读取使用相同的解压规则；使用 subsample 估计分支支持度时，抽样维数若不是有限值，或超过 PCA 后的有效维数，会在清空任何内容之前被拒绝。

### `--metrics-sample-size`

**默认值：** `10000`

指标计算的最大样本数。该上限决定分区列所使用的标签匹配队列；`monophyly_proportion` 在固定输出树上评分，只要写出报告，其数值就不随上限变化。应用上限后仍少于两个带标签样本时不会生成报告。

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

`UPGMA/metrics.csv` 每个 cutoff 一行，只有当应用 `--metrics-sample-size` 上限后仍有至少两个带标签嵌入匹配时才写出。字段分组如下：

| 字段组 | 定义 |
| --- | --- |
| `NMI`、`ARI`、`AMI`、`V_measure`、`BCubed_*`、`purity`、`n_OTUs`、`OTU_species_ratio`、`splitting_index`、`lumping_index`、`silhouette_*`、`WSS` | 单个 cutoff 的分区质量，在 `--metrics-sample-size` 上限作用后的带标签队列上计算。 |
| `monophyly_proportion` | 带标签物种中，其带标签叶恰好构成固定输出树某一演化支全部带标签后代的物种占比。只评估带标签的叶，因此未标注的叶既不会并入也不会拆散某个物种；被表示的单一物种仍计为单系；分母为被匹配标签表示的物种数。与 `--metrics-sample-size`、`--random-state` 和 cutoff 无关。 |

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
