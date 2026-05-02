# OTU-Former

**中文** | [English](README.md)

一个基于图像形态学的 OTU（操作分类单元）划分工具包。提供统一的 `otuformer` 命令行工具，支持自监督预训练、ArcFace 度量学习微调、嵌入向量提取（提供标签时可直接输出质量指标/UMAP）、UPGMA 层次聚类、专家校正标注、群落多样性分析、CAM 可视化以及 ONNX 模型导出等功能。

## 概述

所有功能通过单一入口访问：

```
otuformer <command> [options]
```

| 命令 | 描述 |
|---------|-------------|
| `doctor` | 诊断环境与依赖状态 |
| `pretrain` | 自监督对比学习预训练（DINO/iBOT 风格） |
| `finetune` | ArcFace 度量学习微调 |
| `extract` | 提取图像嵌入向量（支持 ONNX 加速） |
| `cluster` | UPGMA 层次聚类划分形态学 OTU |
| `annotate` | 应用专家校正，生成 refined OTU 标注 |
| `diversity` | 计算群落 alpha 多样性指标（Shannon、Simpson、Chao1、Faith's PD 等） |
| `cam` | 生成 GradCAM 等热力图 |
| `export` | 导出模型为 ONNX 格式 |

## 功能特性

- **统一命令行接口**：单一 `otuformer` 入口，覆盖从预训练到多样性分析的完整流程
- **自监督预训练**：基于 DINO/iBOT 风格的教师-学生 ViT 对比学习，支持全局/局部裁剪与掩码 token 一致性
- **ArcFace 度量学习微调**：利用标注数据优化模型，产出判别性嵌入用于 OTU 聚类
- **多模式嵌入提取**：支持 CLS token、patch-topk、attention-pool 三种模式，可选 ONNX 加速
- **嵌入质量评估**：NMI、ARI、Recall@K、kNN 准确率、mAP@R、轮廓系数、线性探测准确率等
- **UPGMA 层次聚类**：构建距离矩阵与系统发育树，在多个距离阈值下自动划分 OTU
- **PCA 白化与局部缩放**：可选的 PCA 白化与 Mutual-Proximity 风格局部缩放，提升聚类质量
- **Bootstrap 支持率估计**：支持 subsample/bootstrap 模式估计分支支持率
- **专家校正标注**：接受人工校正 CSV，生成 refined OTU 分配与类内距离摘要
- **群落多样性分析**：Richness、Chao1、ACE、Shannon、Simpson、Hill 数（q0/q1/q2）、Pielou 均匀度、Faith's PD（MPD）、加权 PD（MPD_w）等 alpha 多样性指标
- **模型可解释性**：支持 Grad-CAM、Grad-CAM++、LayerCAM、Score-CAM、Eigen-CAM、Ablation-CAM 六种算法
- **ONNX 导出加速**：导出为 ONNX 格式，CPU 推理提速 2-5 倍
- **UMAP 可视化**：嵌入空间降维可视化，直观展示形态学分布
- **并行处理**：多线程图像处理，可配置 CPU 线程数
- **完整日志**：所有命令自动保存运行日志到输出目录的 `logs/` 子目录
- **优雅退出**：Ctrl+C 安全中断，保留已处理结果

## 系统要求

- Python 3.11+
- 操作系统：macOS、Linux、Windows
- 设备：CPU（必选）；CUDA GPU / Apple Silicon MPS（可选）

## 安装

推荐使用隔离的 Python 环境，避免与系统/全局 site-packages 发生依赖冲突。

安装前请先克隆仓库并进入项目目录：

```bash
git clone https://github.com/xtmtd/OTU-Former.git
cd OTU-Former
```

### 隔离环境

可任选其一：

**选项 1：conda**

```bash
conda create -n otuformer python=3.11 -y
conda activate otuformer
pip install -e .
```

**选项 2：uv + venv**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

**选项 3：标准库 venv + pip**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 使用方法

推荐的工作流命令顺序：

1. `doctor` — 检查环境
2. `pretrain` — 自监督预训练
3. `finetune` — ArcFace 微调
4. `extract` — 提取嵌入向量
5. `cluster` — UPGMA 聚类为 OTU
6. `annotate` — 应用专家校正
7. `diversity` — 多样性分析

### doctor 命令

诊断环境与依赖是否满足当前功能需求。

```bash
otuformer doctor
```

报告内容包括：
- Python 版本与可用设备（`cpu`、`cuda`、`mps`）
- 关键依赖的版本与状态（ok/missing/outdated）

**输出**：
- 终端打印诊断报告

---

### pretrain 命令

基于 DINO/iBOT 风格的教师-学生 ViT 对比学习进行自监督预训练，无需标注数据。

```bash
# 基本用法
otuformer pretrain \
    --input-images-dir ./images \
    --out-dir runs/pretrain

# 使用 CSV 指定训练子集
otuformer pretrain \
    --train-data images.csv \
    --input-images-dir ./images \
    --out-dir runs/pretrain

# 自定义模型与训练参数
otuformer pretrain \
    --input-images-dir ./images \
    --model-name vit_small_patch16_224 \
    --max-epochs 100 \
    --lr 1e-3 \
    --batch-size 64 \
    --out-dir runs/pretrain
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--train-data` | 训练数据 CSV（含 `image` 列，路径相对于 `--input-images-dir`）；省略则递归使用所有图像 | 无 |
| `--input-images-dir` | 图像根目录 | 必填 |
| `--out-dir` | 输出目录 | `runs/pretrain` |
| `--model-name` | timm 骨干网络名称 | `vit_tiny_patch16_224` |
| `--out-dim` | SSL 投影器输出维度 | 256 |
| `--max-epochs` | 预训练轮数 | 50 |
| `--lr` | 基础学习率 | 5e-4 |
| `--weight-decay` | AdamW 权重衰减 | 0.05 |
| `--warmup-epochs` | 预热轮数 | 3 |
| `--global-crop-size` | 全局裁剪分辨率 | 224 |
| `--local-crop-size` | 局部裁剪分辨率 | 96 |
| `--local-crops` | 局部裁剪数量 | 6 |
| `--mask-ratio` | 掩码 token 比例 | 0.5 |
| `--lambda-local` | 局部裁剪损失权重 | 1.5 |
| `--lambda-mask` | 掩码 token 损失权重 | 1.0 |
| `--teacher-momentum` | 初始 EMA 动量 | 0.995 |
| `--teacher-momentum-end` | 最终 EMA 动量 | 0.999 |
| `--student-temp` | 学生温度 | 0.1 |
| `--teacher-temp-start` | 初始教师温度 | 0.04 |
| `--teacher-temp-end` | 最终教师温度 | 0.07 |
| `--disable-cross-view-loss` | 禁用跨视图全局损失 | 否 |
| `--resume` | 恢复中断训练的检查点路径 | 无 |
| `--log-every-n-steps` | 每 N 步记录指标 | 50 |
| `--save-every-epochs` | 每 N 轮保存检查点 | 10 |
| `--keep-last-checkpoints` | 保留最近 N 个检查点 | 10 |
| `--visualize-data` | 用于周期指标/UMAP 的 CSV；省略则复用 `--train-data` | 无 |
| `--extract-size` | 嵌入提取图像尺寸（<=0 自动） | 0 |
| `--metrics-sample-size` | 周期指标最大样本数 | 10000 |
| `--umap-n-neighbors` | UMAP 邻居数 | 15 |
| `--umap-min-dist` | UMAP 最小距离 | 0.1 |
| `--umap-metric` | UMAP 距离度量 | `cosine` |
| `--visualize-class-number` | UMAP 最大显示类别数 | 20 |
| `--disable-embedding-metrics` | 禁用周期嵌入指标与 UMAP | 否 |
| `--batch-size` | 批量大小 | 32 |
| `--num-workers` | DataLoader 工作线程数 | 4 |
| `--cpus` | PyTorch/MKL CPU 线程数 | 12 |
| `--device` | `auto`/`cpu`/`cuda`/`mps` | `auto` |
| `--seed` | 随机种子 | 42 |

**输出**：
- `logs/pretrain.log` — 运行日志
- `checkpoints/` — 模型检查点（`SSL_latest.pth`、`SSL_best.pth`）
- `metrics.json` / `metrics.csv` — 训练指标
- `umap.pdf` — UMAP 可视化（未禁用时）

---

### finetune 命令

利用标注数据进行 ArcFace 度量学习微调。

```bash
# 基本用法
otuformer finetune \
    --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --out-dir runs/finetune

# 恢复训练
otuformer finetune \
    --resume runs/finetune/finetune_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --out-dir runs/finetune

# 自定义参数
otuformer finetune \
    --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --model-name vit_small_patch16_224 \
    --finetune-epochs 50 \
    --finetune-lr 3e-4 \
    --freeze-ratio 0.5 \
    --loss arcface \
    --out-dir runs/finetune
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--checkpoint` | 预训练检查点路径（通常为 `runs/pretrain/SSL_latest.pth`） | 必填 |
| `--resume` | 恢复微调检查点路径 | 无 |
| `--train-data` | 训练数据 CSV（含 `image` 和 `label` 列） | 必填 |
| `--input-images-dir` | 图像根目录 | 必填 |
| `--out-dir` | 输出目录 | `runs/finetune` |
| `--model-name` | timm 骨干网络名称 | `vit_tiny_patch16_224` |
| `--metric-embed-dim` | ArcFace 投影器嵌入维度 | 256 |
| `--finetune-epochs` | 微调轮数 | 20 |
| `--finetune-lr` | 微调学习率 | 1e-4 |
| `--freeze-ratio` | 冻结骨干网络比例（0.0=不冻结，1.0=全冻结） | 0.7 |
| `--loss` | 度量学习损失名称 | `arcface` |
| `--batch-size` | 批量大小 | 32 |
| `--num-workers` | DataLoader 工作线程数 | 4 |
| `--cpus` | PyTorch/MKL CPU 线程数 | 12 |
| `--device` | `auto`/`cpu`/`cuda`/`mps` | `auto` |
| `--seed` | 随机种子 | 42 |
| `--log-every-n-steps` | 每 N 步记录指标 | 50 |
| `--save-every-epochs` | 每 N 轮保存检查点 | 10 |
| `--keep-last-checkpoints` | 保留最近 N 个检查点 | 10 |
| `--visualize-data` | 用于周期指标/UMAP 的 CSV；省略则复用 `--train-data` | 无 |
| `--extract-size` | 嵌入提取图像尺寸（<=0 自动） | 0 |
| `--metrics-sample-size` | 周期指标最大样本数 | 10000 |
| `--umap-n-neighbors` | UMAP 邻居数 | 15 |
| `--umap-min-dist` | UMAP 最小距离 | 0.1 |
| `--umap-metric` | UMAP 距离度量 | `cosine` |
| `--visualize-class-number` | UMAP 最大显示类别数 | 20 |
| `--disable-embedding-metrics` | 禁用周期嵌入指标与 UMAP | 否 |

**输出**：
- `logs/finetune.log` — 运行日志
- `checkpoints/` — 模型检查点（`finetune_latest.pth`、`finetune_best.pth`）
- `metrics.json` / `metrics.csv` — 训练指标
- `umap.pdf` — UMAP 可视化（未禁用时）

---

### extract 命令

从图像中提取嵌入向量。支持 ONNX 模型加速 CPU 推理。

```bash
# 使用 PyTorch 检查点（CLS token 模式）
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --out-dir runs/extract

# 使用 ONNX 模型（CPU 推理加速 2-5 倍）
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --onnx-path runs/export/encoder.onnx \
    --out-dir runs/extract

# 带标签 CSV 的 attention-pool 模式
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --label-csv labels.csv \
    --token-mode attention-pool \
    --out-dir runs/extract
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--checkpoint` | 预训练或微调检查点路径；除非提供 `--onnx-path` 否则必填 | 无 |
| `--input-images-dir` | 输入图像目录 | 必填 |
| `--out-dir` | 输出目录 | `runs/extract` |
| `--model-name` | timm 骨干网络名称 | `vit_tiny_patch16_224` |
| `--extract-size` | 提取时图像缩放/裁剪尺寸 | 224 |
| `--use-projector-output` | 使用投影器输出而非 CLS token | 否 |
| `--use-student` | 加载学生权重而非教师（EMA） | 否 |
| `--token-mode` | `cls`/`patch-topk`/`attention-pool` | `cls` |
| `--topk-patches` | patch-topk 模式的 top-K 值 | 20 |
| `--attention-pooling-type` | `lightweight`/`multihead`/`gated` | `lightweight` |
| `--attention-pooling-epochs` | attention-pool 查询微调轮数 | 20 |
| `--label-csv` | 用于评估嵌入质量与生成 UMAP 的标签 CSV | 无 |
| `--metrics-sample-size` | 指标评估最大样本数 | 10000 |
| `--umap-n-neighbors` | UMAP 邻居数 | 15 |
| `--umap-min-dist` | UMAP 最小距离 | 0.1 |
| `--umap-metric` | UMAP 距离度量 | `cosine` |
| `--visualize-class-number` | UMAP 最大显示类别数 | 20 |
| `--disable-umap` | 跳过 UMAP 生成 | 否 |
| `--batch-size` | 批量大小 | 32 |
| `--num-workers` | DataLoader 工作线程数 | 4 |
| `--device` | `auto`/`cpu`/`cuda`/`mps` | `auto` |
| `--onnx-path` | ONNX 模型路径（提供时使用 ONNX Runtime 推理） | 无 |
| `--seed` | 随机种子 | 42 |

**输出**：
- `embeddings.csv` — 嵌入向量（`id`, `dim_0`, `dim_1`, ...）
- `metrics.csv` — 质量指标（提供 `--label-csv` 时）
- `umap.pdf` — UMAP 可视化（提供 `--label-csv` 且未禁用时）

---

### cluster 命令

基于 UPGMA 层次聚类将嵌入向量划分为形态学 OTU。

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

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--embeddings` | 嵌入向量 CSV | 必填 |
| `--out-dir` | 输出目录 | `runs/cluster` |
| `--distance` | 距离度量：`cosine`/`euclidean` | `cosine` |
| `--prefix` | 聚类前缀（用于分区表标签） | `OTU` |
| `--pca-whitening` | 启用 PCA 白化（`true`/`false`） | `false` |
| `--pca-components` | PCA 成分数 | 256 |
| `--local-scaling` | 启用 Mutual-Proximity 局部缩放（`true`/`false`） | `false` |
| `--local-k` | 局部缩放固定 k 值（0=自动） | 0 |
| `--local-k-strategy` | 自动 k 策略：`adaptive`/`sqrt`/`log`/`fixed` | `adaptive` |
| `--cutoff-min` | 最小距离阈值 | 0.05 |
| `--cutoff-max` | 最大距离阈值 | 1.0 |
| `--cutoff-step` | 距离阈值步长 | 0.05 |
| `--custom-cutoffs` | 自定义阈值列表（逗号分隔，覆盖 min/max/step） | 无 |
| `--support-mode` | 支持率估计模式：`subsample`/`bootstrap` | `subsample` |
| `--num-replicates` | 支持率估计重复次数（0=禁用） | 0 |
| `--subsample-ratio` | subsample 模式特征比例 | 0.8 |
| `--support-display-cutoff` | 树上显示支持率标签的最小阈值 | 50.0 |
| `--save-bootstrap-trees` | 保存所有 Bootstrap 树（`true`/`false`） | `false` |
| `--save-distances` | 保存完整距离矩阵（`true`/`false`） | `false` |
| `--max-distance-pairs` | 最大距离矩阵对数 | 1000000 |
| `--label-csv` / `--labels` | 可选标签 CSV，用于分区质量评估 | 无 |
| `--metrics-sample-size` | 指标最大样本数 | 10000 |
| `--cpus` | CPU 线程数 | 8 |
| `--random-state` | 随机种子 | 42 |

**输出**：
- `UPGMA/UPGMA_Cosine.nwk` — Newick 格式系统发育树
- `UPGMA/partitions/partition_scan.csv` — 阈值扫描结果
- `UPGMA/partitions/tables/partition_<cutoff>_assignments.csv` — 各阈值下的 OTU 分配
- `UPGMA/partitions/UPGMA_tree_partitions.pdf` — 树与分区可视化
- `UPGMA/metrics.csv` — 分区质量指标（提供 `--label-csv` 时）
- `UPGMA/metrics_dashboard.pdf` — 指标面板（提供 `--label-csv` 时）
- `distance_statistics/` — 距离统计与分布图

---

### annotate 命令

应用专家校正到聚类分配，生成 refined OTU 标注。

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

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--raw-assignments` | cluster 输出的原始分区分配 CSV | 必填 |
| `--corrections` | 校正 CSV（至少含 `id`/`image` 和 `cluster` 列） | 必填 |
| `--embeddings` | 可选嵌入向量 CSV，用于距离重计算 | 无 |
| `--support-display-cutoff` | 标注 UPGMA 树上显示支持率的最小阈值 | 50.0 |
| `--figure-width` | 标注 UPGMA PDF 宽度（英寸） | 自动 |
| `--annotate-bar-width` | 校正 OTU 颜色条相对宽度 | 0.08 |
| `--show-annotation-bar` | 在标注 UPGMA PDF 中显示校正 OTU 条 | 否 |
| `--show-partitioning-bars` | 在标注 UPGMA PDF 中显示分区条带 | 否 |
| `--out-dir` | 输出目录 | `runs/annotate` |

**输出**：
- `partition_<cutoff>_assignments.csv` — 校正后的分配
- `partition_<cutoff>_assignments_changed_only.csv` — 仅变更行
- `otu_table.csv` — OTU 表
- `pairwise_distance_summary_intra-class.csv` — 类内距离摘要（提供 `--embeddings` 时）
- `UPGMA_tree_partitions_annotated.pdf` — 标注 UPGMA 树（提供 `--embeddings` 时）
- `annotation_summary.json` — 校正摘要

---

### diversity 命令

计算群落 alpha 多样性指标。支持两种输入模式：分区分配 CSV 或 OTU 表 CSV。

```bash
# 从分区分配计算
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --out-dir runs/diversity

# 启用 Faith's PD（需要 Newick 树）
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

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--assignments` | 分区分配 CSV（含 `id`/`image`、`cluster`，可选 `sample`） | 二选一 |
| `--otu-table-csv` | OTU 表 CSV（样本列 + OTU ID 列头） | 二选一 |
| `--otu-table-has-header` | 强制将 OTU 表首行作为列头（OTU ID 为数值时必需） | 否 |
| `--out-dir` | 输出目录 | `runs/diversity` |
| `--min-abundance` | 最小丰度阈值（逗号分隔） | `0,2,5` |
| `--phylo` | 计算 Faith's PD（MPD），需要 `--tree` | 否 |
| `--tree` | Newick 树路径（`--phylo` 时必需） | 无 |

**输出指标**（保存到 `diversity_indices.csv`，含全局与 per-sample）：
- **Richness**：OTU 数量
- **Chao1**：估计丰富度（考虑稀有 OTU）
- **ACE**：基于丰度的覆盖度估计
- **Shannon**：基于熵的多样性（越高 = 越多样）
- **Simpson**：两个个体不同的概率（越高 = 越多样）
- **Hill q0/q1/q2**：Hill 数（丰富度/均匀度/多样性）
- **Pielou_J**：均匀度（Shannon / log(richness)）
- **Faith's PD (MPD)**：形态系统发育多样性（通过 scikit-bio 计算）
- **MPD_w**：丰度加权根 PD（rPD_w）
- **PD_richness_norm**：Faith's PD / 物种丰富度

**输出**：
- `diversity_indices.csv` — 全局多样性指标
- `per-sample/` — 各样本多样性指标（当存在有效 sample 列时）

---

### cam 命令

生成 CAM 热力图用于模型可解释性分析。

```bash
# 基本用法
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --out-dir runs/cam

# 使用 Grad-CAM++ 并保存原始数组
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --cam-method gradcampp \
    --save-npy \
    --out-dir runs/cam

# 先查看模型层结构
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --dump-model-structure \
    --out-dir runs/cam
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--checkpoint` | OTU-Former 检查点路径（`.ckpt` 或 `.pth`） | 必填 |
| `--images-dir` | 待生成 CAM 的图像目录 | 必填 |
| `--label-csv` | 可选标签 CSV（含 `image` 和 `label` 列）；省略则使用目录下所有图像 | 无 |
| `--out-dir` | 输出目录 | `runs/cam` |
| `--cam-method` | `gradcam`/`gradcampp`/`layercam`/`scorecam`/`eigencam`/`ablationcam` | `gradcam` |
| `--arch` | 强制架构类型：`cnn`/`vit`（省略则自动检测） | 自动 |
| `--target-layer-name` | 指定 CAM 目标层（省略则自动选择） | 自动 |
| `--image-weight` | CAM 叠加中原始图像的混合权重（0-1） | 0.5 |
| `--fig-format` | 输出格式：`png`/`jpg`/`pdf` | `png` |
| `--save-npy` | 保存原始 CAM 数组为 NumPy 格式 | 否 |
| `--dump-model-structure` | 输出模型层名称到 `model_layers.txt` | 否 |
| `--max-images` | 最大处理图像数（None = 全部） | 全部 |
| `--cam-batch-size` | CAM 推理批量大小 | 32 |
| `--num-workers` | DataLoader 工作进程数 | 4 |
| `--device` | `auto`/`cpu`/`cuda`/`mps` | `auto` |

**输出**：
- `figures/` — CAM 叠加图像
- `cam_summary.csv` — 元数据摘要
- `arrays/` — 原始 CAM 数组（使用 `--save-npy` 时）
- `model_layers.txt` — 模型层名称（使用 `--dump-model-structure` 时）

---

### export 命令

将 PyTorch 检查点导出为 ONNX 格式用于部署和加速推理。

```bash
# 基本用法
otuformer export \
    --checkpoint runs/finetune/finetune_latest.pth \
    --out-dir runs/export

# 自定义图像尺寸与 opset
otuformer export \
    --checkpoint runs/finetune/finetune_latest.pth \
    --imgsz 224 \
    --opset 17 \
    --out-dir runs/export
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--checkpoint` | 检查点路径 | 必填 |
| `--out-dir` | 输出目录 | `runs/export` |
| `--imgsz` | ONNX 导出输入图像尺寸（省略则自动从骨干推断） | 自动 |
| `--opset` | ONNX opset 版本 | 18 |

**输出**：
- `encoder.onnx` — ONNX 编码器模型

---

## 通用行为

### 日志

所有命令会在输出目录的 `logs/` 子目录中保存日志文件（如 `pretrain.log`、`finetune.log`、`extract.log`、`cluster.log`、`annotate.log`、`diversity.log`、`cam.log`、`export.log`），包含：
- 完整命令行
- 时间戳
- 所有参数值
- 运行时输出

### 优雅退出

按 `Ctrl+C` — 当前图像处理完成后退出，保存部分结果。

### 设备选择

`--device auto` 自动选择：
1. CUDA（如果可用）
2. MPS / Apple Silicon（如果可用）
3. CPU（回退）

### Shell 补全

安装 shell 补全：

```bash
otuformer --install-completion
```

支持的 shell：bash、zsh、fish

### 版本号

查看已安装版本：

```bash
otuformer --version
otuformer -v
```

## 项目结构

```
.
├── src/otuformer/            # 核心包
│   ├── cli/                  # 命令行接口
│   │   ├── main.py           # 入口点调度器
│   │   ├── doctor.py         # otuformer doctor
│   │   ├── pretrain.py       # otuformer pretrain
│   │   ├── finetune.py       # otuformer finetune
│   │   ├── extract.py        # otuformer extract
│   │   ├── cluster.py        # otuformer cluster
│   │   ├── annotate.py       # otuformer annotate
│   │   ├── diversity.py      # otuformer diversity
│   │   ├── cam.py            # otuformer cam
│   │   └── export.py         # otuformer export
│   ├── training/             # 训练逻辑
│   │   ├── model.py          # 模型定义
│   │   ├── dataset.py        # 数据集
│   │   ├── loss.py           # 损失函数
│   │   ├── trainer.py        # 训练器
│   │   └── scheduler.py      # 学习率调度
│   ├── embedding/            # 嵌入向量
│   │   ├── extractor.py      # 嵌入提取器
│   │   └── evaluator.py      # 嵌入评估器
│   ├── delineation/          # OTU 划分
│   │   ├── distance.py       # 距离计算
│   │   ├── tree.py           # UPGMA 树构建
│   │   ├── partition.py      # 分区与指标
│   │   └── annotate.py       # 校正与标注
│   ├── vision/               # 视觉工具
│   │   ├── cam.py            # CAM 可视化
│   │   └── export.py         # ONNX 导出
│   └── utils/                # 工具函数
│       ├── device.py         # 设备检测
│       ├── logging.py        # TeeLogger
│       ├── checkpoint.py     # 检查点处理
│       └── io.py             # CSV/JSON I/O
├── tests/                    # 测试文件
├── examples/                 # 示例数据
├── docs/                     # 文档
└── pyproject.toml            # 项目配置
```

## 许可证

本项目采用 MIT 许可证 — 详见 LICENSE 文件。

## 联系方式

- 邮箱：`xtmtd.zf@gmail.com`

## 引用

如果您在研究中使用 OTU-Former，请引用：

```bibtex
@software{otuformer2026,
  author = {Zhang, Feng},
  title = {OTU-Former: Image-based Morphological OTU Delineation Toolkit},
  year = {2026},
  url = {https://github.com/xtmtd/OTU-Former}
}
```

预印本可查看：https://doi.org/10.64898/2026.04.28.721370
