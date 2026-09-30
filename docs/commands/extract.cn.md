# otuformer extract

[English](extract.md) | [中文](extract.cn.md)

## 目的

用训练好的检查点从图像中提取嵌入向量，产出供 `cluster`、`diversity` 与 `annotate` 使用的 `embeddings.csv`。可运行 PyTorch 模型，也可使用导出的 ONNX 模型（CPU 推理快 2-5 倍）。提供三种 token 模式：原始 CLS、`patch-topk` 与 `attention-pool`；有标签时还会输出质量指标与 UMAP。

## 用法

```bash
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --out-dir runs/extract
```

## 参数

### `--checkpoint`

**默认值：** `None`

预训练或微调检查点路径。除非提供 --onnx-path 否则必填。

### `--input-images-dir`

**默认值：** `Required`

图像目录或父目录。

### `--out-dir`

**默认值：** `runs/extract`

输出目录。

### `--model-name`

**默认值：** `vit_tiny_patch16_224`

timm 骨干网络。 timm 骨干网络名称

### `--extract-size`

**默认值：** `auto` · **允许值：** auto，或整数（常用：224 / 384 / 448；518 仅适用于 patch-14）

提取的缩放/裁剪尺寸。'auto' 使用检查点记录的训练尺寸。常见取值：224、384、448（例如 patch-14 模型用 518）。

### `--eval-transform`

**默认值：** `center-crop` · **允许值：** center-crop / whole-specimen-pad

评估预处理协议：center-crop（Resize + CenterCrop）或 whole-specimen-pad（保持宽高比的方形填充）。默认：center-crop。

### `--use-projector-output`

**默认值：** `No`

对 SSL 检查点使用 SSL 投影头输出；对微调检查点使用 ArcFace 嵌入。适用于微调类别的检索与闭集聚类。

### `--use-student`

**默认值：** `No`

加载学生权重而非教师（EMA 模型）。默认加载教师，其嵌入在下游分析中更好（DINO/iBOT 惯例）。对微调检查点无影响。

### `--token-mode`

**默认值：** `cls` · **允许值：** cls / patch-topk / attention-pool

嵌入 token 模式。cls=原始 CLS token（未见类别、跨数据集迁移与一般形态学的比较基线）；patch-topk=top-K patch token + PCA；attention-pool=对 patch token 的学习式查询池化。

### `--topk-patches`

**默认值：** `20`

patch-topk 模式的 top-K patch token（常用：10/20/30）。

### `--attention-pooling-type`

**默认值：** `lightweight` · **允许值：** lightweight / multihead / gated

attention pooling 类型。可选：lightweight / multihead / gated。

### `--attention-pooling-epochs`

**默认值：** `20`

未找到缓存的 pooling 检查点时的 attention 查询微调轮数。

### `--label-csv`

**默认值：** `None`

含 'image' 列与可选 'label' 列的 CSV，用于选择图像、评估嵌入质量并生成 UMAP。无标签时只生成 UMAP。attention-pool 模式下若缺少缓存的 pooling 权重，该 CSV 必须包含标签。 含 `image` 列及可选 `label` 列的 CSV；仅 image 时生成 UMAP，不计算监督指标

### `--metrics-sample-size`

**默认值：** `10000`

指标与 UMAP 评估的最大样本数（<=0 表示不设上限）。

### `--umap-n-neighbors`

**默认值：** `15`

UMAP n_neighbors。

### `--umap-min-dist`

**默认值：** `0.1`

UMAP min_dist。

### `--umap-metric`

**默认值：** `cosine` · **允许值：** cosine / euclidean

UMAP 投影的距离度量。常见选择：cosine、euclidean。

### `--visualize-class-number`

**默认值：** `20`

UMAP 图中显示的最大类别数。

### `--disable-umap`

**默认值：** `No`

即使提供了 --label-csv 也跳过 UMAP 生成。

### `--batch-size`

**默认值：** `32`

批量大小。

### `--num-workers`

**默认值：** `4`

DataLoader 工作线程数。

### `--device`

**默认值：** `auto` · **允许值：** auto / cpu / cuda / mps

设备：auto / cpu / cuda / mps。

### `--onnx-path`

**默认值：** `None`

用于推理的导出 ONNX 模型路径。提供时使用 ONNX Runtime 而非 PyTorch。仅支持 token-mode 'cls'（默认）。会覆盖 --checkpoint、--model-name、--use-student 与 --device。

### `--seed`

**默认值：** `42`

随机种子。

### `--overwrite`

**默认值：** `No`

清空已存在的非空输出目录。

## 输入

`--input-images-dir` 是图像目录，或各类别子目录的父目录（批处理模式）。可选的 `--label-csv` 需要 `image` 列，可含 `label` 列。

## 输出

- `embeddings.csv` — 嵌入向量（`id`, `dim_0`, `dim_1`, ...）
- `metrics.csv` — 质量指标（`--label-csv` 至少包含两个类别时）
- `umap.pdf` — UMAP 可视化（提供 `--label-csv`、未禁用且抽样后至少有 10 个样本时）

嵌入质量指标使用共享的 [嵌入指标](embedding-metrics.cn.md) 定义与字段名。

## 示例

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

## 说明

可读取 OTU 检查点与旧脚本检查点（`ref/ibot20260115.py`）。嵌入头优先按 `config.embedding_head` 重建，缺失时按权重中的投影器形状推断。

`--label-csv` 需要 `image` 列，并接受可选的 `label` 列。仅提供图像时仍会生成 UMAP，但不产出监督指标；attention-pool 查询训练仍然需要 `label`。

嵌入质量指标使用共享的 [嵌入指标](embedding-metrics.cn.md) 定义与字段名。

`cam` 与 `extract` 使用原生校验：在创建任何输出目录之前，对枚举取值之外的输入直接拒绝。
