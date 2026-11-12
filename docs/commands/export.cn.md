# otuformer export

[English](export.md) | [中文](export.cn.md)

## 目的

把编码器与投影头导出为 ONNX，用于部署与 CPU 加速推理。导出的模型直接以图像为输入、输出嵌入向量。安装 ONNX Runtime 时，导出过程会在导出尺寸上用 CPU PyTorch 校验数值一致性。

## 用法

```bash
otuformer export \
    --checkpoint runs/finetune/finetune_latest.pth \
    --out-dir runs/export
```

## 参数

### `--checkpoint`

**默认值：** `Required`

检查点路径。

### `--out-dir`

**默认值：** `runs/export`

输出目录。

### `--imgsz`

**默认值：** `auto` · **允许值：** auto，或整数（常用：224 / 384 / 448；518 仅适用于 patch-14）

ONNX 导出的输入图像尺寸。'auto' 使用检查点记录的训练尺寸。常见取值：224、384、448（例如 patch-14 模型用 518）。

### `--opset`

**默认值：** `18` · **允许值：** 整数

ONNX opset 版本。

### `--model-name`

**默认值：** `vit_tiny_patch16_224`

检查点未记录模型名时的回退 timm 骨干网络（ref 脚本微调检查点既不记录 config 也不记录 args）。

### `--output-kind`

**默认值：** `embedding` · **可选值：** embedding / spatial

图产物类型：`embedding` 保持既有的 encoder+projector 图与单个 `embedding` 输出；`spatial` 为每个选定的完整 backbone 块导出一个具名输出，并附固定特征契约。

### `--feature-layers`

**默认值：** `None`（最后一个完整块）

仅空间导出。逗号分隔的 backbone 相对完整块路径（例如 `blocks.5,blocks.11`）。省略时取最后一个完整块。

### `--feature-norm`

**默认值：** `backbone` · **可选值：** backbone / none

仅空间导出。对每个导出块施加 backbone 的最终归一化（`backbone`），或导出原始块输出（`none`）。

### `--feature-l2-normalize`

**默认值：** `No`

仅空间导出。在 backbone 归一化之后对每个 token 向量做 L2 归一化，并导出该表示。

### `--use-student`

**默认值：** `No`

仅空间导出。导出学生权重；图中记录实际使用的权重来源。嵌入导出拒绝该选项，因为该图一直导出教师/默认权重。

### `--overwrite`

**默认值：** `No`

清空已存在的非空输出目录。

## 输入

`--checkpoint` 是 OTU 检查点或 ref 脚本检查点（`ref/ibot20260115.py`）。投影头由 `config.embedding_head` 重建，缺失时回退到权重中记录的投影头形状。

## 输出

- `encoder.onnx` — ONNX 编码器模型

## 示例

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

## 说明

可读取 OTU 检查点与旧脚本检查点（`ref/ibot20260115.py`）。投影器优先按 `config.embedding_head` 重建，缺失时按权重中的投影器形状推断。DINO 的 SSL 检查点导出归一化后的 D 维瓶颈：仅用于训练的原型层、其目标 softmax 以及 K 维 CLS 中心都不会出现在计算图中，因此导出宽度是 D（或微调后代的度量宽度），绝不会是 K。

**空间导出（`--output-kind spatial`）。** 图中为每个选定的完整块提供一个具名输出
（`spatial_000`、`spatial_001`……），按批为 BHWC、按图为 HWC，并在图元数据
`otuformer.spatial.v1` 中携带规范化特征契约：模式版本、输出名顺序、块路径、网格与
通道宽度、前缀/patch 栅格、输入尺寸与归一化、特征归一化与 L2（含 epsilon），以及
实际模型身份与权重来源。导出即固定模型、层、归一化与尺寸；读取方用该契约校验显式请求，
而不是从图里猜测。

校验是强制的，覆盖**每一个**声明输出，并在 batch 1 与 2 上与 CPU FP32 PyTorch 比较，
容差为 `atol=1e-4`、`rtol=1e-3`，并逐输出报告最大误差。缺少 ONNX Runtime、动态批维
缺失、契约缺失或不匹配的图都不是空间模型：该次尝试不发布任何文件，先前的图与报告保持
原样。通道宽度不是嵌入的 `out_dim`，逐图几何永不进入图。
