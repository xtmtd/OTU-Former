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

可读取 OTU 检查点与旧脚本检查点（`ref/ibot20260115.py`）。投影器优先按 `config.embedding_head` 重建，缺失时按权重中的投影器形状推断。
