# otuformer cam

[English](cam.md) | [中文](cam.cn.md)

## 目的

生成 CAM 热图，显示模型关注图像的哪些区域。提供六种算法（Grad-CAM、Grad-CAM++、LayerCAM、Score-CAM、Eigen-CAM、Ablation-CAM），并与模型所见相同的预处理做逆变换后叠加，使热图与原始图像对齐。

## 用法

```bash
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --out-dir runs/cam
```

## 参数

### `--checkpoint`

**默认值：** `Required`

OTU-Former 检查点路径（.ckpt 或 .pth）。

### `--images-dir`

**默认值：** `Required`

用于生成 CAM 热图的图像目录。

### `--label-csv`

**默认值：** `None`

含 'image' 与 'label' 列的可选 CSV。省略时使用 --images-dir 下的全部图像。

### `--out-dir`

**默认值：** `runs/cam`

写入 CAM 可视化与产物的目录。

### `--cam-method`

**默认值：** `gradcam` · **允许值：** gradcam / gradcampp / layercam / scorecam / eigencam / ablationcam

CAM 算法：gradcam、gradcampp、layercam、scorecam、eigencam、ablationcam。

### `--arch`

**默认值：** `None` · **允许值：** cnn / vit（省略时自动检测）

强制架构类型（cnn、vit；未指定时根据模型名自动检测）。

### `--target-layer-name`

**默认值：** `None`

CAM 使用的具体模型层（省略时自动选择）。

### `--image-weight`

**默认值：** `0.5`

CAM 叠加图中原图的混合权重（0-1）。

### `--fig-format`

**默认值：** `png` · **允许值：** png / jpg / pdf

CAM 图输出格式：png、jpg、pdf。

### `--save-npy`

**默认值：** `none` · **允许值：** none / raw / normalized

保存 CAM 数组：none（默认）、raw（正向未归一化 CAM）或 normalized（逐图 min-max [0, 1]）。

### `--dump-model-structure`

**默认值：** `No`

将模型层名写入 out-dir/model_layers.txt，供 --target-layer-name 参考。

### `--max-images`

**默认值：** `None`

最多处理的图像数量（None = 全部）。

### `--cam-batch-size`

**默认值：** `32`

CAM 推理的批量大小。

### `--eval-transform`

**默认值：** `center-crop` · **允许值：** center-crop / whole-specimen-pad

评估预处理协议：center-crop（Resize + CenterCrop；热图限于模型视野）或 whole-specimen-pad（保持宽高比的方形填充；热图覆盖整个标本）。默认：center-crop。

### `--num-workers`

**默认值：** `4`

dataloader 工作进程数（保留）。

### `--model-name`

**默认值：** `vit_tiny_patch16_224`

检查点未记录模型名时的回退 timm 骨干网络（ref 脚本微调检查点既不记录 config 也不记录 args）。

### `--device`

**默认值：** `auto` · **允许值：** auto / cpu / cuda / mps

CAM 生成的计算设备：auto、cpu、cuda、mps。

### `--overwrite`

**默认值：** `No`

清空已存在的非空输出目录。

## 输入

`--images-dir` 存放要可视化的图像。可选的 `--label-csv`（含 `image` 与 `label` 列）限定处理范围；不提供时使用目录下全部图像。

## 输出

- `figures/` — CAM 叠加图像
- `cam_summary.csv` — 元数据摘要
- `arrays/` — CAM 数组，仅在 `--save-npy raw` 或 `--save-npy normalized` 时生成（默认不写任何数组）。保存的数组为模型输入分辨率的 `float32`。
- `model_layers.txt` — 模型层名称（使用 `--dump-model-structure` 时）

## 示例

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
    --save-npy raw \
    --out-dir runs/cam

# 先查看模型层结构
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --dump-model-structure \
    --out-dir runs/cam
```

## 说明

可读取 OTU 检查点与旧脚本检查点（`ref/ibot20260115.py`）；CAM 只使用 backbone。

**CAM 数组语义**：
- `--save-npy raw` 保留 ReLU 后、缩放到模型分辨率后的正值未归一化 CAM 幅值（在逐图 min-max 归一化之前）；`--save-npy normalized` 写入逐图 min-max `[0, 1]` 副本，与旧版本的保存数组语义一致。叠加图始终使用独立的归一化副本，PNG 像素级一致不是契约。
- raw 幅值仅在同一模型、同一目标层、同一预处理配置内可比，不适用于跨 backbone、跨层或跨 CAM 方法比较。`eigencam` 的 raw 数值来自符号任意的 SVD 投影，为完整性而导出，但不适合用于响应强度统计。

`cam` 与 `extract` 使用原生校验：在创建任何输出目录之前，对枚举取值之外的输入直接拒绝。
