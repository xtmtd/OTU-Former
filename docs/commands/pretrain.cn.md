# otuformer pretrain

[English](pretrain.md) | [中文](pretrain.cn.md)

## 目的

使用 DINO/iBOT 风格教师-学生 ViT 的自监督对比预训练。它在全局与局部裁剪上训练学生-教师骨干网络，并配合块级目标，无需任何标签。产出的检查点可用于初始化 `finetune`，也可直接交给 `extract` 使用。

## 用法

```bash
otuformer pretrain \
    --input-images-dir ./images \
    --out-dir runs/pretrain
```

## 参数

### `--train-data`

**默认值：** `None`

含 'image' 列的可选 CSV（路径相对于 --input-images-dir）。省略时递归使用 --input-images-dir 下的全部图像。

### `--input-images-dir`

**默认值：** `Required`

图像根目录。

### `--out-dir`

**默认值：** `runs/pretrain`

输出目录。

### `--model-name`

**默认值：** `vit_tiny_patch16_224`

学生/教师编码器使用的 timm 骨干网络名称。

### `--register-tokens`

**默认值：** `none` · **允许值：** none / 0 / 4

适用于单一 CLS 的 timm VisionTransformer 的 register token：'none'（默认）、'0' 或 '4'。'none' 保持骨干网络自身架构（普通 ViT 没有 register，原生 register 模型保持其自身数量），并在保存的 args 中记为 null。显式 '0' 与 'none' 不同：在无 CLS 的骨干网络上会被拒绝。在原生 register 骨干网络上显式取值也会被拒绝。

### `--out-dim`

**默认值：** `256`

SSL 投影头输出维度。

### `--max-epochs`

**默认值：** `50`

SSL 预训练总 epoch 数。

### `--lr`

**默认值：** `0.0005`

warmup/cosine 调度之前的基础学习率。

### `--weight-decay`

**默认值：** `0.05`

AdamW 权重衰减系数。

### `--warmup-epochs`

**默认值：** `3`

cosine 学习率衰减前的 warmup epoch 数。

### `--augmentation`

**默认值：** `None` · **允许值：** global-barcode / color-robust / legacy

数据增强配置：global-barcode、color-robust 或 legacy。新运行默认 global-barcode。省略则在 --resume 时继承已保存的配置。color-robust 可能抑制诊断性体色、色斑或金属光泽；legacy 复现 0.2.1 并保留其历史变换。

### `--orientation-policy`

**默认值：** `None` · **允许值：** invariant / sensitive

每个全局与局部视图的方向策略：invariant 或 sensitive。新运行默认 sensitive。省略则在 --resume 时继承已保存的策略。 所有全局与局部视图的方向策略（对 `legacy` 无变换效果）：`sensitive`（新运行默认；旋转 `[-15°, 15°]`，不做水平翻转）或 `invariant`（可选加入的宽角度旋转与翻转）；`--resume` 时省略则继承已保存策略

### `--global-crop-size`

**默认值：** `auto` · **允许值：** auto，或整数（常用：224 / 384 / 448；518 仅适用于 patch-14）

全局裁剪分辨率。'auto' 在新运行时解析为骨干网络的原始输入尺寸，在 --resume 时解析为检查点记录尺寸。常见取值：224、384、448（例如 patch-14 模型用 518）。

### `--local-crop-size`

**默认值：** `None` · **允许值：** 能被 patch 尺寸整除的整数

局部裁剪分辨率。省略时新运行使用 96，--resume 时继承检查点数值。必须能被骨干网络的 patch 尺寸整除（例如 patch-14 模型用 98 或 112）。

### `--local-crops`

**默认值：** `None`

局部裁剪数量。省略时新运行使用 6，--resume 时继承检查点数值。

### `--patch-loss`

**默认值：** `consistency` · **允许值：** none / consistency / masked-feature / ibot

块级目标。'none'；'consistency'（默认；选取可见的同位置 patch 做归一化余弦回归，即掩码位置一致性，绝不做输入掩码）；'masked-feature'（对教师最后四个块的 patch 目标做真正的连续掩码特征预测）；'ibot'（实验性的原型分布预测）。

### `--masking-strategy`

**默认值：** `random` · **允许值：** random / blockwise / hybrid

掩码几何，仅用于 masked-feature/ibot。'random' 独立采样 patch 位置；'blockwise' 合并有界矩形区域；'hybrid' 取一半 blockwise、一半随机位置。对 none/consistency 不适用；具体几何限制见本文档。

### `--mask-ratio`

**默认值：** `auto` · **允许值：** auto，或 (0, 1) 内的浮点数

'auto' 或 (0, 1) 内的浮点数。所选块级目标使用的 patch 位置比例。新运行会把 'auto' 解析为 0.30（v0.6.x 默认 0.50）。consistency 用它作为可见同位置比例，masked-feature/ibot 用它作为被掩码的学生输入比例。 `auto` 或 (0, 1) 内的浮点数。新运行 `auto` 解析为 0.30（v0.6.x 为 0.50）；旧检查点续训保留其记录值（缺失时为 0.50）

### `--ibot-prototypes`

**默认值：** `512`

实验性 ibot 模式的原型字典大小：任意 >= 2 的整数，默认 512。字典越大越适合更大的数据集（例如 4096 或 16384）；取 2 的幂只是习惯而非要求。对其他块级模式不适用。

### `--lambda-local`

**默认值：** `1.5`

局部裁剪 SSL 损失项的权重。

### `--lambda-mask`

**默认值：** `1.0`

块级损失项的权重。ibot 的交叉熵与余弦块级损失量级不同，因此使用 --patch-loss ibot 时通常要调低（例如 0.25-0.5）。

### `--teacher-momentum`

**默认值：** `0.995`

初始 EMA 动量。

### `--teacher-momentum-end`

**默认值：** `0.999`

最终 EMA 动量。

### `--student-temp`

**默认值：** `0.1`

学生温度。

### `--teacher-temp-start`

**默认值：** `0.04`

初始教师温度。

### `--teacher-temp-end`

**默认值：** `0.07`

最终教师温度。

### `--disable-cross-view-loss`

**默认值：** `No`

关闭跨视图全局损失配对。默认保持全局裁剪之间的完整跨视图匹配。

### `--log-every-n-steps`

**默认值：** `50`

每 N 次迭代记录一次指标。

### `--save-every-epochs`

**默认值：** `10`

每 N 个 epoch 保存一次检查点。

### `--keep-last-checkpoints`

**默认值：** `10`

仅保留最后 N 个检查点。

### `--visualize-data`

**默认值：** `None`

含 'image' 列与可选 'label' 列的 CSV，用于周期性嵌入指标与 UMAP。无标签时只生成 UMAP。省略时若有 --train-data 则复用。

### `--extract-size`

**默认值：** `auto` · **允许值：** auto，或整数（常用：224 / 384 / 448；518 仅适用于 patch-14）

周期性指标与 UMAP 嵌入提取的图像尺寸。'auto' 使用模型的 img_size。常见取值：224、384、448（例如 patch-14 模型用 518）。

### `--metrics-sample-size`

**默认值：** `10000`

周期性指标与 UMAP 的最大样本数（<=0 表示不设上限）。

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

### `--disable-embedding-metrics`

**默认值：** `No`

预训练期间关闭周期性嵌入指标与 UMAP 生成，以减少运行时开销。

### `--batch-size`

**默认值：** `32`

批量大小。

### `--num-workers`

**默认值：** `4`

DataLoader 工作线程数。

### `--cpus`

**默认值：** `12`

PyTorch/MKL 使用的 CPU 线程数。

### `--device`

**默认值：** `auto` · **允许值：** auto / cpu / cuda / mps

设备：auto / cpu / cuda / mps。

### `--seed`

**默认值：** `42`

随机种子。

### `--resume`

**默认值：** `None`

用于恢复被中断预训练的检查点路径。

### `--overwrite`

**默认值：** `No`

清空已存在的非空输出目录。

## 输入

`--input-images-dir` 是图像根目录；可选的 `--train-data` 是含 `image` 列的 CSV，路径相对于该根目录。不提供 `--train-data` 时使用根目录下全部图像，无标签图像会被赋予合成的 `NA_<index>` 标签。`--visualize-data` 是可选的独立 CSV，用于周期性指标与 UMAP。

## 输出

- `logs/pretrain.log` — 运行日志
- `SSL_latest.pth`、`SSL_epoch_*.pth` — 模型检查点，写入 `--out-dir` 根目录（不存在 `checkpoints/` 子目录，也没有 `SSL_best.pth`）
- `logs/metrics.pretrain.csv` — 周期嵌入指标
- `logs/instant_metrics.pretrain.csv` — 每次迭代的训练指标
- `logs/training_curves_pretrain.pdf` — 训练曲线图
- `logs/umap.train.epoch_<N>.pdf` — 周期 UMAP 图（未禁用时）

嵌入质量指标使用共享的 [嵌入指标](embedding-metrics.cn.md) 定义与字段名。

## 示例

```bash
otuformer pretrain --train-data images_union.csv --input-images-dir ./images \
    --out-dir runs/pretrain --resume runs/pretrain/SSL_latest.pth --max-epochs 100
```

```bash
# 基本用法
otuformer pretrain \
    --augmentation global-barcode \
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

## 说明

**patch 级目标（v0.7.0）。** `--patch-loss` 选择 patch 目标：

| `--patch-loss` | student 输入 | 目标 |
|---|---|---|
| `none` | 未遮挡 | 无 patch 目标 |
| `consistency`（默认） | 未遮挡 | 在选中的可见同位 patch 上做归一化余弦回归 |
| `masked-feature` | 遮挡 | continuous masked feature prediction（连续掩码特征预测） |
| `ibot`（实验性） | 遮挡 | 原型分布预测 |

`consistency` 选择完全可见的 patch 位置，因此它是 **masked-position patch
consistency**，而不是 masked image modeling：student 仍能看到被选中的每个像素。
`masked-feature`（即 continuous masked feature prediction）与 `ibot`
（实验性）会在位置编码之前把选中的 student patch embedding 替换为可学习的 mask
token，因此 student 看不到被选中的内容，但位置信息保留。`masked-feature` 的
teacher 目标是 teacher 最后四个 block 输出的 L2 归一化均值。iBOT 预测 teacher 的
中心化原型分布
（`--ibot-prototypes`，默认 512；任意 >= 2 的整数，数据集越大可用越大），使用移动平均中心。
`masked-feature` 与 `ibot` 互斥，不会叠加。

`--masking-strategy` 为 `masked-feature` 与 `ibot` 选择真实遮挡几何；对 `none`
与 `consistency` 无效：

- `random`（默认）——每个样本、每个 global view 独立采样 `round(ratio * N)` 个
  互不重复的 patch 位置。
- `blockwise`——每轮生成二到四个矩形提案，重叠部分合并。合法矩形的边长至少为 2
  个 patch，长宽比在 `[0.5, 2.0]` 内，面积不超过 `floor(0.20 * N)` 个 patch。
  重叠只计一次，最后裁剪或随机填充到恰好 `round(ratio * N)` 个位置。若网格无法
  容纳任何合法矩形（例如 `4 x 4`：最大块面积为 3，而 `2 x 2` 块需要 4），会在
  训练循环开始前报错，而不是静默降级为 `random`。
- `hybrid`——先取恰好 `floor(target / 2)` 个块状位置，再随机填充到目标数量。

所有策略都会把数量限制为至少遮挡一个、至少保留一个可见 patch。

`--lambda-mask`（默认 `1.0`）是 patch 损失的权重。iBOT 的交叉熵与余弦类 patch
损失不在同一量级，因此使用 `--patch-loss ibot` 时通常需要调低（例如 `0.25`-`0.5`）。
不同 patch 模式的损失值不可直接比较。

**迁移提示：`--mask-ratio` 现在默认 `auto`，新运行会解析为 0.30，而 v0.6.x 的
默认值是 0.50。** 恢复旧检查点会保留其记录的比值（缺失时为 0.50），且不能切换到
`masked-feature` 或 `ibot`。对 `consistency`，`--mask-ratio` 是参与损失的可见同位
比例；对 `masked-feature`/`ibot`，它是被替换为 mask token 的 student patch 比例。
显式传入当前模式不使用的选项会被拒绝，而不是静默忽略。

**可选 register token（v0.10.0）。** `--register-tokens` 对合格的单 CLS timm
`VisionTransformer` 接受 `none`（默认）、`0` 或 `4`。`4` 会新增四个可训练 register，并迁移单 CLS
的预训练骨干：register token 保留其原生初始化，而所有 CLS/patch/block 权重与位置
保持不变。`none` 保留骨干网络自身结构，原生 register 模型会保留自身数量；对原生
register 或无 CLS 的骨干传入显式值会被拒绝，DINOv3/Eva 骨干不被该选项支持。
register 永远不是图像 patch：`patch-topk`、`attention-pool` 提取、CAM reshape 与
patch 目标都会跳过所有前缀 token（`backbone.num_prefix_tokens`）。新的 attention-pool
checkpoint 会记录其 patch 策略（`num_prefix_tokens`、`register_tokens`、
`prefix_excluded`、`attention_pooling_type`）并保存到带策略标签的同级文件，因此
在不同 patch 集上训练的 pool 不会被静默复用。ONNX 导出还会在同一输入上比对 CPU
FP32 PyTorch 与 CPU ONNX Runtime（`atol=1e-4`、`rtol=1e-3`），并在
`export_report.json` 中记录 `validation_status`、`max_abs_diff` 与 register 数量。微调运行的 register 布局属于 v0.9.0 伪标签实验同一性的一部分，因此 finetune#2 必须从具有相同 patch 集合的源初始化。

`masked-feature` 与 iBOT 会为每个 global view 增加一次 masked student 前向，在默认
的两 global / 六 local 裁剪布局下，student 编码器计算量约增至 1.65 倍。
extract、finetune、CAM、export 均未改变：默认仍使用原始 CLS token，它们从不加载
训练专用的 patch 状态，也从不调用 mask token、predictor 或 iBOT head。预训练与
预训练续训仅支持标准 timm ViT 骨干网络。v0.7.0 的续训是严格的：patch 模式、解析后的
比值、遮挡策略、原型数量以及固定的 target/center 设置都必须与检查点一致，
`masked-feature`/iBOT 检查点必须携带其 `patch_objective` 状态。v0.7.0 检查点会完整往返 RNG 状态；旧检查点
只能尽力恢复并会给出警告。

这些目标属于自监督上下文特征预测，本身并不建立解剖部位语义或稠密形态学监督，
v0.7.0 也不对稠密表示做出任何声明。

当 `--visualize-data` 没有 `label` 列时，周期性监督指标会被跳过，但仍会基于可视化嵌入生成 UMAP。

如需加入新图像继续预训练，请创建同时包含旧图和新图的 CSV，保持相同的
`--out-dir`，使用最新 checkpoint 的 `--resume`，并增加 `--max-epochs`：

扩展训练可以使用更大的合并数据集；原计划的中断恢复则需要保持原来的
DataLoader 长度。只用新增图像训练虽然可行，但可能遗忘已有数据。

**局部视图的续训继承。** `--resume` 时，省略的 `--local-crop-size` 与 `--local-crops` 继承 checkpoint 已保存的值；显式指定冲突值会报错，修改这些值需要开启新运行。

数据增强与方向策略遵循共享的 [训练数据增强契约](training-augmentation.cn.md)。

嵌入质量指标使用共享的 [嵌入指标](embedding-metrics.cn.md) 定义与字段名。

## 版本注记

- **v0.10.0** — `--register-tokens` 接受 `none`、`0` 或 `4`；解析后的数量记录进检查点，续训与微调时必须一致。
- **v0.7.0** — `--patch-loss` 与 `--masking-strategy` 选择块级目标及其掩码几何；`--mask-ratio` 默认 `auto`，新运行解析为 `0.30`（v0.6.x 为 `0.50`）。
