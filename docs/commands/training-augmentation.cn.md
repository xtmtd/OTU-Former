# otuformer training augmentation

[English](training-augmentation.md) | [中文](training-augmentation.cn.md)

## 目的

`pretrain` 与 `finetune` 共用同一份数据增强契约，由 `--augmentation` 配置与
`--orientation-policy` 方向策略组成。本文说明各配置与各策略的具体行为，以及取值的继承规则。

各 flag 本身（允许值与默认值）在每个受影响命令文档的 `Parameters` 表中**只定义一次**；本文
只做链接，不重复这些默认值与取值表（见文档规范的 4.1 节）。

## 用法

```bash
# pretrain：整标本条形码配置，方向敏感标记
otuformer pretrain --input-images-dir ./images \
    --augmentation global-barcode --orientation-policy sensitive

# finetune：可选加入的 conservative 配置
otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv --input-images-dir ./images \
    --augmentation conservative --orientation-policy sensitive
```

## 适用命令

| 命令 | 其中定义的 flag | 文档 |
|---|---|---|
| `otuformer pretrain` | `--augmentation`、`--orientation-policy` | [pretrain](pretrain.cn.md) |
| `otuformer finetune` | `--augmentation`、`--orientation-policy` | [finetune](finetune.cn.md) |

当 `finetune --augmentation none` 时，方向策略不生效。

## 配置

| 配置 | 阶段 | 说明 |
|---|---|---|
| `global-barcode` | pretrain | 整标本条形码配置；方向处理取决于所选方向策略。温和的光度扰动；保留颜色（不做灰度化）。 |
| `color-robust` | pretrain | 几何与模糊设置与 `global-barcode` 相同，但颜色扰动更强并启用灰度化。**警告**：可能降低模型对诊断性体色、色斑或金属光泽的敏感度。 |
| `legacy` | pretrain | 完全复现 OTU-Former 0.2.1 的数据增强，仅用于旧运行的续训与对比，不建议用于新运行。它会记录所选的任一方向策略，但不改变其历史变换。 |
| `none` | finetune | 保持原有的确定性 `Resize -> CenterCrop -> ToTensor -> Normalize`。 |
| `conservative` | finetune | 实验性可选项，尚无证据表明优于 `none`；请在留出个体和留出物种上评估。不包含裁剪、灰度化、模糊或曝光反转。 |

## 朝向策略

`sensitive` 是调用者显式选择的策略（程序不会自动推断），用于方向敏感标记：预训练的每个
全局/局部视图以及微调 `conservative` 都使用 `[-15°, 15°]` 旋转且不做水平翻转。之所以将其
作为默认值，是因为 50 轮实现对比使用的是当时的默认策略 `invariant`，并未评估 `sensitive`；
该对比未发现宽角度 `invariant` 旋转带来旋转一致性提升，且 ±180° 旋转并不是合理的常规数据增强。

`invariant` 是可选加入的宽角度旋转/翻转策略，用于方向不敏感标记，并保留现有行为。`legacy`
接受任一策略，但保留其历史翻转。

## 续训与继承

省略 `--augmentation` 和 `--orientation-policy` 时，续训会继承已保存的值；显式冲突会报错，
修改某个配置展开后的参数需要开启新运行。旧 pretrain checkpoint 映射为
`legacy`/`invariant`；旧 finetune checkpoint 映射为 `none`/`invariant`。

通过 `--checkpoint` 开始新的微调属于初始化而非续训：它不会继承预训练的数据增强配置，省略
策略时继承预训练 checkpoint 的策略（旧 checkpoint 回退为 `sensitive`）。

## 说明

**生物学约定。** 背面、腹面、侧面、整体以及解剖部位图像属于不同的标记，不得作为同一个标记的
等价视图混用；支持任意平面内朝向。完整标记的要求针对源图像，而不是每一个随机 SSL 裁剪视图。
全局与局部裁剪都是同一完整源标记的部分 SSL 观测，因此数据增强有助于提高嵌入一致性，但并不
保证嵌入不变性。

**输入尺寸。** 两种微调配置都使用 checkpoint 记录的训练输入尺寸；对于未记录尺寸的旧
checkpoint，回退到 `224`。

checkpoint 元数据（`config.augmentation_profile`、`config.augmentation_config`）用于配置
溯源，并不提供逐位确定性的复现。完整的变换级定义见
[训练数据增强设计](../superpowers/specs/2026-09-07-otuformer-training-augmentation-design.md)。

## 版本注记

- **0.2.1** — `legacy` 配置完全复现 OTU-Former 0.2.1 的数据增强，用于旧运行的续训与对比。
