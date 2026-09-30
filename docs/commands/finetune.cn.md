# otuformer finetune

[English](finetune.md) | [中文](finetune.cn.md)

## 目的

用带标签数据对预训练骨干网络做监督度量学习微调。`--loss` 选择目标函数（`arcface`、`supcon`、`subcenter-arcface` 或 `subcenter-arcface-compact`），产出能把已知物种分开、供 OTU 聚类使用的嵌入向量。需要预训练检查点与带标签 CSV；可选的一轮额外流程可为同一批已知物种的未标注图像补充伪标签。

## 用法

```bash
otuformer finetune \
    --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --out-dir runs/finetune
```

## 参数

### `--checkpoint`

**默认值：** `None`

未设置 --resume 时使用的初始化检查点。SSL 预训练检查点会安装全新的嵌入头；头部匹配的微调检查点保留其已训练的投影头。 预训练检查点路径（通常为 `runs/pretrain/SSL_latest.pth`）；在 `--pseudo-label-from` 模式（仅用于定位已记录的 SSL 文件）与 `--resume` 下均可省略

### `--train-data`

**默认值：** `Required`

含 'image' 与 'label' 列的 CSV。

### `--input-images-dir`

**默认值：** `Required`

图像根目录。

### `--out-dir`

**默认值：** `runs/finetune`

输出目录。

### `--model-name`

**默认值：** `vit_tiny_patch16_224`

度量学习编码器的 timm 骨干网络名称。

### `--metric-embed-dim`

**默认值：** `None`

微调嵌入维度（度量嵌入头的输出，而非原始 CLS 维度）。默认：检查点记录的度量维度，否则为预训练投影头维度。 微调嵌入维度（度量嵌入头输出，非原始 CLS）。显式传值会重建该头；对历史 `ProjectionHead` 检查点会报错，其宽度由预训练投影器固定

### `--finetune-epochs`

**默认值：** `20`

微调总 epoch 数。

### `--finetune-lr`

**默认值：** `0.0001`

微调骨干网络的学习率。

### `--metric-head-lr`

**默认值：** `None`

嵌入头（以及原型损失下分类器）的学习率；默认取 --finetune-lr。

### `--weight-decay`

**默认值：** `0.0001`

微调的 AdamW 权重衰减；使用旧脚本设置时传 0.05。 微调使用的 AdamW weight decay；默认 `1e-4` 是保守的监督训练选择，传 `0.05` 可复现旧脚本设置

### `--freeze-ratio`

**默认值：** `0.7`

被冻结的骨干网络块比例（0.0=不冻结，1.0=全部）。--resume 时必须与已保存值一致。

### `--loss`

**默认值：** `arcface` · **允许值：** arcface / supcon / subcenter-arcface / subcenter-arcface-compact

度量学习损失：arcface（默认）、supcon、subcenter-arcface 或 subcenter-arcface-compact。名称未知时在产生任何输出之前拒绝。--resume 时省略 --loss 会继承记录的模式；显式冲突值会报错。 度量学习损失：`arcface`（默认）、`supcon`、`subcenter-arcface` 或 `subcenter-arcface-compact`；参见“度量损失模式（v0.8.0）”

### `--subcenters`

**默认值：** `2`

--loss subcenter-arcface 或 subcenter-arcface-compact 的每类中心数（K）：2 到 8 的整数。arcface 与 supcon 会拒绝该 flag。

### `--compact-weight`

**默认值：** `0.1`

--loss subcenter-arcface-compact 的同类别中心距离 hinge 权重（上限固定为 0.5）。其他损失模式会拒绝该项。

### `--supcon-temperature`

**默认值：** `0.07`

--loss supcon 的温度；必须 > 0。其他损失模式会拒绝该项。

### `--long-tail`

**默认值：** `none` · **允许值：** none / cb-drw

长尾策略：none（默认）或 cb-drw（对 ArcFace 族交叉熵项使用 Class-Balanced Deferred Reweighting）。--loss supcon 时会被拒绝，且伪标签各轮之间必须一致。 长尾策略：`none` 或 `cb-drw`（仅 ArcFace 家族；beta 0.99、cap 3.0、50% 起 10% 斜坡）

### `--pseudo-label-from`

**默认值：** `None`

已完成 finetune#1 的 ArcFace 族检查点，用于生成一轮自动的已知类别伪标签。finetune#2 仍从同一个原始 SSL 检查点初始化；在该模式下 --checkpoint 只用于在该 SSL 文件移动后定位它。省略的实验类参数继承 finetune#1 的解析值而非新运行默认值；显式冲突会报错。

### `--pseudo-similarity-floor`

**默认值：** `0.75`

未校准的胜出前三均值原始 CLS 余弦类别分数下限，取值 [-1, 1]。这不是概率。越高拒绝越多；越低接受越多。[默认: 0.75]

### `--pseudo-min-gap`

**默认值：** `0.1`

胜出分数与次高分数之差的下限，取值 [0, 2]。越高越会拒绝模糊候选；越低接受越多。[默认: 0.10]

### `--pseudo-neighbors`

**默认值：** `15`

非对称、排除自身类别的 mutual-kNN 检查的邻居数 k。越小通常越局部、越严格；越大通常越宽泛、越宽松（取决于数据）。[默认: 15]

### `--pseudo-cap-multiplier`

**默认值：** `3`

每类伪标签安全上限乘数：在 floor/gap/mutual-kNN 规则之后，每类最多接受 min(乘数 * 专家种子数, 绝对上限) 行。必须 >= 1。[默认: 3]

### `--pseudo-absolute-cap`

**默认值：** `50`

乘数之后应用的每类伪标签绝对上限。必须 >= 1。[默认: 50]

### `--trace-batch-ids`

**默认值：** `No`

选择写入 logs/batch_ids.finetune.jsonl，记录每个训练批次的图像 ID 顺序。文件可能很大；不会读取形态学信息。

### `--augmentation`

**默认值：** `None` · **允许值：** none / conservative

数据增强配置：none 或 conservative。新运行默认 none。`conservative` 为实验性，尚无证据优于 `none`。省略则在 --resume 时继承已保存的配置。

### `--orientation-policy`

**默认值：** `None` · **允许值：** invariant / sensitive

方向策略：invariant 或 sensitive。新运行默认 sensitive。影响 conservative；对 none 不生效。初始化时省略则继承检查点的策略。 方向策略：`sensitive`（新运行默认）或 `invariant`；仅影响 `conservative`。`--checkpoint` 初始化时省略则继承预训练 checkpoint 的策略；`--resume` 时省略则继承已保存策略

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

含 'image' 列与可选 'label' 列的 CSV，用于周期性指标与 UMAP。无标签时只生成 UMAP。省略时复用 --train-data。

### `--extract-size`

**默认值：** `auto`

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

微调期间关闭周期性嵌入指标与 UMAP 生成，以减少运行时开销。

### `--resume`

**默认值：** `None`

用于恢复的微调检查点路径（恢复模型/优化器状态；已保存的优化器设置会覆盖学习率与权重衰减选项）。

### `--overwrite`

**默认值：** `No`

清空已存在的非空输出目录。

## 输入

`--input-images-dir` 是图像根目录；`--train-data` 是含 `image` 与 `label` 列的 CSV，路径相对于该根目录。`--visualize-data` 是可选的独立 CSV，用于周期性指标与 UMAP。

## 输出

- `logs/finetune.log` — 运行日志
- `finetune_latest.pth`、`finetune_epoch_*.pth` — 模型检查点，写入 `--out-dir` 根目录（不存在 `checkpoints/` 子目录，也没有 `finetune_best.pth`）
- `logs/metrics.finetune.csv` — 周期嵌入指标
- `logs/instant_metrics.finetune.csv` — 每次迭代的训练指标
- `logs/loss_diagnostics.finetune.csv` — v0.8.0 每轮损失诊断
- `logs/batch_ids.finetune.jsonl` — 训练批次 ID 轨迹（仅当传入 `--trace-batch-ids`）
- `logs/training_curves_finetune.pdf` — 训练曲线图
- `logs/umap.train.epoch_<N>.pdf` — 周期 UMAP 图（未禁用时）

嵌入质量指标使用共享的 [嵌入指标](embedding-metrics.cn.md) 定义与字段名。

## 示例

```bash
otuformer finetune --train-data labels_union.csv --input-images-dir ./images \
    --out-dir runs/finetune --resume runs/finetune/finetune_latest.pth \
    --finetune-epochs 50
```

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
    --augmentation conservative \
    --finetune-epochs 50 \
    --finetune-lr 3e-4 \
    --freeze-ratio 0.5 \
    --loss arcface \
    --out-dir runs/finetune
```

```bash
# finetune#1 记录专家清单与 SSL 身份，供后续复用
otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv --input-images-dir DATA_ROOT/dorsal \
    --out-dir RUN_ROOT/finetune1 --finetune-epochs 20

# finetune#2：从同一原始 SSL 检查点出发的一轮自动伪标签
otuformer finetune --train-data labels.csv --input-images-dir DATA_ROOT/dorsal \
    --pseudo-label-from RUN_ROOT/finetune1/finetune_latest.pth \
    --out-dir RUN_ROOT/finetune2 --finetune-epochs 20
```

## 说明

单独的、仅含图像的 `--visualize-data` CSV 可用于 UMAP；监督指标至少需要两个标签类别。

如需为已有类别加入新图像，请使用同时包含旧图和新图标注的 CSV，保持相同的
`--out-dir`，从最新微调 checkpoint 使用 `--resume`，并增加
`--finetune-epochs`：

恢复时 CSV 必须保留完全相同的标签集合。增加新类别时，应使用
`--checkpoint` 新建微调，而不能使用 `--resume`。

**稀疏标签伪标签反馈。** 可选一轮，自动回收已知物种的未标注图像，无需候选清单输入：候选图是 `--input-images-dir` 下受支持图像减去专家 CSV 引用后的集合。请使用与 run 树分离的专用数据根（`DATA_ROOT/dorsal/` 与 `RUN_ROOT/finetune1/`）；`--out-dir` 不得与图像根重叠，所有文件型输入都必须位于 `--out-dir` 之外。

- 伪标签仅支持 ArcFace 家族损失且仅一轮；finetune#2 检查点不能再作为伪标签来源。
- 省略的实验类选项继承 finetune#1 的解析值，显式冲突会报错。finetune#2 始终从同一原始 SSL 检查点初始化；伪标签模式下 `--checkpoint` 仅用于 SSL 文件迁移后的定位（文件 SHA-256 必须一致）。
- 预处理固定为 `center-crop`；输出 `pseudo_labels.csv`（每个候选一行诊断）与 `pseudo_summary.json`。
- `--pseudo-similarity-floor`（默认 0.75）是未经校准的 raw-CLS 前三均值余弦类别分数，不是概率；floor 与 gap 越大越严格，`--pseudo-neighbors`（默认 15）越小通常越局部、越严格。每个类在三关之后最多接受 `min(--pseudo-cap-multiplier * 专家种子数, --pseudo-absolute-cap)` 行（默认 3 与 50）。
- 诊断 CSV 不是人工审批环节；接受集为空时会在创建输出目录前失败。
- `--long-tail cb-drw` 是独立的长尾选项（仅 ArcFace 家族，beta 0.99、cap 3.0、50% 起、10% 斜坡），两轮之间必须一致。
- `--input-images-dir` 必须是真实目录树（不遍历目录符号链接，并在 summary 中报告）；显式 `--visualize-data` 仍可用于任意 UMAP/指标诊断。
- RUN1/RUN2 对比请使用独立 `HELDOUT_ROOT`：`otuformer extract --input-images-dir HELDOUT_ROOT --label-csv HELDOUT.csv`；留在训练根内的留出图像会成为伪标签候选。

**检查点处理**：

- `--checkpoint` 开启新运行。SSL 预训练检查点会安装全新的嵌入头；头类型与宽度都匹配的微调检查点会保留已训练的投影器。历史 `ProjectionHead` 检查点保留其投影器，因此嵌入宽度由它决定。
- `--resume` 会恢复保存的优化器状态，因此 `--finetune-lr`、`--metric-head-lr`、`--weight-decay` 不生效。`--freeze-ratio` 必须与保存值一致。
- 旧脚本检查点（`ref/ibot20260115.py`）可被 `extract`、`export`、`cam` 读取，但不能被 `finetune` 续训。

**度量损失模式（v0.8.0）。** `--loss` 接受四种显式监督目标；未知名称或与解析出的模式不匹配的设置会在创建任何输出之前报错。ArcFace 仍是默认值，也是 v0.8.0 的参考损失。

| 模式 | 目标 |
|------|------|
| `arcface` | 每个已标注物种一个角度间隔分类中心（默认） |
| `supcon` | 单视图监督对比损失，基于批内图像对 |
| `subcenter-arcface` | 每个物种 K 个中心；样本靠近最近的中心 |
| `subcenter-arcface-compact` | 子中心 ArcFace 加同类别中心距离上限惩罚 |

- 分类中心是**仅训练期**状态：`extract`、`cluster` 及默认下游流程都不会读取它们，形态学元数据也从不进入训练。
- **原始 CLS 仍是对比目标。** 损失作用于微调头，而默认 `extract` 向量是骨干网络的原始 CLS 嵌入，因此训练损失更低并不能单独证明距离更好。跨模式比较应固定 `--freeze-ratio`、骨干与微调头学习率、轮数、数据增强、训练清单、批量大小和随机种子。**开集**比较需要在留出的已知物种上标定每种损失，再不重新调参地应用于未见物种。
- 首轮探索性设置：SupCon 温度 `0.07`、K `2`、余弦距离上限 `0.5`、compact 权重 `0.1`。这些是起始值，不是已验证的最优值；首轮不设参数网格，也不包含 benchmark runner。
- `supcon` 会跳过没有有效正样本或没有不同物种负样本的批次并报告跳过数量；若整个 epoch 都不可用则直接失败，而不是在空信号上保存 checkpoint。
- 新的 v0.8.0 checkpoint 会记录损失名称及其有效设置、随机种子、源 checkpoint 的 SHA-256、`train_manifest_sha256`、优化器布局与原型 weight decay 规则。新的 v0.8.0 ArcFace 运行对前向中 L2 归一化的原型使用**零 weight decay**，而 v0.7.x 会施加请求的 decay（默认 `1e-4`）。由于前向会归一化原型，该 decay 只缩放其范数，而损失不观测范数；在默认 `--finetune-lr 1e-4` 与 `--weight-decay 1e-4` 下，每步缩放低于 float32 分辨率，因此结果与 v0.7.x 数值完全一致。只有当学习率 × weight decay 的乘积更大时，原始原型范数才会改变并扰动轨迹，因此跨版本 checkpoint 不保证是仅损失不同的对照比较。旧 checkpoint 使用 `--resume` 会保留其保存的优化器语义；对 SSL 或无分类器的初始化 checkpoint 使用 `--resume` 会被拒绝。
- `logs/loss_diagnostics.finetune.csv` 每个完成的 epoch 记录一行（可用/跳过批次数、有效锚点数、满足间隔的比例、compact hinge 激活比例与惩罚、每类每中心的分配计数、中心方向余弦）。`logs/batch_ids.finetune.jsonl` 仅在 `--trace-batch-ids` 时写出。既有 `logs/metrics.finetune.csv` 与 `logs/instant_metrics.finetune.csv` 的 schema 不变。

数据增强与方向策略遵循共享的 [训练数据增强契约](training-augmentation.cn.md)。

嵌入质量指标使用共享的 [嵌入指标](embedding-metrics.cn.md) 定义与字段名。

## 版本注记

- **v0.9.0** — `--pseudo-label-from` 增加一轮自动的已知类别伪标签；`--long-tail` 选择长尾策略。
- **v0.8.0** — `--loss` 选择 `arcface`（默认）、`supcon`、`subcenter-arcface` 或 `subcenter-arcface-compact`，并带有各自的参数与损失诊断。
