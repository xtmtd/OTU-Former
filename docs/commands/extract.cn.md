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

**默认值：** `Required`（空间产物读取模式除外）

图像目录或父目录。嵌入与空间图像抽取必填；空间产物读取模式可省略，此时它只用于报告叠加的图像。

### `--output-kind`

**默认值：** `embedding` · **可选值：** embedding / spatial

要写出的产物：`embedding` 保持既有向量输出；`spatial` 写出完整的 backbone 块特征网格，并附带几何与诊断。任何选项都不会隐式切换它。

### `--spatial-input`

**默认值：** `None`

要读取的既有已完成空间产物。需要 `--output-kind spatial`；数值诊断不需要模型或图像，且源产物永不被修改。

### `--feature-layers`

**默认值：** `None`（最后一个完整块）

仅空间模式。逗号分隔的 backbone 相对完整块路径（例如 `blocks.5,blocks.11`）。省略时取最后一个完整块。请求顺序被保留；不支持区间、重复项或取平均。

### `--feature-norm`

**默认值：** `backbone` · **可选值：** backbone / none

仅空间模式。对每个选定块施加 backbone 的最终归一化（`backbone`），或导出原始块输出（`none`）。

### `--feature-l2-normalize`

**默认值：** `No`

仅空间模式。在 backbone 归一化之后对每个 token 的通道向量做 L2 归一化。与 CAM 热图的 min-max 缩放无关。

### `--spatial-layout`

**默认值：** `per-image` · **可选值：** per-image / sharded

仅空间抽取。`per-image` 为每个样本写一个未压缩 NPZ；`sharded` 写有上界的 NPY 分片。两者共用同一个读取器。

### `--spatial-eval`

**默认值：** `No`

启用冻结的同图跨视图对应（协议 spatial-geometry-v1）。在当前构建中，评测任务尚未交付前一律拒绝，绝不静默跳过。

### `--spatial-eval-sample-size`

**默认值：** `100`

仅空间评测。用于对应的按确定性规则抽取的最大图像数。读取模式拒绝该选项；请改用 `--label-csv`。

### `--dump-model-structure`

**默认值：** `No`

仅检查点的空间模式。把 backbone 相对模块路径及其空间可用性写入 `model_layers.txt`，然后继续抽取。

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

**默认值：** 嵌入输出为 `center-crop`，空间输出为 `whole-specimen-pad` · **允许值：** center-crop / whole-specimen-pad

评估预处理协议：center-crop（Resize + CenterCrop）或 whole-specimen-pad（保持宽高比的方形填充）。默认：嵌入输出用 center-crop，空间输出用 whole-specimen-pad；两种模式下显式指定都以请求为准。

### `--use-projector-output`

**默认值：** `No`

对 SSL 检查点使用 SSL 投影头输出；对微调检查点使用 ArcFace 嵌入。适用于微调类别的检索与闭集聚类。

对 DINO 的 SSL 检查点，该选项返回归一化后的 D 维瓶颈；K 维原型 logits 仅用于训练，从不作为提取输出。两种 SSL 方法的默认输出仍是骨干 CLS，微调检查点则保持其度量头宽度。

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

含必需 `image` 列及可选 `label` 与其他元数据的 CSV。提供该选项即启用共享的严格校验数据路径：一个文件同时是选择器、元数据来源与指标/UMAP 的标签来源。仅 image 时生成 UMAP，不计算监督指标；attention-pool 训练仍需要 `label`，也可由缓存的 pooling 权重替代。

### `--coco-instances`

**默认值：** `None`

v2 共享数据路径：COCO 实例标注（box、polygon、RLE）。

### `--coco-keypoints`

**默认值：** `None`

v2 共享数据路径：COCO 关键点标注。

### `--coco-panoptic`

**默认值：** `None`

v2 共享数据路径：COCO panoptic JSON，需同时提供 `--panoptic-masks-dir`。

### `--panoptic-masks-dir`

**默认值：** `None`

panoptic segment-ID PNG 的根目录；按标注中的 `file_name` 原样解析。需同时提供 `--coco-panoptic`。

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

本次运行读取的每个输入——图像根、`--label-csv`、checkpoint、ONNX 图及其外置张量文件、
可复用的 pooling 检查点——在清理任何内容之前先收集完毕；如果清空输出目录会删掉其中之一，
就直接拒绝。ONNX 图的外置数据指针如果含义不唯一（重复的 `location` 条目），或解析到图目录
之外，都会被拒绝：不同读取器对这种图会读到不同文件。链接按内核语义跟随，包括转接链接与链接之后的 `..`，因此即使文件本身在别处，
只要清理会删掉路径上的一级目录项，也会被拒绝。

两条数据路径在一处规则上不同，这是有意保留的：

- **共享路径**（提供 `--label-csv` 或任一 `--coco-*`）：输出目录包含输入、或位于输入内部，
  一律拒绝。
- **legacy 路径**（既无 CSV 也无 `--coco-*`）：只拒绝“删除”方向。`--out-dir` 位于图像根
  **内部**仍然支持（运行只在那里创建文件）；输出目录等于图像根、或包含图像根时拒绝。

**并发边界。** 检查发生在清空之前，因此这是本地单进程契约：它不防御另一个进程在检查与
清空之间改动目录树（经典的 check-then-act 窗口）。运行保证的是**可检测**——清空之后，
每个受保护输入必须仍解析到同一个真实路径；若不是，运行立即停止并指名该输入，而不是沿着
已经不再等于被检查路径的入口继续。

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

**空间产物（`--output-kind spatial`）。**

`--output-kind spatial` 完全不改动嵌入行为，而是写出独立产物，替代 `embeddings.csv`。同一个
`extract` 命令只依据请求本身解析出三种空间模式之一：

| 模式 | 必需输入 | 行为 |
| --- | --- | --- |
| 空间检查点 | `--checkpoint` + 图像（可选 CSV/COCO） | 单次前向读取所选完整块，映射几何与标注，写出诊断 |
| 空间 ONNX | 已导出的空间图 + 图像 | 只读图声明的层与契约；模型、层、归一化与尺寸由图固定 |
| 空间读取 | `--spatial-input` | 校验并重开已存样本；诊断或兼容的重评测写入独立运行目录 |

- **选择。** 提供 `--label-csv` 时，严格按其行的 CSV 顺序选择图像。未提供时递归扫描根目录下
  所有受支持图像，因此根目录与嵌套图像都会被包含。COCO 源只挂接标注，绝不扩大或收窄集合。
  这是空间模式相对 legacy“CSV > COCO > 目录”选择器的专属偏离，嵌入抽取保持不变。
- **不推断分组。** 只有 CSV 显式提供的 `sample` 才会保留。由目录推断出的 `sample` 会从索引
  与报告中抑制，因此不会凭空造出标本分组。
- **存储。** 默认 `--spatial-layout per-image`；`--spatial-layout sharded` 写出于固定未压缩
  载荷预算内封口的 NPY 分片。布局绝不改变逻辑数组、顺序或任何诊断行。
- **支撑记录。** 每个存储视图在模型输入画布上携带 `pixel_geometric_coverage` /
  `pixel_sampling_valid`，在 patch 网格上携带 `geometric_coverage` / `patch_sampling_valid`。
  覆盖率是几何比例，有效性是保守重采样支撑。两者都不是前景或解剖掩码。
- **空间 ONNX。** 使用空间图时 `--onnx-path` 只读取图声明的输出与契约：层、特征归一化、L2
  与输入尺寸在导出时即已固定，因此显式冲突值会直接失败而不会被重新推导；`--model-name`、
  `--use-student`、`--dump-model-structure` 一律拒绝，因为图无法被改名、换权重或列层。几何、
  覆盖与诊断由同一套共享预处理产生，因此对同一批图像，图与检查点得到的逻辑产物一致。
- **读取是显式的。** 不做自动发现、续跑、修补或推断。读取运行永不修改源产物，数值诊断不需要
  模型，并在写出前拒绝抽取类覆盖选项（检查点、层、归一化、布局、尺寸、变换、COCO 源、模型名、
  学生权重）。
- **读取结果是报告，不是产物。** 读取结果目录只保存所选样本的诊断；它不是可复用的空间产物。
  如果 `--spatial-input` 判定它不完整，请把 `--spatial-input` 指向原始的已完成抽取目录
  （含 `spatial_manifest.json` 与 `spatial_index.jsonl` 的那个），该目录由报告的
  `source_product` 溯源信息指明。
- **成本（算术估算，非实测）。** 评测会为每张抽样图像存下一个参考视图加十一个变换视图：14x14 网格、
  C=192 时，100 张图像每层约 172 MiB；224x224 视图下，存储的像素支撑在跨层共享后约 287 MiB。
  分片只改变物理打包方式：每个分片未压缩载荷上限 256 MiB，绝不改变逻辑数组、顺序或任何诊断行。
- **L2 是表示选择。** `--feature-l2-normalize` 记录网格里装的是什么；余弦匹配在特征采样之后总会归一化，
  因此 v1 没有任何评分必须依赖单位范数导出，原始导出与单位化导出是两个不同的契约，不可互换。
- **限制。** 完全有效的 patch 排除了直接的填充污染，但 ViT 的全局注意力仍可能把无效 token 的
  影响传播进来。

- **评测（`--spatial-eval`）。** 启用评测时，被选中的图像（默认最多 100 张，按既有 `--seed`
  以 SHA-256 排序确定性抽取）会额外存下一个参考视图与十一个几何视图，并写出
  `spatial_evaluation_plan.json`、`spatial_matches.csv` 与 `spatial_evaluation.json`。查询取自
  保守有效的源 patch 中心，其映射点与目标插值支撑都必须有效；每个 pair、每个方向最多 64 个，
  用同一套带种子的排序选出，候选集是**全部**有效目标中心——不使用标签、掩码或真值位置做筛选。
  评分为余弦匹配，PCK 阈值为原图对角线的 0.01、0.02、0.05，先按图像聚合再按变换/层/方向做
  宏平均，并明确给出分母。失败匹配（特征无效、无可用候选）仍计入 PCK 分母，且不产生有限误差。
  读取模式复用已存计划：CSV 只能收窄、`--spatial-eval-sample-size` 被拒绝、显式 `--seed` 必须等于
  计划中的种子；产物若无配对视图会以 `missing_paired_views` 报错，而不是静默跳过请求。

可读取 OTU 检查点与旧脚本检查点（`ref/ibot20260115.py`）。嵌入头优先按 `config.embedding_head` 重建，缺失时按权重中的投影器形状推断。

`--label-csv` 需要 `image` 列，并接受可选的 `label` 列。仅提供图像时仍会生成 UMAP，但不产出监督指标；attention-pool 查询训练仍然需要 `label`。

嵌入质量指标使用共享的 [嵌入指标](embedding-metrics.cn.md) 定义与字段名。

`cam` 与 `extract` 使用原生校验：在创建任何输出目录之前，对枚举取值之外的输入直接拒绝。

**共享数据路径（v2）。** 下列选项共同启用严格校验的数据路径，并与旧提取行为保持兼容。

提供 `--label-csv` 或任一 `--coco-*` 选项即切换到共享的严格校验数据路径；一个都不
提供时（仅目录枚举），保持上面描述的旧行为（包括其 pandas CSV 解析方式）。

- **解析。** 共享路径只解析一次选择器，同一份记录供选择、attention-query 训练、评价
  与 provenance 使用。字面量 `NA`、`N/A`、`unknown`、`0` 与前导零标识符按字符串保留，
  只有空单元格才算缺失。
- **身份。** 内部键是相对 `--input-images-dir` 的规范化 POSIX 路径。CSV 选中的行
  沿用原始 `image` 文本作为输出 `id`；仅由 COCO 选中的行使用该键。COCO 的
  `file_name` 按显式相对根路径解析，不做 CSV 的 basename 回退。
- **标注。** COCO 文档按来源独立：`instances`、`keypoints`、`panoptic` 是固定角色，
  每个角色最多一个文件；同一个文件可同时用于实例与关键点角色。类别 ID 只在来源内
  有效，不会变成物种标签。摄入 box、polygon、RLE、keypoint 与 panoptic 只做读取
  与校验，不训练检测器、head 或任务预测。
- **panoptic 根目录。** 每个标注的 `file_name` 按原样拼接到 `--panoptic-masks-dir`。
  不自动加前缀、不自动剥前缀、不搜索备用目录：带前缀的名字需要把父目录作为根。
- **掩码依赖。** polygon 与 RLE 解码需要可选依赖：
  `pip install 'otuformer[annotations]'`。缺失时，需要它的来源会直接报错并给出安装
  指引，而不是跳过校验。panoptic 的 segment-ID PNG 解码与像素校验只依赖 Pillow 与
  NumPy，不需要该可选依赖。
- **校验范围。** 结构性问题（类型、ID、引用、数值负载、RLE 总长度）对整份来源检查；
  与真实图像相关的几何、解码与像素一致性只对选中图像检查；未选中的文件记为“未执行”，
  且格式正确但未被选中的图像不必在本地存在。
- **缺标注不等于生物学缺失。** 图像不在某来源中、图像存在但无标注、空掩码、关键点
  `v=0` 与 `v=1`、panoptic void 是彼此不同的状态；不生成负掩码，也不把未标注像素当作
  可信背景。自定义 `ignore`／`unknown` 字段原样保留但不作解释。

共享路径在 `--out-dir` 下新增三个产物：

| 文件 | 内容 |
|---|---|
| `embeddings.csv` | 与旧路径相同的 embedding 列 |
| `data_records.jsonl` | 每个输出向量一行，按 embedding 顺序，含 `schema_version`、`embedding_row`、`output_id`、`image_key`、尺寸、图像 SHA-256、元数据与 `(source_id, image_id)` 引用 |
| `data_validation.json` | 状态、范围、计数、方法与容差、完整 issue 与 polygon 栅格化诊断 |
| `run_provenance.json` | 运行状态、请求与实际设置、来源身份与哈希、选择与内容摘要、标签 schema、指标协议与产物清单 |

`artifacts` 列表记录本次运行产生的每个文件，含输出内相对路径、字节数与 SHA-256。其中
包括本次训练出的 pooling 检查点（`checkpoints/<backbone>.<policy>-<type>_pooling.pth`）：
它是"本次生成了哪份可复用权重"的唯一记录；若运行在发布前失败，回滚会删除它，而不是留下
未登记的文件。从磁盘复用的 pooling 检查点属于输入，出现在 `used_files` 中，且永远不会被删除。

本次训练出的 pooling 检查点用独占创建发布：绝不覆盖已存在的文件；占位之后任何一步失败，
都会删除本次创建的那个文件。该保证针对目录项，不针对可见性——另一个进程在本运行仍在写入
时读取该路径，可能看到不完整的文件。请只在产出方运行返回之后再读取该缓存。

输出目录通过安全检查后，运行先写入状态为 `running` 的 `run_provenance.json`，随后写
`completed`、`failed` 或 `interrupted`。校验失败只发布 `data_validation.json` 与失败
记录，不发布 embedding/映射对；硬中断可能留下 `running`，它永远不代表成功。不带新
选项的旧运行不会生成这些文件。
