# otuformer extract

[English](extract.md) | [中文](extract.cn.md)

## 目的

从检查点／ONNX 图提取图像级嵌入或空间 patch 网格，也可在不推理的情况下重读已完成的空间产物。embedding 写出供 `cluster`、`diversity`、`annotate` 使用的 `embeddings.csv`；spatial 写出几何、support 和诊断。有标签时计算嵌入指标；`--spatial-eval` 评价同图几何对应。两者衡量的能力不同。

## 用法

```bash
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --out-dir runs/extract
```

| 模式 | 最小输入 | 输出 |
| --- | --- | --- |
| Embedding（默认） | 图像＋checkpoint 或 embedding ONNX | 向量、可选标签指标／UMAP |
| Spatial 图像提取 | `--output-kind spatial`、图像＋checkpoint 或 spatial ONNX | 网格、诊断、可选冻结对应评价 |
| Spatial 读取 | `--output-kind spatial --spatial-input PRODUCT` | 独立目录中的报告，不需要模型 |

```bash
otuformer extract --output-kind spatial --checkpoint model.pt --input-images-dir ./images --spatial-eval --out-dir runs/spatial
otuformer extract --output-kind spatial --spatial-input runs/spatial --spatial-eval --out-dir runs/spatial_report
otuformer extract --checkpoint model.pt --input-images-dir ./images --feature-layers blocks.5 --dump-model-structure --out-dir runs/layer5
```

没有所有模式共同必需的输入选项，所以 help 不显示统一的必填 `*`；程序会在清理输出前校验各模式的条件要求。

## 参数

### `--checkpoint`

**默认值：** `None`

预训练／微调检查点。图像提取需要 checkpoint 或 ONNX；空间产物读取两者都不需要。

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

**默认值：** `None`（spatial：最后一个完整块；embedding：保留既有输出路径）

Spatial 可用逗号分隔多个完整块，例如 `blocks.5,blocks.11`，保留请求顺序。Checkpoint embedding 只接受一个完整的 `blocks.<index>`：施加 backbone 最终 norm 后取第零个 CLS token；要求 `--token-mode cls` 且不使用 projector。拒绝 ONNX、多层、patch-topk、attention-pool 组合。省略时保留既有 embedding 行为，不做多层平均／融合。`--feature-norm`／`--feature-l2-normalize` 仍仅用于 spatial；选层不会额外施加 embedding L2。

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

仅 spatial：启用冻结的同图跨视图对应（`spatial-geometry-v1`）。图像提取保存配对视图与查询计划；读取模式复用已保存计划，不做模型推理。缺少配对视图的产物不能通过补推理参与评价。

### `--spatial-eval-sample-size`

**默认值：** `100`

仅空间评测。用于对应的按确定性规则抽取的最大图像数。读取模式拒绝该选项；请改用 `--label-csv`。

### `--dump-model-structure`

**默认值：** `No`

Checkpoint embedding／spatial 共用：将 backbone 模块路径及空间、embedding-CLS 各自的完整块可用性写入 `model_layers.txt`，然后继续提取。不是无图像的单独列层命令，仍需要图像输入。ONNX 与空间读取显式拒绝。CSV／COCO 路径的有效 provenance 会记录 embedding 所选层。

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

ONNX Runtime 模型来源。Embedding 读取固定 CLS 图输出，保留旧的 checkpoint 覆盖行为；拒绝显式选层与结构导出。Spatial 按图声明的多层／norm／尺寸契约读取，拒绝 checkpoint／model-name／use-student／dump 覆盖，只接受与图契约完全匹配的其他显式值。空间产物读取拒绝 ONNX 输入。

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

# 使用已导出的 embedding ONNX 图
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

<a id="如何理解输出指标"></a>
## 说明

### 如何理解输出指标

Embedding 与 spatial 评价回答不同问题。分数高或张量数值健康，都不等于证明了解剖同源性或独立测试集泛化。

### Embedding：`metrics.csv`

指标实现与 `pretrain` 的 `logs/metrics.pretrain.csv`、`finetune` 的 `logs/metrics.finetune.csv` 共用。Extract 评价选中且可选抽样后的有标签图像；训练 CSV 还包含 epoch／split，比较时必须对齐 split 和样本群体。`metrics.csv` 是 `metric,value` 两列；比例值 `0.8` 表示 80%。

| 字段 | 含义与用途 | 如何判断 |
| --- | --- | --- |
| `Recall@1`、`Recall@5`、`Recall@10` | 排除查询自身后，余弦相似度前 k 个邻居中至少有一个同标签图像的查询比例；衡量检索成功率。 | 0–1，越高越好；只有一个样本的标签仍计入分母且无法成功。 |
| `mAP` | 全部非自身候选排序的平均精度均值，奖励将所有同标签图像排在前面。 | 0–1，越高越好；没有其他同标签项的查询被排除，分母与 Recall 不同。 |
| `kNN_Acc_k1/k5/k20` | 分层交叉验证下的余弦 k 近邻分类准确率。 | 0–1，越高越好；评价标签邻域，不代表 backbone 的独立测试泛化。 |
| `Linear_Probing_Acc` | 冻结向量上逻辑回归的交叉验证准确率。 | 0–1，越高越好；衡量线性标签可分性。 |
| `Linear_Probing_Balanced_Acc` | 同一次线性探测中各类别 recall 的均值。 | 0–1，越高越好；类别数量不均衡时更有参考价值。 |
| `NMI` | 真实标签与 KMeans 分配之间的归一化互信息。 | 0–1，越高越好。 |
| `ARI`／`AMI` | 经随机机会校正的配对一致性／互信息。 | 越高越好；1 完全一致，约 0 相当于随机，可为负。 |
| `Purity` | 每个 KMeans 簇采用多数标签后的正确样本比例。 | 0–1，越高越好；受指定簇数影响。 |
| `Silhouette_Score` | 以真实标签分组、余弦距离计算的簇内紧致／簇间分离程度；不是按预测簇计算。 | −1 到 1，较大表示分离较好，没有通用及格线。 |

KMeans 输入 L2 归一化向量，簇数等于观察到的标签类别数。CV 使用 seed 42 的打乱分层折，折数为 `min(5, 最小类样本数)`。k 不合法、类别仅一个样本或向量退化等可使指标不可用：CSV 空值不等于零；共享路径 provenance 区分 unavailable 和 failed。`--metrics-sample-size` 会改变评价样本；`--disable-umap` 只关闭图，不关闭指标。UMAP 是可视化，不是准确率。可用性和版本规则见[共享嵌入指标](embedding-metrics.cn.md)。图像级 CV 不保证按标本／分组隔离，train split 结果也不是测试集泛化证据。

### Spatial 诊断：`spatial_results.csv`

每行对应一张图像 × 一个视图 × 一层，包括 base 和已保存的评价视图；`spatial_diagnostics.json` 提供汇总。这里检查数值与 support，不是匹配准确率。

| 字段 | 含义与用途 |
| --- | --- |
| `sample_id`、`image_key`、`view_id`、`role`、`layer` | 稳定样本身份、视图与特征来源；按这些键对齐比较。 |
| `shape`、`grid_hw`、`dtype` | H×W×C 特征形状、H×W patch 网格和保存的数据类型。 |
| `finite_count`、`nonfinite_count` | 有限／非有限 token 向量数，不是通道标量数；出现非有限特征会使诊断失败。 |
| `zero_count`、`near_zero_norm_count` | 范数为零／不超过 1e−12 的 token 数，用于发现不可用或接近不可用的向量。 |
| `norm_min`、`norm_mean`、`norm_max` | token L2 范数统计；开启导出 L2 后约等于 1 是预期现象，不证明区分力。 |
| `channel_variance_mean` | token 内部跨通道方差的均值，不是跨位置／跨图片方差，不能单独检测特征塌缩。 |
| `coverage_min`、`coverage_mean`、`coverage_full_valid_fraction` | 像素级几何覆盖及完全覆盖比例，不是前景比例。 |
| `pixel_valid_fraction`、`patch_valid_fraction` | 保守采样有效像素／整个 footprint 有效的 patch 比例；填充、插值会减少 support。 |
| `support_version` | 覆盖／有效性规则版本，应按兼容契约比较。 |

### Spatial 对应：`spatial_matches.csv` 与 `spatial_evaluation.json`

`spatial_matches.csv` 每行是一条查询 × 方向 × 变换 × 层。`image_key`、`pair_id`、`direction`、`layer`、`query_index`、`row`、`column` 标识查询；`candidate_count` 为几何有效候选中心数，`invalid_candidate_count` 为其中特征不可用的数量。`match_status=matched` 只表示找到可用候选，不代表正确。`predicted_row/column` 是候选网格位置，`true_x/y`、`predicted_x/y` 是原图坐标。

- `raw_error`：原图像素单位的欧氏距离，越低越好。
- `normalized_error`：`raw_error / hypot(原图宽, 原图高)`，越低越好；分母是图像对角线，不是标本长度或包围框。
- `quantization_error`：全部有效目标网格中心能达到的最小归一化误差，是纯几何下限，不是模型误差；粗网格可使严格阈值无法达到。
- `pck_0p01`、`pck_0p02`、`pck_0p05`：**逐查询整数标记**，误差不超过 1%／2%／5% 为 1，否则为 0。匹配失败仍以零计入几何分母。例如误差 `0.0246` 对应 `0,0,1`；对标记求平均才得到比例。

JSON 先逐图计算，再按方向对图像做宏平均；两个方向都可用时 `combined` 等权平均。因此 `0.64` 表示 64%；整个 CSV 按行求平均可能与正式宏平均不同。变换／层分别报告，不发布跨变换总分；no-overlap 不可用分支也不等于准确率为零。结合 eligibility、量化下限和大误差尾部解释 PCK，没有通用及格线。比较时固定图像、查询计划、特征契约与基线。同图几何对应不是跨标本解剖对应，也不能代替分类／检索评价。

报告还给出几何 eligibility 和叠加图状态。可选 `spatial_overlays/` 使用固定长度的 SHA-256 键名；图标题／报告标识原图。读取器按 size／mtime／inode 身份验证一次 payload，各数组摘要仍逐次检查；这一优化假定源目录只读，不用于抵御恶意并发重写。匹配复用每个方向的固定候选几何，不改变冻结协议。

### 其他空间契约

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
