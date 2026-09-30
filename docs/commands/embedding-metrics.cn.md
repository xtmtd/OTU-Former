# otuformer embedding metrics

[English](embedding-metrics.md) | [中文](embedding-metrics.cn.md)

## 目的

`pretrain`、`finetune` 与 `extract` 以相同字段名计算嵌入质量指标。本文定义这些共享指标、
各自的不可用条件、写入位置，以及定义发生过哪些变化。

各受影响命令各自记录自己的 flag 与默认值；本文只做链接，不重复这些内容。

## 适用命令

| 命令 | 指标写入位置 | 文档 |
|---|---|---|
| `otuformer pretrain` | `logs/metrics.pretrain.csv` | [pretrain](pretrain.cn.md) |
| `otuformer finetune` | `logs/metrics.finetune.csv` | [finetune](finetune.cn.md) |
| `otuformer extract` | 输出目录下的 `metrics.csv` | [extract](extract.cn.md) |

## 指标

- `Linear_Probing_Acc` 为普通 CV 准确率；`Linear_Probing_Balanced_Acc` 为同一 CV 过程的
  类别均衡准确率。
- 保留历史字段 `kNN_Acc_k1`、`kNN_Acc_k5`、`kNN_Acc_k20`。
- `mAP` 只对非查询样本排序，并排除没有非自身相关项的查询。`Recall@k` 刻意保留全部查询
  （含单例标签）作为分母；两者的单例查询口径不同，用于保持历史可比性。
- `Silhouette_Score` 是相对真实标签的余弦轮廓系数。

## 可用性规则

- kNN 与线性探测共用同一个显式打乱的
  `StratifiedKFold(shuffle=True, random_state=42)`；折数为 `min(5, 最小类别样本数)`，
  两类数据不再被强制为两折。某类只有 1 个样本时，所有 CV 指标不可用。
- 请求的 `k` 超过最小训练折（`Recall@k` 为 `n_samples - 1`）时按不可用处理，不再静默改名。
- 无法计算的指标写为空 CSV 字段，绘图时表现为断点而非 0。当归一化后的不同嵌入数少于
  类别数时，聚类指标按不可用处理，不再伪造单一簇。

## CSV 字段

`extract` 在 `--label-csv` 至少包含两个类别时写入 `metrics.csv`。三个命令使用相同的字段名：
`kNN_Acc_k1`、`kNN_Acc_k5`、`kNN_Acc_k20`、`Linear_Probing_Acc`、
`Linear_Probing_Balanced_Acc`、`mAP`、`Recall@k`、`Silhouette_Score`。

## 版本注记

- **v0.7.1** — CV 折选择、子采样顺序、`mAP` 自包含与不可用值处理均已改变，因此旧值不可直接
  比较。字段名与历史行仍可读取；`--resume` 的 v0.7.0 `metrics.pretrain.csv` /
  `metrics.finetune.csv` 会就地补一列空的 `Linear_Probing_Balanced_Acc`。
