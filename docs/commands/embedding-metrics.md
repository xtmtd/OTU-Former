# otuformer embedding metrics

[English](embedding-metrics.md) | [中文](embedding-metrics.cn.md)

## Purpose

`pretrain`, `finetune`, and `extract` compute the same embedding-quality metrics
under the same field names. This document defines those shared metrics, when each
one is unavailable, where the values are written, and how the definitions changed.

Each affected command documents its own flags and defaults; this document links
to them rather than restating them.

## Affected Commands

| Command | Where the metrics are written | Document |
|---|---|---|
| `otuformer pretrain` | `logs/metrics.pretrain.csv` | [pretrain](pretrain.md) |
| `otuformer finetune` | `logs/metrics.finetune.csv` | [finetune](finetune.md) |
| `otuformer extract` | `metrics.csv` in the output directory | [extract](extract.md) |

## Metrics

- `Linear_Probing_Acc` is the ordinary CV accuracy; `Linear_Probing_Balanced_Acc`
  is the class-balanced accuracy from the same CV pass.
- The historical fields `kNN_Acc_k1`, `kNN_Acc_k5`, and `kNN_Acc_k20` are
  retained.
- `mAP` ranks only non-query samples and excludes queries with no non-self
  relevant item. `Recall@k` intentionally keeps every query in its denominator,
  including singleton labels; the two singleton-query conventions differ on
  purpose for historical comparability.
- `Silhouette_Score` is the cosine silhouette computed against the true labels.

## Availability Rules

- Cross-validated kNN and linear-probe scores share one explicit shuffled
  `StratifiedKFold(shuffle=True, random_state=42)`. The fold count is
  `min(5, smallest class count)`, so a two-class dataset is no longer forced to
  two folds. Every CV metric is unavailable when a class has a single sample.
- A requested `k` larger than the smallest training fold (or than
  `n_samples - 1` for `Recall@k`) is unavailable rather than silently renamed.
- Metrics that cannot be computed are written as empty CSV fields and appear as
  gaps, not zeros, in training plots. Clustering metrics are unavailable when
  there are fewer distinct normalized embeddings than classes, instead of
  fabricating a single cluster.

## CSV Fields

`extract` writes `metrics.csv` when `--label-csv` provides at least two classes.
All three commands use the same field names: `kNN_Acc_k1`, `kNN_Acc_k5`,
`kNN_Acc_k20`, `Linear_Probing_Acc`, `Linear_Probing_Balanced_Acc`, `mAP`,
`Recall@k`, and `Silhouette_Score`.

## Version Notes

- **v0.7.1** — CV fold selection, subsample ordering, `mAP` self-inclusion, and
  unsupported-value handling changed, so older values are not comparable. Field
  names and historical rows stay readable, and a resumed v0.7.0
  `metrics.pretrain.csv`/`metrics.finetune.csv` gains an empty
  `Linear_Probing_Balanced_Acc` column in place.
