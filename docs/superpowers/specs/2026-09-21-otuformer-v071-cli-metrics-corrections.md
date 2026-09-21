# OTU-Former v0.7.1 CLI and embedding-metric contract

Date: 2026-09-21
Version target: `0.7.1`
Implementation plan:
[`docs/superpowers/plans/2026-09-21-otuformer-v071-cli-metrics-corrections.md`](../plans/2026-09-21-otuformer-v071-cli-metrics-corrections.md)

This document is the current, binding description of how `pretrain`,
`finetune`, and `extract` compute embedding-quality metrics and how the CLI
validates query parameters. Historical reports keep their recorded values; see
the evaluator-vintage notes in
`docs/superpowers/specs/2026-03-30-otuformer-pretrain-alignment-design.md` and
`docs/superpowers/specs/2026-09-07-global-barcode-validation.md`.

## Public metric keys

Single shared evaluator: `src/otuformer/embedding/evaluator.py`.
Adapters: `src/otuformer/training/trainer.py` (periodic logs and plots) and
`src/otuformer/cli/extract.py` (standalone `metrics.csv`).

| Key | Adapters | Meaning |
| --- | --- | --- |
| `Recall@1`, `Recall@5`, `Recall@10` | trainer, extract | Top-k retrieval with the query excluded from its own candidate list |
| `kNN_Acc_k1`, `kNN_Acc_k5`, `kNN_Acc_k20` | trainer, extract | Cosine kNN accuracy under stratified CV |
| `Linear_Probing_Acc` | trainer, extract | Ordinary linear-probe CV accuracy |
| `Linear_Probing_Balanced_Acc` | trainer, extract | Class-balanced linear-probe CV accuracy |
| `mAP` | trainer, extract | Mean average precision over non-query candidates |
| `NMI`, `ARI`, `AMI` | extract (trainer: `NMI`, `ARI`) | KMeans agreement with the true labels |
| `Silhouette_Score` | trainer, extract | Cosine silhouette against the true label codes |
| `Purity` | trainer, extract | Cluster-purity of the KMeans assignment |

Field names are stable: existing keys are never renamed or removed, and
`extract` emits every trainer field.

## CV splitter contract

One private helper, `_make_stratified_cv(labels, max_splits=5)`, is the sole
source of CV configuration for kNN and linear probing:

```python
StratifiedKFold(n_splits=min(5, smallest_class_count), shuffle=True, random_state=42)
```

- The fold count is bounded by the smallest class count, never by the number of
  classes: a two-class dataset with at least five samples per class uses five
  folds.
- A singleton class makes stratified CV impossible, so the helper returns
  `None` and every CV metric is unavailable.
- A requested `k` larger than the smallest training fold
  (`len(labels) - ceil(len(labels) / n_splits)`) is unavailable rather than
  silently reduced.
- kNN uses one `cross_val_score` call per `k`; linear probing uses one
  `cross_validate` call with the scorers `{"accuracy": "accuracy",
  "balanced_accuracy": "balanced_accuracy"}` so both probe values come from the
  same pass.

## Unavailable-value behavior

Unsupported results are `None` at the evaluator boundary, `""` in CSV, and gaps
(not zeros) in plots. Concretely:

- `Recall@k` is unavailable for an empty dataset, a one-sample dataset, or
  `k > n_samples - 1`. Each requested `k` is decided independently.
- `mAP` is unavailable when no query has a non-self relevant item.
- Clustering metrics are unavailable when the number of distinct normalized
  embeddings is less than the number of requested clusters. No single-cluster
  fallback is fabricated.
- `Silhouette_Score` is unavailable when silhouette is undefined (all classes
  are singletons).

## Sampling-order rule

Both adapters subsample with the same seeded, sorted index order:

```python
idx = np.sort(np.random.default_rng(seed).choice(n, size=max_samples, replace=False))
```

There is no sample-ID canonicalization pipeline; trainer and extract assume the
same row universe. Sorting keeps fold assignment, KMeans initialization, and
tie-breaking aligned between the two paths.

## Recall and mAP definitions

- `Recall@k`: for each query, rank all other samples by cosine similarity and
  count the query as correct when at least one same-label candidate appears in
  the top `k`. The denominator is every query, including singleton labels.
- `mAP`: for each query, rank only non-query samples, compute average precision
  over the relevant ones, and exclude queries with no relevant non-query item
  from the mean.

The two singleton-query conventions differ on purpose for historical
comparability and must not be treated as interchangeable denominators.

## CLI parameter-source and choice compatibility

`typer>=0.26` vendors Click, so its `ParameterSource` and `ParamType` classes are
not the external `click` classes.

- Parameter-source checks compare by name:
  `getattr(source, "name", None) == "COMMANDLINE"`. `pretrain.py` routes both
  `_format_user_command()` and `explicit_sources` through
  `_source_is_commandline()`.
- No module under `src/otuformer/cli/` imports external `click` or passes a
  Click type through `click_type`. `cam` and `extract` validate enumerated
  options with local `*_CHOICES` tuples and `typer.BadParameter` before
  `prepare_output_dir()` runs, and their `--help` text lists the accepted values
  (`--arch` `cnn, vit`, `--fig-format` `png, jpg, pdf`, `--device`
  `auto, cpu, cuda, mps`).

## Compatibility and migration

- Historical `kNN_Acc_k*`, `Linear_Probing_Acc`, and clustering values are not
  numerically comparable with v0.7.1 outputs because fold selection, sample
  order, mAP self-inclusion, and unsupported-value handling changed.
- A resumed v0.7.0 `logs/metrics.pretrain.csv` or `logs/metrics.finetune.csv`
  is migrated atomically in place: the recognized v0.7.0 header is rewritten
  with an empty `Linear_Probing_Balanced_Acc` column and all existing rows are
  preserved. A header that already matches the current schema is appended to
  without rewriting. Duplicate-column, malformed, and unrecognized headers
  raise `ValueError` before any append.
- `extract` rejects duplicate `image` rows in `--label-csv` with `ValueError`
  before metric computation, so labels cannot silently desynchronize from
  embeddings.
