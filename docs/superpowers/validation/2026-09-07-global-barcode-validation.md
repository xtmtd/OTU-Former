# `global-barcode` implementation comparison — 2026-09-07

This report records a bounded implementation comparison between the historical
baseline checkpoint and a newly trained `global-barcode`/`invariant`
checkpoint. It compares available implementation metrics and rotation/reflection
consistency on one two-species dataset. It does not establish broad biological
suitability of the new default.

## Data identity and sample counts

| Item | Value |
| --- | --- |
| Manifest CSV | `examples/Epidorcus/figs.csv` |
| Manifest SHA-256 | `60c5631699e35247db998edeefbf6a5c33a5b243bf19a04112e6506f35d35482` |
| Rows (images) | 230 |
| Species (labels) | 2 — `Epidorcus_tonkinensis` 128, `Epidorcus_gracilis` 102 |
| Image root | `examples/Epidorcus/images` |
| Images resolved by validation | 230 |

The same manifest and image root were used for baseline training, candidate
training, and the rotation/reflection validation.

## Baseline and candidate checkpoints

| Role | Path | SHA-256 |
| --- | --- | --- |
| Baseline | `runs/old/pretrain/SSL_epoch_0050.pth` | `d0279e93a4081d47bb0bbd92bc39171edf89049c1f9108eab7da902e36d052db` |
| Candidate | `runs/pretrain-global-barcode-50e/SSL_latest.pth` | `e807db8fea848e556b4d40c1fbd0190613baff32b119a10f3cbe54641008dc74` |

Both checkpoints store `config.model_name = vit_tiny_patch16_224`,
`config.out_dim = 256`, resolved training image size 224, and top-level
`epoch = 49` (0-indexed, i.e. 50 completed epochs). Both load with 0 missing and
0 unexpected `model_state_dict` keys.

## Exact commands and environment

### Candidate 50-epoch training (executed command, as recorded in `runs/pretrain-global-barcode-50e/logs/pretrain.log`)

```bash
otuformer pretrain \
  --train-data examples/Epidorcus/figs.csv \
  --input-images-dir examples/Epidorcus/images \
  --out-dir runs/pretrain-global-barcode-50e \
  --augmentation global-barcode \
  --model-name vit_tiny_patch16_224 \
  --out-dim 256 \
  --max-epochs 50 \
  --lr 0.0005 \
  --weight-decay 0.05 \
  --warmup-epochs 3 \
  --global-crop-size 224 \
  --local-crop-size 96 \
  --local-crops 6 \
  --mask-ratio 0.5 \
  --lambda-local 1.5 \
  --lambda-mask 1.0 \
  --teacher-momentum 0.995 \
  --teacher-momentum-end 0.999 \
  --student-temp 0.1 \
  --teacher-temp-start 0.04 \
  --teacher-temp-end 0.07 \
  --batch-size 32 \
  --num-workers 4 \
  --cpus 12 \
  --device auto \
  --seed 42 \
  --metrics-sample-size 10000 \
  --umap-n-neighbors 15 \
  --umap-min-dist 0.1 \
  --umap-metric cosine \
  --visualize-class-number 20 \
  --log-every-n-steps 50 \
  --save-every-epochs 10 \
  --keep-last-checkpoints 10
```

The plan's Step 3 command additionally listed `--extract-size auto`. That flag
was omitted in the executed run; it is equivalent because the CLI default is
`auto` and the run log records `[Info] Auto eval crop size from model img_size:
224`. All other listed data, seed, model, crop, optimizer, batch, and metric
settings match the plan command.

### Rotation/reflection validation

```bash
python scripts/validate_augmentation_rotation.py \
  --baseline-checkpoint runs/old/pretrain/SSL_epoch_0050.pth \
  --candidate-checkpoint runs/pretrain-global-barcode-50e/SSL_latest.pth \
  --images-csv examples/Epidorcus/figs.csv \
  --images-dir examples/Epidorcus/images \
  --output-json runs/pretrain-global-barcode-50e/rotation-validation.json \
  --device auto
```

Output JSON: `runs/pretrain-global-barcode-50e/rotation-validation.json`
(gitignored, not checked in). Resolved device `mps`; effective evaluation image
size 224 for both checkpoints.

### Environment

| Component | Version |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.12.1 |
| torchvision | 0.27.1 |
| timm | 1.0.27 |
| Device | `mps` (resolved by `--device auto`) |

## Resolved augmentation profile/config

- **Candidate:** `augmentation_profile = global-barcode`,
  `augmentation_config.orientation_policy = invariant`. Saved geometric
  contract: rotation `[-180, 180]` degrees, bicubic, `expand=true`,
  edge-median-RGB fill; horizontal flip probability 0.5; vertical flip 0.0;
  global crop 224 at scale `[0.4, 1.0]`; 6 local crops of 96 at scale
  `[0.05, 0.4]`; color jitter 0.8; grayscale 0.0; blur kernels 23 (global) and
  7 (local).
- **Baseline:** the checkpoint has no `augmentation_config` (legacy
  checkpoint); its saved `args` contain no augmentation fields. Its historical
  augmentation contract is not recoverable from the checkpoint.

## Training health

Source: `runs/pretrain-global-barcode-50e/logs/instant_metrics.pretrain.csv`
(7 rows, one per `log_every_n_steps=50`) and `pretrain.log`.

| Check | Result |
| --- | --- |
| 50 epochs completed | Yes. The log records metrics at epochs 10, 20, 30, 40, 50 and `SSL pretraining complete`; both checkpoints store `epoch = 49` (0-indexed). |
| Every logged loss finite | Yes. Logged loss range 3.0735–8.7792; no `NaN`/`Inf` in any logged numeric column. |
| Last available finite positive `feature_std` | `2.8874013423919678` at epoch 42, step 6. This is the last logged value, not a checkpoint-final value. |
| Different-image similarities not all one | Yes. Candidate different-image median 0.6432 over 26,335 pairs, so embeddings did not collapse to a constant. |

## Seven-metric baseline/candidate comparison

Final available rows (`epoch = 50`) from each run's `metrics.pretrain.csv`. All
seven metrics were present for both runs; none is `unavailable`.

| Metric | Baseline (epoch 50) | Candidate (epoch 50) | Delta (candidate − baseline) |
| --- | --- | --- | --- |
| Recall@1 | 0.9739 | 0.9783 | +0.0043 |
| kNN accuracy k=1 | 0.9043 | 0.9043 | 0.0000 |
| kNN accuracy k=5 | 0.9261 | 0.9565 | +0.0304 |
| kNN accuracy k=20 | 0.9478 | 0.9696 | +0.0217 |
| Linear-probe accuracy | 0.9609 | 0.9870 | +0.0261 |
| mAP | 0.7782 | 0.7702 | −0.0080 |
| Silhouette Score | 0.2224 | 0.2271 | +0.0047 |

## Rotation same-image consistency

Per-image cosine similarity between the angle-0 view and each rotated view;
summary over 230 images. Higher means the embedding changes less under rotation.

| Angle | Baseline median | Baseline p10 | Baseline min | Candidate median | Candidate p10 | Candidate min |
| --- | --- | --- | --- | --- | --- | --- |
| 90° | 0.6827 | 0.6189 | 0.5508 | 0.6481 | 0.5795 | 0.5066 |
| 180° | 0.8465 | 0.7842 | 0.6823 | 0.7901 | 0.6960 | 0.5784 |
| 270° | 0.6890 | 0.6184 | 0.5611 | 0.6616 | 0.5845 | 0.5014 |

## Reflection same-image consistency

Per-image cosine similarity between the angle-0 view and its horizontal
reflection; summary over 230 images. Diagnostic only for the candidate
`invariant` policy; not an invariance acceptance target on its own.

| View | Baseline median | Baseline p10 | Baseline min | Candidate median | Candidate p10 | Candidate min |
| --- | --- | --- | --- | --- | --- | --- |
| Horizontal reflection | 0.9691 | 0.9509 | 0.9150 | 0.9688 | 0.9538 | 0.9253 |

## Different-image similarity

Cosine similarity over all distinct-image pairs of angle-0 embeddings (26,335
pairs). This is the collapse baseline: a rotation-invariant constant embedding
would push the different-image distribution toward one.

| Statistic | Baseline | Candidate |
| --- | --- | --- |
| Median | 0.6550 | 0.6432 |
| p90 | 0.8173 | 0.8174 |
| Pair count | 26,335 | 26,335 |

## Technical implementation result: ISSUE

The plan's acceptance condition — candidate 90-degree median same-image
similarity not below the baseline 90-degree median — is **not met**. The
candidate 90° median is 0.6481 versus the baseline 0.6827. The same direction
holds at 180° (0.7901 vs 0.8465) and 270° (0.6616 vs 0.6890), and the candidate
p10/min are also lower at every angle. The candidate is therefore **not**
demonstrated to be more rotation-consistent than the baseline on this dataset;
this report does not describe the candidate as rotation-invariant or as an
improvement.

At the same time, the downstream embedding metrics are comparable or slightly
better for the candidate: Recall@1 +0.0043, kNN@1 unchanged, kNN@5 +0.0304,
kNN@20 +0.0217, linear probe +0.0261, Silhouette +0.0047, with mAP −0.0080.
The reflection distribution is essentially unchanged (median 0.9691 → 0.9688,
p10 slightly higher, minimum higher), and the different-image distribution is
essentially unchanged (median 0.6550 → 0.6432, p90 0.8173 → 0.8174), so the
lower rotation similarity is not explained by embedding collapse.

Scope caveat: this is a single 50-epoch, two-species, seed-42 comparison. It is
not evidence of broad biological suitability, and it does not by itself justify
changing the default. The observed pattern is consistent with the possibility
that `expand=true` rotation introduces scale variation that trades some
rotation consistency for scale robustness, but that hypothesis was not tested
here.

## Scientific suitability: not assessed by this implementation comparison

This report covers implementation-level training health, available embedding
metrics, and deterministic rotation/reflection consistency only. It makes no
claim about taxonomic accuracy, marker appropriateness, or biological
readiness. A biological-readiness decision requires separate marker-specific
evidence.

## Observed limitations and follow-up

- The candidate 90°/180°/270° same-image medians are all below the baseline, so
  the plan's 90° gate failed. This is surfaced as an implementation result, not
  treated as a code defect; no seeds, augmentation settings, or defaults were
  tuned to make the gate pass.
- Single seed (42) and a single two-species dataset. A seed-replicate
  comparison is needed before attributing the rotation-consistency difference
  to the augmentation profile rather than run-to-run variance.
- Candidate training used `--device auto` on MPS, where some bicubic upsample
  backward operators fall back to CPU; this affects speed only, not the
  recorded values.
- The baseline checkpoint has no `augmentation_config`, so its exact historical
  augmentation contract cannot be reconstructed from the artifact; the
  comparison relies on the shared data, seed, model, optimizer, crop, and
  evaluation settings.
- Follow-up issue: investigate whether expanded-rotation scale variation
  (`expand=true`) trades rotation consistency for scale robustness, and whether
  a smaller or `expand=false` rotation would preserve more rotation
  consistency without losing the other metric gains.
- Follow-up issue: run a seed-replicate comparison (at minimum 3 seeds) to
  separate augmentation effect from seed noise on the seven metrics and the
  rotation summaries.

## Artifacts not committed

Generated checkpoints, full training logs, UMAP/curve PDFs, image data, and the
validation JSON are intentionally excluded from git. `runs/` is gitignored.
