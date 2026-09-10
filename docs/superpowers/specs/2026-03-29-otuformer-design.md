# OTU-Former Toolbox — Design Spec (v0.1.0)

Date: 2026-03-29

## 1. Overview

OTU-Former is a Python CLI toolbox for image-based biodiversity analysis. It converts standardised specimen images (e.g. dorsal beetle photos) into morphological barcodes (fixed-dimension embeddings) via a trained encoder, clusters them into morphological OTUs (morphOTUs) using UPGMA hierarchical clustering, and produces diversity data (OTU table + abundance + alpha diversity indices).

The toolbox packages an existing research workflow (ref/ibot20260115.py, ref/embeddings_tree20260206.py, ref/GradCam_heatmap.py, ref/diversity_index.txt) into a single installable Python package with a unified CLI.

**Version:** 0.1.0  
**Python:** ≥ 3.11  
**CLI framework:** Typer (with `--install-completion` built-in)

---

## 2. Architecture

### 2.1 Package structure

```
otuformer/
├── pyproject.toml
├── README.md
├── src/
│   └── otuformer/
│       ├── __init__.py
│       ├── cli/
│       │   ├── main.py          # Typer app, registers all sub-commands
│       │   ├── pretrain.py
│       │   ├── finetune.py
│       │   ├── extract.py
│       │   ├── evaluate.py
│       │   ├── cluster.py
│       │   ├── annotate.py
│       │   ├── diversity.py
│       │   ├── cam.py
│       │   ├── export.py
│       │   └── doctor.py
│       ├── training/
│       │   ├── model.py         # ViT encoder + 3-layer MLP projector
│       │   ├── loss.py          # SSL losses + ArcFace; loss registry for future methods
│       │   ├── trainer.py       # Training loop (pretrain / finetune)
│       │   ├── dataset.py       # Dataset, multi-crop augmentation
│       │   └── scheduler.py     # LR cosine schedule, EMA momentum, teacher temperature warmup
│       ├── embedding/
│       │   ├── extractor.py     # checkpoint → embeddings CSV
│       │   └── evaluator.py     # kNN, linear probe, Recall@K, mAP, NMI/ARI/AMI, UMAP
│       ├── delineation/
│       │   ├── distance.py      # Cosine + Euclidean distance, PCA whitening, local scaling
│       │   ├── tree.py          # UPGMA construction, bootstrap support
│       │   ├── partition.py     # Two-stage dynamic threshold scan, partition export
│       │   ├── annotate.py      # Expert correction write-back
│       │   └── diversity.py     # Alpha diversity indices + MPD
│       ├── vision/
│       │   └── cam.py           # GradCAM, GradCAM++, ScoreCAM, LayerCAM, EigenCAM, AblationCAM
│       └── utils/
│           ├── io.py            # CSV/JSON read-write helpers
│           ├── logging.py       # TeeLogger (stdout → console + file)
│           └── checkpoint.py    # Checkpoint load/save utilities
└── tests/
```

### 2.2 Layer convention

```
CLI (cli/)  →  app service (inline in cli/)  →  core logic (training/ embedding/ delineation/ vision/)
```

The CLI layer is thin (argument parsing + logging scope + result printing). Core logic modules are importable independently of the CLI, enabling future Python API / skill usage.

---

## 3. Sub-commands

### 3.1 `pretrain`
SSL self-supervised pre-training (teacher-student self-distillation + masked token regression).

All parameters migrated from `ref/ibot20260115.py` `get_parser()`, mode=pretrain. Full list:

- `--train-data` : CSV with `image` column (no label required for SSL)
- `--input-images-dir` : root image directory
- `--out-dir` : output directory for checkpoints and logs
- `--model-name` : timm backbone [default: vit_small_patch16_224]
- `--out-dim` : SSL projector output dimension [default: 256]
- `--max-epochs` [default: 50]
- `--lr` [default: 5e-4], `--weight-decay` [default: 0.05], `--warmup-epochs` [default: 3]
- `--global-crop-size` [default: auto; `auto` resolves to the backbone's native `default_cfg["input_size"]` on a new run or the checkpoint's recorded size on `--resume`; explicit values 224/384/448 are common, 518 is patch-14-only]; `--local-crop-size` [default: 96], `--local-crops` [default: 6]. On `--resume`, omitted `--local-crop-size` and `--local-crops` inherit the checkpoint's saved values; explicit conflicting values fail, and changing them requires a new run.
- `--augmentation` : pretraining augmentation profile — `global-barcode` (new-run default), `color-robust`, or `legacy`
- `--orientation-policy` : geometric orientation policy — `sensitive` (new-run default) or `invariant` (opt-in broad rotation and reflection)
- `--mask-ratio` [default: 0.5], `--lambda-local` [default: 1.5], `--lambda-mask` [default: 1.0]
- `--teacher-momentum` [default: 0.995], `--teacher-momentum-end` [default: 0.999]
- `--student-temp` [default: 0.1], `--teacher-temp-start` [default: 0.04], `--teacher-temp-end` [default: 0.07]
- `--disable-cross-view-loss` : disable cross-view pairing (view1×view2)
- `--resume` : checkpoint path to resume from
- `--log-every-n-steps` [default: 50], `--save-every-epochs` [default: 10], `--keep-last-checkpoints` [default: 10]
- `--batch-size` [default: 32], `--num-workers` [default: 4], `--cpus` [default: 12]
- `--device` [default: mps, choices: cpu/cuda/mps], `--seed` [default: 42]
- `--visualize-data` : CSV with `image` and optional `label` columns for periodic evaluation; without labels, only UMAP is generated

**Outputs:** `SSL_latest.pth` (and epoch checkpoints `SSL_epoch_*.pth`), `metrics.pretrain.csv`, `instant_metrics.csv`, training curves PDF, log file. When labels are absent or contain fewer than two classes, supervised metric fields remain empty while the epoch row is retained; an image-only visualization CSV still produces UMAP. `--metrics-sample-size` limits both supervised metric and UMAP inputs when positive.

---

### Training augmentation contract

```text
pretrain: global-barcode (default), color-robust, legacy
finetune: none (default), conservative
orientation-policy: sensitive (default for new runs), invariant
```

- `global-barcode` is the default whole-specimen barcode profile: modest photometric jitter; color is preserved (no grayscale). Its geometry follows the selected orientation policy (`sensitive` by default, `invariant` opt-in).
- `color-robust` uses the same geometry and blur as `global-barcode` with stronger color jitter and grayscale; it may reduce sensitivity to diagnostic body color, color patterns, or metallic sheen.
- `legacy` reproduces OTU-Former 0.2.1 augmentation exactly for old-run continuation and comparison, not new runs; it records either orientation policy without changing its historical transforms.
- `none` is the unchanged deterministic fine-tuning default: `Resize -> CenterCrop -> ToTensor -> Normalize`.
- `conservative` is an experimental opt-in fine-tuning profile, not proven superior to `none`; evaluate it on held-out individuals and held-out species. It adds no crop, grayscale, blur, or solarization.
- `orientation-policy=sensitive` is the new-run default and a caller-selected policy (never inferred) for direction-sensitive markers: rotation `[-15°, 15°]` and no horizontal reflection for every global/local pretraining view and for fine-tuning `conservative`. `invariant` is the opt-in broad-rotation/reflection policy for orientation-insensitive markers and preserves the existing behavior. `legacy` accepts either policy but keeps its historical flips.
- Dorsal, ventral, lateral, whole-body, and anatomical-part images are distinct markers and must not be mixed as interchangeable views of one marker; arbitrary in-plane orientation is supported. The complete-marker requirement applies to the source image, not to every stochastic SSL crop. Augmentation encourages but does not guarantee embedding invariance.

Resume and inheritance: omitted `--augmentation`/`--orientation-policy` inherit on resume; explicit conflicts fail, and changing a profile's expanded parameters requires a new run. Old pretrain checkpoints map to `legacy`/`invariant`; old finetune checkpoints map to `none`/`invariant`. New fine-tuning from `--checkpoint` is initialization, not resume: it never inherits the pretraining augmentation profile, and an omitted policy inherits the pretraining checkpoint policy (fallback `sensitive` for old checkpoints). Both fine-tuning profiles use the checkpoint-recorded training input size with a `224` fallback for old checkpoints. Checkpoint metadata (`config.augmentation_profile`, `config.augmentation_config`) provides configuration traceability, not bitwise deterministic replay.

Transform-level profile definitions live in [`2026-09-07-otuformer-training-augmentation-design.md`](2026-09-07-otuformer-training-augmentation-design.md).

---

### 3.2 `finetune`
ArcFace metric learning supervised fine-tuning on top of a pretrained checkpoint. New fine-tune runs replace the SSL projector with a compact, unnormalized `backbone_dim -> 512 -> metric_embed_dim` embedding head; ArcFace owns feature and classifier normalization. Fine-tuning uses separate backbone and metric-head learning rates, defaults to AdamW `weight_decay=1e-4` as a conservative supervised-training choice rather than a full legacy-script reproduction, and does not apply gradient clipping.

Initialization semantics for `--checkpoint`: an SSL pretrain checkpoint installs a fresh `ArcFaceEmbeddingHead`; a fine-tune checkpoint whose head and width match the head being trained keeps its trained projector; a `ProjectionHead` checkpoint (OTU historical SFT, or `projection_mlp_2048` metadata) keeps its projector and therefore fixes the embedding width.

Checkpoints record the actual `config.embedding_head` (`arcface_mlp_512` for new runs, `projection_mlp_2048` for runs initialized from a historical SFT checkpoint) and `config.freeze_ratio`. The historical `loss_state_dict` marker is recognized when metadata is absent; missing metadata without fine-tune markers remains valid SSL initialization. On resume, the checkpoint's optimizer state restores its saved parameter-group settings, so CLI LR and weight-decay values are inert, and a changed `--freeze-ratio` is rejected because the optimizer parameter groups depend on it.

Ref-script checkpoints (`ref/ibot20260115.py`: `model`/`teacher`/`student` weight keys, `projector.<i>` projector naming, an `args` dict instead of `config`, `loss_func.W` classifier) are readable by `extract`, `export`, and `cam`, but cannot be resumed by `finetune` — see 4.8.

All parameters migrated from `ref/ibot20260115.py` `get_parser()`, mode=finetune. Full list:

- `--checkpoint` : pretrained checkpoint path (auto-detected if empty)
- `--train-data` : CSV with `image` and `label` columns
- `--input-images-dir` : root image directory
- `--out-dir`
- `--model-name` : timm backbone (must match pretrain) [default: vit_small_patch16_224]
- `--metric-embed-dim` : fine-tune embedding dimension (the ArcFace head output, not the raw CLS dimension) [default: inherit the checkpoint's recorded metric dimension, else the pretrained projector dimension]. An explicit value that conflicts with a `ProjectionHead` checkpoint is rejected, because that projector's output width is fixed by the pretrained weights. On `--resume` the value must match the checkpoint (resizing is only valid for a new `--checkpoint` run).
- `--finetune-epochs` [default: 20], `--finetune-lr` : backbone learning rate [default: 1e-4]
- `--metric-head-lr` : learning rate for the ArcFace embedding head and classifier; omitted means inherit `--finetune-lr`. On `--resume`, saved optimizer state takes precedence.
- `--weight-decay` : AdamW weight decay [default: 1e-4, a conservative supervised SFT choice; pass 0.05 for the legacy script's setting]
- `--freeze-ratio` : fraction of transformer blocks to freeze [default: 0.7]; recorded in `config.freeze_ratio` and required to match on `--resume`, because the optimizer only holds `requires_grad` backbone parameters
- `--loss` : loss function [default: arcface] — registry supports future additions (ProxyAnchor planned)
- `--augmentation` : fine-tuning augmentation profile — `none` (new-run default) or `conservative` (experimental)
- `--orientation-policy` : orientation policy — `sensitive` (new-run default) or `invariant` (opt-in broad rotation and reflection); when initializing from `--checkpoint`, an omitted value inherits the pretraining checkpoint's saved policy
- `--batch-size`, `--num-workers`, `--cpus`, `--device`, `--seed`
- `--log-every-n-steps`, `--save-every-epochs`, `--keep-last-checkpoints`
- `--visualize-data` : CSV with `image` and optional `label` columns for periodic evaluation; without labels, only UMAP is generated

**Outputs:** `finetune_latest.pth` (and epoch checkpoints `finetune_epoch_*.pth`), `metrics.finetune.csv`, `instant_metrics.csv`, training curves PDF, log file. When labels are absent or contain fewer than two classes, supervised metric fields remain empty while the epoch row is retained; an image-only visualization CSV still produces UMAP. `--metrics-sample-size` limits both supervised metric and UMAP inputs when positive.

---

### 3.3 `extract`
Extract embeddings from images using a trained checkpoint.

Parameters migrated from `ref/ibot20260115.py` `get_parser()`, mode=extract. Full list:

- `--checkpoint` : path to pretrain or finetune checkpoint
- `--input-images-dir` : root image directory **or** parent directory containing subdirectories (batch mode, auto-detected)
- `--out-dir`
- `--model-name` : timm backbone (must match training) [default: vit_small_patch16_224]
- `--extract-size` [default: auto; `auto` uses the checkpoint's recorded training size; explicit values (224/384/448, 518 = patch-14 models only) must be divisible by the backbone patch size]
- `--eval-transform` : `center-crop` | `whole-specimen-pad` [default: center-crop; deterministic evaluation preprocessing protocol, applied to extraction and CAM]
- `--use-projector-output` : use SSL projector output for SSL checkpoints, or the ArcFace embedding for fine-tune checkpoints; recommended for fine-tuned-class retrieval and closed-set clustering
- `--token-mode` : `cls` | `patch-topk` | `attention-pool` [default: cls]; `cls` is raw CLS and remains the comparison baseline for unseen classes, cross-dataset transfer, and general morphology representation
- `--topk-patches` : for patch-topk mode, choices: 10/20/30 [default: 20]
- `--attention-pooling-type` : `lightweight` | `multihead` | `gated` [default: lightweight]
- `--attention-pooling-epochs` [default: 20]
- `--label-csv` : CSV with required `image` and optional `label`; labels enable quality metrics, while image-only input enables unlabeled UMAP. Attention-pool query training still requires `label`.
- `--metrics-sample-size` [default: 10000; limits metric/UMAP inputs when positive, no cap when <=0]
- `--batch-size`, `--num-workers`, `--device`, `--seed`
- `--prefix` : OTU name prefix applied to cluster IDs downstream [default: OTU]

**Batch mode:** When `--input-images-dir` contains subdirectories, each subdirectory is treated as an independent sample set. All embeddings are extracted and merged into a single CSV with a `sample` column recording the source subdirectory. This ensures consistent OTU naming across datasets when feeding into `cluster`.

**Checkpoint formats:** Accepts OTU checkpoints (`model_state_dict`/`teacher`/`student` + `config`) and ref-script checkpoints (`teacher`/`student`/`model` + `args`). The embedding head and embedding dimension are read from `config.embedding_head` when present and otherwise inferred from the projector shapes, so a missing or wrong metadata value cannot cause a silent mis-load. Ref-script **SFT** checkpoints record no `config`/`args`, so they need an explicit `--model-name`.

**Outputs:** `embeddings.csv` (columns: `id`, `sample` (batch only), then embedding dimensions), optional `umap.pdf`, and `metrics.csv` only when at least two label classes are available. Positive `--metrics-sample-size` limits both metric and UMAP inputs; values <=0 disable the cap.

---

### 3.4 `evaluate`
Evaluate embedding quality from multiple angles.

Parameters partially from `ref/ibot20260115.py` visualisation/metrics block:

- `--embeddings` : embeddings CSV
- `--labels` : CSV with columns `[id, label]`
- `--out-dir`
- `--umap-dims` : 2 or 3 [default: 2]
- `--umap-n-neighbors` [default: 15], `--umap-min-dist` [default: 0.1], `--umap-metric` [default: cosine]
- `--visualize-class-number` : max classes shown in UMAP [default: 20, 0 = all]
- `--knn-k` : comma-separated k values for kNN [default: 1,5,10]
- `--metrics-sample-size` [default: 10000]

**Metrics computed:**
- Classification transferability: kNN accuracy (k=1,5,10), Linear Probing accuracy
- Retrieval: Recall@K (K=1,5,10), mAP
- Clustering quality: NMI, ARI, AMI, Silhouette Score, Purity
- Metric learning diagnostics: Intra-Class Var, Inter-Class Dist, embedding norms
- Visualisation: UMAP 2D/3D plot (PDF)

**Outputs:** `metrics.json`, `metrics.csv`, `umap.pdf`, log file.

---

### 3.5 `cluster`
Compute pairwise distances, build UPGMA tree, scan partitions.

Parameters migrated from `ref/embeddings_tree20260206.py`:

- `--embeddings` : embeddings CSV
- `--out-dir`
- `--distance` : `cosine` | `euclidean` [default: cosine]
- `--prefix` : OTU name prefix [default: OTU]
- `--pca-whitening` / `--pca-components` [default: 256]
- `--local-scaling` : enable k-NN based local scaling of distance matrix
- `--local-k` [default: 0 = auto], `--local-k-strategy` : `adaptive` | `sqrt` | `log` | `fixed` [default: adaptive]
- `--cutoff-min` [default: 0.05], `--cutoff-max` [default: 1.0], `--cutoff-step` [default: 0.05]
- `--custom-cutoffs` : comma-separated values (overrides range)
- `--num-bootstraps` [default: 0], `--bootstrap-subsample-ratio` [default: 0.8], `--bootstrap-display-cutoff` [default: 50.0]
- `--save-distances` : save pairwise distance matrix CSV
- `--max-distance-pairs` [default: 1_000_000]
- `--labels` : optional, for partition quality metrics (NMI, ARI, BCubed-F etc.)
- `--metrics-sample-size` [default: 10000]
- `--cpus` [default: 8], `--random-state` [default: 42]

**Outputs (under `--out-dir`):**
- `upgma.nwk` : Newick tree file
- `distance_stats.json`
- `distance_hist_*.pdf`, `distance_cum_*.pdf` : distribution plots
- `partition_scan.csv` : cluster count vs cutoff
- `partitions/tables/partition_{cutoff}_assignments.csv` : per-sample cluster assignments
- `partitions/tables/partition_{cutoff}_summary.csv` : cluster size summary
- `partitions/upgma_tree_{cutoff}.pdf` : tree + partition bar visualisation
- `metrics.csv` (if `--labels` provided): NMI, ARI, AMI, BCubed-F, V-measure, Silhouette per cutoff
- `intraclass_distances.csv` (if `--labels` provided)

---

### 3.6 `annotate`
Write expert taxonomic corrections back into partition assignments.

- `--assignments` : `partition_{cutoff}_assignments.csv` (columns: `id, cluster`)
- `--corrections` : CSV with columns `id, corrected_cluster`
- `--out-dir`

**Logic:**
- IDs present in `corrections` override the cluster assignment in `assignments`
- IDs absent from `corrections` are kept unchanged
- Output filename: `{original_stem}_annotated.csv`

**Outputs:**
- `partition_{cutoff}_assignments_annotated.csv`
- `annotation_summary.csv` : number of corrections, clusters affected, before/after distribution

---

### 3.7 `diversity`
Compute alpha diversity indices from a (potentially annotated) partition assignments file.

- `--assignments` : assignments CSV (annotated or raw from `cluster`)
- `--out-dir`
- `--prefix` : OTU name prefix (must match `cluster` step) [default: OTU]
- `--min-abundance` : comma-separated filter thresholds [default: 0,2,5]
- `--phylo` : compute MPD (Morphological Dendrogram Diversity); requires `--tree`
- `--tree` : UPGMA Newick file (required if `--phylo`)

**Diversity indices computed (per `--min-abundance` threshold):**

| Category     | Indices                                                                  |
|--------------|--------------------------------------------------------------------------|
| Richness     | Richness, Chao1, ACE, Margalef, Menhinick                                |
| Evenness     | Pielou's J, Heip's E                                                     |
| Diversity    | Shannon H', Simpson 1-D, Inverse Simpson 1/D, Fisher's alpha, Brillouin  |
| Dominance    | Berger-Parker                                                            |
| Hill numbers | q=0, q=1, q=2                                                            |
| Morpho tree  | MPD (optional, `--phylo`)                                                |

**Output format:** Wide table — rows = indices, columns = `min_abundance_0`, `min_abundance_2`, `min_abundance_5` etc.

**Outputs:** `diversity.csv`, `diversity_summary.pdf` (bar charts per index), log file.

**Note:** All diversity calculations in pure Python (scipy/scikit-bio/numpy). No usearch dependency.

---

### 3.8 `cam`
Generate CAM heatmaps for visual explanation of morphological features.

Implementation mirrors `entomokit classify cam` (`/Users/zf/data/coding/entomokit/entomokit/classify/cam.py` and `src/classification/cam.py`), adapted for OTU-Former checkpoints (timm ViT backbone, no AutoGluon).

- `--checkpoint` : pretrain or finetune checkpoint (OTU or ref-script format)
- `--images-dir` : image directory
- `--model-name` : fallback timm backbone for checkpoints that record no model name (ref-script SFT) [default: `vit_tiny_patch16_224`]
- `--label-csv` : optional CSV with `image` and `label` columns; if omitted all images in `--images-dir` are used
- `--out-dir`
- `--cam-method` : `gradcam` | `gradcampp` | `scorecam` | `layercam` | `eigencam` | `ablationcam` [default: gradcam]
- `--arch` : `cnn` | `vit` (auto-detected if not set)
- `--target-layer-name` : specific model layer (auto-selected when omitted)
- `--image-weight` [default: 0.5] : blend weight of original image in overlay
- `--fig-format` : `png` | `jpg` | `pdf` [default: png]
- `--save-npy` : save raw CAM arrays as .npy
- `--dump-model-structure` : write layer names to `model_layers.txt`
- `--max-images` : limit number of images processed
- `--cam-batch-size` [default: 32], `--num-workers` [default: 4], `--device`
- `--eval-transform` : `center-crop` | `whole-specimen-pad` [default: center-crop; deterministic evaluation preprocessing protocol shared with extraction]

Heatmaps are overlaid on the full original image via the correct inverse of the selected preprocessing: with `center-crop` the heatmap is confined to the model's field of view using a binary FOV mask (out-of-view pixels are dimmed directly, never routed through the heatmap colormap); with `whole-specimen-pad` the heatmap covers the whole specimen. The figure is a gradient-attribution heatmap of the raw CLS feature (its argmax treated as a pseudo-class), not a verified class-discriminative CAM; that semantic is tracked separately.

**Outputs:** Per-image overlay images, `cam_summary.csv`, `model_layers.txt` (if `--dump-model-structure`).

---

### 3.9 `export`
Export encoder + projector to ONNX for deployment.

- `--checkpoint` : pretrain or finetune checkpoint (OTU or ref-script format)
- `--out-dir`
- `--imgsz` : input image size [default: auto; `auto` uses the checkpoint's recorded training size]
- `--opset` : ONNX opset version [default: 17]
- `--model-name` : fallback timm backbone for checkpoints that record no model name (ref-script SFT) [default: `vit_tiny_patch16_224`]

**Note:** Always exports encoder + projector only. The projector is rebuilt from the checkpoint's `config.embedding_head`, falling back to the projector shapes, so fine-tune checkpoints with an `ArcFaceEmbeddingHead` export correctly; the ref-script `projector.<i>` naming is normalized on read. Output embedding dimension is read from the checkpoint metadata (set by `--out-dim` or `--metric-embed-dim` at training time) — not hardcoded. A backbone that does not fit the resolved model name raises an error naming whether the conflict is in the checkpoint metadata or in `--model-name`.

**Outputs:** `encoder.onnx`, `export_report.json`.

---

### 3.10 `doctor`
Check environment health.

**Checks:** Python version, PyTorch + CUDA/MPS availability, timm, scikit-bio, umap-learn, grad-cam, onnx, key package versions.

---

## 4. Key Design Decisions

### 4.1 Distance metrics
Both cosine and Euclidean distances support: PCA whitening, k-NN based local scaling (as implemented in `ref/embeddings_tree20260206.py` `apply_local_scaling()`), chunked computation for large datasets. Euclidean distance is sensitive to scale; documentation recommends combining with PCA whitening or L2 normalisation.

### 4.2 Loss registry
`training/loss.py` implements a simple registry pattern:
```python
LOSS_REGISTRY = {"arcface": ArcFaceLoss, ...}
```
ProxyAnchor is planned as the next addition. Adding it requires only registering a new class, no CLI changes needed.

### 4.3 Batch extract
Auto-detection logic in `extractor.py`: if `--input-images-dir` contains at least one subdirectory with images, batch mode is activated. All batches share the same checkpoint and `--prefix`, ensuring consistent OTU naming.

### 4.4 OTU naming
`--prefix` flows from `extract` → `cluster` → `diversity`. Default prefix `OTU` produces `OTU1, OTU2, ...`. Custom prefix `Dun2024` produces `Dun2024_1, Dun2024_2, ...`.

### 4.5 Morphological Dendrogram Diversity (MPD)
MPD is computed using `scikit-bio`'s Faith PD infrastructure with the UPGMA tree as input. Output is clearly labelled `MPD (morphological dendrogram diversity, not true phylogenetic diversity)` to avoid misinterpretation.

### 4.6 UPGMA only
NJ tree construction is removed (present in original `embeddings_tree20260206.py`). Only UPGMA is retained.

### 4.7 Export embedding dimension
The ONNX export output dimension is read directly from checkpoint metadata. It is not hardcoded to 256 — it reflects whatever `--out-dim` (pretrain) or `--metric-embed-dim` (finetune) was used.

### 4.8 Checkpoint consumption contract
The three read-only consumers (`extract`, `export`, `cam`) share one resolver, `otuformer.utils.checkpoint.resolve_checkpoint()`, so their format support cannot drift apart. `finetune` deliberately uses its own `_select_finetune_embedding_head()`: it answers a different question — which head the *new* run should train (SSL initialization installs a fresh ArcFace head; an existing fine-tune checkpoint keeps its own) rather than which head a checkpoint contains.

| Consumer | `model_state_dict` | `teacher`/`student` | `model` (ref-script SFT) | `args` (no `config`) | `projector.<i>` naming |
|---|---|---|---|---|---|
| `extract` | yes | yes | yes | yes | yes |
| `export` | yes | yes | yes | yes | yes |
| `cam` | yes | yes | yes | yes | yes |
| `finetune` | init; `--resume` requires it | rejected | rejected, with reason | rejected | rejected |

The embedding head is taken from `config.embedding_head` when present, otherwise inferred from the projector's first linear width (512 → `arcface_mlp_512`, 2048 → `projection_mlp_2048`); any other width is rejected rather than silently mis-loaded. Buffers that do not fit the rebuilt model (the SSL-only `center`) are dropped, and a backbone whose tensors do not fit the resolved model name is rejected with a message that distinguishes a conflict inside the checkpoint metadata (which outranks `--model-name`) from a missing recorded name that `--model-name` must supply.

Resuming a ref-script checkpoint in `finetune` stays unsupported: its classifier is `loss_func.W`, which is embedding-major and not interchangeable with `ArcFaceLoss.head.weight`. Ref-script **SSL** checkpoints are self-describing (they record `args` with `model_name`/`out_dim`); ref-script **SFT** checkpoints record neither `config` nor `args`, so they require an explicit `--model-name` (`extract`, `export`, and `cam` all expose it).

---

## 5. Dependencies

```toml
[project.dependencies]
typer >= 0.12
torch >= 2.1
timm >= 1.0
torchvision >= 0.16
numpy >= 1.26
pandas >= 2.0
scipy >= 1.12
scikit-learn >= 1.4
scikit-bio >= 0.6
umap-learn >= 0.5
matplotlib >= 3.8
seaborn >= 0.13
grad-cam >= 1.5          # pip install grad-cam; import as pytorch_grad_cam
onnx >= 1.16
onnxruntime >= 1.18      # optional: used for export validation only
tqdm >= 4.66
```

---

## 6. Future Roadmap

- ProxyAnchor loss (metric learning alternative to ArcFace)
- `report` sub-command (HTML/PDF summary of a full analysis run)
- Beta diversity matrix + cross-sample OTU comparison
- Python API / skill for notebook-based non-CLI analysis (after interfaces stabilise)
