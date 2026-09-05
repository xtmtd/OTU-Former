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
- `--global-crop-size` [default: 224], `--local-crop-size` [default: 96], `--local-crops` [default: 6]
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

### 3.2 `finetune`
ArcFace metric learning supervised fine-tuning on top of a pretrained checkpoint.

All parameters migrated from `ref/ibot20260115.py` `get_parser()`, mode=finetune. Full list:

- `--checkpoint` : pretrained checkpoint path (auto-detected if empty)
- `--train-data` : CSV with `image` and `label` columns
- `--input-images-dir` : root image directory
- `--out-dir`
- `--model-name` : timm backbone (must match pretrain) [default: vit_small_patch16_224]
- `--metric-embed-dim` : embedding dimension for metric learning [default: 256]
- `--finetune-epochs` [default: 20], `--finetune-lr` [default: 1e-4]
- `--freeze-ratio` : fraction of transformer blocks to freeze [default: 0.7]
- `--loss` : loss function [default: arcface] — registry supports future additions (ProxyAnchor planned)
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
- `--extract-size` [default: 224]
- `--use-projector-output` : use projector output instead of CLS token
- `--token-mode` : `cls` | `patch-topk` | `attention-pool` [default: cls]
- `--topk-patches` : for patch-topk mode, choices: 10/20/30 [default: 20]
- `--attention-pooling-type` : `lightweight` | `multihead` | `gated` [default: lightweight]
- `--attention-pooling-epochs` [default: 20]
- `--label-csv` : CSV with required `image` and optional `label`; labels enable quality metrics, while image-only input enables unlabeled UMAP. Attention-pool query training still requires `label`.
- `--metrics-sample-size` [default: 10000; limits metric/UMAP inputs when positive, no cap when <=0]
- `--batch-size`, `--num-workers`, `--device`, `--seed`
- `--prefix` : OTU name prefix applied to cluster IDs downstream [default: OTU]

**Batch mode:** When `--input-images-dir` contains subdirectories, each subdirectory is treated as an independent sample set. All embeddings are extracted and merged into a single CSV with a `sample` column recording the source subdirectory. This ensures consistent OTU naming across datasets when feeding into `cluster`.

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

- `--checkpoint` : pretrain or finetune checkpoint
- `--images-dir` : image directory
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

**Outputs:** Per-image overlay images, `cam_summary.csv`, `model_layers.txt` (if `--dump-model-structure`).

---

### 3.9 `export`
Export encoder + projector to ONNX for deployment.

- `--checkpoint` : pretrain or finetune checkpoint
- `--out-dir`
- `--imgsz` : input image size [default: 224]
- `--opset` : ONNX opset version [default: 17]

**Note:** Always exports encoder + projector only. Output embedding dimension is read from the checkpoint metadata (set by `--out-dim` or `--metric-embed-dim` at training time) — not hardcoded.

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
