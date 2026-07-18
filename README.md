# OTU-Former

[中文文档](README.cn.md) | **English**

An image-based morphological OTU (Operational Taxonomic Unit) delineation toolkit. Provides a unified `otuformer` CLI with commands for self-supervised pretraining, ArcFace metric learning finetuning, embedding extraction (including quality metrics/UMAP when labels are provided), UPGMA hierarchical clustering, expert-corrected annotation, community diversity analysis, CAM visualization, and ONNX model export.

## Overview

All functionality is accessed through a single entry point:

```
otuformer <command> [options]
```

| Command | Description |
|---------|-------------|
| `doctor` | Diagnose environment and dependency status |
| `pretrain` | Self-supervised contrastive pretraining (DINO/iBOT style) |
| `finetune` | ArcFace metric learning finetuning |
| `extract` | Extract image embeddings (ONNX-accelerated) |
| `cluster` | UPGMA hierarchical clustering into morphological OTUs |
| `annotate` | Apply expert corrections to produce refined OTU annotations |
| `diversity` | Compute community alpha diversity indices (Shannon, Simpson, Chao1, Faith's PD, etc.) |
| `cam` | Generate GradCAM and other heatmaps |
| `export` | Export model to ONNX format |
| `update` | Check for and install the latest published version |

## Features

- **Unified CLI**: Single `otuformer` entry point — covers the complete pipeline from pretraining to diversity analysis
- **Self-Supervised Pretraining**: DINO/iBOT-style teacher-student ViT contrastive learning with global/local crops and masked-token consistency
- **ArcFace Metric Learning Finetuning**: Optimize model with labelled data to produce discriminative embeddings for OTU clustering
- **Multi-Mode Embedding Extraction**: CLS token, patch-topk, and attention-pool modes; optional ONNX acceleration
- **Embedding Quality Evaluation**: NMI, ARI, Recall@K, kNN accuracy, mAP@R, Silhouette Score, Linear Probing Accuracy, and more
- **UPGMA Hierarchical Clustering**: Build distance matrix and phylogenetic tree, auto-partition OTUs at multiple distance cutoffs
- **PCA Whitening & Local Scaling**: Optional PCA whitening and Mutual-Proximity-style local scaling for improved clustering
- **Bootstrap Support Estimation**: Subsample/bootstrap modes for branch support estimation
- **Expert-Corrected Annotation**: Accept manual correction CSVs to produce refined OTU assignments with intra-class distance summaries
- **Community Diversity Analysis**: Richness, Chao1, ACE, Shannon, Simpson, Hill numbers (q0/q1/q2), Pielou evenness, Faith's PD (MPD), weighted PD (MPD_w), and other alpha diversity metrics; phylogenetic diversity is computed from an OTU-centroid neighbor-joining tree when embeddings are provided
- **Model Interpretability**: Six CAM algorithms — Grad-CAM, Grad-CAM++, LayerCAM, Score-CAM, Eigen-CAM, Ablation-CAM
- **ONNX Export & Acceleration**: Export to ONNX format for 2-5x faster CPU inference
- **UMAP Visualization**: Embedding space dimensionality reduction for intuitive morphological distribution
- **Parallel Processing**: Multi-threaded image processing with configurable CPU threads
- **Comprehensive Logging**: All commands automatically save run logs to the `logs/` subdirectory of the output directory
- **Graceful Shutdown**: Safe Ctrl+C interruption with partial result preservation

## Requirements

- Python 3.11+
- Operating Systems: macOS, Linux, Windows
- Devices: CPU (required); CUDA GPU / Apple Silicon MPS (optional)

## Installation

Recommended: use an isolated Python environment to avoid dependency conflicts with your system/site-packages.

Before installation, clone the repository and enter the project directory:

```bash
git clone https://github.com/xtmtd/OTU-Former.git
cd OTU-Former
```

### Isolated Environment

Choose one of the following:

**Option 1: conda**

```bash
conda create -n otuformer python=3.11 -y
conda activate otuformer
pip install -e .
```

**Option 2: uv + venv**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

**Option 3: stdlib venv + pip**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Recommended workflow command order:

1. `doctor` — Check environment
2. `pretrain` — Self-supervised pretraining
3. `finetune` — ArcFace finetuning
4. `extract` — Extract embeddings
5. `cluster` — UPGMA clustering into OTUs
6. `annotate` — Apply expert corrections
7. `diversity` — Diversity analysis

### Doctor Command

Diagnose environment and dependency readiness.

```bash
otuformer doctor
```

The report includes:
- Python version and available devices (`cpu`, `cuda`, `mps`)
- Key package versions and status (ok/missing/outdated)

**Output**:
- Diagnostic report printed to terminal

### Update Command

```bash
otuformer update
otuformer update --check
otuformer update --yes
```

Checks published Git tags and installs the latest release only after confirmation.

### Output Safety

Commands refuse a non-empty `--out-dir` by default. Pass `--overwrite` to clear
it deliberately. Training `--resume` runs keep and append to their existing
output directory; `--resume` and `--overwrite` cannot be combined.

### Hugging Face Weights

New pretrain and finetune runs initialize from public timm weights. An
unauthenticated Hugging Face warning is expected when those weights are first
downloaded or absent from the local cache; `HF_TOKEN` is optional and only raises
rate limits. CAM, extract, and export load their supplied local checkpoint and
do not require Hugging Face access.

---

### Pretrain Command

Self-supervised contrastive pretraining using DINO/iBOT-style teacher-student ViT, no labels required.

To continue pretraining after adding images, create a CSV containing both old and
new images, keep the same `--out-dir`, pass the latest checkpoint to `--resume`,
and increase `--max-epochs`:

```bash
otuformer pretrain --train-data images_union.csv --input-images-dir ./images \
    --out-dir runs/pretrain --resume runs/pretrain/SSL_latest.pth --max-epochs 100
```

An extended run may use a larger union dataset. A same-plan resume requires the
original DataLoader length. Training on only newly added images is possible but
can forget previously learned data.

```bash
# Basic usage
otuformer pretrain \
    --input-images-dir ./images \
    --out-dir runs/pretrain

# Use CSV to specify training subset
otuformer pretrain \
    --train-data images.csv \
    --input-images-dir ./images \
    --out-dir runs/pretrain

# Custom model and training parameters
otuformer pretrain \
    --input-images-dir ./images \
    --model-name vit_small_patch16_224 \
    --max-epochs 100 \
    --lr 1e-3 \
    --batch-size 64 \
    --out-dir runs/pretrain
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train-data` | Training data CSV (with `image` column, paths relative to `--input-images-dir`); if omitted, all images under `--input-images-dir` are used recursively | None |
| `--input-images-dir` | Root image directory | Required |
| `--out-dir` | Output directory | `runs/pretrain` |
| `--model-name` | timm backbone name | `vit_tiny_patch16_224` |
| `--out-dim` | SSL projector output dimension | 256 |
| `--max-epochs` | Pretraining epochs | 50 |
| `--lr` | Base learning rate | 5e-4 |
| `--weight-decay` | AdamW weight decay | 0.05 |
| `--warmup-epochs` | Warmup epochs | 3 |
| `--global-crop-size` | Global crop resolution | 224 |
| `--local-crop-size` | Local crop resolution | 96 |
| `--local-crops` | Number of local crops | 6 |
| `--mask-ratio` | Masked token ratio | 0.5 |
| `--lambda-local` | Local crop loss weight | 1.5 |
| `--lambda-mask` | Masked token loss weight | 1.0 |
| `--teacher-momentum` | Initial EMA momentum | 0.995 |
| `--teacher-momentum-end` | Final EMA momentum | 0.999 |
| `--student-temp` | Student temperature | 0.1 |
| `--teacher-temp-start` | Initial teacher temperature | 0.04 |
| `--teacher-temp-end` | Final teacher temperature | 0.07 |
| `--disable-cross-view-loss` | Disable cross-view global loss | No |
| `--resume` | Checkpoint path for resuming interrupted training | None |
| `--log-every-n-steps` | Log metrics every N steps | 50 |
| `--save-every-epochs` | Save checkpoint every N epochs | 10 |
| `--keep-last-checkpoints` | Keep only last N checkpoints | 10 |
| `--visualize-data` | CSV for periodic embedding metrics/UMAP; if omitted, `--train-data` is reused | None |
| `--extract-size` | Image size for embedding extraction (<=0 means auto) | 0 |
| `--metrics-sample-size` | Max samples for periodic metrics | 10000 |
| `--umap-n-neighbors` | UMAP n_neighbors | 15 |
| `--umap-min-dist` | UMAP min_dist | 0.1 |
| `--umap-metric` | UMAP distance metric | `cosine` |
| `--visualize-class-number` | Max classes in UMAP plot | 20 |
| `--disable-embedding-metrics` | Disable periodic embedding metrics + UMAP | No |
| `--batch-size` | Batch size | 32 |
| `--num-workers` | DataLoader workers | 4 |
| `--cpus` | PyTorch/MKL CPU threads | 12 |
| `--device` | `auto`/`cpu`/`cuda`/`mps` | `auto` |
| `--seed` | Random seed | 42 |

**Output**:
- `logs/pretrain.log` — Run log
- `checkpoints/` — Model checkpoints (`SSL_latest.pth`, `SSL_best.pth`)
- `metrics.json` / `metrics.csv` — Training metrics
- `umap.pdf` — UMAP visualization (when not disabled)

---

### Finetune Command

ArcFace metric learning finetuning with labelled data.

To add images to known classes, train with a CSV containing the union of old and
new labeled images, keep the same `--out-dir`, resume from the latest fine-tune
checkpoint, and increase `--finetune-epochs`:

```bash
otuformer finetune --train-data labels_union.csv --input-images-dir ./images \
    --out-dir runs/finetune --resume runs/finetune/finetune_latest.pth \
    --finetune-epochs 50
```

The resumed CSV must retain exactly the same label set. New classes require a
new fine-tune initialized with `--checkpoint` rather than `--resume`.

```bash
# Basic usage
otuformer finetune \
    --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --out-dir runs/finetune

# Resume training
otuformer finetune \
    --resume runs/finetune/finetune_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --out-dir runs/finetune

# Custom parameters
otuformer finetune \
    --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv \
    --input-images-dir ./images \
    --model-name vit_small_patch16_224 \
    --finetune-epochs 50 \
    --finetune-lr 3e-4 \
    --freeze-ratio 0.5 \
    --loss arcface \
    --out-dir runs/finetune
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | Pretrained checkpoint path (typically `runs/pretrain/SSL_latest.pth`) | Required |
| `--resume` | Finetune checkpoint path to resume from | None |
| `--train-data` | Training data CSV (with `image` and `label` columns) | Required |
| `--input-images-dir` | Root image directory | Required |
| `--out-dir` | Output directory | `runs/finetune` |
| `--model-name` | timm backbone name | `vit_tiny_patch16_224` |
| `--metric-embed-dim` | ArcFace projector embedding dimension | 256 |
| `--finetune-epochs` | Finetuning epochs | 20 |
| `--finetune-lr` | Finetuning learning rate | 1e-4 |
| `--freeze-ratio` | Fraction of backbone blocks to freeze (0.0=none, 1.0=all) | 0.7 |
| `--loss` | Metric-learning loss name | `arcface` |
| `--batch-size` | Batch size | 32 |
| `--num-workers` | DataLoader workers | 4 |
| `--cpus` | PyTorch/MKL CPU threads | 12 |
| `--device` | `auto`/`cpu`/`cuda`/`mps` | `auto` |
| `--seed` | Random seed | 42 |
| `--log-every-n-steps` | Log metrics every N steps | 50 |
| `--save-every-epochs` | Save checkpoint every N epochs | 10 |
| `--keep-last-checkpoints` | Keep only last N checkpoints | 10 |
| `--visualize-data` | CSV for periodic embedding metrics/UMAP; if omitted, `--train-data` is reused | None |
| `--extract-size` | Image size for embedding extraction (<=0 means auto) | 0 |
| `--metrics-sample-size` | Max samples for periodic metrics | 10000 |
| `--umap-n-neighbors` | UMAP n_neighbors | 15 |
| `--umap-min-dist` | UMAP min_dist | 0.1 |
| `--umap-metric` | UMAP distance metric | `cosine` |
| `--visualize-class-number` | Max classes in UMAP plot | 20 |
| `--disable-embedding-metrics` | Disable periodic embedding metrics + UMAP | No |

**Output**:
- `logs/finetune.log` — Run log
- `checkpoints/` — Model checkpoints (`finetune_latest.pth`, `finetune_best.pth`)
- `metrics.json` / `metrics.csv` — Training metrics
- `umap.pdf` — UMAP visualization (when not disabled)

---

### Extract Command

Extract embeddings from images. Supports ONNX model for accelerated CPU inference.

```bash
# Using PyTorch checkpoint (CLS token mode)
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --out-dir runs/extract

# Using ONNX model (2-5x faster on CPU)
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --onnx-path runs/export/encoder.onnx \
    --out-dir runs/extract

# With label CSV in attention-pool mode
otuformer extract \
    --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images \
    --label-csv labels.csv \
    --token-mode attention-pool \
    --out-dir runs/extract
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | Pretrain or finetune checkpoint path; required unless `--onnx-path` is provided | None |
| `--input-images-dir` | Input image directory | Required |
| `--out-dir` | Output directory | `runs/extract` |
| `--model-name` | timm backbone name | `vit_tiny_patch16_224` |
| `--extract-size` | Resize/crop size for extraction | 224 |
| `--use-projector-output` | Use projector output instead of CLS token | No |
| `--use-student` | Load student weights instead of teacher (EMA) | No |
| `--token-mode` | `cls`/`patch-topk`/`attention-pool` | `cls` |
| `--topk-patches` | Top-K patch tokens for patch-topk mode | 20 |
| `--attention-pooling-type` | `lightweight`/`multihead`/`gated` | `lightweight` |
| `--attention-pooling-epochs` | Epochs to finetune attention query | 20 |
| `--label-csv` | Label CSV for embedding quality evaluation and UMAP generation | None |
| `--metrics-sample-size` | Max samples for metrics evaluation | 10000 |
| `--umap-n-neighbors` | UMAP n_neighbors | 15 |
| `--umap-min-dist` | UMAP min_dist | 0.1 |
| `--umap-metric` | UMAP distance metric | `cosine` |
| `--visualize-class-number` | Max classes in UMAP plot | 20 |
| `--disable-umap` | Skip UMAP generation | No |
| `--batch-size` | Batch size | 32 |
| `--num-workers` | DataLoader workers | 4 |
| `--device` | `auto`/`cpu`/`cuda`/`mps` | `auto` |
| `--onnx-path` | ONNX model path (uses ONNX Runtime when provided) | None |
| `--seed` | Random seed | 42 |

**Output**:
- `embeddings.csv` — Embedding vectors (`id`, `dim_0`, `dim_1`, ...)
- `metrics.csv` — Quality metrics (when `--label-csv` is provided)
- `umap.pdf` — UMAP visualization (when `--label-csv` is provided and not disabled)

---

### Cluster Command

Cluster embeddings into morphological OTUs via UPGMA hierarchical clustering.

```bash
# Basic usage
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --out-dir runs/cluster

# With PCA whitening, local scaling, and bootstrap support
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --pca-whitening true \
    --local-scaling true \
    --num-replicates 100 \
    --out-dir runs/cluster

# Custom distance metric and cutoff range
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --distance euclidean \
    --cutoff-min 0.1 \
    --cutoff-max 0.8 \
    --cutoff-step 0.02 \
    --out-dir runs/cluster

# With label CSV for partition quality evaluation
otuformer cluster \
    --embeddings runs/extract/embeddings.csv \
    --label-csv labels.csv \
    --out-dir runs/cluster
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--embeddings` | Embeddings CSV | Required |
| `--out-dir` | Output directory | `runs/cluster` |
| `--distance` | Distance metric: `cosine`/`euclidean` | `cosine` |
| `--prefix` | Cluster prefix for partition table labels | `OTU` |
| `--pca-whitening` | Enable PCA whitening (`true`/`false`) | `false` |
| `--pca-components` | PCA components | 256 |
| `--local-scaling` | Enable Mutual-Proximity local scaling (`true`/`false`) | `false` |
| `--local-k` | Fixed k for local scaling (0=auto) | 0 |
| `--local-k-strategy` | Auto-k strategy: `adaptive`/`sqrt`/`log`/`fixed` | `adaptive` |
| `--cutoff-min` | Minimum distance cutoff | 0.05 |
| `--cutoff-max` | Maximum distance cutoff | 1.0 |
| `--cutoff-step` | Cutoff step size | 0.05 |
| `--custom-cutoffs` | Custom cutoffs (comma-separated, overrides min/max/step) | None |
| `--support-mode` | Support estimation mode: `subsample`/`bootstrap` | `subsample` |
| `--num-replicates` | Support estimation replicates (0=disabled) | 0 |
| `--subsample-ratio` | Feature fraction for subsample mode | 0.8 |
| `--support-display-cutoff` | Minimum support threshold for tree labels | 50.0 |
| `--save-bootstrap-trees` | Save all bootstrap trees (`true`/`false`) | `false` |
| `--save-distances` | Save full distance matrix (`true`/`false`) | `false` |
| `--max-distance-pairs` | Max distance matrix pairs | 1000000 |
| `--label-csv` / `--labels` | Optional label CSV for partition quality evaluation | None |
| `--metrics-sample-size` | Max samples for metrics | 10000 |
| `--cpus` | CPU threads | 8 |
| `--random-state` | Random seed | 42 |

**Output**:
- `UPGMA/UPGMA_Cosine.nwk` — Newick format phylogenetic tree
- `UPGMA/partitions/partition_scan.csv` — Threshold scan results
- `UPGMA/partitions/tables/partition_<cutoff>_assignments.csv` — OTU assignments at each cutoff
- `UPGMA/partitions/UPGMA_tree_partitions.pdf` — Tree and partition visualization
- `UPGMA/metrics.csv` — Partition quality metrics (when `--label-csv` is provided)
- `UPGMA/metrics_dashboard.pdf` — Metrics dashboard (when `--label-csv` is provided)
- `distance_statistics/` — Distance statistics and distribution plots

---

### Annotate Command

Apply expert corrections to cluster assignments, producing refined OTU annotations.

```bash
# Basic usage
otuformer annotate \
    --raw-assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --corrections corrections.csv \
    --out-dir runs/annotate

# With embeddings (recompute intra-class distances and annotated UPGMA tree)
otuformer annotate \
    --raw-assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --corrections corrections.csv \
    --embeddings runs/extract/embeddings.csv \
    --show-annotation-bar \
    --out-dir runs/annotate
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--raw-assignments` | Raw partition assignment CSV from cluster output | Required |
| `--corrections` | Corrections CSV (minimum: `id`/`image` and `cluster` columns) | Required |
| `--embeddings` | Optional embeddings CSV for distance recomputation | None |
| `--support-display-cutoff` | Minimum support threshold for annotated UPGMA tree labels | 50.0 |
| `--figure-width` | Annotated UPGMA PDF width in inches | Auto |
| `--annotate-bar-width` | Relative width of corrected OTU color bars | 0.08 |
| `--show-annotation-bar` | Show corrected OTU annotation bars in annotated UPGMA PDF | No |
| `--show-partitioning-bars` | Show partitioning bars in annotated UPGMA PDF | No |
| `--out-dir` | Output directory | `runs/annotate` |

**Output**:
- `partition_<cutoff>_assignments.csv` — Corrected assignments
- `partition_<cutoff>_assignments_changed_only.csv` — Changed rows only
- `otu_table.csv` — OTU table
- `pairwise_distance_summary_intra-class.csv` — Intra-class distance summary (with `--embeddings`)
- `UPGMA_tree_partitions_annotated.pdf` — Annotated UPGMA tree (with `--embeddings`)
- `annotation_summary.json` — Correction summary

---

### Diversity Command

Compute community alpha diversity indices. Supports two input modes: partition assignment CSV or OTU table CSV. For Faith's PD, the recommended path is to provide `--embeddings`, which builds an OTU-centroid neighbor-joining (NJ) tree internally. The older `--tree` input remains available as a legacy fallback.

```bash
# From partition assignments
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --out-dir runs/diversity

# With Faith's PD from OTU-centroid NJ (recommended)
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --phylo \
    --embeddings runs/extract/embeddings.csv \
    --save-nj-tree \
    --nj-bootstrap 100 \
    --out-dir runs/diversity

# Legacy Faith's PD path from an existing Newick tree
otuformer diversity \
    --assignments runs/cluster/UPGMA/partitions/tables/partition_0.30_assignments.csv \
    --phylo \
    --tree runs/cluster/UPGMA/UPGMA_Cosine.nwk \
    --out-dir runs/diversity

# From OTU table
otuformer diversity \
    --otu-table-csv otu_table.csv \
    --out-dir runs/diversity
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--assignments` | Partition assignment CSV (`id`/`image`, `cluster`, optional `sample`) | One of two |
| `--otu-table-csv` | OTU table CSV (sample column + OTU ID headers) | One of two |
| `--otu-table-has-header` | Force first row as header (required if OTU IDs are numeric) | No |
| `--out-dir` | Output directory | `runs/diversity` |
| `--min-abundance` | Minimum abundance thresholds (comma-separated) | `0,2,5` |
| `--phylo` | Enable Faith's PD / PD-related metrics | No |
| `--embeddings` | Embeddings CSV; recommended for PD because OTU-centroid NJ is built internally | None |
| `--tree` | Legacy Newick tree path used only when `--embeddings` is absent | None |
| `--save-nj-tree` | Save the inferred OTU-centroid NJ tree to `NJ_OTU.nwk` | No |
| `--nj-bootstrap` | Bootstrap replicates for NJ support; writes `NJ_OTU_bootstrap.nwk` | `0` |
| `--save-nj-centroids` | Save OTU centroid embeddings to `NJ_OTU_centroids.csv` | No |

**Output metrics** (saved to `diversity_indices.csv`, global and per-sample):
- **Richness**: Number of unique OTUs
- **Chao1**: Estimated richness (accounts for rare OTUs)
- **ACE**: Abundance-based coverage estimator
- **Shannon**: Entropy-based diversity (higher = more diverse)
- **Simpson**: Probability two individuals differ (higher = more diverse)
- **Hill q0/q1/q2**: Hill numbers (richness/evenness/diversity at orders 0,1,2)
- **Pielou_J**: Evenness (Shannon / log(richness))
- **Faith's PD (MPD)**: Morphological phylogenetic diversity computed from the OTU-centroid NJ tree (or legacy `--tree` input)
- **MPD_w**: Abundance-weighted rooted PD (rPD_w)
- **PD_richness_norm**: Faith's PD / species richness

**Output**:
- `diversity_indices.csv` — Global diversity indices
- `per-sample/` — Per-sample diversity files (when valid sample column exists)
- `NJ_OTU.nwk` — OTU-centroid NJ tree (with `--embeddings --save-nj-tree`)
- `NJ_OTU_bootstrap.nwk` — NJ tree with bootstrap support labels (with `--nj-bootstrap > 0`)
- `NJ_OTU_centroids.csv` — OTU centroid embeddings (with `--save-nj-centroids`)

---

### CAM Command

Generate CAM heatmaps for model interpretability analysis.

```bash
# Basic usage
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --out-dir runs/cam

# Use Grad-CAM++ and save raw arrays
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --cam-method gradcampp \
    --save-npy \
    --out-dir runs/cam

# Inspect model layer structure first
otuformer cam \
    --checkpoint runs/finetune/finetune_latest.pth \
    --images-dir ./images \
    --dump-model-structure \
    --out-dir runs/cam
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | OTU-Former checkpoint path (`.ckpt` or `.pth`) | Required |
| `--images-dir` | Directory of images to generate CAM for | Required |
| `--label-csv` | Optional CSV with `image` and `label` columns; if omitted, all images in `--images-dir` are used | None |
| `--out-dir` | Output directory | `runs/cam` |
| `--cam-method` | `gradcam`/`gradcampp`/`layercam`/`scorecam`/`eigencam`/`ablationcam` | `gradcam` |
| `--arch` | Force architecture: `cnn`/`vit` (auto-detected if omitted) | Auto |
| `--target-layer-name` | Specific CAM target layer (auto-selected if omitted) | Auto |
| `--image-weight` | Blend weight of original image in CAM overlay (0-1) | 0.5 |
| `--fig-format` | Output format: `png`/`jpg`/`pdf` | `png` |
| `--save-npy` | Save raw CAM arrays as NumPy files | No |
| `--dump-model-structure` | Write model layer names to `model_layers.txt` | No |
| `--max-images` | Maximum images to process (None = all) | All |
| `--cam-batch-size` | Batch size for CAM inference | 32 |
| `--num-workers` | Dataloader worker processes | 4 |
| `--device` | `auto`/`cpu`/`cuda`/`mps` | `auto` |

**Output**:
- `figures/` — CAM overlay images
- `cam_summary.csv` — Metadata summary
- `arrays/` — Raw CAM arrays (with `--save-npy`)
- `model_layers.txt` — Model layer names (with `--dump-model-structure`)

---

### Export Command

Export a PyTorch checkpoint to ONNX format for deployment and accelerated inference.

```bash
# Basic usage
otuformer export \
    --checkpoint runs/finetune/finetune_latest.pth \
    --out-dir runs/export

# Custom image size and opset
otuformer export \
    --checkpoint runs/finetune/finetune_latest.pth \
    --imgsz 224 \
    --opset 17 \
    --out-dir runs/export
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | Checkpoint path | Required |
| `--out-dir` | Output directory | `runs/export` |
| `--imgsz` | Input image size for ONNX export (auto-inferred from backbone if not provided) | Auto |
| `--opset` | ONNX opset version | 18 |

**Output**:
- `encoder.onnx` — ONNX encoder model

---

## Common Behaviours

### Logging

All commands save log files (e.g. `pretrain.log`, `finetune.log`, `extract.log`, `cluster.log`, `annotate.log`, `diversity.log`, `cam.log`, `export.log`) to the `logs/` subdirectory of the output directory, containing:
- Full command line
- Timestamp
- All parameter values
- Runtime output

### Graceful Shutdown

Press `Ctrl+C` — the current image finishes before exiting; partial results are saved.

### Device Selection

`--device auto` chooses automatically:
1. CUDA (if available)
2. MPS / Apple Silicon (if available)
3. CPU (fallback)

### Shell Completion

Install shell completion for otuformer:

```bash
otuformer --install-completion
```

Supported shells: bash, zsh, fish

### Version

Show installed version:

```bash
otuformer --version
otuformer -v
```

## Project Structure

```
.
├── src/otuformer/            # Core package
│   ├── cli/                  # Command-line interface
│   │   ├── main.py           # Entry point dispatcher
│   │   ├── doctor.py         # otuformer doctor
│   │   ├── pretrain.py       # otuformer pretrain
│   │   ├── finetune.py       # otuformer finetune
│   │   ├── extract.py        # otuformer extract
│   │   ├── cluster.py        # otuformer cluster
│   │   ├── annotate.py       # otuformer annotate
│   │   ├── diversity.py      # otuformer diversity
│   │   ├── cam.py            # otuformer cam
│   │   └── export.py         # otuformer export
│   ├── training/             # Training logic
│   │   ├── model.py          # Model definition
│   │   ├── dataset.py        # Dataset
│   │   ├── loss.py           # Loss functions
│   │   ├── trainer.py        # Trainer
│   │   └── scheduler.py      # Learning rate scheduler
│   ├── embedding/            # Embedding extraction
│   │   ├── extractor.py      # Embedding extractor
│   │   └── evaluator.py      # Embedding evaluator
│   ├── delineation/          # OTU delineation
│   │   ├── distance.py       # Distance computation
│   │   ├── tree.py           # UPGMA tree building
│   │   ├── partition.py      # Partitioning and metrics
│   │   └── annotate.py       # Correction and annotation
│   ├── vision/               # Vision utilities
│   │   ├── cam.py            # CAM visualization
│   │   └── export.py         # ONNX export
│   └── utils/                # Utility functions
│       ├── device.py         # Device detection
│       ├── logging.py        # TeeLogger
│       ├── checkpoint.py     # Checkpoint handling
│       └── io.py             # CSV/JSON I/O
├── tests/                    # Test files
├── examples/                 # Sample data assets
├── docs/                     # Documentation
└── pyproject.toml            # Project configuration
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Email: `xtmtd.zf@gmail.com`

## Citation

If you use OTU-Former in your research, please cite:

```bibtex
@software{otuformer2026,
  author = {Zhang, Feng},
  title = {OTU-Former: Image-based Morphological OTU Delineation Toolkit},
  year = {2026},
  url = {https://github.com/xtmtd/OTU-Former}
}
```

A preprint is available at: https://doi.org/10.64898/2026.04.28.721370
