# OTU-Former Toolbox — Design Spec

Date: 2026-03-29

## 1. Overview

OTU-Former is a Python CLI toolbox for image-based biodiversity analysis. It converts standardised specimen images (e.g. dorsal beetle photos) into morphological barcodes (fixed-dimension embeddings) via a trained encoder, clusters them into morphological OTUs (morphOTUs) using UPGMA hierarchical clustering, and produces diversity data (OTU table + abundance + alpha diversity indices).

The toolbox packages an existing research workflow (ref/ibot20260115.py, ref/embeddings_tree20260206.py, ref/GradCam_heatmap.py, ref/diversity_index.txt) into a single installable Python package with a unified CLI.

**Package version:** see `pyproject.toml`

**Python:** ≥ 3.11

**CLI framework:** Typer (with `--install-completion` built-in)

---

## 2. Architecture

### 2.1 Package structure

```text
src/otuformer/
├── cli/          command parsing and orchestration
├── training/     pretraining and fine-tuning
├── embedding/    extraction, evaluation, pseudo-label logic
├── delineation/  distances, trees, partitions, diversity
├── vision/       CAM generation and ONNX export
└── utils/        shared I/O, device, path, size, logging, checkpoint helpers
```

The tree is deliberately directory-level. A file-by-file list was maintained here
until v0.10.1 and had drifted: it listed a `cli/` module that never existed and
omitted several modules that do (`cli/update.py`, `vision/export.py`,
`constants.py`, and four `utils/` modules). The user-facing command reference for these modules lives under
`docs/commands/` (Section 7) and is not duplicated here.

### 2.2 Layer convention

```
CLI (cli/)  →  app service (inline in cli/)  →  core logic (training/ embedding/ delineation/ vision/)
```

The CLI layer is thin (argument parsing + logging scope + result printing). Core logic modules are importable independently of the CLI, enabling future Python API / skill usage.

---

## 3. Sub-commands

Each sub-command below records its purpose, its outputs, the constraints a later
change must not break, and links to the design document that introduced each
behaviour. Flags, defaults, and allowed values are **not** listed here: they are
owned by the user-facing command documents under `docs/commands/` (Section 7),
while the design docs linked below remain the record of *why* each behaviour
exists.

### 3.1 `pretrain`
SSL self-supervised pre-training (teacher-student self-distillation + patch-level objective).
All parameters migrated from `ref/ibot20260115.py` `get_parser()`, mode=pretrain.

**Outputs:** `SSL_latest.pth` (and epoch checkpoints `SSL_epoch_*.pth`), `metrics.pretrain.csv`, `instant_metrics.csv`, training curves PDF, log file. When labels are absent or contain fewer than two classes, supervised metric fields remain empty while the epoch row is retained; an image-only visualization CSV still produces UMAP. `--metrics-sample-size` limits both supervised metric and UMAP inputs when positive.

**Key constraints:**
- `--global-crop-size` : `auto` resolves to the backbone's native `default_cfg["input_size"]` on a new run or the checkpoint's recorded size on `--resume`
- On `--resume`, omitted `--local-crop-size` and `--local-crops` inherit the checkpoint's saved values; explicit conflicting values fail, and changing them requires a new run.
- `--mask-ratio` : `auto` resolves to 0.30 on a new run, while v0.6.x defaulted to 0.50; a legacy `--resume` keeps its recorded value or falls back to 0.50
- `--patch-loss` : patch-level objective — `none`, `consistency` (default; visible same-position cosine consistency, not input masking), `masked-feature` (A+; continuous masked feature prediction), or `ibot` (experimental prototype prediction)
- `--masking-strategy` : real-masking geometry for `masked-feature`/`ibot` — `random` (default), `blockwise`, or `hybrid`
- `--model-name` : a `vit_*` name alone does not establish compatibility (see Section 4.9)
- Transform-level profile definitions live in [`2026-09-07-otuformer-training-augmentation-design.md`](2026-09-07-otuformer-training-augmentation-design.md).

**Design history:**
[pretrain alignment](2026-03-30-otuformer-pretrain-alignment-design.md),
[v0.6 masked pretraining](2026-09-20-otuformer-masked-pretrain-v070-design.md)
(the v0.7.0 masked patch objectives),
[v0.7.1 CLI and metric corrections](2026-09-21-otuformer-v071-cli-metrics-corrections.md),
[optional register tokens](2026-09-28-otuformer-optional-registers-design.md).

### 3.2 `finetune`
Supervised metric-learning fine-tuning on top of a pretrained checkpoint. ArcFace remains the default; v0.8.0 also implements SupCon, Sub-center ArcFace, and compact Sub-center ArcFace. New runs initialized from SSL checkpoints replace the SSL projector with a compact, unnormalized `backbone_dim -> 512 -> metric_embed_dim` embedding head; the selected loss handles its required normalization.

**Outputs:** `finetune_latest.pth` (and epoch checkpoints `finetune_epoch_*.pth`), `metrics.finetune.csv`, `instant_metrics.csv`, training curves PDF, log file. Pseudo mode additionally writes `pseudo_labels.csv` and `pseudo_summary.json` in the finetune#2 directory. When labels are absent or contain fewer than two classes, supervised metric fields remain empty while the epoch row is retained; an image-only visualization CSV still produces UMAP. `--metrics-sample-size` limits both supervised metric and UMAP inputs when positive.

**Key constraints:**
- Fine-tuning uses separate backbone and metric-head learning rates, defaults to AdamW `weight_decay=1e-4` as a conservative supervised-training choice rather than a full legacy-script reproduction, and does not apply gradient clipping.
- Initialization semantics for `--checkpoint`: an SSL pretrain checkpoint installs a fresh `ArcFaceEmbeddingHead`; a fine-tune checkpoint whose head and width match the head being trained keeps its trained projector; a `ProjectionHead` checkpoint (OTU historical SFT, or `projection_mlp_2048` metadata) keeps its projector and therefore fixes the embedding width.
- `--metric-embed-dim` : An explicit value that conflicts with a `ProjectionHead` checkpoint is rejected, because that projector's output width is fixed by the pretrained weights. On `--resume` the value must match the checkpoint (resizing is only valid for a new `--checkpoint` run).
- `--metric-head-lr` : omitted means inherit `--finetune-lr`. On `--resume`, saved optimizer state takes precedence.
- `--orientation-policy` : when initializing from `--checkpoint`, an omitted value inherits the pretraining checkpoint's saved policy
- Checkpoints record the actual `config.embedding_head` (`arcface_mlp_512` for new runs, `projection_mlp_2048` for runs initialized from a historical SFT checkpoint) and `config.freeze_ratio`. The historical `loss_state_dict` marker is recognized when metadata is absent; missing metadata without fine-tune markers remains valid SSL initialization. On resume, the checkpoint's optimizer state restores its saved parameter-group settings, so CLI LR and weight-decay values are inert, and a changed `--freeze-ratio` is rejected because the optimizer parameter groups depend on it.
- Ref-script checkpoints (`ref/ibot20260115.py`: `model`/`teacher`/`student` weight keys, `projector.<i>` projector naming, an `args` dict instead of `config`, `loss_func.W` classifier) are readable by `extract`, `export`, and `cam`, but cannot be resumed by `finetune` — see 4.8.
- `--long-tail` : CB-DRW applies only to ArcFace-family cross-entropy and must match between pseudo rounds
- The following sparse-label extension is implemented (v0.9.0):
- Without `--pseudo-label-from`, ordinary fine-tune semantics are unchanged. Pseudo mode uses two explicit finetune commands. Omitted experiment options inherit finetune#1 resolved values rather than new-run defaults; explicit conflicts fail. Finetune#2 always initializes from the same original SSL checkpoint, never finetune#1. In pseudo mode the existing `--checkpoint` only locates that SSL file after it has moved, and its file SHA-256 must match. Full rules are specified in [the sparse-label pseudo-feedback design](2026-09-26-otuformer-sparse-label-pseudolabel-design.md).
- New ordinary fine-tune checkpoints add `pseudo_round=0`, the existing expert-manifest hash, original SSL resolved path and file SHA-256, resolved experiment identity, and `pseudo_source_eligible` plus an optional reason. Finetune#2 adds `pseudo_round=1`, pseudo-source SHA-256, authoritative accepted pseudo rows and their hash, rule/preprocessing metadata, counts, and long-tail schedule. These fields are additive; existing model/config keys remain authoritative.

**Design history:**
[v0.8.0 metric losses](2026-09-24-otuformer-metric-loss-v080-design.md),
[v0.9.0 sparse-label pseudo-feedback](2026-09-26-otuformer-sparse-label-pseudolabel-design.md),
[training augmentation](2026-09-07-otuformer-training-augmentation-design.md).

### 3.3 `extract`
Extract embeddings from images using a trained checkpoint.

Parameters migrated from `ref/ibot20260115.py` `get_parser()`, mode=extract.

**Batch mode:** When `--input-images-dir` contains subdirectories, each subdirectory is treated as an independent sample set. All embeddings are extracted and merged into a single CSV with a `sample` column recording the source subdirectory. This ensures consistent OTU naming across datasets when feeding into `cluster`.

**Checkpoint formats:** Accepts OTU checkpoints (`model_state_dict`/`teacher`/`student` + `config`) and ref-script checkpoints (`teacher`/`student`/`model` + `args`). The embedding head and embedding dimension are read from `config.embedding_head` when present and otherwise inferred from the projector shapes, so a missing or wrong metadata value cannot cause a silent mis-load. Ref-script **SFT** checkpoints record no `config`/`args`, so they need an explicit `--model-name`.

**Outputs:** `embeddings.csv` (columns: `id`, `sample` (batch only), then embedding dimensions), optional `umap.pdf`, and `metrics.csv` only when at least two label classes are available. Positive `--metrics-sample-size` limits both metric and UMAP inputs; values <=0 disable the cap.

**Key constraints:**
- `--use-projector-output` : use SSL projector output for SSL checkpoints, or the ArcFace embedding for fine-tune checkpoints; recommended for fine-tuned-class retrieval and closed-set clustering
- `--token-mode` : `cls` is raw CLS and remains the comparison baseline for unseen classes, cross-dataset transfer, and general morphology representation
- `--label-csv` : labels enable quality metrics, while image-only input enables unlabeled UMAP. Attention-pool query training still requires `label`.
- `--eval-transform` : `center-crop` | `whole-specimen-pad` [default: center-crop; deterministic evaluation preprocessing protocol, applied to extraction and CAM]

**Design history:**
[v0.7.1 metric corrections](2026-09-21-otuformer-v071-cli-metrics-corrections.md),
[optional register tokens](2026-09-28-otuformer-optional-registers-design.md).

### 3.4 `cluster`
Compute pairwise distances, build UPGMA tree, scan partitions.

Parameters migrated from `ref/embeddings_tree20260206.py`:

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

**Key constraints:**
- `--labels` : optional, for partition quality metrics (NMI, ARI, BCubed-F etc.)

**Design history:**
[v0.7.1 metric corrections](2026-09-21-otuformer-v071-cli-metrics-corrections.md),
[output safety and resume](2026-07-18-output-safety-and-resume-design.md).

### 3.5 `annotate`
Write expert taxonomic corrections back into partition assignments.

**Outputs:**
- `partition_{cutoff}_assignments.csv` and `partition_{cutoff}_assignments_changed_only.csv`
- `otu_table.csv`
- `annotation_summary.json` : number of corrections, clusters affected, before/after distribution
- `pairwise_distance_summary_intra-class.csv` and `UPGMA_tree_partitions_annotated.pdf` (only with `--embeddings`)

**Key constraints:**
- IDs present in `corrections` override the cluster assignment in `raw_assignments`
- IDs absent from `corrections` are kept unchanged

**Design history:**
[the annotate design](2026-04-01-annotate-design.md).

### 3.6 `diversity`
Compute alpha diversity indices from a (potentially annotated) partition assignments file.

**Diversity indices computed (per `--min-abundance` threshold):**

| Category     | Indices                                                                  |
|--------------|--------------------------------------------------------------------------|
| Richness     | Richness, Chao1, ACE, Margalef, Menhinick                                |
| Evenness     | Pielou's J, Heip's E                                                     |
| Diversity    | Shannon H', Simpson 1-D, Inverse Simpson 1/D, Fisher's alpha, Brillouin  |
| Dominance    | Berger-Parker                                                            |
| Hill numbers | q=0, q=1, q=2                                                            |
| Morpho tree  | MPD (optional, `--phylo`)                                                |

**Outputs:** `diversity.csv`, `diversity_summary.pdf` (bar charts per index), log file.

**Key constraints:**
- `--phylo` : compute MPD (Morphological Dendrogram Diversity); requires `--tree`
- `--tree` : UPGMA Newick file (required if `--phylo`)
- All diversity calculations in pure Python (scipy/scikit-bio/numpy). No usearch dependency.

**Output format:** Wide table — rows = indices, columns = `min_abundance_0`, `min_abundance_2`, `min_abundance_5` etc.

**Design history:**
[the diversity design](2026-04-01-diversity-design.md).

### 3.7 `cam`
Generate CAM heatmaps for visual explanation of morphological features.

Implementation mirrors `entomokit classify cam` (`/Users/zf/data/coding/entomokit/entomokit/classify/cam.py` and `src/classification/cam.py`), adapted for OTU-Former checkpoints (timm ViT backbone, no AutoGluon).

**Outputs:** Per-image overlay images, `cam_summary.csv`, `model_layers.txt` (if `--dump-model-structure`).

**Key constraints:**
- Heatmaps are overlaid on the full original image via the correct inverse of the selected preprocessing: with `center-crop` the heatmap is confined to the model's field of view using a binary FOV mask (out-of-view pixels are dimmed directly, never routed through the heatmap colormap); with `whole-specimen-pad` the heatmap covers the whole specimen. The figure is a gradient-attribution heatmap of the raw CLS feature (its argmax treated as a pseudo-class), not a verified class-discriminative CAM; that semantic is tracked separately.
- `--eval-transform` : `center-crop` | `whole-specimen-pad` [default: center-crop; deterministic evaluation preprocessing protocol shared with extraction]

**Design history:**
[optional register tokens](2026-09-28-otuformer-optional-registers-design.md).

### 3.8 `export`
Export encoder + projector to ONNX for deployment.

**Outputs:** `encoder.onnx`, `export_report.json`.

**Key constraints:**
- Always exports encoder + projector only. The projector is rebuilt from the checkpoint's `config.embedding_head`, falling back to the projector shapes, so fine-tune checkpoints with an `ArcFaceEmbeddingHead` export correctly; the ref-script `projector.<i>` naming is normalized on read. Output embedding dimension is read from the checkpoint metadata (set by `--out-dim` or `--metric-embed-dim` at training time) — not hardcoded. A backbone that does not fit the resolved model name raises an error naming whether the conflict is in the checkpoint metadata or in `--model-name`.

**Design history:**
[optional register tokens](2026-09-28-otuformer-optional-registers-design.md).

### 3.9 `doctor`
Check environment health.

**Checks:** Python version, PyTorch + CUDA/MPS availability, timm, scikit-bio, umap-learn, grad-cam, onnx, key package versions.

## 4. Key Design Decisions

### 4.1 Distance metrics
Both cosine and Euclidean distances support: PCA whitening, k-NN based local scaling (as implemented in `ref/embeddings_tree20260206.py` `apply_local_scaling()`), chunked computation for large datasets. Euclidean distance is sensitive to scale; documentation recommends combining with PCA whitening or L2 normalisation.

### 4.2 Loss registry
`training/loss.py` implements a simple registry pattern:
```python
LOSS_REGISTRY = {"arcface": ArcFaceLoss, ...}
```
The v0.8.0 registry implements ArcFace, SupCon, Sub-center ArcFace, and compact
Sub-center ArcFace with loss-aware CLI validation, batch handling, and
checkpoint/resume compatibility; see [the implemented metric-loss design](2026-09-24-otuformer-metric-loss-v080-design.md). CB-DRW is a later opt-in training strategy over existing ArcFace-family per-row cross-entropy; it is not another registered loss.

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

Finetune#2's accepted pseudo rows and provenance are additive top-level/config payload fields used only by training resume. The shared read-only consumers ignore them and continue to resolve architecture and weights from the existing keys, so `extract`, `export`, and `cam` load finetune#2 checkpoints without schema-specific handling.

### 4.9 Backbone compatibility boundary
OTU-Former does not promise support for every timm backbone or every model whose name starts with `vit_`. Pretraining's masked/student-teacher forward follows timm's standard `VisionTransformer` token and block interfaces; compatibility is determined by the actual implementation and required capabilities, not the model-name prefix. The cached `vit_tiny_patch16_224.augreg_in21k_ft_in1k` is a standard `VisionTransformer` in timm 1.0.27. In the same version, `vit_small_patch16_dinov3.lvd1689m` is implemented as `Eva`, with four native registers and a different positional-embedding/rotary-block interface; it is **not** supported by the current pretraining forward or by the [optional-register design](2026-09-28-otuformer-optional-registers-design.md). Supporting that model requires a separately reviewed Eva/DINOv3 adaptation and end-to-end tests, not a broadened `vit_*` name check. The two cached models do not imply that all models in either family are tested or supported.

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

- Metric-loss comparisons for morphOTU delineation: see [the v0.8.0 design](2026-09-24-otuformer-metric-loss-v080-design.md). ProxyAnchor remains a possible later experiment, not a v0.8.0 requirement.
- `report` sub-command (HTML/PDF summary of a full analysis run)
- Beta diversity matrix + cross-sample OTU comparison
- Python API / skill for notebook-based non-CLI analysis (after interfaces stabilise)

---

## 7. Documentation Conventions

The documentation has three layers, each with one owner:

```text
README.md / README.cn.md          front page, installation, operational flags,
                                  and the command table
    └── links to →                docs/commands/<command>.md (+ .cn.md)

docs/superpowers/specs/*.md       design record: purpose, contracts, invariants
docs/superpowers/plans/*.md       execution plans
```

The load-bearing rules:

- **`docs/commands/` owns the analysis-command flag surface.** Defaults, allowed
  values, and worked examples live there. The CLI is the executable source of
  truth and the document must match generated `--help`.
- **The master design links to design specs only**, never to `docs/commands/`.
  Reader path to current flags: README → command document.
- **Design specs may name flags a decision or invariant requires**, but they do
  not duplicate an exhaustive current parameter reference.
- **Documentation is fully bilingual** with the `.cn.md` suffix, enforced by a
  pairing and structure test.
- **Documentation assertions read a corpus**, not fixed filenames, so moving
  reference material cannot silently break a content test.

The full conventions — layer model, language and naming rules, the command
document skeleton, the `--help` boundary, the new-command checklist, and the test
corpus contract — are in
[the documentation conventions design](2026-09-29-otuformer-documentation-conventions-design.md).
