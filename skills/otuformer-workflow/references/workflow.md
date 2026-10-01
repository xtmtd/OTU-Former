# Workflow guidance

Dependency graph (arrows mean "produces input for", not "must run again"):

```text
doctor -> pretrain -> finetune -> extract -> cluster -> annotate (optional)
       -> diversity

checkpoint -> cam / export
exported ONNX -> extract (within the actual CLI contract)
```

Every step: obtain a fresh schema, present the complete parameter card, obtain
approval for that single step, run the approved argv in the verified
environment, inspect exit status, logs, and real artifacts, then report a short
summary plus a recommendation (continue / redo / stop) and keep the task moving
on the user's instruction. Users with compatible artifacts may start at any
stage; validate only the prerequisites of the requested step.

Recommended next step after each command (a recommendation, not an automatic
action; an optional branch may be skipped when the user's goal does not need it):

| After | Recommend | Redo/stop when |
|---|---|---|
| `doctor` | the requested stage, or `pretrain` | dependencies are missing or the device is wrong |
| `pretrain` | `finetune`, or `extract` for an SSL-only comparison | loss/metrics stalled at the start, checkpoint missing |
| `finetune` | `extract` | class balance or metrics look broken, wrong loss mode |
| `extract` | `cluster` | embedding count does not match the images, metrics unusable |
| `cluster` | `annotate` (optional) then `diversity` | no usable cutoff, degenerate partition sizes |
| `annotate` | `diversity` | corrections did not match the partition IDs |
| `diversity` | `cam` / `export`, or wrap up and report | sample column or tree input does not match the data |
| `cam` | `export`, or wrap up | target layer or checkpoint mismatch |
| `export` | `extract` via ONNX, or wrap up | opset/size unusable for the target runtime |

Artifact names below are the documented ones for the installed version. Confirm
the actual files in the run's output directory before reporting success.

## doctor

Environment and dependency health report. No analysis parameters. Its exit code
can be zero while dependencies are missing, so report the actual packages and
available devices. Run it before substantive work in a new conversation or when
the environment is unknown.

## pretrain

SSL self-supervised pre-training (DINO/iBOT-style) that produces a checkpoint for
fine-tuning or direct embedding extraction. Typical artifacts: `SSL_latest.pth`,
`SSL_epoch_*.pth` written to `--out-dir`, plus logs and metrics inside that
directory. `--resume` is supported; combining `--resume` with `--overwrite` is
rejected. Explain possible initial timm weight downloads before approval.

## finetune

Metric-learning fine-tuning from a pretrained checkpoint. Typical artifacts:
`finetune_latest.pth`, `finetune_epoch_*.pth`, metrics and figures in
`--out-dir`. `--resume` is supported; resume and overwrite conflict. Loss-specific
options (`--subcenters`, `--compact-weight`, pseudo-label options) apply only to
their own loss modes; omitted options stay omitted so the resumed checkpoint
keeps its recorded settings. Pseudo-label feedback preserves the original SSL
provenance and must not be turned into an automatic multi-round loop.

## extract

Embedding extraction from images using a checkpoint or an exported ONNX model.
Typical artifacts: `embeddings.csv` (`id` plus `dim_0...`, where `id` is the image
file name), `metrics.csv` and `umap.pdf` when `--label-csv` provides at least two
classes. Requires `--checkpoint` or `--onnx-path`; ONNX takes precedence, ignores
the checkpoint/device, and supports only `--token-mode cls`.

When images sit in subdirectories below `--input-images-dir`, `embeddings.csv`
also carries a `sample` column inferred from the **first directory level** (or
taken verbatim from a `sample` column in the input CSV). For a demo laid out as
`<label>/<image>`, `sample` therefore equals the label directory name.

## cluster

Partitions embeddings into candidate OTU sets. Typical artifacts:
`UPGMA/partitions/partition_scan.csv`,
`UPGMA/partitions/tables/partition_<cutoff>_assignments.csv`,
`UPGMA/metrics.csv` and `UPGMA/metrics_dashboard.pdf` with `--label-csv`, and
`UPGMA_Cosine.nwk`. Report candidate cutoffs; do not present one partition as
scientifically optimal.

## annotate

Applies a reviewed correction table to a selected partition. Typical artifacts:
`partition_<cutoff>_assignments.csv`, `partition_<cutoff>_assignments_changed_only.csv`,
`otu_table.csv`, `annotation_summary.json`, and (with `--embeddings`)
`pairwise_distance_summary_intra-class.csv` and
`UPGMA_tree_partitions_annotated.pdf`. Correction fields are `id` (or `image`)
plus `cluster`; unmatched or ambiguous IDs must be rejected, not guessed.

## diversity

Diversity indices from assignments or an OTU table. Typical artifacts:
`diversity_indices.csv` (one row per index, one column per `--min-abundance`
threshold), `per-sample/` when a valid sample column exists, and with NJ options
`NJ_OTU.nwk`, `NJ_OTU_bootstrap.nwk`, `NJ_OTU_centroids.csv`. Exactly one of
`--assignments` or `--otu-table-csv` is required; `--embeddings` and the legacy
`--tree` may be combined, with `--embeddings` taking precedence. Label classes
are not ecological sample sets: for an example-style `<label>/<image>` layout the
`sample` column in `embeddings.csv` is that label directory name, so per-sample
diversity separates the two label classes rather than two communities. Confirm
which field actually encodes sampling site or unit before interpreting it.

## cam

Class-activation heatmaps for a checkpoint. Typical artifacts: `figures/`
overlays, `cam_summary.csv`, optional `arrays/` for `--save-npy raw|normalized`,
and `model_layers.txt` with `--dump-model-structure`. Raw-feature gradient CAM is
attribution guidance, not a validated class-discriminative explanation.

## export

ONNX export from a checkpoint, producing `encoder.onnx` (plus any observed report
files) in `--out-dir`. Confirm the actual opset and image size from the run, and
explain that downstream ONNX extraction supports only `--token-mode cls`.

## update

Environment maintenance, not an analysis step: `update --check` is read-only but
contacts GitHub, and installation requires separate environment-change approval.
Never add `--yes` to bypass consent, and never treat `--check` as an install.
