# OTU-Former v0.8.0 Metric-Loss Design

Date: 2026-09-24  
Status: Implemented; execution tracked in the v0.8.0 implementation plan  
Target release: v0.8.0

This design extends supervised `otuformer finetune` without changing SSL
pretraining, extraction defaults, cosine distance, UPGMA, or expert corrections.
The goal is to compare how different supervised objectives affect image
embedding geometry: known-class compression, preservation of within-class
multimodality, and transfer of that geometry to species whose labels were not
used during fine-tuning. ArcFace remains the CLI default and the v0.8.0
reference loss. The v0.8.0 new-run optimizer protocol gives normalized class
prototypes zero weight decay; this is not a numerically controlled
continuation of v0.7.x new ArcFace runs. Legacy `--resume` preserves the
saved optimizer semantics. The tool does not infer why a class has multiple
modes.

## 1. Motivation and Evidence

The existing `--loss` registry contains only ArcFace. An unknown loss name
currently falls back silently to ArcFace. The training loss is applied to the
projector output, while periodic fine-tune metrics and default `extract` use
the backbone's raw CLS embedding. Thus lower training loss or improved
projector retrieval does not establish better distances in the embedding
actually used for morphOTU delineation.

In the seven-species *Prosopocoilus* dataset, expert inspection finds distinct
size-associated branches in *P. astacoides*, *P. suturalis*, and
*P. forficula*. Existing raw-CLS cosine distances show substantial within-
species spread, especially for the latter two. At a cosine UPGMA cutoff of
0.38, the recorded analysis yields eight pure OTUs for seven labeled species;
at 0.41, it yields six OTUs with a cross-species merge. Raising the global
cutoff alone is therefore not a satisfactory answer. These observations
motivate an experiment; they do not prove that any proposed loss will improve
unseen-species delineation.

## 2. Scope and Decisions

Four mutually exclusive `finetune --loss` modes are in scope:

| Mode | Role | Training signal |
| --- | --- | --- |
| `arcface` | CLI default and v0.8 reference loss under the new prototype-decay protocol | One angular-margin classifier center per labeled species |
| `supcon` | Direct image-pair baseline | Same-label positives and different-label negatives within a batch |
| `subcenter-arcface` | Multi-center baseline | Each image approaches its closest center of the labeled species |
| `subcenter-arcface-compact` | Experimental multi-center variant | Sub-center ArcFace plus a bounded same-class center-distance penalty |

These names describe the intended CLI interface, not currently supported
commands. Species labels alone determine positive and negative training
relations; size, sex, geography, and other possible causes are not model
targets. The tool must not name the biological cause of a center, force
centers to remain distinct, discard nondominant centers, or treat center
assignment as a species/OTU determination. Ordinary ArcFace has one learned
classifier center per class, although its image embeddings may still contain
multiple local modes.

No changes to `pretrain`, `cluster`, `annotate`, embedding export defaults,
distance calibration, or the OTU assignment algorithm are in scope. No
ProxyAnchor, traditional contrastive plus pair mining, SoftTriple, mixed
ArcFace/SupCon loss, or automatic center-number selection is included in the
first comparison. Additional loss families require a separate result-driven
decision.

The four modes test different hypotheses:

- `arcface`: a single-center angular-margin baseline for supervised fine-tuning;
- `supcon`: whether direct batch-wise sample relations improve the geometry;
- `subcenter-arcface`: whether retaining class multimodality transfers better
  to labels not used during fine-tuning;
- `subcenter-arcface-compact`: whether a controlled same-class center
  contraction is a useful compromise between a single center and unconstrained
  sub-centers.

Similar results across the modes are valid evidence that this dataset and
training protocol do not distinguish the objectives. No mode is presumed to
improve every known-class or unseen-class metric.

## 3. Training Objectives

All modes share the current backbone, fine-tune embedding head, optimizer
structure, `MetricDataset` label mapping, and `finetune` entry point. They
produce ordinary image embeddings; classifier centers exist only in the
training loss and are not required for `extract` or `cluster`. The loss is
applied to the fine-tune head output, while the default downstream artifact
is raw CLS. Consequently, differences between modes in raw-CLS geometry can
arise only through gradients reaching unfrozen backbone parameters; the
SupCon head itself is not used by default extraction. The comparison must
hold `freeze_ratio`, backbone learning rate, epoch budget, augmentation,
training data, batch size, and seed fixed across loss modes.

### 3.1 ArcFace and Sub-center ArcFace

`arcface` keeps its existing normalization, scale, margin, and class-center
layout. For `subcenter-arcface`, use K normalized learned centers per species
and image-to-class similarity equal to the maximum cosine similarity over
that class's K centers. Apply the existing ArcFace margin to the resulting
target class logit. K is an explicit small training setting, an integer
from 2 to 8 shared by the labeled classes for simplicity (initial comparison
K=2; larger K are optional ablations). K is a capacity limit, not a prediction of the
number of biological morphs. A center may receive no assignments.

Standard Sub-center ArcFace relaxes the single-center constraint: it does
not require two centers of the same species, or their assigned images, to be
close. The published noise-cleaning step that discards nondominant centers
must **not** be imported: here they may represent genuine variation.

### 3.2 Compact Multi-center Variant

Add to the Sub-center objective a same-class center-distance hinge. This is
an experimental variant inspired by existing multi-center regularization, not
a claim of a new algorithm:

```text
L = L_subcenter + weight * mean_over_classes_and_center_pairs(
    max(0, cosine_distance(normalize(w[c,i]), normalize(w[c,j])) - cap)
)
```

`cap` is a permitted positive cosine distance in `(0, 2]`, and `weight`
controls the penalty; both must be recorded in checkpoint metadata and logs.
`cap=2` is a valid no-op boundary, although the first comparison uses the
fixed exploratory value recorded by the implementation plan. Below the cap,
the regularizer exerts no pressure to merge centers. It does not force
distinct centers to persist, and too small a cap or too large a weight can
erase the reason to use multiple centers. Compare this variant directly
against the same K with zero center penalty. Center regularization alone
constrains classifier weights, not every pair of images, so improvement in
raw-CLS distances must be measured, not inferred. Compact uses the same K
setting as plain Sub-center (2 to 8) and exposes only one compact penalty
parameter for a controlled comparison; the first-round comparison defaults to
K=2 and the cap is fixed and recorded. The first-round exploratory
settings are temperature `0.07`, cap `0.5`, and compact weight `0.1`; these
are starting values, not validated optima. A compact weight of `0` is allowed
for the implementation-level equivalence test against plain Sub-center, but
is not a ranked first-round mode setting. The tool does not perform a
hyperparameter grid search. Multi-center regularization
has precedents (notably SoftTriple); this combination is an experimental
variant, **not** an unqualified claim of algorithmic novelty.

### 3.3 Supervised Contrastive

Use a single-view supervised contrastive variant on L2-normalized fine-tune
embedding head outputs. For each anchor in a batch, same-species images other
than the anchor are positives and different-species images are negatives. The
loss raises the anchor's similarity to all available positives while lowering
its relative similarity to the negatives through a temperature-controlled
log-softmax denominator. It therefore optimizes batch-wise sample-to-sample
relations rather than sample-to-class-center relations. It does not require
morph, sex, size, or geography labels.

Exclude self-comparisons. Only anchors with at least one other same-species
image in the batch contribute. Skip batches with no valid positive anchors or
no different-species images, count and report such batches, and fail the run
if an entire epoch provides no usable batches; never emit NaN or silently
train on an empty signal. Report valid-anchor counts so ordinary random
shuffling can be assessed. The CLI batch-size default remains 32; do not
lower it to 16 merely to enable SupCon. If random batches prove insufficient,
revisit sampling explicitly rather than hiding a new sampling regime inside
the loss. Temperature is a recorded training setting.

This is not the two-view augmentation protocol from the original SupCon
paper; it is deliberately a simpler single-view variant for the existing
fine-tune data flow. A same-species pair contributes directly only when both
images occur in the same batch. An optional batch-ID trace may record image
IDs for later external analysis of cross-morph pair coverage without making
morphology metadata part of training.

SupCon compares samples directly but can compress useful within-species
structure or bring distinct species too close. It is a candidate, not a
presumed replacement for ArcFace. The traditional contrastive loss with
pair mining reported in Hunt et al. is a different objective.

## 4. CLI, Optimizer, and Checkpoint Contract

`--loss` accepts only the four explicit names and rejects unknown values
before creating a training run. ArcFace remains the default. Every finetune
source must carry recognizable encoder parameters, so a loss state without any
model state is rejected rather than deferred to the trainer. K applies to both
Sub-center modes and the compact-variant settings apply only to
`subcenter-arcface-compact`; incompatible settings fail clearly rather than
being silently ignored. `--subcenters` is an integer from 2 to 8, and
`arcface`/`supcon` reject it. Preserve the
existing ArcFace defaults and existing readable fine-tune checkpoints.

All learnable class prototypes that are L2-normalized in the forward pass
use `weight_decay=0`, for both ordinary ArcFace class weights and Sub-center
weights. Other metric-head and backbone parameters retain the existing
fine-tune weight-decay policy. Because the forward pass L2-normalizes the
prototypes, the requested decay only rescaled their norm, which the loss never
observes; at the default `--finetune-lr 1e-4` and `--weight-decay 1e-4` the
per-step rescaling is below float32 resolution, so a new v0.8.0 ArcFace run is
numerically identical to a v0.7.x run. A larger learning-rate x weight-decay
product does change the raw prototype norm and can perturb the trajectory, so
v0.8.0 ArcFace is the reference baseline for this comparison rather than a
guaranteed loss-only cross-version control; legacy resumes retain their saved
optimizer semantics. This keeps K=1 Sub-center numerically aligned
with ArcFace instead of introducing an optimizer difference. A unit test
must compare K=1 Sub-center logits/loss against ArcFace with identical
weights, inputs, labels, scale, and margin.

New checkpoints record the loss name and applicable loss hyperparameters in
`config`, alongside the training loss state, class-label mapping, embedding
head type, and optimizer state already saved. They also record the seed, the
SSL initialization checkpoint hash, a canonical training-manifest hash,
the optimizer parameter-group description, and the prototype weight-decay
rule. The manifest hash must cover a canonicalized image/label listing, not
only a path. Canonical image references are normalized POSIX references
relative to `input_images_dir`; an absolute CSV reference must resolve inside
that directory and is converted to the same relative reference before hashing.
Normalize relative CSV references lexically (`./img.jpg` and `img.jpg` hash
identically); do not resolve them through filesystem lookup. Recursive image
lookup may load an image from a subdirectory, but the hash retains the
canonical CSV reference, not the discovered file path. Absolute references
must be contained under the image root before conversion; reject those outside.
Canonical rows preserve duplicates, sort by `(image_ref, label)`, and use
compact UTF-8 JSON, so changing the machine's absolute image root does not
change the manifest hash.

For fine-tune checkpoints, loss and head metadata are independent. Apply
this decision to both `--resume` and new-run `--checkpoint` initialization.
An SSL initialization source has a valid encoder `model_state_dict`, no
`loss_state_dict` key, no `class_labels`, and neither `config.loss` nor
`config.embedding_head`; it may still contain an SSL optimizer. A present but
empty `loss_state_dict` key is not the same as an absent key. Check ref-script
format first (`loss_func`/`model`, or `args` without `config`), then fine-tune
markers, before treating an unmarked encoder checkpoint as SSL. Repo SSL
checkpoints also contain `teacher`/`student` and `args`, but have `config`;
those keys alone do not identify a ref-script checkpoint:

| `config.loss` | `config.embedding_head` | `loss_state_dict` | Interpretation |
| --- | --- | --- | --- |
| absent | absent | key absent; `class_labels` absent, valid encoder weights | SSL initialization: `--checkpoint` installs a fresh head; reject `--resume` |
| absent | present | key absent | head-only initialization checkpoint: `--checkpoint` keeps the declared head for a new run; reject `--resume` (there is no classifier or optimizer state to restore) |
| present | present | key present; empty for SupCon, contains `head.weight` for prototype losses | v0.8.0 checkpoint; use each metadata field for its own purpose; reject inconsistent state, never fallback |
| present | absent | any | malformed v0.8.0 fine-tune checkpoint; reject |
| absent | present | key present; contains `head.weight` | legacy ArcFace fine-tune checkpoint; use recorded embedding head |
| absent | present | key present; empty or lacks `head.weight` | ambiguous fine-tune checkpoint; reject |
| absent | absent | key present; contains `head.weight` | historical ArcFace fine-tune checkpoint; infer head from actual projector weights |
| absent | absent | key present; empty or lacks `head.weight` | ambiguous fine-tune checkpoint; reject, including new-run initialization |
| any | any | key absent but `class_labels` or fine-tune loss metadata present | incomplete fine-tune checkpoint; reject |
| any | any | ref-script `loss_func`/`model` format, or `args` without `config` | read-only for extract/export/cam; reject finetune resume and new-run initialization |

`config.loss` is authoritative for v0.8.0 loss selection, while
`config.embedding_head` is authoritative for the embedding head. Infer a
missing head only for the historical ArcFace row with `head.weight` and
recognizable projector weights; do not infer a ProjectionHead from an empty
loss state or absent projector weights. A checkpoint identified as SSL (not a
fine-tune checkpoint) remains valid for `--checkpoint` initialization without
loss/head metadata, but only when it actually carries recognizable encoder
parameters in `model_state_dict`: a weightless checkpoint, or one whose state
dict has no `backbone.`/`projector.` parameter at all, is rejected rather than
mistaken for SSL initialization. Loading additionally requires the backbone to
load completely, so a partially populated state dict cannot silently train a
randomly initialized encoder. The projector must also load completely unless
the head is deliberately replaced: a new run whose source head or width does
not match installs a fresh head, while a source that declares a matching head
without its projector weights, and every `--resume`, is rejected rather than
trained with a randomly re-initialized projector. An SSL source receives a
fresh fine-tune head. A
declared head is head-only only when the `loss_state_dict` key is absent, and a
new `--checkpoint` run keeps that head and installs a fresh loss, because the
old classifier and optimizer state are deliberately not inherited; a present
but empty loss state next to a declared head is ambiguous and rejected, and
`--resume` rejects head-only sources because there is nothing to restore. The
decision-table, loss-metadata, and optimizer-layout rejections above are
enforced by the CLI preflight before any output directory is created, for a
resume source and for a new-run `--checkpoint` source alike; the backbone and
projector weight-completeness checks are enforced when the model loads, after
the output directory is prepared but before any training step.
An SSL checkpoint
cannot be resumed by `finetune`, even when it contains an optimizer and epoch;
the legacy missing-class-label warning does not make it a valid resume.
Resume must use the recorded loss type, K, and relevant loss hyperparameters,
and rebuild the proper loss module before loading its state/optimizer; an
explicit conflict is rejected. New runs use the v0.8.0 prototype decay rule;
legacy resumes preserve their historical optimizer grouping or use an
explicit state migration. The recorded `config.optimizer_groups` is the
authority on resume, so a checkpoint written by a legacy resume keeps its
two-group layout on later resumes instead of being reinterpreted from
`config.loss`. The implementation must never silently change the
optimizer semantics of an old checkpoint. Starting a **new** run from a
compatible fine-tune checkpoint may change loss, but does not inherit its
old classifier/optimizer state. SupCon has no classifier parameters;
checkpoint loading and validation must not assume `loss_state_dict` always
contains `head.weight`. Read-only checkpoint users continue to reconstruct
the embedding head and ignore training-only loss state. The implementation
plan must address these compatibility cases and keep classifier parameters
out of extracted embeddings.

## 5. Training Diagnostics

The implementation records diagnostics that cannot be reconstructed reliably
from a final checkpoint. `logs/loss_diagnostics.finetune.csv` is written for
every fine-tune run and does not alter the loss. The optional batch trace is
written to `logs/batch_ids.finetune.jsonl`; it is opt-in because it can be
large. Morphology metadata never enters the training input or loss:

- per-class, per-center local assignment counts and global argmax hit counts;
- center direction cosine against the previous epoch, since Adam/AdamW momentum
  can move a zero-current-gradient center after earlier activation. A resumed
  run continues this chain from the centers recorded in the checkpoint instead
  of restarting it, so the first resumed epoch reports a cosine too;
- ArcFace sample proportion currently satisfying the angular margin;
- SupCon valid-anchor counts and skipped-batch counts;
- compact center-pair hinge activation and weighted penalty;
- optional batch image-ID trace for external analysis of pair coverage.

A local assignment count of zero and a global argmax count of zero are
separate states. A zero current gradient does not prove that an Adam/AdamW
center is stationary. Compact regularization may reactivate a center that is
not currently a max-selected center, so its empty-center statistics must be
interpreted separately from ordinary Sub-center ArcFace. The loss never reads
morphology metadata; an external evaluator may join the optional batch IDs to
such metadata later.

## 6. Comparison and Interpretation

The downstream comparison holds `extract` raw CLS, cosine distance, and UPGMA
fixed. Because the training objective acts on the fine-tune head while the
primary downstream embedding is raw CLS, raw-CLS differences are mediated
only by the unfrozen backbone path. Hold `freeze_ratio` (default `0.7`),
backbone learning rate, metric-head learning rate, epoch budget, augmentation,
training manifest, batch size, and seed fixed across loss modes. Report
same-species between-morph distance alongside the closest different-species
distances and OTU splitting/merging: a lower within-species mean alone is not sufficient. A gain in compactness at the cost of new
cross-species merges is a failure for this purpose. Projector-output
embeddings are diagnostic only, not a substitute for improvement in raw CLS.
Center assignments and pairwise center distances may be inspected, but the
algorithm must not diagnose size, sex, geography, imaging artifacts, or
taxonomic status from them.

The evaluation protocol is external to the training command. Metrics that can
be calculated later from embeddings and checkpoints belong in external
analysis: pairwise distances, margins, ROC-AUC, kNN connectivity and
fine-tune-vs-SSL kNN overlap, cutoff calibration, specimen-level de-duplication,
and optional morphology-stratified reports. Required metadata are not
inferred by the tool. If specimen or morphology metadata are absent, the
corresponding report is omitted rather than making those fields mandatory for
ordinary label-only fine-tuning.

For a strict open-set comparison, each loss may be calibrated on a held-out
known-species set that did not participate in fine-tuning, then applied without
retuning to unseen species or genera. A deployment-oriented protocol may
calibrate on the available known-class validation data, but its results must
be reported separately: loss ranking must rely on threshold-independent
metrics and the strict protocol, because known-class compression can bias a
cutoff differently for each loss. The tool does not hard-code species roles,
calibration sets, folds, seeds, or benchmark runners.

SSL may use all available unlabeled images, including images later used for
evaluation. This is a transductive SSL setting, not an inductive claim that
the evaluation images were unseen during pretraining. Fine-tuning labels
remain limited to the user-selected training classes. An SSL-only checkpoint
is an external baseline, not another `--loss` mode.

The intended research setting permits SSL pretraining on unlabeled images
from all three genera, fine-tuning with labeled *Prosopocoilus*, and
evaluating the mixed three-genus partition. The user owns experimental
dataset preparation and evaluation; this design does not add a benchmark
runner or hard-code these taxa. Comparisons should hold non-loss training
conditions and downstream analysis constant. Claims of transfer or
biological interpretation require evidence beyond this design's implementation.

## 7. Release and References

The **subsequent implementation plan** must include updating both package
version locations to `0.8.0`, tests for loss behavior and checkpoint/resume
compatibility, and documentation of CLI options. This design approval alone
does not authorize implementation, training runs, or a git commit.

- [Research roadmap, section 9.3](../../../ref/morphotu-research-roadmap.md)
- [Original OTU-Former design](2026-03-29-otuformer-design.md)
- [ArcFace and Sub-center ArcFace](https://arxiv.org/html/1801.07698)
- [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
- [SoftTriple: multiple centers and center regularization](https://arxiv.org/html/1909.05235)
- [Hunt et al.: insect morphological traits and metric-learning losses](https://pmc.ncbi.nlm.nih.gov/articles/PMC12162172/)
