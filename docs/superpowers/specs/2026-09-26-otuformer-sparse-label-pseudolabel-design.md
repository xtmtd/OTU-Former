# OTU-Former Sparse-Label Fine-Tuning Design

Date: 2026-09-26
Status: Implemented in v0.9.0; this document remains the design of record
Target: Future training extension

## 1. Summary

OTU-Former's sparse-label case is an image-level, open-set problem: an expert
CSV labels a small subset of images, while the same image root contains more
images of known species and images of species with no expert label. Pretraining
remains unchanged. Raw CLS remains the primary downstream representation.

This design adds two independent, opt-in capabilities:

1. a conventional long-tail correction selected by one fine-tune option;
2. one automatic, precision-oriented known-class pseudo-label round.

The ordinary CSV fine-tune path remains the default and keeps its current
behavior.

## 2. Scope

### 2.1 Long-tail strategy

Fine-tuning gains one option:

```text
--long-tail none|cb-drw
```

`none` is the default. `cb-drw` applies Class-Balanced Deferred Reweighting to
the classification term of the three ArcFace-family losses. Its initial
constants are implementation defaults rather than user-facing parameters:

```text
beta = 0.99
max class weight = 3.0
start = 50% of optimizer steps
ramp = 10% of optimizer steps
```

`supcon` with `--long-tail cb-drw` fails before output creation because SupCon
has no class-level cross-entropy term.

CB-DRW is a conventional comparison, not the primary answer to sparse expert
labels. Class frequencies come only from expert rows. Pseudo-label rows never
change the frequency estimate.

### 2.2 One automatic pseudo-label round

A first fine-tune run uses the existing required inputs:

```text
--train-data EXPERT.csv
--input-images-dir ROOT
--checkpoint SSL.pth
```

`EXPERT.csv` contains `image,label`. The biological image tree under `ROOT`
must contain only images from the same comparable marker as the expert images;
new pseudo output must be outside and non-overlapping with `ROOT`. Dorsal,
ventral,
lateral, whole-body, and anatomical-part images are separate marker-specific
runs. This is a user data prerequisite; the program cannot infer it.

A second run adds the first fine-tune checkpoint:

```text
otuformer finetune \
  --train-data EXPERT.csv \
  --input-images-dir ROOT \
  --pseudo-label-from RUN1/finetune_latest.pth \
  --out-dir RUN2
```

The first checkpoint records the original SSL checkpoint's resolved path and
file SHA-256. Finetune#2 reuses that path automatically. If the SSL file moved,
the user supplies its new location through the existing `--checkpoint`; its
file SHA-256 must still match. No separate candidate-list path is required.

This activates the complete automatic flow:

1. validate that `FINETUNE1.pth` was trained from the same expert CSV and SSL
   checkpoint;
2. enumerate supported image files recursively under `ROOT` in canonical
   reference order;
3. subtract expert image references to obtain the unlabeled candidate pool;
4. extract raw CLS for expert seeds and candidates from `FINETUNE1.pth`;
5. apply the fixed precision-oriented decision rule below;
6. write a diagnostic CSV containing every candidate and its decision;
7. initialize finetune#2 from the same original SSL checkpoint;
8. train directly on expert rows plus accepted known-class pseudo-label rows.

The two commands are deliberately separate so the user can inspect finetune#1
training diagnostics and downstream checkpoint quality before selecting it with
`--pseudo-label-from`. Use `finetune_latest.pth` after a completed run when no
external held-out comparison selects another completed epoch checkpoint; the
current trainer has no automatic best-checkpoint concept. This is checkpoint
quality control, not a requirement to review or edit individual pseudo-labels.
`otuformer annotate` remains a separate downstream cluster-correction command
and is not part of fine-tuning.

Only one pseudo-label round is supported. A checkpoint produced by finetune#2
cannot be supplied as another pseudo source.

## 3. Public Interface

### 3.1 New fine-tune options

```text
--long-tail none|cb-drw
--pseudo-label-from FINETUNE1.pth
--pseudo-similarity-floor FLOAT
--pseudo-min-gap FLOAT
--pseudo-neighbors INT
```

Defaults:

```text
long-tail = none
pseudo-similarity-floor = 0.75
pseudo-min-gap = 0.10
pseudo-neighbors = 15
pseudo-cap-multiplier = 3
pseudo-absolute-cap = 50
```

`--pseudo-label-from` identifies the finetune#1 model used only to generate
pseudo-labels; finetune#2 still initializes from the matching original SSL
checkpoint. It is an input identity, not a tunable hyperparameter.

The five pseudo-rule options directly control feature-space acceptance and may
need dataset-specific adjustment. Their CLI help must state both the calculation
and the strictness direction:

```text
--pseudo-similarity-floor FLOAT
    Minimum winning known-class cosine score. The score is the mean cosine
    similarity between the candidate's L2-normalized raw CLS and its three
    nearest expert raw-CLS seeds in that class. Higher rejects more; lower
    accepts more. This is not a softmax probability. [default: 0.75]

--pseudo-min-gap FLOAT
    Minimum winning-score minus runner-up-score gap. Higher rejects more
    ambiguous candidates; lower accepts more. [default: 0.10]

--pseudo-neighbors INT
    Neighbor count k for the mutual-kNN check. Smaller is usually more local
    and stricter; larger is usually broader and more permissive. The effect is
    data-dependent and not guaranteed to be monotonic. [default: 15]

--pseudo-cap-multiplier INT
    Per-class acceptance cap multiplier: a class accepts at most
    min(multiplier * expert_seed_count, absolute_cap) rows after the
    floor/gap/mutual-kNN rules. Must be >= 1. [default: 3]

--pseudo-absolute-cap INT
    Absolute per-class acceptance cap, applied after the multiplier. Must be
    >= 1. [default: 50]
```

All five are optional; defaults provide an empirical, uncalibrated starting
point. Validate finite `similarity_floor` in `[-1, 1]`, finite `min_gap` in
`[0, 2]`, and integers `neighbors >= 1`, `cap_multiplier >= 1`, and
`absolute_cap >= 1` before output creation. `k` is exposed because neighborhood
scale depends on expert-seed density and within-class variation, not because it
is more important than floor or gap.

The `0.75` floor is an empirical, uncalibrated starting point. The external
exploratory summaries that motivated it are not repository artifacts and are
therefore not normative evidence for this implementation. It is not claimed to
be conservative, optimal, a confidence probability, or a precision guarantee.
Raw-CLS anisotropy and nearest-seed selection can make different known or
unknown species score highly, so gap, asymmetric mutual-kNN, diagnostics, and
offline held-out evaluation remain necessary.

Record the effective floor plus seed-to-class and candidate score distributions
in diagnostics so users can see where it lies. For classes with at least four
seeds, leave one training seed out and record the remaining-seed within-class
top-three score. Compute its runner-up exactly like the candidate rule: other
eligible classes use top-three mean, while one/two-seed classes use maximum
similarity. Name the fields `training_seed_loo_within_top3_*`,
`training_seed_loo_runner_up_*`, and `training_seed_loo_gap_*`, and label them as
optimistic training-sample diagnostics: ArcFace has already pulled their classes
together and pushed known classes apart. They are useful only beside candidate
top1/gap quantiles for anomaly detection, never as threshold calibration or
evidence about unknown species.

No public options are added for beta, class-weight cap, DRW timing, pseudo row
weight, calibration status, configuration-difference
approval, feature-cache files, SSL agreement, flip agreement, or statistical
reporting. These are either fixed implementation details or deferred research
ideas.

### 3.2 Existing options

Existing fine-tune parameters keep their current meaning. Every finetune CLI
parameter is classified as `experiment`, `operational`, or `control`; a test
fails when a future parameter has no classification.

The CLI-to-identity mapping is exact:

```text
experiment CLI                 resolved identity key(s)
model_name                     model_name
metric_embed_dim               metric_embed_dim, embedding_head
finetune_epochs                finetune_epochs
finetune_lr                    finetune_lr
metric_head_lr                 effective_metric_head_lr
weight_decay                   weight_decay
freeze_ratio                   freeze_ratio
loss                           loss
subcenters                     subcenters
compact_weight                 compact_weight
supcon_temperature             supcon_temperature
augmentation                   augmentation_profile, augmentation_config
orientation_policy             orientation_policy
batch_size                     batch_size
seed                           seed
long_tail                      long_tail

operational CLI                no cross-run identity key
out_dir, device, cpus, num_workers, trace_batch_ids,
log_every_n_steps, save_every_epochs, keep_last_checkpoints,
visualize_data, extract_size, metrics_sample_size, umap_n_neighbors,
umap_min_dist, umap_metric, visualize_class_number,
disable_embedding_metrics, overwrite

control CLI                    validated/recorded by its own flow
train_data                     train_manifest_sha256
input_images_dir               canonical row references; absolute root is not identity
checkpoint                     ssl_initialization_checkpoint_sha256
resume                         finetune#2 checkpoint authority
pseudo_label_from              pseudo_source_checkpoint_sha256
pseudo_similarity_floor        finetune#2 pseudo rule
pseudo_min_gap                 finetune#2 pseudo rule
pseudo_neighbors               finetune#2 pseudo rule
pseudo_cap_multiplier          finetune#2 pseudo rule
pseudo_absolute_cap            finetune#2 pseudo rule
```

Derived identity keys without a direct CLI parameter are `image_size`,
`class_labels`, `optimizer_name`, `optimizer_groups`, `arcface_scale`,
`arcface_margin`, and `compact_cap`. The initial implementation records fixed
`optimizer_name=adamw`, `arcface_scale=64.0`, `arcface_margin=0.5`, and
`compact_cap=0.5`; it has no fine-tune scheduler, warmup, AMP, or
gradient-clipping option. Applicable loss settings use resolved values;
`augmentation_config` is the parsed dictionary, not a path; and `class_labels`
is the sorted classifier mapping. `num_workers` may change between rounds;
matching `seed` states a common policy but not paired batches or bitwise
determinism.

`train_manifest_sha256` retains the existing v0.8.0 definition: hash sorted
`[canonical_ref, label]` rows as compact UTF-8 JSON (`ensure_ascii=False`,
separators `(',', ':')`), preserving duplicate rows. References use the existing
lexical `normpath(...).as_posix()` canonicalization relative to `ROOT`; this
field is deliberately **not** changed to NFC because doing so would change
existing checkpoint identity for NFD paths. Manifest labels continue to use
`MetricDataset.label_names`, while classifier `class_labels` continue to derive
from `MetricDataset.class_to_idx`; pseudo provenance reuses those existing values
without independently re-reading or converting labels. CSV row order and newline
style do not affect identity when parsed rows are unchanged.

Existing ordinary validation remains authoritative: an absolute reference that
escapes `ROOT` raises `ValueError`, as it does today. Pseudo-source eligibility
is evaluated only after the ordinary manifest is valid; there is no null or
empty-string manifest fallback. New pseudo path identity may use NFC-normalized
real paths without redefining `train_manifest_sha256`.

New fine-tune checkpoints record resolved values that were previously absent
(for example the top-level `orientation_policy`) inside the existing `config`
dictionary; that dictionary remains the sole configuration authority, and these
are new keys of it rather than a second configuration record alongside it. In
pseudo mode, omitted experiment options inherit finetune#1 values and explicitly
supplied conflicts fail. CLI help for existing experiment options must note this
pseudo-mode inheritance even when it also displays the ordinary default. The
existing `--checkpoint` option is already optional; pseudo mode may omit it and
reuse the recorded SSL path, or use it only to locate a moved file with the same
SHA-256. Old checkpoints remain readable but cannot serve as pseudo sources
when required provenance or resolved settings are absent. On resume, the
finetune#2 checkpoint is authoritative.

## 4. Automatic Candidate Discovery

The automatic candidate pool is all supported image files recursively under
`ROOT` minus the expert images in `EXPERT.csv`.

Pseudo provenance and generation use one shared path-identity function. It
resolves every file target, requires the target to remain under resolved
`ROOT`, and uses nonzero `(st_dev, st_ino)` when available. If inode is zero or
unavailable, identity is Unicode-NFC `normcase(realpath)`. If a supposedly
shared inode is observed for different real paths with different bytes, inode
identity is marked unreliable for the scan and the **entire scan** switches to
`normalized-realpath`. The content comparison in this collision check only
detects unreliable metadata; content never changes ordinary path identity.
Summary records `file_identity_mode: inode | normalized-realpath` and
`inode_collision_count`. This keeps one identity mode for the whole scan:
different paths with identical bytes remain distinct files, while aliases of
the same resolved file are subtracted when the platform can identify them.

Directory traversal never follows directory symlinks. It records the count and
first ten normalized relative paths in `skipped_directory_symlinks`; README
states that pseudo-label `ROOT` must be a real directory tree. A file symlink is
permitted only when its target stays inside `ROOT`. Its portable canonical
reference is the normalized root-relative **link path**, not the target path;
resolved path/inode identity is only for same-scan subtraction and alias
detection. The same rule applies when a CSV explicitly names a file through an
in-root directory symlink such as `alias/image.jpg`: preserve that link-path
reference, and do not rewrite it to the target directory's path. Directory
symlinks are still not traversed automatically.

The program:

- rejects duplicate/conflicting expert path identities in pseudo mode, but
  expert files with distinct identities and identical bytes remain legal and
  are only counted as an optional diagnostic;
- enumerates supported images deterministically and ignores non-images;
- excludes any candidate path identity already used by an expert;
- rejects duplicate candidate aliases from scoring with
  `duplicate_file_identity` diagnostics rather than choosing one silently;
- assigns every pre-scan rejection, including duplicate identity, escape, and
  invalid-reference rows, to the `none` seed-count bin so all bins sum to the
  discovered candidate count;
- after feature scoring, for candidates with any expert similarity above
  `0.999`, hashes the candidate and every expert seed above that threshold;
  equal SHA-256 rejects the candidate as `expert_content_duplicate`, while
  unequal bytes record `possible_duplicate=true` without rejection. Summary
  records the near-duplicate candidate count/rate and exact-content rejection
  count. This rule is identical in inode and normalized-realpath modes and also
  catches copied expert files.

Ordinary finetune behavior is unchanged. Existing manifest validation runs
first: notably, an out-of-root absolute reference remains a hard error rather
than becoming a provenance warning. After a valid ordinary manifest is built,
the expert-only path-identity function performs a read-only pseudo-source
eligibility check; candidate discovery can never change that result. It prints
one Info line for both eligible and ineligible ordinary runs; an ineligible line
includes the concrete loss/provenance reason and records
`pseudo_source_eligible=false` plus that reason. Finetune#2 repeats the expert
check and rejects an ineligible source before creating output.

### 4.1 Output and input directory safety

The recommended layout separates biological data from run products:

```text
DATA_ROOT/marker_images/   # pass as ROOT
RUN_ROOT/finetune1/
RUN_ROOT/finetune2/
```

Do not use the repository root or a run tree as biological `ROOT`. Before
candidate extraction or `prepare_output_dir`, a new pseudo run resolves all
paths and rejects:

- any ancestor/descendant overlap between `out_dir` and training `ROOT`;
- any file-valued input located inside `out_dir`, including the pseudo-source
  checkpoint, resolved SSL checkpoint, `train_data`, `visualize_data`, and any
  future explicit CSV/config/weight/ONNX input;
- `overwrite` when it could remove a required input (the overlap and input-file
  checks are the primary protection).

Resume keeps the existing `resume`/`overwrite` prohibition and uses its existing
run directory. This release does not add a marker file to ordinary command
outputs, so ordinary pretrain, finetune, extract, CAM, and annotate output
contents remain unchanged.

Candidate traversal's signature recognition is a safety net for historical
artifacts, not a substitute for separating data and run trees. It recognizes
strong command-specific output signatures and skips matching subtrees with a
warning. It records the signature, count, and
first ten paths in summary:

```text
finetune: logs/finetune.log + finetune_latest.pth
pretrain: logs/pretrain.log + SSL_latest.pth
extract: logs/extract.log + embeddings.csv, metrics.csv, or umap.pdf
cam: logs/cam.log + cam_summary.csv or figures/
annotate: logs/annotate.log + annotation_summary.json, otu_table.csv,
          or UPGMA_tree_partitions_annotated.pdf
```

A directory containing only a weak signal such as `.pth`, `logs/`, or an
unrecognized `runs/` name is not silently removed from the candidate pool; it
emits a warning for review. Supported images under such an unrecognized subtree
remain candidates. This avoids treating ordinary user data directories as
OTU-Former outputs and is why `ROOT` must be a dedicated data tree. Skipped
output subtree counts and example paths enter summary, preventing recognized
UMAP, CAM, diagnostic, and checkpoint outputs from entering the candidate pool.

Multiple images from one biological individual are ordinary separate image
units. Other views of an expert individual can be accepted and may add little
independent information; diagnostics and later evaluation must not interpret
image acceptance as individual-level coverage or generalization.

## 5. Pseudo-Label Decision Rule

### 5.1 Representation and class score

Use raw CLS from finetune#1, never ArcFace logits, softmax probabilities, or
projector output. Extraction uses the existing `center-crop` evaluation
transform and the checkpoint's resolved image size.
The first implementation deliberately does not add another preprocessing
choice.

L2-normalize valid vectors for cosine scoring. A nonfinite or zero expert seed
fails generation. A nonfinite or zero candidate is rejected as
`invalid_feature`.

A class with at least three expert images is eligible to receive pseudo-labels.
Its score is the mean cosine similarity between the L2-normalized candidate raw
CLS and its three nearest L2-normalized expert raw-CLS seeds in that class. It
is a cosine class score in `[-1, 1]`, not a calibrated confidence probability.
A class with one or two seeds cannot win; its maximum seed similarity can still
be the runner-up and block an uncertain assignment. This deliberately gives a
small class a stronger blocking score than a top-three mean would, making the
rule more conservative near poorly sampled classes. An eligible class uses its
same top-three mean whether it is winner or runner-up.

Let `top1` be the highest eligible-class score and `runner_up` the highest score
from every other class. Accept the score only when both strict inequalities
hold:

```text
top1 > pseudo_similarity_floor
top1 - runner_up > pseudo_min_gap
```

No competitor means rejection. Equality means rejection.

### 5.2 Mutual k-nearest-neighbor filter

Use `pseudo_neighbors` in two searches:

- candidate side: nearest valid expert seeds across all classes;
- for each winning-class seed, seed side: all valid candidates plus valid expert
  seeds from **other classes**, excluding every seed from the query seed's own
  class.

For candidate `x` with winning class `w`, require the **same** seed to satisfy
both directions:

```text
pass_mutual(x) iff there exists s in seeds(w) such that
    s is in N_k_candidate(x)
    and x is in N_k_seed(s)
```

This is an **asymmetric, own-class-excluded mutual-kNN** rule; the two searches
use different pools, and two different winning-class seeds cannot satisfy one
side each. Excluding same-class training seeds from `N_k_seed(s)` prevents
tightly compressed classes with six or more seeds from occupying all default
top-five slots. Other-class seeds still block when known alternatives are
closer, but an unknown cluster closest to one known class may pass; open-set
defense therefore depends mainly on floor, gap, and offline evaluation.

Each seed can reciprocally recognize at most `k` candidates, so a class has at
most `k*n_c` mutual-kNN passes before the fixed class cap. The effective upper
bound is `min(k*n_c, 3*n_c, 50)`; for `k=1` or `2`, mutual-kNN is tighter than
the nominal cap. In a dense candidate pool, candidate items may occupy most
seed-side slots, weakening the blocking contribution of other-class seeds.

One/two-seed classes may appear in the candidate-side list or block through the
runner-up score, but they cannot win and do not provide a reciprocal seed.
Invalid-feature candidates enter neither search. If a pool has fewer than `k`
valid entries, use all of it. Similarity ties use canonical references as the
secondary key.

### 5.3 Per-class safety cap

After score and mutual-kNN filtering, accept at most:

```text
min(cap_multiplier * expert seed count, absolute_cap)
```

per class, defaulting to `min(3 * expert seed count, 50)`. Rank by winning score
descending, then canonical reference ascending. Both constants are exposed as
`--pseudo-cap-multiplier` and `--pseudo-absolute-cap`; the cap exists to bound a
single class's pseudo-feedback influence and is applied only to rows that
already passed floor, gap, and mutual-kNN.

After the cap, compute SHA-256 only for provisional accepted rows. For identical
content, retain the canonical-reference-first row and reject later rows as
`accepted_content_duplicate`. This makes inode and normalized-realpath scans
agree without hashing the whole candidate pool; the bound is at most 50 hashes
per class. The first implementation does not refill a cap slot after this
precision-oriented rejection.

Candidates failing any rule are excluded from training. An empty accepted set
fails before output-directory creation; running finetune#2 with no feedback
would be indistinguishable from repeating finetune#1. Before exiting, stderr
must print the rejection counts, valid-candidate top1/gap p50/p90/p99, and the
pre-cap counts that would pass all other rules if only the floor, only the gap,
or only mutual-kNN were removed. It also reports candidate count, accepted
count, and acceptance rate by winning-class seed-count bin `none`, `3`, `4–5`,
`6–10`, and `>10`. `none` contains pre-scan rejections, invalid-feature
candidates, and all candidates that never obtain an eligible winner. This gives
actionable tuning information from the failed attempt without a new dry-run
option or creating `out_dir`; a later retry still re-extracts features. The
same summary is written to `pseudo_summary.json` on successful runs.

### 5.4 Library boundary and scale

`generate_pseudo_rows` returns accepted rows, all diagnostics, and summary even
when acceptance is empty; it never prints, creates directories, or writes files.
It uses a pure in-memory extraction helper that returns `(canonical_ref,
raw_cls_vector)` pairs after the existing model loading, evaluation transform,
and batched forward pass. It does not call the high-level extract command,
require a temporary `extract_csv` input file, or depend on extraction order;
results align by canonical reference. The empty-candidate failure path must
leave both `ROOT` and `out_dir` unchanged.

The CLI prints empty-set diagnostics to stderr and exits. Invalid source
checkpoints, identity conflicts, extraction failures, an empty candidate pool,
and invalid expert seed features raise one lightweight
`PseudoLabelError(ValueError)`.

Internal extraction adopts the extract **CLI** runtime defaults `device=auto`,
`batch_size=32`, and `num_workers=4`; the lower-level library currently defaults
`num_workers=0`. These are runtime settings, not new public
pseudo-rule options. Candidate–seed similarities serve class top-three scoring
and both neighbor searches. A dense `100,000 x 3,000` float32 matrix is about
1.2 GB, so implementation uses fixed-size blockwise exact aggregation/top-k and
does not expose another tuning parameter or change deterministic ordering.

## 6. Training and Weighting

### 6.1 Initialization and epochs

Finetune#2 must initialize from the same original SSL checkpoint as finetune#1,
not from finetune#1. This reduces direct self-reinforcement of the first model's
errors.

`--finetune-epochs` retains its current meaning: each run traverses its own
training rows for the requested epochs. Finetune#2 usually has more optimizer
steps because it has more rows. Record the row count, batches per epoch, epochs,
and completed optimizer steps; do not silently match step budgets.

Pseudo rows use source weight one in the initial implementation. The
precision-oriented acceptance rule, not another training-weight parameter,
controls their influence. Classes with only one or two expert seeds cannot
receive pseudo-labels; this intentionally protects the least-supported classes
from self-training errors but means pseudo feedback mainly helps the middle of
the tail and may increase the relative imbalance of the extreme tail. Summary
output records the number of ineligible classes and their share of expert
seeds.

Existing `--visualize-data`, `metrics_sample_size`, UMAP, and embedding-metric
behavior remain available for arbitrary diagnostic CSVs, including pseudo-row
CSVs when a user explicitly wants to inspect them. For RUN1/RUN2 training
comparisons, the recommended diagnostic CSV is the expert true-label CSV so
pseudo-label selection does not make the comparison look artificially better.
The formal RUN1/RUN2 comparison should use labeled data under `HELDOUT_ROOT`
outside training `ROOT`; training metrics and UMAP are diagnostic only.

Pseudo feedback and its finetune#1 source support only ArcFace-family losses,
and both runs use the same loss, applicable settings, and long-tail strategy.
Thus a `none` finetune#1 cannot start a `cb-drw` finetune#2; evaluating
`pseudo + cb-drw` requires a finetune#1 already trained with `cb-drw`. SupCon
pseudo sources are rejected before output creation. Supporting a SupCon feature
source would reintroduce loss-change controls and matched baselines, so it is
deferred.

### 6.2 CB-DRW math

For expert count `n_c`:

```text
raw_w_c = (1 - beta) / (1 - beta ** n_c)
```

Normalize class weights so their mean over expert training rows is one, clip
each target to `3.0`, and linearly interpolate from one using the fixed DRW
schedule. Pseudo rows receive their assigned class's current weight but do not
change expert counts. Thus CB-DRW in pseudo mode remains an expert-frequency
correction, only an approximate description of balance after pseudo rows alter
the actual training distribution.

Let `t` be the zero-based count of completed optimizer steps read immediately
before the next step. With the current `drop_last=False`, define
`T=len(training_loader)*finetune_epochs`, `start=floor(0.5*T)`, and
`ramp=floor(0.1*T)`. For `ramp>0`,
`lambda(t)=clamp((t-start)/ramp,0,1)`; for `ramp=0`, switch immediately from
zero before `start` to one at and after `start`. Thus `T=1` applies the target
weight from the first step and is no longer meaningfully deferred. Resume
restores `t` and the original `T`.

At optimizer step `t`, define `r_bar(t)` over **training rows only**: expert plus
accepted known-pseudo rows. Rejected diagnostic rows are never included.

```text
L(t) = sum_i(class_weight_t(label_i) * CE_i)
       / (actual_batch_size * r_bar(t))
```

For the two-class expert-count example `n=(10,128)`, `beta=0.99`, and cap `3.0`,
clipping the normalized target before the manifest denominator gives
`r_bar≈0.845792` and a final tail-row coefficient `3/r_bar≈3.546971`. The cap is
therefore on the target class weight, not the final normalized coefficient.

`beta=0.99` intentionally saturates effective counts near 100 and is paired with
the fixed cap to provide a moderate conventional baseline, not an exact
reproduction of every class-balanced-loss paper setting.

Compact Sub-center regularization is added after this reduction and is never
class-weighted. When `--long-tail none`, the existing scalar loss path is used
unchanged. The first implementation does not add DDP or gradient accumulation
semantics.

## 7. Outputs and Provenance

Pseudo mode writes inside finetune#2's output directory:

```text
pseudo_labels.csv
pseudo_summary.json
```

`pseudo_labels.csv` contains one row per candidate with diagnostic fields such
as:

```text
image
accepted
pseudo_label
top1_score
max_seed_similarity
runner_up_class
runner_up_score
score_gap
candidate_neighbor_refs
mutual_neighbor
possible_duplicate
class_cap_rank
rejection_reasons
```

These fields support later diagnosis; users do not edit the file to continue
training. The run log carries only progress lines (scanning, extraction, scoring) and
a per-class table of expert seeds plus added pseudo labels; the full
per-candidate diagnostics stay in `pseudo_labels.csv` and never enter the
log. `pseudo_summary.json` records rejection counts, explicitly optimistic
`training_seed_loo_*` and candidate score quantiles, floor/gap/mutual-kNN
counterfactual pass counts, `none/3/4–5/6–10/>10` seed-count-bin acceptance
rates, eligible/ineligible class and seed counts, thresholds, `k`, path-identity
mode, inode collision count, skipped directory symlinks/output subtrees,
near-duplicate counts/rates, `expert_content_duplicate` and
`accepted_content_duplicate` counts, cap constants, preprocessing, and content
hashes.

Every new ordinary fine-tune checkpoint records `pseudo_round=0` and attempts
to record:

- canonical expert image/label hash;
- source SSL checkpoint resolved path and **file SHA-256**;
- resolved training configuration needed for cross-run equality;
- `pseudo_source_eligible=true` only for an ArcFace-family run with complete,
  valid provenance; otherwise `false` plus a reason.

The existing checkpoint `config` is authoritative for both resume and
finetune#1-to-finetune#2 comparison; no second configuration snapshot is
created. Metadata additions are backward-readable and do not change ordinary
training behavior.

Finetune#2 checkpoints additionally record additive training/resume metadata:

- `pseudo_round=1` and the pseudo-source checkpoint file SHA-256;
- the exact accepted pseudo rows plus candidate-pool and accepted-row hashes;
- pseudo rule constants and preprocessing;
- expert/pseudo counts by class;
- long-tail strategy and optimizer-step schedule.

The accepted rows are bounded by the fixed class cap and are authoritative for
resume. They do not alter `model_state_dict`, architecture keys, embedding-head
metadata, or embedding dimension. The read-only `extract`, `export`, and `cam`
consumers ignore these additive fields and continue through the shared
checkpoint architecture/weight resolver.

Hashes detect accidental input drift, not malicious coordinated edits. Their
roles are explicit:

```text
hard gate for a new pseudo run:
  expert manifest, SSL checkpoint bytes, pseudo-source checkpoint bytes,
  experiment identity/class order, preprocessing, pseudo rule constants,
  pseudo_round

hard gate on finetune#2 resume:
  expert rows, checkpointed accepted rows, SSL identity, experiment identity,
  class order, preprocessing

warning-only:
  changed absolute ROOT with unchanged required references,
  changes limited to non-accepted candidates,
  missing or changed pseudo_labels.csv,
  weak output-directory signals

record-only:
  raw pre-L2 feature-byte hash, candidate-pool hash,
  current pseudo_labels.csv hash, skipped-path counts,
  near-duplicate diagnostics
```

Resume takes rule constants and accepted rows from its checkpoint; users do not
resupply the pseudo rule options. Raw feature hashes can vary with device
numerics and never gate resume or regeneration.

The resolved image-root path is recorded. A changed root warns but is allowed
when canonical expert and accepted-row references still match, because the
absolute path is not experiment identity and does not hash image contents.
Finetune#2 resume uses accepted rows stored in its own checkpoint and does not
require the original finetune#1 file or `--pseudo-label-from`. If that option is
supplied again, its file hash must match. Resume never rewrites or clears
`pseudo_labels.csv`; a missing or mismatched diagnostic file only warns and logs
its current hash because checkpointed accepted rows are authoritative. Added or
removed non-accepted candidates also warn rather than block; missing or changed
expert/accepted rows still fail.

The pre-output validation guarantees in this design (pseudo rounds,
identity conflicts, input/output overlap) are stated for the public CLI.
Internal Python entry points such as `run_finetune` re-check the same
conditions as a backstop, but callers invoking them directly are responsible
for preflighting before creating or clearing an output directory.

## 8. Backward Compatibility

Without `--pseudo-label-from` and with `--long-tail none`:

- dataset loading is unchanged;
- loss behavior is unchanged;
- existing checkpoint readers remain compatible with additive metadata;
- resume behavior is unchanged;
- raw CLS extraction is unchanged;
- ordinary command output files and directory layouts remain unchanged. The
  pseudo scanner only recognizes existing command-specific output signatures;
  it does not require a new marker file.

`--visualize-data` retains its existing meaning and may point to any supported
CSV for explicit UMAP/embedding diagnostics. Pseudo mode does not silently
ignore it; the expert-only CSV is a comparison recommendation, not a forced
behavior.

`--long-tail cb-drw` and `--pseudo-label-from` are independent opt-ins. Passing
any pseudo threshold/neighbor option without `--pseudo-label-from` or a
finetune#2 resume is an error rather than a silently ignored setting. Pseudo mode
performs all validation before creating or clearing its output directory, so an
invalid or identity-conflicting round never clears a prior run first. A new
pseudo round may use `--overwrite`; the input-protection check rejects any
required input that lives inside `--out-dir`, so `--overwrite` can never remove
one (see Section 4.1).

## 9. Deferred Evaluation Guidance

Calibration grids, hidden-label precision bounds, held-out reports, SSL-space
agreement, flip stability, transferred thresholds, and matched-step studies may
be useful for research evaluation, but they are not implemented in this first
extension. They must not become required user steps for ordinary fine-tuning.

Recommended offline evaluation on a complete-label research dataset may report:

- first-round checkpoint quality on a held-out expert subset stored under a
  separate `HELDOUT_ROOT` outside training `ROOT`, using
  `otuformer extract --input-images-dir HELDOUT_ROOT --label-csv HELDOUT.csv`;
  training-set metrics are optimistic, and held-out images left inside training
  `ROOT` would become pseudo candidates and invalidate RUN1/RUN2 comparison;
- accepted pseudo-label precision and coverage by class and expert seed count;
- hidden-species false acceptance;
- per-class mutual-kNN candidate-union size;
- downstream balanced accuracy, per-class recall, raw-CLS retrieval, and OTU
  over-merging/over-splitting;
- epochs and optimizer steps for both runs.

Because multiple images or views may come from one individual, acceptance and
image-level intervals can be inflated by correlated views and do not imply
individual-level independence or information gain.

## 10. Required Self-Checks

- the default CSV path and training behavior remain unchanged: the existing
  out-of-root absolute-reference manifest test still hard-fails; a valid
  ordinary manifest is required before pseudo eligibility is evaluated;
  additive checkpoint metadata remains backward-readable;
- `--long-tail cb-drw` rejects SupCon before output creation;
- all-one class weights reproduce the ordinary ArcFace-family loss;
- rejected diagnostic rows never enter `r_bar(t)` or the training dataset;
- partial final batches use their actual size;
- DRW uses completed optimizer steps and restores the same value on resume;
- extracting `compact_penalty()` preserves
  `old_forward == unreduced.mean() + compact_penalty()` at the existing cap and
  weight;
- candidate discovery is deterministic; directory symlinks are never followed
  and are reported; explicit in-root file/directory symlink references keep
  their link-path canonical reference; an inode conflict switches the entire
  scan to normalized-realpath; both modes use path identity only;
- exact candidate copies of any expert above the similarity threshold are
  rejected by the separate gated SHA-256 rule, while duplicate expert content
  remains legal; post-cap accepted content duplicates keep only the
  canonical-reference-first row in both identity modes;
- finetune#1 startup and finetune#2 preflight use the same expert-only identity
  function and emit one-line Info eligibility before training;
- raw CLS is not projector output or pre-normalized;
- invalid seed features fail and invalid candidate features reject;
- floor/gap equality rejects, one eligible class without a competitor rejects,
  eligible runner-ups use top-three mean, small-class runner-ups use max,
  seed-side neighbors exclude the winning seed's own class, the exact same seed
  satisfies both mutual directions, a real candidate for a class with at least
  six compressed training seeds can still pass, invalid candidates enter no
  neighbor pool, and ties are deterministic;
- training-seed leave-one-out diagnostics require at least four class seeds,
  carry `training_seed_loo_*` optimistic naming, and are never calibration;
- changed expert CSV, SSL checkpoint file SHA-256, pseudo source,
  preprocessing, or inherited training configuration fails before output;
  candidate-pool changes during a new generation are recorded, while changes
  during resume only warn unless an expert or accepted row is affected;
- finetune#2 starts from the same SSL checkpoint and cannot start a third round;
- an empty accepted set prints rejection/quantile/counterfactual diagnostics,
  including mutual-kNN removal and the `none` seed-count bin, and fails without
  creating `out_dir` or any file under `ROOT`; invalid sources/extraction/seed
  features raise `PseudoLabelError`, while the library never prints or writes;
- pseudo preflight rejects output/ROOT and every file-input overlap before
  overwrite; candidate discovery skips strong command-specific output
  signatures, warns on weak `.pth`/`logs` signals, and ordinary commands gain no
  marker file;
- a held-out metric set outside training `ROOT` remains absent from the candidate
  pool; placing it inside `ROOT` is documented as contamination; expert labels
  are recommended for RUN1/RUN2 training diagnostics, while an explicit
  `--visualize-data` remains honored for arbitrary CSV metrics/UMAP;
- every finetune CLI parameter is classified as experiment, operational, or
  control and mapped to exact resolved identity keys; adding an unclassified
  parameter fails a test; the existing lexical manifest hash and independent
  dataset label/class-map values remain byte-for-byte stable across the
  refactor, while NFC is limited to new pseudo real-path identity;
- diagnostic CSV decisions and checkpointed accepted-row hashes agree on a new
  run; a missing or edited CSV only warns during resume.

## 11. Decision Record

- Pretraining and raw CLS remain unchanged.
- Public long-tail control is only `--long-tail none|cb-drw`.
- Public pseudo-label controls are `--pseudo-label-from` plus five important
  feature-rule parameters whose help states the strictness direction; there is
  no separate candidate-list input.
- Candidate images are discovered automatically from the image root.
- Pseudo-label generation and finetune#2 are one automatic flow with no manual
  review and no `annotate` dependency.
- The first release implements no calibration or report subcommands.
- Finetune#2 starts from the same SSL checkpoint, supports one round only, and
  preserves the ordinary fine-tune path by default.

## 12. References

- [OTU-Former Toolbox Design](2026-03-29-otuformer-design.md)
- [OTU-Former v0.8.0 Metric-Loss Design](2026-09-24-otuformer-metric-loss-v080-design.md)
- [Sparse-Label Pseudo-Feedback Implementation Plan](../plans/2026-09-26-otuformer-sparse-label-pseudolabel.md)
