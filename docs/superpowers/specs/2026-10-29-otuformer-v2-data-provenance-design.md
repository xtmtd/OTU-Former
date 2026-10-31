# OTU-Former v2 Data and Provenance Foundation — Design

Date: 2026-10-29
Status: Proposed — awaiting review and approval of this document.
Branch: `otuformer-v2`.
Target release: not assigned; “v2” names the development track, not an approved package-version bump.
Authorization: create and revise this design and its contract/link in the master
design only, including the operator-confirmed COCO and external-review revisions.
Permission to revise is not approval of the resulting design. No implementation plan, product
code, dependency installation, data conversion, analysis, README/command/Skill
update, version change, commit, push, or release is authorized by this documentation
step.

## 1. Intent and Roadmap Position

Start the shared foundation described by Sections 4.1 and 4.8 of the local
`ref/otuformer-v2-development-roadmap.md`: S0 boundary definition and the data-side
beginning of S1. This is not completion of S1, S2, or either whole roadmap module.
The local roadmap is discussion material under ignored `ref/`; this tracked design
must stand on its own without requiring that file in a distributed checkout.

The first usable chain is: select images, resolve stable identities, attach optional
metadata and COCO annotations, validate them, and connect the resulting records to
existing global embedding extraction and a machine-readable run record. Shared
readers/validators are importable by other commands; there is no separate `dataset`
command group or standalone check command.

Preserve the global morphological barcode workflow. Annotation ingestion does not
claim a detector, segmenter, keypoint predictor, panoptic model, morphological
measurement engine, anatomical homology, or improved biological performance.

Use this document as the shared reference for later ingestion, identity, annotation
validation and provenance integrations. Add another command's task-specific inputs
only when needed; reuse these contracts instead of inventing a parallel format or
pre-adding options to every command. Record each integration in the master design
with a link to the relevant sections here, documenting only task-specific additions
or explicitly reviewed deviations. This reuse convention does not approve the draft
or authorize implementation.

### 1.1 Coverage and deferred delivery

This table maps the roadmap requirements, not completed implementation. Later
stages require their own reviewed contracts and execution approval.

| Roadmap requirement | This foundation proposes | Deferred delivery |
|---|---|---|
| 4.1 Image identity, labels, metadata, relationships | Root-relative keys; optional string metadata and explicit relationship columns (§3). | Group-safe split and leakage audit: 4.2 / remaining S1; no inferred specimen or duplicate groups. |
| 4.1 Spatial annotations and missing states | Box, polygon/RLE, keypoint and panoptic reading, association and checks; standard availability states (§4–5). | Custom ignore protocols consumed by a task and geometry transforms: 4.3 / S1–S3 and the relevant S4–S5 task. Continuous measurements remain a separately scoped future module. |
| 4.8 Configuration and artifact identity | Effective global-feature settings, hashes, row mapping and status (§7). | Spatial layout/geometry provenance and export diagnostics: 4.3–4.5 / remaining S1–S2; sampled augmentation contracts: S3. |
| 4.8 Split, grouped and per-sample evaluation | Record the existing image-level metric protocol, evaluated rows and limitations. The row mapping is not a task prediction/error table. | Explicit group split/audit: 4.2 / remaining S1; grouped summaries and per-sample predictions/errors when a task is delivered in S4–S5. |
| 4.8 Task metrics, thresholds and test use | Record metrics actually used and explicitly absent contracts; no generic score (§7.2). | Each S4–S5 task specifies its own labels, head/adaptor, metrics, threshold-selection data and held-out-test rules before making evaluation claims. No universal registry is required now. |

## 2. Decisions and Approaches

### 2.1 Operator-confirmed boundaries

- CSV is optional. When supplied, only `image` is required; `label` and every other
  metadata column are optional. No additional ID or absolute-path column is required.
- Runtime absolute paths are resolved against an explicit image root. Dataset-relative
  image keys, raw references, and COCO IDs are distinct identities.
- Selection precedence for the new data path is CSV, then COCO `images`, then a
  recursive image-directory scan. Existing commands retain legacy behavior when
  the new path is not selected.
- CSV owns image-level labels and metadata; COCO owns spatial annotations. Species
  labels and part categories are not interchangeable.
- Missing annotations do not imply biological absence; preserve distinct availability
  states without inventing negative masks or trusted background (§4.5).
- COCO documents are centralized per dataset/task, not a required per-image JSON
  layout. Panoptic annotations additionally use per-image segment-ID PNG files.
- Ingest boxes, polygon/RLE instances, keypoints, and panoptic annotations in this
  foundation. Support means usable reading, association, decoding where needed,
  and validation, not task training or prediction.
- Conversion is external. Do not implement LabelMe, YOLO, VOC, or general format
  conversion, and do not add supervision merely for conversion.
- Continuous measurements and their custom schema are out of scope. Retain standard
  fields such as `area` and `bbox`; derived biological measurements belong to a
  future module with its own definitions and calibration requirements.
- Keep existing examples read-only. The separately supplied larger dataset remains
  external and read-only; the operator prepares COCO annotations outside this
  project. Portable tests use small synthetic fixtures, not workstation paths.
- Self-intersections and adjacent repeated polygon vertices alone do not justify
  rejecting or repairing annotations. Preserve source geometry; continuous versus
  raster area differences are non-blocking diagnostics, not a fixed-percent failure
  or warning rule. External conversion logs remain external evidence.
- On the new data path, selection, attention-query training, evaluation and
  provenance consume the same parsed CSV records; legacy parsing stays unchanged.
- Shared instance/keypoint ingestion accepts nonnegative bbox extents and reports
  zero width/height without repair; individual tasks may require positive extents.
  Negative extents remain errors; panoptic pixel-consistency checks still apply.
- Internal metric diagnostics may be added at the point of decision/failure while
  preserving public return interfaces, score/availability rules and CSV empty-value
  behavior. Do not duplicate eligibility logic in the CLI.
- Design review, plan approval, implementation approval, and commit approval are
  separate gates.

### 2.2 Approaches considered

1. **A universal dataset/conversion subsystem.** Rejected: existing tools already
   convert formats, and the operator does not want a mostly empty dataset CLI.
2. **Make CSV or all metadata compulsory COCO.** Rejected: image-only SSL/extraction
   must remain simple; specimen metadata is not an object-category table.
3. **Thin CSV + task-specific COCO ingestion and shared checks.** Selected: retain
   source records and existing global extraction, add only the common data and
   provenance contracts demonstrated by real consumers.

The detection project's thin CLI/service/adapter layering, explicit source/output
boundaries, and reports are useful references, not runtime dependencies. Do not
copy automatic bbox-convention guessing, silent category renumbering, lossy
polygon round-trips, or broad exception-driven converter fallbacks.

### 2.3 Author proposals subject to design review

Apart from the operator-confirmed boundaries above, Sections 1.1 and 3–9 make the
draft concrete. In particular, optional relationship-column conventions, CLI option
names, the mask dependency, artifact schemas, and remaining validation policies
are author proposals. Approving this document would approve those proposals;
implementation remains separately gated.

## 3. Inputs, Selection, and Identity

### 3.1 Selection and ordering

The shared reader accepts `images_dir`, an optional manifest CSV, and explicit
optional instance, keypoint, and panoptic sources. It never scans unrelated
directories or downloads data.

| Inputs | Selected images and order |
|---|---|
| CSV supplied | Exactly its rows, in CSV order; COCO only attaches annotations. |
| No CSV, one or more COCO sources | Union of their resolved image keys; sort keys lexically. |
| Neither | Supported images recursively under `images_dir`, sorted by relative key. |

Multiple COCO sources need not cover identical images. Merge by resolved image
key, not by source-local numeric ID or array position. Conflicting dimensions for
one image fail. A CSV row absent from COCO is still a valid image-only record.
COCO-only images do not expand a CSV-selected set. Report selected counts and
unused source counts separately.

### 3.2 Paths and internal keys

Resolve an explicit relative path against `images_dir`; accept absolute image
references only when they resolve inside that root. For legacy basename CSVs,
prefer an exact root-relative file, otherwise allow a unique recursive basename
match. Ambiguity is an error, never “first match wins.” COCO `file_name` is resolved
as an explicit root-relative path, without the CSV basename fallback. No URL fetch.

The internal `image_key` is the normalized POSIX path relative to `images_dir` of
the matched image, not merely its basename, CSV row number, or absolute location.
Normalize redundant `.` and `..` components only after checking resolution and
containment; never strip leading characters with a character-set `lstrip`.
It is stable when the dataset root moves, not when images are renamed. Preserve
case and Unicode spelling; do not case-fold IDs. Reject resolved root escapes,
including symlink escapes. Warn about distinct keys resolving to the same physical
file, but do not silently merge metadata or drop rows. Duplicate image keys are
errors; physical-file aliases with distinct keys are warnings, not automatic
specimen links. Local inode/realpath checks are diagnostics, not portable dataset
IDs. Keep an internal symlink's normalized root-relative spelling as its key and
the target's resolved location as its access path.

Two explicit relative paths such as `a/001.png` and `b/001.png` remain distinct.
Raw CSV references and original COCO image IDs are retained alongside the key.
A COCO identity is `(source, image_id)`; IDs are not assumed global across files.
Do not change the existing fine-tune/pseudo-feedback manifest-hash definition.

### 3.3 Metadata and missing values

Read CSV values without automatically treating literal `NA`, `N/A`, `unknown`,
`0`, or numeric-looking identifiers as missing or numbers. An empty cell is
missing; whitespace-only `image` values are invalid. Preserve raw metadata strings
and source column names. Record absent columns separately from present empty cells.
Duplicate columns, malformed CSV row widths, header-only selectors, or rows
resolving to the same image key fail in the new path. An empty final selection
from any input form fails before model loading. Empty optional labels remain
unlabeled, never a shared empty-string class.

A label is not required by the reader. A task declares whether it requires labels
or spatial annotations and fails with an actionable message when requirements are
not met. Unlabeled global extraction is valid; attention-query training and
supervised fine-tuning keep their own label requirements. Never synthesize a
species label or infer it from directories or COCO part categories.

`specimen_id`, `individual_id`, `view`, `marker`, `sex`, `stage`, `collection`,
`camera`, `scale`, and `source` are optional metadata, not a mandatory ontology.
Do not infer specimen relations, physical scale, or ecological sample sets from
names. An explicit legacy `sample` column keeps its existing meaning;
directory-inferred `sample` remains a legacy output convention, not new biological
evidence. Unknown custom columns are retained but confer no automatic task semantics.

Parse the selector once on the new data path and pass the resolved image records
and raw label strings to every consumer, including attention-query training,
metric/UMAP alignment and provenance. Do not re-read that CSV with pandas default
NA/type inference or repeat an independent path lookup in those consumers. Numeric
class indices may be derived from the same strings using the existing class-order
policy; the derived mapping and recorded class values must agree. A task's label
requirement is unchanged, not its legacy parsing implementation. Plain legacy
execution remains on its original reader.

### 3.4 Explicit relationships without a grouping engine

Proposed optional conventions are `specimen_id`, `individual_id`, `publication_id`,
`plate_id`, and `duplicate_group_id`. Equal nonempty values in the same column of
one manifest express a declared relationship, not a verified biological identity.
Keep columns independent: a plate or duplicate group is not a specimen, and
`specimen_id` and `individual_id` are not automatic aliases. Empty cells establish
no relationship; literal tokens retain §3.3 semantics, not an inferred unknown
sentinel. Values are scoped to that manifest/column, not global IDs across datasets;
producers must namespace colliding local values, including plate numbers reused
by different publications.

For example, two views may share `specimen_id=001`, several specimens may share
`plate_id=paperA:plate01`, and independently declared near duplicates may share
`duplicate_group_id=dup01`. Preserve those strings in row metadata and its digest.
Do not infer membership from filenames, hashes or labels, compute a transitive
union across columns, merge rows, or automatically split/evaluate by these fields.
A later group consumer must declare its chosen columns, missing-value policy and
leakage rules. No relationship table, new required field or CLI option is added.

Future continuous measurements may attach through a separate sidecar keyed by
`image_key`, adding `(source_id, image_id, annotation_id)` where a measurement refers
to an instance rather than an image. This is an extension direction, not a sidecar
schema or artifact implemented here: units, calibration, quantity definitions and
compatibility must be reviewed with that module. Do not add placeholder measurement
columns or treat source COCO `area` as a physical measurement. There is no promise
that future schemas will never evolve.

## 4. COCO Sources and Supported Tasks

### 4.1 Organization and preserved fields

Illustrative input layout, not a command or a required file naming convention:

```text
dataset/
  images/
    a/001.png
    b/001.png
  figs.csv                         # optional
  annotations/
    instances.json                 # optional task source
    keypoints.json                 # optional task source
    panoptic.json                  # optional task source
    panoptic_masks/                 # required if panoptic source is supplied
      image_001.png
```

Source IDs are fixed roles (`instances`, `keypoints`, `panoptic`) in this first
version, with at most one file per role. A combined instance/keypoint COCO document
may be supplied for both roles; preserve the role-qualified references rather than
silently dropping either task. Each source requires `images`, `annotations`, and
`categories` arrays. Image records require integer `id`, positive integer width
and height, and nonempty `file_name`; category records require integer `id` and
nonempty `name`. Boolean values are not valid integer IDs or dimensions. IDs are
unique within their corresponding source table, not necessarily contiguous or
one-based. Multiple image records in one source resolving to the same image key
fail even if their numeric IDs differ. Instance/keypoint objects require `id`, `image_id`, `category_id`,
`bbox`, `area`, and `iscrowd`; keypoint objects additionally require their point
fields. Panoptic uses its per-image records instead of mandatory object IDs.

Use explicit task sources, not “pick the first JSON in a directory.” Preserve
`info`, `licenses`, image/category/annotation IDs, source category ordering,
`supercategory`, and extension fields in retained source data. No category filtering
or automatic renumbering. Categories are source-local; equally numbered categories
in different sources are not automatically equal. No compulsory synthetic species
category when only part annotations are supplied.

Index `images`, `categories`, and annotations while retaining the original JSON
payload. Source hashes and provenance identify that payload; do not rewrite it.
Unknown extension fields are preserved but not interpreted as validated semantics.
Do not require dataset-specific descriptive fields such as URLs or license entries
merely to ingest an otherwise usable image annotation.

### 4.2 Boxes and instances

COCO boxes use absolute original-image `[x, y, width, height]` values. Do not guess
xyxy or accept normalized coordinates as if they were standard COCO. Require finite
values, nonnegative box extents, and containment within the image boundary. Shared
instance/keypoint ingestion records zero width or height as non-blocking information,
not a universal format failure. Do not expand, drop or repair such boxes. A task
that needs a nonempty crop/region must explicitly require positive extents and fail
when that requirement is unmet; global extraction does not use boxes as crops.
Negative extents remain errors. Nonempty panoptic segments retain their pixel-derived
area/bbox requirements (§4.4–5), not this permissive instance/keypoint rule.

Box-only objects are valid and do not acquire a segmentation. If segmentation is
provided, accept multipart polygons, uncompressed RLE, and compressed RLE. Polygon
parts are flat even-length finite coordinate arrays with at least three coordinate
pairs. Do not require a simple polygon, distinct vertices or nonzero shoelace area:
self-intersections, an explicit closing vertex and adjacent repeated points are not
format errors by themselves. Use `pycocotools` rasterization and combine parts using
COCO mask-union semantics, not an invented hole ordering or custom fill/repair
algorithm. Validate bounds before rasterization so a library cannot silently clip
out-of-bounds source coordinates. Preserve float values, vertex order and repeats.
A successfully decoded but empty mask follows the availability/task policy below;
decode failures or an invalid output shape remain errors.

Use original-image boundaries `[0, width] x [0, height]` for polygon vertices and
box edges; this is distinct from discrete pixel indices. RLE size must equal
`[height, width]`, counts must be valid, and decoding must produce that shape.
Preserve `iscrowd`; crowd does not mean a verified absence or ordinary background.
A supported but empty segmentation is recorded as absent/empty coverage, not a
negative mask. Malformed supplied geometry is an error.

### 4.3 Keypoints

Read category-defined unique, nonempty point names and optional skeleton edges.
Do not assume human anatomy or a fixed number of points. Each annotation has `3 * K` values for the
category's K points and a consistent `num_keypoints`. Visibility must be integer
`0`, `1`, or `2`: unlabeled (`x=y=0`), labeled but invisible, or labeled and visible.
Count `v > 0` points, not only visible points, in `num_keypoints`.

Validate finite coordinates for labeled points in the original pixel-center domain
`0 <= x < width`, `0 <= y < height`; invisible coordinates remain annotations and
are not moved to the origin. Skeleton edges use COCO's one-based point indices;
validate them against K and preserve their source order. Segmentation is not
required merely because keypoints are present. Preserve point visibility, names,
and source annotations for downstream consumers without automatic augmentation.

### 4.4 Panoptic

A panoptic source is task JSON plus its explicitly supplied PNG root. Resolve each
per-image annotation's `file_name` as written against that root, not against
`images_dir`. In the §4.1 layout, `file_name="image_001.png"` uses root
`annotations/panoptic_masks/`. If instead the stored name is
`"panoptic_masks/image_001.png"`, supply root `annotations/`. The directory layout
alone does not determine whether the JSON name includes a prefix. Do not prepend
or strip a JSON-stem/directory prefix, auto-search alternate roots, or require a
prefix merely because a PNG directory shares the JSON's stem. Apply the same
root-containment checks, including symlink escapes, to mask paths.
Support RGB segment-ID PNGs with `id = R + 256*G + 256^2*B` using unsigned arithmetic.
Do not silently palette-convert or treat a category-color visualization as an ID map.

Match the map to its image using the source's `image_id`. Check actual dimensions,
unique nonzero segment IDs in that image's `segments_info`, category references,
category `isthing`, segment `iscrowd`, area/bbox, and PNG/JSON segment coverage.
Boolean-like flags accept numeric 0/1 only; segment IDs must fit unsigned 24 bits. Segment IDs are image-local, not global
annotation IDs. Nonzero PNG IDs missing from JSON, declared segments absent from
PNG, and duplicate per-image panoptic records fail. ID 0 is void/unknown, never an
invented foreground category or trusted background. Area and bbox checks use the
segment's actual pixels. Preserve overlapping-instance semantics in instance COCO;
do not turn instances into a mutually exclusive panoptic map.

### 4.5 Availability is not biological absence

Distinguish image absent from a source, image present with no annotations, empty
geometry, keypoint `v=0`, keypoint `v=1`, and panoptic void. None automatically means
“this anatomical part does not exist.” Standard COCO alone does not establish an
exhaustive part-annotation protocol. Report counts and availability, not fabricated
negative masks. `extract` consumes image pixels and optional labels; it does not
use spatial annotations as targets in this phase.

The minimal interpreted states are source coverage, empty/nonempty decoded geometry,
keypoint visibility and panoptic void, with `iscrowd` retained separately. None is
an alias for a general `ignore` flag or a license to treat unannotated pixels as
trusted background. Custom `ignore`, `unknown`, occlusion or region-protocol fields
remain unchanged extension data; report their interpretation as not implemented,
without filtering annotations or inventing an ignore mask. This foundation does
not claim a general custom ignore protocol. Before using such data as supervision,
a future task adapter must explicitly define and test field values, scope and
loss/metric exclusion rules; an unsupported required protocol must fail rather
than silently supervise or score ignored/unknown regions.

## 5. Shared Validation and Area Policy

The shared function returns selected records, indexed sources, and structured
issues; command adapters enforce task requirements. Validate before model creation,
weight downloads, or inference. It must be possible to run the reader from Python
without initializing a model or writing output. No separate CLI validation mode.

Check JSON syntax/types, finite numbers, required task fields, duplicate IDs,
dangling references, selected image readability and actual dimensions, geometry,
and panoptic maps. Strict JSON input rejects duplicate object keys and nonfinite
number literals rather than silently accepting an overwritten ID or NaN. Check
source-wide structural references and numeric payloads even when CSV selects a
subset; geometry/actual-image consistency and pixel decoding apply to the selected
set. Only the selected image/mask files need to exist locally. Record unselected
file/geometry checks as not performed, not passed; unselected content is not
advertised as fully checked.

| Issue | Proposed behavior |
|---|---|
| Invalid/ambiguous image path, duplicate identity, conflicting dimensions, corrupt selected image | Error; no inference or embedding outputs. |
| Duplicate source ID, dangling reference, invalid numeric payload, malformed coordinate/RLE structure, out-of-bounds geometry, failed decoding or unsupported task structure | Error; no silent fallback or dropped annotation. |
| Missing optional metadata/source coverage, no part annotations, unused valid categories | Information/warning with counts; not an automatic task failure. |
| A task's required labels/annotations absent | Task error, even though the image-only record itself is valid. |
| Zero-width/height instance or keypoint bbox with otherwise valid values/bounds | Non-blocking information; a task requiring a positive-area region rejects it explicitly. Negative extents remain errors. |
| Self-intersections or repeated adjacent vertices/zero-length edges alone | Not a format failure or repair request; preserve source vertices. No topology scanner is required. |
| Valid source `area` or subpixel bbox differs from polygon-rasterized values | Non-blocking information with source/computed values and method; no fixed-percent warning rule. |
| RLE or panoptic area/bbox disagrees with the actual mask beyond numerical tolerance | Error; retain source values and diagnostic recomputation. |

Area is an image-annotation quantity, not a physical measurement. Require supplied
areas to be finite and nonnegative. For polygons, use `pycocotools` COCO rasterization
for diagnostic mask area/bbox; do not substitute shoelace area or `bbox` area for
rasterized segmentation area. Retain source `area` without asserting its exporter
used a particular formula. Continuous geometry and rasterized pixel area need not
coincide, especially for thin or small regions; a difference alone establishes
neither invalid geometry nor annotation-quality failure. Shoelace-area cancellation
must not reject a self-intersecting polygon whose mask can be decoded.

Record source and rasterized area/bbox, absolute area difference and decoder/version
for selected polygon annotations as diagnostics. There is no polygon percentage
warning/error threshold, and no replacement threshold is proposed. Do not turn
routine rasterization differences into thousands of warnings. If topology counts
are supplied by an external audit, retain its counting unit and predicate in that
evidence; neither crossing counts nor affected-polygon counts are acceptance gates.
Do not add a custom topology detector merely to produce those counts.

RLE/panoptic mask area and pixel-derived bbox retain exact pixel semantics with the
proposed numeric tolerance `1e-6`; a box around a pixel region has extent
`max_index - min_index + 1`. Polygon bbox/vertices must obey §4.2 bounds, but do not
require a continuous source box to equal the pixel-derived bbox. Box-only `area`
has no mask-area consistency assertion. Zero/empty decoded masks are reported;
tasks needing a nonempty mask must reject them rather than converting them to
background. No correction, clipping, deduplication of vertices, source mutation,
or continuous-measurement output is performed. External converter clipping logs
are evidence of that external operation, not permission for ingestion to clip.

Issue records contain severity, a stable issue code, source reference/row or
annotation identifier, optional image key, and a message with relevant values.
Print a compact summary and actionable examples; serialize all issues when the
command has an approved output directory. An invalid request must not bypass
existing output protection merely to write a report. No unrequested
`--ignore-errors`, repair, auto-clamp, or tolerance options.

## 6. Proposed Integration and Dependencies

### 6.1 A real consumer without a new command group

**Approved revision (2026-10-30): one CSV, one option.** The original proposal
added a separate `--manifest PATH` next to `--label-csv`. That split was rejected
during implementation: a second CSV flag for the same file made two options mean
one concept and left "which CSV holds the metadata?" ambiguous. The delivered
contract is:

- `--label-csv` (existing option) is the single CSV selector for the shared path.
  Supplying it selects the new path on its own; one file carries the `image`
  selector, the label column and any further metadata.
- `--coco-instances`, `--coco-keypoints`, `--coco-panoptic` and
  `--panoptic-masks-dir` are added as proposed. `--panoptic-masks-dir` requires
  the panoptic JSON and vice versa.
- There is no `--manifest`. Legacy parsing is now reached only by supplying no CSV
  and no `--coco-*` option, i.e. plain directory enumeration.

The consequence to accept explicitly: a CSV can no longer mean legacy parsing.
Any `--label-csv` run uses strict string semantics. Task label requirements stay unchanged, but
all new-path consumers use the shared parsed records from §3.3 rather than the
legacy label/metric CSV loaders. On this path, the CSV selector also enables the
existing optional metrics/UMAP flow without requiring a duplicate label file, respecting
`--disable-umap`. COCO-only input does not automatically enable that visualization.
Evaluate supervised metrics only on nonempty labels, record included/excluded row
keys, and preserve image-only CSV UMAP behavior. Attention-query training on the new path requires all selected
training rows to have labels; partial-label filtering for that training is not
silently invented. Existing metric availability rules remain authoritative;
§7.2 permits additive internal diagnostics without changing those rules.

The Python reader's neither-CSV-nor-COCO branch provides directory enumeration;
no additional CLI opt-in flag for that branch is proposed. Plain directory-only
`extract` stays on the existing path.

These are additive options on an existing command, not a new command group. They
avoid an extra boolean mode selector, an annotation autodetection framework, and an
otherwise empty dataset CLI. `--label-csv` alone selects new-path parsing without a
COCO source, so no second flag is needed to express "one CSV, strict semantics".
Other commands' names such as `--train-data` retain their own task contracts. This
design does not rename them or add a selector alias. No new public options are added
to other commands in this phase. Reuse their existing path, logging, checkpoint,
and safety helpers
where semantics match; do not migrate all training/resume behavior at once.

Make the data path visible before extraction: parameter cards and logs identify
`legacy data path: existing semantics` (plain directory enumeration) versus
`new data path: strict CSV semantics` (any `--label-csv` or `--coco-*` option),
including the option that selected the path. The workflow Skill derives this card
notice from the option combination; it is explanatory text, not a new field in the
exported CLI schema or proposal payload. Discover the declared options through the
existing schema; do not change its contract or version to represent the derived
mode.

A CSV has exactly one meaning in the CLI after the approved revision above: it
selects the strict path. There is no CLI combination in which the same CSV is read
with legacy semantics, so a card must never say that adding a COCO option changes
parsing for the CLI. What still has two layers is the *API*: `read_dataset()` and
the record/validation layer are the shared implementation, while the Python
`extract_embeddings(extract_csv=...)` and `attention_train_csv` arguments keep their
existing legacy CSV loading for callers who do not pass records. Record the
effective data path and CSV parsing policy in new-path provenance (§7.2); do not add
a mode switch or change the existing `Command:` and requested `Parameters:` log
contracts.

Author proposal for the input seam: the command calls the shared reader once before
model creation and passes its validated, ordered records through a new optional
keyword-only records parameter to `extract_embeddings()`. Extraction and any
attention-query training construct their image/label dataset views from those same
records, preserving resolved paths, output IDs, raw labels and order. With records
supplied, `images_dir` is only the root/containment validation reference and must
match the reader's image root; it must not trigger another path lookup, directory
scan or basename fallback. Both PyTorch and ONNX extraction consume this input
without reopening the CSV or independently resolving its references. Reject combining the records input with `extract_csv` or
`attention_train_csv`; source CSV paths remain provenance, not execution inputs.
Legacy callers omit the records input and retain the existing path-based parameters
and loaders. The plan may choose the minimal dataset-view implementation, but not
replace this seam with another CSV parse or break existing callers.

Keep `extract_embeddings()` returning its existing DataFrame. Internal execution
may return extra resolved metadata to the command through a small internal result
object, but do not break callers or reconstruct “effective” settings from CLI
requests after inference. The shared data module belongs outside `cli/`, with
stdlib JSON/CSV indexing and focused task helpers; no plugin/registry framework.
Mask decoding is lazy and usable through the same reader from Python. Unit tests
for every supported task plus the extraction attachment path are required; a
format name with an unimplemented body does not satisfy support.

### 6.2 Dependencies

Use stdlib `json`/`csv`, existing NumPy and Pillow for indexing, image checks, and
panoptic RGB-ID decoding. Reuse `pycocotools` for polygon/RLE mask operations rather
than writing a codec. Author proposal: an optional `annotations` extra containing
`pycocotools`, imported only by mask-dependent paths. A supplied source requiring
that capability must fail early with installation guidance if it is unavailable;
never silently skip validation. Tests with the extra and without it are both needed.
Dependency versions and supported platforms require verification during approved
implementation; nothing is installed by this design step.

No dependency on detcli, supervision, or panopticapi is required for the selected
data operations. A small unsigned RGB-ID expression is not a replacement panoptic
metrics package. Task training, advanced augmentation, and COCO/PQ/OKS evaluation
are not part of this dependency decision.

## 7. Extraction Outputs and Minimal Provenance

### 7.1 Compatibility and additive artifacts

Without new options, keep current image selection, ordering, `id`, `sample`, vector
values/columns, pooling behavior, metrics, and checkpoint support. Do not route
legacy inputs through new strict checks that change accepted inputs or old hashes.
Existing log `Command:`/requested `Parameters:` contracts remain; in particular do
not disturb cluster logs parsed by annotate.

Keep the existing embedding column schema, while distinguishing each input path's
ID convention rather than claiming one universal legacy rule:

| Input path | Emitted `id` |
|---|---|
| Legacy directory input | Basename, retaining existing batch/`sample` behavior. |
| Legacy CSV input | CSV `image` reference after the existing reader; not forced to basename. |
| New path with CSV selector | Original `image` string from the shared reader. |
| New COCO-only path | Resolved `image_key`; this is an additive new-input contract. |

For example, CSV references `a/001.png` and `b/001.png` remain distinct output IDs;
there is no new global basename-uniqueness requirement. The new reader's literal
string preservation is explicit, not a claim that legacy pandas inference already
has those semantics. Do not add metadata or annotation columns among embedding
dimensions. Use the existing explicit-sample/first-directory behavior when producing
`sample`, record
its origin, and do not present inferred values as ecological evidence. Never emit
an ambiguous new-path ID mapping; fail before inference if IDs cannot identify rows.

Author-proposed additional artifacts under the existing approved `out_dir`:

| File | Contract |
|---|---|
| `data_records.jsonl` | One record per emitted embedding, in exactly that order; includes `schema_version=1`, a one-based `embedding_row`, `output_id`, `image_key`, raw selector reference, width/height, image byte SHA-256, optional metadata, and `(source_id, image_id)` references. Preserve column presence/empty values. |
| `data_validation.json` | `schema_version`, scope, status, counts, methods and applicable tolerances, source IDs, complete issues and per-annotation polygon raster diagnostics. Distinguish non-blocking information from warnings/errors; diagnostics alone do not fail validation. On validation failure there are no embedding/mapping outputs; the report and failed run record may accompany logs when output safety permits. |
| `run_provenance.json` | Versioned run identity, requested/effective settings, source identities, ordered selection digest, output descriptions/hashes, and final status. |

Large annotation payloads stay in their sources, not repeated per embedding row or
inserted into CSV. Original annotations and decoded access are available through
the shared Python reader; source IDs and hashes make their association explicit.
No image/mask copying or feature cache is introduced. Generated paths are checked
against all read-only input roots using existing output-safety conventions.

### 7.2 Required run record for the new data path

Only the new data path described in §6.1 is required to produce the additive
artifacts in §7.1. Legacy extraction does not acquire a run-record requirement
merely because this section defines one. Proposed `schema_version` is integer `1`.
Record:

- Run status (`running`, `completed`, `failed`, or `interrupted`), command, start/end
  timestamps with timezone, package version, Git commit/dirty hint or explicit
  unavailable status, and relevant dependency versions. Git provenance is not
  proof that dirty source bytes are reproducible.
- Effective data path, selector and path-selection trigger; when a CSV is used,
  record its strict parsing policy separately from the requested CSV option name.
- Requested options including omission/explicitness separately from resolved
  device/provider, input size, evaluation transform, token mode, projector/pooling/PCA
  policy, teacher/student or actual state-dict selector, backbone/register layout,
  and actual feature dimension. Record actual choices at their point of resolution,
  not by inference from defaults afterward.
- Distinct effective feature semantics: image preprocessing normalization (including
  mean/std); feature source layer/output and whether backbone output normalization
  has already run; and post-feature operations in execution order (pooling/projector,
  flattening, scaling/PCA and vector L2 normalization, as applicable). Record identity/
  no normalization explicitly when known. Do not conflate input normalization,
  backbone norm and vector normalization, or add a layer selector/normalization
  option. For an ONNX graph or legacy artifact that does not expose an internal
  layer/norm contract, record the actual output name/dimension plus an explicit
  unavailable reason for the unknown details; never infer them from a filename.
- Used checkpoint or ONNX file SHA-256, requested/resolved paths, recorded source
  configuration where available, and hashes of additional used pooling weights.
  Include ONNX external tensor files when used. Describe the existing PCA fitting
  policy (scope, dimensions, solver/seed when applicable); exporting a new reusable
  PCA model is not required here. Ignored checkpoint/device options in ONNX mode
  remain requests, not used inputs.
- CSV/COCO file byte hashes, original references, source-local category definitions,
  ordered selected image keys and metadata digest, and the declared selection rule.
  Hash selected images and panoptic PNGs by bytes; include their content digests in
  a separate ordered data-content digest. Record selected mask paths/hashes in the
  source section, so individual inputs can be checked as well as their aggregate.
  Report the hashing scope explicitly; the cost of reading/hash-checking selected
  files is part of this author proposal, not a previously approved default.
- Per-source identity, image dimensions, annotation types and availability,
  validation result, and a reference to per-row output mapping. Read each image/mask
  hash during validation and compare the bytes actually decoded at consumption
  with the recorded hash; detect input drift rather than relying on timestamps.
  Do not retain the whole image corpus in memory. A later mutation of the source
  file does not change which decoded bytes ran, but is outside a claim that the
  current on-disk dataset still matches the record.
- Metric names/values, label schema (source column, observed class values and
  missing-label handling), distance/normalization and evaluation protocol actually
  used, labeled/evaluated counts, sampled row keys and seeds, and unavailable metric
  reasons. Record any thresholds actually used and their selection data; otherwise
  state not applicable with a reason. State whether a held-out test was supplied
  and how it was used; existing image-level stratified CV is not a specimen-group
  split or held-out biological test. Group split, grouped summaries and per-sample
  task prediction/error tables are not supplied here, even if relationship metadata
  exists. No specimen-safe claim without explicit identities and a separately
  implemented split protocol; see §1.1 for later task evaluation contracts.
- Produced artifact paths, hashes, row counts and dimensions; hash finalized output
  files other than `run_provenance.json` itself. Report failures/interruptions without
  a completed status and do not hide a partially written embedding as a success.

Unavailable metric reasons require an additive internal diagnostic channel, not
changed metric definitions or an all-caller return-type migration. Generate a
structured value/status/reason at the evaluator's actual eligibility, nonfinite
result or caught-failure branch; keep existing public value/`None` return interfaces
and CSV empty-value behavior through compatibility adapters. New-path provenance
consumes those diagnostics from the same evaluation, not a second evaluation or a
CLI copy of CV/class-count logic. Command-level skips such as missing labels are
recorded by the command. Tests must distinguish insufficient classes/class counts,
ineligible k, nonfinite results and caught failures using stable reason codes and
available context, without inventing a cause when details are unavailable. This
permits a small shared internal change, not altered scores, folds or availability
rules in legacy callers, and does not introduce a metric registry.

Reference portable files by output-relative path; local absolute access paths may
be recorded as such, never used as portable identities. File SHA-256 establishes
content equality, not scientific truth, safety to deserialize arbitrary checkpoints,
or proof of an annotation protocol. No authentication/telemetry subsystem.

Absent split, spatial-feature, geometry-transform, task-head, or calibration records
must state `not_provided`/`not_applicable`/`unavailable` with a reason, not invented
IDs or generic scores. Use explicit `{status, reason}` objects for such missing
contracts instead of an ambiguous bare null. Hash logical ordered record lists
with SHA-256 over UTF-8 JSON (`sort_keys=True`, `ensure_ascii=False`, compact
separators, no NaN); exclude machine-local access roots and times. Selection/metadata
and data-content digests have separate names, versioned definitions, and ordered
row keys. Hash input files by their original bytes, not this JSON normalization.
The new schema does not rewrite legacy checkpoint payloads.
Future additions require explicit schema compatibility, not reuse of an old digest
name for a different meaning. Canonical serialization for logical digests is
separate from the presentation/atomic publication of JSON artifacts. The existing
general `utils.io.write_json()` is not atomic; `vision.export` already uses a local
same-directory temporary-file plus `os.replace()` pattern for its report. Reuse
that pattern and, where needed, introduce a small shared atomic writer for these
new artifacts, with interrupted-write cleanup tests; do not assume the general
writer already provides it or migrate unrelated legacy outputs. Report
truncation/interruption honestly. Initialize a `running` record once the output root
passes safety checks; finalize status on
normal completion or caught errors/interruptions. A hard termination may leave
`running`, which is not success. Publish complete embedding/mapping files together
only after extraction succeeds; temporary/partial files must not be advertised as
final results. Do not change unrelated legacy output transaction semantics.

## 8. Small Input/Output Examples

This is a valid minimal selector; it requires no label or specimen ID:

```csv
image
a/001.png
b/001.png
```

An optional metadata selector preserves identifiers such as `001` as strings:

```csv
image,label,specimen_id,sex,view
a/001.png,Species_A,001,male,dorsal
b/001.png,,,unknown,dorsal
```

A valid tiny COCO box source can accompany either selector:

```json
{
  "images": [
    {"id": 17, "file_name": "a/001.png", "width": 8, "height": 6}
  ],
  "categories": [{"id": 4, "name": "Head"}],
  "annotations": [
    {"id": 23, "image_id": 17, "category_id": 4,
     "bbox": [1, 1, 3, 2], "area": 6, "iscrowd": 0}
  ]
}
```

The second CSV row is selected even though this COCO source has no entry for it.
An illustrative emitted mapping (not the full run schema) is:

```json
{"embedding_row":1,"output_id":"a/001.png","image_key":"a/001.png","selector_reference":"a/001.png","width":8,"height":6,"metadata":{"label":"Species_A","specimen_id":"001","sex":"male","view":"dorsal"},"coco_images":[{"source_id":"instances","image_id":17}]}
```

IDs `17`, `4`, and `23` are preserved; no part name becomes a species label.
The corresponding second mapping has an empty `coco_images` list, not a negative
Head mask. Tiny task examples make the remaining readers concrete. A keypoint category may
extend category 4 above with `keypoints: ["left_tip", "right_tip"]` and
`skeleton: [[1, 2]]`. Its object adds `keypoints: [1, 1, 2, 3, 1, 1]` and
`num_keypoints: 2`: one visible point and one annotated invisible point.

A separate panoptic fixture has a 2-by-2 image, category `{id: 4, name: "Head",
isthing: 1}`, and one per-image annotation with `image_id: 17`,
`file_name: "image_001.png"`, and `segments_info: [{id: 1, category_id: 4,
area: 2, bbox: [0, 0, 2, 1], iscrowd: 0}]`. Its RGB PNG is specified by this
segment-ID matrix (ID 1 encodes RGB `[1, 0, 0]`; ID 0 encodes `[0, 0, 0]`):

```text
1 1
0 0
```

The lower row is void, not a labeled background class. These are document examples,
not created files or evidence that the readers already work.

## 9. Acceptance and Regression Boundaries

The following groups are this foundation's engineering acceptance criteria, not
requirements to build downstream S2 features or new task models. Groups 1–5 and 11
cover the promised readers/integration, groups 6 and 8 are compatibility regressions
for existing paths, and group 7 covers new artifacts. Groups 9–10 constrain data
handling and evidence claims; their real-data/real-checkpoint integration runs need
separate analysis approval and are not mandatory portable-test inputs. The geometry,
spatial export, group-safe evaluation and task-performance deliverables deferred in
§1.1 remain outside this exit. Do not defer ingestion tests for formats promised
here merely because those formats' trainers are deferred.

During separately approved implementation:

1. Test image-only CSV, optional/custom metadata, leading-zero IDs, literal missing
   tokens, root-relative/unique-basename resolution, ambiguity, duplicate rows,
   root relocation, symlink escapes, and ordering/selection precedence. Verify that
   `NA`, `N/A`, `unknown`, `0`, `001` and empty labels retain consistent semantics
   across selection, attention-query class mapping, evaluation and provenance,
   without independent reparsing. Exercise the records input through extraction and
   attention-query training while making legacy CSV loaders fail if called (in the
   test); reject mixed records/path inputs. Check parameter-card/log mode notices for
   `--label-csv` alone and with COCO, including new-path provenance of
   parser choice.
   Legacy reader behavior remains unchanged.
2. Test noncontiguous IDs, source-local category mappings, duplicate IDs/JSON keys,
   broken image/category references, NaN/Infinity, corrupt images, declared/actual
   dimension mismatches, and source-wide versus selected-only validation scope.
3. Test boxes, multipart subpixel polygons, compressed/uncompressed RLE, malformed
   counts, edge/out-of-bounds cases, empty masks and task-specific area/bbox policy.
   Include a self-intersecting polygon with zero signed shoelace area but nonempty
   COCO-decoded coverage, adjacent repeated/closing vertices, and thin/small regions
   with rasterization differences above the former 1% rule. These alone must not
   reject ingestion, produce threshold-based warnings or alter source geometry.
   Explicitly test vertices at `x=width`, `y=height`, corners and nearby subpixel
   positions using the selected COCO decoder: output shape is `(height, width)`
   and decoded RLE run lengths cover `height * width` pixels. Keep valid
   boundary coordinates intact and reject coordinates outside the declared domain.
   Test zero-width/height instance/keypoint bbox diagnostics, task-specific positive-
   extent rejection, negative-extent errors and unchanged panoptic pixel checks.
   These are future acceptance checks, not already verified decoder behavior.
   Verify per-annotation diagnostics and retain RLE/panoptic pixel checks at `1e-6`.
   Source bytes, coordinates, IDs, category ordering, and extensions remain intact.
4. Test keypoint visibility/counts/skeletons using non-human point names; invisible
   and unlabeled points stay distinct. Test panoptic RGB IDs, void, multiple images
   reusing segment IDs, missing/extra segments, metadata and actual pixel agreement.
   Test both PNG-root/name layouts in §4.4, root escapes and failure without silent
   prefix repair or alternate-directory search.
5. Test all task readers through Python and annotation attachment through `extract`.
   Global extraction may not accidentally train on masks/keypoints or add a task
   head. Verify one-to-one embedding/mapping order and metadata preservation.
6. Use synthetic/tiny models and mocked weights for portable integration tests,
   including PyTorch/ONNX, CLS/projector and applicable pooling paths. Same-entry
   legacy regressions compare original IDs, order, `sample` and numerical outputs
   unchanged. Cross-entry comparisons align the same selected images through the
   explicit row/image-key mapping before comparing vectors, not basename equality
   or a fabricated universal ID convention. Check first/middle/last rows and test
   CSV paths `a/001.png` / `b/001.png` remain distinct; directory-only legacy ID
   semantics are not silently repaired. Test no annotation dependency is imported
   by legacy startup/extraction.
7. Verify requested/effective settings, source/weight/content hashes, used versus
   ignored options, input drift, output integrity, schema status, failure and
   interruption records, safety checks, and no source writes or silent repair.
   Test internal metric status/reason propagation, compatibility value/`None` returns
   and CSV empty values, with no duplicate eligibility logic or second metric run.
   Test atomic JSON interruption/cleanup separately from canonical digest bytes.
   Verify legacy runs are not required to emit the new artifacts.
8. Keep current training/resume, fine-tune/pseudo-feedback hashes, cluster/annotate
   log parsing, sample conventions, metric availability and global-vector contracts
   unchanged. Run focused regressions then the full suite before completion claims.
9. Keep all 230 existing teaching images untouched. Larger local data inspection
   found 3,260 CSV/image records and 121 label classes with optional `sex` and `view`;
   it had no explicit specimen ID and had singleton classes. A subsequent read-only
   COCO review checked structure, references and image dimensions for 9,764 polygon
   annotations; missing Head coverage, self-intersections and adjacent repeated
   vertices did not establish format failure or require source repair. The operator
   supplied raster-area comparisons and external converter clipping logs; those
   comparisons were not independently reproduced in this environment. These are
   observations, not portable test constants, exhaustive consumer validation or
   scientific ground truth. Revalidate through the approved reader during separately
   approved integration. Do not ship the local absolute path, real images,
   annotations, or private metadata.
10. Document actual versus unverified support. Synthetic task ingestion tests are
    engineering evidence, not real keypoint/panoptic performance or proof of
    leak-free evaluation. A real checkpoint extraction or full-dataset run needs
    separate analysis approval and an explicit output directory.
11. Test independent optional relationship columns, manifest-scoped values and
    absent/empty cells without inferring groups or changing selection. Preserve
    custom ignore/unknown fields without treating them as interpreted masks.
    Verify standard visibility/void states remain distinct. Test separate image,
    backbone and vector normalization records, effective feature outputs, explicit
    unavailable details, label/metric protocol and absent split/task-result reasons.

## 10. Documentation, Scope Exclusions, and Approval Gates

After separate implementation approval, update both READMEs for navigation/scope,
`docs/commands/extract.md` and its `.cn.md` twin for actual flags/inputs/outputs,
and a bilingual shared data-contract reference if needed. Follow the existing
command-document ownership and section conventions. Do not duplicate exhaustive
flags in this design or replace historical version notes.

Update runtime CLI schema discovery/validation, workflow Skill guidance and its
relevant references, protected input paths, per-step parameter cards, input
availability interpretation, and documentation/Skill tests together. The Skill
must not infer permission to convert data, run extraction, or overwrite inputs.
No extra dataset command or phantom workflow stage is introduced. Shared function
reuse by later commands must use their own separately reviewed task requirements.

Excluded: conversion/repair tools; automated group split and near-duplicate
inference; spatial feature/grid export or cache; new geometrical augmentation
machinery; measurement schemas/computation; model heads, task trainers/predictions,
new SSL losses; universal evaluator/metric registry; all-command provenance
migration; dependency installation, release/version changes, or publication now.

Review this design and its author proposals first. Only an explicit approval of
the document authorizes preparation of a separate implementation plan. The plan
must itself be reviewed and its execution explicitly approved. No approval implies
a Git commit, push, tag, release, external-data conversion, or full analysis run.

This draft received author self-review and an external review supplied by the
operator. Its recommendations were checked against the current code and document;
operator-confirmed revisions are incorporated here. Receiving and addressing that
review does not establish an audit pass or approval of the entire revised draft.
No fresh reviewer was delegated by this session; any such delegation requires
operator authorization.

## 11. References

- [Master design](2026-03-29-otuformer-design.md).
- [Output safety and resume](2026-07-18-output-safety-and-resume-design.md).
- [Sparse-label pseudo-feedback and legacy manifest identity](2026-09-26-otuformer-sparse-label-pseudolabel-design.md).
- [Documentation conventions](2026-09-29-otuformer-documentation-conventions-design.md).
- [Workflow Skill](2026-09-30-otuformer-workflow-skill-design.md).
- Official COCO data formats: `https://cocodataset.org/#format-data`;
  source: `https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm`.
- Official COCO mask API: `https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py`.
- Official panoptic representation/checking reference: `https://github.com/cocodataset/panopticapi`.

Official sources were inspected on 2026-10-29. They define format/codec semantics,
not this project's proposed severities, tolerances, CLI names, or provenance schema.
