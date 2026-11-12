# Spatial Geometry, Feature Reading, and Evaluation — Design

Date: 2026-11-12
Status: Approved to enter implementation planning on 2026-11-12. After reviewing the design and audit corrections, the operator requested a plan based on this document. The plan itself and product execution remain separately gated.
Baseline: `OTU-Former-dense` at `d18541e`; the working tree was clean before this draft.
Authorization: the operator requested authoring and self-reviewing the [implementation plan](../plans/2026-11-12-spatial-geometry-features.md), then authorized its audit corrections. Local commit authorization covers one commit per completed and verified implementation task and one documentation-only baseline commit for the approved design/plan before the first implementation task, both without repeated confirmation. On 2026-11-12 the operator explicitly authorized committing these two documents locally and pushing that documentation commit to the existing GitHub branch. Execution scope/method remain separately gated; product code, new training, other pushes, dependency/version change, tag and release are not authorized by this documentation publication.

## 1. Intent and roadmap position

Deliver a usable geometry and spatial-feature loop around existing OTU-Former models: transform and trace an image and its spatial annotations, read complete feature grids, save and reopen them, and evaluate frozen same-image correspondence under known transformations.

The operator prioritized this work over further relational-confidence weighting or adaptive-neighbor research. Keep the existing batch-only relational default (`--relational-queue-size 0`) and optional FIFO implementation unchanged. This priority decision does not establish that FIFO, confidence weighting, or AdaSim cannot help; no such research conclusion or new training is part of this delivery.

The local `ref/otuformer-v2-development-roadmap.md` supplies the roadmap numbering used here. That ignored research note is context, not a shipped runtime dependency. This design is the durable development contract:

| Roadmap area | This delivery | Not claimed complete |
| --- | --- | --- |
| 4.3 Geometry and spatial annotations | Deterministic and existing random spatial transforms, coordinate maps, valid regions, image/mask/point/box mapping. | New perspective/elastic augmentation, anatomical alignment. |
| 4.4 Backbone and feature reading | Explicit single/multiple complete ViT block outputs, real grids, prefix exclusion, normalization provenance, a narrow importable API. | Every timm architecture, arbitrary internal activation export, task heads. |
| 4.5 Feature export and cache | Per-image NPZ and sharded NPY, common index/reader, explicit offline reuse and re-evaluation. | Automatic cache hits, incremental repair, extraction resume, remote feature stores. |
| Necessary 4.8 | Artifact identity, per-sample diagnostics, fixed-protocol correspondence, summaries and failure examples. | A universal metric registry or biological-quality score. |
| S0 / spatial S1 / S2 / geometry S3 | Set shared contracts; deterministic feature export/read first; complete random geometry and frozen evaluation next. | Full S1, full S3 training-extension integration, S4 task lifecycle. |
| 4.2 group split / remaining S1 | Consume externally prepared selections and preserve explicit metadata; generate no split. | A separate future group-split/data-audit design owns generation, reuse and leakage checks. It is not delivered here or by the data/provenance foundation; P0 remains incomplete. |
| 4.7 / S5 | Consume these contracts later. | Coordinate/ranking losses, a new SSL extension framework, their training/resume contracts. |

The [master design](2026-03-29-otuformer-design.md), [data/provenance foundation](2026-10-29-otuformer-v2-data-provenance-design.md), [Coordinate discussion checkpoint](2026-10-31-otuformer-coordinate-temporary-design.md), [training modularity design](2026-11-07-training-modularity-cli-testing-design.md), and [testing policy](2026-11-07-development-testing-policy.md) govern reuse and acceptance. The Coordinate note's unresolved loss decisions remain unresolved; this design settles only the geometry/read-only consumers it explicitly covers.

### Goals and anti-goals

- Make exported grids usable without importing a trainer, recovering a model, or retaining machine-local input roots.
- Preserve old embedding extraction/export and CAM meanings. Reject unsupported combinations before clearing output.
- Distinguish correct geometry, healthy numeric artifacts, and learned correspondence. Do not call a coordinate round trip a learned matching success.
- Retain useful source data and diagnose discarded evaluation support. Do not silently discard boundary tokens from storage.
- Preserve original color for the first correspondence protocol. Do not introduce color invariance, infer foreground, infer specimen identity, or claim cross-specimen homology.
- A passing implementation is not evidence of better morphOTU distance, global classification, or dense morphology. Those require separately authorized data and experiments.

## 2. Decisions, alternatives, and approval boundary

### 2.1 Operator-confirmed decisions

| Decision | Confirmed boundary |
| --- | --- |
| Delivery | 4.3 + 4.4 + 4.5 and required evaluation/provenance; 4.7 later. |
| Public surface | One `extract` command with an explicit embedding/spatial product selector, backed by a shared narrow Python API. No separate spatial top-level command or embedding/spatial subcommands. |
| Help | Task panels modeled on `pretrain -h`; parameter groups reflect real internal responsibilities. Show all options with applicability, not hidden mode-dependent help. |
| Storage | Both per-image NPZ and sharded NPY; per-image default; one spatial layout selector. |
| Lifecycle | Explicit save/open/re-evaluate. No implicit inference, automatic reuse, incremental fill, or resume in read mode. Original products remain read-only. |
| Evaluation | Basic checks always; frozen same-image cross-view evaluation implemented now but explicitly enabled; downstream-task evaluation later. |
| Input view | Embedding retains `center-crop`; spatial defaults to `whole-specimen-pad`; one existing `--eval-transform`, explicit request wins. |
| Layers | Default last complete transformer block; explicit single/multiple blocks, saved and evaluated separately, no fusion. Reuse CAM layer-path lookup/listing. |
| Normalization | Backbone final norm versus none is explicit; default final norm. Feature normalization and CAM heatmap min-max are different. L2 is a separate proposed setting, not a CAM reinterpretation. |
| CAM | Reuse applicable lookup/grid/geometry support; preserve attribution algorithms, scalar target, target-layer default and heatmap `raw`/`normalized` semantics. No new attribution method. |
| Geometry | Preserve existing transform behavior/order; retain complete grids and valid coverage; conservative correspondence eligibility and reported coverage. |
| Evaluation views | Independent versioned geometric protocol; per-transform/layer reports; identity only an engineering check; no extra appearance perturbation. |
| Color | Retain source color. No new color augmentation/invariance requirement or training-profile changes. |
| Backends | Compatible existing standard-ViT PyTorch checkpoints plus explicit spatial ONNX export/read, fixed resolution and exported layer/norm contract, all-output numerical checks. |
| Re-evaluation entry | `extract` can read a spatial artifact explicitly, without a model or images for numeric operations; no implicit feature generation or mutation of the input artifact. |
| Selection | Spatial image mode: CSV selects, otherwise recursively scan all supported images; COCO attaches annotations only. Spatial read mode: CSV selects stored samples, otherwise all indexed samples. Old embedding selection is unchanged. |

### 2.2 Approaches considered

1. **Continue confidence/adaptive-neighbor research first.** Deferred by the operator in favor of usable geometry and spatial diagnostics; existing mixed metrics do not establish a general method ranking.
2. **Separate spatial CLI or two extract subcommands.** Rejected after inspecting parameter overlap. Reuse model/data/runtime/safety arguments and preserve old invocations; output contracts differ without requiring different top-level verbs.
3. **Only add `patch-grid` to `--token-mode`.** Rejected. Token mode remains a vector-construction choice; a complete spatial product has separate layout, geometry, storage and diagnostics.
4. **API-only or geometry-only delivery.** Rejected as insufficiently usable. Ship a CLI/API/export/read/evaluate loop rather than isolated functions.
5. **Per-image only or sharded only.** Both are required; per-image is the default. The common reader prevents layout from changing feature/evaluation semantics.
6. **Automatic cache management.** Deferred. Explicit artifacts are already reusable and have simpler failure and invalidation rules.
7. **PyTorch spatial only.** Revised after review of the current exporter. A shared tensor-returning forward makes bounded spatial ONNX a reasonable part of this delivery; its reader, metadata and validation must ship together.
8. **Add Coordinate loss with the foundation.** Deferred. Existing-model diagnostics are independently useful and provide evidence for the later loss decision.

### 2.3 Author proposals subject to approval of this document

The detailed defaults and names below are proposals, not decisions attributed to the operator: CLI spellings, CSV layer-list syntax, exact coordinates/adapters, independent export L2 (default off), FP32 storage and epsilons, shard byte budget, protocol transforms/budgets/metrics, product filenames and versioned schemas, Python names, CPU ONNX validation bounds, and publication/reader algorithms. In particular, 256 MiB shards and one reference plus 11 transformed views for up to 100 images are unmeasured proposed defaults, not inherited project constants. Complete-design approval covers these proposals; §12 highlights the main review decisions rather than exhaustively repeating them. Concrete Python module/function names remain builder choices under §6.1. No extra tuning flags or pre-plan benchmark project are introduced.

## 3. CLI and mode resolution

### 3.1 Proposed surface

Keep all old valid embedding invocations valid and preserve existing explicit-option applicability unless this design names a spatial-only deviation.

| Option | Proposed meaning |
| --- | --- |
| `--output-kind embedding|spatial` | New on `extract` and `export`; default `embedding`. No implicit switch when another spatial option is given. |
| `--spatial-input PATH` | Existing completed spatial product directory; `extract --output-kind spatial` read mode. |
| `--feature-layers TEXT` | Spatial only, comma-separated backbone-relative complete block paths; omitted means last block for a checkpoint. |
| `--feature-norm backbone|none` | Spatial only; default `backbone` for checkpoint extraction/export. |
| `--feature-l2-normalize` | Spatial only; default false; post-backbone-norm per-token channel L2. |
| `--spatial-layout per-image|sharded` | Spatial extraction only; default `per-image`. No second format/compression option. |
| `--spatial-eval` | Spatial checkpoint, spatial ONNX and artifact read; enable protocol `spatial-geometry-v1` subject to stored-plan requirements. |
| `--spatial-eval-sample-size INTEGER` | Maximum selected images for correspondence; default 100; positive integer. Basic checks still cover the whole product. |
| `--dump-model-structure` | Model-backed feature discovery; list modules and mark spatial-eligible block paths. Same spelling and path convention as CAM. Does not change CAM's established behavior. |

Use one text-valued layer list (`blocks.5,blocks.11`), not a repeating option: current schema rejects repeating options. Whitespace around entries is allowed; empty entries, duplicates, unknown paths, non-block modules, or malformed lists fail. Resolve default last-block to its actual path. Requested order is preserved. No range syntax, negative indexes, `all`, implicit averaging or concatenation in v1. In spatial checkpoint extraction, `--dump-model-structure` additionally writes `model_layers.txt` and continues extraction, matching CAM's existing report behavior; it is not a new image-free/list-only mode. Reject it explicitly for spatial read and ONNX modes.

`--input-images-dir` becomes conditionally required at command-level validation rather than parser-required: still required for every old embedding/image-extraction path; not required for artifact-only numeric read mode. A parser signature change must not introduce a new no-input success path.

| Mode | Inputs | Behavior |
| --- | --- | --- |
| Embedding | Existing checkpoint/ONNX and images contract. | Existing vector output, selection, defaults, metrics and UMAP. Explicit spatial options fail. |
| Spatial checkpoint | Checkpoint + images; optional CSV/COCO. | Extract selected layers, diagnostics, optional correspondence, publish product. |
| Spatial ONNX | Spatial graph + images; optional CSV/COCO. | Read graph-declared outputs and feature contract. No checkpoint execution. |
| Spatial read | Spatial input; optional CSV and image root for image-dependent reports. | Validate/open selected stored samples; diagnostics or compatible re-evaluation; separate result output. |

Read mode rejects explicit checkpoint, ONNX, model source, feature layer/norm/L2, layout and extraction-size/transform changes. It cannot relabel existing tensors. Applicability below is checked before output mutation; an explicitly supplied option counts even when its value equals the CLI default. Omitted defaults for another mode are inactive, not conflicts.

| Options | Spatial image extraction (checkpoint or ONNX) | Spatial artifact read |
| --- | --- | --- |
| `--token-mode`, `--topk-patches`, `--attention-pooling-type`, `--attention-pooling-epochs`, `--use-projector-output` | Reject explicit use; spatial tensors come from complete backbone blocks, never a head or pooled vector. | Reject explicit use. |
| `--metrics-sample-size`, `--umap-n-neighbors`, `--umap-min-dist`, `--umap-metric`, `--visualize-class-number`, `--disable-umap` | Reject explicit use; no embedding metrics/UMAP run. | Reject explicit use. |
| `--spatial-eval` | Allowed on both backends. | Allowed only for selected saved-plan pairs (§8.3). |
| `--spatial-eval-sample-size` | Allowed only with evaluation enabled. | Reject; CSV restricts the stored plan. |
| `--spatial-layout` | Allowed. | Reject; inherit storage. |
| `--feature-layers`, `--feature-norm`, `--feature-l2-normalize`, `--extract-size` | Checkpoint: resolve normally; ONNX: inherit, allow only an explicit exact match. | Reject; inherit stored representation. |
| `--eval-transform`, COCO sources and their mask directory | Allowed. | Reject; use stored geometry/annotations. |
| `--model-name`, `--use-student` | Checkpoint: allowed; spatial ONNX: reject explicit use, even default-valued, because graph metadata fixes model/weights. | Reject explicit use. |
| `--dump-model-structure` | Checkpoint: write the layer report and continue; spatial ONNX: reject explicit use. | Reject explicit use. |

Existing embedding applicability is unchanged and rejects every new spatial-only option. Shared input/output, CSV, runtime, seed and safety options retain their stated roles; runtime defaults never authorize inference in read mode.

Spatial ONNX inherits its layer/norm/L2/size contract. An explicit exact match is allowed and recorded; a conflict fails. Subselecting graph outputs is a Python reader concern initially, not permission to ask for unexported layers. The per-image preprocessing protocol is selected at extraction, with spatial default padding, and recorded separately from the graph's input normalization. A graph whose declared input normalization differs from the supported shared preprocessing fails rather than silently accepting incompatible pixels.

For spatial image mode, checkpoint and ONNX are mutually exclusive. A student request belongs to checkpoint extraction/export; an ONNX graph has already fixed its weight source. Embedding ONNX keeps its old overrides and limitations. Spatial export adds `--use-student` with the shared checkpoint-selection semantics; an explicit true value in embedding export is rejected until separately designed, preserving its old default behavior.

### 3.2 Panels, schema, and internal organization

Proposed extract panels: Input and output; Model and feature source; Preprocessing and geometry; Embedding output; Spatial output and storage; Embedding evaluation and visualization; Spatial evaluation and diagnostics; Runtime and reproducibility; Safety. Use native `rich_help_panel`; declaration order controls panel/option order, so callback reordering and actual positional/signature-caller audit are included. Keep name/flag-keyed consumers stable rather than rewriting them for order. Check narrow/wide rendered help, binding/defaults/explicitness and schema/cards together, following the existing training-modularity §7 approach. Preserve output-before-overwrite safety ordering. Extend export panels only as affected.

Help and bilingual command prose state conditional defaults and applicable modes. Runtime declarations own executable choices; schema/cards derive them. Express applicability in the existing help/shared-restriction fields and conditional resolution in the existing provenance mechanism; do not invent a new `applies_when` field. No schema shape/version change or nested-command traversal is needed. New spatial constraints fail before destructive output preparation, including direct-core callers. A plain `spatial-input` request without spatial output-kind is an error with an actionable message.

CLI resolves requests once and orchestrates. Share data/model/geometry/feature machinery, then dispatch to embedding or spatial product/reporting owners. Do not grow a monolithic mode switch throughout the legacy extractor, import CLI orchestration into core, or create a universal task engine.

## 4. Selection, identity, and annotation reuse

### 4.1 Spatial image mode

Reuse strict `ImageRecord`/`DataBundle`, existing CSV parsing, path resolution, unique keys, source association, hashing and validation. Spatial always uses this validated record path. Unlike legacy directory-only embedding batch mode, which processes child directories, assigns their names as `sample` and can omit root-level images, spatial recursively includes root and nested images and infers no `sample` group from directories. At the spatial record-to-product boundary, retain `sample` only when `sample_origin=="explicit"` (including its missing-value semantics); suppress `first_directory` values from the spatial index and reports. Reuse the reader without changing its legacy inference for other callers. State this difference in both extract command documents; old embedding behavior stays unchanged.

- With `--label-csv`, exactly its rows in CSV order select images; only `image` is required. Labels and other string metadata remain optional under existing missing-value rules.
- Without CSV, recursively enumerate all supported images under the explicit root, sorted by normalized relative image key, even when COCO is supplied.
- COCO images/annotations attach by resolved image key; they never expand or narrow the selected set. Report selected, annotated, unannotated and unused-source counts separately.
- This is an operator-approved spatial-only deviation from the foundation's CSV > COCO-images > directory selector. Implement an explicit internal selection policy in the shared reader, defaulting to its old behavior for existing callers. Do not pass a fabricated CSV or reimplement annotation reading in the CLI.
- Preserve containment, alias warnings, duplicate-key rejection, original references and COCO source-local identities. No deduplication, specimen inference or metadata rewriting.
- Resolve the selected records once and pass the same order to both backends, layouts and evaluation. Reuse verified byte snapshots for decoding/hashing; an image or annotation changing after preflight fails instead of receiving an identity for different bytes. Empty selection, malformed annotation, unreadable input or root escape fails before output cleanup/model export as applicable.

### 4.2 Artifact read mode

Without CSV, select all base image keys in artifact index order. Rule `artifact-key-resolution-v1` uses the existing strict CSV string/duplicate-field conventions, but resolves `image` against stored keys: normalized root-relative exact key first, then a unique basename only for a bare filename; `a/one.png` must never fall back to `b/one.png`. The bare-name limit matches the current filesystem resolver; this deliberately narrower artifact rule additionally rejects absolute references and performs no filesystem/symlink resolution. Reject ambiguity, missing keys, duplicates, absolute references and root escapes. Implement this against stored keys with dedicated resolver tests; it must not require filesystem lookup. Do not require raw images merely to resolve stored identities. Keep CSV order; selecting a sample selects all its stored views/layers.

CSV labels/metadata are a report overlay, not an overwrite of original provenance. Record source artifact identity, overlay hash and colliding fields; overlay values take precedence only in reported sample metadata. Protocol v1 does not group, stratify, select queries or aggregate scores by labels or these fields. Empty values follow existing CSV semantics. Numeric read mode rejects newly supplied COCO sources: stored annotations/coverage can be read; new ingestion requires image extraction.

An optional image root is used only for explicit image-dependent reports. Verify selected image bytes/dimensions against stored identity before drawing overlays. Missing/mismatched images produce unavailable overlay status; numeric metrics remain possible from a valid paired product. Newly supplying images never authorizes inference or missing-view repair.

### 4.3 Spatial annotations

Map existing boxes, polygons/raster masks, keypoints and panoptic labels through the same realized geometry as images. Preserve source IDs, category schemas, visibility, crowd, void and availability states. Missing annotation remains missing, not background/absence. Do not add a segmentation head, format converter, skeleton relabeling or split generator.

Preserve source geometry; derived view geometry has its own visibility/retained-area information. A cropped-away point is out-of-view without changing its original COCO visibility. Transform a box's corners and clip its polygon to valid view support; the derived axis-aligned enclosure is not an exact rotated object box. Expose both if required to avoid presenting enclosure area as object area. Rasterize/transform masks using category-safe nearest-neighbor sampling with the same realized coordinate maps/canvases; keep validity separate so fill cannot become a real background label. Panoptic IDs/void are not continuous intensities. Record the mask sampling convention independently of the image's interpolation kernel.

## 5. Geometry contract

### 5.1 Canonical coordinates and adapters — proposal

Use continuous image-edge coordinates internally: width W and height H define `[0,W] x [0,H]`; pixel `(column i,row j)` has center `(i+0.5,j+0.5)`. Arrays remain row-major `(y,x)`; coordinate pairs and matrices use `(x,y)`. Store sizes as named width/height or explicitly ordered pairs, never an ambiguous `size` tuple.

Adapters are explicit and do not reinterpret raw source fields globally:

- COCO `xywh` boxes and polygon vertices retain the existing continuous edge-coordinate meaning.
- The current reader preserves keypoint values and checks `0 <= x < W`, `0 <= y < H`; that range and its diagnostic name do not establish an origin convention. This design explicitly proposes the Detectron2-style COCO adapter: for labeled keypoints (`v=1/2`), add `(+0.5,+0.5)` from source pixel-index coordinates to internal edge coordinates, subtract it on source-format output, and record `coco-pixel-index-to-edge-v1`. Preserve source floats without rounding/clamping and preserve the unlabeled `v=0, x=y=0` sentinel without shifting. Detectron2 shifts all x/y, including unlabeled points; this adapter deliberately differs for `v=0`. Detectron2's COCO exporter subtracts the offset; its internal shifted records are not raw COCO and must not be ingested as such (labeled points would shift twice; a shifted `(0.5,0.5,0)` sentinel fails the existing reader). This is a chosen adapter, not proof that every producer uses the same convention; do not guess whether input was already shifted. Source-range-valid points outside internal view support remain preserved with `annotation_in_view=false`. Fixtures cover first/last pixel, fractional values, sentinel and exact inverse; reader validation is unchanged.
- Dense array samples have known centers from their array grid. RLE/panoptic remain raster data, not lists of keypoint coordinates.
- No automatic axis swap or normalized-coordinate guessing.

Each view records original/view dimensions, ordered realized operations (including integer crop bounds, resize dimensions/rounding, rotation center/angle/expand, canvas offsets, affine parameters and flips), source-to-view float64 3x3 affine map, and its checked inverse. Matrices map coordinates, not image samples; raster resampling still follows the recorded per-operation kernels and sizes.

For `p_B = M_B inverse(M_A) p_A`, verify p_A/p_B are in the corresponding valid support. Invertible matrices do not restore crop-discarded content. Retain clipping at every intermediate canvas: the final matrix alone cannot represent content lost by an earlier crop/rotation.

The protocol uses forward **content** maps on column vectors, with x right and y down; image resamplers use their inverse. Compose operations by left multiplication in execution order.

| Operation | Forward convention |
| --- | --- |
| Translation | `x'=x+tx`, `y'=y+ty`; positive moves content right/down. |
| Rotation | Positive angle is visually counter-clockwise about `(W/2,H/2)`; centered linear part `[[cos(theta),sin(theta)],[-sin(theta),cos(theta)]]`. |
| Scale | About the same center; scale >1 enlarges content. |
| Reflection | Horizontal `x'=W-x`; vertical `y'=H-y`. |

Required marker fixture: in a 100x100 canvas, `(70,50)` rotated +90 degrees becomes `(50,30)`; translation `(10,0)` then yields `(60,30)`. Record raw library parameters as well as the canonical map: torchvision `rotate` and `affine` use opposite angle conventions, so adapt rather than copying angle signs or altering training behavior. Resize uses actual dimension ratios; expansion/odd-size rounding follows the preserved implementation and fixtures. Primary-source conventions, checked 2026-11-12: Detectron2 `https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/datasets.md` and `https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/coco.py`; torchvision `https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.rotate.html` and `https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.affine.html`. These establish the named adapter/library conventions, not the origin used by every annotation producer.

### 5.2 Existing behavior and coverage

Cover shared deterministic resize/center-crop/pad and the spatial operations currently used by pretrain/finetune: random resized crop, rotations with/without expansion and post-resize, horizontal/vertical reflection, safe-square padding, translation and scale. Track actually executed parameters once, preserving RNG draw count/order and appearance transforms. Color/blur/grayscale do not alter coordinate maps; record their effective settings where the existing caller requires them, without adding them to protocol v1.

Do not switch all image transforms to GPU, replace PIL resampling, or change profiles/defaults/augmentation order to simplify geometry. New geometry-returning paths are explicit. Existing training calls retain their return type/RNG/loss behavior; no new geometry envelope is pushed into the trainer merely to advertise integration. Geometry is consumed now by export diagnostics, annotation mapping, correspondence and CAM; 4.7 will design training handoff separately.

CAM reuses deterministic mapping while retaining visual/array semantics. First test output parity on its supported preprocessing, including crop field-of-view and padding. Preserve its current attribution target selection and square-input support; no new CAM layer/norm option is introduced by this work.

### 5.3 Validity and boundary support

Keep two related records:

1. Geometric source coverage: exact transformed/clipped original-image support, integrated onto and stored on the transformed view's pixel and patch lattices as coverage fractions in `[0,1]`.
2. Conservative sampling validity: whether an output sample's complete image-resampling support contains valid source samples through the entire operation sequence.

Implement geometric coverage by clipping the current valid convex support to each operation's canvas, transforming it, and integrating its intersection with pixel/patch cells. Existing affine transforms preserve convexity. Preserve area fractions and clipping history; do not infer foreground from fill color.

Sampling validity must account for the actual nearest/bilinear/bicubic/antialias footprint and boundary behavior of the image transform. Signed bicubic interpolation of a 0/1 mask is not a coverage fraction. Propagate Boolean support with the resampler's contributing source indices, including actual resize boundary extension semantics. For crop/flip/integer pad this is exact; for rotate/affine/resize define the corresponding support kernels and rounding. Do not declare this complete until synthetic edge tests against each preserved image operation pass. Metadata identifies this support rule as `conservative-resampling-v1`.

Keep all patch vectors in artifacts. Persist per-pixel geometric coverage (FP32) and conservative validity (Boolean) at the transformed model-input view resolution, plus the per-patch coverage/validity arrays. Store them once per stored view and share them across layers/base-reference aliases; do not allocate them at original-image resolution, duplicate them per feature layer or substitute trace-only reconstruction. Store geometric coverage and conservative validity separately; a center-only check is insufficient. A fully valid patch requires all of its input-pixel footprint to be conservatively valid. This prevents direct fill contamination, not ViT global-attention influence from surrounding invalid tokens. Reports state that limitation.

For feature interpolation, use bilinear sampling with explicit `align_corners=False`; a point in view-edge coordinates maps to feature coordinates from the recorded grid lattice, not `sqrt(N)`. For a nonoverlapping patch lattice, center `(c+0.5,r+0.5)` corresponds to input-edge `((c+0.5)*patch_w,(r+0.5)*patch_h)`, including any declared patch padding/offset. V1 supports only verified standard nonoverlapping patch embeddings without untracked automatic padding. With view point `(x,y)`, zero-based feature-index coordinates are `u=x/patch_w-0.5`, `v=y/patch_h-0.5`. For grid width Gw and height Gh, `align_corners=False` normalized coordinates are `gx=2*(u+0.5)/Gw-1`, `gy=2*(v+0.5)/Gh-1`. Require every contributor with nonzero weight to be valid; out-of-bounds support is unavailable rather than zero-padded evidence. Zero-weight neighbors outside the grid do not invalidate an exact edge-center sample.

Name the pixel-level arrays `pixel_geometric_coverage` (fraction) and `pixel_sampling_valid` (Boolean support), and the patch-level arrays `geometric_coverage` (fraction) and `patch_sampling_valid` (Boolean support). Keep `annotation_in_view` as the separate point/annotation geometry concept. Source visibility remains unchanged; none is a biological foreground mask.

Required numeric fixture for the future plan: input 64x64, patch 16x16, grid 4x4; point `(24,40)` maps to `(u,v)=(1,2)` and normalized `(gx,gy)=(-0.25,0.25)`, selecting feature `[row=2,column=1]` exactly. Also test the first/last patch centers and an interpolated interior point. Keep this fixture separate from the COCO point adapter.

## 6. Feature reading and normalization

### 6.1 Shared core, supported models and API

Reuse `resolve_checkpoint()` / `load_checkpoint_encoder()` and existing size/register validation. Support current compatible standard timm ViT pretrain and finetune checkpoints, relational/DINO head identities and legacy layouts already accepted by that loader. Checkpoint loading compatibility alone does not certify intermediate-feature support. Verify actual block sequence, positional embedding path, prefix count, final norm and patch lattice before output mutation.

Do not certify Eva/DINOv3, hierarchical CNNs or arbitrary `vit_*` models by name. Explicit unsupported capability errors are preferable to substituted models/layers. Do not download/init external pretrained models as part of this feature.

One tensor-returning feature forward obtains requested layers in one execution. The tap is the tensor returned by `backbone.blocks[i](tokens)` after that complete module's residual/internal-norm operations, before any next block or final `backbone.norm`. Anchor intermediate tests to that module's forward output; only then apply requested final norm/L2 and remove prefix tokens with the real grid/layout. The final selected block with final norm must agree with the model's ordinary final spatial output in eval mode. Do not reuse a training-only final-four limit as the public API. Avoid persistent hooks that can leak between requests or capture the wrong duplicate module execution; an explicit supported-ViT forward and temporary scoped hooks are builder implementation choices, but ONNX must execute the same tensor semantics.

Proposed narrow core operations: resolve/list feature layers; run a spatial forward; transform/sample geometry; write/open spatial products; evaluate stored/online pairs. Exact module/function names are builder choices. Keep feature/geometry support independent of `vision.cam`, CLI and trainer. Share demonstrated primitives, not an empty plugin registry or general backbone factory.

Extract/read is inference-only. Shared layout/norm primitives must not unconditionally detach tensors or enter `inference_mode`: CAM has its own gradient path. CAM may accept arbitrary attribution module paths under its existing contract even though spatial extraction accepts complete blocks only.

### 6.2 Normalization contract — proposal

For selected block output z:

`a = backbone.norm(z)` when `feature_norm=backbone`, otherwise `a=z`.

`f = a / max(norm_2(a, channel), 1e-12)` when L2 is enabled, otherwise `f=a`.

Apply the checkpoint's same final norm independently to each selected block; do not invent trainable per-layer norms or remove block-internal norms. Its learned parameters, when present, were trained at the final-block position, not calibrated separately for intermediate taps; equal normalization does not imply comparable layer distributions. Default: final norm on, export L2 off. V1 has no scoring consumer that requires storage L2; it is an explicit representation/export proposal, not a prerequisite for cosine matching. The sampler accepts either representation, with different interpolation semantics. Retaining this flag does not authorize a consumer framework. Always store the resolved choices, epsilon and norm module/type identity. Fail for missing/incompatible final norm when requested, rather than silently treating it as identity.

Save one selected representation per layer, not both raw and norm variants automatically. FP32 v1 storage is a proposal; no precision/compression selector. Check finite values and record zero/near-zero counts and channel-norm/dispersal summaries with normalization context. Nonfinite feature arrays fail extraction; a finite zero vector is retained and diagnosed, not converted to an invented match.

Cosine correspondence normalizes after feature sampling, using the fixed epsilon, even when storage L2 was enabled. Interpolation and L2 do not commute, so raw-export and unit-export runs are distinct representation contracts and need not yield identical scores. Layout/backend parity applies only to the same complete feature contract.

### 6.3 CAM integration boundary

Share backbone-relative module lookup/listing, real layout/prefix handling and deterministic geometry where applicable. Preserve CAM's default final-block internal norm target, scalar attribution target, algorithms, gradient behavior and heatmap save modes. `raw` is an un-min-maxed CAM map; `normalized` is per-map min-max, neither controls backbone features.

Do not put LayerNorm/L2 into the CAM library's activation/gradient reshape callback. Layout transformations used by both activation and gradient are not arbitrary nonlinear feature transformations. New attribution targets or normalized-feature CAM require their own design. Existing helper imports remain compatible where they are actually used; tests identify real callers before moving helpers.

## 7. Spatial products, layouts and lifecycle

### 7.1 Common logical record and metadata

A completed product has a versioned manifest and ordered sample/view index independent of physical layout. Each sample retains the original `image_key`, content hash/dimensions and source metadata/annotation references. Each view retains its stable ID, role (`base`, `eval-reference`, `eval-transform`), geometry/support record and appearance protocol. Each layer retains path/index, dimension, grid shape, norm/L2 and tensor reference. Generated numeric file IDs, not raw image paths or basenames, prevent collisions and directory traversal.

Proposed files:

```
run_provenance.json          # running/completed/failed/interrupted and inputs/outputs
spatial_manifest.json        # format/version, layout, feature contract, file hashes
spatial_index.jsonl          # ordered sample/view/layer references and geometry
features/                   # per-image NPZ or NPY shards
annotations.jsonl           # associated source annotation snapshot, when supplied
annotations/                # original categorical raster payloads, when supplied
spatial_diagnostics.json     # all-product numeric/coverage summary
spatial_results.csv          # per sample/view/layer basic diagnostics
spatial_evaluation_plan.json # stored sampled keys/pairs/query plan, when requested
spatial_evaluation.json      # only when evaluation requested
spatial_matches.csv          # evaluated pair/query results, when requested
logs/extract.log
```

The manifest's `files` inventory seals feature/support payloads, source annotation snapshots, index and optional query plan. Reports and the running log are finalized later and listed with final hashes in the run envelope; neither the manifest nor `run_provenance.json` hashes itself. Names and field schemas are proposals. Provenance reuses existing status/reason and canonical hashing helpers, extends spatial-specific sections and preserves old embedding schemas/absence semantics. The logical product identity covers ordered samples/views, input content, model source/hash and weight choice, input normalization, actual geometry, layers/norm, format/protocol versions, evaluation seed/plan and logical array identities; exclude absolute roots, timestamps and physical layout. Physical file hashes separately protect storage. Two layouts with equal logical arrays/contracts have equal logical identity even though their bytes differ. Define each per-record semantic array digest (not an entire physical shard) over a canonical descriptor (dtype including byte order, shape and semantic role) plus little-endian C-order numeric bytes, then hash ordered logical records/contracts using the existing canonical JSON helper. Container names, shard grouping and NPZ headers do not enter logical identity; index/manifest and payload file hashes still protect their actual published bytes.

An artifact is self-contained for numeric reading: indices, mapping/support, arrays and necessary evaluation protocol are included. Original paths/checkpoint names are provenance, not reopening dependencies. When annotations were supplied, persist the selected associated source records/category schemas and original categorical raster payloads in the product, with hashes; unused source images/annotations need not be copied. Preserve annotation availability and geometry-derived view states. Reconstruct transformed annotations from that snapshot and stored operations on read, without the original COCO/mask paths; do not duplicate every full-resolution view mask by default. Relative references must stay within product root, including symlinks; reject absolute paths, escapes, unexpected array keys, duplicate logical IDs, incompatible shapes/dtypes, unsupported versions and incomplete index references. Never use pickle/object arrays (`allow_pickle=False`). Schema version 1 is this product's version, not a bump to the existing CLI or global data-provenance schema.

### 7.2 Physical layouts — proposal

- **Per-image default:** one uncompressed NPZ per base sample, containing its base/evaluation views and selected layers plus numeric support arrays. Names map logical layer/view IDs through the index, not user-controlled archive paths. Keep JSON metadata in the index; do not pack arbitrary objects into arrays. Write per-sample bounded data, not whole-dataset arrays.
- **Sharded:** separate NPY arrays for each compatible `(layer, grid, view/support kind, dtype)` bucket, stored as `[records,Hg,Wg,C]` for features and appropriate numeric support shapes. Index records point to shard-relative path and row. Use generated shard IDs and memory-mappable arrays; selected read order follows the index, not shard order.
- Builder proposal: seal a shard before exceeding 256 MiB of uncompressed array payload. A single larger record occupies a dedicated shard. No shard-size CLI flag; log actual sizes. Variable layouts form different buckets, not padding that is mistaken for valid data.
- Public feature reader returns a per-layer HWC grid plus geometry/support; batched forward uses BHWC. Adapt to BCHW explicitly for tensor samplers. Memory-mapped views are read-only, file lifetimes are owned by a context-managed reader, and selected per-image loads do not materialize an entire shard.

Feature-only evaluation storage is approximately `N_eval * 12 * sum(Hg*Wg*C*4)` bytes across selected layers, plus base grids for other images and support/metadata. For 100 images, a 14x14 grid and C=192, one layer is about 172 MiB; four such layers about 689 MiB, before overhead. Stored pixel coverage/validity add approximately `N_eval * 12 * Hv * Wv * (4+1)` bytes, shared across layers; at 100 images and 224x224 views this is about 287 MiB before patch support, metadata, any base-only records or alias savings. These are arithmetic estimates, not throughput measurements; shard budget is independent of evaluation view count and is not a performance promise.

Validate hash/NPY header/shape before exposing selected arrays; bounds-check NPZ entries and expected payload sizes before allocating. For explicitly selected subsets, validate the whole index's structure but hash/load only referenced selected payload files, reporting verification scope. A full ordinary read validates all selected records. Do not claim untouched payloads were checked. File content hashes do not authenticate untrusted provenance or guarantee scientific validity.

### 7.3 Publication and re-evaluation

Preflight selection, model/graph contract, layer applicability and output/input safety before overwrite. Seal this attempt's payload/index/manifest and validate cross-file references/counts, then generate diagnostics/evaluation from the sealed saved bytes. Finish reports, restore output streams and close the run log before hashing final report/log bytes and publishing the completion record last. No hashed file is appended afterward; `run_provenance.json` itself is excluded from its artifact hashes. Individual rename is not a transaction; a partial interruption must leave no completed product that passes the reader. A close/hash/final-envelope failure must remain non-completed. Record `failed`/`interrupted` and clean only this attempt's files where safe, preserving failure provenance. Do not delete unrelated temporary files.

Output paths cannot be an ancestor of selected input images, source CSV/COCO/masks, model/ONNX dependencies or `spatial-input`. Re-evaluation writes a distinct run directory outside the source product; overwrite never authorizes changing it. No extraction resume or repair of a failed product.

Read-mode checks and scores consume stored values only. If no paired views exist, basic diagnostics succeed but explicit correspondence evaluation fails preflight with `missing_paired_views`; it must not silently skip an explicitly requested capability or run a model. An intentionally empty valid pair within a paired product is a reportable unavailable case, not this missing-input error. A saved evaluation plan deliberately covers only its sampled keys; non-planned selected samples remain available for basic diagnostics and are reported as `not_in_saved_evaluation_plan`, not corrupt/missing pairs.

## 8. Frozen correspondence protocol — author proposal

### 8.1 Protocol and deterministic view generation

Protocol name `spatial-geometry-v1` is fixed and recorded, not a training recipe. It changes no checkpoint, train augmentation or orientation policy. V1 uses the existing resolved square model input `S x S`; original images and intermediate canvases can be non-square. S must meet the verified patch-size contract. Matrix/grid code retains explicit width/height and never assumes a square original or infers grid size by square root. In the table, `round` means integer round-to-nearest with half-to-even ties, and centered integer offsets use floor division. For every view, execute its recorded image-operation sequence from the same decoded original through the shared geometry path; reuse each resulting tensor across selected layers. Intermediate resampling is allowed where that sequence requires it, including the deterministic base construction followed by the table's transform. Do not generate target views from stored/normalized model tensors or reuse an undocumented reference resize; apply input normalization exactly once after geometric operations.

For each selected evaluation image, form a reference using the run's resolved eval transform/size, plus these transformations applied in view-edge coordinates after that deterministic geometric construction and before input normalization:

| Family / IDs | Fixed transformations at the same output canvas size |
| --- | --- |
| Crop/resize / `crop075` | A centered crop with each side 75% of the reference canvas, integer floor sizes/center offsets, resized to input size. |
| Translation / `tx10`, `ty10` | Positive x or y translation by `round(0.10*S)` pixels, zero rotation, unit scale. |
| Scale / `scale090`, `scale110` | Centered isotropic affine scale 0.9 and 1.1, no translation. |
| Rotation / `rot-15`, `rot15` | Centered -15/+15 degrees, fixed canvas. |
| Rotation stress / `rot90` | +90 degrees, separate stress result. |
| Reflection / `hflip`, `vflip` | Horizontal and vertical reflection, separate results. |
| Composition / `combo` | `crop075`, then +15-degree rotation, then positive x translation of `round(0.10*S)`, in that order. |

This yields one reference plus 11 transformed views per sampled image. No photometric augmentation; keep original colors. Use preserved bicubic image interpolation/fill policies and exact implemented rounding. The family definitions must compile to recorded source-to-view operations that reproduce the same geometry without inheriting a checkpoint's training randomness.

The crop/transform protocol observes the resolved base field of view: `center-crop` evaluations cannot claim whole-image coverage; padding is not foreground. Reflection/large-rotation results are stress diagnostics, not a requirement to erase side identity or a change to sensitive training. All geometric operations are tested even if a specific model is not expected to be invariant to them.

Use the existing extract `--seed` (default 42), not a new seed option. Record the resolved integer as `evaluation.seed` in `run_provenance.json` and `seed` in `spatial_evaluation_plan.json`; read mode compares against the latter and never substitutes its inactive CLI default. Default evaluation sample limit 100 images, selected deterministically from already selected samples by seeded SHA-256 ranking. Rank each key by the digest of the canonical JSON tuple `[protocol, seed, "image", image_key]`, breaking a digest tie by image key; take the first limit, then restore sample index order in outputs. If all selected images fit, evaluate all. This does not advance the training/process RNG. Fixed inputs/seed/size/protocol yield identical view records across backends/layers/layouts. Input-size/preprocessing changes define different experiments and cannot be pooled as directly identical protocols.

No retry-to-obtain-overlap. Store generated invalid/no-overlap pairs and their reason so difficult transforms do not disappear from coverage. Identity forward-repeat, transform round-trip and feature store/read checks are engineering fixtures only, not entries in quality averages.

### 8.2 Queries, candidates, matching and metrics

Evaluate reference-to-transformed and transformed-to-reference separately and provide an equal-direction summary. Each query is a conservatively valid source-grid center whose mapped point is valid in the target and whose target feature interpolation has complete valid support. Determine geometric eligibility without model features.

Take at most 64 eligible query centers per direction/pair, independent of layer/backend/layout. Use the same seeded hash ranking with tuple `[protocol, seed, "query", image_key, pair_id, direction, row, column]`, break ties by row-major index, and restore row-major order after selection. Record source valid-center count, mapped-overlap-eligible count and sampled count separately; report geometric eligibility as eligible/source-valid, with zero source-valid marked unavailable, and sample fraction separately from coverage. Keep all valid target grid centers as candidates; do not use true location, a neighborhood around it, labels or object masks to shrink candidate search. Exclude neither a difficult background nor an annotated part by default. Ground-truth mapped points remain continuous; nearest candidate centers quantize predicted locations.

Match each sampled source-center feature against every valid target-center feature by cosine similarity; this v1 retrieval uses grid-center candidates, without continuous prediction refinement. Separately expose bilinear feature sampling for continuous mapped points and use it for target-support checks/numeric diagnostics, not as a ground-truth-informed candidate shortcut. Normalize cosine vectors after sampling. A query with nonfinite features fails numeric validation; a norm <=1e-12 contributes `invalid_query_feature` and a failed match within the geometric query denominator. Invalid candidate vectors cannot win; report their count. All invalid candidates produce `no_usable_candidate_features`, a failed match for otherwise geometrically eligible queries. Do not let a collapsed model improve coverage by silently dropping its queries. Stable ties choose the smallest row-major candidate index.

Map predicted target center and continuous true point to original coordinates and report Euclidean error divided by original image diagonal. Store raw pixel error as well. Report PCK at normalized thresholds 0.01, 0.02 and 0.05 and mean/median normalized error. Invalid-feature failed matches have no finite error: report numeric-error mean over successful finite matches alongside failure fraction and PCK over all geometrically selected queries, never describe that conditional mean as all-query accuracy.

Report candidate-grid quantization lower-bound errors for the same true points, without using those oracle candidates for model scores. This makes low-resolution limits visible rather than loosening tolerance per model. Fixed input/patch sizes are required for direct layer comparisons; different grids show their budgets and ceilings separately.

Aggregate queries within each image/pair/direction, then macro-average images for each transform/layer/direction; report its contributing-image denominator and micro counts. On a CSV subset, the macro denominator is the selected saved-plan images with at least one geometrically eligible query for that row, not the original plan size or all selected base samples. No-overlap images are unavailable and counted separately; all-feature-invalid eligible images remain in the PCK denominator with zero. Conditional finite-error means have their own finite-match/image counts. Labels/overlays never form groups. Combine directions equally only when both are available; otherwise mark the combined row unavailable. Do not publish a weighted score across transform families.

Store query positions, predictions, errors, match status, candidate counts and protocol in `spatial_matches.csv`. Summary/diagnostic CSV row order follows selected index order, then stored view/pair/direction, requested layer and row-major query order; never shard/file order. Keep timestamps/timings outside numeric CSV rows so equal logical records have layout-independent reports. Summaries include image/pair/query coverage, invalid supports/features, per-layer/per-family results and timings. Default error examples: deterministic worst five finite-error image/pairs per layer/family, plus five failure-status examples; matching line overlays require verified original images. Without images, preserve coordinate/numeric examples and mark image overlays unavailable. Do not use labels or test scores to choose the layer/threshold/protocol.

### 8.3 Online and offline use

When first extraction requests evaluation, store the reference/transformed feature grids, complete geometry/support and query plan for sampled images, not only final scores. Non-evaluated samples keep their base view. Reference tensors can alias the base record logically rather than be stored twice. Both layouts represent the same views.

Stored re-evaluation uses the exact protocol, sampled keys/pairs/query plan and layer/norm contract. Basic diagnostics cover the whole read selection. Correspondence evaluates the intersection of selected keys and the saved evaluation plan, preserving that plan; CSV may restrict it but cannot expand it to previously unevaluated samples. With no CSV, an ordinary re-evaluation therefore reproduces all original evaluated samples even when the product contains more than the 100-image evaluation budget. Report selected/planned/evaluated/not-planned counts separately. If the intersection is empty, reject an explicit eval request before output changes and explain that no stored planned pairs were selected. A missing payload for a planned pair is artifact corruption and fails validation; diagnostics-only reads of valid base records remain allowed. Reject an explicit `--spatial-eval-sample-size` in read mode; use CSV to choose a subset instead. Inherit the stored seed; an explicitly matching seed is allowed, a different one fails. Matching scalar thresholds belong to protocol version, not hidden read-time overrides.

Online results must equal offline results for the same bytes/contract (CPU exact equality where appropriate; documented backend numeric tolerance only where execution actually differs). An NPZ-to-NPY rewrite, if later provided, is not required now; tests generate both from the same logical records to check layout parity. Frozen evaluation cannot establish cross-specimen anatomical identity, part segmentation, phylogenetic distance, global barcode protection or held-out biological benefit.

## 9. Spatial ONNX export and inference

Extend existing `export`, not a new exporter command. `--output-kind embedding` retains old encoder+projector output/name and validation behavior. Spatial uses the same selected-feature forward as PyTorch extraction and exports one named output `spatial_000`, `spatial_001`, etc., with a manifest mapping outputs to complete block paths/grids/norm. No extra embedding output is silently appended to spatial graphs.

Fix model/teacher-student choice, input size, layers, norm and L2 at export. V1 requires dynamic batch with exporter/ORT checks at batches 1 and 2; input H/W remains fixed. A model that cannot pass that contract is unsupported for spatial ONNX v1; do not silently publish a fixed-batch fallback. Do not expose unexported layers, runtime norm changes, arbitrary intermediate names, a new provider selector or dynamic-resolution claims. Preserve the current CPU ONNX Runtime inference path; PyTorch continues to use the existing device resolver.

Embed a small canonical JSON feature contract in ONNX model metadata: spatial schema version, ordered output names, layer/block identities, HWC layout and grid, channel dimension, patch/prefix lattice, input size/normalization, feature norm/L2/epsilon, actual model identity and weight source. This is required graph metadata, not a sidecar-only dependency or architecture inference from names. Validate metadata against graph I/O; old graphs missing this contract are accepted only by the old embedding path and rejected as spatial. Treat user-supplied metadata as assertions to validate, not proof that arbitrary graph computation is authentic.

Keep per-image source geometry outside the graph: the shared preprocessor owns it. ONNX receives normalized image tensors, returns the declared spatial arrays, then shares storage/evaluation with PyTorch. Existing graph/external-data hashing and input-safety windows remain; model relocation alone must not alter semantic identity.

Generalize the exporter validator to compare every named spatial output against CPU FP32 PyTorch reference for shape, dtype, finiteness and values, including batch 1 and 2, selected intermediate/final layers and configured norm/L2. Propose initially the existing `atol=1e-4`, `rtol=1e-3`, but these bounds require measured spatial evidence; do not loosen them to conceal failure. Report per-output max errors and environment. Without ORT, the export cannot be marked spatial-validated; missing runtime is a reported unmet acceptance prerequisite, not a pass. A graph/export failure preserves any previous published model/report.

Use a spatial-specific versioned report/contract section without mislabeling channel width as the old embedding `out_dim`. Source selection/checkpoint identity, all outputs and hashes must survive temp-file publication and external tensor packaging. Verify graph hash only after final metadata/external-data serialization. Do not make CPU evidence a CUDA/TensorRT/CoreML guarantee.

## 10. Compatibility, failures and ownership

- Default old `extract`/`export` still produce embeddings; no automatic extra model forwards, spatial files, changed evaluation protocol or conditional default outside spatial.
- Preserve legacy basename/batch-directory and CSV/COCO selector semantics for embedding. The new explicit spatial reader policy must be covered by side-by-side tests so it cannot leak into existing callers.
- Pretrain/finetune objectives, step mechanics, RNG, checkpoint state and resume rules are unchanged. Geometry helpers must preserve the old tensor path; optional FIFO and DINO configurations remain untouched.
- CAM helpers shared by actual callers may move, but imports, targets, gradients, output min-max and mapping/FOV stay compatible. Do not refactor unrelated CAM algorithms.
- Core returns data/results; CLI handles invocation logging and output orchestration. Python APIs perform trust-boundary validation as well; CLI-only validation is insufficient.
- Separate numeric diagnostics from optional evaluation status. Finite-zero, missing annotation and no geometric overlap have distinct reasons; invalid inputs and missing explicitly requested paired data fail preflight.
- Arrays/indices are not updated in place by read/evaluation. Reports identify what was verified and what remains unavailable.
- No new storage/vision dependency is proposed: use installed NumPy, PyTorch, Pillow/torchvision and ONNX tools. If an implementation requires an additional geometry dependency, pause for review rather than adding one silently.

## 11. Delivery stages, documentation and verification

### 11.1 Three bounded stages

1. **Deterministic spatial read/write.** Shared feature/layer contract and norm controls, deterministic geometry/annotation mapping, both artifact layouts/readers, explicit extract modes/CSV selection, basic diagnostics and representative PyTorch checkpoints. Exit: a user can save/reopen complete grids and sample them in original-image coordinates.
2. **Complete existing geometry and backend contracts.** Track existing random spatial operations with valid support; integrate appropriate CAM helpers with parity; spatial ONNX export/read with metadata/all-output validation. Exit: all promised transforms and backends satisfy the same feature/geometry contracts; no training integration claim.
3. **Frozen correspondence and offline re-evaluation.** Versioned protocol, paired artifacts, metrics/coverage/error examples, online/read/layout/backend parity within evidenced tolerances. Exit: an external caller can reproduce numeric evaluation from stored products without raw model/images.

These are milestones in one project, not authorization to claim the later stages complete when only center-crop or final-layer PyTorch works. Stage 1 is reviewable before ONNX completes. Within stage 2, record random-geometry, CAM-parity and ONNX validation separately so one failure does not hide the others' status; all remain required for full delivery. Spatial ONNX remains operator-confirmed scope and must pass §9; roadmap silence does not defer it. The implementation plan is written only after approval of the complete design and its proposals.

### 11.2 Documentation and workflow surfaces

Implementation must update these owned surfaces together, only for affected behavior:

- Master design: concise new design link and extract/export/CAM compatibility invariants, without duplicating a flag catalog. Follow the [output-safety design](2026-07-18-output-safety-and-resume-design.md) for protection/overwrite, [documentation conventions](2026-09-29-otuformer-documentation-conventions-design.md) for command/help structure, and [augmentation design](2026-09-07-otuformer-training-augmentation-design.md) for preserved transform behavior. Add a dated cross-reference or correction there only if that owned contract changes; do not rewrite all three for link symmetry.
- Relevant data/provenance design: dated spatial-only selector deviation and product extension; do not rewrite the historical embedding selector. Coordinate temporary note: foundation status and still-deferred loss choices. FIFO/relational specs/plans need only an explicit cross-reference/status note if this work actually changes their stated boundary; no invented implementation update.
- New implementation plan plus verification record after design approval; do not retrofit this draft into an execution checklist.
- English/Chinese README navigation and operational notes; English/Chinese `extract`, `export`, `cam` and shared preprocessing/metric command documents where affected. State mode applicability, conditional defaults, storage costs, read-mode restrictions and ONNX fixed contracts at point of use.
- Runtime schema/restrictions/cards/argv builders: new options and source-mode requirements, no repeating-option or schema-format expansion. Parameter panels/help and command prose reflect the same mode contracts.
- `skills/otuformer-workflow/SKILL.md` and affected workflow/teaching/troubleshooting/dialog references: discover runtime flags, explain the two output products and explicit read/eval, never turn missing inputs into inference permission, retain confirmation/overwrite protections.
- The ignored local roadmap/research notes may receive separately authorized status notes after implementation. They are not shipped CLI docs or a required runtime test fixture.

Development specs/plans/guides stay English-only; user-facing documentation retains bilingual pairing. No version/release/tag change follows automatically from the name v2.

### 11.3 Required checks

Follow the [testing policy](2026-11-07-development-testing-policy.md), using focused paths while iterating and full regression at shared-format/refactor/feature gates. Do not train on research datasets to validate plumbing. Portable fixtures are synthetic and real inputs remain read-only.

- Geometry: asymmetric grids/markers, odd/even dimensions, non-square sources, crop/resize/expand/no-expand rotation/flip/pad/affine composition; the signed marker and 64x64 sampling fixtures in §5; COCO labeled/sentinel/fractional adapter inverse; raster category IDs/void, derived visibility, clipped boxes/polygons and interpolation-support boundaries. Reflect a labeled left/right marker without claiming bilateral homology.
- Features: last-block parity with ordinary forward; explicit intermediate/multi-layer one-forward behavior; final norm and L2 formulas/zeros; real grid/prefix/register exclusion; unsupported backbones, nonexistent/non-block layers, duplicate/malformed lists; no leaked hooks or RNG/mode changes.
- Storage: both layouts, multiple views/layers/buckets, record larger than shard limit, bounded reads, hashes, root relocation, tampered/partial archives, path/symlink escapes, shape/dtype/version mismatch, no object arrays, completed-state gating, interruption and no unrelated-file deletion.
- Selection: CSV ordering/optional labels; root+nested images without inferred spatial `sample` versus preserved legacy embedding batch mode; COCO annotation-only versus unchanged embedding COCO selection; explicit versus `first_directory` sample origins; `artifact-key-resolution-v1` exact/unique-bare-basename resolution including ambiguity and a non-bare missing path; overlays/subsets, saved-plan denominators and absent paired views; no specimen inference or model execution on read.
- Evaluation: identity excluded; known synthetic feature fields with true and permuted matches; collapsed/zero features; no overlap; tie stability; candidate quantization; no GT-local candidate restriction; macro versus micro denominators; reproducible view/query sampling; color unchanged; per-transform reports; online/offline and NPZ/NPY parity; reports without original images.
- ONNX: old embedding graph unchanged; spatial graph metadata/output names/shapes; all selected layers, norm/L2 configurations, batches 1/2 and legacy/new/register checkpoint identities; all-output numeric checks and input normalization; explicit conflicts/missing metadata; publication failure preserves old outputs. Missing ORT is not evidence of numerical agreement.
- CLI/help/schema/skill: old argv/default/explicitness; §3's full mode matrix including explicitly supplied defaults/projector/UMAP options; conditional inputs/transform, one text layer list, narrow/wide native panel order/coverage, callback keyword/positional bindings and actual signature consumers (`tests/test_docs.py::cli_option_help` / `cli_flags`, affected CLI smoke tests); cards/argv, output protection, docs and required wheel/import boundaries. Reordering alone does not justify rewriting name/flag-keyed tests. Before the help-panel acceptance gate, harden the shared text assertion normalizer to strip ANSI before removing whitespace; retain actual narrow/wide and plain/forced-color cases, rather than fixing width to hide failures.
- Cross-command regressions: representative extract/export/CAM and pretrain/finetune compatibility where shared model/transform helpers move. Training integration devices follow policy; CPU ONNX/precise geometry tests are explicit CPU contracts, not accelerator certification.
- Performance: report startup/model load, base extraction, extra eval forwards, matching, writing/reading and peak memory separately, for both layouts; state device, sample/patch/query counts and artifact sizes. No throughput or storage-saving promise before measurement.

This draft has no implementation, numeric agreement, runtime-cost or biological-benefit evidence. Acceptance records must name exact commands, baseline/final source identity, environment, devices, failures/skips and artifact locations.

## 12. Design approval and extensions outside this delivery

The operator's earlier scope/behavior approvals in §2.1 did not approve detailed proposals automatically. After reviewing this document and its audit corrections, the operator requested a plan based on it, opening the planning phase. The following review decisions form that plan's basis:

- Geometry/read semantics: the named COCO `+0.5` adapter, forward transform signs/sampling fixtures and mode-conflict table.
- Representation/storage: final norm versus none, independent L2 default off, FP32 and 256 MiB shard budget.
- Evaluation: 12 total views, maximum 100 images and 64 queries per pair/direction, thresholds/denominators and the resulting unmeasured storage/runtime cost.

These decisions remain visible for plan review rather than creating a separate per-item approval workflow. No additional blocking operator question is known after the audit revisions. The [implementation plan](../plans/2026-11-12-spatial-geometry-features.md) must itself be reviewed before execution; implementation, research training and publication remain separately authorized.

Possible separately scoped extensions are Coordinate loss (train on known spatial correspondences), appearance robustness evaluation (color/lighting changes), task heads (for example segmentation or keypoints), cross-specimen matching, variable-height/width spatial ONNX (distinct from the required dynamic batch), automatic cache discovery/reuse and additional architectures. This is an exclusion list, not a committed roadmap or a prerequisite for this delivery. Add no placeholders/frameworks for it. Completing the foundation changes no default training objective and establishes no global-quality score or anatomical claim.

### Draft review record — 2026-11-12

Full design/plan author audit (2026-11-12) rechecked §§1–12, all planned task interfaces/dependencies and the existing reader/model/CLI/schema/CAM/export/logging owners. This pass aligns N4's documentation-baseline authorization and clarifies §7's sealed-manifest versus finalized report/log hash order. The implementation plan owns the corrected test/step dependencies and records the findings; no product execution, commit or independent subagent review occurred. The historical draft-only evidence below predates plan authoring and does not describe the current two-document working tree.

Initial author self-review used the builder/reader questions: can a builder determine selected keys, tensor tap/norm/layout, geometry support, offline inputs, evaluation denominators and compatibility without guessing? It resolved the sampling equations, saved-plan boundary, deterministic ranking, snapshots and per-record digests. This was not independent review; no subagent was dispatched.

Across the operator-supplied audit rounds, the author checked disputed claims against reader/path behavior, the model block loop, extract flags/batch behavior, topic contracts and §5.1's primary sources. Corrections are integrated where owned: mode conflicts, chosen COCO adapter/sentinel difference, signed geometry/fixtures, seed, split ownership, taps/norm, help bindings, explicit-only `sample`, artifact-key resolution and report denominators/order. Removed label-grouping semantics and acknowledged that storage L2 is a representation proposal without a required scoring consumer. Kept `M_B inverse(M_A)` and operator-confirmed spatial ONNX. Later passes narrow §12's checklist claim, suppress derived `sample` fields, rename the ranking domain to `image`, name the index-only key rule, distinguish Detectron2's sentinel handling and reject spatial-ONNX model/weight overrides. R9 adds exact terminal/color evidence and ANSI-safe help-assertion acceptance without changing test code. Stage 2 checks are recorded separately without adding stages or scope. The revisions added no API, tuning flag, benchmark project or plan.

Draft verification after the terminal-environment audit: Python 3.11.15, pytest 9.1.1, Rich 15.0.0, Typer 0.25.1 and Click 8.4.2; `TERM=xterm-256color`; subprocess stdin/stdout/stderr are non-TTY pipes. With `FORCE_COLOR`, `NO_COLOR`, `CLICOLOR`, `CLICOLOR_FORCE` and `PY_COLORS` unset, `COLUMNS=120 python -m pytest tests/test_docs.py -q` and the 80-column run each returned **43 passed** (exit 0); the narrow help probe emitted no ANSI. Any non-empty `FORCE_COLOR` forces color in this tested help stack, including the string `0`; at 80 columns both `FORCE_COLOR=0` and `FORCE_COLOR=1` returned **2 failed, 41 passed** (exit 1), limited to the pretrain/finetune documentation-link assertions. `FORCE_COLOR=""` returned **43 passed**. Adding `NO_COLOR=1` did not counteract either non-empty value. At 120 columns, the tested `FORCE_COLOR=1` run still returned **43 passed**. The narrow colored URL contains `training-augment\x1b[0m\x1b[2mation.md`; stripping ANSI makes the expected URL match. Thus width and actual color emission, not TERM alone, explain this reproduced fragility. This existing test issue is not fixed by the design-only edit; §11.3 requires hardening during the help change. Separate read-back checked local links, fences, whitespace, mode-table coverage against the 11 actual embedding-only flags and working-tree scope. Assert-based arithmetic checks confirmed the patch-center and signed-marker examples and the 172/689 MiB estimates; these are design checks, not implementation evidence. Command-document tests do not establish the design's technical correctness. Only this English design file was added. No product/model execution, training, ONNX export, implementation plan, commit or publication was performed. The proposed geometry, numerical tolerances and storage/performance costs remain unmeasured.
