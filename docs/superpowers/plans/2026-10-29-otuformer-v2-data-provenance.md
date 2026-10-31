# OTU-Former v2 Data and Provenance Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline execution, or superpowers:subagent-driven-development only after the operator explicitly authorizes delegation. Steps use checkbox (`- [ ]`) syntax for tracking. Neither execution nor delegation is authorized by this document.

**Goal:** Connect a shared, strictly validated CSV/COCO record set to existing global embedding extraction and additive, machine-readable provenance without changing legacy behavior.

**Architecture:** A small `otuformer.data` package owns source indexing, identity, geometry checks and provenance primitives. The existing extractor gains an explicit records input and shares its inference loops with record-backed datasets; a private execution result carries actual settings to the command while the public function still returns a DataFrame. Only the new CLI path publishes the new artifacts; existing commands and legacy readers remain on their current contracts.

**Tech Stack:** Python >=3.11; stdlib CSV/JSON/hashlib/pathlib/tempfile; installed NumPy, Pillow, pandas, PyTorch, scikit-learn and ONNX Runtime; lazy optional pycocotools for polygon/RLE operations; pytest.

**Spec:** [Data and provenance design](../specs/2026-10-29-otuformer-v2-data-provenance-design.md), including its §2 approval distinctions and §9 acceptance groups. Also read [master §4.10](../specs/2026-03-29-otuformer-design.md#410-proposed-v2-data-and-provenance-foundation), [output safety](../specs/2026-07-18-output-safety-and-resume-design.md) and [documentation conventions](../specs/2026-09-29-otuformer-documentation-conventions-design.md).

**Status:** Proposed — awaiting plan review and explicit execution approval.
**Authorization:** The operator requested this plan document after the design revisions. That request authorizes drafting this plan, not product changes, dependency installation, analyses, commits or publication. The source design still carries Proposed status; this plan does not silently change it or claim an audit pass. Confirm approval of the governing design and this plan before implementation.
**Delivery:** One uncommitted change set after execution approval; no automatic task commits. Commit, push, tag, release and version assignment remain separate decisions. No execution method has been selected. Execution also requires resolving the pending test-environment prerequisite under Global Constraints below.

## Global Constraints

The following constraints quote the spec; every task inherits them.

- “CSV is optional. When supplied, only `image` is required; `label` and every other metadata column are optional.”
- “The internal `image_key` is the normalized POSIX path relative to `images_dir` of the matched image, not merely its basename, CSV row number, or absolute location.”
- “Parse the selector once on the new data path and pass the resolved image records and raw label strings to every consumer, including attention-query training, metric/UMAP alignment and provenance.”
- “Reject resolved root escapes, including symlink escapes.”
- “COCO boxes use absolute original-image `[x, y, width, height]` values.” No bbox-format autodetection or additional bbox option is planned.
- “Self-intersections and adjacent repeated polygon vertices alone do not justify rejecting or repairing annotations.”
- “RLE/panoptic mask area and pixel-derived bbox retain exact pixel semantics with the proposed numeric tolerance `1e-6`.”
- “Keep `extract_embeddings()` returning its existing DataFrame.” Existing positional parameters retain their meaning.
- “The workflow Skill derives this card notice from the option combination; it is explanatory text, not a new field in the exported CLI schema or proposal payload.”
- “Source IDs are fixed roles (`instances`, `keypoints`, `panoptic`) in this first version, with at most one file per role.”
- “Only the new data path described in §6.1 is required to produce the additive artifacts in §7.1.” New artifact `schema_version` is integer `1`; the CLI schema remains version `1`.
- “Do not change the existing fine-tune/pseudo-feedback manifest-hash definition.” Do not migrate checkpoints, training/resume, global vectors or existing log parsers.
- “Portable tests use small synthetic fixtures, not workstation paths.” Keep the 230 teaching images and external data unchanged.
- “No approval implies a Git commit, push, tag, release, external-data conversion, or full analysis run.” No converter, `dataset` command group, task trainer, measurement schema, feature cache or universal metric registry.

Plan-level choices below are proposed implementation details, not additional operator-confirmed requirements. If implementation reveals a needed contract change, stop and return to design review rather than broadening this plan.

**Execution prerequisite — test environment approval (partly resolved):** The operator-supplied review reported pycocotools missing in both inspected interpreters, and this revision did not rerun those external-environment checks. After that review the operator approved a wheel-only install into the existing project environment (Python 3.11.15, NumPy 2.4.6), where `pycocotools` 2.0.11 is now importable while the editable `otuformer` install still points at the legacy checkout. That project environment is the approved interpreter for the commands below unless the operator names a different isolated one; it is a shared environment, not a purpose-built isolated environment. Prefer `pip install pycocotools==<recorded version>` there, and reserve `pip install -e '.[annotations]'` for a dedicated environment: from this checkout it would repoint the shared editable install away from the legacy source tree. Do not treat plan approval alone as installation consent, and do not modify other environments. Because Task 3's red test precedes the extra declaration, an approved install must bootstrap `pycocotools>=2.0.7` there before that test, or select an already provisioned environment. Any additional dependencies, environment creation or upgrades require an explicit scope in that approval. All `python` commands below mean this approved interpreter. Verify `import pycocotools.mask, onnx, onnxruntime` and record versions before starting; without this resolved prerequisite, do not start execution or count dependency failures/skips as successful task gates.

## Review Focus

These five failure classes receive explicit tests in the owning tasks; they are not permission to defer other spec checks.

1. A source changes after validation: decode the same bytes whose hash is compared, then fail without final embedding/mapping publication on a mismatch (Tasks 1, 4, 7).
2. A symlink aliases or escapes a root: preserve an internal alias's lexical key, warn about physical aliases, reject escapes and mismatched extractor roots (Tasks 1, 4).
3. One combined COCO file serves both roles, or an unselected image is absent locally: retain both role-qualified identities and distinguish source-wide checks from selected-file checks (Tasks 2–3).
4. A cached attention pool bypasses query training: image-only extraction remains legal; label requirements apply only when training is actually needed, and provenance names the pool actually consumed (Task 4).
5. Interruption occurs between two final-file renames, or while JSON is written: per-file atomic writes are not a multi-file transaction; no completed record or success message may advertise a partial pair (Tasks 6–7).

---

## File Ownership and Execution Order

Line references below describe the planning baseline and are navigation hints, not patch coordinates. Read current definitions before editing. Do not refactor unrelated large modules.

| Files | Responsibility | Task |
|---|---|---|
| Create `src/otuformer/data/__init__.py`, `records.py`, `paths.py`, `reader.py` | Public reader, ordered records, source/issue types, strict selection and containment | 1–3 |
| Create `src/otuformer/data/coco.py`, `masks.py` | COCO indexing and task validation; lazy mask decoding | 2–3 |
| Create `src/otuformer/data/provenance.py` | Byte snapshots, digest definitions and artifact serialization | 1, 6–7 |
| Create `src/otuformer/embedding/record_dataset.py`; modify `embedding/extractor.py` | Verified record-backed image/label views and actual execution metadata | 4 |
| Modify `src/otuformer/embedding/evaluator.py` | Add internal metric outcomes while retaining existing public returns | 5 |
| Modify `src/otuformer/utils/io.py` | Add an atomic JSON writer; existing writers remain unchanged | 6 |
| Modify `src/otuformer/cli/extract.py`, `cli/constraints.py` | Additive inputs, shared option checks, orchestration and safe publication | 7–8 |
| Modify `pyproject.toml` optional extras only | Lazy mask dependency; no package-version change | 3 |
| Modify `README.md`, `README.cn.md`, `docs/commands/extract.md`, `extract.cn.md` | Navigation and bilingual command contract | 8 |
| Modify `skills/otuformer-workflow/SKILL.md`, `references/workflow.md`, `references/dialog-templates.md`, `references/troubleshooting.md` | Schema-driven option discovery, derived-mode notice and honest results/approval guidance | 8 |
| Create focused tests listed in each task; extend `tests/test_cli_schema.py`, `test_docs.py`, `test_workflow_skill.py` | Portable tests and existing contract checks | 1–9 |

Keep `src/otuformer/cli_schema.py`, its adapter script and their serialized contracts unchanged unless a concrete failing test proves a small existing-contract fix is necessary; obtain approval before any schema-field/version change. New declared options are discovered automatically. Do not create a second schema or a new reference document merely to repeat this spec; user-facing detail fits the bilingual extract documents for this first consumer.

Tasks 1 → 2 → 3 establish the complete reader. Task 4 consumes it; Task 5 owns diagnostics; Task 6 owns artifacts. Task 7 integrates their interfaces. Task 8 completes the same user-facing delivery; Task 9 is the release-independent regression gate. All tasks are required for this plan's completion, not separately shippable claims of full COCO support.

For each later named test in a task, repeat the same red → minimal implementation → green cycle as its first test; do not implement untested branches in advance. Parameterized fixtures are generated under `tmp_path`, not committed private data. Record each gate's actual command/result before checking its box; stop on environment errors rather than installing or switching runtimes without approval. Keep a task-to-file/hunk inventory as work proceeds for Task 9's handoff, especially when later tasks edit the same file; do not recover attribution by reverting earlier work.

## Shared Interface Ledger

These are proposed internal Python contracts, not new CLI schema fields. Use dataclasses and ordinary dict/list payloads, not a plugin system. Define `JsonValue` in `data/records.py` as `str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]` using a Python 3.11-compatible type alias; arrays of image pixels remain NumPy arrays, not JSON.

### Data types (`data/records.py`)

- `FileSnapshot(path: Path, sha256: str, size_bytes: int)`: resolved local access path plus byte identity; paths are not portable keys.
- `CocoImageRef(source_id: str, image_id: int)`; the source ID is one of the three fixed roles.
- `ImageRecord(image_key: str, access_path: Path, output_id: str, selector_reference: str | None, width: int, height: int, image_file: FileSnapshot, metadata: dict[str, str], coco_images: tuple[CocoImageRef, ...], sample: str | None, sample_origin: str)`. A present empty cell stays `""`; an absent column is absent from metadata.
- `SourceIndex(source_id: str, file: FileSnapshot, payload: dict[str, JsonValue], images_by_id: dict[int, dict], categories_by_id: dict[int, dict], annotations_by_image: dict[int, tuple[dict, ...]], image_keys: dict[int, str], mask_files: dict[int, FileSnapshot])`. Preserve the raw payload and category array order; panoptic annotation lookup uses image ID, not a fabricated object ID.
- `ValidationIssue(severity: str, code: str, source_id: str | None, reference: str | int | None, image_key: str | None, message: str, details: dict[str, JsonValue])`. Severity values are `error`, `warning`, `information`; diagnostic measurements are not warnings by default.
- `ValidationReport(issues: list[ValidationIssue], counts: dict[str, JsonValue], scope: dict[str, JsonValue], polygon_diagnostics: list[dict[str, JsonValue]])`; derived `valid: bool` is false iff there is an error.
- `DataBundle(images_dir: Path, records: tuple[ImageRecord, ...], sources: dict[str, SourceIndex], selector: FileSnapshot | None, selector_columns: tuple[str, ...], selection_rule: str, validation: ValidationReport)` with `raise_for_errors() -> None` and `annotations_for(image_key: str, source_id: str) -> tuple[dict, ...]`. `DataValidationError` carries the report. An invalid bundle is inspectable but must not reach inference.

### Reader and byte primitives

```python
read_dataset(images_dir: Path, *, manifest: Path | None = None,
             coco_instances: Path | None = None, coco_keypoints: Path | None = None,
             coco_panoptic: Path | None = None,
             panoptic_masks_dir: Path | None = None) -> DataBundle
resolve_reference(root: Path, reference: str, *, basename_fallback: bool = False
                  ) -> tuple[str, Path]  # lexical root-relative key, resolved access path
snapshot_bytes(path: Path) -> tuple[bytes, FileSnapshot]
read_verified_bytes(snapshot: FileSnapshot) -> bytes
```

`data/__init__.py` re-exports only `read_dataset`, `DataBundle` and `DataValidationError`; keep imports free of torch/CLI. `data/provenance.py` uses `TYPE_CHECKING` for pandas/extraction/outcome annotations and does not import inference code at runtime. Byte primitives must remain usable before any model imports; avoid a reader/provenance/extractor import cycle.

The reader is callable without torch, a model, an output directory or file writes. `snapshot_bytes` and `read_verified_bytes` live in `data/provenance.py` and use only stdlib; the reader and record datasets use them. Decode images/masks from the returned bytes, not by reopening a path. Store snapshots, not a corpus of decoded images. Keep filename strings, digests and original source payloads, not their complete image bytes. `resolve_reference` checks lexical identity and resolved containment; it need not require an unselected COCO file to exist. Only selected files are read/decoded/hashed, while invalid references and source-wide numeric structure still fail.

### Extraction and evaluation

- `extract_embeddings` keeps its present parameters and adds `*, records: DataBundle | None = None`; return type remains `pd.DataFrame`.
- Private `_extract_embeddings_result` takes the same arguments and returns `ExtractionResult(frame: pd.DataFrame, effective: dict[str, JsonValue], used_files: tuple[FileSnapshot, ...])`. Public extraction unwraps `.frame`; the new command path uses the same private engine to obtain the additional result without a second inference. There is one implementation of model loading/pooling/inference, not parallel engines.
- `RecordImageDataset(bundle: DataBundle, extract_size: int, eval_transform: str)` yields the existing image/ID[/sample] batch shape. `RecordLabelDataset(bundle: DataBundle, extract_size: int, eval_transform: str)` uses the same verified-image access, with sorted unique nonempty raw labels mapped to integer class indices. It yields `(image_tensor, class_index)` only after checking all required labels.
- `MetricOutcome(value: float | None, status: str, reason: str | None, context: dict[str, JsonValue])` has statuses `available`, `unavailable`, `failed`.
- Private `_compute_knn_accuracy_result`, `_compute_linear_probing_metrics_result`, `_compute_recall_at_k_result` and `_compute_clustering_metrics_result` keep their corresponding current arguments and return `dict[str, MetricOutcome]`; `_compute_map_result` returns one `MetricOutcome`. Public functions project these to exactly their old value/dict-of-value returns.

The private result entry is the command-facing metadata side of the spec's extraction seam; the public function accepts the same records input. No metadata is smuggled through DataFrame columns/attrs, global state or parsing console logs. `ExtractionResult` is defined in `embedding/extractor.py` and `MetricOutcome` in `embedding/evaluator.py`; neither is imported by the lightweight reader at runtime. The public and command-facing private entry share the same input guards and numerical engine.

## Task 1: Strict CSV selection, stable identity and verified image records

**Files:**
- Create: `src/otuformer/data/__init__.py`, `records.py`, `paths.py`, `reader.py`, `provenance.py`.
- Create: `tests/test_data_reader.py`.
- Read/reuse diagnostics only: `src/otuformer/utils/paths.py:13–52`; do not change its legacy canonicalization/hash behavior.

**Interfaces:** Consumes Pillow and the spec's §3 identity rules. Produces the data types, three byte/path primitives and `read_dataset`'s CSV/directory branches from the ledger. Later tasks extend the same reader, not a second reader.

The first test generates `a/001.png` and `b/001.png` under `images`, with selector rows in that order and labels `NA`, `001`. Its core assertions are:

```python
bundle = read_dataset(images, manifest=selector)
bundle.raise_for_errors()
assert [r.image_key for r in bundle.records] == ["a/001.png", "b/001.png"]
assert [r.output_id for r in bundle.records] == ["a/001.png", "b/001.png"]
assert [r.metadata["label"] for r in bundle.records] == ["NA", "001"]
```

- [ ] **Step 1: Write failing `test_reader_preserves_raw_csv_and_selection`** using temporary 8×6 PNGs and a CSV containing `a/001.png`, `b/001.png`, labels `NA`/`001`, empty metadata and independently optional relationship columns. Assert CSV order, unchanged raw strings, original CSV output IDs, distinct root-relative keys and absent-versus-empty fields; no label is required.
- [ ] **Step 2: Run the red test.** `python -m pytest tests/test_data_reader.py::test_reader_preserves_raw_csv_and_selection -q` must fail because the new reader is absent, not because a dependency is missing.
- [ ] **Step 3: Implement the ledger types and reader's CSV/directory branches.** Use strict stdlib CSV reading from one UTF-8-sig decoded snapshot (BOM tolerated); reject duplicate/blank headers, malformed widths, empty selection and duplicate resolved keys. Treat only empty cells as missing; whitespace-only image references fail. Directory enumeration uses the existing supported image suffix set and lexical keys. Do not import the torch-dependent extractor just to reuse that constant.
- [ ] **Step 4: Add `test_path_identity_and_containment`** parameterized over exact root-relative precedence, unique basename fallback for CSV only, ambiguous basenames, `.`/`..`, absolute inside/outside paths, case/Unicode spelling, internal/escaping symlinks, hardlink aliases and root relocation. Assert `./a/001.png` and `a//001.png` each normalize to `a/001.png` in separate cases; if both occur in one selector they resolve to a duplicate key and fail. Add `.hidden/001.png` and `.../001.png` whose leading-dot directory names must remain intact: the first two normalization cases alone do not detect an erroneous character-set `lstrip('./')`. Assert alias warnings without merging; on systems lacking symlink/hardlink support skip only those cases with an explicit reason. `path_identity()` may identify physical aliases, never define portable keys.
- [ ] **Step 5: Add `test_reader_reports_input_failures_without_side_effects`** for corrupt selected images, invalid image roots, duplicate rows and header-only input. Assert structured codes, source/row/image references and no created output files. Hash and `Image.load()` the same byte buffer. Add `test_verified_bytes_detects_same_size_replacement`: unchanged length/mtime must not defeat a byte-hash mismatch.
- [ ] **Step 6: Run the task gate.** `python -m pytest tests/test_data_reader.py -q` must pass. Snapshot byte functions must round-trip SHA-256 of the exact fixture bytes, not a reserialized image. Review the diff; do not commit.

## Task 2: Source-local COCO box and keypoint ingestion

**Files:**
- Create: `src/otuformer/data/coco.py`, `tests/test_data_coco.py`.
- Modify: `data/reader.py`, `data/records.py` for source indexes and attachment.

**Interfaces:** Consumes Task 1's `DataBundle`, `resolve_reference` and snapshots. Produces `load_coco_source(path: Path) -> tuple[dict, FileSnapshot]`, `index_coco(source_id: str, payload: dict, file: FileSnapshot) -> SourceIndex` and `validate_annotation(annotation: dict, category: dict, width: int, height: int, *, source_id: str) -> list[ValidationIssue]` in `coco.py`. The reader caches loaded payloads by resolved source path and owns source combination, selection, actual-file validation and aggregation; `index_coco` validates source-wide structure/references/finite numeric payloads without requiring unselected files locally. Malformed sources are converted to structured report errors at the reader boundary, not silently ignored.

For the combined-source fixture, image ID 17 is `a/001.png`; the second manifest image has no annotations:

```python
assert [r.image_key for r in bundle.records] == ["a/001.png", "b/001.png"]
assert [(r.source_id, r.image_id) for r in bundle.records[0].coco_images] == [
    ("instances", 17), ("keypoints", 17)
]
assert bundle.records[1].coco_images == ()
```

- [ ] **Step 1: Write failing `test_coco_roles_and_selection_are_source_local`.** Use noncontiguous image/category/object IDs. Supply a combined instance/keypoint JSON for both roles plus a manifest selecting one annotated image and one image absent from COCO. Assert exact CSV selection/order, both qualified references and preserved extension/category ordering; the extra COCO image does not expand selection. Without CSV, assert lexical union of resolved keys across sources.
- [ ] **Step 2: Run red.** `python -m pytest tests/test_data_coco.py::test_coco_roles_and_selection_are_source_local -q` must fail on absent source support.
- [ ] **Step 3: Implement strict JSON/source indexing and box/keypoint checks.** Reject duplicate JSON keys and NaN/Infinity with stdlib loader hooks; reject bool-as-int IDs/dimensions/visibility. Enforce required arrays/fields, source-local unique IDs, image/category references, duplicate image keys and conflicting dimensions. Reuse one parsed payload/snapshot for the same file supplied under two roles, but retain two indexes and hashes qualified by role. COCO names use explicit root-relative resolution without CSV fallback.
- [ ] **Step 4: Add `test_coco_structure_scope`** for invalid unselected references/numbers, locally missing unselected images, conflicting source dimensions, duplicate IDs/JSON keys, unsupported structure, and unknown extensions. Assert source-wide failures versus selected-only geometry/decode checks and explicit unperformed scope/counts. A well-formed unselected image need not exist; it must not be marked decoded or content-hashed.
- [ ] **Step 5: Add `test_box_and_keypoint_validation`** for finite xywh boxes, inclusive continuous box edges, negative extents, zero width/height as information, and non-human keypoint names. Assert `v=0` requires `(0,0)`, `v=1` retains coordinates, labeled points use the strict pixel-center domain, counts use `v>0`, skeleton indices are one-based and valid, names are unique/nonempty. No segmentation is required for box-only/keypoint-only objects. Preserve crowd and custom ignore fields without interpreting them.
- [ ] **Step 6: Run the task gate.** `python -m pytest tests/test_data_reader.py tests/test_data_coco.py -q` must pass. Assert source files are byte-identical and category IDs never become species labels. Review the diff; do not commit.

## Task 3: COCO polygon/RLE and panoptic decoding with explicit scope

**Files:**
- Create: `src/otuformer/data/masks.py`, `tests/test_data_masks.py`.
- Modify: `data/coco.py`, `data/reader.py`, `data/records.py`, `pyproject.toml`.

**Interfaces:** Consumes indexed raw annotations and snapshots. Produces `decode_instance_mask(annotation: dict, width: int, height: int) -> np.ndarray` and `decode_panoptic_map(rgb_bytes: bytes, width: int, height: int) -> np.ndarray` in `masks.py`; reader reports include per-annotation polygon diagnostics, mask snapshots and availability counts. Public decoding returns exact `(height, width)` arrays; a box-only object is not passed off as a fabricated mask.

```python
before = copy.deepcopy(annotation)
mask = decode_instance_mask(annotation, width=8, height=6)
assert mask.shape == (6, 8)
assert annotation == before
assert decode_panoptic_map(two_by_two_png, 2, 2).tolist() == [[1, 1], [0, 0]]
```

- [ ] **Step 1: Write failing `test_polygon_union_preserves_geometry`.** Use multipart subpixel geometry, a bow-tie with zero signed shoelace area but nonempty COCO coverage, adjacent repeated/closing points and a thin region whose continuous/raster area differs by more than 1%. Assert retained vertices/source values, valid shape, COCO union semantics, diagnostics with decoder/version, and no topology or percent-threshold warning/error.
- [ ] **Step 2: Run red in an approved environment with pycocotools available.** `python -m pytest tests/test_data_masks.py::test_polygon_union_preserves_geometry -q` must fail on absent decoder integration. Missing dependency is an environment blocker, not the expected red test. No installation is authorized by writing this plan.
- [ ] **Step 3: Add the optional extra and lazy decoder.** Proposed floor: `annotations = ["pycocotools>=2.0.7"]`; plan-audit evidence on the approved interpreter shows 2.0.11 installing from the `cp311-macosx_10_9_universal2` wheel without a compiler and working with the installed NumPy 2.4.6 (polygons and `x=width`/`y=height` boundary vertices decoded to exact `(height, width)` masks). Keep the floor permissive and record the version actually installed rather than pinning it to the audited one. Keep the base dependencies/version unchanged. Validate polygon numeric structure/bounds before `frPyObjects`/`merge`/`decode`; require valid RLE size/counts and exact decode shape. Accept compressed string and uncompressed list encodings via the library, not a custom codec. Verify uncompressed run totals cover `height * width`; the decoder rejects over-covering totals but silently accepts an under-covering list, so this check cannot be delegated to it. Compressed validity comes from supported COCO decoding and verified shape. Fail early with installation guidance only when the supplied validation actually needs this extra.
- [ ] **Step 4: Add `test_mask_boundaries_and_pixel_policy`.** Parameterize vertices at `x=width`, `y=height`, corners and nearby subpixels; reject values outside `[0,width] × [0,height]` before decoding. Include empty masks, malformed compressed/uncompressed RLE, wrong size and multi-part unions. For RLE area/bbox assert source/pixel differences above `1e-6` error; for polygons assert information-only differences without overwriting source values. Tasks requiring positive/nonempty regions reject explicitly, not in the permissive shared box reader.
- [ ] **Step 5: Add `test_panoptic_rgb_ids_and_root_layouts`.** Use the spec's `[[1,1],[0,0]]` map plus segment IDs needing nonzero G/B bytes. Assert unsigned `R+256*G+65536*B`, RGB PNG only, exact dimensions, numeric 0/1 flags, unique image-local nonzero 24-bit IDs, category `isthing`, void separation and pixel area/bbox. Exercise missing/extra segments, repeated IDs on different images, duplicate per-image records, both explicit root/name layouts, escapes and no prefix repair/search. Decode verified snapshot bytes; keep instance overlaps independent of panoptic exclusivity.
- [ ] **Step 6: Add `test_availability_is_not_absence` and `test_optional_mask_dependency_is_lazy`.** Distinguish source absence, no objects, empty mask, v=0/v=1, crowd and void; custom ignore semantics remain not implemented. In a subprocess/import blocker, base import, box/keypoint reading and legacy startup do not import pycocotools; required polygon/RLE validation without it fails rather than skipping geometry.
- [ ] **Step 7: Run both task gates.** With the approved extra: `python -c "import pycocotools.mask"` then `python -m pytest tests/test_data_reader.py tests/test_data_coco.py tests/test_data_masks.py -q`. Without the extra/import-blocked: run `test_optional_mask_dependency_is_lazy`; create that condition only through the import blocker, and do not uninstall `pycocotools` from the approved environment to simulate it. No skipped mask suite counts as full-reader verification. Review the diff; do not commit.

## Task 4: Record-backed extraction and actual execution metadata

**Files:**
- Create: `src/otuformer/embedding/record_dataset.py`, `tests/test_extract_records.py`.
- Modify: `src/otuformer/embedding/extractor.py` around `_load_model`, CSV inference loops, `_finetune_attention_query`, PCA, ONNX and `extract_embeddings`.
- Existing regression ownership: `tests/test_extractor.py`; no rewrite of its legacy fixtures/readers.

**Interfaces:** Consumes the validated `DataBundle` and verified-byte primitives. Produces the ledger's two dataset views, public records keyword and private `ExtractionResult` entry. For loop reuse, keep `_extract_one_csv` and `_extract_one_csv_onnx` names/current parameters, widen `csv_path` to `Path | None`, and append `*, dataset: RecordImageDataset | None = None`. Accept exactly one CSV path or dataset at these private boundaries. Append `*, records: DataBundle | None = None` to `_extract_with_onnx`. In `_finetune_attention_query`, widen `label_csv` to `Path | None` and append `*, dataset: RecordLabelDataset | None = None`; keep its optimization loop unchanged. Public legacy signatures stay unchanged except the additive extraction keyword. Add optional private metadata plumbing only where settings are actually resolved; no new public configuration options, no parallel numerical implementation.

```python
frame = extract_embeddings(checkpoint, images, device="cpu", records=bundle)
assert isinstance(frame, pd.DataFrame)
assert frame["id"].tolist() == [r.output_id for r in bundle.records]
assert frame.columns.tolist() == ["id", "sample", "dim_0", "dim_1"]  # 2D fake encoder, subdirectory fixture
```

- [ ] **Step 1: Write failing `test_records_extraction_never_reparses_csv`.** Build records from a synthetic image-only manifest, then make pandas CSV loaders, directory scans and basename lookup fail if called in the test. Supply a tiny deterministic encoder with `pretrained=False`; assert public return is a DataFrame, original IDs/order and explicit/inferred samples match the spec, and no metadata enters `dim_*` columns. The private result has the same frame from one inference.
- [ ] **Step 2: Run red.** `python -m pytest tests/test_extract_records.py::test_records_extraction_never_reparses_csv -q` must fail on the missing records keyword/engine, not a download or real checkpoint requirement.
- [ ] **Step 3: Implement the input seam and reuse numerical loops.** Append the keyword-only records parameter after the existing signature. Reject records combined with either CSV path before model creation. Check bundle validity and `images_dir` against its root/containment; do not rebuild identities, rerun CSV resolution or enumerate a directory. Add the two record dataset views and narrowly allow the existing CSV inference loops to consume a prebuilt dataset internally. Legacy calls still construct their original datasets. Route both PyTorch and ONNX through the same record order. Preserve batching and `sample` behavior: use an explicit column as supplied, otherwise first-directory inference; keep origin metadata separately.
- [ ] **Step 4: Add `test_record_attention_query_label_contract`.** At the missing-pool training branch, use the sorted mapping of the same raw label strings and unchanged training loop. Reject any empty required training label; never reread headers or silently filter rows. A compatible cached pool permits image-only records. Test missing/cached/incompatible pool policies, same first/middle/last row identity and duplicate basenames across directories. Preserve pooling hashes/payloads and existing query optimization behavior.
- [ ] **Step 5: Add `test_records_root_and_input_drift`.** A different root, replacement of a pinned access-path component by an escaping symlink, or changed selected-image bytes fails with no final output. Recheck containment of the pinned access path, not CSV resolution or basename lookup; changing an unused lexical alias must not redirect consumption to another file. `RecordImageDataset` reads verified bytes into `BytesIO` and then decodes those bytes; worker processes carry snapshots, not open handles or corpus caches. Include `num_workers=0` and a supported multiprocessing smoke case. Hash checking repeats for images actually consumed each training epoch; no unrequested corpus cache is added. For scale only, a 3,260-image selection over 20 query-training epochs entails up to 65,200 image read/hash/decode visits, plus validation and final extraction (the existing `drop_last=True` may reduce training visits). Hashing adds work to the reads/decodes training already performs; its runtime impact depends on storage, image sizes, workers and model cost. No benchmark establishes a slowdown factor or that I/O dominates. Approval should explicitly accept repeated verification cost; this estimate does not authorize running that dataset.
- [ ] **Step 6: Add `test_execution_metadata_matches_consumed_state`.** Capture requested versus actual size/device, chosen checkpoint key/model/register layout and pooling source at their resolution points. Capture preprocessing mean/std from the actual transform, feature output/layer and backbone normalization from the executed branch, and vector operations in execution order. Record raw CLS versus SSL normalized projector versus unnormalized task head correctly. For patch-topk record effective K, flattening and actual scaler/PCA or small-sample truncation/no-op branch, its fit scope/dimensions/solver/seed; do not recompute or refit for reporting. Unknown backbone internals get an explicit unavailable reason. Compare output vectors, not just shape, with the same-entry baseline.
- [ ] **Step 7: Record and verify used-weight inputs for the new path.** Deserialize the same verified checkpoint/pooling byte buffers whose hashes are recorded; preserve the legacy loader path otherwise. For inline ONNX consume verified graph bytes. For external-data ONNX, inspect the graph with `load_external_data=False`, resolve referenced local tensor files safely relative to the graph, and record graph/tensor paths and whole-file SHA-256 values, preserving external location/offset/length metadata without editing it. Recheck all hashes immediately before creating ORT from the original graph path and immediately after session construction; fail on any observed change. Do not copy/stage model files, rewrite offsets, inline external tensors or alter the source proto. These checks detect observed drift but are not an atomic snapshot against concurrent change-and-restore while ORT opens original paths; record that verification limitation and require source files to remain stable through session loading, rather than claiming exact-buffer verification. If stronger guarantees are required, return to design review. The project's exporter inlines its own external data; retain a tiny third-party-style external-data fixture for provenance coverage. Record actual session provider/output name/dimension; never claim checkpoint/device requests were used in ONNX mode. Test changed graph/tensor files before and during session construction with mocked ORT, unchanged metadata/bytes, and unknown internal feature metadata.
- [ ] **Step 8: Run the task gate.** `python -m pytest tests/test_extract_records.py tests/test_extractor.py -q` must pass without weight downloads. Cover CLS/projector, patch-topk PCA, each supported attention pool and a tiny ONNX model including external data. New-vs-legacy comparisons align selected rows via keys; same-entry legacy IDs/order/sample/numerics remain unchanged. If optional runtime/platform support blocks a case, report it and keep this task open rather than claiming support. Review the diff; do not commit.

## Task 5: Metric reason diagnostics without public behavior changes

**Files:**
- Modify: `src/otuformer/embedding/evaluator.py` at CV eligibility, finite aggregation and the five image-level metric families.
- Extend: `tests/test_evaluator.py`.
- Read callers: `src/otuformer/training/trainer.py` and `cli/extract.py`; do not migrate trainer interfaces.

**Interfaces:** Produces `MetricOutcome` and the private result functions in the ledger. Public metric signatures, keys, folds, normalization, seeds, values and `None` behavior stay unchanged. Use `_make_stratified_cv_result(labels: np.ndarray, max_splits: int = 5) -> tuple[StratifiedKFold | None, str | None, dict[str, JsonValue]]` as the shared internal CV decision helper; `_make_stratified_cv` remains a compatibility projection. Import the sklearn type only as needed to preserve current lazy evaluation imports. This helper is not a registry.

For two classes with one item each, keep public values unavailable while exposing the actual eligibility cause:

```python
outcomes = _compute_knn_accuracy_result(np.eye(2), np.array(["NA", "001"]))
assert compute_knn_accuracy(np.eye(2), np.array(["NA", "001"]))["kNN_Acc_k1"] is None
assert outcomes["kNN_Acc_k1"].reason == "insufficient_class_count"
assert outcomes["kNN_Acc_k1"].value is None
```

- [ ] **Step 1: Write failing `test_metric_outcomes_preserve_public_values`.** Compare public projections and private outcomes on deterministic embeddings. Count calls to `cross_val_score`/`cross_validate`: linear-probe ordinary/balanced accuracy comes from one shared CV run, and obtaining diagnostics never runs evaluation twice.
- [ ] **Step 2: Run red.** `python -m pytest tests/test_evaluator.py -k metric_outcomes -q` must fail on the missing outcome functions.
- [ ] **Step 3: Implement outcomes at actual decision branches.** Stable reasons include `insufficient_samples`, `insufficient_classes`, `insufficient_class_count`, `ineligible_k`, `nonfinite_result`, `no_relevant_pairs`, `insufficient_distinct_vectors`, and `evaluation_failed`; include actual k/fold/class counts or exception type when available. Populate the same outcome context with the actual per-family distance/similarity, normalization, estimator/CV settings and self-exclusion policy required by Task 6 Step 4, once at their resolution sites. Use a reason only on a branch that establishes it; otherwise report unavailable details. Keep existing caught-failure behavior; do not swallow exceptions that previously escaped public calls. Public functions unwrap outcomes, not a second calculation.
- [ ] **Step 4: Add `test_metric_outcome_failure_branches`.** Exercise one class, singleton classes, invalid/oversized k, nonfinite scorer output, caught CV exceptions, recall/mAP empty relevance and clustering/silhouette eligibility. Assert reason codes, exact public `None` returns and unchanged ordinary scores. CLI-level missing labels or skipped families are recorded later by the command, not invented inside an evaluator that never ran.
- [ ] **Step 5: Run the task gate.** `python -m pytest tests/test_evaluator.py -q` must pass, including existing five-fold shuffled stratification and unavailable-value tests. Review trainer call sites and unchanged CSV adapters. Review the diff; do not commit.

## Task 6: Versioned digests, mappings and atomic artifact primitives

**Files:**
- Modify: `src/otuformer/data/provenance.py`, `src/otuformer/utils/io.py`.
- Create: `tests/test_data_provenance.py`.
- Reuse the pattern, not an assumed shared helper: `src/otuformer/vision/export.py` local atomic JSON writer.

**Interfaces:** Consumes `DataBundle`, `ExtractionResult` and `MetricOutcome`.

```python
canonical_digest(value: JsonValue) -> str
build_data_records(bundle: DataBundle, frame: pd.DataFrame) -> list[dict]
build_validation_record(bundle: DataBundle) -> dict
build_run_record(*, requested: dict, bundle: DataBundle | None,
                 execution: ExtractionResult | None, evaluation: dict,
                 status: str, started_at: str, ended_at: str | None,
                 artifacts: list[dict], failure: dict | None = None) -> dict
write_json_atomic(data: dict, path: Path, indent: int = 2) -> None
```

`build_run_record` is a serialization function, not a job manager; Task 7 owns run lifetime. Existing `utils.io.write_json` is not silently replaced. `build_data_records` fails on length/order/ID mismatch rather than joining ambiguous IDs.

```python
rows = build_data_records(bundle, frame)
assert [r["embedding_row"] for r in rows] == list(range(1, len(frame) + 1))
assert [r["output_id"] for r in rows] == frame["id"].tolist()
assert all(r["schema_version"] == 1 for r in rows)
```

- [ ] **Step 1: Write failing `test_provenance_digests_and_mapping`.** Assert one-based `embedding_row`, exact frame order, IDs/keys/raw refs/dimensions/digests/metadata/source-local IDs, `schema_version == 1`, no duplicated annotation payload and absent-versus-empty preservation. Assert ambiguous or missing rows fail.
- [ ] **Step 2: Run red.** `python -m pytest tests/test_data_provenance.py::test_provenance_digests_and_mapping -q` must fail on absent serialization helpers.
- [ ] **Step 3: Implement named digest payloads.** `selection_metadata_sha256_v1` hashes `{selector_columns, records}`, where records is the ordered list of `{image_key, metadata, coco_images}`. Raw selector references and output IDs remain in the mapping, not this logical digest, because absolute CSV references may contain a machine-local root; original CSV bytes have their separate source hash. Preserve metadata strings as data, without guessing whether arbitrary custom cells are paths; `data_content_sha256_v1` hashes ordered `{image_key, image_sha256, masks}` rows, where masks list `{source_id, image_id, sha256}` in fixed role/image-ID order. Canonical bytes are UTF-8 JSON with `sort_keys=True`, `ensure_ascii=False`, `separators=(",", ":")`, `allow_nan=False`. Exclude access roots/times. Preserve original-byte SHA-256 separately for each CSV/JSON/image/mask/used weight file. Tests pin exact canonical bytes and independently calculated digests; no legacy hash is reused under these names.
- [ ] **Step 4: Implement validation/run records.** Preserve all issues and polygon diagnostics, selection/source-wide versus selected-only scope, counts, methods/versions and only applicable tolerances. Record local requested/resolved paths separately from portable output-relative artifact paths; source hashes/category definitions, label schema, selected/evaluated/sampled keys/seeds and all actual feature settings belong to the run. Serialize metric names/values/status/reasons and per-family protocol from the actual Task 5 evaluation context, not a generic claim of cosine/5-fold evaluation: kNN uses L2-normalized vectors and cosine distance; recall/mAP use cosine similarity and exclude the query itself; linear probing uses L2-normalized inputs and LogisticRegression, not a cosine-distance classifier; clustering uses Euclidean KMeans on L2-normalized vectors, while silhouette uses cosine distance with the existing ground-truth label grouping. kNN/probing use shuffled stratified CV with `random_state=42`, at most five folds and actual `n_splits=min(5, min_class_count)` when eligible; record unavailable CV with its reason otherwise. Record actual k/estimator settings, class order, protocol normalization and any thresholds plus their selection data. Current metrics choose no decision threshold, so emit `{status: not_applicable, reason: no_threshold_selected}` rather than inventing one. Test two-fold and five-fold fixtures, unavailable CV and distinct per-family distance semantics. Declare group split, held-out test use, spatial/task/geometry/calibration contracts explicitly absent with `{status, reason}` using the spec's statuses, not guessed IDs/scores. Include UTC-aware timestamps, package/dependency versions and existing best-effort Git dirty/commit hint; do not claim dirty bytes are reproducible.
- [ ] **Step 5: Add `test_atomic_json_preserves_previous_file_on_interruption`.** Use a same-directory temporary file, complete UTF-8 JSON then `os.replace`; inject serialization and replace failures, assert the previous destination is intact and temporary files are removed. Test keyboard interruption cleanup. Keep canonical digest bytes distinct from pretty artifact bytes; outputs reject NaN/Infinity. Do not promise power-loss durability or a group transaction.
- [ ] **Step 6: Run the task gate.** `python -m pytest tests/test_data_provenance.py tests/test_utils.py -q` must pass. Relocating identical input/output roots must preserve portable identities and logical digests; changing an absolute raw reference to the relocated root changes the original CSV file hash/output ID but not the root-relative selection digest. Modifying metadata/order/selected bytes changes the appropriate logical digest. Review the diff; do not commit.

## Task 7: Safe additive extract orchestration, evaluation and publication

**Files:**
- Modify: `src/otuformer/cli/extract.py`, `src/otuformer/cli/constraints.py`.
- Create: `tests/test_extract_data_cli.py`.
- Reuse unchanged: `cli_schema.check_output_safety`, `default_protected_paths`, `utils.io.prepare_output_dir`, `TeeLogger`.

**Interfaces:** Consumes Tasks 1–6. In `cli/constraints.py`, add import-light `validate_extract_data_options(*, label_csv: object = None, coco_instances: object = None, coco_keypoints: object = None, coco_panoptic: object = None, panoptic_masks_dir: object = None) -> None`, called both from `validate_constraints('extract', ...)` and the real CLI preflight. It enforces the panoptic pair rule only (there is one selector, so no selector conflict exists), not geometry or label eligibility. No serialized schema fields are added.

For a fresh nonexistent output fixture:

```python
assert not out.exists()
result = runner.invoke(app, ["extract", "--checkpoint", str(checkpoint),
    "--input-images-dir", str(images), "--out-dir", str(out),
    "--coco-panoptic", str(panoptic_json)])
assert result.exit_code != 0
assert not out.exists()
```

- [ ] **Step 1: Write failing `test_extract_new_path_options_and_preflight`.** Assert the four new declared Path options exist (`--coco-instances`, `--coco-keypoints`, `--coco-panoptic`, `--panoptic-masks-dir`), that `--label-csv` alone selects the shared path, that no `--manifest` option exists, panoptic JSON/root require each other, and invalid combinations fail before output clearing/model loading. For a pre-existing nonempty output fixture, include `--overwrite` and assert its sentinel file/content remains unchanged; `not out.exists()` applies only to the fresh-path case. Planned deviation (approved 2026-10-30): the separate `--manifest` option was dropped because one CSV file already carries the selector, labels and metadata. `--label-csv` now selects the shared path on its own; plain directory runs are the only legacy path. No boolean mode flag or `dataset` command is introduced.
- [ ] **Step 2: Run red.** `python -m pytest tests/test_extract_data_cli.py::test_extract_new_path_options_and_preflight -q` must fail because the new options are not yet declared.
- [ ] **Step 3: Implement the new-path branch and safety order.** Pure option validation → read-only output-overlap check against default protected roots plus images/CSV/COCO/mask roots, checkpoint/ONNX/external tensor inputs and existing read-only pooling inputs → normal output-dir preparation only after those checks → logging and atomic `running` record → shared reader/validation → extraction. A nonexistent generated pool destination under this run's output root is an output, not a protected input; an existing pool that would be consumed must not be deleted by overwrite. File/source validation occurs before model creation/download/inference; it may write a failure report only after output safety succeeds. Do not use `--overwrite` to override containment. An unsafe root produces no report there. Existing legacy branch/selection remains unchanged.
- [ ] **Step 4: Add `test_extract_attaches_each_annotation_role`** with box/polygon/RLE/keypoint/panoptic synthetic fixtures; assert valid sources attach without affecting global vectors, label targets or row order. Add `test_extract_records_evaluation_uses_one_parse`. Derive label/UMAP arrays from the records and frame order, never a second pandas read. Preserve the existing seeded sorted sample cap and UMAP thresholds; record the sampled and excluded keys explicitly. For new partially labeled input, sample the selected rows once for visualization as today, then restrict supervised metrics to the nonempty labels in that sampled set; record labeled rows excluded by sampling separately from missing labels. CSV selectors enable the existing optional UMAP flow; COCO-only does not. Attention requirements remain at the actual missing-pool branch. Call Task 5's private results once per metric family; retain legacy CSV field names and empty unavailable values. Record command skips and caught outer failures without fabricating a reason.
- [ ] **Step 5: Implement status and artifact publication.** Stage `embeddings.csv` and `data_records.jsonl` under a private directory within the approved output root. Check their one-to-one order before promoting either. Write `data_validation.json` atomically, preserving information-only diagnostics. Complete optional metrics/UMAP with their existing failure policy. Promote the embedding/mapping pair only after successful extraction; on a caught promotion failure/interruption, remove only destinations created by this invocation and leave no advertised final pair. A hard kill may leave one file, but the record remains `running`, never success.
- [ ] **Step 6: Finalize the run honestly.** Record actual metrics/visualization status and used/ignored settings; hash finalized produced outputs other than `run_provenance.json`. Close logging and restore stdout/stderr before hashing a log if it is listed as a finalized artifact; never modify a file after recording its final hash. Publish `completed` last and print success only after final publication, to the restored console rather than the closed/hashed log. On `Exception` use `failed`, on `KeyboardInterrupt` use `interrupted`, preserve/re-raise the original failure, clean private staging and any final pair created by this invocation before a completed record was successfully published, restore stdout/stderr and avoid claiming failed metadata writes succeeded. If failure-record publication also fails, report both errors and leave the last successful non-completed record; never mask the original cause. No legacy run is required to emit the three new artifacts.
- [ ] **Step 7: Add `test_extract_failure_safety_and_status`** for invalid annotations, missing optional decoder, label/identity failure, changed image/model/mask bytes, output equals/contains/is inside an input root, symlink overlap, overwrite with protected sources, extraction error, keyboard interruption, rename failure and final-record write failure. Assert no inference on invalid data; no source writes; no false completed status; no final embedding/mapping pair for caught failure; previous protected inputs unchanged. Include no-new-options regressions for logs, CSV and UMAP.
- [ ] **Step 8: Run the task gate.** `python -m pytest tests/test_extract_data_cli.py tests/test_extract_records.py tests/test_data_provenance.py tests/test_cli_smoke.py -q` must pass. Test only synthetic images/tiny or mocked checkpoints; no example extraction or real analysis. Review the diff; do not commit.

## Task 8: Runtime discovery, bilingual docs and workflow Skill contracts

**Files:**
- Modify: `src/otuformer/cli/constraints.py` declared path roles/unresolved validation notes.
- Extend: `tests/test_cli_schema.py`, `tests/test_docs.py`, `tests/test_workflow_skill.py`.
- Modify: `README.md`, `README.cn.md`, `docs/commands/extract.md`, `extract.cn.md`.
- Modify: the four Skill instruction/reference files named in File Ownership.
- Inspect but do not add derived-mode fields to `src/otuformer/cli_schema.py` or `skills/otuformer-workflow/scripts/export_cli_schema.py`.

**Interfaces:** Existing exported schema/proposal structure, `SCHEMA_VERSION == 1`, declared-option discovery and generic path protection remain intact. Use existing path-role vocabulary and existing unresolved-condition lists; do not invent schema properties.

```python
entries = parameter_index("extract")
assert {"coco_instances", "coco_keypoints", "coco_panoptic",
        "panoptic_masks_dir"} <= entries.keys()
assert "manifest" not in entries
assert SCHEMA_VERSION == 1
with pytest.raises(ValueError):
    validate_parameters("extract", {"input_images_dir": "images", "checkpoint": "c.pt",
                                    "coco_panoptic": "panoptic.json"})
```

- [ ] **Step 1: Write failing `test_extract_data_schema_and_protection`.** Assert all five new options have correct optional path/default/help metadata and safe argv omission, paired/conflicting option rules match the CLI, and JSON/CSV/PNG roots are protected even with overwrite. Export schema/proposals without torch/pycocotools/model initialization and without filesystem writes. Assert no derived mode key appears anywhere in schema/proposal serialization and the registered command set is unchanged.
- [ ] **Step 2: Run red.** `python -m pytest tests/test_cli_schema.py -k extract_data -q` must fail on missing new-path constraints/coverage, not fail because schema payload shape was intentionally changed.
- [ ] **Step 3: Update supplemental constraints within the existing contract.** New CLI declarations provide types/defaults; existing schema discovery exports them. Add import-light path roles, pairing/conflict validation and unresolved geometry/attention requirements. Do not resolve model/data-dependent requirements during schema export. Generic `_protected_paths_from_values` protects the newly declared paths; add a code change there only if the test establishes a real gap, preserving its public contract.
- [ ] **Step 4: Add/update bilingual extract documentation.** Cover every new option, interactions, the single-CSV strict mode versus directory enumeration, ID/sample rules, source-local identity, box/polygon/RLE/keypoint/panoptic scope, explicit panoptic roots, lazy extra, validation scope, all three new outputs and failure/status interpretation. Keep the spec as design authority, not as user CLI reference. Keep historical Version Notes intact; do not invent a release version for unreleased behavior. README twins gain short scope/navigation only, with no duplicated full parameter table or new teaching data.
- [ ] **Step 5: Update Skill guidance and tests.** The Skill reads fresh schema and presents the complete existing card, then appends explanatory `legacy data path: existing semantics` or `new data path: strict CSV semantics` based on discovered options and the proposed combination (the legacy label is directory enumeration only, because one CSV always means the strict path); this notice is not a new card schema field or adapter payload key. If installed schema lacks the new options, do not propose them. Explain NA/leading-zero behavior, protected annotation inputs, no biological-absence inference and incomplete/running artifacts. Update the existing extract reference's overly broad basename/two-class UMAP wording, not the teaching recipe or workflow graph. Preserve per-step/overwrite/install consent and no automatic conversion/rerun rules. Skill instruction tests establish static guidance consistency, not proof that an agent will obey the notices or approval gates.
- [ ] **Step 6: Run the task gate.** `python -m pytest tests/test_cli_schema.py tests/test_docs.py tests/test_workflow_skill.py -q` must pass, including existing packaging/installed-copy checks. Do not install/copy the Skill into a global agent directory during execution. Confirm README/command twins and live `extract --help` agree. Review the diff; do not commit.

## Task 9: Foundation regression and handoff gate

**Files:** Verify the whole approved change set. Do not add a generic framework, CI pipeline or release automation to satisfy this gate.

**Interfaces:** All prior task gates and the complete spec. A task with failing tests, skipped promised reader coverage or an unresolved runtime/dependency error remains incomplete.

- [ ] **Step 1: Review complete spec coverage** using the matrix below; inspect first/middle/last row maps and used-file hashes from portable integration artifacts. All four annotation readers must be functional through Python and attached through extract, not stubs.
- [ ] **Step 2: Run focused integration/regression checks** in the separately approved test environment:

```bash
python -c "import pycocotools.mask, onnx, onnxruntime"
python -m pytest tests/test_data_reader.py tests/test_data_coco.py tests/test_data_masks.py tests/test_data_provenance.py tests/test_extract_records.py tests/test_extract_data_cli.py tests/test_extractor.py tests/test_evaluator.py tests/test_cli_schema.py tests/test_workflow_skill.py tests/test_docs.py tests/test_utils.py -q
```

Expected: zero failures; no mask/backend skips treated as support. Repeat the dependency-absent subprocess case from Task 3. Record versions, exact commands and any platform limitation. Installation or changing the interpreter is a separate approval, not an implicit recovery action.

- [ ] **Step 3: Run full regression.** `python -m pytest -q` must pass, preserving training/resume, sparse-label/pseudo-feedback hashes, register/checkpoint compatibility, cluster/annotate log parsing, ONNX, pooling and global-vector behavior. A pre-existing failure must be documented and resolved or explicitly accepted by the operator, never relabeled as a pass.
- [ ] **Step 4: Check repository scope/integrity.** `git diff --check`, `git status --short`, `git diff --stat`, and a read-only comparison of example asset hashes must show no data/example/checkpoint/version edits or accidental generated artifacts. Inspect all new untracked files as well as tracked diffs. Include a task-indexed changed-file/hunk inventory in the handoff report, marking files shared by several tasks and pre-existing operator changes separately; this supports later commit splitting or targeted rollback without creating commits, stashes, resets or a new repository artifact. Do not imply that reverting a shared file cleanly reverts one task. No workstation paths, real annotations or private metadata enter tests/docs. Verify local links, JSON/CSV examples and both documentation languages.
- [ ] **Step 5: Hand off evidence for review.** Report tests actually run, untouched legacy contracts, limitations and outstanding approval gates. Ask the operator to review before any commit. Do not run the external 3,260-image dataset, a real checkpoint, decoder consumer comparisons or a full biological analysis; those require a separate request, parameter review and explicit output directory. Synthetic tests establish ingestion engineering behavior, not anatomical correctness, group-safe evaluation or model performance.

## Spec-to-Task Coverage Matrix

| Design requirement | Implementation/test owner |
|---|---|
| §1–2 scope, approval/proposal distinctions, no new command/converter | Global Constraints; Tasks 7–9 |
| §3.1–3.2 selection, ordering, identities, containment and aliases | Tasks 1–2; Task 4 root/input seam; Task 7 source protection |
| §3.3 raw metadata, once-only parsing and task label requirements | Tasks 1, 4, 7–8 |
| §3.4 independent relationship metadata; measurements deferred | Task 1 metadata/digest tests; Task 6 digests; Task 8 documentation |
| §4.1 source-local roles, preserved payloads and combined files | Task 2 indexing/selection tests |
| §4.2 boxes, polygons, RLE and non-repair policy | Tasks 2–3; no percent/topology hard gate |
| §4.3 keypoint names, visibility, count, skeleton and bounds | Task 2 keypoint tests |
| §4.4 panoptic roots, RGB IDs, void and pixel agreement | Task 3 panoptic tests; Task 7 attachment |
| §4.5 availability/custom ignore boundary | Tasks 2–3 and Skill/docs in Task 8 |
| §5 issue severities, source-wide vs selected-only, raster diagnostics | Tasks 1–3 and Task 6 report serialization |
| §6.1 CSV mode, records/root seam, public DataFrame and actual settings | Tasks 4, 7–8 |
| §6.2 lazy optional dependency, no legacy startup imports | Task 3 dependency-present/absent tests; Task 8 schema import checks |
| §7.1 ID distinctions, sample, mapping and three additive files | Tasks 4, 6–7 |
| §7.2 content/weight hashes, drift, effective feature/normalization operations | Tasks 1, 4, 6–7 |
| §7.2 metric values/reasons, label schema, actual per-family distance/normalization/CV and thresholds | Task 5 outcome context; Task 6 Step 4 serialization/tests; Task 7 selection/skips and unchanged CSV |
| §7.2 canonical digests versus atomic publication and interruption | Tasks 6–7 |
| §8 examples | Synthetic fixtures in Tasks 1–3 and Task 6 mapping test |
| §9 groups 1–5 and 11 (reader/integration) | Tasks 1–8 |
| §9 groups 6–8 (legacy/artifact regressions) | Tasks 4–9 |
| §9 groups 9–10 (evidence limits and read-only real data) | Global Constraints; Task 9 handoff, not a real-analysis task |
| §10 documentation, Skill/schema discovery and independent gates | Tasks 8–9; no automatic install/version/commit |

## Plan Self-Review and Approval Boundary

Author self-review maps the complete spec to tasks above and checks task/interface consistency, review-focus tests, exact paths and proportionality. It explicitly separates private runtime plumbing and digest definitions from CLI schema/version contracts. This is author self-review, not a fresh independent audit. No subagent review is implied or authorized.

The shared data types and exact helper names above are the implementation proposal being reviewed. The separate source design remains the authority for behavior. In particular, schema-field/version changes, an additional public mode selector, bbox-format compatibility, all-command provenance migration or changing legacy label parsing require a new decision, not improvisation during execution.

**Next gate:** operator review of this plan and confirmation of the governing design, then explicit execution approval, execution-method choice and resolution of the isolated-test-environment prerequisite above. Accept repeated attention-query byte verification costs explicitly; keep the external-data ONNX original-path verification limitation visible. Native inline execution avoids delegation; subagents require an explicit operator request. Nothing in this plan authorizes a commit, push, release, dependency installation or real-data analysis.
