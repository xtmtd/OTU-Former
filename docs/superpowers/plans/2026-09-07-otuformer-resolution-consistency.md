# OTU-Former Resolution & Preprocessing Consistency Implementation Plan

Date: 2026-09-07

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the model input resolution and the deterministic evaluation preprocessing consistent across pretraining, fine-tuning, extraction, CAM, and ONNX export; make the three size options auto-resolving by default; add a `--eval-transform` option on `extract` and `cam` to select the evaluation preprocessing protocol; fix the CAM heatmap's spatial mapping for non-square images; and bump the version to 0.3.0. Training augmentation profiles are unchanged.

**Self-contained:** This plan does not depend on the training-augmentation plan. It implements the minimal `_estimate_background_color`, `resolve_training_image_size`, and size-validation helpers itself. The `augmentation_config.*` precedence levels in `resolve_training_image_size` are forward-compatible no-ops until that plan lands.

**Out of scope (explicit):** The *semantics* of the CAM target. The current CAM wrapper returns the raw CLS feature and treats its argmax dimension as a pseudo-class (`pred_class`/`pred_prob` in `cam.py`); this plan fixes only *where* the heatmap is drawn, not *what* it explains. It does not restore the ArcFace head / class mapping, and the resulting figure is a gradient-attribution heatmap, not a verified class-discriminative CAM. This naming and semantics are tracked as a separate follow-up issue, not corrected here.

**Tech Stack:** Python 3.11+, PyTorch 2.1+, torchvision 0.16+, timm 1.0.27, Pillow, Typer, pytest 8+.

---

## Global Constraints

- Introduce two **explicitly named** deterministic evaluation protocols and select between them with a per-subcommand option:
  - `center_crop_eval_transform(size)`: `Resize(size, bicubic)` (shorter side) → `CenterCrop(size)` → `ToTensor` → ImageNet `Normalize`. This is the current `_make_eval_transform` behavior, renamed. **Default.**
  - `whole_specimen_pad_transform(size)`: pad the source image to a square canvas with the edge-median background fill (preserving aspect ratio, no distortion) → `Resize(size, size)` → `ToTensor` → `Normalize`. Implemented and tested now, but **not the default**; reserved for a later promotion once it wins on downstream metrics.
- Keep `_make_eval_transform` as a one-line backward-compatible alias of `center_crop_eval_transform` during the transition; migrate call sites to the new name.
- Provide a shared mapping and builder: `EVAL_TRANSFORMS = {"center-crop": center_crop_eval_transform, "whole-specimen-pad": whole_specimen_pad_transform}` and `build_eval_transform(name, size) -> transforms.Compose` with an explicit error on unknown names.
- Add one selection option `--eval-transform {center-crop,whole-specimen-pad}` (default `center-crop`) to `extract` and `cam` only. It controls the deterministic inference/evaluation preprocessing:
  - `extract`: the dataset transform (both PyTorch and ONNX paths).
  - `cam`: the preprocess and the matching display transform.
- `pretrain` / `finetune` periodic evaluation (metrics/UMAP) stays fixed on `center_crop_eval_transform`; there is no `--eval-transform` there. The finetune training dataset also stays `center_crop_eval_transform`; training-side preprocessing is owned by the (deferred) training-augmentation plan.
- `export` does not preprocess images (it freezes the graph at a fixed input size), so it does not gain `--eval-transform`.
- Unify extraction, periodic evaluation, fine-tuning `none`, and CAM onto `center_crop_eval_transform` (the shared deterministic protocol); CAM stops using `resolve_model_data_config`, which resolves `crop_pct=0.9` for the supported ViTs. `extract` and `cam` may additionally select `whole-specimen-pad` via `--eval-transform`.
- Two sizes are distinct and must not be conflated:
  - `checkpoint_size`: the model-construction `img_size`, always the checkpoint's recorded training size so the positional embeddings load exactly.
  - `requested_input_size`: the transform resolution (extract) or ONNX dummy-input resolution (export). `auto` resolves to `checkpoint_size`; an explicit integer overrides it and relies on `dynamic_img_size=True` to interpolate the positional embeddings at forward time (e.g., a 224-trained checkpoint may be extracted or exported at 384).
- The three size options accept exactly `auto` or a positive integer, and default to `auto`:
  - `--global-crop-size` (pretrain): `auto` → resume inherits the checkpoint's recorded size; a new run uses the backbone's native `default_cfg["input_size"]`. On resume, an explicit value equal to the checkpoint size is allowed; an explicit different value fails before dataset/model construction and requires a new run.
  - `--extract-size` (pretrain / finetune / extract): `auto` → the checkpoint's recorded training size. Whenever `--onnx-path` is used, the ONNX graph governs instead (see the ONNX contract below).
  - `--imgsz` (export): `auto` → the checkpoint's recorded training size (model construction uses `checkpoint_size`; the exported dummy input uses `imgsz`).
  - Help text lists common integer examples (`224`, `384`, `448`) and states the default is `auto`. `518` may be mentioned only as a patch-14 example with an explicit "patch-14 models" qualifier.
- ONNX size contract (applies whenever `--onnx-path` is used, regardless of `--checkpoint`, because ONNX overrides the PyTorch path):
  - `auto` → the static H/W read from the ONNX graph input;
  - an explicit `--extract-size` must equal the graph's static H/W exactly;
  - a dynamic, non-NCHW, or non-square (`H != W`) input is rejected outright (no explicit size can make such a graph valid).
  - Validate the ONNX input is NCHW, `H == W`, and both spatial dims are positive integers.
- A resolved size must also satisfy the backbone's geometric constraint: for patch-based (ViT-like) backbones, the size must be divisible by the patch size. Validate after model resolution and fail with an error that names the model and the patch size (e.g., "input size 518 is not divisible by patch size 16 for vit_tiny_patch16_224; nearest valid: 512"). CNN backbones (no `patch_embed`) skip this check.
- The recorded training size is the single source of truth, resolved from checkpoint metadata (fine-tune `augmentation_config.image_size`, pretrain `augmentation_config.global_crop.size`, this plan's fine-tune `config.image_size`, legacy `args.global_crop_size`, then `224`) — never from the timm model name alone, because `OTUFormerEncoder(img_size=...)` overrides a backbone's native resolution.
- Every `OTUFormerEncoder` instantiation that loads a checkpoint (extract / export / CAM / fine-tuning `run_finetune`) must construct the model with `img_size=checkpoint_size` before `load_state_dict`; a positional-embedding shape mismatch otherwise raises a `RuntimeError` (PyTorch `strict=False` does not ignore size mismatches). The `requested_input_size` may differ and is handled by `dynamic_img_size=True` interpolation at forward time.
- Fine-tuning checkpoints must persist the resolved training size so downstream extraction/export/CAM can recover it.
- CAM heatmaps are overlaid on the **full original image** via the correct inverse of the selected preprocessing; with `center-crop` the heatmap is confined to the model's field of view using a dedicated binary FOV mask (out-of-view pixels are dimmed/desaturated directly, never routed through the heatmap colormap), and with `whole-specimen-pad` the heatmap covers the whole specimen (nothing is cropped). Never stretch a crop heatmap to the full frame.
- Version bumps to `0.3.0` in `pyproject.toml` and `src/otuformer/__init__.py`.
- Code, comments, and documentation remain in English except `README.cn.md`.
- Do not commit unless the user separately approves a commit after implementation and verification.

---

### Task 1: Named protocols, background estimation, and unified preprocessing

**Files:**
- Modify: `src/otuformer/training/dataset.py` (rename `_make_eval_transform` → `center_crop_eval_transform`; add alias, `whole_specimen_pad_transform`, `_estimate_background_color`, `EVAL_TRANSFORMS`, `build_eval_transform`)
- Modify: `src/otuformer/vision/cam.py` (replace `resolve_model_data_config`-based `build_eval_transforms` with `build_eval_transform`)
- Modify: `src/otuformer/training/trainer.py`, `src/otuformer/embedding/extractor.py` (migrate `_make_eval_transform` call sites)
- Test: `tests/test_training_dataset.py`, `tests/test_cam.py`, `tests/test_extractor.py`

**Interfaces:**
- Produces: `center_crop_eval_transform(size: int) -> transforms.Compose`.
- Produces: `whole_specimen_pad_transform(size: int) -> transforms.Compose`.
- Produces: `_estimate_background_color(image: Image.Image) -> tuple[int, int, int]` — per-channel median of all four outer edges; fallback `(124, 116, 104)` (8-bit ImageNet mean) on failure.
- Produces: `EVAL_TRANSFORMS: dict[str, Callable[[int], transforms.Compose]]` and `build_eval_transform(name: str, size: int) -> transforms.Compose`.
- Produces: `_make_eval_transform = center_crop_eval_transform` (transition alias).

- [ ] **Step 1: Add failing protocol tests**

Assert `center_crop_eval_transform` equals the old composition (`Resize`/`CenterCrop`/`ToTensor`/`Normalize` in order); `whole_specimen_pad_transform` preserves aspect ratio (pads with the edge-median fill, no distortion) and returns a fixed-size tensor; `_estimate_background_color` returns the per-channel median of the four edges and the documented fallback on failure; `build_eval_transform` raises on an unknown name and returns the correct callable for both names. Assert CAM's preprocess and extract's transform are the same function for a given protocol.

- [ ] **Step 2: Run the tests and verify they fail**

```bash
pytest tests/test_training_dataset.py tests/test_cam.py tests/test_extractor.py -k "protocol or pad or background or build_eval" -v
```

Expected: collection/assertion failure because the new names and CAM unification do not exist.

- [ ] **Step 3: Implement the named protocols, background estimation, and CAM unification**

Rename `_make_eval_transform` to `center_crop_eval_transform`, keep the alias, add `whole_specimen_pad_transform`, `_estimate_background_color` (edge median over `np.asarray(image.convert("RGB"))`, catching conversion/shape/empty failures → fallback), `EVAL_TRANSFORMS`, and `build_eval_transform`. Replace CAM's `build_eval_transforms` with `build_eval_transform(name, size)` plus a matching display transform (`Resize(size, bicubic) → CenterCrop(size)` for center-crop; `pad-to-square → Resize(size, size)` for whole-specimen-pad). Remove the `resolve_model_data_config` import from `cam.py`.

- [ ] **Step 4: Run protocol tests and verify they pass**

```bash
pytest tests/test_training_dataset.py tests/test_cam.py tests/test_extractor.py -k "protocol or pad or background or build_eval" -v
```

Expected: all selected tests pass.

---

### Task 2: Fix CAM spatial mapping for non-square images

**Files:**
- Modify: `src/otuformer/vision/cam.py` (`process_image`)
- Test: `tests/test_cam.py`

**Interfaces:**
- Produces: a helper that maps the model-resolution CAM back to the original image coordinates for **both** protocols, plus a binary FOV mask for `center-crop`.
- Changes: `process_image` overlays on the full original image via the correct inverse of the selected preprocessing.

- [ ] **Step 1: Add failing geometry tests**

With `center-crop` and a synthetic non-square image (e.g., 400×800) containing a bright centered square patch, assert the heatmap's activation lands on the patch's original-image coordinates and the out-of-view region is dimmed (not colormap-rendered). With `whole-specimen-pad`, assert the heatmap covers the whole specimen region (no crop-confined blanking) and is aspect-correct. Assert square inputs are unchanged for both protocols.

- [ ] **Step 2: Run the tests and verify they fail**

```bash
pytest tests/test_cam.py -k "non_square or mapping or field_of_view or pad" -v
```

Expected: failure because `cam_on_full` currently stretches the crop heatmap to the full frame.

- [ ] **Step 3: Implement correct inverse mapping**

For `center-crop`: compute the center-crop rectangle within the resized image, place the model-resolution CAM into that rectangle on a canvas the size of the resized image, and resize the canvas back to `(W_orig, H_orig)`. Build a **binary FOV mask** from the same rectangle and, when compositing, dim/desaturate out-of-view pixels directly (do not pass them through `show_cam_on_image`'s colormap, which would render CAM=0 as the colormap's lowest color). Overlay the colormap only inside the FOV. For `whole-specimen-pad`: resize the CAM to the padded square, crop the centered original-image region, and overlay on the full original (whole specimen is in view, so no FOV blanking). Keep the combined `original | overlay` figure. Keep the saved `.npy` as the raw model-resolution normalized map (backward compatible).

- [ ] **Step 4: Run CAM tests and verify they pass**

```bash
pytest tests/test_cam.py -v
```

Expected: all CAM tests pass, including the existing ViT reshape/architecture tests.

---

### Task 3: Pass `img_size`, resolve size, and persist it through the whole chain

**Files:**
- Modify: `src/otuformer/training/trainer.py` (`run_finetune`: resolve size, pass `img_size` + `image_size`, persist `config.image_size`; consolidate `_infer_backbone_image_size`)
- Modify: `src/otuformer/embedding/extractor.py` (`_load_model`, dataset defaults, ONNX-only auto size)
- Modify: `src/otuformer/vision/export.py` (`export_to_onnx`, remove duplicated `_infer_backbone_image_size`)
- Modify: `src/otuformer/vision/cam.py` (`load_model_from_checkpoint`)
- Add: `src/otuformer/utils/size.py` (shared resolution + validation helpers)
- Test: `tests/test_extractor.py`, `tests/test_export.py`, `tests/test_cam.py`, `tests/training/test_pretrain_alignment.py`

**Interfaces:**
- Produces: `resolve_training_image_size(checkpoint: dict | None) -> int` — precedence: fine-tune `config.augmentation_config.image_size` → pretrain `config.augmentation_config.global_crop.size` → this plan's fine-tune `config.image_size` → legacy `args.global_crop_size` → `224`; validate positive integer. The first two levels are forward-compatible no-ops until the augmentation plan writes them.
- Produces: `resolve_backbone_native_size(model_name: str) -> int` — read `default_cfg["input_size"]` via a `pretrained=False` timm instantiation (no weight download).
- Produces: `resolve_backbone_patch_size(model) -> int | None` — read the patch size from `model.backbone.patch_embed`; return `None` for CNN backbones (no divisibility constraint).
- Produces: `validate_input_size(size: int, model, model_name: str) -> None` — raise a clear error naming the model and patch size when `size % patch_size != 0`.
- Produces: one shared `infer_backbone_image_size(model, fallback)` replacing the two current copies.

- [ ] **Step 1: Add failing size-propagation and persistence tests**

Create a checkpoint fixture with `args.global_crop_size=32` (legacy). Assert extract/export/CAM instantiate `OTUFormerEncoder` with `img_size=32` and `model.backbone.pos_embed.shape[1]` matches the checkpoint's pos_embed token count. Add a 224-checkpoint + explicit `--extract-size 384` / `--imgsz 384` case asserting the model is still constructed at 224 (pos_embed loads exactly) while the transform/dummy input uses 384. Assert `run_finetune` resolves the pretrain checkpoint size, passes `img_size=32` to `OTUFormerEncoder` and `image_size=32` to `MetricDataset`, and persists `config.image_size == 32` in the finetune checkpoint. Assert a non-224 finetune checkpoint then resolves back to `32` (not `224`) through `resolve_training_image_size`. Assert `resolve_training_image_size` precedence with a table test (fine-tune size, pretrain size, `config.image_size`, legacy args, fallback 224; reject bool/negative/non-integer). Assert `validate_input_size` rejects a non-divisible size (e.g., 518 for patch16) with an error naming the model and patch size, and passes for a CNN backbone.

- [ ] **Step 2: Run the tests and verify they fail**

```bash
pytest tests/test_extractor.py tests/test_export.py tests/test_cam.py tests/training/test_pretrain_alignment.py -k "size or pos_embed or img_size or patch" -v
```

Expected: failure because extract/export/CAM/finetune do not pass `img_size` (a shape mismatch would otherwise raise), and the finetune checkpoint does not persist a size.

- [ ] **Step 3: Implement shared resolution, validation, and wiring**

Add `utils/size.py` with the resolution and validation helpers. Distinguish `checkpoint_size` (model construction) from `requested_input_size` (transform/dummy input). In `_load_model`, `export_to_onnx`, and `load_model_from_checkpoint`, construct the model with `img_size=checkpoint_size` **before** `load_state_dict`, and apply the transform/dummy input at `requested_input_size` (which may differ). In `run_finetune`, resolve `finetune_image_size` from the chosen checkpoint, pass `img_size=finetune_image_size` to `OTUFormerEncoder` and `image_size=finetune_image_size` to `MetricDataset`, and add `config.image_size = finetune_image_size` to the saved checkpoint. `export_to_onnx` must resolve `checkpoint_size` from the checkpoint (not from `patch_embed.img_size`) for model construction, while the exported dummy input uses the resolved `imgsz`. Consolidate the duplicated `_infer_backbone_image_size` into the shared helper and update both trainer and export call sites. Whenever `--onnx-path` is used, resolve the size from the ONNX graph per the ONNX contract (static H/W; explicit value must match; dynamic/non-NCHW/non-square rejected), and validate NCHW / `H == W` / positive integer spatial dims.

- [ ] **Step 4: Run size tests and verify they pass**

```bash
pytest tests/test_extractor.py tests/test_export.py tests/test_cam.py tests/training/test_pretrain_alignment.py -k "size or pos_embed or img_size or patch" -v
```

Expected: all selected tests pass.

---

### Task 4: Auto-size CLI convention and size validation

**Files:**
- Modify: `src/otuformer/cli/pretrain.py`, `src/otuformer/cli/finetune.py`, `src/otuformer/cli/extract.py`, `src/otuformer/cli/export.py`
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Produces: `--global-crop-size`, `--extract-size`, `--imgsz` each accept `auto` or a positive integer; effective default `auto`.
- Produces: a small shared `_parse_size(value, *, stage)` helper returning `int | None` (`None` = auto), raising `typer.BadParameter` on invalid input.
- Produces: `validate_input_size` applied to every resolved explicit size before dataset/model construction.

- [ ] **Step 1: Add failing CLI contract tests**

Assert `--help` for each command documents `auto`, the effective default, and at least `224`, `384`, `448` (with `518` only as a patch-14 example). Assert `--extract-size auto` and `--extract-size 224` both parse; `--extract-size 0` and `--extract-size abc` fail before training; `--global-crop-size 518` fails for a patch-16 model with an error naming the model and patch size. Assert `--resume checkpoint-224.pth --global-crop-size 384` fails before dataset/model construction with a clear "resume cannot change the input size" error, while an explicit value equal to the checkpoint size is allowed. Assert omitted values resolve to `auto` (None) in the forwarded namespace. Assert extraction with `--onnx-path` (with or without `--checkpoint`) resolves the size from the ONNX input shape: `auto` reads the static H/W, an explicit value must match it, and a dynamic/non-NCHW/non-square graph is rejected.

- [ ] **Step 2: Run CLI tests and verify they fail**

```bash
pytest tests/test_cli_smoke.py -k "size or auto or imgsz or extract_size or global_crop or patch" -v
```

Expected: failure because the options are still fixed integers (or `None` for `--imgsz`), help lacks `auto`, and there is no patch-size validation.

- [ ] **Step 3: Implement auto resolution and size validation**

Change each option to a `str` (or union) defaulting to `"auto"`, parse via `_parse_size`. In pretrain, `auto` global-crop-size → resume inherits the checkpoint's recorded size, else `resolve_backbone_native_size(args.model_name)`; on resume, reject an explicit value that differs from the checkpoint size before dataset/model construction. In extract and export, `auto` → `resolve_training_image_size(checkpoint)`; whenever `--onnx-path` is used, apply the ONNX contract (static H/W; explicit must match; dynamic/non-NCHW/non-square rejected). In pretrain/finetune periodic evaluation, `auto` extract-size → the model's own `img_size`. After each explicit size resolves, run `validate_input_size` against the backbone (patch-size divisibility) before dataset/model construction. Update `_format_user_command` handling for the new string values.

- [ ] **Step 4: Run CLI tests and verify they pass**

```bash
pytest tests/test_cli_smoke.py -v
```

Expected: all CLI tests pass.

---

### Task 5: `--eval-transform` selection option (extract and cam only)

**Files:**
- Modify: `src/otuformer/cli/extract.py`, `src/otuformer/cli/cam.py`
- Modify: `src/otuformer/embedding/extractor.py`, `src/otuformer/vision/cam.py` (accept and forward the protocol name)
- Test: `tests/test_cli_smoke.py`, `tests/test_cam.py`, `tests/test_extractor.py`

**Interfaces:**
- Produces: `--eval-transform {center-crop,whole-specimen-pad}` (default `center-crop`) on `extract` and `cam`.
- Produces: `args.eval_transform` forwarded to the dataset/CAM construction via `build_eval_transform(name, size)`.

- [ ] **Step 1: Add failing option tests**

Assert `--help` documents both values and the default. Assert `--eval-transform whole-specimen-pad` reaches `extract`/`cam` construction as the protocol name, and an invalid value fails before execution. Assert the default remains `center-crop`.

- [ ] **Step 2: Run the tests and verify they fail**

```bash
pytest tests/test_cli_smoke.py tests/test_cam.py -k "eval_transform" -v
```

Expected: failure because the option does not exist.

- [ ] **Step 3: Implement the option and forward it**

Add the option to `extract` and `cam` with validation against `EVAL_TRANSFORMS`. Forward the resolved name into `extract_embeddings` and `run_cam`. CAM builds both preprocess and display transform from `build_eval_transform`; extract builds its dataset transforms from it.

- [ ] **Step 4: Run option tests and verify they pass**

```bash
pytest tests/test_cli_smoke.py tests/test_cam.py tests/test_extractor.py -v
```

Expected: all selected tests pass.

---

### Task 6: Version bump and documentation

**Files:**
- Modify: `pyproject.toml` (`version = "0.3.0"`), `src/otuformer/__init__.py` (`__version__ = "0.3.0"`)
- Modify: `docs/superpowers/specs/2026-03-29-otuformer-design.md` (CLI contract for the three size options and the new `--eval-transform` option)
- Modify: `docs/superpowers/specs/2026-09-07-otuformer-training-augmentation-design.md` (update the §14 "pre-existing extract/CAM discrepancy" note into a cross-reference to this plan)
- Modify: `README.md`, `README.cn.md`

- [ ] **Step 1: Bump version**

Update both version locations to `0.3.0`.

- [ ] **Step 2: Update design and README documents**

Document the `auto` default and integer examples for the three size options (with `518` qualified as patch-14 only), the patch-size divisibility requirement, and the ONNX-only auto behavior; document `--eval-transform {center-crop,whole-specimen-pad}` on `extract` and `cam` (default `center-crop`, `whole-specimen-pad` reserved); describe the CAM field-of-view mask (center-crop) versus whole-specimen coverage (pad), and note that the CAM figure is a gradient-attribution heatmap, not a verified class-discriminative CAM. Update the augmentation design's §14 note to point at this plan instead of describing the discrepancy as unaddressed.

- [ ] **Step 3: Check documentation consistency**

```bash
rg -n "auto|center-crop|whole-specimen-pad|center_crop_eval_transform|whole_specimen_pad_transform|0.3.0|eval-transform|extract-size|global-crop-size|imgsz|patch" \
  README.md README.cn.md docs/superpowers/specs/2026-03-29-otuformer-design.md docs/superpowers/specs/2026-09-07-otuformer-training-augmentation-design.md
git diff --check
```

Expected: documents agree on auto defaults, protocol names, patch-size requirement, and the new option; `git diff --check` prints nothing.

---

### Task 7: Regression and non-224 end-to-end smoke

**Files:**
- Modify only if a real failure requires a scoped fix: files already listed in Tasks 1-5
- Regression: all files under `tests/`

- [ ] **Step 1: Run focused tests**

```bash
pytest tests/test_training_dataset.py tests/test_cam.py tests/test_extractor.py tests/test_export.py tests/test_cli_smoke.py tests/training/test_pretrain_alignment.py -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete suite**

```bash
pytest -q
```

Expected: all tests pass. Any failure in augmentation, loss pairing, or model behavior is a regression.

- [ ] **Step 3: Non-224 end-to-end smoke**

Train a tiny non-224 checkpoint (or synthesize one with `args.global_crop_size=32`), then drive pretrain resume → finetune → extract → export → CAM, asserting matching sizes everywhere, matching pos_embed shapes (no shape-mismatch crash), the finetune checkpoint persisting `config.image_size`, and — for both `--eval-transform center-crop` and `--eval-transform whole-specimen-pad` — a CAM figure whose heatmap is spatially correct (confined field of view versus whole-specimen coverage, respectively).

- [ ] **Step 4: Final verification without committing**

```bash
pytest -q
python -m compileall -q src
git diff --check
git status --short
```

Expected: tests and compilation pass; whitespace check empty; status lists only the intended implementation, test, script, design, README, plan, and version files. Stop and request separate approval before any commit.

---

## Acceptance Criteria

The implementation is acceptable when:

- `center_crop_eval_transform` and `whole_specimen_pad_transform` exist, are tested, and the default is `center-crop`.
- extract, periodic evaluation, fine-tuning `none`, and CAM default to the same `center_crop_eval_transform`; CAM no longer uses `resolve_model_data_config`. `extract` and `cam` can additionally select `whole-specimen-pad` via `--eval-transform`.
- `--eval-transform {center-crop,whole-specimen-pad}` (default `center-crop`) exists on `extract` and `cam`.
- CAM heatmaps are spatially correct for non-square images under both protocols (binary-FOV-confined field-of-view for center-crop; whole-specimen coverage for pad), and out-of-view pixels are dimmed without going through the heatmap colormap.
- The three size options default to `auto`, accept positive integers, and help lists `224`/`384`/`448` (with `518` qualified as patch-14 only).
- Model construction always uses `checkpoint_size`; an explicit `--extract-size`/`--imgsz` overrides only the input/export resolution and relies on positional-embedding interpolation (a 224 checkpoint can be extracted or exported at 384 without a shape mismatch).
- Every resolved explicit size is validated against the backbone's patch size with a clear error naming the model and patch size.
- `--global-crop-size auto` on a new run uses the backbone's native `default_cfg["input_size"]`; on resume it inherits the checkpoint size, and an explicit different value fails before dataset/model construction.
- Non-224 checkpoints flow through pretrain → finetune → extract → export → CAM with matching sizes and no positional-embedding shape-mismatch; the finetune checkpoint persists its resolved `config.image_size`.
- Whenever `--onnx-path` is used (with or without `--checkpoint`), the ONNX graph governs size: `auto` reads the static H/W, an explicit value must match it exactly, and dynamic/non-NCHW/non-square inputs are rejected.
- The CAM target semantics are documented as out of scope (gradient-attribution heatmap, not verified class-discriminative CAM).
- Version is `0.3.0`.
- No commit is performed without its separately required approval.
