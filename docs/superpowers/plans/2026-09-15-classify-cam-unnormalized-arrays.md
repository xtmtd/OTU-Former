# CAM Unnormalized Array Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update OTU-Former CAM export so `--save-npy` explicitly selects `none`, `raw`, or `normalized`, with `raw` preserving the positive unnormalized CAM magnitude while overlays continue to use a separate normalized copy.

**Architecture:** Mirror the proven entomokit v0.6.1 approach, adapted to OTU-Former's `src/otuformer/vision/cam.py`, Typer CLI, ViT reshape support, and center-crop/whole-specimen-pad geometry. Subclass each pytorch-grad-cam implementation with one mixin that preserves upstream ReLU, resize, layer aggregation, and model-specific CAM calculations while bypassing upstream `scale_cam_image()` calls. `process_image()` will keep the extractor result as the saved raw array and derive a separate min-max display array.

**Tech Stack:** Python 3.11+, Typer/Click, PyTorch, pytorch-grad-cam, NumPy, OpenCV, Pillow, pytest, TOML metadata.

**Spec:** User-approved requirements recorded in this plan; upstream reference: `/Users/zf/data/coding/entomokit/docs/superpowers/plans/2026-09-15-classify-cam-unnormalized-arrays.md` and entomokit commit `05b30e8` (`v0.6.1`).

## Global Constraints

- `--save-npy` is a required-value option with exactly `none`, `raw`, or `normalized`; the default is `none`; the bare flag is intentionally rejected.
- `raw` means the positive CAM after ReLU and resize to model resolution, before per-image min-max normalization; signed/negative CAM is out of scope.
- `normalized` means a per-image min-max `[0, 1]` copy; it preserves the previous saved-array semantics.
- `none` creates no `arrays/` directory and writes no `.npy` file.
- The overlay always uses its own normalized copy, regardless of save mode.
- Overlay rendering is not a pixel-exact contract: official-vs-raw normalization order may produce bounded float32/uint8 LSB differences after resize and colorization; tests allow that bounded difference.
- Saved arrays remain `float32` at model resolution and are not mapped to original-image coordinates.
- All six CAM methods use the raw subclasses; do not create separate official/raw execution branches.
- Preserve ViT `reshape_transform`, `AblationLayerVit`, CAM method names, target-layer selection, and center-crop/whole-specimen-pad mapping behavior.
- Do not add dependencies or pin `grad-cam`; protect the mirrored upstream seam with structural tests instead.
- Do not modify historical implementation plans merely to remove old syntax; update live source, README files, and current tests only.
- Version bump is `0.5.0` → `0.6.0`, because the CLI contract is intentionally breaking and the feature adds a new observable export contract; this is not a `1.0.0` API guarantee.
- **Commit policy:** never create a commit without the user's explicit approval in the current conversation. Implementation may leave changes unstaged or staged for review, but every commit command in this plan is approval-gated.

## File Map

- Modify `src/otuformer/vision/cam.py` — raw CAM mixin/classes, save-mode type and branching, display normalization.
- Modify `src/otuformer/cli/cam.py` — explicit three-value Typer option and examples/help.
- Modify `tests/test_cam.py` — raw class, normalization seam, save-mode, and overlay tests; update bool call sites.
- Modify `tests/test_cli_smoke.py` — CLI forwarding/default/rejection coverage for `save_npy`.
- Modify `src/otuformer/__init__.py` — set `__version__ = "0.6.0"`.
- Modify `pyproject.toml` — set project version to `0.6.0`.
- Create `tests/test_version.py` — verify package and project metadata versions stay synchronized.
- Modify `README.md` and `README.cn.md` — commands, option tables, output semantics, and comparability limits.
- Review `docs/superpowers/plans/2026-03-29-plan4-vision-export.md` as historical context only; do not rewrite it because it records an earlier implementation state.

---

### Task 1: Lock The Three-Value CLI Contract

**Files:**
- Modify: `src/otuformer/cli/cam.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_cam.py`

**Interfaces:**
- Produces `save_npy: str` with values `"none"`, `"raw"`, or `"normalized"` at the CLI boundary.
- The default invocation forwards `save_npy="none"` to `run_cam`.
- `--save-npy raw` and `--save-npy normalized` forward their exact strings.
- `--save-npy` without a value and unknown values fail during CLI parsing.

- [ ] **Step 1: Add a forwarding test for the default and explicit raw mode**

In `tests/test_cli_smoke.py`, add a test beside the existing CAM forwarding tests. Monkeypatch `otuformer.vision.cam.run_cam` with a function that records `kwargs["save_npy"]`. Invoke the command once without `--save-npy` and once with `--save-npy raw`, using `_make_ckpt(tmp_path)`, an image directory containing one image, and separate temporary output directories. Assert the recorded values are `"none"` and `"raw"` respectively.

Use this assertion shape:

```python
assert seen == ["none", "raw"]
```

- [ ] **Step 2: Add parser rejection tests**

In `tests/test_cli_smoke.py`, add tests that invoke the app with the required `--checkpoint`, `--images-dir`, and `--out-dir` arguments, then pass either `--save-npy` without a value or `--save-npy bogus`. Assert both invocations have a non-zero exit code and report an option parsing error. Do not assert Click's complete wording; only assert that the command fails before the monkeypatched `run_cam` is called.

- [ ] **Step 3: Replace the Typer boolean option**

In `src/otuformer/cli/cam.py`, replace:

```python
save_npy: bool = typer.Option(
    False,
    "--save-npy",
    help="Save raw CAM heatmaps as NumPy arrays.",
),
```

with a string option using Click's existing choice pattern:

```python
save_npy: str = typer.Option(
    "none",
    "--save-npy",
    click_type=click.Choice(["none", "raw", "normalized"]),
    show_choices=False,
    help=(
        "Save CAM arrays: none (default), raw (positive unnormalized CAM), "
        "or normalized (per-image min-max [0, 1])."
    ),
),
```

Change the quick example from `--save-npy` to `--save-npy raw`. Keep `_format_user_command()` unchanged: strings are already emitted as `--save-npy <value>` while booleans receive the special boolean handling.

- [ ] **Step 4: Update direct test call sites to use the new state**

In `tests/test_cam.py`, replace every `save_npy=False` with `save_npy="none"` and replace the existing `save_npy=True` call with `save_npy="raw"`. The latter test will be strengthened in Task 3; this step only makes all direct callers use the new interface.

- [ ] **Step 5: Run the focused CLI tests**

Run:

```bash
pytest tests/test_cli_smoke.py -k 'cam and (save_npy or forwards)' -v
```

Expected: the new tests pass after the option replacement; before the replacement, the default assertion sees `False`, explicit raw parsing fails, and the bare-flag behavior does not match the new contract.

- [ ] **Step 6: Prepare the CLI contract for review; do not commit**

```bash
git diff --check -- src/otuformer/cli/cam.py tests/test_cli_smoke.py tests/test_cam.py
git status --short
```

Stop after verification and wait for the user's explicit approval before creating a commit. If approved, use:

```bash
git add src/otuformer/cli/cam.py tests/test_cli_smoke.py tests/test_cam.py
git commit -m "feat: make CAM array save mode explicit"
```

---

### Task 2: Bypass Internal CAM Normalization With One Raw Class Family

**Files:**
- Modify: `src/otuformer/vision/cam.py`
- Modify: `tests/test_cam.py`

**Interfaces:**
- Produces `UnnormalizedCAMMixin` and explicit raw subclasses for `GradCAM`, `GradCAMPlusPlus`, `LayerCAM`, `ScoreCAM`, `EigenCAM`, and `AblationCAM`.
- `CAM_METHODS[name]` points to the corresponding raw subclass whenever `pytorch-grad-cam` is available.
- A raw extractor preserves `np.maximum(cam, 0)`, resizes to the input target size, averages layers, and does not call `scale_cam_image()`.
- When `pytorch-grad-cam` is unavailable, the existing `_HAS_CAM=False` behavior remains unchanged and `CAM_METHODS` remains empty.

- [ ] **Step 1: Add structural tests before implementation**

In `tests/test_cam.py`, import `UnnormalizedCAMMixin` inside the test functions so the existing optional dependency behavior remains intact. Add:

```python
CAM_METHOD_NAMES = (
    "ablationcam",
    "eigencam",
    "gradcam",
    "gradcampp",
    "layercam",
    "scorecam",
)


def test_cam_methods_use_raw_subclasses():
    pytest.importorskip("pytorch_grad_cam")
    from otuformer.vision.cam import CAM_METHODS, UnnormalizedCAMMixin

    assert set(CAM_METHODS) == set(CAM_METHOD_NAMES)
    for name, cls in CAM_METHODS.items():
        assert issubclass(cls, UnnormalizedCAMMixin), name
```

Add a structural sentinel test that monkeypatches `pytorch_grad_cam.base_cam.scale_cam_image` to raise. Run the official `GradCAM` under the patch and assert it raises, then run `CAM_METHODS["gradcam"]` with a tiny convolutional model, a deterministic `torch.rand` input, and `ClassifierOutputTarget`; assert the raw call returns a finite `(32, 32)` map. This proves the test is not vacuous.

Add a second test comparing the official and raw GradCAM outputs on the same model/input: min-max normalize the raw output and assert it agrees with the official output within `1e-6`, while the official output has maximum approximately `1.0`. Do not assert any environment-specific raw magnitude.

- [ ] **Step 2: Implement the mixin using the installed upstream seam**

In `src/otuformer/vision/cam.py`, import `List` and add the mixin after the optional pytorch-grad-cam import block and before `CAM_METHODS`:

```python
class UnnormalizedCAMMixin:
    """Mirror pytorch-grad-cam 1.5.5 BaseCAM without scale_cam_image().

    ReLU, target-size resize, layer concatenation, and layer averaging remain
    identical to BaseCAM. Only its two min-max normalization calls are skipped.
    Re-read these methods when upgrading grad-cam.
    """

    @staticmethod
    def _resize_batch(cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        import cv2

        return np.stack(
            [
                cv2.resize(
                    np.float32(image), target_size, interpolation=cv2.INTER_LINEAR
                )
                for image in cam
            ]
        )

    def compute_cam_per_layer(self, input_tensor, targets, eigen_smooth):
        if self.detach:
            activations_list = [
                activation.cpu().data.numpy()
                for activation in self.activations_and_grads.activations
            ]
            grads_list = [
                gradient.cpu().data.numpy()
                for gradient in self.activations_and_grads.gradients
            ]
        else:
            activations_list = list(self.activations_and_grads.activations)
            grads_list = list(self.activations_and_grads.gradients)

        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []
        for index, target_layer in enumerate(self.target_layers):
            activations = activations_list[index] if index < len(activations_list) else None
            gradients = grads_list[index] if index < len(grads_list) else None
            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                activations,
                gradients,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            scaled = self._resize_batch(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        return np.mean(cam_per_target_layer, axis=1)
```

The code must mirror the installed `BaseCAM.compute_cam_per_layer()` and `aggregate_multi_layers()` signatures and order. Do not patch `scale_cam_image` globally and do not change the `get_cam_image()` implementations.

- [ ] **Step 3: Define explicit raw subclasses and update `CAM_METHODS`**

When `_HAS_CAM` is true, define:

```python
class RawGradCAM(UnnormalizedCAMMixin, GradCAM):
    pass


class RawGradCAMPlusPlus(UnnormalizedCAMMixin, GradCAMPlusPlus):
    pass


class RawLayerCAM(UnnormalizedCAMMixin, LayerCAM):
    pass


class RawScoreCAM(UnnormalizedCAMMixin, ScoreCAM):
    pass


class RawEigenCAM(UnnormalizedCAMMixin, EigenCAM):
    pass


class RawAblationCAM(UnnormalizedCAMMixin, AblationCAM):
    pass
```

Set the existing six dictionary entries to these classes. If `_HAS_CAM` is false, do not define subclasses of the fallback `object`; retain `CAM_METHODS = {}`. Keep the dictionary keys and insertion order unchanged unless tests require no order assumption.

- [ ] **Step 4: Add six-method invariant coverage**

Add a parametrized test for all six method names. Use a deterministic tiny convolutional model with `model[4]` as the target layer and a `32 x 32` input. Assert shape, `np.float32` dtype, finite values, and `raw.min() >= 0`. Do not assert a fixed maximum because gradients and backend implementations can legitimately produce different magnitudes; only assert positivity for this fixture if the implementation needs a nonzero guard, and document that it is diagnostic rather than a contract.

- [ ] **Step 5: Run the CAM seam tests**

Run:

```bash
pytest tests/test_cam.py -k 'raw or method' -v
```

Expected: all raw-class, sentinel, official-equivalence, and six-method tests pass. The suite must also run with the installed `grad-cam` version and with the dependency absent path still importable.

- [ ] **Step 6: Prepare the raw CAM implementation for review; do not commit**

```bash
git diff --check -- src/otuformer/vision/cam.py tests/test_cam.py
git status --short
```

Stop after verification and wait for the user's explicit approval before creating a commit. If approved, use:

```bash
git add src/otuformer/vision/cam.py tests/test_cam.py
git commit -m "feat: preserve unnormalized CAM magnitudes"
```

---

### Task 3: Separate Saved Arrays From Display Arrays

**Files:**
- Modify: `src/otuformer/vision/cam.py`
- Modify: `tests/test_cam.py`

**Interfaces:**
- `SaveMode = Literal["none", "raw", "normalized"]` is used by `process_image()` and `run_cam()`.
- `process_image()` saves `grayscale_cam` for `raw`, the min-max copy for `normalized`, and nothing for `none`.
- `map_cam_to_original()` and `show_cam_on_image()` always receive the normalized display copy.
- The summary's existing `cam_array_path`/`cam_file` fields and CAM summary columns remain compatible.

- [ ] **Step 1: Add deterministic save-mode tests**

In `tests/test_cam.py`, add a test-owned CAM array with values outside `[0, 1]`; keep it local to the new save-mode tests rather than changing shared CAM fixtures:

```python
FAKE_CAM = np.array([[0.0, 2.0], [1.0, 0.5]], dtype=np.float32)
```

Create a new local helper/test fixture around `process_image()` that uses a 32x32 image, a fake model returning deterministic logits, a zero tensor preprocess function, a fake extractor returning `FAKE_CAM.copy()`, a temporary figure directory, and a temporary array directory. Monkeypatch `show_cam_on_image()` to return a valid uint8 image. The helper's 2x2 fixture is for save-mode semantics; keep the existing separate 32x32 model-resolution test for the output-shape contract. Do not change the shared `_fake_cam_extractor()` helper: it is used by the existing uint8-overflow, center-crop, and pad-full-frame tests, whose input behavior should remain unchanged.

Add tests asserting:

- `save_npy="raw"` writes float32 values exactly equal to `FAKE_CAM` and the result points to the `.npy` file.
- `save_npy="normalized"` writes `[[0.0, 1.0], [0.5, 0.25]]`.
- `save_npy="none"` returns an empty array path and leaves the array directory empty.
- A capturing `show_cam_on_image()` receives a normalized, mapped mask with min/max within `[0, 1]`, equal to `map_cam_to_original(cam_display, original_size, eval_transform, model_size)` where `cam_display` is the explicit min-max normalization of `FAKE_CAM`; do not omit the required `eval_transform` argument. It must not receive the raw peak of `2.0`.
- Add an end-to-end overlay tolerance test parametrized for `gradcam` and `scorecam`. Run the official upstream class and the corresponding raw class on the same deterministic model/input, normalize each returned CAM for display, pass both through the same `map_cam_to_original(cam_norm, original_size, eval_transform, model_size)` path, and render both with `show_cam_on_image()`. Assert every uint8 channel differs by at most 2 LSB, while allowing a small nonzero count of differing pixels caused by normalization-before/after-resize ordering. This protects the overlay path without imposing a pixel-identical PNG contract. Skip the parameter case if the optional CAM dependency is unavailable, consistently with the existing CAM tests.

Strengthen the existing 32x32 raw-array test with this test-owned fixture:

```python
expected_raw = np.zeros((32, 32), dtype=np.float32)
expected_raw[8:16, 10:22] = 2.0
```

Make the fake extractor return `expected_raw.copy()`, then assert the saved array has shape `(32, 32)`, dtype `float32`, and `np.testing.assert_allclose(saved, expected_raw)`. This keeps the model-resolution contract separate from the 2x2 save-mode fixture.

- [ ] **Step 2: Change `process_image()` to derive a display copy**

Keep the existing extraction:

```python
grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]
```

using the actual existing variable name `grayscale_cam`; then add:

```python
cam_display = grayscale_cam - grayscale_cam.min()
if cam_display.max() > 0:
    cam_display = cam_display / cam_display.max()
else:
    cam_display = np.zeros_like(cam_display)
```

Pass `cam_display` to `map_cam_to_original()` and use the resulting mapped mask for the overlay. Do not normalize or resize the saved raw array.

- [ ] **Step 3: Branch saving by explicit mode**

Add `from typing import Literal` and define:

```python
SaveMode = Literal["none", "raw", "normalized"]
```

Use `save_npy: SaveMode` in both `process_image()` and `run_cam()`, with `run_cam()` defaulting to `"none"`. Replace the truthiness checks with explicit checks:

```python
cam_array_path = ""
if save_npy != "none" and array_dir is not None:
    npy_path = array_dir / f"{stem}.npy"
    cam_array = grayscale_cam if save_npy == "raw" else cam_display
    np.save(npy_path, cam_array.astype(np.float32))
    cam_array_path = str(npy_path)
```

In `run_cam()`, create and mkdir `array_dir` only when `save_npy != "none"`. Do not silently treat `False` or `True` as valid internal state; all repository call sites must use strings.

- [ ] **Step 4: Run the full CAM test file**

Run:

```bash
pytest tests/test_cam.py -v
```

Expected: existing crop/pad geometry, uint8 overflow, checkpoint loader, and CAM setup tests pass alongside the new raw/normalized/none tests. No test should require a fixed CAM magnitude from a real model.

- [ ] **Step 5: Run CLI smoke tests and commit**

Run:

```bash
pytest tests/test_cli_smoke.py tests/test_cam.py -v
```

Then stop for review; do not commit:

```bash
git diff --check -- src/otuformer/vision/cam.py tests/test_cam.py tests/test_cli_smoke.py
git status --short
```

Wait for the user's explicit approval before creating a commit. If approved, use:

```bash
git add src/otuformer/vision/cam.py tests/test_cam.py tests/test_cli_smoke.py
git commit -m "fix: keep CAM display and export arrays separate"
```

---

### Task 4: Synchronize Version Metadata To 0.6.0

**Files:**
- Modify: `src/otuformer/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_version.py`

**Interfaces:**
- `otuformer.__version__ == "0.6.0"`.
- `[project].version` in `pyproject.toml` is `"0.6.0"`.
- The existing `otuformer --version` callback continues to report the package version.

- [ ] **Step 1: Add the metadata consistency test**

Create `tests/test_version.py`:

```python
from pathlib import Path
import tomllib

from otuformer import __version__


def test_package_and_project_versions_match():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == "0.6.0"
    assert project["project"]["version"] == __version__
```

- [ ] **Step 2: Update both version sources**

Change only the version values in `src/otuformer/__init__.py` and `pyproject.toml` from `0.5.0` to `0.6.0`. Do not add a runtime version warning or duplicate version file.

- [ ] **Step 3: Verify version behavior**

Run:

```bash
pytest tests/test_version.py tests/test_cli_startup.py -v
python -c "from typer.testing import CliRunner; from otuformer.cli.main import app; print(CliRunner().invoke(app, ['--version']).output, end='')"
```

Expected: the test suite passes and the command output contains `otuformer 0.6.0`.

- [ ] **Step 4: Prepare the version bump for review; do not commit**

```bash
git diff --check -- src/otuformer/__init__.py pyproject.toml tests/test_version.py
git status --short
```

Stop after verification and wait for the user's explicit approval before creating a commit. If approved, use:

```bash
git add src/otuformer/__init__.py pyproject.toml tests/test_version.py
git commit -m "chore: bump version to 0.6.0"
```

---

### Task 5: Update Live CAM Documentation

**Files:**
- Modify: `README.md`
- Modify: `README.cn.md`
- Modify: `src/otuformer/cli/cam.py` if help wording needs final alignment after tests

**Interfaces:**
- Every live CAM example uses an explicit save mode.
- Every live option table documents `none`, `raw`, and `normalized` accurately.
- Documentation states that raw values are positive unnormalized CAM arrays at model resolution and that overlays always use a separate normalized copy.
- Documentation states comparability limits without claiming cross-model or cross-method magnitude comparability.

- [ ] **Step 1: Update the English CAM example and option table**

In `README.md`, change the GradCAM++ example from a bare `--save-npy` to `--save-npy raw`. Replace the option row with a description equivalent to:

```markdown
| `--save-npy` | `none` (default, no array), `raw` (positive unnormalized CAM), or `normalized` (per-image min-max `[0, 1]`) | `none` |
```

Replace the output entry with wording that `arrays/` is created only for `raw` or `normalized`, and that saved arrays remain float32 at model input resolution.

Add a concise paragraph below the output list: raw magnitude is comparable only under the same model, target layer, and preprocessing configuration; `eigencam` raw values are exported but unsuitable for response-strength statistics because its SVD projection sign is arbitrary; normalized exports preserve the old saved-array semantics; overlay display semantics remain normalized and pixel-identical PNG output is not a contract.

- [ ] **Step 2: Mirror the documentation in Chinese**

Apply the same three changes in `README.cn.md`: use `--save-npy raw` in the example, document the three values/default in the table, and describe the raw/normalized/overlay semantics and comparability limits in Chinese.

- [ ] **Step 3: Check live documentation for stale CAM claims**

Run:

```bash
rg -n -i --glob '!docs/superpowers/plans/**' --glob '!**/__pycache__/**' \
  'save-npy|save_npy|CAM array|CAM 数组|原始 CAM' \
  README.md README.cn.md src/otuformer tests
```

Review every match. The only accepted old-looking wording is in tests that explicitly describe a migration or in historical plan documents excluded above. No live example may contain the bare flag, and no live source help may call normalized output “raw”.

- [ ] **Step 4: Prepare the live documentation for review; do not commit**

```bash
git diff --check -- README.md README.cn.md src/otuformer/cli/cam.py
git status --short
```

Stop after verification and wait for the user's explicit approval before creating a commit. If approved, use:

```bash
git add README.md README.cn.md src/otuformer/cli/cam.py
git commit -m "docs: document CAM array save modes"
```

---

### Task 6: Full Verification And Release Readiness

**Files:**
- No source changes expected; fix only findings from verification tasks.

**Interfaces:**
- All CAM behavior, CLI behavior, version metadata, and existing project tests pass together.
- No uncommitted unrelated changes are introduced.

- [ ] **Step 1: Run focused regression tests**

```bash
pytest tests/test_cam.py tests/test_cli_smoke.py tests/test_cli_startup.py tests/test_version.py -v
```

Expected: PASS. If a failure occurs, identify whether it is a real regression or an environment-dependent CAM backend issue; do not weaken tests that protect the raw/display contract.

- [ ] **Step 2: Run the complete test suite**

```bash
pytest -q
```

Expected: PASS for the full repository suite. The plan is not complete while tests are failing.

- [ ] **Step 3: Verify CLI help and migration behavior manually**

```bash
otuformer cam --help
otuformer cam --checkpoint /tmp/missing.ckpt --images-dir /tmp/missing-images --out-dir /tmp/cam-check --save-npy
otuformer cam --checkpoint /tmp/missing.ckpt --images-dir /tmp/missing-images --out-dir /tmp/cam-check --save-npy invalid
```

Use the installed `otuformer` console entry point (`otuformer.cli.main:app`); do not substitute `python -m otuformer.cli.main`, because that module does not invoke the Typer app itself. Expected: help describes all three modes; the two invalid invocations exit nonzero before model loading; no command path accepts the old bare flag. If the editable package is not installed in the verification environment, use the Task 4 `CliRunner` invocation against `otuformer.cli.main.app` instead.

- [ ] **Step 4: Verify the final diff and status**

```bash
git diff --check
rg -n -- '0\.5\.0|save_npy: bool|save_npy=False|save_npy=True' \
  pyproject.toml src tests README.md README.cn.md
rg -n -- '(^|[[:space:]])--save-npy([[:space:]]|$)' \
  README.md README.cn.md src/otuformer tests
git status --short
```

Expected: no stale version or boolean save-mode references in live files, no whitespace errors, and only intended files changed. Historical plans may retain their original examples because they are records, not live contracts.

- [ ] **Step 5: Record final verification-only fixes without committing**

If verification requires a source/test/doc correction, make the smallest focused change and show it in `git diff`. Do not commit it automatically. After the user explicitly approves the commit, use a focused Conventional Commit such as:

```bash
git add <only-the-corrected-files>
git commit -m "fix: address CAM release verification findings"
```

Do not create a release tag or publish package artifacts as part of this plan unless explicitly requested separately.
