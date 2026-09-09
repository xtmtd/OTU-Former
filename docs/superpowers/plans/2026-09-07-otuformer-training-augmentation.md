# OTU-Former Training Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement biologically explicit, reproducible training augmentation profiles on top of the resolution-consistency implementation already present in the repository.

**Architecture:** Keep fixed profile definitions and stochastic training transform construction in `src/otuformer/training/dataset.py`. Reuse the existing `_estimate_background_color`, named deterministic evaluation transforms, checkpoint-size resolution, and `img_size` propagation; do not duplicate that infrastructure. Resolve omitted versus explicit CLI profiles and orientation policies in `src/otuformer/training/trainer.py` before dataset construction, persist the resolved profile plus expanded JSON-serializable configuration in checkpoints, and reject incompatible resume attempts. Add one `--augmentation` option to each training CLI and document the biological contract in the existing design and bilingual README files.

**Tech Stack:** Python 3.11+, PyTorch 2.1+, torchvision 0.16+, Pillow, pandas, Typer/Click, pytest 8+

**Spec:** `docs/superpowers/specs/2026-09-07-otuformer-training-augmentation-design.md`

## Current Code Baseline

This incremental plan starts from `b4a4b3c` and its relevant ancestors `a583c74` and `7a7604b`. Resolution consistency is already implemented: checkpoint training-size resolution and propagation, `OTUFormerEncoder(img_size=...)`, `auto` size options, named `center-crop`/`whole-specimen-pad` evaluation protocols, extract/CAM/ONNX size handling, non-square CAM mapping, and legacy SSL/ArcFace checkpoint compatibility. These are prerequisites, not tasks in this plan. If they change, update the resolution-consistency documentation first and re-audit this plan.

## Global Constraints

- New pretraining runs default to `global-barcode`; allowed profiles are exactly `global-barcode`, `color-robust`, and `legacy`.
- New training runs default to `orientation-policy=sensitive`; allowed policies are exactly `invariant` and `sensitive`. `sensitive` applies to every global/local view of `global-barcode` and `color-robust`, and to fine-tuning `conservative`, as rotation `[-15°, 15°]` with no horizontal flip. `invariant` is the opt-in broad-rotation/reflection policy. When fine-tuning is initialized from a pretraining checkpoint and the option is omitted, inherit that checkpoint's policy.
- New fine-tuning runs default to `none`; allowed profiles are exactly `none` and `conservative`.
- Keep global scale `(0.4, 1.0)`, local scale `(0.05, 0.4)`, torchvision ratio `(3/4, 4/3)`, two global views, and six local crops by default.
- Keep `--global-crop-size`, `--local-crop-size`, and `--local-crops`; add only the high-level `--orientation-policy` option; do not add low-level augmentation-strength flags.
- Keep all profile definitions in `src/otuformer/training/dataset.py`; do not add YAML, a registry framework, a strategy hierarchy, or a dependency.
- Keep global-view loss, `A-A/A-B/B-A/B-B` pairing, masked-token loss, model architecture, extraction, CAM, ONNX, deterministic evaluation transforms, and TTA unchanged. These interfaces were updated by the completed resolution-consistency work and are outside this plan.
- Do not add dense morphology options, heads, losses, outputs, or placeholder interfaces.
- Reuse the existing `_estimate_background_color()` contract for stochastic training fills: one RGB fill per source image from the per-channel median of all four outer edges, with `(124, 116, 104)` as fallback. Do not implement a second helper or alter the evaluation-transform implementation.
- New pretrain geometry order is `RandomResizedCrop -> expanded rotation -> Resize -> horizontal flip`; no vertical flip or solarization. Under `orientation-policy=sensitive`, every global and local view uses the same order with rotation `[-15°, 15°]` and horizontal flip probability `0.0`.
- `legacy` must reproduce OTU-Former 0.2.1 behavior exactly.
- New checkpoints store `config.augmentation_profile` and `config.augmentation_config`; the top-level profile is a quick-identification index, while the nested `profile` makes the expanded config self-contained. `augmentation_config.orientation_policy` is the only stored source of the orientation policy, with no top-level `config.orientation_policy`. The latter may record the resolved training `image_size` for auditability, but size source, precedence, validation, `img_size` propagation, and size-resume compatibility remain owned by the completed resolution-consistency implementation. This metadata provides configuration traceability, not bitwise replay.
- Resume must inherit omitted profile/policy values and reject explicit profile/policy/config conflicts. Old pretrain checkpoints map to `legacy`/`invariant`; old finetune checkpoints map to `none`/`invariant`. Explicit `sensitive` cannot resume an old checkpoint because no saved policy proves compatibility.
- Starting fine-tuning from a pretraining checkpoint with `--checkpoint` is initialization, not resume, and must not inherit the pretraining augmentation profile.
- Both fine-tuning profiles use the existing checkpoint-size resolution contract. Use the resolved value to instantiate the backbone and construct `MetricDataset`; do not add a second size-resolution path to this plan.
- Code, comments, and documentation remain in English except for `README.cn.md`.
- Do not commit unless the user separately approves a commit after implementation and verification.

---

### Task 1: Implement training augmentation profiles in the existing dataset module

**Files:**
- Modify: `src/otuformer/training/dataset.py` (existing training dataset and transform module)
- Test: `tests/test_training_dataset.py:1-end`

The resolution-consistency implementation in this file is already complete. This task adds only training augmentation profiles and their dataset wiring; it must not recreate or alter the existing evaluation-transform and size-propagation interfaces.

**Interfaces:**
- Produces: `PRETRAIN_AUGMENTATIONS: tuple[str, ...] = ("global-barcode", "color-robust", "legacy")`.
- Produces: `FINETUNE_AUGMENTATIONS: tuple[str, ...] = ("none", "conservative")`.
- Produces: `ORIENTATION_POLICIES: tuple[str, ...] = ("invariant", "sensitive")`.
- Produces: `build_pretrain_augmentation_config(profile: str, global_crop_size: int, local_crop_size: int, local_crops: int, orientation_policy: str = "sensitive") -> dict[str, object]`.
- Produces: `build_finetune_augmentation_config(profile: str, image_size: int, orientation_policy: str = "sensitive") -> dict[str, object]`; `image_size` is supplied by the existing resolution-consistency flow and recorded for auditability.
- Reuses: `_estimate_background_color(image: Image.Image) -> tuple[int, int, int]` already implemented by resolution consistency; no new helper is produced.
- Changes: `MultiCropDataset(..., augmentation_profile: str = "global-barcode", orientation_policy: str = "sensitive")` and exposes `augmentation_profile`, `orientation_policy`, plus `augmentation_config` attributes.
- Changes: `MetricDataset(..., image_size: int, augmentation_profile: str = "none", orientation_policy: str = "sensitive")` and exposes `augmentation_profile`, `orientation_policy`, plus `augmentation_config` attributes; both profiles honor the supplied `image_size`.
- Consumes later: Task 2 imports both configuration builders and passes resolved profile/policy values to these datasets; Task 3 uses the allowed profile and orientation-policy values for CLI validation.

- [ ] **Step 1: Add profile configuration tests**

Add `pytest`, torchvision transform inspection, and the new dataset interfaces. Add focused tests that assert exact expanded values rather than random pixels. Keep the existing resolution-consistency tests for `_estimate_background_color`, `center_crop_eval_transform`, and `whole_specimen_pad_transform`; do not duplicate them in this task:

```python
import pytest

from otuformer.training.dataset import (
    FINETUNE_AUGMENTATIONS,
    ORIENTATION_POLICIES,
    PRETRAIN_AUGMENTATIONS,
    MetricDataset,
    MultiCropDataset,
    build_finetune_augmentation_config,
    build_pretrain_augmentation_config,
)


def test_augmentation_profile_names_are_fixed():
    assert PRETRAIN_AUGMENTATIONS == ("global-barcode", "color-robust", "legacy")
    assert FINETUNE_AUGMENTATIONS == ("none", "conservative")
    assert ORIENTATION_POLICIES == ("invariant", "sensitive")


def test_pretrain_profile_configs_are_fully_expanded():
    barcode = build_pretrain_augmentation_config(
        "global-barcode", 224, 96, 6, orientation_policy="invariant"
    )
    robust = build_pretrain_augmentation_config(
        "color-robust", 224, 96, 6, orientation_policy="invariant"
    )
    sensitive = build_pretrain_augmentation_config(
        "global-barcode", 224, 96, 6, orientation_policy="sensitive"
    )
    sensitive_robust = build_pretrain_augmentation_config(
        "color-robust", 224, 96, 6, orientation_policy="sensitive"
    )
    legacy = build_pretrain_augmentation_config("legacy", 224, 96, 6)
    sensitive_legacy = build_pretrain_augmentation_config(
        "legacy", 224, 96, 6, orientation_policy="sensitive"
    )

    assert barcode["global_crop"]["scale"] == [0.4, 1.0]
    assert barcode["local_crop"]["scale"] == [0.05, 0.4]
    assert barcode["crop_ratio"] == [0.75, 4 / 3]
    assert barcode["local_crops"] == 6
    assert barcode["rotation"] == {
        "degrees": [-180.0, 180.0],
        "interpolation": "bicubic",
        "expand": True,
        "fill": "edge-median-rgb",
    }
    assert barcode["profile"] == "global-barcode"
    assert barcode["orientation_policy"] == "invariant"
    assert barcode["horizontal_flip_probability"] == 0.5
    assert barcode["vertical_flip_probability"] == 0.0
    assert sensitive["orientation_policy"] == "sensitive"
    assert sensitive["rotation"]["degrees"] == [-15.0, 15.0]
    assert sensitive["horizontal_flip_probability"] == 0.0
    assert sensitive_robust["orientation_policy"] == "sensitive"
    assert sensitive_robust["rotation"]["degrees"] == [-15.0, 15.0]
    assert sensitive_robust["horizontal_flip_probability"] == 0.0
    assert sensitive_legacy["orientation_policy"] == "sensitive"
    assert sensitive_legacy["rotation"] == legacy["rotation"]
    assert sensitive_legacy["horizontal_flip_probability"] == 0.5
    assert barcode["color_jitter"] == {
        "probability": 0.8,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.1,
        "hue": 0.02,
    }
    assert barcode["grayscale_probability"] == 0.0
    assert barcode["blur"]["probabilities"] == [1.0, 0.1, 0.5]
    assert robust["color_jitter"]["brightness"] == 0.4
    assert robust["color_jitter"]["saturation"] == 0.2
    assert robust["grayscale_probability"] == 0.2
    assert legacy["rotation"]["expand"] is False
    assert legacy["rotation"]["fill"] == [0, 0, 0]
    assert legacy["vertical_flip_probability"] == 0.5
    assert legacy["blur"]["probabilities"] == [1.0, 1.0, 1.0]

    none = build_finetune_augmentation_config("none", image_size=32)
    conservative = build_finetune_augmentation_config(
        "conservative", image_size=32
    )
    sensitive_conservative = build_finetune_augmentation_config(
        "conservative", image_size=32, orientation_policy="sensitive"
    )
    assert none["image_size"] == 32
    assert conservative["image_size"] == 32
    assert sensitive_conservative["orientation_policy"] == "sensitive"
    assert sensitive_conservative["rotation"]["degrees"] == [-15.0, 15.0]
    assert sensitive_conservative["horizontal_flip_probability"] == 0.0
    assert conservative["transform_order"][-3:] == [
        "resize",
        "to_tensor",
        "normalize",
    ]


def test_unknown_augmentation_profiles_fail():
    with pytest.raises(ValueError, match="Unknown pretrain augmentation profile"):
        build_pretrain_augmentation_config("unknown", 224, 96, 6)
    with pytest.raises(ValueError, match="Unknown orientation policy"):
        build_pretrain_augmentation_config(
            "global-barcode", 224, 96, 6, orientation_policy="unknown"
        )
    with pytest.raises(ValueError, match="Unknown finetune augmentation profile"):
        build_finetune_augmentation_config("unknown", 224)
```

The existing resolution-consistency tests cover the documented background-estimation median and fallback; this plan only verifies that training views reuse that helper and does not add duplicate tests.

- [ ] **Step 2: Run the new configuration tests and verify they fail**

Run:

```bash
pytest tests/test_training_dataset.py -k "augmentation_profile_names or profile_configs or unknown_augmentation" -v
```

Expected: failures because the training profile constants/builders do not exist. Existing resolution-consistency tests must remain passing.

- [ ] **Step 3: Add the minimum fixed profile data and JSON-serializable builders**

In `dataset.py`, add two fixed training-profile mappings alongside the existing resolution-consistency helpers. Reuse the module's existing normalization and background-fill constants where available. Return fresh dictionaries from public builders so callers cannot mutate module-level settings. Use only strings, numbers, booleans, lists, and dictionaries in expanded configs; do not place torchvision/Pillow objects in checkpoint metadata.

The pretrain expanded config must include:

```python
{
    "profile": profile,
    "orientation_policy": orientation_policy,
    "global_crop": {"size": global_crop_size, "scale": [0.4, 1.0]},
    "local_crop": {"size": local_crop_size, "scale": [0.05, 0.4]},
    "crop_ratio": [0.75, 4 / 3],
    "local_crops": local_crops,
    "transform_order": [
        "random_resized_crop",
        "rotation",
        "resize",
        "horizontal_flip",
        "color",
        "blur",
        "to_tensor",
        "normalize",
    ],
    "rotation": {
        "degrees": [-180.0, 180.0],
        "interpolation": "bicubic",
        "expand": True,
        "fill": "edge-median-rgb",
    },
    "fallback_fill": [124, 116, 104],
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.0,
    "color_jitter": {...},
    "grayscale_probability": 0.0,
    "blur": {
        "probabilities": [1.0, 0.1, 0.5],
        "global_kernel": resolved_global_kernel,
        "local_kernel": resolved_local_kernel,
        "sigma": [0.1, 2.0],
    },
    "solarization_probability": 0.0,
}
```

For `global-barcode` use jitter probability/strengths `0.8 / 0.2 / 0.2 / 0.1 / 0.02`; for `color-robust` use `0.8 / 0.4 / 0.4 / 0.2 / 0.1` plus grayscale `0.2`. Resolve new-profile blur kernels as the largest odd integer no greater than the standard kernel (`23` global, `7` local) and crop size; reject crop sizes below `3`.

For `legacy`, preserve fixed kernels `23` and `7`, `expand=False`, black fill, both flips at `0.5`, always-applied `ColorJitter(0.4, 0.4, 0.4, 0.1)`, grayscale `0.2`, and blur probabilities `[1.0, 1.0, 1.0]`. Its transform order omits the post-rotation resize. It accepts either recorded orientation policy for metadata consistency, but the policy must not rewrite these historical transform settings.

The finetune config must make `none` describe the existing deterministic transform and make `conservative` include degrees `[-180.0, 180.0]`, translation `[0.05, 0.05]` relative to the safe canvas side, scale `[0.9, 1.1]`, horizontal flip `0.5`, light color settings, no crop/grayscale/blur/solarization, the supplied `image_size`, safe-canvas formula, and fallback fill. With `orientation_policy="sensitive"`, `conservative` records degrees `[-15.0, 15.0]` and horizontal flip `0.0`. The builder receives the already resolved `image_size`; it must not introduce a second default or size-resolution rule. Existing resolution fallback behavior, including `224` for old checkpoints, remains outside this task.

Do not implement `_estimate_background_color()` here. Use the existing helper and its documented fallback from the completed resolution-consistency implementation. Add only a focused regression assertion that profile transforms receive and reuse the helper's fill.

- [ ] **Step 4: Run the configuration tests and verify they pass**

Run:

```bash
pytest tests/test_training_dataset.py -k "augmentation_profile_names or profile_configs or unknown_augmentation" -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Add failing behavioral tests for all dataset profiles**

Add tests that construct rectangular images and assert stable shapes and configuration behavior:

```python
@pytest.mark.parametrize("profile", PRETRAIN_AUGMENTATIONS)
def test_pretrain_profiles_return_fixed_shapes_for_rectangular_images(tmp_path, profile):
    csv_path = make_dummy_images(tmp_path, n=1)
    Image.new("RGB", (91, 47), color=(240, 240, 240)).save(tmp_path / "img_0.jpg")
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
        augmentation_profile=profile,
    )
    views = ds[0]
    assert [tuple(view.shape) for view in views] == [
        (3, 32, 32),
        (3, 32, 32),
        (3, 16, 16),
        (3, 16, 16),
    ]


def test_multicrop_default_still_returns_six_local_views(tmp_path):
    csv_path = make_dummy_images(tmp_path, n=1)
    views = MultiCropDataset(csv_path=csv_path, images_dir=tmp_path)[0]
    assert len(views) == 8


@pytest.mark.parametrize("profile", FINETUNE_AUGMENTATIONS)
def test_finetune_profiles_return_fixed_shape_for_rectangular_images(tmp_path, profile):
    csv_path = make_dummy_labeled_images(tmp_path, n=1)
    Image.new("RGB", (91, 47), color=(240, 240, 240)).save(tmp_path / "img_0.jpg")
    image, label = MetricDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        image_size=32,
        augmentation_profile=profile,
    )[0]
    assert image.shape == (3, 32, 32)
    assert label == 0
```

Add transform-structure assertions through dataset attributes/config rather than exact random pixels:

- `global_tf1`, `global_tf2`, and `local_tf` are separate callables for new pretrain profiles.
- Their blur probabilities are `1.0`, `0.1`, and `0.5`.
- `legacy` includes vertical flip, grayscale, black non-expanded rotation, and no post-rotation resize.
- `none` still stores a `transforms.Compose` containing `Resize`, `CenterCrop`, `ToTensor`, and `Normalize` in that order.
- `conservative` config contains no random resized crop, grayscale, blur, or solarization; its `sensitive` variant records `[-15°, 15°]` and no horizontal reflection.
- `legacy` accepts either orientation policy but its transform settings remain unchanged; the selected policy is recorded in metadata.
- every pretrain global/local view and fine-tuning view follows the selected policy, including sensitive rotation/reflection settings. Construct `global-barcode`, `color-robust`, `legacy`, and `conservative` datasets with `orientation_policy="sensitive"`; assert the relevant `augmentation_config` entries have `[-15.0, 15.0]` and horizontal-flip probability `0.0`, and assert the fine-tuning config has the same sensitive settings. `_PretrainViewTransform` must expose read-only `rotation_degrees` and `horizontal_flip_probability` attributes copied from that config so focused transform tests can verify `global_tf1`, `global_tf2`, and `local_tf` directly. Assert `legacy` keeps its historical transform and `none` remains deterministic.
- directly call `build_finetune_augmentation_config("conservative", image_size=32, orientation_policy="sensitive")` and assert `[-15.0, 15.0]` plus horizontal flip `0.0`.
- generated views whose tensor shape differs from the requested crop/image size raise a clear `ValueError`: monkeypatch one view callable to return a tensor with an incorrect spatial size, call `ds[0]`, and assert the error names the expected and actual shape.
- invalid profile names passed directly to either dataset raise `ValueError`.

To prove one edge estimate is shared across all source-image views, monkeypatch the existing `_estimate_background_color` to return `(11, 22, 33)`, replace each training view callable with a recorder accepting `(image, fill)`, call `ds[0]`, and assert every recorded fill equals `(11, 22, 33)` and estimation ran once. This test must not alter or replace the deterministic evaluation transforms.

- [ ] **Step 6: Run behavioral tests and verify they fail**

Run:

```bash
pytest tests/test_training_dataset.py -k "pretrain_profiles or six_local or finetune_profiles or shares_background or conservative or legacy or orientation_policy or fixed_output" -v
```

Expected: failures because datasets do not yet accept profiles or expose separate fill-aware view transforms.

- [ ] **Step 7: Implement fill-aware pretrain transforms and conservative fine-tuning**

Add one small, module-level, picklable callable class for pretrain views, for example `_PretrainViewTransform`, with `__call__(self, image: Image.Image, fill: tuple[int, int, int]) -> torch.Tensor`. Avoid nested closures because DataLoader workers using spawn must pickle the dataset.

For `global-barcode` and `color-robust`, perform exactly, with `rotation_degrees` and `horizontal_flip_probability` copied from the resolved `augmentation_config`:

```python
image = random_resized_crop(image)
angle = transforms.RandomRotation.get_params(rotation_degrees)
image = TF.rotate(
    image,
    angle,
    interpolation=InterpolationMode.BICUBIC,
    expand=True,
    fill=fill,
)
image = TF.resize(
    image,
    [crop_size, crop_size],
    interpolation=InterpolationMode.BICUBIC,
    antialias=True,
)
image = horizontal_flip(image)  # constructed with horizontal_flip_probability
image = color_transform(image)
image = blur_transform(image)
image = to_tensor_and_normalize(image)
```

Use distinct instances for `global_tf1`, `global_tf2`, and `local_tf`. Use `RandomApply([ColorJitter(...)], p=0.8)`, optional `RandomGrayscale(p=0.2)` only for `color-robust`, and `RandomApply([GaussianBlur(...)], p=...)`. No solarization or vertical flip.

For `legacy`, use the existing transform sequence unchanged. Wrap it in the same two-argument callable contract while ignoring the supplied edge fill. Keep two global callable instances so `MultiCropDataset` has a uniform `global_tf1/global_tf2/local_tf` interface without changing legacy randomness.

For `conservative`, implement `_pad_to_safe_square(image, fill, max_scale=1.1, max_translation_fraction=0.05)`:

```python
diagonal = math.ceil(math.hypot(image.width, image.height))
side = math.ceil(diagonal * max_scale / (1 - 2 * max_translation_fraction))
canvas = Image.new("RGB", (side, side), color=fill)
canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
```

Then apply one `RandomAffine` whose parameters are degrees `180`, translate `(0.05, 0.05)`, scale `(0.9, 1.1)`, bicubic interpolation, and dynamic edge fill; these are parameters of one affine transform, not three sequential geometric operations. Follow it with horizontal flip, light `RandomApply(ColorJitter, p=0.8)`, `Resize((image_size, image_size))`, tensor conversion, and normalization. Instantiate the affine operation per call only where needed to supply the dynamic fill; keep all other transform objects on the dataset/callable.

Update `MultiCropDataset.__getitem__()` and `MetricDataset.__getitem__()` to estimate the background once after RGB conversion and pass the same tuple to every fill-aware callable. After each callable returns, validate a three-channel tensor with the requested square spatial size (`global_crop_size`, `local_crop_size`, or `image_size`); otherwise raise `ValueError` naming the expected and actual shape. `MetricDataset` must skip estimation for `none`, preserving the exact existing deterministic path.

- [ ] **Step 8: Run dataset tests**

Run:

```bash
pytest tests/test_training_dataset.py -v
```

Expected: all dataset tests pass, including existing recursive-path tests.

---

### Task 2: Enforce checkpoint augmentation continuity in the trainer

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Test: `tests/training/test_pretrain_alignment.py`
- Test: `tests/test_cli_smoke.py` for checkpoint payload integration

**Interfaces:**
- Consumes: `build_pretrain_augmentation_config()`, `build_finetune_augmentation_config()`, `MultiCropDataset(..., augmentation_profile=...)`, and `MetricDataset(..., augmentation_profile=...)` from Task 1.
- Consumes: `ORIENTATION_POLICIES` and passes the selected `orientation_policy` to pretraining datasets/configuration.
- Produces: `_select_augmentation_profile(requested_profile: str | None, checkpoint: dict[str, Any] | None, *, stage: str) -> str` and `_select_orientation_policy(requested_policy: str | None, checkpoint: dict[str, Any] | None, *, stage: str) -> str`.
- Produces: `_validate_augmentation_config(profile: str, current_config: dict[str, object], checkpoint: dict[str, Any] | None, *, stage: str) -> dict[str, object]`.
- Produces: `_resolve_pretrain_local_views(requested_local_crop_size: int | None, requested_local_crops: int | None, checkpoint: dict[str, Any] | None) -> tuple[int, int]` for pretrain new-run defaults and resume continuity.
- Uses the existing resolution-consistency size resolver and validation contract; no augmentation-specific image-size resolver is added.
- Produces: `args.augmentation` and `args.orientation_policy` as the resolved effective values before dataset construction.
- Produces: `config.augmentation_profile` and `config.augmentation_config` in every new training checkpoint; `augmentation_config.orientation_policy` is the single stored source of truth for orientation policy.

- [ ] **Step 1: Add failing resume-resolution unit tests**

In `tests/training/test_pretrain_alignment.py`, add a local helper that builds current expanded configs using Task 1 builders. Cover every branch:

```python
@pytest.mark.parametrize(
    ("stage", "expected_profile"),
    [("pretrain", "global-barcode"), ("finetune", "none")],
)
def test_new_run_uses_stage_defaults(stage, expected_profile):
    profile = trainer._select_augmentation_profile(None, None, stage=stage)
    policy = trainer._select_orientation_policy(None, None, stage=stage)
    config = _augmentation_config(stage, profile, policy)
    assert (profile, policy) == (expected_profile, "sensitive")
    assert trainer._validate_augmentation_config(
        profile, config, None, stage=stage
    ) == config


@pytest.mark.parametrize(
    ("stage", "legacy_profile", "expected_policy"),
    [
        ("pretrain", "legacy", "invariant"),
        ("finetune", "none", "sensitive"),
    ],
)
def test_old_checkpoint_maps_to_compatible_augmentation(
    stage, legacy_profile, expected_policy
):
    checkpoint = {"config": {"model_name": "vit_tiny_patch16_224"}}
    profile = trainer._select_augmentation_profile(
        None, checkpoint, stage=stage
    )
    policy = trainer._select_orientation_policy(None, checkpoint, stage=stage)
    config = _augmentation_config(stage, profile, policy)
    assert (profile, policy) == (legacy_profile, expected_policy)
    assert trainer._validate_augmentation_config(
        profile, config, checkpoint, stage=stage
    ) == config
```

Do not duplicate the existing size-precedence or validation matrix from resolution-consistency work. Add only augmentation metadata round-trip assertions: a finetune checkpoint's `augmentation_config.image_size` resolves to that value, and a pretrain checkpoint's `augmentation_config.global_crop.size` resolves to that value when initializing fine-tuning. The remaining augmentation tests assert that the trainer passes the already resolved training size consistently to the backbone, `MetricDataset`, and `build_finetune_augmentation_config()`.

Also assert:

- omitted profile and orientation policy inherit a checkpoint's profile/policy/config when resuming; for new fine-tuning initialized from `--checkpoint`, omitted orientation policy inherits the pretraining checkpoint when present, while the augmentation profile remains independently defaulted to `none`;
- when a pre-augmentation pretraining checkpoint is used with `--checkpoint`, omitted orientation policy resolves to `sensitive` (the new-run default); explicit `--orientation-policy invariant` resolves to `invariant`; explicit `sensitive` is honored for the new fine-tuning run; explicit `--augmentation conservative` initializes successfully without inheriting the pretraining profile;
- new runs with omitted orientation policy resolve to `sensitive`; explicit `sensitive` is preserved;
- explicitly matching profile/policy/config succeeds;
- explicit `sensitive` is accepted for a new `legacy` run and recorded, but cannot change its transform;
- an explicit different profile raises `ValueError` containing `Cannot resume` and `new run`;
- a same-named profile or orientation policy with one changed expanded value raises `ValueError` containing `configuration differs`;
- old pretrain plus explicit non-`legacy` fails;
- old pretrain plus explicit `--orientation-policy invariant` succeeds, while explicit `sensitive` fails on resume;
- pretrain resume inherits saved local-crop size/count when their CLI options are omitted; explicit local-view values must match the saved expanded config for new checkpoints or the saved `args` for old checkpoints, otherwise fail with `Cannot resume`;
- an old checkpoint missing or invalid local-view fields falls back to the historical `96` local crop size or `6` local crops respectively; `local_crop_size` accepts only non-boolean positive integers, while `local_crops` accepts non-boolean non-negative integers and a saved `local_crops=0` is inherited unchanged; explicit values conflicting with those resolved values fail;
- old pretrain used with `--checkpoint` plus omitted policy resolves to `sensitive`, explicit `sensitive` is honored for the new finetune run, and explicit `--augmentation conservative` initializes successfully;
- old finetune plus explicit `conservative` fails;
- old finetune plus explicit `--orientation-policy invariant` succeeds, while explicit `sensitive` fails;
- profile without config, config without profile, non-string profile, or non-dict expanded config raises `ValueError` containing `malformed augmentation metadata`;
- unsupported `stage` raises `ValueError`.

- [ ] **Step 2: Run resume-resolution tests and verify they fail**

Run:

```bash
pytest tests/training/test_pretrain_alignment.py -k augmentation -v
```

Expected: failures because the augmentation profile/policy resolution and validation helpers do not exist.

- [ ] **Step 3: Implement strict augmentation resolution**

Add the augmentation selection, local-view resolution, and config-validation helpers near the existing resume helpers. Reuse the existing resolution-consistency size resolver rather than creating a local copy. The augmentation helpers' combined behavior must be:

```text
new pretrain + omitted profile/policy -> global-barcode/sensitive/current config
new finetune + omitted profile/policy -> none/sensitive/current config
old pretrain resume                  -> legacy/invariant/current legacy config
old finetune resume                  -> none/invariant/current none config
new finetune from --checkpoint        -> none/saved-pretrain-policy/current config
new pretrain + omitted local views   -> local_crop_size=96/local_crops=6
pretrain resume + omitted local views -> saved local-view config/current config
pretrain resume + explicit local views -> require exact saved local-view config
resume + omitted profile/policy      -> saved profile/policy/current config
resume + explicit profile/policy     -> require exact profile/policy/current config
```

Call them in this exact order:

```python
profile = _select_augmentation_profile(requested_profile, checkpoint, stage=stage)
policy = _select_orientation_policy(requested_policy, checkpoint, stage=stage)
if stage == "pretrain":
    local_crop_size, local_crops = _resolve_pretrain_local_views(
        requested_local_crop_size, requested_local_crops, checkpoint
    )
if stage == "pretrain":
    current_config = builder(
        profile, ..., local_crop_size=local_crop_size,
        local_crops=local_crops, orientation_policy=policy,
    )
else:
    current_config = builder(profile, ..., orientation_policy=policy)
augmentation_config = _validate_augmentation_config(
    profile, current_config, checkpoint, stage=stage
)
```

Treat a checkpoint as old only when both augmentation keys are absent from `checkpoint.get("config", {})`; an old checkpoint has no saved orientation policy and maps to `invariant` only for compatibility. `_select_augmentation_profile()` rejects an explicit non-compatible profile for an old checkpoint and rejects a profile/config key mismatch. `_select_orientation_policy()` applies the new-run default, resume inheritance, and fine-tuning `--checkpoint` initialization inheritance without inheriting the pretraining profile. For every pretrain resume, `_resolve_pretrain_local_views()` resolves omitted local-view options from saved metadata (`augmentation_config.local_crop.size` and `augmentation_config.local_crops`) or, for an old checkpoint, saved `args.local_crop_size` and `args.local_crops`; explicit options must match. `local_crop_size` accepts only non-boolean positive integers; `local_crops` accepts non-boolean non-negative integers, including `0` to disable local views. Only a missing or invalid old field falls back to the historical value (`96` or `6`); a saved valid `local_crops=0` remains `0`. Resolve local views before building `current_config`, write the effective values back to `args.local_crop_size` and `args.local_crops`, then perform every later local-view validation and dataset construction through those fields. `_validate_augmentation_config()` rejects wrong metadata types and exact-config differences, including `augmentation_config.orientation_policy`, otherwise returns `current_config`. Compare ordinary JSON-serializable dictionaries with exact equality; do not add profile versions, migrations, tolerances, or deep-diff dependencies.

Use the existing resolution-consistency implementation for fine-tuning input-size precedence and validation. The augmentation task must pass that resolved size to `build_finetune_augmentation_config()` and `MetricDataset`, without deriving it from the timm model name or introducing duplicate metadata rules.

- [ ] **Step 4: Run resume-resolution tests and verify they pass**

Run:

```bash
pytest tests/training/test_pretrain_alignment.py -k augmentation -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Add failing trainer integration tests for profile/policy propagation and checkpoint metadata**

Extend existing one-epoch CLI/trainer smoke tests rather than creating a second training harness. For pretrain and finetune, load the produced latest checkpoint and assert:

```python
saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
assert saved["config"]["augmentation_profile"] == expected_profile
assert saved["config"]["augmentation_config"]["profile"] == expected_profile
assert saved["config"]["augmentation_config"]["orientation_policy"] == expected_policy
assert "orientation_policy" not in saved["config"]
```

Monkeypatch `MultiCropDataset` and `MetricDataset` in focused tests to capture constructor arguments and assert `run_pretrain`/`run_finetune` passes the resolved profile before creating the DataLoader. In the finetune test, put `global_crop_size=32` in a legacy pretrain checkpoint's `args`; assert `OTUFormerEncoder` and `MetricDataset` both receive `image_size`/`img_size=32` and the expanded finetune config records `image_size == 32`. Add separate assertions that `resolve_training_image_size()` round-trips a finetune checkpoint's `augmentation_config.image_size` and resolves a pretrain checkpoint's `augmentation_config.global_crop.size` when initializing fine-tuning.

Add a fine-tune initialization test where the input pretrain checkpoint says `augmentation_profile="color-robust"`, `args.resume` is empty, and `args.augmentation` is `None`; assert `MetricDataset` receives `none`, proving `--checkpoint` does not inherit pretraining augmentation.

Add a focused test that calls `run_pretrain()` or `run_finetune()` directly and uses `capsys` to assert trainer output contains both `Augmentation profile:` and a JSON rendering of `Augmentation config:`. Do not assert against the CLI callback's earlier `Parameters:` block: that block intentionally records the raw option (`null` when omitted), while trainer output records the resolved profile.

- [ ] **Step 6: Run integration tests and verify they fail**

Run:

```bash
pytest tests/test_cli_smoke.py -k "pretrain or finetune" -v
```

Expected: new metadata/propagation assertions fail because trainer wiring is absent.

- [ ] **Step 7: Reuse existing early checkpoint loading and wire resolved profiles**

In `run_pretrain()`, the resolution-consistency baseline already loads `args.resume` with `torch.load()` before constructing `MultiCropDataset` or `DataLoader`; preserve that ordering and reuse `resume_ckpt` for model/optimizer/schedule restoration. Do not replace it with a second load or move already-correct input-size logic.

1. Select the effective profile and orientation policy after that checkpoint is available. Resolve local views before any local-view validation or dataset construction, then immediately set `args.local_crop_size` and `args.local_crops` to those effective values; use `96`/`6` only for a new run whose options are omitted.
2. Build the current pretrain config from the resolved `args.local_crop_size`/`args.local_crops` and validate it against checkpoint metadata.
3. Set `args.augmentation` and `args.orientation_policy` to the resolved values.
4. Print the profile and sorted/indented JSON config.
5. Construct `MultiCropDataset(..., augmentation_profile=profile, orientation_policy=policy)` using the resolved `args` local-view fields.
6. Add both augmentation fields to the existing checkpoint `config` dictionary.

In `run_finetune()`:

1. Keep loading the chosen model checkpoint before model construction.
2. Reuse the already resolved fine-tuning input size from the resolution-consistency flow and instantiate `OTUFormerEncoder(..., img_size=finetune_image_size)`. The current default and existing baseline remain `224`; a checkpoint trained at another recorded global size keeps that actual input contract.
3. Pass checkpoint metadata into augmentation resume resolution only when `resume_path is not None`; pass `None` for profile validation when this is a new run initialized through `--checkpoint`, but retain the pretraining checkpoint as the source for omitted orientation-policy inheritance.
4. Build/validate `none` or `conservative` with `finetune_image_size` and the resolved orientation policy, set `args.augmentation` and `args.orientation_policy`, log the profile/policy/config, and pass `image_size=finetune_image_size`, the profile, and the policy to `MetricDataset`.
5. Reuse `finetune_image_size` for periodic evaluation when `--extract-size` is `auto`; use the existing `center_crop_eval_transform` path and do not change evaluation-transform or CAM behavior.
6. Add both augmentation fields to the finetune checkpoint `config` dictionary; do not add a duplicate top-level orientation-policy field.

Keep all loss calls, global/local view slicing, schedule construction, optimization, EMA updates, periodic evaluation, and save filenames unchanged.

- [ ] **Step 8: Run trainer and checkpoint tests**

Run:

```bash
pytest tests/training/test_pretrain_alignment.py tests/test_cli_smoke.py -v
```

Expected: all tests pass.

---

### Task 3: Add the high-level CLI contract and detailed help

**Files:**
- Modify: `src/otuformer/cli/pretrain.py`
- Modify: `src/otuformer/cli/finetune.py`
- Test: `tests/test_cli_smoke.py` and existing CLI argument-forwarding tests

**Interfaces:**
- Consumes: `PRETRAIN_AUGMENTATIONS` and `FINETUNE_AUGMENTATIONS` from Task 1 through a lazy callback/import so basic CLI startup does not eagerly import torch/torchvision.
- Produces: pretrain `--augmentation [global-barcode|color-robust|legacy]`, internally `str | None`, new-run default described as `global-barcode`.
- Produces: pretrain `--orientation-policy [invariant|sensitive]`, internally `str | None`, default `sensitive` for new runs; it applies to every global and local view.
- Produces: finetune `--orientation-policy [invariant|sensitive]`, internally `str | None`; it affects `conservative` and is a no-op for `none`; omitted initialization inherits the pretraining checkpoint policy.
- Help tests assert that `--orientation-policy`, both policy names, and the exact phrase `default for a new run: sensitive` are present for both commands.
- Produces: finetune `--augmentation [none|conservative]`, internally `str | None`, new-run default described as `none`.
- Produces: `args.augmentation` in the namespace, preserving `None` when omitted so Task 2 can distinguish omission from an explicit value.

- [ ] **Step 1: Add failing CLI help and forwarding tests**

Add tests to `tests/test_cli_smoke.py`:

```python
@pytest.mark.parametrize(
    ("command", "profiles", "default"),
    [
        ("pretrain", ["global-barcode", "color-robust", "legacy"], "global-barcode"),
        ("finetune", ["none", "conservative"], "none"),
    ],
)
def test_training_help_documents_augmentation_contract(command, profiles, default):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    output = result.output.lower()
    assert "--augmentation" in output
    assert "--orientation-policy" in output
    assert "invariant" in output and "sensitive" in output
    assert all(profile in output for profile in profiles)
    assert f"default for a new run: {default}" in output
    assert "default for a new run: sensitive" in output
    assert "dorsal" in output and "ventral" in output and "lateral" in output
    assert "in-plane" in output
    assert "does not guarantee" in output


def test_pretrain_help_warns_about_color_robust_and_legacy():
    output = runner.invoke(app, ["pretrain", "--help"]).output.lower()
    assert "diagnostic" in output
    assert "metallic" in output
    assert "0.2.1" in output
    assert "new run" in output


def test_finetune_help_marks_conservative_experimental():
    output = runner.invoke(app, ["finetune", "--help"]).output.lower()
    assert "experimental" in output
    assert "new run" in output
```

Extend existing monkeypatched argument-forwarding tests so omitted `--augmentation`, `--orientation-policy`, `--local-crop-size`, and `--local-crops` reach trainer as `None`, while explicit values reach trainer unchanged. Add invalid-value invocations for both profile and policy and assert a nonzero exit with an explicit error before mocked trainer execution. Add a fine-tuning initialization case proving that omitted policy inherits from `--checkpoint` while omitted augmentation resolves independently to `none`.

- [ ] **Step 2: Run CLI contract tests and verify they fail**

Run:

```bash
pytest tests/test_cli_smoke.py -k "augmentation_contract or color_robust or conservative_experimental or augmentation_argument" -v
```

Expected: failures because neither CLI exposes `--augmentation`.

- [ ] **Step 3: Implement lazy CLI validation and help**

Add `augmentation: str | None = typer.Option(None, "--augmentation", help=...)` to each callback. Add `orientation_policy: str | None = typer.Option(None, "--orientation-policy", help=...)` to both callbacks. Change pretrain `--local-crop-size` and `--local-crops` to `int | None` options with `None` defaults, preserving omitted values in `argparse.Namespace`; the trainer resolves `96` and `6` only for new pretraining runs, or inherited checkpoint values on resume. Add short local validation helpers invoked before `prepare_output_dir()`:

```python
def _validate_augmentation(value: str | None, *, stage: str) -> None:
    if value is None:
        return
    from otuformer.training.dataset import (
        FINETUNE_AUGMENTATIONS,
        PRETRAIN_AUGMENTATIONS,
    )

    allowed = PRETRAIN_AUGMENTATIONS if stage == "pretrain" else FINETUNE_AUGMENTATIONS
    if value not in allowed:
        choices = ", ".join(allowed)
        raise typer.BadParameter(
            f"Unknown {stage} augmentation profile '{value}'. Choose one of: {choices}."
        )


def _validate_orientation_policy(value: str | None) -> None:
    from otuformer.training.dataset import ORIENTATION_POLICIES

    if value is not None and value not in ORIENTATION_POLICIES:
        raise typer.BadParameter(
            f"Unknown orientation policy '{value}'. Choose one of: {', '.join(ORIENTATION_POLICIES)}."
        )
```

`_validate_orientation_policy()` imports `ORIENTATION_POLICIES` locally. Invoke both validators before `prepare_output_dir()` so invalid profile or policy values cannot call the mocked trainer or alter output directories.

Do not duplicate transform strengths in Python help text. Help must concisely state:

- valid names and effective new-run default;
- marker-view rule: dorsal/ventral/lateral/anatomical-part images are distinct markers, while arbitrary in-plane orientation is supported;
- `color-robust` can suppress diagnostic color/pattern/metallic sheen;
- `conservative` is experimental;
- `orientation-policy=sensitive` is the caller-selected (never inferred) policy and the new-run default for direction-sensitive markers: every new pretraining view and fine-tuning `conservative` view has no horizontal flip and only `[-15°, 15°]` rotation;
- `legacy` reproduces 0.2.1; it records either policy without changing its historical transforms, but an omitted policy can still be inherited by a new fine-tuning `conservative` run from that checkpoint;
- omitted values inherit on resume, conflicts fail, and parameter changes require a new run;
- augmentation encourages but does not guarantee invariance;
- `sensitive` is caller-selected and is not inferred automatically from the image or taxon;
- all global/local pretraining views use the selected policy, so local-to-global training cannot silently reintroduce broad rotation or reflection.

Keep `_format_user_command()` unchanged: it should print `--augmentation` and `--orientation-policy` only when explicitly supplied.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest tests/test_cli_smoke.py -v
```

Expected: all CLI tests pass and existing safety-option ordering remains intact.

---

### Task 4: Update the public design and bilingual training documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-03-29-otuformer-design.md:76-123`
- Modify: `README.md:144-316`
- Modify: `README.cn.md:144-316`
- Do not modify: `docs/superpowers/specs/2026-03-30-otuformer-pretrain-alignment-design.md`

**Interfaces:**
- Consumes: final CLI names/defaults and resume behavior from Tasks 1-3.
- Produces: English and Chinese user documentation with equivalent profile guidance.

- [ ] **Step 1: Update the main OTU-Former design command contract**

In the `pretrain` and `finetune` sections of `2026-03-29-otuformer-design.md`, add `--augmentation` and `--orientation-policy` to the parameter lists and summarize:

```text
pretrain: global-barcode (default), color-robust, legacy
finetune: none (default), conservative
orientation-policy: sensitive (default for new runs), invariant
```

State that omitted profile/policy values inherit on resume, old pretrain checkpoints map to `legacy`/`invariant`, old finetune checkpoints map to `none`/`invariant`, fine-tuning initialization inherits only the pretraining policy, and explicit mismatches fail. Link to `docs/superpowers/specs/2026-09-07-otuformer-training-augmentation-design.md` for transform-level details. Do not copy the full profile configuration into this broader design.

- [ ] **Step 2: Update the English README**

In `README.md`:

- add `--augmentation global-barcode` to the basic pretrain example and `--augmentation conservative` to the custom finetune example;
- add `--augmentation` and `--orientation-policy`, with effective defaults, to both parameter tables;
- explain that both fine-tuning profiles use the checkpoint-recorded training input size, with a `224` fallback for old checkpoints;
- add one compact "Training augmentation profiles" subsection between pretrain and finetune or immediately after the finetune table;
- explain the complete-source-marker versus partial SSL crop trade-off;
- state that dorsal, ventral, lateral, whole-body, and anatomical-part markers must not be mixed as one marker identity;
- explain `global-barcode`, the color-signal risk of `color-robust`, 0.2.1 compatibility of `legacy`, unchanged default `none`, and experimental `conservative`;
- describe profile and orientation-policy inheritance/conflict rules, including that changing expanded profile or policy parameters starts a new run;
- state that checkpoint metadata provides configuration traceability, not bitwise deterministic replay;
- retain existing training, output-safety, and checkpoint instructions.

Do not document a dense profile, low-level augmentation flags, TTA, or an augmentation-evaluation command.

- [ ] **Step 3: Mirror the guidance in the Chinese README**

Apply the same content and examples to `README.cn.md` in natural Chinese. Keep literal CLI values, file names, checkpoint keys, and code identifiers in English. Ensure neither README implies mathematical rotation or color invariance.

- [ ] **Step 4: Check documentation consistency**

Run:

```bash
rg -n "augmentation|orientation-policy|orientation_policy|global-barcode|color-robust|legacy|conservative" \
  README.md README.cn.md docs/superpowers/specs/2026-03-29-otuformer-design.md

git diff --check
```

Expected: each public document contains both relevant defaults and profile warnings; `git diff --check` prints nothing.

---

### Task 5: Run automated regression and smoke training

**Files:**
- Modify only if a real failure requires a scoped fix: files already listed in Tasks 1-3
- Test: `tests/test_training_dataset.py`
- Test: `tests/training/test_pretrain_alignment.py`
- Test: `tests/test_cli_smoke.py`
- Regression: all files under `tests/`

**Interfaces:**
- Consumes: all implementation from Tasks 1-4.
- Produces: command output demonstrating dataset, resume, CLI, augmentation metadata, and unchanged resolution-consistency behavior.

- [ ] **Step 1: Run focused automated tests**

Run:

```bash
pytest tests/test_training_dataset.py tests/training/test_pretrain_alignment.py tests/test_cli_smoke.py -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass. Any failure in extraction, CAM, ONNX, periodic metrics, loss pairing, or model behavior is a regression from the augmentation integration or its interaction with the completed resolution-consistency interfaces; fix only the scoped interaction, not those subsystems' independent behavior.

- [ ] **Step 3: Run pretrain smoke jobs for all profiles**

Use the existing Epidorcus example with a temporary output root and one epoch. Select a batch size that leaves at least one complete batch; disable periodic embedding metrics to keep the smoke bounded:

```bash
for profile in global-barcode color-robust legacy; do
  otuformer pretrain \
    --train-data examples/Epidorcus/figs.csv \
    --input-images-dir examples/Epidorcus/images \
    --out-dir "runs/augmentation-smoke/pretrain-${profile}" \
    --augmentation "$profile" \
    --max-epochs 1 \
    --batch-size 2 \
    --num-workers 0 \
    --device cpu \
    --save-every-epochs 1 \
    --disable-embedding-metrics \
    --overwrite
done
```

For each output, load `SSL_latest.pth` and assert finite logged loss plus matching `config.augmentation_profile`, `config.augmentation_config.profile`, `config.augmentation_config.orientation_policy`, and `config.augmentation_config.global_crop.size`. Resume each checkpoint with profile and policy omitted:

```bash
for profile in global-barcode color-robust legacy; do
  otuformer pretrain \
    --train-data examples/Epidorcus/figs.csv \
    --input-images-dir examples/Epidorcus/images \
    --out-dir "runs/augmentation-smoke/pretrain-${profile}" \
    --resume "runs/augmentation-smoke/pretrain-${profile}/SSL_latest.pth" \
    --max-epochs 2 \
    --batch-size 2 \
    --num-workers 0 \
    --device cpu \
    --save-every-epochs 1 \
    --disable-embedding-metrics
done
```

Verify inheritance and a second saved epoch. Then repeat the `global-barcode` resume with `--augmentation legacy`; expect a nonzero exit before DataLoader/training. Also run one `global-barcode --orientation-policy sensitive` smoke job and assert all global/local views record the sensitive rotation and no-flip settings.

- [ ] **Step 4: Run fine-tune smoke jobs for both profiles**

Use the smoke pretrain checkpoint and the existing labeled `examples/Epidorcus/figs.csv`:

```bash
for profile in none conservative; do
  otuformer finetune \
    --checkpoint runs/augmentation-smoke/pretrain-global-barcode/SSL_latest.pth \
    --train-data examples/Epidorcus/figs.csv \
    --input-images-dir examples/Epidorcus/images \
    --out-dir "runs/augmentation-smoke/finetune-${profile}" \
    --augmentation "$profile" \
    --finetune-epochs 1 \
    --batch-size 2 \
    --num-workers 0 \
    --device cpu \
    --save-every-epochs 1 \
    --disable-embedding-metrics \
    --overwrite
done
```

For each output, verify finite loss and matching metadata, including `augmentation_config.orientation_policy`, then resume with profile and policy omitted:

```bash
for profile in none conservative; do
  otuformer finetune \
    --resume "runs/augmentation-smoke/finetune-${profile}/finetune_latest.pth" \
    --train-data examples/Epidorcus/figs.csv \
    --input-images-dir examples/Epidorcus/images \
    --out-dir "runs/augmentation-smoke/finetune-${profile}" \
    --finetune-epochs 2 \
    --batch-size 2 \
    --num-workers 0 \
    --device cpu \
    --save-every-epochs 1 \
    --disable-embedding-metrics
done
```

Verify a second saved epoch. Repeat the `none` resume with `--augmentation conservative` and expect failure. The first-loop `conservative` run also proves that new fine-tuning initialized from a `global-barcode` pretrain checkpoint saves `conservative` rather than inheriting pretrain augmentation. Add one `conservative --orientation-policy sensitive` smoke invocation and assert the saved config records the narrow rotation/no-flip policy.

- [ ] **Step 5: Inspect smoke logs and repository state**

Run:

```bash
rg -n "Augmentation profile|Augmentation config" \
  runs/augmentation-smoke/*/logs/*.log
! rg -qi '\b(nan|inf)\b|traceback' runs/augmentation-smoke/*/logs/*.log

git status --short
git diff --check
```

Expected: the positive log check finds the resolved profile/config, the inverted case-insensitive error check succeeds because no log contains standalone NaN/Inf values or Traceback (and does not mistake `[Info]` for `Inf`), generated `runs/` artifacts remain ignored, and only approved source/docs/test files are modified.

---

### Task 6: Run an extended implementation comparison for `global-barcode`

**Files:**
- Create: `scripts/validate_augmentation_rotation.py`
- Test: `tests/training/test_pretrain_alignment.py`
- Modify after implementation verification: `pyproject.toml`, `src/otuformer/__init__.py` (version `0.3.0` -> `0.4.0`)
- Create after the run: `docs/superpowers/specs/2026-09-07-global-barcode-validation.md`
- Do not modify production extraction/evaluation code to support validation.

**Interfaces:**
- Consumes: an existing 50-epoch baseline checkpoint, the new 50-epoch `global-barcode` checkpoint, a fixed validation image CSV, and existing `OTUFormerEncoder` plus the named deterministic `center_crop_eval_transform`.
- Produces: a development-only script that reports same-image rotation/reflection and different-image cosine summaries without becoming an `otuformer` CLI command.
- Produces: a checked-in implementation-comparison report containing exact commands, data/checkpoint identities and hashes, available core metrics, health checks, and rotation/reflection-consistency summaries. It does not establish broad biological suitability of the default.

- [ ] **Step 1: Add minimal failing testable rotation/reflection-summary functions**

Create `scripts/validate_augmentation_rotation.py` with pure helpers for rotation/reflection summaries:

```python
def summarize_similarities(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Similarity values must be non-empty and finite.")
    return {
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "minimum": float(np.min(values)),
    }
```

Add small tests in `tests/training/test_pretrain_alignment.py` that load the script with `importlib.util`, pass `[0.1, 0.2, 0.3]`, check the three summary values, and reject empty/non-finite input. Add a reflection-summary case and verify the script accepts the same fixed image size resolved independently from each checkpoint. Run tests before implementing the helpers and expect failure, then add the helpers and expect pass.

- [ ] **Step 2: Implement the bounded validation script**

The script accepts only development-script arguments via `argparse`:

```text
--baseline-checkpoint PATH
--candidate-checkpoint PATH
--images-csv PATH
--images-dir PATH
--output-json PATH
--device auto|cpu|cuda|mps
--image-size INT (optional deterministic evaluation-preprocessing override; model sizes always resolve from checkpoints)
```

For both checkpoints:

1. load the checkpoint, then read `config.model_name` and `config.out_dim`/`config.metric_embed_dim`;
2. resolve its training image size with the existing `resolve_training_image_size()` precedence and record it;
3. instantiate `OTUFormerEncoder(..., img_size=checkpoint_training_size, pretrained=False)`, then load `model_state_dict`;
4. choose deterministic evaluation size: `--image-size` when explicitly supplied, otherwise the checkpoint training size; require the two effective evaluation sizes to match before comparing embeddings, and record both evaluation-size resolutions;
5. use `center_crop_eval_transform(effective_evaluation_size)` unchanged;
6. rotate each source PIL image with `expand=True` and its edge-median fill at exactly `0`, `90`, `180`, and `270` degrees, then apply deterministic evaluation preprocessing;
7. extract raw CLS tokens, L2-normalize them, and compute per-image cosine similarity between angle 0 and each non-zero angle;
8. create horizontally reflected counterparts with the same edge-derived fill, compute paired reflection cosine summaries, and include them in JSON; read each checkpoint's `augmentation_config.orientation_policy` and report it, and when it is `sensitive`, report the saved no-flip/narrow-rotation contract without treating reflection similarity as an invariance target;
9. compute different-image cosine values from deterministic angle-0 embeddings using all upper-triangle pairs, or a fixed-seed sample capped at 100,000 pairs for large datasets;
10. emit JSON containing per-checkpoint median, 10th percentile, and minimum for each rotation angle and reflection, plus median and 90th percentile for different-image similarities.

Reuse the existing recursive image-reference resolution logic (`_resolve_image_path()` with relative-path, unique-basename, and ambiguity rules), rather than resolving `images_dir / image` only. Keep this script independent of training augmentation internals except for the existing background-estimation helper when constructing rotated validation views; it validates output embeddings, not transform-object composition. Do not add it to `otuformer.cli.main`.

- [ ] **Step 3: Run the 50-epoch candidate training**

Use `runs/old/pretrain/SSL_epoch_0050.pth` as the existing baseline. Its saved arguments specify the controlled Epidorcus data, seed, backbone, batch size, optimization parameters, crop settings, and evaluation settings. Verify `runs/pretrain-global-barcode-50e` does not already exist, then run:

```bash
test ! -e runs/pretrain-global-barcode-50e
otuformer pretrain \
  --train-data examples/Epidorcus/figs.csv \
  --input-images-dir examples/Epidorcus/images \
  --out-dir runs/pretrain-global-barcode-50e \
  --augmentation global-barcode \
  --model-name vit_tiny_patch16_224 \
  --out-dim 256 \
  --max-epochs 50 \
  --lr 0.0005 \
  --weight-decay 0.05 \
  --warmup-epochs 3 \
  --global-crop-size 224 \
  --local-crop-size 96 \
  --local-crops 6 \
  --mask-ratio 0.5 \
  --lambda-local 1.5 \
  --lambda-mask 1.0 \
  --teacher-momentum 0.995 \
  --teacher-momentum-end 0.999 \
  --student-temp 0.1 \
  --teacher-temp-start 0.04 \
  --teacher-temp-end 0.07 \
  --batch-size 32 \
  --num-workers 4 \
  --cpus 12 \
  --device auto \
  --seed 42 \
  --extract-size auto \
  --metrics-sample-size 10000 \
  --umap-n-neighbors 15 \
  --umap-min-dist 0.1 \
  --umap-metric cosine \
  --visualize-class-number 20 \
  --log-every-n-steps 50 \
  --save-every-epochs 10 \
  --keep-last-checkpoints 10
```

Record this exact command and SHA-256 hashes for `runs/old/pretrain/SSL_epoch_0050.pth` and the candidate checkpoint in the report. Do not claim comparability if any listed data, seed, model, crop, optimizer, batch, or metric setting differs.

- [ ] **Step 4: Collect training-health and embedding metrics**

From candidate logs/checkpoint, record:

- all 50 epochs completed;
- every logged loss finite;
- the last available finite positive `feature_std` in `instant_metrics.pretrain.csv`, together with its epoch and step; do not describe it as a checkpoint-final value;
- different-image similarities are not all numerically equal to one.

Read the final available rows for Recall@1, kNN accuracy at k=1, k=5, and k=20, linear-probe accuracy, mAP, and Silhouette Score from each run's `metrics.pretrain.csv`. Do not add a new evaluator. If a metric is missing or was not computed, record it as `unavailable` rather than fabricating a value. The report compares the available implementation metrics and does not make a broad biological-readiness decision.

- [ ] **Step 5: Run rotation/reflection-consistency validation**

Run:

```bash
python scripts/validate_augmentation_rotation.py \
  --baseline-checkpoint runs/old/pretrain/SSL_epoch_0050.pth \
  --candidate-checkpoint runs/pretrain-global-barcode-50e/SSL_latest.pth \
  --images-csv examples/Epidorcus/figs.csv \
  --images-dir examples/Epidorcus/images \
  --output-json runs/pretrain-global-barcode-50e/rotation-validation.json \
  --device auto
```

Expected acceptance condition: candidate 90-degree median same-image similarity is not below the baseline 90-degree median. Report 90/180/270-degree and horizontal-reflection median, p10, and minimum for both checkpoints, plus different-image median and p90. Treat reflection results as diagnostic only for `sensitive`: its acceptance criterion is the recorded no-flip/narrow-rotation policy, not high reflection similarity. Do not describe any result as guaranteed or universal rotation/reflection invariance.

- [ ] **Step 6: Write the validation report**

Create `docs/superpowers/specs/2026-09-07-global-barcode-validation.md` with:

```text
Data identity and sample counts
Baseline and candidate checkpoint paths/hashes
Exact commands and environment versions
Resolved augmentation profile/config
Training-health table
Seven-metric baseline/candidate/delta table
Rotation same-image table by angle
Reflection same-image table
Different-image similarity table
Technical implementation result: PASS or ISSUE
Scientific suitability: not assessed by this implementation comparison
Observed limitations and follow-up issue links
```

Do not include generated checkpoints, full logs, image data, or JSON artifacts in git.

- [ ] **Step 7: Bump the release version after implementation verification**

Update `pyproject.toml` and `src/otuformer/__init__.py` from `0.3.0` to `0.4.0`. This is a behavior-changing minor release because new pretraining defaults change from the historical augmentation to `global-barcode`.

- [ ] **Step 8: Run final verification without committing**

Run:

```bash
pytest -q
python -m compileall -q src scripts/validate_augmentation_rotation.py
git diff --check
git status --short
```

Expected: tests and compilation pass, whitespace check is empty, and status lists only the intended implementation, test, script, design, README, plan, and validation-report files. Stop and request separate approval before any commit.
