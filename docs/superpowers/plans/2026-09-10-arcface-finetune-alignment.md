# ArcFace Fine-Tune Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt selected legacy ArcFace metric-learning behavior while retaining checkpoint compatibility, adding targeted optimizer controls, and making extraction semantics explicit.

**Architecture:** SSL pretraining continues to use the existing normalized three-layer `ProjectionHead`. A new fine-tune run replaces that projector with a non-normalizing `ArcFaceEmbeddingHead` (`backbone_dim -> 512 -> metric_embed_dim`), while `ArcFaceHead` remains the classifier and sole normalization owner. The separate metric-head LR is an added control, not a legacy-script requirement. Fine-tune checkpoint metadata records the actual embedding-head type so extraction reconstructs the correct module before loading weights.

**Tech Stack:** Python 3.11, PyTorch, timm, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-03-29-otuformer-design.md`

## Global Constraints

- Do not change SSL pretraining architecture, its projector, or its gradient clipping.
- New fine-tune runs use `ArcFaceEmbeddingHead(hidden_dim -> 512 -> metric_embed_dim)` with no output L2 normalization.
- `--finetune-lr` controls backbone LR; `--metric-head-lr` is an added option that defaults to inheriting it for the metric head and ArcFace classifier.
- `--weight-decay` defaults to conservative supervised SFT value `1e-4`; users can pass `0.05` for the legacy script's setting. This is not a claim of full legacy reproduction.
- Fine-tuning does not apply gradient clipping; there is no SFT clipping option.
- New fine-tune checkpoints record the actual `config.embedding_head` (`arcface_mlp_512` for new runs, `projection_mlp_2048` for historical SFT checkpoints); missing metadata remains compatible with old OTU checkpoints.
- On resume, saved optimizer state takes precedence over CLI LR and weight-decay values.
- Preserve raw CLS as extract default. `--use-projector-output` remains compatible and exports the task embedding for new fine-tune checkpoints.
- Bump both package version locations to `0.5.0`.
- Do not create a git commit.

---

### Task 1: Add the ArcFace embedding-head model contract

**Files:**
- Modify: `src/otuformer/training/model.py`
- Modify: `tests/test_training_model.py`

**Interfaces:**
- Produces: `ArcFaceEmbeddingHead(embed_dim: int, metric_embed_dim: int) -> nn.Module`.
- Produces: an output of shape `(batch, metric_embed_dim)` with no mandatory unit norm.
- Consumed by: `run_finetune()` and extractor checkpoint reconstruction.

- [ ] **Step 1: Write the failing model test**

```python
from otuformer.training.model import ArcFaceEmbeddingHead


def test_arcface_embedding_head_uses_small_unnormalized_mlp():
    head = ArcFaceEmbeddingHead(embed_dim=192, metric_embed_dim=256)
    output = head(torch.randn(4, 192))

    assert tuple(head.net[0].weight.shape) == (512, 192)
    assert tuple(head.net[2].weight.shape) == (256, 512)
    assert output.shape == (4, 256)
    assert not torch.allclose(output.norm(dim=1), torch.ones(4), atol=1e-5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_training_model.py::test_arcface_embedding_head_uses_small_unnormalized_mlp -q`

Expected: FAIL because `ArcFaceEmbeddingHead` does not exist.

- [ ] **Step 3: Add the minimal head**

```python
class ArcFaceEmbeddingHead(nn.Module):
    def __init__(self, embed_dim: int, metric_embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, metric_embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

Do not alter `ProjectionHead`; it remains the SSL-specific normalized head.

- [ ] **Step 4: Run the focused model tests**

Run: `pytest tests/test_training_model.py -q`

Expected: PASS.

### Task 2: Align fine-tune initialization and optimizer groups

**Files:**
- Modify: `src/otuformer/cli/finetune.py`
- Modify: `src/otuformer/training/trainer.py`
- Modify: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `ArcFaceEmbeddingHead` from Task 1.
- Produces: `--metric-head-lr: float | None`, with `None` inheriting `--finetune-lr`.
- Produces: `--weight-decay: float`, default `1e-4`.
- Produces: AdamW parameter groups for `backbone` and `metric_head` (list order matters; the group dicts carry no `name` key), both with the requested decay; the metric-head group contains the embedding head and ArcFace classifier.
- Produces: `--metric-embed-dim: int | None` (default `None` inherits the checkpoint's metric dimension) so the flag actually resizes the ArcFace head.
- Produces: `config.freeze_ratio` in every new checkpoint, validated on resume.

- [ ] **Step 1: Write failing CLI forwarding tests**

```python
def test_finetune_optimizer_arguments_default_and_override(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        "otuformer.training.trainer.run_finetune",
        lambda args: seen.update(vars(args)),
    )

    result = runner.invoke(app, [
        "finetune", "--checkpoint", str(tmp_path / "source.pth"),
        "--train-data", str(tmp_path / "labels.csv"),
        "--input-images-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"),
    ])

    assert result.exit_code == 0
    assert seen["metric_head_lr"] is None
    assert seen["weight_decay"] == 1e-4
```

Add a second invocation with `--finetune-lr 3e-5 --metric-head-lr 1e-4 --weight-decay 0.05` and assert those three values reach the trainer.

- [ ] **Step 2: Run the forwarding test to verify it fails**

Run: `pytest tests/test_cli_smoke.py::test_finetune_optimizer_arguments_default_and_override -q`

Expected: FAIL because the options are not recognized or absent from `args`.

- [ ] **Step 3: Implement the CLI options and optimizer assembly**

Add options:

```python
metric_head_lr: float | None = typer.Option(
    None, "--metric-head-lr",
    help="Learning rate for the ArcFace embedding head and classifier; defaults to --finetune-lr.",
)
weight_decay: float = typer.Option(
    1e-4, "--weight-decay",
    help="AdamW weight decay for fine-tuning; pass 0.05 for the legacy script's setting.",
)
```

In `run_finetune()` after loading the SSL backbone:

```python
# encoder_out_dim sizes the pretrained projector; out_dim is the fine-tune
# embedding width and may differ (ArcFace head only).
model = OTUFormerEncoder(
    model_name=model_name, out_dim=encoder_out_dim, img_size=finetune_image_size
)
state_dict = ckpt["model_state_dict"]
embedding_head = _select_finetune_embedding_head(ckpt)
if embedding_head == ARCFACE_EMBEDDING_HEAD:
    model.projector = ArcFaceEmbeddingHead(model.backbone.num_features, out_dim)
elif embedding_head == PROJECTION_EMBEDDING_HEAD and out_dim != encoder_out_dim:
    raise ValueError("--metric-embed-dim cannot resize a ProjectionHead checkpoint")

# Reuse the saved projector only when it matches the head being trained.
source_head = resolve_checkpoint_embedding_head(ckpt, state_dict)
source_dim = resolve_projector_out_dim(state_dict)
if source_head != embedding_head or (source_dim is not None and source_dim != out_dim):
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("projector.")}
model.load_state_dict(state_dict, strict=False)

_freeze_backbone_blocks(model, args.freeze_ratio)
optimizer = _build_finetune_optimizer(
    model,
    loss_fn,
    args.finetune_lr,
    getattr(args, "metric_head_lr", None),
    getattr(args, "weight_decay", 1e-4),
)
```

Do not call `clip_grad_norm_` in the fine-tune loop. Do not change the pretrain clipping call.

- [ ] **Step 4: Write a focused optimizer-group unit test**

Extract optimizer construction into a small private helper if needed only to keep it directly testable. Assert the helper gives backbone and metric-head groups effective LRs `3e-5` and `1e-4`, and `weight_decay == 1e-4` for each group.

- [ ] **Step 5: Run focused training and CLI tests**

Run: `pytest tests/test_training_model.py tests/test_cli_smoke.py -k "finetune or arcface_embedding" -q`

Expected: PASS.

### Task 3: Persist and restore the fine-tune head type

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Modify: `src/otuformer/embedding/extractor.py`
- Modify: `tests/test_extractor.py`
- Modify: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `config.embedding_head == "arcface_mlp_512"` in a fine-tune checkpoint.
- Produces: a model with `ArcFaceEmbeddingHead` installed before state-dict loading.
- Compatibility: `embedding_head="projection_mlp_2048"` or the current OTU historical `loss_state_dict` marker continues to use the historical `ProjectionHead`; checkpoints without either marker remain valid SSL initialization checkpoints. Ref-script `model`/`loss_func` checkpoints are rejected by `finetune` (see the follow-up section) but readable by `extract`/`export`/`cam`.

- [ ] **Step 1: Write a failing extractor loader test**

Create a small fine-tune checkpoint with:

```python
model = OTUFormerEncoder(..., out_dim=64, pretrained=False)
model.projector = ArcFaceEmbeddingHead(model.backbone.num_features, 64)
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "model_name": "vit_tiny_patch16_224",
        "out_dim": 64,
        "image_size": 224,
        "embedding_head": "arcface_mlp_512",
    },
}, checkpoint_path)
```

Assert `_load_model()` returns a model whose `projector` is `ArcFaceEmbeddingHead`, then verify one state-dict tensor equals the saved head tensor.

- [ ] **Step 2: Run the new loader test to verify it fails**

Run: `pytest tests/test_extractor.py::test_load_model_rebuilds_arcface_embedding_head -q`

Expected: FAIL because `_load_model()` creates `ProjectionHead` before loading.

- [ ] **Step 3: Add checkpoint metadata and extraction reconstruction**

Every new fine-tune checkpoint records the head it actually trained, so readers
reconstruct the module the saved weights fit:

```python
"embedding_head": "arcface_mlp_512",     # new runs
"embedding_head": "projection_mlp_2048",  # runs initialised from a historical SFT checkpoint
```

Readers additionally infer the head from the projector shapes when metadata is
absent, so a missing or wrong value cannot cause a silent mis-load.

In `_load_model()`, after constructing the encoder and before `load_state_dict`, replace `model.projector` with `ArcFaceEmbeddingHead(model.backbone.num_features, out_dim)` only when that field equals `"arcface_mlp_512"`. Historical `projection_mlp_2048` metadata and missing metadata use the existing projector. Reject unsupported non-empty values with a `ValueError` naming the value. Fine-tune initialization detects the current OTU historical `loss_state_dict` marker so it does not discard the trained projector; ref-script `model`/`loss_func` checkpoints are rejected by `finetune` because their classifier format differs (the follow-up section adds read-only support in `extract`/`export`/`cam`).

- [ ] **Step 4: Add fine-tune metadata and historical-head assertions**

Extend `test_finetune_runs_one_epoch` to assert:

```python
assert saved["config"]["embedding_head"] == "arcface_mlp_512"
```

- [ ] **Step 5: Run loader and smoke tests**

Run: `pytest tests/test_extractor.py tests/test_cli_smoke.py -k "arcface_embedding_head or finetune_runs_one_epoch" -q`

Expected: PASS.

### Task 4: Clarify extraction semantics and update documentation

**Files:**
- Modify: `src/otuformer/cli/extract.py`
- Modify: `docs/superpowers/specs/2026-03-29-otuformer-design.md`
- Modify: `pyproject.toml`
- Modify: `src/otuformer/__init__.py`
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Preserves: `--use-projector-output` boolean behavior.
- Documents: raw CLS is the default; SSL projector output and ArcFace task embedding have different downstream uses.
- Produces: package version `0.5.0` from both version sources.

- [ ] **Step 1: Write failing help-text tests**

```python
def test_extract_help_explains_cls_and_task_embedding_use_cases():
    output = runner.invoke(app, ["extract", "--help"]).output.lower()
    assert "raw cls" in output
    assert "arcface embedding" in output
    assert "unseen classes" in output
```

- [ ] **Step 2: Run the help test to verify it fails**

Run: `pytest tests/test_cli_smoke.py::test_extract_help_explains_cls_and_task_embedding_use_cases -q`

Expected: FAIL because the required semantic guidance is absent.

- [ ] **Step 3: Update only user-facing help and design contract**

State in `--use-projector-output` help that it exports SSL projector features for SSL checkpoints and ArcFace embeddings for new fine-tune checkpoints; recommend it for fine-tuned-class retrieval/closed-set clustering. State in `--token-mode` help that `cls` defaults to raw backbone CLS and raw CLS remains the comparison baseline for unseen classes, cross-dataset transfer, or general morphology representation.

Update design sections 3.2 and 3.3 with the new SFT head, deliberately conservative optimizer default, resume optimizer precedence, absence of SFT clipping, actual `embedding_head` metadata, and the two extraction semantic cases. Update the English and Chinese README fine-tune/extract tables. Do not rewrite historical plans.

Set:

```toml
version = "0.5.0"
```

and:

```python
__version__ = "0.5.0"
```

- [ ] **Step 4: Run help, version, and focused regression tests**

Run: `pytest tests/test_cli_smoke.py -k "extract_help or finetune" tests/test_training_model.py tests/test_extractor.py -q`

Expected: PASS.

### Task 5: Full verification and no-commit handoff

**Files:**
- Verify: changed files from Tasks 1-4

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -q`

Expected: PASS with no failures.

- [ ] **Step 2: Inspect the final changes**

Run:

```bash
git diff --check
git diff --stat
git status --short
```

Expected: no whitespace errors, expected source/tests/docs/version changes, and no staged files or commits.

- [ ] **Step 3: Report evidence**

Report exact commands and outcomes; note that no commit was created.

---

## Follow-Up: review round 2 (export/cam, metric dim, head reuse, freeze continuity, legacy reads)

**Goal:** Close the gaps found after the first implementation: new fine-tune checkpoints broke ONNX export and CAM, `--metric-embed-dim` was inert, a trained head was silently discarded, `--freeze-ratio` changes broke resume, and ref-script checkpoints could not be read at all.

**Interfaces:**
- Produces: `otuformer.utils.checkpoint.resolve_checkpoint()` → `CheckpointArchitecture(model_name, embedding_dim, embedding_head, state_dict, source_key)`; `apply_checkpoint_weights(model, architecture)`; `resolve_checkpoint_embedding_head()`; `resolve_projector_out_dim()`; `normalize_legacy_projector_keys()`.
- Consumes: `CheckpointArchitecture` in `embedding/extractor._load_model`, `vision/export.export_to_onnx`, `vision/cam.load_model_from_checkpoint`.
- Produces: `config.freeze_ratio` and `_validate_finetune_freeze_ratio()`.

### Task 6: One shared checkpoint resolver for read-only consumers

- `resolve_checkpoint()` prefers `config`, then a ref-script `args` dict, then projector shapes, then the caller's default model name. It is the only place that decides the embedding head, so extract/export/cam can no longer drift apart.
- `apply_checkpoint_weights()` installs `ArcFaceEmbeddingHead` when the checkpoint holds one and loads the (optionally renamed) state dict with `strict=False`.
- Head resolution prefers `config.embedding_head`, then infers from the projector's first linear width (512 → `arcface_mlp_512`, 2048 → `projection_mlp_2048`), and rejects unrecognised widths.
- `normalize_legacy_projector_keys()` maps ref-script `projector.<i>.*` onto `projector.net.<i>.*` and returns the input object unchanged when nothing needs renaming.

### Task 7: Fix fine-tune dimension, head reuse, and resume continuity

- `encoder_out_dim` (pretrained projector) and `out_dim` (fine-tune embedding) are now separate. `--metric-embed-dim` defaults to `None` and means "inherit the checkpoint's recorded metric dimension"; an explicit value wins. A conflicting explicit value on a `ProjectionHead` checkpoint is rejected instead of silently ignored.
- The saved projector is reused whenever the head being trained matches the checkpoint's head **and** the width matches. `--checkpoint <arcface checkpoint>` therefore continues from the trained head; `--checkpoint <SSL checkpoint>` still starts from a fresh head.
- `config.freeze_ratio` is saved and validated on `--resume`, because the optimizer only holds `requires_grad` backbone parameters and a changed freeze ratio changes a group's parameter count.

### Task 8: Ref-script checkpoints are read-only compatible

- Supported: `model` / `teacher` / `student` weight keys, `projector.<i>` naming, `args`-derived metadata, `loss_func`/`loss_state_dict` markers.
- Not supported: resuming a ref-script checkpoint in `finetune` (`_select_finetune_embedding_head` raises with the reason). The ref script's classifier is `loss_func.W` (embedding-major) and is not interchangeable with `ArcFaceLoss.head.weight`.

### Verification

- `pytest -q` → 432 passed.
- End-to-end repros: `--metric-embed-dim 32` now saves a 32-dim head; `--checkpoint <arcface>` reproduces the trained head with a zeroed optimizer while `--checkpoint <SSL>` does not; a changed `--freeze-ratio` raises a named error; both ref-script checkpoints in `results_SSL_SFT`/`results_SSL` load through `_load_model`.

---

## Follow-Up: review round 3 (resize readability, resume continuity, read-only edge cases)

**Goal:** Close the gaps found in round 2's implementation.

### Task 9: Keep a resized checkpoint readable (P1)

`run_finetune` builds the encoder at `encoder_out_dim` (the pretrained projector width) and only then replaces the projector, so the SSL-only `center` buffer kept the pretrained width while `config.out_dim` recorded the resized metric dimension. The saved checkpoint therefore failed to reload in `extract`/`export`/`cam` with a `center` size mismatch.

- Writer: `run_finetune` resizes `center` to the fine-tune embedding dimension.
- Reader: `apply_checkpoint_weights()` drops buffers that do not fit the rebuilt model, which also rescues checkpoints already written before this fix.
- `finetune` never restores `center` (it is SSL-only training state), so it is dropped from the fine-tune load path.

### Task 10: Reject embedding-dimension changes on resume (P1)

`--resume` accepted a new `--metric-embed-dim` and rebuilt the projector, then restored the saved `loss_state_dict`/optimizer and died in `ArcFaceLoss.load_state_dict`. `_validate_finetune_embedding_dim()` now requires the resume dimension to match the checkpoint; resizing is only valid for a new `--checkpoint` run.

### Task 11: Read-only edge cases (P1)

- `--use-student` no longer raises on a ref-script SFT checkpoint: `prefer_student` is a true preference (`student`, then the normal priority order) and therefore still has no effect on checkpoints that store no student, as the CLI help promises.
- `export` and `cam` gained `--model-name`, and `apply_checkpoint_weights()` now fails with a message naming `--model-name` when the checkpoint's backbone tensors do not fit the resolved model (missing, unexpected, or shape-mismatched). Previously a wrong fallback produced an opaque size-mismatch traceback with no way for the user to correct it.
- Ref-script **SSL** checkpoints are self-describing (`args` carries `model_name`/`out_dim`); ref-script **SFT** checkpoints are not, so they require an explicit `--model-name`.

### Task 12: Correct the checkpoint contract documentation (P2)

Spec 4.8 claimed all four consumers share `resolve_checkpoint()` and that `finetune` supports `teacher`/`student` initialization. Both were wrong: `finetune` uses `_select_finetune_embedding_head()` (a different question — which head the new run should train), and it rejects every checkpoint without `model_state_dict`. The section and its table now distinguish the three read-only consumers from `finetune`.

### Verification

- `pytest -q` → 440 passed.
- End-to-end repros: a `--metric-embed-dim 32` run now saves `center` at `(1, 32)` and reloads through `_load_model`; a checkpoint already written with the stale `center` still loads; `--resume` with a changed dimension raises a named error; `--use-student` loads both ref-script layouts; a vit_small ref-script SFT checkpoint raises the `--model-name` error by default and exports correctly when the override is passed.

---

## Follow-Up: review round 4 (diagnostic accuracy)

Three P2 documentation/diagnostic items, no behaviour change:

- `extractor._load_model`'s docstring quoted a stale weight-key order. It now points at `resolve_checkpoint_state_dict()` as the single owner of that order, so the two cannot drift again.
- The backbone-mismatch error always told the user to pass `--model-name`, which cannot help when the checkpoint itself records a conflicting `model_name` (metadata outranks the CLI fallback). `CheckpointArchitecture` now carries `model_name_from_metadata`, and the error distinguishes "metadata and weights disagree" from "records no model name; pass the correct `--model-name`".
- `README.cn.md`'s export table was missing `--model-name` (the English table and the Chinese CAM table already had it).

`pytest -q` → 441 passed.
