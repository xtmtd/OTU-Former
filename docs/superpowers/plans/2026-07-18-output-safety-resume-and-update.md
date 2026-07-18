# Output Safety, Resume, and Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not commit unless the user explicitly requests it.

**Goal:** Prevent accidental output replacement, provide safe continuous and extended training resume behavior, add tag-pinned self-update, and document continued training with new images.

**Architecture:** A single IO helper owns output-directory preparation and is called by every output-producing command before logging or writes. Pretrain checkpoints persist schedule provenance so same-plan resume reuses the original curve while extension starts at the saved LR and decays over the new data size; finetune remains fixed-LR and validates its saved class labels. The update command is a focused CLI module that reads GitHub tags and invokes pip only after explicit consent.

**Tech Stack:** Python 3.11, Typer/Click, PyTorch, pytest, standard-library `urllib.request` and `subprocess`.

## Global Constraints

- Do not add dependencies.
- Every existing non-empty output directory, including one containing only dotfiles, must fail unless `--overwrite` or allowed `--resume` is explicit.
- `--overwrite` recursively removes the target directory before any command output is written.
- `--resume` and `--overwrite` are mutually exclusive for pretrain and finetune.
- Same-plan pretrain resume requires the checkpoint's original DataLoader length; extension supports a new DataLoader length.
- Pretrain checkpoints are epoch-boundary only; do not claim mid-epoch recovery.
- Update only to the latest valid Git tag, never to main HEAD.

---

## File Structure

- Modify: `src/otuformer/utils/io.py` - shared output-directory safety helper.
- Modify: `src/otuformer/cli/{pretrain,finetune,extract,cluster,annotate,diversity,cam,export}.py` - expose `--overwrite` and invoke the shared helper.
- Modify: `src/otuformer/training/trainer.py` - checkpoint schedule metadata, resume schedule selection, fine-tune resume validation.
- Modify: `src/otuformer/training/model.py` - allow model construction without downloading pretrained weights.
- Modify: `src/otuformer/{vision/cam,vision/export,embedding/extractor}.py` - disable pretrained-weight loading when a local checkpoint is supplied.
- Create: `src/otuformer/cli/update.py` - tag-only update command.
- Modify: `src/otuformer/cli/main.py` - register `update`.
- Modify: `tests/test_utils.py` - helper unit tests.
- Modify: `tests/test_cli_smoke.py` - command safety, resume, and CLI registration tests.
- Create: `tests/test_update.py` - isolated update command behavior tests.
- Modify: `tests/training/test_pretrain_alignment.py` - schedule construction and resume metadata tests.
- Modify: `README.md` and `README.cn.md` - update command, overwrite behavior, and continued-training guidance.

### Task 1: Centralize output-directory protection

**Files:**
- Modify: `src/otuformer/utils/io.py`
- Modify: `tests/test_utils.py`

**Interfaces:**
- Produces: `prepare_output_dir(out_dir: Path, *, overwrite: bool = False, allow_existing: bool = False) -> Path`
- Raises: `FileExistsError` when `out_dir` exists and has any entry but neither exception flag is set.

- [ ] **Step 1: Write failing helper tests**

```python
from otuformer.utils.io import prepare_output_dir


def test_prepare_output_dir_rejects_existing_content(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / ".DS_Store").write_text("metadata", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_output_dir(out_dir)


def test_prepare_output_dir_overwrite_removes_existing_content(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")

    assert prepare_output_dir(out_dir, overwrite=True) == out_dir
    assert out_dir.exists()
    assert list(out_dir.iterdir()) == []


def test_prepare_output_dir_allows_resume_directory(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "SSL_latest.pth").write_text("checkpoint", encoding="utf-8")

    assert prepare_output_dir(out_dir, allow_existing=True) == out_dir
```

- [ ] **Step 2: Verify the tests fail because the helper is absent**

Run: `pytest tests/test_utils.py -q`

Expected: import failure for `prepare_output_dir`.

- [ ] **Step 3: Implement the minimal helper**

```python
import shutil


def prepare_output_dir(
    out_dir: Path, *, overwrite: bool = False, allow_existing: bool = False
) -> Path:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    elif out_dir.exists() and not allow_existing and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {out_dir}. "
            "Choose another --out-dir or pass --overwrite."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
```

- [ ] **Step 4: Verify helper behavior**

Run: `pytest tests/test_utils.py -q`

Expected: PASS.

### Task 2: Apply output protection to non-training commands

**Files:**
- Modify: `src/otuformer/cli/extract.py`
- Modify: `src/otuformer/cli/cluster.py`
- Modify: `src/otuformer/cli/annotate.py`
- Modify: `src/otuformer/cli/diversity.py`
- Modify: `src/otuformer/cli/cam.py`
- Modify: `src/otuformer/cli/export.py`
- Modify: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `prepare_output_dir(out_dir, overwrite=overwrite)` from Task 1.
- Produces: `--overwrite` on the six commands; an occupied output path is rejected before a log is opened.

- [ ] **Step 1: Write failing CLI tests for cluster's representative behavior**

```python
def test_cluster_rejects_existing_output_without_overwrite(tmp_path):
    emb_path, _ = _make_embeddings_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")

    result = runner.invoke(app, [
        "cluster", "--embeddings", str(emb_path), "--out-dir", str(out_dir),
        "--custom-cutoffs", "0.5",
    ])

    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert (out_dir / "stale.txt").exists()


def test_cluster_overwrite_removes_stale_output(tmp_path):
    emb_path, _ = _make_embeddings_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")

    result = runner.invoke(app, [
        "cluster", "--embeddings", str(emb_path), "--out-dir", str(out_dir),
        "--custom-cutoffs", "0.5", "--overwrite",
    ])

    assert result.exit_code == 0
    assert not (out_dir / "stale.txt").exists()
```

- [ ] **Step 2: Verify the rejection test fails under current cluster cleanup behavior**

Run: `pytest tests/test_cli_smoke.py::test_cluster_rejects_existing_output_without_overwrite -q`

Expected: FAIL because cluster silently removes `stale.txt`.

- [ ] **Step 3: Add `--overwrite` and use the helper in each command**

For each command signature, add:

```python
overwrite: bool = typer.Option(
    False, "--overwrite", help="Clear an existing non-empty output directory."
),
```

Include `"overwrite": overwrite` in its logged parameter dictionary. Replace direct directory preparation with:

```python
from otuformer.utils.io import prepare_output_dir

prepare_output_dir(out_dir, overwrite=overwrite)
```

For cluster, remove the existing `shutil.rmtree(out_dir)` branch and its now-unused `shutil` import only if no other cluster code uses it.

- [ ] **Step 4: Add parameterized smoke coverage for the remaining commands**

Add a test parametrized over lightweight command invocations that monkeypatch each command's expensive worker (`extract_embeddings`, `run_cam`, `export_to_onnx`, and domain entry points as needed). For each command, create a non-empty output directory and assert it exits before the worker is called; repeat with `--overwrite` and assert the worker receives the cleared directory. Keep real cluster coverage from Step 1.

- [ ] **Step 5: Verify command safety**

Run: `pytest tests/test_utils.py tests/test_cli_smoke.py -q`

Expected: PASS.

### Task 3: Add pretrain and finetune output/resume validation

**Files:**
- Modify: `src/otuformer/cli/pretrain.py`
- Modify: `src/otuformer/cli/finetune.py`
- Modify: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `prepare_output_dir(..., allow_existing=bool(resume))` from Task 1.
- Produces: mutual-exclusion validation and checkpoint existence validation before output setup.

- [ ] **Step 1: Write failing CLI tests**

```python
@pytest.mark.parametrize("command,checkpoint_flag", [
    ("pretrain", "--resume"),
    ("finetune", "--resume"),
])
def test_training_rejects_resume_with_overwrite(
    tmp_path, command, checkpoint_flag
):
    result = runner.invoke(app, [
        command, checkpoint_flag, str(tmp_path / "missing.pth"),
        "--overwrite",
        "--out-dir", str(tmp_path / "out"),
        "--input-images-dir", str(tmp_path),
        *(["--train-data", str(tmp_path / "data.csv")] if command == "pretrain" else [
            "--train-data", str(tmp_path / "labels.csv"),
        ]),
    ])
    assert result.exit_code != 0
    assert "cannot be used together" in result.output


def test_pretrain_resume_allows_existing_output_and_appends_log(tmp_path, monkeypatch):
    checkpoint = tmp_path / "resume.pth"
    torch.save({}, checkpoint)
    out_dir = tmp_path / "pretrain_out"
    out_dir.mkdir()
    log_path = out_dir / "logs" / "pretrain.log"
    log_path.parent.mkdir()
    log_path.write_text("old run\n", encoding="utf-8")
    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", lambda args: None)

    result = runner.invoke(app, [
        "pretrain", "--resume", str(checkpoint), "--train-data", str(tmp_path / "data.csv"),
        "--input-images-dir", str(tmp_path), "--out-dir", str(out_dir),
    ])

    assert result.exit_code == 0
    assert log_path.read_text(encoding="utf-8").startswith("old run\n")


def test_finetune_checkpoint_initialization_requires_empty_or_overwrite(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "pretrained.pth"
    torch.save({}, checkpoint)
    out_dir = tmp_path / "finetune_out"
    out_dir.mkdir()
    stale = out_dir / "stale.txt"
    stale.write_text("stale", encoding="utf-8")
    monkeypatch.setattr("otuformer.training.trainer.run_finetune", lambda args: None)

    base_args = [
        "finetune", "--checkpoint", str(checkpoint),
        "--train-data", str(tmp_path / "labels.csv"),
        "--input-images-dir", str(tmp_path), "--out-dir", str(out_dir),
    ]
    rejected = runner.invoke(app, base_args)
    assert rejected.exit_code != 0
    assert stale.exists()

    overwritten = runner.invoke(app, [*base_args, "--overwrite"])
    assert overwritten.exit_code == 0
    assert not stale.exists()
```

- [ ] **Step 2: Verify the mutual-exclusion test fails**

Run: `pytest tests/test_cli_smoke.py::test_training_rejects_resume_with_overwrite tests/test_cli_smoke.py::test_finetune_checkpoint_initialization_requires_empty_or_overwrite -q`

Expected: FAIL because `--overwrite` is not yet recognized and finetune accepts occupied output directories.

- [ ] **Step 3: Implement CLI validation and preparation order**

In both callbacks, add `overwrite: bool = typer.Option(False, "--overwrite", ...)`. Before `TeeLogger` construction:

```python
if resume and overwrite:
    raise typer.BadParameter("--resume and --overwrite cannot be used together.")
if resume and not Path(resume).is_file():
    raise typer.BadParameter(f"Resume checkpoint not found: {resume}")

prepare_output_dir(out_dir, overwrite=overwrite, allow_existing=bool(resume))
```

Pass `overwrite` into the namespace and logged parameter dictionary. Keep `append=bool(resume)` unchanged.

- [ ] **Step 4: Verify training CLI behavior**

Run: `pytest tests/test_cli_smoke.py -q`

Expected: PASS.

### Task 4: Persist and apply pretrain schedule state

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Modify: `tests/training/test_pretrain_alignment.py`

**Interfaces:**
- Produces: `_build_pretrain_schedules(args, schedule_state: dict[str, int | float] | None, *, steps_per_epoch: int, global_step: int, completed_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]`
- `schedule_state` stores `original_max_epochs`, `total_steps`, `steps_per_epoch`, `warmup_steps`, `last_lr`, and `final_lr`.
- Raises: `ValueError` for unsupported epoch requests, same-plan loader mismatch, and legacy-checkpoint extension.

- [ ] **Step 1: Write failing pure schedule tests**

```python
def test_pretrain_extension_starts_at_saved_lr_and_never_increases():
    args = argparse.Namespace(
        max_epochs=5, lr=0.001, warmup_epochs=1,
        teacher_momentum=0.995, teacher_momentum_end=0.999,
        student_temp=0.1, teacher_temp_start=0.04, teacher_temp_end=0.07,
    )
    saved = {
        "original_max_epochs": 2,
        "total_steps": 8,
        "steps_per_epoch": 4,
        "warmup_steps": 4,
        "last_lr": 0.0002,
        "final_lr": 0.00001,
    }

    lr, momentum, student_temp, teacher_temp, metadata = trainer._build_pretrain_schedules(
        args, saved, steps_per_epoch=6, global_step=8, completed_epochs=2
    )

    extension = lr[8:]
    assert extension[0] == pytest.approx(0.0002)
    assert all(left >= right for left, right in zip(extension, extension[1:]))
    assert momentum[8] == pytest.approx(args.teacher_momentum_end)
    assert teacher_temp[8] == pytest.approx(args.teacher_temp_end)
    assert metadata["steps_per_epoch"] == 6


def test_same_plan_pretrain_resume_rejects_changed_loader_length():
    args = argparse.Namespace(max_epochs=2, lr=0.001, warmup_epochs=1)
    saved = {"original_max_epochs": 2, "total_steps": 8, "steps_per_epoch": 4,
             "warmup_steps": 4, "last_lr": 0.0002, "final_lr": 0.00001}

    with pytest.raises(ValueError, match="same-plan resume"):
        trainer._build_pretrain_schedules(
            args, saved, steps_per_epoch=5, global_step=4, completed_epochs=1
        )


def test_legacy_pretrain_checkpoint_rejects_extension():
    args = argparse.Namespace(max_epochs=5, lr=0.001, warmup_epochs=1)

    with pytest.raises(ValueError, match="schedule metadata"):
        trainer._build_pretrain_schedules(args, None, steps_per_epoch=4, global_step=8,
                                          completed_epochs=2)
```

- [ ] **Step 2: Verify schedule tests fail because the builder is absent**

Run: `pytest tests/training/test_pretrain_alignment.py -q`

Expected: FAIL with missing `_build_pretrain_schedules`.

- [ ] **Step 3: Implement the schedule builder and checkpoint payload**

Implement `_build_pretrain_schedules` next to `_cosine_scheduler`, using these branches:

```python
if schedule_state is None and args.max_epochs > completed_epochs:
    raise ValueError("Cannot extend a legacy checkpoint without schedule metadata.")
if schedule_state is not None and args.max_epochs < schedule_state["original_max_epochs"]:
    raise ValueError("Cannot decrease the original pretrain epoch plan.")
if args.max_epochs <= completed_epochs:
    raise ValueError("--max-epochs must exceed the completed checkpoint epoch.")
```

The helper receives only the extracted checkpoint `schedule` dictionary (or `None` for a legacy checkpoint); `run_pretrain` remains responsible for extracting `checkpoint.get("schedule")`, `checkpoint["epoch"]`, and `checkpoint["iteration"]` before it calls the helper. For same-plan resume, reconstruct original LR, momentum, and temperature arrays with the saved original totals and require saved/current `steps_per_epoch` equality. For extension, build an array whose prefix ends at `global_step`, whose next LR is saved `last_lr`, and whose remaining values cosine-decay to saved `final_lr`; fill continuation momentum and teacher temperature with their final values. Before returning, reject `global_step >= len(lr_schedule)` with `ValueError("Checkpoint iteration exceeds the available pretrain schedule.")`; do not silently clamp the index. Save the returned metadata in every pretrain checkpoint and set `last_lr` to `optimizer.param_groups[0]["lr"]` after the final optimizer update in the saved epoch.

- [ ] **Step 4: Route `run_pretrain` through the builder**

Replace the current warmup adjustment and direct `_cosine_scheduler` calls at `trainer.py` around the existing `start_epoch`/`global_step` logic with the builder. Pass the extracted schedule dictionary, completed epoch count, current loader length, and saved global step. Use the produced LR schedule in the training loop without `min()` clamping; the builder has already rejected an invalid schedule index. Preserve the existing checkpoint keys for backward loading.

- [ ] **Step 5: Verify schedule behavior**

Run: `pytest tests/training/test_pretrain_alignment.py -q`

Expected: PASS.

### Task 5: Validate finetune extension and labels

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Modify: `tests/test_cli_smoke.py`

**Interfaces:**
- Produces: fine-tune checkpoint key `class_labels: list[str]`.
- Raises: `ValueError` before loss-state loading if requested epochs are not greater than completed epochs, or saved/current class labels differ.

- [ ] **Step 1: Write failing validation tests**

```python
def test_finetune_resume_rejects_completed_epoch_target(tmp_path):
    checkpoint = tmp_path / "resume.pth"
    torch.save({"epoch": 2, "model_state_dict": {}, "config": {}}, checkpoint)

    with pytest.raises(ValueError, match="finetune-epochs must exceed"):
        trainer._validate_finetune_resume(
            checkpoint={"epoch": 2, "class_labels": ["a", "b"]},
            current_class_labels=["a", "b"],
            finetune_epochs=3,
        )


def test_finetune_resume_rejects_different_class_mapping():
    with pytest.raises(ValueError, match="class labels differ"):
        trainer._validate_finetune_resume(
            checkpoint={"epoch": 0, "class_labels": ["a", "b"]},
            current_class_labels=["a", "c"],
            finetune_epochs=2,
        )
```

- [ ] **Step 2: Verify the tests fail because validation is absent**

Run: `pytest tests/test_cli_smoke.py -q`

Expected: FAIL with missing `_validate_finetune_resume`.

- [ ] **Step 3: Implement validation before state loading**

Add:

```python
def _validate_finetune_resume(
    checkpoint: dict[str, Any], current_class_labels: list[str], finetune_epochs: int
) -> None:
    completed_epochs = int(checkpoint.get("epoch", -1)) + 1
    if finetune_epochs <= completed_epochs:
        raise ValueError(
            "--finetune-epochs must exceed the completed checkpoint epoch."
        )
    saved_labels = checkpoint.get("class_labels")
    if saved_labels is None:
        print("[Warning] Resume checkpoint lacks class labels; validating class count only.")
        return
    if list(saved_labels) != current_class_labels:
        raise ValueError("Cannot resume: training class labels differ from checkpoint.")
```

Call it after `MetricDataset` creation and `loss_fn` construction, before either `optimizer.load_state_dict` or `loss_fn.load_state_dict`. For legacy checkpoints, compare the loss-head class dimension with `len(current_class_labels)` before loading either state. Save `sorted(str(label) for label in ds.class_to_idx)` as `class_labels` in new finetune checkpoints.

- [ ] **Step 4: Verify fine-tune guards**

Run: `pytest tests/test_cli_smoke.py -q`

Expected: PASS.

### Task 6: Add tag-pinned self-update

**Files:**
- Create: `src/otuformer/cli/update.py`
- Modify: `src/otuformer/cli/main.py`
- Create: `tests/test_update.py`

**Interfaces:**
- Produces: `fetch_remote_version(timeout: int = 10) -> str`, `_parse_version(ver: str) -> _SemVer`, and Typer callback `update(check: bool, yes: bool) -> None`.
- Uses: `https://api.github.com/repos/xtmtd/OTU-Former/tags` and installs `git+https://github.com/xtmtd/OTU-Former.git@v{remote_ver}`.

- [ ] **Step 1: Write failing unit tests**

```python
def test_fetch_remote_version_uses_highest_semver_tag(monkeypatch):
    response = _response([{"name": "v0.1.1"}, {"name": "v0.2.0"}, {"name": "nightly"}])
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)

    assert fetch_remote_version() == "0.2.0"


def test_update_check_never_runs_pip(monkeypatch):
    monkeypatch.setattr("otuformer.cli.update.fetch_remote_version", lambda: "9.9.9")
    run = MagicMock()
    monkeypatch.setattr("subprocess.run", run)

    result = runner.invoke(app, ["update", "--check"])

    assert result.exit_code == 0
    run.assert_not_called()


def test_update_yes_installs_exact_tag(monkeypatch):
    monkeypatch.setattr("otuformer.cli.update.fetch_remote_version", lambda: "9.9.9")
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr("subprocess.run", run)

    result = runner.invoke(app, ["update", "--yes"])

    assert result.exit_code == 0
    assert run.call_args.args[0][-1].endswith("OTU-Former.git@v9.9.9")
```

- [ ] **Step 2: Verify tests fail because the module and command are absent**

Run: `pytest tests/test_update.py -q`

Expected: import or unknown-command failure.

- [ ] **Step 3: Implement the focused update module**

Use EntomoKit's SemVer ordering pattern, but make one request only:

```python
_REPO = "xtmtd/OTU-Former"
_TAGS_API_URL = f"https://api.github.com/repos/{_REPO}/tags"
_INSTALL_URL = f"git+https://github.com/{_REPO}.git@v{{version}}"
```

`fetch_remote_version` requests `_TAGS_API_URL`, filters valid tag names with an optional leading `v`, and returns the highest parsed version without the prefix. The callback prints current/latest versions, exits on current versions, respects `--check`, prompts unless `--yes`, and invokes:

```python
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade", _INSTALL_URL.format(version=remote_ver)],
    check=False,
)
```

On network or pip failure, print an error to stderr and `raise typer.Exit(code=1)` or the pip return code. Register `update` in `main.py` without importing heavy ML modules.

- [ ] **Step 4: Add error-path tests and verify**

Add tests that make `urlopen` raise `OSError` and `subprocess.run` return nonzero. Assert the CLI has nonzero exit code and clear error output.

Run: `pytest tests/test_update.py tests/test_cli_startup.py -q`

Expected: PASS.

### Task 7: Load checkpoint inference models offline and group safety options

**Files:**
- Modify: `src/otuformer/training/model.py`
- Modify: `src/otuformer/vision/cam.py`
- Modify: `src/otuformer/embedding/extractor.py`
- Modify: `src/otuformer/vision/export.py`
- Modify: `src/otuformer/cli/{pretrain,finetune,extract,cluster,annotate,diversity,cam,export}.py`
- Modify: `tests/test_training_model.py`
- Modify: `tests/test_cam.py`
- Modify: `tests/test_extractor.py`
- Modify: `tests/test_export.py`
- Modify: `tests/test_cli_smoke.py`

**Interfaces:**
- Produces: `OTUFormerEncoder(..., pretrained: bool = True)`.
- Consumes: local checkpoint paths in CAM, extract, and export loaders.
- Guarantees: checkpoint inference passes `pretrained=False`; new training retains the default `pretrained=True`.

- [ ] **Step 1: Write failing model and checkpoint-loader tests**

```python
def test_encoder_forwards_pretrained_flag_to_timm(monkeypatch):
    calls = []

    def fake_create_model(*args, **kwargs):
        calls.append(kwargs)
        return _tiny_backbone()

    monkeypatch.setattr(model_module.timm, "create_model", fake_create_model)
    model_module.OTUFormerEncoder(pretrained=False)

    assert calls[0]["pretrained"] is False
```

Add one test each for CAM, extraction, and export that monkeypatches its
`OTUFormerEncoder` reference, invokes the local-checkpoint loader, and asserts
the constructor receives `pretrained=False`.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `pytest tests/test_training_model.py tests/test_cam.py tests/test_extractor.py tests/test_export.py -q`

Expected: FAIL because `OTUFormerEncoder` does not accept `pretrained` and
checkpoint loaders do not pass it.

- [ ] **Step 3: Implement checkpoint-only construction**

Add the optional constructor parameter and replace both hard-coded timm calls,
including the fallback branch:

```python
def __init__(..., pretrained: bool = True) -> None:
    # ...
    self.backbone = timm.create_model(..., pretrained=pretrained, ...)
```

Use `pretrained=False` in `load_model_from_checkpoint`, `_load_model`, and
`export_to_onnx`. Leave all training constructors unchanged so their default
remains `True`.

- [ ] **Step 4: Move safety controls to the end of callback signatures**

Move `--resume`, then `--overwrite`, to the final callback parameters of
`pretrain` and `finetune`. Move `--overwrite` to the final callback parameter
of `extract`, `cluster`, `annotate`, `diversity`, `cam`, and `export`. Do not
change option names, defaults, or namespace values.

- [ ] **Step 5: Add help-order tests**

```python
@pytest.mark.parametrize("command,options", [
    ("pretrain", ["--resume", "--overwrite"]),
    ("finetune", ["--resume", "--overwrite"]),
    ("extract", ["--overwrite"]),
    ("cluster", ["--overwrite"]),
    ("annotate", ["--overwrite"]),
    ("diversity", ["--overwrite"]),
    ("cam", ["--overwrite"]),
    ("export", ["--overwrite"]),
])
def test_help_lists_safety_options_last(command, options):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    positions = [result.output.rfind(option) for option in options]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    assert positions[-1] > result.output.rfind("--out-dir")
```

The final assertion guarantees that the safety controls are below the normal
output-directory option, rather than only ordered relative to one another.

- [ ] **Step 6: Verify offline loading and help order**

Run: `pytest tests/test_training_model.py tests/test_cam.py tests/test_extractor.py tests/test_export.py tests/test_cli_smoke.py -q`

Expected: PASS.

### Task 8: Document safe outputs, updates, continued training, and Hugging Face initialization

**Files:**
- Modify: `README.md`
- Modify: `README.cn.md`

**Interfaces:**
- Documents: `--overwrite`, `update`, pretrain extension, and fine-tune extension constraints implemented in Tasks 2-6.

- [ ] **Step 1: Write documentation assertions**

Add a small test in `tests/test_cli_smoke.py`:

```python
def test_readmes_document_update_and_continued_training():
    for path in [Path("README.md"), Path("README.cn.md")]:
        text = path.read_text(encoding="utf-8")
        assert "otuformer update" in text
        assert "--overwrite" in text
        assert "--resume" in text
```

- [ ] **Step 2: Verify the test fails before documentation is added**

Run: `pytest tests/test_cli_smoke.py::test_readmes_document_update_and_continued_training -q`

Expected: FAIL because the update command is not yet documented.

- [ ] **Step 3: Update both READMEs**

Add `update` to each command table and a compact command section with:

```bash
otuformer update
otuformer update --check
otuformer update --yes
```

Near pretrain and finetune usage, add continuation examples that keep the same `--out-dir`, pass the latest checkpoint to `--resume`, use the union CSV of old and new images, and increase the total epoch value. State that pretrain extensions may use a larger union data set; same-plan resume requires the original batch count. State that finetune resume requires exactly the same sorted class labels; new classes need fresh initialization from `--checkpoint`. Add a general output note: existing non-empty output directories require `--overwrite`, which clears the directory, except valid training `--resume` runs.

State that an unauthenticated Hugging Face warning during new pretrain or
finetune initialization is expected for public timm weights, that first use may
need network access when no local cache exists, and that `HF_TOKEN` is optional
for higher rate limits. State that CAM, extract, and export use only their local
checkpoint weights and do not require Hugging Face access.

- [ ] **Step 4: Verify README and focused tests**

Run: `pytest tests/test_cli_smoke.py::test_readmes_document_update_and_continued_training -q`

Expected: PASS.

- [ ] **Step 5: Run the full verification suite**

Run: `pytest -q`

Expected: PASS.

## Final Verification

- [ ] Run `pytest -q` and confirm all tests pass.
- [ ] Run `otuformer --help` and verify `update` is listed.
- [ ] Run `otuformer update --check` only if network access is available; it must not invoke pip.
- [ ] Run `git diff --check` and confirm no whitespace errors.
- [ ] Inspect `git status --short` before any final commit to avoid staging unrelated changes.
