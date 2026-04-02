# OTU-Former Pretrain Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `otuformer pretrain` with the reference SSL workflow in `ref/ibot20260115.py`, while fixing the train-only UMAP labeling issue and preserving the current package structure.

**Architecture:** Keep the implementation centered in `src/otuformer/training/trainer.py` and `src/otuformer/embedding/evaluator.py`. Treat the trainer as the owner of SSL loop semantics and evaluation orchestration, and the evaluator as the owner of metric computation plus caller-driven UMAP rendering. Only touch `src/otuformer/training/model.py` if the encoder cannot expose a clean `(normalized_projector_output, patch_tokens)` contract for pretraining.

**Tech Stack:** Python 3.11+, PyTorch, timm, NumPy, pandas, matplotlib, seaborn, umap-learn, pytest

---

## File Map

- Modify: `src/otuformer/training/trainer.py`
  - Align teacher-temperature schedule, momentum schedule, center update timing, global/local loss aggregation, masked-token regression path, and periodic pretrain evaluation orchestration.
- Modify: `src/otuformer/embedding/evaluator.py`
  - Make UMAP title/label behavior caller-controlled and remove train-only split suffix noise.
- Modify if required: `src/otuformer/training/model.py`
  - Expose a stable pretrain encoder contract only if `trainer.py` cannot already obtain `(normalized_projector_output, patch_tokens)` cleanly.
- Create: `tests/training/test_pretrain_alignment.py`
  - Unit tests for schedule construction, center update behavior, global/local loss pairing, and representation selection.
- Create: `tests/embedding/test_umap_rendering.py`
  - Unit tests for train-only UMAP label/title semantics.

## Task 1: Confirm Config-Parity Prerequisites

**Files:**
- Inspect: `src/otuformer/cli/pretrain.py`
- Inspect baseline artifact: `runs/pretrain/logs/pretrain.log`

- [ ] **Step 1: Compare current CLI defaults to the spec parity values**

Check `src/otuformer/cli/pretrain.py` against the spec values for:

- `model_name=vit_tiny_patch16_224`
- `out_dim=256`
- `max_epochs=50`
- `global_crop_size=224`
- `local_crop_size=96`
- `local_crops=6`
- `student_temp=0.1`
- `teacher_temp_start=0.04`
- `teacher_temp_end=0.07`
- `teacher_momentum=0.995`
- `teacher_momentum_end=0.999`
- `lambda_local=1.5`
- `lambda_mask=1.0`
- `mask_ratio=0.5`
- `batch_size=32`
- `save_every_epochs=10`

- [ ] **Step 2: Compare those values with the existing baseline run log**

Inspect the current baseline artifact at `runs/pretrain/logs/pretrain.log` and confirm it records the same effective values for the parity-sensitive fields above.

- [ ] **Step 3: Confirm no CLI default change is required for this plan**

Decision rule:

- if current defaults already match the parity values, do not edit `cli/pretrain.py`
- if a default differs, do not change it in this plan unless the current CLI cannot express the reference-comparable run via explicit command-line arguments

- [ ] **Step 4: Record the parity conclusion in implementation notes**

Expected note: `cli/pretrain.py` remains unchanged unless a missing option blocks the reference-comparable command.

## Task 2: Align Teacher Temperature Schedule

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Test: `tests/training/test_pretrain_alignment.py`

- [ ] **Step 1: Write a failing schedule test with explicit assertions**

```python
def test_teacher_temp_schedule_uses_70_percent_warmup():
    schedule = _build_teacher_temp_schedule(
        total_iters=10,
        teacher_temp_start=0.04,
        teacher_temp_end=0.07,
    )
    assert len(schedule) == 10
    assert schedule[0] == pytest.approx(0.04)
    assert schedule[6] == pytest.approx(0.07)
    assert schedule[7] == pytest.approx(0.07)
    assert schedule[9] == pytest.approx(0.07)
```

- [ ] **Step 2: Run the targeted test to confirm failure**

Run: `pytest tests/training/test_pretrain_alignment.py -k "teacher_temp_schedule" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the minimal schedule change in `trainer.py`**

Implementation note:

- add a private helper `_build_teacher_temp_schedule(...)` in `trainer.py` to make the schedule logic testable

- [ ] **Step 4: Run the targeted test again**

Run: `pytest tests/training/test_pretrain_alignment.py -k "teacher_temp_schedule" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/trainer.py tests/training/test_pretrain_alignment.py
git commit -m "fix: align teacher temperature schedule"
```

## Task 3: Align Momentum Update And Center Semantics

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Test: `tests/training/test_pretrain_alignment.py`

- [ ] **Step 1: Write failing tests for momentum ordering and center update**

```python
def test_teacher_momentum_updates_after_optimizer_step():
    call_order = []
    ...
    assert call_order == ["optimizer_step", "ema_update"]

def test_teacher_center_update_uses_current_iteration_global_outputs():
    teacher_center = torch.zeros(1, 2)
    teacher_global = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    updated = _update_teacher_center(teacher_center, teacher_global)
    assert updated.shape == (1, 2)
    assert updated[0, 0] == pytest.approx(0.05)
    assert updated[0, 1] == pytest.approx(0.05)
```

- [ ] **Step 2: Run the targeted tests to confirm failure**

Run: `pytest tests/training/test_pretrain_alignment.py -k "momentum or center" -v`
Expected: at least one failing assertion or missing test target.

- [ ] **Step 3: Align momentum ordering and center update in `trainer.py`**

Implement exactly these rules from `ref/ibot20260115.py`:

- momentum uses cosine schedule over total iterations
- center subtraction happens before teacher normalization
- center update uses concatenated teacher global outputs from the current iteration
- momentum update happens after `optimizer.step()`

Implementation note:

- add a private helper `_update_teacher_center(center, teacher_global)` in `trainer.py` to make the center update testable

- [ ] **Step 4: Run the targeted tests again**

Run: `pytest tests/training/test_pretrain_alignment.py -k "momentum or center" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/trainer.py tests/training/test_pretrain_alignment.py
git commit -m "fix: align pretrain momentum and center updates"
```

## Task 4: Align Global Loss Pairing

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Test: `tests/training/test_pretrain_alignment.py`

- [ ] **Step 1: Write a failing test for global loss aggregation**

```python
def test_global_loss_uses_cartesian_pairing_when_cross_view_enabled(monkeypatch):
    student = [torch.tensor([[1.0]]), torch.tensor([[2.0]])]
    teacher = [torch.tensor([[3.0]]), torch.tensor([[4.0]])]
    calls = []

    def fake_ssl_loss(s, t, *_):
        calls.append((float(s.item()), float(t.item())))
        return torch.tensor(0.0)

    monkeypatch.setattr(trainer, "_ssl_loss", fake_ssl_loss)
    _ = _compute_global_loss(student, teacher, 0.1, 0.04, disable_cross_view_loss=False)
    assert calls == [(1.0, 3.0), (1.0, 4.0), (2.0, 3.0), (2.0, 4.0)]
```

- [ ] **Step 2: Run the targeted tests to confirm failure**

Run: `pytest tests/training/test_pretrain_alignment.py -k "global_loss" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the minimal trainer change for global loss pairing**

Requirements:

- when `disable_cross_view_loss` is false, global loss averages over the full Cartesian product of student-global and teacher-global outputs
- when it is true, global loss averages only matching view pairs

Implementation note:

- add a private helper `_compute_global_loss(...)` in `trainer.py` to make pairing behavior testable

- [ ] **Step 4: Run the targeted tests again**

Run: `pytest tests/training/test_pretrain_alignment.py -k "global_loss" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/trainer.py tests/training/test_pretrain_alignment.py
git commit -m "fix: align pretrain global loss pairing"
```

## Task 5: Align Local Loss Pairing

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Test: `tests/training/test_pretrain_alignment.py`

- [ ] **Step 1: Write a failing test for local loss aggregation**

```python
def test_local_loss_averages_all_local_to_teacher_global_pairs(monkeypatch):
    local_student = [torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])]
    teacher = [torch.tensor([[4.0]]), torch.tensor([[5.0]])]
    calls = []

    def fake_ssl_loss(s, t, *_):
        calls.append((float(s.item()), float(t.item())))
        return torch.tensor(0.0)

    monkeypatch.setattr(trainer, "_ssl_loss", fake_ssl_loss)
    _ = _compute_local_loss(local_student, teacher, 0.1, 0.04)
    assert len(calls) == 6
```

- [ ] **Step 2: Run the targeted test to confirm failure**

Run: `pytest tests/training/test_pretrain_alignment.py -k "local_loss" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the minimal trainer change for local loss pairing**

Requirement:

- local loss averages over all `local_student x teacher_global` pairs

Implementation note:

- add a private helper `_compute_local_loss(...)` in `trainer.py` to make pairing behavior testable

- [ ] **Step 4: Run the targeted test again**

Run: `pytest tests/training/test_pretrain_alignment.py -k "local_loss" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/trainer.py tests/training/test_pretrain_alignment.py
git commit -m "fix: align pretrain local loss pairing"
```

## Task 6: Align Masked-Token Regression

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Test: `tests/training/test_pretrain_alignment.py`

- [ ] **Step 1: Write a failing test for masked-token behavior**

```python
def test_masked_token_loss_matches_reference_branching(monkeypatch):
    torch.manual_seed(0)
    student_tokens = torch.randn(2, 8, 4)
    teacher_tokens = torch.randn(2, 8, 4)
    loss = _masked_token_loss(student_tokens, teacher_tokens, 0.5, "vit_tiny_patch16_224")
    assert loss >= 0
```

- [ ] **Step 2: Run the targeted tests to confirm failure**

Run: `pytest tests/training/test_pretrain_alignment.py -k "masked_token_loss_matches_reference_branching" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the minimal masked-token change**

Requirements:

- masked-token loss uses per-sample random mask index selection from the reference implementation
- EVA and non-EVA loss branches match the reference behavior

Implementation note:

- add a private helper `_sample_mask_indices(...)` in `trainer.py` if needed to make mask selection testable

- [ ] **Step 4: Run the targeted tests again**

Run: `pytest tests/training/test_pretrain_alignment.py -k "masked_token_loss_matches_reference_branching" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/trainer.py tests/training/test_pretrain_alignment.py
git commit -m "fix: align masked-token pretrain path"
```

## Task 7: Tighten The Pretrain Encoder Contract If Needed

**Files:**
- Modify if required: `src/otuformer/training/model.py`
- Test: `tests/training/test_pretrain_alignment.py`

- [ ] **Step 1: Write a failing contract test only if the current encoder interface is ambiguous**

```python
def test_pretrain_encoder_exposes_projector_output_and_patch_tokens():
    model = OTUFormerEncoder(return_patch_tokens=True)
    proj, patch_tokens = model(torch.randn(2, 3, 224, 224))
    assert proj.ndim == 2
    assert patch_tokens.ndim == 3
```

- [ ] **Step 2: Run the targeted test if a contract change is needed**

Run: `pytest tests/training/test_pretrain_alignment.py -k "encoder_exposes_projector_output_and_patch_tokens" -v`
Expected: FAIL only if the current interface is not explicit enough for pretraining.

- [ ] **Step 3: Tighten `model.py` only if the trainer cannot already consume the current interface cleanly**

- [ ] **Step 4: Run the targeted test again if `model.py` changed**

Run: `pytest tests/training/test_pretrain_alignment.py -k "encoder_exposes_projector_output_and_patch_tokens" -v`
Expected: PASS.

- [ ] **Step 5: Commit only if `model.py` changed**

```bash
git add src/otuformer/training/model.py tests/training/test_pretrain_alignment.py
git commit -m "refactor: clarify pretrain encoder contract"
```

## Task 8: Make Periodic Pretrain Evaluation Use The Reference Representation

**Files:**
- Modify: `src/otuformer/training/trainer.py`
- Test: `tests/training/test_pretrain_alignment.py`

- [ ] **Step 1: Write a failing test for evaluation CLS-token selection**

```python
def test_pretrain_periodic_evaluation_uses_cls_token_features(monkeypatch):
    captured = {}
    def fake_extract(*args, **kwargs):
        captured["source"] = kwargs.get("source")
        return np.zeros((2, 4))
    monkeypatch.setattr(trainer, "_extract_eval_embeddings", fake_extract)
    _ = _compute_and_log_all_metrics(...)
    assert captured["source"] == "cls_token"
```

- [ ] **Step 2: Run the targeted test to confirm failure**

Run: `pytest tests/training/test_pretrain_alignment.py -k "evaluation_uses_cls_token_features" -v`
Expected: FAIL.

- [ ] **Step 3: Update `trainer.py` to keep CLS-token periodic SSL evaluation**

Implementation notes:

- keep evaluator metric formulas stable unless a reference mismatch forces a change
- if evaluator-side formulas do change, note that explicitly in code comments or verification notes so the metric movement is not mistaken for training-only improvement

- [ ] **Step 4: Run the targeted test again**

Run: `pytest tests/training/test_pretrain_alignment.py -k "evaluation_uses_cls_token_features" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/trainer.py tests/training/test_pretrain_alignment.py
git commit -m "test: lock pretrain periodic evaluation to cls-token features"
```

## Task 9: Verify Evaluator Metric Mapping Against The Reference Semantics

**Files:**
- Modify if required: `src/otuformer/embedding/evaluator.py`
- Test: `tests/training/test_pretrain_alignment.py`

- [ ] **Step 1: Write a failing test for evaluator metric naming/value mapping**

```python
def test_pretrain_metric_names_and_values_match_expected_mapping():
    ...
```

- [ ] **Step 2: Run the targeted test to confirm failure**

Run: `pytest tests/training/test_pretrain_alignment.py -k "metric_names_and_values_match_expected_mapping" -v`
Expected: FAIL if naming or value mapping still drifts from the reference assumptions.

- [ ] **Step 3: Compare `src/otuformer/embedding/evaluator.py` with reference log semantics and fix only required drift**

Requirements:

- inspect metric keys written to `metrics.pretrain.csv`
- confirm the trainer-to-evaluator mapping matches the reference log semantics for `NMI`, `ARI`, `Recall@K`, `kNN_Acc_k*`, `Linear_Probing_Acc`, `mAP`, `Silhouette_Score`, `Purity`
- do not broaden this into adding or renaming unrelated metrics

- [ ] **Step 4: Run the targeted test again**

Run: `pytest tests/training/test_pretrain_alignment.py -k "metric_names_and_values_match_expected_mapping" -v`
Expected: PASS.

- [ ] **Step 5: Commit only if evaluator mapping changed**

```bash
git add src/otuformer/embedding/evaluator.py tests/training/test_pretrain_alignment.py
git commit -m "fix: align pretrain evaluator metric mapping"
```

## Task 10: Fix Train-Only UMAP Labeling Semantics

**Files:**
- Modify: `src/otuformer/embedding/evaluator.py`
- Modify: `src/otuformer/training/trainer.py`
- Test: `tests/embedding/test_umap_rendering.py`

- [ ] **Step 1: Write failing tests for train-only UMAP rendering**

```python
def test_train_only_umap_does_not_append_split_suffix_to_labels():
    labels = ["A", "B"]
    rendered_labels = _format_umap_labels(labels, split=None)
    assert rendered_labels == ["A", "B"]

def test_umap_title_is_caller_controlled():
    title = _resolve_umap_title("UMAP Train - Epoch 10")
    assert title == "UMAP Train - Epoch 10"
```

- [ ] **Step 2: Run the targeted tests to confirm failure**

Run: `pytest tests/embedding/test_umap_rendering.py -v`
Expected: FAIL.

- [ ] **Step 3: Make UMAP rendering caller-driven**

Requirements:

- `run_umap()` accepts explicit title / split-display context
- train-only plots use raw class labels without `(train)`
- `trainer.py` passes the correct train-only metadata for pretrain checkpoints
- do not broaden this into a larger plotting redesign

- [ ] **Step 4: Run the targeted tests again**

Run: `pytest tests/embedding/test_umap_rendering.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/embedding/evaluator.py src/otuformer/training/trainer.py tests/embedding/test_umap_rendering.py
git commit -m "fix: clean up train-only umap labels"
```

## Task 11: End-To-End Verification Against The Reference Run

**Files:**
- Modify as needed from prior tasks only
- Reference: `ref/ibot20260115.py`
- Reference logs:
  - `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/log.txt`
  - `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/metrics.pretrain.csv`
  - `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/umap.train.epoch_0050.pdf`
  - `/Users/zf/data/DL_morphology/Lucaninae/experiments/SSL_genus/Epidorcus/SSL/training/logs/training_curves_pretrain.pdf`

- [ ] **Step 1: Run the targeted test suite**

Run: `pytest tests/training/test_pretrain_alignment.py tests/embedding/test_umap_rendering.py -v`
Expected: PASS.

- [ ] **Step 2: Preserve the pre-change baseline artifacts before rerunning pretrain**

Copy these existing files to a temporary baseline location such as `runs/pretrain/logs_baseline_2026-03-30/`:

- `runs/pretrain/logs/pretrain.log`
- `runs/pretrain/logs/metrics.pretrain.csv`
- `runs/pretrain/logs/umap.train.epoch_0050.pdf`
- `runs/pretrain/logs/training_curves_pretrain.pdf`

- [ ] **Step 3: Run the example pretrain workflow**

Run: `otuformer pretrain --train-data examples/Epidorcus/figs.csv --input-images-dir examples/Epidorcus/images --max-epochs 50 --log-every-n-steps 50 --save-every-epochs 10`
Expected: completes successfully and rewrites these exact artifacts:

- `runs/pretrain/logs/pretrain.log`
- `runs/pretrain/logs/metrics.pretrain.csv`
- `runs/pretrain/logs/umap.train.epoch_0050.pdf`
- `runs/pretrain/logs/training_curves_pretrain.pdf`

- [ ] **Step 4: Compare generated outputs to the preserved baseline and reference artifacts**

Check:

- epoch-50 values for `Recall@1`, `kNN_Acc_k1`, `kNN_Acc_k5`, `kNN_Acc_k20`, `Linear_Probing_Acc`, `mAP`, `Silhouette_Score`
- trend direction across epochs 10, 20, 30, 40, 50
- UMAP labels no longer include unnecessary `(train)` suffixes
- `runs/pretrain/logs/pretrain.log` confirms the expected command/default configuration is still in effect
- compare current outputs against `runs/pretrain/logs_baseline_2026-03-30/metrics.pretrain.csv` before comparing to the external reference
- compare current `runs/pretrain/logs/metrics.pretrain.csv` to the external reference metrics file only after confirming the baseline comparison above
- code/log inspection confirms the schedule, center, loss, and representation rules from the spec
- inspect `runs/pretrain/logs/training_curves_pretrain.pdf` and confirm the `Structure Quality` lines are clearly distinguishable; if the file is unchanged, record that it still satisfies this check in the final summary

- [ ] **Step 5: Record verification outcome in the final summary**

Include:

- which metrics moved closer to the reference
- whether any evaluator formula changed
- whether any tracked metric regressed
- whether at least 4 of the 7 tracked epoch-50 metrics moved closer to the reference
- whether more than 1 of the 7 tracked epoch-50 metrics regressed relative to the pre-change baseline
- whether UMAP labeling is fixed
- whether `training_curves_pretrain.pdf` color handling was intentionally left unchanged unless touched incidentally

- [ ] **Step 6: Commit**

```bash
git add src/otuformer/training/trainer.py src/otuformer/embedding/evaluator.py src/otuformer/training/model.py tests/training/test_pretrain_alignment.py tests/embedding/test_umap_rendering.py
git commit -m "fix: align pretrain behavior with reference workflow"
```
