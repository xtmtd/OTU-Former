# OTU-Former Plan 2: Training Module (pretrain + finetune)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `training/` subpackage and wire up the `pretrain` and `finetune` CLI commands, migrating logic directly from `ref/ibot20260115.py`.

**Architecture:** `training/` contains model, loss, dataset, scheduler, and trainer modules. The CLI layer (`cli/pretrain.py`, `cli/finetune.py`) is thin: parse args, call `run_pretrain()` / `run_finetune()` service functions in `training/trainer.py`. Logic is ported from `ref/ibot20260115.py` which uses `--mode pretrain|finetune|extract`. Extract mode is handled in Plan 3.

**Tech Stack:** PyTorch ≥ 2.1, timm ≥ 1.0, torchvision, tqdm

**Prerequisites:** Plan 1 complete (`otuformer` installable, CLI stubs in place).

**Reference source:** `ref/ibot20260115.py` — all training logic originates here.

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/otuformer/training/model.py` | `OTUFormerEncoder`: ViT backbone + 3-layer MLP projector; `ArcFaceHead` |
| `src/otuformer/training/loss.py` | SSL losses (global distillation, local-to-global, masked patch regression); `ArcFaceLoss`; `LOSS_REGISTRY` |
| `src/otuformer/training/dataset.py` | `MultiCropDataset` for pretrain (multi-crop augmentation); `MetricDataset` for finetune |
| `src/otuformer/training/scheduler.py` | Cosine LR schedule, EMA momentum schedule, teacher temperature warmup |
| `src/otuformer/training/trainer.py` | `run_pretrain()`, `run_finetune()` — main training loops; `EnhancedMetricsLogger`, `InstantMetricsLogger` |
| `src/otuformer/cli/pretrain.py` | Replace stub body: call `run_pretrain()` |
| `src/otuformer/cli/finetune.py` | Replace stub body: call `run_finetune()` |
| `tests/test_training_model.py` | Unit tests for model forward pass shapes |
| `tests/test_training_loss.py` | Unit tests for loss computations |
| `tests/test_training_dataset.py` | Unit tests for dataset item shapes |
| `tests/test_training_scheduler.py` | Unit tests for schedule values |

---

## Task 1: `training/model.py` — encoder + projector + ArcFace head

Port model architecture from `ref/ibot20260115.py`.

**Files:**
- Create: `src/otuformer/training/model.py`
- Create: `tests/test_training_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training_model.py
import torch
import pytest
from otuformer.training.model import OTUFormerEncoder, ArcFaceHead


def test_encoder_output_shape():
    """CLS token output shape matches out_dim."""
    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=128)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 128)


def test_encoder_output_is_l2_normalized():
    """Projector output should be L2 normalized."""
    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=128)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_arcface_head_output_shape():
    """ArcFace head output shape = (batch, num_classes)."""
    head = ArcFaceHead(embed_dim=128, num_classes=10)
    x = torch.randn(4, 128)
    out = head(x)
    assert out.shape == (4, 10)


def test_encoder_patch_tokens_available():
    """Model can return both CLS and patch tokens."""
    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=128, return_patch_tokens=True)
    x = torch.randn(2, 3, 224, 224)
    cls_out, patch_tokens = model(x)
    assert cls_out.shape == (2, 128)
    assert patch_tokens.ndim == 3  # (batch, num_patches, hidden_dim)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training_model.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `training/model.py`**

Port from `ref/ibot20260115.py`. Key classes:

```python
# src/otuformer/training/model.py
"""OTU-Former model: ViT encoder + MLP projector + ArcFace head."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


class ProjectionHead(nn.Module):
    """3-layer MLP projector with GELU, outputs L2-normalized embedding."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)


class OTUFormerEncoder(nn.Module):
    """ViT backbone + projection head.

    Args:
        model_name: timm backbone identifier.
        out_dim: Projector output dimension (L2-normalized).
        return_patch_tokens: If True, forward() returns (cls_emb, patch_tokens).
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        out_dim: int = 256,
        return_patch_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.return_patch_tokens = return_patch_tokens
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        hidden_dim = self.backbone.num_features
        self.projector = ProjectionHead(hidden_dim, hidden_dim, out_dim)

    def forward(self, x: torch.Tensor):
        features = self.backbone.forward_features(x)
        # CLS token is index 0 for ViT
        cls_token = features[:, 0]
        cls_emb = self.projector(cls_token)
        if self.return_patch_tokens:
            patch_tokens = features[:, 1:]
            return cls_emb, patch_tokens
        return cls_emb


class ArcFaceHead(nn.Module):
    """ArcFace classification head.

    Args:
        embed_dim: Input embedding dimension.
        num_classes: Number of classes.
        s: Scale factor [default: 64.0].
        m: Angular margin in radians [default: 0.5].
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.5,
    ) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if labels is None:
            return cosine * self.s
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_training_model.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/model.py tests/test_training_model.py
git commit -m "feat: add OTUFormerEncoder and ArcFaceHead"
```

---

## Task 2: `training/loss.py` — SSL losses + ArcFace loss registry

Port from `ref/ibot20260115.py`.

**Files:**
- Create: `src/otuformer/training/loss.py`
- Create: `tests/test_training_loss.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training_loss.py
import torch
import pytest
from otuformer.training.loss import (
    GlobalDistillationLoss,
    LocalToGlobalLoss,
    MaskedPatchRegressionLoss,
    ArcFaceLoss,
    LOSS_REGISTRY,
)


def test_global_distillation_loss_shape():
    student = torch.randn(4, 128)
    teacher = torch.randn(4, 128)
    center = torch.zeros(128)
    loss_fn = GlobalDistillationLoss(out_dim=128)
    loss = loss_fn(student, teacher, center, teacher_temp=0.07, student_temp=0.1)
    assert loss.ndim == 0  # scalar


def test_masked_patch_regression_loss():
    student_patches = torch.randn(4, 196, 384)
    teacher_patches = torch.randn(4, 196, 384)
    mask = torch.zeros(4, 196, dtype=torch.bool)
    mask[:, :50] = True
    loss_fn = MaskedPatchRegressionLoss()
    loss = loss_fn(student_patches, teacher_patches, mask)
    assert loss.ndim == 0


def test_arcface_loss_forward():
    loss_fn = ArcFaceLoss(embed_dim=64, num_classes=5)
    x = torch.randn(8, 64)
    labels = torch.randint(0, 5, (8,))
    loss = loss_fn(x, labels)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_loss_registry_contains_arcface():
    assert "arcface" in LOSS_REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training_loss.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `training/loss.py`**

Port from `ref/ibot20260115.py`. Key implementations:

```python
# src/otuformer/training/loss.py
"""SSL and metric learning loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from otuformer.training.model import ArcFaceHead


class GlobalDistillationLoss(nn.Module):
    """Cross-entropy between student and teacher CLS token distributions (DINO-style)."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        center: torch.Tensor,
        teacher_temp: float,
        student_temp: float,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)
        teacher_probs = F.softmax((teacher_out - center) / teacher_temp, dim=-1).detach()
        return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


class LocalToGlobalLoss(nn.Module):
    """Align local-view student embeddings to teacher global embeddings."""

    def forward(
        self,
        local_student_outs: list[torch.Tensor],
        teacher_out: torch.Tensor,
        center: torch.Tensor,
        teacher_temp: float,
        student_temp: float,
    ) -> torch.Tensor:
        teacher_probs = F.softmax((teacher_out - center) / teacher_temp, dim=-1).detach()
        total_loss = torch.tensor(0.0, device=teacher_out.device)
        for local_out in local_student_outs:
            student_log_probs = F.log_softmax(local_out / student_temp, dim=-1)
            total_loss += -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        return total_loss / max(len(local_student_outs), 1)


class MaskedPatchRegressionLoss(nn.Module):
    """MSE between student and teacher patch tokens at masked positions (iBOT-style)."""

    def forward(
        self,
        student_patches: torch.Tensor,
        teacher_patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=student_patches.device)
        s = student_patches[mask]
        t = teacher_patches[mask].detach()
        s = F.normalize(s, dim=-1)
        t = F.normalize(t, dim=-1)
        return F.mse_loss(s, t)


class ArcFaceLoss(nn.Module):
    """ArcFace metric learning loss."""

    def __init__(self, embed_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5) -> None:
        super().__init__()
        self.head = ArcFaceHead(embed_dim, num_classes, s=s, m=m)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.head(embeddings, labels)
        return self.ce(logits, labels)


LOSS_REGISTRY: dict[str, type] = {
    "arcface": ArcFaceLoss,
    # "proxy_anchor": ProxyAnchorLoss,  # planned
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_training_loss.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/loss.py tests/test_training_loss.py
git commit -m "feat: add SSL losses and ArcFace loss registry"
```

---

## Task 3: `training/scheduler.py` — LR, EMA, teacher temperature

**Files:**
- Create: `src/otuformer/training/scheduler.py`
- Create: `tests/test_training_scheduler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training_scheduler.py
from otuformer.training.scheduler import (
    cosine_lr_schedule,
    ema_momentum_schedule,
    teacher_temp_schedule,
)


def test_cosine_lr_at_boundaries():
    lr = cosine_lr_schedule(epoch=0, max_epochs=100, base_lr=1e-3, min_lr=1e-6, warmup_epochs=5)
    assert lr < 1e-3  # warmup not done yet
    lr_end = cosine_lr_schedule(epoch=99, max_epochs=100, base_lr=1e-3, min_lr=1e-6, warmup_epochs=5)
    assert abs(lr_end - 1e-6) < 1e-7


def test_ema_momentum_increases():
    m0 = ema_momentum_schedule(step=0, total_steps=1000, start=0.995, end=0.999)
    m1 = ema_momentum_schedule(step=999, total_steps=1000, start=0.995, end=0.999)
    assert m1 > m0
    assert m1 <= 0.999


def test_teacher_temp_warmup():
    t0 = teacher_temp_schedule(epoch=0, warmup_epochs=10, start=0.04, end=0.07)
    t5 = teacher_temp_schedule(epoch=5, warmup_epochs=10, start=0.04, end=0.07)
    t10 = teacher_temp_schedule(epoch=10, warmup_epochs=10, start=0.04, end=0.07)
    assert t0 < t5 < t10
    assert abs(t10 - 0.07) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training_scheduler.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `training/scheduler.py`**

```python
# src/otuformer/training/scheduler.py
"""Learning rate, EMA momentum, and teacher temperature schedules."""

from __future__ import annotations

import math


def cosine_lr_schedule(
    epoch: int,
    max_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
    warmup_epochs: int = 0,
) -> float:
    """Cosine LR with linear warmup."""
    if epoch < warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def ema_momentum_schedule(
    step: int,
    total_steps: int,
    start: float = 0.995,
    end: float = 0.999,
) -> float:
    """Cosine EMA momentum schedule from start to end."""
    progress = step / max(total_steps, 1)
    return end - (end - start) * 0.5 * (1 + math.cos(math.pi * progress))


def teacher_temp_schedule(
    epoch: int,
    warmup_epochs: int,
    start: float = 0.04,
    end: float = 0.07,
) -> float:
    """Linear warmup for teacher temperature."""
    if epoch >= warmup_epochs:
        return end
    return start + (end - start) * epoch / max(warmup_epochs, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_training_scheduler.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/scheduler.py tests/test_training_scheduler.py
git commit -m "feat: add LR/EMA/temperature schedules"
```

---

## Task 4: `training/dataset.py` — multi-crop dataset + metric dataset

**Files:**
- Create: `src/otuformer/training/dataset.py`
- Create: `tests/test_training_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training_dataset.py
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from otuformer.training.dataset import MultiCropDataset, MetricDataset


def make_dummy_images(tmp_path, n=4):
    records = []
    for i in range(n):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 50, 0, 0)).save(img_path)
        records.append({"image": f"img_{i}.jpg"})
    csv_path = tmp_path / "images.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


def make_dummy_labeled_images(tmp_path, n=4):
    records = []
    for i in range(n):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 50, 0, 0)).save(img_path)
        records.append({"image": f"img_{i}.jpg", "label": f"class_{i % 2}"})
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


def test_multicrop_dataset_returns_list(tmp_path):
    csv_path = make_dummy_images(tmp_path)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    views = ds[0]
    assert isinstance(views, list)
    assert len(views) == 4  # 2 global + 2 local


def test_multicrop_dataset_view_shapes(tmp_path):
    csv_path = make_dummy_images(tmp_path)
    ds = MultiCropDataset(
        csv_path=csv_path,
        images_dir=tmp_path,
        global_crop_size=32,
        local_crop_size=16,
        local_crops=2,
    )
    views = ds[0]
    for v in views[:2]:
        assert v.shape == (3, 32, 32)
    for v in views[2:]:
        assert v.shape == (3, 16, 16)


def test_metric_dataset_returns_image_label(tmp_path):
    csv_path = make_dummy_labeled_images(tmp_path)
    ds = MetricDataset(csv_path=csv_path, images_dir=tmp_path, image_size=32)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training_dataset.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `training/dataset.py`**

Port augmentation pipeline from `ref/ibot20260115.py` (geometric + color augmentations including large rotations for pretrain; standard transforms for finetune/extract):

```python
# src/otuformer/training/dataset.py
"""Datasets for SSL pretraining and metric learning fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _make_global_transform(crop_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _make_local_transform(crop_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.05, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _make_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class MultiCropDataset(Dataset):
    """SSL pretraining dataset with multi-crop augmentation.

    Returns list of views: [global_1, global_2, local_1, ..., local_n].
    CSV must have an 'image' column with filenames relative to images_dir.
    """

    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        local_crops: int = 6,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.image_paths = [Path(images_dir) / row for row in df["image"]]
        self.global_tf = _make_global_transform(global_crop_size)
        self.local_tf = _make_local_transform(local_crop_size)
        self.local_crops = local_crops

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        views = [self.global_tf(img), self.global_tf(img)]
        for _ in range(self.local_crops):
            views.append(self.local_tf(img))
        return views


class MetricDataset(Dataset):
    """Fine-tuning dataset for metric learning.

    CSV must have 'image' and 'label' columns.
    Labels are encoded as integers via a stable sorted mapping.
    """

    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        image_size: int = 224,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.image_paths = [Path(images_dir) / row for row in df["image"]]
        classes = sorted(df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.class_to_idx[l] for l in df["label"]]
        self.transform = _make_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_training_dataset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/dataset.py tests/test_training_dataset.py
git commit -m "feat: add MultiCropDataset and MetricDataset"
```

---

## Task 5: `training/trainer.py` — pretrain + finetune training loops

Port training loops, metrics loggers from `ref/ibot20260115.py`.

**Files:**
- Create: `src/otuformer/training/trainer.py`

Note: `trainer.py` is the largest file in this plan. It contains `run_pretrain()`, `run_finetune()`, `EnhancedMetricsLogger`, `InstantMetricsLogger` (ported from `ref/ibot20260115.py`), and EMA teacher update logic. No unit tests for the full training loop (too expensive); integration test via CLI smoke in Task 6.

- [ ] **Step 1: Implement `EnhancedMetricsLogger` and `InstantMetricsLogger`**

Port directly from `ref/ibot20260115.py` classes `EnhancedMetricsLogger` and `InstantMetricsLogger` (lines ~71–461). These handle CSV logging and training curve plotting.

- [ ] **Step 2: Implement EMA teacher update**

```python
@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """Update teacher parameters as EMA of student parameters."""
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
```

- [ ] **Step 3: Implement `run_pretrain()`**

Port `train_ssl()` from `ref/ibot20260115.py`. Function signature:

```python
def run_pretrain(args: argparse.Namespace) -> None:
    """Run SSL pretraining. All hyperparameters come from args namespace."""
```

Key steps inside (matching original):
1. Set seed, device, thread count
2. Build `MultiCropDataset`, DataLoader
3. Build student `OTUFormerEncoder`, copy to teacher, disable teacher grad
4. Set up `GlobalDistillationLoss`, `LocalToGlobalLoss`, `MaskedPatchRegressionLoss`
5. Set up AdamW optimizer on student
6. Training loop: forward both global views through student and teacher, compute 3 losses, backward, clip grad, EMA update teacher, center update (0.9 EMA), log metrics
7. Every `eval_every` epochs: compute comprehensive metrics (NMI, ARI, Recall@K, kNN), save checkpoint
8. Save `best.pt` (best NMI on eval set if available, else last), `last.pt`

- [ ] **Step 4: Implement `run_finetune()`**

Port `train_arcface()` from `ref/ibot20260115.py`. Function signature:

```python
def run_finetune(args: argparse.Namespace) -> None:
    """Run ArcFace fine-tuning. All hyperparameters come from args namespace."""
```

Key steps:
1. Load pretrained checkpoint, build `OTUFormerEncoder`
2. Freeze first `freeze_ratio` fraction of transformer blocks
3. Build `MetricDataset`, DataLoader
4. Instantiate loss from `LOSS_REGISTRY[args.loss]`
5. Training loop: forward, loss, backward, log
6. Evaluate every N epochs: kNN, NMI, ARI, Intra/Inter-class distances
7. Save `best.pt`, `last.pt`

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/training/trainer.py
git commit -m "feat: add pretrain and finetune training loops"
```

---

## Task 6: Wire CLI — `pretrain` and `finetune` commands

**Files:**
- Modify: `src/otuformer/cli/pretrain.py`
- Modify: `src/otuformer/cli/finetune.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Add integration smoke tests**

```python
# append to tests/test_cli_smoke.py
import pandas as pd
from PIL import Image


def _make_tiny_pretrain_data(tmp_path):
    """Create minimal dataset for pretrain smoke test."""
    for i in range(4):
        Image.new("RGB", (32, 32)).save(tmp_path / f"img_{i}.jpg")
    df = pd.DataFrame({"image": [f"img_{i}.jpg" for i in range(4)]})
    csv_path = tmp_path / "images.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_pretrain_runs_one_epoch(tmp_path):
    csv_path = _make_tiny_pretrain_data(tmp_path)
    result = runner.invoke(app, [
        "pretrain",
        "--train-data", str(csv_path),
        "--input-images-dir", str(tmp_path),
        "--out-dir", str(tmp_path / "out"),
        "--model-name", "vit_tiny_patch16_224",
        "--max-epochs", "1",
        "--batch-size", "2",
        "--device", "cpu",
    ])
    assert result.exit_code == 0
    assert (tmp_path / "out" / "last.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli_smoke.py::test_pretrain_runs_one_epoch -v
```

Expected: FAIL (stub raises `typer.Exit(1)`).

- [ ] **Step 3: Wire `cli/pretrain.py` to `run_pretrain()`**

Replace the stub body with:

```python
import argparse
from otuformer.training.trainer import run_pretrain
from otuformer.utils.logging import TeeLogger
import sys

out_dir.mkdir(parents=True, exist_ok=True)
log_file = out_dir / "logs" / "pretrain.log"
tee = TeeLogger(log_file)
sys.stdout = tee

# Convert Typer options to argparse.Namespace for trainer compatibility
ns = argparse.Namespace(
    train_data=str(train_data),
    input_images_dir=str(input_images_dir),
    out_dir=str(out_dir),
    model_name=model_name,
    out_dim=out_dim,
    max_epochs=max_epochs,
    lr=lr,
    weight_decay=weight_decay,
    warmup_epochs=warmup_epochs,
    global_crop_size=global_crop_size,
    local_crop_size=local_crop_size,
    local_crops=local_crops,
    mask_ratio=mask_ratio,
    lambda_local=lambda_local,
    lambda_mask=lambda_mask,
    teacher_momentum=teacher_momentum,
    teacher_momentum_end=teacher_momentum_end,
    student_temp=student_temp,
    teacher_temp_start=teacher_temp_start,
    teacher_temp_end=teacher_temp_end,
    disable_cross_view_loss=disable_cross_view_loss,
    resume=resume,
    log_every_n_steps=log_every_n_steps,
    save_every_epochs=save_every_epochs,
    keep_last_checkpoints=keep_last_checkpoints,
    batch_size=batch_size,
    num_workers=num_workers,
    cpus=cpus,
    device=device,
    seed=seed,
)
try:
    run_pretrain(ns)
finally:
    sys.stdout = tee.terminal
    tee.close()
```

- [ ] **Step 4: Wire `cli/finetune.py` similarly**

Same pattern: build `argparse.Namespace`, call `run_finetune(ns)`.

Also add a finetune smoke test to `tests/test_cli_smoke.py`:

```python
def _make_tiny_finetune_data(tmp_path):
    """Create minimal labeled dataset for finetune smoke test."""
    for i in range(4):
        Image.new("RGB", (32, 32)).save(tmp_path / f"img_{i}.jpg")
    df = pd.DataFrame({
        "image": [f"img_{i}.jpg" for i in range(4)],
        "label": ["classA", "classA", "classB", "classB"],
    })
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_finetune_runs_one_epoch(tmp_path):
    # First create a pretrain checkpoint
    csv_path = _make_tiny_pretrain_data(tmp_path)
    runner.invoke(app, [
        "pretrain",
        "--train-data", str(csv_path),
        "--input-images-dir", str(tmp_path),
        "--out-dir", str(tmp_path / "pretrain_out"),
        "--model-name", "vit_tiny_patch16_224",
        "--max-epochs", "1",
        "--batch-size", "2",
        "--device", "cpu",
    ])
    ckpt = tmp_path / "pretrain_out" / "last.pt"
    labels_csv = _make_tiny_finetune_data(tmp_path)
    result = runner.invoke(app, [
        "finetune",
        "--checkpoint", str(ckpt),
        "--train-data", str(labels_csv),
        "--input-images-dir", str(tmp_path),
        "--out-dir", str(tmp_path / "finetune_out"),
        "--model-name", "vit_tiny_patch16_224",
        "--finetune-epochs", "1",
        "--batch-size", "2",
        "--device", "cpu",
    ])
    assert result.exit_code == 0
    assert (tmp_path / "finetune_out" / "last.pt").exists()
```

- [ ] **Step 5: Run smoke tests**

```bash
pytest tests/test_cli_smoke.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/otuformer/cli/pretrain.py src/otuformer/cli/finetune.py tests/test_cli_smoke.py
git commit -m "feat: wire pretrain and finetune CLI commands to training loop"
```

---

## Task 7: Final integration check

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 2: Manual CLI check**

```bash
otuformer pretrain --help
otuformer finetune --help
```

Expected: full argument lists shown.

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: plan 2 complete — training module"
```
