import inspect

import torch
import torch.nn.functional as F
import pytest

from otuformer.training.loss import (
    ArcFaceLoss,
    GlobalDistillationLoss,
    LOSS_REGISTRY,
    LocalToGlobalLoss,
    ibot_patch_loss,
    masked_patch_cosine_loss,
)


def test_global_distillation_loss_shape():
    student = torch.randn(4, 128)
    teacher = torch.randn(4, 128)
    center = torch.zeros(128)
    loss_fn = GlobalDistillationLoss(out_dim=128)
    loss = loss_fn(student, teacher, center, teacher_temp=0.07, student_temp=0.1)
    assert loss.ndim == 0


def test_local_to_global_loss_shape():
    local_student = [torch.randn(4, 128), torch.randn(4, 128)]
    teacher = torch.randn(4, 128)
    center = torch.zeros(128)
    loss_fn = LocalToGlobalLoss()
    loss = loss_fn(local_student, teacher, center, teacher_temp=0.07, student_temp=0.1)
    assert loss.ndim == 0


def _reference_cosine(student, teacher, mask):
    s = F.normalize(student[mask], dim=-1)
    t = F.normalize(teacher[mask].detach(), dim=-1)
    return (2 - 2 * (s * t).sum(-1)).mean()


def test_masked_patch_cosine_loss_matches_reference():
    torch.manual_seed(0)
    student = torch.randn(3, 7, 5)
    teacher = torch.randn(3, 7, 5)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[:, :3] = True

    loss = masked_patch_cosine_loss(student, teacher, mask)

    assert loss.ndim == 0
    assert torch.allclose(loss, _reference_cosine(student, teacher, mask))


def test_masked_patch_cosine_loss_ignores_unselected_positions():
    student = torch.randn(2, 6, 4)
    teacher = torch.randn(2, 6, 4)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 4] = True

    perturbed_student = student.clone()
    perturbed_teacher = teacher.clone()
    perturbed_student[~mask] = 100.0
    perturbed_teacher[~mask] = -100.0

    base = masked_patch_cosine_loss(student, teacher, mask)
    assert torch.allclose(base, _reference_cosine(student, teacher, mask))
    assert torch.allclose(
        base, masked_patch_cosine_loss(perturbed_student, perturbed_teacher, mask)
    )
    # The authoritative loss has no EVA-specific branch.
    assert "eva" not in inspect.getsource(masked_patch_cosine_loss).lower()


def test_masked_patch_cosine_loss_is_teacher_detached():
    student = torch.randn(2, 4, 3, requires_grad=True)
    teacher = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.ones(2, 4, dtype=torch.bool)

    masked_patch_cosine_loss(student, teacher, mask).backward()

    assert student.grad is not None
    assert teacher.grad is None


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(2, 4, dtype=torch.bool),
        torch.zeros(2, 5, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.float32),
    ],
)
def test_masked_patch_cosine_loss_rejects_invalid_masks(mask):
    with pytest.raises(ValueError, match="mask"):
        masked_patch_cosine_loss(torch.randn(2, 4, 3), torch.randn(2, 4, 3), mask)


def _reference_ibot(student_logits, teacher_logits, mask, center, s_temp, t_temp):
    teacher_probs = F.softmax(
        (teacher_logits[mask].detach() - center) / t_temp, dim=-1
    )
    student_log_probs = F.log_softmax(student_logits[mask] / s_temp, dim=-1)
    return -(teacher_probs * student_log_probs).sum(-1).mean(), teacher_probs


def test_ibot_patch_loss_matches_reference_without_log_k_scaling():
    torch.manual_seed(0)
    k = 5
    student_logits = torch.randn(2, 6, k)
    teacher_logits = torch.randn(2, 6, k)
    center = torch.randn(1, k)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 2:5] = True

    loss, probs = ibot_patch_loss(
        student_logits, teacher_logits, mask, center, 0.1, 0.04
    )
    expected, expected_probs = _reference_ibot(
        student_logits, teacher_logits, mask, center, 0.1, 0.04
    )

    assert loss.ndim == 0
    assert torch.allclose(loss, expected)
    assert torch.allclose(probs, expected_probs)
    assert not probs.requires_grad
    assert probs.shape == (int(mask.sum()), k)


def test_ibot_patch_loss_uses_only_masked_positions():
    torch.manual_seed(1)
    student_logits = torch.randn(2, 5, 4)
    teacher_logits = torch.randn(2, 5, 4)
    center = torch.zeros(1, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 3] = True

    perturbed = teacher_logits.clone()
    perturbed[~mask] = 50.0

    base, _ = ibot_patch_loss(student_logits, teacher_logits, mask, center, 0.1, 0.04)
    other, _ = ibot_patch_loss(student_logits, perturbed, mask, center, 0.1, 0.04)

    assert torch.allclose(base, other)


def test_ibot_patch_loss_rejects_invalid_masks():
    with pytest.raises(ValueError, match="mask"):
        ibot_patch_loss(
            torch.randn(2, 4, 3),
            torch.randn(2, 4, 3),
            torch.zeros(2, 4, dtype=torch.bool),
            torch.zeros(1, 3),
            0.1,
            0.04,
        )


def test_arcface_loss_forward():
    loss_fn = ArcFaceLoss(embed_dim=64, num_classes=5)
    x = torch.randn(8, 64)
    labels = torch.randint(0, 5, (8,))
    loss = loss_fn(x, labels)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_loss_registry_contains_arcface():
    assert "arcface" in LOSS_REGISTRY
