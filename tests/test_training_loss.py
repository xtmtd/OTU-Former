import torch

from otuformer.training.loss import (
    ArcFaceLoss,
    GlobalDistillationLoss,
    LOSS_REGISTRY,
    LocalToGlobalLoss,
    MaskedPatchRegressionLoss,
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
