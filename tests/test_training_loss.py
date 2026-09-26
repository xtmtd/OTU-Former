import inspect
import math

import torch
import torch.nn.functional as F
import pytest

from otuformer.training.loss import (
    ArcFaceLoss,
    GlobalDistillationLoss,
    LOSS_REGISTRY,
    LocalToGlobalLoss,
    SupConLoss,
    SubCenterArcFaceLoss,
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


def test_loss_registry_has_exactly_the_four_v080_modes():
    assert set(LOSS_REGISTRY) == {
        "arcface",
        "supcon",
        "subcenter-arcface",
        "subcenter-arcface-compact",
    }


# --- Sub-center ArcFace -----------------------------------------------------


def test_subcenter_loss_k1_matches_arcface_loss_and_gradients():
    torch.manual_seed(0)
    labels = torch.tensor([0, 2, 1, 1])
    arc = ArcFaceLoss(embed_dim=8, num_classes=3, s=16.0, m=0.25)
    sub = SubCenterArcFaceLoss(embed_dim=8, num_classes=3, k=1, s=16.0, m=0.25)
    with torch.no_grad():
        sub.head.weight.copy_(arc.head.weight.unsqueeze(1))

    x_arc = torch.randn(4, 8, requires_grad=True)
    x_sub = x_arc.detach().clone().requires_grad_(True)
    arc_loss = arc(x_arc, labels)
    sub_loss = sub(x_sub, labels)

    assert torch.allclose(arc_loss, sub_loss, atol=1e-6)
    arc_loss.backward()
    sub_loss.backward()
    assert torch.allclose(x_arc.grad, x_sub.grad, atol=1e-6)
    assert torch.allclose(
        arc.head.weight.grad, sub.head.weight.grad.squeeze(1), atol=1e-6
    )


def _set_centers(loss_fn, centers):
    with torch.no_grad():
        loss_fn.head.weight.copy_(F.normalize(centers, dim=-1))


def _two_centers(second_cosine):
    """One class, two centers whose cosine similarity is ``second_cosine``."""
    centers = torch.zeros(1, 2, 4)
    centers[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    centers[0, 1] = torch.tensor(
        [second_cosine, math.sqrt(1.0 - second_cosine**2), 0.0, 0.0]
    )
    return centers


def test_compact_zero_weight_is_plain_subcenter_ce():
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    labels = torch.tensor([0, 0])
    loss_fn = SubCenterArcFaceLoss(
        embed_dim=4, num_classes=1, k=2, compact_weight=0.0, cap=0.5
    )
    _set_centers(loss_fn, _two_centers(0.7))

    expected = F.cross_entropy(loss_fn.head(x, labels), labels)
    assert torch.allclose(loss_fn(x, labels), expected, atol=1e-6)


def test_compact_hinge_is_inactive_below_cap():
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    labels = torch.tensor([0, 0])
    plain = SubCenterArcFaceLoss(
        embed_dim=4, num_classes=1, k=2, compact_weight=0.0, cap=0.5
    )
    _set_centers(plain, _two_centers(0.7))  # cosine distance 0.3 < cap
    penalized = SubCenterArcFaceLoss(
        embed_dim=4, num_classes=1, k=2, compact_weight=0.3, cap=0.5
    )
    _set_centers(penalized, _two_centers(0.7))

    assert torch.allclose(penalized(x, labels), plain(x, labels), atol=1e-6)


def test_compact_hinge_adds_weight_times_excess_distance():
    torch.manual_seed(0)
    x = torch.randn(2, 4, requires_grad=True)
    labels = torch.tensor([0, 0])
    plain = SubCenterArcFaceLoss(
        embed_dim=4, num_classes=1, k=2, compact_weight=0.0, cap=0.5
    )
    _set_centers(plain, _two_centers(0.2))  # cosine distance 0.8 > cap
    penalized = SubCenterArcFaceLoss(
        embed_dim=4, num_classes=1, k=2, compact_weight=0.3, cap=0.5
    )
    _set_centers(penalized, _two_centers(0.2))

    penalty = penalized(x, labels) - plain(x, labels)
    assert torch.allclose(penalty, torch.tensor(0.3 * (0.8 - 0.5)), atol=1e-5)

    penalized(x, labels).backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(penalized.head.weight.grad).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": -1},
        {"compact_weight": -0.1},
        {"compact_weight": float("nan")},
        {"compact_weight": float("inf")},
        {"cap": 0.0},
        {"cap": 2.5},
        {"cap": float("nan")},
    ],
)
def test_subcenter_loss_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SubCenterArcFaceLoss(embed_dim=4, num_classes=2, **kwargs)


def test_subcenter_loss_accepts_cap_two_as_noop_boundary():
    loss_fn = SubCenterArcFaceLoss(
        embed_dim=4, num_classes=2, k=2, compact_weight=0.5, cap=2.0
    )
    x = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    loss = loss_fn(x, labels)
    assert torch.isfinite(loss)


# --- Supervised contrastive -------------------------------------------------


def _reference_supcon(z, labels, temperature):
    zn = F.normalize(z, dim=-1)
    logits = zn @ zn.T / temperature
    n = z.shape[0]
    self_mask = ~torch.eye(n, dtype=torch.bool)
    pos_mask = labels[:, None].eq(labels[None, :]) & self_mask
    logsumexp = torch.logsumexp(
        logits.masked_fill(~self_mask, float("-inf")), dim=1
    )
    pos_counts = pos_mask.sum(dim=1)
    pos_mean = (logits * pos_mask).sum(dim=1) / pos_counts.clamp(min=1)
    valid = pos_counts > 0
    return (logsumexp[valid] - pos_mean[valid]).mean()


def test_supcon_loss_matches_reference_forward_and_backward():
    torch.manual_seed(0)
    labels = torch.tensor([0, 0, 1, 1])
    z_loss = torch.randn(4, 6, requires_grad=True)
    z_ref = z_loss.detach().clone().requires_grad_(True)
    loss_fn = SupConLoss(temperature=0.07)

    loss = loss_fn(z_loss, labels)
    expected = _reference_supcon(z_ref, labels, 0.07)

    assert loss is not None
    assert torch.allclose(loss, expected, atol=1e-6)
    assert loss_fn.last_valid_anchors == 4
    loss.backward()
    expected.backward()
    assert torch.allclose(z_loss.grad, z_ref.grad, atol=1e-6)


def test_supcon_skips_batches_without_positives_or_negatives():
    loss_fn = SupConLoss(temperature=0.1)

    assert loss_fn(torch.randn(3, 4), torch.tensor([0, 1, 2])) is None
    assert loss_fn.last_valid_anchors == 0

    assert loss_fn(torch.randn(3, 4), torch.tensor([0, 0, 0])) is None
    assert loss_fn.last_valid_anchors == 0

    loss = loss_fn(torch.randn(3, 4), torch.tensor([0, 0, 1]))
    assert loss is not None
    assert loss_fn.last_valid_anchors == 2


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan"), float("inf")])
def test_supcon_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        SupConLoss(temperature=temperature)
