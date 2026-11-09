"""Pretraining loss terms and their per-step diagnostics."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple
import torch
import torch.nn.functional as F
from otuformer.training.loss import (
    ArcFaceLoss,
    LOSS_REGISTRY,
    SubCenterArcFaceLoss,
    SupConLoss,
    dino_distillation_loss,
    ibot_patch_loss,
    koleo_loss,
    masked_patch_cosine_loss,
)
from .patch_objective import (
    PATCH_CENTER_MOMENTUM,
)

_RELATIONAL_PAIR_SETS: dict[str, tuple[tuple[int, int], ...]] = {
    "cross": ((0, 1), (1, 0)),
    "full": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "same": ((0, 0), (1, 1)),
}


def _compute_global_loss(
    student_global: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    temp_student: float,
    temp_teacher: float,
    *,
    relational_pairing: str,
) -> torch.Tensor:
    """Average the selected global student/teacher view pairs.

    ``cross`` averages A-to-B and B-to-A, ``full`` averages the whole Cartesian
    product, and the internal ``same`` pairing keeps the historical
    matching-index terms for legacy checkpoints.
    """
    pairs = _RELATIONAL_PAIR_SETS.get(relational_pairing)
    if pairs is None:
        raise ValueError(
            f"Unsupported relational_pairing {relational_pairing!r}; expected one "
            f"of: {', '.join(_RELATIONAL_PAIR_SETS)}."
        )
    if len(student_global) != 2 or len(teacher_global) != 2:
        raise ValueError(
            "global relational loss needs exactly two student and two teacher "
            f"features, got {len(student_global)} and {len(teacher_global)}."
        )
    loss = torch.tensor(0.0, device=student_global[0].device)
    for student_index, teacher_index in pairs:
        loss = loss + _ssl_loss(
            student_global[student_index],
            teacher_global[teacher_index],
            temp_student,
            temp_teacher,
        )
    return loss / len(pairs)


def _compute_local_loss(
    local_views_student: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    temp_student: float,
    temp_teacher: float,
) -> torch.Tensor:
    """Compute local-to-global distillation loss following ref/ibot20260115.py.

    Averages over all local_student x teacher_global pairs.
    """
    if not local_views_student:
        return torch.tensor(0.0, device=teacher_global[0].device)
    device = local_views_student[0].device
    loss = torch.tensor(0.0, device=device)
    for lv in local_views_student:
        for t in teacher_global:
            loss += _ssl_loss(lv, t, temp_student, temp_teacher)
    loss /= max(1, len(local_views_student) * len(teacher_global))
    return loss


def _compute_dino_global_loss(
    student_logits: list[torch.Tensor],
    teacher_targets: list[torch.Tensor],
    temp_student: float,
) -> torch.Tensor:
    """Average the two cross-view pairs: student A -> teacher B, B -> A."""
    losses = [
        dino_distillation_loss(student_logits[0], teacher_targets[1], temp_student),
        dino_distillation_loss(student_logits[1], teacher_targets[0], temp_student),
    ]
    return torch.stack(losses).mean()


def _compute_dino_local_loss(
    local_logits: list[torch.Tensor],
    teacher_targets: list[torch.Tensor],
    temp_student: float,
) -> torch.Tensor:
    """Average every student-local -> teacher-global pair; zero without locals."""
    if not local_logits:
        return torch.zeros((), device=teacher_targets[0].device)
    losses = [
        dino_distillation_loss(logits, target, temp_student)
        for logits in local_logits
        for target in teacher_targets
    ]
    return torch.stack(losses).mean()


def _compute_koleo_raw(
    student_tokens: list[torch.Tensor],
    koleo_config: dict[str, Any],
) -> tuple[torch.Tensor, list[int]]:
    """Sum the per-global-crop KoLeo losses over the unmasked backbone CLS.

    Uses the existing unmasked global forwards, never locals and never a pooled
    cross-crop neighbor set, so two augmentations of one image cannot repel each
    other. Crops with fewer than two samples are skipped and counted.
    """
    device = student_tokens[0].device
    total = torch.zeros((), device=device)
    sizes: list[int] = []
    for tokens in student_tokens:
        backbone_cls = tokens[:, 0]
        count = int(backbone_cls.shape[0])
        sizes.append(count)
        if count < 2:
            continue
        total = total + koleo_loss(
            backbone_cls,
            normalize_eps=float(koleo_config["koleo_normalize_eps"]),
            distance_eps=float(koleo_config["koleo_distance_eps"]),
            log_eps=float(koleo_config["koleo_log_eps"]),
        )
    return total, sizes


def _dino_cls_diagnostics(
    teacher_targets: list[torch.Tensor], prototypes: int
) -> dict[str, float]:
    """CLS-target diagnostics over the cached assignments of both globals.

    Entropy and perplexity use finite-safe arithmetic: zero probabilities
    contribute zero entropy instead of NaN. These describe the teacher targets
    only; they are not morphology or collapse detection.
    """
    probabilities = torch.cat([t.detach().float() for t in teacher_targets], dim=0)
    positive = probabilities > 0
    log_probabilities = torch.where(
        positive, probabilities.log(), torch.zeros_like(probabilities)
    )
    entropy = float(-(probabilities * log_probabilities).sum(dim=-1).mean())

    marginal = probabilities.mean(dim=0)
    total = marginal.sum().clamp_min(1e-12)
    marginal = marginal / total
    marginal_positive = marginal > 0
    marginal_log = torch.where(
        marginal_positive, marginal.log(), torch.zeros_like(marginal)
    )
    perplexity = float(torch.exp(-(marginal * marginal_log).sum()))

    counts = torch.bincount(
        probabilities.argmax(dim=-1), minlength=prototypes
    ).float()
    active_ratio = float((counts > 0).float().mean())
    occupancy = float(counts.max() / counts.sum().clamp_min(1.0))
    return {
        "cls_assignment_entropy": entropy,
        "cls_prototype_perplexity": perplexity,
        "cls_active_prototype_ratio": active_ratio,
        "cls_max_prototype_occupancy": occupancy,
    }


def _koleo_instant_metrics(
    ssl_config: dict[str, Any],
    enabled: bool,
    koleo_raw: torch.Tensor,
    sizes: list[int],
) -> dict[str, Any]:
    """Status and counts for the optional KoLeo term (blank when N/A)."""
    weight = float(ssl_config["koleo_weight"])
    if not enabled:
        return {
            "koleo_status": "disabled",
            "koleo_loss_raw": None,
            "koleo_loss_weighted": None,
        }
    eligible = sum(size for size in sizes if size >= 2)
    skipped = sum(size for size in sizes if size < 2)
    if eligible == 0:
        # A skipped batch contributes exactly zero, still numeric.
        return {
            "koleo_status": "skipped",
            "koleo_loss_raw": 0.0,
            "koleo_loss_weighted": 0.0,
            "koleo_eligible_samples": 0,
            "koleo_skipped_samples": skipped,
        }
    raw = float(koleo_raw.item())
    return {
        "koleo_status": "active",
        "koleo_loss_raw": raw,
        "koleo_loss_weighted": weight * raw,
        "koleo_eligible_samples": eligible,
        "koleo_skipped_samples": skipped,
    }


def _update_teacher_center(
    center: torch.Tensor,
    teacher_global: torch.Tensor,
) -> torch.Tensor:
    """Update teacher center buffer with EMA of current-iteration teacher global outputs.

    Uses fixed 0.9/0.1 EMA coefficients matching ref/ibot20260115.py.
    """
    batch_mean = teacher_global.mean(dim=0, keepdim=True)
    return center * 0.9 + batch_mean * 0.1


def _ssl_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    temp_student: float,
    temp_teacher: float,
) -> torch.Tensor:
    student_sim = student_out @ teacher_out.T / temp_student
    with torch.no_grad():
        teacher_sim = teacher_out @ teacher_out.T / temp_teacher
        teacher_sim = teacher_sim - teacher_sim.mean(dim=1, keepdim=True)
        teacher_probs = F.softmax(teacher_sim, dim=1)
    student_log_probs = F.log_softmax(student_sim, dim=1)
    return -(teacher_probs * student_log_probs).sum(dim=1).mean()


def _update_patch_center(
    patch_center: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """EMA of the iBOT patch center from current pre-EMA teacher logits."""
    return patch_center * PATCH_CENTER_MOMENTUM + teacher_logits.mean(
        dim=0, keepdim=True
    ) * (1.0 - PATCH_CENTER_MOMENTUM)
