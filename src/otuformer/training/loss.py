"""SSL and metric learning loss functions."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from otuformer.training.model import ArcFaceHead, SubCenterArcFaceHead


class GlobalDistillationLoss(nn.Module):
    """Cross-entropy between student and teacher CLS distributions."""

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
        teacher_probs = F.softmax(
            (teacher_out - center) / teacher_temp, dim=-1
        ).detach()
        return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


class LocalToGlobalLoss(nn.Module):
    """Align local-view student embeddings to global teacher embeddings."""

    def forward(
        self,
        local_student_outs: list[torch.Tensor],
        teacher_out: torch.Tensor,
        center: torch.Tensor,
        teacher_temp: float,
        student_temp: float,
    ) -> torch.Tensor:
        teacher_probs = F.softmax(
            (teacher_out - center) / teacher_temp, dim=-1
        ).detach()
        total_loss = torch.tensor(0.0, device=teacher_out.device)
        for local_out in local_student_outs:
            student_log_probs = F.log_softmax(local_out / student_temp, dim=-1)
            total_loss += -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        return total_loss / max(len(local_student_outs), 1)


def _validate_patch_mask(mask: torch.Tensor, tokens: torch.Tensor) -> None:
    """Reject empty, wrongly shaped, or non-boolean patch masks."""
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be a boolean tensor, got {mask.dtype}.")
    if mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"mask must match the patch token grid {tuple(tokens.shape[:2])}, "
            f"got {tuple(mask.shape)}."
        )
    if int(mask.sum()) == 0:
        raise ValueError("mask selects no patch positions; refusing to return 0.")


def masked_patch_cosine_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Normalized cosine regression at selected patch positions.

    Used by both ``consistency`` (selected fully-visible positions) and
    ``masked-feature`` (truly masked student positions). The teacher side is
    always stop-gradient and the loss is normalized by the selected count.
    """
    if student.shape != teacher.shape:
        raise ValueError(
            f"student and teacher patch tokens must match, got "
            f"{tuple(student.shape)} and {tuple(teacher.shape)}."
        )
    _validate_patch_mask(mask, student)
    s = F.normalize(student[mask], dim=-1)
    t = F.normalize(teacher[mask].detach(), dim=-1)
    return (2.0 - 2.0 * (s * t).sum(dim=-1)).mean()


def ibot_patch_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    patch_center: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centered prototype cross-entropy at masked positions.

    Returns ``(loss, teacher_probs)`` where ``teacher_probs`` is detached and
    holds the pre-EMA teacher probabilities used by diagnostics and by the
    patch-center update. The cross-entropy is never divided by ``log(K)``.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student and teacher patch logits must match, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )
    _validate_patch_mask(mask, student_logits)
    teacher_probs = F.softmax(
        (teacher_logits[mask].detach() - patch_center) / teacher_temp, dim=-1
    )
    student_log_probs = F.log_softmax(student_logits[mask] / student_temp, dim=-1)
    loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
    return loss, teacher_probs


class ArcFaceLoss(nn.Module):
    """ArcFace metric learning loss wrapper."""

    def __init__(
        self, embed_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5
    ) -> None:
        super().__init__()
        self.head = ArcFaceHead(embed_dim, num_classes, s=s, m=m)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.head(embeddings, labels)
        return self.ce(logits, labels)


class SubCenterArcFaceLoss(nn.Module):
    """Sub-center ArcFace, plus an optional same-class center-distance hinge.

    ``compact_weight=0`` is plain Sub-center ArcFace and matches the CE of its
    head logits exactly. The compact penalty averages
    ``relu(cosine_distance(center_i, center_j) - cap)`` over every distinct
    same-class center pair, so a center distance below ``cap`` exerts no
    pressure to merge. It constrains classifier weights only, never images.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        k: int = 2,
        s: float = 64.0,
        m: float = 0.5,
        compact_weight: float = 0.0,
        cap: float = 0.5,
    ) -> None:
        super().__init__()
        if not math.isfinite(cap) or cap <= 0.0 or cap > 2.0:
            raise ValueError(f"cap must be in (0, 2], got {cap}.")
        if not math.isfinite(compact_weight) or compact_weight < 0.0:
            raise ValueError(
                f"compact_weight must be a finite value >= 0, got {compact_weight}."
            )
        self.head = SubCenterArcFaceHead(embed_dim, num_classes, k=k, s=s, m=m)
        self.compact_weight = float(compact_weight)
        self.cap = float(cap)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.ce(self.head(embeddings, labels), labels)
        if self.compact_weight == 0.0 or self.head.k < 2:
            return loss
        centers = F.normalize(self.head.weight, dim=-1)
        total = embeddings.new_zeros(())
        pairs = 0
        for i in range(self.head.k):
            for j in range(i + 1, self.head.k):
                distance = 1.0 - (centers[:, i] * centers[:, j]).sum(dim=-1)
                total = total + F.relu(distance - self.cap).sum()
                pairs += centers.shape[0]
        return loss + self.compact_weight * total / pairs


class SupConLoss(nn.Module):
    """Single-view supervised contrastive loss on normalized embeddings.

    For each anchor, same-species images other than itself are positives and
    different-species images are negatives; the temperature-controlled
    log-softmax denominator spans every non-self pair. Batches with no valid
    positive anchor, or with no different-species images, return ``None``
    instead of a differentiable zero so the trainer can skip them. Morphology
    metadata is never read.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"temperature must be a finite value > 0, got {temperature}."
            )
        self.temperature = float(temperature)
        self.last_valid_anchors = 0

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor | None:
        self.last_valid_anchors = 0
        n = embeddings.shape[0]
        if n < 2:
            return None
        features = F.normalize(embeddings, dim=-1)
        logits = features @ features.T / self.temperature
        self_mask = ~torch.eye(n, dtype=torch.bool, device=features.device)
        pos_mask = labels[:, None].eq(labels[None, :]) & self_mask
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if not bool(valid.any()) or not bool((pos_counts < n - 1).any()):
            return None
        self.last_valid_anchors = int(valid.sum())
        logsumexp = torch.logsumexp(
            logits.masked_fill(~self_mask, float("-inf")), dim=1
        )
        pos_mean = (logits * pos_mask).sum(dim=1) / pos_counts.clamp(min=1)
        return (logsumexp[valid] - pos_mean[valid]).mean()


LOSS_REGISTRY: dict[str, type] = {
    "arcface": ArcFaceLoss,
    "supcon": SupConLoss,
    "subcenter-arcface": SubCenterArcFaceLoss,
    "subcenter-arcface-compact": SubCenterArcFaceLoss,
}
