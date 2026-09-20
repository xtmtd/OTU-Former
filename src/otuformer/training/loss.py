"""SSL and metric learning loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from otuformer.training.model import ArcFaceHead


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


LOSS_REGISTRY: dict[str, type] = {
    "arcface": ArcFaceLoss,
}
