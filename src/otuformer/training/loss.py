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


class MaskedPatchRegressionLoss(nn.Module):
    """MSE between normalized student and teacher patch tokens at masked positions."""

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
