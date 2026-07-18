"""OTU-Former model: ViT encoder + projector + ArcFace head."""

from __future__ import annotations

import math
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Three-layer MLP projector with normalized output."""

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
    """ViT backbone plus projection head."""

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        out_dim: int = 256,
        return_patch_tokens: bool = False,
        img_size: int = 224,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.return_patch_tokens = return_patch_tokens
        self.model_name = model_name
        try:
            self.backbone = timm.create_model(
                model_name,
                img_size=img_size,
                num_classes=0,
                global_pool="",
                pretrained=pretrained,
                dynamic_img_size=True,
            )
        except Exception:
            try:
                self.backbone = timm.create_model(
                    model_name,
                    img_size=img_size,
                    pretrained=pretrained,
                    dynamic_img_size=True,
                )
                if hasattr(self.backbone, "head"):
                    self.backbone.head = nn.Identity()
                if hasattr(self.backbone, "fc_norm"):
                    self.backbone.fc_norm = nn.Identity()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load pretrained backbone weights for '{model_name}'. "
                    "Repair or remove the corrupted timm/Hugging Face cache, then retry."
                ) from exc
        hidden_dim = self.backbone.num_features
        proj_hidden_dim = 2048
        self.projector = ProjectionHead(hidden_dim, proj_hidden_dim, out_dim)
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, x: torch.Tensor):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        cls_emb = self.projector(cls_token)
        if self.return_patch_tokens:
            patch_tokens = features[:, 1:]
            return cls_emb, patch_tokens
        return cls_emb


class ArcFaceHead(nn.Module):
    """ArcFace classification head."""

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

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if labels is None:
            return cosine * self.s
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s
