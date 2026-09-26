"""OTU-Former model: ViT encoder + projector + ArcFace head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_TARGET_LAYERS = 4
"""Number of trailing transformer blocks averaged into the A+ target (fixed)."""


@dataclass
class PretrainForward:
    """Outputs of the training-only single-pass ViT forward."""

    cls: torch.Tensor
    tokens: torch.Tensor
    grid_size: tuple[int, int]
    final_four: list[torch.Tensor] | None = None


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


class ArcFaceEmbeddingHead(nn.Module):
    """Compact, unnormalized embedding head used for ArcFace fine-tuning."""

    def __init__(self, embed_dim: int, metric_embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, metric_embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
            except TypeError as exc:
                if "unexpected keyword argument" in str(exc):
                    raise RuntimeError(
                        f"Backbone '{model_name}' is not supported: it rejects the "
                        "ViT-only options (img_size / dynamic_img_size), i.e. it is "
                        "a CNN-style model. The OTU-Former pipeline is built around "
                        "ViT patch tokens; CNN backbones are not supported."
                    ) from exc
                raise
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

    # --- Training-only masked pretraining path (v0.7.0) ---------------------

    def validate_pretrain_backbone(self, require_intermediates: bool = False) -> None:
        """Fail fast when the backbone cannot run the masked training path.

        Resolves and caches the single final-token normalization module used by
        the A+ target so callers never redo the attribute lookup. Only standard
        timm ViT backbones are supported by v0.7.0 pretraining.
        """
        missing: list[str] = []
        backbone = self.backbone
        for name in ("patch_embed", "_pos_embed", "norm", "patch_drop", "norm_pre"):
            value = getattr(backbone, name, None)
            if value is None:
                missing.append(name)
            elif not callable(value):
                missing.append(f"{name} (not callable)")
        blocks = getattr(backbone, "blocks", None)
        if blocks is None:
            missing.append("blocks")
        elif not hasattr(blocks, "__len__"):
            missing.append("blocks (not a sequence)")
        if not isinstance(getattr(backbone, "num_prefix_tokens", None), int):
            missing.append("num_prefix_tokens")
        if not callable(getattr(backbone, "forward_features", None)):
            missing.append("forward_features")
        patch_embed = getattr(backbone, "patch_embed", None)
        if patch_embed is not None and not hasattr(patch_embed, "patch_size"):
            missing.append("patch_embed.patch_size")
        if missing:
            raise ValueError(
                f"Backbone '{self.model_name}' is not supported for v0.7.0 masked "
                f"pretraining: missing capability {', '.join(missing)}. Standard "
                "timm ViT backbones are required."
            )
        if require_intermediates and len(blocks) < PATCH_TARGET_LAYERS:
            raise ValueError(
                f"Backbone '{self.model_name}' has {len(blocks)} transformer blocks; "
                f"masked-feature (A+) needs the final four block outputs."
            )
        self._pretrain_final_norm = backbone.norm

    @property
    def pretrain_final_norm(self) -> nn.Module:
        """Final token normalization resolved by ``validate_pretrain_backbone``."""
        norm = getattr(self, "_pretrain_final_norm", None)
        if norm is None:
            raise RuntimeError(
                "validate_pretrain_backbone() must run before pretrain_final_norm "
                "is read."
            )
        return norm

    def _patch_grid_size(self, x: torch.Tensor) -> tuple[int, int]:
        patch_size = self.backbone.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        return int(x.shape[-2]) // patch_h, int(x.shape[-1]) // patch_w

    def _replace_patches(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        mask_token: torch.Tensor,
        patch_count: int,
    ) -> torch.Tensor:
        """Replace selected patch embeddings with the learnable mask token.

        Operates before prefix/positional embedding, so only patch tokens can be
        masked and every masked position keeps its own positional encoding.
        """
        if mask_token is None:
            raise ValueError("mask_token is required when mask is provided.")
        if mask.shape != (tokens.shape[0], patch_count):
            raise ValueError(
                f"mask must have shape {(tokens.shape[0], patch_count)}, got "
                f"{tuple(mask.shape)}."
            )
        fill = mask_token.to(dtype=tokens.dtype, device=tokens.device)
        return torch.where(mask.unsqueeze(-1), fill.expand_as(tokens), tokens)

    def forward_pretrain(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> PretrainForward:
        """Single-pass training forward that can replace patch embeddings.

        Follows the timm ``VisionTransformer`` sequence exactly: patch embed,
        optional mask replacement, prefix + positional embedding, patch drop,
        pre-normalization, blocks, final norm. The mask token is injected here
        rather than on the general encoder so read-only consumers never need it.
        """
        backbone = self.backbone
        patches = backbone.patch_embed(x)
        if patches.dim() == 4:
            # timm's dynamic-size ViT keeps the NHWC grid; preserve it so
            # ``_pos_embed`` can still resample positional embeddings.
            batch, grid_h, grid_w, channels = patches.shape
            grid_size = (int(grid_h), int(grid_w))
            patch_count = int(grid_h) * int(grid_w)
            flat = patches.reshape(batch, patch_count, channels)
            if mask is not None:
                flat = self._replace_patches(flat, mask, mask_token, patch_count)
                patches = flat.reshape(batch, int(grid_h), int(grid_w), channels)
        else:
            grid_size = self._patch_grid_size(x)
            patch_count = grid_size[0] * grid_size[1]
            if patches.shape[1] != patch_count:
                raise ValueError(
                    "patch embedding produced "
                    f"{patches.shape[1]} tokens but the resolved grid "
                    f"{grid_size} expects {patch_count}."
                )
            if mask is not None:
                patches = self._replace_patches(
                    patches, mask, mask_token, patch_count
                )

        tokens = backbone._pos_embed(patches)
        tokens = backbone.patch_drop(tokens)
        tokens = backbone.norm_pre(tokens)

        final_four: list[torch.Tensor] | None = [] if return_intermediates else None
        capture_from = len(backbone.blocks) - PATCH_TARGET_LAYERS
        for index, block in enumerate(backbone.blocks):
            tokens = block(tokens)
            if final_four is not None and index >= capture_from:
                final_four.append(tokens)

        tokens = backbone.norm(tokens)
        cls_emb = self.projector(tokens[:, 0])
        return PretrainForward(
            cls=cls_emb,
            tokens=tokens,
            grid_size=grid_size,
            final_four=final_four,
        )


def _build_patch_head(hidden_dim: int, out_dim: int | None) -> nn.Sequential:
    """LayerNorm -> Linear -> GELU -> Linear patch head."""
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, int(out_dim)),
    )


class PatchObjective(nn.Module):
    """Training-only patch state for v0.7.0 masked pretraining.

    Owns the student-only mask token, the A+ predictor, and the iBOT heads and
    patch center. Read-only consumers (extract, fine-tune, CAM, export) never
    need this module, so its state lives outside the general encoder and is
    stored under ``patch_objective`` in checkpoints.
    """

    def __init__(
        self,
        mode: str,
        hidden_dim: int,
        ibot_prototypes: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("none", "consistency", "masked-feature", "ibot"):
            raise ValueError(f"Unsupported patch objective mode {mode!r}.")
        self.mode = mode
        self.hidden_dim = int(hidden_dim)

        if mode in ("masked-feature", "ibot"):
            mask_token = torch.empty(1, 1, self.hidden_dim)
            nn.init.trunc_normal_(mask_token, std=0.02)
            self.mask_token: nn.Parameter | None = nn.Parameter(mask_token)
        else:
            self.register_parameter("mask_token", None)

        if mode == "masked-feature":
            # LayerNorm -> Linear -> GELU -> Linear, with L2 normalization in
            # ``predict_masked_features`` so the predictor has no extra weight.
            self.predictor: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.predictor = None

        if mode == "ibot":
            if ibot_prototypes is None:
                raise ValueError("ibot_prototypes is required for ibot mode.")
            self.ibot_prototypes = int(ibot_prototypes)
            self.student_ibot_head = _build_patch_head(
                self.hidden_dim, self.ibot_prototypes
            )
            self.teacher_ibot_head = _build_patch_head(
                self.hidden_dim, self.ibot_prototypes
            )
            self.teacher_ibot_head.load_state_dict(
                self.student_ibot_head.state_dict()
            )
            for param in self.teacher_ibot_head.parameters():
                param.requires_grad = False
            self.register_buffer(
                "patch_center", torch.zeros(1, self.ibot_prototypes)
            )
        else:
            self.ibot_prototypes = None
            self.student_ibot_head = None
            self.teacher_ibot_head = None
            self.register_buffer("patch_center", None)

    def predict_masked_features(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.predictor is None:
            raise RuntimeError("The A+ predictor exists only in masked-feature mode.")
        return F.normalize(self.predictor(tokens), dim=-1)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Student-side parameters that join the existing optimizer."""
        params: list[nn.Parameter] = []
        if self.mask_token is not None:
            params.append(self.mask_token)
        if self.predictor is not None:
            params.extend(self.predictor.parameters())
        if self.student_ibot_head is not None:
            params.extend(self.student_ibot_head.parameters())
        return params


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


class SubCenterArcFaceHead(nn.Module):
    """Sub-center ArcFace head: ``k`` normalized centers per class.

    Image-to-class similarity is the maximum cosine similarity over that
    class's centers; the ArcFace angular margin is applied to the resulting
    target-class logit. ``k=1`` is numerically identical to ``ArcFaceHead``.
    Centers are training-only state and are never used by ``extract``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        k: int = 2,
        s: float = 64.0,
        m: float = 0.5,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        self.s = s
        self.m = m
        self.k = int(k)
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, self.k, embed_dim)
        )
        # Per-center initialization matching ArcFaceHead's (C, D) layout.
        nn.init.xavier_uniform_(self.weight.view(-1, embed_dim))
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        weight = F.normalize(self.weight, dim=-1)
        cosine = F.linear(F.normalize(x), weight.reshape(-1, weight.shape[-1]))
        cosine = cosine.reshape(
            x.shape[0], weight.shape[0], self.k
        ).max(dim=-1).values
        if labels is None:
            return cosine * self.s
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=weight.shape[0]).float()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s
