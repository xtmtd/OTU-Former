"""Training-only DINO CLS objective and its weight-normalized prototype layer.

The prototype layer is kept out of the encoder: extraction still reads the
normalized D-dimensional bottleneck MLP, while only training sees K logits.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from otuformer.constants import DINO_CENTERING_MODES
from otuformer.training.loss import sinkhorn_knopp_targets

CENTER_MOMENTUM = 0.9

# Fixed architecture/stability settings. These are recorded in effective config
# but never inherited from a checkpoint; a conflicting recorded value is an
# error, not a silent migration.
DINO_FIXED_CONFIG = {
    "dino_hidden_dim": 2048,
    "dino_mlp_init_std": 0.02,
    "dino_mlp_init_bias": 0.0,
    "dino_weight_norm_api": "explicit",
    "dino_prototype_scale": 1.0,
    "dino_freeze_last_layer_epochs": 1,
    "dino_target_dtype": "float32",
    "koleo_normalize_eps": 1e-8,
    "koleo_distance_eps": 1e-8,
    "koleo_log_eps": 1e-8,
}

# Operator-configurable DINO settings: inherited on resume when omitted.
DINO_DEFAULTS = {
    "dino_prototypes": 64,
    "dino_centering": "ema",
    "koleo_weight": 0.0,
}

# Earlier releases recorded the deprecated torch.nn.utils.weight_norm helper.
# The parameterisation is identical (same state keys, same forward, same
# gradients), so a recipe recorded by those releases stays resumable.
_WEIGHT_NORM_API_EQUIVALENTS = {
    DINO_FIXED_CONFIG["dino_weight_norm_api"],
    "legacy",
}


def _validate_out_dim(out_dim: int) -> int:
    if isinstance(out_dim, bool) or not isinstance(out_dim, int) or out_dim <= 0:
        raise ValueError(f"out_dim must be a positive integer, got {out_dim!r}.")
    return out_dim


def _validate_prototypes(prototypes: int) -> int:
    if (
        isinstance(prototypes, bool)
        or not isinstance(prototypes, int)
        or prototypes < 2
    ):
        raise ValueError(
            f"dino_prototypes must be an integer >= 2, got {prototypes!r}."
        )
    return prototypes


class WeightNormLinear(nn.Module):
    """Bias-free linear layer with explicit ``w = g * v / ||v||`` scaling.

    Deliberately not ``torch.nn.utils.weight_norm`` (deprecated since PyTorch
    2.9): this keeps the canonical ``weight_g``/``weight_v`` parameter layout and
    the identical forward/gradient while depending only on ``F.linear``. The
    parametrization API is not a substitute -- it stores
    ``parametrizations.weight.original0/original1``, which would break every
    checkpoint written with the ``weight_g``/``weight_v`` keys.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_v = nn.Parameter(torch.empty(out_features, in_features))
        # ``g`` holds the row norms, so the derived weight equals ``v`` at init.
        self.weight_g = nn.Parameter(torch.zeros(out_features, 1))
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        with torch.no_grad():
            self.weight_g.copy_(
                self.weight_v.norm(dim=1, keepdim=True, p=2).clamp_min(1e-12)
            )

    @property
    def weight(self) -> torch.Tensor:
        """The effective weight, recomputed from ``g`` and ``v`` on every read."""
        norm = self.weight_v.norm(dim=1, keepdim=True, p=2).clamp_min(1e-12)
        return (self.weight_v / norm) * self.weight_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


def _prototype_layer(out_dim: int, prototypes: int) -> WeightNormLinear:
    """Prototype layer whose scale is frozen at one for the whole run."""
    layer = WeightNormLinear(out_dim, prototypes)
    with torch.no_grad():
        layer.weight_g.fill_(1.0)
    layer.weight_g.requires_grad_(False)
    return layer


class DINOObjective(nn.Module):
    """Student/teacher prototype layers plus the optional K-dimensional center.

    Both layers are built independently and the teacher is initialized by a
    strict state copy: the layer keeps no cached non-leaf ``weight``, so only the
    parameter tensors are ever stored.
    """

    def __init__(self, out_dim: int, prototypes: int, centering: str) -> None:
        super().__init__()
        if centering not in DINO_CENTERING_MODES:
            raise ValueError(
                f"Unsupported DINO centering {centering!r}; expected one of "
                f"{DINO_CENTERING_MODES}."
            )
        self.out_dim = _validate_out_dim(out_dim)
        self.prototypes = _validate_prototypes(prototypes)
        self.centering = centering
        self.student_prototypes = _prototype_layer(self.out_dim, self.prototypes)
        self.teacher_prototypes = _prototype_layer(self.out_dim, self.prototypes)
        self.teacher_prototypes.load_state_dict(
            self.student_prototypes.state_dict()
        )
        for parameter in self.teacher_prototypes.parameters():
            parameter.requires_grad_(False)
        # ``None`` registers an inactive center that is absent from state_dict.
        self.register_buffer(
            "cls_center",
            torch.zeros(1, self.prototypes) if centering == "ema" else None,
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        """The only trainable direction; scales stay fixed at one."""
        return [self.student_prototypes.weight_v]

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Return raw prototype logits, never normalized or centered."""
        return self.student_prototypes(bottleneck)

    def teacher_logits(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return self.teacher_prototypes(bottleneck)

    def _stacked(self, teacher_logits: list[torch.Tensor]) -> torch.Tensor:
        if len(teacher_logits) != 2:
            raise ValueError(
                "DINO target construction expects exactly two teacher global "
                f"crops, got {len(teacher_logits)}."
            )
        shapes = {tuple(t.shape) for t in teacher_logits}
        if len(shapes) != 1:
            raise ValueError(
                f"teacher global crops must share a shape, got {sorted(shapes)}."
            )
        for tensor in teacher_logits:
            if tensor.ndim != 2:
                raise ValueError(
                    "teacher logits must be B-by-K matrices, got shape "
                    f"{tuple(tensor.shape)}."
                )
        return torch.cat([t.detach().float() for t in teacher_logits], dim=0)

    def teacher_targets(
        self, teacher_logits: list[torch.Tensor], teacher_temp: float
    ) -> list[torch.Tensor]:
        """Build detached float32 teacher assignments once, split by view."""
        if not math.isfinite(teacher_temp) or teacher_temp <= 0.0:
            raise ValueError(
                f"teacher_temp must be a finite value > 0, got {teacher_temp}."
            )
        stacked = self._stacked(teacher_logits)
        if self.centering == "ema":
            probabilities = torch.softmax(
                (stacked - self.cls_center.detach().float()) / teacher_temp, dim=-1
            )
        else:
            probabilities = sinkhorn_knopp_targets(stacked, teacher_temp)
        per_view = teacher_logits[0].shape[0]
        return list(probabilities.split(per_view))

    def update_center(self, teacher_logits: list[torch.Tensor]) -> None:
        """EMA-update the K-dimensional center from raw cached teacher logits."""
        if self.centering != "ema" or self.cls_center is None:
            return
        stacked = self._stacked(teacher_logits)
        with torch.no_grad():
            self.cls_center.mul_(CENTER_MOMENTUM).add_(
                (1.0 - CENTER_MOMENTUM) * stacked.mean(dim=0, keepdim=True)
            )


def validate_dino_prototype_layer(layer: nn.Module) -> None:
    """Assert the canonical prototype contract on a constructed layer.

    Building the layer no longer depends on a deprecated PyTorch helper, so this
    checks the contract itself (keys, frozen unit scale, forward and backward
    connectivity) instead of probing an external API. The resolver probes this
    contract before output setup; training also checks its actual device layers.
    """
    keys = set(layer.state_dict())
    if keys != {"weight_g", "weight_v"}:
        raise ValueError(
            "DINO prototype layer must expose exactly the canonical weight_g and "
            f"weight_v state keys, got {sorted(keys)}."
        )
    if layer.weight_g.requires_grad is not False:
        raise ValueError(
            "DINO prototype weight_g must stay frozen (requires_grad=False)."
        )
    if not torch.equal(layer.weight_g.detach(), torch.ones_like(layer.weight_g)):
        raise ValueError(
            "DINO prototype weight_g must be exactly one; the fixed prototype "
            "scale is never inherited or migrated."
        )
    for tensor in layer.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("DINO prototype layer stores non-finite parameters.")
    # Probe on the layer's own device and dtype: the objective is built with
    # ``.to(device)``, so a CPU probe tensor would fail on every accelerator.
    reference = layer.weight_v
    output = layer(
        torch.zeros(
            2, layer.in_features, device=reference.device, dtype=reference.dtype
        )
    )
    if tuple(output.shape) != (2, layer.out_features):
        raise ValueError(
            "DINO prototype layer forward returned "
            f"{tuple(output.shape)} instead of (2, {layer.out_features})."
        )
    if layer.weight_v.requires_grad:
        gradient = torch.autograd.grad(
            output.sum(), layer.weight_v, allow_unused=True
        )[0]
        if gradient is None:
            raise ValueError(
                "DINO prototype layer backward produced no weight_v gradient."
            )


def preflight_dino_prototypes(out_dim: int, prototypes: int, centering: str) -> None:
    """Probe the actual prototype implementation without advancing caller RNG."""
    from otuformer.utils.checkpoint import preserved_rng

    with preserved_rng():
        objective = DINOObjective(out_dim, prototypes, centering)
        validate_dino_prototype_layer(objective.student_prototypes)
        validate_dino_prototype_layer(objective.teacher_prototypes)


def check_dino_weight_norm_api_record(recorded: Any) -> bool:
    """True when a recorded ``dino_weight_norm_api`` is this parameterisation.

    Earlier releases recorded ``legacy`` for the same ``weight_g``/``weight_v``
    layout, so those recipes remain resumable.
    """
    return recorded in _WEIGHT_NORM_API_EQUIVALENTS
