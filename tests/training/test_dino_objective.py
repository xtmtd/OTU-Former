"""Training-only DINO objective: canonical state, freeze, contract and KoLeo scope."""

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from otuformer.training.dino import (
    DINOObjective,
    WeightNormLinear,
    check_dino_weight_norm_api_record,
    validate_dino_prototype_layer,
)
from otuformer.training.loss import koleo_loss

CANONICAL_KEYS = {
    "student_prototypes.weight_g",
    "student_prototypes.weight_v",
    "teacher_prototypes.weight_g",
    "teacher_prototypes.weight_v",
}


def test_dino_objective_canonical_roundtrip():
    objective = DINOObjective(out_dim=17, prototypes=11, centering="ema")

    assert objective.student_prototypes.weight_g.shape == (11, 1)
    assert objective.student_prototypes.weight_v.shape == (11, 17)
    state = objective.state_dict()
    assert set(state) == CANONICAL_KEYS | {"cls_center"}
    assert state["cls_center"].shape == (1, 11)
    # Prototype layers wrap a bias-free Linear, so no bias keys exist.
    assert not any("bias" in key for key in state)

    z = torch.nn.functional.normalize(torch.randn(4, 17), dim=-1)
    with torch.no_grad():
        assert torch.allclose(
            objective.student_prototypes(z), objective.teacher_prototypes(z)
        )

    restored = DINOObjective(out_dim=17, prototypes=11, centering="ema")
    restored.load_state_dict(state)
    with torch.no_grad():
        torch.testing.assert_close(
            restored.student_prototypes(z), objective.student_prototypes(z)
        )
    # Independent modules, never a deepcopy of a weight-normalized layer.
    assert (
        restored.student_prototypes.weight_v is not restored.teacher_prototypes.weight_v
    )

    sk = DINOObjective(out_dim=17, prototypes=11, centering="sinkhorn-knopp")
    assert objective.cls_center is not None
    assert sk.cls_center is None
    assert ("cls_center" in objective.state_dict()) is True
    assert ("cls_center" in sk.state_dict()) is False


def test_dino_fixed_scale_ema():
    objective = DINOObjective(out_dim=17, prototypes=11, centering="ema")

    assert torch.equal(
        objective.teacher_prototypes.weight_g, torch.ones(11, 1)
    )
    assert torch.equal(
        objective.student_prototypes.weight_g, torch.ones(11, 1)
    )
    trainable = objective.trainable_parameters()
    assert len(trainable) == 1
    assert trainable[0] is objective.student_prototypes.weight_v
    for layer in (objective.student_prototypes, objective.teacher_prototypes):
        assert layer.weight_g.requires_grad is False
        assert layer.weight_v.requires_grad is (
            layer is objective.student_prototypes
        )

    logits = [torch.randn(3, 11), torch.randn(3, 11)]
    objective.update_center(logits)
    torch.testing.assert_close(
        objective.cls_center,
        torch.cat(logits).mean(dim=0, keepdim=True) * 0.1,
        rtol=1e-5,
        atol=1e-6,
    )
    objective.load_state_dict(objective.state_dict())
    for layer in (objective.student_prototypes, objective.teacher_prototypes):
        assert torch.equal(layer.weight_g, torch.ones(11, 1))

    # SK registers no active CLS center and its update is a no-op.
    sk = DINOObjective(out_dim=17, prototypes=11, centering="sinkhorn-knopp")
    sk.update_center(logits)
    assert sk.cls_center is None
    assert "cls_center" not in sk.state_dict()


def test_dino_targets_are_detached_float32_and_follow_the_mode():
    objective = DINOObjective(out_dim=8, prototypes=5, centering="ema")
    teacher = [
        torch.randn(2, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 5, dtype=torch.float64, requires_grad=True),
    ]
    center = objective.cls_center.detach().clone()
    targets = objective.teacher_targets(teacher, teacher_temp=0.07)

    assert len(targets) == 2
    for view, target in zip(teacher, targets):
        assert target.shape == (2, 5)
        assert target.dtype == torch.float32
        assert target.requires_grad is False
        torch.testing.assert_close(
            target,
            torch.softmax((view.detach().float() - center) / 0.07, dim=-1),
            rtol=1e-5,
            atol=1e-6,
        )
    # No state update in the target builder.
    assert torch.equal(objective.cls_center, center)

    with pytest.raises(ValueError):
        objective.teacher_targets(teacher[:1], teacher_temp=0.07)


def test_prototype_contract_validation_preserves_rng():
    """Building and checking the layer must not consume randomness."""
    objective = DINOObjective(out_dim=8, prototypes=5, centering="ema")
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    before = (
        random.getstate(),
        np.random.get_state()[1].copy(),
        torch.get_rng_state().clone(),
    )

    validate_dino_prototype_layer(objective.student_prototypes)
    validate_dino_prototype_layer(objective.teacher_prototypes)

    assert random.getstate() == before[0]
    assert (np.random.get_state()[1] == before[1]).all()
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.parametrize("centering", ["ema", "sinkhorn-knopp"])
def test_dino_preflight_probe_preserves_rng(centering):
    from otuformer.training.dino import preflight_dino_prototypes

    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    preflight_dino_prototypes(8, 5, centering)
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0]
    np.testing.assert_array_equal(after[1], before[1][1])
    assert after[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


def test_prototype_contract_rejects_noncanonical_state():
    objective = DINOObjective(out_dim=8, prototypes=5, centering="ema")
    validate_dino_prototype_layer(objective.student_prototypes)

    # A layer whose keys differ from the canonical pair must be rejected loudly.
    plain = nn.Linear(8, 5, bias=False)
    with pytest.raises(ValueError, match="weight_g|weight_v"):
        validate_dino_prototype_layer(plain)


def test_prototype_contract_validation_follows_the_layer_device():
    """The contract check must run on the layer's own device, not on the CPU.

    DINO builds the objective with ``.to(device)``, so a CPU probe tensor fails on
    every accelerator with "input is on cpu but expected on mps/cuda".
    """
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    for device in devices:
        objective = DINOObjective(out_dim=8, prototypes=5, centering="ema").to(device)
        validate_dino_prototype_layer(objective.student_prototypes)
        assert objective.student_prototypes.weight_v.device.type == device.type


def test_prototype_contract_rejects_a_trainable_scale():
    objective = DINOObjective(out_dim=8, prototypes=5, centering="ema")
    objective.student_prototypes.weight_g.requires_grad_(True)
    with pytest.raises(ValueError, match="weight_g must stay frozen"):
        validate_dino_prototype_layer(objective.student_prototypes)


def test_prototype_layer_matches_the_legacy_parameterisation(monkeypatch):
    """The explicit layer is interchangeable with the deprecated API.

    Same state keys, same forward, same gradients: that equivalence is what keeps
    checkpoints written by earlier releases resumable, so it is asserted rather
    than assumed.
    """
    import warnings

    import torch.nn.utils as torch_utils

    legacy = getattr(torch_utils, "weight_norm", None)
    if legacy is None:  # pragma: no cover - only on a PyTorch that removed it
        pytest.skip("this PyTorch no longer ships the deprecated helper")
    torch.manual_seed(0)
    # The deprecated helper still emits its own warning; this test is the only
    # place in the suite that calls it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        reference = legacy(nn.Linear(8, 5, bias=False))

    mine = WeightNormLinear(8, 5)
    assert sorted(reference.state_dict()) == sorted(mine.state_dict())
    mine.load_state_dict(reference.state_dict(), strict=True)

    x = torch.randn(4, 8)
    upstream = torch.randn(4, 5)
    ref_out = reference(x)
    my_out = mine(x)
    torch.testing.assert_close(ref_out, my_out, rtol=1e-6, atol=1e-6)

    ref_out.backward(upstream)
    my_out.backward(upstream)
    for name in ("weight_g", "weight_v"):
        torch.testing.assert_close(
            getattr(reference, name).grad,
            getattr(mine, name).grad,
            rtol=1e-6,
            atol=1e-6,
        )


def test_no_deprecated_weight_norm_is_used_at_runtime():
    """Training must not call the deprecated helper, so no warning is emitted."""
    import warnings

    from otuformer.training.dino import DINOObjective

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DINOObjective(out_dim=8, prototypes=5, centering="ema")

    assert [w for w in caught if "weight_norm" in str(w.message)] == []


def test_recorded_api_accepts_both_equivalent_names():
    assert check_dino_weight_norm_api_record("explicit")
    # Releases before the explicit layer recorded this name for the same layout.
    assert check_dino_weight_norm_api_record("legacy")
    assert not check_dino_weight_norm_api_record("parametrizations")
    assert not check_dino_weight_norm_api_record(None)


def test_koleo_only_backward_excludes_teacher_and_prototypes():
    torch.manual_seed(0)
    objective = DINOObjective(out_dim=8, prototypes=7, centering="ema")
    student_backbone = nn.Linear(8, 8)
    teacher_backbone = nn.Linear(8, 8)

    for parameter in list(objective.parameters()) + list(
        student_backbone.parameters()
    ) + list(teacher_backbone.parameters()):
        parameter.grad = None

    backbone_cls = student_backbone(torch.randn(5, 8))
    loss = koleo_loss(backbone_cls)
    loss.backward()

    assert student_backbone.weight.grad is not None
    assert torch.isfinite(student_backbone.weight.grad).all()
    assert student_backbone.weight.grad.abs().sum() > 0
    for name, parameter in objective.named_parameters():
        assert parameter.grad is None, name
    for name, parameter in teacher_backbone.named_parameters():
        assert parameter.grad is None, name
