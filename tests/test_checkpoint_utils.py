"""Unit tests for the shared checkpoint architecture resolver."""

import pytest
import torch

from otuformer.training.model import ArcFaceEmbeddingHead
from otuformer.utils.checkpoint import (
    ARCFACE_EMBEDDING_HEAD,
    PROJECTION_EMBEDDING_HEAD,
    load_checkpoint_encoder,
    normalize_legacy_projector_keys,
    reject_noncanonical_register_keys,
    resolve_checkpoint,
    resolve_checkpoint_embedding_head,
    resolve_checkpoint_state_dict,
    resolve_projector_out_dim,
    resolve_register_layout,
)


def _arcface_weights(embed_dim: int = 192, out_dim: int = 256) -> dict:
    head = ArcFaceEmbeddingHead(embed_dim, out_dim)
    return {
        "projector.net.0.weight": head.net[0].weight.detach(),
        "projector.net.2.weight": head.net[2].weight.detach(),
    }


def _projection_weights(embed_dim: int = 192, out_dim: int = 256) -> dict:
    return {
        "projector.net.0.weight": torch.zeros(2048, embed_dim),
        "projector.net.4.weight": torch.zeros(out_dim, 2048),
    }


def test_head_is_inferred_from_projector_width_when_metadata_is_absent():
    assert resolve_checkpoint_embedding_head({}, _arcface_weights()) == (
        ARCFACE_EMBEDDING_HEAD
    )
    assert resolve_checkpoint_embedding_head({}, _projection_weights()) == (
        PROJECTION_EMBEDDING_HEAD
    )
    # No projector weights at all: historical default is ProjectionHead.
    assert resolve_checkpoint_embedding_head({}, {}) == PROJECTION_EMBEDDING_HEAD


def test_declared_head_wins_even_when_shapes_disagree():
    assert resolve_checkpoint_embedding_head(
        {"config": {"embedding_head": ARCFACE_EMBEDDING_HEAD}}, _projection_weights()
    ) == ARCFACE_EMBEDDING_HEAD


def test_unknown_declared_head_and_unknown_projector_width_are_rejected():
    with pytest.raises(ValueError, match="Unsupported embedding head"):
        resolve_checkpoint_embedding_head(
            {"config": {"embedding_head": "mystery_head"}}, _projection_weights()
        )
    with pytest.raises(ValueError, match="Unrecognised projector width"):
        resolve_checkpoint_embedding_head(
            {}, {"projector.net.0.weight": torch.zeros(768, 192)}
        )


def test_resolve_checkpoint_prefers_student_only_when_requested():
    student = _arcface_weights()
    checkpoint = {
        "teacher": _projection_weights(),
        "student": student,
        "model_state_dict": {},
    }
    assert resolve_checkpoint_state_dict(checkpoint, prefer_student=True)[1] == "student"
    assert resolve_checkpoint_state_dict(checkpoint)[1] == "teacher"


def test_prefer_student_still_falls_back_for_ref_script_sft():
    """``--use-student`` must have no effect on checkpoints that hold no student."""
    ref_script_sft = {"model": {"projector.0.weight": torch.zeros(512, 192)}}

    state_dict, source_key = resolve_checkpoint_state_dict(
        ref_script_sft, prefer_student=True
    )

    assert source_key == "model"
    assert state_dict is ref_script_sft["model"]


def test_apply_checkpoint_weights_rejects_an_incompatible_backbone():
    from otuformer.training.model import OTUFormerEncoder
    from otuformer.utils.checkpoint import (
        CheckpointArchitecture,
        apply_checkpoint_weights,
    )

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    architecture = CheckpointArchitecture(
        model_name="vit_tiny_patch16_224",
        embedding_dim=8,
        embedding_head=PROJECTION_EMBEDDING_HEAD,
        state_dict={"backbone.not_a_real_layer.weight": torch.zeros(2, 2)},
        source_key="model_state_dict",
    )

    with pytest.raises(ValueError, match="pass the correct --model-name"):
        apply_checkpoint_weights(model, architecture)


def test_apply_checkpoint_weights_blames_metadata_when_it_conflicts_with_weights():
    """A recorded model name outranks --model-name, so the hint must say so."""
    from otuformer.training.model import OTUFormerEncoder
    from otuformer.utils.checkpoint import (
        CheckpointArchitecture,
        apply_checkpoint_weights,
    )

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    architecture = CheckpointArchitecture(
        model_name="vit_tiny_patch16_224",
        embedding_dim=8,
        embedding_head=PROJECTION_EMBEDDING_HEAD,
        state_dict={"backbone.not_a_real_layer.weight": torch.zeros(2, 2)},
        source_key="model_state_dict",
        model_name_from_metadata=True,
    )

    with pytest.raises(ValueError, match="precedence over --model-name"):
        apply_checkpoint_weights(model, architecture)


def test_apply_checkpoint_weights_accepts_a_matching_backbone():
    from otuformer.training.model import OTUFormerEncoder
    from otuformer.utils.checkpoint import (
        CheckpointArchitecture,
        apply_checkpoint_weights,
    )

    source = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    architecture = CheckpointArchitecture(
        model_name="vit_tiny_patch16_224",
        embedding_dim=8,
        embedding_head=PROJECTION_EMBEDDING_HEAD,
        state_dict=source.state_dict(),
        source_key="model_state_dict",
    )

    apply_checkpoint_weights(model, architecture)

    assert torch.equal(
        model.backbone.cls_token, source.backbone.cls_token
    )


def test_resolve_checkpoint_uses_legacy_args_when_config_is_absent():
    checkpoint = {
        "teacher": _projection_weights(out_dim=256),
        "args": {"model_name": "vit_small_patch16_224", "out_dim": 256},
    }

    resolved = resolve_checkpoint(checkpoint, "vit_tiny_patch16_224")

    assert resolved.model_name == "vit_small_patch16_224"
    assert resolved.embedding_dim == 256
    assert resolved.embedding_head == PROJECTION_EMBEDDING_HEAD
    assert resolved.source_key == "teacher"


def test_resolve_checkpoint_infers_dim_and_head_from_ref_script_arcface_weights():
    head = ArcFaceEmbeddingHead(192, 64)
    checkpoint = {
        "model": {
            "projector.0.weight": head.net[0].weight.detach(),
            "projector.2.weight": head.net[2].weight.detach(),
        }
    }

    resolved = resolve_checkpoint(checkpoint, "vit_tiny_patch16_224")

    assert resolved.source_key == "model"
    assert resolved.model_name == "vit_tiny_patch16_224"  # no config, no args
    assert resolved.embedding_dim == 64
    assert resolved.embedding_head == ARCFACE_EMBEDDING_HEAD
    assert set(normalize_legacy_projector_keys(resolved.state_dict)) == {
        "projector.net.0.weight",
        "projector.net.2.weight",
    }


def test_normalize_legacy_projector_keys_leaves_current_names_untouched():
    state = {
        "backbone.cls_token": torch.zeros(1, 1, 1),
        "projector.net.0.weight": torch.zeros(2, 2),
    }
    assert normalize_legacy_projector_keys(state) is state


def test_projector_out_dim_is_none_without_a_projector():
    assert resolve_projector_out_dim({"backbone.cls_token": torch.zeros(1)}) is None


def test_general_encoder_checkpoint_ignores_patch_objective_state():
    """Read-only consumers load the encoder and never touch training-only state."""
    from otuformer.training.model import OTUFormerEncoder
    from otuformer.utils.checkpoint import apply_checkpoint_weights

    source = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    checkpoint = {
        "model_state_dict": source.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 8},
        "patch_objective": {
            "mask_token": torch.zeros(1, 1, source.backbone.num_features),
            "predictor.0.weight": torch.zeros(
                source.backbone.num_features, source.backbone.num_features
            ),
        },
        "rng_state": {"python": None},
    }

    resolved = resolve_checkpoint(checkpoint, "vit_tiny_patch16_224")
    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False
    )
    apply_checkpoint_weights(model, resolved)

    assert torch.equal(model.backbone.cls_token, source.backbone.cls_token)
    assert model.projector.net[4].weight.shape[0] == 8
    assert all(not key.startswith(("mask_token", "predictor.")) for key in resolved.state_dict)


# --- v0.10.0 register layout resolution ---------------------------------------


def _tiny_vit_model(native_regs=0, class_token=True, embed_dim=32):
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(1)
    return VisionTransformer(
        img_size=32, patch_size=16, in_chans=3, embed_dim=embed_dim, depth=2,
        num_heads=2, mlp_ratio=2.0, num_classes=0, global_pool="",
        class_token=class_token, reg_tokens=native_regs, dynamic_img_size=True,
    )


def _full_tiny_state(native_regs=0, register_count=None):
    """A complete ``backbone.*`` state dict is required by the mismatch check."""
    model = _tiny_vit_model(
        native_regs=native_regs if register_count is None else register_count
    )
    return {f"backbone.{key}": value for key, value in model.state_dict().items()}


def _install_tiny_factory(monkeypatch, native_regs=0, class_token=True):
    import otuformer.training.model as model_module
    from timm.models.vision_transformer import VisionTransformer

    calls = []

    def factory(_model_name, **kwargs):
        calls.append(dict(kwargs))
        requested = kwargs.get("reg_tokens")
        count = native_regs if requested is None else int(requested)
        torch.manual_seed(1)
        return VisionTransformer(
            img_size=32, patch_size=16, in_chans=3, embed_dim=32, depth=2,
            num_heads=2, mlp_ratio=2.0, num_classes=0, global_pool="",
            class_token=class_token, reg_tokens=count, dynamic_img_size=True,
        )

    monkeypatch.setattr(model_module.timm, "create_model", factory)
    return calls


@pytest.mark.parametrize(
    "state_dict",
    [
        {"_orig_mod.backbone.reg_token": torch.zeros(1, 4, 32)},
        {"module.backbone.reg_token": torch.zeros(1, 4, 32)},
        {"student.backbone.reg_token": torch.zeros(1, 4, 32)},
    ],
)
def test_noncanonical_register_keys_are_rejected(state_dict):
    with pytest.raises(ValueError, match="backbone.reg_token"):
        reject_noncanonical_register_keys(state_dict)


def test_canonical_and_absent_register_keys_pass_the_prescan():
    reject_noncanonical_register_keys({})
    reject_noncanonical_register_keys(
        {"backbone.reg_token": torch.zeros(1, 4, 32), "backbone.cls_token": None}
    )


@pytest.mark.parametrize(
    "native,recorded,shape,expected",
    [
        (0, None, None, (0, 0)),
        (0, 0, None, (0, 0)),
        (0, None, (1, 4, 32), "reject"),
        (0, 0, (1, 4, 32), "reject"),
        (0, 4, (1, 4, 32), (4, 4)),
        (0, 4, None, "reject"),
        (0, 4, (1, 3, 32), "reject"),
        (0, 4, (1, 4, 16), "reject"),
        (0, 3, None, "reject"),
        (4, None, (1, 4, 32), (4, 0)),
        (4, 4, (1, 4, 32), (4, 0)),
        (4, 2, (1, 4, 32), "reject"),
        (4, None, None, "reject"),
        (2, None, (1, 2, 32), (2, 0)),
        (2, 4, (1, 2, 32), "reject"),
    ],
)
def test_resolve_register_layout_follows_the_design_table(
    native, recorded, shape, expected
):
    backbone = _tiny_vit_model(native_regs=native)
    state = {} if shape is None else {"backbone.reg_token": torch.zeros(*shape)}
    checkpoint = {} if recorded is None else {"config": {"register_tokens": recorded}}

    if expected == "reject":
        with pytest.raises(ValueError):
            resolve_register_layout(checkpoint, state, backbone)
    else:
        assert resolve_register_layout(checkpoint, state, backbone) == expected


def test_register_count_never_falls_back_to_invocation_args():
    backbone = _tiny_vit_model(native_regs=0)
    checkpoint = {"args": {"register_tokens": 4}, "config": {"model_name": "x"}}
    assert resolve_register_layout(checkpoint, {}, backbone) == (0, 0)


def test_recorded_count_rebuilds_four_when_args_say_zero_or_none():
    backbone = _tiny_vit_model(native_regs=0)
    state = {"backbone.reg_token": torch.zeros(1, 4, 32)}
    for invocation in (0, None):
        checkpoint = {
            "config": {"register_tokens": 4},
            "args": {"register_tokens": invocation},
        }
        assert resolve_register_layout(checkpoint, state, backbone) == (4, 4)


def test_no_cls_zero_register_layout_is_legacy_but_rejects_registers():
    backbone = _tiny_vit_model(native_regs=0, class_token=False)
    assert resolve_register_layout({}, {}, backbone) == (0, 0)
    assert resolve_register_layout(
        {"config": {"register_tokens": 0}}, {}, backbone
    ) == (0, 0)
    with pytest.raises(ValueError):
        resolve_register_layout({"config": {"register_tokens": 4}}, {}, backbone)
    with pytest.raises(ValueError):
        resolve_register_layout(
            {}, {"backbone.reg_token": torch.zeros(1, 4, 32)}, backbone
        )


def test_no_cls_native_register_model_is_rejected_by_the_constructor(monkeypatch):
    from otuformer.training.model import OTUFormerEncoder

    _install_tiny_factory(monkeypatch, native_regs=4, class_token=False)
    with pytest.raises(ValueError, match="no CLS token"):
        OTUFormerEncoder(
            model_name="tiny-real", out_dim=8, return_patch_tokens=True,
            img_size=32, pretrained=False,
        )


def test_loader_rejects_a_noncanonical_key_before_constructing_anything(monkeypatch):
    calls = _install_tiny_factory(monkeypatch)
    checkpoint = {"model_state_dict": {"_orig_mod.backbone.reg_token": torch.zeros(1, 4, 32)}}
    architecture = resolve_checkpoint(checkpoint, "tiny-real")

    with pytest.raises(ValueError, match="backbone.reg_token"):
        load_checkpoint_encoder(checkpoint, architecture, image_size=32)

    assert calls == []


def test_loader_rejects_a_native_register_model_that_is_not_a_vit(monkeypatch):
    import otuformer.training.model as model_module

    class _NotAVit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reg_token = torch.nn.Parameter(torch.zeros(1, 4, 32))
            self.num_reg_tokens = 4
            self.num_prefix_tokens = 5
            self.num_features = 32

    monkeypatch.setattr(
        model_module.timm, "create_model", lambda _name, **kwargs: _NotAVit()
    )
    checkpoint = {"model_state_dict": {}}
    architecture = resolve_checkpoint(checkpoint, "vit_small_patch16_dinov3.lvd1689m")

    with pytest.raises(ValueError, match="standard timm VisionTransformer"):
        load_checkpoint_encoder(checkpoint, architecture, image_size=32)


def test_loader_reuses_the_native_encoder_and_fills_both_counts(monkeypatch):
    calls = _install_tiny_factory(monkeypatch)
    checkpoint = {
        "model_state_dict": _full_tiny_state(),
        "config": {"model_name": "tiny-real", "out_dim": 8},
    }
    architecture = resolve_checkpoint(checkpoint, "tiny-real")

    encoder, resolved = load_checkpoint_encoder(checkpoint, architecture, image_size=32)

    assert len(calls) == 1
    assert resolved.actual_register_tokens == 0
    assert resolved.added_register_tokens == 0
    assert encoder.backbone.num_prefix_tokens == 1
    assert encoder.return_patch_tokens is True


def test_loader_builds_added_four_and_fills_both_counts(monkeypatch):
    calls = _install_tiny_factory(monkeypatch)
    checkpoint = {
        "model_state_dict": _full_tiny_state(register_count=4),
        "config": {"model_name": "tiny-real", "out_dim": 8, "register_tokens": 4},
    }
    architecture = resolve_checkpoint(checkpoint, "tiny-real")

    encoder, resolved = load_checkpoint_encoder(checkpoint, architecture, image_size=32)

    # One probe plus the registered construction, which re-probes so a direct
    # caller can never bypass the native-register rejection.
    assert len(calls) == 3
    assert "reg_tokens" not in calls[0]
    assert calls[-1]["reg_tokens"] == 4
    assert resolved.actual_register_tokens == 4
    assert resolved.added_register_tokens == 4
    assert encoder.backbone.num_prefix_tokens == 5
    assert torch.equal(
        encoder.backbone.reg_token,
        checkpoint["model_state_dict"]["backbone.reg_token"],
    )


def test_loader_reuses_a_native_register_encoder(monkeypatch):
    calls = _install_tiny_factory(monkeypatch, native_regs=2)
    checkpoint = {
        "model_state_dict": _full_tiny_state(native_regs=2),
        "config": {"model_name": "tiny-real", "out_dim": 8},
    }
    architecture = resolve_checkpoint(checkpoint, "tiny-real")

    encoder, resolved = load_checkpoint_encoder(checkpoint, architecture, image_size=32)

    assert len(calls) == 1
    assert (resolved.actual_register_tokens, resolved.added_register_tokens) == (2, 0)
    assert encoder.backbone.num_prefix_tokens == 3


def test_loader_preserves_python_numpy_and_torch_rng_state(monkeypatch):
    import random

    import numpy as np

    _install_tiny_factory(monkeypatch)
    checkpoint = {
        "model_state_dict": _full_tiny_state(register_count=4),
        "config": {"model_name": "tiny-real", "out_dim": 8, "register_tokens": 4},
    }
    architecture = resolve_checkpoint(checkpoint, "tiny-real")
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    before = (
        random.getstate(),
        np.random.get_state()[1].copy(),
        torch.get_rng_state().clone(),
    )

    load_checkpoint_encoder(checkpoint, architecture, image_size=32)

    assert random.getstate() == before[0]
    assert (np.random.get_state()[1] == before[1]).all()
    assert torch.equal(torch.get_rng_state(), before[2])


def test_loader_ignores_alias_and_training_only_namespaces(monkeypatch):
    _install_tiny_factory(monkeypatch)
    state = _full_tiny_state()
    state["_pretrain_final_norm.weight"] = torch.zeros(32)
    state["_pretrain_final_norm.bias"] = torch.zeros(32)
    state["center"] = torch.zeros(1, 8)
    checkpoint = {
        "model_state_dict": state,
        "config": {"model_name": "tiny-real", "out_dim": 8},
        "patch_objective": {"mask_token": torch.zeros(1, 1, 32)},
    }
    architecture = resolve_checkpoint(checkpoint, "tiny-real")

    encoder, resolved = load_checkpoint_encoder(checkpoint, architecture, image_size=32)

    assert resolved.actual_register_tokens == 0
    assert encoder.backbone.num_prefix_tokens == 1
