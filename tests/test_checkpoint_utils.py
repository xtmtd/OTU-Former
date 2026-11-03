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
    resolve_checkpoint_ssl_identity,
    resolve_checkpoint_ssl_source,
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
    """A complete ProjectionHead state (three linear layers, each with a bias)."""
    from otuformer.training.model import ProjectionHead

    head = ProjectionHead(embed_dim, 2048, out_dim)
    return {f"projector.{key}": value for key, value in head.state_dict().items()}


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


@pytest.mark.parametrize("head", [ARCFACE_EMBEDDING_HEAD, PROJECTION_EMBEDDING_HEAD])
@pytest.mark.parametrize("corruption", ["missing_weight", "missing_bias", "wrong_bias", "wrong_input"])
def test_metric_checkpoint_loading_rejects_incomplete_projector(head, corruption):
    from otuformer.training.model import OTUFormerEncoder
    from otuformer.utils.checkpoint import apply_checkpoint_weights

    source = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, img_size=32, pretrained=False
    )
    if head == ARCFACE_EMBEDDING_HEAD:
        source.projector = ArcFaceEmbeddingHead(192, 8)
    state = dict(source.state_dict())
    last = 2 if head == ARCFACE_EMBEDDING_HEAD else 4
    if corruption == "missing_weight":
        del state["projector.net.2.weight"]
    elif corruption == "missing_bias":
        del state[f"projector.net.{last}.bias"]
    elif corruption == "wrong_bias":
        state[f"projector.net.{last}.bias"] = torch.zeros(7)
    else:
        state["projector.net.0.weight"] = torch.zeros(
            state["projector.net.0.weight"].shape[0], 1
        )
    checkpoint = {"model_state_dict": state, "config": {
        "model_name": "vit_tiny_patch16_224", "out_dim": 8,
        "embedding_head": head,
        "ssl_source_method": "dino", "ssl_source_head": "dino_mlp_2048",
        "ssl_initialization_checkpoint_sha256": "a" * 64,
    }}
    architecture = resolve_checkpoint(checkpoint, "vit_tiny_patch16_224")
    target = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, img_size=32, pretrained=False
    )
    with pytest.raises(ValueError, match="projector"):
        apply_checkpoint_weights(target, architecture)


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


# ---------------------------------------------------------------------------
# SSL identity vs embedding-layout identity (CLS distillation, Task 1)
# ---------------------------------------------------------------------------

_SSL_SHA = "0123456789abcdef" * 4


def _ssl_payload(
    method=None,
    head=None,
    out_dim=17,
    weights=None,
    config=None,
    **extra,
):
    cfg = {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim}
    if method is not None:
        cfg["ssl_method"] = method
    if head is not None:
        cfg["ssl_head"] = head
    cfg.update(config or {})
    payload = {
        "teacher": _projection_weights(out_dim=out_dim)
        if weights is None
        else weights,
        "config": cfg,
    }
    payload.update(extra)
    return payload


def _metric_payload(**config):
    cfg = {
        "model_name": "vit_tiny_patch16_224",
        "out_dim": 32,
        "embedding_head": ARCFACE_EMBEDDING_HEAD,
    }
    cfg.update(config)
    return {"model": _arcface_weights(out_dim=32), "config": cfg}


def test_ssl_identity_is_separate_from_metric_layout():
    dino = resolve_checkpoint(
        _ssl_payload("dino", "dino_mlp_2048"), "vit_tiny_patch16_224"
    )
    # The extractable layout stays the shared normalized MLP; the SSL method is
    # training identity, not an inference layout.
    assert dino.embedding_head == PROJECTION_EMBEDDING_HEAD
    assert dino.ssl_identity == ("dino", "dino_mlp_2048")
    assert dino.ssl_source_identity is None

    relational = resolve_checkpoint(
        _ssl_payload("relational", "projection_mlp_2048"), "vit_tiny_patch16_224"
    )
    assert relational.ssl_identity == ("relational", "projection_mlp_2048")
    assert relational.embedding_head == PROJECTION_EMBEDDING_HEAD

    state, _ = resolve_checkpoint_state_dict(
        _ssl_payload("dino", "dino_mlp_2048")
    )
    assert resolve_checkpoint_ssl_identity(
        {"config": {"out_dim": 17, "ssl_method": "dino", "ssl_head": "dino_mlp_2048"}},
        state,
    ) == ("dino", "dino_mlp_2048")

    # Partial, unknown, mismatched, shape-inconsistent and contradictory
    # identities must fail instead of degrading to a historical inference.
    for payload in (
        _ssl_payload("dino"),
        _ssl_payload(head="dino_mlp_2048"),
        _ssl_payload("mystery", "dino_mlp_2048"),
        _ssl_payload("dino", "mystery_mlp_2048"),
        _ssl_payload("relational", "dino_mlp_2048"),
        _ssl_payload(dino_objective={}),
        _ssl_payload(
            "dino",
            "dino_mlp_2048",
            config={"embedding_head": ARCFACE_EMBEDDING_HEAD},
        ),
        _ssl_payload("dino", "dino_mlp_2048", config={"loss": {}}),
        # D out_dim disagrees with the recorded D-dimensional MLP.
        _ssl_payload(
            "dino", "dino_mlp_2048", out_dim=17, weights=_projection_weights(out_dim=32)
        ),
        # 512-wide projector cannot carry an SSL identity.
        _ssl_payload("dino", "dino_mlp_2048", out_dim=17, weights=_arcface_weights(out_dim=17)),
    ):
        with pytest.raises(ValueError):
            resolve_checkpoint(payload, "vit_tiny_patch16_224")


def test_dino_source_metric_checkpoint_is_valid():
    payload = _metric_payload(
        ssl_source_method="dino",
        ssl_source_head="dino_mlp_2048",
        ssl_initialization_checkpoint_sha256=_SSL_SHA,
    )
    architecture = resolve_checkpoint(payload, "vit_tiny_patch16_224")

    assert architecture.ssl_identity is None
    assert architecture.embedding_head == ARCFACE_EMBEDDING_HEAD
    assert architecture.ssl_source_identity == ("dino", "dino_mlp_2048")
    assert resolve_checkpoint_ssl_source(payload) == ("dino", "dino_mlp_2048")


def test_source_identity_requires_sha_anchor():
    unanchored_metric_checkpoint = _metric_payload(
        ssl_source_method="dino", ssl_source_head="dino_mlp_2048"
    )
    with pytest.raises(ValueError, match="SHA|source|identity"):
        resolve_checkpoint(unanchored_metric_checkpoint, "vit_tiny_patch16_224")

    for bad in (
        _metric_payload(
            ssl_source_method="dino",
            ssl_source_head="dino_mlp_2048",
            ssl_initialization_checkpoint_sha256="not-a-sha",
        ),
        _metric_payload(
            ssl_source_method="dino", ssl_initialization_checkpoint_sha256=_SSL_SHA
        ),
        _metric_payload(
            ssl_source_head="dino_mlp_2048", ssl_initialization_checkpoint_sha256=_SSL_SHA
        ),
        _metric_payload(
            ssl_source_method="mystery",
            ssl_source_head="dino_mlp_2048",
            ssl_initialization_checkpoint_sha256=_SSL_SHA,
        ),
        _metric_payload(
            ssl_source_method="relational",
            ssl_source_head="dino_mlp_2048",
            ssl_initialization_checkpoint_sha256=_SSL_SHA,
        ),
    ):
        with pytest.raises(ValueError, match="SHA|source|identity"):
            resolve_checkpoint(bad, "vit_tiny_patch16_224")

    # A missing pair stays valid, including historical SHA-only provenance.
    assert (
        resolve_checkpoint(_metric_payload(), "vit_tiny_patch16_224").ssl_source_identity
        is None
    )
    sha_only = _metric_payload(ssl_initialization_checkpoint_sha256=_SSL_SHA)
    assert resolve_checkpoint(sha_only, "vit_tiny_patch16_224").ssl_source_identity is None


def test_ssl_identity_payload_precedence():
    # Rule 1: metric/head-only/legacy markers keep the payload out of SSL
    # identity resolution, whatever the projector width looks like.
    metric_with_ssl_looking_weights = resolve_checkpoint(
        {"model": _projection_weights(out_dim=17), "config": {"out_dim": 17}},
        "vit_tiny_patch16_224",
    )
    assert metric_with_ssl_looking_weights.ssl_identity is None

    # A declared 2048-wide head-only file is still head-only, not relational.
    declared_head_only = resolve_checkpoint(
        {
            "teacher": _projection_weights(out_dim=17),
            "config": {"embedding_head": PROJECTION_EMBEDDING_HEAD, "out_dim": 17},
        },
        "vit_tiny_patch16_224",
    )
    assert declared_head_only.ssl_identity is None
    assert declared_head_only.embedding_head == PROJECTION_EMBEDDING_HEAD

    for legacy_marker in (
        {"loss_state_dict": {}},  # key presence counts even when empty
        {"class_labels": ["species_a"]},
        {"loss_func": "arcface"},
        {"config": {"out_dim": 17, "loss": {}}},
    ):
        payload = {"teacher": _projection_weights(out_dim=17), **legacy_marker}
        assert (
            resolve_checkpoint(payload, "vit_tiny_patch16_224").ssl_identity is None
        ), legacy_marker

    # Those markers still reject a contradictory active SSL identity/state.
    for contradictory in (
        {
            "model": _projection_weights(out_dim=17),
            "config": {"ssl_method": "dino", "ssl_head": "dino_mlp_2048"},
        },
        {"model": _projection_weights(out_dim=17), "dino_objective": {}},
        {
            "teacher": _projection_weights(out_dim=17),
            "config": {"loss": {}, "ssl_method": "relational", "ssl_head": "projection_mlp_2048"},
        },
        {"teacher": _projection_weights(out_dim=17), "loss_state_dict": {}, "dino_objective": {}},
    ):
        with pytest.raises(ValueError):
            resolve_checkpoint(contradictory, "vit_tiny_patch16_224")

    # Rule 2: a 512-wide first projector marks a metric layout.
    width_512 = resolve_checkpoint(
        {"teacher": _arcface_weights(out_dim=17)}, "vit_tiny_patch16_224"
    )
    assert width_512.ssl_identity is None
    assert width_512.embedding_head == ARCFACE_EMBEDDING_HEAD

    # Rule 5: a complete historical 2048-hidden MLP infers relational.
    historical = resolve_checkpoint(
        {
            "teacher": _projection_weights(out_dim=17),
            "args": {"model_name": "vit_tiny_patch16_224", "out_dim": 17},
        },
        "vit_tiny_patch16_224",
    )
    assert historical.ssl_identity == ("relational", "projection_mlp_2048")

    # Rule 6: no recognizable projector leaves the identity unknown.
    backbone_only = resolve_checkpoint(
        {"teacher": {"backbone.cls_token": torch.zeros(1, 1, 192)}},
        "vit_tiny_patch16_224",
    )
    assert backbone_only.ssl_identity is None


def test_recorded_ssl_identity_requires_the_complete_projector():
    """A recorded identity may not load with a randomly initialised middle layer."""
    complete = _projection_weights(out_dim=17)
    assert resolve_checkpoint(
        _ssl_payload("dino", "dino_mlp_2048", weights=complete),
        "vit_tiny_patch16_224",
    ).ssl_identity == ("dino", "dino_mlp_2048")

    for missing in (
        "projector.net.2.weight",
        "projector.net.2.bias",
        "projector.net.0.bias",
    ):
        partial = dict(complete)
        partial.pop(missing)
        with pytest.raises(ValueError, match="projector|complete|MLP"):
            resolve_checkpoint(
                _ssl_payload("dino", "dino_mlp_2048", weights=partial),
                "vit_tiny_patch16_224",
            )

    # A shape-inconsistent middle layer is equally unusable.
    wrong_middle = dict(complete)
    wrong_middle["projector.net.2.weight"] = torch.zeros(64, 2048)
    with pytest.raises(ValueError, match="projector|shape|MLP"):
        resolve_checkpoint(
            _ssl_payload("dino", "dino_mlp_2048", weights=wrong_middle),
            "vit_tiny_patch16_224",
        )

    # The output bias must agree with the output width.
    wrong_bias = dict(complete)
    wrong_bias["projector.net.4.bias"] = torch.zeros(5)
    with pytest.raises(ValueError, match="entries|rows|bias"):
        resolve_checkpoint(
            _ssl_payload("dino", "dino_mlp_2048", weights=wrong_bias),
            "vit_tiny_patch16_224",
        )

    # Historical (unrecorded) payloads keep their permissive inference.
    historical = {"teacher": {"projector.net.0.weight": torch.zeros(2048, 192),
                             "projector.net.4.weight": torch.zeros(17, 2048)}}
    assert resolve_checkpoint(
        historical, "vit_tiny_patch16_224"
    ).ssl_identity == ("relational", "projection_mlp_2048")


def test_relational_identity_rejects_dino_only_state():
    payload = _ssl_payload("relational", "projection_mlp_2048")
    assert resolve_checkpoint(payload, "vit_tiny_patch16_224").ssl_identity == (
        "relational",
        "projection_mlp_2048",
    )

    with_objective = _ssl_payload(
        "relational", "projection_mlp_2048", dino_objective={}
    )
    with_config = _ssl_payload(
        "relational",
        "projection_mlp_2048",
        config={"dino_prototypes": 11},
    )
    for corrupt in (with_objective, with_config):
        with pytest.raises(ValueError, match="relational|DINO|inconsistent"):
            resolve_checkpoint(corrupt, "vit_tiny_patch16_224")
