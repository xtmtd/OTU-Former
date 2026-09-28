import copy
import math

import pytest
import torch
import torch.nn.functional as F

from otuformer.training.loss import (
    ArcFaceLoss,
    SubCenterArcFaceLoss,
    SupConLoss,
)
from otuformer.training.model import (
    ArcFaceEmbeddingHead,
    ArcFaceHead,
    OTUFormerEncoder,
    SubCenterArcFaceHead,
)
from otuformer.training.trainer import (
    _build_finetune_optimizer,
    _build_finetune_optimizer_for_layout,
    _finetune_optimizer_layout,
    _select_finetune_embedding_head,
)


def test_encoder_does_not_silently_fall_back_to_random_weights(monkeypatch):
    import otuformer.training.model as model_module

    calls = []

    def fail_pretrained(_model_name, **kwargs):
        calls.append(kwargs)
        raise OSError("corrupt pretrained-weight cache")

    monkeypatch.setattr(model_module.timm, "create_model", fail_pretrained)

    with pytest.raises(RuntimeError, match="pretrained backbone weights"):
        OTUFormerEncoder(model_name="vit_tiny_patch16_224")

    assert calls
    assert all(kwargs["pretrained"] for kwargs in calls)


def test_encoder_forwards_pretrained_false_to_all_timm_branches(monkeypatch):
    import otuformer.training.model as model_module

    calls = []

    def fail_create_model(_model_name, **kwargs):
        calls.append(kwargs)
        raise OSError("no local weights required")

    monkeypatch.setattr(model_module.timm, "create_model", fail_create_model)

    with pytest.raises(RuntimeError, match="Could not load pretrained backbone weights"):
        OTUFormerEncoder(model_name="vit_tiny_patch16_224", pretrained=False)

    assert calls
    assert all(kwargs["pretrained"] is False for kwargs in calls)


def test_encoder_rejects_cnn_style_backbone_with_clear_error():
    """A CNN-style backbone (rejects the ViT-only kwargs) must produce an
    honest error instead of blaming the timm/Hugging Face cache."""
    with pytest.raises(RuntimeError, match="CNN-style model"):
        OTUFormerEncoder(model_name="convnextv2_femto", out_dim=8, pretrained=False)


def test_encoder_output_shape():
    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=128)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 128)


def test_encoder_output_is_l2_normalized():
    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=128)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_arcface_embedding_head_uses_small_unnormalized_mlp():
    head = ArcFaceEmbeddingHead(embed_dim=192, metric_embed_dim=256)
    output = head(torch.randn(4, 192))

    assert tuple(head.net[0].weight.shape) == (512, 192)
    assert tuple(head.net[2].weight.shape) == (256, 512)
    assert output.shape == (4, 256)
    assert not torch.allclose(output.norm(dim=1), torch.ones(4), atol=1e-5)


def test_finetune_head_selection_preserves_historical_sft_projector():
    # Bare encoder state with no fine-tune markers is SSL initialization.
    assert _select_finetune_embedding_head(
        {"model_state_dict": {"backbone.cls_token": torch.zeros(1, 1, 4)}}
    ) == "arcface_mlp_512"
    # Without real encoder weights it is not a valid SSL source.
    with pytest.raises(ValueError, match="encoder weights"):
        _select_finetune_embedding_head({})
    with pytest.raises(ValueError, match="encoder weights"):
        _select_finetune_embedding_head({"model_state_dict": {}})
    # An empty loss state with no model state is rejected outright.
    with pytest.raises(ValueError, match="encoder weights"):
        _select_finetune_embedding_head({"loss_state_dict": {}})
    with pytest.raises(ValueError, match="legacy fine-tune checkpoint format"):
        _select_finetune_embedding_head({"loss_func": {}})
    # A declared head without projector weights is still usable for --checkpoint.
    assert _select_finetune_embedding_head(
        {
            "model_state_dict": {"backbone.cls_token": torch.zeros(1, 1, 192)},
            "config": {"embedding_head": "arcface_mlp_512"},
        }
    ) == "arcface_mlp_512"
    # Historical ProjectionHead coverage uses a real legacy ArcFace shape.
    legacy = {
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
        "model_state_dict": {
            "projector.net.0.weight": torch.zeros(2048, 4),
            "projector.net.4.weight": torch.zeros(16, 2048),
        },
    }
    assert _select_finetune_embedding_head(legacy) == "projection_mlp_2048"


def _linear_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.projector = torch.nn.Linear(4, 2)

    return Model()


def _ids(params):
    return [id(p) for p in params]


def test_finetune_optimizer_layout_selects_recorded_policy():
    v080_arcface = {
        "config": {"loss": "arcface"},
        "optimizer": {"param_groups": [{}, {}, {}]},
    }
    v080_supcon = {
        "config": {"loss": "supcon"},
        "optimizer": {"param_groups": [{}, {}]},
    }
    legacy_one_group = {"optimizer": {"param_groups": [{}]}}
    legacy_two_groups = {"optimizer": {"param_groups": [{}, {}]}}

    assert _finetune_optimizer_layout(None, False, "arcface") == "v080_prototype"
    assert _finetune_optimizer_layout(None, False, "supcon") == "v080_supcon"
    assert (
        _finetune_optimizer_layout(v080_arcface, True, "arcface") == "v080_prototype"
    )
    assert _finetune_optimizer_layout(v080_supcon, True, "supcon") == "v080_supcon"
    assert (
        _finetune_optimizer_layout(legacy_one_group, True, "arcface")
        == "legacy_single"
    )
    assert (
        _finetune_optimizer_layout(legacy_two_groups, True, "arcface")
        == "legacy_split"
    )


@pytest.mark.parametrize(
    "checkpoint",
    [
        # v0.8.0 prototype policy recorded with the wrong group count
        {"config": {"loss": "arcface"}, "optimizer": {"param_groups": [{}, {}]}},
        # v0.8.0 SupCon policy recorded with the wrong group count
        {
            "config": {"loss": "supcon"},
            "optimizer": {"param_groups": [{}, {}, {}]},
        },
        # unmarked three-group optimizer cannot be attributed to a policy
        {"optimizer": {"param_groups": [{}, {}, {}]}},
        {"optimizer": {"param_groups": []}},
    ],
)
def test_finetune_optimizer_layout_rejects_inconsistent_group_counts(checkpoint):
    with pytest.raises(ValueError, match="optimizer"):
        _finetune_optimizer_layout(checkpoint, True, "arcface")


def test_finetune_optimizer_uses_distinct_backbone_and_head_learning_rates():
    model = _linear_model()
    loss = ArcFaceLoss(embed_dim=2, num_classes=3)

    optimizer = _build_finetune_optimizer(model, loss, 3e-5, 1e-4, 1e-4)

    assert [group["lr"] for group in optimizer.param_groups] == [3e-5, 1e-4, 1e-4]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        1e-4,
        1e-4,
        0.0,
    ]
    assert len(optimizer.param_groups) == 3
    assert _ids(optimizer.param_groups[0]["params"]) == _ids(
        [p for p in model.backbone.parameters() if p.requires_grad]
    )
    assert _ids(optimizer.param_groups[1]["params"]) == _ids(
        model.projector.parameters()
    )
    assert _ids(optimizer.param_groups[2]["params"]) == _ids(loss.parameters())

    supcon = SupConLoss(temperature=0.07)
    supcon_optimizer = _build_finetune_optimizer_for_layout(
        "v080_supcon", model, supcon, 3e-5, 1e-4, 1e-4
    )

    assert len(supcon_optimizer.param_groups) == 2
    assert [group["lr"] for group in supcon_optimizer.param_groups] == [3e-5, 1e-4]
    assert [
        group["weight_decay"] for group in supcon_optimizer.param_groups
    ] == [1e-4, 1e-4]
    assert all(group["params"] for group in supcon_optimizer.param_groups)


def test_subcenter_prototype_is_the_only_zero_decay_group():
    model = _linear_model()
    loss = SubCenterArcFaceLoss(embed_dim=2, num_classes=3, k=2)

    optimizer = _build_finetune_optimizer(model, loss, 1e-4, None, 1e-4)

    assert _ids(optimizer.param_groups[2]["params"]) == [id(loss.head.weight)]
    assert optimizer.param_groups[2]["weight_decay"] == 0.0
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-4, 1e-4]


def test_legacy_optimizer_layouts_rebuild_and_load_unchanged():
    model = _linear_model()
    loss = ArcFaceLoss(embed_dim=2, num_classes=3)
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    projector = list(model.projector.parameters())
    prototype = list(loss.parameters())

    historical_single = torch.optim.AdamW(
        backbone + projector + prototype, lr=3e-5, weight_decay=1e-4
    )
    single_state = copy.deepcopy(historical_single.state_dict())
    rebuilt_single = _build_finetune_optimizer_for_layout(
        "legacy_single", model, loss, 1e-4, None, 1e-4
    )
    rebuilt_single.load_state_dict(single_state)

    assert [group["lr"] for group in rebuilt_single.param_groups] == [3e-5]
    # The historical prototype decay (1e-4) survives; the new zero-decay rule
    # is never imposed on a legacy resume.
    assert rebuilt_single.param_groups[0]["weight_decay"] == 1e-4

    historical_split = torch.optim.AdamW(
        [
            {"params": backbone, "lr": 3e-5, "weight_decay": 1e-4},
            {"params": projector + prototype, "lr": 1e-4, "weight_decay": 1e-4},
        ]
    )
    split_state = copy.deepcopy(historical_split.state_dict())
    rebuilt_split = _build_finetune_optimizer_for_layout(
        "legacy_split", model, loss, 1e-4, 1e-4, 1e-4
    )

    assert _ids(rebuilt_split.param_groups[1]["params"]) == _ids(
        projector + prototype
    )
    rebuilt_split.load_state_dict(split_state)

    assert [group["lr"] for group in rebuilt_split.param_groups] == [3e-5, 1e-4]
    assert [group["weight_decay"] for group in rebuilt_split.param_groups] == [
        1e-4,
        1e-4,
    ]


def test_arcface_head_output_shape():
    head = ArcFaceHead(embed_dim=128, num_classes=10)
    x = torch.randn(4, 128)
    out = head(x)
    assert out.shape == (4, 10)


def test_encoder_patch_tokens_available():
    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=128,
        return_patch_tokens=True,
    )
    x = torch.randn(2, 3, 224, 224)
    cls_out, patch_tokens = model(x)
    assert cls_out.shape == (2, 128)
    assert patch_tokens.ndim == 3


def test_encoder_accepts_local_crop_size():
    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=64)
    x = torch.randn(2, 3, 96, 96)
    out = model(x)
    assert out.shape == (2, 64)


def test_encoder_inference_contract_is_unchanged_by_pretrain_work():
    """v0.7.0 must not alter the normal forward/extraction contract."""
    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=32,
        return_patch_tokens=True,
        pretrained=False,
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        projected, patch_tokens = model(x)
        raw = model.backbone.forward_features(x)

    assert projected.shape == (2, 32)
    assert torch.allclose(
        projected.norm(dim=1), torch.ones(2), atol=1e-5
    )
    # Patch tokens stay raw final-norm backbone tokens.
    assert torch.allclose(patch_tokens, raw[:, 1:], atol=1e-6)
    assert raw[:, 0].shape[1] == model.backbone.num_features
    assert patch_tokens.shape[1] == (224 // 16) ** 2


def test_encoder_patch_token_count_follows_input_size():
    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=16,
        return_patch_tokens=True,
    )
    model.eval()
    for size, expected in ((224, 196), (96, 36)):
        with torch.no_grad():
            _, tokens = model(torch.randn(1, 3, size, size))
        assert tokens.shape == (1, expected, model.backbone.num_features)


def test_subcenter_head_weight_shape():
    head = SubCenterArcFaceHead(embed_dim=8, num_classes=5, k=3)
    assert tuple(head.weight.shape) == (5, 3, 8)


def test_subcenter_head_rejects_invalid_k():
    with pytest.raises(ValueError):
        SubCenterArcFaceHead(embed_dim=4, num_classes=2, k=0)
    with pytest.raises(ValueError):
        SubCenterArcFaceHead(embed_dim=4, num_classes=2, k=-2)


def test_subcenter_head_k1_matches_arcface_head_logits_and_gradients():
    torch.manual_seed(0)
    arc = ArcFaceHead(embed_dim=8, num_classes=3, s=16.0, m=0.25)
    sub = SubCenterArcFaceHead(embed_dim=8, num_classes=3, k=1, s=16.0, m=0.25)
    with torch.no_grad():
        sub.weight.copy_(arc.weight.unsqueeze(1))

    x_arc = torch.randn(4, 8, requires_grad=True)
    x_sub = x_arc.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 2, 1, 1])
    arc_logits = arc(x_arc, labels)
    sub_logits = sub(x_sub, labels)

    assert torch.allclose(arc_logits, sub_logits, atol=1e-6)
    arc_logits.sum().backward()
    sub_logits.sum().backward()
    assert torch.allclose(x_arc.grad, x_sub.grad, atol=1e-6)
    assert torch.allclose(
        arc.weight.grad, sub.weight.grad.squeeze(1), atol=1e-6
    )


def test_subcenter_head_k2_applies_margin_to_max_target_center():
    head = SubCenterArcFaceHead(embed_dim=4, num_classes=2, k=2, s=32.0, m=0.3)
    with torch.no_grad():
        # Class 0: slot 1 (index 1) is the closest center to e0.
        head.weight[0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        head.weight[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # Class 1: slot 2 is closest to e3, the other center is orthogonal.
        head.weight[1, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        head.weight[1, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])

    x = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], requires_grad=True
    )
    labels = torch.tensor([0, 1])

    cosine = F.normalize(x) @ F.normalize(head.weight.reshape(-1, 4)).T
    cosine = cosine.reshape(2, 2, 2)
    cosine_max = cosine.max(dim=2).values
    assert cosine_max.gather(1, labels[:, None]).squeeze(1).tolist() == [1.0, 1.0]
    # The winning target center lives in the second slot.
    assert cosine[0, 0].argmax().item() == 1
    assert cosine[1, 1].argmax().item() == 1

    cos_m, sin_m = math.cos(0.3), math.sin(0.3)
    th = math.cos(math.pi - 0.3)
    mm = math.sin(math.pi - 0.3) * 0.3
    target = cosine_max.gather(1, labels[:, None]).squeeze(1)
    phi = target * cos_m - torch.sqrt(1.0 - target.pow(2).clamp(0, 1)) * sin_m
    phi = torch.where(target > th, phi, target - mm)
    expected = cosine_max.clone()
    expected[torch.arange(2), labels] = phi

    assert torch.allclose(head(x, labels), expected * 32.0, atol=1e-6)


def test_subcenter_head_without_labels_returns_scaled_cosine():
    torch.manual_seed(3)
    head = SubCenterArcFaceHead(embed_dim=6, num_classes=3, k=2, s=8.0)
    x = torch.randn(5, 6)
    cosine = F.normalize(x) @ F.normalize(head.weight.reshape(-1, 6)).T
    cosine = cosine.reshape(5, 3, 2).max(dim=2).values
    assert torch.allclose(head(x), cosine * 8.0, atol=1e-6)


# --- v0.10.0 optional register tokens -----------------------------------------

_FACTORY_SEED = 0

_TINY_SPECS = {
    "tiny_onecls": {"regs": 0, "cls": True},
    "tiny_native_reg4": {"regs": 4, "cls": True},
    "tiny_native_reg2": {"regs": 2, "cls": True},
    "tiny_gap": {"regs": 0, "cls": False},
    "tiny_gap_reg4": {"regs": 4, "cls": False},
    "tiny_onecls_noembed": {"regs": 0, "cls": True, "no_embed_class": True},
}


def _tensor_hash(tensor: torch.Tensor) -> str:
    import hashlib

    payload = tensor.detach().to(torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


class _VitNamedButNotAVit(torch.nn.Module):
    """Reports native registers without being a timm VisionTransformer."""

    def __init__(self) -> None:
        super().__init__()
        self.reg_token = torch.nn.Parameter(torch.zeros(1, 4, 32))
        self.num_reg_tokens = 4
        self.num_prefix_tokens = 5
        self.has_class_token = False
        self.num_features = 32


def tiny_vit(name="tiny_onecls", pretrained=False, regs=None, embed_dim=32,
             img_size=32, patch_size=16, depth=2, num_heads=2):
    """Build a real tiny timm VisionTransformer, optionally with synthetic weights.

    ``pretrained=True`` fills every tensor with a name-derived constant and makes
    ``pos_embed`` carry its position index, so tests assert exact values and
    positions without any network download.
    """
    from timm.models.vision_transformer import VisionTransformer

    spec = _TINY_SPECS[name]
    torch.manual_seed(_FACTORY_SEED)
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=2.0,
        num_classes=0,
        global_pool="",
        class_token=spec["cls"],
        reg_tokens=spec["regs"] if regs is None else int(regs),
        no_embed_class=bool(spec.get("no_embed_class", False)),
        dynamic_img_size=True,
    )
    if pretrained:
        for tensor_name, tensor in model.state_dict().items():
            if tensor_name == "pos_embed":
                tensor.copy_(
                    torch.arange(tensor.shape[1], dtype=tensor.dtype)
                    .view(1, -1, 1)
                    .expand_as(tensor)
                )
            else:
                tensor.fill_(float(len(tensor_name) % 9) + 1.0)
    return model


def install_tiny_factory(monkeypatch, register_embed_dim=None, calls=None):
    """Route ``timm.create_model`` to :func:`tiny_vit` and record the kwargs."""
    import otuformer.training.model as model_module

    calls = [] if calls is None else calls

    def factory(model_name, **kwargs):
        calls.append(dict(kwargs))
        if model_name not in _TINY_SPECS:
            return _VitNamedButNotAVit()
        spec = _TINY_SPECS[model_name]
        requested = kwargs.get("reg_tokens")
        embed_dim = 32
        if requested is not None and register_embed_dim is not None:
            embed_dim = register_embed_dim
        return tiny_vit(
            model_name,
            pretrained=bool(kwargs.get("pretrained", False)),
            regs=spec["regs"] if requested is None else int(requested),
            embed_dim=embed_dim,
            img_size=int(kwargs.get("img_size", 32)),
        )

    monkeypatch.setattr(model_module.timm, "create_model", factory)
    return calls


def build_tiny_encoder(monkeypatch, model_name="tiny_onecls", **kwargs):
    install_tiny_factory(monkeypatch)
    kwargs.setdefault("out_dim", 8)
    kwargs.setdefault("return_patch_tokens", True)
    kwargs.setdefault("img_size", 32)
    return OTUFormerEncoder(model_name=model_name, **kwargs)


def test_omitted_register_option_preserves_pre_change_construction():
    """Guards the zero-register path against extra construction or RNG use.

    Baseline captured from the unmodified code before this feature
    (workspace ``baseline.json``): with seed S the student projector and with
    seed S+1 the teacher projector initialize to these exact tensors, and the
    two are independently initialized. The projector stream is identical for
    ``pretrained=True`` and ``pretrained=False``, so this offline recipe
    represents the real trainer path.
    """
    from otuformer.training.trainer import _set_seed

    _set_seed(1234)
    student = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=64,
        return_patch_tokens=True, img_size=32, pretrained=False,
    )
    rng_state = torch.get_rng_state()
    _set_seed(1235)
    teacher = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=64,
        return_patch_tokens=True, img_size=32, pretrained=False,
    )
    torch.set_rng_state(rng_state)

    assert _tensor_hash(student.projector.net[0].weight) == "25a3dddafffdaaab"
    assert _tensor_hash(student.projector.net[2].weight) == "fd71bf8c97fea6d8"
    assert _tensor_hash(student.projector.net[4].weight) == "e0bbcff873a45e07"
    assert _tensor_hash(teacher.projector.net[0].weight) == "86651199182904cb"
    assert _tensor_hash(teacher.projector.net[2].weight) == "c3168ca1fdad0fd2"
    assert _tensor_hash(teacher.projector.net[4].weight) == "90aef90fe9a5f239"

    _set_seed(1234)
    finetune = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=64, img_size=32, pretrained=False
    )
    head = ArcFaceEmbeddingHead(finetune.backbone.num_features, 64)
    assert _tensor_hash(finetune.projector.net[0].weight) == "25a3dddafffdaaab"
    assert _tensor_hash(head.net[0].weight) == "f0692685b48e8447"
    assert _tensor_hash(head.net[2].weight) == "7c6eb2860dfa6f9e"


def test_register_option_four_migrates_pretrained_weights_exactly(monkeypatch):
    calls = install_tiny_factory(monkeypatch)
    source = tiny_vit("tiny_onecls", pretrained=True)
    encoder = OTUFormerEncoder(
        model_name="tiny_onecls", out_dim=8, img_size=32,
        reg_tokens=4, pretrained=True,
    )
    target = encoder.backbone
    source_state = source.state_dict()
    target_state = target.state_dict()

    assert set(target_state) - set(source_state) == {"reg_token"}
    assert set(source_state) - set(target_state) == set()
    for key, tensor in source_state.items():
        if key == "pos_embed":
            continue
        assert tensor.shape == target_state[key].shape
        assert torch.equal(tensor, target_state[key])
    assert source_state["pos_embed"].shape[1] + 4 == target_state["pos_embed"].shape[1]
    assert target.num_prefix_tokens == 5
    assert target.num_reg_tokens == 4
    # Native one-CLS construction first, then the registered target without a
    # second pretrained load.
    assert [call["pretrained"] for call in calls] == [True, False]
    assert "reg_tokens" not in calls[0]
    assert calls[1]["reg_tokens"] == 4


def test_register_positions_keep_cls_patches_and_native_register_entries(monkeypatch):
    install_tiny_factory(monkeypatch)
    source = tiny_vit("tiny_onecls", pretrained=True)
    native_registered = tiny_vit("tiny_onecls", regs=4)
    encoder = OTUFormerEncoder(
        model_name="tiny_onecls", out_dim=8, img_size=32,
        reg_tokens=4, pretrained=True,
    )
    pos = encoder.backbone.pos_embed
    assert torch.equal(pos[0, 0], source.pos_embed[0, 0])
    assert torch.equal(pos[0, 5:], source.pos_embed[0, 1:])
    assert torch.equal(pos[0, 1:5], native_registered.pos_embed[0, 1:5])
    assert not torch.equal(pos[0, 1:5], source.pos_embed[0, 1:5])


def test_register_option_four_without_embed_class_keeps_patch_only_pos_embed(monkeypatch):
    install_tiny_factory(monkeypatch)
    source = tiny_vit("tiny_onecls_noembed", pretrained=True)
    encoder = OTUFormerEncoder(
        model_name="tiny_onecls_noembed", out_dim=8, img_size=32,
        reg_tokens=4, pretrained=True,
    )
    target = encoder.backbone
    assert source.pos_embed.shape == target.pos_embed.shape
    assert torch.equal(source.pos_embed, target.pos_embed)
    assert target.pos_embed.shape[1] == 4
    assert target.num_prefix_tokens == 5
    assert target.reg_token is not None


def test_register_option_four_without_pretrained_builds_registered_architecture(monkeypatch):
    calls = install_tiny_factory(monkeypatch)
    encoder = OTUFormerEncoder(
        model_name="tiny_onecls", out_dim=8, img_size=32,
        reg_tokens=4, pretrained=False,
    )
    assert calls and all(call["pretrained"] is False for call in calls)
    assert "reg_tokens" not in calls[0]
    assert calls[-1]["reg_tokens"] == 4
    assert encoder.backbone.num_prefix_tokens == 5
    assert encoder.backbone.reg_token is not None


def test_register_option_zero_preserves_zero_register_construction(monkeypatch):
    calls = install_tiny_factory(monkeypatch)
    encoder = OTUFormerEncoder(
        model_name="tiny_onecls", out_dim=8, img_size=32,
        reg_tokens=0, pretrained=True,
    )
    assert len(calls) == 1
    assert "reg_tokens" not in calls[0]
    assert encoder.backbone.num_prefix_tokens == 1
    assert encoder.backbone.reg_token is None


def test_invalid_register_option_is_rejected(monkeypatch):
    install_tiny_factory(monkeypatch)
    with pytest.raises(ValueError, match="0 or 4"):
        OTUFormerEncoder(
            model_name="tiny_onecls", out_dim=8, img_size=32,
            reg_tokens=2, pretrained=False,
        )


def test_explicit_register_option_rejected_on_native_register_model(monkeypatch):
    install_tiny_factory(monkeypatch)
    for value in (0, 4):
        with pytest.raises(ValueError, match="native register tokens"):
            OTUFormerEncoder(
                model_name="tiny_native_reg4", out_dim=8, img_size=32,
                reg_tokens=value, pretrained=False,
            )


def test_explicit_register_option_rejected_on_no_cls_model(monkeypatch):
    install_tiny_factory(monkeypatch)
    for value in (0, 4):
        with pytest.raises(ValueError, match="no CLS token"):
            OTUFormerEncoder(
                model_name="tiny_gap", out_dim=8, img_size=32,
                reg_tokens=value, pretrained=False,
            )


def test_native_register_model_without_cls_rejected_even_when_omitted(monkeypatch):
    install_tiny_factory(monkeypatch)
    with pytest.raises(ValueError, match="no CLS token"):
        OTUFormerEncoder(
            model_name="tiny_gap_reg4", out_dim=8, img_size=32,
            reg_tokens=None, pretrained=False,
        )


def test_vit_named_non_vision_transformer_with_registers_is_rejected(monkeypatch):
    install_tiny_factory(monkeypatch)
    with pytest.raises(ValueError, match="standard timm VisionTransformer"):
        OTUFormerEncoder(
            model_name="vit_small_patch16_dinov3.lvd1689m", out_dim=8,
            img_size=32, reg_tokens=None, pretrained=False,
        )


def test_migration_rejects_incompatible_target_weights(monkeypatch):
    install_tiny_factory(monkeypatch, register_embed_dim=64)
    with pytest.raises(ValueError, match="migrat"):
        OTUFormerEncoder(
            model_name="tiny_onecls", out_dim=8, img_size=32,
            reg_tokens=4, pretrained=True,
        )


def test_forward_excludes_every_prefix_token(monkeypatch):
    install_tiny_factory(monkeypatch)
    plain = OTUFormerEncoder(
        model_name="tiny_onecls", out_dim=8, return_patch_tokens=True,
        img_size=32, pretrained=False,
    )
    registered = OTUFormerEncoder(
        model_name="tiny_onecls", out_dim=8, return_patch_tokens=True,
        img_size=32, reg_tokens=4, pretrained=False,
    )
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        _, plain_patches = plain(x)
        _, registered_patches = registered(x)
    assert plain_patches.shape == (2, 4, 32)
    assert registered_patches.shape == (2, 4, 32)
    assert registered.backbone.num_prefix_tokens == 5


def test_pretrain_final_norm_binds_the_migrated_backbone(monkeypatch):
    install_tiny_factory(monkeypatch)
    encoder = OTUFormerEncoder(
        model_name="tiny_onecls", out_dim=8, return_patch_tokens=True,
        img_size=32, reg_tokens=4, pretrained=True,
    )
    encoder.validate_pretrain_backbone()
    assert encoder._pretrain_final_norm is encoder.backbone.norm
    assert encoder.backbone.num_prefix_tokens == 5


def test_timm_without_reg_tokens_support_raises_a_capability_error(monkeypatch):
    """An older timm must fail with a clear capability error, not a CNN error."""
    import otuformer.training.model as model_module

    def factory(model_name, **kwargs):
        if "reg_tokens" in kwargs:
            raise TypeError(
                "create_model() got an unexpected keyword argument 'reg_tokens'"
            )
        return tiny_vit(model_name)

    monkeypatch.setattr(model_module.timm, "create_model", factory)
    with pytest.raises(RuntimeError, match="upgrade timm"):
        OTUFormerEncoder(
            model_name="tiny_onecls", out_dim=8, img_size=32,
            reg_tokens=4, pretrained=False,
        )


def test_capability_guard_rejects_a_backbone_that_omits_the_registers(monkeypatch):
    """A timm that ignores reg_tokens must be caught, not silently accepted."""
    import otuformer.training.model as model_module

    def factory(model_name, **kwargs):
        requested = kwargs.get("reg_tokens")
        model = tiny_vit(
            model_name,
            pretrained=bool(kwargs.get("pretrained", False)),
            regs=0 if requested is None else int(requested),
        )
        if requested is not None:
            model.num_prefix_tokens = 1
        return model

    monkeypatch.setattr(model_module.timm, "create_model", factory)
    with pytest.raises(ValueError, match="prefix tokens"):
        OTUFormerEncoder(
            model_name="tiny_onecls", out_dim=8, img_size=32,
            reg_tokens=4, pretrained=False,
        )


def test_capability_guard_checks_the_pos_embed_forward_shape(monkeypatch):
    """The guard must probe ``_pos_embed``, not just trust constructor kwargs."""
    import otuformer.training.model as model_module

    def factory(model_name, **kwargs):
        requested = kwargs.get("reg_tokens")
        model = tiny_vit(
            model_name,
            pretrained=bool(kwargs.get("pretrained", False)),
            regs=0 if requested is None else int(requested),
        )
        if requested is not None:
            original = model._pos_embed
            model._pos_embed = lambda patches: original(patches)[:, :1]
        return model

    monkeypatch.setattr(model_module.timm, "create_model", factory)
    with pytest.raises(ValueError, match="_pos_embed"):
        OTUFormerEncoder(
            model_name="tiny_onecls", out_dim=8, img_size=32,
            reg_tokens=4, pretrained=False,
        )
