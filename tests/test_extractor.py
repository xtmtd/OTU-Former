from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from otuformer.embedding.extractor import (
    GatedAttentionPooling,
    detect_batch_mode,
    extract_embeddings,
    _iter_trainable_params,
    _load_model,
)
from otuformer.utils.size import resolve_onnx_input_size


def make_checkpoint(tmp_path: Path, out_dim: int = 64) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=out_dim)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim},
    }
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, p)
    return p


def make_images(directory: Path, n: int = 3) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (224, 224), color=(i * 80, 0, 0)).save(
            directory / f"img_{i}.jpg"
        )


def make_legacy_checkpoint(
    tmp_path: Path, out_dim: int = 64, global_crop_size: int = 32
) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=out_dim,
        pretrained=False,
        img_size=global_crop_size,
    )
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim},
        "args": {"global_crop_size": global_crop_size},
    }
    p = tmp_path / "legacy_ckpt.pt"
    torch.save(ckpt, p)
    return p


def test_load_model_keeps_historical_projection_head_metadata(monkeypatch, tmp_path):
    import otuformer.embedding.extractor as extractor_module

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.backbone = type("Backbone", (), {"num_features": 192})()
            self.projector = torch.nn.Identity()

    monkeypatch.setattr(extractor_module, "OTUFormerEncoder", FakeEncoder)
    # The shared read-only loader owns the construction site now.
    import otuformer.utils.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "OTUFormerEncoder", FakeEncoder)
    checkpoint = tmp_path / "historical-sft.pth"
    torch.save(
        {
            "model_state_dict": {},
            "config": {"out_dim": 64, "embedding_head": "projection_mlp_2048"},
        },
        checkpoint,
    )
    monkeypatch.setattr(extractor_module, "load_checkpoint", lambda _: torch.load(checkpoint))

    model, _ = extractor_module._load_model(
        checkpoint, "vit_tiny_patch16_224", torch.device("cpu")
    )

    assert isinstance(model.projector, torch.nn.Identity)


def test_load_model_ignores_parameter_free_supcon_loss_state(tmp_path):
    """A v0.8.0 SupCon checkpoint has an empty loss state and no classifier."""
    from otuformer.training.model import ArcFaceEmbeddingHead, OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=16, pretrained=False, img_size=32
    )
    encoder.projector = ArcFaceEmbeddingHead(encoder.backbone.num_features, 16)
    checkpoint = tmp_path / "supcon.pth"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "loss_state_dict": {},
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 16,
                "metric_embed_dim": 16,
                "image_size": 32,
                "embedding_head": "arcface_mlp_512",
                "loss": "supcon",
                "supcon_temperature": 0.07,
            },
        },
        checkpoint,
    )

    model, size = _load_model(
        checkpoint, "vit_tiny_patch16_224", torch.device("cpu")
    )

    assert isinstance(model.projector, ArcFaceEmbeddingHead)
    assert size == 32
    # Default extraction vector stays the raw CLS feature.
    assert model.backbone.num_features == encoder.backbone.num_features


def test_load_model_rebuilds_arcface_embedding_head(monkeypatch, tmp_path):
    import otuformer.embedding.extractor as extractor_module
    from otuformer.training.model import ArcFaceEmbeddingHead

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.backbone = type("Backbone", (), {"num_features": 192})()
            self.projector = torch.nn.Identity()

    monkeypatch.setattr(extractor_module, "OTUFormerEncoder", FakeEncoder)
    # The shared read-only loader owns the construction site now.
    import otuformer.utils.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "OTUFormerEncoder", FakeEncoder)
    source = ArcFaceEmbeddingHead(192, 64)
    checkpoint = tmp_path / "arcface.pth"
    torch.save(
        {
            "model_state_dict": {"projector.net.0.weight": source.net[0].weight.detach()},
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 64,
                "embedding_head": "arcface_mlp_512",
            },
        },
        checkpoint,
    )
    monkeypatch.setattr(extractor_module, "load_checkpoint", lambda _: torch.load(checkpoint))

    model, _ = extractor_module._load_model(
        checkpoint, "vit_tiny_patch16_224", torch.device("cpu")
    )

    assert isinstance(model.projector, ArcFaceEmbeddingHead)
    assert torch.equal(model.projector.net[0].weight, source.net[0].weight)


def test_load_model_constructs_at_recorded_checkpoint_size(monkeypatch, tmp_path):
    import otuformer.embedding.extractor as extractor_module

    seen = {}

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)
            self.backbone = torch.nn.Identity()

        def load_state_dict(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        extractor_module,
        "load_checkpoint",
        lambda _path: {"model_state_dict": {}, "args": {"global_crop_size": 32}},
    )
    monkeypatch.setattr(extractor_module, "OTUFormerEncoder", FakeEncoder)
    # The shared read-only loader owns the construction site now.
    import otuformer.utils.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "OTUFormerEncoder", FakeEncoder)

    model, size = extractor_module._load_model(
        tmp_path / "model.pth", "vit_tiny_patch16_224", torch.device("cpu")
    )

    assert seen["img_size"] == 32
    assert size == 32


def test_extract_pos_embed_matches_legacy_checkpoint_size(tmp_path):
    ckpt = make_legacy_checkpoint(tmp_path, global_crop_size=32)
    model, size = _load_model(ckpt, "vit_tiny_patch16_224", torch.device("cpu"))
    assert size == 32
    # (32 / 16) ** 2 + 1 CLS token = 5 tokens
    assert model.backbone.pos_embed.shape[1] == 5


def test_extract_224_checkpoint_at_384_uses_dynamic_interpolation(tmp_path):
    ckpt = make_checkpoint(tmp_path)  # 224 checkpoint, no recorded size
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)

    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        extract_size=384,
        token_mode="cls",
    )
    assert len(out) == 2


def test_extract_auto_size_resolves_to_checkpoint_size(tmp_path):
    ckpt = make_legacy_checkpoint(tmp_path, global_crop_size=32)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        token_mode="cls",
        extract_size=None,
    )
    assert len(out) == 2


def test_extract_size_error_names_resolved_model_not_cli_arg(tmp_path):
    """Size-validation errors must name the checkpoint-resolved model."""
    ckpt = make_checkpoint(tmp_path)  # config model_name = vit_tiny_patch16_224
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    with pytest.raises(ValueError, match="for vit_tiny_patch16_224"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            model_name="some_unrelated_backbone",
            extract_size=518,
            token_mode="cls",
        )


def _make_onnx_graph_proto(dims):
    import onnx
    from onnx import TensorProto, helper

    value_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, dims)
    node = helper.make_node("Identity", ["input"], ["out"])
    graph = helper.make_graph(
        [node], "g", [value_info], [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])]
    )
    return onnx.ModelProto(ir_version=8, graph=graph)


def test_resolve_onnx_input_size_reads_static_square_nchw(monkeypatch, tmp_path):
    import onnx

    monkeypatch.setattr(
        onnx, "load", lambda p: _make_onnx_graph_proto([1, 3, 224, 224])
    )
    assert resolve_onnx_input_size(tmp_path / "m.onnx") == 224

    monkeypatch.setattr(
        onnx, "load", lambda p: _make_onnx_graph_proto([1, 3, 224, 256])
    )
    with pytest.raises(ValueError, match="square"):
        resolve_onnx_input_size(tmp_path / "m.onnx")

    monkeypatch.setattr(
        onnx, "load", lambda p: _make_onnx_graph_proto([1, 1, 224, 224])
    )
    with pytest.raises(ValueError, match="NCHW"):
        resolve_onnx_input_size(tmp_path / "m.onnx")

    monkeypatch.setattr(
        onnx, "load", lambda p: _make_onnx_graph_proto([1, 3, "h", 224])
    )
    with pytest.raises(ValueError, match="static"):
        resolve_onnx_input_size(tmp_path / "m.onnx")


def test_extract_onnx_auto_size_reads_graph_and_explicit_must_match(tmp_path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    onnx_path = make_onnx(tmp_path)  # exported at 224 (static input)

    auto_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        token_mode="cls",
        onnx_path=onnx_path,
        extract_size=None,
    )
    assert len(auto_out) == 2

    match_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        token_mode="cls",
        onnx_path=onnx_path,
        extract_size=224,
    )
    assert len(match_out) == 2

    with pytest.raises(ValueError, match="does not match ONNX input size"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            token_mode="cls",
            onnx_path=onnx_path,
            extract_size=384,
        )


def test_extractor_checkpoint_loader_disables_pretrained_weights(monkeypatch, tmp_path):
    import otuformer.embedding.extractor as extractor_module

    seen = {}

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)
            self.backbone = torch.nn.Identity()

        def load_state_dict(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(extractor_module, "load_checkpoint", lambda _path: {"model_state_dict": {}})
    monkeypatch.setattr(extractor_module, "OTUFormerEncoder", FakeEncoder)
    # The shared read-only loader owns the construction site now.
    import otuformer.utils.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "OTUFormerEncoder", FakeEncoder)

    _load_model(tmp_path / "model.pth", "vit_tiny_patch16_224", torch.device("cpu"))

    assert seen["pretrained"] is False


def test_extract_single_dir(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir)
    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
    )
    assert isinstance(out, pd.DataFrame)
    assert "id" in out.columns
    assert "sample" not in out.columns
    assert len(out) == 3
    assert out.shape[1] > 2


def test_detect_batch_mode(tmp_path: Path):
    img_dir = tmp_path / "single"
    make_images(img_dir)
    assert detect_batch_mode(img_dir) is False

    parent = tmp_path / "multi"
    make_images(parent / "site_a")
    make_images(parent / "site_b")
    assert detect_batch_mode(parent) is True


def test_extract_batch_mode(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    parent = tmp_path / "multi"
    make_images(parent / "site_a", n=2)
    make_images(parent / "site_b", n=3)
    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=parent,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
    )
    assert "sample" in out.columns
    assert len(out) == 5
    assert set(out["sample"].unique()) == {"site_a", "site_b"}


def test_extract_patch_topk_mode_changes_embedding_shape(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)

    cls_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        token_mode="cls",
    )
    topk_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        token_mode="patch-topk",
        topk_patches=4,
    )

    cls_dim = len([c for c in cls_out.columns if c.startswith("dim_")])
    topk_dim = len([c for c in topk_out.columns if c.startswith("dim_")])
    assert topk_dim != cls_dim
    assert topk_dim <= 512


def test_extract_attention_pool_requires_training_csv_without_finetuned_pool(
    tmp_path: Path,
):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)

    with pytest.raises(ValueError, match="label-csv"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            token_mode="attention-pool",
        )


def test_extract_attention_pool_rejects_image_only_training_csv(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    csv_path = tmp_path / "images.csv"
    pd.DataFrame({"image": ["img_0.jpg", "img_1.jpg"]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="image.*label"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            token_mode="attention-pool",
            attention_train_csv=csv_path,
        )


def test_extract_attention_pool_validates_csv_before_start_message(
    tmp_path: Path, capsys
):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    csv_path = tmp_path / "images.csv"
    pd.DataFrame({"image": ["img_0.jpg", "img_1.jpg"]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="image.*label"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            token_mode="attention-pool",
            attention_train_csv=csv_path,
        )

    assert "Starting query finetuning" not in capsys.readouterr().out


# --- v0.10.0 patch boundaries and attention-pool policy -----------------------


def _prefix_backbone(prefix_tokens, dim=16, registers=None):
    backbone = torch.nn.Identity()
    backbone.num_prefix_tokens = prefix_tokens
    backbone.num_features = dim
    backbone.num_reg_tokens = (
        max(0, prefix_tokens - 1) if registers is None else registers
    )
    backbone.reg_token = None
    return backbone


def test_patch_features_excludes_registers_and_keeps_the_first_patch_when_cls_free():
    import otuformer.embedding.extractor as extractor_module

    feats = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)

    registered = extractor_module._patch_features(_prefix_backbone(5), feats)
    assert torch.equal(registered, feats[:, 5:])

    cls_only = extractor_module._patch_features(_prefix_backbone(1), feats)
    assert torch.equal(cls_only, feats[:, 1:])

    # A CLS-free/no-register backbone keeps its first patch.
    cls_free = extractor_module._patch_features(_prefix_backbone(0), feats)
    assert cls_free.shape == (2, 7, 3)
    assert torch.equal(cls_free[:, 0], feats[:, 0])


def test_pool_policy_and_tagged_sibling_filename():
    import otuformer.embedding.extractor as extractor_module

    model = type("Model", (), {"backbone": _prefix_backbone(5, 32, registers=4)})()
    policy = extractor_module._attention_pool_policy(model, "gated")

    assert policy == {
        "num_prefix_tokens": 5,
        "register_tokens": 4,
        "prefix_excluded": True,
        "attention_pooling_type": "gated",
    }
    assert extractor_module._tagged_pooling_checkpoint_path(
        Path("runs/pretrain/SSL_latest.gated_pooling.pth"), policy
    ) == Path("runs/pretrain/SSL_latest.prefix5-reg4-gated_pooling.pth")


def test_pool_candidates_search_the_tagged_sibling_first():
    import otuformer.embedding.extractor as extractor_module

    policy = {
        "num_prefix_tokens": 5,
        "register_tokens": 4,
        "prefix_excluded": True,
        "attention_pooling_type": "lightweight",
    }
    default = Path("d/SSL.pth")

    assert extractor_module._attention_pooling_checkpoint_candidates(
        default, "lightweight", None, policy=policy
    ) == [
        Path("d/SSL.prefix5-reg4-lightweight_pooling.pth"),
        Path("d/SSL.lightweight_pooling.pth"),
    ]

    requested = Path("d/stale_pool.pth")
    assert extractor_module._attention_pooling_checkpoint_candidates(
        default, "lightweight", requested, policy=policy
    ) == [
        Path("d/stale_pool.prefix5-reg4-lightweight_pooling.pth"),
        Path("d/stale_pool.pth"),
        Path("d/SSL.prefix5-reg4-lightweight_pooling.pth"),
        Path("d/SSL.lightweight_pooling.pth"),
    ]

    # Without a policy the historical path order is unchanged.
    assert extractor_module._attention_pooling_checkpoint_candidates(
        default, "gated", None
    ) == [Path("d/SSL.gated_pooling.pth")]
    assert extractor_module._attention_pooling_checkpoint_candidates(
        default, "gated", requested
    ) == [requested, Path("d/SSL.gated_pooling.pth")]


def test_pool_policy_compatibility_table():
    import otuformer.embedding.extractor as extractor_module

    policy = {
        "num_prefix_tokens": 5,
        "register_tokens": 4,
        "prefix_excluded": True,
        "attention_pooling_type": "lightweight",
    }
    assert extractor_module._pool_policy_compatible({"args": dict(policy)}, policy)[0]

    for key, value in (
        ("num_prefix_tokens", 1),
        ("register_tokens", 0),
        ("prefix_excluded", False),
        ("attention_pooling_type", "gated"),
    ):
        broken = dict(policy)
        broken[key] = value
        compatible, reason = extractor_module._pool_policy_compatible(
            {"args": broken}, policy
        )
        assert not compatible and key in reason

    # A policy-less pool is reusable only for one prefix and zero registers.
    legacy_policy = dict(policy, num_prefix_tokens=1, register_tokens=0)
    assert extractor_module._pool_policy_compatible({"args": {}}, legacy_policy)[0]
    assert extractor_module._pool_policy_compatible({"teacher": {}}, legacy_policy)[0]
    assert not extractor_module._pool_policy_compatible({"args": {}}, policy)[0]
    assert not extractor_module._pool_policy_compatible({}, policy)[0]


def test_partially_recorded_pool_policy_is_never_reused():
    import otuformer.embedding.extractor as extractor_module

    policy = {
        "num_prefix_tokens": 1,
        "register_tokens": 0,
        "prefix_excluded": True,
        "attention_pooling_type": "lightweight",
    }
    # A pre-policy file whose recorded pooling type differs must not be reused.
    ok, reason = extractor_module._pool_policy_compatible(
        {"args": {"attention_pooling_type": "gated"}}, policy
    )
    assert not ok and "attention_pooling_type" in reason
    # A pre-policy file with the matching type stays readable.
    assert extractor_module._pool_policy_compatible(
        {"args": {"attention_pooling_type": "lightweight"}}, policy
    )[0]
    # A half-written patch policy is rejected even when the fields it has match.
    ok, reason = extractor_module._pool_policy_compatible(
        {"args": {"num_prefix_tokens": 1, "register_tokens": 0}}, policy
    )
    assert not ok and "prefix_excluded" in reason
    # A complete patch trio without the pooling type cannot be verified either,
    # including for a registered backbone (prefix 5 / 4 registers).
    registered_policy = {
        "num_prefix_tokens": 5,
        "register_tokens": 4,
        "prefix_excluded": True,
        "attention_pooling_type": "gated",
    }
    ok, reason = extractor_module._pool_policy_compatible(
        {
            "args": {
                "num_prefix_tokens": 5,
                "register_tokens": 4,
                "prefix_excluded": True,
            }
        },
        registered_policy,
    )
    assert not ok and "attention_pooling_type" in reason


def test_attention_pool_tagged_checkpoint_is_written_and_discovered(tmp_path: Path):
    import otuformer.embedding.extractor as extractor_module

    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {"image": ["img_0.jpg", "img_1.jpg"], "label": [0, 1]}
    ).to_csv(csv_path, index=False)

    extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=1,
        token_mode="attention-pool",
        attention_train_csv=csv_path,
        attention_pooling_epochs=1,
        seed=1,
    )

    legacy = extractor_module._attention_pooling_checkpoint_path(ckpt, "lightweight")
    policy = {
        "num_prefix_tokens": 1,
        "register_tokens": 0,
        "prefix_excluded": True,
        "attention_pooling_type": "lightweight",
    }
    tagged = extractor_module._tagged_pooling_checkpoint_path(legacy, policy)
    assert tagged.exists()
    assert not legacy.exists()
    payload = torch.load(tagged, map_location="cpu", weights_only=False)
    assert payload["args"]["num_prefix_tokens"] == 1
    assert payload["args"]["register_tokens"] == 0
    assert payload["args"]["prefix_excluded"] is True

    # A later extraction with no training CSV must discover the tagged sibling.
    reloaded = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=1,
        token_mode="attention-pool",
    )
    assert len(reloaded) > 0


def test_tagged_sibling_with_a_mismatched_policy_is_rejected_and_preserved(
    tmp_path: Path,
):
    import otuformer.embedding.extractor as extractor_module

    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    policy = {
        "num_prefix_tokens": 1,
        "register_tokens": 0,
        "prefix_excluded": True,
        "attention_pooling_type": "lightweight",
    }
    legacy = extractor_module._attention_pooling_checkpoint_path(ckpt, "lightweight")
    tagged = extractor_module._tagged_pooling_checkpoint_path(legacy, policy)
    pool = extractor_module._build_attention_pooling(
        "lightweight", ckpt and 192
    )
    torch.save(
        {
            "attention_pool_state_dict": pool.state_dict(),
            "args": dict(policy, num_prefix_tokens=5),
        },
        tagged,
    )
    before = tagged.read_bytes()

    with pytest.raises(ValueError, match="different patch set"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=1,
            token_mode="attention-pool",
        )

    assert tagged.read_bytes() == before


def test_incompatible_tagged_sibling_is_fatal_even_with_a_compatible_legacy_file(
    tmp_path: Path,
):
    """The plan requires failing with the sibling's path, not falling through.

    A compatible policy-less legacy pool is present, so the only reason to raise
    is the policy-tagged sibling itself: choosing the legacy file instead would
    silently answer the same request from a different artifact.
    """
    import otuformer.embedding.extractor as extractor_module

    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    policy = {
        "num_prefix_tokens": 1,
        "register_tokens": 0,
        "prefix_excluded": True,
        "attention_pooling_type": "lightweight",
    }
    legacy = extractor_module._attention_pooling_checkpoint_path(ckpt, "lightweight")
    tagged = extractor_module._tagged_pooling_checkpoint_path(legacy, policy)
    pool = extractor_module._build_attention_pooling("lightweight", 192)
    torch.save({"attention_pool_state_dict": pool.state_dict()}, legacy)
    torch.save(
        {
            "attention_pool_state_dict": pool.state_dict(),
            "args": dict(policy, num_prefix_tokens=5),
        },
        tagged,
    )

    with pytest.raises(ValueError, match="prefix1-reg0-lightweight_pooling"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=1,
            token_mode="attention-pool",
        )


def test_legacy_pool_without_policy_stays_readable(tmp_path: Path):
    import otuformer.embedding.extractor as extractor_module

    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    legacy = extractor_module._attention_pooling_checkpoint_path(ckpt, "lightweight")
    pool = extractor_module._build_attention_pooling("lightweight", 192)
    torch.save({"attention_pool_state_dict": pool.state_dict()}, legacy)

    result = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=1,
        token_mode="attention-pool",
    )

    assert len(result) > 0


def test_extract_csv_preserves_csv_order(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=3)

    csv_path = tmp_path / "subset.csv"
    pd.DataFrame({"image": ["img_2.jpg", "img_0.jpg"]}).to_csv(csv_path, index=False)

    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        token_mode="cls",
        extract_csv=csv_path,
    )
    assert out["id"].tolist() == ["img_2.jpg", "img_0.jpg"]


def test_extract_csv_infers_sample_for_subdirs(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    parent = tmp_path / "multi"
    make_images(parent / "site_a", n=1)
    make_images(parent / "site_b", n=1)

    csv_path = tmp_path / "subset.csv"
    pd.DataFrame(
        {
            "image": ["site_a/img_0.jpg", "site_b/img_0.jpg"],
            "label": ["x", "y"],
        }
    ).to_csv(csv_path, index=False)

    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=parent,
        device="cpu",
        batch_size=2,
        token_mode="cls",
        extract_csv=csv_path,
    )
    assert "sample" in out.columns
    assert out["sample"].tolist() == ["site_a", "site_b"]


def test_gated_attention_has_trainable_params():
    pool = GatedAttentionPooling(dim=192)
    trainable = list(_iter_trainable_params(pool))
    assert len(trainable) > 0


def make_onnx(tmp_path: Path, out_dim: int = 64) -> Path:
    from otuformer.vision.export import export_to_onnx

    ckpt = make_checkpoint(tmp_path, out_dim=out_dim)
    onnx_path = tmp_path / "encoder.onnx"
    export_to_onnx(checkpoint_path=ckpt, out_path=onnx_path, imgsz=224, opset=18)
    return onnx_path


def test_extract_with_onnx_produces_same_shape(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=3)

    torch_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
        use_projector_output=True,
    )

    onnx_path = make_onnx(tmp_path)
    onnx_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
        onnx_path=onnx_path,
    )

    assert isinstance(onnx_out, pd.DataFrame)
    assert "id" in onnx_out.columns
    assert len(onnx_out) == len(torch_out)
    dim_cols = [c for c in onnx_out.columns if c.startswith("dim_")]
    assert len(dim_cols) == len([c for c in torch_out.columns if c.startswith("dim_")])


def test_extract_with_onnx_matches_pytorch_values(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=3)

    onnx_path = make_onnx(tmp_path)

    torch_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
        use_projector_output=True,
    )

    onnx_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
        onnx_path=onnx_path,
    )

    dim_cols = [c for c in torch_out.columns if c.startswith("dim_")]
    torch_vals = torch_out[dim_cols].values
    onnx_vals = onnx_out[dim_cols].values

    import numpy as np

    np.testing.assert_allclose(torch_vals, onnx_vals, rtol=1e-4, atol=1e-4)


def test_extract_onnx_rejects_patch_topk(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    onnx_path = make_onnx(tmp_path)

    with pytest.raises(ValueError, match="(?i)onnx"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            token_mode="patch-topk",
            onnx_path=onnx_path,
        )


def test_load_model_reads_ref_script_arcface_checkpoint(tmp_path: Path):
    """Ref-script SFT layout: 'model' weights, no config, projector.<i> names."""
    from otuformer.training.model import ArcFaceEmbeddingHead, OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=64, pretrained=False
    )
    encoder.projector = ArcFaceEmbeddingHead(encoder.backbone.num_features, 64)
    expected = encoder.projector.net[0].weight.detach().clone()

    state = dict(encoder.state_dict())
    for i in (0, 2):
        state[f"projector.{i}.weight"] = state.pop(f"projector.net.{i}.weight")
        state[f"projector.{i}.bias"] = state.pop(f"projector.net.{i}.bias")

    checkpoint = tmp_path / "arcface_epoch_0020.pth"
    torch.save(
        {"epoch": 0, "model": state, "loss_func": {"W": torch.zeros(64, 2)}},
        checkpoint,
    )

    model, size = _load_model(checkpoint, "vit_tiny_patch16_224", torch.device("cpu"))

    assert isinstance(model.projector, ArcFaceEmbeddingHead)
    assert torch.equal(model.projector.net[0].weight, expected)
    assert size == 224
    # ``--use-student`` has no effect on a checkpoint that stores no student.
    student_model, _ = _load_model(
        checkpoint, "vit_tiny_patch16_224", torch.device("cpu"), use_student=True
    )
    assert torch.equal(student_model.projector.net[0].weight, expected)


def test_load_model_reads_ref_script_ssl_checkpoint_metadata(tmp_path: Path):
    """Ref-script SSL layout: 'teacher' weights and an ``args`` dict instead of config."""
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=48, pretrained=False, img_size=224
    )
    checkpoint = tmp_path / "SSL_epoch_0020.pth"
    torch.save(
        {
            "teacher": model.state_dict(),
            "student": model.state_dict(),
            "args": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 48,
                "global_crop_size": 224,
            },
        },
        checkpoint,
    )

    # A wrong CLI default must not win over the checkpoint's recorded model name.
    loaded, size = _load_model(checkpoint, "vit_small_patch16_224", torch.device("cpu"))

    assert loaded.model_name == "vit_tiny_patch16_224"
    assert loaded.backbone.num_features == model.backbone.num_features
    assert size == 224
    assert torch.equal(loaded.projector.net[0].weight, model.projector.net[0].weight)


def test_pseudo_payload_fields_do_not_affect_readonly_consumers(tmp_path):
    from otuformer.embedding.extractor import _load_model
    from otuformer.training.model import ArcFaceEmbeddingHead, OTUFormerEncoder
    from otuformer.utils.device import resolve_device

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False, img_size=32
    )
    encoder.projector = ArcFaceEmbeddingHead(encoder.backbone.num_features, 8)
    checkpoint = tmp_path / "ft2.pth"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 8,
                "metric_embed_dim": 8,
                "embedding_head": "arcface_mlp_512",
                "image_size": 32,
                "pseudo_round": 1,
            },
            "accepted_pseudo_rows": [{"image": "a.jpg", "label": "A"}],
            "pseudo_source_checkpoint_sha256": "deadbeef",
        },
        checkpoint,
    )

    model, size = _load_model(checkpoint, "vit_tiny_patch16_224", resolve_device("cpu"))

    assert size == 32
    assert model.projector is not None
