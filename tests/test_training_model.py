import pytest
import torch

from otuformer.training.loss import ArcFaceLoss
from otuformer.training.model import ArcFaceEmbeddingHead, ArcFaceHead, OTUFormerEncoder
from otuformer.training.trainer import (
    _build_finetune_optimizer,
    _select_finetune_embedding_head,
    _use_split_finetune_optimizer,
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
    assert _select_finetune_embedding_head({"model_state_dict": {}}) == (
        "arcface_mlp_512"
    )
    assert _select_finetune_embedding_head({"loss_state_dict": {}}) == (
        "projection_mlp_2048"
    )
    with pytest.raises(ValueError, match="legacy fine-tune checkpoint format"):
        _select_finetune_embedding_head({"loss_func": {}})
    assert _select_finetune_embedding_head(
        {"config": {"embedding_head": "arcface_mlp_512"}}
    ) == "arcface_mlp_512"


def test_resume_reuses_saved_finetune_optimizer_group_count():
    historical_sft = {
        "config": {"embedding_head": "projection_mlp_2048"},
        "optimizer": {"param_groups": [{}, {}]},
    }
    assert _use_split_finetune_optimizer(historical_sft, resume=True)
    assert not _use_split_finetune_optimizer(
        {"config": {"embedding_head": "projection_mlp_2048"},
         "optimizer": {"param_groups": [{}]}},
        resume=True,
    )


def test_finetune_optimizer_uses_distinct_backbone_and_head_learning_rates():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.projector = torch.nn.Linear(4, 2)

    optimizer = _build_finetune_optimizer(
        Model(), ArcFaceLoss(embed_dim=2, num_classes=3), 3e-5, 1e-4, 1e-4
    )

    assert [group["lr"] for group in optimizer.param_groups] == [3e-5, 1e-4]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [1e-4, 1e-4]


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
