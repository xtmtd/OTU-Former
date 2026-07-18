import pytest
import torch

from otuformer.training.model import ArcFaceHead, OTUFormerEncoder


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
