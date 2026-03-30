import torch

from otuformer.training.model import ArcFaceHead, OTUFormerEncoder


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
