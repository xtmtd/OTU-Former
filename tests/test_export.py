from pathlib import Path

import numpy as np
import pytest
import torch

from otuformer.vision.export import export_to_onnx, load_exported_onnx


def test_export_checkpoint_loader_disables_pretrained_weights(monkeypatch, tmp_path: Path):
    import otuformer.vision.export as export_module

    seen = {}

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)

        def load_state_dict(self, *_args, **_kwargs):
            return None

        def eval(self):
            return self

    monkeypatch.setattr(export_module, "load_checkpoint", lambda _path: {"model_state_dict": {}})
    monkeypatch.setattr(export_module, "OTUFormerEncoder", FakeEncoder)
    monkeypatch.setattr(export_module.torch.onnx, "export", lambda *_args, **_kwargs: None)

    export_to_onnx(tmp_path / "model.pth", tmp_path / "encoder.onnx", imgsz=224)

    assert seen["pretrained"] is False


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


def test_export_creates_onnx_file(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    out_path = tmp_path / "encoder.onnx"
    report = export_to_onnx(
        checkpoint_path=ckpt,
        out_path=out_path,
        imgsz=224,
        opset=17,
    )
    assert out_path.exists()
    assert report["out_dim"] == 64
    assert report["model_name"] == "vit_tiny_patch16_224"


def make_legacy_checkpoint(tmp_path: Path, global_crop_size: int = 32) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=64,
        pretrained=False,
        img_size=global_crop_size,
    )
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 64},
        "args": {"global_crop_size": global_crop_size},
    }
    p = tmp_path / "legacy_ckpt.pt"
    torch.save(ckpt, p)
    return p


def test_export_constructs_model_at_recorded_checkpoint_size(monkeypatch, tmp_path):
    import otuformer.vision.export as export_module

    seen = {}

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)

        def load_state_dict(self, *_args, **_kwargs):
            return None

        def eval(self):
            return self

    monkeypatch.setattr(
        export_module,
        "load_checkpoint",
        lambda _path: {
            "model_state_dict": {},
            "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 64},
            "args": {"global_crop_size": 32},
        },
    )
    monkeypatch.setattr(export_module, "OTUFormerEncoder", FakeEncoder)
    monkeypatch.setattr(export_module.torch.onnx, "export", lambda *_a, **_k: None)

    export_module.export_to_onnx(tmp_path / "model.pth", tmp_path / "encoder.onnx")

    assert seen["img_size"] == 32


def test_export_auto_imgsz_uses_checkpoint_size(tmp_path: Path):
    ckpt = make_legacy_checkpoint(tmp_path, global_crop_size=32)
    out_path = tmp_path / "encoder32.onnx"
    report = export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=None)
    assert report["input_shape"] == [1, 3, 32, 32]


def test_export_224_checkpoint_at_384_dummy_input(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)  # 224 checkpoint
    out_path = tmp_path / "encoder384.onnx"
    report = export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=384)
    assert report["input_shape"] == [1, 3, 384, 384]


def test_onnx_output_shape(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path, out_dim=64)
    out_path = tmp_path / "encoder.onnx"
    export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)
    session = load_exported_onnx(out_path)
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {"input": dummy})
    assert outputs[0].shape == (1, 64)


def make_arcface_checkpoint(tmp_path: Path, out_dim: int = 64) -> Path:
    """A new-style fine-tune checkpoint (ArcFaceEmbeddingHead projector)."""
    from otuformer.training.model import ArcFaceEmbeddingHead, OTUFormerEncoder

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=out_dim, pretrained=False
    )
    model.projector = ArcFaceEmbeddingHead(model.backbone.num_features, out_dim)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": "vit_tiny_patch16_224",
            "out_dim": out_dim,
            "metric_embed_dim": out_dim,
            "embedding_head": "arcface_mlp_512",
        },
    }
    p = tmp_path / "arcface.pt"
    torch.save(ckpt, p)
    return p


def test_export_handles_arcface_embedding_head_checkpoint(tmp_path: Path):
    ckpt = make_arcface_checkpoint(tmp_path)
    out_path = tmp_path / "arcface.onnx"

    report = export_to_onnx(
        checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17
    )

    assert report["out_dim"] == 64
    assert out_path.exists()


def make_ref_script_sft_checkpoint(tmp_path: Path, model_name: str, out_dim: int = 64) -> Path:
    """Ref-script SFT layout: no config, no args, backbone fallback required."""
    from otuformer.training.model import ArcFaceEmbeddingHead, OTUFormerEncoder

    model = OTUFormerEncoder(model_name=model_name, out_dim=out_dim, pretrained=False)
    model.projector = ArcFaceEmbeddingHead(model.backbone.num_features, out_dim)
    p = tmp_path / "arcface_epoch_0020.pth"
    torch.save(
        {"model": model.state_dict(), "loss_func": {"W": torch.zeros(out_dim, 2)}},
        p,
    )
    return p


def test_export_accepts_an_explicit_model_name_for_a_ref_script_checkpoint(tmp_path: Path):
    ckpt = make_ref_script_sft_checkpoint(tmp_path, "vit_tiny_patch16_224")
    out_path = tmp_path / "refscript.onnx"

    report = export_to_onnx(
        checkpoint_path=ckpt,
        out_path=out_path,
        imgsz=224,
        opset=17,
        model_name="vit_tiny_patch16_224",
    )

    assert report["model_name"] == "vit_tiny_patch16_224"
    assert out_path.exists()


def test_export_names_the_backbone_mismatch_for_an_unresolvable_checkpoint(tmp_path: Path):
    """The ref script's own default backbone is vit_small, which the tiny fallback cannot fit."""
    ckpt = make_ref_script_sft_checkpoint(tmp_path, "vit_small_patch16_224")

    with pytest.raises(ValueError, match="--model-name"):
        export_to_onnx(checkpoint_path=ckpt, out_path=tmp_path / "bad.onnx", imgsz=224)
