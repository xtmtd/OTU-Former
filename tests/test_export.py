from pathlib import Path

import numpy as np
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


def test_onnx_output_shape(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path, out_dim=64)
    out_path = tmp_path / "encoder.onnx"
    export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)
    session = load_exported_onnx(out_path)
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {"input": dummy})
    assert outputs[0].shape == (1, 64)
