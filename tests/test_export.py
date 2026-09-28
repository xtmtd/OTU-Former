from pathlib import Path

import numpy as np
import pytest
import torch

from otuformer.vision.export import export_to_onnx, load_exported_onnx


def test_export_checkpoint_loader_disables_pretrained_weights(monkeypatch, tmp_path: Path):
    import otuformer.utils.checkpoint as checkpoint_module
    import otuformer.vision.export as export_module

    seen = {}

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)
            self.backbone = torch.nn.Identity()

        def load_state_dict(self, *_args, **_kwargs):
            return None

        def eval(self):
            return self

    monkeypatch.setattr(export_module, "load_checkpoint", lambda _path: {"model_state_dict": {}})
    # The shared read-only loader owns the construction site now.
    monkeypatch.setattr(checkpoint_module, "OTUFormerEncoder", FakeEncoder)
    monkeypatch.setattr(export_module.torch.onnx, "export", lambda *_args, **_kwargs: None)
    # Both the graph and its publication are stubbed; this test only asserts
    # the construction kwargs.
    monkeypatch.setattr(export_module.os, "replace", lambda *_a, **_k: None)
    monkeypatch.setattr(
        export_module,
        "_validate_numeric_agreement",
        lambda *_a, **_k: {
            "validated": False,
            "validation_status": "skipped",
            "max_abs_diff": None,
            "output_shape": [1, 64],
            "validation_note": "stubbed",
        },
    )

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
    import otuformer.utils.checkpoint as checkpoint_module
    import otuformer.vision.export as export_module

    seen = {}

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)
            self.backbone = torch.nn.Identity()

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
    monkeypatch.setattr(checkpoint_module, "OTUFormerEncoder", FakeEncoder)
    monkeypatch.setattr(export_module.torch.onnx, "export", lambda *_a, **_k: None)
    monkeypatch.setattr(export_module.os, "replace", lambda *_a, **_k: None)
    monkeypatch.setattr(
        export_module,
        "_validate_numeric_agreement",
        lambda *_a, **_k: {
            "validated": False,
            "validation_status": "skipped",
            "max_abs_diff": None,
            "output_shape": [1, 64],
            "validation_note": "stubbed",
        },
    )

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


# --- v0.10.0 numeric validation ------------------------------------------------


def test_export_reports_numeric_validation_status(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path, out_dim=64)
    out_path = tmp_path / "encoder.onnx"

    report = export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)

    assert report["validated"] is True
    assert report["validation_status"] == "passed"
    assert report["atol"] == 1e-4
    assert report["rtol"] == 1e-3
    assert report["register_tokens"] == 0
    assert report["max_abs_diff"] is not None and report["max_abs_diff"] < 1e-4
    assert report["output_shape"] == [1, 64]
    assert report["onnx_path"] == str(out_path)
    assert not (tmp_path / "encoder.part.onnx").exists()


def test_export_skips_numeric_validation_without_onnxruntime(tmp_path: Path, monkeypatch):
    import sys

    ckpt = make_checkpoint(tmp_path, out_dim=64)
    out_path = tmp_path / "encoder.onnx"
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    report = export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224)

    assert out_path.exists()
    assert report["validated"] is False
    assert report["validation_status"] == "skipped"
    assert report["max_abs_diff"] is None
    assert "ONNX Runtime" in report["validation_note"]


def test_export_numeric_mismatch_preserves_the_published_outputs(
    tmp_path: Path, monkeypatch
):
    import json

    import onnxruntime as ort

    import otuformer.vision.export as export_module

    ckpt = make_checkpoint(tmp_path, out_dim=64)
    out_path = tmp_path / "encoder.onnx"
    report_path = tmp_path / "export_report.json"
    export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)
    before_model = out_path.read_bytes()
    before_report = report_path.read_bytes()

    class PerturbedSession:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, *_args, **_kwargs):
            return [np.full((1, 64), 5.0, dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", PerturbedSession)

    with pytest.raises(ValueError, match="disagrees"):
        export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)

    assert out_path.read_bytes() == before_model
    assert report_path.read_bytes() == before_report
    assert not (tmp_path / "encoder.part.onnx").exists()
    assert not (tmp_path / "encoder.part.onnx.data").exists()
    assert json.loads(report_path.read_text())["validation_status"] == "passed"
    assert export_module.ATOL == 1e-4 and export_module.RTOL == 1e-3


def test_export_publication_failure_leaves_an_invalidated_report(
    tmp_path: Path, monkeypatch
):
    import json
    import os

    import otuformer.vision.export as export_module

    ckpt = make_checkpoint(tmp_path, out_dim=64)
    out_path = tmp_path / "encoder.onnx"
    report_path = tmp_path / "export_report.json"
    export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)
    before_model = out_path.read_bytes()

    real_replace = os.replace

    def selective_replace(src, dst, *args, **kwargs):
        if str(src).endswith("encoder.part.onnx"):
            raise OSError("publication failed")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(export_module.os, "replace", selective_replace)

    with pytest.raises(OSError, match="publication failed"):
        export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)

    assert out_path.read_bytes() == before_model
    report = json.loads(report_path.read_text())
    assert report["validated"] is False
    assert report["validation_status"] == "invalidated"
    assert report["onnx_path"] == str(out_path)
    # A failed publication must not leave this attempt's temporary graph behind.
    assert not (tmp_path / "encoder.part.onnx").exists()
    assert not (tmp_path / "encoder.part.onnx.data").exists()


def test_export_reconstructs_a_registered_checkpoint(tmp_path: Path, monkeypatch):
    import timm as timm_module
    from timm.models.vision_transformer import VisionTransformer

    import otuformer.training.model as model_module

    def factory(_model_name, **kwargs):
        requested = kwargs.get("reg_tokens")
        return VisionTransformer(
            img_size=32, patch_size=16, in_chans=3, embed_dim=16, depth=2,
            num_heads=2, num_classes=0, global_pool="",
            reg_tokens=0 if requested is None else int(requested),
            dynamic_img_size=True,
        )

    monkeypatch.setattr(timm_module, "create_model", factory)
    model = model_module.OTUFormerEncoder(
        model_name="tiny-vit", out_dim=8, pretrained=False, reg_tokens=4, img_size=32
    )
    ckpt_path = tmp_path / "registered.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": "tiny-vit",
                "out_dim": 8,
                "image_size": 32,
                "register_tokens": 4,
            },
        },
        ckpt_path,
    )
    out_path = tmp_path / "registered.onnx"

    report = export_to_onnx(
        checkpoint_path=ckpt_path, out_path=out_path, imgsz=32, opset=17
    )

    assert report["register_tokens"] == 4
    assert report["output_shape"] == [1, 8]
    assert report["validation_status"] == "passed"


def test_export_rejects_a_noncanonical_register_key(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path, out_dim=64)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload["model_state_dict"] = {
        key: value for key, value in payload["model_state_dict"].items()
    }
    payload["model_state_dict"]["_orig_mod.backbone.reg_token"] = torch.zeros(1, 4, 192)
    torch.save(payload, ckpt)

    with pytest.raises(ValueError, match="backbone.reg_token"):
        export_to_onnx(checkpoint_path=ckpt, out_path=tmp_path / "bad.onnx", imgsz=224)


def test_export_raises_when_the_installed_runtime_fails(tmp_path: Path, monkeypatch):
    import onnxruntime as ort

    ckpt = make_checkpoint(tmp_path, out_dim=64)
    out_path = tmp_path / "encoder.onnx"
    report_path = tmp_path / "export_report.json"
    export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)
    before_model = out_path.read_bytes()
    before_report = report_path.read_bytes()

    class FailingSession:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, *_args, **_kwargs):
            raise RuntimeError("runtime exploded")

    monkeypatch.setattr(ort, "InferenceSession", FailingSession)

    with pytest.raises(RuntimeError, match="runtime exploded"):
        export_to_onnx(checkpoint_path=ckpt, out_path=out_path, imgsz=224, opset=17)

    assert out_path.read_bytes() == before_model
    assert report_path.read_bytes() == before_report
    assert not (tmp_path / "encoder.part.onnx").exists()


def _install_native_register_factory(monkeypatch, native_regs=2, class_token=True):
    import timm as timm_module
    from timm.models.vision_transformer import VisionTransformer

    def factory(_model_name, **kwargs):
        requested = kwargs.get("reg_tokens")
        return VisionTransformer(
            img_size=32, patch_size=16, in_chans=3, embed_dim=16, depth=2,
            num_heads=2, num_classes=0, global_pool="", class_token=class_token,
            reg_tokens=native_regs if requested is None else int(requested),
            dynamic_img_size=True,
        )

    monkeypatch.setattr(timm_module, "create_model", factory)
    return VisionTransformer


def test_export_reconstructs_a_native_register_checkpoint(tmp_path: Path, monkeypatch):
    import otuformer.training.model as model_module

    _install_native_register_factory(monkeypatch, native_regs=2)
    model = model_module.OTUFormerEncoder(
        model_name="tiny-native", out_dim=8, pretrained=False, img_size=32
    )
    ckpt_path = tmp_path / "native.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": "tiny-native",
                "out_dim": 8,
                "image_size": 32,
            },
        },
        ckpt_path,
    )
    out_path = tmp_path / "native.onnx"

    report = export_to_onnx(
        checkpoint_path=ckpt_path, out_path=out_path, imgsz=32, opset=17
    )

    assert report["register_tokens"] == 2
    assert report["validation_status"] == "passed"


def test_export_rejects_a_no_cls_native_register_checkpoint(
    tmp_path: Path, monkeypatch
):
    _install_native_register_factory(monkeypatch, native_regs=4, class_token=False)
    from timm.models.vision_transformer import VisionTransformer

    raw = VisionTransformer(
        img_size=32, patch_size=16, in_chans=3, embed_dim=16, depth=2,
        num_heads=2, num_classes=0, global_pool="", class_token=False,
        reg_tokens=4, dynamic_img_size=True,
    )
    ckpt_path = tmp_path / "gap_reg4.pth"
    torch.save(
        {
            "model_state_dict": {
                f"backbone.{key}": value for key, value in raw.state_dict().items()
            },
            "config": {
                "model_name": "tiny-gap-reg4",
                "out_dim": 8,
                "image_size": 32,
            },
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="no CLS token"):
        export_to_onnx(
            checkpoint_path=ckpt_path,
            out_path=tmp_path / "bad.onnx",
            imgsz=32,
            opset=17,
        )
