import torch

from otuformer.vision.cam import (
    CAM_METHODS,
    default_vit_target,
    get_module_by_name,
    infer_architecture,
    vit_reshape_transform,
)


def test_cam_checkpoint_loader_disables_pretrained_weights(monkeypatch, tmp_path):
    import otuformer.vision.cam as cam_module

    seen = {}

    class FakeBackbone(torch.nn.Module):
        def forward_features(self, x):
            return x

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)
            self.backbone = FakeBackbone()

        def load_state_dict(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(cam_module, "load_checkpoint", lambda _path: {"model_state_dict": {}})
    monkeypatch.setattr(cam_module, "OTUFormerEncoder", FakeEncoder)

    cam_module.load_model_from_checkpoint(tmp_path / "model.pth", "vit_tiny_patch16_224", torch.device("cpu"))

    assert seen["pretrained"] is False


def make_tiny_vit():
    import timm

    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)


def test_cam_methods_complete_if_dependency_available():
    if len(CAM_METHODS) == 0:
        return
    expected = {
        "gradcam",
        "gradcampp",
        "scorecam",
        "layercam",
        "eigencam",
        "ablationcam",
    }
    assert expected == set(CAM_METHODS.keys())


def test_infer_architecture_vit():
    model = make_tiny_vit()
    arch = infer_architecture("vit_tiny_patch16_224", model)
    assert arch == "vit"


def test_infer_architecture_cnn():
    import timm

    model = timm.create_model("convnextv2_femto", pretrained=False)
    arch = infer_architecture("convnextv2_femto", model)
    assert arch == "cnn"


def test_default_vit_target_returns_module():
    model = make_tiny_vit()
    target = default_vit_target(model)
    assert isinstance(target, torch.nn.Module)


def test_vit_reshape_transform_shape():
    tensor = torch.randn(2, 197, 192)
    out = vit_reshape_transform(tensor)
    assert out.shape == (2, 192, 14, 14)


def test_get_module_by_name():
    model = make_tiny_vit()
    mod = get_module_by_name(model, "blocks.0")
    assert isinstance(mod, torch.nn.Module)
