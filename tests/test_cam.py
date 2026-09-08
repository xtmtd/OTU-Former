import numpy as np
import pytest
import torch
from PIL import Image

from otuformer.vision.cam import (
    CAM_METHODS,
    build_display_transform,
    default_vit_target,
    get_module_by_name,
    infer_architecture,
    load_model_from_checkpoint,
    map_cam_to_original,
    vit_reshape_transform,
)


def test_cam_checkpoint_loader_accepts_legacy_arcface_model_key(monkeypatch, tmp_path):
    import otuformer.vision.cam as cam_module

    seen = {}

    class FakeBackbone(torch.nn.Module):
        def forward_features(self, x):
            return x

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.backbone = FakeBackbone()

        def load_state_dict(self, state_dict, **_kwargs):
            seen["state_dict"] = state_dict

    legacy_weights = {"backbone.cls_token": torch.zeros(1, 1, 1)}
    monkeypatch.setattr(cam_module, "load_checkpoint", lambda _path: {"model": legacy_weights})
    monkeypatch.setattr(cam_module, "OTUFormerEncoder", FakeEncoder)

    cam_module.load_model_from_checkpoint(
        tmp_path / "arcface_epoch_0020.pth", "vit_tiny_patch16_224", torch.device("cpu")
    )

    assert seen["state_dict"] is legacy_weights


def test_cam_checkpoint_loader_prefers_legacy_ssl_teacher_key(monkeypatch, tmp_path):
    import otuformer.vision.cam as cam_module

    seen = {}

    class FakeBackbone(torch.nn.Module):
        def forward_features(self, x):
            return x

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.backbone = FakeBackbone()

        def load_state_dict(self, state_dict, **_kwargs):
            seen["state_dict"] = state_dict

    teacher_weights = {"backbone.cls_token": torch.ones(1, 1, 1)}
    student_weights = {"backbone.cls_token": torch.zeros(1, 1, 1)}
    monkeypatch.setattr(
        cam_module,
        "load_checkpoint",
        lambda _path: {"teacher": teacher_weights, "student": student_weights},
    )
    monkeypatch.setattr(cam_module, "OTUFormerEncoder", FakeEncoder)

    cam_module.load_model_from_checkpoint(
        tmp_path / "SSL_epoch_0020.pth", "vit_tiny_patch16_224", torch.device("cpu")
    )

    assert seen["state_dict"] is teacher_weights


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


def test_cam_load_model_constructs_at_checkpoint_size(monkeypatch, tmp_path):
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

    monkeypatch.setattr(
        cam_module,
        "load_checkpoint",
        lambda _path: {
            "model_state_dict": {},
            "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 64},
            "args": {"global_crop_size": 32},
        },
    )
    monkeypatch.setattr(cam_module, "OTUFormerEncoder", FakeEncoder)

    model, size, name = cam_module.load_model_from_checkpoint(
        tmp_path / "model.pth", "vit_tiny_patch16_224", torch.device("cpu")
    )

    assert seen["img_size"] == 32
    assert size == 32
    assert name == "vit_tiny_patch16_224"
    assert isinstance(model, torch.nn.Module)


def test_cam_arch_is_inferred_from_checkpoint_model_name(monkeypatch, tmp_path):
    """A CNN checkpoint must not be misjudged as ViT via the CLI default name."""
    import otuformer.vision.cam as cam_module

    seen = {}

    class FakeModel(torch.nn.Module):
        backbone = torch.nn.Module()

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (32, 32)).save(img_dir / "img.jpg")

    monkeypatch.setattr(
        cam_module,
        "load_model_from_checkpoint",
        lambda _ckpt, _name, _dev: (FakeModel(), 32, "convnextv2_femto"),
    )

    def fake_prepare_cam(model, arch, target_layer_name, cam_name, cam_batch_size):
        seen["arch"] = arch
        return "cam", ("layer",), None

    monkeypatch.setattr(cam_module, "prepare_cam", fake_prepare_cam)

    cam_module.run_cam(
        checkpoint=tmp_path / "ckpt.pth",
        images_dir=img_dir,
        out_dir=tmp_path / "out",
        device="cpu",
    )

    assert seen["arch"] == "cnn"


def test_build_display_transform_matches_protocol_and_rejects_unknown():
    import torchvision.transforms as transforms

    from otuformer.training.dataset import EVAL_TRANSFORMS, build_eval_transform

    for name in EVAL_TRANSFORMS:
        preprocess = build_eval_transform(name, 224)
        display = build_display_transform(name, 224)
        # the display transform is exactly the geometric part of the preprocess
        assert [type(t) for t in display.transforms] == [
            type(t) for t in preprocess.transforms[: len(display.transforms)]
        ]
        assert len(preprocess.transforms) == len(display.transforms) + 2  # + ToTensor, Normalize
    with pytest.raises(ValueError, match="Unknown eval transform 'bogus'"):
        build_display_transform("bogus", 224)
    assert "center-crop" in EVAL_TRANSFORMS
    assert "whole-specimen-pad" in EVAL_TRANSFORMS


def test_map_cam_center_crop_confines_to_field_of_view():
    model_size = 32
    cam = np.zeros((model_size, model_size), dtype=np.float32)
    cam[17:19, 8:24] = 1.0  # centered activation

    cam_on_full, fov = map_cam_to_original(cam, (100, 200), "center-crop", model_size)

    assert cam_on_full.shape == (200, 100)
    assert fov.shape == (200, 100)
    # portrait 100x200 -> resize (32, 64), center crop 32x32 at rows 16..48
    # -> FOV is the middle vertical band of the original
    assert not fov[20, 50]
    assert not fov[170, 50]
    assert fov[100, 50]
    # the centered activation maps to the vertical center of the original
    ys, xs = np.nonzero(cam_on_full > 0.9 * cam_on_full.max())
    assert 100 <= ys.mean() <= 112
    assert 40 <= xs.mean() <= 60


def test_map_cam_center_crop_matches_torchvision_resize_rounding():
    # torchvision Resize(32) truncates the long side with int(): 101x333 -> 32x105
    import torchvision.transforms as transforms

    tf = transforms.Resize(32, interpolation=Image.BICUBIC)
    assert tf(Image.new("RGB", (101, 333))).size == (32, 105)

    model_size = 32
    cam = np.zeros((model_size, model_size), dtype=np.float32)
    cam[:, :] = 1.0
    cam_on_full, fov = map_cam_to_original(
        cam, (101, 333), "center-crop", model_size
    )
    assert cam_on_full.shape == (333, 101)
    # crop y-offset mirrors CenterCrop: round((105 - 32) / 2) = 36
    first = int(np.nonzero(fov)[0].min())
    expected_first = int(round((105 - model_size) / 2.0) * 333 / 105)
    assert abs(first - expected_first) <= 1.5


def test_process_image_uint8_overlay_does_not_overflow(monkeypatch, tmp_path):
    """show_cam_on_image returns uint8 [0, 255]; rescaling must not wrap."""
    import cv2  # noqa: F401
    import otuformer.vision.cam as cam_module

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 100), (200, 200, 200)).save(img_path)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 192)

    monkeypatch.setattr(cam_module, "show_cam_on_image", _fake_show)

    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    cam_module.process_image(
        img_path=img_path,
        label="",
        model=FakeModel(),
        preprocess=cam_module.build_eval_transform("center-crop", 32),
        display_transform=cam_module.build_display_transform("center-crop", 32),
        cam_extractor=_fake_cam_extractor(),
        device=torch.device("cpu"),
        fig_dir=fig_dir,
        array_dir=None,
        image_weight=0.8,
        fig_format="png",
        save_npy=False,
        eval_transform="center-crop",
        model_size=32,
    )
    overlay = np.array(Image.open(fig_dir / "img_cam.png").convert("RGB"))
    assert overlay.max() > 200  # bright heatmap pixels survive without uint8 wrap


def test_map_cam_center_crop_square_unchanged():
    model_size = 32
    cam = np.zeros((model_size, model_size), dtype=np.float32)
    cam[8:24, 8:24] = 1.0
    cam_on_full, fov = map_cam_to_original(cam, (100, 100), "center-crop", model_size)
    assert cam_on_full.shape == (100, 100)
    assert fov.all()


def test_map_cam_whole_specimen_pad_covers_full_frame():
    model_size = 32
    cam = np.zeros((model_size, model_size), dtype=np.float32)
    cam[8:24, 8:24] = 1.0

    cam_on_full, fov = map_cam_to_original(
        cam, (100, 200), "whole-specimen-pad", model_size
    )

    assert cam_on_full.shape == (200, 100)
    assert fov.all()  # whole specimen in view, no crop-confined blanking
    ys, xs = np.nonzero(cam_on_full > 0.9 * cam_on_full.max())
    assert abs(ys.mean() - 100) <= 3
    assert abs(xs.mean() - 50) <= 3


def test_map_cam_whole_specimen_pad_square_unchanged():
    model_size = 32
    cam = np.zeros((model_size, model_size), dtype=np.float32)
    cam[8:24, 8:24] = 1.0
    cam_on_full, fov = map_cam_to_original(
        cam, (100, 100), "whole-specimen-pad", model_size
    )
    assert cam_on_full.shape == (100, 100)
    assert fov.all()


def _fake_cam_extractor():
    def extractor(input_tensor=None, targets=None):
        cam = np.zeros((32, 32), dtype=np.float32)
        cam[14:18, 10:22] = 1.0
        return [cam]

    return extractor


def _fake_show(rgb, cam_full, image_weight=0.8, **kwargs):
    """Emulate show_cam_on_image, matching its real return contract: uint8 [0, 255].

    cam=0 renders as the colormap's lowest color (black-ish here), so an
    out-of-view overlay that was routed through the colormap would stay at
    ``image_weight * rgb``; the FOV fix must replace it with a dimmed gray.
    """
    heat = np.clip(cam_full, 0, 1)[..., None] * np.array([0.0, 0.3, 1.0])
    blended = (rgb * image_weight + heat * (1.0 - image_weight)).astype(np.float32)
    return np.uint8(255 * np.clip(blended, 0, 1))


def test_process_image_dims_out_of_view_for_center_crop(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    import cv2  # noqa: F401
    import otuformer.vision.cam as cam_module

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 200), (200, 200, 200)).save(img_path)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 192)

    monkeypatch.setattr(cam_module, "show_cam_on_image", _fake_show)

    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    result = cam_module.process_image(
        img_path=img_path,
        label="",
        model=FakeModel(),
        preprocess=cam_module.build_eval_transform("center-crop", 32),
        display_transform=cam_module.build_display_transform("center-crop", 32),
        cam_extractor=_fake_cam_extractor(),
        device=torch.device("cpu"),
        fig_dir=fig_dir,
        array_dir=None,
        image_weight=0.8,
        fig_format="png",
        save_npy=False,
        eval_transform="center-crop",
        model_size=32,
    )
    assert result["figure_path"]

    overlay = np.array(Image.open(fig_dir / "img_cam.png").convert("RGB"))[:, 100:, :]
    overlay = overlay.astype(np.float32) / 255.0
    # row 30 is out of view (dimmed directly, not colormap-rendered), row 100
    # is inside the model's field of view
    assert overlay[30, :, :].mean() < 0.55
    assert overlay[100, :, :].mean() > overlay[30, :, :].mean() + 0.05


def test_process_image_whole_specimen_pad_keeps_full_frame(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    import cv2  # noqa: F401
    import otuformer.vision.cam as cam_module

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 200), (200, 200, 200)).save(img_path)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 192)

    monkeypatch.setattr(cam_module, "show_cam_on_image", _fake_show)

    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    result = cam_module.process_image(
        img_path=img_path,
        label="",
        model=FakeModel(),
        preprocess=cam_module.build_eval_transform("whole-specimen-pad", 32),
        display_transform=cam_module.build_display_transform("whole-specimen-pad", 32),
        cam_extractor=_fake_cam_extractor(),
        device=torch.device("cpu"),
        fig_dir=fig_dir,
        array_dir=None,
        image_weight=0.8,
        fig_format="png",
        save_npy=False,
        eval_transform="whole-specimen-pad",
        model_size=32,
    )
    assert result["figure_path"]

    overlay = np.array(Image.open(fig_dir / "img_cam.png").convert("RGB"))[:, 100:, :]
    overlay = overlay.astype(np.float32) / 255.0
    # whole frame in view: the top band is colormap-rendered, NOT dimmed
    assert overlay[30, :, :].mean() > 0.6
    assert overlay[100, :, :].mean() > overlay[30, :, :].mean()


def test_process_image_saves_raw_model_resolution_npy(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    import cv2  # noqa: F401
    import otuformer.vision.cam as cam_module

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 100), (200, 200, 200)).save(img_path)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 192)

    monkeypatch.setattr(
        cam_module,
        "show_cam_on_image",
        lambda rgb, cam_full, **kwargs: (
            (rgb * 0.5 + np.clip(cam_full, 0, 1)[..., None] * 0.5).astype(np.float32)
        ),
    )

    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    arr_dir = tmp_path / "arrays"
    arr_dir.mkdir()
    result = cam_module.process_image(
        img_path=img_path,
        label="",
        model=FakeModel(),
        preprocess=cam_module.build_eval_transform("center-crop", 32),
        display_transform=cam_module.build_display_transform("center-crop", 32),
        cam_extractor=_fake_cam_extractor(),
        device=torch.device("cpu"),
        fig_dir=fig_dir,
        array_dir=arr_dir,
        image_weight=0.5,
        fig_format="png",
        save_npy=True,
        eval_transform="center-crop",
        model_size=32,
    )
    saved = np.load(result["cam_array_path"])
    assert saved.shape == (32, 32)  # raw model-resolution map, backward compatible
    assert float(saved.max()) == 1.0
