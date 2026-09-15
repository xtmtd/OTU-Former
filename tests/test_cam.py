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


CAM_METHOD_NAMES = (
    "ablationcam",
    "eigencam",
    "gradcam",
    "gradcampp",
    "layercam",
    "scorecam",
)


def _tiny_conv_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 3),
    )


def _fixed_cam_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 3, 32, 32)


def test_cam_methods_use_raw_subclasses():
    pytest.importorskip("pytorch_grad_cam")
    from otuformer.vision.cam import CAM_METHODS, UnnormalizedCAMMixin

    assert set(CAM_METHODS) == set(CAM_METHOD_NAMES)
    for name, cls in CAM_METHODS.items():
        assert issubclass(cls, UnnormalizedCAMMixin), name


def test_raw_cam_does_not_call_scale_cam_image(monkeypatch):
    """Structural proof: the raw path never reaches upstream normalization.

    The official class must raise under the same patch, otherwise this test
    would pass vacuously.
    """
    pytest.importorskip("pytorch_grad_cam")
    import pytorch_grad_cam.base_cam as base_cam_module
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from otuformer.vision.cam import CAM_METHODS

    def _boom(*_args, **_kwargs):
        raise AssertionError("scale_cam_image must not be called")

    monkeypatch.setattr(base_cam_module, "scale_cam_image", _boom)
    tensor = _fixed_cam_input()
    targets = [ClassifierOutputTarget(1)]

    torch.manual_seed(0)
    official_model = _tiny_conv_model().eval()
    with pytest.raises(AssertionError):
        GradCAM(model=official_model, target_layers=[official_model[4]])(
            input_tensor=tensor, targets=targets
        )

    torch.manual_seed(0)
    raw_model = _tiny_conv_model().eval()
    raw = CAM_METHODS["gradcam"](model=raw_model, target_layers=[raw_model[4]])(
        input_tensor=tensor, targets=targets
    )[0]

    assert raw.shape == (32, 32)


def test_raw_cam_agrees_with_official_cam_after_min_max():
    """Protect display equivalence; does not pin any CAM magnitude.

    Catches a changed resize, a changed ReLU policy, a changed aggregation, a
    new non-affine upstream step, and an upstream that stops normalizing (via
    ``official.max()``). It does not catch a pure magnitude scaling of raw CAMs,
    which min-max would hide.
    """
    pytest.importorskip("pytorch_grad_cam")
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from otuformer.vision.cam import CAM_METHODS

    torch.manual_seed(0)
    model = _tiny_conv_model().eval()
    tensor = _fixed_cam_input()
    targets = [ClassifierOutputTarget(1)]

    official = GradCAM(model=model, target_layers=[model[4]])(
        input_tensor=tensor, targets=targets
    )[0]
    raw = CAM_METHODS["gradcam"](model=model, target_layers=[model[4]])(
        input_tensor=tensor, targets=targets
    )[0]

    display = raw - raw.min()
    display = display / display.max()

    assert official.max() == pytest.approx(1.0, abs=1e-6)
    assert np.abs(display - official).max() < 1e-6


@pytest.mark.parametrize("name", CAM_METHOD_NAMES)
def test_every_cam_method_returns_unnormalized_map(name):
    pytest.importorskip("pytorch_grad_cam")
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from otuformer.vision.cam import CAM_METHODS

    torch.manual_seed(0)
    model = _tiny_conv_model().eval()
    cam = CAM_METHODS[name](model=model, target_layers=[model[4]])

    raw = cam(input_tensor=_fixed_cam_input(), targets=[ClassifierOutputTarget(1)])[0]

    # Structural invariants only: no value here is an acceptance criterion.
    assert raw.shape == (32, 32)
    assert raw.dtype == np.float32
    assert np.isfinite(raw).all()
    assert raw.min() >= 0.0
    # No positivity assertion. A fully ReLU'd-zero map is legal (scorecam returns
    # exactly zero on this tiny fixture), and eigencam is additionally exposed to
    # the SVD sign. Magnitude is environment/backend dependent and is never
    # acceptance evidence; the binding evidence is the raw-subclass mapping, the
    # scale_cam_image sentinel, and the test-owned save-path array.


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
        save_npy="none",
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
        save_npy="none",
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
        save_npy="none",
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

    expected_raw = np.zeros((32, 32), dtype=np.float32)
    expected_raw[8:16, 10:22] = 2.0

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
        cam_extractor=lambda **_kwargs: [expected_raw.copy()],
        device=torch.device("cpu"),
        fig_dir=fig_dir,
        array_dir=arr_dir,
        image_weight=0.5,
        fig_format="png",
        save_npy="raw",
        eval_transform="center-crop",
        model_size=32,
    )
    saved = np.load(result["cam_array_path"])
    assert saved.shape == (32, 32)
    assert saved.dtype == np.float32
    np.testing.assert_allclose(saved, expected_raw)


FAKE_CAM = np.array([[0.0, 2.0], [1.0, 0.5]], dtype=np.float32)


def _run_fake_cam_process_image(tmp_path, monkeypatch, save_npy):
    import otuformer.vision.cam as cam_module

    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(image_path)
    array_dir = tmp_path / "arrays"
    array_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(
        cam_module,
        "show_cam_on_image",
        lambda *_args, **_kwargs: np.full((32, 32, 3), 240, dtype=np.uint8),
    )

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 1.0]], device=x.device).repeat(x.shape[0], 1)

    return cam_module.process_image(
        img_path=image_path,
        label="x",
        model=FakeModel(),
        preprocess=lambda _image: torch.zeros(3, 32, 32),
        display_transform=cam_module.build_display_transform("center-crop", 2),
        cam_extractor=lambda **_kwargs: [FAKE_CAM.copy()],
        device=torch.device("cpu"),
        fig_dir=tmp_path,
        array_dir=array_dir,
        image_weight=0.5,
        fig_format="png",
        save_npy=save_npy,
        eval_transform="center-crop",
        model_size=2,
    )


def test_process_image_saves_unnormalized_raw_array(tmp_path, monkeypatch):
    record = _run_fake_cam_process_image(tmp_path, monkeypatch, "raw")

    saved = np.load(tmp_path / "arrays" / "image.npy")
    assert saved.dtype == np.float32
    np.testing.assert_allclose(saved, FAKE_CAM)
    assert float(saved.max()) == 2.0
    assert record["cam_array_path"].endswith("image.npy")


def test_process_image_saves_normalized_array_on_request(tmp_path, monkeypatch):
    _run_fake_cam_process_image(tmp_path, monkeypatch, "normalized")

    saved = np.load(tmp_path / "arrays" / "image.npy")
    np.testing.assert_allclose(
        saved, np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    )


def test_process_image_writes_no_array_when_disabled(tmp_path, monkeypatch):
    record = _run_fake_cam_process_image(tmp_path, monkeypatch, "none")

    assert record["cam_array_path"] == ""
    assert list((tmp_path / "arrays").iterdir()) == []


def test_process_image_overlay_receives_normalized_mask(tmp_path, monkeypatch):
    """The overlay must always receive the normalized, mapped copy.

    ``FAKE_CAM`` peaks at ``2.0``; if the raw array leaked into the overlay the
    mapped mask would peak at ``2.0`` and the bound below would fail.
    """
    import otuformer.vision.cam as cam_module

    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(image_path)
    array_dir = tmp_path / "arrays"
    array_dir.mkdir()

    captured = {}

    def _capture(_rgb, mask, **_kwargs):
        captured["mask"] = np.asarray(mask).copy()
        return np.full((32, 32, 3), 240, dtype=np.uint8)

    monkeypatch.setattr(cam_module, "show_cam_on_image", _capture)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 1.0]], device=x.device).repeat(x.shape[0], 1)

    cam_module.process_image(
        img_path=image_path,
        label="x",
        model=FakeModel(),
        preprocess=lambda _image: torch.zeros(3, 32, 32),
        display_transform=cam_module.build_display_transform("center-crop", 2),
        cam_extractor=lambda **_kwargs: [FAKE_CAM.copy()],
        device=torch.device("cpu"),
        fig_dir=tmp_path,
        array_dir=array_dir,
        image_weight=0.5,
        fig_format="png",
        save_npy="raw",
        eval_transform="center-crop",
        model_size=2,
    )

    cam_display = FAKE_CAM - FAKE_CAM.min()
    cam_display = cam_display / cam_display.max()
    expected, _fov = cam_module.map_cam_to_original(
        cam_display, (32, 32), "center-crop", 2
    )

    mask = captured["mask"]
    assert mask.shape == (32, 32)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    np.testing.assert_allclose(mask, expected)


@pytest.mark.parametrize("name", ["gradcam", "scorecam"])
def test_overlay_figure_matches_between_official_and_raw_cam(name):
    """Overlay rendering must not change beyond a bounded uint8 LSB difference.

    ``scorecam`` is included deliberately: it is the method most exposed to the
    normalization-before/after-resize ordering, so a gradcam-only check could
    not detect the regression this test exists for. Pixel identity is not a
    contract; the bound is a tolerance, not equality.
    """
    pytest.importorskip("pytorch_grad_cam")
    import pytorch_grad_cam
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    import otuformer.vision.cam as cam_module

    official_cls = {
        "gradcam": pytorch_grad_cam.GradCAM,
        "scorecam": pytorch_grad_cam.ScoreCAM,
    }[name]
    torch.manual_seed(0)
    tensor = torch.rand(1, 3, 64, 64)
    targets = [ClassifierOutputTarget(1)]
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)

    def render(cls):
        torch.manual_seed(0)
        model = _tiny_conv_model().eval()
        cam_map = cls(model=model, target_layers=[model[4]])(
            input_tensor=tensor, targets=targets
        )[0]
        display = cam_map - cam_map.min()
        if display.max() > 0:
            display = display / display.max()
        mapped, _fov = cam_module.map_cam_to_original(
            display, (64, 64), "center-crop", 64
        )
        return show_cam_on_image(rgb, mapped, use_rgb=True)

    official = render(official_cls).astype(np.int16)
    raw = render(cam_module.CAM_METHODS[name]).astype(np.int16)
    difference = np.abs(official - raw)

    assert difference.max() <= 2
    assert (difference > 0).mean() < 0.01


def test_run_cam_rejects_invalid_save_npy(tmp_path):
    """A bool or unknown string must fail at the API boundary, not save arrays."""
    from otuformer.vision.cam import process_image, run_cam

    for invalid in (True, False, "bogus", None):
        with pytest.raises(ValueError, match="save_npy"):
            run_cam(
                checkpoint=tmp_path / "missing.ckpt",
                images_dir=tmp_path,
                out_dir=tmp_path / "out",
                save_npy=invalid,
            )

    with pytest.raises(ValueError, match="save_npy"):
        process_image(
            img_path=tmp_path / "missing.png",
            label="",
            model=None,
            preprocess=None,
            display_transform=None,
            cam_extractor=None,
            device=None,
            fig_dir=tmp_path,
            array_dir=None,
            image_weight=0.5,
            fig_format="png",
            save_npy="bogus",
        )


def test_cam_loads_arcface_embedding_head_checkpoint(tmp_path):
    from otuformer.training.model import ArcFaceEmbeddingHead, OTUFormerEncoder

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=64, pretrained=False
    )
    model.projector = ArcFaceEmbeddingHead(model.backbone.num_features, 64)
    checkpoint = tmp_path / "arcface.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 64,
                "embedding_head": "arcface_mlp_512",
            },
        },
        checkpoint,
    )

    cam_model, size, name = load_model_from_checkpoint(
        checkpoint, "vit_tiny_patch16_224", torch.device("cpu")
    )

    assert name == "vit_tiny_patch16_224"
    assert size == 224
    assert isinstance(cam_model, torch.nn.Module)
