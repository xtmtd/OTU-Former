import importlib
import sys


def test_main_import_does_not_eager_load_heavy_modules():
    modules_to_clear = [
        "otuformer.cli.main",
        "otuformer.cli.cam",
        "otuformer.cli.cluster",
        "otuformer.vision.cam",
        "scipy.cluster",
        "pytorch_grad_cam",
    ]
    for module_name in modules_to_clear:
        sys.modules.pop(module_name, None)

    importlib.import_module("otuformer.cli.main")

    assert "otuformer.vision.cam" not in sys.modules
    assert "scipy.cluster" not in sys.modules
    assert "pytorch_grad_cam" not in sys.modules
