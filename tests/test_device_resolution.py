import otuformer.training.pretrain_loop as pretrain_loop_module
from otuformer.training import trainer
import otuformer.training.runtime as runtime_module
import otuformer.training.pretrain_recipe as pretrain_recipe_module


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(pretrain_recipe_module.torch.cuda, "is_available", lambda: True)
    if hasattr(pretrain_recipe_module.torch.backends, "mps"):
        monkeypatch.setattr(pretrain_recipe_module.torch.backends.mps, "is_available", lambda: True)

    dev = pretrain_loop_module._resolve_device("auto")
    assert dev.type == "cuda"


def test_resolve_device_auto_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr(pretrain_recipe_module.torch.cuda, "is_available", lambda: False)
    has_mps = hasattr(pretrain_recipe_module.torch.backends, "mps")
    if has_mps:
        monkeypatch.setattr(pretrain_recipe_module.torch.backends.mps, "is_available", lambda: True)

    dev = pretrain_loop_module._resolve_device("auto")
    assert dev.type == ("mps" if has_mps else "cpu")


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(pretrain_recipe_module.torch.cuda, "is_available", lambda: False)
    if hasattr(pretrain_recipe_module.torch.backends, "mps"):
        monkeypatch.setattr(pretrain_recipe_module.torch.backends.mps, "is_available", lambda: False)

    dev = pretrain_loop_module._resolve_device("auto")
    assert dev.type == "cpu"
