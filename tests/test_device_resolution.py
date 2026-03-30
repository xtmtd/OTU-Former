from otuformer.training import trainer


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    if hasattr(trainer.torch.backends, "mps"):
        monkeypatch.setattr(trainer.torch.backends.mps, "is_available", lambda: True)

    dev = trainer._resolve_device("auto")
    assert dev.type == "cuda"


def test_resolve_device_auto_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    has_mps = hasattr(trainer.torch.backends, "mps")
    if has_mps:
        monkeypatch.setattr(trainer.torch.backends.mps, "is_available", lambda: True)

    dev = trainer._resolve_device("auto")
    assert dev.type == ("mps" if has_mps else "cpu")


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    if hasattr(trainer.torch.backends, "mps"):
        monkeypatch.setattr(trainer.torch.backends.mps, "is_available", lambda: False)

    dev = trainer._resolve_device("auto")
    assert dev.type == "cpu"
