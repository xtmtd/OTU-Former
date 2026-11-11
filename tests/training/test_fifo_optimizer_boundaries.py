"""Regression checks for FIFO admission and the successful-update boundary."""

import pytest
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from typer.testing import CliRunner

from otuformer.cli.main import app
from otuformer.training import model, pretrain_recipe as recipe


def _group(**changes):
    return {
        "params": [0], "lr": 0.001, "eps": 1e-8,
        "betas": (0.9, 0.999), "weight_decay": 0.0, **changes,
    }


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("coefficients", [
    {"betas": (1 - 1e-12, 0.999)}, {"eps": 1e100},
])
def test_fifo_accepts_usable_scalar_coefficients(device, coefficients):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    group = _group(**coefficients)
    recipe._validate_queue_optimizer_coefficients(
        group, group_index=0, lr_bound=0.001,
        target_device=torch.device(device),
    )
    parameter = torch.nn.Parameter(torch.ones(4, device=device))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    saved = optimizer.state_dict()
    saved["param_groups"][0].update({k: v for k, v in group.items() if k != "params"})
    saved["state"][0] = {
        "step": torch.tensor(1.0), "exp_avg": torch.zeros(4),
        "exp_avg_sq": torch.zeros(4),
    }
    optimizer.load_state_dict(saved)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()


def test_fifo_rejects_mps_subnormal_eps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    with pytest.raises(ValueError, match="optimizer"):
        recipe._validate_queue_optimizer_coefficients(
            _group(eps=1e-38), group_index=0, lr_bound=0.001,
            target_device=torch.device("mps"),
        )
    recipe._validate_queue_optimizer_coefficients(
        _group(eps=1e-38), group_index=0, lr_bound=0.001,
        target_device=torch.device("cpu"),
    )


def test_fifo_rejects_zero_lr_differentiable_update():
    with pytest.raises(ValueError, match="optimizer"):
        recipe._validate_queue_optimizer_coefficients(
            _group(differentiable=True), group_index=0, lr_bound=0.0,
            target_device=torch.device("cpu"),
        )
    recipe._validate_queue_optimizer_coefficients(
        _group(), group_index=0, lr_bound=0.0,
        target_device=torch.device("cpu"),
    )


def test_fifo_resume_probes_the_remaining_schedule_not_unused_endpoints():
    checkpoint = {
        "optimizer": {"param_groups": [_group(differentiable=True)]},
        "schedule": {
            "original_max_epochs": 2, "steps_per_epoch": 2, "total_steps": 4,
            "warmup_steps": 0, "last_lr": 0.001, "final_lr": 0.0,
        },
        "epoch": 0, "iteration": 2,
    }
    # The cosine sequence never reaches its zero final endpoint in this plan.
    recipe._validate_queue_optimizer_precision(
        checkpoint, requested_lr=0.001, requested_weight_decay=0.0,
        requested_max_epochs=2,
    )


def test_fifo_resume_checks_saved_weight_decay_only():
    checkpoint = {"optimizer": {"param_groups": [_group()]}}
    recipe._validate_queue_optimizer_precision(
        checkpoint, requested_lr=0.001, requested_weight_decay=1e100,
    )


@pytest.mark.parametrize("flags, step", [
    ({"foreach": True}, torch.tensor(1, dtype=torch.bfloat16)),
    ({}, torch.tensor(1.0, requires_grad=True)),
    ({"foreach": True}, torch.tensor(1.0, requires_grad=True)),
    ({"fused": True}, torch.tensor(1.0, requires_grad=True)),
])
def test_fifo_accepts_working_cpu_step_formats(flags, step):
    flags = {**dict.fromkeys(("fused", "foreach", "capturable", "differentiable"), False), **flags}
    assert recipe._validate_fifo_step_format(
        step, flags=flags, target_device=torch.device("cpu"), label="optimizer step",
    ) == 1.0
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    state = optimizer.state_dict()
    state["param_groups"][0].update(flags)
    state["state"][0] = {
        "step": step.clone(), "exp_avg": torch.zeros(4), "exp_avg_sq": torch.zeros(4),
    }
    optimizer.load_state_dict(state)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()


@pytest.fixture
def fifo_cli(tmp_path, monkeypatch):
    def tiny(*args, **kwargs):
        return VisionTransformer(
            img_size=64, patch_size=8, embed_dim=16, depth=4, num_heads=2,
            num_classes=0, global_pool="", dynamic_img_size=True,
        )
    monkeypatch.setattr(model.timm, "create_model", tiny)
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64), (42, 5, 8)).save(images / "one.jpg")
    return [
        "pretrain", "--input-images-dir", str(images), "--batch-size", "1",
        "--out-dim", "8", "--relational-queue-size", "1", "--global-crop-size", "64",
        "--local-crops", "0", "--patch-loss", "none", "--device", "cpu",
        "--num-workers", "0", "--cpus", "2", "--disable-embedding-metrics",
    ]


@pytest.mark.parametrize("weight_decay", ["1e100", "-1e-100"])
def test_fifo_cli_invalid_decay_preserves_outputs(tmp_path, fifo_cli, weight_decay):
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    (out / "keep.txt").write_bytes(b"original sentinel")
    (out / "logs" / "pretrain.log").write_bytes(b"original log\n")
    def snapshot():
        return {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    before = snapshot()
    result = CliRunner().invoke(app, fifo_cli + [
        "--out-dir", str(out), "--overwrite", "--max-epochs", "1",
        "--weight-decay", weight_decay,
    ])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert snapshot() == before


@pytest.mark.parametrize("device, lr, changes", [
    ("cpu", "0", {"differentiable": True}),
    ("mps", "0.001", {"eps": 1e-38}),
])
def test_fifo_cli_bad_update_preserves_resume_outputs(tmp_path, fifo_cli, device, lr, changes):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    source = tmp_path / "source"
    runner = CliRunner()
    first = runner.invoke(app, fifo_cli + ["--out-dir", str(source), "--max-epochs", "1", "--lr", lr])
    assert first.exit_code == 0, (first.output, repr(first.exception))
    saved = torch.load(source / "SSL_latest.pth", map_location="cpu", weights_only=False)
    for group in saved["optimizer"]["param_groups"]:
        group.update(changes)
    checkpoint = tmp_path / "bad.pth"
    torch.save(saved, checkpoint)
    out = tmp_path / "continued"
    (out / "logs").mkdir(parents=True)
    log = out / "logs" / "pretrain.log"
    log.write_bytes(b"old log\n")
    resumed = runner.invoke(app, fifo_cli + [
        "--out-dir", str(out), "--resume", str(checkpoint), "--max-epochs", "2",
        "--device", device, "--lr", lr,
    ])
    assert resumed.exit_code == 2, (resumed.output, repr(resumed.exception))
    assert log.read_bytes() == b"old log\n"
    assert not (out / "SSL_latest.pth").exists()


@pytest.mark.parametrize("step_format", ["default", "foreach_bfloat16", "requires_grad"])
def test_fifo_cli_accepts_effective_saved_coefficients(
    tmp_path, fifo_cli, step_format,
):
    source = tmp_path / "source"
    runner = CliRunner()
    first = runner.invoke(app, fifo_cli + [
        "--out-dir", str(source), "--max-epochs", "1", "--weight-decay", "0",
    ])
    assert first.exit_code == 0, (first.output, repr(first.exception))
    saved = torch.load(source / "SSL_latest.pth", map_location="cpu", weights_only=False)
    if step_format == "foreach_bfloat16":
        for group in saved["optimizer"]["param_groups"]:
            group["foreach"] = True
        for entry in saved["optimizer"]["state"].values():
            entry["step"] = entry["step"].to(torch.bfloat16)
    elif step_format == "requires_grad":
        for entry in saved["optimizer"]["state"].values():
            entry["step"].requires_grad_(True)
    checkpoint = tmp_path / "usable.pth"
    torch.save(saved, checkpoint)
    out = tmp_path / "continued"
    resumed = runner.invoke(app, fifo_cli + [
        "--out-dir", str(out), "--resume", str(checkpoint), "--max-epochs", "2",
        "--weight-decay", "1e100",
    ])
    assert resumed.exit_code == 0, (resumed.output, repr(resumed.exception))
    result = torch.load(out / "SSL_latest.pth", map_location="cpu", weights_only=False)
    assert result["relational_queue"]["completed_steps"] == 2
    assert all(group["weight_decay"] == 0.0 for group in result["optimizer"]["param_groups"])
    assert all(torch.isfinite(value).all() for value in result["student"].values())


def test_fifo_nonfinite_optimizer_result_does_not_commit_or_save(tmp_path, monkeypatch, fifo_cli):
    from otuformer.training.relational_queue import RelationalQueue
    real_step = torch.optim.AdamW.step
    committed = []
    real_commit = RelationalQueue.commit
    def commit(self, *args, **kwargs):
        committed.append(kwargs["step"])
        return real_commit(self, *args, **kwargs)
    def corrupt(self, *args, **kwargs):
        result = real_step(self, *args, **kwargs)
        if any(p.numel() > 4 for group in self.param_groups for p in group["params"]):
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("nan"))
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt)
    monkeypatch.setattr(RelationalQueue, "commit", commit)
    out = tmp_path / "out"
    result = CliRunner().invoke(app, fifo_cli + ["--out-dir", str(out), "--max-epochs", "1"])
    assert result.exit_code != 0
    assert "optimizer update became non-finite" in result.output
    assert committed == []
    assert not list(out.glob("SSL_*.pth"))
