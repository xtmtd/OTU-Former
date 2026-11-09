"""KoLeo must reach the optimizer, and update order must be observed, not read.

These checks fail if the regularizer is computed and logged but never added to
the optimized loss, and if the step/EMA/center ordering is rearranged. Both are
behavioural: no source-text matching.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from otuformer.training import pretrain_loop as trainer


def _images(root: Path, rows: int = 4) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index in range(rows):
        Image.new("RGB", (32, 32), (index * 47 % 255, 10, 90)).save(
            images / f"{index}.png"
        )
    manifest = root / "data.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        for index in range(rows):
            writer.writerow([f"{index}.png"])
    return images, manifest


def _args(
    root: Path, *, koleo: float, patch_loss: str = "consistency", device: str = "auto"
):
    images, manifest = _images(root)
    return argparse.Namespace(
        train_data=str(manifest),
        input_images_dir=str(images),
        out_dir=str(root / "out"),
        overwrite=False,
        model_name="vit_tiny_patch16_224",
        register_tokens=None,
        model_name_explicit=False,
        out_dim=8,
        out_dim_explicit=True,
        ssl_method="relational",
        ssl_method_explicit=True,
        koleo_weight=koleo,
        koleo_weight_explicit=True,
        max_epochs=1,
        lr=5e-4,
        weight_decay=0.0,
        warmup_epochs=0,
        no_norm_bias_weight_decay=True,
        no_norm_bias_weight_decay_explicit=False,
        augmentation="global-barcode",
        orientation_policy="sensitive",
        global_crop_size=32,
        local_crop_size=96,
        local_crops=0,
        mask_ratio="auto",
        lambda_local=1.5,
        lambda_mask=1.0,
        patch_loss=patch_loss,
        patch_loss_explicit=False,
        masking_strategy="random",
        masking_strategy_explicit=False,
        mask_ratio_explicit=False,
        ibot_prototypes=512,
        ibot_prototypes_explicit=False,
        lambda_mask_explicit=False,
        teacher_momentum=0.995,
        teacher_momentum_end=0.999,
        student_temp=0.1,
        teacher_temp_start=0.04,
        teacher_temp_end=0.07,
        teacher_temp_warmup_ratio=0.7,
        teacher_temp_warmup_ratio_explicit=False,
        relational_teacher_views="weak",
        relational_center_source=None,
        relational_pairing=None,
        log_every_n_steps=1000,
        save_every_epochs=1,
        keep_last_checkpoints=2,
        visualize_data=None,
        extract_size=None,
        metrics_sample_size=100,
        umap_n_neighbors=5,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=0,
        compute_embedding_metrics=False,
        batch_size=4,
        num_workers=0,
        cpus=1,
        device=device,
        seed=42,
        resume="",
    )


def _train(root: Path, **kwargs) -> dict:
    trainer.run_pretrain(_args(root, **kwargs))
    return torch.load(
        root / "out" / "SSL_latest.pth", map_location="cpu", weights_only=False
    )


def test_training_args_default_to_auto_and_allow_cpu_replay(tmp_path):
    assert _args(tmp_path, koleo=0.0).device == "auto"
    assert _args(tmp_path, koleo=0.0, device="cpu").device == "cpu"


def test_koleo_weight_changes_parameters_and_reaches_backbone_cls(tmp_path, monkeypatch):
    """The real loop must backpropagate KoLeo and change optimized parameters."""
    # Isolate the regularizer from backend variation in this exact comparison.
    off = _train(tmp_path / "off", koleo=0.0, device="cpu")
    gradients = []
    real = trainer._compute_koleo_raw

    def observe_gradients(student_tokens, ssl_config):
        total, sizes = real(student_tokens, ssl_config)
        gradients.extend(
            torch.autograd.grad(total, student_tokens, retain_graph=True)
        )
        return total, sizes

    monkeypatch.setattr(trainer, "_compute_koleo_raw", observe_gradients)
    on = _train(tmp_path / "on", koleo=0.5, device="cpu")

    assert len(gradients) == 2
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient[:, 0].abs().sum() > 0
        assert torch.count_nonzero(gradient[:, 1:]) == 0

    off_state = off["student"]
    on_state = on["student"]
    shared = [key for key in off_state if not key.startswith("backbone.blocks.")]
    assert shared, "expected shared student parameters"
    differing = [
        key for key in shared if not torch.equal(off_state[key], on_state[key])
    ]
    # A regularizer that never reaches backward leaves every shared weight
    # bit-identical, so an empty list here means the term is not wired in.
    assert differing, (
        "KoLeo weight changed nothing in the trained student; the term is not "
        "part of the optimized objective"
    )


def test_default_and_explicit_zero_koleo_skip_work_and_match(tmp_path, monkeypatch):
    """Omitting KoLeo and explicitly disabling it preserve the same CPU state."""
    def unexpected_koleo(*args, **kwargs):
        pytest.fail("disabled KoLeo must not compute neighbors")

    monkeypatch.setattr(trainer, "_compute_koleo_raw", unexpected_koleo)
    args = _args(tmp_path / "default", koleo=0.0, device="cpu")
    del args.koleo_weight
    del args.koleo_weight_explicit
    trainer.run_pretrain(args)
    default = torch.load(
        tmp_path / "default" / "out" / "SSL_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    disabled = _train(tmp_path / "zero", koleo=0.0, device="cpu")

    for key, value in default["student"].items():
        assert torch.equal(value, disabled["student"][key]), key


def test_optimizer_step_precedes_ema_and_relational_center_precedes_the_step(
    tmp_path, monkeypatch
):
    """Relational updates its feature center before the step, EMA after it."""
    events: list[str] = []
    real_step = torch.optim.AdamW.step
    real_ema = trainer.update_teacher
    real_center = trainer._update_teacher_center

    def spy_step(self, *args, **kwargs):
        events.append("optimizer_step")
        return real_step(self, *args, **kwargs)

    def spy_ema(student, teacher, momentum):
        events.append("ema")
        return real_ema(student, teacher, momentum)

    def spy_center(center, inputs):
        events.append("relational_center")
        return real_center(center, inputs)

    monkeypatch.setattr(torch.optim.AdamW, "step", spy_step)
    monkeypatch.setattr(trainer, "update_teacher", spy_ema)
    monkeypatch.setattr(trainer, "_update_teacher_center", spy_center)

    _train(tmp_path, koleo=0.0)

    assert events, "no optimizer step recorded"
    assert events.count("optimizer_step") == 1
    step = events.index("optimizer_step")
    assert "relational_center" in events, "relational center update missing"
    assert events.index("relational_center") < step, (
        "relational's feature center must be updated before the step, from the "
        "cached pre-step features"
    )
    assert "ema" in events
    assert step < events.index("ema"), "encoder EMA must follow the step"
