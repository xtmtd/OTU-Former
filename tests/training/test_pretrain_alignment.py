import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from otuformer.training import trainer
from otuformer.training.dataset import (
    build_finetune_augmentation_config,
    build_pretrain_augmentation_config,
)
from otuformer.utils.size import (
    resolve_training_image_size,
    validate_input_size,
)


def _schedule_args(max_epochs=5):
    return argparse.Namespace(
        max_epochs=max_epochs,
        lr=0.001,
        warmup_epochs=1,
        teacher_momentum=0.995,
        teacher_momentum_end=0.999,
        student_temp=0.1,
        teacher_temp_start=0.04,
        teacher_temp_end=0.07,
    )


def test_pretrain_extension_starts_at_saved_lr_and_never_increases():
    saved = {
        "original_max_epochs": 2,
        "total_steps": 8,
        "steps_per_epoch": 4,
        "warmup_steps": 4,
        "last_lr": 0.0002,
        "final_lr": 0.00001,
    }
    lr, momentum, _, teacher_temp, metadata = trainer._build_pretrain_schedules(
        _schedule_args(), saved, steps_per_epoch=6, global_step=8, completed_epochs=2
    )

    extension = lr[8:]
    assert extension[0] == pytest.approx(0.0002)
    assert all(left >= right for left, right in zip(extension, extension[1:]))
    assert momentum[8] == pytest.approx(0.999)
    assert teacher_temp[8] == pytest.approx(0.07)
    assert metadata["steps_per_epoch"] == 6


def test_same_plan_pretrain_resume_rejects_changed_loader_length():
    saved = {
        "original_max_epochs": 2,
        "total_steps": 8,
        "steps_per_epoch": 4,
        "warmup_steps": 4,
        "last_lr": 0.0002,
        "final_lr": 0.00001,
    }

    with pytest.raises(ValueError, match="same-plan resume"):
        trainer._build_pretrain_schedules(
            _schedule_args(2), saved, steps_per_epoch=5, global_step=4, completed_epochs=1
        )


def test_finetune_resume_rejects_different_class_mapping():
    with pytest.raises(ValueError, match="class labels differ"):
        trainer._validate_finetune_resume(
            {"epoch": 0, "class_labels": ["a", "b"]}, ["a", "c"], 2
        )


def test_legacy_pretrain_same_plan_resume_is_allowed():
    lr, *_ = trainer._build_pretrain_schedules(
        _schedule_args(5),
        None,
        steps_per_epoch=4,
        global_step=8,
        completed_epochs=2,
        original_max_epochs=5,
    )

    assert len(lr) == 20


def test_legacy_pretrain_resume_rejects_missing_schedule_metadata():
    with pytest.raises(ValueError, match="legacy checkpoint"):
        trainer._build_pretrain_schedules(
            _schedule_args(6),
            None,
            steps_per_epoch=4,
            global_step=8,
            completed_epochs=2,
            original_max_epochs=5,
        )


def test_finetune_resume_rejects_completed_epoch_target():
    with pytest.raises(ValueError, match="finetune-epochs must exceed"):
        trainer._validate_finetune_resume(
            {"epoch": 2, "class_labels": ["a", "b"]}, ["a", "b"], 3
        )


def test_teacher_temp_schedule_uses_70_percent_warmup():
    schedule = trainer._build_teacher_temp_schedule(
        total_iters=10,
        teacher_temp_start=0.04,
        teacher_temp_end=0.07,
    )
    assert len(schedule) == 10
    assert schedule[0] == pytest.approx(0.04)
    assert schedule[6] == pytest.approx(0.07)
    assert schedule[7] == pytest.approx(0.07)
    assert schedule[9] == pytest.approx(0.07)


def test_teacher_center_update_uses_current_iteration_global_outputs():
    teacher_center = torch.zeros(1, 2)
    teacher_global = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    updated = trainer._update_teacher_center(teacher_center, teacher_global)
    assert updated.shape == (1, 2)
    assert updated[0, 0] == pytest.approx(0.05)
    assert updated[0, 1] == pytest.approx(0.05)


def test_global_loss_uses_cartesian_pairing_when_cross_view_enabled(monkeypatch):
    student = [torch.tensor([[1.0]]), torch.tensor([[2.0]])]
    teacher = [torch.tensor([[3.0]]), torch.tensor([[4.0]])]
    calls = []

    def fake_ssl_loss(s, t, *_):
        calls.append((float(s.item()), float(t.item())))
        return torch.tensor(0.0)

    monkeypatch.setattr(trainer, "_ssl_loss", fake_ssl_loss)
    trainer._compute_global_loss(
        student, teacher, 0.1, 0.04, disable_cross_view_loss=False
    )
    assert set(calls) == {(1.0, 3.0), (1.0, 4.0), (2.0, 3.0), (2.0, 4.0)}


def test_local_loss_averages_all_local_to_teacher_global_pairs(monkeypatch):
    local_student = [
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[3.0]]),
    ]
    teacher = [torch.tensor([[4.0]]), torch.tensor([[5.0]])]
    calls = []

    def fake_ssl_loss(s, t, *_):
        calls.append((float(s.item()), float(t.item())))
        return torch.tensor(0.0)

    monkeypatch.setattr(trainer, "_ssl_loss", fake_ssl_loss)
    trainer._compute_local_loss(local_student, teacher, 0.1, 0.04)
    assert len(calls) == 6


def test_pretrain_metric_names_and_values_match_expected_mapping():
    """Metric keys in metrics.pretrain.csv must match reference log column names exactly."""
    import inspect

    src = inspect.getsource(trainer._compute_all_metrics)
    required_keys = [
        "NMI",
        "ARI",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "kNN_Acc_k1",
        "kNN_Acc_k5",
        "kNN_Acc_k20",
        "Linear_Probing_Acc",
        "mAP",
        "Silhouette_Score",
        "Purity",
    ]
    for key in required_keys:
        assert f'"{key}"' in src or f"'{key}'" in src, (
            f"Metric key '{key}' not found in _compute_all_metrics"
        )


def test_pretrain_encoder_exposes_projector_output_and_patch_tokens():
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(return_patch_tokens=True)
    x = torch.randn(2, 3, 224, 224)
    result = model(x)
    proj, patch_tokens = result
    assert proj.ndim == 2
    assert patch_tokens.ndim == 3


def test_pretrain_periodic_evaluation_uses_cls_token_features():
    """Embedding extraction for eval uses CLS token features from backbone tokens."""
    import inspect

    src = inspect.getsource(trainer._compute_embeddings_from_csv)
    assert "tokens = _extract_tokens(model, batch)" in src
    assert "emb = tokens[:, 0]" in src


def test_unlabeled_periodic_evaluation_still_generates_umap(
    tmp_path, monkeypatch, capsys
):
    calls = []

    class MetricsLogger:
        def log(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(
        trainer,
        "_compute_embeddings_from_csv",
        lambda **kwargs: (np.zeros((10, 4), dtype=np.float32), None),
    )
    monkeypatch.setattr(
        trainer,
        "run_umap",
        lambda embeddings, labels, out_path, **kwargs: calls.append(
            (embeddings, labels, out_path)
        ),
    )
    args = argparse.Namespace(
        visualize_data="images.csv",
        train_data="",
        input_images_dir="images",
        batch_size=2,
        num_workers=0,
        metrics_sample_size=100,
        seed=42,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=20,
    )

    trainer._compute_and_log_all_metrics(
        args=args,
        model=object(),
        device=torch.device("cpu"),
        epoch=0,
        logs_dir=tmp_path,
        metrics_logger=MetricsLogger(),
        eval_image_size=224,
    )

    assert len(calls) == 2
    assert calls[1][0].shape[0] == 10
    assert calls[1][1] is None
    assert "no labels provided" in capsys.readouterr().out


def test_zero_metrics_sample_size_does_not_subsample():
    embeddings = np.arange(20, dtype=np.float32).reshape(10, 2)

    sampled, labels = trainer._maybe_subsample_for_metrics(
        embeddings, None, max_samples=0, seed=42
    )

    assert sampled is embeddings
    assert labels is None


def test_single_class_periodic_evaluation_skips_supervised_metrics():
    fields = trainer._compute_all_metrics(
        np.zeros((4, 3), dtype=np.float32),
        np.array(["one"] * 4),
        compute_linear_probe=True,
    )

    assert all(value == "" for value in fields.values())


def test_unlabeled_periodic_evaluation_skips_small_sampled_umap(
    tmp_path, monkeypatch, capsys
):
    calls = []

    class MetricsLogger:
        def log(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        trainer,
        "_compute_embeddings_from_csv",
        lambda **kwargs: (np.zeros((20, 4), dtype=np.float32), None),
    )
    monkeypatch.setattr(
        trainer,
        "run_umap",
        lambda *args, **kwargs: calls.append(args),
    )
    args = argparse.Namespace(
        visualize_data="images.csv",
        train_data="",
        input_images_dir="images",
        batch_size=2,
        num_workers=0,
        metrics_sample_size=5,
        seed=42,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=20,
    )

    trainer._compute_and_log_all_metrics(
        args=args,
        model=object(),
        device=torch.device("cpu"),
        epoch=0,
        logs_dir=tmp_path,
        metrics_logger=MetricsLogger(),
        eval_image_size=224,
    )

    assert not calls
    assert "Skipping UMAP" in capsys.readouterr().out


def test_unlabeled_periodic_evaluation_uses_sampled_umap_features(
    tmp_path, monkeypatch
):
    calls = []

    class MetricsLogger:
        def log(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        trainer,
        "_compute_embeddings_from_csv",
        lambda **kwargs: (np.zeros((20, 4), dtype=np.float32), None),
    )
    monkeypatch.setattr(
        trainer,
        "run_umap",
        lambda embeddings, labels, out_path, **kwargs: calls.append(embeddings),
    )
    args = argparse.Namespace(
        visualize_data="images.csv",
        train_data="",
        input_images_dir="images",
        batch_size=2,
        num_workers=0,
        metrics_sample_size=10,
        seed=42,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=20,
    )

    trainer._compute_and_log_all_metrics(
        args=args,
        model=object(),
        device=torch.device("cpu"),
        epoch=0,
        logs_dir=tmp_path,
        metrics_logger=MetricsLogger(),
        eval_image_size=224,
    )

    assert calls[0].shape[0] == 10


def test_sampled_single_class_evaluation_reports_skipped_metrics(
    tmp_path, monkeypatch, capsys
):
    class MetricsLogger:
        def log(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        trainer,
        "_compute_embeddings_from_csv",
        lambda **kwargs: (
            np.zeros((10, 4), dtype=np.float32),
            np.array(["one"] * 10),
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_maybe_subsample_for_metrics",
        lambda embeddings, labels, max_samples, seed: (
            embeddings[:5],
            np.array(["one"] * 5),
        ),
    )
    args = argparse.Namespace(
        visualize_data="images.csv",
        train_data="",
        input_images_dir="images",
        batch_size=2,
        num_workers=0,
        metrics_sample_size=5,
        seed=42,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=20,
    )

    trainer._compute_and_log_all_metrics(
        args=args,
        model=object(),
        device=torch.device("cpu"),
        epoch=0,
        logs_dir=tmp_path,
        metrics_logger=MetricsLogger(),
        eval_image_size=224,
    )

    assert "fewer than two label classes after sampling" in capsys.readouterr().out


def test_teacher_momentum_updates_after_optimizer_step():
    """Momentum update must happen after optimizer.step(), not before."""
    # Verify ordering by inspecting the pretrain loop source for the update_teacher call
    import inspect

    src = inspect.getsource(trainer.run_pretrain)
    opt_pos = src.find("optimizer.step()")
    ema_pos = src.find("update_teacher(student, teacher")
    assert opt_pos != -1, "optimizer.step() not found in run_pretrain"
    assert ema_pos != -1, "update_teacher call not found in run_pretrain"
    assert opt_pos < ema_pos, "EMA update must occur after optimizer.step()"


def test_resolve_training_image_size_precedence():
    # 1. fine-tune augmentation_config.image_size
    assert (
        resolve_training_image_size(
            {
                "config": {
                    "augmentation_config": {"image_size": 384},
                    "image_size": 224,
                },
                "args": {"global_crop_size": 96},
            }
        )
        == 384
    )
    # 2. pretrain augmentation_config.global_crop.size
    assert (
        resolve_training_image_size(
            {
                "config": {
                    "augmentation_config": {"global_crop": {"size": 448}},
                    "image_size": 224,
                }
            }
        )
        == 448
    )
    # 3. this plan's fine-tune config.image_size
    assert resolve_training_image_size({"config": {"image_size": 352}}) == 352
    # 4. legacy args.global_crop_size
    assert resolve_training_image_size({"args": {"global_crop_size": 288}}) == 288
    # 5. fallback 224
    assert resolve_training_image_size({}) == 224
    assert resolve_training_image_size(None) == 224
    # invalid (bool / negative / non-integer) values are rejected (treated as absent)
    assert resolve_training_image_size({"config": {"image_size": True}}) == 224
    assert resolve_training_image_size({"config": {"image_size": -4}}) == 224
    assert resolve_training_image_size({"config": {"image_size": "big"}}) == 224


def test_validate_input_size_rejects_non_divisible_with_model_name():
    import timm

    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    with pytest.raises(
        ValueError,
        match="518 is not divisible by patch size 16 for vit_tiny_patch16_224; nearest valid: 512",
    ):
        validate_input_size(518, model, "vit_tiny_patch16_224")


def test_validate_input_size_passes_cnn_backbone_without_patch_embed():
    import timm

    model = timm.create_model("convnextv2_femto", pretrained=False)
    validate_input_size(224, model, "convnextv2_femto")  # no raise


def test_finetune_resolves_and_persists_checkpoint_size(tmp_path):
    from otuformer.training.model import OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=16,
        pretrained=False,
        img_size=32,
    )
    ckpt = {
        "model_state_dict": encoder.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 16},
        "args": {"global_crop_size": 32},
    }
    ckpt_path = tmp_path / "pretrain_32.pth"
    torch.save(ckpt, ckpt_path)

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (64, 64), color=(i * 50, 0, 0)).save(img_dir / f"img_{i}.jpg")
    df = pd.DataFrame(
        {
            "image": [f"img_{i}.jpg" for i in range(4)],
            "label": ["classA", "classA", "classB", "classB"],
        }
    )
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)

    args = argparse.Namespace(
        seed=42,
        cpus=0,
        device="cpu",
        out_dir=str(tmp_path / "ft_out"),
        checkpoint=str(ckpt_path),
        resume="",
        train_data=str(csv_path),
        input_images_dir=str(img_dir),
        model_name="vit_tiny_patch16_224",
        metric_embed_dim=16,
        finetune_epochs=1,
        finetune_lr=1e-4,
        freeze_ratio=0.7,
        loss="arcface",
        batch_size=2,
        num_workers=0,
        log_every_n_steps=50,
        save_every_epochs=1,
        keep_last_checkpoints=0,
        compute_embedding_metrics=False,
        extract_size=None,
        visualize_data="",
        metrics_sample_size=10000,
    )
    trainer.run_finetune(args)

    saved = torch.load(
        tmp_path / "ft_out" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["config"]["image_size"] == 32
    assert saved["config"]["augmentation_profile"] == "none"
    assert saved["config"]["augmentation_config"]["image_size"] == 32
    assert saved["config"]["augmentation_config"]["orientation_policy"] == "sensitive"
    assert "orientation_policy" not in saved["config"]
    # and a finetune checkpoint resolves back to 32 (not 224)
    assert resolve_training_image_size(saved) == 32


# --- Augmentation continuity ---------------------------------------------------------


class _DatasetConstructionReached(Exception):
    """Raised by fake datasets to stop a trainer before the DataLoader runs."""


def _augmentation_config(
    stage,
    profile,
    policy,
    *,
    global_crop_size=224,
    local_crop_size=96,
    local_crops=6,
    image_size=224,
):
    if stage == "pretrain":
        return build_pretrain_augmentation_config(
            profile,
            global_crop_size,
            local_crop_size,
            local_crops,
            orientation_policy=policy,
        )
    return build_finetune_augmentation_config(
        profile, image_size, orientation_policy=policy
    )


def _augmented_checkpoint(stage, profile, policy, **kwargs):
    config = _augmentation_config(stage, profile, policy, **kwargs)
    return {
        "config": {
            "model_name": "vit_tiny_patch16_224",
            "augmentation_profile": profile,
            "augmentation_config": config,
        }
    }


def _write_pretrain_checkpoint(
    path,
    *,
    image_size=32,
    config_extra=None,
    args_extra=None,
    model_name="vit_tiny_patch16_224",
    out_dim=16,
):
    from otuformer.training.model import OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name=model_name,
        out_dim=out_dim,
        pretrained=False,
        img_size=image_size,
    )
    config = {"model_name": model_name, "out_dim": out_dim}
    if config_extra:
        config.update(config_extra)
    args = {"global_crop_size": image_size}
    if args_extra:
        args.update(args_extra)
    torch.save(
        {"model_state_dict": encoder.state_dict(), "config": config, "args": args},
        path,
    )
    return path


def _write_historical_sft_checkpoint(path, *, image_size=32, out_dim=16):
    from otuformer.training.loss import ArcFaceLoss
    from otuformer.training.model import OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=out_dim,
        pretrained=False,
        img_size=image_size,
    )
    loss_fn = ArcFaceLoss(embed_dim=out_dim, num_classes=2)
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": encoder.state_dict(),
            "loss_state_dict": loss_fn.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": out_dim,
                "image_size": image_size,
            },
            "class_labels": ["classA", "classB"],
        },
        path,
    )
    return path


def _finetune_args(tmp_path, checkpoint_path, **overrides):
    values = dict(
        seed=42,
        cpus=0,
        device="cpu",
        out_dir=str(tmp_path / "ft_out"),
        checkpoint=str(checkpoint_path),
        resume="",
        train_data="labels.csv",
        input_images_dir=str(tmp_path),
        model_name="vit_tiny_patch16_224",
        metric_embed_dim=16,
        finetune_lr=1e-4,
        loss="arcface",
        freeze_ratio=0.7,
        extract_size=None,
        compute_embedding_metrics=False,
        augmentation=None,
        orientation_policy=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_historical_sft_new_run_can_resume_saved_projection_checkpoint(tmp_path):
    checkpoint = _write_historical_sft_checkpoint(tmp_path / "historical.pth")
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (64, 64), color=(i * 50, 0, 0)).save(img_dir / f"img_{i}.jpg")
    labels = pd.DataFrame(
        {
            "image": [f"img_{i}.jpg" for i in range(4)],
            "label": ["classA", "classA", "classB", "classB"],
        }
    )
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    first_args = _finetune_args(
        tmp_path,
        checkpoint,
        out_dir=str(tmp_path / "first"),
        train_data=str(labels_path),
        input_images_dir=str(img_dir),
        finetune_epochs=1,
        batch_size=2,
        num_workers=0,
        log_every_n_steps=100,
        save_every_epochs=1,
        keep_last_checkpoints=0,
        weight_decay=1e-4,
        metric_head_lr=None,
    )
    trainer.run_finetune(first_args)
    first_path = tmp_path / "first" / "finetune_latest.pth"
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    assert first["config"]["embedding_head"] == "projection_mlp_2048"
    assert len(first["optimizer"]["param_groups"]) == 3

    resume_args = _finetune_args(
        tmp_path,
        checkpoint,
        out_dir=str(tmp_path / "second"),
        resume=str(first_path),
        train_data=str(labels_path),
        input_images_dir=str(img_dir),
        finetune_epochs=2,
        batch_size=2,
        num_workers=0,
        log_every_n_steps=100,
        save_every_epochs=1,
        keep_last_checkpoints=0,
        weight_decay=1e-4,
        metric_head_lr=None,
    )
    trainer.run_finetune(resume_args)
    second = torch.load(
        tmp_path / "second" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert second["config"]["embedding_head"] == "projection_mlp_2048"
    assert len(second["optimizer"]["param_groups"]) == 3


@pytest.mark.parametrize(
    ("stage", "expected_profile"),
    [("pretrain", "global-barcode"), ("finetune", "none")],
)
def test_new_run_uses_stage_defaults(stage, expected_profile):
    profile = trainer._select_augmentation_profile(None, None, stage=stage)
    policy = trainer._select_orientation_policy(None, None, stage=stage)
    config = _augmentation_config(stage, profile, policy)
    assert (profile, policy) == (expected_profile, "sensitive")
    assert trainer._validate_augmentation_config(
        profile, config, None, stage=stage
    ) == config


@pytest.mark.parametrize(
    ("stage", "legacy_profile", "expected_policy"),
    [
        ("pretrain", "legacy", "invariant"),
        ("finetune", "none", "sensitive"),
    ],
)
def test_old_checkpoint_maps_to_compatible_augmentation(
    stage, legacy_profile, expected_policy
):
    checkpoint = {"config": {"model_name": "vit_tiny_patch16_224"}}
    profile = trainer._select_augmentation_profile(None, checkpoint, stage=stage)
    policy = trainer._select_orientation_policy(None, checkpoint, stage=stage)
    config = _augmentation_config(stage, profile, policy)
    assert (profile, policy) == (legacy_profile, expected_policy)
    assert trainer._validate_augmentation_config(
        profile, config, checkpoint, stage=stage
    ) == config


def test_pretrain_resume_inherits_saved_augmentation():
    checkpoint = _augmented_checkpoint("pretrain", "global-barcode", "invariant")
    profile = trainer._select_augmentation_profile(None, checkpoint, stage="pretrain")
    policy = trainer._select_orientation_policy(
        None, checkpoint, stage="pretrain", resume=True
    )
    config = _augmentation_config("pretrain", profile, policy)
    assert (profile, policy) == ("global-barcode", "invariant")
    assert (
        trainer._validate_augmentation_config(
            profile, config, checkpoint, stage="pretrain"
        )
        == config
    )


def test_pretrain_resume_accepts_explicit_matching_augmentation():
    checkpoint = _augmented_checkpoint("pretrain", "color-robust", "sensitive")
    profile = trainer._select_augmentation_profile(
        "color-robust", checkpoint, stage="pretrain"
    )
    policy = trainer._select_orientation_policy(
        "sensitive", checkpoint, stage="pretrain", resume=True
    )
    config = _augmentation_config("pretrain", profile, policy)
    assert (profile, policy) == ("color-robust", "sensitive")
    assert (
        trainer._validate_augmentation_config(
            profile, config, checkpoint, stage="pretrain"
        )
        == config
    )


def test_pretrain_resume_rejects_different_profile_and_policy():
    checkpoint = _augmented_checkpoint("pretrain", "global-barcode", "invariant")
    with pytest.raises(ValueError, match="Cannot resume") as excinfo:
        trainer._select_augmentation_profile(
            "color-robust", checkpoint, stage="pretrain"
        )
    assert "new run" in str(excinfo.value)
    with pytest.raises(ValueError, match="Cannot resume") as excinfo:
        trainer._select_orientation_policy(
            "sensitive", checkpoint, stage="pretrain", resume=True
        )
    assert "new run" in str(excinfo.value)


def test_finetune_resume_inherits_and_matches_saved_augmentation():
    checkpoint = _augmented_checkpoint(
        "finetune", "conservative", "sensitive", image_size=224
    )
    profile = trainer._select_augmentation_profile(None, checkpoint, stage="finetune")
    policy = trainer._select_orientation_policy(
        None, checkpoint, stage="finetune", resume=True
    )
    config = _augmentation_config("finetune", profile, policy, image_size=224)
    assert (profile, policy) == ("conservative", "sensitive")
    assert (
        trainer._validate_augmentation_config(
            profile, config, checkpoint, stage="finetune"
        )
        == config
    )
    assert (
        trainer._select_augmentation_profile(
            "conservative", checkpoint, stage="finetune"
        )
        == "conservative"
    )
    assert (
        trainer._select_orientation_policy(
            "sensitive", checkpoint, stage="finetune", resume=True
        )
        == "sensitive"
    )


def test_resume_rejects_changed_expanded_configuration():
    checkpoint = _augmented_checkpoint("pretrain", "global-barcode", "invariant")
    saved = checkpoint["config"]["augmentation_config"]
    changed = json.loads(json.dumps(saved))
    changed["color_jitter"]["brightness"] = 0.9
    with pytest.raises(ValueError, match="configuration differs"):
        trainer._validate_augmentation_config(
            "global-barcode", changed, checkpoint, stage="pretrain"
        )
    changed_policy = json.loads(json.dumps(saved))
    changed_policy["orientation_policy"] = "sensitive"
    with pytest.raises(ValueError, match="configuration differs"):
        trainer._validate_augmentation_config(
            "global-barcode", changed_policy, checkpoint, stage="pretrain"
        )


def test_new_run_orientation_policy_defaults_and_override():
    assert (
        trainer._select_orientation_policy(None, None, stage="pretrain")
        == "sensitive"
    )
    assert (
        trainer._select_orientation_policy("sensitive", None, stage="pretrain")
        == "sensitive"
    )
    assert (
        trainer._select_orientation_policy("invariant", None, stage="pretrain")
        == "invariant"
    )
    assert (
        trainer._select_orientation_policy(None, None, stage="finetune")
        == "sensitive"
    )
    assert (
        trainer._select_orientation_policy("sensitive", None, stage="finetune")
        == "sensitive"
    )
    assert (
        trainer._select_orientation_policy("invariant", None, stage="finetune")
        == "invariant"
    )


def test_new_legacy_run_records_explicit_sensitive_without_changing_transform():
    profile = trainer._select_augmentation_profile("legacy", None, stage="pretrain")
    policy = trainer._select_orientation_policy("sensitive", None, stage="pretrain")
    config = _augmentation_config("pretrain", profile, policy)
    assert (profile, policy) == ("legacy", "sensitive")
    assert config["orientation_policy"] == "sensitive"
    assert config["rotation"]["degrees"] == [-180.0, 180.0]
    assert config["horizontal_flip_probability"] == 0.5


def test_old_pretrain_resume_profile_and_policy_compatibility():
    checkpoint = {"config": {"model_name": "vit_tiny_patch16_224"}}
    assert (
        trainer._select_augmentation_profile(None, checkpoint, stage="pretrain")
        == "legacy"
    )
    assert (
        trainer._select_augmentation_profile("legacy", checkpoint, stage="pretrain")
        == "legacy"
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer._select_augmentation_profile(
            "global-barcode", checkpoint, stage="pretrain"
        )
    assert (
        trainer._select_orientation_policy(
            None, checkpoint, stage="pretrain", resume=True
        )
        == "invariant"
    )
    assert (
        trainer._select_orientation_policy(
            "invariant", checkpoint, stage="pretrain", resume=True
        )
        == "invariant"
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer._select_orientation_policy(
            "sensitive", checkpoint, stage="pretrain", resume=True
        )


def test_old_finetune_resume_compatibility():
    checkpoint = {"config": {"model_name": "vit_tiny_patch16_224"}}
    assert (
        trainer._select_augmentation_profile(None, checkpoint, stage="finetune")
        == "none"
    )
    assert (
        trainer._select_augmentation_profile("none", checkpoint, stage="finetune")
        == "none"
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer._select_augmentation_profile(
            "conservative", checkpoint, stage="finetune"
        )
    assert (
        trainer._select_orientation_policy(
            None, checkpoint, stage="finetune", resume=True
        )
        == "invariant"
    )
    assert (
        trainer._select_orientation_policy(
            "invariant", checkpoint, stage="finetune", resume=True
        )
        == "invariant"
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer._select_orientation_policy(
            "sensitive", checkpoint, stage="finetune", resume=True
        )


def test_old_pretrain_checkpoint_finetune_initialization():
    checkpoint = {"config": {"model_name": "vit_tiny_patch16_224"}}
    assert trainer._select_augmentation_profile(None, None, stage="finetune") == "none"
    assert (
        trainer._select_augmentation_profile("conservative", None, stage="finetune")
        == "conservative"
    )
    assert (
        trainer._select_orientation_policy(
            None, checkpoint, stage="finetune", resume=False
        )
        == "sensitive"
    )
    assert (
        trainer._select_orientation_policy(
            "invariant", checkpoint, stage="finetune", resume=False
        )
        == "invariant"
    )
    assert (
        trainer._select_orientation_policy(
            "sensitive", checkpoint, stage="finetune", resume=False
        )
        == "sensitive"
    )


def test_new_style_pretrain_checkpoint_finetune_initialization_inherits_policy_only():
    checkpoint = _augmented_checkpoint("pretrain", "color-robust", "sensitive")
    assert trainer._select_augmentation_profile(None, None, stage="finetune") == "none"
    assert (
        trainer._select_orientation_policy(
            None, checkpoint, stage="finetune", resume=False
        )
        == "sensitive"
    )
    assert (
        trainer._select_orientation_policy(
            "invariant", checkpoint, stage="finetune", resume=False
        )
        == "invariant"
    )


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"config": {"augmentation_profile": "legacy"}},
        {"config": {"augmentation_config": {"profile": "legacy"}}},
        {
            "config": {
                "augmentation_profile": 7,
                "augmentation_config": {"profile": "legacy"},
            }
        },
        {"config": {"augmentation_profile": "legacy", "augmentation_config": ["nope"]}},
        # top-level profile disagrees with the self-contained nested profile
        {
            "config": {
                "augmentation_profile": "legacy",
                "augmentation_config": build_pretrain_augmentation_config(
                    "global-barcode", 224, 96, 6
                ),
            }
        },
        # nested config omits its self-contained profile
        {
            "config": {
                "augmentation_profile": "legacy",
                "augmentation_config": {"orientation_policy": "invariant"},
            }
        },
        # saved orientation policy is not a known policy
        {
            "config": {
                "augmentation_profile": "legacy",
                "augmentation_config": {
                    "profile": "legacy",
                    "orientation_policy": "bogus",
                },
            }
        },
        # saved orientation policy is missing / not a string
        {
            "config": {
                "augmentation_profile": "legacy",
                "augmentation_config": {
                    "profile": "legacy",
                    "orientation_policy": None,
                },
            }
        },
    ],
)
def test_malformed_augmentation_metadata(checkpoint):
    with pytest.raises(ValueError, match="malformed augmentation metadata"):
        trainer._select_augmentation_profile(None, checkpoint, stage="pretrain")
    with pytest.raises(ValueError, match="malformed augmentation metadata"):
        trainer._select_orientation_policy(
            None, checkpoint, stage="pretrain", resume=True
        )
    with pytest.raises(ValueError, match="malformed augmentation metadata"):
        trainer._validate_augmentation_config(
            "legacy",
            _augmentation_config("pretrain", "legacy", "invariant"),
            checkpoint,
            stage="pretrain",
        )


def test_saved_profile_must_match_stage():
    pretrain_style = {
        "config": {
            "augmentation_profile": "global-barcode",
            "augmentation_config": build_pretrain_augmentation_config(
                "global-barcode", 224, 96, 6
            ),
        }
    }
    finetune_style = {
        "config": {
            "augmentation_profile": "none",
            "augmentation_config": build_finetune_augmentation_config("none", 224),
        }
    }

    with pytest.raises(ValueError, match="not valid for finetune"):
        trainer._select_augmentation_profile(None, pretrain_style, stage="finetune")
    with pytest.raises(ValueError, match="not valid for pretrain"):
        trainer._select_augmentation_profile(None, finetune_style, stage="pretrain")


@pytest.mark.parametrize("stage", ["other", "pretraining", ""])
def test_unsupported_stage_raises(stage):
    with pytest.raises(ValueError, match="Unsupported augmentation stage"):
        trainer._select_augmentation_profile(None, None, stage=stage)
    with pytest.raises(ValueError, match="Unsupported augmentation stage"):
        trainer._select_orientation_policy(None, None, stage=stage)
    with pytest.raises(ValueError, match="Unsupported augmentation stage"):
        trainer._validate_augmentation_config(
            "none",
            _augmentation_config("finetune", "none", "invariant"),
            None,
            stage=stage,
        )


def test_pretrain_local_views_new_run_defaults_and_validation():
    assert trainer._resolve_pretrain_local_views(None, None, None) == (96, 6)
    assert trainer._resolve_pretrain_local_views(128, 4, None) == (128, 4)
    assert trainer._resolve_pretrain_local_views(128, 0, None) == (128, 0)
    for bad_size in (0, -1, True, 1.5, "96"):
        with pytest.raises(ValueError):
            trainer._resolve_pretrain_local_views(bad_size, None, None)
    for bad_crops in (-1, False, 1.5, "6"):
        with pytest.raises(ValueError):
            trainer._resolve_pretrain_local_views(None, bad_crops, None)


def test_pretrain_resume_inherits_saved_local_views():
    checkpoint = _augmented_checkpoint(
        "pretrain",
        "global-barcode",
        "invariant",
        local_crop_size=128,
        local_crops=4,
    )
    assert trainer._resolve_pretrain_local_views(None, None, checkpoint) == (128, 4)
    assert trainer._resolve_pretrain_local_views(128, 4, checkpoint) == (128, 4)
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer._resolve_pretrain_local_views(96, None, checkpoint)
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer._resolve_pretrain_local_views(None, 6, checkpoint)


def test_old_pretrain_local_views_inherit_and_fallback():
    checkpoint = {
        "config": {"model_name": "vit_tiny_patch16_224"},
        "args": {"local_crop_size": 128, "local_crops": 0},
    }
    assert trainer._resolve_pretrain_local_views(None, None, checkpoint) == (128, 0)
    assert trainer._resolve_pretrain_local_views(128, 0, checkpoint) == (128, 0)

    missing = {"config": {"model_name": "vit_tiny_patch16_224"}, "args": {}}
    assert trainer._resolve_pretrain_local_views(None, None, missing) == (96, 6)
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer._resolve_pretrain_local_views(128, None, missing)

    invalid = {
        "config": {"model_name": "vit_tiny_patch16_224"},
        "args": {"local_crop_size": 0, "local_crops": True},
    }
    assert trainer._resolve_pretrain_local_views(None, None, invalid) == (96, 6)
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer._resolve_pretrain_local_views(None, 0, invalid)


def test_augmentation_metadata_round_trips_training_size():
    ft_config = build_finetune_augmentation_config("none", 384)
    ft_checkpoint = {
        "config": {"augmentation_profile": "none", "augmentation_config": ft_config}
    }
    assert resolve_training_image_size(ft_checkpoint) == 384
    pt_config = build_pretrain_augmentation_config("global-barcode", 448, 96, 6)
    pt_checkpoint = {
        "config": {
            "augmentation_profile": "global-barcode",
            "augmentation_config": pt_config,
        }
    }
    assert resolve_training_image_size(pt_checkpoint) == 448


def test_pretrain_passes_resolved_augmentation_to_dataset(
    tmp_path, monkeypatch, capsys
):
    captured = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        raise _DatasetConstructionReached

    monkeypatch.setattr(trainer, "MultiCropDataset", fake_dataset)
    args = argparse.Namespace(
        seed=42,
        cpus=0,
        device="cpu",
        out_dir=str(tmp_path / "pretrain_out"),
        model_name="vit_tiny_patch16_224",
        out_dim=16,
        global_crop_size=224,
        local_crop_size=96,
        local_crops=2,
        resume="",
        augmentation=None,
        orientation_policy=None,
        train_data="",
        input_images_dir=str(tmp_path),
    )

    with pytest.raises(_DatasetConstructionReached):
        trainer.run_pretrain(args)

    assert captured["augmentation_profile"] == "global-barcode"
    assert captured["orientation_policy"] == "sensitive"
    assert captured["local_crop_size"] == 96
    assert captured["local_crops"] == 2
    out = capsys.readouterr().out
    assert "Augmentation profile: global-barcode" in out
    assert "Augmentation config:" in out
    assert '"profile": "global-barcode"' in out
    assert '"orientation_policy": "sensitive"' in out


def test_pretrain_passes_explicit_augmentation_to_dataset(tmp_path, monkeypatch):
    captured = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        raise _DatasetConstructionReached

    monkeypatch.setattr(trainer, "MultiCropDataset", fake_dataset)
    args = argparse.Namespace(
        seed=42,
        cpus=0,
        device="cpu",
        out_dir=str(tmp_path / "pretrain_out"),
        model_name="vit_tiny_patch16_224",
        out_dim=16,
        global_crop_size=224,
        local_crop_size=96,
        local_crops=0,
        resume="",
        augmentation="legacy",
        orientation_policy="sensitive",
        train_data="",
        input_images_dir=str(tmp_path),
    )

    with pytest.raises(_DatasetConstructionReached):
        trainer.run_pretrain(args)

    assert captured["augmentation_profile"] == "legacy"
    assert captured["orientation_policy"] == "sensitive"
    assert captured["local_crops"] == 0


def test_finetune_initialization_does_not_inherit_pretrain_augmentation(
    tmp_path, monkeypatch, capsys
):
    pretrain_config = build_pretrain_augmentation_config(
        "color-robust", 32, 96, 2, orientation_policy="sensitive"
    )
    checkpoint_path = _write_pretrain_checkpoint(
        tmp_path / "pretrain.pth",
        image_size=32,
        config_extra={
            "augmentation_profile": "color-robust",
            "augmentation_config": pretrain_config,
        },
    )
    captured = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        raise _DatasetConstructionReached

    monkeypatch.setattr(trainer, "MetricDataset", fake_dataset)
    args = _finetune_args(tmp_path, checkpoint_path)

    with pytest.raises(_DatasetConstructionReached):
        trainer.run_finetune(args)

    assert captured["augmentation_profile"] == "none"
    assert captured["orientation_policy"] == "sensitive"
    assert captured["image_size"] == 32
    out = capsys.readouterr().out
    assert "Augmentation profile: none" in out
    assert '"orientation_policy": "sensitive"' in out


def test_finetune_passes_resolved_training_size_to_model_and_dataset(
    tmp_path, monkeypatch, capsys
):
    checkpoint_path = _write_pretrain_checkpoint(tmp_path / "legacy_pretrain.pth")
    captured_model = {}
    captured_dataset = {}

    real_encoder = trainer.OTUFormerEncoder

    def fake_encoder(*args, **kwargs):
        captured_model["img_size"] = kwargs.get("img_size")
        return real_encoder(*args, **kwargs)

    def fake_dataset(**kwargs):
        captured_dataset.update(kwargs)
        raise _DatasetConstructionReached

    monkeypatch.setattr(trainer, "OTUFormerEncoder", fake_encoder)
    monkeypatch.setattr(trainer, "MetricDataset", fake_dataset)
    args = _finetune_args(tmp_path, checkpoint_path)

    with pytest.raises(_DatasetConstructionReached):
        trainer.run_finetune(args)

    assert captured_model["img_size"] == 32
    assert captured_dataset["image_size"] == 32
    assert captured_dataset["augmentation_profile"] == "none"
    assert captured_dataset["orientation_policy"] == "sensitive"
    out = capsys.readouterr().out
    assert "Augmentation profile: none" in out
    assert '"image_size": 32' in out


# --- Rotation/reflection validation script ------------------------------------------

_VALIDATION_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "validate_augmentation_rotation.py"
)


def _load_rotation_validation_script():
    spec = importlib.util.spec_from_file_location(
        "validate_augmentation_rotation", _VALIDATION_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rotation_validation_script_summarizes_similarities():
    script = _load_rotation_validation_script()
    summary = script.summarize_similarities(np.array([0.1, 0.2, 0.3]))
    assert summary["median"] == pytest.approx(0.2)
    assert summary["p10"] == pytest.approx(0.12)
    assert summary["minimum"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "values",
    [np.array([]), np.array([0.1, np.nan]), np.array([0.2, np.inf])],
)
def test_rotation_validation_script_rejects_empty_or_non_finite(values):
    script = _load_rotation_validation_script()
    with pytest.raises(ValueError, match="non-empty and finite"):
        script.summarize_similarities(values)


def test_rotation_validation_script_reflection_summary():
    script = _load_rotation_validation_script()
    reference = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    reflected = np.array([[1.0, 0.0], [1.0, 0.0], [0.6, 0.8]])

    summary = script.summarize_paired_similarities(reference, reflected)

    assert summary["median"] == pytest.approx(0.6)
    assert summary["p10"] == pytest.approx(0.12)
    assert summary["minimum"] == pytest.approx(0.0)


def test_rotation_validation_script_accepts_shared_fixed_evaluation_size():
    script = _load_rotation_validation_script()
    baseline = {
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 16},
        "args": {"global_crop_size": 224},
    }
    candidate = {
        "config": {
            "model_name": "vit_tiny_patch16_224",
            "out_dim": 16,
            "augmentation_config": {"global_crop": {"size": 448}},
        }
    }

    assert script.resolve_evaluation_size(baseline) == 224
    assert script.resolve_evaluation_size(candidate) == 448
    with pytest.raises(ValueError, match="differ"):
        script.require_matching_evaluation_sizes(224, 448)

    shared = script.resolve_evaluation_size(baseline, 224)
    assert shared == 224
    assert script.resolve_evaluation_size(candidate, 224) == 224
    assert script.require_matching_evaluation_sizes(shared, 224) == 224

    for bad in (True, 224.0, 0, -4):
        with pytest.raises(ValueError, match="positive integer"):
            script.resolve_evaluation_size(baseline, bad)


def test_different_image_sampling_cap_is_fixed_seed_deterministic():
    script = _load_rotation_validation_script()
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(600, 8))
    assert 600 * 599 // 2 > script.DIFFERENT_IMAGE_MAX_PAIRS

    first = script.different_image_similarity_summary(embeddings, seed=7)
    second = script.different_image_similarity_summary(embeddings, seed=7)
    third = script.different_image_similarity_summary(embeddings, seed=8)

    assert first == second
    assert 0 < first["pair_count"] <= script.DIFFERENT_IMAGE_MAX_PAIRS
    assert first != third


def test_orientation_policy_note_describes_policies_and_legacy_sensitive():
    script = _load_rotation_validation_script()
    assert "[-15, 15]" in script.orientation_policy_note("sensitive", "global-barcode")
    assert "full rotation" in script.orientation_policy_note("invariant", "global-barcode")
    assert "legacy checkpoint" in script.orientation_policy_note(None, None)

    legacy_note = script.orientation_policy_note("sensitive", "legacy")
    assert "legacy" in legacy_note
    assert "historical" in legacy_note
    assert "does not apply" in legacy_note


def _write_tiny_ft_data(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    for i in range(4):
        Image.new("RGB", (64, 64), color=(i * 50, 0, 0)).save(img_dir / f"img_{i}.jpg")
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "image": [f"img_{i}.jpg" for i in range(4)],
            "label": ["classA", "classA", "classB", "classB"],
        }
    ).to_csv(labels_path, index=False)
    return img_dir, labels_path


def _write_arcface_checkpoint(path, *, image_size=32, out_dim=16):
    from otuformer.training.model import ArcFaceEmbeddingHead, OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=out_dim,
        pretrained=False,
        img_size=image_size,
    )
    encoder.projector = ArcFaceEmbeddingHead(encoder.backbone.num_features, out_dim)
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": out_dim,
                "metric_embed_dim": out_dim,
                "image_size": image_size,
                "embedding_head": "arcface_mlp_512",
            },
        },
        path,
    )
    return path


def _finetune_overrides(tmp_path, out_name="ft_out", **overrides):
    img_dir, labels_path = _write_tiny_ft_data(tmp_path)
    values = dict(
        out_dir=str(tmp_path / out_name),
        train_data=str(labels_path),
        input_images_dir=str(img_dir),
        batch_size=2,
        num_workers=0,
        log_every_n_steps=100,
        save_every_epochs=1,
        keep_last_checkpoints=0,
        weight_decay=1e-4,
        metric_head_lr=None,
    )
    values.update(overrides)
    return values


def _run_frozen_finetune(tmp_path, checkpoint_path, out_name, **overrides):
    """One fine-tune epoch with a zeroed optimizer.

    With ``lr=0`` and ``weight_decay=0`` the optimizer leaves parameters
    untouched, so the saved head is exactly the head that was loaded.
    """
    args = _finetune_args(
        tmp_path,
        checkpoint_path,
        finetune_epochs=1,
        **_finetune_overrides(
            tmp_path,
            out_name,
            finetune_lr=0.0,
            metric_head_lr=0.0,
            weight_decay=0.0,
            **overrides,
        ),
    )
    trainer.run_finetune(args)
    return torch.load(
        Path(args.out_dir) / "finetune_latest.pth", map_location="cpu", weights_only=False
    )


def test_arcface_checkpoint_init_keeps_trained_embedding_head(tmp_path):
    source = _write_arcface_checkpoint(tmp_path / "src.pth", image_size=32, out_dim=16)
    source_state = torch.load(source, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]

    saved = _run_frozen_finetune(tmp_path, source, "from_arcface")

    assert saved["config"]["embedding_head"] == "arcface_mlp_512"
    assert torch.equal(
        saved["model_state_dict"]["projector.net.0.weight"],
        source_state["projector.net.0.weight"],
    )
    assert torch.equal(
        saved["model_state_dict"]["projector.net.2.weight"],
        source_state["projector.net.2.weight"],
    )


def test_metric_embed_dim_resizes_the_arcface_head(tmp_path):
    from otuformer.embedding.extractor import _load_model

    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)

    saved = _run_frozen_finetune(tmp_path, ssl, "resized", metric_embed_dim=8)

    assert saved["config"]["out_dim"] == 8
    assert saved["config"]["metric_embed_dim"] == 8
    assert saved["model_state_dict"]["projector.net.2.weight"].shape == (8, 512)
    assert saved["loss_state_dict"]["head.weight"].shape == (2, 8)
    # A resized checkpoint must stay readable by extract/export/cam.
    assert saved["model_state_dict"]["center"].shape == (1, 8)
    loaded, _ = _load_model(
        tmp_path / "resized" / "finetune_latest.pth",
        "vit_tiny_patch16_224",
        torch.device("cpu"),
    )
    assert loaded.projector.net[2].weight.shape == (8, 512)


def test_finetune_resume_rejects_a_changed_embedding_dim(tmp_path):
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)
    first = _finetune_args(
        tmp_path, ssl, finetune_epochs=1, **_finetune_overrides(tmp_path, "f1")
    )
    trainer.run_finetune(first)

    resume = _finetune_args(
        tmp_path,
        ssl,
        resume=str(tmp_path / "f1" / "finetune_latest.pth"),
        finetune_epochs=2,
        metric_embed_dim=8,
        **_finetune_overrides(tmp_path, "f2"),
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer.run_finetune(resume)


def test_metric_embed_dim_is_rejected_for_a_projection_head_checkpoint(tmp_path):
    sft = _write_historical_sft_checkpoint(tmp_path / "sft.pth", image_size=32, out_dim=16)

    args = _finetune_args(
        tmp_path,
        sft,
        finetune_epochs=1,
        metric_embed_dim=8,
        **_finetune_overrides(tmp_path, "bad_dim"),
    )

    with pytest.raises(ValueError, match="cannot be applied to a ProjectionHead"):
        trainer.run_finetune(args)


def test_finetune_resume_rejects_a_changed_freeze_ratio(tmp_path):
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)

    first = _finetune_args(
        tmp_path,
        ssl,
        finetune_epochs=1,
        freeze_ratio=0.5,
        **_finetune_overrides(tmp_path, "f1"),
    )
    trainer.run_finetune(first)
    saved = torch.load(
        tmp_path / "f1" / "finetune_latest.pth", map_location="cpu", weights_only=False
    )
    assert saved["config"]["freeze_ratio"] == 0.5

    resume = _finetune_args(
        tmp_path,
        ssl,
        resume=str(tmp_path / "f1" / "finetune_latest.pth"),
        finetune_epochs=2,
        freeze_ratio=0.7,
        **_finetune_overrides(tmp_path, "f2"),
    )
    with pytest.raises(ValueError, match="freeze_ratio"):
        trainer.run_finetune(resume)


_V070_ENHANCED_HEADER = [
    "epoch",
    "split",
    "NMI",
    "ARI",
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "kNN_Acc_k1",
    "kNN_Acc_k5",
    "kNN_Acc_k20",
    "Linear_Probing_Acc",
    "mAP",
    "Silhouette_Score",
    "Purity",
]


def _write_v070_enhanced_csv(path):
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_V070_ENHANCED_HEADER)
        writer.writerow(["1", "train"] + ["0.10"] * (len(_V070_ENHANCED_HEADER) - 2))


def test_maybe_subsample_for_metrics_returns_sorted_indices():
    embeddings = np.arange(30, dtype=np.float32).reshape(-1, 1)
    labels = np.arange(30)

    sampled, sampled_labels = trainer._maybe_subsample_for_metrics(
        embeddings, labels, max_samples=10, seed=42
    )

    expected = np.sort(np.random.default_rng(42).choice(30, size=10, replace=False))
    assert np.array_equal(sampled[:, 0].astype(int), expected)
    assert np.array_equal(sampled_labels, expected)


def test_compute_all_metrics_uses_empty_strings_for_unavailable_metrics():
    embeddings = np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32)
    labels = np.array(["a", "a", "a", "b", "b", "b"])

    fields = trainer._compute_all_metrics(embeddings, labels, compute_linear_probe=True)

    assert fields["Recall@10"] == ""
    assert fields["kNN_Acc_k20"] == ""
    assert isinstance(fields["Linear_Probing_Acc"], float)
    assert isinstance(fields["Linear_Probing_Balanced_Acc"], float)
    for value in fields.values():
        assert value == "" or isinstance(value, float)


def test_enhanced_metrics_logger_keeps_history_and_adds_balanced_probe(tmp_path):
    logger = trainer.EnhancedMetricsLogger(
        tmp_path / "metrics.pretrain.csv", mode="pretrain"
    )

    for field in (
        "kNN_Acc_k1",
        "kNN_Acc_k5",
        "kNN_Acc_k20",
        "Linear_Probing_Acc",
        "Linear_Probing_Balanced_Acc",
        "mAP",
    ):
        assert field in logger.fieldnames


@pytest.mark.parametrize("filename", ["metrics.pretrain.csv", "metrics.finetune.csv"])
def test_v070_enhanced_header_is_migrated_atomically(tmp_path, filename):
    import csv

    path = tmp_path / filename
    _write_v070_enhanced_csv(path)

    logger = trainer.EnhancedMetricsLogger(path, mode="pretrain")
    logger.log(2, "train", {"NMI": 0.5})

    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == logger.fieldnames
    assert "Linear_Probing_Balanced_Acc" in rows[0]
    assert len(rows) == 3
    for row in rows[1:]:
        assert len(row) == len(logger.fieldnames)
    balanced_col = rows[0].index("Linear_Probing_Balanced_Acc")
    assert rows[1][balanced_col] == ""
    assert rows[2][balanced_col] == ""
    assert rows[1][rows[0].index("NMI")] == "0.10"
    assert rows[2][rows[0].index("NMI")] == "0.5"
    assert list(tmp_path.glob("*.tmp")) == []


def test_current_enhanced_schema_is_appended_without_migration(tmp_path):
    import csv

    path = tmp_path / "metrics.pretrain.csv"
    logger = trainer.EnhancedMetricsLogger(path, mode="pretrain")
    logger.log(1, "train", {"NMI": 0.2})

    logger_again = trainer.EnhancedMetricsLogger(path, mode="pretrain")
    logger_again.log(2, "train", {"NMI": 0.3})

    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == logger.fieldnames
    assert len(rows) == 3
    assert all(len(row) == len(rows[0]) for row in rows)


@pytest.mark.parametrize(
    "content",
    [
        "epoch,split,NMI,epoch\n1,train,0.1,2\n",
        "epoch,split,unknown_metric\n1,train,0.1\n",
        "epoch,split,NMI\n1,train\n",
    ],
)
def test_malformed_enhanced_schema_fails_without_touching_the_file(tmp_path, content):
    path = tmp_path / "metrics.pretrain.csv"
    path.write_text(content, encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ValueError, match="schema"):
        trainer.EnhancedMetricsLogger(path, mode="pretrain")

    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


# --- v0.8.0 loss configuration resolver -------------------------------------


def test_resolve_finetune_loss_mode_new_run_defaults(tmp_path):
    args = _finetune_args(tmp_path, tmp_path / "source.pth")

    resolved = trainer._resolve_finetune_loss_config(args, None)

    assert resolved["loss"] == "arcface"
    assert resolved["subcenters"] is None
    assert resolved["compact_weight"] is None
    assert resolved["compact_cap"] is None
    assert resolved["supcon_temperature"] is None


def test_resolve_finetune_loss_mode_applies_only_applicable_settings(tmp_path):
    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="supcon", supcon_temperature=0.1
    )
    resolved = trainer._resolve_finetune_loss_config(
        args, None, explicit_options={"loss", "supcon_temperature"}
    )
    assert resolved["loss"] == "supcon"
    assert resolved["supcon_temperature"] == 0.1
    assert resolved["subcenters"] is None
    assert resolved["compact_weight"] is None
    assert resolved["compact_cap"] is None

    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="subcenter-arcface", subcenters=3
    )
    resolved = trainer._resolve_finetune_loss_config(
        args, None, explicit_options={"loss", "subcenters"}
    )
    assert resolved["subcenters"] == 3
    assert resolved["compact_weight"] is None
    assert resolved["supcon_temperature"] is None

    args = _finetune_args(
        tmp_path,
        tmp_path / "source.pth",
        loss="subcenter-arcface-compact",
        compact_weight=0.25,
    )
    resolved = trainer._resolve_finetune_loss_config(
        args, None, explicit_options={"loss", "compact_weight"}
    )
    assert resolved["compact_weight"] == 0.25
    assert resolved["compact_cap"] == 0.5
    assert resolved["subcenters"] == 2
    assert resolved["supcon_temperature"] is None


@pytest.mark.parametrize(
    ("overrides", "explicit"),
    [
        ({"loss": "bogus"}, {"loss"}),
        ({"loss": "supcon", "subcenters": 3}, {"loss", "subcenters"}),
        (
            {"loss": "arcface", "supcon_temperature": 0.1},
            {"loss", "supcon_temperature"},
        ),
        (
            {"loss": "subcenter-arcface", "compact_weight": 0.2},
            {"loss", "compact_weight"},
        ),
        ({"loss": "subcenter-arcface", "subcenters": 9}, {"loss", "subcenters"}),
        ({"loss": "subcenter-arcface", "subcenters": 0}, {"loss", "subcenters"}),
        ({"loss": "supcon", "supcon_temperature": 0.0}, {"loss", "supcon_temperature"}),
        (
            {"loss": "subcenter-arcface-compact", "compact_weight": -0.1},
            {"loss", "compact_weight"},
        ),
    ],
)
def test_resolve_finetune_loss_option_rejects_inapplicable_or_invalid(
    tmp_path, overrides, explicit
):
    args = _finetune_args(tmp_path, tmp_path / "source.pth", **overrides)

    with pytest.raises(ValueError):
        trainer._resolve_finetune_loss_config(args, None, explicit_options=explicit)


@pytest.mark.parametrize(
    ("overrides", "explicit"),
    [
        (
            {"loss": "subcenter-arcface-compact", "compact_weight": float("nan")},
            {"loss", "compact_weight"},
        ),
        (
            {"loss": "subcenter-arcface-compact", "compact_weight": float("inf")},
            {"loss", "compact_weight"},
        ),
        (
            {"loss": "supcon", "supcon_temperature": float("nan")},
            {"loss", "supcon_temperature"},
        ),
        (
            {"loss": "supcon", "supcon_temperature": float("inf")},
            {"loss", "supcon_temperature"},
        ),
    ],
)
def test_resolve_finetune_loss_option_rejects_non_finite_values(
    tmp_path, overrides, explicit
):
    args = _finetune_args(tmp_path, tmp_path / "source.pth", **overrides)

    with pytest.raises(ValueError):
        trainer._resolve_finetune_loss_config(args, None, explicit_options=explicit)


def test_resolve_finetune_loss_mode_resume_inherits_saved_mode(tmp_path):
    checkpoint = {
        "config": {"loss": "supcon", "supcon_temperature": 0.05},
    }
    # ``--loss`` omitted: the Namespace still carries the CLI default, but the
    # option is not explicit, so the recorded mode must win.
    args = _finetune_args(tmp_path, tmp_path / "source.pth", loss="arcface")

    resolved = trainer._resolve_finetune_loss_config(args, checkpoint)

    assert resolved["loss"] == "supcon"
    assert resolved["supcon_temperature"] == 0.05
    assert resolved["subcenters"] is None


def test_resolve_finetune_loss_mode_resume_allows_equal_and_rejects_conflict(tmp_path):
    checkpoint = {"config": {"loss": "supcon", "supcon_temperature": 0.07}}

    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="supcon", supcon_temperature=0.07
    )
    resolved = trainer._resolve_finetune_loss_config(
        args, checkpoint, explicit_options={"loss", "supcon_temperature"}
    )
    assert resolved["loss"] == "supcon"
    assert resolved["supcon_temperature"] == 0.07

    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="supcon", supcon_temperature=0.1
    )
    with pytest.raises(ValueError):
        trainer._resolve_finetune_loss_config(
            args, checkpoint, explicit_options={"loss", "supcon_temperature"}
        )

    args = _finetune_args(tmp_path, tmp_path / "source.pth", loss="arcface")
    with pytest.raises(ValueError):
        trainer._resolve_finetune_loss_config(
            args, checkpoint, explicit_options={"loss"}
        )


def test_resolve_compact_subcenters_uses_the_requested_k(tmp_path):
    """compact shares the plain Sub-center K range instead of fixing K=2."""
    args = _finetune_args(
        tmp_path,
        tmp_path / "source.pth",
        loss="subcenter-arcface-compact",
        subcenters=3,
        compact_weight=0.25,
    )

    resolved = trainer._resolve_finetune_loss_config(
        args, None, explicit_options={"loss", "subcenters", "compact_weight"}
    )

    assert resolved["loss"] == "subcenter-arcface-compact"
    assert resolved["subcenters"] == 3
    assert resolved["compact_weight"] == 0.25
    assert resolved["compact_cap"] == 0.5


@pytest.mark.parametrize("k", [0, 1, 9, -1])
def test_resolve_subcenters_rejects_k_outside_the_bound(tmp_path, k):
    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="subcenter-arcface", subcenters=k
    )
    with pytest.raises(ValueError, match="--subcenters"):
        trainer._resolve_finetune_loss_config(
            args, None, explicit_options={"subcenters"}
        )


def test_resume_rejects_explicit_k_against_a_legacy_compact_checkpoint(tmp_path):
    """A compact checkpoint with subcenters=None was trained with the fixed K=2."""
    checkpoint = {
        "config": {"loss": "subcenter-arcface-compact", "compact_weight": 0.1},
    }
    args = _finetune_args(
        tmp_path,
        tmp_path / "source.pth",
        loss="subcenter-arcface-compact",
        subcenters=3,
    )
    with pytest.raises(ValueError, match="subcenters"):
        trainer._resolve_finetune_loss_config(
            args, checkpoint, explicit_options={"subcenters"}
        )

    # Omitting K still resolves the recorded K=2, not an arbitrary default.
    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="subcenter-arcface-compact"
    )
    resolved = trainer._resolve_finetune_loss_config(args, checkpoint)
    assert resolved["subcenters"] == 2


def test_resolve_finetune_loss_mode_rejects_new_mode_on_legacy_resume(tmp_path):
    legacy = {"config": {}, "loss_state_dict": {"head.weight": torch.zeros(3, 4)}}

    args = _finetune_args(tmp_path, tmp_path / "source.pth", loss="supcon")
    with pytest.raises(ValueError, match="legacy"):
        trainer._resolve_finetune_loss_config(
            args, legacy, explicit_options={"loss"}
        )

    args = _finetune_args(tmp_path, tmp_path / "source.pth", loss="arcface")
    resolved = trainer._resolve_finetune_loss_config(
        args, legacy, explicit_options={"loss"}
    )
    assert resolved["loss"] == "arcface"


# --- v0.8.0 checkpoint decision table ---------------------------------------


def _ssl_like_state():
    return {"projector.net.0.weight": torch.zeros(2048, 4)}


def test_finetune_source_decision_table_classifies_sources():
    ssl = {
        "model_state_dict": _ssl_like_state(),
        "config": {"model_name": "m", "out_dim": 16},
    }
    assert trainer._classify_finetune_source(ssl) == "ssl"
    assert trainer._select_finetune_embedding_head(ssl) == "arcface_mlp_512"

    v080_supcon = {
        "model_state_dict": _ssl_like_state(),
        "config": {"loss": "supcon", "embedding_head": "arcface_mlp_512"},
        "loss_state_dict": {},
    }
    assert trainer._classify_finetune_source(v080_supcon) == "v080"
    assert (
        trainer._select_finetune_embedding_head(v080_supcon) == "arcface_mlp_512"
    )

    v080_arcface = {
        "model_state_dict": _ssl_like_state(),
        "config": {"loss": "arcface", "embedding_head": "arcface_mlp_512"},
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
    }
    assert trainer._classify_finetune_source(v080_arcface) == "v080"

    legacy_with_head = {
        "model_state_dict": _ssl_like_state(),
        "config": {"embedding_head": "projection_mlp_2048"},
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
    }
    assert trainer._classify_finetune_source(legacy_with_head) == "legacy_arcface"
    assert (
        trainer._select_finetune_embedding_head(legacy_with_head)
        == "projection_mlp_2048"
    )

    historical = {
        "model_state_dict": _ssl_like_state(),
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
    }
    assert trainer._classify_finetune_source(historical) == "legacy_arcface"
    assert (
        trainer._select_finetune_embedding_head(historical) == "projection_mlp_2048"
    )

    head_only = {
        "config": {"embedding_head": "arcface_mlp_512"},
        "model_state_dict": _ssl_like_state(),
    }
    assert trainer._classify_finetune_source(head_only) == "head_only"
    assert (
        trainer._select_finetune_embedding_head(head_only) == "arcface_mlp_512"
    )

    # Repo SSL checkpoints also carry teacher/student and args; only `config`
    # plus the absence of fine-tune markers makes them SSL.
    repo_ssl = {
        "model_state_dict": _ssl_like_state(),
        "student": {},
        "teacher": {},
        "args": {"model_name": "m"},
        "config": {"model_name": "m", "out_dim": 16},
    }
    assert trainer._classify_finetune_source(repo_ssl) == "ssl"


@pytest.mark.parametrize(
    "checkpoint",
    [
        # empty loss state, no head metadata: ambiguous, never ProjectionHead
        {"loss_state_dict": {}},
        # fine-tune marker without any classifier or head metadata
        {"loss_state_dict": {}, "class_labels": ["classA"]},
        # v0.8.0 loss metadata without an embedding head
        {"config": {"loss": "arcface"}},
        # arcface needs a classifier state; supcon must not have one
        {
            "config": {"loss": "arcface", "embedding_head": "arcface_mlp_512"},
            "loss_state_dict": {},
        },
        {
            "config": {"loss": "supcon", "embedding_head": "arcface_mlp_512"},
            "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
        },
        # prototype shape that cannot be a prototype classifier
        {
            "config": {
                "loss": "subcenter-arcface",
                "embedding_head": "arcface_mlp_512",
            },
            "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
        },
        # ref-script formats are read-only for finetune
        {"model": {}, "loss_func": {}},
        {"config": {"loss": "arcface", "embedding_head": "arcface_mlp_512"}, "loss_func": {}},
        # ref-script signature: args without config
        {"model_state_dict": _ssl_like_state(), "args": {"model_name": "m"}},
        # no encoder weights: not a valid SSL initialization source
        {},
        {"model_state_dict": {}},
        # non-empty state without any recognizable encoder parameter
        {"model_state_dict": {"nonexistent.weight": torch.zeros(1)}},
        # a declared head with a *present* empty loss state is ambiguous, not head-only
        {"config": {"embedding_head": "arcface_mlp_512"}, "loss_state_dict": {}},
        # recorded K / embedding width must match the classifier state
        {
            "config": {
                "loss": "subcenter-arcface",
                "embedding_head": "arcface_mlp_512",
                "subcenters": 3,
            },
            "loss_state_dict": {"head.weight": torch.zeros(2, 2, 16)},
        },
        {
            "config": {
                "loss": "arcface",
                "embedding_head": "arcface_mlp_512",
                "metric_embed_dim": 32,
            },
            "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
        },
        # a v0.8.0 source without any model state cannot be resumed or initialized
        {
            "config": {"loss": "supcon", "embedding_head": "arcface_mlp_512"},
            "loss_state_dict": {},
        },
        {
            "config": {"loss": "arcface", "embedding_head": "arcface_mlp_512"},
            "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
        },
        # head-only without model state
        {"config": {"embedding_head": "arcface_mlp_512"}},
    ],
)
def test_finetune_source_decision_table_rejects(checkpoint):
    with pytest.raises(ValueError):
        trainer._classify_finetune_source(checkpoint)


def test_finetune_legacy_inference_requires_projector_weights():
    # No model state at all: rejected before head inference is attempted.
    weightless = {
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
        "config": {},
    }
    with pytest.raises(ValueError, match="encoder weights"):
        trainer._select_finetune_embedding_head(weightless)

    # Encoder weights present, but no projector weights and no declared head:
    # there is nothing to infer the embedding head from.
    no_projector = {
        "model_state_dict": {"backbone.cls_token": torch.zeros(1, 1, 192)},
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
        "config": {},
    }
    with pytest.raises(ValueError, match="projector"):
        trainer._select_finetune_embedding_head(no_projector)


# --- manifest hashing -------------------------------------------------------


def test_train_manifest_hash_is_permutation_invariant_and_content_sensitive(
    tmp_path,
):
    root = tmp_path / "images"
    refs = ["b.jpg", "a.jpg", "a.jpg"]
    labels = ["classB", "classA", "classA"]
    baseline = trainer._train_manifest_sha256(refs, labels, root)

    assert (
        trainer._train_manifest_sha256(
            ["a.jpg", "b.jpg", "a.jpg"], ["classA", "classB", "classA"], root
        )
        == baseline
    )
    # ``./img.jpg`` and ``img.jpg`` are the same canonical reference.
    assert (
        trainer._train_manifest_sha256(
            ["./a.jpg", "a.jpg", "b.jpg"], ["classA", "classA", "classB"], root
        )
        == baseline
    )
    # Duplicate rows and label assignments are part of the manifest.
    assert (
        trainer._train_manifest_sha256(
            refs + ["a.jpg"], labels + ["classA"], root
        )
        != baseline
    )
    assert (
        trainer._train_manifest_sha256(
            refs, ["classB", "classA", "classB"], root
        )
        != baseline
    )


def test_train_manifest_hash_ignores_the_absolute_root(tmp_path):
    labels = ["classA", "classA"]
    one = tmp_path / "one"
    two = tmp_path / "two"

    first = trainer._train_manifest_sha256(
        [str(one / "a.jpg"), "b.jpg"], labels, one
    )
    second = trainer._train_manifest_sha256(
        [str(two / "a.jpg"), "b.jpg"], labels, two
    )

    assert first == second


def test_train_manifest_hash_rejects_references_outside_the_root(tmp_path):
    with pytest.raises(ValueError, match="input-images-dir"):
        trainer._train_manifest_sha256(
            [str(tmp_path / "elsewhere" / "a.jpg")], ["classA"], tmp_path / "images"
        )


def test_metric_dataset_retains_original_image_refs(tmp_path):
    img_dir, labels_path = _write_tiny_ft_data(tmp_path)
    from otuformer.training.dataset import MetricDataset

    ds = MetricDataset(
        csv_path=labels_path, images_dir=img_dir, image_size=32
    )

    assert ds.image_refs == [
        "img_0.jpg",
        "img_1.jpg",
        "img_2.jpg",
        "img_3.jpg",
    ]
    assert ds.label_names == ["classA", "classA", "classB", "classB"]


def test_metric_dataset_retains_recursive_csv_refs(tmp_path):
    """A CSV ref found only by recursive lookup keeps its original form."""
    from otuformer.training.dataset import MetricDataset

    img_dir = tmp_path / "images"
    (img_dir / "nested").mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (64, 64), color=(i * 40, 0, 0)).save(
            img_dir / "nested" / f"img_{i}.jpg"
        )
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {"image": ["img_0.jpg", "img_1.jpg"], "label": ["classA", "classB"]}
    ).to_csv(labels_path, index=False)

    ds = MetricDataset(csv_path=labels_path, images_dir=img_dir, image_size=32)

    # The image is discovered under nested/, but the CSV reference is retained,
    # so the manifest hash covers the reference, not the discovered path.
    assert ds.image_refs == ["img_0.jpg", "img_1.jpg"]
    assert trainer._train_manifest_sha256(
        ds.image_refs, ds.label_names, img_dir
    ) == trainer._train_manifest_sha256(
        ["img_0.jpg", "img_1.jpg"], ["classA", "classB"], img_dir
    )


# --- checkpoint provenance --------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "payload.bin"
    path.write_bytes(b"otuformer" * 5000)

    assert trainer._sha256_file(path) == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _expected_manifest(tmp_path):
    return trainer._train_manifest_sha256(
        [f"img_{i}.jpg" for i in range(4)],
        ["classA", "classA", "classB", "classB"],
        tmp_path / "images",
    )


def test_finetune_records_provenance_and_effective_config(tmp_path):
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)

    saved = _run_frozen_finetune(tmp_path, ssl, "provenance")

    config = saved["config"]
    ssl_sha = trainer._sha256_file(ssl)
    assert config["loss"] == "arcface"
    assert config["subcenters"] is None
    assert config["compact_weight"] is None
    assert config["compact_cap"] is None
    assert config["supcon_temperature"] is None
    assert config["seed"] == 42
    assert config["optimizer_groups"] == "v080_prototype"
    assert config["prototype_weight_decay"] == 0.0
    assert config["initialization_checkpoint_sha256"] == ssl_sha
    assert config["ssl_initialization_checkpoint_sha256"] == ssl_sha
    assert config["train_manifest_sha256"] == _expected_manifest(tmp_path)

    finetune_path = tmp_path / "provenance" / "finetune_latest.pth"
    second = _run_frozen_finetune(tmp_path, finetune_path, "provenance2")
    second_config = second["config"]
    assert second_config["initialization_checkpoint_sha256"] == (
        trainer._sha256_file(finetune_path)
    )
    assert second_config["ssl_initialization_checkpoint_sha256"] == ssl_sha


def test_finetune_new_run_can_switch_loss_without_inheriting_classifier(tmp_path):
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)

    arcface_run = _run_frozen_finetune(tmp_path, ssl, "switch_arcface")
    assert arcface_run["config"]["loss"] == "arcface"
    assert "head.weight" in arcface_run["loss_state_dict"]
    assert len(arcface_run["optimizer"]["param_groups"]) == 3

    supcon_run = _run_frozen_finetune(
        tmp_path,
        tmp_path / "switch_arcface" / "finetune_latest.pth",
        "switch_supcon",
        loss="supcon",
        supcon_temperature=0.1,
        batch_size=4,
    )

    assert supcon_run["config"]["loss"] == "supcon"
    # A direct Python caller has no explicit options, so the recorded value is
    # the first-round default, not the unused argument above.
    assert supcon_run["config"]["supcon_temperature"] == 0.07
    assert supcon_run["config"]["subcenters"] is None
    assert supcon_run["loss_state_dict"] == {}
    assert len(supcon_run["optimizer"]["param_groups"]) == 2


def test_build_finetune_loss_uses_effective_mode_kwargs():
    from otuformer.training.loss import (
        ArcFaceLoss,
        SubCenterArcFaceLoss,
        SupConLoss,
    )

    arcface = trainer._build_finetune_loss(
        {
            "loss": "arcface",
            "subcenters": None,
            "compact_weight": None,
            "compact_cap": None,
            "supcon_temperature": None,
        },
        8,
        3,
    )
    assert isinstance(arcface, ArcFaceLoss)
    assert tuple(arcface.head.weight.shape) == (3, 8)

    supcon = trainer._build_finetune_loss(
        {
            "loss": "supcon",
            "subcenters": None,
            "compact_weight": None,
            "compact_cap": None,
            "supcon_temperature": 0.1,
        },
        8,
        3,
    )
    assert isinstance(supcon, SupConLoss)
    assert supcon.temperature == 0.1

    subcenter = trainer._build_finetune_loss(
        {
            "loss": "subcenter-arcface",
            "subcenters": 3,
            "compact_weight": None,
            "compact_cap": None,
            "supcon_temperature": None,
        },
        8,
        3,
    )
    assert isinstance(subcenter, SubCenterArcFaceLoss)
    assert subcenter.head.k == 3
    assert subcenter.compact_weight == 0.0

    compact = trainer._build_finetune_loss(
        {
            "loss": "subcenter-arcface-compact",
            "subcenters": 3,
            "compact_weight": 0.25,
            "compact_cap": 0.5,
            "supcon_temperature": None,
        },
        8,
        3,
    )
    assert isinstance(compact, SubCenterArcFaceLoss)
    assert compact.head.k == 3
    assert compact.compact_weight == 0.25
    assert compact.cap == 0.5


def test_compact_checkpoint_k_comes_from_recorded_subcenters():
    trainer._validate_v080_loss_state(
        {"loss": "subcenter-arcface-compact", "subcenters": 3},
        {"head.weight": torch.zeros(4, 3, 16)},
    )
    with pytest.raises(ValueError, match="centers"):
        trainer._validate_v080_loss_state(
            {"loss": "subcenter-arcface-compact", "subcenters": 3},
            {"head.weight": torch.zeros(4, 2, 16)},
        )
    # A compact checkpoint written before K was recorded still expects K=2.
    trainer._validate_v080_loss_state(
        {"loss": "subcenter-arcface-compact"},
        {"head.weight": torch.zeros(4, 2, 16)},
    )


def test_finetune_resume_rejects_ssl_and_head_only_sources(tmp_path):
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)
    real_ssl = tmp_path / "SSL_latest.pth"
    torch.save(
        {
            "model_state_dict": _ssl_like_state(),
            "optimizer": {"state": {}, "param_groups": [{}]},
            "epoch": 3,
            "args": {"model_name": "m"},
            "config": {"model_name": "m", "out_dim": 16},
        },
        real_ssl,
    )
    head_only = tmp_path / "head_only.pth"
    torch.save(
        {
            "model_state_dict": _ssl_like_state(),
            "config": {"embedding_head": "arcface_mlp_512", "out_dim": 16},
        },
        head_only,
    )

    for index, source in enumerate((ssl, real_ssl, head_only)):
        args = _finetune_args(
            tmp_path,
            source,
            resume=str(source),
            finetune_epochs=2,
            **_finetune_overrides(tmp_path, f"resume_{index}"),
        )
        with pytest.raises(ValueError, match="SSL"):
            trainer.run_finetune(args)


# --- one epoch per loss mode, skips, diagnostics and traces ----------------

_LOSS_MODES = (
    "arcface",
    "supcon",
    "subcenter-arcface",
    "subcenter-arcface-compact",
)


def _run_loss_mode_epoch(tmp_path, mode, out_name, **extra):
    ssl = _write_pretrain_checkpoint(
        tmp_path / f"ssl_{mode}.pth", image_size=32, out_dim=16
    )
    args = _finetune_args(
        tmp_path,
        ssl,
        finetune_epochs=1,
        loss=mode,
        **_finetune_overrides(
            tmp_path,
            out_name,
            batch_size=4,
            # Frozen optimizer: the saved head is exactly the head that was
            # loaded, so the resume test can assert it was not re-initialized.
            finetune_lr=0.0,
            metric_head_lr=0.0,
            weight_decay=0.0,
            **extra,
        ),
    )
    trainer.run_finetune(args)
    return torch.load(
        tmp_path / out_name / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )


@pytest.mark.parametrize("mode", _LOSS_MODES)
def test_finetune_one_epoch_then_resume_for_each_loss_mode(tmp_path, mode, capsys):
    import otuformer.embedding.extractor as extractor

    saved = _run_loss_mode_epoch(tmp_path, mode, f"one_{mode}")
    expected_groups = 2 if mode == "supcon" else 3
    assert saved["config"]["loss"] == mode
    assert len(saved["optimizer"]["param_groups"]) == expected_groups
    if mode == "supcon":
        assert saved["loss_state_dict"] == {}
    else:
        assert "head.weight" in saved["loss_state_dict"]

    out_dir = tmp_path / f"one_{mode}"
    diagnostics = pd.read_csv(out_dir / "logs" / "loss_diagnostics.finetune.csv")
    assert len(diagnostics) == 1
    assert diagnostics.loc[0, "mode"] == mode

    # Training-only loss state is ignored by the read-only consumer.
    model, _ = extractor._load_model(
        out_dir / "finetune_latest.pth", "vit_tiny_patch16_224", torch.device("cpu")
    )
    assert model.backbone.num_features > 0

    resume_args = _finetune_args(
        tmp_path,
        tmp_path / f"ssl_{mode}.pth",
        resume=str(out_dir / "finetune_latest.pth"),
        finetune_epochs=2,
        loss=mode,
        **_finetune_overrides(
            tmp_path,
            f"two_{mode}",
            batch_size=4,
            finetune_lr=0.0,
            metric_head_lr=0.0,
            weight_decay=0.0,
        ),
    )
    trainer.run_finetune(resume_args)
    # The SupCon-only batch/anchor line must not clutter the other modes.
    epoch_log = capsys.readouterr().out
    if mode == "supcon":
        assert "valid_anchors=" in epoch_log
    else:
        assert "valid_anchors=" not in epoch_log
    resumed = torch.load(
        tmp_path / f"two_{mode}" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )

    assert resumed["epoch"] == 1
    assert len(resumed["optimizer"]["param_groups"]) == expected_groups
    if mode != "supcon":
        # Resuming never re-initializes the trained classifier.
        assert torch.equal(
            resumed["loss_state_dict"]["head.weight"],
            saved["loss_state_dict"]["head.weight"],
        )
    resumed_diagnostics = pd.read_csv(
        tmp_path / f"two_{mode}" / "logs" / "loss_diagnostics.finetune.csv"
    )
    assert len(resumed_diagnostics) == 1
    assert resumed_diagnostics.loc[0, "mode"] == mode


def test_finetune_supcon_epoch_without_negatives_fails_without_checkpoint(tmp_path):
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)
    img_dir = tmp_path / "single_images"
    img_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (64, 64), color=(i * 40, 0, 0)).save(
            img_dir / f"img_{i}.jpg"
        )
    labels_path = tmp_path / "one_class.csv"
    pd.DataFrame(
        {
            "image": [f"img_{i}.jpg" for i in range(4)],
            "label": ["classA"] * 4,
        }
    ).to_csv(labels_path, index=False)

    args = _finetune_args(
        tmp_path,
        ssl,
        finetune_epochs=1,
        loss="supcon",
        out_dir=str(tmp_path / "one_class_out"),
        train_data=str(labels_path),
        input_images_dir=str(img_dir),
        batch_size=4,
        num_workers=0,
        log_every_n_steps=100,
        save_every_epochs=1,
        keep_last_checkpoints=0,
        weight_decay=1e-4,
        metric_head_lr=None,
    )

    with pytest.raises(ValueError, match="usable"):
        trainer.run_finetune(args)

    assert not (tmp_path / "one_class_out" / "finetune_latest.pth").exists()


def _write_legacy_finetune_checkpoint(path, *, groups: int):
    """A pre-v0.8.0 finetune checkpoint with a historical optimizer layout.

    One or two parameter groups, both with nonzero prototype decay, exactly as
    the pre-v0.8.0 constructors produced them.
    """
    from otuformer.training.loss import ArcFaceLoss
    from otuformer.training.model import OTUFormerEncoder

    _write_historical_sft_checkpoint(path, image_size=32, out_dim=16)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["config"]["embedding_head"] = "projection_mlp_2048"
    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=16, pretrained=False, img_size=32
    )
    loss = ArcFaceLoss(embed_dim=16, num_classes=2)
    trainer._freeze_backbone_blocks(encoder, 0.7)
    if groups == 1:
        optimizer = torch.optim.AdamW(
            [p for p in encoder.parameters() if p.requires_grad]
            + list(loss.parameters()),
            lr=3e-5,
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p for p in encoder.backbone.parameters() if p.requires_grad
                    ],
                    "lr": 3e-5,
                    "weight_decay": 1e-4,
                },
                {
                    "params": list(encoder.projector.parameters())
                    + list(loss.parameters()),
                    "lr": 1e-4,
                    "weight_decay": 1e-4,
                },
            ]
        )
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["epoch"] = 0
    torch.save(checkpoint, path)
    return path


@pytest.mark.parametrize(
    ("groups", "expected_lrs", "expected_decays"),
    [(1, [3e-5], [1e-4]), (2, [3e-5, 1e-4], [1e-4, 1e-4])],
)
def test_finetune_legacy_resume_keeps_historical_layout(
    tmp_path, groups, expected_lrs, expected_decays
):
    """Review Focus 3: a genuine pre-v0.8.0 resume keeps its own optimizer."""
    source = _write_legacy_finetune_checkpoint(
        tmp_path / f"legacy_{groups}.pth", groups=groups
    )
    out_name = f"legacy_resume_{groups}"
    args = _finetune_args(
        tmp_path,
        source,
        resume=str(source),
        finetune_epochs=2,
        **_finetune_overrides(tmp_path, out_name, batch_size=4),
    )

    trainer.run_finetune(args)

    resumed = torch.load(
        tmp_path / out_name / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert len(resumed["optimizer"]["param_groups"]) == groups
    # The historical decay and learning rates survive; the v0.8.0 zero-decay
    # prototype rule is never imposed on a legacy resume.
    assert [g["lr"] for g in resumed["optimizer"]["param_groups"]] == expected_lrs
    assert [
        g["weight_decay"] for g in resumed["optimizer"]["param_groups"]
    ] == expected_decays
    assert resumed["config"]["loss"] == "arcface"


def test_run_finetune_uses_a_preloaded_resume_checkpoint(tmp_path, monkeypatch):
    """The CLI preflight's read must not be repeated inside run_finetune."""
    source = _write_legacy_finetune_checkpoint(tmp_path / "preloaded.pth", groups=2)
    preloaded = torch.load(source, map_location="cpu", weights_only=False)

    def fail(*_args, **_kwargs):
        raise AssertionError("resume checkpoint was loaded a second time")

    monkeypatch.setattr(trainer, "load_checkpoint", fail)
    args = _finetune_args(
        tmp_path,
        source,
        resume=str(source),
        finetune_epochs=2,
        **_finetune_overrides(tmp_path, "preloaded_resume", batch_size=4),
    )

    trainer.run_finetune(args, source_checkpoint=preloaded)

    assert (tmp_path / "preloaded_resume" / "finetune_latest.pth").exists()


def test_center_direction_cosine_uses_post_epoch_boundaries(tmp_path):
    """The recorded cosine must compare consecutive post-epoch snapshots."""
    torch.manual_seed(0)
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)
    args = _finetune_args(
        tmp_path,
        ssl,
        finetune_epochs=2,
        loss="subcenter-arcface-compact",
        **_finetune_overrides(tmp_path, "post_epoch", batch_size=4),
    )

    trainer.run_finetune(args)

    diagnostics = pd.read_csv(
        tmp_path / "post_epoch" / "logs" / "loss_diagnostics.finetune.csv"
    )
    assert pd.isna(diagnostics.loc[0, "center_direction_cosine"])

    def saved_centers(name):
        state = torch.load(
            tmp_path / "post_epoch" / name, map_location="cpu", weights_only=False
        )
        return torch.nn.functional.normalize(
            state["loss_state_dict"]["head.weight"], dim=-1
        )

    expected = trainer._center_direction_cosine(
        saved_centers("finetune_epoch_0001.pth"),
        saved_centers("finetune_epoch_0002.pth"),
    )
    recorded = json.loads(diagnostics.loc[1, "center_direction_cosine"])
    assert [value for row in recorded for value in row] == pytest.approx(
        [value for row in expected for value in row]
    )


def test_batch_id_trace_appends_on_resume_into_the_same_output_dir(tmp_path):
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)
    first = _finetune_args(
        tmp_path,
        ssl,
        finetune_epochs=1,
        trace_batch_ids=True,
        **_finetune_overrides(tmp_path, "trace_resume", batch_size=4),
    )
    trainer.run_finetune(first)
    trace_path = tmp_path / "trace_resume" / "logs" / "batch_ids.finetune.jsonl"
    first_lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert first_lines

    resume = _finetune_args(
        tmp_path,
        ssl,
        resume=str(tmp_path / "trace_resume" / "finetune_latest.pth"),
        finetune_epochs=2,
        trace_batch_ids=True,
        **_finetune_overrides(tmp_path, "trace_resume", batch_size=4),
    )
    trainer.run_finetune(resume)

    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    # The resume appended to the existing trace instead of truncating it.
    assert len(records) == len(first_lines) + 1
    assert [record["epoch"] for record in records] == [0, 1]


def test_finetune_legacy_resume_twice_keeps_the_recorded_layout(tmp_path):
    """A legacy resume's own checkpoint stays resumable with its 2-group layout."""
    source = _write_legacy_finetune_checkpoint(tmp_path / "legacy_twice.pth", groups=2)
    first_args = _finetune_args(
        tmp_path,
        source,
        resume=str(source),
        finetune_epochs=2,
        **_finetune_overrides(tmp_path, "legacy_twice_1", batch_size=4),
    )
    trainer.run_finetune(first_args)
    first_path = tmp_path / "legacy_twice_1" / "finetune_latest.pth"
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    assert first["config"]["loss"] == "arcface"
    assert first["config"]["optimizer_groups"] == "legacy_split"
    assert len(first["optimizer"]["param_groups"]) == 2

    second_args = _finetune_args(
        tmp_path,
        source,
        resume=str(first_path),
        finetune_epochs=3,
        **_finetune_overrides(tmp_path, "legacy_twice_2", batch_size=4),
    )
    trainer.run_finetune(second_args)
    second = torch.load(
        tmp_path / "legacy_twice_2" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )

    assert second["config"]["optimizer_groups"] == "legacy_split"
    assert len(second["optimizer"]["param_groups"]) == 2
    assert [
        g["weight_decay"] for g in second["optimizer"]["param_groups"]
    ] == [1e-4, 1e-4]


def test_center_direction_cosine_continues_across_a_resume(tmp_path):
    """The first continued epoch must compare against the checkpoint centers."""
    out_name = "continued_cosine"
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)
    first = _finetune_args(
        tmp_path,
        ssl,
        finetune_epochs=1,
        loss="subcenter-arcface-compact",
        **_finetune_overrides(tmp_path, out_name, batch_size=4),
    )
    trainer.run_finetune(first)

    resume = _finetune_args(
        tmp_path,
        ssl,
        resume=str(tmp_path / out_name / "finetune_latest.pth"),
        finetune_epochs=2,
        loss="subcenter-arcface-compact",
        **_finetune_overrides(tmp_path, out_name, batch_size=4),
    )
    trainer.run_finetune(resume)

    diagnostics = pd.read_csv(
        tmp_path / out_name / "logs" / "loss_diagnostics.finetune.csv"
    )
    assert list(diagnostics["epoch"]) == [0, 1]
    assert pd.isna(diagnostics.loc[0, "center_direction_cosine"])

    def saved_centers(name):
        state = torch.load(
            tmp_path / out_name / name, map_location="cpu", weights_only=False
        )
        return torch.nn.functional.normalize(
            state["loss_state_dict"]["head.weight"], dim=-1
        )

    expected = trainer._center_direction_cosine(
        saved_centers("finetune_epoch_0001.pth"),
        saved_centers("finetune_epoch_0002.pth"),
    )
    recorded = json.loads(diagnostics.loc[1, "center_direction_cosine"])
    assert [value for row in recorded for value in row] == pytest.approx(
        [value for row in expected for value in row]
    )


def test_finetune_rejects_a_partial_encoder_source(tmp_path):
    """A source with only a few backbone keys must not train a random encoder."""
    source = tmp_path / "partial.pth"
    torch.save(
        {
            "model_state_dict": {"backbone.cls_token": torch.zeros(1, 1, 192)},
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 16,
                "image_size": 32,
            },
        },
        source,
    )
    args = _finetune_args(
        tmp_path,
        source,
        finetune_epochs=1,
        **_finetune_overrides(tmp_path, "partial_source", batch_size=4),
    )

    with pytest.raises(ValueError, match="encoder weights"):
        trainer.run_finetune(args)


def test_finetune_resume_rejects_a_source_without_projector_weights(tmp_path):
    """--resume must restore the trained projector, never re-initialize it."""
    saved = _run_loss_mode_epoch(tmp_path, "arcface", "no_projector_src")
    source = tmp_path / "no_projector.pth"
    saved["model_state_dict"] = {
        key: value
        for key, value in saved["model_state_dict"].items()
        if not key.startswith("projector.")
    }
    torch.save(saved, source)

    args = _finetune_args(
        tmp_path,
        source,
        resume=str(source),
        finetune_epochs=2,
        **_finetune_overrides(tmp_path, "no_projector_resume", batch_size=4),
    )

    with pytest.raises(ValueError, match="encoder weights"):
        trainer.run_finetune(args)


def test_finetune_resume_rejects_a_partially_populated_backbone(tmp_path):
    """A state dict missing backbone parameters must not train a random encoder."""
    saved = _run_loss_mode_epoch(tmp_path, "arcface", "partial_backbone_src")
    source = tmp_path / "partial_backbone.pth"
    dropped = set(
        sorted(k for k in saved["model_state_dict"] if k.startswith("backbone."))[:2]
    )
    saved["model_state_dict"] = {
        key: value
        for key, value in saved["model_state_dict"].items()
        if key not in dropped
    }
    torch.save(saved, source)

    args = _finetune_args(
        tmp_path,
        source,
        resume=str(source),
        finetune_epochs=2,
        **_finetune_overrides(tmp_path, "partial_backbone_resume", batch_size=4),
    )

    with pytest.raises(ValueError, match="complete encoder weights"):
        trainer.run_finetune(args)


def test_finetune_rejects_a_declared_head_without_projector_weights(tmp_path):
    """A declared matching head must still come with its trained projector."""
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)
    state = torch.load(ssl, map_location="cpu", weights_only=False)
    state["model_state_dict"] = {
        key: value
        for key, value in state["model_state_dict"].items()
        if not key.startswith("projector.")
    }
    state["config"]["embedding_head"] = "arcface_mlp_512"
    source = tmp_path / "declared_head_no_projector.pth"
    torch.save(state, source)

    args = _finetune_args(
        tmp_path,
        source,
        finetune_epochs=1,
        **_finetune_overrides(tmp_path, "declared_head_src", batch_size=4),
    )

    with pytest.raises(ValueError, match="encoder weights"):
        trainer.run_finetune(args)


def test_loss_diagnostics_counts_local_and_global_winners_differently():
    from otuformer.training.loss import SubCenterArcFaceLoss

    loss = SubCenterArcFaceLoss(embed_dim=4, num_classes=2, k=2)
    with torch.no_grad():
        # Target-local winner is class 0 / center 0; the globally winning
        # class-center is class 1 / center 0.
        loss.head.weight[0, 0] = torch.tensor([0.99, 0.14, 0.0, 0.0])
        loss.head.weight[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        loss.head.weight[1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        loss.head.weight[1, 1] = torch.tensor([0.0, 0.0, 1.0, 0.0])

    counts = trainer._empty_loss_diagnostic_counts(2, 2)
    trainer._accumulate_loss_diagnostics(
        counts, loss, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([0])
    )

    assert counts["local"] == [[1, 0], [0, 0]]
    assert counts["global"] == [[0, 0], [1, 0]]
    assert counts["samples"] == 1


def test_loss_diagnostics_margin_hits_use_margin_adjusted_target_logit():
    from otuformer.training.loss import ArcFaceLoss

    loss = ArcFaceLoss(embed_dim=4, num_classes=2)
    with torch.no_grad():
        loss.head.weight[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        loss.head.weight[1] = torch.tensor([-1.0, 0.0, 0.0, 0.0])

    counts = trainer._empty_loss_diagnostic_counts(2, 1)
    trainer._accumulate_loss_diagnostics(
        counts,
        loss,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([0, 1]),
    )

    # Only the sample whose margin-adjusted target logit beats the best rival.
    assert counts["margin_hits"] == 1
    assert counts["samples"] == 2


def test_loss_diagnostics_compact_hinge_stats_match_prescribed_centers():
    from otuformer.training.loss import ArcFaceLoss, SubCenterArcFaceLoss

    loss = SubCenterArcFaceLoss(
        embed_dim=4, num_classes=2, k=2, compact_weight=0.1, cap=0.5
    )
    with torch.no_grad():
        loss.head.weight[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        loss.head.weight[0, 1] = torch.tensor([0.7, math.sqrt(1 - 0.49), 0.0, 0.0])
        loss.head.weight[1, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        loss.head.weight[1, 1] = torch.tensor([math.sqrt(1 - 0.01), 0.1, 0.0, 0.0])

    fraction, mean_penalty = trainer._compact_hinge_stats(loss)

    assert fraction == pytest.approx(0.5)
    assert mean_penalty == pytest.approx(0.1 * 0.4 / 2)
    assert trainer._compact_hinge_stats(ArcFaceLoss(embed_dim=4, num_classes=2)) == (
        None,
        None,
    )


def test_center_direction_cosine_is_empty_before_the_second_epoch():
    from otuformer.training.loss import ArcFaceLoss, SupConLoss

    loss = ArcFaceLoss(embed_dim=4, num_classes=2)
    first = trainer._prototype_centers(loss)
    assert first is not None
    assert trainer._center_direction_cosine(None, first) == []

    with torch.no_grad():
        loss.head.weight.mul_(2.0)
    second = trainer._prototype_centers(loss)
    scaled = trainer._center_direction_cosine(first, second)
    assert scaled[0][0] == pytest.approx(1.0)
    assert scaled[1][0] == pytest.approx(1.0)

    with torch.no_grad():
        loss.head.weight[0].neg_()
    third = trainer._prototype_centers(loss)
    cosines = trainer._center_direction_cosine(second, third)
    assert cosines[0][0] == pytest.approx(-1.0)
    assert cosines[1][0] == pytest.approx(1.0)

    assert trainer._prototype_centers(SupConLoss(temperature=0.07)) is None


def test_loss_diagnostics_logger_creates_the_file_and_rejects_unknown_schema(
    tmp_path,
):
    path = tmp_path / "nested" / "loss_diagnostics.finetune.csv"
    logger = trainer.LossDiagnosticsLogger(path)
    logger.log(epoch=0, mode="arcface", usable_batches=2)

    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(trainer._LOSS_DIAGNOSTIC_FIELDS)
    assert "arcface" in text

    broken = tmp_path / "broken.csv"
    broken.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        trainer.LossDiagnosticsLogger(broken)
    assert broken.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_indexed_dataset_emits_the_actual_dataset_index(tmp_path):
    from torch.utils.data import DataLoader

    from otuformer.training.dataset import IndexedDataset, MetricDataset

    img_dir, labels_path = _write_tiny_ft_data(tmp_path)
    dataset = MetricDataset(csv_path=labels_path, images_dir=img_dir, image_size=32)
    assert dataset.image_refs == [f"img_{i}.jpg" for i in range(4)]

    loader = DataLoader(
        IndexedDataset(dataset), batch_size=2, shuffle=True, num_workers=0
    )
    seen = []
    for _images, _labels, indices in loader:
        seen.extend(indices.tolist())

    assert sorted(seen) == [0, 1, 2, 3]


def test_batch_id_trace_is_off_by_default_and_follows_shuffled_order(tmp_path):
    _run_loss_mode_epoch(tmp_path, "arcface", "trace_off")
    assert not (
        tmp_path / "trace_off" / "logs" / "batch_ids.finetune.jsonl"
    ).exists()

    _run_loss_mode_epoch(tmp_path, "arcface", "trace_on", trace_batch_ids=True)
    lines = (
        tmp_path / "trace_on" / "logs" / "batch_ids.finetune.jsonl"
    ).read_text(encoding="utf-8").splitlines()

    assert lines
    for line in lines:
        record = json.loads(line)
        assert set(record) == {"epoch", "step", "indices", "refs"}
        assert sorted(record["indices"]) != []
        assert record["refs"] == [f"img_{i}.jpg" for i in record["indices"]]
    seen = [index for line in lines for index in json.loads(line)["indices"]]
    assert sorted(seen) == [0, 1, 2, 3]
