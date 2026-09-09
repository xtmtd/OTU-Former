import argparse
import importlib.util
import json
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


def test_masked_token_loss_matches_reference_branching():
    torch.manual_seed(0)
    student_tokens = torch.randn(2, 8, 4)
    teacher_tokens = torch.randn(2, 8, 4)
    # non-EVA model: cosine loss, must be >= 0
    loss = trainer._masked_token_loss(
        student_tokens, teacher_tokens, 0.5, "vit_tiny_patch16_224"
    )
    assert float(loss) >= 0
    # EVA model: mse loss, must be >= 0
    loss_eva = trainer._masked_token_loss(
        student_tokens, teacher_tokens, 0.5, "eva_tiny"
    )
    assert float(loss_eva) >= 0
    # non-EVA and EVA should produce different values for same inputs
    assert float(loss) != pytest.approx(float(loss_eva))


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
        freeze_ratio=0.7,
        extract_size=None,
        compute_embedding_metrics=False,
        augmentation=None,
        orientation_policy=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


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
