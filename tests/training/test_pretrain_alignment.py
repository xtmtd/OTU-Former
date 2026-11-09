import argparse
import importlib.util
import json
import math
import os
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
import otuformer.training.dataset as dataset_module
import otuformer.training.finetune_loop as finetune_loop_module
import otuformer.training.model as model_module
import otuformer.training.objectives as objectives_module
import otuformer.training.pretrain_recipe as pretrain_recipe_module
import otuformer.training.pretrain_loop as pretrain_loop_module
import otuformer.training.reporting as reporting_module
import otuformer.training.augmentation as augmentation_module
import otuformer.training.finetune_support as finetune_support_module


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
    lr, momentum, _, teacher_temp, metadata = pretrain_loop_module._build_pretrain_schedules(
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
        pretrain_loop_module._build_pretrain_schedules(
            _schedule_args(2), saved, steps_per_epoch=5, global_step=4, completed_epochs=1
        )


@pytest.mark.parametrize("steps_per_epoch,global_step,total_steps", [(2, 4, 12), (8, 10, 42)])
def test_extended_schedule_rejects_return_to_original_plan(
    steps_per_epoch, global_step, total_steps
):
    saved = {
        "original_max_epochs": 4,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": 0,
        "last_lr": 0.0002,
        "final_lr": 0.00001,
    }
    with pytest.raises(ValueError, match="extended.*original"):
        pretrain_loop_module._build_pretrain_schedules(
            _schedule_args(4), saved, steps_per_epoch=steps_per_epoch,
            global_step=global_step, completed_epochs=2, original_max_epochs=6,
        )


def test_finetune_resume_rejects_different_class_mapping():
    with pytest.raises(ValueError, match="class labels differ"):
        finetune_loop_module._validate_finetune_resume(
            {"epoch": 0, "class_labels": ["a", "b"]}, ["a", "c"], 2
        )


def test_legacy_pretrain_same_plan_resume_is_allowed():
    lr, *_ = pretrain_loop_module._build_pretrain_schedules(
        _schedule_args(5),
        None,
        steps_per_epoch=4,
        global_step=8,
        completed_epochs=2,
        original_max_epochs=5,
    )

    assert len(lr) == 20


@pytest.mark.parametrize("recorded_epochs", [5, "5"])
def test_legacy_pretrain_same_target_remains_valid(recorded_epochs):
    pretrain_recipe_module._validate_pretrain_epoch_target(5, 2, None, recorded_epochs)


def test_legacy_pretrain_resume_rejects_missing_schedule_metadata():
    with pytest.raises(ValueError, match="legacy checkpoint"):
        pretrain_loop_module._build_pretrain_schedules(
            _schedule_args(6),
            None,
            steps_per_epoch=4,
            global_step=8,
            completed_epochs=2,
            original_max_epochs=5,
        )


def test_finetune_resume_rejects_completed_epoch_target():
    with pytest.raises(ValueError, match="finetune-epochs must exceed"):
        finetune_loop_module._validate_finetune_resume(
            {"epoch": 2, "class_labels": ["a", "b"]}, ["a", "b"], 3
        )


def test_teacher_temp_schedule_uses_70_percent_warmup():
    schedule = pretrain_recipe_module._build_teacher_temp_schedule(
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
    updated = pretrain_loop_module._update_teacher_center(teacher_center, teacher_global)
    assert updated.shape == (1, 2)
    assert updated[0, 0] == pytest.approx(0.05)
    assert updated[0, 1] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("pairing", "expected"),
    [
        ("cross", {(1.0, 4.0), (2.0, 3.0)}),
        ("full", {(1.0, 3.0), (1.0, 4.0), (2.0, 3.0), (2.0, 4.0)}),
        ("same", {(1.0, 3.0), (2.0, 4.0)}),
    ],
)
def test_relational_global_pair_sets(monkeypatch, pairing, expected):
    student = [torch.tensor([[1.0]]), torch.tensor([[2.0]])]
    teacher = [torch.tensor([[3.0]]), torch.tensor([[4.0]])]
    calls = []

    def fake_ssl_loss(s, t, *_):
        calls.append((float(s.item()), float(t.item())))
        return torch.tensor(1.0)

    monkeypatch.setattr(objectives_module, "_ssl_loss", fake_ssl_loss)
    loss = pretrain_loop_module._compute_global_loss(
        student, teacher, 0.1, 0.04, relational_pairing=pairing
    )
    assert set(calls) == expected
    # every policy averages over the pairs it actually selected
    assert float(loss) == pytest.approx(1.0)
    assert len(calls) == len(expected)


def test_relational_global_pair_sets_reject_unknown_or_wrong_arity():
    student = [torch.tensor([[1.0]]), torch.tensor([[2.0]])]
    teacher = [torch.tensor([[3.0]]), torch.tensor([[4.0]])]
    with pytest.raises(ValueError, match="relational_pairing"):
        pretrain_loop_module._compute_global_loss(
            student, teacher, 0.1, 0.04, relational_pairing="zip"
        )
    with pytest.raises(ValueError, match="exactly two"):
        pretrain_loop_module._compute_global_loss(
            student + student, teacher, 0.1, 0.04, relational_pairing="cross"
        )


def test_relational_loss_keeps_candidate_diagonal():
    torch.manual_seed(0)
    teacher = torch.nn.functional.normalize(torch.randn(4, 6), dim=-1)
    student = torch.nn.functional.normalize(torch.randn(4, 6), dim=-1)
    student.requires_grad_(True)
    loss = objectives_module._ssl_loss(student, teacher, 0.1, 0.04)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()

    # manual full-batch softmax including the same-image diagonal
    teacher_sim = teacher @ teacher.T / 0.04
    teacher_sim = teacher_sim - teacher_sim.mean(dim=1, keepdim=True)
    probs = torch.softmax(teacher_sim, dim=1)
    student_sim = student.detach() @ teacher.T / 0.1
    manual = -(probs * torch.log_softmax(student_sim, dim=1)).sum(dim=1).mean()
    assert float(loss.detach()) == pytest.approx(float(manual), rel=1e-6)

    # subtracting each row's mean leaves the softmax unchanged
    shifted = torch.softmax(teacher_sim + 3.5, dim=1)
    assert torch.allclose(probs, shifted, atol=1e-6)


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

    monkeypatch.setattr(objectives_module, "_ssl_loss", fake_ssl_loss)
    pretrain_loop_module._compute_local_loss(local_student, teacher, 0.1, 0.04)
    assert len(calls) == 6


def test_pretrain_metric_names_and_values_match_expected_mapping(monkeypatch):
    """Each reported name carries the value its own computation produced.

    Every downstream metric is stubbed with a distinct sentinel, so a swapped or
    aliased mapping fails even when both names are present: comparing NMI with
    ARI alone would let a mutual swap through.
    """
    sentinels = {
        "NMI": 0.11,
        "ARI": 0.22,
        "Silhouette_Score": 0.33,
        "Purity": 0.44,
        "mAP": 0.55,
    }
    monkeypatch.setattr(
        reporting_module, "compute_recall_at_k",
        lambda *a, **k: {"Recall@1": 0.61, "Recall@5": 0.62, "Recall@10": 0.63},
    )
    monkeypatch.setattr(
        reporting_module, "compute_knn_accuracy",
        lambda *a, **k: {"kNN_Acc_k1": 0.71, "kNN_Acc_k5": 0.72, "kNN_Acc_k20": 0.73},
    )
    monkeypatch.setattr(reporting_module, "compute_map", lambda *a, **k: sentinels["mAP"])
    monkeypatch.setattr(
        reporting_module, "compute_clustering_metrics", lambda *a, **k: dict(sentinels)
    )

    labels = np.array([0, 0, 1, 1, 2, 2])
    embeddings = np.arange(12, dtype=float).reshape(6, 2)
    metrics = reporting_module._compute_all_metrics(
        embeddings, labels, compute_linear_probe=False
    )

    for key, value in {
        **sentinels,
        "Recall@1": 0.61,
        "Recall@5": 0.62,
        "Recall@10": 0.63,
        "kNN_Acc_k1": 0.71,
        "kNN_Acc_k5": 0.72,
        "kNN_Acc_k20": 0.73,
    }.items():
        assert metrics[key] == value, key


def test_pretrain_encoder_exposes_projector_output_and_patch_tokens():
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(return_patch_tokens=True)
    x = torch.randn(2, 3, 224, 224)
    result = model(x)
    proj, patch_tokens = result
    assert proj.ndim == 2
    assert patch_tokens.ndim == 3


def test_pretrain_periodic_evaluation_uses_cls_token_features(tmp_path, monkeypatch):
    """Evaluation embeddings come from the CLS token, observed on real tensors.

    The previous source-string check passed even when the extraction selected
    patch tokens and left a CLS statement behind. This drives the function with a
    stub model whose CLS and patch rows are distinguishable.
    """
    seen = {}

    class StubBackbone:
        def forward_features(self, images):
            batch = images.shape[0]
            tokens = torch.zeros(batch, 5, 4)
            tokens[:, 0] = 1.0  # CLS row
            tokens[:, 1:] = 2.0  # patch rows
            return tokens

    class StubModel:
        backbone = StubBackbone()

        def eval(self):
            return self

        def __call__(self, images):
            return torch.zeros(images.shape[0], 4)

    images = tmp_path / "images"
    images.mkdir()
    for index in range(3):
        Image.new("RGB", (32, 32), color=(index * 40, 0, 0)).save(
            images / f"img_{index}.jpg"
        )
    csv_path = tmp_path / "eval.csv"
    pd.DataFrame(
        {"image": [f"img_{i}.jpg" for i in range(3)], "label": ["A", "A", "B"]}
    ).to_csv(csv_path, index=False)

    features, labels = reporting_module._compute_embeddings_from_csv(
        model=StubModel(),
        csv_path=csv_path,
        root_dir=images,
        image_size=32,
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
    )
    assert len(labels) == 3
    # The CLS row is the constant 1.0 row; a patch-token selection would yield
    # 2.0, and a projector path would yield zeros.
    assert torch.as_tensor(features).shape == (3, 4)
    assert torch.allclose(torch.as_tensor(features), torch.ones(3, 4))


def test_unlabeled_periodic_evaluation_still_generates_umap(
    tmp_path, monkeypatch, capsys
):
    calls = []

    class MetricsLogger:
        def log(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(
        reporting_module, "_compute_embeddings_from_csv",
        lambda **kwargs: (np.zeros((10, 4), dtype=np.float32), None),
    )
    monkeypatch.setattr(
        reporting_module, "run_umap",
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

    pretrain_loop_module._compute_and_log_all_metrics(
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

    sampled, labels = reporting_module._maybe_subsample_for_metrics(
        embeddings, None, max_samples=0, seed=42
    )

    assert sampled is embeddings
    assert labels is None


def test_single_class_periodic_evaluation_skips_supervised_metrics():
    fields = reporting_module._compute_all_metrics(
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
        reporting_module, "_compute_embeddings_from_csv",
        lambda **kwargs: (np.zeros((20, 4), dtype=np.float32), None),
    )
    monkeypatch.setattr(
        reporting_module, "run_umap",
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

    pretrain_loop_module._compute_and_log_all_metrics(
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
        reporting_module, "_compute_embeddings_from_csv",
        lambda **kwargs: (np.zeros((20, 4), dtype=np.float32), None),
    )
    monkeypatch.setattr(
        reporting_module, "run_umap",
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

    pretrain_loop_module._compute_and_log_all_metrics(
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
        reporting_module, "_compute_embeddings_from_csv",
        lambda **kwargs: (
            np.zeros((10, 4), dtype=np.float32),
            np.array(["one"] * 10),
        ),
    )
    monkeypatch.setattr(
        reporting_module, "_maybe_subsample_for_metrics",
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

    pretrain_loop_module._compute_and_log_all_metrics(
        args=args,
        model=object(),
        device=torch.device("cpu"),
        epoch=0,
        logs_dir=tmp_path,
        metrics_logger=MetricsLogger(),
        eval_image_size=224,
    )

    assert "fewer than two label classes after sampling" in capsys.readouterr().out


def test_teacher_momentum_updates_after_optimizer_step(tmp_path, monkeypatch):
    """The EMA update happens after ``optimizer.step()``, observed at runtime.

    Source position is not evidence: a comment, a branch or a reordering inside
    the step could satisfy it. Record the real event order instead.
    """
    events = []
    real_step = torch.optim.AdamW.step
    real_update = pretrain_loop_module.update_teacher

    def step_spy(self, *args, **kwargs):
        events.append("step")
        return real_step(self, *args, **kwargs)

    def update_spy(student, teacher, momentum):
        events.append("ema")
        return real_update(student, teacher, momentum)

    monkeypatch.setattr(torch.optim.AdamW, "step", step_spy)
    monkeypatch.setattr(pretrain_loop_module, "update_teacher", update_spy)

    args = argparse.Namespace(
        seed=42, cpus=0, device="cpu", out_dir=str(tmp_path / "out"),
        model_name="vit_tiny_patch16_224", register_tokens=None,
        out_dim=16, ssl_method="relational", ssl_method_explicit=False,
        koleo_weight=0.0, koleo_weight_explicit=False,
        max_epochs=1, lr=1e-3, weight_decay=0.05, warmup_epochs=0,
        augmentation=None, orientation_policy=None,
        global_crop_size=224, local_crop_size=96, local_crops=0,
        mask_ratio="auto", lambda_local=1.5, lambda_mask=1.0,
        patch_loss="none", masking_strategy="random",
        teacher_momentum=0.995, teacher_momentum_end=0.999,
        student_temp=0.1, teacher_temp_start=0.04, teacher_temp_end=0.07,
        teacher_temp_warmup_ratio=None, resume="", log_every_n_steps=1000,
        save_every_epochs=1, keep_last_checkpoints=1, visualize_data="",
        extract_size=None, metrics_sample_size=10, umap_n_neighbors=5,
        umap_min_dist=0.1, umap_metric="cosine", visualize_class_number=5,
        compute_embedding_metrics=False, batch_size=2, num_workers=0,
        train_data=str(tmp_path / "data.csv"),
        input_images_dir=str(tmp_path / "images"),
    )
    images = tmp_path / "images"
    images.mkdir()
    for index in range(4):
        Image.new("RGB", (64, 64), color=(index * 50, 0, 0)).save(
            images / f"img_{index}.jpg"
        )
    pd.DataFrame({"image": [f"img_{i}.jpg" for i in range(4)]}).to_csv(
        tmp_path / "data.csv", index=False
    )
    pretrain_loop_module.run_pretrain(args)

    # Per optimizer step the teacher is refreshed once, and never before it.
    assert events.count("step") == events.count("ema")
    assert events[:2] == ["step", "ema"]
    for index, event in enumerate(events):
        if event == "ema":
            assert events[index - 1] == "step"


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
    finetune_loop_module.run_finetune(args)

    saved = torch.load(
        tmp_path / "ft_out" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["config"]["image_size"] == 32
    assert saved["config"]["augmentation_profile"] == "none"
    assert saved["config"]["augmentation_config"]["image_size"] == 32
    assert saved["config"]["augmentation_config"]["orientation_policy"] == "sensitive"
    assert saved["config"]["orientation_policy"] == "sensitive"
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
    finetune_loop_module.run_finetune(first_args)
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
    finetune_loop_module.run_finetune(resume_args)
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
    profile = finetune_loop_module._select_augmentation_profile(None, None, stage=stage)
    policy = finetune_loop_module._select_orientation_policy(None, None, stage=stage)
    config = _augmentation_config(stage, profile, policy)
    assert (profile, policy) == (expected_profile, "sensitive")
    assert finetune_loop_module._validate_augmentation_config(
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
    profile = finetune_loop_module._select_augmentation_profile(None, checkpoint, stage=stage)
    policy = finetune_loop_module._select_orientation_policy(None, checkpoint, stage=stage)
    config = _augmentation_config(stage, profile, policy)
    assert (profile, policy) == (legacy_profile, expected_policy)
    assert finetune_loop_module._validate_augmentation_config(
        profile, config, checkpoint, stage=stage
    ) == config


def test_pretrain_resume_inherits_saved_augmentation():
    checkpoint = _augmented_checkpoint("pretrain", "global-barcode", "invariant")
    profile = finetune_loop_module._select_augmentation_profile(None, checkpoint, stage="pretrain")
    policy = finetune_loop_module._select_orientation_policy(
        None, checkpoint, stage="pretrain", resume=True
    )
    config = _augmentation_config("pretrain", profile, policy)
    assert (profile, policy) == ("global-barcode", "invariant")
    assert (
        finetune_loop_module._validate_augmentation_config(
            profile, config, checkpoint, stage="pretrain"
        )
        == config
    )


def test_pretrain_resume_accepts_explicit_matching_augmentation():
    checkpoint = _augmented_checkpoint("pretrain", "color-robust", "sensitive")
    profile = finetune_loop_module._select_augmentation_profile(
        "color-robust", checkpoint, stage="pretrain"
    )
    policy = finetune_loop_module._select_orientation_policy(
        "sensitive", checkpoint, stage="pretrain", resume=True
    )
    config = _augmentation_config("pretrain", profile, policy)
    assert (profile, policy) == ("color-robust", "sensitive")
    assert (
        finetune_loop_module._validate_augmentation_config(
            profile, config, checkpoint, stage="pretrain"
        )
        == config
    )


def test_pretrain_resume_rejects_different_profile_and_policy():
    checkpoint = _augmented_checkpoint("pretrain", "global-barcode", "invariant")
    with pytest.raises(ValueError, match="Cannot resume") as excinfo:
        finetune_loop_module._select_augmentation_profile(
            "color-robust", checkpoint, stage="pretrain"
        )
    assert "new run" in str(excinfo.value)
    with pytest.raises(ValueError, match="Cannot resume") as excinfo:
        finetune_loop_module._select_orientation_policy(
            "sensitive", checkpoint, stage="pretrain", resume=True
        )
    assert "new run" in str(excinfo.value)


def test_finetune_resume_inherits_and_matches_saved_augmentation():
    checkpoint = _augmented_checkpoint(
        "finetune", "conservative", "sensitive", image_size=224
    )
    profile = finetune_loop_module._select_augmentation_profile(None, checkpoint, stage="finetune")
    policy = finetune_loop_module._select_orientation_policy(
        None, checkpoint, stage="finetune", resume=True
    )
    config = _augmentation_config("finetune", profile, policy, image_size=224)
    assert (profile, policy) == ("conservative", "sensitive")
    assert (
        finetune_loop_module._validate_augmentation_config(
            profile, config, checkpoint, stage="finetune"
        )
        == config
    )
    assert (
        finetune_loop_module._select_augmentation_profile(
            "conservative", checkpoint, stage="finetune"
        )
        == "conservative"
    )
    assert (
        finetune_loop_module._select_orientation_policy(
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
        finetune_loop_module._validate_augmentation_config(
            "global-barcode", changed, checkpoint, stage="pretrain"
        )
    changed_policy = json.loads(json.dumps(saved))
    changed_policy["orientation_policy"] = "sensitive"
    with pytest.raises(ValueError, match="configuration differs"):
        finetune_loop_module._validate_augmentation_config(
            "global-barcode", changed_policy, checkpoint, stage="pretrain"
        )


def test_new_run_orientation_policy_defaults_and_override():
    assert (
        finetune_loop_module._select_orientation_policy(None, None, stage="pretrain")
        == "sensitive"
    )
    assert (
        finetune_loop_module._select_orientation_policy("sensitive", None, stage="pretrain")
        == "sensitive"
    )
    assert (
        finetune_loop_module._select_orientation_policy("invariant", None, stage="pretrain")
        == "invariant"
    )
    assert (
        finetune_loop_module._select_orientation_policy(None, None, stage="finetune")
        == "sensitive"
    )
    assert (
        finetune_loop_module._select_orientation_policy("sensitive", None, stage="finetune")
        == "sensitive"
    )
    assert (
        finetune_loop_module._select_orientation_policy("invariant", None, stage="finetune")
        == "invariant"
    )


def test_new_legacy_run_records_explicit_sensitive_without_changing_transform():
    profile = finetune_loop_module._select_augmentation_profile("legacy", None, stage="pretrain")
    policy = finetune_loop_module._select_orientation_policy("sensitive", None, stage="pretrain")
    config = _augmentation_config("pretrain", profile, policy)
    assert (profile, policy) == ("legacy", "sensitive")
    assert config["orientation_policy"] == "sensitive"
    assert config["rotation"]["degrees"] == [-180.0, 180.0]
    assert config["horizontal_flip_probability"] == 0.5


def test_old_pretrain_resume_profile_and_policy_compatibility():
    checkpoint = {"config": {"model_name": "vit_tiny_patch16_224"}}
    assert (
        finetune_loop_module._select_augmentation_profile(None, checkpoint, stage="pretrain")
        == "legacy"
    )
    assert (
        finetune_loop_module._select_augmentation_profile("legacy", checkpoint, stage="pretrain")
        == "legacy"
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        finetune_loop_module._select_augmentation_profile(
            "global-barcode", checkpoint, stage="pretrain"
        )
    assert (
        finetune_loop_module._select_orientation_policy(
            None, checkpoint, stage="pretrain", resume=True
        )
        == "invariant"
    )
    assert (
        finetune_loop_module._select_orientation_policy(
            "invariant", checkpoint, stage="pretrain", resume=True
        )
        == "invariant"
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        finetune_loop_module._select_orientation_policy(
            "sensitive", checkpoint, stage="pretrain", resume=True
        )


def test_old_finetune_resume_compatibility():
    checkpoint = {"config": {"model_name": "vit_tiny_patch16_224"}}
    assert (
        finetune_loop_module._select_augmentation_profile(None, checkpoint, stage="finetune")
        == "none"
    )
    assert (
        finetune_loop_module._select_augmentation_profile("none", checkpoint, stage="finetune")
        == "none"
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        finetune_loop_module._select_augmentation_profile(
            "conservative", checkpoint, stage="finetune"
        )
    assert (
        finetune_loop_module._select_orientation_policy(
            None, checkpoint, stage="finetune", resume=True
        )
        == "invariant"
    )
    assert (
        finetune_loop_module._select_orientation_policy(
            "invariant", checkpoint, stage="finetune", resume=True
        )
        == "invariant"
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        finetune_loop_module._select_orientation_policy(
            "sensitive", checkpoint, stage="finetune", resume=True
        )


def test_old_pretrain_checkpoint_finetune_initialization():
    checkpoint = {"config": {"model_name": "vit_tiny_patch16_224"}}
    assert finetune_loop_module._select_augmentation_profile(None, None, stage="finetune") == "none"
    assert (
        finetune_loop_module._select_augmentation_profile("conservative", None, stage="finetune")
        == "conservative"
    )
    assert (
        finetune_loop_module._select_orientation_policy(
            None, checkpoint, stage="finetune", resume=False
        )
        == "sensitive"
    )
    assert (
        finetune_loop_module._select_orientation_policy(
            "invariant", checkpoint, stage="finetune", resume=False
        )
        == "invariant"
    )
    assert (
        finetune_loop_module._select_orientation_policy(
            "sensitive", checkpoint, stage="finetune", resume=False
        )
        == "sensitive"
    )


def test_new_style_pretrain_checkpoint_finetune_initialization_inherits_policy_only():
    checkpoint = _augmented_checkpoint("pretrain", "color-robust", "sensitive")
    assert finetune_loop_module._select_augmentation_profile(None, None, stage="finetune") == "none"
    assert (
        finetune_loop_module._select_orientation_policy(
            None, checkpoint, stage="finetune", resume=False
        )
        == "sensitive"
    )
    assert (
        finetune_loop_module._select_orientation_policy(
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
        finetune_loop_module._select_augmentation_profile(None, checkpoint, stage="pretrain")
    with pytest.raises(ValueError, match="malformed augmentation metadata"):
        finetune_loop_module._select_orientation_policy(
            None, checkpoint, stage="pretrain", resume=True
        )
    with pytest.raises(ValueError, match="malformed augmentation metadata"):
        finetune_loop_module._validate_augmentation_config(
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
        finetune_loop_module._select_augmentation_profile(None, pretrain_style, stage="finetune")
    with pytest.raises(ValueError, match="not valid for pretrain"):
        finetune_loop_module._select_augmentation_profile(None, finetune_style, stage="pretrain")


@pytest.mark.parametrize("stage", ["other", "pretraining", ""])
def test_unsupported_stage_raises(stage):
    with pytest.raises(ValueError, match="Unsupported augmentation stage"):
        finetune_loop_module._select_augmentation_profile(None, None, stage=stage)
    with pytest.raises(ValueError, match="Unsupported augmentation stage"):
        finetune_loop_module._select_orientation_policy(None, None, stage=stage)
    with pytest.raises(ValueError, match="Unsupported augmentation stage"):
        finetune_loop_module._validate_augmentation_config(
            "none",
            _augmentation_config("finetune", "none", "invariant"),
            None,
            stage=stage,
        )


def test_pretrain_local_views_new_run_defaults_and_validation():
    assert pretrain_recipe_module._resolve_pretrain_local_views(None, None, None) == (96, 6)
    assert pretrain_recipe_module._resolve_pretrain_local_views(128, 4, None) == (128, 4)
    assert pretrain_recipe_module._resolve_pretrain_local_views(128, 0, None) == (128, 0)
    for bad_size in (0, -1, True, 1.5, "96"):
        with pytest.raises(ValueError):
            pretrain_recipe_module._resolve_pretrain_local_views(bad_size, None, None)
    for bad_crops in (-1, False, 1.5, "6"):
        with pytest.raises(ValueError):
            pretrain_recipe_module._resolve_pretrain_local_views(None, bad_crops, None)


def test_pretrain_resume_inherits_saved_local_views():
    checkpoint = _augmented_checkpoint(
        "pretrain",
        "global-barcode",
        "invariant",
        local_crop_size=128,
        local_crops=4,
    )
    assert pretrain_recipe_module._resolve_pretrain_local_views(None, None, checkpoint) == (128, 4)
    assert pretrain_recipe_module._resolve_pretrain_local_views(128, 4, checkpoint) == (128, 4)
    with pytest.raises(ValueError, match="Cannot resume"):
        pretrain_recipe_module._resolve_pretrain_local_views(96, None, checkpoint)
    with pytest.raises(ValueError, match="Cannot resume"):
        pretrain_recipe_module._resolve_pretrain_local_views(None, 6, checkpoint)


def test_old_pretrain_local_views_inherit_and_fallback():
    checkpoint = {
        "config": {"model_name": "vit_tiny_patch16_224"},
        "args": {"local_crop_size": 128, "local_crops": 0},
    }
    assert pretrain_recipe_module._resolve_pretrain_local_views(None, None, checkpoint) == (128, 0)
    assert pretrain_recipe_module._resolve_pretrain_local_views(128, 0, checkpoint) == (128, 0)

    missing = {"config": {"model_name": "vit_tiny_patch16_224"}, "args": {}}
    assert pretrain_recipe_module._resolve_pretrain_local_views(None, None, missing) == (96, 6)
    with pytest.raises(ValueError, match="Cannot resume"):
        pretrain_recipe_module._resolve_pretrain_local_views(128, None, missing)

    invalid = {
        "config": {"model_name": "vit_tiny_patch16_224"},
        "args": {"local_crop_size": 0, "local_crops": True},
    }
    assert pretrain_recipe_module._resolve_pretrain_local_views(None, None, invalid) == (96, 6)
    with pytest.raises(ValueError, match="Cannot resume"):
        pretrain_recipe_module._resolve_pretrain_local_views(None, 0, invalid)


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

    monkeypatch.setattr(pretrain_loop_module, "MultiCropDataset", fake_dataset)
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
        pretrain_loop_module.run_pretrain(args)

    assert captured["augmentation_profile"] == "global-barcode"
    assert captured["orientation_policy"] == "sensitive"
    assert captured["local_crop_size"] == 96
    assert captured["local_crops"] == 2
    out = capsys.readouterr().out
    assert "Resolved device: cpu" in out
    assert "Augmentation profile: global-barcode" in out
    assert "Augmentation config:" in out
    assert '"profile": "global-barcode"' in out
    assert '"orientation_policy": "sensitive"' in out


def test_pretrain_passes_explicit_augmentation_to_dataset(tmp_path, monkeypatch):
    captured = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        raise _DatasetConstructionReached

    monkeypatch.setattr(pretrain_loop_module, "MultiCropDataset", fake_dataset)
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
        pretrain_loop_module.run_pretrain(args)

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

    monkeypatch.setattr(finetune_loop_module, "MetricDataset", fake_dataset)
    args = _finetune_args(tmp_path, checkpoint_path)

    with pytest.raises(_DatasetConstructionReached):
        finetune_loop_module.run_finetune(args)

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

    real_encoder = finetune_loop_module.OTUFormerEncoder

    def fake_encoder(*args, **kwargs):
        captured_model["img_size"] = kwargs.get("img_size")
        return real_encoder(*args, **kwargs)

    def fake_dataset(**kwargs):
        captured_dataset.update(kwargs)
        raise _DatasetConstructionReached

    monkeypatch.setattr(finetune_loop_module, "OTUFormerEncoder", fake_encoder)
    monkeypatch.setattr(finetune_loop_module, "MetricDataset", fake_dataset)
    args = _finetune_args(tmp_path, checkpoint_path)

    with pytest.raises(_DatasetConstructionReached):
        finetune_loop_module.run_finetune(args)

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
    finetune_loop_module.run_finetune(args)
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
    finetune_loop_module.run_finetune(first)

    resume = _finetune_args(
        tmp_path,
        ssl,
        resume=str(tmp_path / "f1" / "finetune_latest.pth"),
        finetune_epochs=2,
        metric_embed_dim=8,
        **_finetune_overrides(tmp_path, "f2"),
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        finetune_loop_module.run_finetune(resume)


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
        finetune_loop_module.run_finetune(args)


def test_finetune_resume_rejects_a_changed_freeze_ratio(tmp_path):
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)

    first = _finetune_args(
        tmp_path,
        ssl,
        finetune_epochs=1,
        freeze_ratio=0.5,
        **_finetune_overrides(tmp_path, "f1"),
    )
    finetune_loop_module.run_finetune(first)
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
        finetune_loop_module.run_finetune(resume)


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

    sampled, sampled_labels = reporting_module._maybe_subsample_for_metrics(
        embeddings, labels, max_samples=10, seed=42
    )

    expected = np.sort(np.random.default_rng(42).choice(30, size=10, replace=False))
    assert np.array_equal(sampled[:, 0].astype(int), expected)
    assert np.array_equal(sampled_labels, expected)


def test_compute_all_metrics_uses_empty_strings_for_unavailable_metrics():
    embeddings = np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32)
    labels = np.array(["a", "a", "a", "b", "b", "b"])

    fields = reporting_module._compute_all_metrics(embeddings, labels, compute_linear_probe=True)

    assert fields["Recall@10"] == ""
    assert fields["kNN_Acc_k20"] == ""
    assert isinstance(fields["Linear_Probing_Acc"], float)
    assert isinstance(fields["Linear_Probing_Balanced_Acc"], float)
    for value in fields.values():
        assert value == "" or isinstance(value, float)


def test_enhanced_metrics_logger_keeps_history_and_adds_balanced_probe(tmp_path):
    logger = pretrain_loop_module.EnhancedMetricsLogger(
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

    logger = pretrain_loop_module.EnhancedMetricsLogger(path, mode="pretrain")
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
    logger = pretrain_loop_module.EnhancedMetricsLogger(path, mode="pretrain")
    logger.log(1, "train", {"NMI": 0.2})

    logger_again = pretrain_loop_module.EnhancedMetricsLogger(path, mode="pretrain")
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
        pretrain_loop_module.EnhancedMetricsLogger(path, mode="pretrain")

    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


# --- v0.8.0 loss configuration resolver -------------------------------------


def test_resolve_finetune_loss_mode_new_run_defaults(tmp_path):
    args = _finetune_args(tmp_path, tmp_path / "source.pth")

    resolved = finetune_loop_module._resolve_finetune_loss_config(args, None)

    assert resolved["loss"] == "arcface"
    assert resolved["subcenters"] is None
    assert resolved["compact_weight"] is None
    assert resolved["compact_cap"] is None
    assert resolved["supcon_temperature"] is None


def test_resolve_finetune_loss_mode_applies_only_applicable_settings(tmp_path):
    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="supcon", supcon_temperature=0.1
    )
    resolved = finetune_loop_module._resolve_finetune_loss_config(
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
    resolved = finetune_loop_module._resolve_finetune_loss_config(
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
    resolved = finetune_loop_module._resolve_finetune_loss_config(
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
        finetune_loop_module._resolve_finetune_loss_config(args, None, explicit_options=explicit)


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
        finetune_loop_module._resolve_finetune_loss_config(args, None, explicit_options=explicit)


def test_resolve_finetune_loss_mode_resume_inherits_saved_mode(tmp_path):
    checkpoint = {
        "config": {"loss": "supcon", "supcon_temperature": 0.05},
    }
    # ``--loss`` omitted: the Namespace still carries the CLI default, but the
    # option is not explicit, so the recorded mode must win.
    args = _finetune_args(tmp_path, tmp_path / "source.pth", loss="arcface")

    resolved = finetune_loop_module._resolve_finetune_loss_config(args, checkpoint)

    assert resolved["loss"] == "supcon"
    assert resolved["supcon_temperature"] == 0.05
    assert resolved["subcenters"] is None


def test_resolve_finetune_loss_mode_resume_allows_equal_and_rejects_conflict(tmp_path):
    checkpoint = {"config": {"loss": "supcon", "supcon_temperature": 0.07}}

    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="supcon", supcon_temperature=0.07
    )
    resolved = finetune_loop_module._resolve_finetune_loss_config(
        args, checkpoint, explicit_options={"loss", "supcon_temperature"}
    )
    assert resolved["loss"] == "supcon"
    assert resolved["supcon_temperature"] == 0.07

    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="supcon", supcon_temperature=0.1
    )
    with pytest.raises(ValueError):
        finetune_loop_module._resolve_finetune_loss_config(
            args, checkpoint, explicit_options={"loss", "supcon_temperature"}
        )

    args = _finetune_args(tmp_path, tmp_path / "source.pth", loss="arcface")
    with pytest.raises(ValueError):
        finetune_loop_module._resolve_finetune_loss_config(
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

    resolved = finetune_loop_module._resolve_finetune_loss_config(
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
        finetune_loop_module._resolve_finetune_loss_config(
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
        finetune_loop_module._resolve_finetune_loss_config(
            args, checkpoint, explicit_options={"subcenters"}
        )

    # Omitting K still resolves the recorded K=2, not an arbitrary default.
    args = _finetune_args(
        tmp_path, tmp_path / "source.pth", loss="subcenter-arcface-compact"
    )
    resolved = finetune_loop_module._resolve_finetune_loss_config(args, checkpoint)
    assert resolved["subcenters"] == 2


def test_resolve_finetune_loss_mode_rejects_new_mode_on_legacy_resume(tmp_path):
    legacy = {"config": {}, "loss_state_dict": {"head.weight": torch.zeros(3, 4)}}

    args = _finetune_args(tmp_path, tmp_path / "source.pth", loss="supcon")
    with pytest.raises(ValueError, match="legacy"):
        finetune_loop_module._resolve_finetune_loss_config(
            args, legacy, explicit_options={"loss"}
        )

    args = _finetune_args(tmp_path, tmp_path / "source.pth", loss="arcface")
    resolved = finetune_loop_module._resolve_finetune_loss_config(
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
    assert finetune_loop_module._classify_finetune_source(ssl) == "ssl"
    assert finetune_loop_module._select_finetune_embedding_head(ssl) == "arcface_mlp_512"

    v080_supcon = {
        "model_state_dict": _ssl_like_state(),
        "config": {"loss": "supcon", "embedding_head": "arcface_mlp_512"},
        "loss_state_dict": {},
    }
    assert finetune_loop_module._classify_finetune_source(v080_supcon) == "v080"
    assert (
        finetune_loop_module._select_finetune_embedding_head(v080_supcon) == "arcface_mlp_512"
    )

    v080_arcface = {
        "model_state_dict": _ssl_like_state(),
        "config": {"loss": "arcface", "embedding_head": "arcface_mlp_512"},
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
    }
    assert finetune_loop_module._classify_finetune_source(v080_arcface) == "v080"

    legacy_with_head = {
        "model_state_dict": _ssl_like_state(),
        "config": {"embedding_head": "projection_mlp_2048"},
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
    }
    assert finetune_loop_module._classify_finetune_source(legacy_with_head) == "legacy_arcface"
    assert (
        finetune_loop_module._select_finetune_embedding_head(legacy_with_head)
        == "projection_mlp_2048"
    )

    historical = {
        "model_state_dict": _ssl_like_state(),
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
    }
    assert finetune_loop_module._classify_finetune_source(historical) == "legacy_arcface"
    assert (
        finetune_loop_module._select_finetune_embedding_head(historical) == "projection_mlp_2048"
    )

    head_only = {
        "config": {"embedding_head": "arcface_mlp_512"},
        "model_state_dict": _ssl_like_state(),
    }
    assert finetune_loop_module._classify_finetune_source(head_only) == "head_only"
    assert (
        finetune_loop_module._select_finetune_embedding_head(head_only) == "arcface_mlp_512"
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
    assert finetune_loop_module._classify_finetune_source(repo_ssl) == "ssl"


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
        finetune_loop_module._classify_finetune_source(checkpoint)


def test_finetune_legacy_inference_requires_projector_weights():
    # No model state at all: rejected before head inference is attempted.
    weightless = {
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
        "config": {},
    }
    with pytest.raises(ValueError, match="encoder weights"):
        finetune_loop_module._select_finetune_embedding_head(weightless)

    # Encoder weights present, but no projector weights and no declared head:
    # there is nothing to infer the embedding head from.
    no_projector = {
        "model_state_dict": {"backbone.cls_token": torch.zeros(1, 1, 192)},
        "loss_state_dict": {"head.weight": torch.zeros(2, 16)},
        "config": {},
    }
    with pytest.raises(ValueError, match="projector"):
        finetune_loop_module._select_finetune_embedding_head(no_projector)


# --- manifest hashing -------------------------------------------------------


def test_train_manifest_hash_is_permutation_invariant_and_content_sensitive(
    tmp_path,
):
    root = tmp_path / "images"
    refs = ["b.jpg", "a.jpg", "a.jpg"]
    labels = ["classB", "classA", "classA"]
    baseline = finetune_loop_module._train_manifest_sha256(refs, labels, root)

    assert (
        finetune_loop_module._train_manifest_sha256(
            ["a.jpg", "b.jpg", "a.jpg"], ["classA", "classB", "classA"], root
        )
        == baseline
    )
    # ``./img.jpg`` and ``img.jpg`` are the same canonical reference.
    assert (
        finetune_loop_module._train_manifest_sha256(
            ["./a.jpg", "a.jpg", "b.jpg"], ["classA", "classA", "classB"], root
        )
        == baseline
    )
    # Duplicate rows and label assignments are part of the manifest.
    assert (
        finetune_loop_module._train_manifest_sha256(
            refs + ["a.jpg"], labels + ["classA"], root
        )
        != baseline
    )
    assert (
        finetune_loop_module._train_manifest_sha256(
            refs, ["classB", "classA", "classB"], root
        )
        != baseline
    )


def test_train_manifest_hash_ignores_the_absolute_root(tmp_path):
    labels = ["classA", "classA"]
    one = tmp_path / "one"
    two = tmp_path / "two"

    first = finetune_loop_module._train_manifest_sha256(
        [str(one / "a.jpg"), "b.jpg"], labels, one
    )
    second = finetune_loop_module._train_manifest_sha256(
        [str(two / "a.jpg"), "b.jpg"], labels, two
    )

    assert first == second


def test_train_manifest_hash_rejects_references_outside_the_root(tmp_path):
    with pytest.raises(ValueError, match="input-images-dir"):
        finetune_loop_module._train_manifest_sha256(
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
    assert finetune_loop_module._train_manifest_sha256(
        ds.image_refs, ds.label_names, img_dir
    ) == finetune_loop_module._train_manifest_sha256(
        ["img_0.jpg", "img_1.jpg"], ["classA", "classB"], img_dir
    )


# --- checkpoint provenance --------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "payload.bin"
    path.write_bytes(b"otuformer" * 5000)

    assert finetune_loop_module._sha256_file(path) == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _expected_manifest(tmp_path):
    return finetune_loop_module._train_manifest_sha256(
        [f"img_{i}.jpg" for i in range(4)],
        ["classA", "classA", "classB", "classB"],
        tmp_path / "images",
    )


def test_finetune_records_provenance_and_effective_config(tmp_path):
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth", image_size=32, out_dim=16)

    saved = _run_frozen_finetune(tmp_path, ssl, "provenance")

    config = saved["config"]
    ssl_sha = finetune_loop_module._sha256_file(ssl)
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
        finetune_loop_module._sha256_file(finetune_path)
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

    arcface = finetune_loop_module._build_finetune_loss(
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

    supcon = finetune_loop_module._build_finetune_loss(
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

    subcenter = finetune_loop_module._build_finetune_loss(
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

    compact = finetune_loop_module._build_finetune_loss(
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
    finetune_support_module._validate_v080_loss_state(
        {"loss": "subcenter-arcface-compact", "subcenters": 3},
        {"head.weight": torch.zeros(4, 3, 16)},
    )
    with pytest.raises(ValueError, match="centers"):
        finetune_support_module._validate_v080_loss_state(
            {"loss": "subcenter-arcface-compact", "subcenters": 3},
            {"head.weight": torch.zeros(4, 2, 16)},
        )
    # A compact checkpoint written before K was recorded still expects K=2.
    finetune_support_module._validate_v080_loss_state(
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
            finetune_loop_module.run_finetune(args)


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
    finetune_loop_module.run_finetune(args)
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
    finetune_loop_module.run_finetune(resume_args)
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
        finetune_loop_module.run_finetune(args)

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
    finetune_loop_module._freeze_backbone_blocks(encoder, 0.7)
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

    finetune_loop_module.run_finetune(args)

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

    monkeypatch.setattr(finetune_loop_module, "load_checkpoint", fail)
    args = _finetune_args(
        tmp_path,
        source,
        resume=str(source),
        finetune_epochs=2,
        **_finetune_overrides(tmp_path, "preloaded_resume", batch_size=4),
    )

    finetune_loop_module.run_finetune(args, source_checkpoint=preloaded)

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

    finetune_loop_module.run_finetune(args)

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

    expected = finetune_loop_module._center_direction_cosine(
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
    finetune_loop_module.run_finetune(first)
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
    finetune_loop_module.run_finetune(resume)

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
    finetune_loop_module.run_finetune(first_args)
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
    finetune_loop_module.run_finetune(second_args)
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
    finetune_loop_module.run_finetune(first)

    resume = _finetune_args(
        tmp_path,
        ssl,
        resume=str(tmp_path / out_name / "finetune_latest.pth"),
        finetune_epochs=2,
        loss="subcenter-arcface-compact",
        **_finetune_overrides(tmp_path, out_name, batch_size=4),
    )
    finetune_loop_module.run_finetune(resume)

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

    expected = finetune_loop_module._center_direction_cosine(
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
        finetune_loop_module.run_finetune(args)


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
        finetune_loop_module.run_finetune(args)


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
        finetune_loop_module.run_finetune(args)


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
        finetune_loop_module.run_finetune(args)


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

    counts = finetune_loop_module._empty_loss_diagnostic_counts(2, 2)
    finetune_loop_module._accumulate_loss_diagnostics(
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

    counts = finetune_loop_module._empty_loss_diagnostic_counts(2, 1)
    finetune_loop_module._accumulate_loss_diagnostics(
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

    fraction, mean_penalty = finetune_loop_module._compact_hinge_stats(loss)

    assert fraction == pytest.approx(0.5)
    assert mean_penalty == pytest.approx(0.1 * 0.4 / 2)
    assert finetune_loop_module._compact_hinge_stats(ArcFaceLoss(embed_dim=4, num_classes=2)) == (
        None,
        None,
    )


def test_center_direction_cosine_is_empty_before_the_second_epoch():
    from otuformer.training.loss import ArcFaceLoss, SupConLoss

    loss = ArcFaceLoss(embed_dim=4, num_classes=2)
    first = finetune_loop_module._prototype_centers(loss)
    assert first is not None
    assert finetune_loop_module._center_direction_cosine(None, first) == []

    with torch.no_grad():
        loss.head.weight.mul_(2.0)
    second = finetune_loop_module._prototype_centers(loss)
    scaled = finetune_loop_module._center_direction_cosine(first, second)
    assert scaled[0][0] == pytest.approx(1.0)
    assert scaled[1][0] == pytest.approx(1.0)

    with torch.no_grad():
        loss.head.weight[0].neg_()
    third = finetune_loop_module._prototype_centers(loss)
    cosines = finetune_loop_module._center_direction_cosine(second, third)
    assert cosines[0][0] == pytest.approx(-1.0)
    assert cosines[1][0] == pytest.approx(1.0)

    assert finetune_loop_module._prototype_centers(SupConLoss(temperature=0.07)) is None


def test_loss_diagnostics_logger_creates_the_file_and_rejects_unknown_schema(
    tmp_path,
):
    path = tmp_path / "nested" / "loss_diagnostics.finetune.csv"
    logger = finetune_loop_module.LossDiagnosticsLogger(path)
    logger.log(epoch=0, mode="arcface", usable_batches=2)

    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(finetune_support_module._LOSS_DIAGNOSTIC_FIELDS)
    assert "arcface" in text

    broken = tmp_path / "broken.csv"
    broken.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        finetune_loop_module.LossDiagnosticsLogger(broken)
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


def test_cb_drw_resume_restores_the_original_horizon(tmp_path):
    img_dir, labels_path = _write_tiny_ft_data(tmp_path)
    checkpoint = _write_pretrain_checkpoint(tmp_path / "ssl.pth")
    args = _finetune_args(
        tmp_path,
        checkpoint,
        out_dir=str(tmp_path / "ft_cbdrw"),
        train_data=str(labels_path),
        input_images_dir=str(img_dir),
        finetune_epochs=1,
        batch_size=4,
        num_workers=0,
        log_every_n_steps=100,
        save_every_epochs=1,
        keep_last_checkpoints=0,
        long_tail="cb-drw",
    )
    finetune_loop_module.run_finetune(args)
    saved = torch.load(
        tmp_path / "ft_cbdrw" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )

    assert saved["config"]["long_tail"] == "cb-drw"
    record = saved["config"]["cb_drw"]
    assert record["total_steps"] == 1  # one batch per epoch, one epoch
    assert record["start_step"] == 0
    assert record["ramp_steps"] == 0  # T=1 applies the target from the first step
    assert record["expert_counts"] == [2, 2]
    assert record["target_weights"] == [1.0, 1.0]
    assert record["r_bar_step0"] == 1.0

    resume_args = _finetune_args(
        tmp_path,
        checkpoint,
        out_dir=str(tmp_path / "ft_cbdrw_resume"),
        resume=str(tmp_path / "ft_cbdrw" / "finetune_latest.pth"),
        train_data=str(labels_path),
        input_images_dir=str(img_dir),
        finetune_epochs=2,
        batch_size=4,
        num_workers=0,
        log_every_n_steps=100,
        save_every_epochs=1,
        keep_last_checkpoints=0,
        long_tail="cb-drw",
    )
    # Resume restores the original horizon T; extra epochs keep the target
    # weights instead of being rejected.
    finetune_loop_module.run_finetune(resume_args)
    resumed = torch.load(
        tmp_path / "ft_cbdrw_resume" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    record = resumed["config"]["cb_drw"]
    assert record["total_steps"] == 1
    # The recorded schedule must match the schedule actually applied: both come
    # from the restored T, not from the extended epoch count.
    assert record["start_step"] == int(1 * 0.5)
    assert record["ramp_steps"] == int(1 * 0.1)


def test_cb_drw_is_rejected_for_supcon(tmp_path):
    img_dir, labels_path = _write_tiny_ft_data(tmp_path)
    checkpoint = _write_pretrain_checkpoint(tmp_path / "ssl.pth")
    args = _finetune_args(
        tmp_path,
        checkpoint,
        out_dir=str(tmp_path / "ft_supcon"),
        train_data=str(labels_path),
        input_images_dir=str(img_dir),
        finetune_epochs=1,
        batch_size=2,
        num_workers=0,
        log_every_n_steps=100,
        save_every_epochs=1,
        keep_last_checkpoints=0,
        loss="supcon",
        long_tail="cb-drw",
    )
    with pytest.raises(ValueError, match="SupCon"):
        finetune_loop_module.run_finetune(args)


def _write_pseudo_ft_data(tmp_path, *, per_class=5):
    """Dedicated fixture with >=3 expert seeds per class, as the rule requires."""
    img_dir = tmp_path / "pseudo_images"
    img_dir.mkdir()
    rows = []
    for index in range(per_class * 2):
        label = "classA" if index < per_class else "classB"
        color = (index * 20 % 256, 40, 160) if label == "classA" else (30, index * 20 % 256, 90)
        Image.new("RGB", (64, 64), color=color).save(img_dir / f"img_{index}.jpg")
        rows.append({"image": f"img_{index}.jpg", "label": label})
    labels_path = tmp_path / "pseudo_labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    return img_dir, labels_path


def test_pseudo_round_end_to_end_and_third_round_rejected(tmp_path):
    img_dir, labels_path = _write_pseudo_ft_data(tmp_path)
    # Candidates are classA-like images with one changed pixel, so their bytes
    # differ from every expert seed while the raw-CLS features stay close.
    for index in range(2):
        image = Image.new("RGB", (64, 64), color=(index * 20 % 256, 40, 160))
        image.putpixel((0, 0), (1, 1, 1))
        image.save(img_dir / f"cand_{index}.jpg")
    # A hard-linked duplicate candidate exercises pre-scan rejection counting.
    os.link(img_dir / "cand_0.jpg", img_dir / "cand_2.jpg")
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth")
    finetune_loop_module.run_finetune(
        _finetune_args(
            tmp_path,
            ssl,
            out_dir=str(tmp_path / "ft1"),
            train_data=str(labels_path),
            input_images_dir=str(img_dir),
            finetune_epochs=1,
            batch_size=4,
            num_workers=0,
            log_every_n_steps=100,
            save_every_epochs=1,
            keep_last_checkpoints=0,
        )
    )
    ft1 = tmp_path / "ft1" / "finetune_latest.pth"

    pseudo = finetune_support_module.prepare_pseudo_round(
        pseudo_label_from=ft1,
        train_data=labels_path,
        input_images_dir=img_dir,
        out_dir=tmp_path / "ft2",
        similarity_floor=-1.0,
        min_gap=0.0,
        neighbors=3,
        long_tail="none",
        loss=None,
        checkpoint=None,
        device="cpu",
        batch_size=2,
        num_workers=0,
    )
    assert any(row["source"] == "known-pseudo" for row in pseudo["rows"])
    assert not (tmp_path / "ft2").exists()
    assert pseudo["summary"]["candidate_count"] == len(pseudo["diagnostics"])
    assert (
        pseudo["summary"]["rejection_reason_counts"].get("duplicate_file_identity")
        == 1
    )

    finetune_loop_module.run_finetune(
        _finetune_args(
            tmp_path,
            ssl,
            out_dir=str(tmp_path / "ft2"),
            train_data=str(labels_path),
            input_images_dir=str(img_dir),
            finetune_epochs=1,
            batch_size=4,
            num_workers=0,
            log_every_n_steps=100,
            save_every_epochs=1,
            keep_last_checkpoints=0,
            pseudo_round_data=pseudo,
        )
    )
    saved = torch.load(
        tmp_path / "ft2" / "finetune_latest.pth", map_location="cpu", weights_only=False
    )
    assert saved["config"]["pseudo_round"] == 1
    assert saved["config"]["accepted_pseudo_rows"]
    assert saved["config"]["pseudo_source_checkpoint_sha256"]
    assert (tmp_path / "ft2" / "pseudo_labels.csv").exists()
    assert (tmp_path / "ft2" / "pseudo_summary.json").exists()
    assert saved["config"]["accepted_pseudo_rows_sha256"]
    assert saved["config"]["pseudo_rule"]["eval_transform"] == "center-crop"
    assert saved["config"]["input_images_dir"] == str(img_dir.resolve())
    assert saved["config"]["pseudo_labels_sha256"]
    training = saved["config"]["pseudo_training_summary"]
    assert training["epochs"] == 1
    assert training["training_rows"] == (
        sum(training["expert_counts_by_class"].values())
        + sum(training["pseudo_counts_by_class"].values())
    )
    assert training["batches_per_epoch"] >= 1
    assert training["completed_optimizer_steps"] >= 1
    assert set(training["expert_counts_by_class"]) == {"classA", "classB"}
    assert saved["config"]["preprocessing"]["eval_transform"] == "center-crop"

    # A finetune#2 resume must also fix the experiment identity: an explicit
    # batch-size change is rejected even with unchanged training rows.
    with pytest.raises(ValueError, match="batch_size"):
        finetune_loop_module.run_finetune(
            _finetune_args(
                tmp_path,
                ssl,
                out_dir=str(tmp_path / "ft2_resume_bs"),
                resume=str(tmp_path / "ft2" / "finetune_latest.pth"),
                train_data=str(labels_path),
                input_images_dir=str(img_dir),
                finetune_epochs=1,
                batch_size=2,
                num_workers=0,
                log_every_n_steps=100,
                save_every_epochs=1,
                keep_last_checkpoints=0,
            )
        )

    # Resume rejects a changed expert manifest instead of silently continuing.
    changed = tmp_path / "changed_labels.csv"
    changed_frame = pd.read_csv(labels_path)
    changed_frame.loc[0, "label"] = "classB"
    changed_frame.to_csv(changed, index=False)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        finetune_loop_module.run_finetune(
            _finetune_args(
                tmp_path,
                ssl,
                out_dir=str(tmp_path / "ft2_resume"),
                resume=str(tmp_path / "ft2" / "finetune_latest.pth"),
                train_data=str(changed),
                input_images_dir=str(img_dir),
                finetune_epochs=1,
                batch_size=4,
                num_workers=0,
                log_every_n_steps=100,
                save_every_epochs=1,
                keep_last_checkpoints=0,
            )
        )

    with pytest.raises(ValueError, match="finetune#2"):
        finetune_support_module.prepare_pseudo_round(
            pseudo_label_from=tmp_path / "ft2" / "finetune_latest.pth",
            train_data=labels_path,
            input_images_dir=img_dir,
            out_dir=tmp_path / "ft3",
            similarity_floor=-1.0,
            min_gap=0.0,
            neighbors=3,
            long_tail="none",
            loss=None,
            checkpoint=None,
            device="cpu",
            batch_size=2,
            num_workers=0,
        )


def test_pseudo_mode_rejects_experiment_identity_mismatch(tmp_path):
    img_dir, labels_path = _write_pseudo_ft_data(tmp_path)
    image = Image.new("RGB", (64, 64), color=(5, 40, 160))
    image.putpixel((0, 0), (1, 1, 1))
    image.save(img_dir / "cand_0.jpg")
    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth")
    finetune_loop_module.run_finetune(
        _finetune_args(
            tmp_path,
            ssl,
            out_dir=str(tmp_path / "ft1"),
            train_data=str(labels_path),
            input_images_dir=str(img_dir),
            finetune_epochs=1,
            batch_size=4,
            num_workers=0,
            log_every_n_steps=100,
            save_every_epochs=1,
            keep_last_checkpoints=0,
        )
    )
    pseudo = finetune_support_module.prepare_pseudo_round(
        pseudo_label_from=tmp_path / "ft1" / "finetune_latest.pth",
        train_data=labels_path,
        input_images_dir=img_dir,
        out_dir=tmp_path / "ft2",
        similarity_floor=-1.0,
        min_gap=0.0,
        neighbors=3,
        long_tail="none",
        loss=None,
        checkpoint=None,
        device="cpu",
        batch_size=2,
        num_workers=0,
    )

    with pytest.raises(ValueError, match="freeze_ratio"):
        finetune_loop_module.run_finetune(
            _finetune_args(
                tmp_path,
                ssl,
                out_dir=str(tmp_path / "ft2"),
                train_data=str(labels_path),
                input_images_dir=str(img_dir),
                finetune_epochs=1,
                batch_size=4,
                num_workers=0,
                log_every_n_steps=100,
                save_every_epochs=1,
                keep_last_checkpoints=0,
                freeze_ratio=0.5,
                pseudo_round_data=pseudo,
            )
        )


def test_pseudo_round_preserves_numeric_class_order_across_resume(tmp_path):
    img_dir = tmp_path / "num_images"
    img_dir.mkdir()
    rows = []
    for index in range(10):
        label = 2 if index < 5 else 10
        color = (index * 20 % 256, 40, 160) if label == 2 else (30, index * 20 % 256, 90)
        Image.new("RGB", (64, 64), color=color).save(img_dir / f"img_{index}.jpg")
        rows.append({"image": f"img_{index}.jpg", "label": label})
    labels_path = tmp_path / "num_labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    for index in range(2):
        image = Image.new("RGB", (64, 64), color=(index * 20 % 256, 40, 160))
        image.putpixel((0, 0), (1, 1, 1))
        image.save(img_dir / f"cand_{index}.jpg")

    ssl = _write_pretrain_checkpoint(tmp_path / "ssl.pth")
    finetune_loop_module.run_finetune(
        _finetune_args(
            tmp_path, ssl, out_dir=str(tmp_path / "num_ft1"),
            train_data=str(labels_path), input_images_dir=str(img_dir),
            finetune_epochs=1, batch_size=4, num_workers=0,
            log_every_n_steps=100, save_every_epochs=1, keep_last_checkpoints=0,
        )
    )
    ft1 = tmp_path / "num_ft1" / "finetune_latest.pth"
    # The real ft1 checkpoint records class_order ["2", "10"] (numeric) while
    # class_labels keeps the historical sorted-string ["10", "2"].
    source = torch.load(ft1, map_location="cpu", weights_only=False)
    assert source["class_order"] == ["2", "10"]
    assert source["class_labels"] == ["10", "2"]

    pseudo = finetune_support_module.prepare_pseudo_round(
        pseudo_label_from=ft1, train_data=labels_path, input_images_dir=img_dir,
        out_dir=tmp_path / "num_ft2", similarity_floor=-1.0, min_gap=0.0,
        neighbors=3, long_tail="none", loss=None, checkpoint=None,
        device="cpu", batch_size=2, num_workers=0,
    )
    assert pseudo["class_labels"] == ["2", "10"]

    finetune_loop_module.run_finetune(
        _finetune_args(
            tmp_path, ssl, out_dir=str(tmp_path / "num_ft2"),
            train_data=str(labels_path), input_images_dir=str(img_dir),
            finetune_epochs=1, batch_size=4, num_workers=0,
            log_every_n_steps=100, save_every_epochs=1, keep_last_checkpoints=0,
            pseudo_round_data=pseudo,
        )
    )
    ft2 = tmp_path / "num_ft2" / "finetune_latest.pth"
    saved = torch.load(ft2, map_location="cpu", weights_only=False)
    # Save must not re-sort the numeric class table into ["10", "2"].
    assert saved["class_labels"] == ["2", "10"]
    assert saved["class_order"] == ["2", "10"]

    finetune_loop_module.run_finetune(
        _finetune_args(
            tmp_path, ssl, out_dir=str(tmp_path / "num_ft2_resume"),
            resume=str(ft2), train_data=str(labels_path),
            input_images_dir=str(img_dir), finetune_epochs=2, batch_size=4,
            num_workers=0, log_every_n_steps=100, save_every_epochs=1,
            keep_last_checkpoints=0,
        )
    )
    resumed = torch.load(
        tmp_path / "num_ft2_resume" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed["class_labels"] == ["2", "10"]


def test_pseudo_cli_preflight_accepts_numeric_class_order(tmp_path, monkeypatch):
    """The public CLI preflight must accept labels whose sorted string order
    differs from the classifier row order (2 before 10)."""
    from typer.testing import CliRunner

    from otuformer.cli.main import app
    from otuformer.embedding import pseudo_label as pl_module

    img_dir = tmp_path / "num_cli_images"
    img_dir.mkdir()
    rows = []
    for index in range(10):
        label = 2 if index < 5 else 10
        Image.new("RGB", (64, 64), color=(index * 20 % 256, 40, 160)).save(
            img_dir / f"img_{index}.jpg"
        )
        rows.append({"image": f"img_{index}.jpg", "label": label})
    for index in range(2):
        image = Image.new("RGB", (64, 64), color=(index * 20 % 256, 40, 160))
        image.putpixel((0, 0), (1, 1, 1))
        image.save(img_dir / f"cand_{index}.jpg")
    labels_path = tmp_path / "num_cli_labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)

    ssl = _write_pretrain_checkpoint(tmp_path / "num_cli_ssl.pth")
    finetune_loop_module.run_finetune(
        _finetune_args(
            tmp_path, ssl, out_dir=str(tmp_path / "num_cli_ft1"),
            train_data=str(labels_path), input_images_dir=str(img_dir),
            finetune_epochs=1, batch_size=4, num_workers=0,
            log_every_n_steps=100, save_every_epochs=1, keep_last_checkpoints=0,
        )
    )
    ft1 = tmp_path / "num_cli_ft1" / "finetune_latest.pth"

    monkeypatch.setattr(
        pl_module,
        "generate_pseudo_rows",
        lambda **kwargs: (
            [{"image": "cand_0.jpg", "label": "2"}],
            [{"image": "cand_0.jpg", "accepted": True, "rejection_reasons": []}],
            {
                "thresholds": {"similarity_floor": 0.0, "min_gap": 0.0, "neighbors": 3},
                "candidate_count": 1,
                "accepted_count": 1,
                "acceptance_rate": 1.0,
                "rejection_reason_counts": {},
                "seed_count_bins": {},
            },
        ),
    )
    monkeypatch.setattr(finetune_loop_module, "run_finetune", lambda args, **kwargs: None)

    result = CliRunner().invoke(
        app,
        [
            "finetune",
            "--train-data", str(labels_path),
            "--input-images-dir", str(img_dir),
            "--out-dir", str(tmp_path / "num_cli_ft2"),
            "--pseudo-label-from", str(ft1),
            "--device", "cpu",
            "--batch-size", "4",
            "--num-workers", "0",
            "--save-every-epochs", "1",
            "--log-every-n-steps", "100",
            "--keep-last-checkpoints", "0",
        ],
    )

    assert result.exit_code == 0, result.output


def test_resolve_pretrain_settings_is_pure_and_single_valued():
    """One resolution, no checkpoint mutation, no RNG consumption."""
    args = argparse.Namespace(
        model_name="vit_tiny_patch16_224",
        global_crop_size=None,
        local_crop_size=None,
        local_crops=None,
    )
    torch.manual_seed(0)
    before = torch.get_rng_state()

    settings = pretrain_loop_module.resolve_pretrain_settings(args)
    assert torch.equal(torch.get_rng_state(), before), "resolution advanced the RNG"
    pretrain_loop_module.preflight_pretrain_run(args, settings)
    assert torch.equal(torch.get_rng_state(), before), "preflight advanced the RNG"
    assert vars(args) == {
        "model_name": "vit_tiny_patch16_224",
        "global_crop_size": None,
        "local_crop_size": None,
        "local_crops": None,
    }
    assert settings.global_crop_size == 224
    assert (settings.local_crop_size, settings.local_crops) == (96, 6)
    assert settings.patch_mode == "consistency"

    explicit = argparse.Namespace(
        model_name="vit_tiny_patch16_224",
        global_crop_size=224,
        local_crop_size=96,
        local_crops=6,
    )
    auto = pretrain_loop_module.resolve_pretrain_settings(args)
    assert pretrain_loop_module.resolve_pretrain_settings(explicit) == auto
    assert pretrain_loop_module.resolve_pretrain_settings(args) == auto


def test_run_pretrain_reuses_a_handed_over_resolution(tmp_path, monkeypatch):
    """A handed-over resolution must not be resolved or preflighted twice."""
    settings = pretrain_recipe_module.PretrainSettings(
        model_name="vit_tiny_patch16_224",
        global_crop_size=224,
        local_crop_size=96,
        local_crops=0,
        patch_config={
            "patch_loss": "none",
            "masking_strategy": None,
            "mask_ratio_requested": None,
            "mask_ratio_resolved": None,
            "ibot_prototypes": None,
            "patch_target_layers": None,
            "patch_center_momentum": None,
        },
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the handed-over resolution was redone")

    monkeypatch.setattr(pretrain_loop_module, "resolve_pretrain_settings", forbidden)
    monkeypatch.setattr(pretrain_loop_module, "preflight_pretrain_run", forbidden)

    class DatasetReached(Exception):
        """Stop the run at the first trainer-owned step after preflight."""

    monkeypatch.setattr(
        pretrain_loop_module, "MultiCropDataset",
        lambda **_kwargs: (_ for _ in ()).throw(DatasetReached()),
    )
    args = argparse.Namespace(
        out_dir=str(tmp_path / "handoff_out"),
        seed=42,
        cpus=0,
        device="cpu",
        register_tokens=None,
        train_data="",
        input_images_dir=str(tmp_path),
        batch_size=2,
        num_workers=0,
        global_crop_size=None,
        local_crop_size=None,
        local_crops=None,
        augmentation=None,
        orientation_policy=None,
        resume="",
        out_dim=16,
        pretrain_settings=settings,
    )

    with pytest.raises(DatasetReached):
        pretrain_loop_module.run_pretrain(args)

    # The resolved values still drive the trainer's own bookkeeping.
    assert args.global_crop_size == 224
    assert (args.local_crop_size, args.local_crops) == (96, 0)


# --- relational target recipe resolution ------------------------------------

def _recipe_args(**overrides):
    base = {
        "batch_size": 8,
        "augmentation": None,
        "orientation_policy": None,
        "relational_teacher_views": None,
        "relational_center_source": None,
        "relational_pairing": None,
        "teacher_temp_start": None,
        "teacher_temp_end": None,
        "teacher_temp_warmup_ratio": 0.7,
        "teacher_temp_warmup_ratio_explicit": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _recipe_settings(global_crop_size=32, local_crop_size=16, local_crops=0):
    return pretrain_recipe_module.PretrainSettings(
        model_name="vit_tiny_patch16_224",
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        local_crops=local_crops,
        patch_config={
            "patch_loss": "none",
            "masking_strategy": None,
            "mask_ratio_requested": None,
            "mask_ratio_resolved": None,
            "ibot_prototypes": None,
            "patch_target_layers": None,
            "patch_center_momentum": None,
        },
    )


def _recipe(overrides=None, checkpoint=None, settings=None):
    return pretrain_loop_module._resolve_pretrain_recipe(
        _recipe_args(**(overrides or {})),
        checkpoint,
        settings=settings or _recipe_settings(),
    )


def _bundle(**overrides):
    bundle = {
        "version": 1,
        "relational_teacher_views": "ordinary",
        "relational_pairing": "full",
        "relational_center_source": "ordinary",
        "teacher_temp_start": 0.04,
        "teacher_temp_end": 0.07,
        "weak_teacher_augmentation_config": None,
    }
    bundle.update(overrides)
    return bundle


_DEFAULT_BUNDLE = object()


def _versioned_checkpoint(bundle=_DEFAULT_BUNDLE, args_extra=None):
    saved_args = {
        "relational_teacher_views": "ordinary",
        "relational_pairing": "full",
        "relational_center_source": "ordinary",
        "teacher_temp_start": 0.04,
        "teacher_temp_end": 0.07,
        "teacher_temp_warmup_ratio": 0.7,
    }
    saved_args.update(args_extra or {})
    resolved_bundle = _bundle() if bundle is _DEFAULT_BUNDLE else bundle
    return {
        "config": {
            "augmentation_profile": "global-barcode",
            "augmentation_config": build_pretrain_augmentation_config(
                "global-barcode", 32, 16, 0
            ),
            pretrain_loop_module.RELATIONAL_TARGET_CONFIG_KEY: resolved_bundle,
        },
        "args": saved_args,
    }


def _legacy_checkpoint(args_extra=None):
    saved_args = {
        "augmentation": "global-barcode",
        "disable_cross_view_loss": False,
        "teacher_temp_start": 0.04,
        "teacher_temp_end": 0.07,
    }
    saved_args.update(args_extra or {})
    return {"config": {}, "args": saved_args}


def test_relational_recipe_resolves_requests_and_history():
    recipe = _recipe()
    assert recipe.relational_target_config == {
        "version": 1,
        "relational_teacher_views": "weak",
        "relational_pairing": "cross",
        "relational_center_source": "weak",
        "teacher_temp_start": 0.08,
        "teacher_temp_end": 0.11,
        "weak_teacher_augmentation_config": recipe.weak_teacher_augmentation_config,
    }
    assert recipe.weak_teacher_augmentation_config is not None
    assert recipe.teacher_temp_warmup_ratio == 0.7

    ordinary = _recipe({"relational_teacher_views": "ordinary"})
    assert ordinary.relational_target_config["relational_center_source"] == "ordinary"
    assert ordinary.weak_teacher_augmentation_config is None

    both = _recipe(
        {
            "relational_teacher_views": "weak",
            "relational_center_source": "ordinary",
            "relational_pairing": "full",
            "teacher_temp_start": 0.05,
            "teacher_temp_end": 0.09,
        }
    )
    assert both.relational_target_config["relational_center_source"] == "ordinary"
    assert both.relational_target_config["relational_pairing"] == "full"
    assert both.relational_target_config["teacher_temp_start"] == 0.05

    # the ordinary config is never mutated and never gains normalize keys
    assert "normalize" not in both.augmentation_config

    for bad in (
        {"relational_teacher_views": "ordinary", "relational_center_source": "weak"},
        {"relational_pairing": "same"},
        {"relational_pairing": "zip"},
        {"relational_teacher_views": "strong"},
        {"teacher_temp_start": True},
        {"teacher_temp_start": float("inf")},
        {"teacher_temp_start": -0.04},
        {"teacher_temp_start": 0.0},
        {"teacher_temp_start": 0.0009},
        {"teacher_temp_end": 101.0},
    ):
        with pytest.raises(ValueError):
            _recipe(bad)

    # deterministic conservative admission bound at 2e-38: fresh requests are
    # banded, so admission is exercised through an inherited saved endpoint
    for batch_size, should_pass in ((6, True), (7, False)):
        checkpoint = _versioned_checkpoint(
            _bundle(teacher_temp_start=2e-38),
            {"teacher_temp_start": 2e-38},
        )
        if should_pass:
            assert _recipe(
                {"batch_size": batch_size}, checkpoint=checkpoint
            ).relational_target_config["teacher_temp_start"] == 2e-38
        else:
            with pytest.raises(ValueError, match="overflows"):
                _recipe({"batch_size": batch_size}, checkpoint=checkpoint)
    # 2/tau alone is finite at 6e-39 and B=6, but the mean reduction overflows
    six = _versioned_checkpoint(
        _bundle(teacher_temp_start=6e-39), {"teacher_temp_start": 6e-39}
    )
    with pytest.raises(ValueError, match="overflows"):
        _recipe({"batch_size": 6}, checkpoint=six)
    # a double-precision quotient that is finite but float32-infinite
    assert math.isfinite(7 / 2e-38)


def test_relational_recipe_legacy_and_versioned_resolution():
    legacy = _recipe(checkpoint=_legacy_checkpoint())
    assert legacy.relational_target_config["relational_teacher_views"] == "ordinary"
    assert legacy.relational_target_config["relational_center_source"] == "ordinary"
    assert legacy.relational_target_config["relational_pairing"] == "full"
    assert legacy.relational_target_config["teacher_temp_start"] == 0.04
    assert legacy.relational_target_config["teacher_temp_end"] == 0.07
    assert legacy.teacher_temp_warmup_ratio == 0.7

    same = _recipe(
        checkpoint=_legacy_checkpoint({"disable_cross_view_loss": True})
    )
    assert same.relational_target_config["relational_pairing"] == "same"

    assumed = _recipe(checkpoint={"config": {}, "args": {}})
    assert assumed.relational_target_config["relational_pairing"] == "full"
    assert any("disable_cross_view_loss" in w for w in assumed.warnings)
    assert any("teacher_temp_start" in w for w in assumed.warnings)

    # explicit requests equal to the effective values are accepted
    accepted = _recipe(
        {"teacher_temp_start": 0.04, "teacher_temp_end": 0.07, "relational_pairing": "full"},
        checkpoint=_legacy_checkpoint(),
    )
    assert accepted.relational_target_config["teacher_temp_start"] == 0.04

    # explicit numeric 1 matches a saved integer 1 after normalization
    integer_saved = _legacy_checkpoint(
        {"teacher_temp_start": 1, "teacher_temp_end": 0.07}
    )
    matched = _recipe({"teacher_temp_start": 1}, checkpoint=integer_saved)
    assert matched.relational_target_config["teacher_temp_start"] == 1.0

    # exact numeric matching, not an absolute tolerance
    tiny = _legacy_checkpoint(
        {"teacher_temp_start": 2e-20, "teacher_temp_end": 0.07}
    )
    with pytest.raises(ValueError, match="conflicts"):
        _recipe({"teacher_temp_start": 2e-13}, checkpoint=tiny)
    assert _recipe({"teacher_temp_start": 2e-20}, checkpoint=tiny).relational_target_config[
        "teacher_temp_start"
    ] == 2e-20

    # versioned bundle wins; explicit conflicts fail, including new-run defaults
    versioned = _recipe(checkpoint=_versioned_checkpoint())
    assert versioned.relational_target_config["relational_pairing"] == "full"
    for conflict in (
        {"relational_teacher_views": "weak"},
        {"relational_pairing": "cross"},
        {"teacher_temp_start": 0.08},
    ):
        with pytest.raises(ValueError, match="conflicts"):
            _recipe(conflict, checkpoint=_versioned_checkpoint())

    # a non-default recorded ratio is inherited, and an equal request passes
    ratio_ckpt = _versioned_checkpoint(
        args_extra={"teacher_temp_warmup_ratio": 0.3}
    )
    assert _recipe(checkpoint=ratio_ckpt).teacher_temp_warmup_ratio == 0.3
    assert _recipe(
        {"teacher_temp_warmup_ratio": 0.3}, checkpoint=ratio_ckpt
    ).teacher_temp_warmup_ratio == 0.3
    with pytest.raises(ValueError, match="0.3"):
        _recipe(
            {
                "teacher_temp_warmup_ratio": 0.7,
                "teacher_temp_warmup_ratio_explicit": True,
            },
            checkpoint=ratio_ckpt,
        )

    # an unrecorded ratio warns and rejects an explicit non-default
    missing = _versioned_checkpoint()
    missing["args"].pop("teacher_temp_warmup_ratio")
    assert _recipe(checkpoint=missing).teacher_temp_warmup_ratio == 0.7
    assert any("teacher_temp_warmup_ratio" in w for w in _recipe(checkpoint=missing).warnings)
    with pytest.raises(ValueError, match="warmup ratio|records no"):
        _recipe({"teacher_temp_warmup_ratio": 0.3}, checkpoint=missing)

    # a format-legal saved endpoint outside the band is inherited, and an
    # explicit request equal to it is accepted
    outside = _versioned_checkpoint(
        _bundle(teacher_temp_start=0.0009, teacher_temp_end=0.07),
        {"teacher_temp_start": 0.0009},
    )
    assert _recipe(checkpoint=outside).relational_target_config[
        "teacher_temp_start"
    ] == 0.0009
    assert _recipe(
        {"teacher_temp_start": 0.0009}, checkpoint=outside
    ).relational_target_config["teacher_temp_start"] == 0.0009

    # requests and the checkpoint are not mutated
    args = _recipe_args()
    checkpoint = _versioned_checkpoint()
    snapshot = json.dumps(checkpoint, sort_keys=True)
    pretrain_recipe_module._resolve_relational_target_config(args, checkpoint)
    assert args.relational_teacher_views is None
    assert json.dumps(checkpoint, sort_keys=True) == snapshot


def test_relational_recipe_rejects_bad_metadata():
    for bad_bundle in (
        None,
        {},
        {"version": True},
        {"version": 2},
        _bundle(relational_center_source="weak"),
        _bundle(relational_pairing="same", relational_teacher_views="weak"),
        _bundle(teacher_temp_start=True),
        _bundle(teacher_temp_start=0.0),
        _bundle(teacher_temp_start=float("nan")),
        _bundle(weak_teacher_augmentation_config={"profile": "legacy"}),
    ):
        checkpoint = _versioned_checkpoint(bad_bundle)
        with pytest.raises(ValueError):
            _recipe(checkpoint=checkpoint)

    # the retired spelling is not an alias
    retired = _versioned_checkpoint()
    del retired["config"][pretrain_loop_module.RELATIONAL_TARGET_CONFIG_KEY][
        "relational_center_source"
    ]
    retired["config"][pretrain_loop_module.RELATIONAL_TARGET_CONFIG_KEY][
        "teacher_center_source"
    ] = "ordinary"
    with pytest.raises(ValueError, match="incomplete"):
        _recipe(checkpoint=retired)

    # duplicate disagreement between bundle and saved args
    disagreeing = _versioned_checkpoint()
    disagreeing["args"]["relational_pairing"] = "cross"
    with pytest.raises(ValueError, match="disagrees"):
        _recipe(checkpoint=disagreeing)

    # type confusion: args true must not equal bundle 1.0
    typed = _versioned_checkpoint(
        _bundle(teacher_temp_start=1.0), {"teacher_temp_start": True}
    )
    with pytest.raises(ValueError, match="disagrees"):
        _recipe(checkpoint=typed)

    # partial new-style markers without a bundle
    for marker, value in (
        ("relational_teacher_views", "weak"),
        ("relational_pairing", "cross"),
        ("relational_center_source", "weak"),
        ("teacher_center_source", "weak"),
        ("weak_teacher_augmentation_config", None),
        ("weak_relational_teacher_views", False),
    ):
        partial = {"config": {}, "args": {marker: value}}
        with pytest.raises(ValueError, match="partial"):
            _recipe(checkpoint=partial)

    # weak-config drift is detected against the current derivation
    weak_bundle = _bundle(
        relational_teacher_views="weak",
        relational_center_source="weak",
        relational_pairing="cross",
        weak_teacher_augmentation_config={
            **build_pretrain_augmentation_config("global-barcode", 32, 16, 0),
            "color_jitter": {
                "probability": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "saturation": 0.0,
                "hue": 0.0,
            },
            "grayscale_probability": 0.0,
            "blur": {
                **build_pretrain_augmentation_config("global-barcode", 32, 16, 0)["blur"],
                "probabilities": [0.0, 0.0, 0.5],
            },
            "normalize": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        },
    )
    weak_ckpt = _versioned_checkpoint(
        weak_bundle,
        {
            "relational_teacher_views": "weak",
            "relational_center_source": "weak",
            "relational_pairing": "cross",
        },
    )
    with pytest.raises(ValueError, match="weak teacher augmentation"):
        _recipe(checkpoint=weak_ckpt)


def test_relational_recipe_accepts_v2_marked_resume():
    """A v2/DINO-marked payload is no longer rejected by key presence.

    Cross-method resume stays rejected, but by the SSL resolver that owns the
    method identity, not by the relational recipe resolver.
    """
    for payload in (
        {"config": {"ssl_method": "dino"}},
        {"config": {"ssl_head": "dino"}},
        {"dino_objective": {}},
        {"config": {"ssl_method": None}},
        {"dino_objective": None},
        {
            "config": {
                "ssl_method": "relational",
                pretrain_loop_module.RELATIONAL_TARGET_CONFIG_KEY: _bundle(),
            },
            "args": {},
        },
    ):
        assert _recipe(checkpoint=payload) is not None
    # marker-free legacy main still passes
    assert _recipe(checkpoint=_legacy_checkpoint()) is not None


# --- teacher temperature schedule after the ratio option -------------------

def test_relational_resume_teacher_schedule():
    """Ratio boundaries, one-step ramp semantics and the extension tail."""
    base = dict(
        teacher_temp_start=0.08,
        teacher_temp_end=0.11,
    )
    assert pretrain_recipe_module._build_teacher_temp_schedule(4, **base, teacher_temp_warmup_ratio=0.0) == pytest.approx(
        [0.11] * 4
    )
    full_ramp = pretrain_recipe_module._build_teacher_temp_schedule(
        4, **base, teacher_temp_warmup_ratio=1.0
    )
    assert full_ramp == pytest.approx([0.08, 0.09, 0.10, 0.11])
    # one planned step at ratio 1 keeps the start value, not the end
    assert pretrain_recipe_module._build_teacher_temp_schedule(
        1, **base, teacher_temp_warmup_ratio=1.0
    ) == pytest.approx([0.08])
    # a multi-step plan whose ramp truncates to one step also keeps the start
    assert pretrain_recipe_module._build_teacher_temp_schedule(
        10, **base, teacher_temp_warmup_ratio=0.1
    )[0] == pytest.approx(0.08)
    # truncation to zero ramp steps uses the end temperature throughout
    assert pretrain_recipe_module._build_teacher_temp_schedule(
        9, **base, teacher_temp_warmup_ratio=0.1
    ) == pytest.approx([0.11] * 9)
    # the default reproduces the pre-change 0.7 schedule bit for bit
    assert pretrain_recipe_module._build_teacher_temp_schedule(
        10, 0.04, 0.07
    ) == pytest.approx(
        pretrain_recipe_module._build_teacher_temp_schedule(
            10, 0.04, 0.07, teacher_temp_warmup_ratio=0.7
        )
    )


def test_relational_same_plan_schedule_uses_the_inherited_ratio():
    saved = {
        "original_max_epochs": 2,
        "total_steps": 4,
        "steps_per_epoch": 2,
        "warmup_steps": 0,
        "last_lr": 0.0005,
        "final_lr": 0.00001,
    }
    args = argparse.Namespace(
        max_epochs=2,
        lr=0.001,
        warmup_epochs=0,
        teacher_momentum=0.995,
        teacher_momentum_end=0.999,
        student_temp=0.1,
        teacher_temp_start=0.08,
        teacher_temp_end=0.11,
        teacher_temp_warmup_ratio=1.0,
    )
    _, _, _, teacher_temp, _ = pretrain_loop_module._build_pretrain_schedules(
        args, saved, steps_per_epoch=2, global_step=2, completed_epochs=1
    )
    assert teacher_temp == pytest.approx([0.08, 0.09, 0.10, 0.11])

    extended = argparse.Namespace(**{**vars(args), "max_epochs": 3})
    _, _, _, tail, _ = pretrain_loop_module._build_pretrain_schedules(
        extended, saved, steps_per_epoch=2, global_step=4, completed_epochs=2
    )
    assert tail[4:] == pytest.approx([0.11] * len(tail[4:]))


def test_relational_zero_ratio_reaches_the_real_schedule():
    """A legal zero ratio must not be replaced by the 0.7 fallback."""
    args = argparse.Namespace(
        max_epochs=5,
        lr=0.001,
        warmup_epochs=0,
        teacher_momentum=0.995,
        teacher_momentum_end=0.999,
        student_temp=0.1,
        teacher_temp_start=0.08,
        teacher_temp_end=0.11,
        teacher_temp_warmup_ratio=0.0,
    )
    _, _, _, fresh, _ = pretrain_loop_module._build_pretrain_schedules(
        args, None, steps_per_epoch=2, global_step=0, completed_epochs=0
    )
    assert fresh == pytest.approx([0.11] * len(fresh))

    saved = {
        "original_max_epochs": 5,
        "total_steps": 10,
        "steps_per_epoch": 2,
        "warmup_steps": 0,
        "last_lr": 0.0005,
        "final_lr": 0.00001,
    }
    _, _, _, same_plan, _ = pretrain_loop_module._build_pretrain_schedules(
        args, saved, steps_per_epoch=2, global_step=4, completed_epochs=2
    )
    assert same_plan == pytest.approx([0.11] * len(same_plan))

    # a missing attribute still falls back to the historical ramp
    legacy_args = argparse.Namespace(**vars(args))
    del legacy_args.teacher_temp_warmup_ratio
    ramped = pretrain_recipe_module._build_teacher_temp_schedule(
        10, 0.08, 0.11, pretrain_recipe_module._teacher_temp_warmup_ratio(legacy_args)
    )
    assert ramped[0] == pytest.approx(0.08)
    assert ramped[7] == pytest.approx(0.11)


def test_relational_interrupted_extension_diverges_without_reproducing():
    """The documented limitation: a rewritten plan is not the original plan."""
    # 5 epochs x 10 steps, extended to a total of 10 epochs at completed 2/step 20
    # rewrites total_steps to 100 while original_max_epochs stays 5.
    saved = {
        "original_max_epochs": 5,
        "total_steps": 100,
        "steps_per_epoch": 10,
        "warmup_steps": 0,
        "last_lr": 0.0005,
        "final_lr": 0.00001,
    }

    def args(max_epochs):
        return argparse.Namespace(
            max_epochs=max_epochs,
            lr=0.001,
            warmup_epochs=0,
            teacher_momentum=0.995,
            teacher_momentum_end=0.999,
            student_temp=0.1,
            teacher_temp_start=0.04,
            teacher_temp_end=0.07,
            teacher_temp_warmup_ratio=0.7,
        )

    # An interrupted extension saved at completed epoch 3 / step 30 can return
    # to --max-epochs 5, but rebuilds the ramp over 100 steps.
    _, _, _, diverged, _ = pretrain_loop_module._build_pretrain_schedules(
        args(5), saved, steps_per_epoch=10, global_step=30, completed_epochs=3
    )
    assert float(diverged[40]) == pytest.approx(0.0573913045, abs=1e-9)
    # the original 50-step plan plateaued at the end temperature
    original = pretrain_recipe_module._build_teacher_temp_schedule(50, 0.04, 0.07)
    assert float(original[40]) == pytest.approx(0.07)
    # endpoints and recipe are unchanged; only the step-wise schedule differs
    assert float(diverged[-1]) == pytest.approx(0.07, abs=1e-6)

    # A completed 10-epoch extension rejects the return outright.
    with pytest.raises(ValueError, match="must exceed the completed checkpoint epoch"):
        pretrain_loop_module._build_pretrain_schedules(
            args(5), saved, steps_per_epoch=10, global_step=100, completed_epochs=10
        )


def test_relational_checkpoint_metadata_is_additive(tmp_path):
    """Downstream loaders tolerate the extra config bundle."""
    from otuformer.training.model import OTUFormerEncoder
    from otuformer.utils.checkpoint import load_checkpoint

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=16, pretrained=False, img_size=224
    )
    base = {
        "model_state_dict": encoder.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 16},
        "args": {"max_epochs": 1},
    }
    plain = tmp_path / "plain.pth"
    torch.save(base, plain)
    annotated = tmp_path / "annotated.pth"
    torch.save(
        {
            **base,
            "config": {
                **base["config"],
                pretrain_loop_module.RELATIONAL_TARGET_CONFIG_KEY: _bundle(),
            },
        },
        annotated,
    )
    left = load_checkpoint(plain)
    right = load_checkpoint(annotated)
    assert set(left["model_state_dict"]) == set(right["model_state_dict"])
    for key, value in left["model_state_dict"].items():
        assert torch.equal(value, right["model_state_dict"][key])
    # the additive bundle is inert metadata, not a weight or architecture change
    assert right["config"][pretrain_loop_module.RELATIONAL_TARGET_CONFIG_KEY] == _bundle()


def test_relational_recipe_rejects_present_null_metadata():
    """Only an absent key may fall back; a stored null is damaged provenance."""
    for extra in (
        {"teacher_temp_start": None},
        {"teacher_temp_end": None},
        {"disable_cross_view_loss": None},
        {"teacher_temp_warmup_ratio": None},
    ):
        with pytest.raises(ValueError):
            _recipe(checkpoint=_legacy_checkpoint(extra))

    # the same rule on the versioned path
    for extra in ({"teacher_temp_start": None}, {"teacher_temp_warmup_ratio": None}):
        with pytest.raises(ValueError):
            _recipe(checkpoint=_versioned_checkpoint(args_extra=extra))

    # an absent key still assumes the historical value with a warning
    absent = _legacy_checkpoint()
    for key in ("teacher_temp_start", "teacher_temp_end", "disable_cross_view_loss"):
        absent["args"].pop(key)
    resolved = _recipe(checkpoint=absent)
    assert resolved.relational_target_config["teacher_temp_start"] == 0.04
    assert resolved.relational_target_config["relational_pairing"] == "full"
    assert len(resolved.warnings) >= 3


def test_relational_recipe_rejects_null_weak_record():
    """A weak version-1 bundle must attest its transform record."""
    weak_bundle = _bundle(
        relational_teacher_views="weak",
        relational_center_source="weak",
        relational_pairing="cross",
        weak_teacher_augmentation_config=None,
    )
    with pytest.raises(ValueError, match="must be a mapping for weak"):
        _recipe(
            checkpoint=_versioned_checkpoint(
                weak_bundle,
                {
                    "relational_teacher_views": "weak",
                    "relational_center_source": "weak",
                    "relational_pairing": "cross",
                },
            )
        )


def test_relational_recipe_rejects_underflowing_temperature():
    """1e-50 casts to float32 0.0 and must not raise ZeroDivisionError."""
    checkpoint = _versioned_checkpoint(
        _bundle(teacher_temp_start=1e-50), {"teacher_temp_start": 1e-50}
    )
    with pytest.raises(ValueError, match="usable float32 temperature"):
        _recipe(checkpoint=checkpoint)


def test_relational_admission_boundary_is_adjacent():
    """The admission limit is inclusive and the neighbouring value flips it."""
    f32 = np.finfo(np.float32)
    limit = float(f32.max) * (1.0 - 4.0 * float(f32.eps))
    for batch_size in (6, 7):
        exact = float(max(batch_size, 2)) / limit
        # a temperature that sits just inside the bound is admitted
        assert (
            pretrain_recipe_module._admissible_temperature(
                np.nextafter(np.float32(exact), np.float32(np.inf)).item(),
                batch_size,
                "--teacher-temp-start",
            )
            is not None
        )
        # one float32 step smaller crosses the bound and is rejected
        with pytest.raises(ValueError, match="overflows"):
            pretrain_recipe_module._admissible_temperature(
                np.nextafter(np.float32(exact), np.float32(0.0)).item(),
                batch_size,
                "--teacher-temp-start",
            )


def test_finetune_does_not_copy_the_ssl_relational_bundle(tmp_path):
    """Finetune builds its own config; the SSL recipe must not leak in."""
    source = _write_legacy_finetune_checkpoint(
        tmp_path / "annotated_sft.pth", groups=2
    )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["config"][pretrain_loop_module.RELATIONAL_TARGET_CONFIG_KEY] = _bundle(
        relational_teacher_views="weak",
        relational_center_source="weak",
        relational_pairing="cross",
        weak_teacher_augmentation_config=build_pretrain_augmentation_config(
            "global-barcode", 32, 16, 0
        ),
    )
    torch.save(checkpoint, source)

    args = _finetune_args(
        tmp_path,
        source,
        finetune_epochs=1,
        **_finetune_overrides(tmp_path, "annotated_ft", batch_size=4),
    )
    finetune_loop_module.run_finetune(args)
    saved = torch.load(
        tmp_path / "annotated_ft" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert pretrain_loop_module.RELATIONAL_TARGET_CONFIG_KEY not in saved["config"]
    assert saved["config"]["loss"] == "arcface"
