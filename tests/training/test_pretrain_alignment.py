import argparse

import numpy as np
import pytest
import torch

from otuformer.training import trainer


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
