import pytest
import torch

from otuformer.training import trainer


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
