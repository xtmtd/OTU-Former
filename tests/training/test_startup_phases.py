"""Startup phase records must open before the slow work they name.

A phase start that appears after dataset construction attributes the real
path/CSV resolution cost to nothing, which is exactly the silence the operator
reported. These checks assert event order, not elapsed time.
"""

from __future__ import annotations

import argparse
import csv
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from PIL import Image

from otuformer.training import pretrain_loop as trainer
from otuformer.training import reporting


def _images(root: Path, rows: int = 4) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index in range(rows):
        Image.new("RGB", (32, 32), (index * 40, 10, 90)).save(images / f"{index}.png")
    manifest = root / "data.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        for index in range(rows):
            writer.writerow([f"{index}.png"])
    return images, manifest


def _pretrain_args(root: Path) -> argparse.Namespace:
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
        koleo_weight=0.0,
        koleo_weight_explicit=True,
        max_epochs=1,
        lr=5e-4,
        weight_decay=0.05,
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
        patch_loss="consistency",
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
        batch_size=2,
        num_workers=0,
        cpus=1,
        device="auto",
        seed=42,
        resume="",
    )


def test_pretrain_dataset_and_objective_phases_open_before_their_work(
    tmp_path, monkeypatch
):
    events: list[str] = []
    # Record what the log already contained when each slow constructor started:
    # that is the only way to see whether the phase was open beforehand.
    open_before: dict[str, bool] = {}
    captured = io.StringIO()
    real_dataset = trainer.MultiCropDataset
    real_objective = trainer.PatchObjective

    def already_logged(label: str) -> bool:
        return f"[Phase] {label}: start" in captured.getvalue()

    def dataset(*args, **kwargs):
        events.append("dataset")
        open_before["dataset"] = already_logged("dataset indexing and loader")
        return real_dataset(*args, **kwargs)

    class Objective(real_objective):
        def __init__(self, *args, **kwargs):
            events.append("objective")
            open_before["objective"] = already_logged(
                "objective, optimizer and recovery"
            )
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(trainer, "MultiCropDataset", dataset)
    monkeypatch.setattr(trainer, "PatchObjective", Objective)

    with redirect_stdout(captured):
        trainer.run_pretrain(_pretrain_args(tmp_path))
    output = captured.getvalue()
    from otuformer.training.runtime import _resolve_device

    assert f"Resolved device: {_resolve_device('auto')}" in output

    dataset_start = output.index("[Phase] dataset indexing and loader: start")
    dataset_done = output.index("[Phase] dataset indexing and loader: done")
    preparation_start = output.index(
        "[Phase] objective, optimizer and recovery: start"
    )
    first_batch_start = output.index("[Phase] first batch and initial forwards: start")

    # The dataset constructor is the real indexing work: the phase must already
    # be open when it runs.
    assert events.index("dataset") == 0
    assert dataset_start < dataset_done
    # The phase must already be open when the constructor runs, otherwise the
    # real path/CSV resolution is attributed to nothing.
    assert open_before["dataset"] is True, (
        "dataset indexing phase opened after the constructor ran"
    )
    # The objective belongs to the preparation block, which opens first.
    assert events.index("objective") == 1
    assert open_before["objective"] is True
    assert preparation_start < first_batch_start
    # Every start has a matching completion.
    for label in (
        "dataset indexing and loader",
        "student construction",
        "teacher construction",
        "objective, optimizer and recovery",
        "first batch and initial forwards",
    ):
        assert output.count(f"[Phase] {label}: start") == 1, label
        assert output.count(f"[Phase] {label}: done in ") == 1, label


def test_pretrain_phase_start_survives_a_failure_inside_the_phase(tmp_path):
    output = io.StringIO()
    args = _pretrain_args(tmp_path)
    args.input_images_dir = "/nonexistent-images-dir"
    args.train_data = ""
    with pytest.raises(Exception):
        with redirect_stdout(output):
            trainer.run_pretrain(args)
    text = output.getvalue()
    assert "[Phase] dataset indexing and loader: start" in text
    assert "[Phase] dataset indexing and loader: done" not in text


def _bad_eval_csv(root: Path) -> Path:
    csv_path = root / "bad_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        writer.writerow(["missing.png"])
    return csv_path


def test_evaluation_manifest_phase_warns_once_and_training_continues(
    tmp_path, capsys
):
    args = _pretrain_args(tmp_path)
    args.compute_embedding_metrics = True
    args.visualize_data = str(_bad_eval_csv(tmp_path))

    trainer.run_pretrain(args)

    output = capsys.readouterr().out
    assert output.count("[Phase] evaluation manifest: start") == 1
    assert output.count("[Phase] evaluation manifest: done in ") == 1
    assert output.index("[Phase] evaluation manifest: start") < output.index(
        "[Phase] evaluation manifest: done"
    )
    assert output.count("[Warning] Periodic embedding evaluation is disabled") == 1
    # The epoch logger and its curves input survive the disabled evaluator.
    assert (tmp_path / "out" / "logs" / "metrics.pretrain.csv").exists()
    assert (tmp_path / "out" / "SSL_latest.pth").exists()


def test_evaluation_manifest_is_prepared_once_and_handed_to_the_evaluator(
    tmp_path, monkeypatch
):
    args = _pretrain_args(tmp_path)
    args.compute_embedding_metrics = True
    args.visualize_data = str(tmp_path / "data.csv")
    captured = []
    monkeypatch.setattr(
        trainer,
        "_compute_and_log_all_metrics",
        lambda **kwargs: captured.append(kwargs),
    )

    trainer.run_pretrain(args)

    assert len(captured) == 1
    manifest = captured[0]["manifest"]
    assert isinstance(manifest, reporting.PreparedEvaluationManifest)
    assert len(manifest.paths) == 4
    assert all(path.is_file() for path in manifest.paths)


def test_disabled_metrics_skip_manifest_preparation(tmp_path, monkeypatch):
    args = _pretrain_args(tmp_path)  # compute_embedding_metrics=False
    args.visualize_data = str(_bad_eval_csv(tmp_path))

    def boom(*args, **kwargs):
        raise AssertionError("manifest prepared while embedding metrics are disabled")

    monkeypatch.setattr(trainer, "prepare_evaluation_manifest", boom)
    trainer.run_pretrain(args)

    assert not (tmp_path / "out" / "logs" / "metrics.pretrain.csv").exists()


def test_evaluation_manifest_phase_keeps_the_recovery_timer(tmp_path, monkeypatch):
    """The nested phase must not overwrite the enclosing recovery start."""
    starts = {}
    done = []
    real_start = trainer._phase_start
    real_done = trainer._phase_done

    def start(label):
        token = real_start(label)
        starts[label] = token
        return token

    def finish(label, token):
        done.append((label, token))
        return real_done(label, token)

    monkeypatch.setattr(trainer, "_phase_start", start)
    monkeypatch.setattr(trainer, "_phase_done", finish)
    args = _pretrain_args(tmp_path)
    args.compute_embedding_metrics = True
    args.visualize_data = str(tmp_path / "data.csv")

    trainer.run_pretrain(args)

    recovery = [
        token for label, token in done if label == "objective, optimizer and recovery"
    ]
    assert recovery == [starts["objective, optimizer and recovery"]]
    assert any(label == "evaluation manifest" for label, _ in done)


def test_evaluation_manifest_reuses_the_training_selection_when_it_matches(
    tmp_path, monkeypatch
):
    from otuformer.training.dataset import resolve_training_images

    args = _pretrain_args(tmp_path)
    args.compute_embedding_metrics = True
    args.visualize_data = str(tmp_path / "data.csv")
    args.pretrain_resolved_images = resolve_training_images(
        tmp_path / "data.csv", tmp_path / "images"
    )
    paths = args.pretrain_resolved_images.paths
    captured = {}

    def fake_prepare(csv_path, root_dir, **kwargs):
        captured.update(kwargs)
        return reporting.PreparedEvaluationManifest(
            csv_path=csv_path, paths=paths, labels=None
        )

    monkeypatch.setattr(trainer, "prepare_evaluation_manifest", fake_prepare)
    monkeypatch.setattr(trainer, "_compute_and_log_all_metrics", lambda **k: None)

    trainer.run_pretrain(args)

    assert captured["image_paths"] == paths
    assert captured["refs"] == ("0.png", "1.png", "2.png", "3.png")


def test_dataset_reuses_a_matching_prepared_selection(tmp_path, monkeypatch):
    from dataclasses import replace

    from otuformer.training import dataset as dataset_module

    images, manifest = _images(tmp_path)
    real = dataset_module.resolve_dataset_image_paths
    calls = []
    monkeypatch.setattr(
        dataset_module,
        "resolve_dataset_image_paths",
        lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs),
    )
    paths = tuple(real(manifest, images))
    carrier = dataset_module.ResolvedTrainingImages(
        csv_path=str(manifest), images_dir=str(images), paths=paths
    )

    def build(prepared_images):
        return dataset_module.MultiCropDataset(
            csv_path=manifest,
            images_dir=images,
            global_crop_size=32,
            local_crop_size=16,
            local_crops=0,
            augmentation_profile="global-barcode",
            orientation_policy="sensitive",
            prepared_images=prepared_images,
        )

    dataset = build(carrier)
    assert calls == []
    assert dataset.image_paths == list(paths)

    build(replace(carrier, csv_path=str(tmp_path / "other.csv")))
    assert calls == [1]


def test_ordinary_run_resolves_the_training_selection_once(tmp_path, monkeypatch):
    """The dataset and the evaluation manifest share one resolution."""
    from otuformer.training import dataset as dataset_module

    args = _pretrain_args(tmp_path)
    args.compute_embedding_metrics = True
    args.visualize_data = str(tmp_path / "data.csv")
    calls = []
    real_dataset = dataset_module.resolve_dataset_image_paths
    real_reporting = reporting.resolve_dataset_image_paths
    monkeypatch.setattr(
        dataset_module,
        "resolve_dataset_image_paths",
        lambda *a, **k: calls.append(1) or real_dataset(*a, **k),
    )
    monkeypatch.setattr(
        reporting,
        "resolve_dataset_image_paths",
        lambda *a, **k: calls.append(1) or real_reporting(*a, **k),
    )
    captured = {}
    real_prepare = reporting.prepare_evaluation_manifest

    def spy(csv_path, root_dir, **kwargs):
        captured.update(kwargs)
        return real_prepare(csv_path, root_dir, **kwargs)

    monkeypatch.setattr(trainer, "prepare_evaluation_manifest", spy)
    monkeypatch.setattr(trainer, "_compute_and_log_all_metrics", lambda **k: None)

    trainer.run_pretrain(args)

    assert len(calls) == 1
    assert captured["image_paths"] is not None
    assert len(captured["image_paths"]) == 4
