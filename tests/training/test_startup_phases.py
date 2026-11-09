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
