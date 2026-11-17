"""Live Coordinate pretraining: end-to-end save, resume and all-off parity."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from otuformer.training import dense_recipe
from otuformer.training import pretrain_loop as pretrain_loop_module

SIZE = 32


def _dataset(tmp_path, rows=4):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for index in range(rows):
        Image.new("RGB", (64, 64), color=(index * 40, 10, 10)).save(
            images / f"img_{index}.jpg"
        )
    frame = pd.DataFrame({"image": [f"img_{i}.jpg" for i in range(rows)]})
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)
    return csv, images


def _args(tmp_path, out_name, **overrides):
    base = dict(
        seed=42,
        cpus=0,
        device="cpu",
        out_dir=str(tmp_path / out_name),
        model_name="vit_tiny_patch16_224",
        register_tokens=None,
        out_dim=16,
        ssl_method="relational",
        ssl_method_explicit=False,
        koleo_weight=0.0,
        koleo_weight_explicit=False,
        max_epochs=1,
        lr=1e-3,
        weight_decay=0.05,
        warmup_epochs=0,
        augmentation=None,
        orientation_policy=None,
        global_crop_size=SIZE,
        local_crop_size=16,
        local_crops=1,
        mask_ratio="auto",
        lambda_local=1.5,
        lambda_mask=1.0,
        patch_loss="none",
        masking_strategy="random",
        teacher_momentum=0.995,
        teacher_momentum_end=0.999,
        student_temp=0.1,
        teacher_temp_start=0.04,
        teacher_temp_end=0.07,
        teacher_temp_warmup_ratio=None,
        relational_teacher_views="weak",
        relational_center_source=None,
        relational_pairing="cross",
        relational_queue_size=0,
        resume="",
        log_every_n_steps=1000,
        save_every_epochs=1,
        keep_last_checkpoints=1,
        visualize_data="",
        extract_size=None,
        metrics_sample_size=10,
        umap_n_neighbors=5,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=5,
        compute_embedding_metrics=False,
        batch_size=2,
        num_workers=0,
        coordinate_weight=0.0,
        coordinate_weight_explicit=False,
        dense_query_points=49,
        dense_query_points_explicit=False,
        ranking_weight=0.0,
        ranking_weight_explicit=False,
        ranking_reference_points=16,
        ranking_reference_points_explicit=False,
    )
    csv, images = _dataset(tmp_path)
    base.update(
        train_data=str(csv),
        input_images_dir=str(images),
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_coordinate_run_saves_bundles_and_reports(tmp_path):
    args = _args(
        tmp_path,
        "dense",
        coordinate_weight=0.5,
        coordinate_weight_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)

    saved = torch.load(
        Path(args.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    config = saved["config"]["dense_training_config"]
    assert config["objectives"] == ["coordinate"]
    assert config["coordinate_weight"] == pytest.approx(0.5)
    assert config["ranking_weight"] == 0.0
    assert "ranking_config" not in config
    state = saved["dense_training"]
    assert state["completed_steps"] == saved["iteration"]
    assert "ranking" not in state
    assert torch.is_tensor(state["query_rng"])


def test_all_off_run_omits_both_bundles(tmp_path):
    args = _args(tmp_path, "off")
    pretrain_loop_module.run_pretrain(args)
    saved = torch.load(
        Path(args.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    assert "dense_training" not in saved
    assert "dense_training_config" not in saved["config"]


def test_all_off_batch_is_still_a_plain_list(tmp_path):
    """The disabled path must not switch the loader to the geometry envelope."""
    csv, images = _dataset(tmp_path)
    from otuformer.training.dataset import MultiCropDataset

    dataset = MultiCropDataset(
        csv, images, global_crop_size=SIZE, local_crop_size=16, local_crops=1
    )
    row = dataset[0]
    assert isinstance(row, list)
    assert len(row) == 3


def test_coordinate_resume_is_strict(tmp_path):
    args = _args(
        tmp_path,
        "resume",
        coordinate_weight=0.4,
        coordinate_weight_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"

    # Same recipe resumes.
    resumed = _args(
        tmp_path,
        "resume",
        max_epochs=2,
        resume=str(latest),
        coordinate_weight=0.4,
        coordinate_weight_explicit=True,
    )
    pretrain_loop_module.run_pretrain(resumed)

    # A changed weight is a new run, not a resume.
    conflicting = _args(
        tmp_path,
        "resume",
        max_epochs=2,
        resume=str(latest),
        coordinate_weight=0.9,
        coordinate_weight_explicit=True,
    )
    with pytest.raises(ValueError, match="coordinate_weight"):
        pretrain_loop_module.run_pretrain(conflicting)

    # Enabling a previously disabled objective is rejected too.
    off = _args(tmp_path, "offrun")
    pretrain_loop_module.run_pretrain(off)
    enabling = _args(
        tmp_path,
        "offrun",
        max_epochs=2,
        resume=str(Path(off.out_dir) / "SSL_latest.pth"),
        coordinate_weight=0.5,
        coordinate_weight_explicit=True,
    )
    with pytest.raises(ValueError, match="dense"):
        pretrain_loop_module.run_pretrain(enabling)


def test_coordinate_weight_zero_is_the_historical_recipe(tmp_path):
    settings = dense_recipe.resolve_dense_settings(
        argparse.Namespace(
            coordinate_weight=0.0,
            coordinate_weight_explicit=True,
            ranking_weight=0.0,
            ranking_weight_explicit=False,
        ),
        None,
    )
    assert settings.enabled is False


def test_enabled_run_writes_dense_columns(tmp_path):
    """An enabled run logs its own columns; the disabled run keeps the old header."""
    import csv as _csv

    on = _args(tmp_path, "logged", coordinate_weight=0.5,
               coordinate_weight_explicit=True, dense_query_points=4,
               dense_query_points_explicit=True,
               # A larger global view keeps full patch support at 16px patches.
               global_crop_size=64, local_crop_size=32)
    pretrain_loop_module.run_pretrain(on)
    header = (Path(on.out_dir) / "logs" / "instant_metrics.pretrain.csv").read_text().splitlines()[0]
    assert "dense_coordinate_loss_raw" in header
    assert "dense_min_rho" in header
    rows = list(_csv.DictReader((Path(on.out_dir) / "logs" / "instant_metrics.pretrain.csv").open()))
    assert rows and rows[0]["dense_coordinate_loss_raw"] != ""
    assert rows[0]["dense_coordinate_status"] in ("observed", "skipped")

    off = _args(tmp_path, "unlogged")
    pretrain_loop_module.run_pretrain(off)
    off_header = (Path(off.out_dir) / "logs" / "instant_metrics.pretrain.csv").read_text().splitlines()[0]
    assert "dense_coordinate_loss_raw" not in off_header
