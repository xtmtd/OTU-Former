import argparse
import pytest
import pandas as pd
from PIL import Image
import torch
import os
import logging
from pathlib import Path
from typer.testing import CliRunner

from otuformer.cli.main import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "otu-former" in result.output.lower() or "otuformer" in result.output.lower()


@pytest.mark.parametrize(
    "command,options",
    [
        ("pretrain", ["--resume", "--overwrite"]),
        ("finetune", ["--resume", "--overwrite"]),
        ("extract", ["--overwrite"]),
        ("cluster", ["--overwrite"]),
        ("annotate", ["--overwrite"]),
        ("diversity", ["--overwrite"]),
        ("cam", ["--overwrite"]),
        ("export", ["--overwrite"]),
    ],
)
def test_help_lists_safety_options_after_out_dir(command, options):
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0
    positions = [result.output.rfind(option) for option in options]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    assert positions[-1] > result.output.rfind("--out-dir")


def test_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.output
    assert "torch" in result.output.lower()


@pytest.mark.parametrize(
    "cmd",
    [
        ["pretrain", "--help"],
        ["finetune", "--help"],
        ["extract", "--help"],
        ["cluster", "--help"],
        ["annotate", "--help"],
        ["diversity", "--help"],
        ["cam", "--help"],
        ["export", "--help"],
    ],
)
def test_subcommand_help(cmd):
    result = runner.invoke(app, cmd)
    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("cmd", "expected_flags"),
    [
        ("pretrain", ["--train-data", "--model-name", "--mask-ratio"]),
        ("finetune", ["--checkpoint", "--freeze-ratio", "--loss"]),
        (
            "extract",
            [
                "--checkpoint",
                "--token-mode",
                "--topk-patches",
                "--umap-metric",
                "--disable-umap",
            ],
        ),
        ("cluster", ["--embeddings", "--distance", "--custom-cutoffs"]),
        (
            "annotate",
            [
                "--raw-assignments",
                "--corrections",
                "--embeddings",
                "--support-display-cutoff",
                "--annotate-bar-width",
                "--out-dir",
            ],
        ),
        ("diversity", ["--assignments", "--min-abundance", "--phylo"]),
        ("cam", ["--checkpoint", "--images-dir", "--cam-method"]),
        ("export", ["--checkpoint", "--imgsz", "--opset"]),
    ],
)
def test_subcommand_help_includes_key_flags(cmd, expected_flags):
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
    for flag in expected_flags:
        assert flag in result.output


def test_cluster_help_includes_save_bootstrap_trees_and_detailed_text():
    result = runner.invoke(app, ["cluster", "--help"])
    assert result.exit_code == 0
    assert "save-bootstrap" in result.output
    assert "--num-replicates" in result.output
    assert "--support-mode" in result.output
    assert "subsample" in result.output
    assert "bootstrap" in result.output
    assert "local-k-strategy" in result.output
    assert "adaptive" in result.output
    output = result.output.lower()
    assert "bootstrap" in output
    assert "partition" in output


def test_extract_help_lists_attention_pooling_type_choices():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "--attention-pooling-type" in result.output
    assert "lightweight" in result.output
    assert "multihead" in result.output
    assert "gated" in result.output


def _make_tiny_pretrain_data(tmp_path):
    for i in range(4):
        Image.new("RGB", (224, 224), color=(i * 60, 20, 10)).save(
            tmp_path / f"img_{i}.jpg"
        )
    df = pd.DataFrame({"image": [f"img_{i}.jpg" for i in range(4)]})
    csv_path = tmp_path / "images.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_tiny_finetune_data(tmp_path):
    for i in range(4):
        Image.new("RGB", (224, 224), color=(i * 60, 20, 10)).save(
            tmp_path / f"img_{i}.jpg"
        )
    df = pd.DataFrame(
        {
            "image": [f"img_{i}.jpg" for i in range(4)],
            "label": ["classA", "classA", "classB", "classB"],
        }
    )
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_pretrain_runs_one_epoch(tmp_path):
    csv_path = _make_tiny_pretrain_data(tmp_path)
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(csv_path),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
            "--model-name",
            "vit_tiny_patch16_224",
            "--out-dim",
            "64",
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--global-crop-size",
            "224",
            "--local-crop-size",
            "96",
            "--local-crops",
            "2",
            "--disable-embedding-metrics",
            "--disable-cross-view-loss",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert not (tmp_path / "pretrain_out" / "last.pt").exists()
    assert (tmp_path / "pretrain_out" / "SSL_epoch_0001.pth").exists()
    assert (tmp_path / "pretrain_out" / "SSL_latest.pth").exists()
    assert (
        tmp_path / "pretrain_out" / "logs" / "instant_metrics.pretrain.csv"
    ).exists()
    assert not (tmp_path / "pretrain_out" / "logs" / "metrics_instant.jsonl").exists()
    assert not (tmp_path / "pretrain_out" / "best.pt").exists()
    saved = torch.load(
        tmp_path / "pretrain_out" / "SSL_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["config"]["augmentation_profile"] == "global-barcode"
    assert saved["config"]["augmentation_config"]["profile"] == "global-barcode"
    assert saved["config"]["augmentation_config"]["orientation_policy"] == "sensitive"
    assert "orientation_policy" not in saved["config"]


def test_pretrain_runs_one_epoch_without_train_data(tmp_path):
    _make_tiny_pretrain_data(tmp_path)
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out_no_csv"),
            "--model-name",
            "vit_tiny_patch16_224",
            "--out-dim",
            "64",
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--global-crop-size",
            "224",
            "--local-crop-size",
            "96",
            "--local-crops",
            "2",
            "--disable-embedding-metrics",
            "--disable-cross-view-loss",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert not (tmp_path / "pretrain_out_no_csv" / "last.pt").exists()
    assert (tmp_path / "pretrain_out_no_csv" / "SSL_latest.pth").exists()


@pytest.mark.parametrize("command", ["pretrain", "finetune"])
def test_training_rejects_resume_with_overwrite(tmp_path, command):
    args = [
        command,
        "--resume",
        str(tmp_path / "missing.pth"),
        "--overwrite",
        "--out-dir",
        str(tmp_path / "out"),
        "--input-images-dir",
        str(tmp_path),
        "--train-data",
        str(tmp_path / "data.csv"),
    ]
    result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert "cannot be used together" in result.output


def test_pretrain_resume_allows_existing_output_and_appends_log(tmp_path, monkeypatch):
    checkpoint = tmp_path / "resume.pth"
    torch.save({}, checkpoint)
    out_dir = tmp_path / "pretrain_out"
    log_path = out_dir / "logs" / "pretrain.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("old run\n", encoding="utf-8")
    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", lambda _args: None)

    result = runner.invoke(
        app,
        [
            "pretrain", "--resume", str(checkpoint), "--train-data", str(tmp_path / "data.csv"),
            "--input-images-dir", str(tmp_path), "--out-dir", str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert log_path.read_text(encoding="utf-8").startswith("old run\n")


def test_finetune_checkpoint_initialization_requires_empty_or_overwrite(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pretrained.pth"
    torch.save({}, checkpoint)
    out_dir = tmp_path / "finetune_out"
    out_dir.mkdir()
    stale = out_dir / "stale.txt"
    stale.write_text("stale", encoding="utf-8")
    monkeypatch.setattr("otuformer.training.trainer.run_finetune", lambda _args: None)
    args = [
        "finetune", "--checkpoint", str(checkpoint), "--train-data", str(tmp_path / "labels.csv"),
        "--input-images-dir", str(tmp_path), "--out-dir", str(out_dir),
    ]

    rejected = runner.invoke(app, args)
    assert rejected.exit_code != 0
    assert stale.exists()

    overwritten = runner.invoke(app, [*args, "--overwrite"])
    assert overwritten.exit_code == 0
    assert not stale.exists()


def test_readmes_document_update_and_continued_training():
    for path in ["README.md", "README.cn.md"]:
        text = Path(path).read_text(encoding="utf-8")
        assert "otuformer update" in text
        assert "--overwrite" in text
        assert "--resume" in text
        assert "HF_TOKEN" in text


def test_finetune_runs_one_epoch(tmp_path):
    pretrain_csv = _make_tiny_pretrain_data(tmp_path)
    pretrain_result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(pretrain_csv),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
            "--model-name",
            "vit_tiny_patch16_224",
            "--out-dim",
            "64",
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--global-crop-size",
            "224",
            "--local-crop-size",
            "96",
            "--local-crops",
            "2",
            "--disable-embedding-metrics",
            "--disable-cross-view-loss",
            "--device",
            "cpu",
        ],
    )
    assert pretrain_result.exit_code == 0

    labels_csv = _make_tiny_finetune_data(tmp_path)
    ckpt = tmp_path / "pretrain_out" / "SSL_latest.pth"
    result = runner.invoke(
        app,
        [
            "finetune",
            "--checkpoint",
            str(ckpt),
            "--train-data",
            str(labels_csv),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "finetune_out"),
            "--model-name",
            "vit_tiny_patch16_224",
            "--metric-embed-dim",
            "64",
            "--finetune-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "finetune_out" / "finetune_latest.pth").exists()
    saved = torch.load(
        tmp_path / "finetune_out" / "finetune_latest.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["config"]["augmentation_profile"] == "none"
    assert saved["config"]["augmentation_config"]["profile"] == "none"
    assert saved["config"]["augmentation_config"]["orientation_policy"] == "sensitive"
    assert "orientation_policy" not in saved["config"]


def _make_ckpt(tmp_path, out_dim=64):
    from otuformer.training.model import OTUFormerEncoder

    m = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=out_dim)
    ckpt = {
        "model_state_dict": m.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim},
    }
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, p)
    return p


def test_extract_command(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224)).save(img_dir / "test.jpg")
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(ckpt),
            "--input-images-dir",
            str(img_dir),
            "--out-dir",
            str(tmp_path / "extract_out"),
            "--model-name",
            "vit_tiny_patch16_224",
            "--extract-size",
            "224",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "extract_out" / "embeddings.csv").exists()


def test_extract_image_only_csv_generates_umap_without_metrics(tmp_path, monkeypatch):
    ckpt = _make_ckpt(tmp_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        Image.new("RGB", (224, 224), color=(i * 10, 0, 0)).save(
            img_dir / f"img_{i}.jpg"
        )
    csv_path = tmp_path / "images.csv"
    pd.DataFrame({"image": [f"img_{i}.jpg" for i in range(10)]}).to_csv(
        csv_path, index=False
    )

    def fake_umap(embeddings, labels, out_path, **kwargs):
        assert len(embeddings) == 10
        assert labels is None
        out_path.write_text("umap")

    monkeypatch.setattr("otuformer.embedding.evaluator.run_umap", fake_umap)
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(ckpt),
            "--input-images-dir",
            str(img_dir),
            "--label-csv",
            str(csv_path),
            "--out-dir",
            str(tmp_path / "extract_image_only"),
            "--extract-size",
            "224",
            "--batch-size",
            "10",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "extract_image_only" / "umap.pdf").exists()
    assert not (tmp_path / "extract_image_only" / "metrics.csv").exists()


def test_extract_small_visualization_sample_reports_skipped_umap(
    tmp_path, monkeypatch
):
    ckpt = _make_ckpt(tmp_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        Image.new("RGB", (224, 224), color=(i * 20, 20, 10)).save(
            img_dir / f"img_{i}.jpg"
        )
    csv_path = tmp_path / "images.csv"
    pd.DataFrame({"image": [f"img_{i}.jpg" for i in range(10)]}).to_csv(
        csv_path, index=False
    )

    monkeypatch.setattr(
        "otuformer.embedding.evaluator.run_umap",
        lambda *args, **kwargs: pytest.fail("UMAP should be skipped"),
    )
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(ckpt),
            "--input-images-dir",
            str(img_dir),
            "--label-csv",
            str(csv_path),
            "--out-dir",
            str(tmp_path / "extract_small_sample"),
            "--metrics-sample-size",
            "5",
            "--extract-size",
            "224",
            "--batch-size",
            "10",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Skipping UMAP: fewer than 10 samples after sampling" in result.output
    assert not (tmp_path / "extract_small_sample" / "umap.pdf").exists()


def test_extract_single_class_csv_skips_metrics_but_generates_umap(
    tmp_path, monkeypatch
):
    ckpt = _make_ckpt(tmp_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        Image.new("RGB", (224, 224), color=(i * 20, 20, 10)).save(
            img_dir / f"img_{i}.jpg"
        )
    csv_path = tmp_path / "single_class.csv"
    pd.DataFrame(
        {
            "image": [f"img_{i}.jpg" for i in range(10)],
            "label": ["one"] * 10,
        }
    ).to_csv(csv_path, index=False)

    def fake_umap(embeddings, labels, out_path, **kwargs):
        assert len(embeddings) == 10
        assert set(labels) == {"one"}
        out_path.write_text("umap")

    monkeypatch.setattr("otuformer.embedding.evaluator.run_umap", fake_umap)
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(ckpt),
            "--input-images-dir",
            str(img_dir),
            "--label-csv",
            str(csv_path),
            "--out-dir",
            str(tmp_path / "extract_single_class"),
            "--extract-size",
            "224",
            "--batch-size",
            "10",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "fewer than two classes" in result.output
    assert (tmp_path / "extract_single_class" / "umap.pdf").exists()
    assert not (tmp_path / "extract_single_class" / "metrics.csv").exists()


def test_export_command(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--out-dir",
            str(tmp_path / "export_out"),
            "--imgsz",
            "224",
            "--opset",
            "17",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "export_out" / "encoder.onnx").exists()
    assert (tmp_path / "export_out" / "logs" / "export.log").exists()


def _make_embeddings_and_labels(tmp_path):
    emb = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "dim_0": [1.0, 0.9, -1.0, -0.9],
            "dim_1": [0.1, 0.0, -0.1, 0.0],
            "dim_2": [0.2, 0.2, -0.2, -0.2],
        }
    )
    labels = pd.DataFrame({"id": ["a", "b", "c", "d"], "label": ["x", "x", "y", "y"]})
    emb_path = tmp_path / "embeddings.csv"
    labels_path = tmp_path / "labels.csv"
    emb.to_csv(emb_path, index=False)
    labels.to_csv(labels_path, index=False)
    return emb_path, labels_path


def _make_embeddings_with_sample_and_labels(tmp_path):
    emb = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "sample": ["s1", "s1", "s2", "s2"],
            "dim_0": [1.0, 0.9, -1.0, -0.9],
            "dim_1": [0.1, 0.0, -0.1, 0.0],
            "dim_2": [0.2, 0.2, -0.2, -0.2],
        }
    )
    labels = pd.DataFrame({"id": ["a", "b", "c", "d"], "label": ["x", "x", "y", "y"]})
    emb_path = tmp_path / "embeddings_with_sample.csv"
    labels_path = tmp_path / "labels_with_sample.csv"
    emb.to_csv(emb_path, index=False)
    labels.to_csv(labels_path, index=False)
    return emb_path, labels_path


def test_cluster_command(tmp_path):
    emb_path, labels_path = _make_embeddings_and_labels(tmp_path)
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(tmp_path / "cluster_out"),
            "--labels",
            str(labels_path),
            "--custom-cutoffs",
            "0.2,0.5",
            "--distance",
            "cosine",
        ],
    )
    assert result.exit_code == 0
    assert any(
        (tmp_path / "cluster_out" / "UPGMA" / "partitions" / "tables").glob(
            "partition_*_assignments.csv"
        )
    )


def test_cluster_accepts_label_csv_with_image_column(tmp_path):
    emb_path, _ = _make_embeddings_and_labels(tmp_path)
    label_csv = tmp_path / "labels_image.csv"
    pd.DataFrame(
        {
            "image": ["a", "b", "c", "d"],
            "label": ["x", "x", "y", "y"],
        }
    ).to_csv(label_csv, index=False)

    out_dir = tmp_path / "cluster_out_label_csv"
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--label-csv",
            str(label_csv),
            "--custom-cutoffs",
            "0.5",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "UPGMA" / "metrics.csv").exists()
    assert (out_dir / "logs" / "cluster.log").exists()


def test_cluster_bool_options_accept_true_false(tmp_path):
    emb_path, labels_path = _make_embeddings_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out_bools"
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--labels",
            str(labels_path),
            "--custom-cutoffs",
            "0.5",
            "--distance",
            "cosine",
            "--pca-whitening",
            "true",
            "--local-scaling",
            "false",
            "--save-distances",
            "true",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "distance_statistics" / "distance_matrix.csv").exists()


def test_cluster_bool_options_reject_invalid_values(tmp_path):
    emb_path, _ = _make_embeddings_and_labels(tmp_path)
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(tmp_path / "cluster_out_invalid_bool"),
            "--custom-cutoffs",
            "0.5",
            "--pca-whitening",
            "maybe",
        ],
    )
    assert result.exit_code != 0
    output = result.output.lower()
    assert "use true or" in output
    assert "false" in output


def test_cluster_respects_max_distance_pairs(tmp_path):
    emb_path, _ = _make_embeddings_and_labels(tmp_path)
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(tmp_path / "cluster_out_limited"),
            "--distance",
            "cosine",
            "--max-distance-pairs",
            "2",
        ],
    )
    assert result.exit_code != 0
    assert "max-distance-pairs" in result.output.lower()


def test_cluster_without_label_csv_skips_partition_metrics(tmp_path):
    emb_path, _ = _make_embeddings_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out_no_labels"
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--custom-cutoffs",
            "0.5",
        ],
    )
    assert result.exit_code == 0
    assert not (out_dir / "partition_metrics.csv").exists()


def test_cluster_writes_ref_like_structure_and_csv_stats(tmp_path):
    emb_path, labels_path = _make_embeddings_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out_ref_like"
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--labels",
            str(labels_path),
            "--distance",
            "cosine",
            "--cutoff-min",
            "0.2",
            "--cutoff-max",
            "0.4",
            "--cutoff-step",
            "0.02",
            "--num-replicates",
            "5",
            "--support-mode",
            "subsample",
            "--subsample-ratio",
            "0.8",
        ],
    )
    assert result.exit_code == 0

    root_dirs = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert root_dirs == ["UPGMA", "distance_statistics", "logs"]
    assert not (out_dir / "logs" / "log.txt").exists()
    assert (out_dir / "logs" / "cluster.log").exists()

    assert (out_dir / "distance_statistics" / "distance_stats.csv").exists()
    assert (out_dir / "distance_statistics" / "distance_hist_raw_cosine.pdf").exists()
    assert (out_dir / "distance_statistics" / "distance_cum_raw_cosine.pdf").exists()
    assert (
        out_dir / "distance_statistics" / "distance_hist_raw_cosine_log.pdf"
    ).exists()
    assert not (out_dir / "distance_statistics" / "distance_stats.json").exists()

    assert (out_dir / "UPGMA" / "UPGMA_Cosine.nwk").exists()
    assert (out_dir / "UPGMA" / "UPGMA_Cosine_bootstrap.nwk").exists()
    assert (out_dir / "UPGMA" / "metrics_dashboard.pdf").exists()
    assert (out_dir / "UPGMA" / "partitions" / "partition_scan.csv").exists()
    assert (out_dir / "UPGMA" / "partitions" / "partition_scan.pdf").exists()
    assert (out_dir / "UPGMA" / "partitions" / "UPGMA_tree_partitions.pdf").exists()
    assert any(
        (out_dir / "UPGMA" / "partitions" / "tables").glob(
            "partition_*_assignments.csv"
        )
    )
    assert (
        out_dir / "UPGMA" / "partitions" / "tables" / "partition_0.4_assignments.csv"
    ).exists()
    assert (
        out_dir / "UPGMA" / "partitions" / "tables" / "partition_0.4_summary.csv"
    ).exists()
    assert (out_dir / "UPGMA" / "metrics.csv").exists()
    assert not (out_dir / "UPGMA" / "partition_metrics.json").exists()

    bootstrap_newick = (out_dir / "UPGMA" / "UPGMA_Cosine_bootstrap.nwk").read_text(
        encoding="utf-8"
    )
    assert ")" in bootstrap_newick
    assert any(ch.isdigit() for ch in bootstrap_newick.split(")")[-2])

    cluster_log = (out_dir / "logs" / "cluster.log").read_text(encoding="utf-8")
    assert "[2/7] SKIP PCA whitening" in cluster_log
    assert "[4/7] SKIP local scaling" in cluster_log

    metrics_df = pd.read_csv(out_dir / "UPGMA" / "metrics.csv")
    expected_cutoffs = [
        0.2,
        0.22,
        0.24,
        0.26,
        0.28,
        0.3,
        0.32,
        0.34,
        0.36,
        0.38,
        0.4,
    ]
    assert [round(v, 2) for v in metrics_df["cutoff"].tolist()] == expected_cutoffs
    for col in [
        "BCubed_precision",
        "BCubed_recall",
        "BCubed_fscore",
        "monophyly_proportion",
        "v_measure",
    ]:
        assert col in metrics_df.columns


def test_cluster_old_bootstrap_flag_is_rejected(tmp_path):
    emb_path, labels_path = _make_embeddings_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out_reject_old_bootstrap"
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--labels",
            str(labels_path),
            "--distance",
            "cosine",
            "--cutoff-min",
            "0.2",
            "--cutoff-max",
            "0.4",
            "--cutoff-step",
            "0.02",
            "--num-bootstraps",
            "5",
        ],
    )
    assert result.exit_code != 0


def test_cluster_rejects_old_outputs_without_overwrite(tmp_path):
    emb_path, labels_path = _make_embeddings_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out_cleanup"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "old_should_disappear.txt"
    stale.write_text("stale", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--label-csv",
            str(labels_path),
            "--custom-cutoffs",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    assert "--overwrite" in str(result.exception)
    assert stale.exists()


def test_cluster_overwrite_cleans_old_outputs_before_writing(tmp_path):
    emb_path, labels_path = _make_embeddings_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out_cleanup"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "old_should_disappear.txt"
    stale.write_text("stale", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--label-csv",
            str(labels_path),
            "--custom-cutoffs",
            "0.5",
            "--overwrite",
        ],
    )
    assert result.exit_code == 0
    assert not stale.exists()


def test_cluster_save_bootstrap_trees_option(tmp_path):
    emb_path, labels_path = _make_embeddings_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out_bootstrap_trees"

    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--label-csv",
            str(labels_path),
            "--custom-cutoffs",
            "0.5",
            "--num-replicates",
            "3",
            "--support-mode",
            "subsample",
            "--subsample-ratio",
            "0.8",
            "--save-bootstrap-trees",
            "true",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "UPGMA" / "bootstrap_trees.nwk").exists()


def test_cluster_partition_tables_include_sample_column(tmp_path):
    emb_path, labels_path = _make_embeddings_with_sample_and_labels(tmp_path)
    out_dir = tmp_path / "cluster_out_sample_columns"
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--label-csv",
            str(labels_path),
            "--custom-cutoffs",
            "0.5",
        ],
    )
    assert result.exit_code == 0

    assign_path = (
        out_dir / "UPGMA" / "partitions" / "tables" / "partition_0.5_assignments.csv"
    )
    summary_path = (
        out_dir / "UPGMA" / "partitions" / "tables" / "partition_0.5_summary.csv"
    )
    assert assign_path.exists()
    assert summary_path.exists()

    assign_df = pd.read_csv(assign_path)
    summary_df = pd.read_csv(summary_path)
    assert "sample" in assign_df.columns
    assert "sample" in summary_df.columns
    assert set(assign_df["sample"]) == {"s1", "s2"}


def test_cluster_euclidean_uses_l2_normalized_embeddings(tmp_path):
    emb = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "dim_0": [100.0, 0.0, -100.0],
            "dim_1": [0.0, 100.0, 0.0],
        }
    )
    labels = pd.DataFrame({"id": ["a", "b", "c"], "label": ["x", "y", "z"]})
    emb_path = tmp_path / "emb.csv"
    labels_path = tmp_path / "lbl.csv"
    emb.to_csv(emb_path, index=False)
    labels.to_csv(labels_path, index=False)

    out_dir = tmp_path / "cluster_out_euclidean_norm"
    result = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
            "--distance",
            "euclidean",
            "--custom-cutoffs",
            "0.2",
            "--label-csv",
            str(labels_path),
            "--save-distances",
            "true",
        ],
    )
    assert result.exit_code == 0
    dist = pd.read_csv(
        out_dir / "distance_statistics" / "distance_matrix.csv", index_col=0
    )
    vals = dist.to_numpy()
    assert vals.max() <= 2.000001


def test_annotate_command(tmp_path):
    runs_dir = tmp_path / "runs"
    cluster_dir = runs_dir / "cluster"
    tables_dir = cluster_dir / "UPGMA" / "partitions" / "tables"
    logs_dir = cluster_dir / "logs"
    extract_dir = runs_dir / "extract"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    assignments = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "cluster": ["OTU_1", "OTU_1", "OTU_2"],
            "sample": ["S1", "S1", "S2"],
        }
    )
    assign_path = tables_dir / "partition_0.2_assignments.csv"
    assignments.to_csv(assign_path, index=False)
    assignments.to_csv(tables_dir / "partition_0.25_assignments.csv", index=False)

    emb = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "dim_0": [0.1, 0.2, 0.8],
            "dim_1": [0.2, 0.1, 0.9],
        }
    )
    emb.to_csv(extract_dir / "embeddings.csv", index=False)

    cluster_log = "\n".join(
        [
            "[2026-04-01 00:00:00] Parameters:",
            "[2026-04-01 00:00:00] {",
            '[2026-04-01 00:00:00]   "distance": "cosine",',
            '[2026-04-01 00:00:00]   "embeddings": "runs/extract/embeddings.csv",',
            '[2026-04-01 00:00:00]   "pca_whitening": false,',
            '[2026-04-01 00:00:00]   "pca_components": 256,',
            '[2026-04-01 00:00:00]   "local_scaling": false,',
            '[2026-04-01 00:00:00]   "local_k": 0,',
            '[2026-04-01 00:00:00]   "local_k_strategy": "adaptive"',
            "[2026-04-01 00:00:00] }",
        ]
    )
    (logs_dir / "cluster.log").write_text(cluster_log, encoding="utf-8")

    corrections = pd.DataFrame({"id": ["b"], "cluster": ["OTU_2"]})
    corr_path = tmp_path / "corrections.csv"
    corrections.to_csv(corr_path, index=False)

    result = runner.invoke(
        app,
        [
            "annotate",
            "--raw-assignments",
            str(assign_path),
            "--corrections",
            str(corr_path),
            "--embeddings",
            str(extract_dir / "embeddings.csv"),
            "--out-dir",
            str(tmp_path / "annotate_out"),
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "annotate_out" / "partition_0.2_assignments.csv").exists()
    assert (tmp_path / "annotate_out" / "logs" / "annotate.log").exists()


def test_diversity_command(tmp_path):
    assignments = pd.DataFrame(
        {
            "id": [f"img{i}" for i in range(10)],
            "cluster": ["OTU_1"] * 6 + ["OTU_2"] * 4,
        }
    )
    assign_path = tmp_path / "assignments.csv"
    assignments.to_csv(assign_path, index=False)
    result = runner.invoke(
        app,
        [
            "diversity",
            "--assignments",
            str(assign_path),
            "--out-dir",
            str(tmp_path / "diversity_out"),
            "--min-abundance",
            "0,2",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "diversity_out" / "diversity_indices.csv").exists()
    assert (tmp_path / "diversity_out" / "logs" / "diversity.log").exists()


def test_diversity_log_command_uses_flag_style_for_bools(tmp_path):
    assignments = pd.DataFrame(
        {
            "id": ["img1", "img2", "img3", "img4"],
            "cluster": ["OTU_1", "OTU_1", "OTU_2", "OTU_3"],
        }
    )
    embeddings = pd.DataFrame(
        {
            "id": ["img1", "img2", "img3", "img4"],
            "dim_0": [1.0, 0.9, 0.0, 0.0],
            "dim_1": [0.0, 0.1, 1.0, 0.9],
        }
    )
    assign_path = tmp_path / "assignments.csv"
    emb_path = tmp_path / "embeddings.csv"
    out_dir = tmp_path / "diversity_out_flags"
    assignments.to_csv(assign_path, index=False)
    embeddings.to_csv(emb_path, index=False)

    result = runner.invoke(
        app,
        [
            "diversity",
            "--assignments",
            str(assign_path),
            "--out-dir",
            str(out_dir),
            "--min-abundance",
            "0",
            "--phylo",
            "--embeddings",
            str(emb_path),
            "--save-nj-tree",
        ],
    )

    assert result.exit_code == 0
    log_text = (out_dir / "logs" / "diversity.log").read_text(encoding="utf-8")
    command_line = next(
        (line for line in log_text.splitlines() if "Command:" in line), ""
    )
    assert "--phylo" in command_line
    assert "--save-nj-tree" in command_line
    assert "--phylo true" not in command_line
    assert "--save-nj-tree true" not in command_line


def test_cam_command(tmp_path):
    pytest.importorskip("pytorch_grad_cam")
    pytest.importorskip("cv2")

    ckpt = _make_ckpt(tmp_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224)).save(img_dir / "test.jpg")

    result = runner.invoke(
        app,
        [
            "cam",
            "--checkpoint",
            str(ckpt),
            "--images-dir",
            str(img_dir),
            "--out-dir",
            str(tmp_path / "cam_out"),
            "--cam-method",
            "eigencam",
            "--max-images",
            "1",
            "--cam-batch-size",
            "1",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "cam_out" / "cam_summary.csv").exists()
    assert (tmp_path / "cam_out" / "logs" / "cam.log").exists()


def test_cam_command_does_not_leave_closed_logging_stream(monkeypatch, tmp_path):
    from otuformer.utils.logging import TeeLogger

    def fake_run_cam(**kwargs):
        out_dir = kwargs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"image": "x", "status": "ok"}]).to_csv(
            out_dir / "cam_summary.csv", index=False
        )

    monkeypatch.setattr("otuformer.vision.cam.run_cam", fake_run_cam)

    ckpt = _make_ckpt(tmp_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224)).save(img_dir / "test.jpg")

    result = runner.invoke(
        app,
        [
            "cam",
            "--checkpoint",
            str(ckpt),
            "--images-dir",
            str(img_dir),
            "--out-dir",
            str(tmp_path / "cam_out_cleanup"),
            "--cam-method",
            "eigencam",
            "--max-images",
            "1",
            "--cam-batch-size",
            "1",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0

    for handler in logging.getLogger().handlers:
        stream = getattr(handler, "stream", None)
        assert not isinstance(stream, TeeLogger)


def test_pretrain_default_model_name_and_device_auto(tmp_path, monkeypatch):
    seen = {}

    def fake_run_pretrain(args):
        seen["model_name"] = args.model_name
        seen["device"] = args.device
        seen["extract_size"] = args.extract_size
        seen["augmentation"] = args.augmentation
        seen["orientation_policy"] = args.orientation_policy
        seen["local_crop_size"] = args.local_crop_size
        seen["local_crops"] = args.local_crops

    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", fake_run_pretrain)

    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(tmp_path / "images.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
        ],
    )
    assert result.exit_code == 0
    assert seen["model_name"] == "vit_tiny_patch16_224"
    assert seen["device"] == "auto"
    assert seen["extract_size"] is None
    assert seen["augmentation"] is None
    assert seen["orientation_policy"] is None
    assert seen["local_crop_size"] is None
    assert seen["local_crops"] is None


def test_pretrain_help_mentions_umap_metric_choices_and_extract_auto():
    result = runner.invoke(app, ["pretrain", "--help"])
    assert result.exit_code == 0
    assert "cosine" in result.output
    assert "euclidean" in result.output
    assert "backbone" in result.output.lower()
    assert "--no-disable-cross-view-loss" not in result.output
    assert "--no-compute-embedding-metrics" not in result.output


def test_pretrain_save_every_epochs_and_console_progress(tmp_path):
    csv_path = _make_tiny_pretrain_data(tmp_path)
    out_dir = tmp_path / "pretrain_save_every"
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(csv_path),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--model-name",
            "vit_tiny_patch16_224",
            "--out-dim",
            "64",
            "--max-epochs",
            "2",
            "--save-every-epochs",
            "1",
            "--log-every-n-steps",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--global-crop-size",
            "224",
            "--local-crop-size",
            "96",
            "--local-crops",
            "2",
            "--disable-embedding-metrics",
            "--disable-cross-view-loss",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert "Epoch 1/2" in result.output
    assert (out_dir / "SSL_epoch_0001.pth").exists()
    assert (out_dir / "SSL_epoch_0002.pth").exists()
    assert (out_dir / "SSL_latest.pth").exists()


def test_pretrain_embedding_metrics_and_umap_toggle(tmp_path):
    records = []
    for i in range(10):
        Image.new("RGB", (224, 224), color=(i * 20, 30, 10)).save(
            tmp_path / f"img_{i}.jpg"
        )
        records.append({"image": f"img_{i}.jpg", "label": f"class_{i % 2}"})
    labeled_csv = tmp_path / "labeled.csv"
    pd.DataFrame(records).to_csv(labeled_csv, index=False)

    out_dir = tmp_path / "pretrain_metrics_on"
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(labeled_csv),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--model-name",
            "vit_tiny_patch16_224",
            "--out-dim",
            "64",
            "--max-epochs",
            "1",
            "--save-every-epochs",
            "1",
            "--log-every-n-steps",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--global-crop-size",
            "224",
            "--local-crop-size",
            "96",
            "--local-crops",
            "2",
            "--disable-cross-view-loss",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "logs" / "metrics.pretrain.csv").exists()
    curves_path = out_dir / "logs" / "training_curves_pretrain.pdf"
    assert curves_path.exists() or "skip training curves" in result.output
    assert "unexpected keyword argument" not in result.output
    umap_epoch = out_dir / "logs" / "umap.train.epoch_0001.pdf"
    assert (
        umap_epoch.exists()
        or "Missing optional dependency 'umap-learn'" in result.output
    )

    out_dir_off = tmp_path / "pretrain_metrics_off"
    result_off = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(labeled_csv),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir_off),
            "--model-name",
            "vit_tiny_patch16_224",
            "--out-dim",
            "64",
            "--max-epochs",
            "1",
            "--save-every-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--global-crop-size",
            "224",
            "--local-crop-size",
            "96",
            "--local-crops",
            "2",
            "--disable-embedding-metrics",
            "--disable-cross-view-loss",
            "--device",
            "cpu",
        ],
    )
    assert result_off.exit_code == 0
    assert not (out_dir_off / "logs" / "metrics.pretrain.csv").exists()


def test_pretrain_log_contains_command_params_and_stderr_on_error(
    tmp_path, monkeypatch
):
    def fake_run_pretrain(_args):
        print("stdout-line")
        import sys

        sys.stderr.write("\x1b[31mstderr-line\x1b[0m\n")
        raise RuntimeError("boom")

    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", fake_run_pretrain)

    out_dir = tmp_path / "pretrain_out"
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(tmp_path / "images.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0

    log_path = out_dir / "logs" / "pretrain.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    command_line = next(
        (line for line in log_text.splitlines() if line.startswith("Command:")), ""
    )
    assert "Command:" in log_text
    assert "otuformer pretrain" in log_text
    assert "--model-name" not in command_line
    assert "Parameters:" in log_text
    assert "model_name" in log_text
    assert "stdout-line" in log_text
    assert "stderr-line" in log_text
    assert "Traceback (most recent call last):" in log_text
    assert "RuntimeError: boom" in log_text


def test_pretrain_enables_mps_fallback_env_for_auto(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    seen = {}

    def fake_run_pretrain(_args):
        seen["fallback"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")

    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", fake_run_pretrain)

    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(tmp_path / "images.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
            "--device",
            "auto",
        ],
    )

    assert result.exit_code == 0
    assert seen["fallback"] == "1"


@pytest.mark.parametrize(
    ("command", "run_attr", "augmentation"),
    [
        ("pretrain", "otuformer.training.trainer.run_pretrain", "global-barcode"),
        ("finetune", "otuformer.training.trainer.run_finetune", "conservative"),
    ],
)
def test_training_enables_mps_fallback_before_augmentation_validation(
    tmp_path, monkeypatch, command, run_attr, augmentation
):
    """PyTorch latches PYTORCH_ENABLE_MPS_FALLBACK at import time, so the CLI
    must set it before the augmentation validators import torch."""
    prior_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
    seen = {}

    def spy_validate_augmentation(value, *, stage):
        seen["fallback"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")

    monkeypatch.setattr(
        f"otuformer.cli.{command}._validate_augmentation", spy_validate_augmentation
    )
    monkeypatch.setattr(run_attr, lambda _args: None)

    try:
        result = runner.invoke(
            app,
            [
                command,
                "--train-data",
                str(tmp_path / "data.csv"),
                "--input-images-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--device",
                "mps",
                "--augmentation",
                augmentation,
            ],
        )
    finally:
        if prior_fallback is None:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = prior_fallback

    assert result.exit_code == 0, result.output
    assert seen.get("fallback") == "1"


# ---- Resolution & preprocessing consistency (2026-09-07 plan) ----


@pytest.mark.parametrize("cmd", ["pretrain", "finetune", "extract", "export"])
def test_size_options_document_auto_and_examples(cmd):
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
    assert "auto" in result.output
    for example in ["224", "384", "448"]:
        assert example in result.output


def test_size_options_qualify_518_as_patch14_example():
    for cmd in ["pretrain", "finetune", "extract", "export"]:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0
        if "518" in result.output:
            assert "patch-14" in result.output


def test_extract_size_auto_and_explicit_parse(tmp_path, monkeypatch):
    seen = {}

    def fake_extract(**kwargs):
        seen["extract_size"] = kwargs["extract_size"]
        return pd.DataFrame({"id": [], "dim_0": []})

    monkeypatch.setattr("otuformer.embedding.extractor.extract_embeddings", fake_extract)

    _make_tiny_pretrain_data(tmp_path)
    for flag in ["auto", "224"]:
        result = runner.invoke(
            app,
            [
                "extract",
                "--input-images-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / f"out_{flag}"),
                "--extract-size",
                flag,
            ],
        )
        assert result.exit_code == 0
        expected = None if flag == "auto" else 224
        assert seen["extract_size"] == expected


@pytest.mark.parametrize("flag", ["0", "abc", "-8"])
def test_extract_size_rejects_invalid_before_execution(tmp_path, flag):
    result = runner.invoke(
        app,
        [
            "extract",
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--extract-size",
            flag,
        ],
    )
    assert result.exit_code != 0
    assert "positive integer" in result.output


def test_global_crop_size_rejects_non_divisible_patch16(tmp_path):
    csv_path = _make_tiny_pretrain_data(tmp_path)
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(csv_path),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
            "--global-crop-size",
            "518",
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--disable-embedding-metrics",
            "--disable-cross-view-loss",
        ],
    )
    assert result.exit_code != 0
    assert "518 is not divisible by patch size 16" in result.output
    assert "vit_tiny_patch16_224" in result.output


def test_pretrain_resume_rejects_changed_input_size(tmp_path):
    ckpt = _make_ckpt(tmp_path)  # 224 checkpoint
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--resume",
            str(ckpt),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
            "--train-data",
            str(tmp_path / "data.csv"),
            "--global-crop-size",
            "384",
        ],
    )
    assert result.exit_code != 0
    assert "resume cannot change the input size" in result.output


def test_pretrain_resume_allows_equal_input_size(tmp_path, monkeypatch):
    ckpt = _make_ckpt(tmp_path)  # 224 checkpoint
    seen = {}

    def fake_run_pretrain(args):
        seen["global_crop_size"] = args.global_crop_size

    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", fake_run_pretrain)

    result = runner.invoke(
        app,
        [
            "pretrain",
            "--resume",
            str(ckpt),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
            "--train-data",
            str(tmp_path / "data.csv"),
            "--global-crop-size",
            "224",
        ],
    )
    assert result.exit_code == 0
    assert seen["global_crop_size"] == 224


def test_pretrain_auto_global_crop_size_forwards_none(tmp_path, monkeypatch):
    seen = {}

    def fake_run_pretrain(args):
        seen["global_crop_size"] = args.global_crop_size
        seen["extract_size"] = args.extract_size

    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", fake_run_pretrain)

    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(tmp_path / "images.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
        ],
    )
    assert result.exit_code == 0
    assert seen["global_crop_size"] is None
    assert seen["extract_size"] is None


def test_export_imgsz_auto_and_explicit(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--out-dir",
            str(tmp_path / "export_auto"),
            "--imgsz",
            "auto",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "export_auto" / "encoder.onnx").exists()

    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--out-dir",
            str(tmp_path / "export_224"),
            "--imgsz",
            "224",
        ],
    )
    assert result.exit_code == 0


def test_extract_help_documents_eval_transform():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "eval-transform" in result.output
    assert "center-crop" in result.output
    assert "whole-specimen-pad" in result.output


def test_cam_help_documents_eval_transform():
    result = runner.invoke(app, ["cam", "--help"])
    assert result.exit_code == 0
    assert "eval-transform" in result.output
    assert "center-crop" in result.output
    assert "whole-specimen-pad" in result.output


def test_extract_rejects_invalid_eval_transform(tmp_path):
    result = runner.invoke(
        app,
        [
            "extract",
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--eval-transform",
            "bogus",
        ],
    )
    assert result.exit_code != 0
    assert "bogus" in result.output


def test_cam_rejects_invalid_eval_transform(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    result = runner.invoke(
        app,
        [
            "cam",
            "--checkpoint",
            str(ckpt),
            "--images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--eval-transform",
            "bogus",
        ],
    )
    assert result.exit_code != 0
    assert "bogus" in result.output


def test_extract_forwards_eval_transform(monkeypatch, tmp_path):
    seen = {}

    def fake_extract(**kwargs):
        seen["eval_transform"] = kwargs["eval_transform"]
        return pd.DataFrame({"id": [], "dim_0": []})

    monkeypatch.setattr("otuformer.embedding.extractor.extract_embeddings", fake_extract)

    _make_tiny_pretrain_data(tmp_path)
    result = runner.invoke(
        app,
        [
            "extract",
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--eval-transform",
            "whole-specimen-pad",
        ],
    )
    assert result.exit_code == 0
    assert seen["eval_transform"] == "whole-specimen-pad"


def test_extract_default_eval_transform_is_center_crop(monkeypatch, tmp_path):
    seen = {}

    def fake_extract(**kwargs):
        seen["eval_transform"] = kwargs["eval_transform"]
        return pd.DataFrame({"id": [], "dim_0": []})

    monkeypatch.setattr("otuformer.embedding.extractor.extract_embeddings", fake_extract)

    _make_tiny_pretrain_data(tmp_path)
    result = runner.invoke(
        app,
        [
            "extract",
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0
    assert seen["eval_transform"] == "center-crop"


def test_cam_forwards_eval_transform(monkeypatch, tmp_path):
    seen = {}

    def fake_run_cam(**kwargs):
        seen["eval_transform"] = kwargs["eval_transform"]
        (kwargs["out_dir"] / "figures").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("otuformer.vision.cam.run_cam", fake_run_cam)

    ckpt = _make_ckpt(tmp_path)
    result = runner.invoke(
        app,
        [
            "cam",
            "--checkpoint",
            str(ckpt),
            "--images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--eval-transform",
            "whole-specimen-pad",
        ],
    )
    assert result.exit_code == 0
    assert seen["eval_transform"] == "whole-specimen-pad"


def test_pretrain_local_crop_size_validated_against_patch(tmp_path):
    csv_path = _make_tiny_pretrain_data(tmp_path)
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(csv_path),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
            "--local-crop-size",
            "97",
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--disable-embedding-metrics",
            "--disable-cross-view-loss",
        ],
    )
    assert result.exit_code != 0
    assert "97 is not divisible by patch size 16" in result.output
    assert "vit_tiny_patch16_224" in result.output


def test_pretrain_local_crops_zero_skips_local_size_validation(tmp_path):
    """--local-crops 0 never uses local_crop_size, so an invalid local size
    must not be rejected (mirrors the patch-14 + local_crops=0 config)."""
    csv_path = _make_tiny_pretrain_data(tmp_path)
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(csv_path),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
            "--global-crop-size",
            "224",
            "--local-crop-size",
            "97",  # invalid for patch-16, but unused when local_crops=0
            "--local-crops",
            "0",
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--disable-embedding-metrics",
            "--disable-cross-view-loss",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "pretrain_out" / "SSL_latest.pth").exists()


# ---- Training augmentation CLI contract (2026-09-07 plan) ----


@pytest.mark.parametrize(
    ("command", "profiles", "default"),
    [
        ("pretrain", ["global-barcode", "color-robust", "legacy"], "global-barcode"),
        ("finetune", ["none", "conservative"], "none"),
    ],
)
def test_training_help_documents_augmentation_contract(command, profiles, default):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    output = result.output.lower()
    assert "--augmentation" in output
    assert "--orientation-policy" in output
    assert "invariant" in output and "sensitive" in output
    assert all(profile in output for profile in profiles)
    assert f"default for a new run: {default}" in output
    assert "default for a new run: sensitive" in output
    assert "dorsal" in output and "ventral" in output and "lateral" in output
    assert "in-plane" in output
    assert "does not guarantee" in output


def test_pretrain_help_warns_about_color_robust_and_legacy():
    output = runner.invoke(app, ["pretrain", "--help"]).output.lower()
    assert "diagnostic" in output
    assert "metallic" in output
    assert "0.2.1" in output
    assert "new run" in output


def test_finetune_help_marks_conservative_experimental():
    output = runner.invoke(app, ["finetune", "--help"]).output.lower()
    assert "experimental" in output
    assert "new run" in output


def _write_augmented_pretrain_checkpoint(
    tmp_path,
    *,
    local_crop_size,
    local_crops,
    profile="global-barcode",
    policy="invariant",
    image_size=224,
):
    from otuformer.training.dataset import build_pretrain_augmentation_config
    from otuformer.training.model import OTUFormerEncoder

    config = build_pretrain_augmentation_config(
        profile,
        image_size,
        local_crop_size,
        local_crops,
        orientation_policy=policy,
    )
    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=64,
        pretrained=False,
        img_size=image_size,
    )
    path = tmp_path / "augmented_resume.pth"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 64,
                "augmentation_profile": profile,
                "augmentation_config": config,
            },
            "args": {
                "global_crop_size": image_size,
                "local_crop_size": local_crop_size,
                "local_crops": local_crops,
            },
        },
        path,
    )
    return path


def test_pretrain_augmentation_argument_forwarding_defaults_to_none(
    tmp_path, monkeypatch
):
    seen = {}

    def fake_run_pretrain(args):
        seen["augmentation"] = args.augmentation
        seen["orientation_policy"] = args.orientation_policy
        seen["local_crop_size"] = args.local_crop_size
        seen["local_crops"] = args.local_crops

    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", fake_run_pretrain)

    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(tmp_path / "images.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {
        "augmentation": None,
        "orientation_policy": None,
        "local_crop_size": None,
        "local_crops": None,
    }


def test_pretrain_augmentation_argument_forwarding_explicit_values(
    tmp_path, monkeypatch
):
    seen = {}

    def fake_run_pretrain(args):
        seen["augmentation"] = args.augmentation
        seen["orientation_policy"] = args.orientation_policy
        seen["local_crop_size"] = args.local_crop_size
        seen["local_crops"] = args.local_crops

    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", fake_run_pretrain)

    result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(tmp_path / "images.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "pretrain_out"),
            "--augmentation",
            "legacy",
            "--orientation-policy",
            "sensitive",
            "--local-crop-size",
            "112",
            "--local-crops",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {
        "augmentation": "legacy",
        "orientation_policy": "sensitive",
        "local_crop_size": 112,
        "local_crops": 0,
    }


def test_format_user_command_omits_unset_augmentation_flags(tmp_path, monkeypatch):
    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", lambda _args: None)

    def command_line(out_dir):
        log_text = (out_dir / "logs" / "pretrain.log").read_text(encoding="utf-8")
        return next(line for line in log_text.splitlines() if "Command:" in line)

    default_out = tmp_path / "format_default"
    default_result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(tmp_path / "images.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(default_out),
        ],
    )
    assert default_result.exit_code == 0, default_result.output
    default_line = command_line(default_out)
    assert "--augmentation" not in default_line
    assert "--orientation-policy" not in default_line

    explicit_out = tmp_path / "format_explicit"
    explicit_result = runner.invoke(
        app,
        [
            "pretrain",
            "--train-data",
            str(tmp_path / "images.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(explicit_out),
            "--augmentation",
            "color-robust",
            "--orientation-policy",
            "invariant",
        ],
    )
    assert explicit_result.exit_code == 0, explicit_result.output
    explicit_line = command_line(explicit_out)
    assert "--augmentation color-robust" in explicit_line
    assert "--orientation-policy invariant" in explicit_line


def test_finetune_augmentation_argument_forwarding_defaults_to_none(
    tmp_path, monkeypatch
):
    seen = {}

    def fake_run_finetune(args):
        seen["augmentation"] = args.augmentation
        seen["orientation_policy"] = args.orientation_policy

    monkeypatch.setattr("otuformer.training.trainer.run_finetune", fake_run_finetune)

    result = runner.invoke(
        app,
        [
            "finetune",
            "--train-data",
            str(tmp_path / "labels.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "finetune_out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {"augmentation": None, "orientation_policy": None}


def test_finetune_augmentation_argument_forwarding_explicit_values(
    tmp_path, monkeypatch
):
    seen = {}

    def fake_run_finetune(args):
        seen["augmentation"] = args.augmentation
        seen["orientation_policy"] = args.orientation_policy

    monkeypatch.setattr("otuformer.training.trainer.run_finetune", fake_run_finetune)

    result = runner.invoke(
        app,
        [
            "finetune",
            "--train-data",
            str(tmp_path / "labels.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "finetune_out"),
            "--augmentation",
            "conservative",
            "--orientation-policy",
            "sensitive",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {"augmentation": "conservative", "orientation_policy": "sensitive"}


def test_finetune_initialization_inherits_policy_but_not_profile(
    tmp_path, monkeypatch
):
    from otuformer.training import trainer

    ckpt = _write_augmented_pretrain_checkpoint(
        tmp_path,
        local_crop_size=96,
        local_crops=2,
        profile="color-robust",
        policy="sensitive",
    )
    seen = {}

    def fake_run_finetune(args):
        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        seen["augmentation"] = args.augmentation
        seen["orientation_policy"] = args.orientation_policy
        seen["profile"] = trainer._select_augmentation_profile(
            args.augmentation, None, stage="finetune"
        )
        seen["policy"] = trainer._select_orientation_policy(
            args.orientation_policy, loaded, stage="finetune", resume=False
        )

    monkeypatch.setattr("otuformer.training.trainer.run_finetune", fake_run_finetune)

    result = runner.invoke(
        app,
        [
            "finetune",
            "--checkpoint",
            str(ckpt),
            "--train-data",
            str(tmp_path / "labels.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "finetune_out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["augmentation"] is None
    assert seen["orientation_policy"] is None
    assert seen["profile"] == "none"
    assert seen["policy"] == "sensitive"


@pytest.mark.parametrize(
    ("command", "flag", "value"),
    [
        ("pretrain", "--augmentation", "bogus"),
        ("pretrain", "--orientation-policy", "bogus"),
        ("finetune", "--augmentation", "bogus"),
        ("finetune", "--orientation-policy", "bogus"),
    ],
)
def test_training_augmentation_argument_rejects_invalid_before_training(
    tmp_path, monkeypatch, command, flag, value
):
    called = []
    monkeypatch.setattr(
        "otuformer.training.trainer.run_pretrain", lambda _args: called.append("p")
    )
    monkeypatch.setattr(
        "otuformer.training.trainer.run_finetune", lambda _args: called.append("f")
    )
    out_dir = tmp_path / "augmentation_out"

    result = runner.invoke(
        app,
        [
            command,
            "--train-data",
            str(tmp_path / "data.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            flag,
            value,
        ],
    )

    assert result.exit_code != 0
    assert value in result.output
    assert called == []
    assert not out_dir.exists()


def test_pretrain_resume_inherits_saved_local_views_without_explicit_flags(
    tmp_path, monkeypatch
):
    from otuformer.training import trainer

    ckpt = _write_augmented_pretrain_checkpoint(
        tmp_path, local_crop_size=112, local_crops=0
    )
    seen = {}

    def fake_run_pretrain(args):
        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        seen["local_crop_size"] = args.local_crop_size
        seen["local_crops"] = args.local_crops
        seen["resolved"] = trainer._resolve_pretrain_local_views(
            args.local_crop_size, args.local_crops, loaded
        )

    monkeypatch.setattr("otuformer.training.trainer.run_pretrain", fake_run_pretrain)

    result = runner.invoke(
        app,
        [
            "pretrain",
            "--resume",
            str(ckpt),
            "--train-data",
            str(tmp_path / "data.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "resume_out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["local_crop_size"] is None
    assert seen["local_crops"] is None
    assert seen["resolved"] == (112, 0)


def test_pretrain_resume_rejects_conflicting_local_views(tmp_path):
    ckpt = _write_augmented_pretrain_checkpoint(
        tmp_path, local_crop_size=112, local_crops=0
    )

    result = runner.invoke(
        app,
        [
            "pretrain",
            "--resume",
            str(ckpt),
            "--train-data",
            str(tmp_path / "data.csv"),
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "resume_out"),
            "--local-crops",
            "6",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "Cannot resume" in str(result.exception)
    assert not (tmp_path / "resume_out" / "SSL_latest.pth").exists()


class _DatasetCaptured(Exception):
    """Stop run_pretrain right after dataset construction."""


def test_run_pretrain_resume_inherits_new_style_augmentation_into_dataset(
    tmp_path, monkeypatch
):
    """run_pretrain must forward the profile, policy, and local-view settings
    inherited from a new-style checkpoint to the pretrain dataset."""
    from otuformer.training import trainer

    ckpt = _write_augmented_pretrain_checkpoint(
        tmp_path,
        local_crop_size=112,
        local_crops=3,
        profile="color-robust",
        policy="sensitive",
    )
    seen = {}

    def capturing_multi_crop_dataset(**kwargs):
        seen.update(kwargs)
        raise _DatasetCaptured

    monkeypatch.setattr(trainer, "MultiCropDataset", capturing_multi_crop_dataset)

    args = argparse.Namespace(
        seed=42,
        cpus=1,
        device="cpu",
        out_dir=str(tmp_path / "resume_out"),
        resume=str(ckpt),
        train_data=str(tmp_path / "data.csv"),
        input_images_dir=str(tmp_path),
        model_name="vit_tiny_patch16_224",
        global_crop_size=None,
        augmentation=None,
        orientation_policy=None,
        local_crop_size=None,
        local_crops=None,
        batch_size=2,
        num_workers=0,
    )
    with pytest.raises(_DatasetCaptured):
        trainer.run_pretrain(args)

    assert seen["augmentation_profile"] == "color-robust"
    assert seen["orientation_policy"] == "sensitive"
    assert seen["local_crop_size"] == 112
    assert seen["local_crops"] == 3
    assert seen["global_crop_size"] == 224
