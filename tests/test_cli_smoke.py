import pytest
import pandas as pd
from PIL import Image
import torch
import os
from typer.testing import CliRunner

from otuformer.cli.main import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "otu-former" in result.output.lower() or "otuformer" in result.output.lower()


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
        ["evaluate", "--help"],
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
        ("extract", ["--checkpoint", "--token-mode", "--topk-patches"]),
        ("evaluate", ["--embeddings", "--labels", "--knn-k"]),
        ("cluster", ["--embeddings", "--distance", "--custom-cutoffs"]),
        ("annotate", ["--assignments", "--corrections", "--out-dir"]),
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


def _make_tiny_pretrain_data(tmp_path):
    for i in range(4):
        Image.new("RGB", (224, 224)).save(tmp_path / f"img_{i}.jpg")
    df = pd.DataFrame({"image": [f"img_{i}.jpg" for i in range(4)]})
    csv_path = tmp_path / "images.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_tiny_finetune_data(tmp_path):
    for i in range(4):
        Image.new("RGB", (224, 224)).save(tmp_path / f"img_{i}.jpg")
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
    assert (tmp_path / "finetune_out" / "last.pt").exists()


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


def test_evaluate_command(tmp_path):
    emb_path, labels_path = _make_embeddings_and_labels(tmp_path)
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--embeddings",
            str(emb_path),
            "--labels",
            str(labels_path),
            "--out-dir",
            str(tmp_path / "evaluate_out"),
            "--knn-k",
            "1,2",
            "--umap-dims",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "evaluate_out" / "metrics.json").exists()
    assert (tmp_path / "evaluate_out" / "metrics.csv").exists()


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
    assert any((tmp_path / "cluster_out").glob("partition_*_assignments.csv"))


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


def test_annotate_command(tmp_path):
    assignments = pd.DataFrame({"id": ["a", "b"], "cluster": ["OTU_1", "OTU_1"]})
    corrections = pd.DataFrame({"id": ["b"], "corrected_cluster": ["OTU_2"]})
    assign_path = tmp_path / "assignments.csv"
    corr_path = tmp_path / "corrections.csv"
    assignments.to_csv(assign_path, index=False)
    corrections.to_csv(corr_path, index=False)

    result = runner.invoke(
        app,
        [
            "annotate",
            "--assignments",
            str(assign_path),
            "--corrections",
            str(corr_path),
            "--out-dir",
            str(tmp_path / "annotate_out"),
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "annotate_out" / "assignments_annotated.csv").exists()


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


def test_pretrain_default_model_name_and_device_auto(tmp_path, monkeypatch):
    seen = {}

    def fake_run_pretrain(args):
        seen["model_name"] = args.model_name
        seen["device"] = args.device
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
    assert seen["model_name"] == "vit_tiny_patch16_224"
    assert seen["device"] == "auto"
    assert seen["extract_size"] == 0


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
        Image.new("RGB", (224, 224)).save(tmp_path / f"img_{i}.jpg")
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
