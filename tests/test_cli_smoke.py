import pytest
import pandas as pd
from PIL import Image
import torch
import os
import logging
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
    assert (tmp_path / "finetune_out" / "finetune_latest.pth").exists()


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


def test_cluster_cleans_old_outputs_before_writing(tmp_path):
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
