from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from otuformer.cli.main import app
from otuformer.training.model import OTUFormerEncoder


runner = CliRunner()


def _make_ckpt(tmp_path: Path, out_dim: int = 32) -> Path:
    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=out_dim)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim},
    }
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, p)
    return p


def _prepare_example_subset(tmp_path: Path) -> tuple[Path, Path]:
    examples_root = Path("/Users/zf/data/coding/OTU-Former/examples/Epidorcus/images")
    src_a = sorted((examples_root / "Epidorcus_gracilis").glob("*.jpg"))[0]
    src_b = sorted((examples_root / "Epidorcus_tonkinensis").glob("*.jpg"))[0]

    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dst_a = image_dir / "gracilis_1.jpg"
    dst_b = image_dir / "tonkinensis_1.jpg"
    Image.open(src_a).convert("RGB").save(dst_a)
    Image.open(src_b).convert("RGB").save(dst_b)

    labels = pd.DataFrame(
        {
            "image": [dst_a.name, dst_b.name],
            "label": ["Epidorcus_gracilis", "Epidorcus_tonkinensis"],
        }
    )
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    # id-based labels for cluster command (id/label columns)
    id_labels = pd.DataFrame(
        {
            "id": [dst_a.name, dst_b.name],
            "label": ["Epidorcus_gracilis", "Epidorcus_tonkinensis"],
        }
    )
    id_labels_path = tmp_path / "id_labels.csv"
    id_labels.to_csv(id_labels_path, index=False)

    return image_dir, labels_path, id_labels_path


def test_examples_epidorcus_e2e_low_memory(tmp_path: Path):
    ckpt = _make_ckpt(tmp_path)
    image_dir, labels_path, id_labels_path = _prepare_example_subset(tmp_path)

    extract_out = tmp_path / "extract_out"
    res = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(ckpt),
            "--input-images-dir",
            str(image_dir),
            "--out-dir",
            str(extract_out),
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
            "--label-csv",
            str(labels_path),
            "--metrics-sample-size",
            "2",
            "--disable-umap",
        ],
    )
    assert res.exit_code == 0
    emb_path = extract_out / "embeddings.csv"
    assert emb_path.exists()
    assert (extract_out / "metrics.csv").exists()

    cluster_out = tmp_path / "cluster_out"
    res = runner.invoke(
        app,
        [
            "cluster",
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(cluster_out),
            "--labels",
            str(id_labels_path),
            "--distance",
            "cosine",
            "--custom-cutoffs",
            "0.5",
            "--max-distance-pairs",
            "10",
            "--cpus",
            "1",
        ],
    )
    assert res.exit_code == 0

    assignments = (
        cluster_out
        / "UPGMA"
        / "partitions"
        / "tables"
        / "partition_0.5_assignments.csv"
    )
    assert assignments.exists()
    assert not (cluster_out / "partition_0.5_assignments.csv").exists()

    diversity_out = tmp_path / "diversity_out"
    res = runner.invoke(
        app,
        [
            "diversity",
            "--assignments",
            str(assignments),
            "--out-dir",
            str(diversity_out),
            "--min-abundance",
            "0,1",
        ],
    )
    assert res.exit_code == 0
    assert (diversity_out / "diversity_indices.csv").exists()

    export_out = tmp_path / "export_out"
    res = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--out-dir",
            str(export_out),
            "--imgsz",
            "224",
            "--opset",
            "17",
        ],
    )
    assert res.exit_code == 0
    assert (export_out / "encoder.onnx").exists()
