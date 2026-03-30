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
            "id": [dst_a.name, dst_b.name],
            "label": ["Epidorcus_gracilis", "Epidorcus_tonkinensis"],
        }
    )
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    return image_dir, labels_path


def test_examples_epidorcus_e2e_low_memory(tmp_path: Path):
    ckpt = _make_ckpt(tmp_path)
    image_dir, labels_path = _prepare_example_subset(tmp_path)

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
        ],
    )
    assert res.exit_code == 0
    emb_path = extract_out / "embeddings.csv"
    assert emb_path.exists()

    eval_out = tmp_path / "evaluate_out"
    res = runner.invoke(
        app,
        [
            "evaluate",
            "--embeddings",
            str(emb_path),
            "--labels",
            str(labels_path),
            "--out-dir",
            str(eval_out),
            "--knn-k",
            "1",
            "--umap-dims",
            "2",
            "--metrics-sample-size",
            "2",
        ],
    )
    assert res.exit_code == 0
    assert (eval_out / "metrics.json").exists()

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
            str(labels_path),
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

    assignments = cluster_out / "partition_0.5_assignments.csv"
    assert assignments.exists()

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
