from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from otuformer.embedding.extractor import (
    GatedAttentionPooling,
    detect_batch_mode,
    extract_embeddings,
    _iter_trainable_params,
    _load_model,
)


def make_checkpoint(tmp_path: Path, out_dim: int = 64) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=out_dim)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim},
    }
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, p)
    return p


def make_images(directory: Path, n: int = 3) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (224, 224), color=(i * 80, 0, 0)).save(
            directory / f"img_{i}.jpg"
        )


def test_extractor_checkpoint_loader_disables_pretrained_weights(monkeypatch, tmp_path):
    import otuformer.embedding.extractor as extractor_module

    seen = {}

    class FakeEncoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)

        def load_state_dict(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(extractor_module, "load_checkpoint", lambda _path: {"model_state_dict": {}})
    monkeypatch.setattr(extractor_module, "OTUFormerEncoder", FakeEncoder)

    _load_model(tmp_path / "model.pth", "vit_tiny_patch16_224", torch.device("cpu"))

    assert seen["pretrained"] is False


def test_extract_single_dir(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir)
    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
    )
    assert isinstance(out, pd.DataFrame)
    assert "id" in out.columns
    assert "sample" not in out.columns
    assert len(out) == 3
    assert out.shape[1] > 2


def test_detect_batch_mode(tmp_path: Path):
    img_dir = tmp_path / "single"
    make_images(img_dir)
    assert detect_batch_mode(img_dir) is False

    parent = tmp_path / "multi"
    make_images(parent / "site_a")
    make_images(parent / "site_b")
    assert detect_batch_mode(parent) is True


def test_extract_batch_mode(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    parent = tmp_path / "multi"
    make_images(parent / "site_a", n=2)
    make_images(parent / "site_b", n=3)
    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=parent,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
    )
    assert "sample" in out.columns
    assert len(out) == 5
    assert set(out["sample"].unique()) == {"site_a", "site_b"}


def test_extract_patch_topk_mode_changes_embedding_shape(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)

    cls_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        token_mode="cls",
    )
    topk_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        token_mode="patch-topk",
        topk_patches=4,
    )

    cls_dim = len([c for c in cls_out.columns if c.startswith("dim_")])
    topk_dim = len([c for c in topk_out.columns if c.startswith("dim_")])
    assert topk_dim != cls_dim
    assert topk_dim <= 512


def test_extract_attention_pool_requires_training_csv_without_finetuned_pool(
    tmp_path: Path,
):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)

    with pytest.raises(ValueError, match="label-csv"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            token_mode="attention-pool",
        )


def test_extract_attention_pool_rejects_image_only_training_csv(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    csv_path = tmp_path / "images.csv"
    pd.DataFrame({"image": ["img_0.jpg", "img_1.jpg"]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="image.*label"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            token_mode="attention-pool",
            attention_train_csv=csv_path,
        )


def test_extract_attention_pool_validates_csv_before_start_message(
    tmp_path: Path, capsys
):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    csv_path = tmp_path / "images.csv"
    pd.DataFrame({"image": ["img_0.jpg", "img_1.jpg"]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="image.*label"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            token_mode="attention-pool",
            attention_train_csv=csv_path,
        )

    assert "Starting query finetuning" not in capsys.readouterr().out


def test_extract_csv_preserves_csv_order(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=3)

    csv_path = tmp_path / "subset.csv"
    pd.DataFrame({"image": ["img_2.jpg", "img_0.jpg"]}).to_csv(csv_path, index=False)

    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        device="cpu",
        batch_size=2,
        token_mode="cls",
        extract_csv=csv_path,
    )
    assert out["id"].tolist() == ["img_2.jpg", "img_0.jpg"]


def test_extract_csv_infers_sample_for_subdirs(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    parent = tmp_path / "multi"
    make_images(parent / "site_a", n=1)
    make_images(parent / "site_b", n=1)

    csv_path = tmp_path / "subset.csv"
    pd.DataFrame(
        {
            "image": ["site_a/img_0.jpg", "site_b/img_0.jpg"],
            "label": ["x", "y"],
        }
    ).to_csv(csv_path, index=False)

    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=parent,
        device="cpu",
        batch_size=2,
        token_mode="cls",
        extract_csv=csv_path,
    )
    assert "sample" in out.columns
    assert out["sample"].tolist() == ["site_a", "site_b"]


def test_gated_attention_has_trainable_params():
    pool = GatedAttentionPooling(dim=192)
    trainable = list(_iter_trainable_params(pool))
    assert len(trainable) > 0


def make_onnx(tmp_path: Path, out_dim: int = 64) -> Path:
    from otuformer.vision.export import export_to_onnx

    ckpt = make_checkpoint(tmp_path, out_dim=out_dim)
    onnx_path = tmp_path / "encoder.onnx"
    export_to_onnx(checkpoint_path=ckpt, out_path=onnx_path, imgsz=224, opset=18)
    return onnx_path


def test_extract_with_onnx_produces_same_shape(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=3)

    torch_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
        use_projector_output=True,
    )

    onnx_path = make_onnx(tmp_path)
    onnx_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
        onnx_path=onnx_path,
    )

    assert isinstance(onnx_out, pd.DataFrame)
    assert "id" in onnx_out.columns
    assert len(onnx_out) == len(torch_out)
    dim_cols = [c for c in onnx_out.columns if c.startswith("dim_")]
    assert len(dim_cols) == len([c for c in torch_out.columns if c.startswith("dim_")])


def test_extract_with_onnx_matches_pytorch_values(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=3)

    onnx_path = make_onnx(tmp_path)

    torch_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
        use_projector_output=True,
    )

    onnx_out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=224,
        batch_size=2,
        device="cpu",
        onnx_path=onnx_path,
    )

    dim_cols = [c for c in torch_out.columns if c.startswith("dim_")]
    torch_vals = torch_out[dim_cols].values
    onnx_vals = onnx_out[dim_cols].values

    import numpy as np

    np.testing.assert_allclose(torch_vals, onnx_vals, rtol=1e-4, atol=1e-4)


def test_extract_onnx_rejects_patch_topk(tmp_path: Path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir, n=2)
    onnx_path = make_onnx(tmp_path)

    with pytest.raises(ValueError, match="(?i)onnx"):
        extract_embeddings(
            checkpoint_path=ckpt,
            images_dir=img_dir,
            device="cpu",
            batch_size=2,
            token_mode="patch-topk",
            onnx_path=onnx_path,
        )
