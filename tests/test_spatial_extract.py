"""Spatial request resolution, checkpoint extraction and offline read."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from otuformer.spatial.extractor import (
    SpatialRequestError,
    extract_spatial,
    prepare_spatial_request,
    reevaluate_spatial,
)
from otuformer.spatial.request import spatial_mode, validate_extract_mode_options


def write_images(root: Path, keys=("a.png", "nested/b.png")) -> Path:
    for key in keys:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 40), (30, 60, 90)).save(path)
    return root


def write_checkpoint(path: Path, image_size: int = 32) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(out_dim=8, pretrained=False, img_size=image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 8,
                "image_size": image_size,
            },
        },
        path,
    )
    return path


def base_values(tmp_path: Path, **overrides) -> dict:
    values = {
        "output_kind": "spatial",
        "checkpoint": None,
        "onnx_path": None,
        "input_images_dir": None,
        "out_dir": str(tmp_path / "out"),
        "model_name": "vit_tiny_patch16_224",
        "extract_size": "auto",
        "eval_transform": "whole-specimen-pad",
        "use_student": False,
        "feature_layers": None,
        "feature_norm": "backbone",
        "feature_l2_normalize": False,
        "spatial_layout": "per-image",
        "spatial_eval": False,
        "spatial_eval_sample_size": 100,
        "spatial_input": None,
        "label_csv": None,
        "coco_instances": None,
        "coco_keypoints": None,
        "coco_panoptic": None,
        "panoptic_masks_dir": None,
        "dump_model_structure": False,
        "batch_size": 2,
        "num_workers": 0,
        "device": "cpu",
        "seed": 42,
        "overwrite": False,
    }
    values.update(overrides)
    return values


def prepared(tmp_path: Path, **overrides):
    values = base_values(tmp_path, **overrides)
    explicit = frozenset(name for name, value in overrides.items() if value is not None)
    return values, prepare_spatial_request(values, explicit_options=explicit)


def test_mode_resolution_follows_the_request_only():
    assert spatial_mode({"output_kind": "embedding"}) == "embedding"
    assert spatial_mode({"output_kind": "spatial", "spatial_input": "p"}) == "read"
    assert spatial_mode({"output_kind": "spatial", "onnx_path": "g.onnx"}) == "onnx"
    assert spatial_mode({"output_kind": "spatial", "checkpoint": "c.pt"}) == "checkpoint"


@pytest.mark.parametrize(
    "name",
    [
        "token_mode",
        "topk_patches",
        "attention_pooling_type",
        "attention_pooling_epochs",
        "use_projector_output",
        "metrics_sample_size",
        "umap_n_neighbors",
        "umap_min_dist",
        "umap_metric",
        "visualize_class_number",
        "disable_umap",
    ],
)
def test_explicit_embedding_only_options_reject_spatial_even_at_default(name):
    values = base_values(Path("/tmp"), checkpoint="c.pt", input_images_dir="i")
    values[name] = values.get(name, True)
    with pytest.raises(ValueError, match="embedding-only options"):
        validate_extract_mode_options(values, explicit_options=frozenset({name}))
    # Omitted for this mode is inactive, not a conflict.
    assert validate_extract_mode_options(
        values, explicit_options=frozenset()
    ) == []


@pytest.mark.parametrize(
    "name",
    ["feature_layers", "feature_norm", "spatial_layout", "spatial_eval"],
)
def test_explicit_spatial_only_options_reject_embedding(name):
    values = base_values(Path("/tmp"), output_kind="embedding")
    with pytest.raises(ValueError, match="spatial-only"):
        validate_extract_mode_options(values, explicit_options=frozenset({name}))


def test_plain_spatial_input_without_spatial_output_kind_is_actionable():
    values = base_values(Path("/tmp"), output_kind="embedding", spatial_input="p")
    with pytest.raises(ValueError, match="requires --output-kind spatial"):
        validate_extract_mode_options(values, explicit_options=frozenset({"spatial_input"}))


def test_read_mode_rejects_extraction_overrides():
    values = base_values(Path("/tmp"), spatial_input="p")
    for name in (
        "checkpoint",
        "onnx_path",
        "model_name",
        "use_student",
        "feature_layers",
        "feature_norm",
        "spatial_layout",
        "extract_size",
        "eval_transform",
        "dump_model_structure",
        "spatial_eval_sample_size",
    ):
        with pytest.raises(ValueError, match="cannot be supplied|read mode"):
            validate_extract_mode_options(values, explicit_options=frozenset({name}))
    # Read mode needs no images and no model.
    notes = validate_extract_mode_options(
        values, explicit_options=frozenset({"input_images_dir"})
    )
    assert notes and "seed" in notes[0]


def test_spatial_extraction_requires_a_source_and_images():
    values = base_values(Path("/tmp"))
    with pytest.raises(ValueError, match="needs a feature source"):
        validate_extract_mode_options(values, explicit_options=frozenset())
    values = base_values(Path("/tmp"), checkpoint="c.pt")
    with pytest.raises(ValueError, match="--input-images-dir is required"):
        validate_extract_mode_options(values, explicit_options=frozenset())


def test_later_stage_requests_fail_before_any_output(tmp_path):
    values = base_values(
        tmp_path, checkpoint="c.pt", input_images_dir="i", spatial_eval=True
    )
    out = Path(values["out_dir"])
    with pytest.raises(SpatialRequestError, match="--spatial-eval is not enabled"):
        prepare_spatial_request(values, explicit_options=frozenset({"spatial_eval"}))
    assert not out.exists()

    values = base_values(
        tmp_path, onnx_path=str(tmp_path / "missing.onnx"), input_images_dir=str(tmp_path)
    )
    with pytest.raises(SpatialRequestError, match="spatial graph does not exist"):
        prepare_spatial_request(values, explicit_options=frozenset({"onnx_path"}))
    assert not out.exists()

    open_spatial_request = base_values(tmp_path, spatial_input=str(tmp_path / "missing"))
    with pytest.raises(SpatialRequestError, match="does not exist"):
        prepare_spatial_request(
            open_spatial_request, explicit_options=frozenset({"spatial_input"})
        )


def test_prepare_protects_selected_inputs_and_records_the_contract(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    values, request = prepared(
        tmp_path, checkpoint=str(checkpoint), input_images_dir=str(images)
    )
    try:
        assert request.mode == "checkpoint"
        assert request.image_size == 32
        assert request.eval_transform == "whole-specimen-pad"
        assert [sample.image_key for sample in request.samples] == [
            "a.png",
            "nested/b.png",
        ]
        protected = {snapshot.path.resolve() for snapshot in request.protected_files}
        assert checkpoint.resolve() in protected
        assert (images / "a.png").resolve() in protected
        assert request.contract is not None
        assert request.contract.layer_paths == ("blocks.11",)
        assert request.contract.model_identity["checkpoint_sha256"]
    finally:
        request.close()


def test_extraction_writes_both_layouts_and_reads_without_model_or_images(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    products = {}
    for layout in ("per-image", "sharded"):
        values, request = prepared(
            tmp_path,
            checkpoint=str(checkpoint),
            input_images_dir=str(images),
            spatial_layout=layout,
            out_dir=str(tmp_path / layout),
        )
        try:
            result = extract_spatial(request)
        finally:
            request.close()
        assert result["status"] == "completed"
        assert (Path(result["out_dir"]) / "spatial_manifest.json").is_file()
        assert (Path(result["out_dir"]) / "spatial_results.csv").is_file()
        assert (Path(result["out_dir"]) / "spatial_diagnostics.json").is_file()
        assert (Path(result["out_dir"]) / "run_provenance.json").is_file()
        products[layout] = result

    left = products["per-image"]
    right = products["sharded"]
    assert left["manifest"]["product_id"] == right["manifest"]["product_id"]
    assert [row["image_key"] for row in left["rows"]] == [
        row["image_key"] for row in right["rows"]
    ]
    assert left["rows"] == right["rows"]

    # Delete the model and the raw images: numeric read must still work.
    checkpoint.unlink()
    import shutil

    shutil.rmtree(images)

    values = base_values(tmp_path, spatial_input=left["out_dir"], out_dir=str(tmp_path / "read"))
    read_request = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input"})
    )
    try:
        result = reevaluate_spatial(read_request)
    finally:
        read_request.close()
    assert result["status"] == "completed"
    assert result["rows"] == left["rows"]
    envelope = json.loads(
        (Path(result["out_dir"]) / "run_provenance.json").read_text(encoding="utf-8")
    )
    assert envelope["status"] == "completed"
    assert envelope["source_product"]["product_id"] == left["manifest"]["product_id"]
    paths = {item["path"] for item in envelope["artifacts"]}
    assert "spatial_manifest.json" not in paths  # read reports are a separate product
    assert "spatial_results.csv" in paths


def test_read_mode_reuses_the_stored_csv_subset(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    _, request = prepared(
        tmp_path, checkpoint=str(checkpoint), input_images_dir=str(images)
    )
    try:
        result = extract_spatial(request)
    finally:
        request.close()
    product = Path(result["out_dir"])

    subset = tmp_path / "subset.csv"
    subset.write_text("image\nnested/b.png\n", encoding="utf-8")
    values = base_values(
        tmp_path,
        spatial_input=str(product),
        label_csv=str(subset),
        out_dir=str(tmp_path / "subset-out"),
    )
    read_request = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "label_csv"})
    )
    try:
        subset_result = reevaluate_spatial(read_request)
    finally:
        read_request.close()
    assert {row["image_key"] for row in subset_result["rows"]} == {"nested/b.png"}
    assert len(subset_result["rows"]) == 1


def test_read_mode_rejects_outputs_that_touch_the_source_product(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    _, request = prepared(
        tmp_path, checkpoint=str(checkpoint), input_images_dir=str(images)
    )
    try:
        result = extract_spatial(request)
    finally:
        request.close()
    product = Path(result["out_dir"])

    for target in (product, product / "inner", product.parent):
        values = base_values(
            tmp_path, spatial_input=str(product), out_dir=str(target / "out" if target != product else product)
        )
        with pytest.raises(SpatialRequestError, match="source product"):
            prepare_spatial_request(
                values, explicit_options=frozenset({"spatial_input", "out_dir"})
            )


def test_dump_model_structure_writes_layers_and_continues(tmp_path):
    images = write_images(tmp_path / "images", keys=("a.png",))
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    values, request = prepared(
        tmp_path,
        checkpoint=str(checkpoint),
        input_images_dir=str(images),
        dump_model_structure=True,
    )
    try:
        result = extract_spatial(request)
    finally:
        request.close()
    report = Path(result["out_dir"]) / "model_layers.txt"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "blocks.0" in text and "blocks.11" in text
    assert "spatial-eligible" in text
    assert result["rows"]  # extraction still completed


def test_extraction_rejects_a_bad_selection_before_writing(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    values = base_values(
        tmp_path, checkpoint=str(checkpoint), input_images_dir=str(images)
    )
    with pytest.raises(Exception, match="resolved no images"):
        prepare_spatial_request(
            values, explicit_options=frozenset({"checkpoint", "input_images_dir"})
        )
    assert not (Path(values["out_dir"]) / "spatial_manifest.json").exists()
    assert not (Path(values["out_dir"]) / "run_provenance.json").exists()


def test_unsupported_feature_layer_fails_before_output(tmp_path):
    images = write_images(tmp_path / "images", keys=("a.png",))
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    values = base_values(
        tmp_path,
        checkpoint=str(checkpoint),
        input_images_dir=str(images),
        feature_layers="blocks.99",
    )
    with pytest.raises(ValueError, match="unknown feature layer"):
        prepare_spatial_request(
            values, explicit_options=frozenset({"checkpoint", "feature_layers"})
        )
    assert not Path(values["out_dir"]).exists()


def test_sheared_input_is_not_silently_accepted(tmp_path):
    from otuformer.vision.geometry import GeometryOp, apply_geometry

    image = Image.new("RGB", (8, 8))
    with pytest.raises(ValueError, match="shear is not part of spatial v1"):
        apply_geometry(
            image,
            [
                GeometryOp(
                    "affine",
                    {
                        "from_wh": [8, 8],
                        "center": [4.0, 4.0],
                        "angle": 0.0,
                        "translate": [0.0, 0.0],
                        "scale": 1.0,
                        "shear": [0.1, 0.0],
                        "interpolation": "bilinear",
                        "fill": [0, 0, 0],
                        "canvas_wh": [8, 8],
                    },
                )
            ],
            track=True,
        )
