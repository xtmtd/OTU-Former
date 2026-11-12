"""Spatial ONNX extraction through the same records, storage and diagnostics.

The graph and a checkpoint produce the same logical product semantics; only the
physical bytes and the backend identity differ.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from otuformer.cli.main import app
from otuformer.spatial.diagnostics import diagnose_views
from otuformer.spatial.extractor import (
    SpatialRequestError,
    extract_spatial,
    prepare_spatial_request,
)
from otuformer.spatial.storage import open_spatial_product
from otuformer.vision.export import export_to_onnx

pytest.importorskip("onnxruntime")
runner = CliRunner()


def write_images(root: Path, keys=("a.png", "nested/b.png")) -> Path:
    for key in keys:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 40), (20, 60, 90)).save(path)
    return root


def write_checkpoint(path: Path, image_size: int = 32) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    torch.manual_seed(0)
    model = OTUFormerEncoder(out_dim=8, pretrained=False, img_size=image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 8,
                "image_size": image_size,
                "register_tokens": 0,
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


@pytest.fixture
def graph(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    out = tmp_path / "export" / "encoder.onnx"
    report = export_to_onnx(
        checkpoint, out, imgsz=32, output_kind="spatial",
        feature_layers="blocks.0,blocks.11",
    )
    assert report["validated"] is True
    return checkpoint, out


def run_backend(tmp_path, *, values, explicit):
    request = prepare_spatial_request(values, explicit_options=frozenset(explicit))
    try:
        return extract_spatial(request)
    finally:
        request.close()


@pytest.mark.parametrize("layout", ["per-image", "sharded"])
def test_onnx_and_checkpoint_share_spatial_product_semantics(tmp_path, graph, layout):
    checkpoint, onnx_path = graph
    images = write_images(tmp_path / "images")

    checkpoint_result = run_backend(
        tmp_path,
        values=base_values(
            tmp_path,
            checkpoint=str(checkpoint),
            input_images_dir=str(images),
            feature_layers="blocks.0,blocks.11",
            spatial_layout=layout,
            out_dir=str(tmp_path / f"ckpt-{layout}"),
        ),
        explicit=["checkpoint", "input_images_dir", "feature_layers"],
    )
    onnx_result = run_backend(
        tmp_path,
        values=base_values(
            tmp_path,
            onnx_path=str(onnx_path),
            input_images_dir=str(images),
            spatial_layout=layout,
            out_dir=str(tmp_path / f"onnx-{layout}"),
        ),
        explicit=["onnx_path", "input_images_dir"],
    )

    assert checkpoint_result["status"] == onnx_result["status"] == "completed"
    # The backends differ, so the physical identity differs; the logical contract
    # (layers, norm, L2, size) and every stored array agree.
    assert checkpoint_result["envelope"]["effective"]["feature_contract"] == (
        onnx_result["envelope"]["effective"]["feature_contract"]
    )
    assert onnx_result["envelope"]["effective"]["backend"] == "onnxruntime-cpu"

    left = open_spatial_product(Path(checkpoint_result["out_dir"]))
    right = open_spatial_product(Path(onnx_result["out_dir"]))
    try:
        assert left.image_keys == right.image_keys
        left_views = list(left.iter_views())
        right_views = list(right.iter_views())
        assert [view.image_key for view in left_views] == [
            view.image_key for view in right_views
        ]
        for one, two in zip(left_views, right_views):
            assert one.view_id == two.view_id and one.role == two.role
            for layer in one.features:
                assert one.features[layer].shape == two.features[layer].shape
                assert one.features[layer].dtype == two.features[layer].dtype
                np.testing.assert_allclose(
                    one.features[layer], two.features[layer], atol=1e-4, rtol=1e-3
                )
            # Geometry and support come from the shared preprocessor, not the graph.
            np.testing.assert_array_equal(
                one.pixel_geometric_coverage, two.pixel_geometric_coverage
            )
            np.testing.assert_array_equal(
                one.pixel_sampling_valid, two.pixel_sampling_valid
            )
            np.testing.assert_allclose(
                one.trace.source_to_view, two.trace.source_to_view, atol=1e-12
            )
            np.testing.assert_array_equal(
                one.geometric_coverage, two.geometric_coverage
            )
        # Diagnostics are numerically equal but the artifact identity is not.
        left_rows, _ = diagnose_views(left.iter_views(), left.contract)
        right_rows, _ = diagnose_views(right.iter_views(), right.contract)
        assert [row["image_key"] for row in left_rows] == [
            row["image_key"] for row in right_rows
        ]
        for one, two in zip(left_rows, right_rows):
            assert one["shape"] == two["shape"]
            assert one["dtype"] == two["dtype"]
            assert one["coverage_min"] == pytest.approx(two["coverage_min"])
            assert one["norm_mean"] == pytest.approx(two["norm_mean"], rel=1e-4)
    finally:
        left.close()
        right.close()

    assert checkpoint_result["manifest"]["product_id"] != (
        onnx_result["manifest"]["product_id"]
    )


def test_onnx_normalization_and_l2_come_from_the_graph(tmp_path, graph):
    checkpoint, _ = graph
    images = write_images(tmp_path / "images", keys=("a.png",))
    unit_graph = tmp_path / "unit" / "encoder.onnx"
    export_to_onnx(
        checkpoint, unit_graph, imgsz=32, output_kind="spatial",
        feature_norm="none", feature_l2_normalize=True,
    )
    result = run_backend(
        tmp_path,
        values=base_values(
            tmp_path,
            onnx_path=str(unit_graph),
            input_images_dir=str(images),
            out_dir=str(tmp_path / "unit-out"),
        ),
        explicit=["onnx_path", "input_images_dir"],
    )
    contract = result["envelope"]["effective"]["feature_contract"]
    assert contract["norm"] == "none"
    assert contract["l2"] is True
    reader = open_spatial_product(Path(result["out_dir"]))
    try:
        (view,) = list(reader.iter_views())
        grid = view.features["blocks.11"]
        norms = np.linalg.norm(grid.reshape(-1, grid.shape[-1]), axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    finally:
        reader.close()


@pytest.mark.parametrize(
    "overrides,explicit,match",
    [
        ({"feature_norm": "none"}, ["feature_norm"], "does not match the graph"),
        ({"feature_l2_normalize": True}, ["feature_l2_normalize"], "does not match"),
        ({"feature_layers": "blocks.5"}, ["feature_layers"], "does not match the graph"),
        ({"extract_size": "40"}, ["extract_size"], "does not match the graph"),
    ],
)
def test_explicit_values_must_match_the_graph(tmp_path, graph, overrides, explicit, match):
    _, onnx_path = graph
    images = write_images(tmp_path / "images", keys=("a.png",))
    values = base_values(
        tmp_path,
        onnx_path=str(onnx_path),
        input_images_dir=str(images),
        **overrides,
    )
    with pytest.raises(SpatialRequestError, match=match):
        prepare_spatial_request(
            values, explicit_options=frozenset({"onnx_path", "input_images_dir", *explicit})
        )
    assert not Path(values["out_dir"]).exists()


def test_an_explicit_exact_match_is_accepted_and_recorded(tmp_path, graph):
    _, onnx_path = graph
    images = write_images(tmp_path / "images", keys=("a.png",))
    result = run_backend(
        tmp_path,
        values=base_values(
            tmp_path,
            onnx_path=str(onnx_path),
            input_images_dir=str(images),
            feature_layers="blocks.0,blocks.11",
            feature_norm="backbone",
            feature_l2_normalize=False,
            extract_size="32",
            out_dir=str(tmp_path / "exact"),
        ),
        explicit=[
            "onnx_path", "input_images_dir", "feature_layers", "feature_norm",
            "feature_l2_normalize", "extract_size",
        ],
    )
    assert result["status"] == "completed"


def test_graph_and_sidecar_are_protected_inputs(tmp_path, graph):
    _, onnx_path = graph
    images = write_images(tmp_path / "images", keys=("a.png",))
    request = prepare_spatial_request(
        base_values(
            tmp_path,
            onnx_path=str(onnx_path),
            input_images_dir=str(images),
        ),
        explicit_options=frozenset({"onnx_path", "input_images_dir"}),
    )
    try:
        protected = {snapshot.path.resolve() for snapshot in request.protected_files}
        assert onnx_path.resolve() in protected
        assert (images / "a.png").resolve() in protected
    finally:
        request.close()


def test_an_embedding_graph_is_not_accepted_as_spatial(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    embedding_graph = tmp_path / "embedding" / "encoder.onnx"
    export_to_onnx(checkpoint, embedding_graph, imgsz=32)
    images = write_images(tmp_path / "images", keys=("a.png",))
    values = base_values(
        tmp_path, onnx_path=str(embedding_graph), input_images_dir=str(images)
    )
    with pytest.raises(ValueError, match="no spatial feature contract"):
        prepare_spatial_request(
            values, explicit_options=frozenset({"onnx_path", "input_images_dir"})
        )
    assert not Path(values["out_dir"]).exists()


def test_cli_runs_spatial_onnx_extraction_and_reads_it_back(tmp_path, graph):
    checkpoint, onnx_path = graph
    images = write_images(tmp_path / "images")
    out = tmp_path / "onnx-out"
    result = runner.invoke(
        app,
        [
            "extract",
            "--output-kind",
            "spatial",
            "--onnx-path",
            str(onnx_path),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    envelope = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    assert envelope["status"] == "completed"
    assert envelope["effective"]["backend"] == "onnxruntime-cpu"

    # The checkpoint and the raw images are gone: the product still reads.
    checkpoint.unlink()
    shutil.rmtree(images)
    read_out = tmp_path / "read-out"
    read = runner.invoke(
        app,
        [
            "extract",
            "--output-kind",
            "spatial",
            "--spatial-input",
            str(out),
            "--out-dir",
            str(read_out),
        ],
    )
    assert read.exit_code == 0, read.output
    assert (read_out / "spatial_results.csv").is_file()


def test_cli_rejects_graph_overrides_before_writing(tmp_path, graph):
    _, onnx_path = graph
    images = write_images(tmp_path / "images", keys=("a.png",))
    out = tmp_path / "out"
    for extra, match in (
        (["--model-name", "vit_small_patch16_224"], "cannot relabel a graph"),
        (["--use-student"], "cannot choose weights in a graph"),
        (["--dump-model-structure"], "already declares its exported layers"),
    ):
        result = runner.invoke(
            app,
            [
                "extract",
                "--output-kind",
                "spatial",
                "--onnx-path",
                str(onnx_path),
                "--input-images-dir",
                str(images),
                "--out-dir",
                str(out),
                *extra,
            ],
        )
        assert result.exit_code != 0, extra
        assert match in result.output.replace("\n", " "), extra
        assert not out.exists()
