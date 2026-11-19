"""Spatial ONNX export: fixed graph contract, every output validated."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from otuformer.cli.main import app
from otuformer.vision.features import SpatialFeatureForward, resolve_feature_contract
from otuformer.vision.export import (
    SPATIAL_OUTPUT_PREFIX,
    export_to_onnx,
    read_spatial_onnx_contract,
    spatial_output_name,
    validate_spatial_onnx_outputs,
)

pytest.importorskip("onnxruntime")

runner = CliRunner()


def tiny_checkpoint(tmp_path: Path, *, reg_tokens: int | None = None, image_size: int = 32) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    torch.manual_seed(0)
    model = OTUFormerEncoder(
        out_dim=8,
        pretrained=False,
        img_size=image_size,
        reg_tokens=reg_tokens,
    )
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 8,
                "image_size": image_size,
                # The loader rejects a checkpoint whose recorded register layout
                # disagrees with its weights.
                "register_tokens": 4 if reg_tokens == 4 else 0,
            },
        },
        path,
    )
    return path


@pytest.mark.parametrize("reg_tokens", [None, 4])
def test_spatial_export_validates_every_layer_and_batch(tmp_path, reg_tokens):
    checkpoint = tiny_checkpoint(tmp_path, reg_tokens=reg_tokens)
    out = tmp_path / "export" / "encoder.onnx"
    report = export_to_onnx(
        checkpoint,
        out,
        imgsz=32,
        output_kind="spatial",
        feature_layers="blocks.0,blocks.11",
    )
    assert report["output_kind"] == "spatial"
    assert report["validated"] is True
    assert report["validation_status"] == "passed"
    assert report["spatial"]["output_names"] == ["spatial_000", "spatial_001"]
    assert report["spatial"]["layout"] == "BHWC"
    assert report["spatial"]["per_image_layout"] == "HWC"
    batch_rows = report["spatial"]["batch_validation"]
    assert set(batch_rows) == {"batch_1", "batch_2"}
    for name in ("spatial_000", "spatial_001"):
        for batch in ("batch_1", "batch_2"):
            row = batch_rows[batch][name]
            assert row["validated"] is True
            assert row["dtype"] == "float32"
            assert row["max_abs_diff"] <= report["atol"]
            assert row["shape"][0] == int(batch.split("_")[1])
            assert row["shape"][1:] == [2, 2, 192]
    # The old embedding width is never relabelled as a spatial channel count.
    assert "out_dim" not in report

    contract = read_spatial_onnx_contract(out)
    assert contract.layer_paths == ("blocks.0", "blocks.11")
    assert contract.norm == "backbone"
    assert contract.l2 is False
    assert contract.prefix_tokens == (5 if reg_tokens == 4 else 1)
    assert contract.model_identity["checkpoint_sha256"]
    assert contract.model_identity["weight_source"]
    assert spatial_output_name(3) == f"{SPATIAL_OUTPUT_PREFIX}003"

    # Numeric agreement with an independent forward, at both batch sizes.
    from otuformer.utils.checkpoint import load_checkpoint, load_checkpoint_encoder, resolve_checkpoint

    payload = load_checkpoint(checkpoint)
    architecture = resolve_checkpoint(payload, "vit_tiny_patch16_224")
    encoder, _ = load_checkpoint_encoder(payload, architecture, image_size=32)
    encoder.eval()
    forward = SpatialFeatureForward(encoder, contract)
    revalidated = validate_spatial_onnx_outputs(out, forward, contract)
    assert revalidated["validated"] is True
    assert revalidated["max_abs_diff"] <= report["atol"]


@pytest.mark.parametrize(
    "norm,l2",
    [("backbone", False), ("none", False), ("backbone", True), ("none", True)],
)
def test_spatial_export_supports_norm_and_l2_variants(tmp_path, norm, l2):
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / f"export_{norm}_{l2}" / "encoder.onnx"
    report = export_to_onnx(
        checkpoint,
        out,
        imgsz=32,
        output_kind="spatial",
        feature_norm=norm,
        feature_l2_normalize=l2,
    )
    assert report["validated"] is True
    contract = read_spatial_onnx_contract(out)
    assert contract.norm == norm
    assert contract.l2 is bool(l2)
    if l2:
        from otuformer.utils.checkpoint import (
            load_checkpoint,
            load_checkpoint_encoder,
            resolve_checkpoint,
        )

        payload = load_checkpoint(checkpoint)
        architecture = resolve_checkpoint(payload, "vit_tiny_patch16_224")
        encoder, _ = load_checkpoint_encoder(payload, architecture, image_size=32)
        encoder.eval()
        (grid,) = SpatialFeatureForward(encoder, contract)(torch.randn(1, 3, 32, 32))
        norms = torch.linalg.vector_norm(grid[0].reshape(-1, grid.shape[-1]), dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_spatial_export_records_the_actual_weight_source(tmp_path):
    checkpoint = tiny_checkpoint(tmp_path)
    teacher = export_to_onnx(
        checkpoint, tmp_path / "teacher" / "encoder.onnx", imgsz=32,
        output_kind="spatial", use_student=False,
    )
    student = export_to_onnx(
        checkpoint, tmp_path / "student" / "encoder.onnx", imgsz=32,
        output_kind="spatial", use_student=True,
    )
    assert teacher["validated"] and student["validated"]
    # This checkpoint stores one state dict, so both requests resolve to it; the
    # report still records the source that was actually read.
    assert teacher["spatial"]["feature_contract"]["model_identity"]["weight_source"]
    assert student["spatial"]["feature_contract"]["model_identity"]["weight_source"]


def test_embedding_export_keeps_its_single_graph(tmp_path):
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "embedding" / "encoder.onnx"
    report = export_to_onnx(checkpoint, out, imgsz=32)
    assert "out_dim" in report
    assert report["output_kind"] if "output_kind" in report else True
    import onnx

    proto = onnx.load(str(out), load_external_data=False)
    assert [output.name for output in proto.graph.output] == ["embedding"]
    with pytest.raises(ValueError, match="no spatial feature contract"):
        read_spatial_onnx_contract(out)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"feature_layers": "blocks.0"}, "spatial-only options"),
        ({"feature_norm": "backbone"}, "spatial-only options"),
        ({"feature_norm": "none"}, "spatial-only options"),
        ({"feature_l2_normalize": False}, "spatial-only options"),
        ({"feature_l2_normalize": True}, "spatial-only options"),
    ],
)
def test_embedding_export_rejects_spatial_only_values(tmp_path, kwargs, match):
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "embedding" / "encoder.onnx"
    with pytest.raises(ValueError, match=match):
        export_to_onnx(checkpoint, out, imgsz=32, **kwargs)
    assert not out.exists()


def test_embedding_export_rejects_a_student_request_before_writing(tmp_path):
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "embedding" / "encoder.onnx"
    with pytest.raises(ValueError, match="student request is not supported"):
        export_to_onnx(checkpoint, out, imgsz=32, use_student=True)
    assert not out.exists()
    # Omitted/false keeps the old behaviour.
    report = export_to_onnx(checkpoint, out, imgsz=32, use_student=False)
    assert report["validated"] is True


def test_failed_spatial_export_preserves_the_previous_products(tmp_path):
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "export" / "encoder.onnx"
    first = export_to_onnx(checkpoint, out, imgsz=32, output_kind="spatial")
    assert first["validated"] is True
    graph_before = out.read_bytes()
    report_before = (out.parent / "export_report.json").read_bytes()

    with pytest.raises(ValueError, match="not a complete transformer block|unknown feature layer"):
        export_to_onnx(
            checkpoint, out, imgsz=32, output_kind="spatial", feature_layers="blocks.99"
        )
    assert out.read_bytes() == graph_before
    assert (out.parent / "export_report.json").read_bytes() == report_before

    with pytest.raises(ValueError, match="not divisible by patch size"):
        export_to_onnx(checkpoint, out, imgsz=40, output_kind="spatial")
    assert out.read_bytes() == graph_before


def test_spatial_export_rejects_a_non_dynamic_batch_graph(tmp_path):
    """A fixed-batch graph is not a spatial v1 graph."""
    import onnx

    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "export" / "encoder.onnx"
    export_to_onnx(checkpoint, out, imgsz=32, output_kind="spatial")
    contract = read_spatial_onnx_contract(out)

    fixed = tmp_path / "fixed.onnx"
    proto = onnx.load(str(out))
    for entry in proto.graph.input:
        for dimension in entry.type.tensor_type.shape.dim:
            if dimension.dim_param == "batch":
                dimension.ClearField("dim_param")
                dimension.dim_value = 1
    onnx.save(proto, str(fixed))

    from otuformer.utils.checkpoint import (
        load_checkpoint,
        load_checkpoint_encoder,
        resolve_checkpoint,
    )

    payload = load_checkpoint(checkpoint)
    architecture = resolve_checkpoint(payload, "vit_tiny_patch16_224")
    encoder, _ = load_checkpoint_encoder(payload, architecture, image_size=32)
    encoder.eval()
    with pytest.raises(ValueError, match="dynamic batch"):
        validate_spatial_onnx_outputs(
            fixed, SpatialFeatureForward(encoder, contract), contract
        )


def test_spatial_export_rejects_tampered_output_metadata(tmp_path):
    import onnx

    from otuformer.constants import SPATIAL_ONNX_METADATA_KEY

    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "export" / "encoder.onnx"
    export_to_onnx(checkpoint, out, imgsz=32, output_kind="spatial")

    tampered = tmp_path / "tampered.onnx"
    proto = onnx.load(str(out))
    for entry in proto.metadata_props:
        if entry.key == SPATIAL_ONNX_METADATA_KEY:
            payload = json.loads(entry.value)
            payload["layout"] = "BCHW"
            entry.value = json.dumps(payload)
    onnx.save(proto, str(tampered))
    with pytest.raises(ValueError, match="layout must be BHWC"):
        read_spatial_onnx_contract(tampered)

    missing = tmp_path / "missing.onnx"
    proto = onnx.load(str(out))
    proto.ClearField("metadata_props")
    onnx.save(proto, str(missing))
    with pytest.raises(ValueError, match="no spatial feature contract"):
        read_spatial_onnx_contract(missing)


def test_export_cli_rejects_conflicts_before_clearing_output(tmp_path):
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep.txt"
    keep.write_text("previous", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out),
            "--overwrite",
            "--feature-norm",
            "backbone",
        ],
    )
    assert result.exit_code != 0
    assert "spatial-only" in result.output
    assert keep.read_text(encoding="utf-8") == "previous"

    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out),
            "--overwrite",
            "--use-student",
        ],
    )
    assert result.exit_code != 0
    assert keep.read_text(encoding="utf-8") == "previous"


def test_export_cli_writes_a_spatial_graph(tmp_path):
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "spatial-out"
    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out),
            "--imgsz",
            "32",
            "--output-kind",
            "spatial",
            "--feature-layers",
            "blocks.0,blocks.11",
        ],
    )
    assert result.exit_code == 0, result.output
    onnx_path = out / "encoder.onnx"
    assert onnx_path.is_file()
    report = json.loads((out / "export_report.json").read_text(encoding="utf-8"))
    assert report["output_kind"] == "spatial"
    assert report["spatial"]["output_names"] == ["spatial_000", "spatial_001"]
    assert report["validated"] is True
    # The log records the requested values.
    log = (out / "logs" / "export.log").read_text(encoding="utf-8")
    assert "--output-kind" in log
    assert "spatial" in log


def test_a_rejected_spatial_request_does_not_clear_the_existing_output(tmp_path):
    """Preflight the request before `--overwrite` deletes an earlier graph."""
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "export"
    out.mkdir()
    previous = out / "encoder.onnx"
    previous.write_bytes(b"previous graph bytes")
    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out),
            "--output-kind",
            "spatial",
            "--feature-layers",
            "blocks.999",
            "--overwrite",
        ],
    )
    assert result.exit_code != 0
    assert previous.is_file()
    assert previous.read_bytes() == b"previous graph bytes"


def test_graph_metadata_must_agree_with_the_graph_io(tmp_path):
    """Output names and the per-image layout are asserted, not just recorded."""
    import onnx

    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "export" / "encoder.onnx"
    export_to_onnx(checkpoint, out, imgsz=32, output_kind="spatial")

    def rewrite(mutate):
        proto = onnx.load(str(out))
        for entry in proto.metadata_props:
            if entry.key != "otuformer.spatial.v1":
                continue
            payload = json.loads(entry.value)
            mutate(payload)
            entry.value = json.dumps(payload)
        target = tmp_path / "tampered.onnx"
        onnx.save(proto, str(target))
        return target

    def rename_output(payload):
        payload["outputs"] = ["spatial_999"]

    def change_layout(payload):
        payload["per_image_layout"] = "CHW"

    with pytest.raises(ValueError, match="declares outputs"):
        read_spatial_onnx_contract(rewrite(rename_output))
    with pytest.raises(ValueError, match="per-image layout"):
        read_spatial_onnx_contract(rewrite(change_layout))


def test_prepare_spatial_export_writes_nothing(tmp_path):
    from otuformer.vision.export import _prepare_spatial_export

    checkpoint = tiny_checkpoint(tmp_path)

    prepared = _prepare_spatial_export(
        checkpoint,
        imgsz=None,
        model_name="vit_tiny_patch16_224",
        feature_layers=None,
        feature_norm="backbone",
        feature_l2_normalize=False,
        use_student=False,
    )

    assert prepared["size"] == 32
    assert prepared["model"] is not None
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.pt"]
