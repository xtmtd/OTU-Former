import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from otuformer.data.provenance import build_run_record
from otuformer.data.reader import read_dataset
from otuformer.embedding.record_dataset import RecordImageDataset
from otuformer.embedding.extractor import (
    _extract_embeddings_result,
    _onnx_external_files,
    _require_unchanged,
    extract_embeddings,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def write_png(path: Path, size=(32, 32), color=(10, 20, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def write_csv(path: Path, header, rows) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def tiny_checkpoint(tmp_path: Path, out_dim: int = 2, size: int = 32) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=out_dim,
        pretrained=False,
        img_size=size,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim},
        "args": {"global_crop_size": size},
    }
    path = tmp_path / "tiny.ckpt"
    torch.save(checkpoint, path)
    return path


def images_and_manifest(tmp_path, rows=4, labels=None, header=None):
    images = tmp_path / "images"
    for index in range(rows):
        write_png(images / "plate1" / f"{index:03d}.png", color=(index * 10, 0, 0))
    if header is None:
        header, body = ["image"], [[f"plate1/{index:03d}.png"] for index in range(rows)]
    else:
        body = [
            [f"plate1/{index:03d}.png"]
            + [(labels or ["NA", "001", "", "species"][index % 4])] * (len(header) - 1)
            for index in range(rows)
        ]
    selector = write_csv(tmp_path / "sel.csv", header, body)
    return images, selector


def bundle_for(images, selector, **kwargs):
    bundle = read_dataset(images, manifest=selector, **kwargs)
    bundle.raise_for_errors()
    return bundle


def test_records_extraction_never_reparses_csv(tmp_path, monkeypatch):
    import otuformer.embedding.extractor as extractor

    images, selector = images_and_manifest(tmp_path)
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the legacy CSV/directory reader was used")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(extractor, "CSVImageDataset", forbidden)
    monkeypatch.setattr(extractor, "LabelCSVImageDataset", forbidden)
    monkeypatch.setattr(extractor, "ImageFolderDataset", forbidden)
    monkeypatch.setattr(extractor, "detect_batch_mode", forbidden)
    monkeypatch.setattr(extractor, "_build_recursive_index", forbidden)

    frame = extract_embeddings(
        checkpoint, images, device="cpu", records=bundle, num_workers=0
    )

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [record.output_id for record in bundle.records]
    assert frame.columns[:2].tolist() == ["id", "sample"]
    assert all(name.startswith("dim_") for name in frame.columns[2:])
    assert "label" not in frame.columns
    assert frame["sample"].tolist() == ["plate1"] * 4

    result = _extract_embeddings_result(
        checkpoint, images, device="cpu", records=bundle, num_workers=0
    )
    pd.testing.assert_frame_equal(result.frame, frame)
    assert result.effective["data_path"] == "records"
    assert result.effective["selection"]["records"] == 4
    # The checkpoint and the selector bytes are recorded as used inputs.
    used = {snapshot.path.name for snapshot in result.used_files}
    assert {"tiny.ckpt", "sel.csv"} <= used
    assert any(
        snapshot.sha256 == hashlib.sha256(selector.read_bytes()).hexdigest()
        for snapshot in result.used_files
    )


def test_explicit_sample_column_wins_over_inference(tmp_path):
    images = tmp_path / "images"
    write_png(images / "plate1" / "001.png")
    selector = write_csv(
        tmp_path / "sel.csv", ["image", "sample"], [["plate1/001.png", "manual"]]
    )
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    frame = extract_embeddings(
        checkpoint, images, device="cpu", records=bundle, num_workers=0
    )

    assert frame["sample"].tolist() == ["manual"]
    assert bundle.records[0].sample_origin == "explicit"


def test_record_attention_query_label_contract(tmp_path, monkeypatch):
    images = tmp_path / "images"
    for index in range(4):
        write_png(images / "p" / f"{index}.png", color=(index * 5, 0, 0))
    selector = write_csv(
        tmp_path / "labelled.csv",
        ["image", "label"],
        [
            ["p/0.png", "NA"],
            ["p/1.png", "001"],
            ["p/2.png", "NA"],
            ["p/3.png", "001"],
        ],
    )
    labelled = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path, out_dim=2)

    pool_path = tmp_path / "pool" / "tiny.lightweight_pooling.pth"
    frame = extract_embeddings(
        checkpoint,
        images,
        device="cpu",
        records=labelled,
        token_mode="attention-pool",
        attention_pooling_epochs=1,
        batch_size=2,
        num_workers=0,
        attention_pooling_checkpoint_path=pool_path,
    )
    assert len(frame) == 4
    assert list((tmp_path / "pool").glob("*.pth"))

    # A cached pool lets an image-only manifest run without training.
    image_only = write_csv(
        tmp_path / "image_only.csv", ["image"], [[f"p/{index}.png"] for index in range(4)]
    )
    unlabelled = bundle_for(images, image_only)
    cached = extract_embeddings(
        checkpoint,
        images,
        device="cpu",
        records=unlabelled,
        token_mode="attention-pool",
        batch_size=2,
        num_workers=0,
        attention_pooling_checkpoint_path=pool_path,
    )
    assert len(cached) == 4

    # Without a cached pool, an image-only manifest cannot silently train a probe.
    fresh_pool = tmp_path / "fresh" / "tiny.lightweight_pooling.pth"
    with pytest.raises(ValueError, match="no label"):
        extract_embeddings(
            checkpoint,
            images,
            device="cpu",
            records=unlabelled,
            token_mode="attention-pool",
            attention_pooling_epochs=1,
            batch_size=2,
            num_workers=0,
            attention_pooling_checkpoint_path=fresh_pool,
        )


def test_records_root_and_input_drift(tmp_path):
    images, selector = images_and_manifest(tmp_path, rows=2)
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    elsewhere = tmp_path / "elsewhere"
    write_png(elsewhere / "plate1" / "000.png")
    with pytest.raises(ValueError, match="records were built from"):
        extract_embeddings(
            checkpoint, elsewhere, device="cpu", records=bundle, num_workers=0
        )

    # Replacing a selected image by different bytes invalidates the snapshot.
    target = images / "plate1" / "000.png"
    write_png(target, color=(200, 0, 0))
    with pytest.raises(RuntimeError, match="input changed after validation"):
        extract_embeddings(
            checkpoint, images, device="cpu", records=bundle, num_workers=0
        )


def test_records_drift_via_escaping_symlink(tmp_path):
    images, selector = images_and_manifest(tmp_path, rows=1)
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    outside = write_png(tmp_path / "outside.png", color=(7, 7, 7))
    target = images / "plate1" / "000.png"
    target.unlink()
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    # Containment is re-resolved before the bytes are read, and the symlink points
    # at an identical copy, so only the boundary check can catch this.
    with pytest.raises(ValueError, match="resolves outside the root"):
        extract_embeddings(
            checkpoint, images, device="cpu", records=bundle, num_workers=0
        )


def test_execution_metadata_matches_consumed_state(tmp_path):
    images, selector = images_and_manifest(tmp_path, rows=4)
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    result = _extract_embeddings_result(
        checkpoint, images, device="cpu", records=bundle, num_workers=0
    )
    effective = result.effective

    assert effective["device"] == "cpu"
    assert effective["requested"]["extract_size"] is None
    assert effective["input_size"] == 32
    assert effective["checkpoint"]["model_name"] == "vit_tiny_patch16_224"
    assert "register_tokens" in effective["checkpoint"]
    assert effective["input_normalization"]["mean"] == IMAGENET_MEAN
    assert effective["input_normalization"]["std"] == IMAGENET_STD
    assert effective["feature_output"] == "backbone_cls_token"
    assert effective["post_feature_ops"] == ["cls_token_index_0"]
    # timm's forward_features applies the final norm; the hook observes that
    # rather than assuming extraction added nothing.
    assert effective["backbone_output_norm"]["status"] == "applied"
    assert "backbone.norm" in effective["backbone_output_norm"]["layers"][0]
    assert effective["vector_normalization"]["status"] == "none"
    # The raw backbone CLS width, not the projector's out_dim.
    assert effective["feature_chain"]["output_dim"] == 192

    projector = _extract_embeddings_result(
        checkpoint,
        images,
        device="cpu",
        records=bundle,
        use_projector_output=True,
        num_workers=0,
    )
    assert projector.effective["feature_output"] == "projector_output"
    # ProjectionHead ends in F.normalize, so the executed chain says so.
    assert projector.effective["post_feature_ops"] == ["projector", "l2_normalize"]
    assert projector.effective["vector_normalization"]["status"] == "applied"
    assert projector.effective["feature_chain"]["input_dim"] == 192

    patch_topk = _extract_embeddings_result(
        checkpoint,
        images,
        device="cpu",
        records=bundle,
        token_mode="patch-topk",
        topk_patches=1,
        num_workers=0,
    )
    policy = patch_topk.effective["patch_topk_pca"]
    assert policy["mode"] == "fit_transform"
    assert policy["fit_scope"] == "all extracted rows in this run"
    assert policy["solver"] == "full"
    assert policy["seed"] == 42
    assert policy["output_dim"] == 3
    assert patch_topk.frame.filter(like="dim_").shape[1] == 3


def test_records_match_legacy_csv_vectors(tmp_path):
    images, selector = images_and_manifest(tmp_path, rows=3)
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    legacy = extract_embeddings(
        checkpoint, images, device="cpu", extract_csv=selector, num_workers=0
    )
    modern = extract_embeddings(
        checkpoint, images, device="cpu", records=bundle, num_workers=0
    )

    assert legacy["id"].tolist() == modern["id"].tolist()
    legacy_dims = legacy.filter(like="dim_").to_numpy()
    modern_dims = modern.filter(like="dim_").to_numpy()
    assert np.allclose(legacy_dims, modern_dims, atol=1e-6)


def test_records_deserialize_verified_checkpoint_bytes(tmp_path, monkeypatch):
    import otuformer.embedding.extractor as extractor

    images, selector = images_and_manifest(tmp_path, rows=1)
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the checkpoint path loader was used on the records path")

    monkeypatch.setattr(extractor, "load_checkpoint", forbidden)

    frame = extract_embeddings(
        checkpoint, images, device="cpu", records=bundle, num_workers=0
    )
    assert len(frame) == 1


def test_multiprocessing_dataset_carries_snapshots(tmp_path):
    images, selector = images_and_manifest(tmp_path, rows=4)
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    single = extract_embeddings(
        checkpoint, images, device="cpu", records=bundle, num_workers=0
    )
    parallel = extract_embeddings(
        checkpoint, images, device="cpu", records=bundle, num_workers=2
    )

    assert single["id"].tolist() == parallel["id"].tolist()
    assert np.allclose(
        single.filter(like="dim_").to_numpy(),
        parallel.filter(like="dim_").to_numpy(),
        atol=1e-6,
    )


def _tiny_onnx(tmp_path: Path, external: bool) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = (np.arange(2 * 3 * 32 * 32, dtype=np.float32) / 1000.0).reshape(3 * 32 * 32, 2)
    bias = np.zeros((2,), dtype=np.float32)
    initializers = [
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(bias, "bias"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["input"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "weight", "bias"], ["embedding"], alpha=1.0),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 2])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / ("external.onnx" if external else "inline.onnx")
    if external:
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="external.data",
            size_threshold=0,
        )
    else:
        onnx.save_model(model, str(path))
    return path


def test_inline_onnx_session_uses_verified_bytes(tmp_path, monkeypatch):
    import onnxruntime as ort

    images, selector = images_and_manifest(tmp_path, rows=2)
    bundle = bundle_for(images, selector)
    onnx_path = _tiny_onnx(tmp_path, external=False)

    seen: list[object] = []
    real_session = ort.InferenceSession

    def spy(source, *args, **kwargs):
        seen.append(type(source).__name__)
        return real_session(source, *args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", spy)

    result = _extract_embeddings_result(
        None, images, device="cpu", onnx_path=onnx_path, records=bundle, num_workers=0
    )

    assert seen == ["bytes"]
    assert result.effective["onnx"]["provider"] == "CPUExecutionProvider"
    assert result.effective["onnx"]["internal_feature_contract"]["status"] == "unavailable"
    assert any(snapshot.path == onnx_path.resolve() for snapshot in result.used_files)
    assert len(result.frame) == 2


def test_external_onnx_files_are_hashed_and_rechecked(tmp_path):
    onnx_path = _tiny_onnx(tmp_path, external=True)
    externals = _onnx_external_files(onnx_path)
    assert [path.name for path in externals] == ["external.data"]

    from otuformer.data.provenance import snapshot_bytes

    _data, snapshot = snapshot_bytes(externals[0])
    assert snapshot.sha256 == hashlib.sha256(externals[0].read_bytes()).hexdigest()

    externals[0].write_bytes(externals[0].read_bytes() + b"\x00")
    with pytest.raises(RuntimeError, match="changed while the session was loading"):
        _require_unchanged([snapshot])


def test_onnx_external_reference_must_stay_next_to_the_graph(tmp_path):
    from onnx import TensorProto, helper, numpy_helper
    import onnx

    value = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), "weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["embedding"])],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 2])],
        [value],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    path = graphs / "escape.onnx"
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="link.data",
        size_threshold=0,
    )
    outside = tmp_path / "outside.data"
    (graphs / "link.data").rename(outside)
    try:
        (graphs / "link.data").symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    with pytest.raises(ValueError, match="outside the graph directory"):
        _onnx_external_files(path)


def test_provenance_keeps_the_hashes_of_the_bytes_that_ran(tmp_path):
    images, selector = images_and_manifest(tmp_path, rows=2)
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    result = _extract_embeddings_result(
        checkpoint, images, device="cpu", records=bundle, num_workers=0
    )
    record = build_run_record(
        requested={"token_mode": "cls"},
        bundle=bundle,
        execution=result,
        evaluation={},
        status="completed",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        artifacts=[],
    )

    # The real checkpoint and selector hashes must survive into the record.
    hashes = {row["sha256"] for row in record["used_files"]}
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() in hashes
    assert hashlib.sha256(selector.read_bytes()).hexdigest() in hashes
    assert record["feature_dimensions"]["status"] == "available"
    assert record["feature_dimensions"]["input_dim"] == 192
    assert record["feature_dimensions"]["output_dim"] == 192


def test_external_onnx_graph_drift_during_the_session_is_detected(
    tmp_path, monkeypatch
):
    import onnxruntime as ort

    images, selector = images_and_manifest(tmp_path, rows=1)
    bundle = bundle_for(images, selector)
    onnx_path = _tiny_onnx(tmp_path, external=True)

    import onnx

    real_session = ort.InferenceSession

    def drifting_session(source, *args, **kwargs):
        session = real_session(source, *args, **kwargs)
        # Replace the graph with a valid, different one while keeping the tensors
        # byte-identical: only the graph hash can notice this.
        proto = onnx.load(str(onnx_path), load_external_data=False)
        proto.producer_name = "swapped"
        staged = tmp_path / "staged.onnx"
        onnx.save_model(
            proto,
            str(staged),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="external.data",
            size_threshold=0,
        )
        onnx_path.write_bytes(staged.read_bytes())
        return session

    monkeypatch.setattr(ort, "InferenceSession", drifting_session)

    with pytest.raises(RuntimeError, match="changed while the session was loading"):
        _extract_embeddings_result(
            None,
            images,
            device="cpu",
            onnx_path=onnx_path,
            records=bundle,
            num_workers=0,
        )


def test_patch_topk_never_claims_the_unused_projector(tmp_path):
    images, selector = images_and_manifest(tmp_path, rows=2)
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    result = _extract_embeddings_result(
        checkpoint,
        images,
        device="cpu",
        records=bundle,
        token_mode="patch-topk",
        topk_patches=1,
        use_projector_output=True,
        num_workers=0,
    )

    # The flag is ignored on this branch, so the projector must not be recorded as
    # executed and the vectors must not be described as unit-normalized.
    assert "projector" not in result.effective["post_feature_ops"]
    assert result.effective["vector_normalization"]["status"] == "none"
    values = result.frame[[c for c in result.frame.columns if c.startswith("dim_")]]
    assert np.linalg.norm(values.to_numpy(), axis=1).min() > 2.0
    # Two samples cannot fit a PCA, so its steps must not be claimed either.
    assert result.effective["patch_topk_pca"]["mode"] == "none"
    assert "pca" not in result.effective["feature_chain"]["ops"]
    assert "standard_scaler" not in result.effective["feature_chain"]["ops"]


def test_datasets_recheck_containment_when_consuming(tmp_path):
    images, selector = images_and_manifest(tmp_path, rows=1)
    bundle = bundle_for(images, selector)
    outside = write_png(tmp_path / "outside.png", color=(1, 2, 3))

    dataset = RecordImageDataset(bundle, 32)
    target = images / "plate1" / "000.png"
    target.unlink()
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    # Reading through the dataset must refuse the escaped path, not just hash it.
    with pytest.raises(ValueError, match="resolves outside the root"):
        dataset[0]


def test_constant_attribute_external_tensor_is_discovered(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(
        np.arange(4, dtype=np.float32).reshape(2, 2), "weight"
    )
    payload = weight.raw_data
    (tmp_path / "constant.data").write_bytes(payload)
    # onnx.save_model does not externalize attribute tensors, so mark the Constant
    # payload external by hand: a legal model ORT can load from its path.
    weight.ClearField("raw_data")
    weight.data_location = TensorProto.EXTERNAL
    for key, value in (("location", "constant.data"), ("length", str(len(payload)))):
        entry = weight.external_data.add()
        entry.key = key
        entry.value = value

    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["embedding"]),
            helper.make_node("Constant", [], ["weight"], value=weight),
        ],
        "constant",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 2])],
        [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "constant.onnx"
    onnx.save_model(model, str(path))

    import onnxruntime as ort

    # The model is loadable, so treating it as self-contained would be wrong.
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [entry.name for entry in _onnx_external_files(path)] == ["constant.data"]


def _function_default_onnx(tmp_path, data_dir, location="default.data"):
    """A model whose local function carries an external tensor as a default attribute."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.array([6.0, 9.0], dtype=np.float32), "w")
    payload = weight.raw_data
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "default.data").write_bytes(payload)
    weight.ClearField("raw_data")
    weight.data_location = TensorProto.EXTERNAL
    for key, value in (("location", location), ("length", str(len(payload)))):
        entry = weight.external_data.add()
        entry.key = key
        entry.value = value

    default = onnx.AttributeProto()
    default.name = "w"
    default.type = onnx.AttributeProto.TENSOR
    default.t.CopyFrom(weight)

    constant = helper.make_node("Constant", [], ["weights"])
    reference = onnx.AttributeProto()
    reference.name = "value"
    reference.type = onnx.AttributeProto.TENSOR
    reference.ref_attr_name = "w"
    constant.attribute.extend([reference])

    function = helper.make_function(
        "custom",
        "Scale",
        ["input"],
        ["output"],
        [constant, helper.make_node("Mul", ["input", "weights"], ["output"])],
        [helper.make_opsetid("", 17)],
    )
    function.attribute_proto.extend([default])
    function.attribute.extend(["w"])

    graph = helper.make_graph(
        [helper.make_node("Scale", ["input"], ["output"], domain="custom")],
        "function-default",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("custom", 1)],
    )
    model.ir_version = 9
    model.functions.extend([function])
    # The graph sits beside its payload: an external location is graph-relative.
    path = data_dir / "fn.onnx"
    onnx.save_model(model, str(path))
    return path, data_dir / "default.data"


def test_function_default_attribute_tensor_is_discovered(tmp_path):
    import onnx
    import numpy as np
    import onnxruntime as ort

    path, data = _function_default_onnx(tmp_path, tmp_path / "graph")
    onnx.checker.check_model(str(path))

    # The model loads and runs from its path, and the payload is used from the default
    # attribute the function body reaches through ref_attr_name.
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"input": np.ones((1, 2), dtype=np.float32)})
    assert output[0].tolist() == [[6.0, 9.0]]
    assert _onnx_external_files(path) == [data.resolve()]


def _alias_graph(tmp_path, real_payload, alias_payload, other_tag=None):
    """A graph at ``real/`` plus an ``alias/`` symlink with its own payload.

    ``other_tag`` additionally writes a second, byte-different graph so a symlink
    can be repointed at a model this run never hashed.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    declared = np.ones((3 * 32 * 32, 2), dtype=np.float32)

    def write(directory, payload, tag):
        directory.mkdir(parents=True, exist_ok=True)
        graph = helper.make_graph(
            [
                helper.make_node("Flatten", ["input"], ["flat"], axis=1),
                helper.make_node("MatMul", ["flat", "weight"], ["embedding"]),
            ],
            "tiny",
            [
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, ["batch", 3, 32, 32]
                )
            ],
            [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 2])],
            [numpy_helper.from_array(declared, "weight")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        model.doc_string = tag
        path = directory / "model.onnx"
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.data",
            size_threshold=0,
        )
        (directory / "weights.data").write_bytes(
            np.full_like(declared, payload).tobytes()
        )
        return path

    real_graph = write(tmp_path / "real", real_payload, "real")
    write(tmp_path / "alias", alias_payload, "alias")
    alias_graph = tmp_path / "alias" / "model.onnx"
    alias_graph.unlink()
    try:
        alias_graph.symlink_to(real_graph)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")
    other = write(tmp_path / "other", real_payload, other_tag) if other_tag else None
    return alias_graph, real_graph, other


def test_onnx_records_the_weights_that_were_actually_loaded(tmp_path):
    alias_graph, real_graph, _ = _alias_graph(
        tmp_path, real_payload=1.0, alias_payload=5.0
    )
    images, selector = images_and_manifest(tmp_path / "data", rows=2)
    bundle = bundle_for(images, selector)

    via_alias = _extract_embeddings_result(
        None, images, device="cpu", onnx_path=alias_graph, records=bundle, num_workers=0
    )
    via_real = _extract_embeddings_result(
        None, images, device="cpu", onnx_path=real_graph, records=bundle, num_workers=0
    )

    recorded = {Path(entry.path) for entry in via_alias.used_files}
    # onnxruntime resolves external data next to the path it was given, so the
    # record must name the alias directory -- not the symlink's target directory.
    assert alias_graph in recorded
    assert alias_graph.parent / "weights.data" in recorded
    assert not any(path.parent == real_graph.parent for path in recorded)
    assert via_alias.effective["onnx"]["external_tensor_files"] == [
        str(alias_graph.parent / "weights.data")
    ]
    assert via_alias.effective["onnx"]["graph_file"] == str(alias_graph)

    # Different payloads must produce different vectors: the record is not cosmetic.
    dims = [column for column in via_alias.frame.columns if column.startswith("dim_")]
    assert not np.allclose(
        via_alias.frame[dims].to_numpy(), via_real.frame[dims].to_numpy()
    )


def test_onnx_graph_symlink_repointed_before_session_is_detected(tmp_path, monkeypatch):
    import onnxruntime as ort

    alias_graph, _real_graph, other_graph = _alias_graph(
        tmp_path, real_payload=1.0, alias_payload=5.0, other_tag="other"
    )
    images, selector = images_and_manifest(tmp_path / "data", rows=2)
    bundle = bundle_for(images, selector)

    original = ort.InferenceSession

    def repointing_session(*args, **kwargs):
        # A permanent switch, not a change-and-restore: the hash of the old target
        # must never be treated as evidence for the new one.
        alias_graph.unlink()
        alias_graph.symlink_to(other_graph)
        return original(*args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", repointing_session)

    with pytest.raises(RuntimeError, match="changed while the session was loading"):
        _extract_embeddings_result(
            None,
            images,
            device="cpu",
            onnx_path=alias_graph,
            records=bundle,
            num_workers=0,
        )


def _dotdot_location_graph(tmp_path):
    """A graph whose external location walks through a symlink before ``..``."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    root = tmp_path / "graph"
    (root / "a" / "sub").mkdir(parents=True)
    (root / "b" / "sub").mkdir(parents=True)
    declared = np.ones((3 * 32 * 32, 2), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["input"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weight"], ["embedding"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 2])],
        [numpy_helper.from_array(declared, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    link = root / "link"
    try:
        link.symlink_to(root / "a" / "sub")
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")
    path = root / "model.onnx"
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="link/../weights.data",
        size_threshold=0,
    )
    # onnx writes the payload at the lexically folded position, which is exactly the
    # mismatch under test. Remove it so the fixture holds only what the runtime can
    # reach; the caller writes the payload where the runtime actually looks.
    (root / "weights.data").unlink(missing_ok=True)
    return root, path, declared


def test_external_location_dotdot_follows_the_link_without_a_decoy(tmp_path):
    import onnxruntime as ort

    root, path, declared = _dotdot_location_graph(tmp_path)
    # The runtime follows the link first, so `link/..` is a/, and nothing exists at
    # the lexically folded root position.
    real = root / "a" / "weights.data"
    real.write_bytes(np.full_like(declared, 5.0).tobytes())
    assert not (root / "weights.data").exists()

    enumerated = _onnx_external_files(path)
    assert [entry.resolve() for entry in enumerated] == [real]
    # The spelling keeps the location the graph wrote, so a later redirect of the
    # intermediate link changes what a re-read sees.
    assert enumerated == [root / "link" / ".." / "weights.data"]

    images, selector = images_and_manifest(tmp_path / "data", rows=2)
    bundle = bundle_for(images, selector)
    # Reading the input size must not need the weights: this is the run that fails
    # if the size check loads external data through its own resolution.
    result = _extract_embeddings_result(
        None, images, device="cpu", onnx_path=path, records=bundle, num_workers=0
    )

    recorded = {Path(entry.path) for entry in result.used_files}
    assert root / "link" / ".." / "weights.data" in recorded
    assert result.effective["onnx"]["external_tensor_files"] == [
        str(root / "link" / ".." / "weights.data")
    ]
    # The recorded path names the bytes the runtime consumed.
    payload = np.frombuffer(
        Path(next(iter(enumerated))).read_bytes(), dtype=np.float32
    )
    assert payload.max() == pytest.approx(5.0)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(
        None, {"input": np.ones((1, 3, 32, 32), dtype=np.float32)}
    )[0]
    assert output[0][0] == pytest.approx(5.0 * 3 * 32 * 32)


def test_external_location_redirected_during_session_is_detected(tmp_path, monkeypatch):
    import onnxruntime as ort

    root, path, declared = _dotdot_location_graph(tmp_path)
    (root / "a" / "weights.data").write_bytes(np.ones_like(declared).tobytes())
    (root / "b" / "weights.data").write_bytes(np.full_like(declared, 5.0).tobytes())
    link = root / "link"
    images, selector = images_and_manifest(tmp_path / "data", rows=2)
    bundle = bundle_for(images, selector)

    original = ort.InferenceSession

    def redirecting_session(*args, **kwargs):
        # Permanent, not change-and-restore: the link keeps pointing at b/.
        link.unlink()
        link.symlink_to(root / "b" / "sub")
        return original(*args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", redirecting_session)

    # The snapshot is bound to the location expression, so re-reading it follows the
    # new link and the different bytes are caught.
    with pytest.raises(RuntimeError, match="changed while the session was loading"):
        _extract_embeddings_result(
            None,
            images,
            device="cpu",
            onnx_path=path,
            records=bundle,
            num_workers=0,
        )


def _duplicate_location_graph(tmp_path):
    """One tensor with two ``location`` keys: onnx reads the first, ORT the last."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    root = tmp_path / "graph"
    (root / "out").mkdir(parents=True)
    declared = np.ones((3 * 32 * 32, 2), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["input"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weight"], ["embedding"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 2])],
        [numpy_helper.from_array(declared, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = root / "model.onnx"
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="first.data",
        size_threshold=0,
    )
    (root / "first.data").write_bytes(np.ones_like(declared).tobytes())
    actual = root / "out" / "actual.data"
    actual.write_bytes(np.full_like(declared, 5.0).tobytes())
    proto = onnx.load(str(path), load_external_data=False)
    duplicate = proto.graph.initializer[0].external_data.add()
    duplicate.key = "location"
    duplicate.value = "out/actual.data"
    onnx.save_model(proto, str(path))
    return path, root / "first.data", actual


def test_duplicate_external_location_is_refused(tmp_path):
    path, first, actual = _duplicate_location_graph(tmp_path)

    # The readers disagree, so there is no honest file list to record.
    with pytest.raises(ValueError, match="declares 2 'location' entries"):
        _onnx_external_files(path)
    assert first.is_file() and actual.is_file()


def test_duplicate_external_location_publishes_no_provenance(tmp_path):
    path, _first, _actual = _duplicate_location_graph(tmp_path)
    images, selector = images_and_manifest(tmp_path / "data", rows=2)
    bundle = bundle_for(images, selector)

    with pytest.raises(ValueError, match="declares 2 'location' entries"):
        _extract_embeddings_result(
            None, images, device="cpu", onnx_path=path, records=bundle, num_workers=0
        )


def _pool_inputs(tmp_path):
    """Inputs only: building them writes a checkpoint, which save-fault tests must
    not intercept."""
    images, selector = images_and_manifest(
        tmp_path / "data", rows=4, header=["image", "label"], labels="A"
    )
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)
    return images, bundle, checkpoint


def _pool_run(tmp_path, pool_path, pool_type="lightweight", **kwargs):
    images, bundle, checkpoint = _pool_inputs(tmp_path)
    settings = dict(
        checkpoint_path=checkpoint,
        images_dir=images,
        device="cpu",
        records=bundle,
        token_mode="attention-pool",
        attention_pooling_type=pool_type,
        attention_pooling_epochs=1,
        attention_pooling_checkpoint_path=pool_path,
        batch_size=2,
        num_workers=0,
        seed=0,
    )
    settings.update(kwargs)
    return _extract_embeddings_result(**settings)


@pytest.mark.parametrize("pool_type", ["lightweight", "multihead", "gated"])
def test_trained_pool_weights_are_saved_and_reloaded(tmp_path, pool_type):
    import torch

    pool_path = tmp_path / "cache" / "pool.pth"
    first = _pool_run(tmp_path, pool_path, pool_type)
    written = Path(first.effective["attention_pooling"]["written_file"])
    assert written.is_file()

    # The saved state must contain the trained pool, not just the backbone.
    payload = torch.load(written, map_location="cpu", weights_only=False)
    pool_keys = [key for key in payload["teacher"] if key.startswith("attention_pool.")]
    assert pool_keys, sorted(payload["teacher"])[:5]

    # Reloading that file must reproduce the vectors exactly, which only holds when
    # the checkpoint carried the trained parameters.
    second = _pool_run(tmp_path, pool_path, pool_type)
    assert second.effective["attention_pooling"]["source"]["status"] == "cached"
    dims = [column for column in first.frame.columns if column.startswith("dim_")]
    assert np.allclose(
        first.frame[dims].to_numpy(), second.frame[dims].to_numpy(), atol=1e-6
    )


def test_checkpoint_without_trained_pool_parameters_is_refused(tmp_path):
    import torch

    pool_path = tmp_path / "cache" / "pool.pth"
    first = _pool_run(tmp_path, pool_path)
    written = Path(first.effective["attention_pooling"]["written_file"])

    # Reproduce the files written by the old save-before-attach order.
    payload = torch.load(written, map_location="cpu", weights_only=False)
    stripped = {
        key: value
        for key, value in payload["teacher"].items()
        if not key.startswith("attention_pool.")
    }
    payload["teacher"] = stripped
    torch.save(payload, written)

    with pytest.raises(ValueError, match="contains no trained pool parameters"):
        _pool_run(tmp_path, pool_path)


def test_failure_before_return_leaves_no_pooling_file(tmp_path, monkeypatch):
    import otuformer.embedding.extractor as extractor_module

    pool_path = tmp_path / "cache" / "pool.pth"
    images, selector = images_and_manifest(
        tmp_path / "data", rows=4, header=["image", "label"], labels="A"
    )
    bundle = bundle_for(images, selector)
    checkpoint = tiny_checkpoint(tmp_path)

    def exploding_extract(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(extractor_module, "_extract_one_csv", exploding_extract)

    with pytest.raises(RuntimeError, match="inference failed"):
        _extract_embeddings_result(
            checkpoint_path=checkpoint,
            images_dir=images,
            device="cpu",
            records=bundle,
            token_mode="attention-pool",
            attention_pooling_type="lightweight",
            attention_pooling_epochs=1,
            attention_pooling_checkpoint_path=pool_path,
            batch_size=2,
            num_workers=0,
        )

    # The extractor owns the weights until it returns them: a failure must not
    # leave a file the caller never learns about.
    assert not list(pool_path.parent.glob("*_pooling.pth"))


def test_half_written_pooling_file_is_removed(tmp_path, monkeypatch):
    import otuformer.embedding.extractor as extractor_module

    pool_path = tmp_path / "cache" / "pool.pth"
    # Build the inputs first: the fixture writes a checkpoint through torch.save, so
    # patching before this point would make the test pass without reaching the pool.
    images, bundle, checkpoint = _pool_inputs(tmp_path)
    real_save = extractor_module.torch.save
    injected: list[list[str]] = []

    def failing_save(payload, target):
        # The real save receives an open handle, so the reservation already
        # happened; observe the reserved entry from the filesystem side.
        assert hasattr(target, "write"), type(target)
        injected.append(
            [p.name for p in pool_path.parent.glob("*_pooling.pth")]
        )
        target.write(b"x" * 32)  # a partial checkpoint on disk
        raise OSError("disk full")

    monkeypatch.setattr(extractor_module.torch, "save", failing_save)

    with pytest.raises(OSError, match="disk full"):
        _extract_embeddings_result(
            checkpoint_path=checkpoint,
            images_dir=images,
            device="cpu",
            records=bundle,
            token_mode="attention-pool",
            attention_pooling_type="lightweight",
            attention_pooling_epochs=1,
            attention_pooling_checkpoint_path=pool_path,
            batch_size=2,
            num_workers=0,
        )

    # The injection hit the pooling write (a reserved entry existed at that moment),
    # and the partial file is gone afterwards.
    assert injected and injected[0], injected
    assert injected[0][0].endswith("_pooling.pth")
    assert not list(pool_path.parent.glob("*_pooling.pth"))


def test_failure_after_saving_still_removes_the_pooling_file(tmp_path, monkeypatch):
    """The cleanup scope covers reporting, not only the write itself."""
    import builtins

    pool_path = tmp_path / "cache" / "pool.pth"
    real_print = builtins.print

    def failing_print(*args, **kwargs):
        if args and "Saved finetuned attention pooling checkpoint" in str(args[0]):
            raise OSError(28, "No space left on device")
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", failing_print)

    with pytest.raises(OSError, match="No space left"):
        _pool_run(tmp_path, pool_path)

    # The file was written but never handed over, so it must not survive.
    assert not list(pool_path.parent.glob("*_pooling.pth"))


def test_publish_never_overwrites_a_concurrent_file(tmp_path, monkeypatch):
    """A file that appears between the check and the write is not clobbered."""
    import otuformer.embedding.extractor as extractor_module

    pool_path = tmp_path / "cache" / "pool.pth"
    real_open = extractor_module.os.open

    def racing_open(path, flags, *args, **kwargs):
        # Another publisher wins the race just before this call reserves the entry.
        if Path(path).name.endswith("_pooling.pth"):
            Path(path).write_bytes(b"other publisher's weights")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(extractor_module.os, "open", racing_open)

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _pool_run(tmp_path, pool_path)

    survivors = list(pool_path.parent.glob("*_pooling.pth"))
    assert len(survivors) == 1
    assert survivors[0].read_bytes() == b"other publisher's weights"


def test_failed_publish_cannot_delete_another_publishers_file(tmp_path, monkeypatch):
    import otuformer.embedding.extractor as extractor_module

    pool_path = tmp_path / "cache" / "pool.pth"
    real_open = extractor_module.os.open

    def failing_open(path, flags, *args, **kwargs):
        if Path(path).name.endswith("_pooling.pth"):
            Path(path).write_bytes(b"other publisher's weights")
            raise OSError(28, "No space left on device")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(extractor_module.os, "open", failing_open)

    with pytest.raises(OSError, match="No space left"):
        _pool_run(tmp_path, pool_path)

    # We never owned the entry, so the other file stays exactly as it was.
    survivors = list(pool_path.parent.glob("*_pooling.pth"))
    assert len(survivors) == 1
    assert survivors[0].read_bytes() == b"other publisher's weights"


def test_failure_right_after_reservation_leaves_no_empty_file(tmp_path, monkeypatch):
    """Ownership starts at the exclusive create, not at the successful wrap."""
    import otuformer.embedding.extractor as extractor_module

    pool_path = tmp_path / "cache" / "pool.pth"
    images, bundle, checkpoint = _pool_inputs(tmp_path)
    real_fdopen = extractor_module.os.fdopen
    closed: list[int] = []

    def failing_fdopen(fd, *args, **kwargs):
        closed.append(fd)
        raise MemoryError("cannot wrap descriptor")

    monkeypatch.setattr(extractor_module.os, "fdopen", failing_fdopen)

    with pytest.raises(MemoryError):
        _extract_embeddings_result(
            checkpoint_path=checkpoint,
            images_dir=images,
            device="cpu",
            records=bundle,
            token_mode="attention-pool",
            attention_pooling_type="lightweight",
            attention_pooling_epochs=1,
            attention_pooling_checkpoint_path=pool_path,
            batch_size=2,
            num_workers=0,
        )

    monkeypatch.setattr(extractor_module.os, "fdopen", real_fdopen)
    # No zero-byte cache left behind, and the reserved descriptor was closed.
    assert not list(pool_path.parent.glob("*_pooling.pth"))
    assert closed
    for fd in closed:
        with pytest.raises(OSError):
            extractor_module.os.fstat(fd)

    # A retry is not blocked by a stranded cache entry.
    result = _pool_run(tmp_path, pool_path)
    assert Path(result.effective["attention_pooling"]["written_file"]).is_file()


def test_interrupt_after_reservation_is_cleaned_up(tmp_path, monkeypatch):
    """The protected scope starts before the descriptor exists.

    An interruption arriving once the entry is created -- here at the descriptor
    wrap, i.e. the first line inside the scope -- must leave no empty cache and no
    open descriptor, and the original interruption must reach the caller.
    """
    import otuformer.embedding.extractor as extractor_module

    pool_path = tmp_path / "cache" / "pool.pth"
    images, bundle, checkpoint = _pool_inputs(tmp_path)
    real_fdopen = extractor_module.os.fdopen
    reserved: list[int] = []

    def interrupted_fdopen(fd, *args, **kwargs):
        reserved.append(fd)
        raise KeyboardInterrupt

    monkeypatch.setattr(extractor_module.os, "fdopen", interrupted_fdopen)

    with pytest.raises(KeyboardInterrupt):
        _extract_embeddings_result(
            checkpoint_path=checkpoint,
            images_dir=images,
            device="cpu",
            records=bundle,
            token_mode="attention-pool",
            attention_pooling_type="lightweight",
            attention_pooling_epochs=1,
            attention_pooling_checkpoint_path=pool_path,
            batch_size=2,
            num_workers=0,
        )

    monkeypatch.setattr(extractor_module.os, "fdopen", real_fdopen)
    assert not list(pool_path.parent.glob("*_pooling.pth"))
    assert reserved
    for fd in reserved:
        with pytest.raises(OSError):
            extractor_module.os.fstat(fd)

    # The cache is not left blocking a retry.
    result = _pool_run(tmp_path, pool_path)
    assert Path(result.effective["attention_pooling"]["written_file"]).is_file()


def test_failed_removal_keeps_the_original_error(tmp_path, monkeypatch, capsys):
    """A cleanup failure is reported, never substituted for the real failure."""
    import pathlib as pathlib_module

    import otuformer.embedding.extractor as extractor_module

    pool_path = tmp_path / "cache" / "pool.pth"
    images, bundle, checkpoint = _pool_inputs(tmp_path)
    real_save = extractor_module.torch.save

    def failing_save(payload, target):
        target.write(b"x" * 31)
        raise KeyboardInterrupt

    def locked_unlink(self, missing_ok=False):
        raise PermissionError(13, "Operation not permitted", str(self))

    monkeypatch.setattr(extractor_module.torch, "save", failing_save)
    monkeypatch.setattr(pathlib_module.Path, "unlink", locked_unlink)

    with pytest.raises(KeyboardInterrupt):
        _extract_embeddings_result(
            checkpoint_path=checkpoint,
            images_dir=images,
            device="cpu",
            records=bundle,
            token_mode="attention-pool",
            attention_pooling_type="lightweight",
            attention_pooling_epochs=1,
            attention_pooling_checkpoint_path=pool_path,
            batch_size=2,
            num_workers=0,
        )

    monkeypatch.setattr(extractor_module.torch, "save", real_save)
    monkeypatch.undo()
    output = capsys.readouterr().out
    # The leftover file is named, so an operator can remove it by hand.
    assert "could not remove the partial attention pooling checkpoint" in output
    assert str(pool_path.parent) in output
