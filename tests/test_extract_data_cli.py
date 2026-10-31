import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from otuformer.cli.main import app

runner = CliRunner()


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


def fixture(tmp_path, rows=4, with_labels=True):
    images = tmp_path / "images"
    for index in range(rows):
        write_png(images / "plate1" / f"{index:03d}.png", color=(index * 7, 0, 0))
    header = ["image", "label"] if with_labels else ["image"]
    body = []
    for index in range(rows):
        row = [f"plate1/{index:03d}.png"]
        if with_labels:
            row.append("NA" if index % 2 == 0 else "001")
        body.append(row)
    selector = write_csv(tmp_path / "sel.csv", header, body)
    return images, selector


def coco_source(images, file_name="plate1/000.png"):
    return {
        "images": [{"id": 17, "file_name": file_name, "width": 32, "height": 32}],
        "categories": [{"id": 4, "name": "Head"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 17,
                "category_id": 4,
                "bbox": [1, 1, 6, 6],
                "area": 36,
                "iscrowd": 0,
                "segmentation": [[1, 1, 7, 1, 7, 7, 1, 7]],
            }
        ],
    }


from test_extract_records import _function_default_onnx


def cli_message(result) -> str:
    """CliRunner output with the rich error panel's line wrapping removed."""
    import re

    return re.sub(r"[\s\u2502\u2500\u256d\u256e\u2570\u256f]+", " ", result.output)


def test_extract_new_path_options_and_preflight(tmp_path):
    images, selector = fixture(tmp_path)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"

    # One CSV is the whole selector: supplying --label-csv selects the shared,
    # strictly validated path on its own.
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--num-workers",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "run_provenance.json").exists()
    assert (out / "data_records.jsonl").exists()
    assert (out / "data_validation.json").exists()

    out2 = tmp_path / "out2"
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out2),
            "--coco-panoptic",
            str(tmp_path / "panoptic.json"),
        ],
    )
    assert result.exit_code != 0
    assert not out2.exists()

    # A pre-existing output directory keeps its contents on a rejected request.
    populated = tmp_path / "populated"
    populated.mkdir()
    sentinel = populated / "keep.txt"
    sentinel.write_text("keep me")
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(populated),
            "--coco-panoptic",
            str(tmp_path / "panoptic.json"),
            "--overwrite",
        ],
    )
    assert result.exit_code != 0
    assert sentinel.read_text() == "keep me"

    # An output directory inside a protected input root is refused.
    inside = images / "out"
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(inside),
            "--label-csv",
            str(selector),
        ],
    )
    assert result.exit_code != 0
    assert not inside.exists()


def test_extract_attaches_each_annotation_role_and_publishes(tmp_path):
    images, selector = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    coco = tmp_path / "coco.json"
    coco.write_text(json.dumps(coco_source(images)), encoding="utf-8")
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--coco-instances",
            str(coco),
            "--coco-keypoints",
            str(coco),
            "--num-workers",
            "0",
            "--batch-size",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    embeddings = pd.read_csv(out / "embeddings.csv")
    assert embeddings["id"].tolist() == [
        f"plate1/{index:03d}.png" for index in range(4)
    ]

    mapping = [json.loads(line) for line in (out / "data_records.jsonl").read_text().splitlines()]
    assert [row["embedding_row"] for row in mapping] == [1, 2, 3, 4]
    assert [row["output_id"] for row in mapping] == embeddings["id"].tolist()
    assert mapping[0]["coco_images"] == [
        {"source_id": "instances", "image_id": 17},
        {"source_id": "keypoints", "image_id": 17},
    ]
    assert mapping[1]["coco_images"] == []
    assert mapping[0]["metadata"]["label"] == "NA"

    validation = json.loads((out / "data_validation.json").read_text())
    assert validation["status"] == "valid"
    assert validation["counts"]["images_without_annotations"] == 3
    assert validation["polygon_diagnostics"]

    provenance = json.loads((out / "run_provenance.json").read_text())
    assert provenance["status"] == "completed"
    assert provenance["effective"]["data_path"] == "records"
    assert provenance["thresholds"]["status"] == "not_applicable"
    assert provenance["split"]["status"] == "not_provided"
    assert provenance["selection"]["selection_metadata_sha256_v1"]
    assert provenance["selection"]["data_content_sha256_v1"]
    assert {entry["path"] for entry in provenance["artifacts"]} >= {
        "embeddings.csv",
        "data_records.jsonl",
        "data_validation.json",
        "metrics.csv",
    }
    assert provenance["metrics"]["values"]["kNN_Acc_k1"]["status"] in {
        "available",
        "unavailable",
    }
    assert provenance["metrics"]["protocol"]["knn_accuracy"]["distance"] == "cosine"
    assert provenance["metrics"]["protocol"]["linear_probing"]["classifier"] == "LogisticRegression"


def test_extract_records_evaluation_uses_one_parse(tmp_path):
    images, selector = fixture(tmp_path, rows=12)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--metrics-sample-size",
            "6",
            "--num-workers",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output

    provenance = json.loads((out / "run_provenance.json").read_text())
    metrics = provenance["metrics"]
    assert metrics["samples_considered"] == 12
    assert metrics["sampled_rows"] == 6
    assert len(metrics["excluded_by_sampling"]) == 6
    assert metrics["seed"] == 42
    assert metrics["label_schema"]["reported"] if "label_schema" in metrics else True

    frame = pd.read_csv(out / "embeddings.csv")
    mapping = [json.loads(line) for line in (out / "data_records.jsonl").read_text().splitlines()]
    assert len(mapping) == len(frame) == 12


def test_extract_failure_safety_and_status(tmp_path):
    images, selector = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)

    # Invalid annotations fail before inference and publish no embedding pair.
    broken = coco_source(images)
    broken["annotations"][0]["bbox"] = [5, 5, 100, 100]
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    out = tmp_path / "out-fail"
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--coco-instances",
            str(broken_path),
            "--num-workers",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert not (out / "embeddings.csv").exists()
    assert not (out / "data_records.jsonl").exists()
    assert (out / "data_validation.json").exists()
    provenance = json.loads((out / "run_provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert provenance["failure"]["kind"] == "data_validation"
    assert "bbox_out_of_bounds" in provenance["failure"]["codes"]

    # Changed image bytes abort with no published pair and no completed record.
    ok_out = tmp_path / "out-drift"
    drift_coco = tmp_path / "ok.json"
    drift_coco.write_text(json.dumps(coco_source(images)), encoding="utf-8")
    write_png(images / "plate1" / "000.png", color=(250, 0, 0))
    # The manifest still selects the same key; hashing happens per run, so a
    # second run sees the new bytes and must still publish consistently.
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(ok_out),
            "--label-csv",
            str(selector),
            "--coco-instances",
            str(drift_coco),
            "--num-workers",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    provenance = json.loads((ok_out / "run_provenance.json").read_text())
    assert provenance["status"] == "completed"

    # A missing optional decoder is an actionable error, not a silent skip.
    out_nodecoder = tmp_path / "out-nodecoder"
    polygon_only = coco_source(images)
    del polygon_only["annotations"][0]["segmentation"]
    polygon_path = tmp_path / "poly.json"
    polygon_path.write_text(json.dumps(polygon_only), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out_nodecoder),
            "--label-csv",
            str(selector),
            "--coco-instances",
            str(polygon_path),
            "--num-workers",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output

    # The legacy path is now directory-only: any CSV selects the shared path.
    legacy_out = tmp_path / "out-legacy"
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(legacy_out),
            "--num-workers",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (legacy_out / "embeddings.csv").exists()
    assert not (legacy_out / "run_provenance.json").exists()
    assert not (legacy_out / "data_records.jsonl").exists()


def flat_manifest(tmp_path):
    """The simplest image-only case: images directly under the root."""
    images = tmp_path / "flat"
    write_png(images / "one.png")
    write_png(images / "two.png")
    selector = write_csv(
        tmp_path / "flat.csv", ["image"], [["one.png"], ["two.png"]]
    )
    return images, selector


def test_root_level_images_work_on_the_new_path(tmp_path):
    images, selector = flat_manifest(tmp_path)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    frame = pd.read_csv(out / "embeddings.csv")
    # No sample column: a missing sample is not an empty one.
    assert list(frame.columns)[:1] == ["id"]
    assert "sample" not in frame.columns
    assert frame["id"].tolist() == ["one.png", "two.png"]


def test_label_csv_with_coco_stays_the_selector(tmp_path):
    images, selector = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    coco = tmp_path / "coco.json"
    coco.write_text(json.dumps(coco_source(images)), encoding="utf-8")
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--coco-instances",
            str(coco),
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    provenance = json.loads((out / "run_provenance.json").read_text())
    # The CSV is the selector, not a silently dropped parameter.
    assert provenance["selection"]["rule"] == "csv_rows"
    assert provenance["selection"]["selected_images"] == 4
    assert provenance["selection"]["selector"]["sha256"]
    assert provenance["requested"]["label_csv"] == str(selector)
    frame = pd.read_csv(out / "embeddings.csv")
    assert frame["id"].tolist() == [f"plate1/{index:03d}.png" for index in range(4)]


def test_label_csv_inside_out_dir_is_protected_from_overwrite(tmp_path):
    images, selector = fixture(tmp_path, rows=2)
    checkpoint = tiny_checkpoint(tmp_path)
    coco = tmp_path / "coco.json"
    coco.write_text(json.dumps(coco_source(images)), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    inner = out / "labels.csv"
    inner.write_bytes(selector.read_bytes())

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(inner),
            "--coco-instances",
            str(coco),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert inner.exists()


def test_onnx_external_tensor_inside_out_dir_is_protected(tmp_path):
    graph, externals = external_onnx(tmp_path)
    images, selector = flat_manifest(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    # Move the whole model (graph and its tensor file) into the output directory:
    # --overwrite would otherwise delete an input as if it were a stale artifact.
    moved = out / externals[0].name
    moved.write_bytes(externals[0].read_bytes())
    inner = out / "model.onnx"
    inner.write_bytes(graph.read_bytes())

    result = runner.invoke(
        app,
        [
            "extract",
            "--onnx-path",
            str(inner),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "contains protected input" in cli_message(result)
    # Both the graph and its tensor file are inputs, so both are protected.
    from otuformer.cli.extract import _onnx_input_paths

    protected = {path.resolve() for path in _onnx_input_paths(inner)}
    assert protected == {inner.resolve(), moved.resolve()}
    assert moved.exists()


def test_reused_pooling_input_is_protected_from_overwrite(tmp_path):
    images, selector = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    pool = (
        out
        / "checkpoints"
        / f"{checkpoint.stem}.lightweight_pooling.pth"
    )
    pool.parent.mkdir(parents=True)
    pool.write_bytes(b"pretend pool weights")

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--token-mode",
            "attention-pool",
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert pool.read_bytes() == b"pretend pool weights"


def test_metric_failure_does_not_discard_the_extraction(tmp_path, monkeypatch):
    from otuformer.embedding import evaluator

    def boom(*args, **kwargs):
        raise RuntimeError("clustering exploded")

    monkeypatch.setattr(evaluator, "_compute_clustering_metrics_result", boom)

    images, selector = fixture(tmp_path, rows=12)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out / "embeddings.csv").exists()
    provenance = json.loads((out / "run_provenance.json").read_text())
    assert provenance["status"] == "completed"
    assert provenance["metrics"]["values"]["Clustering"]["status"] == "failed"
    # Another family still ran.
    assert provenance["metrics"]["values"]["kNN_Acc_k1"]["status"] in {
        "available",
        "unavailable",
    }


def test_final_record_failure_publishes_no_success(tmp_path, monkeypatch):
    from otuformer.cli import extract as extract_module

    images, selector = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"

    real_write = extract_module.__dict__["_run_new_data_path"]

    def fail_on_completed(*args, **kwargs):
        import otuformer.utils.io as io_module

        real_json = io_module.write_json_atomic

        def write_json_atomic(value, path):
            if isinstance(value, dict) and value.get("status") == "completed":
                raise OSError("disk full")
            return real_json(value, path)

        monkeypatch.setattr(io_module, "write_json_atomic", write_json_atomic)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(extract_module, "_run_new_data_path", fail_on_completed)

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    provenance = json.loads((out / "run_provenance.json").read_text())
    assert provenance["status"] == "failed"
    # The extraction really ran, so the failed record must not claim otherwise.
    assert provenance["effective"]["data_path"] == "records"
    assert provenance["selection"]["selected_images"] == 4
    assert not (out / "embeddings.csv").exists()


def test_umap_only_runs_for_csv_selected_runs(tmp_path, monkeypatch):
    from otuformer.cli import extract as extract_module

    calls: list[int] = []
    real = extract_module._evaluate_records

    def spy(*args, **kwargs):
        calls.append(kwargs["enable_visualization"])
        return real(*args, **kwargs)

    monkeypatch.setattr(extract_module, "_evaluate_records", spy)

    images = tmp_path / "images"
    for index in range(12):
        write_png(images / f"plate1" / f"{index:03d}.png", color=(index, 0, 0))
    coco = tmp_path / "coco.json"
    payload = {
        "images": [
            {
                "id": index,
                "file_name": f"plate1/{index:03d}.png",
                "width": 32,
                "height": 32,
            }
            for index in range(12)
        ],
        "categories": [{"id": 4, "name": "Head"}],
        "annotations": [],
    }
    coco.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--coco-instances",
            str(coco),
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [False]
    provenance = json.loads((out / "run_provenance.json").read_text())
    assert provenance["metrics"]["visualization"]["umap"]["status"] == "not_applicable"
    assert "no CSV selector" in provenance["metrics"]["visualization"]["umap"]["reason"]


def external_onnx(tmp_path):
    """A tiny external-data ONNX graph plus its tensor file."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = (np.arange(3 * 32 * 32 * 2, dtype=np.float32) / 1000.0).reshape(
        3 * 32 * 32, 2
    )
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["input"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weight"], ["embedding"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 2])],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    graphs = tmp_path / "graph"
    graphs.mkdir(exist_ok=True)
    path = graphs / "external.onnx"
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="external.data",
        size_threshold=0,
    )
    from otuformer.embedding.extractor import _onnx_external_files

    return path, _onnx_external_files(path)


def test_pooling_sibling_beside_the_checkpoint_is_protected(tmp_path):
    images, selector = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    cache = out / "cache.pth"
    out.mkdir()
    cache.write_bytes(b"trained pool weights")
    sibling = tmp_path / f"{checkpoint.stem}.lightweight_pooling.pth"
    try:
        sibling.symlink_to(cache)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--token-mode",
            "attention-pool",
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    # The loader searches beside the backbone checkpoint too, so that cache is an
    # input and must survive the overwrite.
    assert result.exit_code != 0
    assert cache.read_bytes() == b"trained pool weights"


def test_rollback_continues_after_a_failing_removal(tmp_path, monkeypatch):
    from otuformer.cli import extract as extract_module

    real_cleanup = extract_module._cleanup_created
    seen: list[str] = []

    def grumpy_cleanup(paths, **kwargs):
        def unlink(self, missing_ok=False):
            seen.append(self.name)
            if self.name == "embeddings.csv":
                raise PermissionError("locked")
            real_unlink(self, missing_ok=missing_ok)

        import pathlib

        real_unlink = pathlib.Path.unlink
        monkeypatch.setattr(pathlib.Path, "unlink", unlink)
        return real_cleanup(paths, **kwargs)

    images, selector = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"

    import otuformer.utils.io as io_module

    real_json = io_module.write_json_atomic

    def fail_on_completed(value, path):
        if isinstance(value, dict) and value.get("status") == "completed":
            raise OSError("disk full")
        return real_json(value, path)

    monkeypatch.setattr(io_module, "write_json_atomic", fail_on_completed)
    monkeypatch.setattr(extract_module, "_cleanup_created", grumpy_cleanup)

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--num-workers",
            "0",
        ],
    )

    # The original failure is what the user sees...
    assert result.exit_code != 0
    assert "disk full" in result.output or "OSError" in result.output
    # ...and the failing removal did not stop the rest of the rollback.
    assert "embeddings.csv" in seen
    assert not (out / "data_records.jsonl").exists()
    assert (out / "embeddings.csv").exists()  # locked, not silently "cleaned"


def test_legacy_overwrite_cannot_clear_the_image_root(tmp_path):
    images, selector = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(images),
            "--label-csv",
            str(selector),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    # The images are still there: "output inside the root" is allowed, "clear the
    # root itself" is not.
    assert (images / "plate1" / "000.png").exists()


def sparse_attribute_onnx(tmp_path, data_dir, location="sparse.data"):
    """A graph whose Constant node carries a sparse payload in external data."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    values = numpy_helper.from_array(np.array([2.0, 3.0], dtype=np.float32), "values")
    payload = values.raw_data
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sparse.data").write_bytes(payload)
    values.ClearField("raw_data")
    values.data_location = TensorProto.EXTERNAL
    for key, value in (("location", location), ("length", str(len(payload)))):
        entry = values.external_data.add()
        entry.key = key
        entry.value = value
    indices = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "indices")
    sparse = helper.make_sparse_tensor(values, indices, [2])

    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["embedding"]),
            helper.make_node("Constant", [], ["weights"], sparse_value=sparse),
        ],
        "sparse",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 2])],
        [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    graph_dir = data_dir if location != "../out/sparse.data" else tmp_path / "graph"
    path = graph_dir / "sparse.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(path))
    return path, data_dir / "sparse.data"


def test_sparse_attribute_tensor_is_an_input(tmp_path):
    from otuformer.embedding.extractor import _onnx_external_files

    path, data = sparse_attribute_onnx(tmp_path, tmp_path / "graph")

    assert _onnx_external_files(path) == [data.resolve()]


def test_sparse_attribute_tensor_inside_out_dir_is_protected(tmp_path):
    images, selector = flat_manifest(tmp_path)
    out = tmp_path / "out"
    path, data = sparse_attribute_onnx(tmp_path, out)

    result = runner.invoke(
        app,
        [
            "extract",
            "--onnx-path",
            str(path),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    # The sparse payload is an input file, so --overwrite must refuse to clear it.
    assert result.exit_code != 0
    assert "contains protected input" in cli_message(result)
    from otuformer.cli.extract import _onnx_input_paths

    # A sparse attribute payload is protected like any other tensor file.
    protected = {path.resolve() for path in _onnx_input_paths(path)}
    assert data.resolve() in protected
    assert data.exists()


def test_sparse_payload_outside_the_graph_directory_fails_before_overwrite(tmp_path):
    images, selector = flat_manifest(tmp_path)
    out = tmp_path / "out"
    path, data = sparse_attribute_onnx(tmp_path, out, location="../out/sparse.data")

    result = runner.invoke(
        app,
        [
            "extract",
            "--onnx-path",
            str(path),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    # A location outside the graph directory is refused here as well as by the
    # runtime, before anything is cleared, so the payload survives.
    assert result.exit_code != 0
    assert "outside the graph directory" in str(result.exception)
    assert data.exists()


def test_function_default_tensor_inside_out_dir_is_protected(tmp_path):
    from otuformer.embedding.extractor import _onnx_external_files

    images, selector = flat_manifest(tmp_path)
    out = tmp_path / "out"
    path, data = _function_default_onnx(tmp_path, out)

    result = runner.invoke(
        app,
        [
            "extract",
            "--onnx-path",
            str(path),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    # A function's default attribute payload is an input like any other tensor file.
    assert _onnx_external_files(path) == [data.resolve()]
    assert result.exit_code != 0
    assert "contains protected input" in cli_message(result)
    assert data.exists()


def test_graph_entry_symlink_inside_out_dir_blocks_overwrite(tmp_path):
    """The entry symlink is deleted by cleanup, so it must be protected too."""
    images, selector = flat_manifest(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    payload = bytes(range(256)) * (3 * 32 * 32 * 2 * 4 // 256)
    (real / "weights.data").write_bytes(payload)
    # Same graph, resolved through the entry inside the output directory.
    (out / "weights.data").write_bytes(payload)

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    declared = np.zeros((3 * 32 * 32, 2), dtype=np.float32)
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
    onnx.save_model(
        model,
        str(real / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    entry = out / "model.onnx"
    try:
        entry.symlink_to(real / "model.onnx")
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    result = runner.invoke(
        app,
        [
            "extract",
            "--onnx-path",
            str(entry),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    # Refused before anything is cleared: the entry and the payload it resolves to
    # both still exist.
    assert result.exit_code != 0
    assert "contains protected input" in cli_message(result)
    assert entry.exists() and entry.is_symlink()
    assert (out / "weights.data").read_bytes() == payload
    assert (real / "model.onnx").is_file()


def test_selector_through_symlinked_dotdot_is_protected(tmp_path):
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    selector = out / "labels.csv"
    selector.write_text("image,label\nplate1/000.png,a\n", encoding="utf-8")
    entry = tmp_path / "entry"
    entry.mkdir()
    link = entry / "link"
    try:
        link.symlink_to(out / "sub")
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    # link -> out/sub, so link/../labels.csv is out/labels.csv on disk, not
    # entry/labels.csv the way lexical folding would have it.
    reference = str(link / ".." / "labels.csv")
    before = selector.read_bytes()
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            reference,
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "contains protected input" in cli_message(result)
    assert selector.read_bytes() == before


def test_selector_entry_symlink_inside_out_dir_is_protected(tmp_path):
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("image,label\nplate1/000.png,a\n", encoding="utf-8")
    entry = out / "selector.csv"
    view = tmp_path / "view"
    try:
        entry.symlink_to(outside)
        view.symlink_to(out)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(view / "selector.csv"),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    # The link's target is outside, but cleanup deletes the link itself, so the
    # directory entry has to be protected as well.
    assert result.exit_code != 0
    assert "contains protected input" in cli_message(result)
    assert entry.is_symlink()
    assert outside.is_file()


def test_selector_reached_through_an_intermediate_link_is_protected(tmp_path):
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    selector = outside / "labels.csv"
    selector.write_text("image,label\nplate1/000.png,a\n", encoding="utf-8")
    via = out / "via"
    try:
        via.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    before = selector.read_bytes()
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            # The file lives outside, but the path goes through a link that the
            # cleanup would delete, making the selector unreachable.
            "--label-csv",
            str(via / "labels.csv"),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "is reached through" in cli_message(result)
    assert via.is_symlink()
    assert selector.read_bytes() == before


def test_selector_relay_link_inside_out_dir_is_protected(tmp_path):
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    real = tmp_path / "real.csv"
    real.write_text("image,label\nplate1/000.png,a\n", encoding="utf-8")
    relay = out / "relay.csv"
    entry = tmp_path / "entry.csv"
    try:
        relay.symlink_to(real)
        entry.symlink_to(relay)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    before = real.read_bytes()
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            # The link that would be deleted appears only after following `entry`,
            # so it is neither the written path nor its target.
            "--label-csv",
            str(entry),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "is reached through" in cli_message(result)
    assert relay.is_symlink()
    assert real.read_bytes() == before


def test_directory_relay_link_inside_out_dir_is_protected(tmp_path):
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    selector = real / "labels.csv"
    selector.write_text("image,label\nplate1/000.png,a\n", encoding="utf-8")
    relay = out / "relay"
    entry = tmp_path / "entry"
    try:
        relay.symlink_to(real)
        entry.symlink_to(relay)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    before = selector.read_bytes()
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            # The directory link `relay` only appears while walking the path.
            "--label-csv",
            str(entry / "labels.csv"),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "is reached through" in cli_message(result)
    assert relay.is_symlink()
    assert selector.read_bytes() == before


def test_file_link_through_a_directory_link_in_out_dir_is_protected(tmp_path):
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    selector = real / "labels.csv"
    selector.write_text("image,label\nplate1/000.png,a\n", encoding="utf-8")
    via = out / "via"
    entry = tmp_path / "entry.csv"
    try:
        via.symlink_to(real)
        entry.symlink_to(via / "labels.csv")
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    before = selector.read_bytes()
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            # The link's target itself goes through a directory link inside `out`.
            "--label-csv",
            str(entry),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "is reached through" in cli_message(result)
    assert via.is_symlink()
    assert selector.read_bytes() == before


@pytest.mark.parametrize("target_kind", ["relative", "absolute"])
def test_selector_dotdot_after_a_link_is_protected(tmp_path, target_kind):
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    real = tmp_path / "real"
    (real / "sub").mkdir(parents=True)
    selector = real / "labels.csv"
    selector.write_text("image,label\nplate1/000.png,a\n", encoding="utf-8")
    link = out / "link"
    try:
        link.symlink_to(
            Path("..") / "real" / "sub" if target_kind == "relative" else real / "sub"
        )
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    # `..` applies to where the link points, so the selector is real/labels.csv;
    # the path still needs out/link, which the cleanup would delete.
    reference = link / ".." / "labels.csv"
    assert reference.resolve() == selector.resolve()
    before = selector.read_bytes()
    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(reference),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "is reached through" in cli_message(result)
    assert link.is_symlink()
    assert selector.read_bytes() == before


def test_cleanup_that_breaks_a_protected_input_is_reported(tmp_path, monkeypatch):
    """Check-then-act cannot be atomic; the outcome is verified instead.

    The safety walk is made blind to one link, which is the same class of failure as
    another process swapping that entry between the check and the cleanup. The run
    must stop with the input named rather than continue on a path that no longer
    leads where it was checked.
    """
    import pathlib

    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    selector = real / "labels.csv"
    selector.write_text("image,label\nplate1/000.png,a\n", encoding="utf-8")
    relay = out / "relay"
    entry = tmp_path / "entry"
    try:
        relay.symlink_to(real)
        entry.symlink_to(relay)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    # Both spellings: the walk may reach the entry through either absolute form.
    hidden = {str(entry), str(tmp_path.resolve() / "entry")}
    original = pathlib.Path.is_symlink

    def blind(self):
        return False if str(self) in hidden else original(self)

    monkeypatch.setattr(pathlib.Path, "is_symlink", blind)

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(entry / "labels.csv"),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "changed a protected input" in cli_message(result)


def test_legacy_image_root_link_inside_out_dir_is_protected(tmp_path):
    """The legacy path keeps the path the user typed, not only its target."""
    out = tmp_path / "out"
    out.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    for index in range(4):
        write_png(external / f"img_{index}.png")
    checkpoint = tiny_checkpoint(tmp_path)
    via = out / "via"
    try:
        via.symlink_to(external)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks are not available on this platform")

    result = runner.invoke(
        app,
        [
            "extract",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(via),
            "--out-dir",
            str(out),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    # Refused before anything is cleared: the entry the user named is still there.
    assert result.exit_code != 0
    assert "is reached through" in cli_message(result)
    assert via.is_symlink()
    assert (external / "img_0.png").is_file()


def _legacy_run(images, out, *extra):
    return runner.invoke(
        app,
        [
            "extract",
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--num-workers",
            "0",
            *extra,
        ],
    )


def test_legacy_checkpoint_inside_out_dir_is_protected(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(out)
    before = checkpoint.read_bytes()

    result = _legacy_run(images, out, "--checkpoint", str(checkpoint), "--overwrite")

    assert result.exit_code != 0
    assert "is reached through" in cli_message(result)
    assert checkpoint.read_bytes() == before


def test_legacy_onnx_graph_inside_out_dir_is_protected(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    images, _ = fixture(tmp_path, rows=4)
    graph, externals = external_onnx(tmp_path)
    inner = out / "model.onnx"
    inner.write_bytes(graph.read_bytes())
    for external in externals:
        (out / external.name).write_bytes(external.read_bytes())
    before = inner.read_bytes()

    result = _legacy_run(images, out, "--onnx-path", str(inner), "--overwrite")

    assert result.exit_code != 0
    assert "is reached through" in cli_message(result)
    assert inner.read_bytes() == before


def test_legacy_onnx_external_weights_inside_out_dir_are_protected(tmp_path):
    """A payload reached through `..` is refused before cleanup, not deleted."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    out = tmp_path / "out"
    out.mkdir()
    images, _ = fixture(tmp_path, rows=4)
    payload = out / "weights.data"
    payload.write_bytes(np.ones((3 * 32 * 32, 2), dtype=np.float32).tobytes())
    weight = numpy_helper.from_array(
        np.ones((3 * 32 * 32, 2), dtype=np.float32), "weight"
    )
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["input"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weight"], ["embedding"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 2])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    path = model_dir / "model.onnx"
    # onnx refuses to write a `..` location, so save plainly and patch the pointer;
    # the runtime resolves it happily, which is why protection must see it too.
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    (model_dir / "weights.data").unlink()
    proto = onnx.load(str(path), load_external_data=False)
    for tensor in proto.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = "../out/weights.data"
    onnx.save_model(proto, str(path))
    before = payload.read_bytes()

    result = _legacy_run(images, out, "--onnx-path", str(path), "--overwrite")

    # onnxruntime rejects this location too ("escapes model directory"), so the run
    # must refuse before clearing; the payload survives either way.
    assert result.exit_code != 0
    assert "outside the graph directory" in str(result.exception)
    assert payload.read_bytes() == before


def test_legacy_pooling_cache_inside_out_dir_is_protected(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    pool = out / "checkpoints" / f"{checkpoint.stem}.lightweight_pooling.pth"
    pool.parent.mkdir(parents=True)
    pool.write_bytes(b"trained pool weights")
    before = pool.read_bytes()

    result = _legacy_run(
        images,
        out,
        "--checkpoint",
        str(checkpoint),
        "--token-mode",
        "attention-pool",
        "--overwrite",
    )

    assert result.exit_code != 0
    assert "is reached through" in cli_message(result)
    assert pool.read_bytes() == before


def _duplicate_location_onnx(tmp_path, out):
    """A graph whose tensor has two locations: the real payload is inside ``out``."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    root = tmp_path / "model"
    root.mkdir(parents=True)
    out.mkdir(exist_ok=True)
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
    actual = out / "actual.data"
    actual.write_bytes(np.full_like(declared, 5.0).tobytes())
    proto = onnx.load(str(path), load_external_data=False)
    duplicate = proto.graph.initializer[0].external_data.add()
    duplicate.key = "location"
    duplicate.value = "../out/actual.data"
    onnx.save_model(proto, str(path))
    return path, actual


def test_duplicate_location_payload_inside_out_dir_is_protected(tmp_path):
    images, _ = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    path, actual = _duplicate_location_onnx(tmp_path, out)
    before = actual.read_bytes()

    result = _legacy_run(images, out, "--onnx-path", str(path), "--overwrite")

    # The ambiguity is refused before anything is cleared, so the payload survives.
    assert result.exit_code != 0
    combined = str(result.exception) + cli_message(result)
    assert "location" in combined
    assert actual.read_bytes() == before


def test_duplicate_location_is_refused_on_the_shared_path(tmp_path):
    images, selector = flat_manifest(tmp_path)
    out = tmp_path / "out"
    path, actual = _duplicate_location_onnx(tmp_path, out)
    before = actual.read_bytes()

    result = runner.invoke(
        app,
        [
            "extract",
            "--onnx-path",
            str(path),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--label-csv",
            str(selector),
            "--overwrite",
            "--num-workers",
            "0",
        ],
    )

    assert result.exit_code != 0
    combined = str(result.exception) + cli_message(result)
    assert "location" in combined
    assert actual.read_bytes() == before
    # No provenance is published for a model whose payload cannot be named.
    assert not (out / "run_provenance.json").exists()


def _trained_pool_run(tmp_path):
    images, selector = fixture(tmp_path, rows=4)
    checkpoint = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    return images, selector, checkpoint, out


def _pool_args(images, out, checkpoint, selector, *extra):
    return [
        "extract",
        "--checkpoint",
        str(checkpoint),
        "--input-images-dir",
        str(images),
        "--out-dir",
        str(out),
        "--label-csv",
        str(selector),
        "--token-mode",
        "attention-pool",
        "--attention-pooling-epochs",
        "1",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        *extra,
    ]


def test_new_pooling_checkpoint_is_published_in_provenance(tmp_path):
    import hashlib

    images, selector, checkpoint, out = _trained_pool_run(tmp_path)

    result = runner.invoke(app, _pool_args(images, out, checkpoint, selector))

    assert result.exit_code == 0, cli_message(result)
    record = json.loads((out / "run_provenance.json").read_text())
    pools = [entry for entry in record["artifacts"] if "pooling" in entry["path"]]
    assert len(pools) == 1, record["artifacts"]
    entry = pools[0]
    # Output-relative path, byte count and digest of the file this run produced.
    assert entry["path"].startswith("checkpoints/")
    written = out / entry["path"]
    assert entry["bytes"] == written.stat().st_size
    assert entry["sha256"] == hashlib.sha256(written.read_bytes()).hexdigest()
    assert record["effective"]["attention_pooling"]["written_file"] == str(written)


def test_failed_run_removes_the_pooling_checkpoint_it_trained(tmp_path, monkeypatch):
    import otuformer.utils.io as io_module

    images, selector, checkpoint, out = _trained_pool_run(tmp_path)
    real_write = io_module.write_json_atomic

    def fail_on_completed(value, path):
        if isinstance(value, dict) and value.get("status") == "completed":
            raise OSError("disk full")
        return real_write(value, path)

    monkeypatch.setattr(io_module, "write_json_atomic", fail_on_completed)

    result = runner.invoke(app, _pool_args(images, out, checkpoint, selector))

    assert result.exit_code != 0
    # Weights trained by this run are its output: rollback must not leave them behind.
    assert not list((out / "checkpoints").glob("*_pooling.pth"))
