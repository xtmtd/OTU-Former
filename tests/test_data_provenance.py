import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from otuformer.data.provenance import (
    build_data_records,
    build_run_record,
    build_validation_record,
    canonical_digest,
    canonical_json_bytes,
    data_content_digest,
    selection_metadata_digest,
    serialize_metric_outcomes,
)
from otuformer.data.reader import read_dataset
from otuformer.embedding.evaluator import MetricOutcome
from otuformer.utils.io import write_json_atomic


def write_png(path: Path, size=(8, 6), color=(10, 20, 30)) -> Path:
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


def make_bundle(tmp_path, rows=2):
    images = tmp_path / "images"
    for index in range(rows):
        write_png(images / "a" / f"{index:03d}.png", color=(index * 5, 0, 0))
    selector = write_csv(
        tmp_path / "sel.csv",
        ["image", "label", "sex"],
        [[f"a/{index:03d}.png", "NA" if index == 0 else "", ""] for index in range(rows)],
    )
    bundle = read_dataset(images, manifest=selector)
    bundle.raise_for_errors()
    frame = pd.DataFrame(
        {
            "id": [record.output_id for record in bundle.records],
            "sample": ["a"] * rows,
            "dim_0": [0.1] * rows,
            "dim_1": [0.2] * rows,
        }
    )
    return bundle, frame


def test_provenance_digests_and_mapping(tmp_path):
    bundle, frame = make_bundle(tmp_path)

    rows = build_data_records(bundle, frame)

    assert [row["embedding_row"] for row in rows] == [1, 2]
    assert [row["output_id"] for row in rows] == frame["id"].tolist()
    assert [row["image_key"] for row in rows] == ["a/000.png", "a/001.png"]
    assert all(row["schema_version"] == 1 for row in rows)
    assert rows[0]["metadata"]["label"] == "NA"  # literal token preserved
    assert rows[1]["metadata"]["label"] == ""  # present but empty
    assert rows[0]["image_sha256"] == bundle.records[0].image_file.sha256
    assert (rows[0]["width"], rows[0]["height"]) == (8, 6)
    assert rows[0]["coco_images"] == []
    # No annotation payload is duplicated into the mapping.
    assert "annotations" not in json.dumps(rows)

    # Reordering or dropping rows is refused rather than silently joined.
    with pytest.raises(ValueError, match="order does not match"):
        build_data_records(bundle, frame.iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="rows but the selection"):
        build_data_records(bundle, frame.iloc[:1])

    # Canonical bytes are pinned rather than derived from the pretty artifact.
    payload = {"b": 1, "a": [2, 3]}
    assert canonical_json_bytes(payload) == b'{"a":[2,3],"b":1}'
    assert canonical_digest(payload) == hashlib.sha256(b'{"a":[2,3],"b":1}').hexdigest()
    assert selection_metadata_digest(bundle) == canonical_digest(
        {
            "selector_columns": ["image", "label", "sex"],
            "records": [
                {
                    "image_key": record.image_key,
                    "metadata": dict(record.metadata),
                    "coco_images": [],
                }
                for record in bundle.records
            ],
        }
    )


def test_digests_distinguish_selection_from_content(tmp_path):
    bundle, _frame = make_bundle(tmp_path)
    baseline_selection = selection_metadata_digest(bundle)
    baseline_content = data_content_digest(bundle)

    # Root relocation keeps both logical digests.
    relocated = tmp_path / "moved"
    shutil.copytree(tmp_path / "images", relocated / "images")
    shutil.copy(tmp_path / "sel.csv", relocated / "sel.csv")
    moved = read_dataset(relocated / "images", manifest=relocated / "sel.csv")
    moved.raise_for_errors()
    assert selection_metadata_digest(moved) == baseline_selection
    assert data_content_digest(moved) == baseline_content

    # Changing selected bytes moves only the content digest.
    write_png(relocated / "images" / "a" / "000.png", color=(99, 1, 1))
    changed_bytes = read_dataset(relocated / "images", manifest=relocated / "sel.csv")
    changed_bytes.raise_for_errors()
    assert selection_metadata_digest(changed_bytes) == baseline_selection
    assert data_content_digest(changed_bytes) != baseline_content

    # Changing metadata moves only the selection digest.
    write_csv(
        relocated / "sel2.csv",
        ["image", "label", "sex"],
        [["a/000.png", "species", ""], ["a/001.png", "", ""]],
    )
    changed_metadata = read_dataset(relocated / "images", manifest=relocated / "sel2.csv")
    changed_metadata.raise_for_errors()
    assert selection_metadata_digest(changed_metadata) != baseline_selection


def test_validation_and_run_records(tmp_path):
    bundle, _frame = make_bundle(tmp_path)

    validation = build_validation_record(bundle)
    assert validation["schema_version"] == 1
    assert validation["status"] == "valid"
    # No mask check ran, so no tolerance is claimed.
    assert validation["methods"]["tolerances"]["status"] == "not_applicable"
    assert "polygon_rasterization" not in validation["methods"]
    assert validation["scope"]["unselected_file_and_geometry"] == "not_performed"
    assert validation["scope"]["custom_ignore_protocol"]["status"] == "not_implemented"
    assert validation["issues"] == []
    assert validation["counts"]["selected_images"] == 2

    outcomes = {
        "NMI": MetricOutcome(0.5, "available", None, {"family": "clustering", "n_clusters": 2}),
        "kNN_Acc_k1": MetricOutcome(None, "unavailable", "insufficient_class_count", {"family": "knn_accuracy"}),
    }
    metrics = serialize_metric_outcomes(outcomes)
    assert metrics["values"]["NMI"]["value"] == 0.5
    assert metrics["values"]["kNN_Acc_k1"]["reason"] == "insufficient_class_count"
    assert metrics["protocol"]["clustering"]["n_clusters"] == 2

    record = build_run_record(
        requested={"token_mode": "cls"},
        bundle=bundle,
        execution=None,
        evaluation=metrics,
        status="running",
        started_at="2026-10-29T00:00:00+00:00",
        ended_at=None,
        artifacts=[],
    )
    assert record["schema_version"] == 1
    assert record["status"] == "running"
    assert record["effective"]["status"] == "unavailable"
    assert record["thresholds"] == {
        "status": "not_applicable",
        "reason": "no_threshold_selected",
    }
    assert record["split"]["status"] == "not_provided"
    assert record["held_out_test"]["existing_protocol"].startswith("image-level")
    assert record["label_schema"]["observed_classes"] == ["NA"]
    assert record["label_schema"]["unlabeled_rows"] == 1
    assert record["selection"]["selection_metadata_sha256_v1"] == selection_metadata_digest(bundle)
    assert record["selection"]["data_content_sha256_v1"] == data_content_digest(bundle)
    assert record["package"]["git"]["reproducible_from_dirty_tree"] is False
    with pytest.raises(ValueError, match="unknown run status"):
        build_run_record(
            requested={},
            bundle=bundle,
            execution=None,
            evaluation={},
            status="done",
            started_at="now",
            ended_at=None,
            artifacts=[],
        )


def test_atomic_json_preserves_previous_file_on_interruption(tmp_path, monkeypatch):
    path = tmp_path / "run_provenance.json"
    write_json_atomic({"status": "running"}, path)
    assert json.loads(path.read_text()) == {"status": "running"}

    # NaN is rejected before the destination is touched.
    with pytest.raises(ValueError):
        write_json_atomic({"value": float("nan")}, path)
    assert json.loads(path.read_text()) == {"status": "running"}
    assert not (tmp_path / "run_provenance.json.tmp").exists()

    # A failure during replacement leaves the previous file intact.
    import otuformer.utils.io as io_module

    def boom(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(io_module.os, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        write_json_atomic({"status": "completed"}, path)
    assert json.loads(path.read_text()) == {"status": "running"}
    assert not (tmp_path / "run_provenance.json.tmp").exists()

    # Keyboard interruption during serialization is cleaned up too.
    monkeypatch.undo()

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(io_module.json, "dump", interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_json_atomic({"status": "completed"}, path)
    assert json.loads(path.read_text()) == {"status": "running"}
    assert not (tmp_path / "run_provenance.json.tmp").exists()


def test_git_provenance_uses_the_package_checkout(tmp_path, monkeypatch):
    from otuformer.data import provenance

    # A run started from an unrelated repository must not borrow its identity.
    monkeypatch.chdir(tmp_path)
    record = provenance._git_provenance()
    assert record["status"] in {"available", "unavailable"}
    if record["status"] == "available":
        assert record["root"] != str(tmp_path)
        assert (tmp_path / ".git").exists() is False
    else:
        assert record["reason"]


def test_validation_record_serializes_category_definitions_and_mask_paths(tmp_path):
    from otuformer.data.provenance import build_validation_record

    images = tmp_path / "images"
    write_png(images / "a" / "001.png", size=(1, 1))
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("RGB", (1, 1), (1, 0, 0)).save(masks / "one.png")
    payload = {
        "images": [{"id": 17, "file_name": "a/001.png", "width": 1, "height": 1}],
        "categories": [{"id": 4, "name": "Head", "isthing": 1}],
        "annotations": [
            {
                "image_id": 17,
                "file_name": "one.png",
                "segments_info": [
                    {"id": 1, "category_id": 4, "area": 1, "bbox": [0, 0, 1, 1], "iscrowd": 0}
                ],
            }
        ],
    }
    source = tmp_path / "panoptic.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    bundle = read_dataset(
        images, manifest=selector, coco_panoptic=source, panoptic_masks_dir=masks
    )
    bundle.raise_for_errors()
    validation = build_validation_record(bundle)

    category = validation["sources"][0]["categories"][0]
    assert category["definition"]["isthing"] == 1
    mask = validation["sources"][0]["masks"]["17"]
    assert mask["path"].endswith("one.png")
    assert mask["sha256"] == hashlib.sha256((masks / "one.png").read_bytes()).hexdigest()
    assert mask["bytes"] == (masks / "one.png").stat().st_size
    assert validation["methods"]["panoptic_decode"]
    assert "polygon_rasterization" not in validation["methods"]
    assert validation["methods"]["tolerances"]["rle_panoptic_pixel"] == 1e-6
