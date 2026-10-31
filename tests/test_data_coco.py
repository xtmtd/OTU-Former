import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from otuformer.data.reader import read_dataset


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


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def combined_source():
    return {
        "info": {"description": "tiny fixture"},
        "licenses": [],
        "images": [
            {
                "id": 17,
                "file_name": "a/001.png",
                "width": 8,
                "height": 6,
                "custom_flag": "keep-me",
            }
        ],
        "categories": [
            {
                "id": 4,
                "name": "Head",
                "supercategory": "body",
                "keypoints": ["left_tip", "right_tip"],
                "skeleton": [[1, 2]],
            },
            {"id": 9, "name": "Wing", "supercategory": "body"},
        ],
        "annotations": [
            {
                "id": 23,
                "image_id": 17,
                "category_id": 4,
                "bbox": [1, 1, 3, 2],
                "area": 6,
                "iscrowd": 0,
                "keypoints": [1, 1, 2, 3, 1, 1],
                "num_keypoints": 2,
            },
            {
                "id": 31,
                "image_id": 17,
                "category_id": 9,
                "bbox": [0, 0, 8, 6],
                "area": 48,
                "iscrowd": 0,
            },
        ],
    }


def test_coco_roles_and_selection_are_source_local(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    write_png(images / "b" / "001.png")
    combined = write_json(tmp_path / "ann" / "instances.json", combined_source())
    selector = write_csv(
        tmp_path / "sel.csv", ["image", "label"], [["a/001.png", "Species_A"], ["b/001.png", ""]]
    )

    bundle = read_dataset(
        images,
        manifest=selector,
        coco_instances=combined,
        coco_keypoints=combined,
    )
    bundle.raise_for_errors()

    assert [r.image_key for r in bundle.records] == ["a/001.png", "b/001.png"]
    assert [(ref.source_id, ref.image_id) for ref in bundle.records[0].coco_images] == [
        ("instances", 17),
        ("keypoints", 17),
    ]
    assert bundle.records[1].coco_images == ()
    assert bundle.selection_rule == "csv_rows"

    # One parse, two role-qualified indexes with the same payload bytes.
    assert set(bundle.sources) == {"instances", "keypoints"}
    assert bundle.sources["instances"].file.sha256 == bundle.sources["keypoints"].file.sha256
    assert bundle.sources["instances"].file.sha256 == hashlib.sha256(
        combined.read_bytes()
    ).hexdigest()

    # Extension fields and category order survive; categories stay source-local.
    assert bundle.sources["instances"].payload["images"][0]["custom_flag"] == "keep-me"
    assert list(bundle.sources["instances"].categories_by_id) == [4, 9]
    assert bundle.records[0].metadata["label"] == "Species_A"

    # No CSV: the union of source image keys, sorted lexically.
    second = copy.deepcopy(combined_source())
    second["images"] = [{"id": 5, "file_name": "b/001.png", "width": 8, "height": 6}]
    second["annotations"] = []
    keypoints = write_json(tmp_path / "ann" / "keypoints.json", second)
    union = read_dataset(images, coco_instances=combined, coco_keypoints=keypoints)
    union.raise_for_errors()
    assert [r.image_key for r in union.records] == ["a/001.png", "b/001.png"]
    assert union.selection_rule == "coco_union"


def test_coco_selection_does_not_expand_csv_rows(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    write_png(images / "b" / "001.png")
    source = write_json(tmp_path / "ann.json", combined_source())
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["b/001.png"]])

    bundle = read_dataset(images, manifest=selector, coco_instances=source)
    bundle.raise_for_errors()

    assert [r.image_key for r in bundle.records] == ["b/001.png"]
    assert bundle.records[0].coco_images == ()
    assert bundle.validation.counts["unused_source_images"] == {"instances": 1}


def test_coco_structure_scope(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])

    # An unselected image that is absent locally is not an error and is not read.
    payload = combined_source()
    payload["images"].append(
        {"id": 18, "file_name": "ghost/999.png", "width": 8, "height": 6}
    )
    clean = write_json(tmp_path / "clean.json", payload)
    bundle = read_dataset(images, manifest=selector, coco_instances=clean)
    bundle.raise_for_errors()
    assert [r.image_key for r in bundle.records] == ["a/001.png"]
    assert bundle.validation.counts["unused_source_images"] == {"instances": 1}
    assert "ghost/999.png" not in {r.image_key for r in bundle.records}

    def codes_for(mutate, name="bad.json"):
        broken = combined_source()
        mutate(broken)
        path = write_json(tmp_path / name, broken)
        result = read_dataset(images, manifest=selector, coco_instances=path)
        return {issue.code for issue in result.validation.by_severity("error")}

    def unselected_bad_area(payload):
        payload["annotations"][0]["area"] = "big"

    assert "invalid_area" in codes_for(unselected_bad_area, "area.json")

    # NaN cannot be serialized as strict JSON, so write the literal text instead.
    nan_path = tmp_path / "nan.json"
    nan_text = json.dumps(combined_source()).replace('"bbox": [1, 1, 3, 2]', '"bbox": [0, 0, NaN, 1]')
    nan_path.write_text(nan_text, encoding="utf-8")
    assert "invalid_coco_json" in {
        issue.code
        for issue in read_dataset(
            images, manifest=selector, coco_instances=nan_path
        ).validation.by_severity("error")
    }

    def duplicate_id(payload):
        payload["annotations"][1]["id"] = 23

    assert "duplicate_annotation_id" in codes_for(duplicate_id, "dupid.json")

    def dangling_reference(payload):
        payload["annotations"][0]["image_id"] = 99

    assert "dangling_image_reference" in codes_for(dangling_reference, "dangling.json")

    def missing_table(payload):
        del payload["categories"]

    assert "missing_coco_table" in codes_for(missing_table, "notable.json")

    def bool_id(payload):
        payload["images"][0]["id"] = True

    assert "invalid_image_id" in codes_for(bool_id, "boolid.json")

    # Duplicate JSON object keys are rejected instead of silently overwritten.
    dup_key_path = tmp_path / "dupkey.json"
    dup_key_path.write_text(
        '{"images": [], "images": [], "annotations": [], "categories": []}',
        encoding="utf-8",
    )
    assert "invalid_coco_json" in {
        issue.code
        for issue in read_dataset(
            images, manifest=selector, coco_instances=dup_key_path
        ).validation.by_severity("error")
    }

    # Conflicting dimensions across two sources for one image key.
    other = copy.deepcopy(combined_source())
    other["images"][0]["width"] = 9
    conflict = write_json(tmp_path / "conflict.json", other)
    assert "conflicting_dimensions" in {
        issue.code
        for issue in read_dataset(
            images,
            manifest=selector,
            coco_instances=clean,
            coco_keypoints=conflict,
        ).validation.by_severity("error")
    }

    # Declared dimensions that disagree with the decoded image.
    lying = copy.deepcopy(combined_source())
    lying["images"][0]["width"] = 7
    lying_path = write_json(tmp_path / "lying.json", lying)
    assert "declared_dimension_mismatch" in {
        issue.code
        for issue in read_dataset(
            images, manifest=selector, coco_instances=lying_path
        ).validation.by_severity("error")
    }


def test_box_and_keypoint_validation(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])

    def bundle_for(mutate):
        payload = combined_source()
        mutate(payload)
        path = write_json(tmp_path / f"{abs(hash(mutate))}.json", payload)
        return read_dataset(
            images, manifest=selector, coco_instances=path, coco_keypoints=path
        )

    clean = bundle_for(lambda payload: None)
    clean.raise_for_errors()
    assert [r.image_key for r in clean.records] == ["a/001.png"]
    annotations = clean.annotations_for("a/001.png", "instances")
    assert len(annotations) == 2
    # Custom ignore/crowd fields are preserved but not interpreted.
    assert "custom_flag" in clean.sources["instances"].payload["images"][0]

    def inclusive_edges(payload):
        payload["annotations"][1]["bbox"] = [0, 0, 8, 6]

    inclusive = bundle_for(inclusive_edges)
    assert inclusive.validation.valid

    def negative_extent(payload):
        payload["annotations"][1]["bbox"] = [1, 1, -2, 3]

    assert "negative_bbox_extent" in {
        i.code for i in bundle_for(negative_extent).validation.issues
    }

    def zero_extent(payload):
        payload["annotations"][1]["bbox"] = [1, 1, 0, 3]
        payload["annotations"][1]["area"] = 0

    zero = bundle_for(zero_extent)
    zero.raise_for_errors()  # information, not an error
    info = [i for i in zero.validation.issues if i.code == "zero_bbox_extent"]
    assert len(info) == 1 and info[0].severity == "information"

    def out_of_bounds(payload):
        payload["annotations"][1]["bbox"] = [7, 1, 3, 2]

    assert "bbox_out_of_bounds" in {
        i.code for i in bundle_for(out_of_bounds).validation.issues
    }

    def keypoint_at_width(payload):
        payload["annotations"][0]["keypoints"] = [8, 1, 2, 3, 1, 1]

    assert "keypoint_out_of_bounds" in {
        i.code for i in bundle_for(keypoint_at_width).validation.issues
    }

    def visible_at_last_pixel(payload):
        payload["annotations"][0]["keypoints"] = [7, 5, 2, 3, 1, 1]

    assert bundle_for(visible_at_last_pixel).validation.valid

    def invisible_keeps_coordinates(payload):
        payload["annotations"][0]["keypoints"] = [1, 1, 2, 6, 4, 1]

    assert bundle_for(invisible_keeps_coordinates).validation.valid

    def unlabelled_off_origin(payload):
        payload["annotations"][0]["keypoints"] = [1, 1, 2, 3, 1, 0]

    assert "unlabelled_keypoint_offset" in {
        i.code for i in bundle_for(unlabelled_off_origin).validation.by_severity("error")
    }

    def wrong_count(payload):
        payload["annotations"][0]["num_keypoints"] = 1

    assert "keypoint_count_mismatch" in {
        i.code for i in bundle_for(wrong_count).validation.by_severity("error")
    }

    def zero_based_skeleton(payload):
        payload["categories"][0]["skeleton"] = [[0, 1]]

    assert "invalid_skeleton" in {
        i.code for i in bundle_for(zero_based_skeleton).validation.by_severity("error")
    }

    def duplicate_names(payload):
        payload["categories"][0]["keypoints"] = ["tip", "tip"]

    assert "duplicate_keypoint_name" in {
        i.code for i in bundle_for(duplicate_names).validation.by_severity("error")
    }

    def blank_name(payload):
        payload["categories"][0]["keypoints"] = ["tip", " "]

    assert "invalid_keypoint_names" in {
        i.code for i in bundle_for(blank_name).validation.by_severity("error")
    }


def test_sources_are_never_rewritten(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    source = write_json(tmp_path / "ann.json", combined_source())
    before = source.read_bytes()

    bundle = read_dataset(images, coco_instances=source)
    bundle.raise_for_errors()

    assert source.read_bytes() == before
    assert bundle.sources["instances"].payload == json.loads(before)
    # Category IDs never become species labels.
    assert all("Head" not in record.metadata.values() for record in bundle.records)


def test_unreferenced_category_is_still_structurally_checked(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])

    # A keypoint category that no annotation references is still part of the
    # source, so its own definition is checked.
    payload = combined_source()
    payload["categories"].append(
        {"id": 77, "name": "Orphan", "keypoints": ["tip", "tip"], "skeleton": [[1, 2]]}
    )
    path = write_json(tmp_path / "orphan.json", payload)
    codes = {
        issue.code
        for issue in read_dataset(
            images, manifest=selector, coco_instances=path
        ).validation.by_severity("error")
    }
    assert "duplicate_keypoint_name" in codes

    # A malformed skeleton on the same orphan category is also an error.
    payload = combined_source()
    payload["categories"].append(
        {"id": 77, "name": "Orphan", "keypoints": ["tip", "base"], "skeleton": [[0, 1]]}
    )
    path = write_json(tmp_path / "orphan2.json", payload)
    codes = {
        issue.code
        for issue in read_dataset(
            images, manifest=selector, coco_instances=path
        ).validation.by_severity("error")
    }
    assert "invalid_skeleton" in codes


def test_panoptic_flags_must_be_numeric(tmp_path):
    import json as json_module

    images = tmp_path / "images"
    write_png(images / "a" / "001.png", size=(2, 1))
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    payload = {
        "images": [{"id": 17, "file_name": "a/001.png", "width": 2, "height": 1}],
        "categories": [{"id": 4, "name": "Head", "isthing": True}],
        "annotations": [
            {
                "image_id": 17,
                "file_name": "one.png",
                "segments_info": [
                    {"id": 1, "category_id": 4, "area": 1, "bbox": [0, 0, 1, 1], "iscrowd": True}
                ],
            }
        ],
    }
    path = tmp_path / "panoptic.json"
    path.write_text(json_module.dumps(payload), encoding="utf-8")
    masks = tmp_path / "masks"
    masks.mkdir()
    write_png(masks / "one.png", size=(2, 1))

    bundle = read_dataset(
        images, manifest=selector, coco_panoptic=path, panoptic_masks_dir=masks
    )
    codes = {issue.code for issue in bundle.validation.by_severity("error")}
    assert "invalid_isthing" in codes
    assert "invalid_iscrowd" in codes


def test_overflowing_number_literal_is_rejected(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    text = json.dumps(combined_source()).replace(
        '"bbox": [1, 1, 3, 2]', '"bbox": [1e999, 1, 3, 2]'
    )
    path = tmp_path / "overflow.json"
    path.write_text(text, encoding="utf-8")

    bundle = read_dataset(images, manifest=selector, coco_instances=path)

    assert "invalid_coco_json" in {
        issue.code for issue in bundle.validation.by_severity("error")
    }
