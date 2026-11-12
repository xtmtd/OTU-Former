"""Explicit spatial image selection must not change legacy selection.

Spatial image mode selects CSV rows, otherwise every supported image under the
root (recursively), while COCO sources attach annotations only. Legacy callers
keep the historical CSV > COCO-images > directory selector.
"""

from __future__ import annotations

import json
import os
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


def write_coco(path: Path, images, annotations=()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "images": images,
                "categories": [{"id": 4, "name": "Head", "supercategory": "part"}],
                "annotations": list(annotations),
            }
        ),
        encoding="utf-8",
    )
    return path


def keys(bundle) -> tuple[str, ...]:
    return tuple(record.image_key for record in bundle.records)


def test_spatial_csv_or_directory_preserves_order_and_annotation_scope(tmp_path):
    root = tmp_path / "images"
    write_png(root / "a.png")
    write_png(root / "b.png")
    write_png(root / "nested" / "c.png")
    coco = write_coco(
        tmp_path / "instances.json",
        [{"id": 7, "file_name": "a.png", "width": 8, "height": 6}],
        [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 4,
                "bbox": [1.0, 1.0, 3.0, 3.0],
                "area": 9.0,
                "iscrowd": 0,
            }
        ],
    )

    legacy = read_dataset(root, coco_instances=coco)
    legacy.raise_for_errors()
    assert keys(legacy) == ("a.png",)
    assert legacy.selection_rule == "coco_union"

    spatial = read_dataset(root, coco_instances=coco, selection_policy="csv-or-directory")
    spatial.raise_for_errors()
    assert keys(spatial) == ("a.png", "b.png", "nested/c.png")
    assert spatial.selection_rule == "directory_scan"
    # COCO still attaches by resolved key; it does not expand or narrow anything.
    assert spatial.annotations_for("a.png", "instances")
    assert spatial.annotations_for("b.png", "instances") == ()
    assert spatial.annotations_for("nested/c.png", "instances") == ()

    selector = write_csv(
        tmp_path / "figs.csv",
        ["image", "label"],
        [["nested/c.png", "NA"], ["a.png", ""]],
    )
    ordered = read_dataset(
        root, manifest=selector, coco_instances=coco, selection_policy="csv-or-directory"
    )
    ordered.raise_for_errors()
    assert keys(ordered) == ("nested/c.png", "a.png")
    assert ordered.selection_rule == "csv_rows"
    assert ordered.records[0].metadata == {"label": "NA"}
    assert ordered.records[1].metadata == {"label": ""}
    assert ordered.annotations_for("nested/c.png", "instances") == ()
    assert ordered.annotations_for("a.png", "instances")
    # Selecting with a CSV never changes legacy behaviour for other callers.
    assert keys(read_dataset(root, manifest=selector, coco_instances=coco)) == (
        "nested/c.png",
        "a.png",
    )


def test_spatial_directory_policy_keeps_root_and_nested_images(tmp_path):
    root = tmp_path / "images"
    write_png(root / "top.png")
    write_png(root / "plate" / "one.png")
    write_png(root / "plate" / "deep" / "two.png")

    bundle = read_dataset(root, selection_policy="csv-or-directory")
    bundle.raise_for_errors()
    assert keys(bundle) == ("plate/deep/two.png", "plate/one.png", "top.png")
    # Directory-derived sample metadata stays a reader fact, never a product group.
    assert bundle.records[0].sample == "plate"
    assert bundle.records[0].sample_origin == "first_directory"
    assert bundle.records[-1].sample is None
    assert bundle.records[-1].sample_origin == "none"


def test_spatial_policy_keeps_explicit_sample_and_aliases(tmp_path):
    root = tmp_path / "images"
    write_png(root / "plate" / "one.png")
    os.symlink(root / "plate" / "one.png", root / "alias.png")
    selector = write_csv(
        tmp_path / "figs.csv",
        ["image", "sample", "explicit_empty"],
        [["plate/one.png", "specimen-9", ""], ["alias.png", "", ""]],
    )

    bundle = read_dataset(root, manifest=selector, selection_policy="csv-or-directory")
    bundle.raise_for_errors()
    assert keys(bundle) == ("plate/one.png", "alias.png")
    assert bundle.records[0].sample == "specimen-9"
    assert bundle.records[0].sample_origin == "explicit"
    # An explicitly empty sample is a present-but-empty value, not inference.
    assert bundle.records[1].sample is None
    assert bundle.records[1].sample_origin == "explicit"
    # The alias addresses the same bytes but keeps its own key identity.
    assert bundle.records[0].image_file.sha256 == bundle.records[1].image_file.sha256


def test_spatial_policy_rejects_escapes_and_dotdot(tmp_path):
    root = tmp_path / "images"
    write_png(root / "a.png")
    write_png(tmp_path / "outside.png")
    escape = write_csv(tmp_path / "escape.csv", ["image"], [["../outside.png"]])
    dotdot = write_csv(tmp_path / "dotdot.csv", ["image"], [["sub/../a.png"]])

    escaped = read_dataset(root, manifest=escape, selection_policy="csv-or-directory")
    assert not escaped.validation.valid
    assert any(issue.code == "root_escape" for issue in escaped.validation.issues)

    # A folded in-root reference keeps the key of the file it actually reaches.
    folded = read_dataset(root, manifest=dotdot, selection_policy="csv-or-directory")
    folded.raise_for_errors()
    assert keys(folded) == ("a.png",)

    above = write_csv(tmp_path / "above.csv", ["image"], [["sub/../../outside.png"]])
    escaped_above = read_dataset(root, manifest=above, selection_policy="csv-or-directory")
    assert not escaped_above.validation.valid
    assert any(issue.code == "root_escape" for issue in escaped_above.validation.issues)

    symlinked = root / "linked.png"
    os.symlink(tmp_path / "outside.png", symlinked)
    escaped_dir = read_dataset(root, selection_policy="csv-or-directory")
    assert not escaped_dir.validation.valid
    assert any(issue.code == "root_escape" for issue in escaped_dir.validation.issues)


def test_spatial_policy_rejects_unreadable_image_and_empty_selection(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    broken = root / "broken.png"
    broken.write_bytes(b"not a png")

    bundle = read_dataset(root, selection_policy="csv-or-directory")
    assert not bundle.validation.valid
    assert any(issue.code == "image_decode_failed" for issue in bundle.validation.issues)

    empty = tmp_path / "empty"
    empty.mkdir()
    empty_bundle = read_dataset(empty, selection_policy="csv-or-directory")
    assert not empty_bundle.validation.valid
    assert any(issue.code == "empty_selection" for issue in empty_bundle.validation.issues)


def test_spatial_policy_survives_root_relocation(tmp_path):
    root = tmp_path / "images"
    write_png(root / "plate" / "one.png")
    write_png(root / "two.png")

    before = read_dataset(root, selection_policy="csv-or-directory")
    before.raise_for_errors()
    moved = tmp_path / "relocated"
    root.rename(moved)
    after = read_dataset(moved, selection_policy="csv-or-directory")
    after.raise_for_errors()
    assert keys(before) == keys(after)
    assert [record.image_file.sha256 for record in before.records] == [
        record.image_file.sha256 for record in after.records
    ]


def test_unknown_selection_policy_is_rejected(tmp_path):
    root = tmp_path / "images"
    write_png(root / "a.png")
    with pytest.raises(ValueError):
        read_dataset(root, selection_policy="anything-else")
