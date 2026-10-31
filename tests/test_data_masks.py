import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from otuformer.data.masks import decode_instance_mask, decode_panoptic_map
from otuformer.data.reader import read_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def instance_source(annotations, image_size=(8, 6)):
    return {
        "images": [
            {
                "id": 17,
                "file_name": "a/001.png",
                "width": image_size[0],
                "height": image_size[1],
            }
        ],
        "categories": [{"id": 4, "name": "Head"}],
        "annotations": annotations,
    }


def annotation(annotation_id=1, **overrides):
    base = {
        "id": annotation_id,
        "image_id": 17,
        "category_id": 4,
        "bbox": [0, 0, 8, 6],
        "area": 48,
        "iscrowd": 0,
    }
    base.update(overrides)
    return base


def fixture(tmp_path, annotations):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    source = write_json(tmp_path / "ann.json", instance_source(annotations))
    return images, selector, source


def test_polygon_union_preserves_geometry(tmp_path):
    parts = [
        [1.0, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 4.0],
        [5.0, 2.0, 7.0, 2.0, 7.0, 5.0],
    ]
    bow_tie = [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0, 4.0]
    repeated = [1.0, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0]
    thin = [0.0, 2.2, 7.7, 2.4, 7.7, 2.6, 0.0, 2.4]
    images, selector, source = fixture(
        tmp_path,
        [
            annotation(1, segmentation=[parts[0], parts[1]], area=20.0),
            annotation(2, segmentation=[bow_tie], area=0.0),
            annotation(3, segmentation=[repeated], area=9.0),
            # Continuous area differs from the raster by far more than 1%.
            annotation(4, segmentation=[thin], area=40.0),
        ],
    )
    before = source.read_bytes()

    bundle = read_dataset(images, manifest=selector, coco_instances=source)
    bundle.raise_for_errors()

    annotations = bundle.annotations_for("a/001.png", "instances")
    assert annotations[0]["segmentation"] == [parts[0], parts[1]]
    assert source.read_bytes() == before

    mask = decode_instance_mask(annotations[0], width=8, height=6)
    assert mask.shape == (6, 8) and mask.dtype == np.uint8
    first = decode_instance_mask({"segmentation": [parts[0]]}, 8, 6)
    second = decode_instance_mask({"segmentation": [parts[1]]}, 8, 6)
    assert np.array_equal(mask, np.clip(first.astype(int) + second.astype(int), 0, 1).astype(np.uint8))

    assert decode_instance_mask(annotations[1], 8, 6).sum() > 0  # bow-tie coverage
    assert not any(issue.severity == "error" for issue in bundle.validation.issues)
    assert not any("threshold" in issue.code for issue in bundle.validation.issues)

    diagnostics = {entry["annotation_id"]: entry for entry in bundle.validation.polygon_diagnostics}
    assert set(diagnostics) == {1, 2, 3, 4}
    assert diagnostics[4]["decoder"].startswith("pycocotools-")
    assert diagnostics[4]["area_delta"] > 1.0
    assert bundle.validation.counts["polygon_diagnostics"] == 4
    # A zero signed shoelace area is not a failure.
    assert "empty_decoded_mask" not in {
        issue.code for issue in bundle.validation.issues
    }


def test_mask_boundaries_and_pixel_policy(tmp_path):
    images, selector, source = fixture(
        tmp_path,
        [
            annotation(1, segmentation=[[4.0, 1.0, 8.0, 1.0, 8.0, 3.0, 4.0, 3.0]], area=8.0, bbox=[4, 1, 4, 2]),
            annotation(2, segmentation=[[1.0, 3.0, 4.0, 3.0, 4.0, 6.0, 1.0, 6.0]], area=9.0, bbox=[1, 3, 3, 3]),
            annotation(3, segmentation=[[0.0, 0.0, 8.0, 0.0, 8.0, 6.0, 0.0, 6.0]], area=48.0),
            annotation(4, segmentation=[[0.5, 0.5, 3.7, 0.3, 3.7, 2.6, 0.5, 2.6]], area=8.0, bbox=[0, 0, 4, 3]),
            # A valid RLE with no foreground decodes to an empty mask.
            annotation(5, segmentation={"size": [6, 8], "counts": [48]}, area=0.0, bbox=[0, 0, 0, 0]),
        ],
    )
    bundle = read_dataset(images, manifest=selector, coco_instances=source)
    bundle.raise_for_errors()

    annotations = bundle.annotations_for("a/001.png", "instances")
    for index, expected in enumerate([8, 9, 48]):
        mask = decode_instance_mask(annotations[index], 8, 6)
        assert mask.shape == (6, 8) and int(mask.sum()) == expected
    assert decode_instance_mask(annotations[4], 8, 6).sum() == 0
    assert "empty_decoded_mask" in {issue.code for issue in bundle.validation.issues}

    # Polygon differences stay informational; RLE differences are errors.
    assert not any(issue.code == "rle_area_mismatch" for issue in bundle.validation.issues)

    rle_fail = [annotation(6, segmentation={"size": [6, 8], "counts": [0, 47]}, area=47.0)]
    images2, selector2, source2 = fixture(tmp_path / "rle", rle_fail)
    broken = read_dataset(images2, manifest=selector2, coco_instances=source2)
    assert "rle_run_total_mismatch" in {
        issue.code for issue in broken.validation.by_severity("error")
    }

    wrong_size = [annotation(7, segmentation={"size": [6, 7], "counts": [0, 42]}, area=42.0)]
    images3, selector3, source3 = fixture(tmp_path / "size", wrong_size)
    assert "rle_size_mismatch" in {
        issue.code
        for issue in read_dataset(
            images3, manifest=selector3, coco_instances=source3
        ).validation.by_severity("error")
    }

    out_of_bounds = [
        annotation(8, segmentation=[[7.0, 1.0, 8.5, 1.0, 8.5, 3.0, 7.0, 3.0]])
    ]
    images4, selector4, source4 = fixture(tmp_path / "oob", out_of_bounds)
    assert "polygon_out_of_bounds" in {
        issue.code
        for issue in read_dataset(
            images4, manifest=selector4, coco_instances=source4
        ).validation.by_severity("error")
    }

    malformed = [annotation(9, segmentation={"size": [6, 8], "counts": "!!!"})]
    images5, selector5, source5 = fixture(tmp_path / "bad", malformed)
    assert {
        "invalid_rle",
        "rle_area_mismatch",
        "decode_failed",
    } & {
        issue.code
        for issue in read_dataset(
            images5, manifest=selector5, coco_instances=source5
        ).validation.by_severity("error")
    }


def panoptic_payload(file_name="image_001.png", segments=None):
    return {
        "images": [{"id": 17, "file_name": "a/001.png", "width": 2, "height": 2}],
        "categories": [{"id": 4, "name": "Head", "supercategory": "body", "isthing": 1}],
        "annotations": [
            {
                "image_id": 17,
                "file_name": file_name,
                "segments_info": segments
                or [{"id": 1, "category_id": 4, "area": 2, "bbox": [0, 0, 2, 1], "iscrowd": 0}],
            }
        ],
    }


def write_id_png(path: Path, ids) -> Path:
    array = np.asarray(ids, dtype=np.uint32)
    rgb = np.stack(
        [
            (array % 256).astype(np.uint8),
            ((array // 256) % 256).astype(np.uint8),
            ((array // (256 * 256)) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


def panoptic_fixture(tmp_path, ids=((1, 1), (0, 0)), **kwargs):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png", size=(2, 2))
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    masks = tmp_path / "masks"
    write_id_png(masks / "image_001.png", ids)
    source = write_json(tmp_path / "panoptic.json", panoptic_payload(**kwargs))
    return images, selector, source, masks


def test_panoptic_rgb_ids_and_root_layouts(tmp_path):
    images, selector, source, masks = panoptic_fixture(tmp_path)
    bundle = read_dataset(
        images,
        manifest=selector,
        coco_panoptic=source,
        panoptic_masks_dir=masks,
    )
    bundle.raise_for_errors()

    assert bundle.selection_rule == "csv_rows"
    assert bundle.sources["panoptic"].mask_files[17].size_bytes > 0
    decoded = decode_panoptic_map(
        bundle.sources["panoptic"].mask_files[17].path.read_bytes(), 2, 2
    )
    assert decoded.tolist() == [[1, 1], [0, 0]]
    # Void (0) is not a foreground segment and not an invented category.
    assert bundle.sources["panoptic"].payload["categories"][0]["isthing"] == 1

    # Segment ids with non-zero green/blue bytes encode as unsigned 24-bit ids.
    big_id = 1 + 256 * 2 + 65536 * 3
    images2, selector2, source2, masks2 = panoptic_fixture(
        tmp_path / "big",
        ids=((big_id, 0), (0, 0)),
        segments=[
            {"id": big_id, "category_id": 4, "area": 1, "bbox": [0, 0, 1, 1], "iscrowd": 0}
        ],
    )
    big = read_dataset(
        images2, manifest=selector2, coco_panoptic=source2, panoptic_masks_dir=masks2
    )
    big.raise_for_errors()
    assert decode_panoptic_map((masks2 / "image_001.png").read_bytes(), 2, 2)[0][0] == big_id

    # A prefixed file_name is resolved as written against the supplied root.
    images3, selector3, source3, masks3 = panoptic_fixture(
        tmp_path / "prefixed", file_name="panoptic_masks/image_001.png"
    )
    (masks3 / "panoptic_masks").mkdir()
    (masks3 / "image_001.png").rename(masks3 / "panoptic_masks" / "image_001.png")
    prefixed = read_dataset(
        images3, manifest=selector3, coco_panoptic=source3, panoptic_masks_dir=masks3
    )
    prefixed.raise_for_errors()

    # No prefix repair and no alternate-directory search.
    no_search = read_dataset(
        images3,
        manifest=selector3,
        coco_panoptic=source3,
        panoptic_masks_dir=masks3 / "panoptic_masks",
    )
    assert "missing_panoptic_mask" in {
        issue.code for issue in no_search.validation.by_severity("error")
    }

    # Palette PNGs are not segment-ID maps.
    palette = tmp_path / "palette"
    (palette / "masks").mkdir(parents=True)
    Image.fromarray(np.asarray([[0, 1], [1, 0]], dtype=np.uint8), mode="P").save(
        palette / "masks" / "image_001.png"
    )
    write_png(palette / "images" / "a" / "001.png", size=(2, 2))
    source4 = write_json(palette / "panoptic.json", panoptic_payload())
    palette_read = read_dataset(
        palette / "images",
        manifest=write_csv(palette / "sel.csv", ["image"], [["a/001.png"]]),
        coco_panoptic=source4,
        panoptic_masks_dir=palette / "masks",
    )
    assert "panoptic_not_rgb" in {
        issue.code for issue in palette_read.validation.by_severity("error")
    }

    # Missing / extra segments and area or bbox disagreements fail.
    extra = read_dataset(
        images,
        manifest=selector,
        coco_panoptic=source,
        panoptic_masks_dir=masks,
    )
    extra.raise_for_errors()

    images5, selector5, source5, masks5 = panoptic_fixture(
        tmp_path / "missing",
        ids=((1, 2), (0, 0)),
        segments=[
            {"id": 1, "category_id": 4, "area": 1, "bbox": [0, 0, 1, 1], "iscrowd": 0},
            {"id": 3, "category_id": 4, "area": 1, "bbox": [1, 0, 1, 1], "iscrowd": 0},
        ],
    )
    missing = read_dataset(
        images5, manifest=selector5, coco_panoptic=source5, panoptic_masks_dir=masks5
    )
    codes = {issue.code for issue in missing.validation.by_severity("error")}
    assert "panoptic_id_missing_from_json" in codes
    assert "panoptic_segment_missing_from_png" in codes

    images6, selector6, source6, masks6 = panoptic_fixture(
        tmp_path / "area",
        segments=[
            {"id": 1, "category_id": 4, "area": 3, "bbox": [0, 0, 2, 2], "iscrowd": 0}
        ],
    )
    wrong_area = read_dataset(
        images6, manifest=selector6, coco_panoptic=source6, panoptic_masks_dir=masks6
    )
    assert {"panoptic_area_mismatch", "panoptic_bbox_mismatch"} <= {
        issue.code for issue in wrong_area.validation.by_severity("error")
    }

    # A duplicate per-image record is a structural failure.
    duplicated = panoptic_payload()
    duplicated["annotations"].append(copy.deepcopy(duplicated["annotations"][0]))
    duplicate_source = write_json(tmp_path / "dup.json", duplicated)
    assert "duplicate_panoptic_record" in {
        issue.code
        for issue in read_dataset(
            images,
            manifest=selector,
            coco_panoptic=duplicate_source,
            panoptic_masks_dir=masks,
        ).validation.by_severity("error")
    }

    # The mask pair is required in both directions.
    assert "panoptic_pair_required" in {
        issue.code
        for issue in read_dataset(
            images, manifest=selector, coco_panoptic=source
        ).validation.by_severity("error")
    }
    assert "panoptic_pair_required" in {
        issue.code
        for issue in read_dataset(
            images, manifest=selector, panoptic_masks_dir=masks
        ).validation.by_severity("error")
    }


def test_panoptic_segment_ids_are_image_local(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png", size=(2, 2))
    write_png(images / "b" / "002.png", size=(2, 2))
    selector = write_csv(
        tmp_path / "sel.csv", ["image"], [["a/001.png"], ["b/002.png"]]
    )
    masks = tmp_path / "masks"
    write_id_png(masks / "one.png", ((1, 1), (0, 0)))
    write_id_png(masks / "two.png", ((0, 0), (1, 1)))
    payload = {
        "images": [
            {"id": 17, "file_name": "a/001.png", "width": 2, "height": 2},
            {"id": 18, "file_name": "b/002.png", "width": 2, "height": 2},
        ],
        "categories": [{"id": 4, "name": "Head", "isthing": 1}],
        "annotations": [
            {
                "image_id": 17,
                "file_name": "one.png",
                "segments_info": [
                    {"id": 1, "category_id": 4, "area": 2, "bbox": [0, 0, 2, 1], "iscrowd": 0}
                ],
            },
            {
                "image_id": 18,
                "file_name": "two.png",
                "segments_info": [
                    {"id": 1, "category_id": 4, "area": 2, "bbox": [0, 1, 2, 1], "iscrowd": 0}
                ],
            },
        ],
    }
    source = write_json(tmp_path / "panoptic.json", payload)

    bundle = read_dataset(
        images, manifest=selector, coco_panoptic=source, panoptic_masks_dir=masks
    )
    bundle.raise_for_errors()

    assert [record.image_key for record in bundle.records] == ["a/001.png", "b/002.png"]
    assert len(bundle.sources["panoptic"].annotations_by_image) == 2


def test_availability_is_not_absence(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    write_png(images / "b" / "002.png")
    selector = write_csv(
        tmp_path / "sel.csv", ["image"], [["a/001.png"], ["b/002.png"]]
    )
    payload = {
        "images": [{"id": 17, "file_name": "a/001.png", "width": 8, "height": 6}],
        "categories": [
            {
                "id": 4,
                "name": "Head",
                "keypoints": ["tip", "base"],
                "skeleton": [[1, 2]],
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 17,
                "category_id": 4,
                "bbox": [1, 1, 2, 2],
                "area": 4,
                "iscrowd": 1,
                "keypoints": [1, 1, 2, 0, 0, 0],
                "num_keypoints": 1,
                "ignore": True,
                "region_protocol": "custom-v1",
            }
        ],
    }
    source = write_json(tmp_path / "ann.json", payload)

    bundle = read_dataset(images, manifest=selector, coco_instances=source, coco_keypoints=source)
    bundle.raise_for_errors()

    # The image without a COCO record is present-but-unannotated, not a negative.
    assert bundle.validation.counts["images_without_annotations"] == 1
    assert bundle.validation.counts["keypoints_by_visibility"] == {0: 1, 1: 0, 2: 1}
    annotation = bundle.annotations_for("a/001.png", "instances")[0]
    assert annotation["iscrowd"] == 1
    assert annotation["ignore"] is True
    assert annotation["region_protocol"] == "custom-v1"
    assert not any("ignore" in issue.code for issue in bundle.validation.issues)


def test_optional_mask_dependency_is_lazy(tmp_path):
    script = f"""
import sys, types
sys.modules["pycocotools"] = None  # simulate a missing optional extra
from otuformer.data.reader import read_dataset
from otuformer.data.masks import MaskDependencyError
print("base-import-ok")
try:
    import pycocotools.mask  # noqa: F401
except ImportError:
    print("decoder-missing-ok")
else:
    raise SystemExit("expected the decoder to be unavailable")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert "base-import-ok" in result.stdout
    assert "decoder-missing-ok" in result.stdout


def test_decode_errors_reach_the_validation_report(tmp_path):
    # A corrupt panoptic file is a validation error, not an escaping exception.
    images = tmp_path / "images"
    write_png(images / "a" / "001.png", size=(2, 1))
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    masks = tmp_path / "masks"
    masks.mkdir()
    corrupt = masks / "broken.png"
    corrupt.write_bytes(b"not an image at all")
    source = write_json(
        tmp_path / "panoptic.json",
        {
            "images": [{"id": 17, "file_name": "a/001.png", "width": 2, "height": 1}],
            "categories": [{"id": 4, "name": "Head", "isthing": 1}],
            "annotations": [
                {
                    "image_id": 17,
                    "file_name": "broken.png",
                    "segments_info": [
                        {"id": 1, "category_id": 4, "area": 1, "bbox": [0, 0, 1, 1], "iscrowd": 0}
                    ],
                }
            ],
        },
    )
    bundle = read_dataset(
        images, manifest=selector, coco_panoptic=source, panoptic_masks_dir=masks
    )
    assert "panoptic_decode_failed" in {
        issue.code for issue in bundle.validation.by_severity("error")
    }

    # A non-ASCII compact RLE string is a decode error with one stable code.
    from otuformer.data.masks import MaskDecodeError, decode_instance_mask

    with pytest.raises(MaskDecodeError) as error:
        decode_instance_mask(
            {"segmentation": {"size": [1, 2], "counts": "é"}}, width=2, height=1
        )
    assert error.value.code == "invalid_rle_counts"


def test_panoptic_mask_must_really_be_png(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png", size=(2, 1))
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    masks = tmp_path / "masks"
    # An RGB BMP saved under a .png name satisfies "RGB" but not the format.
    masks.mkdir()
    Image.new("RGB", (2, 1), (1, 0, 0)).save(masks / "one.png", format="BMP")
    source = write_json(
        tmp_path / "panoptic.json",
        {
            "images": [{"id": 17, "file_name": "a/001.png", "width": 2, "height": 1}],
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
        },
    )
    bundle = read_dataset(
        images, manifest=selector, coco_panoptic=source, panoptic_masks_dir=masks
    )
    assert "panoptic_not_png" in {
        issue.code for issue in bundle.validation.by_severity("error")
    }


def test_out_of_bounds_polygon_never_reaches_the_rasterizer(tmp_path, monkeypatch):
    from otuformer.data import reader as reader_module

    calls: list[object] = []
    real = reader_module.rasterized_diagnostics

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(reader_module, "rasterized_diagnostics", spy)

    images = tmp_path / "images"
    write_png(images / "a" / "001.png")
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    source = write_json(
        tmp_path / "coco.json",
        instance_source(
            [annotation(segmentation=[[-1, 0, 4, 0, 4, 4, -1, 4]])]
        ),
    )

    bundle = read_dataset(images, manifest=selector, coco_instances=source)

    assert "polygon_out_of_bounds" in {
        issue.code for issue in bundle.validation.by_severity("error")
    }
    assert calls == []
    assert bundle.validation.counts["geometry_rejected_before_rasterization"] == 1


def test_panoptic_void_pixels_are_counted(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png", size=(2, 2))
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    masks = tmp_path / "masks"
    masks.mkdir()
    write_id_png(masks / "one.png", ((1, 0), (0, 0)))
    source = write_json(
        tmp_path / "panoptic.json",
        {
            "images": [{"id": 17, "file_name": "a/001.png", "width": 2, "height": 2}],
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
        },
    )
    bundle = read_dataset(
        images, manifest=selector, coco_panoptic=source, panoptic_masks_dir=masks
    )
    bundle.raise_for_errors()

    # A void map must not read as "nothing survived validation".
    assert bundle.validation.counts["panoptic_void_pixels"] == 3


def test_void_count_does_not_re_decode_after_validation(tmp_path):
    images = tmp_path / "images"
    write_png(images / "a" / "001.png", size=(2, 2))
    selector = write_csv(tmp_path / "sel.csv", ["image"], [["a/001.png"]])
    masks = tmp_path / "masks"
    masks.mkdir()
    write_id_png(masks / "one.png", ((1, 0), (0, 0)))
    source = write_json(
        tmp_path / "panoptic.json",
        {
            "images": [{"id": 17, "file_name": "a/001.png", "width": 2, "height": 2}],
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
        },
    )
    bundle = read_dataset(
        images, manifest=selector, coco_panoptic=source, panoptic_masks_dir=masks
    )
    bundle.raise_for_errors()
    assert bundle.validation.counts["panoptic_void_pixels"] == 3
    assert bundle.validation.scope["panoptic_void"]["status"] == "available"

    from otuformer.data.provenance import build_validation_record

    # Corrupting the mask after validation cannot turn a real count into zero.
    (masks / "one.png").write_bytes(b"not an image")
    bundle.raise_for_errors()
    assert bundle.validation.counts["panoptic_void_pixels"] == 3
    assert build_validation_record(bundle)["counts"]["panoptic_void_pixels"] == 3


def _panoptic_bundle(tmp_path, images, masks):
    """Build a bundle; ``masks`` maps image id to (file name, ids or None)."""
    images_dir = tmp_path / "images"
    directory = tmp_path / "masks"
    directory.mkdir(exist_ok=True)
    for image in images:
        write_png(images_dir / image["file_name"], size=(2, 2))
    for file_name, ids in masks.values():
        if ids is not None:
            write_id_png(directory / file_name, ids)
    source = write_json(
        tmp_path / "panoptic.json",
        {
            "images": images,
            "categories": [{"id": 4, "name": "Head", "isthing": 1}],
            "annotations": [
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "segments_info": [
                        {
                            "id": 1,
                            "category_id": 4,
                            "area": 1,
                            "bbox": [0, 0, 1, 1],
                            "iscrowd": 0,
                        }
                    ],
                }
                for image_id, (file_name, _ids) in masks.items()
            ],
        },
    )
    selector = write_csv(
        tmp_path / "sel.csv", ["image"], [[image["file_name"]] for image in images]
    )
    return read_dataset(
        images_dir,
        manifest=selector,
        coco_panoptic=source,
        panoptic_masks_dir=directory,
    )


def test_void_total_is_not_published_as_complete_when_a_mask_is_missing(tmp_path):
    bundle = _panoptic_bundle(
        tmp_path,
        [
            {"id": 17, "file_name": "a/001.png", "width": 2, "height": 2},
            {"id": 18, "file_name": "a/002.png", "width": 2, "height": 2},
        ],
        {17: ("one.png", ((1, 0), (0, 0))), 18: ("two.png", None)},
    )

    assert not bundle.validation.valid
    scope = bundle.validation.scope["panoptic_void"]
    # One of the two expected masks never reached the count, so the total is a
    # lower bound and must not be advertised as available.
    assert scope["status"] != "available"
    assert scope["masks_counted"] == 1
    assert scope["masks_expected"] == 2


def test_void_total_is_not_published_as_complete_after_snapshot_drift(tmp_path):
    from otuformer.data import reader as reader_module
    from otuformer.data.provenance import SnapshotDriftError

    real_read = reader_module.read_verified_bytes

    def drifting_read(snapshot):
        raise SnapshotDriftError(snapshot, "content changed")

    reader_module.read_verified_bytes = drifting_read
    try:
        bundle = _panoptic_bundle(
            tmp_path,
            [{"id": 17, "file_name": "a/001.png", "width": 2, "height": 2}],
            {17: ("one.png", ((1, 0), (0, 0)))},
        )
    finally:
        reader_module.read_verified_bytes = real_read

    assert "input_drift" in {issue.code for issue in bundle.validation.issues}
    scope = bundle.validation.scope["panoptic_void"]
    # Nothing reached the count, so the reported number is not a complete total.
    assert scope["status"] == "unavailable"
    assert scope["masks_counted"] == 0
    assert scope["masks_expected"] == 1


def test_decode_silences_only_the_pycocotools_deprecation():
    import warnings

    from otuformer.data.masks import decode_instance_mask, quiet_pycocotools_decode

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask = decode_instance_mask(
            {"segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]]}, 4, 4
        )

    assert mask.sum() > 0
    # pycocotools 2.0.11's NumPy 2 ``__array__`` deprecation is silenced here...
    assert not [w for w in caught if "__array__" in str(w.message)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quiet_pycocotools_decode():
            warnings.warn("unrelated deprecation", DeprecationWarning, stacklevel=1)
            warnings.warn("unrelated user warning", UserWarning, stacklevel=1)

    # ...and nothing else is: the filter is message-specific, not a blanket mute.
    assert sorted(str(w.message) for w in caught) == [
        "unrelated deprecation",
        "unrelated user warning",
    ]
