"""Annotation mapping through one realized geometry."""

from __future__ import annotations

import numpy as np
import pytest

from otuformer.vision.geometry import (
    GeometryOp,
    apply_geometry,
    trace_operations,
    transform_annotations,
)

CATEGORIES = {
    4: {"id": 4, "name": "Head", "supercategory": "part", "keypoints": ["left", "right"]}
}


def identity_trace(width: int = 100, height: int = 100):
    return trace_operations((width, height), [])


def resize_trace(width: int, height: int, to_width: int, to_height: int):
    return trace_operations(
        (width, height),
        [
            GeometryOp(
                "resize",
                {
                    "from_wh": [width, height],
                    "to_wh": [to_width, to_height],
                    "interpolation": "nearest",
                    "antialias": False,
                },
            )
        ],
    )


def rotation_trace(angle: float, size: int = 100):
    return trace_operations(
        (size, size),
        [
            GeometryOp(
                "rotate",
                {
                    "from_wh": [size, size],
                    "angle": angle,
                    "center": [size / 2.0, size / 2.0],
                    "expand": False,
                    "interpolation": "nearest",
                    "fill": [0, 0, 0],
                    "canvas_wh": [size, size],
                },
            )
        ],
    )


def keypoint_annotation(*points: tuple[float, float, int]) -> dict:
    values: list[float] = []
    for x, y, visibility in points:
        values.extend([float(x), float(y), float(visibility)])
    return {
        "id": 11,
        "image_id": 7,
        "category_id": 4,
        "keypoints": values,
        "num_keypoints": sum(1 for p in points if p[2] in (1, 2)),
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "area": 100.0,
        "iscrowd": 0,
    }


def test_coco_point_adapter_shifts_labelled_points_and_keeps_the_sentinel():
    (record,) = transform_annotations(
        [keypoint_annotation((0.0, 0.0, 2), (0.0, 0.0, 0))],
        CATEGORIES,
        identity_trace(),
        role="keypoints",
    )
    assert record["coordinate_adapter"] == "coco-pixel-index-to-edge-v1"
    assert record["keypoints"] == [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
    assert record["keypoints_in_view"] == [True, False]
    # Source visibility is untouched by the adapter.
    assert record["source"]["keypoints"][2] == 2.0


def test_point_adapter_round_trips_and_handles_fractional_and_last_pixel():
    trace = resize_trace(10, 10, 20, 20)
    (record,) = transform_annotations(
        [keypoint_annotation((4.5, 9.0, 2), (9.0, 9.0, 1))],
        CATEGORIES,
        trace,
        role="keypoints",
    )
    mapped = record["keypoints"]
    # Internal edge coordinate (x + 0.5) scaled by 2, then shifted back: 9.5 and 18.5.
    assert mapped[0][:2] == pytest.approx([9.5, 18.5])
    assert mapped[1][:2] == pytest.approx([18.5, 18.5])
    # Exact inverse through the trace recovers the source convention.
    from otuformer.vision.geometry import map_points

    back = map_points(
        trace,
        np.array([[value + 0.5 for value in mapped[0][:2]]]),
        inverse=True,
    )[0]
    assert back == pytest.approx([5.0, 9.5])


def test_out_of_view_is_not_the_same_as_source_visibility():
    trace = resize_trace(100, 100, 10, 10)
    (record,) = transform_annotations(
        [keypoint_annotation((90.0, 90.0, 2), (1.0, 1.0, 2))],
        CATEGORIES,
        trace,
        role="keypoints",
    )
    assert record["source"]["keypoints"][2] == 2.0  # source visibility unchanged
    assert record["keypoints_in_view"] == [True, True]
    assert record["keypoints"][0][:2] == pytest.approx([8.55, 8.55])

    offset = trace_operations(
        (100, 100),
        [GeometryOp("crop", {"from_wh": [100, 100], "left": 0, "top": 0, "width": 10, "height": 10})],
    )
    outside = keypoint_annotation((90.0, 90.0, 2))
    outside["bbox"] = [60.0, 60.0, 20.0, 20.0]  # fully outside the 10x10 crop
    (cropped,) = transform_annotations([outside], CATEGORIES, offset, role="keypoints")
    assert cropped["keypoints_in_view"] == [False]
    assert cropped["source"]["keypoints"][2] == 2.0
    assert cropped["annotation_in_view"] is False


def test_rotated_box_exposes_polygon_and_enclosure_separately():
    trace = rotation_trace(45.0)
    annotation = {
        "id": 3,
        "image_id": 7,
        "category_id": 4,
        "bbox": [25.0, 25.0, 20.0, 20.0],
        "area": 400.0,
        "iscrowd": 0,
        "segmentation": [[25.0, 25.0, 45.0, 25.0, 45.0, 45.0, 25.0, 45.0]],
    }
    (record,) = transform_annotations([annotation], CATEGORIES, trace, role="instances")
    assert record["source"]["bbox"] == [25.0, 25.0, 20.0, 20.0]  # source preserved
    assert record["polygon_xy"] is not None and record["bbox_xyxy"] is not None
    assert len(record["polygon_xy"]) >= 4
    # Retained area comes from the clipped polygon, not from the enclosure.
    polygon_area = 400.0
    x0, y0, x1, y1 = record["bbox_xyxy"]
    enclosure_area = (x1 - x0) * (y1 - y0)
    assert record["area_retained"] == pytest.approx(polygon_area, rel=1e-6)
    assert enclosure_area > polygon_area * 1.2
    assert record["annotation_in_view"] is True
    assert record["bbox_xy"][2] == pytest.approx(x1 - x0)


def test_rotated_box_retained_area_shrinks_when_clipped():
    trace = rotation_trace(45.0)
    annotation = {
        "id": 4,
        "image_id": 7,
        "category_id": 4,
        "bbox": [-20.0, -20.0, 40.0, 40.0],
        "area": 1600.0,
        "iscrowd": 0,
    }
    (record,) = transform_annotations([annotation], CATEGORIES, trace, role="instances")
    # The corner box mostly rotates out of the canvas; retained < source.
    assert record["area_retained"] is not None
    assert 0.0 < record["area_retained"] < 1600.0
    assert record["annotation_in_view"] is True


def test_categorical_mask_keeps_ids_void_and_crowd():
    raster = np.zeros((10, 10), dtype=np.uint8)
    raster[2:8, 2:8] = 5  # a real panoptic/category id
    raster[0, 0] = 255  # void
    annotation = {
        "id": 9,
        "image_id": 7,
        "category_id": 4,
        "iscrowd": 1,
        "raster": raster,
    }
    (record,) = transform_annotations(
        [annotation], CATEGORIES, resize_trace(10, 10, 5, 5), role="panoptic"
    )
    mask = record["mask"]
    assert mask.shape == (5, 5)
    assert set(np.unique(mask)).issubset({0, 5, 255})
    # Category ids are sampled with nearest neighbours: no averaged intensities.
    assert 5 in np.unique(mask)
    assert record["iscrowd"] == 1
    assert record["mask_support"] is not None and record["mask_support"].all()


def test_missing_annotation_stays_missing():
    annotation = {"id": 12, "image_id": 7, "category_id": 4, "segmentation": [], "area": 0.0}
    (record,) = transform_annotations([annotation], CATEGORIES, identity_trace(), role="instances")
    assert record["bbox_xy"] is None and record["polygon_xy"] is None
    assert record["annotation_in_view"] is None
    assert record["area_retained"] is None
    assert record["mask"] is None


def test_rle_mask_is_reported_unavailable_rather_than_fabricated():
    annotation = {
        "id": 13,
        "image_id": 7,
        "category_id": 4,
        "segmentation": {"size": [10, 10], "counts": "abc"},
    }
    (record,) = transform_annotations(
        [annotation], CATEGORIES, identity_trace(10, 10), role="instances"
    )
    assert record["mask"] is None
    assert record["mask_unavailable"] == "rle_mask_requires_explicit_decode"


def test_polygon_annotation_maps_through_the_realized_geometry():
    annotation = {
        "id": 21,
        "image_id": 7,
        "category_id": 4,
        "bbox": [10.0, 10.0, 20.0, 20.0],
        "segmentation": [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]],
        "area": 400.0,
    }
    (record,) = transform_annotations(
        [annotation], CATEGORIES, resize_trace(50, 50, 100, 100), role="instances"
    )
    assert record["area_source"] == pytest.approx(400.0)
    assert record["area_retained"] == pytest.approx(1600.0)
    xs = sorted({round(point[0], 6) for point in record["polygon_xy"]})
    assert xs[0] == pytest.approx(20.0) and xs[-1] == pytest.approx(60.0)


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="unknown annotation role"):
        transform_annotations([], CATEGORIES, identity_trace(), role="detection")
