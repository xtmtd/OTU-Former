"""COCO source loading, source-local indexing and annotation validation.

Structure and numeric payloads are validated source-wide; geometry against the
actual image is validated only for selected images. Nothing here renumbers
categories, rewrites the payload or interprets custom ignore protocols.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import snapshot_bytes
from .records import FileSnapshot, SourceIndex, ValidationIssue

ROLE_INSTANCES = "instances"
ROLE_KEYPOINTS = "keypoints"
ROLE_PANOPTIC = "panoptic"
SOURCE_ROLES = (ROLE_INSTANCES, ROLE_KEYPOINTS, ROLE_PANOPTIC)


class CocoSourceError(ValueError):
    """Malformed source: the reader converts these to report errors."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        first = issues[0].message if issues else "invalid COCO source"
        super().__init__(first)
        self.issues = issues


def _issue(code: str, message: str, *, source_id: str, reference=None, **details
           ) -> ValidationIssue:
    return ValidationIssue(
        severity="error",
        code=code,
        message=message,
        source_id=source_id,
        reference=reference,
        details=details,
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _is_flag(value: Any) -> bool:
    """Numeric COCO 0/1; JSON ``true``/``false`` is not the same thing."""
    return _is_int(value) and value in (0, 1)


def _check_category(category: dict[str, Any], source_id: str) -> list[ValidationIssue]:
    """Structural checks that apply even when no object references a category.

    A category nobody annotates is still part of the source, so a malformed
    keypoint definition or skeleton is a source-wide error rather than dead data.
    """
    issues: list[ValidationIssue] = []
    category_id = category.get("id")
    names = category.get("keypoints")
    if names is not None:
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name.strip() for name in names)
        ):
            issues.append(
                _issue(
                    "invalid_keypoint_names",
                    f"keypoint category {category_id} needs nonempty point names",
                    source_id=source_id,
                    reference=category_id,
                )
            )
            return issues
        if len(set(names)) != len(names):
            issues.append(
                _issue(
                    "duplicate_keypoint_name",
                    f"keypoint category {category_id} repeats a point name",
                    source_id=source_id,
                    reference=category_id,
                )
            )
        skeleton = category.get("skeleton")
        if skeleton is not None and (
            not isinstance(skeleton, list)
            or not all(
                isinstance(edge, list)
                and len(edge) == 2
                and all(_is_int(point) and 1 <= point <= len(names) for point in edge)
                for edge in skeleton
            )
        ):
            issues.append(
                _issue(
                    "invalid_skeleton",
                    f"keypoint category {category_id} has an invalid one-based skeleton",
                    source_id=source_id,
                    reference=category_id,
                )
            )
    return issues


def load_coco_source(path: Path) -> tuple[dict[str, Any], FileSnapshot]:
    """Read and strictly parse one COCO JSON document."""
    try:
        data, snapshot = snapshot_bytes(Path(path))
    except OSError as exc:
        raise CocoSourceError(
            [
                ValidationIssue(
                    severity="error",
                    code="unreadable_coco_source",
                    message=f"cannot read COCO source: {exc}",
                )
            ]
        ) from exc
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CocoSourceError(
            [
                ValidationIssue(
                    severity="error",
                    code="coco_source_not_utf8",
                    message=f"COCO source is not UTF-8: {exc}",
                )
            ]
        ) from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
        )
    except ValueError as exc:
        raise CocoSourceError(
            [
                ValidationIssue(
                    severity="error",
                    code="invalid_coco_json",
                    message=f"COCO source is not valid JSON: {exc}",
                )
            ]
        ) from exc
    if not isinstance(payload, dict):
        raise CocoSourceError(
            [
                ValidationIssue(
                    severity="error",
                    code="invalid_coco_structure",
                    message="COCO source must be a JSON object",
                )
            ]
        )
    return payload, snapshot


def _reject_duplicate_keys(pairs):
    seen = set()
    for key, _value in pairs:
        if key in seen:
            raise ValueError(f"duplicate object key: {key!r}")
        seen.add(key)
    return dict(pairs)


def _reject_constant(text: str):
    raise ValueError(f"non-finite number literal: {text}")


def _reject_float(text: str):
    """Reject finite-looking literals that overflow to inf, e.g. ``1e999``."""
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"number literal is not finite: {text}")
    return value


def index_coco(
    source_id: str, payload: dict[str, Any], file: FileSnapshot
) -> SourceIndex:
    """Index one source, raising ``CocoSourceError`` for structural problems."""
    if source_id not in SOURCE_ROLES:
        raise ValueError(f"unknown COCO role: {source_id!r}")
    issues: list[ValidationIssue] = []

    tables = {}
    for name in ("images", "annotations", "categories"):
        value = payload.get(name)
        if not isinstance(value, list):
            issues.append(
                _issue(
                    "missing_coco_table",
                    f"COCO source requires a '{name}' array",
                    source_id=source_id,
                )
            )
        tables[name] = value if isinstance(value, list) else []

    images_by_id: dict[int, dict[str, Any]] = {}
    for image in tables["images"]:
        if not isinstance(image, dict):
            issues.append(_issue("invalid_coco_image", "image record must be an object", source_id=source_id))
            continue
        image_id = image.get("id")
        if not _is_int(image_id):
            issues.append(_issue("invalid_image_id", "image 'id' must be an integer", source_id=source_id, reference=image_id))
            continue
        for field in ("width", "height"):
            if not _is_int(image.get(field)) or image[field] <= 0:
                issues.append(
                    _issue(
                        "invalid_image_dimensions",
                        f"image {image_id} needs a positive integer '{field}'",
                        source_id=source_id,
                        reference=image_id,
                    )
                )
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            issues.append(
                _issue(
                    "invalid_file_name",
                    f"image {image_id} needs a nonempty 'file_name'",
                    source_id=source_id,
                    reference=image_id,
                )
            )
        if image_id in images_by_id:
            issues.append(
                _issue(
                    "duplicate_image_id",
                    f"image id {image_id} appears more than once",
                    source_id=source_id,
                    reference=image_id,
                )
            )
        images_by_id[image_id] = image

    categories_by_id: dict[int, dict[str, Any]] = {}
    for category in tables["categories"]:
        if not isinstance(category, dict):
            issues.append(_issue("invalid_category", "category record must be an object", source_id=source_id))
            continue
        category_id = category.get("id")
        if not _is_int(category_id):
            issues.append(_issue("invalid_category_id", "category 'id' must be an integer", source_id=source_id, reference=category_id))
            continue
        name = category.get("name")
        if not isinstance(name, str) or not name.strip():
            issues.append(
                _issue(
                    "invalid_category_name",
                    f"category {category_id} needs a nonempty 'name'",
                    source_id=source_id,
                    reference=category_id,
                )
            )
        if category_id in categories_by_id:
            issues.append(
                _issue(
                    "duplicate_category_id",
                    f"category id {category_id} appears more than once",
                    source_id=source_id,
                    reference=category_id,
                )
            )
        categories_by_id[category_id] = category
        issues.extend(_check_category(category, source_id))
        if source_id == ROLE_PANOPTIC:
            # A panoptic source must state these numerically; a missing or JSON
            # truthy value is not the numeric 0/1 the format defines.
            for field in ("isthing", "iscrowd"):
                if field in category and not _is_flag(category[field]):
                    issues.append(
                        _issue(
                            f"invalid_{field}",
                            f"category {category_id} '{field}' must be 0 or 1",
                            source_id=source_id,
                            reference=category_id,
                        )
                    )

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    if source_id != ROLE_PANOPTIC:
        issues.extend(
            _index_object_annotations(
                source_id,
                tables["annotations"],
                images_by_id,
                categories_by_id,
                annotations_by_image,
            )
        )
    else:
        issues.extend(
            _index_panoptic_annotations(
                source_id,
                tables["annotations"],
                images_by_id,
                categories_by_id,
                annotations_by_image,
            )
        )

    if issues:
        raise CocoSourceError(issues)

    return SourceIndex(
        source_id=source_id,
        file=file,
        payload=payload,
        images_by_id=images_by_id,
        categories_by_id=categories_by_id,
        annotations_by_image={
            image_id: tuple(items) for image_id, items in annotations_by_image.items()
        },
        image_keys={},
    )


def _index_object_annotations(
    source_id: str,
    annotations: list[Any],
    images_by_id: dict[int, dict[str, Any]],
    categories_by_id: dict[int, dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids: set[int] = set()
    required = ("id", "image_id", "category_id", "bbox", "area", "iscrowd")
    for annotation in annotations:
        if not isinstance(annotation, dict):
            issues.append(_issue("invalid_annotation", "annotation must be an object", source_id=source_id))
            continue
        annotation_id = annotation.get("id")
        if not _is_int(annotation_id):
            issues.append(_issue("invalid_annotation_id", "annotation 'id' must be an integer", source_id=source_id, reference=annotation_id))
            continue
        if annotation_id in seen_ids:
            issues.append(_issue("duplicate_annotation_id", f"annotation id {annotation_id} appears more than once", source_id=source_id, reference=annotation_id))
            continue
        seen_ids.add(annotation_id)
        missing = [field for field in required if field not in annotation]
        if missing:
            issues.append(
                _issue(
                    "missing_annotation_field",
                    f"annotation {annotation_id} is missing {', '.join(missing)}",
                    source_id=source_id,
                    reference=annotation_id,
                )
            )
            continue
        image_id = annotation["image_id"]
        if not _is_int(image_id) or image_id not in images_by_id:
            issues.append(_issue("dangling_image_reference", f"annotation {annotation_id} references unknown image {image_id}", source_id=source_id, reference=annotation_id))
            continue
        category_id = annotation["category_id"]
        if not _is_int(category_id) or category_id not in categories_by_id:
            issues.append(_issue("dangling_category_reference", f"annotation {annotation_id} references unknown category {category_id}", source_id=source_id, reference=annotation_id))
            continue
        if annotation["iscrowd"] not in (0, 1):
            issues.append(_issue("invalid_iscrowd", f"annotation {annotation_id} 'iscrowd' must be 0 or 1", source_id=source_id, reference=annotation_id))
        bbox = annotation["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(_is_finite_number(value) for value in bbox)
        ):
            issues.append(_issue("invalid_bbox", f"annotation {annotation_id} 'bbox' must be 4 finite numbers", source_id=source_id, reference=annotation_id))
        area = annotation["area"]
        if not _is_finite_number(area) or area < 0:
            issues.append(_issue("invalid_area", f"annotation {annotation_id} 'area' must be finite and nonnegative", source_id=source_id, reference=annotation_id))
        if source_id == ROLE_KEYPOINTS:
            issues.extend(
                _index_keypoints(
                    annotation,
                    annotation_id,
                    categories_by_id[category_id],
                )
            )
        issues.extend(
            _check_segmentation_structure(annotation, annotation_id, source_id)
        )
        annotations_by_image.setdefault(image_id, []).append(annotation)
    return issues


def _check_segmentation_structure(
    annotation: dict[str, Any], annotation_id: int, source_id: str
) -> list[ValidationIssue]:
    """Structural polygon/RLE checks that need no image and no decoder."""
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        return []  # box-only objects are valid and acquire no segmentation
    issues: list[ValidationIssue] = []
    if isinstance(segmentation, list):
        if not segmentation:
            issues.append(
                _issue(
                    "empty_segmentation",
                    f"annotation {annotation_id} has an empty polygon list",
                    source_id=source_id,
                    reference=annotation_id,
                )
            )
            return issues
        for part in segmentation:
            if (
                not isinstance(part, list)
                or len(part) % 2
                or len(part) < 6
                or not all(_is_finite_number(value) for value in part)
            ):
                issues.append(
                    _issue(
                        "invalid_polygon",
                        f"annotation {annotation_id} has a polygon part that is not a flat "
                        "even-length finite array of at least three coordinate pairs",
                        source_id=source_id,
                        reference=annotation_id,
                    )
                )
                return issues
        return issues
    if isinstance(segmentation, dict):
        size = segmentation.get("size")
        counts = segmentation.get("counts")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or not all(_is_int(value) and value > 0 for value in size)
        ):
            issues.append(
                _issue(
                    "invalid_rle_size",
                    f"annotation {annotation_id} RLE 'size' must be two positive integers",
                    source_id=source_id,
                    reference=annotation_id,
                )
            )
            return issues
        height, width = int(size[0]), int(size[1])
        if isinstance(counts, list):
            if not all(_is_int(value) and value >= 0 for value in counts):
                issues.append(
                    _issue(
                        "invalid_rle_counts",
                        f"annotation {annotation_id} RLE counts must be nonnegative integers",
                        source_id=source_id,
                        reference=annotation_id,
                    )
                )
            elif sum(counts) != height * width:
                issues.append(
                    _issue(
                        "rle_run_total_mismatch",
                        f"annotation {annotation_id} RLE runs cover {sum(counts)} pixels, "
                        f"expected {height * width}",
                        source_id=source_id,
                        reference=annotation_id,
                    )
                )
        elif not (isinstance(counts, str) and counts):
            issues.append(
                _issue(
                    "invalid_rle_counts",
                    f"annotation {annotation_id} RLE 'counts' must be a nonempty string or list",
                    source_id=source_id,
                    reference=annotation_id,
                )
            )
        return issues
    issues.append(
        _issue(
            "invalid_segmentation",
            f"annotation {annotation_id} segmentation must be a polygon list or RLE object",
            source_id=source_id,
            reference=annotation_id,
        )
    )
    return issues


def _index_keypoints(
    annotation: dict[str, Any], annotation_id: int, category: dict[str, Any]
) -> list[ValidationIssue]:
    source_id = ROLE_KEYPOINTS
    issues: list[ValidationIssue] = []
    names = category.get("keypoints")
    if names is None:
        # A combined source may carry object categories without keypoints;
        # their point fields are extension data, not keypoint annotations.
        return issues
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name.strip() for name in names)
    ):
        issues.append(
            _issue(
                "invalid_keypoint_names",
                f"keypoint category {category.get('id')} needs unique nonempty point names",
                source_id=source_id,
                reference=category.get("id"),
            )
        )
        return issues
    if len(set(names)) != len(names):
        issues.append(
            _issue(
                "duplicate_keypoint_name",
                f"keypoint category {category.get('id')} repeats a point name",
                source_id=source_id,
                reference=category.get("id"),
            )
        )
    if "keypoints" not in annotation or "num_keypoints" not in annotation:
        issues.append(
            _issue(
                "missing_keypoint_field",
                f"annotation {annotation_id} needs 'keypoints' and 'num_keypoints'",
                source_id=source_id,
                reference=annotation_id,
            )
        )
        return issues
    values = annotation["keypoints"]
    expected = 3 * len(names)
    if (
        not isinstance(values, list)
        or len(values) != expected
        or not all(_is_finite_number(value) for value in values)
    ):
        issues.append(
            _issue(
                "invalid_keypoints",
                f"annotation {annotation_id} needs {expected} finite keypoint values",
                source_id=source_id,
                reference=annotation_id,
            )
        )
        return issues
    labelled = 0
    for index in range(len(names)):
        x, y, visibility = values[3 * index : 3 * index + 3]
        if not _is_int(visibility) or visibility not in (0, 1, 2):
            issues.append(
                _issue(
                    "invalid_keypoint_visibility",
                    f"annotation {annotation_id} point {names[index]!r} has visibility {visibility!r}",
                    source_id=source_id,
                    reference=annotation_id,
                )
            )
            continue
        if visibility > 0:
            labelled += 1
        elif x != 0 or y != 0:
            issues.append(
                _issue(
                    "unlabelled_keypoint_offset",
                    f"annotation {annotation_id} unlabelled point {names[index]!r} must be (0, 0)",
                    source_id=source_id,
                    reference=annotation_id,
                )
            )
    count = annotation["num_keypoints"]
    if not _is_int(count) or count != labelled:
        issues.append(
            _issue(
                "keypoint_count_mismatch",
                f"annotation {annotation_id} num_keypoints={count!r} but {labelled} point(s) have v>0",
                source_id=source_id,
                reference=annotation_id,
            )
        )
    skeleton = category.get("skeleton")
    if skeleton is not None:
        if not isinstance(skeleton, list) or not all(
            isinstance(edge, list)
            and len(edge) == 2
            and all(_is_int(point) and 1 <= point <= len(names) for point in edge)
            for edge in skeleton
        ):
            issues.append(
                _issue(
                    "invalid_skeleton",
                    f"keypoint category {category.get('id')} has an invalid one-based skeleton",
                    source_id=source_id,
                    reference=category.get("id"),
                )
            )
    return issues


def _index_panoptic_annotations(
    source_id: str,
    annotations: list[Any],
    images_by_id: dict[int, dict[str, Any]],
    categories_by_id: dict[int, dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
) -> list[ValidationIssue]:
    """Per-image panoptic records; mask decoding and pixel checks live in masks.py."""
    issues: list[ValidationIssue] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            issues.append(_issue("invalid_panoptic_annotation", "panoptic annotation must be an object", source_id=source_id))
            continue
        image_id = annotation.get("image_id")
        if not _is_int(image_id) or image_id not in images_by_id:
            issues.append(_issue("dangling_image_reference", f"panoptic annotation references unknown image {image_id}", source_id=source_id, reference=image_id))
            continue
        if image_id in annotations_by_image:
            issues.append(_issue("duplicate_panoptic_record", f"image {image_id} has more than one panoptic record", source_id=source_id, reference=image_id))
            continue
        file_name = annotation.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            issues.append(_issue("invalid_file_name", f"panoptic record for image {image_id} needs a nonempty 'file_name'", source_id=source_id, reference=image_id))
        segments = annotation.get("segments_info")
        if not isinstance(segments, list):
            issues.append(_issue("invalid_segments_info", f"panoptic record for image {image_id} needs a 'segments_info' array", source_id=source_id, reference=image_id))
            continue
        seen: set[int] = set()
        for segment in segments:
            if not isinstance(segment, dict):
                issues.append(_issue("invalid_segment", f"panoptic segment for image {image_id} must be an object", source_id=source_id, reference=image_id))
                continue
            segment_id = segment.get("id")
            if not _is_int(segment_id) or not 0 < segment_id < 256**3:
                issues.append(_issue("invalid_segment_id", f"panoptic segment id {segment_id!r} must fit unsigned 24 bits", source_id=source_id, reference=image_id))
                continue
            if segment_id in seen:
                issues.append(_issue("duplicate_segment_id", f"panoptic segment id {segment_id} repeats in image {image_id}", source_id=source_id, reference=image_id))
            seen.add(segment_id)
            if "iscrowd" in segment and not _is_flag(segment["iscrowd"]):
                issues.append(_issue("invalid_iscrowd", f"panoptic segment {segment_id} 'iscrowd' must be 0 or 1", source_id=source_id, reference=image_id))
            category_id = segment.get("category_id")
            if not _is_int(category_id) or category_id not in categories_by_id:
                issues.append(_issue("dangling_category_reference", f"panoptic segment {segment_id} references unknown category {category_id}", source_id=source_id, reference=image_id))
            segment_area = segment.get("area")
            if not _is_finite_number(segment_area) or segment_area < 0:
                issues.append(_issue("invalid_area", f"panoptic segment {segment_id} 'area' must be finite and nonnegative", source_id=source_id, reference=image_id))
            segment_bbox = segment.get("bbox")
            if (
                not isinstance(segment_bbox, list)
                or len(segment_bbox) != 4
                or not all(_is_finite_number(value) for value in segment_bbox)
            ):
                issues.append(_issue("invalid_bbox", f"panoptic segment {segment_id} 'bbox' must be 4 finite numbers", source_id=source_id, reference=image_id))
        annotations_by_image.setdefault(image_id, []).append(annotation)
    return issues


def validate_annotation(
    annotation: dict[str, Any],
    category: dict[str, Any],
    width: int,
    height: int,
    *,
    source_id: str,
) -> list[ValidationIssue]:
    """Selected-image geometry checks for one instance/keypoint annotation."""
    issues: list[ValidationIssue] = []
    reference = annotation.get("id")
    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, box_width, box_height = bbox
        if box_width < 0 or box_height < 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="negative_bbox_extent",
                    message=f"annotation {reference} has a negative box extent",
                    source_id=source_id,
                    reference=reference,
                )
            )
        elif box_width == 0 or box_height == 0:
            issues.append(
                ValidationIssue(
                    severity="information",
                    code="zero_bbox_extent",
                    message=f"annotation {reference} has a zero-width or zero-height box",
                    source_id=source_id,
                    reference=reference,
                )
            )
        if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="bbox_out_of_bounds",
                    message=(
                        f"annotation {reference} box {bbox} is outside "
                        f"[0,{width}] x [0,{height}]"
                    ),
                    source_id=source_id,
                    reference=reference,
                )
            )
    if source_id == ROLE_KEYPOINTS:
        names = category.get("keypoints") or []
        values = annotation.get("keypoints") or []
        for index, name in enumerate(names):
            if 3 * index + 2 >= len(values):
                break
            x, y, visibility = values[3 * index : 3 * index + 3]
            if visibility not in (1, 2):
                continue
            if not (0 <= x < width and 0 <= y < height):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="keypoint_out_of_bounds",
                        message=(
                            f"annotation {reference} point {name!r} {(x, y)} is outside "
                            f"the pixel-center domain [0,{width}) x [0,{height})"
                        ),
                        source_id=source_id,
                        reference=reference,
                    )
                )
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        for part_index, part in enumerate(segmentation):
            if not isinstance(part, list):
                continue
            for offset in range(0, len(part) - 1, 2):
                x, y = part[offset], part[offset + 1]
                if not (0 <= x <= width and 0 <= y <= height):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="polygon_out_of_bounds",
                            message=(
                                f"annotation {reference} polygon {part_index} vertex "
                                f"{(x, y)} is outside [0,{width}] x [0,{height}]"
                            ),
                            source_id=source_id,
                            reference=reference,
                        )
                    )
                    break
    return issues
