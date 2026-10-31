"""Strict dataset reader for the v2 data path.

Selection is CSV rows (in file order), otherwise the union of COCO image keys
(sorted), otherwise a recursive image-directory scan. COCO sources only attach
annotations; they never expand a CSV-selected set. Every selected image is hashed
and decoded once here, and consumers later decode the bytes they verify.
"""

from __future__ import annotations

import csv
import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .coco import (
    ROLE_INSTANCES,
    ROLE_KEYPOINTS,
    ROLE_PANOPTIC,
    CocoSourceError,
    load_coco_source,
    index_coco,
    validate_annotation,
)
from .paths import (
    ReferenceError,
    is_within,
    iter_image_files,
    lexical_key,
    relative_key,
    require_contained,
    resolve_reference,
)
from .masks import (
    MaskDecodeError,
    MaskDependencyError,
    decode_panoptic_map,
    panoptic_segment_stats,
    polygon_within_bounds,
    rasterized_diagnostics,
)
from .provenance import SnapshotDriftError, read_verified_bytes, snapshot_bytes
from .records import (
    CocoImageRef,
    DataBundle,
    FileSnapshot,
    ImageRecord,
    SourceIndex,
    ValidationIssue,
    ValidationReport,
)

SELECTION_CSV = "csv_rows"
SELECTION_COCO = "coco_union"
SELECTION_DIRECTORY = "directory_scan"


def read_dataset(
    images_dir: Path,
    *,
    manifest: Path | None = None,
    coco_instances: Path | None = None,
    coco_keypoints: Path | None = None,
    coco_panoptic: Path | None = None,
    panoptic_masks_dir: Path | None = None,
) -> DataBundle:
    # Keep the spelling the caller used: an absolute reference may name the root
    # through the same symlink, and that spelling carries the alias identity.
    root_written = Path(images_dir).expanduser()
    root = root_written.resolve()
    roles = {
        ROLE_INSTANCES: coco_instances,
        ROLE_KEYPOINTS: coco_keypoints,
        ROLE_PANOPTIC: coco_panoptic,
    }
    supplied = {role: path for role, path in roles.items() if path is not None}
    report = ValidationReport()

    if (panoptic_masks_dir is None) != (coco_panoptic is None):
        report.add(
            ValidationIssue(
                severity="error",
                code="panoptic_pair_required",
                message="--coco-panoptic and --panoptic-masks-dir require each other",
            )
        )

    if not root.is_dir():
        report.add(
            ValidationIssue(
                severity="error",
                code="invalid_images_root",
                message=f"image root is not a directory: {root}",
            )
        )
        return _empty_bundle(root, report)

    sources: dict[str, SourceIndex] = {}
    if supplied:
        sources, source_issues = _load_sources(supplied)
        report.issues.extend(source_issues)
        if not report.valid:
            return _bundle(root, [], sources, manifest, None, (), report)

    key_refs, declared = _merge_source_identity(root, sources, report, root_written)
    if not report.valid:
        return _bundle(root, [], sources, manifest, None, (), report)

    if manifest is not None:
        records, snapshot, columns = _read_csv_selection(
            root, Path(manifest), report, key_refs, declared, root_written
        )
        rule = SELECTION_CSV
    elif sources:
        records, snapshot, columns = _read_coco_selection(
            root, report, key_refs, declared
        )
        rule = SELECTION_COCO
    else:
        records, snapshot, columns = _read_directory_selection(
            root, report, key_refs, declared
        )
        rule = SELECTION_DIRECTORY

    if ROLE_PANOPTIC in sources and panoptic_masks_dir is not None:
        sources = _resolve_panoptic_masks(panoptic_masks_dir, sources, records, report)
    _validate_selected_annotations(report, sources, records)
    _validate_panoptic_masks(report, sources, records)
    _collect_geometry_diagnostics(report, sources, records)
    _check_unselected_source_images(report, root, sources, records)
    _report_availability(report, sources, records)
    _report_source_coverage(report, sources, records)
    _report_check_scope(report, sources)
    if not records and report.valid:
        report.add(
            ValidationIssue(
                severity="error",
                code="empty_selection",
                message="the selection resolved no images",
            )
        )
    return _bundle(root, records, sources, manifest, snapshot, columns, report, rule)


def _bundle(
    root: Path,
    records: list[ImageRecord],
    sources: dict[str, SourceIndex],
    manifest: Path | None,
    snapshot: FileSnapshot | None,
    columns: tuple[str, ...],
    report: ValidationReport,
    rule: str | None = None,
) -> DataBundle:
    if rule is None:
        rule = SELECTION_CSV if manifest is not None else (
            SELECTION_COCO if sources else SELECTION_DIRECTORY
        )
    report.scope.setdefault("selection_rule", rule)
    report.counts.setdefault("selected_images", len(records))
    return DataBundle(
        images_dir=root,
        records=tuple(records),
        sources=sources,
        selector=snapshot,
        selector_columns=columns,
        selection_rule=rule,
        validation=report,
    )


def _empty_bundle(root: Path, report: ValidationReport) -> DataBundle:
    return _bundle(root, [], {}, None, None, (), report)


def _load_sources(
    supplied: dict[str, Path],
) -> tuple[dict[str, SourceIndex], list[ValidationIssue]]:
    """Index each role once; a combined file is parsed once but indexed twice."""
    loaded: dict[Path, tuple[dict[str, Any], FileSnapshot]] = {}
    sources: dict[str, SourceIndex] = {}
    issues: list[ValidationIssue] = []
    for role, path in supplied.items():
        resolved = Path(path).expanduser().resolve()
        if resolved not in loaded:
            try:
                loaded[resolved] = load_coco_source(resolved)
            except CocoSourceError as exc:
                for issue in exc.issues:
                    issues.append(
                        replace(issue, source_id=issue.source_id or role)
                    )
                continue
        payload, snapshot = loaded[resolved]
        try:
            sources[role] = index_coco(role, payload, snapshot)
        except CocoSourceError as exc:
            for issue in exc.issues:
                issues.append(replace(issue, source_id=issue.source_id or role))
    return sources, issues


def _merge_source_identity(
    root: Path,
    sources: dict[str, SourceIndex],
    report: ValidationReport,
    root_written: Path | None = None,
) -> tuple[dict[str, list[CocoImageRef]], dict[str, tuple[int, int]]]:
    """Resolve COCO ``file_name`` keys, detect duplicates and conflicts."""
    key_refs: dict[str, list[CocoImageRef]] = {}
    declared: dict[str, tuple[int, int]] = {}
    for role, index in list(sources.items()):
        keys: dict[int, str] = {}
        claimed: dict[str, int] = {}
        for image_id, image in index.images_by_id.items():
            try:
                key = lexical_key(
                    root if root_written is None else root_written,
                    image["file_name"],
                    resolved_root=root,
                )
            except ReferenceError as exc:
                report.add(
                    ValidationIssue(
                        severity="error",
                        code=exc.code,
                        message=str(exc),
                        source_id=role,
                        reference=image_id,
                    )
                )
                continue
            if key in claimed:
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="duplicate_source_image_key",
                        message=(
                            f"{role} images {claimed[key]} and {image_id} resolve to {key!r}"
                        ),
                        source_id=role,
                        reference=image_id,
                        image_key=key,
                    )
                )
                continue
            claimed[key] = image_id
            keys[image_id] = key
            key_refs.setdefault(key, []).append(CocoImageRef(role, image_id))
            size = (int(image["width"]), int(image["height"]))
            previous = declared.get(key)
            if previous is not None and previous != size:
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="conflicting_dimensions",
                        message=(
                            f"{key!r} is declared as {previous[0]}x{previous[1]} and "
                            f"{size[0]}x{size[1]}"
                        ),
                        source_id=role,
                        reference=image_id,
                        image_key=key,
                    )
                )
            else:
                declared[key] = size
        sources[role] = replace(index, image_keys=keys)
    return key_refs, declared


def _read_csv_selection(
    root: Path,
    manifest: Path,
    report: ValidationReport,
    key_refs: dict[str, list[CocoImageRef]],
    declared: dict[str, tuple[int, int]],
    root_written: Path | None = None,
) -> tuple[list[ImageRecord], FileSnapshot | None, tuple[str, ...]]:
    if not manifest.is_file():
        report.add(
            ValidationIssue(
                severity="error",
                code="missing_selector",
                message=f"manifest CSV does not exist: {manifest}",
            )
        )
        return [], None, ()
    try:
        data, snapshot = snapshot_bytes(manifest)
    except OSError as exc:
        report.add(
            ValidationIssue(
                severity="error",
                code="unreadable_selector",
                message=f"cannot read manifest CSV: {exc}",
            )
        )
        return [], None, ()

    header, rows = _parse_strict_csv(data, report)
    if header is None:
        return [], snapshot, ()
    image_index = header.index("image")

    records: list[ImageRecord] = []
    seen: dict[str, int] = {}
    identities: dict[object, str] = {}
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            report.add(
                ValidationIssue(
                    severity="error",
                    code="malformed_row",
                    message=f"row {row_number} has {len(row)} fields, expected {len(header)}",
                    reference=row_number,
                )
            )
            continue
        reference = row[image_index]
        if not reference.strip():
            report.add(
                ValidationIssue(
                    severity="error",
                    code="blank_image_reference",
                    message=f"row {row_number} has a blank 'image' value",
                    reference=row_number,
                )
            )
            continue
        try:
            key, access_path = resolve_reference(
                root if root_written is None else root_written,
                reference,
                basename_fallback=True,
                resolved_root=root,
            )
        except ReferenceError as exc:
            report.add(
                ValidationIssue(
                    severity="error",
                    code=exc.code,
                    message=str(exc),
                    reference=row_number,
                )
            )
            continue
        record = _build_record(
            report,
            root=root,
            key=key,
            access_path=access_path,
            output_id=reference,
            selector_reference=reference,
            metadata={
                column: row[index]
                for index, column in enumerate(header)
                if column != "image"
            },
            row_number=row_number,
            seen=seen,
            identities=identities,
            declared_size=declared.get(key),
            coco_images=tuple(key_refs.get(key, ())),
        )
        if record is not None:
            records.append(record)

    report.counts["selector_rows"] = len(rows)
    return records, snapshot, tuple(header)


def _read_coco_selection(
    root: Path,
    report: ValidationReport,
    key_refs: dict[str, list[CocoImageRef]],
    declared: dict[str, tuple[int, int]],
) -> tuple[list[ImageRecord], FileSnapshot | None, tuple[str, ...]]:
    records: list[ImageRecord] = []
    seen: dict[str, int] = {}
    identities: dict[object, str] = {}
    for key in sorted(key_refs):
        try:
            access_path = (root / key).resolve()
        except OSError as exc:  # pragma: no cover - defensive
            report.add(
                ValidationIssue(
                    severity="error",
                    code="unreadable_image",
                    message=f"cannot resolve {key!r}: {exc}",
                    image_key=key,
                )
            )
            continue
        if not access_path.is_file():
            report.add(
                ValidationIssue(
                    severity="error",
                    code="missing_image",
                    message=f"COCO image is not present locally: {key}",
                    image_key=key,
                )
            )
            continue
        record = _build_record(
            report,
            root=root,
            key=key,
            access_path=access_path,
            output_id=key,
            selector_reference=None,
            metadata={},
            row_number=None,
            seen=seen,
            identities=identities,
            declared_size=declared.get(key),
            coco_images=tuple(key_refs.get(key, ())),
        )
        if record is not None:
            records.append(record)
    return records, None, ()


def _read_directory_selection(
    root: Path,
    report: ValidationReport,
    key_refs: dict[str, list[CocoImageRef]],
    declared: dict[str, tuple[int, int]],
) -> tuple[list[ImageRecord], FileSnapshot | None, tuple[str, ...]]:
    records: list[ImageRecord] = []
    seen: dict[str, int] = {}
    identities: dict[object, str] = {}
    for path in iter_image_files(root):
        key = relative_key(root, path)
        record = _build_record(
            report,
            root=root,
            key=key,
            access_path=path.resolve(),
            output_id=key,
            selector_reference=None,
            metadata={},
            row_number=None,
            seen=seen,
            identities=identities,
            declared_size=declared.get(key),
            coco_images=tuple(key_refs.get(key, ())),
        )
        if record is not None:
            records.append(record)
    return records, None, ()


def _selected_annotations(
    sources: dict[str, SourceIndex], records: list[ImageRecord]
):
    """Yield each selected annotation once, with its most specific role.

    One combined file may serve two roles, so annotations are grouped by object
    identity: the keypoint interpretation wins when any supplied role has it.
    """
    grouped: dict[int, tuple[dict[str, Any], ImageRecord, set[str]]] = {}
    for record in records:
        for ref in record.coco_images:
            if ref.source_id == ROLE_PANOPTIC:
                continue
            index = sources[ref.source_id]
            for annotation in index.annotations_by_image.get(ref.image_id, ()):
                entry = grouped.get(id(annotation))
                if entry is None:
                    grouped[id(annotation)] = (annotation, record, {ref.source_id})
                else:
                    entry[2].add(ref.source_id)
    for annotation, record, roles in grouped.values():
        role = ROLE_KEYPOINTS if ROLE_KEYPOINTS in roles else ROLE_INSTANCES
        yield annotation, record, role


def _validate_selected_annotations(
    report: ValidationReport,
    sources: dict[str, SourceIndex],
    records: list[ImageRecord],
) -> None:
    for annotation, record, role in _selected_annotations(sources, records):
        category = sources[role].categories_by_id[annotation["category_id"]]
        report.issues.extend(
            validate_annotation(
                annotation,
                category,
                record.width,
                record.height,
                source_id=role,
            )
        )


def _resolve_panoptic_masks(
    masks_dir: Path,
    sources: dict[str, SourceIndex],
    records: list[ImageRecord],
    report: ValidationReport,
) -> dict[str, SourceIndex]:
    root = Path(masks_dir).expanduser().resolve()
    if not root.is_dir():
        report.add(
            ValidationIssue(
                severity="error",
                code="invalid_panoptic_masks_root",
                message=f"panoptic mask root is not a directory: {root}",
            )
        )
        return sources
    index = sources[ROLE_PANOPTIC]
    mask_files: dict[int, FileSnapshot] = {}
    for record in records:
        for ref in record.coco_images:
            if ref.source_id != ROLE_PANOPTIC:
                continue
            annotations = index.annotations_by_image.get(ref.image_id, ())
            if not annotations:
                continue
            file_name = annotations[0]["file_name"]
            try:
                _key, path = resolve_reference(root, file_name)
            except ReferenceError as exc:
                missing = exc.code == "missing_image"
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="missing_panoptic_mask" if missing else exc.code,
                        message=(
                            f"panoptic mask is missing: {file_name}"
                            if missing
                            else str(exc)
                        ),
                        source_id=ROLE_PANOPTIC,
                        reference=ref.image_id,
                        image_key=record.image_key,
                    )
                )
                continue
            if not path.is_file():
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="missing_panoptic_mask",
                        message=f"panoptic mask is missing: {file_name}",
                        source_id=ROLE_PANOPTIC,
                        reference=ref.image_id,
                        image_key=record.image_key,
                    )
                )
                continue
            try:
                _data, snapshot = snapshot_bytes(path)
            except OSError as exc:
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="unreadable_panoptic_mask",
                        message=f"cannot read panoptic mask: {exc}",
                        source_id=ROLE_PANOPTIC,
                        reference=ref.image_id,
                        image_key=record.image_key,
                    )
                )
                continue
            mask_files[ref.image_id] = snapshot
    sources[ROLE_PANOPTIC] = replace(index, mask_files=mask_files)
    return sources


PANOPTIC_TOLERANCE = 1e-6


def _validate_panoptic_masks(
    report: ValidationReport,
    sources: dict[str, SourceIndex],
    records: list[ImageRecord],
) -> None:
    index = sources.get(ROLE_PANOPTIC)
    if index is None:
        return
    # Void pixels are counted from the decode validation already performed; a second
    # decode could fail and silently turn a real count into a trustworthy-looking 0.
    void_pixels = 0
    void_expected = 0
    void_counted = 0
    for record in records:
        for ref in record.coco_images:
            if ref.source_id != ROLE_PANOPTIC:
                continue
            snapshot = index.mask_files.get(ref.image_id)
            annotation = (index.annotations_by_image.get(ref.image_id) or (None,))[0]
            if annotation is None:
                continue
            void_expected += 1
            if snapshot is None:
                # A mask this record expects is absent: the total is incomplete.
                continue
            try:
                data = read_verified_bytes(snapshot)
            except SnapshotDriftError as exc:
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="input_drift",
                        message=str(exc),
                        source_id=ROLE_PANOPTIC,
                        reference=ref.image_id,
                        image_key=record.image_key,
                    )
                )
                continue
            try:
                ids = decode_panoptic_map(data, record.width, record.height)
            except MaskDecodeError as exc:
                report.add(
                    ValidationIssue(
                        severity="error",
                        code=exc.code,
                        message=str(exc),
                        source_id=ROLE_PANOPTIC,
                        reference=ref.image_id,
                        image_key=record.image_key,
                    )
                )
                continue
            void_pixels += int(np.count_nonzero(ids == 0))
            void_counted += 1
            declared = {int(seg["id"]): seg for seg in annotation["segments_info"]}
            present = {int(value) for value in np.unique(ids)}
            present.discard(0)  # void is not a foreground segment
            for segment_id in sorted(present - set(declared)):
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="panoptic_id_missing_from_json",
                        message=f"segment {segment_id} is in the PNG but not in segments_info",
                        source_id=ROLE_PANOPTIC,
                        reference=ref.image_id,
                        image_key=record.image_key,
                    )
                )
            for segment_id in sorted(set(declared) - present):
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="panoptic_segment_missing_from_png",
                        message=f"segment {segment_id} is declared but absent from the PNG",
                        source_id=ROLE_PANOPTIC,
                        reference=ref.image_id,
                        image_key=record.image_key,
                    )
                )
            for segment_id, stats in panoptic_segment_stats(ids, list(declared)).items():
                segment = declared[segment_id]
                if abs(stats["area"] - float(segment["area"])) > PANOPTIC_TOLERANCE:
                    report.add(
                        ValidationIssue(
                            severity="error",
                            code="panoptic_area_mismatch",
                            message=(
                                f"segment {segment_id} declares area {segment['area']} but "
                                f"the PNG has {stats['area']}"
                            ),
                            source_id=ROLE_PANOPTIC,
                            reference=ref.image_id,
                            image_key=record.image_key,
                        )
                    )
                if any(
                    abs(actual - float(declared_value)) > PANOPTIC_TOLERANCE
                    for actual, declared_value in zip(stats["bbox"], segment["bbox"])
                ):
                    report.add(
                        ValidationIssue(
                            severity="error",
                            code="panoptic_bbox_mismatch",
                            message=(
                                f"segment {segment_id} declares bbox {segment['bbox']} but "
                                f"the PNG yields {stats['bbox']}"
                            ),
                            source_id=ROLE_PANOPTIC,
                            reference=ref.image_id,
                            image_key=record.image_key,
                        )
                    )


    skipped = void_expected - void_counted
    report.counts["panoptic_void_pixels"] = void_pixels
    if void_expected == 0:
        return
    if skipped == 0:
        report.scope["panoptic_void"] = {"status": "available", "unit": "pixels"}
    else:
        # Missing, drifted or undecodable masks make this a lower bound, never a
        # real total: say so instead of publishing a plausible number.
        report.scope["panoptic_void"] = {
            "status": "partial" if void_counted else "unavailable",
            "unit": "pixels",
            "masks_counted": void_counted,
            "masks_expected": void_expected,
            "reason": (
                f"{skipped} panoptic mask(s) could not be read or decoded, so the "
                "total is a lower bound"
            ),
        }


def _collect_geometry_diagnostics(
    report: ValidationReport,
    sources: dict[str, SourceIndex],
    records: list[ImageRecord],
) -> None:
    """Non-blocking polygon diagnostics; differences never gate ingestion."""
    diagnostics: list[dict[str, Any]] = []
    empty = 0
    dependency_error: str | None = None
    rejected: set[int] = set()
    for annotation, record, role in _selected_annotations(sources, records):
        if annotation.get("segmentation") is None:
            continue
        if not polygon_within_bounds(
            annotation.get("segmentation"), record.width, record.height
        ):
            # Out-of-bounds geometry is already an error; rasterizing it would only
            # produce numbers for a mask that must not be used.
            rejected.add(id(annotation))
            continue
        if dependency_error is not None:
            continue
        try:
            measured = rasterized_diagnostics(annotation, record.width, record.height)
        except MaskDependencyError as exc:
            dependency_error = str(exc)
            continue
        except MaskDecodeError as exc:
            report.add(
                ValidationIssue(
                    severity="error",
                    code=exc.code,
                    message=str(exc),
                    source_id=role,
                    reference=annotation.get("id"),
                    image_key=record.image_key,
                )
            )
            continue
        if isinstance(annotation.get("segmentation"), dict):
            # RLE keeps exact pixel semantics: a disagreement is an error.
            if abs(measured["rasterized_area"] - float(annotation.get("area") or 0.0)) > PANOPTIC_TOLERANCE:
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="rle_area_mismatch",
                        message=(
                            f"annotation {annotation.get('id')} declares area "
                            f"{annotation.get('area')} but decodes to "
                            f"{measured['rasterized_area']}"
                        ),
                        source_id=role,
                        reference=annotation.get("id"),
                        image_key=record.image_key,
                    )
                )
            declared_bbox = annotation.get("bbox") or []
            if len(declared_bbox) == 4 and any(
                abs(actual - float(expected)) > PANOPTIC_TOLERANCE
                for actual, expected in zip(measured["rasterized_bbox"], declared_bbox)
            ):
                report.add(
                    ValidationIssue(
                        severity="error",
                        code="rle_bbox_mismatch",
                        message=(
                            f"annotation {annotation.get('id')} declares bbox "
                            f"{declared_bbox} but decodes to {measured['rasterized_bbox']}"
                        ),
                        source_id=role,
                        reference=annotation.get("id"),
                        image_key=record.image_key,
                    )
                )
        entry: dict[str, Any] = {
            "source_id": role,
            "image_key": record.image_key,
            "annotation_id": annotation.get("id"),
            "source_area": annotation.get("area"),
            "source_bbox": annotation.get("bbox"),
            **measured,
        }
        entry["area_delta"] = abs(float(measured["rasterized_area"]) - float(annotation.get("area") or 0.0))
        diagnostics.append(entry)
        if measured["rasterized_area"] == 0:
            empty += 1
            report.add(
                ValidationIssue(
                    severity="information",
                    code="empty_decoded_mask",
                    message=(
                        f"annotation {annotation.get('id')} decodes to an empty mask"
                    ),
                    source_id=role,
                    reference=annotation.get("id"),
                    image_key=record.image_key,
                )
            )
    if dependency_error is not None:
        report.add(
            ValidationIssue(
                severity="error",
                code="mask_decoder_unavailable",
                message=dependency_error,
            )
        )
    report.polygon_diagnostics = diagnostics
    report.counts["geometry_rejected_before_rasterization"] = len(rejected)
    report.counts["empty_segmentations"] = empty
    report.counts["polygon_diagnostics"] = len(diagnostics)


def _check_unselected_source_images(
    report: ValidationReport,
    root: Path,
    sources: dict[str, SourceIndex],
    records: list[ImageRecord],
) -> None:
    """Containment for source references that were not selected.

    Unselected files need not exist and their geometry is never checked, but a
    reference that resolves outside the image root is an error, selected or not.
    """
    selected = {record.image_key for record in records}
    unselected = 0
    for role, index in sources.items():
        for image_id, key in sorted(index.image_keys.items()):
            if key in selected:
                continue
            unselected += 1
            try:
                require_contained(root, Path(root) / key, key)
            except ReferenceError as exc:
                report.add(
                    ValidationIssue(
                        severity="error",
                        code=exc.code,
                        message=str(exc),
                        source_id=role,
                        reference=image_id,
                        image_key=key,
                    )
                )
    report.counts["unselected_source_images"] = unselected
    report.scope["unselected_source_images"] = {
        "count": unselected,
        "file_and_geometry_checks": "not_performed",
    }


def _report_availability(
    report: ValidationReport,
    sources: dict[str, SourceIndex],
    records: list[ImageRecord],
) -> None:
    visibility = {0: 0, 1: 0, 2: 0}
    for annotation, _record, role in _selected_annotations(sources, records):
        if role != ROLE_KEYPOINTS:
            continue
        values = annotation.get("keypoints") or []
        for index in range(2, len(values), 3):
            value = values[index]
            if value in visibility:
                visibility[value] += 1
    report.counts["images_without_annotations"] = sum(
        1 for record in records if not _has_annotations(sources, record)
    )
    report.counts["images_without_source_reference"] = sum(
        1 for record in records if not record.coco_images
    )
    report.counts["keypoints_by_visibility"] = visibility
    report.scope["custom_ignore_protocol"] = {
        "status": "not_implemented",
        "reason": (
            "custom ignore/unknown/occlusion fields are preserved extension data "
            "and are never interpreted as a supervision mask"
        ),
    }


def _has_annotations(sources: dict[str, SourceIndex], record: ImageRecord) -> bool:
    """True when a present source actually carries objects for this image."""
    for ref in record.coco_images:
        source = sources.get(ref.source_id)
        if source is not None and source.annotations_by_image.get(ref.image_id):
            return True
    return False


def _report_check_scope(
    report: ValidationReport, sources: dict[str, SourceIndex]
) -> None:
    """Separate what was actually checked from what was not.

    A ``valid`` status only promises the performed checks; reading it as "the whole
    source is fully verified" is what this block prevents.
    """
    report.scope["source_structure"] = (
        "performed" if sources else "not_applicable"
    )
    report.scope["selected_image_decode_and_geometry"] = "performed"
    report.scope["unselected_file_and_geometry"] = "not_performed"


def _report_source_coverage(
    report: ValidationReport,
    sources: dict[str, SourceIndex],
    records: list[ImageRecord],
) -> None:
    unused_images: dict[str, int] = {}
    unused_annotations: dict[str, int] = {}
    for role, index in sources.items():
        used_ids = {
            ref.image_id for record in records for ref in record.coco_images
            if ref.source_id == role
        }
        unused_images[role] = len(index.images_by_id) - len(used_ids)
        total = sum(len(items) for items in index.annotations_by_image.values())
        used = sum(
            len(index.annotations_by_image.get(image_id, ())) for image_id in used_ids
        )
        unused_annotations[role] = total - used
    report.counts["unused_source_images"] = unused_images
    report.counts["unused_annotations"] = unused_annotations
    report.counts["source_files"] = len(sources)


def _parse_strict_csv(
    data: bytes, report: ValidationReport
) -> tuple[list[str] | None, list[list[str]]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        report.add(
            ValidationIssue(
                severity="error",
                code="selector_not_utf8",
                message=f"manifest is not UTF-8: {exc}",
            )
        )
        return None, []
    rows = [row for row in csv.reader(io.StringIO(text, newline=""))]
    if not rows:
        report.add(
            ValidationIssue(
                severity="error",
                code="empty_selector",
                message="manifest CSV has no header row",
            )
        )
        return None, []
    header = list(rows[0])
    if any(not column.strip() for column in header):
        report.add(
            ValidationIssue(
                severity="error",
                code="blank_header",
                message="manifest CSV has a blank column name",
            )
        )
        return None, []
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        report.add(
            ValidationIssue(
                severity="error",
                code="duplicate_header",
                message=f"manifest CSV repeats column(s): {', '.join(duplicates)}",
            )
        )
        return None, []
    if "image" not in header:
        report.add(
            ValidationIssue(
                severity="error",
                code="missing_image_column",
                message="manifest CSV must contain an 'image' column",
            )
        )
        return None, []
    body = rows[1:]
    if not body:
        report.add(
            ValidationIssue(
                severity="error",
                code="empty_selection",
                message="manifest CSV has no data rows",
            )
        )
        return None, []
    return header, body


def _build_record(
    report: ValidationReport,
    *,
    root: Path,
    key: str,
    access_path: Path,
    output_id: str,
    selector_reference: str | None,
    metadata: dict[str, str],
    row_number: int | None,
    seen: dict[str, int],
    identities: dict[object, str],
    declared_size: tuple[int, int] | None = None,
    coco_images: tuple[CocoImageRef, ...] = (),
) -> ImageRecord | None:
    from otuformer.utils.paths import path_identity

    reference = row_number if row_number is not None else key
    if not is_within(root, access_path):
        report.add(
            ValidationIssue(
                severity="error",
                code="root_escape",
                message=f"image resolves outside the root: {key}",
                reference=reference,
                image_key=key,
            )
        )
        return None

    if key in seen:
        report.add(
            ValidationIssue(
                severity="error",
                code="duplicate_image_key",
                message=f"image key {key!r} is selected more than once",
                reference=reference,
                image_key=key,
            )
        )
        return None
    seen[key] = reference

    identity = path_identity(access_path)
    other_key = identities.get(identity)
    if other_key is not None and other_key != key:
        report.add(
            ValidationIssue(
                severity="warning",
                code="physical_alias",
                message=f"keys {other_key!r} and {key!r} resolve to the same physical file",
                image_key=key,
                details={"other_image_key": other_key},
            )
        )
    else:
        identities[identity] = key

    try:
        data, snapshot = snapshot_bytes(access_path)
    except OSError as exc:
        report.add(
            ValidationIssue(
                severity="error",
                code="unreadable_image",
                message=f"cannot read image: {exc}",
                reference=reference,
                image_key=key,
            )
        )
        return None

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
    except Exception as exc:  # noqa: BLE001 - decoder errors are input errors
        report.add(
            ValidationIssue(
                severity="error",
                code="image_decode_failed",
                message=f"cannot decode image: {exc}",
                reference=reference,
                image_key=key,
            )
        )
        return None
    if width <= 0 or height <= 0:
        report.add(
            ValidationIssue(
                severity="error",
                code="invalid_image_dimensions",
                message=f"image reports non-positive size {width}x{height}",
                reference=reference,
                image_key=key,
            )
        )
        return None
    if declared_size is not None and declared_size != (int(width), int(height)):
        report.add(
            ValidationIssue(
                severity="error",
                code="declared_dimension_mismatch",
                message=(
                    f"declared {declared_size[0]}x{declared_size[1]} but the image is "
                    f"{width}x{height}"
                ),
                reference=reference,
                image_key=key,
            )
        )

    sample, sample_origin = _resolve_sample(key, metadata)
    return ImageRecord(
        image_key=key,
        access_path=access_path,
        output_id=output_id,
        selector_reference=selector_reference,
        width=int(width),
        height=int(height),
        image_file=snapshot,
        metadata=dict(metadata),
        coco_images=coco_images,
        sample=sample,
        sample_origin=sample_origin,
    )


def _resolve_sample(key: str, metadata: dict[str, str]) -> tuple[str | None, str]:
    if "sample" in metadata:
        value = metadata["sample"]
        return (value if value != "" else None), "explicit"
    parts = key.split("/")
    if len(parts) >= 2:
        return parts[0], "first_directory"
    return None, "none"
