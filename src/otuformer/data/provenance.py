"""Byte snapshots and digest/privacy primitives for provenance (stdlib only).

A snapshot pins the bytes that validation saw. Consumers decode the bytes they
verify, so a later replacement cannot silently change what ran.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .records import FileSnapshot


class SnapshotDriftError(RuntimeError):
    """A pinned input no longer matches the bytes that were validated."""

    def __init__(self, snapshot: FileSnapshot, reason: str) -> None:
        super().__init__(f"input changed after validation ({reason}): {snapshot.path}")
        self.snapshot = snapshot
        self.reason = reason


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_verified_bytes(snapshot: FileSnapshot) -> bytes:
    """Read ``snapshot.path`` and fail unless the bytes still match the snapshot."""
    try:
        data = snapshot.path.read_bytes()
    except OSError as exc:  # removed or unreadable after validation
        raise SnapshotDriftError(snapshot, f"unreadable: {exc}") from exc
    if len(data) != snapshot.size_bytes:
        raise SnapshotDriftError(snapshot, "size")
    if sha256_bytes(data) != snapshot.sha256:
        raise SnapshotDriftError(snapshot, "sha256")
    return data


def snapshot_bytes(
    path: Path, *, follow_symlink: bool = True
) -> tuple[bytes, FileSnapshot]:
    """Read ``path`` once and return its bytes with a content-identity snapshot.

    ``follow_symlink=False`` records the entry that was opened rather than its real
    target. An ONNX graph resolves its external data against the path the runtime
    was given, so that spelling is the one that explains what was actually loaded.
    """
    target = Path(path).expanduser()
    if follow_symlink:
        target = target.resolve()
    elif not target.is_absolute():
        target = Path.cwd() / target
    data = target.read_bytes()
    snapshot = FileSnapshot(
        path=target, sha256=sha256_bytes(data), size_bytes=len(data)
    )
    return data, snapshot


# --- canonical serialization and digests ----------------------------------

SCHEMA_VERSION = 1
RLE_PANOPTIC_TOLERANCE = 1e-6


def canonical_json_bytes(value) -> bytes:
    """Canonical bytes for logical digests (deterministic, no NaN)."""
    import json

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def selection_metadata_payload(bundle) -> dict:
    """Ordered selection identity; excludes machine-local roots and selectors."""
    return {
        "selector_columns": list(bundle.selector_columns),
        "records": [
            {
                "image_key": record.image_key,
                "metadata": dict(record.metadata),
                "coco_images": [
                    {"source_id": ref.source_id, "image_id": int(ref.image_id)}
                    for ref in record.coco_images
                ],
            }
            for record in bundle.records
        ],
    }


def data_content_payload(bundle) -> dict:
    """Ordered content identity: image bytes plus the masks that belong to them."""
    return {
        "records": [
            {
                "image_key": record.image_key,
                "image_sha256": record.image_file.sha256,
                "masks": _mask_rows(bundle, record),
            }
            for record in bundle.records
        ],
    }


def _mask_rows(bundle, record) -> list[dict]:
    rows: list[dict] = []
    for ref in record.coco_images:
        source = bundle.sources.get(ref.source_id)
        if source is None:
            continue
        snapshot = source.mask_files.get(ref.image_id)
        if snapshot is None:
            continue
        rows.append(
            {
                "source_id": ref.source_id,
                "image_id": int(ref.image_id),
                "sha256": snapshot.sha256,
            }
        )
    return sorted(rows, key=lambda row: (row["source_id"], row["image_id"]))


def selection_metadata_digest(bundle) -> str:
    return canonical_digest(selection_metadata_payload(bundle))


def data_content_digest(bundle) -> str:
    return canonical_digest(data_content_payload(bundle))


# --- per-row mapping and validation report --------------------------------


def _issue_payload(issue) -> dict:
    return {
        "severity": issue.severity,
        "code": issue.code,
        "message": issue.message,
        "source_id": issue.source_id,
        "reference": issue.reference,
        "image_key": issue.image_key,
        "details": dict(issue.details),
    }


def build_data_records(bundle, frame) -> list[dict]:
    """One mapping record per emitted embedding, in exactly that order."""
    if len(frame) != len(bundle.records):
        raise ValueError(
            f"frame has {len(frame)} rows but the selection has {len(bundle.records)} records"
        )
    emitted = frame["id"].tolist()
    rows: list[dict] = []
    for index, record in enumerate(bundle.records):
        if emitted[index] != record.output_id:
            raise ValueError(
                "embedding row order does not match the selection: "
                f"row {index + 1} is {emitted[index]!r}, expected {record.output_id!r}"
            )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "embedding_row": index + 1,
                "output_id": record.output_id,
                "image_key": record.image_key,
                "selector_reference": record.selector_reference,
                "width": int(record.width),
                "height": int(record.height),
                "image_sha256": record.image_file.sha256,
                "image_size_bytes": int(record.image_file.size_bytes),
                "metadata": dict(record.metadata),
                "coco_images": [
                    {"source_id": ref.source_id, "image_id": int(ref.image_id)}
                    for ref in record.coco_images
                ],
            }
        )
    return rows


def build_validation_record(bundle) -> dict:
    report = bundle.validation
    used = _decoder_usage(bundle)
    sources = []
    for source_id, index in sorted(bundle.sources.items()):
        sources.append(
            {
                "source_id": source_id,
                "file": str(index.file.path),
                "sha256": index.file.sha256,
                "images": len(index.images_by_id),
                "annotations": sum(len(items) for items in index.annotations_by_image.values()),
                "categories": [
                    {
                        "id": int(category_id),
                        "name": category.get("name"),
                        "supercategory": category.get("supercategory"),
                        "definition": {
                            key: category[key]
                            for key in ("keypoints", "skeleton", "isthing", "iscrowd")
                            if key in category
                        },
                    }
                    for category_id, category in sorted(index.categories_by_id.items())
                ],
                "masks": {
                    str(image_id): {
                        "path": str(snapshot.path),
                        "sha256": snapshot.sha256,
                        "bytes": int(snapshot.size_bytes),
                    }
                    for image_id, snapshot in sorted(index.mask_files.items())
                },
            }
        )
    # Only declare checks that actually ran, with the tolerance each one used.
    methods: dict = {"image_decode": "Pillow"}
    if used["polygon_rasterization"]:
        methods["polygon_rasterization"] = "pycocotools (COCO mask union)"
        methods.setdefault("tolerances", {})["polygon_area"] = "no fixed threshold"
    if used["rle"]:
        methods.setdefault("tolerances", {})["rle_pixel"] = RLE_PANOPTIC_TOLERANCE
    if used["panoptic"]:
        methods["panoptic_decode"] = "Pillow RGB to 24-bit segment ids"
        methods.setdefault("tolerances", {})["rle_panoptic_pixel"] = (
            RLE_PANOPTIC_TOLERANCE
        )
    if not methods.get("tolerances"):
        methods["tolerances"] = {"status": "not_applicable", "reason": "no mask check ran"}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "valid" if report.valid else "invalid",
        "scope": dict(report.scope),
        "counts": dict(report.counts),
        "methods": methods,
        "sources": sources,
        "issues": [_issue_payload(issue) for issue in report.issues],
        "polygon_diagnostics": list(report.polygon_diagnostics),
    }


def _decoder_usage(bundle) -> dict[str, bool]:
    """Which decoders the selected annotations actually require."""
    usage = {"polygon_rasterization": False, "rle": False, "panoptic": False}
    for record in bundle.records:
        for ref in record.coco_images:
            if ref.source_id == "panoptic":
                usage["panoptic"] = True
                continue
            source = bundle.sources.get(ref.source_id)
            if source is None:
                continue
            for annotation in source.annotations_by_image.get(ref.image_id, ()):
                segmentation = annotation.get("segmentation")
                if isinstance(segmentation, list):
                    usage["polygon_rasterization"] = True
                elif isinstance(segmentation, dict):
                    usage["rle"] = True
    return usage


# --- run record -----------------------------------------------------------


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("otuformer")
    except Exception:  # pragma: no cover - metadata always present when installed
        return "unknown"


def _git_provenance() -> dict:
    """Git identity of the checkout that this package was imported from.

    The calling directory is irrelevant: a run started from an unrelated repository
    must never report that repository's commit. The existing source-root locator and
    project-identity check from ``utils/logging`` are reused for the same reason.
    """
    import subprocess

    from otuformer.utils.logging import _project_name_at

    source_dir = Path(__file__).resolve().parent

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(source_dir), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:  # pragma: no cover - git absent
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    toplevel = run("rev-parse", "--show-toplevel")
    commit = run("rev-parse", "HEAD") if toplevel else None
    if not toplevel or not commit:
        return {
            "status": "unavailable",
            "reason": "no git checkout for the imported package",
        }
    if _project_name_at(Path(toplevel)) != "otuformer":
        return {
            "status": "unavailable",
            "reason": "the enclosing repository does not declare this project",
            "root": toplevel,
        }
    dirty_status = run("status", "--porcelain")
    return {
        "status": "available",
        "root": toplevel,
        "commit": commit,
        "dirty": bool(dirty_status),
        "reproducible_from_dirty_tree": False,
    }


def _dependency_versions() -> dict:
    from importlib.metadata import version

    versions = {}
    for name in ("torch", "timm", "numpy", "pillow", "pandas", "scikit-learn"):
        try:
            versions[name] = version(name)
        except Exception:  # pragma: no cover - optional package absent
            versions[name] = "unavailable"
    try:
        versions["pycocotools"] = version("pycocotools")
    except Exception:
        versions["pycocotools"] = {"status": "unavailable", "reason": "optional extra not installed"}
    return versions


def _feature_dimensions(effective: dict) -> dict:
    """Actual feature widths, with the missing half named rather than hidden."""
    chain = effective.get("feature_chain")
    if not chain:
        return {"status": "unavailable", "reason": "extraction did not run"}
    input_dim = chain.get("input_dim")
    output_dim = chain.get("output_dim")
    if input_dim is None or output_dim is None:
        missing = [
            name
            for name, value in (("input_dim", input_dim), ("output_dim", output_dim))
            if value is None
        ]
        return {
            "status": "partial",
            "reason": f"not recorded by this backend: {', '.join(missing)}",
            "input_dim": input_dim,
            "output_dim": output_dim,
        }
    return {
        "status": "available",
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
    }


def _label_schema(bundle) -> dict:
    if "label" not in bundle.selector_columns:
        return {
            "status": "not_provided",
            "reason": "the selector has no 'label' column",
            "source_column": None,
            "observed_classes": [],
        }
    labels = [record.metadata.get("label", "") for record in bundle.records]
    observed = sorted({label for label in labels if label != ""})
    return {
        "status": "provided",
        "source_column": "label",
        "observed_classes": observed,
        "labeled_rows": sum(1 for label in labels if label != ""),
        "unlabeled_rows": sum(1 for label in labels if label == ""),
        "missing_label_handling": "an empty cell is unlabeled, never a shared class",
    }


def serialize_metric_outcomes(outcomes: dict) -> dict:
    """Serialize MetricOutcome values with their per-family protocol context."""
    values: dict[str, dict] = {}
    families: dict[str, dict] = {}
    for name, outcome in outcomes.items():
        values[name] = {
            "value": outcome.value,
            "status": outcome.status,
            "reason": outcome.reason,
        }
        family = outcome.context.get("family")
        if family is not None and family not in families:
            families[family] = {
                key: value
                for key, value in outcome.context.items()
                if key not in {"family"}
            }
    return {"values": values, "protocol": families}


def build_run_record(
    *,
    requested: dict,
    bundle,
    execution,
    evaluation: dict,
    status: str,
    started_at: str,
    ended_at: str | None,
    artifacts: list[dict],
    failure: dict | None = None,
) -> dict:
    """Serialize one run; artwork publication and lifetime belong to the command."""
    if status not in ("running", "completed", "failed", "interrupted"):
        raise ValueError(f"unknown run status: {status!r}")
    effective = (
        execution.effective
        if execution is not None
        else {"status": "unavailable", "reason": "extraction did not run"}
    )
    # The bytes actually consumed: re-read nothing, serialize what extraction kept.
    used_files = [
        {
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
            "bytes": int(snapshot.size_bytes),
        }
        for snapshot in (execution.used_files if execution is not None else ())
    ]
    selection = {
        "rule": bundle.selection_rule,
        "selected_images": len(bundle.records),
        "selector": (
            {"path": str(bundle.selector.path), "sha256": bundle.selector.sha256}
            if bundle.selector is not None
            else {"status": "not_provided", "reason": "no selector CSV"}
        ),
        "selector_columns": list(bundle.selector_columns),
        "selection_metadata_sha256_v1": selection_metadata_digest(bundle),
        "data_content_sha256_v1": data_content_digest(bundle),
    }
    unavailable = lambda reason: {"status": "not_provided", "reason": reason}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "command": "extract",
        "started_at": started_at,
        "ended_at": ended_at,
        "package": {"name": "otuformer", "version": _package_version(), "git": _git_provenance()},
        "dependencies": _dependency_versions(),
        "requested": requested,
        "effective": effective,
        "selection": selection,
        "sources": [
            {
                "source_id": source_id,
                "file": str(index.file.path),
                "sha256": index.file.sha256,
                "images": len(index.images_by_id),
            }
            for source_id, index in sorted(bundle.sources.items())
        ],
        "label_schema": _label_schema(bundle),
        "feature_dimensions": _feature_dimensions(effective),
        "used_files": used_files,
        "metrics": evaluation or unavailable("no evaluation ran"),
        "thresholds": {"status": "not_applicable", "reason": "no_threshold_selected"},
        "split": unavailable("group-safe split is a later delivery (roadmap 4.2)"),
        "held_out_test": {
            "status": "not_provided",
            "reason": "no held-out test set was supplied or used",
            "existing_protocol": "image-level stratified cross-validation only",
        },
        "geometry_transform": unavailable("sampled augmentation contracts are a later delivery (roadmap 4.3)"),
        "spatial_features": unavailable("spatial feature export is not part of this foundation"),
        "task_head": unavailable("no task head is trained or applied here"),
        "calibration": unavailable("no calibration procedure was applied"),
        "artifacts": artifacts,
        "failure": failure,
    }
