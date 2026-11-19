"""Resolve, run and publish an explicit spatial extraction or read.

The core owns behaviour and validation: it never imports the CLI. Preparation
resolves the request once (mode, inputs, contract, forward, protected files) and
the execution step consumes that resolved state, so a Python caller and the CLI
follow the same path and a direct call cannot bypass the mode checks.

Publication order matters: payload → index → manifest → reports → completed run
envelope. Nothing is written to a hashed file after the envelope is published,
and an interrupted attempt never leaves a readable completed product.
"""

from __future__ import annotations

import io
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image

from ..constants import (
    EVAL_TRANSFORM_CHOICES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SPATIAL_EVAL_SAMPLE_SIZE,
    SPATIAL_PROTOCOL,
    SPATIAL_QUERY_LIMIT,
)
from ..data.provenance import build_spatial_run_record, canonical_json_bytes, sha256_bytes
from ..data.reader import read_dataset
from ..data.records import DataValidationError, FileSnapshot
from ..spatial.diagnostics import RESULT_FIELDS, diagnose_views
from ..spatial.records import (
    DIRECTORY_FEATURES,
    FILENAME_DIAGNOSTICS,
    FILENAME_EVALUATION,
    FILENAME_INDEX,
    FILENAME_LOG,
    FILENAME_MANIFEST,
    FILENAME_MATCHES,
    FILENAME_MODEL_LAYERS,
    FILENAME_OVERLAYS,
    FILENAME_PROVENANCE,
    FILENAME_RESULTS,
    SpatialRecordError,
    SpatialSample,
    SpatialView,
    samples_from_records,
)
from ..spatial.protocol import select_evaluation_keys
from ..spatial.request import (
    MODE_CHECKPOINT,
    MODE_EMBEDDING,
    MODE_ONNX,
    MODE_READ,
    spatial_mode,
    validate_extract_mode_options,
)
from ..spatial.storage import (
    SpatialReader,
    SpatialWriter,
    open_spatial_product,
    write_csv_rows,
)
from ..vision.features import (
    FeatureContract,
    SpatialFeatureForward,
    describe_feature_layers,
    resolve_feature_contract,
    resolve_feature_layers,
)
from ..vision.geometry import build_eval_view, patch_support, pixel_support


class SpatialRequestError(ValueError):
    """The spatial request cannot run as asked."""


@dataclass(frozen=True)
class PreparedSpatialRequest:
    """Attempt-local resolved state; only :func:`prepare_spatial_request` makes one."""

    mode: str
    values: dict[str, Any]
    explicit_options: frozenset[str]
    out_dir: Path
    bundle: Any | None = None
    reader: SpatialReader | None = None
    contract: FeatureContract | None = None
    forward: Any | None = None
    encoder: Any | None = None
    samples: tuple[SpatialSample, ...] = ()
    annotations: dict[str, Any] | None = None
    protected_files: tuple[FileSnapshot, ...] = ()
    notes: tuple[str, ...] = ()
    source_product: dict[str, Any] | None = None
    image_size: int | None = None
    eval_transform: str | None = None
    device: str = "cpu"
    batch_size: int = 16
    layout: str = "per-image"

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _resolve_size(values: Mapping[str, Any], checkpoint_payload: dict) -> int:
    from ..utils.size import resolve_training_image_size, validate_input_size

    requested = values.get("extract_size")
    if requested in (None, "", "auto"):
        return int(resolve_training_image_size(checkpoint_payload))
    try:
        size = int(requested)
    except (TypeError, ValueError):
        raise SpatialRequestError(
            f"--extract-size must be 'auto' or a positive integer, got {requested!r}"
        ) from None
    if size <= 0:
        raise SpatialRequestError("--extract-size must be positive")
    return size


def _load_checkpoint_encoder(values: Mapping[str, Any]):
    from ..data.provenance import snapshot_bytes
    from ..utils.checkpoint import (
        load_checkpoint_bytes,
        load_checkpoint_encoder,
        resolve_checkpoint,
    )

    checkpoint_path = _as_path(values.get("checkpoint"))
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise SpatialRequestError(
            f"checkpoint does not exist: {values.get('checkpoint')!r}"
        )
    # One read: the digest, the deserialized model and the protected-input
    # snapshot all describe the same bytes.
    data, checkpoint_snapshot = snapshot_bytes(checkpoint_path)
    payload = load_checkpoint_bytes(data)
    architecture = resolve_checkpoint(
        payload,
        str(values.get("model_name") or "vit_tiny_patch16_224"),
        prefer_student=bool(values.get("use_student")),
    )
    image_size = _resolve_size(values, payload)
    encoder, resolved = load_checkpoint_encoder(
        payload, architecture, image_size=image_size
    )
    encoder.eval()
    from ..utils.size import validate_input_size

    validate_input_size(image_size, encoder, architecture.model_name)
    return (
        encoder,
        payload,
        architecture,
        resolved,
        image_size,
        checkpoint_path,
        checkpoint_snapshot,
    )


def _protected_inputs(bundle) -> list[FileSnapshot]:
    """Every input a cleanup of the output directory could destroy.

    Source JSON alone is not enough: a panoptic product reads the selected mask
    PNGs through ``SourceIndex.mask_files``, and an output directory pointed at
    the mask root would clear them while the guard only saw the COCO file. The
    snapshots carry the real paths, whose symlink hops the shared
    ``input_dependency_positions`` policy already resolves.
    """
    protected: list[FileSnapshot] = [record.image_file for record in bundle.records]
    if bundle.selector is not None:
        protected.append(bundle.selector)
    for index in bundle.sources.values():
        protected.append(index.file)
        protected.extend(index.mask_files.values())
    return protected


def _onnx_protected_inputs(graph_path: Path) -> list[FileSnapshot]:
    """The graph and every external tensor file it declares.

    The payload names are read from the graph instead of guessed: a legitimate
    model may put its weights anywhere, and a guessed sidecar name leaves a real
    input unprotected against ``--overwrite``.
    """
    from ..data.provenance import snapshot_bytes
    from ..embedding.extractor import _onnx_external_files

    protected = [snapshot_bytes(graph_path)[1]]
    for external in _onnx_external_files(graph_path):
        if external.is_file():
            protected.append(snapshot_bytes(external)[1])
    return protected


def _annotation_snapshot(bundle) -> dict[str, Any] | None:
    """Snapshot the selected source annotations and their categorical rasters.

    Instances and keypoints are copied as source records. A panoptic source is a
    per-image categorical raster plus its ``segments_info``: the source pixel grid
    is read here (from the already verified mask snapshot) and stored with the
    product, so view rasters can be rebuilt after the original COCO and mask paths
    are gone. Nothing is decoded into per-view masks at snapshot time.
    """
    if not bundle.sources:
        return None
    from ..data.coco import ROLE_INSTANCES, ROLE_KEYPOINTS, ROLE_PANOPTIC
    from ..data.masks import decode_panoptic_map
    from ..data.provenance import read_verified_bytes

    records: list[dict[str, Any]] = []
    categories: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    rasters: dict[str, Any] = {}
    for role in (ROLE_INSTANCES, ROLE_KEYPOINTS, ROLE_PANOPTIC):
        index = bundle.sources.get(role)
        if index is None:
            continue
        sources[role] = {
            "path": str(index.file.path),
            "sha256": index.file.sha256,
            "images": len(index.images_by_id),
        }
        # Category IDs are source-local: instances and keypoints may both use id
        # 1 for different things, so the namespace is ``role`` first. Without it
        # one source's names and keypoint schema silently overwrite the other's.
        for category_id, category in index.categories_by_id.items():
            categories[f"{role}:{int(category_id)}"] = dict(category)
        for record in bundle.records:
            for reference in record.coco_images:
                if reference.source_id != role:
                    continue
                annotations = index.annotations_by_image.get(reference.image_id, ())
                for annotation in annotations:
                    role_name = role
                    payload = dict(annotation)
                    if role == ROLE_PANOPTIC:
                        snapshot = index.mask_files.get(reference.image_id)
                        raster = decode_panoptic_map(
                            read_verified_bytes(snapshot),
                            record.width,
                            record.height,
                        )
                        rasters[record.image_key] = raster
                        # One stored entry per image keeps the segment table with
                        # its own raster instead of one record per segment.
                        role_name = ROLE_PANOPTIC
                    records.append(
                        {
                            "annotation_id": annotation.get("id"),
                            "image_key": record.image_key,
                            "category_id": annotation.get("category_id"),
                            "role": role_name,
                            "annotation": payload,
                        }
                    )
    if not records and not sources:
        return None
    return {
        "records": records,
        "categories": categories,
        "sources": sources,
        "rasters": rasters,
    }


def prepare_spatial_request(
    values: Mapping[str, object], *, explicit_options: frozenset[str]
) -> PreparedSpatialRequest:
    """Validate the mode matrix and resolve one attempt-local request."""
    notes = validate_extract_mode_options(
        values, explicit_options=explicit_options
    )
    resolved = dict(values)
    mode = spatial_mode(resolved)
    if mode == MODE_EMBEDDING:
        raise SpatialRequestError(
            "prepare_spatial_request handles spatial output only; pass "
            "--output-kind spatial"
        )
    out_dir = Path(str(resolved.get("out_dir") or "runs/extract")).expanduser()
    if mode == MODE_READ:
        prepared = _prepare_read(resolved, explicit_options, out_dir, notes)
    elif mode == MODE_ONNX:
        prepared = _prepare_onnx(resolved, explicit_options, out_dir, notes)
    else:
        prepared = _prepare_checkpoint(resolved, explicit_options, out_dir, notes)
    try:
        _require_separate_output(prepared)
    except Exception:
        prepared.close()
        raise
    return prepared


def _overlay_inputs(images_dir: Path, reader: Any) -> list[FileSnapshot]:
    """The selected original images an overlay would read.

    Containment is decided *before* any byte is read: a canonical key can still
    name a symlink inside the root that resolves to a file outside it, and
    hashing it here would already have read the escaping file.
    """
    from ..data.paths import require_contained
    from ..data.provenance import snapshot_bytes

    protected: list[FileSnapshot] = []
    for key in reader.image_keys:
        path = images_dir / key
        if not path.is_file():
            continue
        try:
            resolved = require_contained(images_dir, path, key)
        except ValueError:
            # The overlay reports this image as unavailable; nothing to protect
            # and nothing to read.
            continue
        try:
            protected.append(snapshot_bytes(resolved)[1])
        except OSError:  # pragma: no cover - unreadable image is reported later
            continue
    return protected


def _require_separate_output(prepared: PreparedSpatialRequest) -> None:
    """Refuse an output directory that would write over a declared input.

    The CLI clears the directory before the run, so a Python caller was the only
    path that could reach a report write with an input inside it - for example a
    selector CSV placed where ``spatial_results.csv`` goes, which the report then
    overwrote while the file sat in ``protected_files``. Collecting the protection
    list is not enough; the core executes the same non-overlap rule.
    """
    from ..utils.paths import input_dependency_positions

    target = prepared.out_dir.expanduser().resolve()
    for snapshot in prepared.protected_files:
        entry = Path(snapshot.path)
        for position in input_dependency_positions(entry):
            if position == target or target in position.parents:
                raise SpatialRequestError(
                    f"--out-dir {target} contains the input {position}, which this "
                    f"run reads through {entry}; choose a directory that cannot "
                    "clear or overwrite it"
                )
        resolved = entry.resolve()
        if target == resolved or target in resolved.parents:
            raise SpatialRequestError(
                f"--out-dir {target} is inside the protected input {resolved}"
            )


def _prepare_read(
    values: dict[str, Any],
    explicit_options: frozenset[str],
    out_dir: Path,
    notes: list[str],
) -> PreparedSpatialRequest:
    source = _as_path(values.get("spatial_input"))
    if source is None:
        raise SpatialRequestError("--spatial-input needs a product directory")
    if not source.is_dir():
        raise SpatialRequestError(f"spatial product does not exist: {source}")
    selector = _as_path(values.get("label_csv"))
    reader = open_spatial_product(source, selector_csv=selector)
    source_resolved = source.resolve()
    target = out_dir.resolve()
    if target == source_resolved or source_resolved in target.parents:
        reader.close()
        raise SpatialRequestError(
            f"--out-dir {target} is inside the source product {source_resolved}; "
            "re-evaluation writes a separate run directory outside it"
        )
    if target in source_resolved.parents:
        reader.close()
        raise SpatialRequestError(
            f"--out-dir {target} contains the source product {source_resolved}; "
            "choose a directory that cannot clear it"
        )
    protected: list[FileSnapshot] = []
    if selector is not None and selector.is_file():
        from ..data.provenance import snapshot_bytes

        protected.append(snapshot_bytes(selector)[1])
    for item in reader.manifest["files"]:
        path = source / item["path"]
        if path.is_file():
            protected.append(
                FileSnapshot(
                    path=path, sha256=item["sha256"], size_bytes=int(item["bytes"])
                )
            )
    manifest_path = source / FILENAME_MANIFEST
    protected.append(
        FileSnapshot(
            path=manifest_path,
            sha256=sha256_bytes(manifest_path.read_bytes()),
            size_bytes=manifest_path.stat().st_size,
        )
    )
    plan = reader.evaluation_plan
    if plan is not None and "seed" in explicit_options:
        stored_seed = plan.get("seed")
        if stored_seed is not None and int(values.get("seed")) != int(stored_seed):
            reader.close()
            raise SpatialRequestError(
                f"--seed {values.get('seed')} does not match the saved evaluation "
                f"plan seed {stored_seed}; a stored plan fixes its seed"
            )
    if values.get("spatial_eval"):
        if plan is None:
            reader.close()
            raise SpatialRequestError(
                "missing_paired_views: this product has no stored paired views or "
                "evaluation plan, so --spatial-eval cannot be satisfied from it"
            )
        # An explicit evaluation must be decidable before the output changes: the
        # selection/plan intersection is what the report will cover.
        planned = set(plan.get("selected_keys", []))
        if not [key for key in reader.image_keys if key in planned]:
            reader.close()
            raise SpatialRequestError(
                "no stored planned pairs were selected; the saved evaluation plan "
                f"covers {sorted(planned)[:3]}{'...' if len(planned) > 3 else ''} "
                "and the selection contains none of them"
            )
    images = _as_path(values.get("input_images_dir"))
    if images is not None:
        # The overlay is drawn from these files, and its figures are written under
        # the output, so the selected originals must be protected exactly like any
        # other input: an out-dir whose overlay directory holds them would replace
        # the image being measured.
        protected.extend(_overlay_inputs(images, reader))
        notes.append(
            f"optional image overlay requested from {images}; stored numeric "
            "results stay readable when images are missing or mismatched"
        )
    return PreparedSpatialRequest(
        mode=MODE_READ,
        values=values,
        explicit_options=explicit_options,
        out_dir=out_dir,
        reader=reader,
        contract=reader.contract,
        protected_files=tuple(protected),
        notes=tuple(notes),
        source_product={
            "path": str(source),
            "product_id": reader.product_id,
            "layout": reader.manifest["layout"],
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        },
        eval_transform=None,
        image_size=reader.contract.input_hw[0],
    )


def _prepare_checkpoint(
    values: dict[str, Any],
    explicit_options: frozenset[str],
    out_dir: Path,
    notes: list[str],
) -> PreparedSpatialRequest:
    images_dir = _as_path(values.get("input_images_dir"))
    if images_dir is None:
        raise SpatialRequestError(
            "--input-images-dir is required for spatial image extraction"
        )
    encoder, payload, architecture, resolved, image_size, checkpoint_path, (
        checkpoint_snapshot
    ) = _load_checkpoint_encoder(values)
    bundle = read_dataset(
        images_dir,
        manifest=_as_path(values.get("label_csv")),
        coco_instances=_as_path(values.get("coco_instances")),
        coco_keypoints=_as_path(values.get("coco_keypoints")),
        coco_panoptic=_as_path(values.get("coco_panoptic")),
        panoptic_masks_dir=_as_path(values.get("panoptic_masks_dir")),
        selection_policy="csv-or-directory",
    )
    if not bundle.validation.valid:
        raise DataValidationError(bundle.validation)
    layers = resolve_feature_layers(encoder.backbone, values.get("feature_layers"))
    contract = resolve_feature_contract(
        encoder,
        image_size,
        layers,
        norm=str(values.get("feature_norm") or "backbone"),
        l2=bool(values.get("feature_l2_normalize")),
        model_identity={
            "architecture": architecture.model_name,
            "checkpoint_sha256": checkpoint_snapshot.sha256,
            "weight_source": resolved.source_key,
            "register_tokens": resolved.actual_register_tokens,
            "added_register_tokens": resolved.added_register_tokens,
            "ssl_identity": list(resolved.ssl_identity or ()),
        },
        normalization={
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "space": "rgb01",
        },
    )
    # The embedding default stays center-crop; spatial defaults to
    # whole-specimen-pad and an explicit request wins.
    if "eval_transform" in explicit_options:
        eval_transform = str(values.get("eval_transform"))
    else:
        eval_transform = "whole-specimen-pad"
    if eval_transform not in EVAL_TRANSFORM_CHOICES:
        raise SpatialRequestError(
            f"unknown --eval-transform {eval_transform!r}; expected one of "
            f"{EVAL_TRANSFORM_CHOICES}"
        )
    samples = tuple(samples_from_records(bundle.records))
    _require_evaluable_selection(values, samples)
    protected = _protected_inputs(bundle)
    protected.append(checkpoint_snapshot)
    return PreparedSpatialRequest(
        mode=MODE_CHECKPOINT,
        values=values,
        explicit_options=explicit_options,
        out_dir=out_dir,
        bundle=bundle,
        contract=contract,
        forward=SpatialFeatureForward(encoder, contract),
        encoder=encoder,
        samples=samples,
        annotations=_annotation_snapshot(bundle),
        protected_files=tuple(protected),
        notes=tuple(notes),
        image_size=image_size,
        eval_transform=eval_transform,
        device=str(values.get("device") or "cpu"),
        batch_size=int(values.get("batch_size") or 16),
        layout=str(values.get("spatial_layout") or "per-image"),
    )


def _effective_checkpoint(prepared: PreparedSpatialRequest) -> dict[str, Any]:
    identity = prepared.contract.model_identity
    return {
        "mode": MODE_CHECKPOINT,
        "input_size": prepared.image_size,
        "eval_transform": prepared.eval_transform,
        "layout": prepared.layout,
        "feature_contract": prepared.contract.to_json(),
        "input_normalization": dict(prepared.contract.normalization),
        "device": prepared.device,
        "batch_size": prepared.batch_size,
        "checkpoint": str(prepared.values.get("checkpoint")),
        "model_name": identity.get("architecture"),
        "checkpoint_sha256": identity.get("checkpoint_sha256"),
        "weight_source": identity.get("weight_source"),
        "register_tokens": identity.get("register_tokens"),
        "ssl_identity": identity.get("ssl_identity"),
        "notes": list(prepared.notes),
    }


def _prepare_onnx(
    values: dict[str, Any],
    explicit_options: frozenset[str],
    out_dir: Path,
    notes: list[str],
) -> PreparedSpatialRequest:
    """Resolve a spatial graph: the exported contract fixes the representation."""
    from ..constants import IMAGENET_MEAN, IMAGENET_STD
    from ..data.provenance import snapshot_bytes
    from ..vision.export import OnnxSpatialForward

    graph_path = _as_path(values.get("onnx_path"))
    if graph_path is None or not graph_path.is_file():
        raise SpatialRequestError(f"spatial graph does not exist: {graph_path}")
    images_dir = _as_path(values.get("input_images_dir"))
    if images_dir is None:
        raise SpatialRequestError(
            "--input-images-dir is required for spatial image extraction"
        )
    forward = OnnxSpatialForward(graph_path)
    contract = forward.contract

    # The graph owns layers, normalization and size: an explicit request must
    # match it exactly rather than silently inherit a different meaning.
    if "feature_layers" in explicit_options:
        requested = [
            entry.strip()
            for entry in str(values.get("feature_layers") or "").split(",")
            if entry.strip()
        ]
        if requested != list(contract.layer_paths):
            raise SpatialRequestError(
                f"--feature-layers {requested} does not match the graph's exported "
                f"layers {list(contract.layer_paths)}"
            )
    if "feature_norm" in explicit_options and str(
        values.get("feature_norm")
    ) != contract.norm:
        raise SpatialRequestError(
            f"--feature-norm {values.get('feature_norm')!r} does not match the graph's "
            f"exported normalization {contract.norm!r}"
        )
    if "feature_l2_normalize" in explicit_options and bool(
        values.get("feature_l2_normalize")
    ) != bool(contract.l2):
        raise SpatialRequestError(
            f"--feature-l2-normalize {bool(values.get('feature_l2_normalize'))} does "
            f"not match the graph's exported L2 setting {contract.l2}"
        )
    if "extract_size" in explicit_options:
        requested_size = values.get("extract_size")
        if str(requested_size) not in ("auto", "", "None"):
            if int(requested_size) != int(contract.input_hw[0]):
                raise SpatialRequestError(
                    f"--extract-size {requested_size} does not match the graph's fixed "
                    f"input size {contract.input_hw[0]}"
                )
    declared = dict(contract.normalization or {})
    # The whole normalization is compared, not only the mean: a graph that changes
    # the std or the pixel space receives a different input than this build
    # produces, so inference would silently differ from the graph's declaration.
    expected_normalization = {
        "mean": [float(v) for v in IMAGENET_MEAN],
        "std": [float(v) for v in IMAGENET_STD],
        "space": "rgb01",
    }
    for field, expected in expected_normalization.items():
        if field not in declared:
            raise SpatialRequestError(
                f"the graph declares no input {field!r}; this build applies "
                f"{field}={expected}"
            )
        declared_value = declared[field]
        if field == "space":
            matched = str(declared_value) == expected
        else:
            try:
                matched = [float(v) for v in declared_value] == expected
            except (TypeError, ValueError):
                matched = False
        if not matched:
            raise SpatialRequestError(
                f"the graph declares input {field}={declared_value!r}, which this "
                f"build does not apply; supported {field}={expected}"
            )

    bundle = read_dataset(
        images_dir,
        manifest=_as_path(values.get("label_csv")),
        coco_instances=_as_path(values.get("coco_instances")),
        coco_keypoints=_as_path(values.get("coco_keypoints")),
        coco_panoptic=_as_path(values.get("coco_panoptic")),
        panoptic_masks_dir=_as_path(values.get("panoptic_masks_dir")),
        selection_policy="csv-or-directory",
    )
    if not bundle.validation.valid:
        raise DataValidationError(bundle.validation)
    eval_transform = (
        str(values.get("eval_transform"))
        if "eval_transform" in explicit_options
        else "whole-specimen-pad"
    )
    if eval_transform not in EVAL_TRANSFORM_CHOICES:
        raise SpatialRequestError(
            f"unknown --eval-transform {eval_transform!r}; expected one of "
            f"{EVAL_TRANSFORM_CHOICES}"
        )
    protected: list[FileSnapshot] = _protected_inputs(bundle)
    protected.extend(_onnx_protected_inputs(graph_path))
    notes.append(
        "the graph's declared contract fixes layers, normalization, L2 and input "
        "size; numeric agreement was validated at export time"
    )
    onnx_samples = tuple(samples_from_records(bundle.records))
    _require_evaluable_selection(values, onnx_samples)
    return PreparedSpatialRequest(
        mode=MODE_ONNX,
        values=values,
        explicit_options=explicit_options,
        out_dir=out_dir,
        bundle=bundle,
        contract=contract,
        forward=forward,
        samples=onnx_samples,
        annotations=_annotation_snapshot(bundle),
        protected_files=tuple(protected),
        notes=tuple(notes),
        image_size=int(contract.input_hw[0]),
        eval_transform=eval_transform,
        device="cpu (onnxruntime)",
        batch_size=int(values.get("batch_size") or 16),
        layout=str(values.get("spatial_layout") or "per-image"),
    )


def _effective_onnx(prepared: PreparedSpatialRequest) -> dict[str, Any]:
    return {
        "mode": MODE_ONNX,
        "input_size": prepared.image_size,
        "eval_transform": prepared.eval_transform,
        "layout": prepared.layout,
        "feature_contract": prepared.contract.to_json(),
        "input_normalization": dict(prepared.contract.normalization),
        "device": prepared.device,
        "batch_size": prepared.batch_size,
        "onnx_path": str(prepared.values.get("onnx_path")),
        "backend": "onnxruntime-cpu",
        "notes": list(prepared.notes),
    }


def _decode(snapshot: FileSnapshot) -> Image.Image:
    from ..data.provenance import read_verified_bytes

    payload = read_verified_bytes(snapshot)
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB")


@contextmanager
def _backend_context(prepared: PreparedSpatialRequest) -> Iterator[Any]:
    """Torch inference context for a checkpoint; a pass-through for ONNX."""
    if prepared.mode == MODE_ONNX:
        yield prepared.device
        return
    with _torch_inference(prepared) as device:
        yield device


@contextmanager
def _torch_inference(prepared: PreparedSpatialRequest) -> Iterator[Any]:
    import torch

    from ..utils.device import resolve_device

    device = resolve_device(prepared.device)
    model = prepared.encoder
    was_training = model.training
    model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            yield device
    finally:
        model.train(was_training)


def _write_model_layers(out_dir: Path, encoder: Any) -> Path:
    path = out_dir / FILENAME_MODEL_LAYERS
    from ..vision.features import write_model_layers

    return write_model_layers(path, encoder.backbone)


def _run_checkpoint_extraction(
    prepared: PreparedSpatialRequest, writer: SpatialWriter,
) -> tuple[SpatialWriter, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Write payload/index/manifest, then return the sealed writer and reports."""
    import torch

    assert prepared.bundle is not None
    assert prepared.contract is not None
    out_dir = prepared.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(prepared.bundle.records)
    # Resolve the evaluation subset before the first payload is written: an empty
    # intersection or a rejected plan must not reach an output that was already
    # cleared.
    keep_keys = set(_evaluation_keys(prepared, [record.image_key for record in records]))
    base_views: dict[str, SpatialView] = {}
    try:
        with _backend_context(prepared) as device:
            batch: list[tuple[Any, Any, Any, str]] = []
            for record in records:
                image = _decode(record.image_file)
                tensor, trace = build_eval_view(
                    image, name=str(prepared.eval_transform), size=int(prepared.image_size)
                )
                batch.append((record, tensor, trace, "base"))
                if len(batch) >= prepared.batch_size:
                    base_views.update(
                        _flush_batch(writer, prepared, batch, device, keep=keep_keys)
                    )
                    batch = []
            if batch:
                base_views.update(
                    _flush_batch(writer, prepared, batch, device, keep=keep_keys)
                )
        if keep_keys:
            store_evaluation_views(
                prepared,
                writer,
                base_views=base_views,
                image_limit=_evaluation_limit(prepared),
            )
    except Exception:
        writer.abort({"reason": "extraction failed"})
        raise
    manifest = writer.finish()
    if prepared.values.get("dump_model_structure"):
        _write_model_layers(out_dir, prepared.encoder)
    reader = writer.reader()
    try:
        rows, summary = diagnose_views(
            reader.iter_views(), prepared.contract, plan=reader.evaluation_plan
        )
    finally:
        reader.close()
    _attach_sample_ids(rows, out_dir)
    write_csv_rows(out_dir / FILENAME_RESULTS, rows, RESULT_FIELDS)
    (out_dir / FILENAME_DIAGNOSTICS).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return writer, manifest, rows, summary


def _attach_sample_ids(rows: list[dict[str, Any]], out_dir: Path) -> None:
    order: dict[str, int] = {}
    index_path = out_dir / FILENAME_INDEX
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            order.setdefault(entry["sample"]["image_key"], int(entry["sample_id"]))
    for row in rows:
        row["sample_id"] = order.get(row["image_key"])


def _run_forward(
    prepared: PreparedSpatialRequest, tensors, device: Any
) -> tuple[np.ndarray, ...]:
    """One batch through whichever backend this request resolved to."""
    import torch

    return _forward_numpy(
        prepared, torch.stack([item[1] for item in tensors]), device
    )


def _evaluation_keys(
    prepared: PreparedSpatialRequest, keys: Sequence[str]
) -> tuple[str, ...]:
    """The evaluated subset, resolved before any output changes.

    An explicit evaluation may only cover images that will be sampled, so the
    empty case is a rejected request rather than a failure reached after payloads
    were already written and an existing output had been cleared.
    """
    if not prepared.values.get("spatial_eval"):
        return ()
    sampled = select_evaluation_keys(
        list(keys),
        seed=_evaluation_seed(prepared),
        limit=_evaluation_limit(prepared),
    )
    if not sampled:
        raise SpatialRequestError(
            "the selection resolved no image for spatial evaluation; "
            "--spatial-eval needs at least one selected image"
        )
    return tuple(sampled)


def _require_evaluable_selection(
    values: Mapping[str, Any], samples: Sequence[SpatialSample]
) -> None:
    """Fail a request that cannot evaluate anything before its output is cleared."""
    if not values.get("spatial_eval"):
        return
    raw_limit = values.get("spatial_eval_sample_size")
    limit = SPATIAL_EVAL_SAMPLE_SIZE if raw_limit is None else int(raw_limit)
    raw_seed = values.get("seed")
    seed = 42 if raw_seed is None else int(raw_seed)
    if not select_evaluation_keys(
        [sample.image_key for sample in samples], seed=seed, limit=limit
    ):
        raise SpatialRequestError(
            "--spatial-eval needs at least one selected image; this request "
            "selects none"
        )


def _evaluation_seed(prepared: PreparedSpatialRequest) -> int:
    raw_seed = prepared.values.get("seed")
    return 42 if raw_seed is None else int(raw_seed)


def _evaluation_limit(prepared: PreparedSpatialRequest) -> int:
    raw_limit = prepared.values.get("spatial_eval_sample_size")
    return SPATIAL_EVAL_SAMPLE_SIZE if raw_limit is None else int(raw_limit)


def _base_view_views(
    record: Any,
    trace: Any,
    contract: FeatureContract,
    outputs: Sequence[np.ndarray],
    position: int,
    view_id: str = "base",
) -> SpatialView:
    """One base view from already-computed batch output."""
    grid_h, grid_w = (
        int(contract.input_hw[0]) // int(contract.patch_hw[0]),
        int(contract.input_hw[1]) // int(contract.patch_hw[1]),
    )
    coverage, pixel_valid = pixel_support(trace)
    patch_coverage, patch_valid = patch_support(
        trace,
        pixel_sampling_valid=pixel_valid,
        grid_hw=(grid_h, grid_w),
        patch_hw=contract.patch_hw,
    )
    return SpatialView(
        image_key=record.image_key,
        view_id=view_id,
        role="base",
        trace=trace,
        features={
            path: np.asarray(outputs[index][position], dtype=np.float32)
            for index, path in enumerate(contract.layer_paths)
        },
        pixel_geometric_coverage=coverage,
        pixel_sampling_valid=pixel_valid,
        geometric_coverage=patch_coverage,
        patch_sampling_valid=patch_valid,
    )


def _flush_batch(
    writer: SpatialWriter,
    prepared: PreparedSpatialRequest,
    batch: Sequence[tuple[Any, Any, Any, str]],
    device: Any,
    *,
    keep: set[str] | None = None,
) -> dict[str, SpatialView]:
    """Write one batch and return the base views the caller asked to keep.

    ``keep`` bounds retention: only evaluation-sampled base grids survive the
    call, because every other base view is already sealed in its payload. Holding
    all of them would grow with the dataset rather than with the evaluation
    budget.
    """
    contract = prepared.contract
    assert contract is not None
    outputs = _run_forward(prepared, list(batch), device)
    written: dict[str, SpatialView] = {}
    for position, (record, _tensor, trace, _view_id) in enumerate(batch):
        view = _base_view_views(record, trace, contract, outputs, position)
        writer.write(view)
        if keep is None or record.image_key in keep:
            written[record.image_key] = view
    return written


def _evaluate_sealed_product(
    prepared: PreparedSpatialRequest, writer: SpatialWriter, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the plan from the sealed bytes, via the writer's staging reader.

    The same matcher serves online and offline evaluation, so an online score and
    a later re-evaluation of the same product are the same computation.
    """
    from .evaluation import evaluate_spatial, write_spatial_evaluation

    reader = writer.reader()
    try:
        keys = list(
            dict.fromkeys(
                entry["sample"]["image_key"] for entry in reader._index  # noqa: SLF001
            )
        )
        eval_rows, eval_summary = evaluate_spatial(reader, selected_keys=keys)
        write_spatial_evaluation(
            prepared.out_dir,
            eval_rows,
            eval_summary,
            reader=reader,
            images_dir=_as_path(prepared.values.get("input_images_dir")),
        )
        return {
            "status": "completed",
            "protocol": eval_summary["protocol"],
            "seed": eval_summary["seed"],
            "counts": eval_summary["counts"],
            "matches": str(Path(prepared.out_dir) / FILENAME_MATCHES),
        }
    finally:
        reader.close()


def _artifact_rows(out_dir: Path, *, include_log: bool) -> list[dict[str, Any]]:
    """Final artifacts with their on-disk hashes, excluding the envelope itself.

    Only files this attempt owns are recorded: a symlink is never followed (a
    planted one would make the run hash and register a file outside the output
    directory), and an entry whose real path leaves the directory is refused.
    """
    root = Path(out_dir)
    resolved_root = root.resolve()
    owned = {
        FILENAME_RESULTS, FILENAME_DIAGNOSTICS, FILENAME_EVALUATION,
        FILENAME_MATCHES, FILENAME_MODEL_LAYERS,
    }
    if include_log:
        owned.add(FILENAME_LOG)
    manifest_path = root / FILENAME_MANIFEST
    if manifest_path.is_file() and not manifest_path.is_symlink():
        _refuse_symlink(manifest_path, root)
        manifest = json.loads(manifest_path.read_bytes())
        owned.add(FILENAME_MANIFEST)
        owned.update(item["path"] for item in manifest["files"])
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in owned and not relative.startswith(f"{FILENAME_OVERLAYS}/"):
            continue
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - defensive
            continue
        if resolved_root not in resolved.parents:
            continue
        payload = path.read_bytes()
        rows.append(
            {"path": relative, "sha256": sha256_bytes(payload), "bytes": len(payload)}
        )
    return rows


def publish_spatial_run(
    out_dir: Path, record: Mapping[str, Any], *, include_log: bool = True
) -> dict[str, Any]:
    """Hash final artifacts and publish the completed envelope atomically."""
    root = Path(out_dir)
    final = dict(record)
    if final.get("status") == "completed":
        final["artifacts"] = _artifact_rows(root, include_log=include_log)
    envelope = root / FILENAME_PROVENANCE
    temporary = envelope.with_suffix(".json.part")
    # Publication is still part of this attempt's ownership: a symlink planted at
    # either name would otherwise send the envelope bytes outside the directory.
    for path in (temporary, envelope):
        _refuse_symlink(path, root)
    temporary.write_bytes(canonical_json_bytes(final))
    temporary.replace(envelope)
    return final


def _refuse_symlink(path: Path, root: Path) -> None:
    """Reject a symlink or an escaping parent before writing an attempt file."""
    if path.is_symlink():
        raise SpatialRequestError(
            f"refusing to write through the symlink {path}; a run owns its own files"
        )
    try:
        resolved_root = root.resolve()
        parent = path.parent.resolve()
    except OSError as error:  # pragma: no cover - defensive
        raise SpatialRequestError(f"cannot resolve {path}: {error}") from error
    if parent != resolved_root and resolved_root not in parent.parents:
        raise SpatialRequestError(
            f"refusing to write {path}; it resolves outside {resolved_root}"
        )


def extract_spatial(
    prepared: PreparedSpatialRequest, *, defer_publication: bool = False
) -> dict[str, Any]:
    """Run one spatial extraction and publish its completed run."""
    if prepared.mode not in (MODE_CHECKPOINT, MODE_ONNX):
        raise SpatialRequestError(
            "extract_spatial needs a checkpoint or graph request, not "
            f"{prepared.mode!r}"
        )
    if not prepared.samples:
        raise SpatialRequestError(
            "the selection resolved no images; nothing would be stored"
        )
    started = _utc_now()
    out_dir = prepared.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _claim_report_directory(out_dir)
    # Refusal to claim a previous product must not write a new failure over it.
    writer = SpatialWriter(
        out_dir,
        contract=prepared.contract,
        layout=prepared.layout,
        samples=list(prepared.samples),
        annotations=prepared.annotations,
    )
    try:
        writer, manifest, rows, summary = _run_checkpoint_extraction(prepared, writer)
        evaluation_summary = None
        if prepared.values.get("spatial_eval"):
            evaluation_summary = _evaluate_sealed_product(prepared, writer, manifest)
        effective = (
            _effective_onnx(prepared)
            if prepared.mode == MODE_ONNX
            else _effective_checkpoint(prepared)
        )
        record = build_spatial_run_record(
            requested=dict(prepared.values),
            bundle=prepared.bundle,
            source_product=None,
            effective=effective,
            used_files=prepared.protected_files,
            evaluation=evaluation_summary,
            status="completed",
            started_at=started,
            ended_at=_utc_now(),
            artifacts=[],
        )
        if defer_publication:
            return {
                "status": "completed",
                "manifest": manifest,
                "rows": rows,
                "summary": summary,
                "evaluation": evaluation_summary,
                "envelope": record,
                "out_dir": str(out_dir),
                "deferred": True,
            }
        published = publish_spatial_run(out_dir, record, include_log=False)
        return {
            "status": "completed",
            "manifest": manifest,
            "rows": rows,
            "summary": summary,
            "evaluation": evaluation_summary,
            "envelope": published,
            "out_dir": str(out_dir),
            "deferred": False,
        }
    except BaseException as error:
        _record_spatial_failure(out_dir, prepared, started, error)
        raise


def reevaluate_spatial(
    prepared: PreparedSpatialRequest, *, defer_publication: bool = False
) -> dict[str, Any]:
    """Read a stored product and write report-only diagnostics beside it.

    A public Python caller reaches this without the CLI's output preparation, so
    the report directory is claimed here: existing files are refused rather than
    overwritten, and a failure is recorded as this attempt's envelope instead of
    leaving the previous attempt's status to describe the new run.
    """
    if prepared.mode != MODE_READ or prepared.reader is None:
        raise SpatialRequestError("reevaluate_spatial needs a spatial read request")
    reader = prepared.reader
    out_dir = prepared.out_dir
    started = _utc_now()
    _claim_report_directory(out_dir)
    try:
        written = _write_read_reports(prepared, reader, out_dir, started)
    except BaseException as error:
        _record_spatial_failure(out_dir, prepared, started, error)
        raise
    rows, summary, evaluation_summary, record = written
    if defer_publication:
        return {
            "status": "completed",
            "rows": rows,
            "summary": summary,
            "evaluation": evaluation_summary,
            "envelope": record,
            "out_dir": str(out_dir),
            "deferred": True,
        }
    # Publication also hashes and renames attempt files, so it can fail too; the
    # failure belongs to this attempt's status, not outside the handler above.
    try:
        published = publish_spatial_run(out_dir, record, include_log=False)
    except BaseException as error:
        _record_spatial_failure(out_dir, prepared, started, error)
        raise
    return {
        "status": "completed",
        "rows": rows,
        "summary": summary,
        "evaluation": evaluation_summary,
        "envelope": published,
        "out_dir": str(out_dir),
        "deferred": False,
    }


def _claim_report_directory(out_dir: Path) -> None:
    """Refuse a report directory that already holds this attempt's outputs.

    The claim covers every name the attempt writes - reports, the publication
    temporary and the overlay directory - and refuses a symlink at any of them.
    Checking only the published report names let a planted
    ``run_provenance.json.part`` symlink receive the completed envelope anywhere
    on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    claimed = (
        FILENAME_RESULTS,
        FILENAME_DIAGNOSTICS,
        FILENAME_EVALUATION,
        FILENAME_MATCHES,
        FILENAME_PROVENANCE,
        f"{Path(FILENAME_PROVENANCE).stem}.json.part",
        FILENAME_OVERLAYS,
        FILENAME_MODEL_LAYERS,
    )
    occupied: list[str] = []
    for name in claimed:
        path = out_dir / name
        # ``lexists`` sees a dangling symlink that ``exists`` reports as absent.
        if path.is_symlink() or path.exists():
            occupied.append(name)
    if occupied:
        raise SpatialRequestError(
            f"the read-report directory {out_dir} already holds "
            + ", ".join(occupied)
            + "; choose another --out-dir or clear it explicitly"
        )


def _record_spatial_failure(
    out_dir: Path,
    prepared: PreparedSpatialRequest,
    started: str,
    error: BaseException,
) -> None:
    """Publish this attempt's failed envelope, so no earlier status survives."""
    reason = (
        "interrupted by the user"
        if isinstance(error, KeyboardInterrupt)
        else str(error)
    )
    record = build_spatial_run_record(
        requested=dict(prepared.values),
        bundle=None,
        source_product=prepared.source_product,
        effective={"mode": prepared.mode},
        used_files=prepared.protected_files,
        evaluation=None,
        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
        failure={"reason": reason, "type": type(error).__name__},
        started_at=started,
        ended_at=_utc_now(),
        artifacts=[],
    )
    try:
        publish_spatial_run(out_dir, record, include_log=False)
    except OSError:  # pragma: no cover - the original failure is what matters
        pass


def _write_read_reports(
    prepared: PreparedSpatialRequest,
    reader: Any,
    out_dir: Path,
    started: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Compute and write the read reports; returns rows, summary and record."""
    rows, summary = diagnose_views(
        reader.iter_views(), reader.contract, plan=reader.evaluation_plan
    )
    _attach_reader_sample_ids(rows, reader)
    # A selector CSV is a report overlay, so every read report carries it - not
    # only the evaluation report. Diagnostics-only re-reads used to accept the CSV
    # and drop its labels.
    summary["sample_metadata"] = reader.overlay_report()
    write_csv_rows(out_dir / FILENAME_RESULTS, rows, RESULT_FIELDS)
    (out_dir / FILENAME_DIAGNOSTICS).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    evaluation_summary = None
    if prepared.values.get("spatial_eval"):
        from .evaluation import evaluate_spatial, write_spatial_evaluation

        eval_rows, eval_summary = evaluate_spatial(
            reader, selected_keys=list(reader.image_keys)
        )
        write_spatial_evaluation(
            out_dir,
            eval_rows,
            eval_summary,
            reader=reader,
            images_dir=_as_path(prepared.values.get("input_images_dir")),
        )
        evaluation_summary = {
            "status": "completed",
            "protocol": eval_summary["protocol"],
            "seed": eval_summary["seed"],
            "counts": eval_summary["counts"],
            "matches": str(out_dir / FILENAME_MATCHES),
        }
    effective = {
        "mode": MODE_READ,
        "source_product": str(prepared.values.get("spatial_input")),
        "product_id": reader.product_id,
        "layout": reader.manifest["layout"],
        "feature_contract": reader.contract.to_json(),
        "verification_scope": reader.verification_scope().to_json(),
        "sample_selection": {
            "rule": "artifact-key-resolution-v1 if --label-csv supplied, else index order",
            "selected_keys": list(reader.image_keys),
        },
        "metadata_overlay": reader.overlay_report(),
        "notes": list(prepared.notes),
    }
    record = build_spatial_run_record(
        requested=dict(prepared.values),
        bundle=None,
        source_product=prepared.source_product,
        effective=effective,
        used_files=prepared.protected_files,
        evaluation=evaluation_summary,
        status="completed",
        started_at=started,
        ended_at=_utc_now(),
        artifacts=[],
    )
    return rows, summary, evaluation_summary, record


def _forward_numpy(
    prepared: PreparedSpatialRequest, tensor_batch, device: Any
) -> tuple[np.ndarray, ...]:
    """One stacked batch through whichever backend this request resolved to."""
    if prepared.mode == MODE_ONNX:
        return tuple(
            prepared.forward(np.stack([tensor.numpy() for tensor in tensor_batch]))
        )
    produced = prepared.forward(tensor_batch.to(device))
    return tuple(
        value.detach().to("cpu").numpy().astype(np.float32) for value in produced
    )


def _assemble_view(
    prepared: PreparedSpatialRequest,
    image_key: str,
    view_id: str,
    role: str,
    trace: Any,
    outputs: Sequence[np.ndarray],
    position: int,
) -> SpatialView:
    """One stored view: shared support plus this backend's feature grids."""
    contract = prepared.contract
    assert contract is not None
    grid_h, grid_w = (
        int(contract.input_hw[0]) // int(contract.patch_hw[0]),
        int(contract.input_hw[1]) // int(contract.patch_hw[1]),
    )
    coverage, pixel_valid = pixel_support(trace)
    patch_coverage, patch_valid = patch_support(
        trace,
        pixel_sampling_valid=pixel_valid,
        grid_hw=(grid_h, grid_w),
        patch_hw=contract.patch_hw,
    )
    features = {
        path: np.asarray(outputs[index][position], dtype=np.float32)
        for index, path in enumerate(contract.layer_paths)
    }
    return SpatialView(
        image_key=image_key,
        view_id=view_id,
        role=role,
        trace=trace,
        features=features,
        pixel_geometric_coverage=coverage,
        pixel_sampling_valid=pixel_valid,
        geometric_coverage=patch_coverage,
        patch_sampling_valid=patch_valid,
    )


def store_evaluation_views(
    prepared: PreparedSpatialRequest,
    writer: SpatialWriter,
    *,
    base_views: Mapping[str, SpatialView],
    image_limit: int = SPATIAL_EVAL_SAMPLE_SIZE,
    query_limit: int = SPATIAL_QUERY_LIMIT,
) -> dict[str, Any]:
    """Store the frozen protocol views for the sampled keys and seal the plan.

    Only evaluated keys get extra views: their stored base view is the protocol
    reference (no second copy), plus the eleven transforms. The plan is built from
    stored support only, so no feature value, label or true location influences
    which images or queries are used.
    """
    import torch

    from ..spatial.protocol import build_evaluation_plan, build_evaluation_views

    assert prepared.contract is not None
    seed = _evaluation_seed(prepared)
    records = {record.image_key: record for record in prepared.bundle.records}
    selected = select_evaluation_keys(list(base_views), seed=seed, limit=image_limit)
    views_by_key: dict[str, list[SpatialView]] = {}
    with _backend_context(prepared) as device:
        for key in selected:
            image = _decode(records[key].image_file)
            built = build_evaluation_views(
                image,
                image_key=key,
                eval_transform=str(prepared.eval_transform),
                size=int(prepared.image_size),
            )
            transformed = [
                (view_id, tensor, trace)
                for view_id, tensor, trace in built
                if view_id != "reference"
            ]
            tensor_batch = torch.stack([tensor for _, tensor, _ in transformed])
            outputs = _forward_numpy(prepared, tensor_batch, device)
            views = [
                _assemble_view(
                    prepared, key, view_id, "eval-transform", trace, outputs, position
                )
                for position, (view_id, _tensor, trace) in enumerate(transformed)
            ]
            for view in views:
                writer.write(view)
            views_by_key[key] = [base_views[key], *views]
    plan = build_evaluation_plan(
        views_by_key, seed=seed, image_limit=image_limit, query_limit=query_limit
    )
    writer.set_evaluation_plan(plan)
    return plan


def _attach_reader_sample_ids(rows: list[dict[str, Any]], reader: SpatialReader) -> None:
    order: dict[str, int] = {}
    for entry in reader._index:  # noqa: SLF001 - the reader owns its index order
        order.setdefault(entry["sample"]["image_key"], int(entry["sample_id"]))
    for row in rows:
        row["sample_id"] = order.get(row["image_key"])


__all__ = [
    "PreparedSpatialRequest",
    "SpatialRequestError",
    "extract_spatial",
    "prepare_spatial_request",
    "publish_spatial_run",
    "reevaluate_spatial",
]
