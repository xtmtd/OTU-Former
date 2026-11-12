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
    FILENAME_INDEX,
    FILENAME_LOG,
    FILENAME_MANIFEST,
    FILENAME_MATCHES,
    FILENAME_MODEL_LAYERS,
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
    from ..utils.checkpoint import (
        load_checkpoint,
        load_checkpoint_encoder,
        resolve_checkpoint,
    )

    checkpoint_path = _as_path(values.get("checkpoint"))
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise SpatialRequestError(
            f"checkpoint does not exist: {values.get('checkpoint')!r}"
        )
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
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
    return encoder, payload, architecture, resolved, image_size, checkpoint_path


def _annotation_snapshot(bundle) -> dict[str, Any] | None:
    if not bundle.sources:
        return None
    from ..data.coco import ROLE_INSTANCES, ROLE_KEYPOINTS
    from ..vision.geometry import transform_annotations

    records: list[dict[str, Any]] = []
    categories: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for role in (ROLE_INSTANCES, ROLE_KEYPOINTS):
        index = bundle.sources.get(role)
        if index is None:
            continue
        sources[role] = {
            "path": str(index.file.path),
            "sha256": index.file.sha256,
            "images": len(index.images_by_id),
        }
        for category_id, category in index.categories_by_id.items():
            categories[str(category_id)] = dict(category)
        for record in bundle.records:
            for reference in record.coco_images:
                if reference.source_id != role:
                    continue
                for annotation in index.annotations_by_image.get(reference.image_id, ()):
                    entry = {
                        "annotation_id": annotation.get("id"),
                        "image_key": record.image_key,
                        "category_id": annotation.get("category_id"),
                        "role": role,
                        "annotation": dict(annotation),
                    }
                    records.append(entry)
    if not records and not sources:
        return None
    return {"records": records, "categories": categories, "sources": sources}


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
        return _prepare_read(resolved, explicit_options, out_dir, notes)
    if mode == MODE_ONNX:
        return _prepare_onnx(resolved, explicit_options, out_dir, notes)
    return _prepare_checkpoint(resolved, explicit_options, out_dir, notes)


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
        if int(values.get("seed") or 0) != int(plan.get("seed", 0)):
            reader.close()
            raise SpatialRequestError(
                f"--seed {values.get('seed')} does not match the saved evaluation "
                f"plan seed {plan.get('seed')}; a stored plan fixes its seed"
            )
    if values.get("spatial_eval") and plan is None:
        reader.close()
        raise SpatialRequestError(
            "missing_paired_views: this product has no stored paired views or "
            "evaluation plan, so --spatial-eval cannot be satisfied from it"
        )
    images = _as_path(values.get("input_images_dir"))
    if images is not None:
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
    encoder, payload, architecture, resolved, image_size, checkpoint_path = (
        _load_checkpoint_encoder(values)
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
    layers = resolve_feature_layers(encoder.backbone, values.get("feature_layers"))
    contract = resolve_feature_contract(
        encoder,
        image_size,
        layers,
        norm=str(values.get("feature_norm") or "backbone"),
        l2=bool(values.get("feature_l2_normalize")),
        model_identity={
            "architecture": architecture.model_name,
            "checkpoint_sha256": sha256_bytes(checkpoint_path.read_bytes()),
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
    protected = [record.image_file for record in bundle.records]
    if bundle.selector is not None:
        protected.append(bundle.selector)
    for index in bundle.sources.values():
        protected.append(index.file)
    from ..data.provenance import snapshot_bytes

    protected.append(snapshot_bytes(checkpoint_path)[1])
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
    if declared.get("mean") and [
        float(v) for v in declared["mean"]
    ] != [float(v) for v in IMAGENET_MEAN]:
        raise SpatialRequestError(
            "the graph declares an input normalization this build does not apply; "
            f"graph={declared.get('mean')}, supported={list(IMAGENET_MEAN)}"
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
    protected: list[FileSnapshot] = [record.image_file for record in bundle.records]
    if bundle.selector is not None:
        protected.append(bundle.selector)
    for index in bundle.sources.values():
        protected.append(index.file)
    protected.append(snapshot_bytes(graph_path)[1])
    for sidecar in (
        graph_path.with_name(graph_path.name + ".data"),
        graph_path.with_suffix(".onnx.data"),
    ):
        if sidecar.is_file():
            protected.append(snapshot_bytes(sidecar)[1])
    notes.append(
        "the graph's declared contract fixes layers, normalization, L2 and input "
        "size; numeric agreement was validated at export time"
    )
    return PreparedSpatialRequest(
        mode=MODE_ONNX,
        values=values,
        explicit_options=explicit_options,
        out_dir=out_dir,
        bundle=bundle,
        contract=contract,
        forward=forward,
        samples=tuple(samples_from_records(bundle.records)),
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
    lines = [
        "# Named modules for --feature-layers",
        "# Format: <name>\\t<class>\\t<spatial-eligible>",
    ]
    for row in describe_feature_layers(encoder.backbone):
        lines.append(
            f"{row['path']}\\t{row['module_class'].rsplit('.', 1)[-1]}"
            f"\\t{'yes' if row['spatial_eligible'] else ''}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_checkpoint_extraction(
    prepared: PreparedSpatialRequest,
) -> tuple[SpatialWriter, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Write payload/index/manifest, then return the sealed writer and reports."""
    import torch

    assert prepared.bundle is not None
    assert prepared.contract is not None
    out_dir = prepared.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SpatialWriter(
        out_dir,
        contract=prepared.contract,
        layout=prepared.layout,
        samples=list(prepared.samples),
        annotations=prepared.annotations,
    )
    records = list(prepared.bundle.records)
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
                        _flush_batch(writer, prepared, batch, device)
                    )
                    batch = []
            if batch:
                base_views.update(_flush_batch(writer, prepared, batch, device))
        if prepared.values.get("spatial_eval"):
            store_evaluation_views(
                prepared,
                writer,
                base_views=base_views,
                image_limit=int(
                    prepared.values.get("spatial_eval_sample_size")
                    or SPATIAL_EVAL_SAMPLE_SIZE
                ),
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


def _flush_batch(
    writer: SpatialWriter,
    prepared: PreparedSpatialRequest,
    batch: Sequence[tuple[Any, Any, Any, str]],
    device: Any,
) -> dict[str, SpatialView]:
    contract = prepared.contract
    outputs = _run_forward(prepared, list(batch), device)
    written: dict[str, SpatialView] = {}
    for position, (record, tensor, trace, view_id) in enumerate(batch):
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
        view = SpatialView(
            image_key=record.image_key,
            view_id=view_id,
            role="base",
            trace=trace,
            features=features,
            pixel_geometric_coverage=coverage,
            pixel_sampling_valid=pixel_valid,
            geometric_coverage=patch_coverage,
            patch_sampling_valid=patch_valid,
        )
        writer.write(view)
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
    """Final artifacts with their on-disk hashes, excluding the envelope itself."""
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(out_dir).as_posix()
        if relative == FILENAME_PROVENANCE:
            continue
        if relative == FILENAME_LOG and not include_log:
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
    final = dict(record)
    if final.get("status") == "completed":
        final["artifacts"] = _artifact_rows(Path(out_dir), include_log=include_log)
    envelope = Path(out_dir) / FILENAME_PROVENANCE
    temporary = envelope.with_suffix(".json.part")
    temporary.write_bytes(canonical_json_bytes(final))
    temporary.replace(envelope)
    return final


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
    writer, manifest, rows, summary = _run_checkpoint_extraction(prepared)
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


def reevaluate_spatial(
    prepared: PreparedSpatialRequest, *, defer_publication: bool = False
) -> dict[str, Any]:
    """Read a stored product and write report-only diagnostics beside it."""
    if prepared.mode != MODE_READ or prepared.reader is None:
        raise SpatialRequestError("reevaluate_spatial needs a spatial read request")
    reader = prepared.reader
    out_dir = prepared.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    rows, summary = diagnose_views(
        reader.iter_views(), reader.contract, plan=reader.evaluation_plan
    )
    _attach_reader_sample_ids(rows, reader)
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
    published = publish_spatial_run(out_dir, record, include_log=False)
    return {
        "status": "completed",
        "rows": rows,
        "summary": summary,
        "evaluation": evaluation_summary,
        "envelope": published,
        "out_dir": str(out_dir),
        "deferred": False,
    }


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
    seed = int(prepared.values.get("seed") or 42)
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
