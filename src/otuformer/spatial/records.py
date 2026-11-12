"""Small logical records for spatial products.

Stdlib + NumPy only: importing this module must never pull torch, ONNX Runtime,
the CLI or a trainer.

A spatial product is a versioned manifest, an ordered sample/view index and the
feature/support payloads those entries reference. Physical layout (per-image NPZ
or bounded NPY shards) never changes a logical record, so two layouts holding the
same arrays have the same logical identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..constants import (
    SPATIAL_LAYOUT_CHOICES,
    SPATIAL_SCHEMA_VERSION,
    SPATIAL_SUPPORT_VERSION,
    SPATIAL_VIEW_ROLES,
)
from ..vision.features import FeatureContract
from ..vision.geometry import GeometryTrace

#: Roles of the four stored support arrays and the per-layer feature grids.
PIXEL_COVERAGE = "pixel_geometric_coverage"
PIXEL_VALID = "pixel_sampling_valid"
PATCH_COVERAGE = "geometric_coverage"
PATCH_VALID = "patch_sampling_valid"
SUPPORT_ARRAY_NAMES = (PIXEL_COVERAGE, PIXEL_VALID, PATCH_COVERAGE, PATCH_VALID)
FEATURE_ARRAY_PREFIX = "feature:"

#: Filenames owned by design §7.1; no task invents an alternative.
FILENAME_PROVENANCE = "run_provenance.json"
FILENAME_MANIFEST = "spatial_manifest.json"
FILENAME_INDEX = "spatial_index.jsonl"
FILENAME_DIAGNOSTICS = "spatial_diagnostics.json"
FILENAME_RESULTS = "spatial_results.csv"
FILENAME_ANNOTATIONS = "annotations.jsonl"
FILENAME_EVALUATION_PLAN = "spatial_evaluation_plan.json"
FILENAME_EVALUATION = "spatial_evaluation.json"
FILENAME_MATCHES = "spatial_matches.csv"
FILENAME_LOG = "logs/extract.log"
FILENAME_MODEL_LAYERS = "model_layers.txt"
DIRECTORY_FEATURES = "features"
DIRECTORY_ANNOTATIONS = "annotations"


class SpatialRecordError(ValueError):
    """A record, product or layout violates the version-1 contract."""


@dataclass(frozen=True)
class SpatialSample:
    """One selected source image, as the product stores it.

    ``metadata`` carries only explicitly supplied CSV values; a directory-derived
    ``sample`` is never promoted into a product field.
    """

    image_key: str
    image_sha256: str
    width: int
    height: int
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_key:
            raise SpatialRecordError("a spatial sample needs a non-empty image key")
        if "/" in self.image_key and self.image_key.startswith("/"):
            raise SpatialRecordError("image keys are root-relative, never absolute")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise SpatialRecordError("a spatial sample needs positive dimensions")
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SpatialRecordError(
                    "sample metadata must be a mapping of strings to strings"
                )

    def to_json(self) -> dict[str, Any]:
        return {
            "image_key": self.image_key,
            "image_sha256": self.image_sha256,
            "width": int(self.width),
            "height": int(self.height),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SpatialView:
    """One stored view: geometry, support arrays and per-layer feature grids."""

    image_key: str
    view_id: str
    role: str
    trace: GeometryTrace
    features: dict[str, np.ndarray]
    pixel_geometric_coverage: np.ndarray
    pixel_sampling_valid: np.ndarray
    geometric_coverage: np.ndarray
    patch_sampling_valid: np.ndarray

    def __post_init__(self) -> None:
        if self.role not in SPATIAL_VIEW_ROLES:
            raise SpatialRecordError(
                f"unknown view role {self.role!r}; expected {SPATIAL_VIEW_ROLES}"
            )
        if not self.view_id:
            raise SpatialRecordError("a spatial view needs a non-empty view id")

    @property
    def support_version(self) -> str:
        return SPATIAL_SUPPORT_VERSION


def samples_from_records(records: Iterable[Any]) -> list[SpatialSample]:
    """Project reader records onto the product's sample contract.

    Only explicitly supplied CSV metadata survives; a ``first_directory`` sample
    inference stays a reader fact and never becomes product metadata.
    """
    samples: list[SpatialSample] = []
    seen: set[str] = set()
    for record in records:
        if record.image_key in seen:
            raise SpatialRecordError(
                f"duplicate image key in the selection: {record.image_key!r}"
            )
        seen.add(record.image_key)
        metadata = dict(record.metadata) if record.metadata else {}
        if getattr(record, "sample_origin", "none") != "explicit":
            metadata.pop("sample", None)
        samples.append(
            SpatialSample(
                image_key=record.image_key,
                image_sha256=record.image_file.sha256,
                width=int(record.width),
                height=int(record.height),
                metadata=metadata,
            )
        )
    return samples


def validate_view(view: SpatialView, contract: FeatureContract) -> None:
    """Reject a view whose arrays do not match the frozen feature contract.

    Shapes, dtypes, value bounds, support separation and finiteness are checked
    before anything is written: a product must never publish a grid a reader
    cannot interpret.
    """
    view_width, view_height = (int(v) for v in contract.input_hw)
    if tuple(int(v) for v in view.trace.view_wh) != (view_width, view_height):
        raise SpatialRecordError(
            f"view {view.view_id!r} has canvas {tuple(view.trace.view_wh)} but the "
            f"feature contract fixes {contract.input_hw}"
        )
    grid_hw = (
        view_width // int(contract.patch_hw[0]),
        view_height // int(contract.patch_hw[1]),
    )
    layer_paths = tuple(contract.layer_paths)
    if tuple(view.features) != layer_paths:
        raise SpatialRecordError(
            f"view {view.view_id!r} stores layers {tuple(view.features)!r} but the "
            f"contract requires {layer_paths!r} in that order"
        )
    for path in layer_paths:
        array = np.asarray(view.features[path])
        expected = (grid_hw[0], grid_hw[1], _channels(contract, path))
        if array.shape != expected:
            raise SpatialRecordError(
                f"layer {path!r} has shape {array.shape}, expected {expected}"
            )
        if array.dtype != np.float32:
            raise SpatialRecordError(
                f"layer {path!r} must be FP32, got {array.dtype}"
            )
        if not np.all(np.isfinite(array)):
            raise SpatialRecordError(f"layer {path!r} has non-finite values")

    coverage = np.asarray(view.pixel_geometric_coverage)
    valid = np.asarray(view.pixel_sampling_valid)
    if coverage.shape != (view_height, view_width) or coverage.dtype != np.float32:
        raise SpatialRecordError(
            f"pixel coverage must be FP32 {(view_height, view_width)}, got "
            f"{coverage.dtype} {coverage.shape}"
        )
    if valid.shape != (view_height, view_width) or valid.dtype != np.bool_:
        raise SpatialRecordError(
            f"pixel validity must be Boolean {(view_height, view_width)}, got "
            f"{valid.dtype} {valid.shape}"
        )
    if not np.all(np.isfinite(coverage)):
        raise SpatialRecordError("pixel coverage has non-finite values")
    if float(coverage.min()) < 0.0 or float(coverage.max()) > 1.0:
        raise SpatialRecordError("pixel coverage must stay within [0, 1]")

    patch_coverage = np.asarray(view.geometric_coverage)
    patch_valid = np.asarray(view.patch_sampling_valid)
    if patch_coverage.shape != grid_hw or patch_coverage.dtype != np.float32:
        raise SpatialRecordError(
            f"patch coverage must be FP32 {grid_hw}, got "
            f"{patch_coverage.dtype} {patch_coverage.shape}"
        )
    if patch_valid.shape != grid_hw or patch_valid.dtype != np.bool_:
        raise SpatialRecordError(
            f"patch validity must be Boolean {grid_hw}, got "
            f"{patch_valid.dtype} {patch_valid.shape}"
        )
    if not np.all(np.isfinite(patch_coverage)):
        raise SpatialRecordError("patch coverage has non-finite values")
    if float(patch_coverage.min()) < 0.0 or float(patch_coverage.max()) > 1.0:
        raise SpatialRecordError("patch coverage must stay within [0, 1]")


def _channels(contract: FeatureContract, path: str) -> int:
    for layer in contract.layers:
        if layer.path == path:
            return int(layer.channels)
    raise SpatialRecordError(f"layer {path!r} is not part of the feature contract")


def view_arrays(view: SpatialView) -> list[tuple[str, str, np.ndarray]]:
    """Ordered ``(logical name, role, array)`` triples stored once per view."""
    triples: list[tuple[str, str, np.ndarray]] = []
    for path, array in view.features.items():
        triples.append((f"{FEATURE_ARRAY_PREFIX}{path}", "feature", np.asarray(array)))
    triples.append((PIXEL_COVERAGE, PIXEL_COVERAGE, np.asarray(view.pixel_geometric_coverage)))
    triples.append((PIXEL_VALID, PIXEL_VALID, np.asarray(view.pixel_sampling_valid)))
    triples.append((PATCH_COVERAGE, PATCH_COVERAGE, np.asarray(view.geometric_coverage)))
    triples.append((PATCH_VALID, PATCH_VALID, np.asarray(view.patch_sampling_valid)))
    for name, role, array in triples:
        if array.dtype == object:
            raise SpatialRecordError(
                f"array {name!r} is an object array; version 1 forbids pickled "
                "payloads"
            )
    return triples


def layout_is_valid(layout: str) -> bool:
    return layout in SPATIAL_LAYOUT_CHOICES


def require_schema_version(value: Any) -> int:
    version = int(value)
    if version != SPATIAL_SCHEMA_VERSION:
        raise SpatialRecordError(
            f"unsupported spatial schema version {version}; this build reads "
            f"version {SPATIAL_SCHEMA_VERSION}"
        )
    return version


def require_support_version(value: Any) -> str:
    if value != SPATIAL_SUPPORT_VERSION:
        raise SpatialRecordError(
            f"unsupported support rule {value!r}; this build implements "
            f"{SPATIAL_SUPPORT_VERSION}"
        )
    return SPATIAL_SUPPORT_VERSION


def require_sequence_of_arrays(
    payload: Mapping[str, Any], names: Sequence[str]
) -> None:
    missing = [name for name in names if name not in payload]
    if missing:
        raise SpatialRecordError(
            f"payload is missing array(s): {', '.join(sorted(missing))}"
        )
