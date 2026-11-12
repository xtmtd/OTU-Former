"""Sealed spatial products: ordered index plus per-image NPZ or NPY shards.

The writer owns one attempt directory. It validates every view against the
frozen feature contract, seals payloads and the ordered index, then seals the
manifest last. Reports and the run envelope are written afterwards by the caller
(the CLI/`spatial.extractor`), which is why completion lives outside the
manifest: neither the manifest nor the run envelope hashes itself.

The reader is read-only. It validates the manifest/index/payload chain for the
records it is asked for, reports how much it verified, and never mutates,
repairs or infers anything.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..constants import SPATIAL_KEY_RULE, SPATIAL_SCHEMA_VERSION, SPATIAL_SUPPORT_VERSION
from ..data.provenance import (
    canonical_digest,
    canonical_json_bytes,
    read_verified_bytes,
    sha256_bytes,
)
from ..data.records import FileSnapshot
from ..vision.features import FeatureContract
from ..vision.geometry import GeometryTrace
from .records import (
    DIRECTORY_FEATURES,
    FILENAME_ANNOTATIONS,
    FILENAME_EVALUATION_PLAN,
    FILENAME_INDEX,
    FILENAME_MANIFEST,
    FEATURE_ARRAY_PREFIX,
    PATCH_COVERAGE,
    PATCH_VALID,
    PIXEL_COVERAGE,
    PIXEL_VALID,
    SpatialRecordError,
    SpatialSample,
    SpatialView,
    layout_is_valid,
    require_schema_version,
    require_support_version,
    validate_view,
    view_arrays,
)

#: Uncompressed array payload a shard may reach before it is sealed.
SHARD_BYTES = 256 * 1024 * 1024

APPEARANCE_PROTOCOL = {"photometric": "none"}


def _canonical_array(array: np.ndarray) -> np.ndarray:
    """Little-endian C-order copy, so a digest never depends on host order."""
    values = np.asarray(array)
    if values.dtype == object:
        raise SpatialRecordError("object arrays are forbidden in version 1")
    little = values.dtype.newbyteorder("<")
    return np.ascontiguousarray(values.astype(little, copy=False))


def array_digest(role: str, array: np.ndarray) -> str:
    """Per-record semantic digest: canonical descriptor plus numeric bytes."""
    canonical = _canonical_array(array)
    descriptor = {
        "role": role,
        "dtype": canonical.dtype.str,
        "shape": [int(v) for v in canonical.shape],
    }
    return sha256_bytes(canonical_json_bytes(descriptor) + b"\x00" + canonical.tobytes(order="C"))


def array_descriptor(role: str, array: np.ndarray) -> dict[str, Any]:
    canonical = _canonical_array(array)
    return {
        "role": role,
        "dtype": canonical.dtype.str,
        "shape": [int(v) for v in canonical.shape],
        "digest": array_digest(role, array),
    }


class SpatialWriter:
    """Write one spatial product into an attempt-owned directory."""

    def __init__(
        self,
        out_dir: Path,
        *,
        contract: FeatureContract,
        layout: str,
        samples: Sequence[SpatialSample],
        annotations: Mapping[str, Any] | None = None,
        evaluation_plan: Mapping[str, Any] | None = None,
        shard_bytes: int | None = None,
    ) -> None:
        if not layout_is_valid(layout):
            raise SpatialRecordError(f"unsupported spatial layout {layout!r}")
        if not samples:
            raise SpatialRecordError("a spatial product needs at least one sample")
        keys = [sample.image_key for sample in samples]
        if len(set(keys)) != len(keys):
            raise SpatialRecordError("sample keys must be unique")
        self.out_dir = Path(out_dir)
        self.contract = contract
        self.layout = layout
        self.samples = list(samples)
        self._sample_index = {sample.image_key: index for index, sample in enumerate(samples)}
        self._annotations = dict(annotations or {})
        self._evaluation_plan = dict(evaluation_plan) if evaluation_plan else None
        self._entries: list[dict[str, Any]] = []
        self._files: list[dict[str, Any]] = []
        self._created: list[Path] = []
        self._written_views: set[tuple[str, str]] = set()
        self._sample_payloads: dict[int, Path] = {}
        self._pending_sample: int | None = None
        self._pending_arrays: dict[str, np.ndarray] = {}
        # Bounded shard budget. ``shard_bytes`` is a private test seam, never a
        # CLI option: version 1 has one budget.
        self._shard_bytes = int(shard_bytes or SHARD_BYTES)
        self._staging: dict[tuple[Any, ...], list[np.ndarray]] = {}
        self._staging_bytes: dict[tuple[Any, ...], int] = {}
        self._shard_numbers: dict[tuple[Any, ...], int] = {}
        self._shard_paths: dict[tuple[Any, ...], str] = {}
        self._shards: list[dict[str, Any]] = []
        self._closed = False
        self._manifest: dict[str, Any] | None = None
        (self.out_dir / DIRECTORY_FEATURES).mkdir(parents=True, exist_ok=True)

    # -- payload writing -------------------------------------------------
    def _write_npz(self, sample_id: int, arrays: Mapping[str, np.ndarray]) -> Path:
        path = self.out_dir / DIRECTORY_FEATURES / f"{sample_id:06d}.npz"
        buffer = io.BytesIO()
        np.savez(buffer, **{name: _canonical_array(array) for name, array in arrays.items()})
        payload = buffer.getvalue()
        path.write_bytes(payload)
        self._created.append(path)
        self._sample_payloads[sample_id] = path
        # A sample's archive can be extended after its first flush (protocol views
        # are written later), so the manifest keeps exactly one record per payload
        # file and always the hash of its final bytes.
        relative = path.relative_to(self.out_dir).as_posix()
        self._files = [
            record for record in self._files if record.get("path") != relative
        ]
        self._files.append(
            {
                "role": "payload",
                "path": path.relative_to(self.out_dir).as_posix(),
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
                "uncompressed_bytes": int(
                    sum(_canonical_array(array).nbytes for array in arrays.values())
                ),
                "records": 1,
                "rows": len(arrays),
                "oversized_record": False,
            }
        )
        return path

    def write(self, view: SpatialView) -> None:
        if self._closed:
            raise SpatialRecordError("this writer is already sealed")
        if view.image_key not in self._sample_index:
            raise SpatialRecordError(
                f"view for {view.image_key!r} does not belong to a selected sample"
            )
        identity = (view.image_key, view.view_id)
        if identity in self._written_views:
            raise SpatialRecordError(
                f"view {view.view_id!r} of {view.image_key!r} was written twice"
            )
        validate_view(view, self.contract)
        sample_id = self._sample_index[view.image_key] + 1
        slot = sum(1 for entry in self._entries if entry["sample"]["image_key"] == view.image_key)
        arrays: dict[str, np.ndarray] = {}
        logical: dict[str, dict[str, Any]] = {}
        refs: dict[str, dict[str, Any]] = {}
        for logical_name, role, array in view_arrays(view):
            logical[logical_name] = array_descriptor(role, array)
            if self.layout == "sharded":
                refs[logical_name] = self._stage_row(logical_name, role, array)
                continue
            key = f"v{slot:04d}.{logical_name}"
            arrays[key] = np.asarray(array)
            refs[logical_name] = {
                "kind": "npz",
                "path": f"{DIRECTORY_FEATURES}/{sample_id:06d}.npz",
                "key": key,
            }
        entry = {
            "schema_version": SPATIAL_SCHEMA_VERSION,
            "sample_id": sample_id,
            "sample": self.samples[sample_id - 1].to_json(),
            "view_id": view.view_id,
            "role": view.role,
            "appearance": dict(APPEARANCE_PROTOCOL),
            "support_version": SPATIAL_SUPPORT_VERSION,
            "coordinate_adapter": None,
            "geometry": view.trace.to_json(),
            "feature_layers": list(self.contract.layer_paths),
            "arrays": logical,
            "refs": refs,
        }
        self._entries.append(entry)
        self._written_views.add(identity)
        if self.layout == "sharded":
            return
        # One NPZ per sample. Views arrive grouped by sample, so the archive is
        # written once when the sample's views are complete (or at finish()), and
        # a reader never observes a half-updated sample.
        if self._pending_sample is not None and self._pending_sample != sample_id:
            self._flush_pending()
        self._pending_sample = sample_id
        self._pending_arrays.update(arrays)

    def _flush_pending(self) -> None:
        """Write this sample's archive, merged with anything already sealed for it.

        A sample's views can arrive in more than one group (base views first, the
        protocol views later), so the archive is extended rather than replaced: its
        earlier arrays must not disappear from the file an index entry points at.
        """
        if self._pending_sample is None or not self._pending_arrays:
            return
        existing: dict[str, np.ndarray] = {}
        path = self._sample_payloads.get(self._pending_sample)
        if path is not None and path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                existing = {name: archive[name] for name in archive.files}
        existing.update(self._pending_arrays)
        self._write_npz(self._pending_sample, existing)
        self._pending_sample = None
        self._pending_arrays = {}

    # -- bounded shards --------------------------------------------------
    def _bucket_key(
        self, logical_name: str, role: str, array: np.ndarray
    ) -> tuple[Any, ...]:
        """One bucket per compatible (kind, layer/support role, grid, dtype)."""
        canonical = _canonical_array(array)
        if logical_name.startswith(FEATURE_ARRAY_PREFIX):
            return ("feature", logical_name, tuple(canonical.shape), canonical.dtype.str)
        return ("support", role, tuple(canonical.shape), canonical.dtype.str)

    def _shard_path(self, bucket: tuple[Any, ...]) -> str:
        number = self._shard_numbers.get(bucket, 0)
        if bucket[0] == "feature":
            label = bucket[1].replace(FEATURE_ARRAY_PREFIX, "feature-")
        else:
            label = f"support-{bucket[1]}"
        label = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in label
        )
        return f"{DIRECTORY_FEATURES}/{label}-{number:06d}.npy"

    def _stage_row(
        self, logical_name: str, role: str, array: np.ndarray
    ) -> dict[str, Any]:
        canonical = _canonical_array(array)
        bucket = self._bucket_key(logical_name, role, array)
        if bucket not in self._staging:
            self._staging[bucket] = []
            self._staging_bytes[bucket] = 0
            self._shard_numbers[bucket] = 0
            self._shard_paths[bucket] = self._shard_path(bucket)
        if (
            self._staging[bucket]
            and self._staging_bytes[bucket] + canonical.nbytes > self._shard_bytes
        ):
            self._seal_bucket(bucket)
        row = len(self._staging[bucket])
        self._staging[bucket].append(canonical)
        self._staging_bytes[bucket] += int(canonical.nbytes)
        return {
            "kind": "npy",
            "path": self._shard_paths[bucket],
            "row": row,
            "bucket": "|".join(str(part) for part in bucket[:-1]),
        }

    def _seal_bucket(self, bucket: tuple[Any, ...]) -> None:
        rows = self._staging.get(bucket) or []
        if not rows:
            return
        stacked = np.stack(rows, axis=0)
        relative = self._shard_paths[bucket]
        path = self.out_dir / relative
        payload = io.BytesIO()
        np.save(payload, stacked, allow_pickle=False)
        body = payload.getvalue()
        path.write_bytes(body)
        self._created.append(path)
        oversized = len(rows) == 1 and stacked.nbytes > self._shard_bytes
        self._shards.append(
            {
                "path": relative,
                "bucket": "|".join(str(part) for part in bucket[:-1]),
                "rows": len(rows),
                "uncompressed_bytes": int(stacked.nbytes),
                "physical_bytes": len(body),
                "oversized_record": bool(oversized),
                "dtype": stacked.dtype.str,
                "record_shape": [int(v) for v in stacked.shape[1:]],
            }
        )
        self._files.append(
            {
                "role": "payload",
                "path": relative,
                "sha256": sha256_bytes(body),
                "bytes": len(body),
                "uncompressed_bytes": int(stacked.nbytes),
                "records": len(rows),
                "rows": len(rows),
                "bucket": "|".join(str(part) for part in bucket[:-1]),
                "oversized_record": bool(oversized),
            }
        )
        self._staging[bucket] = []
        self._staging_bytes[bucket] = 0
        self._shard_numbers[bucket] = self._shard_numbers.get(bucket, 0) + 1
        next_path = self._shard_path(bucket)
        while (self.out_dir / next_path).exists():
            self._shard_numbers[bucket] = self._shard_numbers.get(bucket, 0) + 1
            next_path = self._shard_path(bucket)
        self._shard_paths[bucket] = next_path

    def _seal_all_buckets(self) -> None:
        for bucket in list(self._staging):
            self._seal_bucket(bucket)

    def shard_report(self) -> list[dict[str, Any]]:
        """Sealed shard sizes for the run log (physical and payload bytes)."""
        return [dict(entry) for entry in self._shards]

    # -- plan and sealing ------------------------------------------------
    def set_evaluation_plan(self, plan: Mapping[str, Any]) -> None:
        if self._closed:
            raise SpatialRecordError("this writer is already sealed")
        if self._evaluation_plan is not None:
            raise SpatialRecordError("the evaluation plan was already supplied")
        self._evaluation_plan = dict(plan)

    def finish(self) -> dict[str, Any]:
        if self._closed:
            raise SpatialRecordError("this writer is already sealed")
        self._flush_pending()
        self._seal_all_buckets()
        missing = [
            sample.image_key
            for sample in self.samples
            if not any(
                entry["sample"]["image_key"] == sample.image_key
                for entry in self._entries
            )
        ]
        if missing:
            raise SpatialRecordError(
                "every sample needs at least its base view; missing "
                f"{', '.join(missing[:3])}"
            )
        index_bytes = b"".join(
            canonical_json_bytes(entry) + b"\n" for entry in self._entries
        )
        index_path = self.out_dir / FILENAME_INDEX
        index_path.write_bytes(index_bytes)
        self._created.append(index_path)
        self._files.append(
            {
                "role": "index",
                "path": FILENAME_INDEX,
                "sha256": sha256_bytes(index_bytes),
                "bytes": len(index_bytes),
                "records": len(self._entries),
            }
        )
        annotations = {"status": "absent"}
        if self._annotations:
            payload = b"".join(
                canonical_json_bytes(record) + b"\n"
                for record in self._annotations.get("records", [])
            )
            annotations_path = self.out_dir / FILENAME_ANNOTATIONS
            annotations_path.write_bytes(payload)
            self._created.append(annotations_path)
            self._files.append(
                {
                    "role": "annotations",
                    "path": FILENAME_ANNOTATIONS,
                    "sha256": sha256_bytes(payload),
                    "bytes": len(payload),
                    "records": len(self._annotations.get("records", [])),
                }
            )
            annotations = {
                "status": "present",
                "path": FILENAME_ANNOTATIONS,
                "sha256": sha256_bytes(payload),
                "records": len(self._annotations.get("records", [])),
                "categories": dict(self._annotations.get("categories", {})),
                "sources": dict(self._annotations.get("sources", {})),
            }
        plan = {"status": "absent"}
        if self._evaluation_plan is not None:
            plan_bytes = canonical_json_bytes(self._evaluation_plan)
            plan_path = self.out_dir / FILENAME_EVALUATION_PLAN
            plan_path.write_bytes(plan_bytes)
            self._created.append(plan_path)
            self._files.append(
                {
                    "role": "evaluation-plan",
                    "path": FILENAME_EVALUATION_PLAN,
                    "sha256": sha256_bytes(plan_bytes),
                    "bytes": len(plan_bytes),
                    "records": 1,
                }
            )
            plan = {
                "status": "present",
                "path": FILENAME_EVALUATION_PLAN,
                "sha256": sha256_bytes(plan_bytes),
            }
        manifest = {
            "schema_version": SPATIAL_SCHEMA_VERSION,
            "layout": self.layout,
            "feature_contract": self.contract.to_json(),
            "product_id": self.logical_id(),
            "samples": len(self.samples),
            "views": len(self._entries),
            "index": {
                "path": FILENAME_INDEX,
                "sha256": sha256_bytes(index_bytes),
                "bytes": len(index_bytes),
                "entries": len(self._entries),
            },
            "files": list(self._files),
            "annotations": annotations,
            "evaluation_plan": plan,
            "support_version": SPATIAL_SUPPORT_VERSION,
            "appearance": dict(APPEARANCE_PROTOCOL),
            "key_rule": SPATIAL_KEY_RULE,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_path = self.out_dir / FILENAME_MANIFEST
        manifest_path.write_bytes(manifest_bytes)
        self._created.append(manifest_path)
        self._manifest = manifest
        self._closed = True
        return manifest

    def logical_id(self) -> str:
        """Layout-independent product identity over ordered logical records."""
        payload = {
            "schema_version": SPATIAL_SCHEMA_VERSION,
            "feature_contract": self.contract.to_json(),
            "support_version": SPATIAL_SUPPORT_VERSION,
            "appearance": dict(APPEARANCE_PROTOCOL),
            "entries": [
                {
                    "sample_id": entry["sample_id"],
                    "sample": entry["sample"],
                    "view_id": entry["view_id"],
                    "role": entry["role"],
                    "geometry": entry["geometry"],
                    "feature_layers": entry["feature_layers"],
                    "arrays": entry["arrays"],
                }
                for entry in self._entries
            ],
        }
        return canonical_digest(payload)

    def reader(self) -> "SpatialReader":
        """Open this writer's sealed staging files (never a public escape hatch)."""
        if self._manifest is None:
            raise SpatialRecordError(
                "finish() must seal the payload, index and manifest before "
                "diagnostics can read them"
            )
        return SpatialReader(self.out_dir, require_completion=False)

    def abort(self, reason: Mapping[str, Any] | None = None) -> list[str]:
        """Delete only files this attempt created; keep failure provenance."""
        removed: list[str] = []
        for path in sorted(self._created, reverse=True):
            try:
                if path.is_file():
                    path.unlink()
                    removed.append(str(path.name))
            except OSError:  # pragma: no cover - best effort cleanup
                continue
        features = self.out_dir / DIRECTORY_FEATURES
        try:
            if features.is_dir() and not any(features.iterdir()):
                features.rmdir()
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        self._closed = True
        return removed


@dataclass(frozen=True)
class VerificationScope:
    """What one read actually verified, so a report never overclaims."""

    selected_keys: tuple[str, ...]
    index_entries: int
    payload_files_verified: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "selected_keys": list(self.selected_keys),
            "index_entries": int(self.index_entries),
            "payload_files_verified": list(self.payload_files_verified),
        }


class SpatialReader:
    """Read-only access to a sealed spatial product."""

    def __init__(self, path: Path, *, require_completion: bool = True) -> None:
        self.path = Path(path)
        self._require_completion = require_completion
        self.manifest = self._load_manifest()
        self._index = self._load_index()
        self._selected: tuple[str, ...] = tuple(
            dict.fromkeys(entry["sample"]["image_key"] for entry in self._index)
        )
        self._verified_payloads: list[str] = []
        self._open_npz: dict[str, Any] = {}
        self._open_shards: dict[str, Any] = {}
        self.annotations = self._load_annotations()
        self.evaluation_plan = self._load_evaluation_plan()

    # -- validation ------------------------------------------------------
    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.path / FILENAME_MANIFEST
        if not manifest_path.is_file():
            raise SpatialRecordError(
                f"not a spatial product: {FILENAME_MANIFEST} is missing"
            )
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        require_schema_version(manifest.get("schema_version"))
        layout = manifest.get("layout")
        if not layout_is_valid(layout):
            raise SpatialRecordError(f"unsupported spatial layout {layout!r}")
        require_support_version(manifest.get("support_version"))
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise SpatialRecordError("the manifest does not inventory any files")
        if self._require_completion:
            self._require_completed_envelope(manifest_path, raw)
        listed = {item["path"] for item in files}
        if manifest["index"]["path"] not in listed:
            raise SpatialRecordError("the manifest does not seal its own index")
        for item in files:
            self._check_relative(item["path"])
        self.contract = FeatureContract.from_json(manifest["feature_contract"])
        self._manifest_bytes = raw
        return manifest

    def _require_completed_envelope(self, manifest_path: Path, manifest_bytes: bytes) -> None:
        from .records import FILENAME_PROVENANCE

        envelope = self.path / FILENAME_PROVENANCE
        if not envelope.is_file():
            raise SpatialRecordError(
                "this directory has no completed run envelope; a spatial product "
                "is only readable after run_provenance.json reports completion"
            )
        record = json.loads(envelope.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            raise SpatialRecordError(
                f"run provenance status is {record.get('status')!r}, not 'completed'"
            )
        artifacts = record.get("artifacts") or []
        expected = sha256_bytes(manifest_bytes)
        for artifact in artifacts:
            if artifact.get("path") == FILENAME_MANIFEST:
                if artifact.get("sha256") != expected:
                    raise SpatialRecordError(
                        "run provenance does not match the sealed manifest bytes"
                    )
                return
        raise SpatialRecordError(
            "run provenance does not list the sealed manifest artifact"
        )

    def _check_relative(self, relative: str) -> None:
        if not isinstance(relative, str) or not relative:
            raise SpatialRecordError("file paths must be non-empty strings")
        if relative.startswith("/") or os.path.isabs(relative):
            raise SpatialRecordError(f"absolute product path is forbidden: {relative}")
        candidate = (self.path / relative).resolve()
        root = self.path.resolve()
        if root != candidate and root not in candidate.parents:
            raise SpatialRecordError(f"product path escapes the product root: {relative}")

    def _load_index(self) -> list[dict[str, Any]]:
        index_path = self.path / FILENAME_INDEX
        payload = index_path.read_bytes()
        expected = self.manifest["index"]["sha256"]
        if sha256_bytes(payload) != expected:
            raise SpatialRecordError("spatial index bytes do not match the manifest")
        if len(payload) != int(self.manifest["index"]["bytes"]):
            raise SpatialRecordError("spatial index byte count does not match")
        entries: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for line in payload.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line.decode("utf-8"))
            require_schema_version(entry.get("schema_version"))
            require_support_version(entry.get("support_version"))
            identity = (entry["sample"]["image_key"], entry["view_id"])
            if identity in seen:
                raise SpatialRecordError(
                    f"duplicate logical view {entry['view_id']!r} for "
                    f"{entry['sample']['image_key']!r}"
                )
            seen.add(identity)
            if int(entry["sample_id"]) < 1:
                raise SpatialRecordError("sample ids start at 1")
            if entry.get("feature_layers") != list(self.contract.layer_paths):
                raise SpatialRecordError(
                    f"index entry {identity} lists layers that differ from the "
                    "feature contract"
                )
            for name, descriptor in entry["arrays"].items():
                if not name.startswith(FEATURE_ARRAY_PREFIX) and name not in (
                    PIXEL_COVERAGE,
                    PIXEL_VALID,
                    PATCH_COVERAGE,
                    PATCH_VALID,
                ):
                    raise SpatialRecordError(f"unexpected array {name!r}")
                self._check_relative(entry["refs"][name]["path"])
            entries.append(entry)
        if len(entries) != int(self.manifest["index"]["entries"]):
            raise SpatialRecordError("index entry count does not match the manifest")
        return entries

    def _load_annotations(self) -> dict[str, Any]:
        status = self.manifest["annotations"]["status"]
        if status == "absent":
            return {"status": "absent", "records": []}
        path = self.path / self.manifest["annotations"]["path"]
        payload = path.read_bytes()
        if sha256_bytes(payload) != self.manifest["annotations"]["sha256"]:
            raise SpatialRecordError("annotation snapshot bytes do not match")
        records = [
            json.loads(line.decode("utf-8"))
            for line in payload.splitlines()
            if line.strip()
        ]
        return {
            "status": "present",
            "records": records,
            "categories": dict(self.manifest["annotations"].get("categories", {})),
            "sources": dict(self.manifest["annotations"].get("sources", {})),
        }

    def _load_evaluation_plan(self) -> dict[str, Any] | None:
        from ..constants import SPATIAL_PROTOCOL

        status = self.manifest["evaluation_plan"]["status"]
        if status == "absent":
            return None
        path = self.path / self.manifest["evaluation_plan"]["path"]
        payload = path.read_bytes()
        if sha256_bytes(payload) != self.manifest["evaluation_plan"]["sha256"]:
            raise SpatialRecordError("evaluation plan bytes do not match")
        plan = json.loads(payload.decode("utf-8"))
        if plan.get("protocol") != SPATIAL_PROTOCOL:
            raise SpatialRecordError(
                f"saved plan protocol {plan.get('protocol')!r} is not {SPATIAL_PROTOCOL!r}"
            )
        require_schema_version(plan.get("schema_version"))
        require_support_version(plan.get("support_version"))
        stored = {(entry["sample"]["image_key"], entry["view_id"]) for entry in self._index}
        keys = {entry["sample"]["image_key"] for entry in self._index}
        missing_keys = [key for key in plan.get("selected_keys", []) if key not in keys]
        if missing_keys:
            raise SpatialRecordError(
                "the saved plan selects keys this product does not contain: "
                + ", ".join(missing_keys[:3])
            )
        for pair in plan.get("pairs", []):
            image_key = pair.get("image_key")
            target = pair.get("target_view")
            if (image_key, target) not in stored:
                raise SpatialRecordError(
                    f"a planned pair references a missing stored view: "
                    f"{image_key!r}/{target!r}"
                )
        return plan

    # -- access ----------------------------------------------------------
    @property
    def image_keys(self) -> tuple[str, ...]:
        return self._selected

    @property
    def product_id(self) -> str:
        return str(self.manifest["product_id"])

    def select(self, keys: Sequence[str] | None) -> tuple[str, ...]:
        if keys is None:
            return self._selected
        available = set(self._selected)
        missing = [key for key in keys if key not in available]
        if missing:
            raise SpatialRecordError(
                f"selected key(s) not in this product: {', '.join(missing[:3])}"
            )
        return tuple(keys)

    def iter_views(self, keys: Sequence[str] | None = None) -> Iterator[SpatialView]:
        """Yield stored views in *selected* order, each sample in index order."""
        selection = self.select(keys)
        by_key: dict[str, list[dict[str, Any]]] = {}
        for entry in self._index:
            by_key.setdefault(entry["sample"]["image_key"], []).append(entry)
        for key in selection:
            for entry in by_key[key]:
                yield self._build_view(entry)

    def get_view(self, image_key: str, view_id: str) -> SpatialView:
        for entry in self._index:
            if (
                entry["sample"]["image_key"] == image_key
                and entry["view_id"] == view_id
            ):
                return self._build_view(entry)
        raise SpatialRecordError(f"no stored view {view_id!r} for {image_key!r}")

    def view_ids(self, image_key: str) -> tuple[str, ...]:
        return tuple(
            entry["view_id"]
            for entry in self._index
            if entry["sample"]["image_key"] == image_key
        )

    def _build_view(self, entry: Mapping[str, Any]) -> SpatialView:
        arrays = self._load_arrays(entry)
        layer_paths = list(entry["feature_layers"])
        features = {path: arrays[f"{FEATURE_ARRAY_PREFIX}{path}"] for path in layer_paths}
        trace = GeometryTrace.from_json(entry["geometry"])
        view = SpatialView(
            image_key=entry["sample"]["image_key"],
            view_id=entry["view_id"],
            role=entry["role"],
            trace=trace,
            features=features,
            pixel_geometric_coverage=arrays[PIXEL_COVERAGE],
            pixel_sampling_valid=arrays[PIXEL_VALID],
            geometric_coverage=arrays[PATCH_COVERAGE],
            patch_sampling_valid=arrays[PATCH_VALID],
        )
        # The reader re-applies the writer's semantic checks: a product whose
        # coverage left [0, 1] or whose validity is not Boolean is not readable
        # just because its bytes are internally consistent.
        validate_view(view, self.contract)
        return view

    def _load_arrays(self, entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
        loaded: dict[str, np.ndarray] = {}
        for name, descriptor in entry["arrays"].items():
            ref = entry["refs"][name]
            if ref["kind"] == "npz":
                archive = self._open_archive(ref["path"])
                if ref["key"] not in archive.files:
                    raise SpatialRecordError(
                        f"payload {ref['path']} has no array {ref['key']!r}"
                    )
                array = archive[ref["key"]]
                label = ref["key"]
            elif ref["kind"] == "npy":
                shard = self._open_shard(ref["path"])
                row = int(ref["row"])
                if row < 0 or row >= int(shard.shape[0]):
                    raise SpatialRecordError(
                        f"shard {ref['path']} has no row {row}"
                    )
                # One bounded copy: the memmap stays read-only and no caller can
                # write through a materialized view.
                array = np.array(shard[row], copy=True)
                label = f"{ref['path']}[{row}]"
            else:
                raise SpatialRecordError(
                    f"unsupported payload kind {ref['kind']!r} for this reader"
                )
            self._check_descriptor(array, descriptor, label)
            loaded[name] = array
        return loaded

    def _open_shard(self, relative: str):
        cached = self._open_shards.get(relative)
        if cached is not None:
            return cached
        self._verify_payload(relative)
        shard = np.load(self.path / relative, mmap_mode="r", allow_pickle=False)
        self._open_shards[relative] = shard
        return shard

    def _check_descriptor(
        self, array: np.ndarray, descriptor: Mapping[str, Any], name: str
    ) -> None:
        shape = tuple(int(v) for v in descriptor["shape"])
        if tuple(int(v) for v in array.shape) != shape:
            raise SpatialRecordError(
                f"array {name!r} has shape {array.shape}, expected {shape}"
            )
        if array.dtype.str != descriptor["dtype"]:
            raise SpatialRecordError(
                f"array {name!r} has dtype {array.dtype.str}, expected "
                f"{descriptor['dtype']}"
            )
        if array_digest(descriptor["role"], array) != descriptor["digest"]:
            raise SpatialRecordError(f"array {name!r} does not match its digest")

    def _open_archive(self, relative: str):
        cached = self._open_npz.get(relative)
        if cached is not None:
            return cached
        self._verify_payload(relative)
        archive = np.load(self.path / relative, allow_pickle=False)
        self._open_npz[relative] = archive
        return archive

    def _verify_payload(self, relative: str) -> None:
        item = next(
            (entry for entry in self.manifest["files"] if entry["path"] == relative),
            None,
        )
        if item is None:
            raise SpatialRecordError(
                f"payload {relative} is not sealed by the manifest"
            )
        path = self.path / relative
        if not path.is_file():
            raise SpatialRecordError(f"sealed payload is missing: {relative}")
        digest = sha256_bytes(path.read_bytes())
        if digest != item["sha256"]:
            raise SpatialRecordError(f"payload bytes do not match: {relative}")
        if relative not in self._verified_payloads:
            self._verified_payloads.append(relative)

    def verification_scope(self, keys: Sequence[str] | None = None) -> VerificationScope:
        return VerificationScope(
            selected_keys=tuple(self.select(keys)),
            index_entries=len(self._index),
            payload_files_verified=tuple(self._verified_payloads),
        )

    def close(self) -> None:
        for archive in self._open_npz.values():
            try:
                archive.close()
            except Exception:  # pragma: no cover - archive already closed
                pass
        self._open_npz.clear()
        for shard in self._open_shards.values():
            try:
                del shard
            except Exception:  # pragma: no cover - memmap already released
                pass
        self._open_shards.clear()

    def __enter__(self) -> "SpatialReader":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def open_spatial_product(
    path: Path, *, selector_csv: Path | None = None
) -> SpatialReader:
    """Open a completed spatial product read-only.

    Completion is required, so an interrupted attempt is never readable. An
    optional CSV restricts the stored keys with ``artifact-key-resolution-v1``;
    it never adds filesystem lookup or an inference step.
    """
    reader = SpatialReader(Path(path), require_completion=True)
    if selector_csv is None:
        return reader
    keys = read_selector_keys(Path(selector_csv), reader.image_keys)
    reader._selected = keys  # noqa: SLF001 - the reader owns its selection order
    return reader


def read_selector_keys(path: Path, available: Sequence[str]) -> tuple[str, ...]:
    """``artifact-key-resolution-v1`` selection against stored keys."""
    if not path.is_file():
        raise SpatialRecordError(f"selector CSV does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "image" not in reader.fieldnames:
        raise SpatialRecordError("selector CSV needs an 'image' column")
    available_set = set(available)
    basenames: dict[str, list[str]] = {}
    for key in available:
        basenames.setdefault(key.rsplit("/", 1)[-1], []).append(key)
    selected: list[str] = []
    for row in reader:
        reference = (row.get("image") or "").strip()
        if not reference:
            raise SpatialRecordError("selector CSV has a blank 'image' value")
        key = resolve_stored_key(reference, available_set, basenames)
        selected.append(key)
    if not selected:
        raise SpatialRecordError("selector CSV selected no stored keys")
    if len(set(selected)) != len(selected):
        raise SpatialRecordError("selector CSV selects the same stored key twice")
    return tuple(selected)


def resolve_stored_key(
    reference: str, available: set[str], basenames: Mapping[str, list[str]]
) -> str:
    if reference.startswith("/") or os.path.isabs(reference):
        raise SpatialRecordError(
            f"absolute references are rejected for stored keys: {reference}"
        )
    normalized = reference.replace("\\", "/")
    if normalized.startswith("../") or "/../" in normalized:
        raise SpatialRecordError(f"reference escapes the product: {reference}")
    if normalized in available:
        return normalized
    if "/" not in normalized and normalized in basenames:
        matches = basenames[normalized]
        if len(matches) == 1:
            return matches[0]
        raise SpatialRecordError(
            f"bare basename {reference!r} is ambiguous: {', '.join(sorted(matches))}"
        )
    raise SpatialRecordError(f"stored key not found: {reference}")


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Deterministic CSV writer used by diagnostics/evaluation reports."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def manifest_bytes(path: Path) -> bytes:
    return (Path(path) / FILENAME_MANIFEST).read_bytes()


def verify_file_snapshot(snapshot: FileSnapshot) -> bytes:
    return read_verified_bytes(snapshot)
