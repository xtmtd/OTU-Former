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
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..constants import (
    SPATIAL_COORDINATE_ADAPTER,
    SPATIAL_VIEW_ROLES,
    SPATIAL_KEY_RULE,
    SPATIAL_MASK_SAMPLING,
    SPATIAL_SCHEMA_VERSION,
    SPATIAL_SUPPORT_VERSION,
)
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
    DIRECTORY_ANNOTATIONS,
    DIRECTORY_FEATURES,
    FILENAME_ANNOTATIONS,
    FILENAME_EVALUATION_PLAN,
    FILENAME_INDEX,
    FILENAME_MANIFEST,
    FILENAME_PROVENANCE,
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

#: Files whose presence means the directory already belongs to a sealed product.
SEALED_MARKERS = (FILENAME_MANIFEST, FILENAME_INDEX, FILENAME_PROVENANCE)

#: Open payload handles a reader keeps before closing the least recently used.
OPEN_PAYLOAD_LIMIT = 4

APPEARANCE_PROTOCOL = {"photometric": "none"}


def _npy_header(zip_file: Any, name: str) -> tuple[tuple[int, ...], str]:
    """Shape and dtype from an NPZ member's NPY header, without the payload.

    ``np.load`` materializes an array from the header it finds, so a header that
    lies about its shape turns a 2.6 KB product into a multi-terabyte allocation
    request. Reading the header first lets the index descriptor be compared before
    any array memory is reserved.
    """
    member = f"{name}.npy" if not name.endswith(".npy") else name
    try:
        info = zip_file.getinfo(member)
    except KeyError:
        raise SpatialRecordError(f"payload has no NPY member {member!r}") from None
    with zip_file.open(member) as handle:
        prefix = handle.read(10)
        if len(prefix) != 10 or prefix[:6] != b"\x93NUMPY":
            raise SpatialRecordError(f"payload member {member!r} is not an NPY array")
        major = prefix[6]
        header_length = int.from_bytes(
            prefix[8:10], "little" if major == 1 else "big"
        )
        if header_length <= 0 or header_length > 1 << 20:
            raise SpatialRecordError(
                f"payload member {member!r} declares an unusable NPY header "
                f"length {header_length}"
            )
        raw = handle.read(header_length)
    if len(raw) != header_length:
        raise SpatialRecordError(f"payload member {member!r} has a truncated header")
    import ast

    try:
        header = ast.literal_eval(raw.decode("latin1"))
    except (SyntaxError, ValueError):
        raise SpatialRecordError(
            f"payload member {member!r} has an unreadable NPY header"
        ) from None
    if not isinstance(header, dict) or "shape" not in header or "descr" not in header:
        raise SpatialRecordError(
            f"payload member {member!r} has an unexpected NPY header"
        )
    shape = tuple(int(v) for v in header["shape"])
    descriptor = np.dtype(str(header["descr"]))
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * descriptor.itemsize
    if expected_bytes != info.file_size - 10 - header_length:
        raise SpatialRecordError(
            f"payload member {member!r} declares {expected_bytes} bytes but holds "
            f"{info.file_size - 10 - header_length}"
        )
    return shape, descriptor.newbyteorder("<").str


def _checked_npz_array(
    archive: Any,
    key: str,
    *,
    label: str,
    declared_shape: Any = None,
    declared_dtype: Any = None,
) -> np.ndarray:
    """Load one NPZ array after its header and byte length have been verified.

    ``np.load`` allocates from the header it finds, so a header that lies about the
    shape turns a few hundred bytes into a huge allocation request. The header is
    compared with whatever the manifest declares and with the member's real size
    first, and only then is the array materialized.
    """
    archive_file = getattr(archive, "zip", None)
    if archive_file is not None:
        header_shape, header_dtype = _npy_header(archive_file, key)
        if declared_shape is not None and list(header_shape) != [
            int(v) for v in declared_shape
        ]:
            raise SpatialRecordError(
                f"{label} declares shape {list(header_shape)} on disk, but the "
                f"manifest seals shape {[int(v) for v in declared_shape]}"
            )
        if declared_dtype is not None and header_dtype != str(declared_dtype):
            raise SpatialRecordError(
                f"{label} declares dtype {header_dtype} on disk, but the manifest "
                f"seals dtype {declared_dtype}"
            )
    return np.array(archive[key], copy=True)


def _close_archive(archive: Any) -> None:
    try:
        archive.close()
    except Exception:  # pragma: no cover - archive already closed
        pass


def _release_memmap(shard: Any) -> None:
    try:
        del shard
    except Exception:  # pragma: no cover - memmap already released
        pass


def _validated_categorical_raster(
    raster: Any, *, image_key: str, size: tuple[int, int] | None
) -> np.ndarray:
    """Check a categorical raster before it is sealed or exposed.

    A label raster is not an intensity image: ids must be integers, the grid must
    be the sample's own pixel grid, and a NaN or a float would silently become a
    "valid" category. The same function guards the write and read boundaries so a
    hand-built product cannot smuggle one past either.
    """
    array = np.asarray(raster)
    if array.ndim != 2:
        raise SpatialRecordError(
            f"a categorical raster for {image_key!r} must be 2-D, got shape {array.shape}"
        )
    if array.dtype == object or array.dtype.kind not in ("u", "i"):
        raise SpatialRecordError(
            f"a categorical raster for {image_key!r} must hold integer ids, got "
            f"dtype {array.dtype}"
        )
    if array.size == 0:
        raise SpatialRecordError(
            f"a categorical raster for {image_key!r} is empty"
        )
    if size is not None:
        width, height = int(size[0]), int(size[1])
        if array.shape != (height, width):
            raise SpatialRecordError(
                f"a categorical raster for {image_key!r} is {array.shape}, but the "
                f"sample is {(height, width)} (height, width)"
            )
    if int(array.min()) < 0:
        raise SpatialRecordError(
            f"a categorical raster for {image_key!r} holds a negative id "
            f"({int(array.min())})"
        )
    return array


@dataclass(frozen=True)
class SelectorOverlay:
    """A selector CSV's label/metadata overlay.

    The overlay is a *report* input: it records the file identity, the fields it
    carries and the fields where it collides with stored sample metadata, and it
    supplies values only to reported sample metadata. It never rewrites the stored
    record, selects queries or forms groups.
    """

    path: str
    sha256: str
    fields: tuple[str, ...]
    rows: dict[str, dict[str, str]]
    stored_fields: tuple[str, ...] = ()
    colliding_fields: tuple[str, ...] = ()

    def metadata_for(self, key: str, stored: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = dict(stored or {})
        for field, value in self.rows.get(key, {}).items():
            # Empty values keep the existing CSV missing-value semantics.
            if value != "":
                merged[field] = value
        return merged

    def report(
        self, keys: Sequence[str], stored: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        return {
            "status": "present",
            "path": self.path,
            "sha256": self.sha256,
            "fields": list(self.fields),
            "stored_metadata_fields": list(self.stored_fields),
            "colliding_fields": list(self.colliding_fields),
            "samples": {key: self.metadata_for(key, stored.get(key)) for key in keys},
            "precedence": "overlay values win in reported sample metadata only",
        }


def _owned_array(array: np.ndarray) -> np.ndarray:
    """A private, canonical copy of a caller's array.

    ``write()`` must take ownership of the values it seals: a caller that reuses
    the buffer afterwards would otherwise change the payload while the descriptor
    kept the digest of the old bytes.
    """
    return _canonical_array(np.array(array, copy=True))


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
        self._annotation_payloads: dict[str, dict[str, Any]] = dict(
            self._annotations.get("rasters", {}) or {}
        )
        self._annotation_refs: dict[str, dict[str, Any]] = {}
        self._claim_attempt_directory()

    # -- attempt ownership -----------------------------------------------
    def _claim_attempt_directory(self) -> None:
        """Own the target directory or refuse it before a byte is written.

        The CLI clears an existing output first, so a public Python caller was the
        only path that could reach here on a live product. A core writer that
        cannot prove the directory belongs to this attempt must not overwrite the
        payloads, index or manifest of an earlier one - and ``abort()`` must never
        be able to delete them.

        Ownership also covers the physical target: a ``features`` directory that is
        a symlink (including a dangling one) would send this attempt's payloads
        outside the product root, so every directory the writer reuses must resolve
        inside the root it just claimed.
        """
        if self.out_dir.is_symlink() and not self.out_dir.exists():
            raise SpatialRecordError(
                f"{self.out_dir} is a dangling symlink; a spatial attempt needs a "
                "real output directory"
            )
        root = self.out_dir.resolve()
        self._root = root
        for name in SEALED_MARKERS:
            # lexists: a dangling link at a sealed name is still an occupied name.
            if os.path.lexists(root / name):
                raise SpatialRecordError(
                    f"{self.out_dir} already holds a spatial product ({name}); "
                    "write into a fresh attempt directory instead of "
                    "overwriting a sealed one"
                )
        self._owned_directory(DIRECTORY_FEATURES)
        existing = sorted(
            path.name
            for path in (root / DIRECTORY_FEATURES).iterdir()
            if path.is_file()
        )
        if existing:
            raise SpatialRecordError(
                f"{root / DIRECTORY_FEATURES} already holds payload files "
                f"({', '.join(existing[:3])}); this attempt cannot claim them"
            )

    def _owned_directory(self, name: str) -> Path:
        """Create/validate one writer-owned subdirectory inside the product root."""
        directory = self._root / name
        if directory.is_symlink():
            raise SpatialRecordError(
                f"{directory} is a symbolic link; this attempt writes only inside "
                "its own product directory"
            )
        directory.mkdir(parents=True, exist_ok=True)
        resolved = directory.resolve()
        if resolved != directory or resolved.parent != self._root:
            raise SpatialRecordError(
                f"{directory} resolves to {resolved}, outside the product root "
                f"{self._root}"
            )
        return directory

    def _claim_path(self, path: Path) -> None:
        """A written path must be new, unlinked and inside the product root.

        ``exists()`` follows symlinks, so a dangling link at a claimed name looks
        absent while ``write_bytes`` would still follow it. ``lexists`` catches the
        link itself: an attempt may only create regular new files inside its own
        root.
        """
        if path.is_symlink():
            raise SpatialRecordError(
                f"refusing to write {path}: the name is a symbolic link, and this "
                "attempt writes only its own regular files"
            )
        resolved = path.resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise SpatialRecordError(
                f"refusing to write {path}: it resolves to {resolved}, outside the "
                f"product root {self._root}"
            )
        if os.path.lexists(path) and path not in self._created:
            raise SpatialRecordError(
                f"refusing to overwrite {path} that this attempt did not create"
            )

    # -- payload writing -------------------------------------------------
    def _write_npz(self, sample_id: int, arrays: Mapping[str, np.ndarray]) -> Path:
        path = self._root / DIRECTORY_FEATURES / f"{sample_id:06d}.npz"
        self._claim_path(path)
        buffer = io.BytesIO()
        np.savez(buffer, **{name: _canonical_array(array) for name, array in arrays.items()})
        payload = buffer.getvalue()
        path.write_bytes(payload)
        self._created.append(path)
        self._sample_payloads[sample_id] = path
        # A sample's archive can be extended after its first flush (protocol views
        # are written later), so the manifest keeps exactly one record per payload
        # file and always the hash of its final bytes.
        relative = path.relative_to(self._root).as_posix()
        self._files = [
            record for record in self._files if record.get("path") != relative
        ]
        self._files.append(
            {
                "role": "payload",
                "path": path.relative_to(self._root).as_posix(),
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
            # Take an independent snapshot: the caller may reuse its buffer after
            # write() returns, and a product whose bytes moved while its descriptor
            # stayed fixed is unreadable for everyone.
            owned = _owned_array(array)
            logical[logical_name] = array_descriptor(role, owned)
            if self.layout == "sharded":
                refs[logical_name] = self._stage_row(logical_name, role, owned)
                continue
            key = f"v{slot:04d}.{logical_name}"
            arrays[key] = owned
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
        path = self._root / relative
        self._claim_path(path)
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
        while (self._root / next_path).exists():
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
        index_path = self._root / FILENAME_INDEX
        self._claim_path(index_path)
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
            self._write_annotation_payloads()
            records = []
            for record in self._annotations.get("records", []):
                entry = dict(record)
                reference = self._annotation_refs.get(str(entry.get("image_key")))
                if reference is not None and str(entry.get("role")) == "panoptic":
                    entry["raster"] = dict(reference)
                records.append(entry)
            payload = b"".join(
                canonical_json_bytes(record) + b"\n" for record in records
            )
            annotations_path = self._root / FILENAME_ANNOTATIONS
            self._claim_path(annotations_path)
            annotations_path.write_bytes(payload)
            self._created.append(annotations_path)
            self._files.append(
                {
                    "role": "annotations",
                    "path": FILENAME_ANNOTATIONS,
                    "sha256": sha256_bytes(payload),
                    "bytes": len(payload),
                    "records": len(records),
                }
            )
            annotations = {
                "status": "present",
                "path": FILENAME_ANNOTATIONS,
                "sha256": sha256_bytes(payload),
                "records": len(records),
                "categories": dict(self._annotations.get("categories", {})),
                "sources": dict(self._annotations.get("sources", {})),
                "coordinate_adapter": SPATIAL_COORDINATE_ADAPTER,
                "mask_sampling": SPATIAL_MASK_SAMPLING,
                "rasters": {
                    key: dict(reference)
                    for key, reference in sorted(self._annotation_refs.items())
                },
            }
        plan = {"status": "absent"}
        if self._evaluation_plan is not None:
            plan_bytes = canonical_json_bytes(self._evaluation_plan)
            plan_path = self._root / FILENAME_EVALUATION_PLAN
            self._claim_path(plan_path)
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
        manifest_path = self._root / FILENAME_MANIFEST
        self._claim_path(manifest_path)
        manifest_path.write_bytes(manifest_bytes)
        self._created.append(manifest_path)
        self._manifest = manifest
        self._closed = True
        return manifest

    def _write_annotation_payloads(self) -> None:
        """Seal original categorical rasters under ``annotations/``.

        A panoptic mask is a categorical raster, not an intensity image: it is
        stored as an uncompressed ``.npz`` array with its own hash and shape, so a
        product keeps the supplied labels after the original COCO/mask paths are
        gone. Nothing is transformed here - the stored raster is the source pixel
        grid, and view rasters are rebuilt from it on read.
        """
        if not self._annotation_payloads:
            return
        self._owned_directory(DIRECTORY_ANNOTATIONS)
        for image_key, raster in self._annotation_payloads.items():
            sample_id = self._sample_index.get(str(image_key))
            if sample_id is None:
                raise SpatialRecordError(
                    f"annotation raster for {image_key!r} has no stored sample"
                )
            array = _validated_categorical_raster(
                raster,
                image_key=str(image_key),
                size=(self.samples[sample_id].width, self.samples[sample_id].height),
            )
            relative = f"{DIRECTORY_ANNOTATIONS}/{sample_id + 1:06d}.npz"
            path = self._root / relative
            self._claim_path(path)
            buffer = io.BytesIO()
            np.savez(buffer, raster=_canonical_array(array))
            payload = buffer.getvalue()
            path.write_bytes(payload)
            self._created.append(path)
            self._files.append(
                {
                    "role": "annotation-raster",
                    "path": relative,
                    "sha256": sha256_bytes(payload),
                    "bytes": len(payload),
                    "uncompressed_bytes": int(_canonical_array(array).nbytes),
                    "records": 1,
                    "rows": 1,
                    "oversized_record": False,
                }
            )
            self._annotation_refs[str(image_key)] = {
                "kind": "npz",
                "path": relative,
                "key": "raster",
                "sha256": sha256_bytes(payload),
                "shape": [int(v) for v in array.shape],
                "dtype": _canonical_array(array).dtype.str,
                "sampling": SPATIAL_MASK_SAMPLING,
            }

    def annotation_identity(self) -> dict[str, Any] | None:
        """Semantic identity of the stored annotations, without any path.

        A product that keeps the same feature grids but different source
        annotations is not the same logical product, so the normalized records,
        the category tables and the categorical rasters enter the identity through
        their own digests. Absolute roots, mask paths and physical file names are
        deliberately excluded, exactly like the layout for the views.
        """
        if not self._annotations:
            return None
        records = [
            {
                "image_key": record.get("image_key"),
                "role": record.get("role"),
                "annotation_id": record.get("annotation_id"),
                "category_id": record.get("category_id"),
                "annotation": record.get("annotation"),
            }
            for record in self._annotations.get("records", [])
        ]
        sources = {
            str(role): {"sha256": value.get("sha256"), "images": value.get("images")}
            for role, value in (self._annotations.get("sources") or {}).items()
        }
        rasters = {
            str(key): array_digest("annotation-raster", np.asarray(raster))
            for key, raster in self._annotation_payloads.items()
        }
        return canonical_digest(
            {
                "records": records,
                "categories": {
                    str(key): value
                    for key, value in (self._annotations.get("categories") or {}).items()
                },
                "sources": sources,
                "rasters": rasters,
                "coordinate_adapter": SPATIAL_COORDINATE_ADAPTER,
                "mask_sampling": SPATIAL_MASK_SAMPLING,
            }
        )

    def logical_id(self) -> str:
        """Layout-independent product identity over ordered logical records.

        The stored evaluation contract (protocol, seed, sampled keys and query
        plan) and the stored annotation content are part of the identity: two runs
        that store the same views under different plans, or with different labels,
        rasters or category tables, are not the same product.
        """
        payload = {
            "schema_version": SPATIAL_SCHEMA_VERSION,
            "feature_contract": self.contract.to_json(),
            "support_version": SPATIAL_SUPPORT_VERSION,
            "appearance": dict(APPEARANCE_PROTOCOL),
            "annotation_identity": self.annotation_identity(),
            "evaluation_plan": (
                canonical_digest(self._evaluation_plan)
                if self._evaluation_plan is not None
                else None
            ),
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
        return SpatialReader(self._root, _staging=True)

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
        features = self._root / DIRECTORY_FEATURES
        try:
            if not features.is_symlink() and features.is_dir() and not any(
                features.iterdir()
            ):
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

    def __init__(
        self,
        path: Path,
        *,
        _staging: bool = False,
        open_payload_limit: int = OPEN_PAYLOAD_LIMIT,
        selector_csv: Path | None = None,
    ) -> None:
        self.path = Path(path)
        # Only the writer's own ``reader()`` may open its just-sealed files before
        # the completion envelope exists. The public entry point never can.
        self._require_completion = not _staging
        self._open_limit = max(1, int(open_payload_limit))
        self.manifest = self._load_manifest()
        self._index = self._load_index()
        self._selected: tuple[str, ...] = tuple(
            dict.fromkeys(entry["sample"]["image_key"] for entry in self._index)
        )
        self.overlay: SelectorOverlay | None = None
        # Resolve the selector first: plan validation reads stored support, and it
        # must only read payloads of the selected samples. The whole index is
        # still schema-checked; this narrows payload bytes, not structure.
        if selector_csv is not None:
            keys, overlay = read_selector(
                Path(selector_csv), self._selected, self
            )
            self._selected = keys
            self.overlay = overlay
        self._valid_patch_cache: dict[tuple[str, str], int] = {}
        self._support_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._view_cache: dict[tuple[str, str], SpatialView] = {}
        self._verified_payloads: list[str] = []
        self._verified_stats: dict[str, tuple[int, int, int]] = {}
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

    def _check_descriptor_schema(
        self, name: str, descriptor: Any, identity: tuple[str, str]
    ) -> None:
        """Every array descriptor must be complete and match the contract."""
        if not isinstance(descriptor, Mapping):
            raise SpatialRecordError(
                f"index entry {identity} has a non-object descriptor for {name!r}"
            )
        for field in ("role", "dtype", "shape", "digest"):
            if field not in descriptor:
                raise SpatialRecordError(
                    f"index entry {identity} descriptor for {name!r} has no {field!r}"
                )
        digest = descriptor["digest"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise SpatialRecordError(
                f"index entry {identity} descriptor for {name!r} has no sha256 digest"
            )
        try:
            dtype = np.dtype(str(descriptor["dtype"]))
        except TypeError:
            raise SpatialRecordError(
                f"index entry {identity} descriptor for {name!r} has an unusable "
                f"dtype {descriptor['dtype']!r}"
            ) from None
        shape = descriptor["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or not all(
                isinstance(v, int) and not isinstance(v, bool) and v > 0
                for v in shape
            )
        ):
            raise SpatialRecordError(
                f"index entry {identity} descriptor for {name!r} has an unusable "
                f"shape {shape!r}"
            )
        first = self.contract.layers[0]
        grid_hw = [int(first.grid_hw[0]), int(first.grid_hw[1])]
        expect_fp32 = name.startswith(FEATURE_ARRAY_PREFIX)
        if expect_fp32:
            if dtype != np.dtype(np.float32):
                raise SpatialRecordError(
                    f"index entry {identity} declares feature {name!r} dtype "
                    f"{dtype}, but the contract stores FP32"
                )
            path = name[len(FEATURE_ARRAY_PREFIX):]
            spec = next(
                (layer for layer in self.contract.layers if layer.path == path), None
            )
            if spec is None:
                raise SpatialRecordError(
                    f"index entry {identity} declares an unknown feature layer "
                    f"{path!r}"
                )
            grid_hw = [int(spec.grid_hw[0]), int(spec.grid_hw[1])]
            channels = int(spec.channels)
            if list(shape) != [grid_hw[0], grid_hw[1], channels]:
                raise SpatialRecordError(
                    f"index entry {identity} declares {path!r} shape {shape}, but the "
                    f"contract grid/channels are {[grid_hw[0], grid_hw[1], channels]}"
                )
            return
        if name in (PIXEL_COVERAGE,):
            if dtype != np.dtype(np.float32):
                raise SpatialRecordError(
                    f"index entry {identity} declares {name!r} dtype {dtype}, not FP32"
                )
        elif name in (PIXEL_VALID, PATCH_VALID):
            if dtype != np.dtype(bool):
                raise SpatialRecordError(
                    f"index entry {identity} declares {name!r} dtype {dtype}, not "
                    "Boolean"
                )
        elif name in (PATCH_COVERAGE,) and dtype != np.dtype(np.float32):
            raise SpatialRecordError(
                f"index entry {identity} declares {name!r} dtype {dtype}, not FP32"
            )
        if name in (PATCH_COVERAGE, PATCH_VALID):
            if list(shape) != [grid_hw[0], grid_hw[1]]:
                raise SpatialRecordError(
                    f"index entry {identity} declares {name!r} shape {shape}, but the "
                    f"contract grid is {grid_hw}"
                )

    def _check_ref_schema(
        self, name: str, ref: Any, identity: tuple[str, str]
    ) -> None:
        """Every reference must be a supported, contained payload locator."""
        if not isinstance(ref, Mapping):
            raise SpatialRecordError(
                f"index entry {identity} has a non-object ref for {name!r}"
            )
        kind = ref.get("kind")
        if kind not in ("npz", "npy"):
            raise SpatialRecordError(
                f"index entry {identity} ref for {name!r} has unsupported kind {kind!r}"
            )
        if not isinstance(ref.get("path"), str) or not ref["path"]:
            raise SpatialRecordError(
                f"index entry {identity} ref for {name!r} has no payload path"
            )
        self._check_relative(ref["path"])
        if kind == "npz":
            if not isinstance(ref.get("key"), str) or not ref["key"]:
                raise SpatialRecordError(
                    f"index entry {identity} ref for {name!r} has no NPZ key"
                )
        else:
            row = ref.get("row")
            if not isinstance(row, int) or isinstance(row, bool) or row < 0:
                raise SpatialRecordError(
                    f"index entry {identity} ref for {name!r} has no NPY row index"
                )

    def _check_trace_structure(
        self, entry: Mapping[str, Any], identity: tuple[str, str]
    ) -> None:
        """Every entry's geometry must be a valid, invertible trace.

        Unselected payloads are not hashed, but their metadata is: a singular map or
        a malformed trace there would make the product's index describe a geometry
        no reader could interpret, and would only surface once that sample is
        selected.
        """
        try:
            trace = GeometryTrace.from_json(entry["geometry"])
        except (KeyError, TypeError, ValueError) as error:
            raise SpatialRecordError(
                f"index entry {identity} has an unusable geometry trace: {error}"
            ) from None
        if trace.view_wh != tuple(int(v) for v in self.contract.input_hw):
            raise SpatialRecordError(
                f"index entry {identity} stores canvas {trace.view_wh} but the "
                f"feature contract fixes {self.contract.input_hw}"
            )
        for name in ("pixel_geometric_coverage", "pixel_sampling_valid"):
            descriptor = entry["arrays"][name]
            shape = [int(v) for v in descriptor["shape"]]
            if shape != [int(trace.view_wh[1]), int(trace.view_wh[0])]:
                raise SpatialRecordError(
                    f"index entry {identity} declares {name} shape {shape}, but its "
                    f"canvas is {tuple(trace.view_wh)} (height, width)"
                )
        for path in self.contract.layer_paths:
            descriptor = entry["arrays"][f"{FEATURE_ARRAY_PREFIX}{path}"]
            shape = [int(v) for v in descriptor["shape"]]
            if len(shape) != 3 or [
                int(v) for v in self.contract.layers[0].grid_hw
            ] != shape[:2]:
                raise SpatialRecordError(
                    f"index entry {identity} declares {path} shape {shape}, which does "
                    "not match the contract grid"
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

    def _sealed_companion(self, relative: Any, *, role: str) -> bytes:
        """Bytes of a manifest-sealed companion file, checked like any payload.

        The annotation snapshot and the saved plan are *referenced* by the
        manifest, so their paths must satisfy the same containment, inventory and
        digest rules as payloads. Following ``annotations.path`` on its own would
        let a manifest point outside the product root.
        """
        if not isinstance(relative, str) or not relative:
            raise SpatialRecordError(f"the manifest {role} entry has no usable path")
        self._check_relative(relative)
        item = next(
            (
                entry
                for entry in self.manifest["files"]
                if entry.get("path") == relative
            ),
            None,
        )
        if item is None:
            raise SpatialRecordError(
                f"the {role} file {relative} is not sealed by the manifest inventory"
            )
        if item.get("role") != role:
            raise SpatialRecordError(
                f"manifest file {relative} is listed as {item.get('role')!r}, not "
                f"{role!r}"
            )
        path = self.path / relative
        if not path.is_file():
            raise SpatialRecordError(f"sealed {role} file is missing: {relative}")
        payload = path.read_bytes()
        if sha256_bytes(payload) != item["sha256"]:
            raise SpatialRecordError(
                f"the {role} file {relative} does not match its manifest inventory digest"
            )
        return payload

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
        sample_ids: dict[int, str] = {}
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
            sample_id = int(entry["sample_id"])
            if sample_id < 1:
                raise SpatialRecordError("sample ids start at 1")
            # A sample id identifies one sample: two keys sharing it would make the
            # write-time identity and the read-time grouping disagree.
            claimed = sample_ids.setdefault(sample_id, identity[0])
            if claimed != identity[0]:
                raise SpatialRecordError(
                    f"sample id {sample_id} is claimed by both {claimed!r} and "
                    f"{identity[0]!r}"
                )
            if entry.get("role") not in SPATIAL_VIEW_ROLES:
                raise SpatialRecordError(
                    f"index entry {identity} has unknown view role "
                    f"{entry.get('role')!r}"
                )
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
            # Required arrays and a usable trace are structural, so they are checked
            # for every entry - selected or not. Only the payload *bytes* are
            # skipped, which is what "whole index, selected payloads" promised.
            for required in (
                PIXEL_COVERAGE,
                PIXEL_VALID,
                PATCH_COVERAGE,
                PATCH_VALID,
            ):
                if required not in entry["arrays"]:
                    raise SpatialRecordError(
                        f"index entry {identity} has no {required!r} array"
                    )
            for path in self.contract.layer_paths:
                if f"{FEATURE_ARRAY_PREFIX}{path}" not in entry["arrays"]:
                    raise SpatialRecordError(
                        f"index entry {identity} has no feature array for {path!r}"
                    )
            if set(entry["refs"]) != set(entry["arrays"]):
                raise SpatialRecordError(
                    f"index entry {identity} does not reference every stored array"
                )
            for name, descriptor in entry["arrays"].items():
                self._check_descriptor_schema(name, descriptor, identity)
            for name, ref in entry["refs"].items():
                self._check_ref_schema(name, ref, identity)
            self._check_trace_structure(entry, identity)
            entries.append(entry)
        if len(entries) != int(self.manifest["index"]["entries"]):
            raise SpatialRecordError("index entry count does not match the manifest")
        return entries

    def _load_annotations(self) -> dict[str, Any]:
        entry = self.manifest["annotations"]
        status = entry["status"]
        if status == "absent":
            return {"status": "absent", "records": [], "rasters": {}}
        if status != "present":
            raise SpatialRecordError(
                f"unsupported annotation status {status!r} in the manifest"
            )
        payload = self._sealed_companion(entry.get("path"), role="annotations")
        if sha256_bytes(payload) != entry["sha256"]:
            raise SpatialRecordError("annotation snapshot bytes do not match")
        records = [
            json.loads(line.decode("utf-8"))
            for line in payload.splitlines()
            if line.strip()
        ]
        rasters = dict(entry.get("rasters", {}) or {})
        for image_key, reference in rasters.items():
            if not isinstance(reference, dict) or "path" not in reference:
                raise SpatialRecordError(
                    f"the annotation raster for {image_key!r} has no sealed path"
                )
            self._check_relative(reference["path"])
        return {
            "status": "present",
            "records": records,
            "categories": dict(entry.get("categories", {})),
            "sources": dict(entry.get("sources", {})),
            "coordinate_adapter": entry.get("coordinate_adapter"),
            "mask_sampling": entry.get("mask_sampling"),
            "rasters": rasters,
        }

    def _load_evaluation_plan(self) -> dict[str, Any] | None:
        entry = self.manifest["evaluation_plan"]
        status = entry["status"]
        if status == "absent":
            return None
        if status != "present":
            raise SpatialRecordError(
                f"unsupported evaluation-plan status {status!r} in the manifest"
            )
        payload = self._sealed_companion(entry.get("path"), role="evaluation-plan")
        if sha256_bytes(payload) != entry["sha256"]:
            raise SpatialRecordError("evaluation plan bytes do not match")
        plan = json.loads(payload.decode("utf-8"))
        self._validate_plan(plan)
        return plan

    def _validate_plan(self, plan: Mapping[str, Any]) -> None:
        """Complete structural check of a saved plan, at open time.

        A plan is the stored evaluation contract, so every field evaluation reads
        is checked here instead of failing later with a ``KeyError`` or silently
        reporting zero queries: the required top-level fields, the frozen protocol
        bounds (image and query limits), the exact set of directions per pair, the
        stored view references, and every query's grid bounds and points.
        """
        from ..constants import (
            SPATIAL_EVAL_SAMPLE_SIZE,
            SPATIAL_PROTOCOL,
            SPATIAL_QUERY_LIMIT,
        )
        from .protocol import (
            DIRECTIONS,
            PROTOCOL_VIEW_IDS,
            REFERENCE_ALIASES,
            grid_center_point,
        )

        if plan.get("protocol") != SPATIAL_PROTOCOL:
            raise SpatialRecordError(
                f"saved plan protocol {plan.get('protocol')!r} is not {SPATIAL_PROTOCOL!r}"
            )
        require_schema_version(plan.get("schema_version"))
        require_support_version(plan.get("support_version"))
        required = {
            "protocol",
            "schema_version",
            "support_version",
            "seed",
            "image_limit",
            "query_limit",
            "view_ids",
            "selected_keys",
            "pairs",
        }
        missing = sorted(required - set(plan))
        if missing:
            raise SpatialRecordError(
                "the saved evaluation plan is missing required field(s): "
                + ", ".join(missing)
            )
        selected = plan["selected_keys"]
        if not isinstance(selected, list) or not selected:
            raise SpatialRecordError(
                "the saved evaluation plan selects no keys, so it describes no "
                "evaluation"
            )
        pairs = plan["pairs"]
        if not isinstance(pairs, list) or not pairs:
            raise SpatialRecordError(
                "the saved evaluation plan holds no pairs, so it describes no "
                "evaluation"
            )
        for name in ("seed", "image_limit", "query_limit"):
            value = plan[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise SpatialRecordError(
                    f"saved plan {name} must be an integer, got {value!r}"
                )
        view_ids = plan["view_ids"]
        if not isinstance(view_ids, list) or not view_ids:
            raise SpatialRecordError("saved plan view_ids must be a non-empty list")
        image_limit = plan["image_limit"]
        query_limit = plan["query_limit"]
        if image_limit <= 0 or image_limit > SPATIAL_EVAL_SAMPLE_SIZE:
            raise SpatialRecordError(
                f"saved plan image_limit {image_limit} is outside the frozen "
                f"protocol bound (1..{SPATIAL_EVAL_SAMPLE_SIZE})"
            )
        if query_limit <= 0 or query_limit > SPATIAL_QUERY_LIMIT:
            raise SpatialRecordError(
                f"saved plan query_limit {query_limit} is outside the frozen "
                f"protocol bound (1..{SPATIAL_QUERY_LIMIT})"
            )
        if len(selected) > image_limit:
            raise SpatialRecordError(
                f"the saved plan selects {len(selected)} keys but its image_limit is "
                f"{image_limit}"
            )
        stored = {
            (entry["sample"]["image_key"], entry["view_id"]) for entry in self._index
        }
        keys = {entry["sample"]["image_key"] for entry in self._index}
        missing_keys = [key for key in selected if key not in keys]
        if missing_keys:
            raise SpatialRecordError(
                "the saved plan selects keys this product does not contain: "
                + ", ".join(missing_keys[:3])
            )
        covered: dict[str, set[str]] = {key: set() for key in selected}
        selected_set = set(self._selected)
        seen_pairs: set[tuple[str, str]] = set()
        # The frozen protocol lives in the code, never in the input being checked:
        # a plan that shrinks `view_ids` must not be able to redefine the set.
        if set(view_ids) != set(PROTOCOL_VIEW_IDS):
            raise SpatialRecordError(
                "the saved plan view_ids do not hold the frozen protocol views: "
                f"missing {sorted(set(PROTOCOL_VIEW_IDS) - set(view_ids))}, "
                f"unexpected {sorted(set(view_ids) - set(PROTOCOL_VIEW_IDS))}"
            )
        expected_targets = set(PROTOCOL_VIEW_IDS) - set(REFERENCE_ALIASES)
        for pair in pairs:
            # Only the selected samples' payload bytes may be read here; the rest of
            # the plan is still checked structurally, above and below.
            checked = pair.get("image_key") in covered and pair.get(
                "image_key"
            ) in selected_set
            for field in (
                "image_key",
                "pair_id",
                "reference_view",
                "target_view",
                "patch_hw",
                "directions",
                "geometry",
            ):
                if field not in pair:
                    raise SpatialRecordError(
                        f"a saved plan pair is missing required field {field!r}"
                    )
            image_key = pair["image_key"]
            target = pair["target_view"]
            if not isinstance(image_key, str) or not isinstance(target, str):
                raise SpatialRecordError(
                    "a saved plan pair needs string image_key/target_view values"
                )
            if image_key not in covered:
                raise SpatialRecordError(
                    f"a planned pair covers {image_key!r}, which the plan does not select"
                )
            identity = (image_key, str(pair["pair_id"]))
            if identity in seen_pairs:
                raise SpatialRecordError(
                    f"the saved plan lists pair {identity[1]!r} for {image_key!r} twice, "
                    "which would score the same queries again"
                )
            seen_pairs.add(identity)
            # The frozen protocol fixes one pair per transform, named by its target.
            if identity[1] != target:
                raise SpatialRecordError(
                    f"a saved plan pair id {identity[1]!r} does not match its target "
                    f"view {target!r}"
                )
            if (image_key, target) not in stored:
                raise SpatialRecordError(
                    f"a planned pair references a missing stored view: "
                    f"{image_key!r}/{target!r}"
                )
            reference = pair["reference_view"]
            if (image_key, reference) not in stored:
                raise SpatialRecordError(
                    f"a planned pair references a missing reference view: "
                    f"{image_key!r}/{reference!r}"
                )
            # A pair must compare two different views, and the source side must be
            # the stored reference. Otherwise a transform "evaluated against
            # itself" would score a perfect match under the frozen protocol name.
            if reference == target:
                raise SpatialRecordError(
                    f"a planned pair for {image_key!r}/{identity[1]!r} uses "
                    f"{reference!r} as both its reference and its target, which is "
                    "not a transform evaluation"
                )
            reference_role = self._view_role(image_key, reference)
            if reference_role not in ("base", "eval-reference"):
                raise SpatialRecordError(
                    f"a planned pair for {image_key!r}/{identity[1]!r} names "
                    f"{reference!r} as its reference, but that stored view has role "
                    f"{reference_role!r}"
                )
            patch_hw = pair["patch_hw"]
            if (
                not isinstance(patch_hw, list)
                or len(patch_hw) != 2
                or not all(
                    isinstance(v, int) and not isinstance(v, bool) and v > 0
                    for v in patch_hw
                )
            ):
                raise SpatialRecordError(
                    f"a saved plan pair for {image_key!r}/{identity[1]!r} needs a "
                    f"positive integer patch_hw, got {patch_hw!r}"
                )
            # A positive size is not enough: evaluation consumes this value as the
            # patch lattice, so a different one silently moves every predicted
            # centre. It must be the stored feature contract's patch size.
            pair_patch_hw = patch_hw
            contract_patch = [int(v) for v in self.contract.patch_hw]
            if patch_hw != contract_patch:
                raise SpatialRecordError(
                    f"a saved plan pair for {image_key!r}/{identity[1]!r} declares "
                    f"patch_hw {patch_hw}, but the stored feature contract is "
                    f"{contract_patch}"
                )
            self._validate_pair_geometry(pair, image_key=image_key, target=target)
            branches = pair["directions"]
            if not isinstance(branches, dict) or set(branches) != set(DIRECTIONS):
                raise SpatialRecordError(
                    f"a planned pair for {image_key!r}/{pair['pair_id']!r} must hold "
                    f"exactly the directions {list(DIRECTIONS)}, got "
                    f"{sorted(branches) if isinstance(branches, dict) else branches!r}"
                )
            covered[image_key].add(identity[1])
            grid = self._view_grid(image_key, target)
            for direction in DIRECTIONS:
                branch = branches[direction]
                if not isinstance(branch, dict) or "queries" not in branch:
                    raise SpatialRecordError(
                        f"a saved plan {direction} branch of "
                        f"{image_key!r}/{pair['pair_id']!r} has no query list"
                    )
                # The branch names the pair it runs in; it must be the pair's own
                # views, in the order this direction reads them.
                expected_source = (
                    reference
                    if direction == "reference_to_target"
                    else target
                )
                expected_destination = (
                    target
                    if direction == "reference_to_target"
                    else reference
                )
                for field, expected in (
                    ("source_view", expected_source),
                    ("target_view", expected_destination),
                ):
                    if field not in branch:
                        raise SpatialRecordError(
                            f"a saved plan {direction} branch is missing {field!r}"
                        )
                    if branch[field] != expected:
                        raise SpatialRecordError(
                            f"a saved plan {direction} branch of "
                            f"{image_key!r}/{identity[1]!r} names {field} "
                            f"{branch[field]!r}, but the pair stores {expected!r}"
                        )
                queries = branch["queries"]
                if not isinstance(queries, list):
                    raise SpatialRecordError(
                        f"a saved plan {direction} branch has a non-list query field"
                    )
                if len(queries) > query_limit:
                    raise SpatialRecordError(
                        f"a saved plan branch holds {len(queries)} queries, more than "
                        f"its frozen limit {query_limit}"
                    )
                self._validate_branch_counts(
                    branch,
                    image_key=image_key,
                    pair_id=identity[1],
                    direction=direction,
                    queries=queries,
                    query_limit=query_limit,
                    check_support=checked,
                )
                cells = set()
                for query in queries:
                    cell = (query.get("row"), query.get("column"))
                    if cell in cells:
                        raise SpatialRecordError(
                            f"the saved plan {direction} branch of "
                            f"{image_key!r}/{identity[1]!r} lists cell {cell} twice, "
                            "which would score the same query again"
                        )
                    cells.add(cell)
                if grid is None:
                    continue
                source_trace = self._view_trace(image_key, expected_source)
                target_trace = self._view_trace(image_key, expected_destination)
                pair_patch = (
                    int(pair_patch_hw[0]),
                    int(pair_patch_hw[1]),
                )
                for query in queries:
                    row, column = query.get("row"), query.get("column")
                    if not isinstance(row, int) or not isinstance(column, int):
                        raise SpatialRecordError(
                            "a saved plan query has no integer row/column"
                        )
                    if not (0 <= row < grid[0] and 0 <= column < grid[1]):
                        raise SpatialRecordError(
                            f"a saved plan query ({row},{column}) is outside the "
                            f"stored {grid[0]}x{grid[1]} grid"
                        )
                    for name in ("xy_source", "xy_target"):
                        point = query.get(name)
                        if (
                            not isinstance(point, list)
                            or len(point) != 2
                            or not all(
                                isinstance(value, (int, float))
                                and not isinstance(value, bool)
                                and math.isfinite(float(value))
                                for value in point
                            )
                        ):
                            raise SpatialRecordError(
                                f"a saved plan query has no finite {name} point"
                            )
                    self._validate_query_geometry(
                        query,
                        image_key=image_key,
                        pair_id=identity[1],
                        direction=direction,
                        patch_hw=pair_patch,
                        source_trace=source_trace,
                        target_trace=target_trace,
                    )
                    # Only the selected samples' support bytes may be read.
                    if not checked:
                        continue
                    self._validate_query_support(
                        query,
                        image_key=image_key,
                        pair_id=identity[1],
                        direction=direction,
                        patch_hw=pair_patch,
                        source_view=expected_source,
                        target_view=expected_destination,
                    )
            if checked:
                self._planned_eligibility(
                    pair,
                    branches,
                    image_key=image_key,
                    pair_id=identity[1],
                    patch_hw=pair_patch,
                    query_limit=query_limit,
                    seed=int(plan["seed"]),
                )
        for key, pair_ids in covered.items():
            if not pair_ids:
                raise SpatialRecordError(
                    f"the saved plan selects {key!r} but stores no pair for it"
                )
            # The frozen protocol evaluates one pair per transformed view, so a
            # product that silently dropped a transform is not the same contract.
            missing_targets = sorted(expected_targets - pair_ids)
            if missing_targets:
                raise SpatialRecordError(
                    f"the saved plan for {key!r} is missing transform(s) "
                    f"{missing_targets}, so it does not hold the frozen view set"
                )

    def _validate_pair_geometry(
        self, pair: Mapping[str, Any], *, image_key: str, target: str
    ) -> None:
        """A stored pair's geometry signature must agree with the stored views."""
        geometry = pair["geometry"]
        if not isinstance(geometry, dict) or set(geometry) != {"reference", "target"}:
            raise SpatialRecordError(
                f"a saved plan pair for {image_key!r}/{target!r} needs a geometry "
                "record with exactly 'reference' and 'target'"
            )
        for name, expected_view in (
            ("reference", pair["reference_view"]),
            ("target", target),
        ):
            signature = geometry[name]
            if not isinstance(signature, dict):
                raise SpatialRecordError(
                    f"a saved plan {name} geometry signature is not an object"
                )
            entry = next(
                (
                    item
                    for item in self._index
                    if item["sample"]["image_key"] == image_key
                    and item["view_id"] == expected_view
                ),
                None,
            )
            if entry is None:
                continue
            trace = entry["geometry"]
            expected = {
                "view_id": expected_view,
                "source_wh": [int(v) for v in trace["source_wh"]],
                "view_wh": [int(v) for v in trace["view_wh"]],
            }
            for field, value in expected.items():
                if signature.get(field) != value:
                    raise SpatialRecordError(
                        f"a saved plan {name} geometry signature {field} does not "
                        f"match the stored view {image_key!r}/{expected_view!r}"
                    )
            patch_hw = signature.get("patch_hw")
            grid = signature.get("grid_hw")
            stored_grid = self._view_grid(image_key, expected_view)
            if stored_grid is not None and grid != [
                int(stored_grid[0]),
                int(stored_grid[1]),
            ]:
                raise SpatialRecordError(
                    f"a saved plan {name} geometry signature declares grid {grid}, "
                    f"but the stored view {image_key!r}/{expected_view!r} has "
                    f"{list(stored_grid)}"
                )
            contract_patch = [int(v) for v in self.contract.patch_hw]
            if patch_hw != contract_patch:
                raise SpatialRecordError(
                    f"a saved plan {name} geometry signature declares patch_hw "
                    f"{patch_hw}, but the stored feature contract is {contract_patch}"
                )
            # The stored trace is the source of truth for the maps: a signature
            # that disagrees with it would describe a geometry the product does
            # not hold, and evaluation reads the signature.
            expected_matrices = {
                "source_to_view": [
                    [float(v) for v in row] for row in trace["source_to_view"]
                ]
            }
            for field, value in expected_matrices.items():
                declared = signature.get(field)
                if declared is None:
                    # The stored map is what evaluation reads, so its absence is a
                    # missing required signature rather than "nothing to check".
                    raise SpatialRecordError(
                        f"a saved plan {name} geometry signature has no {field!r}"
                    )
                if not isinstance(declared, list) or len(declared) != 3:
                    raise SpatialRecordError(
                        f"a saved plan {name} geometry signature {field} is not a "
                        "3x3 matrix"
                    )
                if not np.allclose(
                    np.asarray(declared, dtype=np.float64),
                    np.asarray(value, dtype=np.float64),
                    atol=1e-9,
                ):
                    raise SpatialRecordError(
                        f"a saved plan {name} geometry signature {field} does not "
                        f"match the stored view {image_key!r}/{expected_view!r}"
                    )

    def _view_trace(self, image_key: str, view_id: str) -> dict[str, Any] | None:
        for entry in self._index:
            if (
                entry["sample"]["image_key"] == image_key
                and entry["view_id"] == view_id
            ):
                return entry["geometry"]
        return None

    def _validate_query_geometry(
        self,
        query: Mapping[str, Any],
        *,
        image_key: str,
        pair_id: str,
        direction: str,
        patch_hw: tuple[int, int],
        source_trace: Mapping[str, Any] | None,
        target_trace: Mapping[str, Any] | None,
    ) -> None:
        """A query's two points must follow from its lattice cell and the maps.

        ``xy_source`` is the source grid centre of ``(row, column)`` and
        ``xy_target`` is that point carried into the target view through the stored
        traces. Evaluation derives ground truth from ``xy_target``, so a finite but
        arbitrary point would become a wrong truth while every digest still matched.
        """
        from ..vision.geometry_support import transform_points
        from .protocol import grid_center_point

        expected_source = list(grid_center_point(int(query["row"]), int(query["column"]), patch_hw))
        source = [float(v) for v in query["xy_source"]]
        if not np.allclose(source, expected_source, atol=1e-6):
            raise SpatialRecordError(
                f"a saved plan query for {image_key!r}/{pair_id!r}/{direction} has "
                f"xy_source {source}, but its grid centre is {expected_source}"
            )
        if source_trace is None or target_trace is None:
            return
        source_to_view = np.asarray(source_trace["source_to_view"], dtype=np.float64)
        view_to_source = np.asarray(source_trace["view_to_source"], dtype=np.float64)
        target_to_view = np.asarray(target_trace["source_to_view"], dtype=np.float64)
        composed = target_to_view @ view_to_source
        expected_target = transform_points(composed, np.asarray([source]))[0]
        target = [float(v) for v in query["xy_target"]]
        if not np.allclose(target, expected_target, atol=1e-6):
            raise SpatialRecordError(
                f"a saved plan query for {image_key!r}/{pair_id!r}/{direction} has "
                f"xy_target {target}, but the stored geometry maps its source point "
                f"to {[float(v) for v in expected_target]}"
            )

    def _valid_patch_count(self, image_key: str, view_id: str) -> int | None:
        """Valid patch cells of one stored view, from its stored support array."""
        cached = self._valid_patch_cache.get((image_key, view_id))
        if cached is not None:
            return cached if cached >= 0 else None
        support = self._view_support(image_key, view_id)
        if support is None:
            return None
        count = int(np.count_nonzero(support[0]))
        self._valid_patch_cache[(image_key, view_id)] = count
        return count

    def _stored_view(self, image_key: str, view_id: str) -> SpatialView | None:
        """One stored view rebuilt from its payload, for plan validation.

        At most two views are kept: a pair needs its reference and target in both
        directions, and retaining every planned view would hold the whole plan's
        feature grids resident, which the reader exists to avoid.
        """
        key = (image_key, view_id)
        if key in self._view_cache:
            return self._view_cache[key]
        entry = next(
            (
                item
                for item in self._index
                if item["sample"]["image_key"] == image_key
                and item["view_id"] == view_id
            ),
            None,
        )
        if entry is None:
            return None
        view = self._build_view(entry)
        self._view_cache[key] = view
        while len(self._view_cache) > 2:
            self._view_cache.pop(next(iter(self._view_cache)))
        return view

    def _planned_eligibility(
        self,
        pair: Mapping[str, Any],
        branches: Mapping[str, Any],
        *,
        image_key: str,
        pair_id: str,
        patch_hw: tuple[int, int],
        query_limit: int,
        seed: int,
    ) -> None:
        """Recompute each branch's eligibility and frozen sample from stored bytes.

        The stored counts are what a report publishes, and the checks below them
        can only compare numbers with each other: a plan that makes every branch
        consistently unavailable (`eligible_count` 0, no queries, `no_overlap`)
        stays arithmetically perfect while dropping the whole report, and a
        target-support check that only looks at the target pixel grid accepts a
        point whose feature interpolation support is incomplete. Recomputing the
        protocol's own eligibility and seeded ranking against the stored support
        and geometry pins both, and needs no payload beyond the selected samples.
        """
        from .protocol import _eligible_queries, _rank_queries

        reference = self._stored_view(image_key, str(pair["reference_view"]))
        target = self._stored_view(image_key, str(pair["target_view"]))
        if reference is None or target is None:
            return
        for direction, source, destination in (
            ("reference_to_target", reference, target),
            ("target_to_reference", target, reference),
        ):
            branch = branches[direction]
            queries, counts = _eligible_queries(source, destination, patch_hw)
            for field, actual in (
                ("source_valid_count", counts["source_valid_count"]),
                ("eligible_count", counts["eligible_count"]),
            ):
                if int(branch[field]) != int(actual):
                    raise SpatialRecordError(
                        f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} "
                        f"declares {field} {branch[field]}, but the stored support "
                        f"gives {actual}"
                    )
            expected = _rank_queries(
                queries,
                seed=int(seed),
                image_key=image_key,
                pair_id=pair_id,
                direction=direction,
                limit=query_limit,
            )
            stored_cells = [
                (int(query["row"]), int(query["column"]))
                for query in branch["queries"]
            ]
            if stored_cells != [
                (int(query["row"]), int(query["column"])) for query in expected
            ]:
                raise SpatialRecordError(
                    f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} "
                    "does not store the query set the frozen protocol samples "
                    "from the stored support"
                )

    def _view_support(self, image_key: str, view_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Stored patch and pixel support of one view."""
        cached = self._support_cache.get((image_key, view_id))
        if cached is not None:
            return cached
        entry = next(
            (
                item
                for item in self._index
                if item["sample"]["image_key"] == image_key
                and item["view_id"] == view_id
            ),
            None,
        )
        if entry is None:
            return None
        wanted = (PATCH_VALID, PIXEL_VALID)
        if any(name not in entry["arrays"] for name in wanted):
            return None
        arrays = self._load_arrays(
            {
                "arrays": {name: entry["arrays"][name] for name in wanted},
                "refs": {name: entry["refs"][name] for name in wanted},
            }
        )
        support = (
            np.asarray(arrays[PATCH_VALID], dtype=bool),
            np.asarray(arrays[PIXEL_VALID], dtype=bool),
        )
        self._support_cache[(image_key, view_id)] = support
        return support

    def _validate_query_support(
        self,
        query: Mapping[str, Any],
        *,
        image_key: str,
        pair_id: str,
        direction: str,
        patch_hw: tuple[int, int],
        source_view: str,
        target_view: str,
    ) -> None:
        """A planned query must sit on valid stored support in both views."""
        from .protocol import pixel_point_support

        source_support = self._view_support(image_key, source_view)
        target_support = self._view_support(image_key, target_view)
        if source_support is None or target_support is None:
            return
        patch_valid, _pixel_valid = source_support
        row, column = int(query["row"]), int(query["column"])
        if not bool(patch_valid[row, column]):
            raise SpatialRecordError(
                f"a saved plan query for {image_key!r}/{pair_id!r}/{direction} sits "
                f"on an invalid source patch ({row},{column})"
            )
        _target_patch, target_pixel = target_support
        if not pixel_point_support(target_pixel, query["xy_target"]):
            raise SpatialRecordError(
                f"a saved plan query for {image_key!r}/{pair_id!r}/{direction} maps "
                "outside the stored target pixel support"
            )

    def _view_role(self, image_key: str, view_id: str) -> str | None:
        for entry in self._index:
            if (
                entry["sample"]["image_key"] == image_key
                and entry["view_id"] == view_id
            ):
                return str(entry.get("role"))
        return None

    def _validate_branch_counts(
        self,
        branch: Mapping[str, Any],
        *,
        image_key: str,
        pair_id: str,
        direction: str,
        queries: Sequence[Any],
        query_limit: int,
        check_support: bool = True,
    ) -> None:
        """The stored counts must describe the stored queries and eligibility.

        The counts are what a report publishes about geometric eligibility, so a
        branch that keeps ``sampled_count`` while its query list was emptied (or
        inflated) would claim coverage the product does not hold.
        """
        counts: dict[str, int] = {}
        for field in ("source_valid_count", "eligible_count", "sampled_count"):
            value = branch.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SpatialRecordError(
                    f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} "
                    f"needs a non-negative integer {field}, got {value!r}"
                )
            counts[field] = value
        if counts["sampled_count"] != len(queries):
            raise SpatialRecordError(
                f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} "
                f"declares sampled_count {counts['sampled_count']} but stores "
                f"{len(queries)} queries"
            )
        # The protocol samples ``min(eligible, limit)``, so an eligible branch that
        # stores no query cannot be the frozen evaluation: erasing the queries while
        # keeping a positive eligibility would silently drop the whole report.
        expected_sampled = min(counts["eligible_count"], query_limit)
        if counts["sampled_count"] != expected_sampled:
            raise SpatialRecordError(
                f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} "
                f"declares sampled_count {counts['sampled_count']}, but the frozen "
                f"protocol samples min(eligible={counts['eligible_count']}, "
                f"limit={query_limit}) = {expected_sampled}"
            )
        # The upper bound is not decorative: the source-valid centres are the grid
        # cells of the source view, so a count above that grid describes coverage
        # the product does not hold.
        grid_cells = None
        source_view = branch.get("source_view")
        if isinstance(source_view, str):
            source_grid = self._view_grid(image_key, source_view)
            if source_grid is not None:
                grid_cells = int(source_grid[0]) * int(source_grid[1])
        if grid_cells is not None and counts["source_valid_count"] > grid_cells:
            raise SpatialRecordError(
                f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} "
                f"declares source_valid_count {counts['source_valid_count']}, more "
                f"than the {grid_cells} cells of its source view"
            )
        # The counts must describe the *stored support*, not just each other: the
        # source-valid centres are exactly the valid patch cells of the source
        # view, so a fabricated smaller number would fake a better eligibility
        # fraction and an inflated one would claim coverage the product lacks.
        # ``_planned_eligibility`` recomputes both counts from the stored bytes;
        # this check is the payload-free fallback for an unselected sample.
        if check_support and isinstance(source_view, str):
            actual = self._valid_patch_count(image_key, source_view)
            if actual is not None and counts["source_valid_count"] != actual:
                raise SpatialRecordError(
                    f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} "
                    f"declares source_valid_count {counts['source_valid_count']}, but "
                    f"the stored source view has {actual} valid patch cells"
                )
        if not (
            counts["source_valid_count"]
            >= counts["eligible_count"]
            >= counts["sampled_count"]
        ):
            raise SpatialRecordError(
                f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} has "
                "inconsistent counts: source_valid >= eligible >= sampled is required"
            )
        unavailable = branch.get("unavailable")
        if not isinstance(unavailable, bool):
            raise SpatialRecordError(
                f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} needs "
                f"a boolean 'unavailable' flag, got {unavailable!r}"
            )
        if unavailable != (counts["eligible_count"] == 0):
            raise SpatialRecordError(
                f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} "
                "declares an unavailable state that does not match its eligible count"
            )
        reason = branch.get("unavailable_reason")
        if unavailable:
            expected = (
                "no_overlap"
                if counts["source_valid_count"] > 0
                else "no_source_valid_center"
            )
            if reason != expected:
                raise SpatialRecordError(
                    f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} "
                    f"gives unavailable_reason {reason!r}, expected {expected!r}"
                )
        elif reason is not None:
            raise SpatialRecordError(
                f"a saved plan {direction} branch of {image_key!r}/{pair_id!r} gives "
                "an unavailable_reason for an available branch"
            )

    def _view_grid(
        self, image_key: str, view_id: str
    ) -> tuple[int, int] | None:
        """Stored feature grid of one view, so a plan query can be bounds-checked."""
        for entry in self._index:
            if entry["sample"]["image_key"] != image_key or entry["view_id"] != view_id:
                continue
            for descriptor in entry["arrays"].values():
                if descriptor["role"] == "feature":
                    shape = [int(v) for v in descriptor["shape"]]
                    return (shape[0], shape[1])
        return None

    # -- access ----------------------------------------------------------
    @property
    def image_keys(self) -> tuple[str, ...]:
        return self._selected

    @property
    def product_id(self) -> str:
        return str(self.manifest["product_id"])

    def sample_metadata(self, image_key: str) -> dict[str, Any]:
        """Stored sample metadata, with an optional selector overlay applied.

        The overlay only changes what a report says about a sample; the stored
        record, its hashes and the evaluation contract stay untouched.
        """
        stored: dict[str, Any] = {}
        for entry in self._index:
            if entry["sample"]["image_key"] == image_key:
                stored = dict(entry["sample"].get("metadata") or {})
                break
        if self.overlay is None:
            return stored
        return self.overlay.metadata_for(image_key, stored)

    def overlay_report(self) -> dict[str, Any]:
        """Manifest-level report of the selector overlay, if one was supplied."""
        if self.overlay is None:
            return {"status": "absent"}
        stored = {
            entry["sample"]["image_key"]: dict(
                entry["sample"].get("metadata") or {}
            )
            for entry in self._index
        }
        return self.overlay.report(tuple(self._selected), stored)

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

    # -- annotations -----------------------------------------------------
    def annotation_records(self, image_key: str) -> list[dict[str, Any]]:
        """Stored source annotations of one sample, without any view geometry."""
        return [
            dict(record)
            for record in self.annotations.get("records", [])
            if record.get("image_key") == image_key
        ]

    def annotations_for_view(
        self, image_key: str, view_id: str
    ) -> list[dict[str, Any]]:
        """Rebuild one view's annotations from the snapshot and stored geometry.

        This is what makes the product self-contained for labels: the categorical
        raster and the ordered source annotations are stored, and the view's
        realized operations come from its index record, so no original COCO file,
        mask directory or image is consulted.
        """
        from ..vision.geometry import transform_annotations

        if self.annotations.get("status") != "present":
            return []
        entry = next(
            (
                item
                for item in self._index
                if item["sample"]["image_key"] == image_key
                and item["view_id"] == view_id
            ),
            None,
        )
        if entry is None:
            raise SpatialRecordError(f"no stored view {view_id!r} for {image_key!r}")
        trace = GeometryTrace.from_json(entry["geometry"])
        # Categories stay in their stored namespace (``role:id``), because a
        # stored product may hold two sources that both use category id 1.
        categories = {
            key: dict(value)
            for key, value in (self.annotations.get("categories") or {}).items()
        }
        raster = None
        by_role: dict[str, list[dict[str, Any]]] = {}
        for record in self.annotation_records(image_key):
            role = str(record.get("role"))
            payload = dict(record.get("annotation") or {})
            if role == "panoptic":
                if raster is None:
                    raster = self._load_annotation_raster(image_key)
                payload["raster"] = raster
            by_role.setdefault(role, []).append(payload)
        rebuilt: list[dict[str, Any]] = []
        for role, payloads in by_role.items():
            rebuilt.extend(
                transform_annotations(payloads, categories, trace, role=role)
            )
        return rebuilt

    def _load_annotation_raster(self, image_key: str) -> np.ndarray:
        reference = (self.annotations.get("rasters") or {}).get(image_key)
        if not isinstance(reference, dict):
            raise SpatialRecordError(
                f"the product stores no categorical raster for {image_key!r}"
            )
        payload = self._sealed_companion(reference["path"], role="annotation-raster")
        if reference.get("sha256") and sha256_bytes(payload) != reference["sha256"]:
            raise SpatialRecordError(
                f"the categorical raster for {image_key!r} does not match its digest"
            )
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if "raster" not in archive.files:
                raise SpatialRecordError(
                    f"the categorical raster for {image_key!r} has no 'raster' array"
                )
            # Same allocation-before-validation guard as the feature payloads: the
            # NPY header is checked against the manifest and against the member's
            # real byte length before numpy reserves array memory.
            stored = _checked_npz_array(
                archive,
                "raster",
                label=f"the categorical raster for {image_key!r}",
                declared_shape=reference.get("shape"),
                declared_dtype=reference.get("dtype"),
            )
        declared_shape = reference.get("shape")
        declared_dtype = reference.get("dtype")
        if declared_shape is not None and [int(v) for v in stored.shape] != [
            int(v) for v in declared_shape
        ]:
            raise SpatialRecordError(
                f"the categorical raster for {image_key!r} is {stored.shape}, but the "
                f"manifest declares {list(declared_shape)}"
            )
        if declared_dtype is not None and _canonical_array(stored).dtype.str != str(
            declared_dtype
        ):
            raise SpatialRecordError(
                f"the categorical raster for {image_key!r} is "
                f"{_canonical_array(stored).dtype.str}, but the manifest declares "
                f"{declared_dtype}"
            )
        # Same rules as the write boundary: a product cannot smuggle a float or a
        # NaN raster past either side.
        return _validated_categorical_raster(
            stored,
            image_key=image_key,
            size=self._sample_size(image_key),
        )

    def _sample_size(self, image_key: str) -> tuple[int, int] | None:
        for entry in self._index:
            sample = entry["sample"]
            if sample["image_key"] == image_key:
                return (int(sample["width"]), int(sample["height"]))
        return None

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
                array = _checked_npz_array(
                    archive,
                    ref["key"],
                    label=f"array {ref['key']!r}",
                    declared_shape=descriptor["shape"],
                    declared_dtype=descriptor["dtype"],
                )
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
        return self._cached_open(
            self._open_shards,
            relative,
            lambda: np.load(self.path / relative, mmap_mode="r", allow_pickle=False),
            _release_memmap,
        )

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
        return self._cached_open(
            self._open_npz,
            relative,
            lambda: np.load(self.path / relative, allow_pickle=False),
            _close_archive,
        )

    def _cached_open(
        self,
        cache: dict[str, Any],
        relative: str,
        loader: Any,
        closer: Any,
    ):
        """Open one payload handle, keeping at most ``_open_limit`` alive.

        A whole-product read touches every archive and shard; an unbounded cache
        would keep one descriptor per file open until ``close()``. The cache is a
        small LRU, so a long read stays within a fixed descriptor budget and each
        evicted handle is released immediately.
        """
        cached = cache.pop(relative, None)
        if cached is None:
            self._ensure_payload(relative)
            cached = loader()
        cache[relative] = cached
        while len(cache) > self._open_limit:
            oldest = next(iter(cache))
            closer(cache.pop(oldest))
        return cached

    def _ensure_payload(self, relative: str) -> None:
        """Verify this payload once, unless its file identity changed since.

        Verification reads and hashes the whole file, and the handle cache is
        bounded below the number of payload files. Re-verifying on every reopen
        scaled the hash work with reads instead of with files, so a known
        identity is kept and only a changed size/mtime/inode re-reads.
        """
        item = next(
            (entry for entry in self.manifest["files"] if entry["path"] == relative),
            None,
        )
        if item is None:
            raise SpatialRecordError(
                f"payload {relative} is not sealed by the manifest"
            )
        path = self.path / relative
        try:
            stat = path.stat()
        except OSError:
            raise SpatialRecordError(f"sealed payload is missing: {relative}") from None
        if not path.is_file():
            raise SpatialRecordError(f"sealed payload is missing: {relative}")
        identity = (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))
        if self._verified_stats.get(relative) == identity:
            return
        self._verified_stats.pop(relative, None)
        digest = sha256_bytes(path.read_bytes())
        if digest != item["sha256"]:
            raise SpatialRecordError(f"payload bytes do not match: {relative}")
        self._verified_stats[relative] = identity
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
            _close_archive(archive)
        self._open_npz.clear()
        for shard in self._open_shards.values():
            _release_memmap(shard)
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
    return SpatialReader(Path(path), selector_csv=selector_csv)


def read_selector_keys(path: Path, available: Sequence[str]) -> tuple[str, ...]:
    """``artifact-key-resolution-v1`` selection against stored keys."""
    return read_selector(path, available, None)[0]


def read_selector(
    path: Path, available: Sequence[str], reader: "SpatialReader | None"
) -> tuple[tuple[str, ...], SelectorOverlay]:
    """Resolve a selector CSV and keep its remaining columns as an overlay.

    Only ``image`` participates in selection. Every other column is label or
    string metadata: the design makes it a report overlay, so it is validated,
    hash-recorded and merged into reported metadata - never written into the
    stored records or used to group queries and scores. A repeated column name
    would silently collapse two values, so it is rejected.
    """
    if not path.is_file():
        raise SpatialRecordError(f"selector CSV does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    reader_csv = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader_csv.fieldnames or [])
    if not fieldnames or "image" not in fieldnames:
        raise SpatialRecordError("selector CSV needs an 'image' column")
    if len(set(fieldnames)) != len(fieldnames):
        raise SpatialRecordError("selector CSV repeats a column name")
    available_set = set(available)
    basenames: dict[str, list[str]] = {}
    for key in available:
        basenames.setdefault(key.rsplit("/", 1)[-1], []).append(key)
    selected: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    for row in reader_csv:
        reference = (row.get("image") or "").strip()
        if not reference:
            raise SpatialRecordError("selector CSV has a blank 'image' value")
        key = resolve_stored_key(reference, available_set, basenames)
        selected.append(key)
        rows[key] = {
            field: (row.get(field) or "")
            for field in fieldnames
            if field != "image"
        }
    if not selected:
        raise SpatialRecordError("selector CSV selected no stored keys")
    if len(set(selected)) != len(selected):
        raise SpatialRecordError("selector CSV selects the same stored key twice")
    overlay_fields = tuple(field for field in fieldnames if field != "image")
    stored_fields: tuple[str, ...] = ()
    colliding: tuple[str, ...] = ()
    if reader is not None:
        # The collision set is computed over the *selected* samples, because those
        # are the rows the overlay rewrites. Reading only the first index entry
        # reported no collision when the first sample had no metadata and a later
        # selected one already used the same label.
        selected_set = set(selected)
        stored_metadata: dict[str, dict[str, Any]] = {}
        for entry in reader._index:  # noqa: SLF001 - the reader owns its index
            sample = entry["sample"]
            key = sample["image_key"]
            if key in selected_set and key not in stored_metadata:
                stored_metadata[key] = dict(sample.get("metadata") or {})
        stored_fields = tuple(
            sorted({field for metadata in stored_metadata.values() for field in metadata})
        )
        colliding = tuple(
            sorted(
                {
                    field
                    for metadata in stored_metadata.values()
                    for field in overlay_fields
                    # Only a non-empty overlay value actually replaces a stored one.
                    if field in metadata and any(
                        rows.get(key, {}).get(field) for key in selected_set if key in rows
                    )
                }
            )
        )
    overlay = SelectorOverlay(
        path=str(path),
        sha256=sha256_bytes(path.read_bytes()),
        fields=overlay_fields,
        rows=rows,
        stored_fields=stored_fields,
        colliding_fields=colliding,
    )
    return tuple(selected), overlay


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
