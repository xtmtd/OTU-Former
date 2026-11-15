"""Per-image spatial product: sealing, reopening and trust boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from otuformer.data.provenance import sha256_bytes
from otuformer.spatial.records import (
    FILENAME_MANIFEST,
    FILENAME_PROVENANCE,
    PATCH_COVERAGE,
    PATCH_VALID,
    PIXEL_COVERAGE,
    PIXEL_VALID,
    SpatialRecordError,
    SpatialSample,
    SpatialView,
    samples_from_records,
)
from otuformer.spatial.storage import (
    SpatialReader,
    SpatialWriter,
    open_spatial_product,
    read_selector_keys,
    resolve_stored_key,
)
from otuformer.vision.features import FeatureContract, LayerSpec
from otuformer.vision.geometry import GeometryOp, trace_operations

LAYERS = ("blocks.0", "blocks.1")


def contract_for(
    layers=LAYERS, *, image_size: int = 32, patch: int = 16, channels: int = 3
) -> FeatureContract:
    grid = image_size // patch
    return FeatureContract(
        layers=tuple(
            LayerSpec(path=path, block_index=index, grid_hw=(grid, grid), channels=channels)
            for index, path in enumerate(layers)
        ),
        input_hw=(image_size, image_size),
        patch_hw=(patch, patch),
        prefix_tokens=1,
        norm="backbone",
        norm_module={"path": "norm", "qualname": "timm.layers.norm.LayerNorm"},
        l2=False,
        epsilon=1e-12,
        normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        model_identity={"architecture": "test-vit", "weights": "unit"},
    )


def make_sample(key: str, *, width: int = 640, height: int = 480, metadata=None) -> SpatialSample:
    return SpatialSample(
        image_key=key,
        image_sha256=sha256_bytes(key.encode()),
        width=width,
        height=height,
        metadata=dict(metadata or {}),
    )


def make_view(
    sample: SpatialSample,
    contract: FeatureContract,
    *,
    view_id: str = "base",
    role: str = "base",
    seed: int = 0,
    zero: bool = False,
) -> SpatialView:
    generator = np.random.default_rng(seed)
    grid_h, grid_w = contract.layers[0].grid_hw
    features = {}
    for layer in contract.layers:
        values = np.zeros((grid_h, grid_w, layer.channels), dtype=np.float32)
        if not zero:
            values = generator.normal(size=(grid_h, grid_w, layer.channels)).astype(np.float32)
        features[layer.path] = values
    view_w, view_h = (int(v) for v in contract.input_hw)
    trace = trace_operations(
        (sample.width, sample.height),
        [
            GeometryOp(
                "resize",
                {
                    "from_wh": [sample.width, sample.height],
                    "to_wh": [view_w, view_h],
                    "interpolation": "bicubic",
                    "antialias": True,
                },
            )
        ],
    )
    return SpatialView(
        image_key=sample.image_key,
        view_id=view_id,
        role=role,
        trace=trace,
        features=features,
        pixel_geometric_coverage=np.ones((view_h, view_w), dtype=np.float32),
        pixel_sampling_valid=np.ones((view_h, view_w), dtype=bool),
        geometric_coverage=np.ones((grid_h, grid_w), dtype=np.float32),
        patch_sampling_valid=np.ones((grid_h, grid_w), dtype=bool),
    )


def write_envelope(root: Path, *, status: str = "completed") -> None:
    manifest = (root / FILENAME_MANIFEST).read_bytes()
    record = {
        "status": status,
        "artifacts": [
            {"path": FILENAME_MANIFEST, "sha256": sha256_bytes(manifest)},
        ],
    }
    (root / FILENAME_PROVENANCE).write_text(json.dumps(record, indent=2), encoding="utf-8")


def build_product(
    root: Path,
    *,
    keys=("a/one.png", "b/one.png"),
    layout: str = "per-image",
    contract: FeatureContract | None = None,
    views_per_key: int = 1,
    annotations=None,
    completed: bool = True,
) -> Path:
    contract = contract or contract_for()
    samples = [make_sample(key) for key in keys]
    writer = SpatialWriter(
        root,
        contract=contract,
        layout=layout,
        samples=samples,
        annotations=annotations,
    )
    for seed, sample in enumerate(samples):
        for extra in range(views_per_key):
            view_id = "base" if extra == 0 else f"eval-{extra}"
            role = "base" if extra == 0 else "eval-transform"
            writer.write(make_view(sample, contract, view_id=view_id, role=role, seed=seed + extra))
    writer.finish()
    if completed:
        write_envelope(root)
    return root


def test_npz_roundtrip_is_self_contained_and_root_relocatable(tmp_path):
    root = tmp_path / "product"
    annotations = {
        "records": [{"annotation_id": 1, "image_key": "a/one.png", "category_id": 4}],
        "categories": {"4": {"name": "Head"}},
        "sources": {"instances": {"sha256": "abc"}},
    }
    build_product(root, annotations=annotations)

    # No model or raw images are needed to reopen the product.
    relocated = tmp_path / "moved"
    root.rename(relocated)

    reader = open_spatial_product(relocated)
    try:
        assert reader.manifest["schema_version"] == 1
        assert reader.manifest["layout"] == "per-image"
        assert reader.image_keys == ("a/one.png", "b/one.png")
        assert reader.product_id
        assert reader.annotations["status"] == "present"
        assert reader.annotations["records"][0]["annotation_id"] == 1
        assert reader.evaluation_plan is None

        views = list(reader.iter_views())
        assert len(views) == 2
        first = views[0]
        assert first.view_id == "base" and first.role == "base"
        assert first.features["blocks.0"].dtype == np.float32
        assert first.features["blocks.0"].shape == (2, 2, 3)
        # Pixel support lives at model-input resolution, not at 640x480.
        assert first.pixel_geometric_coverage.shape == (32, 32)
        assert first.pixel_sampling_valid.dtype == np.bool_
        assert first.geometric_coverage.shape == (2, 2)
        assert first.patch_sampling_valid.dtype == np.bool_

        single = reader.get_view("b/one.png", "base")
        assert single.image_key == "b/one.png"
        np.testing.assert_allclose(
            single.features["blocks.1"], views[1].features["blocks.1"]
        )
        # Iterating again reopens payloads instead of reusing an exhausted one.
        assert len(list(reader.iter_views())) == 2
        scope = reader.verification_scope()
        assert scope.index_entries == 2
        assert len(scope.payload_files_verified) == 2
    finally:
        reader.close()

    selector = relocated / ".." / "subset.csv"
    selector.write_text("image\nb/one.png\na/one.png\n", encoding="utf-8")
    reordered = open_spatial_product(relocated, selector_csv=selector)
    try:
        assert reordered.image_keys == ("b/one.png", "a/one.png")
        ordered = list(reordered.iter_views())
        assert [view.image_key for view in ordered] == ["b/one.png", "a/one.png"]
    finally:
        reordered.close()


def test_missing_completion_is_not_a_readable_product(tmp_path):
    root = tmp_path / "product"
    build_product(root, completed=False)
    with pytest.raises(SpatialRecordError, match="no completed run envelope"):
        open_spatial_product(root)
    # The writer-owned staging reader still opens its own sealed files.
    reader = SpatialReader(root, _staging=True)
    reader.close()

    write_envelope(root, status="failed")
    with pytest.raises(SpatialRecordError, match="not 'completed'"):
        open_spatial_product(root)


def test_spatial_sample_origin_is_not_inferred_in_outputs(tmp_path):
    from otuformer.data.records import FileSnapshot, ImageRecord

    def record(key: str, sample_origin: str, metadata: dict[str, str]) -> ImageRecord:
        return ImageRecord(
            image_key=key,
            access_path=Path("/tmp") / key,
            output_id=key,
            selector_reference=None,
            width=640,
            height=480,
            image_file=FileSnapshot(path=Path("/tmp") / key, sha256="0" * 64, size_bytes=1),
            metadata=metadata,
            sample="plate" if sample_origin == "first_directory" else metadata.get("sample") or None,
            sample_origin=sample_origin,
        )

    samples = samples_from_records(
        [
            record("plate/one.png", "first_directory", {}),
            record("two.png", "explicit", {"sample": "specimen-7", "label": "NA"}),
            record("three.png", "explicit", {"sample": ""}),
        ]
    )
    assert samples[0].metadata == {}  # directory-derived sample suppressed
    assert samples[1].metadata == {"sample": "specimen-7", "label": "NA"}
    assert samples[2].metadata == {"sample": ""}  # explicitly empty stays empty

    root = tmp_path / "product"
    writer = SpatialWriter(
        root, contract=contract_for(), layout="per-image", samples=samples
    )
    for sample in samples:
        writer.write(make_view(sample, contract_for()))
    writer.finish()
    write_envelope(root)
    index = (root / "spatial_index.jsonl").read_text(encoding="utf-8")
    assert "first_directory" not in index
    assert "specimen-7" in index
    reader = open_spatial_product(root)
    try:
        assert reader.image_keys == ("plate/one.png", "two.png", "three.png")
    finally:
        reader.close()


def test_artifact_key_resolution_rule():
    available = ["a/one.png", "b/one.png", "c/two.png"]
    basenames: dict[str, list[str]] = {}
    for key in available:
        basenames.setdefault(key.rsplit("/", 1)[-1], []).append(key)
    keys = set(available)

    assert resolve_stored_key("a/one.png", keys, basenames) == "a/one.png"
    assert resolve_stored_key("c/two.png", keys, basenames) == "c/two.png"
    with pytest.raises(SpatialRecordError, match="ambiguous"):
        resolve_stored_key("one.png", keys, basenames)
    with pytest.raises(SpatialRecordError, match="stored key not found"):
        resolve_stored_key("d/one.png", keys, basenames)
    with pytest.raises(SpatialRecordError, match="absolute"):
        resolve_stored_key("/a/one.png", keys, basenames)
    with pytest.raises(SpatialRecordError, match="escapes"):
        resolve_stored_key("../a/one.png", keys, basenames)
    with pytest.raises(SpatialRecordError, match="stored key not found"):
        resolve_stored_key("missing.png", keys, basenames)


def test_selector_csv_rejects_duplicates_blanks_and_empty_selection(tmp_path):
    available = ("a/one.png", "c/two.png")
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("image\na/one.png\na/one.png\n", encoding="utf-8")
    with pytest.raises(SpatialRecordError, match="twice"):
        read_selector_keys(duplicate, available)

    blank = tmp_path / "blank.csv"
    blank.write_text("image\n  \n", encoding="utf-8")
    with pytest.raises(SpatialRecordError, match="blank"):
        read_selector_keys(blank, available)

    empty = tmp_path / "empty.csv"
    empty.write_text("image\n", encoding="utf-8")
    with pytest.raises(SpatialRecordError, match="selected no stored keys"):
        read_selector_keys(empty, available)

    no_column = tmp_path / "nocolumn.csv"
    no_column.write_text("file\na/one.png\n", encoding="utf-8")
    with pytest.raises(SpatialRecordError, match="needs an 'image' column"):
        read_selector_keys(no_column, available)


def _index_lines(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / "spatial_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tamper_array(root: Path, entry: dict, name: str, array: np.ndarray) -> None:
    """Rewrite one stored array and keep the descriptor/file hashes consistent."""
    from otuformer.spatial.storage import array_digest

    ref = entry["refs"][name]
    payload = root / ref["path"]
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[ref["key"]] = array
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload.write_bytes(buffer.getvalue())
    entry["arrays"][name]["digest"] = array_digest(entry["arrays"][name]["role"], array)
    manifest_path = root / FILENAME_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if item["path"] == ref["path"]:
            item["sha256"] = sha256_bytes(payload.read_bytes())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_index(root: Path, entries: list[dict]) -> None:
    payload = b"".join(
        __import__("json").dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for entry in entries
    )
    (root / "spatial_index.jsonl").write_bytes(payload)
    manifest_path = root / FILENAME_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index"]["sha256"] = sha256_bytes(payload)
    manifest["index"]["bytes"] = len(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    write_envelope(root)


def test_duplicate_logical_view_is_rejected(tmp_path):
    root = tmp_path / "product"
    build_product(root)
    entries = _index_lines(root)
    entries.append(dict(entries[0]))
    _rewrite_index(root, entries)
    with pytest.raises(SpatialRecordError, match="duplicate logical view"):
        open_spatial_product(root)


def test_unsupported_schema_and_support_versions_are_rejected(tmp_path):
    root = tmp_path / "product"
    build_product(root)
    entries = _index_lines(root)
    entries[0]["schema_version"] = 99
    _rewrite_index(root, entries)
    with pytest.raises(SpatialRecordError, match="unsupported spatial schema version"):
        open_spatial_product(root)

    root2 = tmp_path / "product2"
    build_product(root2)
    entries = _index_lines(root2)
    entries[0]["support_version"] = "guessed-v0"
    _rewrite_index(root2, entries)
    with pytest.raises(SpatialRecordError, match="unsupported support rule"):
        open_spatial_product(root2)


@pytest.mark.parametrize(
    "mutate, message",
    [
        ("feature_shape", "shape"),
        ("feature_dtype", "dtype"),
        ("feature_digest", "does not match its digest"),
        ("pixel_dtype", "dtype"),
        ("coverage_bounds", "must stay within"),
        ("valid_dtype", "dtype"),
        ("path_escape", "escapes the product root"),
        ("absolute_path", "absolute product path"),
        ("unexpected_array", "unexpected array"),
        ("layers_mismatch", "lists layers that differ"),
        ("array_object", "object array"),
    ],
)
def test_reader_rejects_tampered_or_unsafe_products(tmp_path, mutate, message):
    root = tmp_path / "product"
    # Duplicate basename fixtures ("a/one.png" and "b/one.png") exercise key identity.
    build_product(root)
    entries = _index_lines(root)
    entry = entries[0]
    if mutate == "feature_shape":
        entry["arrays"]["feature:blocks.0"]["shape"] = [2, 2, 9]
    elif mutate == "feature_dtype":
        entry["arrays"]["feature:blocks.0"]["dtype"] = "<f8"
    elif mutate == "feature_digest":
        entry["arrays"]["feature:blocks.0"]["digest"] = "0" * 64
    elif mutate == "pixel_dtype":
        entry["arrays"][PIXEL_COVERAGE]["dtype"] = "<f8"
    elif mutate == "coverage_bounds":
        tampered = np.full((32, 32), 2.0, dtype=np.float32)
        _tamper_array(root, entry, PIXEL_COVERAGE, tampered)
    elif mutate == "valid_dtype":
        entry["arrays"][PIXEL_VALID]["dtype"] = "<f4"
    elif mutate == "path_escape":
        entry["refs"]["feature:blocks.0"]["path"] = "../escape.npz"
    elif mutate == "absolute_path":
        entry["refs"]["feature:blocks.0"]["path"] = "/tmp/escape.npz"
    elif mutate == "unexpected_array":
        entry["arrays"]["surprise"] = entry["arrays"][PATCH_COVERAGE]
        entry["refs"]["surprise"] = entry["refs"][PATCH_COVERAGE]
    elif mutate == "layers_mismatch":
        entry["feature_layers"] = ["blocks.0"]
    elif mutate == "array_object":
        # Rewrite the payload with an object array under the same key.
        payload = root / entry["refs"]["feature:blocks.0"]["path"]
        with np.load(payload, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays[entry["refs"]["feature:blocks.0"]["key"]] = np.array(
            [object()], dtype=object
        )
        import io

        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        payload.write_bytes(buffer.getvalue())
        manifest_path = root / FILENAME_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            if item["path"] == entry["refs"]["feature:blocks.0"]["path"]:
                item["sha256"] = sha256_bytes(payload.read_bytes())
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        write_envelope(root)
        with pytest.raises(Exception):
            open_spatial_product(root).iter_views().__next__()
        return
    _rewrite_index(root, entries)
    with pytest.raises(SpatialRecordError, match=message):
        reader = open_spatial_product(root)
        reader.close()
        list(reader.iter_views())


def test_missing_payload_or_manifest_entry_is_corruption(tmp_path):
    root = tmp_path / "product"
    build_product(root)
    payload = next((root / "features").iterdir())
    payload.unlink()
    reader = open_spatial_product(root)
    with pytest.raises(SpatialRecordError, match="sealed payload is missing"):
        list(reader.iter_views())
    reader.close()


def test_invalid_trace_is_rejected_before_reading(tmp_path):
    """A singular map is structural: the whole index is checked at open."""
    root = tmp_path / "product"
    build_product(root)
    entries = _index_lines(root)
    entries[0]["geometry"]["source_to_view"] = np.zeros((3, 3)).tolist()
    _rewrite_index(root, entries)
    with pytest.raises(SpatialRecordError, match="singular"):
        open_spatial_product(root)


def test_writer_validates_views_before_publishing(tmp_path):
    contract = contract_for()
    sample = make_sample("a.png")
    writer = SpatialWriter(tmp_path / "p", contract=contract, layout="per-image", samples=[sample])
    bad = make_view(sample, contract)
    broken = SpatialView(
        image_key=bad.image_key,
        view_id=bad.view_id,
        role=bad.role,
        trace=bad.trace,
        features={**bad.features, "blocks.0": np.full((2, 2, 3), np.nan, dtype=np.float32)},
        pixel_geometric_coverage=bad.pixel_geometric_coverage,
        pixel_sampling_valid=bad.pixel_sampling_valid,
        geometric_coverage=bad.geometric_coverage,
        patch_sampling_valid=bad.patch_sampling_valid,
    )
    with pytest.raises(SpatialRecordError, match="non-finite"):
        writer.write(broken)

    with pytest.raises(SpatialRecordError, match="unknown view role"):
        SpatialView(
            image_key=bad.image_key,
            view_id=bad.view_id,
            role="diagnostic",
            trace=bad.trace,
            features=bad.features,
            pixel_geometric_coverage=bad.pixel_geometric_coverage,
            pixel_sampling_valid=bad.pixel_sampling_valid,
            geometric_coverage=bad.geometric_coverage,
            patch_sampling_valid=bad.patch_sampling_valid,
        )

    with pytest.raises(SpatialRecordError, match="at least its base view"):
        writer.finish()


def test_abort_removes_only_attempt_files(tmp_path):
    root = tmp_path / "product"
    root.mkdir()
    keep = root / "keep.txt"
    keep.write_text("previous attempt data", encoding="utf-8")
    contract = contract_for()
    samples = [make_sample("a.png"), make_sample("b.png")]
    writer = SpatialWriter(root, contract=contract, layout="per-image", samples=samples)
    writer.write(make_view(samples[0], contract))
    writer.write(make_view(samples[1], contract))  # flushes the first sample's NPZ
    created = sorted(path.name for path in (root / "features").iterdir())
    assert created
    removed = writer.abort({"reason": "interrupted"})
    assert removed
    assert keep.read_text(encoding="utf-8") == "previous attempt data"
    assert not (root / "features").exists()
    with pytest.raises(SpatialRecordError):
        writer.write(make_view(samples[0], contract))


def test_previous_completed_product_survives_a_refused_attempt(tmp_path):
    root = tmp_path / "product"
    build_product(root)
    before = (root / FILENAME_MANIFEST).read_bytes()
    # A second attempt into the same directory is refused before any byte moves:
    # the core writer owns an attempt directory, it never adopts a sealed product.
    contract = contract_for()
    with pytest.raises(SpatialRecordError, match="already holds a spatial product"):
        SpatialWriter(
            root,
            contract=contract,
            layout="per-image",
            samples=[make_sample("a/one.png")],
        )
    assert (root / FILENAME_MANIFEST).read_bytes() == before
    reader = open_spatial_product(root)
    try:
        assert len(list(reader.iter_views())) == 2
    finally:
        reader.close()


def test_writer_refuses_a_directory_with_foreign_payload_files(tmp_path):
    """A public ``write``/``finish``/``abort`` cycle cannot delete old payloads."""
    root = tmp_path / "product"
    (root / "features").mkdir(parents=True)
    stale = root / "features" / "000001.npz"
    stale.write_bytes(b"previous attempt")
    with pytest.raises(SpatialRecordError, match="already holds payload files"):
        SpatialWriter(
            root,
            contract=contract_for(),
            layout="per-image",
            samples=[make_sample("a.png")],
        )
    assert stale.read_bytes() == b"previous attempt"


def test_unsupported_layout_and_empty_samples_are_rejected(tmp_path):
    with pytest.raises(SpatialRecordError, match="unsupported spatial layout"):
        SpatialWriter(
            tmp_path / "p", contract=contract_for(), layout="parquet", samples=[make_sample("a")]
        )
    with pytest.raises(SpatialRecordError, match="at least one sample"):
        SpatialWriter(tmp_path / "q", contract=contract_for(), layout="per-image", samples=[])


def test_writes_reject_unknown_and_duplicated_views(tmp_path):
    contract = contract_for()
    sample = make_sample("a.png")
    writer = SpatialWriter(tmp_path / "p", contract=contract, layout="per-image", samples=[sample])
    writer.write(make_view(sample, contract))
    with pytest.raises(SpatialRecordError, match="written twice"):
        writer.write(make_view(sample, contract))
    unknown = make_sample("other.png")
    with pytest.raises(SpatialRecordError, match="does not belong"):
        writer.write(make_view(unknown, contract))


def test_product_id_is_layout_independent(tmp_path):
    first = tmp_path / "npz"
    second = tmp_path / "npz-copy"
    build_product(first)
    build_product(second)
    one = open_spatial_product(first)
    two = open_spatial_product(second)
    try:
        assert one.product_id == two.product_id
    finally:
        one.close()
        two.close()
    # Different logical content yields a different identity.
    third = tmp_path / "npz-other"
    build_product(third, views_per_key=2)
    other = open_spatial_product(third)
    try:
        assert other.product_id != one.product_id
    finally:
        other.close()


def test_v1_index_entry_fields_are_pinned(tmp_path):
    """A tiny canonical metadata fixture pins the adopted version-1 table."""
    root = tmp_path / "product"
    build_product(root, keys=("a/one.png",))
    (entry,) = _index_lines(root)
    assert set(entry) == {
        "schema_version",
        "sample_id",
        "sample",
        "view_id",
        "role",
        "appearance",
        "support_version",
        "coordinate_adapter",
        "geometry",
        "feature_layers",
        "arrays",
        "refs",
    }
    assert entry["schema_version"] == 1
    assert entry["sample_id"] == 1
    assert set(entry["sample"]) == {
        "image_key",
        "image_sha256",
        "width",
        "height",
        "metadata",
    }
    assert entry["view_id"] == "base" and entry["role"] == "base"
    assert entry["appearance"] == {"photometric": "none"}
    assert entry["support_version"] == "conservative-resampling-v1"
    assert entry["feature_layers"] == ["blocks.0", "blocks.1"]
    assert set(entry["arrays"]) == {
        "feature:blocks.0",
        "feature:blocks.1",
        PIXEL_COVERAGE,
        PIXEL_VALID,
        PATCH_COVERAGE,
        PATCH_VALID,
    }
    for name, descriptor in entry["arrays"].items():
        assert set(descriptor) == {"role", "dtype", "shape", "digest"}
        assert descriptor["digest"] and len(descriptor["digest"]) == 64
        assert descriptor["dtype"] in {"<f4", "|b1"}
    assert entry["arrays"][PIXEL_COVERAGE] == {
        "role": PIXEL_COVERAGE,
        "dtype": "<f4",
        "shape": [32, 32],
        "digest": entry["arrays"][PIXEL_COVERAGE]["digest"],
    }
    for name, ref in entry["refs"].items():
        assert set(ref) == {"kind", "path", "key"}
        assert ref["kind"] == "npz"
        assert ref["path"].startswith("features/") and ref["path"].endswith(".npz")
        assert ".." not in ref["path"]
    manifest = json.loads((root / FILENAME_MANIFEST).read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "layout",
        "feature_contract",
        "product_id",
        "samples",
        "views",
        "index",
        "files",
        "annotations",
        "evaluation_plan",
        "support_version",
        "appearance",
        "key_rule",
    }
    assert manifest["annotations"] == {"status": "absent"}
    assert manifest["evaluation_plan"] == {"status": "absent"}
    assert manifest["key_rule"] == "artifact-key-resolution-v1"
    assert manifest["samples"] == 1 and manifest["views"] == 1


def test_a_stored_plan_is_carried_and_preflighted(tmp_path):
    """The optional paired plan persists, and a planned pair needs its payload."""
    from otuformer.spatial.protocol import (
        PROTOCOL_VIEW_IDS,
        build_evaluation_plan,
        protocol_operations,
    )
    from otuformer.vision.geometry import apply_geometry, evaluation_view_ops

    root = tmp_path / "product"
    feature_contract = contract_for()
    samples = [make_sample("a.png", width=64, height=64)]
    writer = SpatialWriter(
        root, contract=feature_contract, layout="per-image", samples=samples
    )
    image = Image.new("RGB", (64, 64), (30, 60, 90))
    base_ops = evaluation_view_ops(image, name="whole-specimen-pad", size=32)
    stored = []
    for view_id in PROTOCOL_VIEW_IDS:
        if view_id == "reference":
            ops = base_ops
            stored_id, role = "base", "base"
        else:
            ops = [*base_ops, *protocol_operations(32, view_id)]
            stored_id, role = view_id, "eval-transform"
        _, trace = apply_geometry(image, ops, track=True)
        view = make_view(samples[0], feature_contract, view_id=stored_id, role=role)
        view = SpatialView(
            image_key=view.image_key,
            view_id=stored_id,
            role=role,
            trace=trace,
            features=view.features,
            pixel_geometric_coverage=view.pixel_geometric_coverage,
            pixel_sampling_valid=view.pixel_sampling_valid,
            geometric_coverage=view.geometric_coverage,
            patch_sampling_valid=view.patch_sampling_valid,
        )
        writer.write(view)
        stored.append(view)
    plan = build_evaluation_plan({"a.png": stored}, seed=42)
    writer.set_evaluation_plan(plan)
    manifest = writer.finish()
    write_envelope(root)

    assert manifest["evaluation_plan"]["status"] == "present"
    reader = open_spatial_product(root)
    try:
        assert reader.evaluation_plan is not None
        assert reader.evaluation_plan["protocol"] == "spatial-geometry-v1"
        assert reader.evaluation_plan["selected_keys"] == ["a.png"]
        assert len(reader.evaluation_plan["pairs"]) == len(PROTOCOL_VIEW_IDS) - 1
        assert len(reader.view_ids("a.png")) == len(PROTOCOL_VIEW_IDS)
        # The base view is the reference alias: no second saved copy.
        assert reader.view_ids("a.png")[0] == "base"
    finally:
        reader.close()

    # A planned view that was never stored is corruption, not a normal subset: the
    # index and its count are consistent, but the plan references a missing payload.
    second = tmp_path / "truncated"
    truncated_writer = SpatialWriter(
        second, contract=feature_contract, layout="per-image", samples=samples
    )
    for view in stored:
        if view.view_id == "rot90":
            continue
        truncated_writer.write(view)
    truncated_writer.set_evaluation_plan(plan)
    truncated_writer.finish()
    write_envelope(second)
    with pytest.raises(SpatialRecordError, match="missing stored view"):
        open_spatial_product(second)


def test_a_plan_with_a_foreign_protocol_is_rejected(tmp_path):
    from otuformer.spatial.protocol import PROTOCOL_VIEW_IDS, build_evaluation_plan, protocol_operations
    from otuformer.vision.geometry import apply_geometry, evaluation_view_ops

    root = tmp_path / "product"
    feature_contract = contract_for()
    samples = [make_sample("a.png", width=64, height=64)]
    writer = SpatialWriter(
        root, contract=feature_contract, layout="per-image", samples=samples
    )
    image = Image.new("RGB", (64, 64), (30, 60, 90))
    base_ops = evaluation_view_ops(image, name="whole-specimen-pad", size=32)
    stored = []
    for view_id in PROTOCOL_VIEW_IDS:
        if view_id == "reference":
            ops, stored_id, role = base_ops, "base", "base"
        else:
            ops = [*base_ops, *protocol_operations(32, view_id)]
            stored_id, role = view_id, "eval-transform"
        _, trace = apply_geometry(image, ops, track=True)
        template = make_view(samples[0], feature_contract, view_id=stored_id, role=role)
        view = SpatialView(
            image_key=template.image_key,
            view_id=stored_id,
            role=role,
            trace=trace,
            features=template.features,
            pixel_geometric_coverage=template.pixel_geometric_coverage,
            pixel_sampling_valid=template.pixel_sampling_valid,
            geometric_coverage=template.geometric_coverage,
            patch_sampling_valid=template.patch_sampling_valid,
        )
        writer.write(view)
        stored.append(view)
    writer.set_evaluation_plan(build_evaluation_plan({"a.png": stored}, seed=42))
    writer.finish()

    from otuformer.data.provenance import canonical_json_bytes

    plan_path = root / "spatial_evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["protocol"] = "some-other-protocol"
    payload = canonical_json_bytes(plan)
    plan_path.write_bytes(payload)
    manifest_path = root / FILENAME_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_plan"]["sha256"] = sha256_bytes(payload)
    # The manifest inventories the plan file too, so a consistent rewrite has to
    # update both digests; only then is the protocol the next check to fail.
    for item in manifest["files"]:
        if item["path"] == "spatial_evaluation_plan.json":
            item["sha256"] = sha256_bytes(payload)
            item["bytes"] = len(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    write_envelope(root)
    with pytest.raises(SpatialRecordError, match="is not 'spatial-geometry-v1'"):
        open_spatial_product(root)


def _rewrite_manifest(root: Path) -> dict:
    path = root / FILENAME_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def _store_manifest(root: Path, manifest: dict) -> None:
    (root / FILENAME_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    write_envelope(root)


def test_annotation_snapshot_outside_the_product_root_is_refused(tmp_path):
    """An ``annotations.path`` outside the inventory/root is a trust-boundary break."""
    root = tmp_path / "product"
    build_product(
        root,
        annotations={
            "records": [{"annotation_id": 1, "image_key": "a/one.png", "category_id": 4}],
            "categories": {"4": {"name": "Head"}},
            "sources": {},
        },
    )
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"annotation_id": 9}\n', encoding="utf-8")
    manifest = _rewrite_manifest(root)
    # A digest consistent with the outside file is not enough: the path escapes.
    manifest["annotations"]["path"] = "../outside.jsonl"
    manifest["annotations"]["sha256"] = sha256_bytes(outside.read_bytes())
    _store_manifest(root, manifest)
    with pytest.raises(SpatialRecordError, match="escapes the product root"):
        open_spatial_product(root)


def test_annotation_snapshot_must_be_in_the_manifest_inventory(tmp_path):
    root = tmp_path / "product"
    build_product(
        root,
        annotations={
            "records": [{"annotation_id": 1, "image_key": "a/one.png", "category_id": 4}],
            "categories": {},
            "sources": {},
        },
    )
    # A second, untouched copy inside the root but absent from files[].
    spare = root / "spare_annotations.jsonl"
    spare.write_bytes((root / "annotations.jsonl").read_bytes())
    manifest = _rewrite_manifest(root)
    manifest["annotations"]["path"] = "spare_annotations.jsonl"
    _store_manifest(root, manifest)
    with pytest.raises(SpatialRecordError, match="not sealed by the manifest inventory"):
        open_spatial_product(root)


def test_a_plan_with_out_of_bounds_or_unknown_queries_is_refused(tmp_path):
    """Structural plan validation happens at open, before any evaluation."""
    from otuformer.spatial.protocol import (
        PROTOCOL_VIEW_IDS,
        build_evaluation_plan,
        protocol_operations,
    )
    from otuformer.vision.geometry import apply_geometry, evaluation_view_ops

    root = tmp_path / "product"
    contract = contract_for()
    sample = make_sample("a.png")
    # A complete protocol view set, so the reader can resolve each pair.
    base_ops = evaluation_view_ops(Image.new("RGB", (640, 480)), name="center-crop", size=32)
    from otuformer.vision.geometry import apply_geometry

    image = Image.new("RGB", (640, 480), (10, 20, 30))
    writer = SpatialWriter(root, contract=contract, layout="per-image", samples=[sample])
    stored = []
    for view_id in PROTOCOL_VIEW_IDS:
        if view_id == "reference":
            ops, role = base_ops, "base"
            stored_id = "base"
        else:
            ops = [*base_ops, *protocol_operations(32, view_id)]
            stored_id, role = view_id, "eval-transform"
        _, trace = apply_geometry(image, ops, track=True)
        template = make_view(sample, contract, view_id=stored_id, role=role)
        view = SpatialView(
            image_key=template.image_key,
            view_id=stored_id,
            role=role,
            trace=trace,
            features=template.features,
            pixel_geometric_coverage=template.pixel_geometric_coverage,
            pixel_sampling_valid=template.pixel_sampling_valid,
            geometric_coverage=template.geometric_coverage,
            patch_sampling_valid=template.patch_sampling_valid,
        )
        writer.write(view)
        stored.append(view)
    writer.set_evaluation_plan(build_evaluation_plan({"a.png": stored}, seed=42))
    writer.finish()
    write_envelope(root)

    plan_path = root / "spatial_evaluation_plan.json"
    original = json.loads(plan_path.read_text(encoding="utf-8"))
    from otuformer.data.provenance import canonical_json_bytes

    grid = contract.layers[0].grid_hw
    cases = (
        ("query_bounds", grid[0] + 8, 0, None),
        ("query_direction", 0, 0, "sideways"),
    )
    for _name, row, column, direction in cases:
        plan = json.loads(json.dumps(original))
        pair = plan["pairs"][0]
        branch = pair["directions"]["reference_to_target"]
        branch["queries"] = [
            {
                "row": row,
                "column": column,
                "xy_source": [1.0, 1.0],
                "xy_target": [2.0, 2.0],
            }
        ]
        # Keep the counts consistent with the stored support so the bounds check is
        # the next failure, not a count mismatch.
        valid = int(
            original["pairs"][0]["directions"]["reference_to_target"][
                "source_valid_count"
            ]
        )
        branch["sampled_count"] = 1
        branch["eligible_count"] = 1
        branch["source_valid_count"] = valid
        branch["unavailable"] = False
        branch["unavailable_reason"] = None
        if direction is not None:
            branch = pair["directions"].pop("reference_to_target")
            pair["directions"][direction] = branch
        payload = canonical_json_bytes(plan)
        plan_path.write_bytes(payload)
        manifest = _rewrite_manifest(root)
        manifest["evaluation_plan"]["sha256"] = sha256_bytes(payload)
        manifest["evaluation_plan"]["bytes"] = len(payload)
        for item in manifest["files"]:
            if item["path"] == "spatial_evaluation_plan.json":
                item["sha256"] = sha256_bytes(payload)
                item["bytes"] = len(payload)
        _store_manifest(root, manifest)
        expected = (
            "outside the stored"
            if direction is None
            else "must hold exactly the directions"
        )
        with pytest.raises(SpatialRecordError, match=expected):
            open_spatial_product(root)


def test_open_payload_handles_are_bounded(tmp_path):
    """A full read must not keep one descriptor per payload file open."""
    root = tmp_path / "product"
    build_product(root, keys=("a.png", "b.png", "c.png", "d.png", "e.png", "f.png"))
    reader = SpatialReader(root, _staging=True, open_payload_limit=2)
    try:
        for view in reader.iter_views():
            assert view.features
            assert len(reader._open_npz) <= 2
    finally:
        reader.close()
    assert not reader._open_npz


def test_a_verified_payload_is_not_rehashed_when_its_handle_is_reopened(tmp_path):
    """Verification is per file identity, not per open: the cache is bounded.

    With one payload per sample and a smaller handle budget, a full read evicts
    and reopens the same archives. Re-reading each file for its SHA-256 made the
    hash work scale with reads instead of with files.
    """
    from otuformer.spatial import storage as storage_module

    root = tmp_path / "product"
    keys = ("a.png", "b.png", "c.png", "d.png", "e.png", "f.png")
    build_product(root, keys=keys)
    reader = SpatialReader(root, _staging=True, open_payload_limit=2)
    try:
        payload_bytes = {
            (root / item["path"]).stat().st_size
            for item in reader.manifest["files"]
            if item["path"].startswith("features/")
        }
        hashed: list[int] = []
        original = storage_module.sha256_bytes

        def counting(data: bytes) -> str:
            hashed.append(len(data))
            return original(data)

        storage_module.sha256_bytes = counting
        try:
            views = list(reader.iter_views())
        finally:
            storage_module.sha256_bytes = original
    finally:
        reader.close()
    assert len(views) == len(keys)
    # One hash per payload file, not one per (evicted and reopened) handle.
    assert len([size for size in hashed if size in payload_bytes]) == len(keys)


def test_changed_payload_identity_is_checked_even_on_a_cached_handle(tmp_path):
    root = tmp_path / "product"
    build_product(root, keys=("a.png",))
    with SpatialReader(root, _staging=True) as reader:
        reader.get_view("a.png", "base")
        relative = next(iter(reader._open_npz))
        # A ZIP permits trailing bytes, so array digests alone cannot detect this.
        with (root / relative).open("ab") as handle:
            handle.write(b"changed sealed trailer")
        with pytest.raises(SpatialRecordError, match="payload bytes do not match"):
            reader.get_view("a.png", "base")


def test_selector_overlay_reports_labels_without_rewriting_the_product(tmp_path):
    root = tmp_path / "product"
    build_product(root)
    stored_before = (root / "spatial_index.jsonl").read_bytes()
    selector = tmp_path / "labels.csv"
    selector.write_text(
        "image,label,site\nb/one.png,beetle,lab\na/one.png,,\n", encoding="utf-8"
    )
    reader = open_spatial_product(root, selector_csv=selector)
    try:
        assert reader.image_keys == ("b/one.png", "a/one.png")
        assert reader.sample_metadata("b/one.png") == {"label": "beetle", "site": "lab"}
        # Empty CSV values keep the existing missing-value semantics.
        assert reader.sample_metadata("a/one.png") == {}
        report = reader.overlay_report()
        assert report["status"] == "present"
        assert report["fields"] == ["label", "site"]
        assert report["sha256"] == sha256_bytes(selector.read_bytes())
        assert report["samples"]["b/one.png"]["label"] == "beetle"
    finally:
        reader.close()
    assert (root / "spatial_index.jsonl").read_bytes() == stored_before


def test_selector_csv_rejects_a_repeated_column(tmp_path):
    root = tmp_path / "product"
    build_product(root)
    selector = tmp_path / "labels.csv"
    selector.write_text("image,label,label\nb/one.png,x,y\n", encoding="utf-8")
    with pytest.raises(SpatialRecordError, match="repeats a column name"):
        open_spatial_product(root, selector_csv=selector)


def test_product_id_covers_the_evaluation_plan(tmp_path):
    from otuformer.spatial.protocol import (
        PROTOCOL_VIEW_IDS,
        build_evaluation_plan,
        protocol_operations,
    )
    from otuformer.vision.geometry import apply_geometry, evaluation_view_ops

    def build(root: Path, seed: int) -> str:
        contract = contract_for()
        sample = make_sample("a.png")
        image = Image.new("RGB", (640, 480), (10, 20, 30))
        from otuformer.vision.geometry import apply_geometry

        base_ops = evaluation_view_ops(image, name="center-crop", size=32)
        writer = SpatialWriter(root, contract=contract, layout="per-image", samples=[sample])
        stored = []
        for view_id in PROTOCOL_VIEW_IDS:
            if view_id == "reference":
                ops, role, stored_id = base_ops, "base", "base"
            else:
                ops = [*base_ops, *protocol_operations(32, view_id)]
                role, stored_id = "eval-transform", view_id
            _, trace = apply_geometry(image, ops, track=True)
            template = make_view(sample, contract, view_id=stored_id, role=role)
            view = SpatialView(
                image_key=sample.image_key,
                view_id=stored_id,
                role=role,
                trace=trace,
                features=template.features,
                pixel_geometric_coverage=template.pixel_geometric_coverage,
                pixel_sampling_valid=template.pixel_sampling_valid,
                geometric_coverage=template.geometric_coverage,
                patch_sampling_valid=template.patch_sampling_valid,
            )
            writer.write(view)
            stored.append(view)
        writer.set_evaluation_plan(
            build_evaluation_plan({"a.png": stored}, seed=seed)
        )
        return writer.finish()["product_id"]

    first = build(tmp_path / "one", 42)
    same = build(tmp_path / "two", 42)
    other = build(tmp_path / "three", 7)
    assert first == same
    # The same views under a different evaluation contract are not one product.
    assert other != first


def test_writer_refuses_symlinked_owned_directories(tmp_path):
    """A symlinked ``features`` would send payloads outside the product root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "product"
    root.mkdir()
    (root / "features").symlink_to(outside)
    contract = contract_for()
    with pytest.raises(SpatialRecordError, match="symbolic link"):
        SpatialWriter(
            root, contract=contract, layout="per-image", samples=[make_sample("a.png")]
        )
    assert not list(outside.iterdir())

    # A dangling target is refused the same way, not silently created elsewhere.
    dangling = tmp_path / "dangling"
    dangling.mkdir()
    (dangling / "features").symlink_to(tmp_path / "missing-directory")
    with pytest.raises(SpatialRecordError, match="symbolic link"):
        SpatialWriter(
            dangling, contract=contract, layout="per-image", samples=[make_sample("a.png")]
        )


def test_writer_refuses_a_symlinked_output_directory(tmp_path):
    link = tmp_path / "dangling-output"
    link.symlink_to(tmp_path / "nowhere")
    with pytest.raises(SpatialRecordError, match="dangling symlink"):
        SpatialWriter(
            link, contract=contract_for(), layout="per-image", samples=[make_sample("a.png")]
        )


def test_product_id_covers_annotation_content(tmp_path):
    def build(annotations) -> str:
        contract = contract_for()
        sample = make_sample("a.png", width=2, height=2)
        writer = SpatialWriter(
            contract=contract,
            out_dir=tmp_path / f"p{len(list(tmp_path.iterdir()))}",
            layout="per-image",
            samples=[sample],
            annotations=annotations,
        )
        writer.write(make_view(sample, contract))
        return writer.finish()["product_id"]

    base = {
        "records": [
            {
                "annotation_id": 1,
                "image_key": "a.png",
                "category_id": 4,
                "role": "instances",
                "annotation": {"id": 1, "category_id": 4, "bbox": [0, 0, 2, 2]},
            }
        ],
        "categories": {"instances:4": {"name": "Head"}},
        "sources": {"instances": {"sha256": "abc", "images": 1}},
    }
    first = build(base)
    assert build(base) == first
    assert build({**base, "categories": {"instances:4": {"name": "Body"}}}) != first
    assert (
        build(
            {
                **base,
                "records": [
                    {**base["records"][0], "annotation_id": 2}
                ],
            }
        )
        != first
    )
    raster = np.zeros((2, 2), dtype=np.uint8)
    with_raster = build({**base, "rasters": {"a.png": raster}})
    assert with_raster != first
    assert build({**base, "rasters": {"a.png": raster + 1}}) != with_raster


def test_plan_validation_rejects_incomplete_or_unbounded_plans(tmp_path):
    from otuformer.data.provenance import canonical_json_bytes
    from otuformer.spatial.protocol import (
        PROTOCOL_VIEW_IDS,
        build_evaluation_plan,
        protocol_operations,
    )
    from otuformer.vision.geometry import apply_geometry, evaluation_view_ops

    root = tmp_path / "product"
    contract = contract_for()
    sample = make_sample("a.png")
    image = Image.new("RGB", (640, 480), (10, 20, 30))
    base_ops = evaluation_view_ops(image, name="center-crop", size=32)
    writer = SpatialWriter(root, contract=contract, layout="per-image", samples=[sample])
    stored = []
    for view_id in PROTOCOL_VIEW_IDS:
        if view_id == "reference":
            ops, role, stored_id = base_ops, "base", "base"
        else:
            ops = [*base_ops, *protocol_operations(32, view_id)]
            role, stored_id = "eval-transform", view_id
        _, trace = apply_geometry(image, ops, track=True)
        template = make_view(sample, contract, view_id=stored_id, role=role)
        view = SpatialView(
            image_key=sample.image_key,
            view_id=stored_id,
            role=role,
            trace=trace,
            features=template.features,
            pixel_geometric_coverage=template.pixel_geometric_coverage,
            pixel_sampling_valid=template.pixel_sampling_valid,
            geometric_coverage=template.geometric_coverage,
            patch_sampling_valid=template.patch_sampling_valid,
        )
        writer.write(view)
        stored.append(view)
    writer.set_evaluation_plan(build_evaluation_plan({"a.png": stored}, seed=42))
    writer.finish()
    write_envelope(root)
    original = json.loads((root / "spatial_evaluation_plan.json").read_text(encoding="utf-8"))

    def store(plan: dict) -> None:
        payload = canonical_json_bytes(plan)
        (root / "spatial_evaluation_plan.json").write_bytes(payload)
        manifest = _rewrite_manifest(root)
        manifest["evaluation_plan"]["sha256"] = sha256_bytes(payload)
        manifest["evaluation_plan"]["bytes"] = len(payload)
        for item in manifest["files"]:
            if item["path"] == "spatial_evaluation_plan.json":
                item["sha256"] = sha256_bytes(payload)
                item["bytes"] = len(payload)
        _store_manifest(root, manifest)

    def edit(mutate) -> dict:
        plan = json.loads(json.dumps(original))
        mutate(plan)
        return plan

    cases = (
        ("missing direction", lambda p: p["pairs"][0]["directions"].pop("target_to_reference"), "exactly the directions"),
        ("no pairs", lambda p: p.update({"pairs": []}), "holds no pairs"),
        ("query limit above the protocol", lambda p: p.update({"query_limit": 999}), "outside the frozen protocol bound"),
        ("image limit above the protocol", lambda p: p.update({"image_limit": 1000}), "outside the frozen protocol bound"),
        ("missing seed", lambda p: p.pop("seed"), "missing required field"),
        ("missing patch_hw", lambda p: p["pairs"][0].pop("patch_hw"), "missing required field"),
        ("missing query list", lambda p: p["pairs"][0]["directions"]["reference_to_target"].pop("queries"), "no query list"),
        ("pair for an unselected key", lambda p: p["pairs"][0].update({"image_key": "ghost.png"}), "does not select"),
    )
    for _name, mutate, message in cases:
        store(edit(mutate))
        with pytest.raises(SpatialRecordError, match=message):
            open_spatial_product(root)
    store(original)
    reader = open_spatial_product(root)
    reader.close()


def test_dangling_symlink_cannot_redirect_a_sealed_file(tmp_path):
    """Every sealed write goes through ``_claim_path``, dangling links included."""
    outside = tmp_path / "outside"
    outside.mkdir()
    base = {
        "records": [{"annotation_id": 1, "image_key": "a.png", "category_id": 4}],
        "categories": {"instances:4": {"name": "Head"}},
        "sources": {},
    }
    plan = {"protocol": "spatial-geometry-v1"}
    for name in (
        "spatial_index.jsonl",
        "spatial_manifest.json",
        "annotations.jsonl",
        "spatial_evaluation_plan.json",
    ):
        root = tmp_path / f"p-{name}"
        root.mkdir()
        (root / name).symlink_to(outside / f"{name}.target")
        contract = contract_for()
        sample = make_sample("a.png")
        if name in ("spatial_index.jsonl", "spatial_manifest.json"):
            # A sealed name is refused at construction; the dangling link counts.
            with pytest.raises(SpatialRecordError, match="already holds"):
                SpatialWriter(
                    root,
                    contract=contract,
                    layout="per-image",
                    samples=[sample],
                    annotations=base,
                    evaluation_plan=plan,
                )
            continue
        writer = SpatialWriter(
            root,
            contract=contract,
            layout="per-image",
            samples=[sample],
            annotations=base,
            evaluation_plan=plan,
        )
        writer.write(make_view(sample, contract))
        with pytest.raises(SpatialRecordError, match="symbolic link"):
            writer.finish()
    # Nothing was created outside a product root.
    assert not list(outside.iterdir())


def test_categorical_raster_must_match_the_sample_and_hold_integer_ids(tmp_path):
    contract = contract_for()

    def attempt(raster, *, width=2, height=2, key="a.png"):
        sample = make_sample(key, width=width, height=height)
        writer = SpatialWriter(
            tmp_path / f"p-{len(list(tmp_path.iterdir()))}",
            contract=contract,
            layout="per-image",
            samples=[sample],
            annotations={
                "records": [
                    {"annotation_id": 1, "image_key": key, "category_id": 4, "role": "panoptic"}
                ],
                "categories": {"panoptic:4": {"name": "Head"}},
                "sources": {},
                "rasters": {key: raster},
            },
        )
        writer.write(make_view(sample, contract))
        return writer.finish()

    # A 16x16 sample cannot carry a 1x1 raster.
    with pytest.raises(SpatialRecordError, match="but the sample is"):
        attempt(np.zeros((1, 1), dtype=np.uint8), width=16, height=16)
    # A float raster is not a categorical raster, whatever its shape.
    with pytest.raises(SpatialRecordError, match="must hold integer ids"):
        attempt(np.zeros((2, 2), dtype=np.float32))
    # NaN in a float raster is refused before it can be called a valid label.
    with pytest.raises(SpatialRecordError, match="must hold integer ids"):
        attempt(np.full((2, 2), np.nan, dtype=np.float32))
    # Negative ids are not categories either.
    with pytest.raises(SpatialRecordError, match="negative id"):
        attempt(np.full((2, 2), -1, dtype=np.int32))
    # A one-dimensional array is not a raster.
    with pytest.raises(SpatialRecordError, match="must be 2-D"):
        attempt(np.zeros((2,), dtype=np.uint8))
    # A well-formed raster still seals.
    manifest = attempt(np.zeros((2, 2), dtype=np.uint8))
    assert manifest["product_id"]


def test_reader_rejects_a_hand_built_raster_that_disagrees_with_the_manifest(tmp_path):
    from otuformer.data.provenance import canonical_json_bytes

    root = tmp_path / "product"
    contract = contract_for()
    sample = make_sample("a.png", width=2, height=2)
    writer = SpatialWriter(
        root,
        contract=contract,
        layout="per-image",
        samples=[sample],
        annotations={
            "records": [
                {"annotation_id": 1, "image_key": "a.png", "category_id": 4, "role": "panoptic"}
            ],
            "categories": {"panoptic:4": {"name": "Head"}},
            "sources": {},
            "rasters": {"a.png": np.zeros((2, 2), dtype=np.uint8)},
        },
    )
    writer.write(make_view(sample, contract))
    writer.finish()

    # Replace the sealed raster with a float one and re-seal every digest, so only
    # the semantic check can catch it.
    import io as _io

    import numpy as _np

    relative = "annotations/000001.npz"
    buffer = _io.BytesIO()
    _np.savez(buffer, raster=_np.full((2, 2), _np.nan, dtype=_np.float32))
    payload = buffer.getvalue()
    (root / relative).write_bytes(payload)
    manifest = _rewrite_manifest(root)
    manifest["annotations"]["rasters"]["a.png"]["sha256"] = sha256_bytes(payload)
    manifest["annotations"]["rasters"]["a.png"]["dtype"] = "<f4"
    for item in manifest["files"]:
        if item["path"] == relative:
            item["sha256"] = sha256_bytes(payload)
            item["bytes"] = len(payload)
    _store_manifest(root, manifest)
    reader = open_spatial_product(root)
    try:
        with pytest.raises(SpatialRecordError, match="must hold integer ids"):
            reader.annotations_for_view("a.png", "base")
    finally:
        reader.close()


def test_reader_rejects_a_raster_whose_declared_shape_disagrees(tmp_path):
    root = tmp_path / "product"
    contract = contract_for()
    sample = make_sample("a.png", width=2, height=2)
    writer = SpatialWriter(
        root,
        contract=contract,
        layout="per-image",
        samples=[sample],
        annotations={
            "records": [
                {"annotation_id": 1, "image_key": "a.png", "category_id": 4, "role": "panoptic"}
            ],
            "categories": {"panoptic:4": {"name": "Head"}},
            "sources": {},
            "rasters": {"a.png": np.zeros((2, 2), dtype=np.uint8)},
        },
    )
    writer.write(make_view(sample, contract))
    writer.finish()
    manifest = _rewrite_manifest(root)
    manifest["annotations"]["rasters"]["a.png"]["shape"] = [4, 4]
    _store_manifest(root, manifest)
    reader = open_spatial_product(root)
    try:
        with pytest.raises(SpatialRecordError, match="manifest (declares|seals)"):
            reader.annotations_for_view("a.png", "base")
    finally:
        reader.close()


def test_full_plan_contract_is_checked_not_only_field_presence(tmp_path):
    """Duplicate pairs, dropped transforms, bad ids and stale geometry all fail."""
    from otuformer.data.provenance import canonical_json_bytes
    from otuformer.spatial.protocol import (
        PROTOCOL_VIEW_IDS,
        build_evaluation_plan,
        protocol_operations,
    )
    from otuformer.vision.geometry import apply_geometry, evaluation_view_ops

    root = tmp_path / "product"
    contract = contract_for()
    sample = make_sample("a.png")
    image = Image.new("RGB", (640, 480), (10, 20, 30))
    base_ops = evaluation_view_ops(image, name="center-crop", size=32)
    writer = SpatialWriter(root, contract=contract, layout="per-image", samples=[sample])
    stored = []
    for view_id in PROTOCOL_VIEW_IDS:
        if view_id == "reference":
            ops, role, stored_id = base_ops, "base", "base"
        else:
            ops = [*base_ops, *protocol_operations(32, view_id)]
            role, stored_id = "eval-transform", view_id
        _, trace = apply_geometry(image, ops, track=True)
        template = make_view(sample, contract, view_id=stored_id, role=role)
        view = SpatialView(
            image_key=sample.image_key,
            view_id=stored_id,
            role=role,
            trace=trace,
            features=template.features,
            pixel_geometric_coverage=template.pixel_geometric_coverage,
            pixel_sampling_valid=template.pixel_sampling_valid,
            geometric_coverage=template.geometric_coverage,
            patch_sampling_valid=template.patch_sampling_valid,
        )
        writer.write(view)
        stored.append(view)
    writer.set_evaluation_plan(build_evaluation_plan({"a.png": stored}, seed=42))
    writer.finish()
    write_envelope(root)
    original = json.loads((root / "spatial_evaluation_plan.json").read_text(encoding="utf-8"))

    def store(plan: dict) -> None:
        payload = canonical_json_bytes(plan)
        (root / "spatial_evaluation_plan.json").write_bytes(payload)
        manifest = _rewrite_manifest(root)
        manifest["evaluation_plan"]["sha256"] = sha256_bytes(payload)
        manifest["evaluation_plan"]["bytes"] = len(payload)
        for item in manifest["files"]:
            if item["path"] == "spatial_evaluation_plan.json":
                item["sha256"] = sha256_bytes(payload)
                item["bytes"] = len(payload)
        _store_manifest(root, manifest)

    def edit(mutate) -> dict:
        plan = json.loads(json.dumps(original))
        mutate(plan)
        return plan

    def duplicate_pair(plan):
        plan["pairs"].append(json.loads(json.dumps(plan["pairs"][0])))

    def drop_transforms(plan):
        plan["pairs"] = plan["pairs"][:1]

    def zero_patch(plan):
        plan["pairs"][0]["patch_hw"] = [0, 0]

    def drop_geometry(plan):
        plan["pairs"][0].pop("geometry")

    def string_seed(plan):
        plan["seed"] = "forty-two"

    def bad_branch_view(plan):
        plan["pairs"][0]["directions"]["reference_to_target"]["source_view"] = "ghost"

    def pair_id_mismatch(plan):
        plan["pairs"][0]["pair_id"] = "not-the-target"

    def stale_geometry(plan):
        plan["pairs"][0]["geometry"]["target"]["view_wh"] = [99, 99]

    def wrong_patch_size(plan):
        plan["pairs"][0]["patch_hw"] = [1, 1]
        plan["pairs"][0]["geometry"]["reference"]["patch_hw"] = [1, 1]
        plan["pairs"][0]["geometry"]["target"]["patch_hw"] = [1, 1]

    def arbitrary_truth(plan):
        for pair in plan["pairs"]:
            queries = pair["directions"]["reference_to_target"]["queries"]
            if queries:
                queries[0]["xy_target"] = [999.0, 999.0]
                return

    def stale_matrix(plan):
        plan["pairs"][0]["geometry"]["target"]["source_to_view"][0][2] = 3.0

    def shrunk_protocol(plan):
        plan["view_ids"] = ["reference", "tx10"]
        plan["pairs"] = [
            pair for pair in plan["pairs"] if pair["target_view"] == "tx10"
        ]

    cases = (
        (duplicate_pair, "twice"),
        (drop_transforms, "missing transform"),
        (zero_patch, "positive integer patch_hw"),
        (drop_geometry, "missing required field"),
        (string_seed, "must be an integer"),
        (bad_branch_view, "names source_view"),
        (pair_id_mismatch, "does not match its target"),
        (stale_geometry, "does not match the stored view"),
        (wrong_patch_size, "stored feature contract is"),
        (arbitrary_truth, "stored geometry maps its source point"),
        (stale_matrix, "does not match the stored view"),
        (shrunk_protocol, "do not hold the frozen protocol views"),
    )
    for mutate, message in cases:
        store(edit(mutate))
        with pytest.raises(SpatialRecordError, match=message):
            open_spatial_product(root)
    store(original)
    reader = open_spatial_product(root)
    reader.close()


def test_wide_integer_category_ids_round_trip_exactly(tmp_path):
    """The backend works in int32; ids wider than that must not be truncated."""
    from otuformer.data.provenance import canonical_json_bytes

    values = (
        (np.uint64, 4294967297),
        (np.int64, 2147483648),
        (np.uint32, 16777215),
        (np.uint8, 255),
    )
    for dtype, value in values:
        root = tmp_path / f"p-{dtype.__name__}-{value}"
        contract = contract_for()
        sample = make_sample("a.png", width=2, height=2)
        raster = np.full((2, 2), value, dtype=dtype)
        writer = SpatialWriter(
            root,
            contract=contract,
            layout="per-image",
            samples=[sample],
            annotations={
                "records": [
                    {
                        "annotation_id": 1,
                        "image_key": "a.png",
                        "category_id": 4,
                        "role": "panoptic",
                    }
                ],
                "categories": {"panoptic:4": {"name": "Head"}},
                "sources": {},
                "rasters": {"a.png": raster},
            },
        )
        writer.write(make_view(sample, contract))
        writer.finish()
        write_envelope(root)
        reader = open_spatial_product(root)
        try:
            (record,) = reader.annotations_for_view("a.png", "base")
            assert set(np.unique(record["mask"]).tolist()) == {value}, dtype
            assert record["mask_support"].all(), dtype
        finally:
            reader.close()


def test_a_lying_npy_header_is_rejected_before_allocation(tmp_path):
    """A sealed 2.6 KB product must not become a terabyte allocation request."""
    import io as _io
    import struct
    import zipfile

    root = tmp_path / "product"
    build_product(root, keys=("a/one.png",))
    payload_path = root / "features" / "000001.npz"
    with np.load(payload_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    key = "v0000.feature:blocks.0"
    declared_shape = [int(v) for v in arrays[key].shape]

    buffer = _io.BytesIO()
    np.savez(buffer, **arrays)
    out = _io.BytesIO()
    with zipfile.ZipFile(_io.BytesIO(buffer.getvalue())) as source, zipfile.ZipFile(
        out, "w"
    ) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == f"{key}.npy":
                header_length = struct.unpack("<H", data[8:10])[0]
                body = data[10 + header_length :]
                header = (
                    "{'descr': '<f4', 'fortran_order': False, "
                    "'shape': (1000000, 1000000, 4), }"
                )
                padding = 64 - ((10 + len(header) + 1) % 64)
                header = header + " " * padding + "\n"
                data = (
                    data[:8]
                    + struct.pack("<H", len(header))
                    + header.encode("latin1")
                    + body
                )
            target.writestr(item, data)
    payload = out.getvalue()
    payload_path.write_bytes(payload)
    manifest = _rewrite_manifest(root)
    for item in manifest["files"]:
        if item["path"] == "features/000001.npz":
            item["sha256"] = sha256_bytes(payload)
            item["bytes"] = len(payload)
    _store_manifest(root, manifest)

    reader = open_spatial_product(root)
    try:
        with pytest.raises(SpatialRecordError, match="declares|seals"):
            reader.get_view("a/one.png", "base")
        # The index still describes the honest array.
        assert declared_shape
    finally:
        reader.close()


def test_annotation_raster_header_is_checked_before_allocation(tmp_path):
    """A tiny archive must not become a huge allocation through its header."""
    import io as _io
    import struct
    import zipfile

    root = tmp_path / "product"
    contract = contract_for()
    sample = make_sample("a.png", width=2, height=2)
    writer = SpatialWriter(
        root,
        contract=contract,
        layout="per-image",
        samples=[sample],
        annotations={
            "records": [
                {
                    "annotation_id": 1,
                    "image_key": "a.png",
                    "category_id": 4,
                    "role": "panoptic",
                }
            ],
            "categories": {"panoptic:4": {"name": "Head"}},
            "sources": {},
            "rasters": {"a.png": np.zeros((2, 2), dtype=np.uint8)},
        },
    )
    writer.write(make_view(sample, contract))
    writer.finish()
    relative = "annotations/000001.npz"
    header = "{'descr': '|u1', 'fortran_order': False, 'shape': (1000000, 4000000), }"
    padding = 64 - ((10 + len(header) + 1) % 64)
    header = header + " " * padding + "\n"
    npy = (
        b"\x93NUMPY"
        + bytes([1, 0])
        + struct.pack("<H", len(header))
        + header.encode("latin1")
        + b"\x00\x00\x00\x00"
    )
    buffer = _io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("raster.npy", npy)
    payload = buffer.getvalue()
    (root / relative).write_bytes(payload)
    manifest = _rewrite_manifest(root)
    manifest["annotations"]["rasters"]["a.png"]["sha256"] = sha256_bytes(payload)
    for item in manifest["files"]:
        if item["path"] == relative:
            item["sha256"] = sha256_bytes(payload)
            item["bytes"] = len(payload)
    _store_manifest(root, manifest)

    reader = open_spatial_product(root)
    try:
        with pytest.raises(SpatialRecordError, match="manifest|declares"):
            reader.annotations_for_view("a.png", "base")
    finally:
        reader.close()


def test_whole_index_checks_descriptor_and_ref_schema(tmp_path):
    root = tmp_path / "product"
    build_product(root)
    entries = _index_lines(root)

    def edit(mutate) -> None:
        copy = json.loads(json.dumps(entries))
        for entry in copy:
            if entry["sample"]["image_key"] == "b/one.png":
                mutate(entry)
        _rewrite_index(root, copy)
        write_envelope(root)

    cases = (
        (lambda e: e["refs"]["pixel_geometric_coverage"].update({"kind": "parquet"}), "unsupported kind"),
        (lambda e: e["refs"]["pixel_geometric_coverage"].pop("key"), "no NPZ key"),
        (lambda e: e["arrays"]["feature:blocks.0"].update({"shape": [2, 2, 9]}), "channels"),
        (lambda e: e["arrays"]["feature:blocks.0"].update({"dtype": "<f8"}), "dtype"),
        (lambda e: e["arrays"]["geometric_coverage"].update({"shape": [3, 3]}), "grid"),
        (lambda e: e["arrays"]["feature:blocks.1"].pop("digest"), "no 'digest'"),
    )
    for mutate, message in cases:
        edit(mutate)
        with pytest.raises(SpatialRecordError, match=message):
            open_spatial_product(root)
        _rewrite_index(root, entries)
        write_envelope(root)
    reader = open_spatial_product(root)
    reader.close()


def test_plan_counts_must_describe_the_stored_support(tmp_path):
    """Erasing queries or shrinking source-valid counts is not a valid plan."""
    from otuformer.data.provenance import canonical_json_bytes
    from otuformer.spatial.protocol import (
        PROTOCOL_VIEW_IDS,
        build_evaluation_plan,
        protocol_operations,
    )
    from otuformer.vision.geometry import apply_geometry, evaluation_view_ops

    root = tmp_path / "product"
    contract = contract_for()
    sample = make_sample("a.png")
    image = Image.new("RGB", (640, 480), (10, 20, 30))
    base_ops = evaluation_view_ops(image, name="center-crop", size=32)
    writer = SpatialWriter(root, contract=contract, layout="per-image", samples=[sample])
    stored = []
    for view_id in PROTOCOL_VIEW_IDS:
        if view_id == "reference":
            ops, role, stored_id = base_ops, "base", "base"
        else:
            ops = [*base_ops, *protocol_operations(32, view_id)]
            role, stored_id = "eval-transform", view_id
        _, trace = apply_geometry(image, ops, track=True)
        template = make_view(sample, contract, view_id=stored_id, role=role)
        view = SpatialView(
            image_key=sample.image_key,
            view_id=stored_id,
            role=role,
            trace=trace,
            features=template.features,
            pixel_geometric_coverage=template.pixel_geometric_coverage,
            pixel_sampling_valid=template.pixel_sampling_valid,
            geometric_coverage=template.geometric_coverage,
            patch_sampling_valid=template.patch_sampling_valid,
        )
        writer.write(view)
        stored.append(view)
    writer.set_evaluation_plan(build_evaluation_plan({"a.png": stored}, seed=42))
    writer.finish()
    write_envelope(root)
    original = json.loads((root / "spatial_evaluation_plan.json").read_text(encoding="utf-8"))

    def store(plan: dict) -> None:
        payload = canonical_json_bytes(plan)
        (root / "spatial_evaluation_plan.json").write_bytes(payload)
        manifest = _rewrite_manifest(root)
        manifest["evaluation_plan"]["sha256"] = sha256_bytes(payload)
        manifest["evaluation_plan"]["bytes"] = len(payload)
        for item in manifest["files"]:
            if item["path"] == "spatial_evaluation_plan.json":
                item["sha256"] = sha256_bytes(payload)
                item["bytes"] = len(payload)
        _store_manifest(root, manifest)

    def first_branch(plan):
        for pair in plan["pairs"]:
            for branch in pair["directions"].values():
                if branch["queries"]:
                    return branch
        raise AssertionError("fixture needs a branch with queries")

    def erase_queries(plan):
        branch = first_branch(plan)
        branch["queries"] = []
        branch["sampled_count"] = 0

    def fake_source_valid(plan):
        branch = first_branch(plan)
        branch["source_valid_count"] = 1

    def consistent_zero(plan):
        # Every branch agrees with every other and with the real source support:
        # only recomputing eligibility from the stored geometry catches it.
        for pair in plan["pairs"]:
            for branch in pair["directions"].values():
                branch["queries"] = []
                branch["sampled_count"] = 0
                branch["eligible_count"] = 0
                branch["unavailable"] = True
                branch["unavailable_reason"] = "no_overlap"

    def drop_signature_matrix(plan):
        plan["pairs"][0]["geometry"]["target"].pop("source_to_view")

    cases = (
        (erase_queries, "frozen protocol samples|stored support"),
        (fake_source_valid, "stored support|valid patch cells"),
        (consistent_zero, "stored support|frozen protocol samples"),
        (drop_signature_matrix, "source_to_view"),
    )
    for mutate, message in cases:
        plan = json.loads(json.dumps(original))
        mutate(plan)
        store(plan)
        with pytest.raises(SpatialRecordError, match=message):
            open_spatial_product(root)
    store(original)
    reader = open_spatial_product(root)
    reader.close()


def _protocol_product(tmp_path, *, keys=("a.png",), views_per_key=None):
    """One product carrying the frozen protocol views and a real saved plan."""
    from otuformer.spatial.protocol import (
        PROTOCOL_VIEW_IDS,
        build_evaluation_plan,
        protocol_operations,
    )
    from otuformer.vision.geometry import apply_geometry, evaluation_view_ops

    root = tmp_path / "product"
    contract = contract_for()

    def build(selected):
        samples = [make_sample(key) for key in selected]
        image = Image.new("RGB", (640, 480), (10, 20, 30))
        base_ops = evaluation_view_ops(image, name="center-crop", size=32)
        writer = SpatialWriter(
            root, contract=contract, layout="per-image", samples=samples
        )
        stored: dict[str, list[SpatialView]] = {key: [] for key in selected}
        for sample in samples:
            for view_id in PROTOCOL_VIEW_IDS:
                if view_id == "reference":
                    ops, role, stored_id = base_ops, "base", "base"
                else:
                    ops = [*base_ops, *protocol_operations(32, view_id)]
                    role, stored_id = "eval-transform", view_id
                _, trace = apply_geometry(image, ops, track=True)
                template = make_view(sample, contract, view_id=stored_id, role=role)
                view = SpatialView(
                    image_key=sample.image_key,
                    view_id=stored_id,
                    role=role,
                    trace=trace,
                    features=template.features,
                    pixel_geometric_coverage=template.pixel_geometric_coverage,
                    pixel_sampling_valid=template.pixel_sampling_valid,
                    geometric_coverage=template.geometric_coverage,
                    patch_sampling_valid=template.patch_sampling_valid,
                )
                writer.write(view)
                stored[sample.image_key].append(view)
        writer.set_evaluation_plan(build_evaluation_plan(stored, seed=42))
        writer.finish()
        write_envelope(root)

    build(tuple(keys))
    return root


def test_a_selected_read_does_not_load_unselected_payloads(tmp_path):
    """Plan validation reads support bytes only for the selected samples."""
    root = _protocol_product(tmp_path, keys=("a.png", "b.png"))
    selector = tmp_path / "selector.csv"
    selector.write_text("image\na.png\n", encoding="utf-8")
    # Deleting an unselected payload makes the difference observable: a whole-plan
    # support check would fail on it even though the report never covers it.
    b_payloads = sorted((root / "features").glob("*.npz"))
    removed = 0
    index = [
        json.loads(line)
        for line in (root / "spatial_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    b_paths = {
        entry["refs"][name]["path"]
        for entry in index
        if entry["sample"]["image_key"] == "b.png"
        for name in entry["refs"]
    }
    for name in b_paths:
        (root / name).unlink()
        removed += 1
    assert removed
    reader = open_spatial_product(root, selector_csv=selector)
    try:
        assert reader.image_keys == ("a.png",)
        assert {view.image_key for view in reader.iter_views()} == {"a.png"}
    finally:
        reader.close()
    assert b_payloads  # the fixture really had unselected payloads


def test_a_tampered_plan_is_recomputed_not_trusted_from_its_counts(tmp_path):
    """Consistently forged counts, a dropped query and a bad grid support fail.

    The stored counts are what a report publishes. Checking only their arithmetic
    let a plan make every branch `eligible_count=0`/`no_overlap` and erase the
    whole evaluation, and the target-support check tested the pixel grid only, so
    a point whose feature interpolation support was incomplete still opened.
    """
    from otuformer.data.provenance import canonical_json_bytes

    root = _protocol_product(tmp_path, keys=("a.png",))
    plan_path = root / "spatial_evaluation_plan.json"
    original = json.loads(plan_path.read_text(encoding="utf-8"))

    def store(plan: dict) -> None:
        payload = canonical_json_bytes(plan)
        plan_path.write_bytes(payload)
        manifest = _rewrite_manifest(root)
        manifest["evaluation_plan"]["sha256"] = sha256_bytes(payload)
        manifest["evaluation_plan"]["bytes"] = len(payload)
        for item in manifest["files"]:
            if item["path"] == "spatial_evaluation_plan.json":
                item["sha256"] = sha256_bytes(payload)
                item["bytes"] = len(payload)
        _store_manifest(root, manifest)

    def first_branch(plan):
        for pair in plan["pairs"]:
            for branch in pair["directions"].values():
                if branch["queries"]:
                    return branch
        raise AssertionError("fixture needs a branch with queries")

    def erase_every_branch(plan):
        for pair in plan["pairs"]:
            for branch in pair["directions"].values():
                branch["queries"] = []
                branch["sampled_count"] = 0
                branch["eligible_count"] = 0
                branch["unavailable"] = True
                branch["unavailable_reason"] = "no_overlap"

    def drop_one_query(plan):
        branch = first_branch(plan)
        branch["queries"] = branch["queries"][1:]
        branch["sampled_count"] = len(branch["queries"])

    def swap_one_query(plan):
        branch = first_branch(plan)
        kept = branch["queries"][0]
        dropped = branch["queries"][1] if len(branch["queries"]) > 1 else None
        if dropped is None:
            raise AssertionError("fixture needs two queries")
        branch["queries"] = [dropped, kept]
        branch["queries"][0]["row"], branch["queries"][0]["column"] = 0, 0
        branch["sampled_count"] = len(branch["queries"])

    cases = (
        (erase_every_branch, "stored support"),
        (drop_one_query, "frozen protocol samples"),
        (swap_one_query, "frozen protocol samples"),
    )
    for mutate, message in cases:
        plan = json.loads(json.dumps(original))
        mutate(plan)
        store(plan)
        with pytest.raises(SpatialRecordError, match=message):
            open_spatial_product(root)
    store(original)
    reader = open_spatial_product(root)
    reader.close()


def test_an_escaping_image_key_is_refused_at_both_boundaries(tmp_path):
    """A key becomes a filesystem path for overlays, so it must stay relative."""
    from otuformer.spatial.records import SpatialSample

    for key in ("../outside.png", "/abs.png", "a/../b.png", "a//b.png", "C:/x.png"):
        with pytest.raises(SpatialRecordError):
            SpatialSample(
                image_key=key, image_sha256="a" * 64, width=8, height=8
            )
    # The reader re-checks the stored index, not only the writer's records.
    root = tmp_path / "product"
    build_product(root)
    index_path = root / "spatial_index.jsonl"
    lines = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lines[0]["sample"]["image_key"] = "../outside.png"
    payload = b"\n".join(
        json.dumps(line, sort_keys=True).encode("utf-8") for line in lines
    ) + b"\n"
    index_path.write_bytes(payload)
    manifest = _rewrite_manifest(root)
    manifest["index"]["sha256"] = sha256_bytes(payload)
    manifest["index"]["bytes"] = len(payload)
    for item in manifest["files"]:
        if item["path"] == "spatial_index.jsonl":
            item["sha256"] = sha256_bytes(payload)
            item["bytes"] = len(payload)
    _store_manifest(root, manifest)
    with pytest.raises(SpatialRecordError, match="canonical relative path"):
        open_spatial_product(root)


def test_sample_identity_and_trace_source_are_strict(tmp_path):
    """A readable product's ids, digests, sizes and trace source agree."""
    root = tmp_path / "product"
    build_product(root)
    original = (root / "spatial_index.jsonl").read_text(encoding="utf-8")

    def store(lines):
        payload = b"\n".join(
            json.dumps(line, sort_keys=True).encode("utf-8") for line in lines
        ) + b"\n"
        (root / "spatial_index.jsonl").write_bytes(payload)
        manifest = _rewrite_manifest(root)
        manifest["index"]["sha256"] = sha256_bytes(payload)
        manifest["index"]["bytes"] = len(payload)
        for item in manifest["files"]:
            if item["path"] == "spatial_index.jsonl":
                item["sha256"] = sha256_bytes(payload)
                item["bytes"] = len(payload)
        _store_manifest(root, manifest)

    def lines():
        return [
            json.loads(line)
            for line in original.splitlines()
            if line.strip()
        ]

    cases = (
        ({"sample_id": 1.9}, "unusable sample id"),
        ({"img": {"image_sha256": "not-a-digest"}}, "unusable image_sha256"),
        ({"img": {"width": 0}}, "unusable sample width"),
        (
            {"sample": {"width": 1, "height": 1, "image_sha256": "a" * 64,
                        "image_key": "a/one.png", "metadata": {}}},
            "declares|traces a source",
        ),
    )
    for patch, message in cases:
        mutated = lines()
        for name in ("sample_id", "sample"):
            if name in patch:
                mutated[0][name] = patch[name]
        if "img" in patch:
            mutated[0]["sample"].update(patch["img"])
        store(mutated)
        with pytest.raises(SpatialRecordError, match=message):
            open_spatial_product(root)
    # The untouched index still opens.
    (root / "spatial_index.jsonl").write_text(original, encoding="utf-8")
    store(lines())
    reader = open_spatial_product(root)
    reader.close()


def test_every_view_of_a_key_must_share_one_sample_identity(tmp_path):
    """A key describes one source image; its views cannot disagree about it."""
    root = tmp_path / "product"
    build_product(root, views_per_key=2)
    original = (root / "spatial_index.jsonl").read_text(encoding="utf-8")

    def store(lines):
        payload = b"\n".join(
            json.dumps(line, sort_keys=True).encode("utf-8") for line in lines
        ) + b"\n"
        (root / "spatial_index.jsonl").write_bytes(payload)
        manifest = _rewrite_manifest(root)
        manifest["index"]["sha256"] = sha256_bytes(payload)
        manifest["index"]["bytes"] = len(payload)
        for item in manifest["files"]:
            if item["path"] == "spatial_index.jsonl":
                item["sha256"] = sha256_bytes(payload)
                item["bytes"] = len(payload)
        _store_manifest(root, manifest)

    def lines():
        return [json.loads(line) for line in original.splitlines() if line.strip()]

    cases = (
        ({"sample_id": 2}, "disagree about their sample identity"),
        ({"sample": {"image_sha256": "b" * 64}}, "disagree about their sample identity"),
        ({"sample": {"metadata": {"label": "other"}}}, "disagree about their sample identity"),
    )
    for patch, message in cases:
        mutated = lines()
        second = mutated[1]
        if "sample_id" in patch:
            second["sample_id"] = patch["sample_id"]
        else:
            second["sample"].update(patch["sample"])
        store(mutated)
        with pytest.raises(SpatialRecordError, match=message):
            open_spatial_product(root)
    (root / "spatial_index.jsonl").write_text(original, encoding="utf-8")
    store(lines())
    reader = open_spatial_product(root)
    reader.close()
