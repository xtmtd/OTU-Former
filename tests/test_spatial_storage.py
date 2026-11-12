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
    reader = SpatialReader(root, require_completion=False)
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
        ("feature_shape", "has shape"),
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
    root = tmp_path / "product"
    build_product(root)
    entries = _index_lines(root)
    entries[0]["geometry"]["source_to_view"] = np.zeros((3, 3)).tolist()
    _rewrite_index(root, entries)
    reader = open_spatial_product(root)
    with pytest.raises(ValueError, match="singular"):
        list(reader.iter_views())
    reader.close()


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


def test_previous_completed_product_survives_a_failed_attempt(tmp_path):
    root = tmp_path / "product"
    build_product(root)
    before = (root / FILENAME_MANIFEST).read_bytes()
    # A second attempt into the same directory fails before sealing.
    contract = contract_for()
    writer = SpatialWriter(
        root, contract=contract, layout="per-image", samples=[make_sample("a/one.png")]
    )
    with pytest.raises(SpatialRecordError):
        writer.finish()
    writer.abort({"reason": "no views"})
    assert (root / FILENAME_MANIFEST).read_bytes() == before
    reader = open_spatial_product(root)
    try:
        assert len(list(reader.iter_views())) == 2
    finally:
        reader.close()


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
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    write_envelope(root)
    with pytest.raises(SpatialRecordError, match="is not 'spatial-geometry-v1'"):
        open_spatial_product(root)
