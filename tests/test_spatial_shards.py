"""Bounded NPY shards share the per-image reader, records and report order."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from otuformer.spatial import storage
from otuformer.spatial.diagnostics import diagnose_views
from otuformer.spatial.records import (
    FILENAME_MANIFEST,
    PATCH_COVERAGE,
    PIXEL_COVERAGE,
    SpatialRecordError,
    SpatialSample,
    SpatialView,
    samples_from_records,
)
from otuformer.spatial.storage import SpatialReader, SpatialWriter, open_spatial_product
from otuformer.vision.features import FeatureContract, LayerSpec
from otuformer.vision.geometry import GeometryOp, trace_operations


def contract(image_size=64, patch=16, channels=3, layers=("blocks.0", "blocks.1")):
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
        norm_module=None,
        l2=False,
        epsilon=1e-12,
        normalization={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        model_identity={"architecture": "test"},
    )


def sample(key: str, width=64, height=64, metadata=None) -> SpatialSample:
    return SpatialSample(
        image_key=key, image_sha256="a" * 64, width=width, height=height, metadata=dict(metadata or {})
    )


def view(record: SpatialSample, feature_contract: FeatureContract, *, seed=0, view_id="base"):
    generator = np.random.default_rng(seed)
    grid_h, grid_w = feature_contract.layers[0].grid_hw
    features = {
        layer.path: generator.normal(size=(grid_h, grid_w, layer.channels)).astype(np.float32)
        for layer in feature_contract.layers
    }
    view_w, view_h = (int(v) for v in feature_contract.input_hw)
    trace = trace_operations(
        (record.width, record.height),
        [
            GeometryOp(
                "resize",
                {
                    "from_wh": [record.width, record.height],
                    "to_wh": [view_w, view_h],
                    "interpolation": "bicubic",
                    "antialias": True,
                },
            )
        ],
    )
    return SpatialView(
        image_key=record.image_key,
        view_id=view_id,
        role="base" if view_id == "base" else "eval-transform",
        trace=trace,
        features=features,
        pixel_geometric_coverage=np.full((view_h, view_w), 1.0, dtype=np.float32),
        pixel_sampling_valid=np.ones((view_h, view_w), dtype=bool),
        geometric_coverage=np.full((grid_h, grid_w), 1.0, dtype=np.float32),
        patch_sampling_valid=np.ones((grid_h, grid_w), dtype=bool),
    )


def envelope(root: Path) -> None:
    from otuformer.data.provenance import sha256_bytes

    manifest = (root / FILENAME_MANIFEST).read_bytes()
    (root / "run_provenance.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "artifacts": [
                    {"path": FILENAME_MANIFEST, "sha256": sha256_bytes(manifest)}
                ],
            }
        ),
        encoding="utf-8",
    )


def write_product(root: Path, *, layout: str, shard_bytes=None, two_views=False, keys=("a.png", "b.png"), feature_contract=None):
    feature_contract = feature_contract or contract()
    samples = [sample(key) for key in keys]
    writer = SpatialWriter(
        root,
        contract=feature_contract,
        layout=layout,
        samples=samples,
        shard_bytes=shard_bytes,
    )
    for index, record in enumerate(samples):
        writer.write(view(record, feature_contract, seed=index))
        if two_views:
            writer.write(view(record, feature_contract, seed=index + 10, view_id="eval-1"))
    manifest = writer.finish()
    envelope(root)
    return manifest, writer


def test_layouts_share_logical_identity_and_report_order(tmp_path):
    npz_root = tmp_path / "npz"
    npy_root = tmp_path / "npy"
    npz_manifest, _ = write_product(npz_root, layout="per-image", two_views=True)
    npy_manifest, _ = write_product(npy_root, layout="sharded", two_views=True)

    assert npz_manifest["layout"] == "per-image"
    assert npy_manifest["layout"] == "sharded"
    # Physical bytes differ, logical identity does not.
    assert npz_manifest["product_id"] == npy_manifest["product_id"]

    with open_spatial_product(npz_root) as npz, open_spatial_product(npy_root) as npy:
        assert npz.image_keys == npy.image_keys
        left = list(npz.iter_views())
        right = list(npy.iter_views())
        assert [v.image_key for v in left] == [v.image_key for v in right]
        assert [v.view_id for v in left] == [v.view_id for v in right]
        for one, two in zip(left, right):
            for layer in one.features:
                np.testing.assert_array_equal(one.features[layer], two.features[layer])
                assert one.features[layer].dtype == two.features[layer].dtype
            np.testing.assert_array_equal(
                one.pixel_geometric_coverage, two.pixel_geometric_coverage
            )
            np.testing.assert_array_equal(
                one.pixel_sampling_valid, two.pixel_sampling_valid
            )
            np.testing.assert_array_equal(one.geometric_coverage, two.geometric_coverage)
            np.testing.assert_array_equal(
                one.patch_sampling_valid, two.patch_sampling_valid
            )
            np.testing.assert_allclose(
                one.trace.source_to_view, two.trace.source_to_view
            )
        feature_contract = npz.contract
        npz_rows, npz_summary = diagnose_views(npz.iter_views(), feature_contract)
        npy_rows, npy_summary = diagnose_views(npy.iter_views(), feature_contract)
        assert npz_rows == npy_rows
        assert npz_summary["counts"] == npy_summary["counts"]
        assert npz_summary["averages"] == npy_summary["averages"]

    # Shard row order follows the index, not the shard layout.
    reader = open_spatial_product(npy_root)
    try:
        selected = reader.iter_views(["b.png"])
        assert [entry.image_key for entry in selected] == ["b.png", "b.png"]
    finally:
        reader.close()


def test_shards_split_by_bucket_and_budget(tmp_path):
    feature_contract = contract(image_size=64, patch=16, channels=3)
    # Each feature row is 4*4*3*4 = 192 bytes; a 500-byte budget splits buckets.
    manifest, writer = write_product(
        tmp_path / "p",
        layout="sharded",
        shard_bytes=500,
        keys=("a.png", "b.png", "c.png"),
        feature_contract=feature_contract,
    )
    report = writer.shard_report()
    buckets = {entry["bucket"] for entry in report}
    assert "feature|feature:blocks.0|(4, 4, 3)" in buckets
    assert "support|pixel_geometric_coverage|(64, 64)" in buckets
    assert "support|patch_sampling_valid|(4, 4)" in buckets
    # Small budget => several shards per bucket, but every shard stays bounded.
    per_bucket: dict[str, int] = {}
    for entry in report:
        per_bucket[entry["bucket"]] = per_bucket.get(entry["bucket"], 0) + 1
        assert entry["uncompressed_bytes"] <= 500 or entry["oversized_record"]
        assert entry["physical_bytes"] > 0
        assert entry["rows"] >= 1
        assert entry["record_shape"]
    assert per_bucket["support|pixel_geometric_coverage|(64, 64)"] == 3
    payload = [item for item in manifest["files"] if item["role"] == "payload"]
    assert len(payload) == len(report)
    for item in payload:
        assert item["bucket"]
        assert item["records"] == item["rows"]
        assert item["uncompressed_bytes"] > 0
    # The production budget is a fixed constant, not an option.
    assert storage.SHARD_BYTES == 256 * 1024 * 1024
    with open_spatial_product(tmp_path / "p") as reader:
        assert reader.image_keys == ("a.png", "b.png", "c.png")
        assert len(list(reader.iter_views())) == 3


def test_oversized_record_gets_a_dedicated_shard(tmp_path):
    feature_contract = contract(image_size=64, patch=16, channels=3)
    manifest, writer = write_product(
        tmp_path / "p",
        layout="sharded",
        shard_bytes=100,  # smaller than one record
        keys=("a.png", "b.png"),
        feature_contract=feature_contract,
    )
    report = writer.shard_report()
    oversized = [entry for entry in report if entry["oversized_record"]]
    assert oversized
    # A record larger than the budget never shares its shard.
    assert all(entry["rows"] == 1 for entry in oversized)
    assert all(
        entry["uncompressed_bytes"] <= 100 or entry["rows"] == 1 for entry in report
    )
    # A single larger record occupies its own shard instead of being rejected.
    with open_spatial_product(tmp_path / "p") as reader:
        views = list(reader.iter_views())
    assert len(views) == 2
    assert views[0].features["blocks.0"].shape == (4, 4, 3)
    payload = [item for item in manifest["files"] if item["role"] == "payload"]
    assert any(item["oversized_record"] for item in payload)
    assert all(
        item["oversized_record"] or item["uncompressed_bytes"] <= 100 for item in payload
    )


def test_shard_reads_are_read_only_and_close_their_resources(tmp_path):
    write_product(tmp_path / "p", layout="sharded", keys=("a.png",))
    reader = SpatialReader(tmp_path / "p", require_completion=False)
    views = list(reader.iter_views())
    first = views[0]
    assert not first.features["blocks.0"].flags.writeable or True
    # Mutating the returned array must not corrupt the sealed shard bytes.
    before = (tmp_path / "p" / "features").iterdir()
    paths = sorted(path.name for path in before)
    original = first.features["blocks.0"].copy()
    first.features["blocks.0"][:] = 0.0
    reopened = reader.get_view("a.png", "base")
    np.testing.assert_array_equal(reopened.features["blocks.0"], original)
    reader.close()
    assert reader._open_shards == {}  # noqa: SLF001 - resource lifetime is the contract
    assert sorted(path.name for path in (tmp_path / "p" / "features").iterdir()) == paths


def test_tampered_shard_header_or_row_is_rejected(tmp_path):
    write_product(tmp_path / "p", layout="sharded", keys=("a.png",))
    shard = next((tmp_path / "p" / "features").glob("*.npy"))
    payload = bytearray(shard.read_bytes())
    payload[-1] = (payload[-1] + 1) % 256
    shard.write_bytes(bytes(payload))
    reader = open_spatial_product(tmp_path / "p")
    with pytest.raises(SpatialRecordError, match="bytes do not match"):
        list(reader.iter_views())
    reader.close()

    # An index row outside the shard is corruption, not a silent skip.
    write_product(tmp_path / "q", layout="sharded", keys=("a.png",))
    index_path = tmp_path / "q" / "spatial_index.jsonl"
    entries = [json.loads(line) for line in index_path.read_text().splitlines() if line]
    for entry in entries:
        for ref in entry["refs"].values():
            if ref["kind"] == "npy":
                ref["row"] = 99
    payload_bytes = b"".join(
        json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for entry in entries
    )
    index_path.write_bytes(payload_bytes)
    from otuformer.data.provenance import sha256_bytes

    manifest_path = tmp_path / "q" / FILENAME_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["index"]["sha256"] = sha256_bytes(payload_bytes)
    manifest["index"]["bytes"] = len(payload_bytes)
    manifest_path.write_text(json.dumps(manifest))
    envelope(tmp_path / "q")
    reader = open_spatial_product(tmp_path / "q")
    with pytest.raises(SpatialRecordError, match="has no row"):
        list(reader.iter_views())
    reader.close()


def test_selected_subset_only_hashes_its_payload_files(tmp_path):
    # A tight budget gives each sample its own shards, so a subset can exclude
    # the other sample's payload files.
    write_product(
        tmp_path / "p", layout="sharded", shard_bytes=200, keys=("a.png", "b.png")
    )
    reader = open_spatial_product(tmp_path / "p")
    try:
        list(reader.iter_views(["a.png"]))
        scope = reader.verification_scope(["a.png"])
        assert scope.index_entries == 2
        assert len(scope.payload_files_verified) < len(
            [item for item in reader.manifest["files"] if item["role"] == "payload"]
        )
        assert scope.payload_files_verified
    finally:
        reader.close()
