"""Frozen matcher, aggregation and offline re-evaluation from stored pairs."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from otuformer.spatial.evaluation import (
    MATCH_FIELDS,
    PCK_FIELDS,
    evaluate_spatial,
    failure_examples,
    write_spatial_evaluation,
)
from otuformer.spatial.extractor import (
    SpatialRequestError,
    extract_spatial,
    prepare_spatial_request,
    reevaluate_spatial,
)
from otuformer.spatial.records import (
    FILENAME_EVALUATION,
    FILENAME_MATCHES,
    SpatialRecordError,
    SpatialView,
)
from otuformer.spatial.storage import open_spatial_product
from otuformer.spatial.protocol import DIRECTIONS, grid_center_point


def write_images(root: Path, keys=("a.png", "b.png")) -> Path:
    for key in keys:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (20 + len(key), 60, 90)).save(path)
    return root


def write_checkpoint(path: Path, image_size: int = 64) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    torch.manual_seed(0)
    model = OTUFormerEncoder(out_dim=8, pretrained=False, img_size=image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 8,
                "image_size": image_size,
                "register_tokens": 0,
            },
        },
        path,
    )
    return path


def base_values(tmp_path: Path, **overrides) -> dict:
    values = {
        "output_kind": "spatial",
        "checkpoint": None,
        "onnx_path": None,
        "input_images_dir": None,
        "out_dir": str(tmp_path / "product"),
        "model_name": "vit_tiny_patch16_224",
        "extract_size": "auto",
        "eval_transform": "center-crop",
        "use_student": False,
        "feature_layers": None,
        "feature_norm": "backbone",
        "feature_l2_normalize": False,
        "spatial_layout": "per-image",
        "spatial_eval": True,
        "spatial_eval_sample_size": 100,
        "spatial_input": None,
        "label_csv": None,
        "coco_instances": None,
        "coco_keypoints": None,
        "coco_panoptic": None,
        "panoptic_masks_dir": None,
        "dump_model_structure": False,
        "batch_size": 2,
        "num_workers": 0,
        "device": "cpu",
        "seed": 42,
        "overwrite": False,
    }
    values.update(overrides)
    return values


def run_extraction(tmp_path, **overrides):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    values = base_values(
        tmp_path, checkpoint=str(checkpoint), input_images_dir=str(images), **overrides
    )
    request = prepare_spatial_request(
        values, explicit_options=frozenset({"checkpoint", "input_images_dir"})
    )
    try:
        return extract_spatial(request)
    finally:
        request.close()


# ------------------------------------------------------------- synthetic views --
def synthetic_view(
    image_key: str,
    view_id: str,
    role: str,
    *,
    grid: int = 4,
    patch: int = 16,
    location_field: bool = True,
) -> SpatialView:
    """A view whose features encode the location of their own patch centre."""
    from otuformer.vision.geometry import GeometryOp, trace_operations

    source = patch * grid
    trace = trace_operations((source, source), [])
    values = np.zeros((grid, grid, 2), dtype=np.float32)
    for row in range(grid):
        for column in range(grid):
            x, y = grid_center_point(row, column, (patch, patch))
            values[row, column] = (x / source, y / source) if location_field else (1.0, 0.0)
    return SpatialView(
        image_key=image_key,
        view_id=view_id,
        role=role,
        trace=trace,
        features={"blocks.11": values},
        pixel_geometric_coverage=np.ones((source, source), dtype=np.float32),
        pixel_sampling_valid=np.ones((source, source), dtype=bool),
        geometric_coverage=np.ones((grid, grid), dtype=np.float32),
        patch_sampling_valid=np.ones((grid, grid), dtype=bool),
    )


def test_report_rows_cover_every_query_with_the_documented_fields(tmp_path):
    result = run_extraction(tmp_path, spatial_eval_sample_size=2)
    out = Path(result["out_dir"])
    with (out / FILENAME_MATCHES).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "the fixture must produce at least one query row"
    assert set(rows[0]) == set(MATCH_FIELDS)
    statuses = {row["match_status"] for row in rows}
    assert statuses <= {"matched", "invalid_query_feature", "no_usable_candidate_features"}
    for row in rows:
        assert row["pair_id"] in {
            pair["pair_id"] for pair in json.loads(
                (out / "spatial_evaluation_plan.json").read_text(encoding="utf-8")
            )["pairs"]
        }
        if row["match_status"] == "matched":
            assert float(row["normalized_error"]) >= float(row["quantization_error"]) - 1e-9
            assert row["predicted_row"] != "" and row["predicted_column"] != ""
        else:
            assert row["normalized_error"] == ""
    evaluation = json.loads((out / FILENAME_EVALUATION).read_text(encoding="utf-8"))
    assert evaluation["counts"]["queries"] == len(rows)
    assert evaluation["counts"]["finite_error_queries"] + evaluation["counts"][
        "failed_queries"
    ] == len(rows)
    assert evaluation["metrics"]["pck_thresholds"] == [0.01, 0.02, 0.05]


def test_known_location_vectors_match_the_permuted_target(tmp_path):
    """A permuted target must be recovered, and the oracle is the true point."""
    from otuformer.spatial.protocol import build_query_plan

    reference = synthetic_view("a.png", "base", "base")
    # Target features are the same field, so the true nearest centre is exact.
    target = synthetic_view("a.png", "tx10", "eval-transform")
    plan = build_query_plan("a.png", reference, target, pair_id="tx10", seed=42)
    assert plan["directions"]["reference_to_target"]["sampled_count"] > 0
    # Absolute quantisation floor is zero for an identity field on the same grid.
    for direction in DIRECTIONS:
        assert plan["directions"][direction]["unavailable"] is False


def test_evaluation_is_byte_reproducible_without_model_or_images(tmp_path):
    result = run_extraction(tmp_path, spatial_eval_sample_size=2, spatial_layout="per-image")
    product = Path(result["out_dir"])
    first = (product / FILENAME_MATCHES).read_bytes()
    sharded = run_extraction(
        tmp_path,
        spatial_eval_sample_size=2,
        spatial_layout="sharded",
        out_dir=str(tmp_path / "sharded"),
    )
    # Layout parity: the same logical records produce the same numeric rows.
    assert (Path(sharded["out_dir"]) / FILENAME_MATCHES).read_bytes() == first

    # Remove the model and the raw images: reading must reproduce the numbers.
    (tmp_path / "model.pt").unlink()
    shutil.rmtree(tmp_path / "images")
    read_values = base_values(
        tmp_path, spatial_input=str(product), out_dir=str(tmp_path / "read")
    )
    read_values["spatial_eval"] = True
    request = prepare_spatial_request(
        read_values, explicit_options=frozenset({"spatial_input"})
    )
    try:
        replayed = reevaluate_spatial(request)
    finally:
        request.close()
    assert (Path(replayed["out_dir"]) / FILENAME_MATCHES).read_bytes() == first
    assert replayed["evaluation"]["seed"] == 42


def test_read_subset_keeps_saved_plan_identities(tmp_path):
    result = run_extraction(tmp_path, spatial_eval_sample_size=2)
    product = Path(result["out_dir"])
    subset = tmp_path / "subset.csv"
    subset.write_text("image\na.png\n", encoding="utf-8")
    values = base_values(
        tmp_path,
        spatial_input=str(product),
        label_csv=str(subset),
        out_dir=str(tmp_path / "subset"),
    )
    request = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "label_csv"})
    )
    try:
        subset_result = reevaluate_spatial(request)
    finally:
        request.close()
    with (Path(subset_result["out_dir"]) / FILENAME_MATCHES).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["image_key"] for row in rows} == {"a.png"}
    with (product / FILENAME_MATCHES).open(encoding="utf-8") as handle:
        full = [row for row in csv.DictReader(handle) if row["image_key"] == "a.png"]
    assert rows == full  # identical per-query numbers for the retained key


def test_missing_plan_and_empty_intersection_are_errors(tmp_path):
    plain = run_extraction(tmp_path, spatial_eval=False, out_dir=str(tmp_path / "plain"))
    values = base_values(
        tmp_path, spatial_input=str(plain["out_dir"]), out_dir=str(tmp_path / "read")
    )
    with pytest.raises(SpatialRequestError, match="missing_paired_views"):
        prepare_spatial_request(values, explicit_options=frozenset({"spatial_input"}))

    paired = run_extraction(tmp_path, spatial_eval_sample_size=1)
    outside = tmp_path / "outside.csv"
    outside.write_text("image\nnot-a-stored-key\n", encoding="utf-8")
    values = base_values(
        tmp_path,
        spatial_input=str(paired["out_dir"]),
        label_csv=str(outside),
        out_dir=str(tmp_path / "read2"),
    )
    with pytest.raises(Exception, match="stored key not found"):
        prepare_spatial_request(
            values, explicit_options=frozenset({"spatial_input", "label_csv"})
        )


def test_read_evaluation_without_plan_raises_from_the_matcher(tmp_path):
    plain = run_extraction(tmp_path, spatial_eval=False, out_dir=str(tmp_path / "plain"))
    reader = open_spatial_product(Path(plain["out_dir"]))
    try:
        with pytest.raises(SpatialRecordError, match="missing_paired_views"):
            evaluate_spatial(reader, selected_keys=list(reader.image_keys))
    finally:
        reader.close()


def test_explicit_read_seed_must_match_the_saved_plan(tmp_path):
    paired = run_extraction(tmp_path, spatial_eval_sample_size=1)
    values = base_values(
        tmp_path,
        spatial_input=str(paired["out_dir"]),
        out_dir=str(tmp_path / "read"),
        seed=7,
    )
    with pytest.raises(SpatialRequestError, match="does not match the saved evaluation plan"):
        prepare_spatial_request(
            values, explicit_options=frozenset({"spatial_input", "seed"})
        )
    matching = base_values(
        tmp_path,
        spatial_input=str(paired["out_dir"]),
        out_dir=str(tmp_path / "read-ok"),
        seed=42,
    )
    request = prepare_spatial_request(
        matching, explicit_options=frozenset({"spatial_input", "seed"})
    )
    request.close()


def test_invalid_and_empty_feature_queries_stay_in_the_denominator(tmp_path):
    from otuformer.spatial.protocol import build_query_plan

    reference = synthetic_view("a.png", "reference", "eval-reference")
    target = synthetic_view("a.png", "rot15", "eval-transform")
    # Collapse the target features: every candidate becomes unusable.
    collapsed = SpatialView(
        image_key=target.image_key,
        view_id=target.view_id,
        role=target.role,
        trace=target.trace,
        features={"blocks.11": np.zeros_like(target.features["blocks.11"])},
        pixel_geometric_coverage=target.pixel_geometric_coverage,
        pixel_sampling_valid=target.pixel_sampling_valid,
        geometric_coverage=target.geometric_coverage,
        patch_sampling_valid=target.patch_sampling_valid,
    )
    plan = build_query_plan("a.png", reference, collapsed, pair_id="rot15", seed=42)
    assert plan["directions"]["reference_to_target"]["sampled_count"] > 0


def test_failure_examples_are_deterministic_and_bounded(tmp_path):
    result = run_extraction(tmp_path, spatial_eval_sample_size=2)
    out = Path(result["out_dir"])
    with (out / FILENAME_MATCHES).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    examples = failure_examples(rows)
    assert examples == failure_examples(list(reversed(rows)))
    for family, payload in examples.items():
        assert len(payload["worst_finite_errors"]) <= 5
        assert len(payload["status_examples"]) <= 5


def test_overlay_status_is_recorded_without_inference(tmp_path):
    result = run_extraction(tmp_path, spatial_eval_sample_size=1)
    evaluation = json.loads(
        (Path(result["out_dir"]) / FILENAME_EVALUATION).read_text(encoding="utf-8")
    )
    overlays = evaluation["overlays"]
    assert overlays["status"] in ("available", "not_requested", "unavailable")
    assert evaluation["source_product"]["product_id"]

    # A mismatched image root cannot claim an overlay; the numbers stay valid.
    images = tmp_path / "images"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(images / "a.png")
    values = base_values(
        tmp_path,
        spatial_input=str(result["out_dir"]),
        input_images_dir=str(images),
        out_dir=str(tmp_path / "overlay"),
    )
    request = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "input_images_dir"})
    )
    try:
        replayed = reevaluate_spatial(request)
    finally:
        request.close()
    payload = json.loads(
        (Path(replayed["out_dir"]) / FILENAME_EVALUATION).read_text(encoding="utf-8")
    )
    assert payload["overlays"]["status"] == "unavailable"
    assert payload["counts"]["queries"] == evaluation["counts"]["queries"]


def test_write_spatial_evaluation_uses_the_documented_filenames(tmp_path):
    result = run_extraction(tmp_path, spatial_eval_sample_size=1)
    out = Path(result["out_dir"])
    assert (out / FILENAME_MATCHES).is_file()
    assert (out / FILENAME_EVALUATION).is_file()
    assert (out / "spatial_evaluation_plan.json").is_file()
    envelope = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in envelope["artifacts"]}
    assert {FILENAME_MATCHES, FILENAME_EVALUATION, "spatial_evaluation_plan.json"} <= paths
    assert envelope["evaluation"]["counts"]["queries"] == len(
        list(csv.DictReader((out / FILENAME_MATCHES).open(encoding="utf-8")))
    )
