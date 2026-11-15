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


def test_candidate_geometry_is_built_once_per_direction(monkeypatch):
    from otuformer.spatial import evaluation
    from otuformer.spatial.protocol import build_query_plan

    reference = synthetic_view("a.png", "reference", "eval-reference")
    target = synthetic_view("a.png", "rot15", "eval-transform")
    for view in (reference, target):
        view.features["blocks.11"] = np.eye(16, dtype=np.float32).reshape(4, 4, 16)
    pair = build_query_plan("a.png", reference, target, pair_id="rot15", seed=42)
    original = evaluation.transform_points
    candidate_calls = []

    def count_candidates(matrix, points):
        if len(points) > 1:
            candidate_calls.append(len(points))
        return original(matrix, points)

    monkeypatch.setattr(evaluation, "transform_points", count_candidates)
    rows = evaluation._evaluate_pair_direction(
        reference, target, pair=pair, direction="reference_to_target",
        layer="blocks.11", query_limit=64,
    )
    assert len(rows) == 16
    assert all(row["normalized_error"] == 0 for row in rows)
    assert all(row["pck_0p05"] == 1 for row in rows)
    assert candidate_calls == [16]


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
    """A permuted target must be recovered by the real matcher, not just planned."""
    from otuformer.spatial.evaluation import _evaluate_pair_direction
    from otuformer.spatial.protocol import build_query_plan

    grid = 4
    patch = 16
    source = grid * patch

    def one_hot_view(view_id: str, role: str, *, permute: bool) -> SpatialView:
        from otuformer.vision.geometry import trace_operations

        trace = trace_operations((source, source), [])
        values = np.zeros((grid, grid, grid * grid), dtype=np.float32)
        for row in range(grid):
            for column in range(grid):
                target = (
                    (grid - 1 - row, grid - 1 - column) if permute else (row, column)
                )
                values[target[0], target[1], row * grid + column] = 1.0
        return SpatialView(
            image_key="a.png",
            view_id=view_id,
            role=role,
            trace=trace,
            features={"blocks.11": values},
            pixel_geometric_coverage=np.ones((source, source), dtype=np.float32),
            pixel_sampling_valid=np.ones((source, source), dtype=bool),
            geometric_coverage=np.ones((grid, grid), dtype=np.float32),
            patch_sampling_valid=np.ones((grid, grid), dtype=bool),
        )

    reference = one_hot_view("reference", "eval-reference", permute=False)
    target = one_hot_view("tx10", "eval-transform", permute=True)
    plan = build_query_plan("a.png", reference, target, pair_id="tx10", seed=42)
    sampled = int(plan["directions"]["reference_to_target"]["sampled_count"])
    assert sampled > 0
    rows = _evaluate_pair_direction(
        reference,
        target,
        pair={
            "image_key": "a.png",
            "pair_id": "tx10",
            "patch_hw": [patch, patch],
            "directions": plan["directions"],
        },
        direction="reference_to_target",
        layer="blocks.11",
        query_limit=sampled,
    )
    assert len(rows) == sampled
    for row in rows:
        # One-hot features make the permuted cell the unique best candidate, so a
        # matcher that ignored the permutation would report the unpermuted cell.
        assert (int(row["predicted_row"]), int(row["predicted_column"])) == (
            grid - 1 - int(row["row"]),
            grid - 1 - int(row["column"]),
        )
        assert row["match_status"] == "matched"
        assert float(row["raw_error"]) > 0.0

    # Unpermuted, the same fixture must recover the true cell exactly.
    identity_rows = _evaluate_pair_direction(
        reference,
        one_hot_view("tx10", "eval-transform", permute=False),
        pair={
            "image_key": "a.png",
            "pair_id": "tx10",
            "patch_hw": [patch, patch],
            "directions": plan["directions"],
        },
        direction="reference_to_target",
        layer="blocks.11",
        query_limit=sampled,
    )
    for row in identity_rows:
        assert (int(row["predicted_row"]), int(row["predicted_column"])) == (
            int(row["row"]),
            int(row["column"]),
        )
        assert float(row["quantization_error"]) == pytest.approx(0.0, abs=1e-9)


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


def test_empty_evaluation_intersection_fails_before_any_output(tmp_path):
    """A selected subset outside the saved plan is rejected at preparation."""
    # Two images stored, but only one of them is in the saved evaluation plan.
    paired = run_extraction(tmp_path, spatial_eval_sample_size=1)
    product = Path(paired["out_dir"])
    reader = open_spatial_product(product)
    try:
        planned = set(reader.evaluation_plan["selected_keys"])
        unplanned = [key for key in reader.image_keys if key not in planned]
    finally:
        reader.close()
    assert unplanned
    subset = tmp_path / "unplanned.csv"
    subset.write_text("image\n" + "\n".join(unplanned) + "\n", encoding="utf-8")
    values = base_values(
        tmp_path,
        spatial_input=str(product),
        label_csv=str(subset),
        spatial_eval=True,
        out_dir=str(tmp_path / "read"),
    )
    with pytest.raises(SpatialRequestError, match="no stored planned pairs were selected"):
        prepare_spatial_request(
            values,
            explicit_options=frozenset({"spatial_input", "label_csv", "spatial_eval"}),
        )
    # Diagnostics-only reads of those same stored keys remain allowed.
    values["spatial_eval"] = False
    request = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "label_csv"})
    )
    request.close()


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
    from otuformer.spatial.evaluation import _evaluate_pair_direction
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
    branch = plan["directions"]["reference_to_target"]
    sampled = int(branch["sampled_count"])
    assert sampled > 0
    rows = _evaluate_pair_direction(
        reference,
        collapsed,
        pair={
            "image_key": "a.png",
            "pair_id": "rot15",
            "patch_hw": [16, 16],
            "directions": plan["directions"],
        },
        direction="reference_to_target",
        layer="blocks.11",
        query_limit=sampled,
    )
    # Every query stays in the report and the PCK denominator with zero hits,
    # and the failure is reported as unusable candidates rather than a lower
    # geometric-eligibility count.
    assert len(rows) == sampled
    for row in rows:
        assert row["match_status"] == "no_usable_candidate_features"
        assert row["invalid_candidate_count"] == row["candidate_count"] > 0
        for field in PCK_FIELDS:
            assert int(row[field]) == 0

    # A collapsed source query is a failed match inside the same denominator too.
    collapsed_reference = SpatialView(
        image_key=reference.image_key,
        view_id=reference.view_id,
        role=reference.role,
        trace=reference.trace,
        features={"blocks.11": np.zeros_like(reference.features["blocks.11"])},
        pixel_geometric_coverage=reference.pixel_geometric_coverage,
        pixel_sampling_valid=reference.pixel_sampling_valid,
        geometric_coverage=reference.geometric_coverage,
        patch_sampling_valid=reference.patch_sampling_valid,
    )
    source_rows = _evaluate_pair_direction(
        collapsed_reference,
        target,
        pair={
            "image_key": "a.png",
            "pair_id": "rot15",
            "patch_hw": [16, 16],
            "directions": plan["directions"],
        },
        direction="reference_to_target",
        layer="blocks.11",
        query_limit=sampled,
    )
    assert len(source_rows) == sampled
    assert all(row["match_status"] == "invalid_query_feature" for row in source_rows)


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


def test_subset_evaluation_follows_the_csv_selection_order(tmp_path):
    """A CSV subset is the report order: [b.png, a.png] must stay [b, a]."""
    result = run_extraction(tmp_path, spatial_eval_sample_size=2)
    product = Path(result["out_dir"])
    subset = tmp_path / "subset.csv"
    subset.write_text("image\nb.png\na.png\n", encoding="utf-8")
    values = base_values(
        tmp_path,
        spatial_input=str(product),
        label_csv=str(subset),
        out_dir=str(tmp_path / "ordered"),
    )
    request = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "label_csv"})
    )
    try:
        replayed = reevaluate_spatial(request)
    finally:
        request.close()
    with (Path(replayed["out_dir"]) / FILENAME_MATCHES).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [key for key in dict.fromkeys(row["image_key"] for row in rows)] == [
        "b.png",
        "a.png",
    ]


def test_summary_denominators_count_images_not_rows(tmp_path):
    result = run_extraction(tmp_path, spatial_eval_sample_size=2)
    evaluation = json.loads(
        (Path(result["out_dir"]) / FILENAME_EVALUATION).read_text(encoding="utf-8")
    )
    metrics = evaluation["metrics"]
    # No pooled cross-transform score is published.
    assert "overall" not in metrics
    assert "denominators" in metrics
    for layer, payload in metrics["per_layer"].items():
        assert "overall" not in payload
        for pair_id, transform in payload["transforms"].items():
            for direction in DIRECTIONS:
                branch = transform["directions"][direction]
                if not branch.get("available"):
                    continue
                # Two images, two directions: a row count of 4 is not 4 images.
                assert branch["contributing_images"] <= 2
                assert branch["rows"] >= branch["contributing_images"]
                assert branch["contributing_images"] == len(
                    branch["contributing_image_keys"]
                )


def test_diagnostics_only_read_reports_the_selector_overlay(tmp_path):
    """A selector CSV is a report overlay for every read, not only evaluation."""
    result = run_extraction(tmp_path, spatial_eval=False, out_dir=str(tmp_path / "plain"))
    subset = tmp_path / "labels.csv"
    subset.write_text("image,label\nb.png,beetle\n", encoding="utf-8")
    values = base_values(
        tmp_path,
        spatial_input=str(result["out_dir"]),
        label_csv=str(subset),
        spatial_eval=False,
        out_dir=str(tmp_path / "read"),
    )
    request = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "label_csv"})
    )
    try:
        replayed = reevaluate_spatial(request)
    finally:
        request.close()
    out = Path(replayed["out_dir"])
    diagnostics = json.loads((out / "spatial_diagnostics.json").read_text(encoding="utf-8"))
    overlay = diagnostics["sample_metadata"]
    assert overlay["status"] == "present"
    assert overlay["samples"]["b.png"] == {"label": "beetle"}
    envelope = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    assert envelope["effective"]["metadata_overlay"]["status"] == "present"
    assert envelope["effective"]["metadata_overlay"]["fields"] == ["label"]


def test_overlay_only_requires_the_selected_images_and_their_real_size(tmp_path):
    """An unselected image cannot make the selected overlay unavailable."""
    result = run_extraction(tmp_path, spatial_eval_sample_size=2)
    images = tmp_path / "images"
    subset = tmp_path / "subset.csv"
    subset.write_text("image\na.png\n", encoding="utf-8")
    # Remove the unselected image from the supplied root.
    (images / "b.png").unlink()
    values = base_values(
        tmp_path,
        spatial_input=str(result["out_dir"]),
        label_csv=str(subset),
        input_images_dir=str(images),
        spatial_eval=True,
        out_dir=str(tmp_path / "overlay"),
    )
    request = prepare_spatial_request(
        values,
        explicit_options=frozenset(
            {"spatial_input", "label_csv", "input_images_dir"}
        ),
    )
    try:
        replayed = reevaluate_spatial(request)
    finally:
        request.close()
    payload = json.loads(
        (Path(replayed["out_dir"]) / FILENAME_EVALUATION).read_text(encoding="utf-8")
    )
    assert payload["overlays"]["status"] == "available"
    assert payload["overlays"]["verified_images"] == ["a.png"]
    assert payload["overlays"]["missing_images"] == []
    # The documented matching-line figure is produced, not just a status flag.
    figures = payload["overlays"]["figures"]
    assert figures["status"] == "available"
    assert figures["figures"] and Path(figures["figures"][0]).is_file()


def test_overlay_rejects_bytes_that_hide_a_wrong_stored_size(tmp_path):
    from PIL import Image

    from otuformer.spatial.evaluation import _verify_overlay_images

    root = tmp_path / "product"
    build = run_extraction(tmp_path, spatial_eval_sample_size=1, out_dir=str(root))
    images = tmp_path / "wrong"
    images.mkdir()
    Image.new("RGB", (8, 8), (5, 6, 7)).save(images / "a.png")
    reader = open_spatial_product(Path(build["out_dir"]))
    try:
        # Match the stored hash so only the dimension check can catch it.
        import hashlib

        stored = reader._index[0]["sample"]  # noqa: SLF001 - reader owns its index
        payload = (images / "a.png").read_bytes()
        stored["image_sha256"] = hashlib.sha256(payload).hexdigest()
        stored["width"], stored["height"] = 999, 888
        report = _verify_overlay_images(reader, images)
    finally:
        reader.close()
    assert report["status"] == "unavailable"
    assert report["wrong_size_images"] == ["a.png"]


def test_plan_geometry_is_keyed_by_image_and_reports_eligibility_counts(tmp_path):
    result = run_extraction(tmp_path, spatial_eval_sample_size=2)
    payload = json.loads(
        (Path(result["out_dir"]) / FILENAME_EVALUATION).read_text(encoding="utf-8")
    )
    # Both images have a tx10 pair; neither may shadow the other.
    assert set(payload["plan_geometry"]) == {"a.png", "b.png"}
    for image_key, pairs in payload["plan_geometry"].items():
        assert "tx10" in pairs
        assert pairs["tx10"]["reference_signature"]["view_id"]
    eligibility = payload["geometric_eligibility"]
    assert set(eligibility["totals"]) == {
        "source_valid_count",
        "eligible_count",
        "sampled_count",
    }
    assert eligibility["totals"]["source_valid_count"] >= eligibility["totals"][
        "eligible_count"
    ]
    branch = next(
        branch
        for pairs in eligibility["pairs"].values()
        for pair in pairs.values()
        for branch in pair.values()
        if branch.get("available")
    )
    assert branch["eligibility_fraction"] is not None
    assert branch["sample_fraction"] is not None
    # A no-overlap branch keeps its reason instead of only `available: false`.
    unavailable = [
        branch
        for pairs in eligibility["pairs"].values()
        for pair in pairs.values()
        for branch in pair.values()
        if not branch.get("available")
    ]
    assert unavailable
    assert all(branch.get("unavailable_reason") for branch in unavailable)


def test_overlay_names_do_not_collide_and_keep_every_family(tmp_path):
    """`a_b.png` and `a/b.png` must not share a figure; families are not dropped."""
    from otuformer.spatial.evaluation import draw_match_overlays

    keys = (
        "a_b.png", "a/b.png", "a.jpg", "a.jpg.png/b.jpg",
        "/".join(["segment" * 8] * 5) + "/" + "昆虫" * 30 + ".png",
    )

    class FakeReader:
        image_keys = keys

    rows = []
    for index, key in enumerate(keys):
        for layer in ("blocks.0", "blocks.11"):
            for family in ("tx10", "rot15"):
                for query in range(6):
                    rows.append(
                        {
                            "image_key": key,
                            "layer": layer,
                            "pair_id": family,
                            "direction": "reference_to_target",
                            "query_index": query,
                            "normalized_error": str(0.1 + index + query * 0.01),
                            "true_x": 1.0,
                            "true_y": 1.0,
                            "predicted_x": 2.0,
                            "predicted_y": 2.0,
                        }
                    )
    images = tmp_path / "images"
    for key in keys:
        (images / key).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (9, 9, 9)).save(images / key)
    overlay = {"status": "available", "verified_images": list(keys)}
    report = draw_match_overlays(
        tmp_path / "out",
        rows,
        reader=FakeReader(),  # type: ignore[arg-type]
        overlay=overlay,
        images_dir=images,
    )
    assert report["status"] == "available"
    paths = report["figures"]
    assert len(paths) == len(keys)
    assert len(set(paths)) == len(keys)  # no silent overwrite
    assert all(Path(path).is_file() for path in paths)
    # `a.jpg` must not occupy the path `a.jpg.png/b.jpg` needs as a directory.
    assert all(Path(path).parent.name == "spatial_overlays" for path in paths)
    assert all(len(Path(path).name.encode("utf-8")) <= 255 for path in paths)
    # Each family/layer keeps its own worst five: the loss cap is per group.
    assert report["per_image_lines"] == 5


def test_core_read_protects_the_optional_overlay_images(tmp_path):
    """A core read whose out-dir holds the originals must be refused."""
    result = run_extraction(tmp_path, spatial_eval_sample_size=1)
    # Put an original image where the overlay figure would be written.
    overlays = tmp_path / "read" / "spatial_overlays"
    overlays.mkdir(parents=True)
    Image.new("RGB", (64, 64), (11, 22, 33)).save(overlays / "a.png")
    values = base_values(
        tmp_path,
        spatial_input=str(result["out_dir"]),
        input_images_dir=str(overlays),
        spatial_eval=True,
        out_dir=str(tmp_path / "read"),
    )
    with pytest.raises(SpatialRequestError, match="contains the input"):
        prepare_spatial_request(
            values, explicit_options=frozenset({"spatial_input", "input_images_dir"})
        )


def test_an_overlay_refuses_a_key_that_resolves_outside_the_image_root(tmp_path):
    """A stored key is data: even a canonical one must resolve under the root."""
    from otuformer.spatial.evaluation import _verify_overlay_images

    outside = tmp_path / "outside"
    outside.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(outside / "stolen.png")
    # A symlink inside the root that resolves outside is the same escape.
    (images / "linked.png").symlink_to(outside / "stolen.png")

    class FakeReader:
        image_keys = ("linked.png",)

        _index = [
            {
                "sample": {
                    "image_key": "linked.png",
                    "image_sha256": __import__("otuformer.data.provenance", fromlist=["x"]).sha256_bytes(
                        (outside / "stolen.png").read_bytes()
                    ),
                    "width": 16,
                    "height": 16,
                }
            }
        ]

    report = _verify_overlay_images(FakeReader(), images)  # type: ignore[arg-type]
    assert report["status"] == "unavailable"
    assert report["verified_images"] == []


def test_a_core_read_refuses_to_overwrite_an_existing_report(tmp_path):
    """The public Python path has no CLI `--overwrite`, so it must not clear."""
    from otuformer.spatial.extractor import (
        SpatialRequestError,
        prepare_spatial_request,
        reevaluate_spatial,
    )

    result = run_extraction(tmp_path, spatial_eval_sample_size=1)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    previous = report_dir / "spatial_results.csv"
    previous.write_text("image_key\nprevious-run\n", encoding="utf-8")
    values = base_values(
        tmp_path,
        spatial_input=str(result["out_dir"]),
        input_images_dir=None,
        out_dir=str(report_dir),
    )
    explicit = frozenset({"spatial_input", "out_dir"})
    prepared = prepare_spatial_request(values, explicit_options=explicit)
    try:
        with pytest.raises(SpatialRequestError, match="already holds"):
            reevaluate_spatial(prepared)
    finally:
        prepared.close()
    assert previous.read_text(encoding="utf-8") == "image_key\nprevious-run\n"


def test_a_core_read_records_its_own_failure_envelope(tmp_path, monkeypatch):
    """A failed read leaves this attempt's status, not an earlier one."""
    from otuformer.spatial.extractor import (
        prepare_spatial_request,
        reevaluate_spatial,
    )

    result = run_extraction(tmp_path, spatial_eval_sample_size=1)
    report_dir = tmp_path / "report"
    values = base_values(
        tmp_path,
        spatial_input=str(result["out_dir"]),
        input_images_dir=None,
        out_dir=str(report_dir),
    )
    prepared = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "out_dir"})
    )
    with monkeypatch.context() as patch:
        import otuformer.spatial.extractor as module

        def boom(*args, **kwargs):
            raise OSError("simulated write failure")

        patch.setattr(module, "write_csv_rows", boom)
        try:
            with pytest.raises(OSError):
                reevaluate_spatial(prepared)
        finally:
            prepared.close()
    envelope = json.loads((report_dir / "run_provenance.json").read_text())
    assert envelope["status"] == "failed"
    assert "simulated write failure" in json.dumps(envelope)


def test_a_core_read_refuses_a_planted_publication_symlink(tmp_path):
    """The attempt owns every name it writes, including the publication temporary."""
    from otuformer.spatial.extractor import (
        SpatialRequestError,
        prepare_spatial_request,
        reevaluate_spatial,
    )

    result = run_extraction(tmp_path, spatial_eval_sample_size=1)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    outside = tmp_path / "user.json"
    outside.write_text("keep-me", encoding="utf-8")
    (report_dir / "run_provenance.json.part").symlink_to(outside)
    values = base_values(
        tmp_path,
        spatial_input=str(result["out_dir"]),
        input_images_dir=None,
        out_dir=str(report_dir),
    )
    prepared = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "out_dir"})
    )
    try:
        with pytest.raises(SpatialRequestError, match="already holds"):
            reevaluate_spatial(prepared)
    finally:
        prepared.close()
    assert outside.read_text(encoding="utf-8") == "keep-me"


def test_a_failed_publication_still_records_the_attempt_status(tmp_path, monkeypatch):
    """The final hash/publish is part of the attempt, not outside its handler."""
    from otuformer.spatial.extractor import (
        prepare_spatial_request,
        reevaluate_spatial,
    )
    from otuformer.spatial import extractor as module

    result = run_extraction(tmp_path, spatial_eval_sample_size=1)
    report_dir = tmp_path / "report"
    values = base_values(
        tmp_path,
        spatial_input=str(result["out_dir"]),
        input_images_dir=None,
        out_dir=str(report_dir),
    )
    prepared = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "out_dir"})
    )

    def boom(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(module, "_artifact_rows", boom)
    try:
        with pytest.raises(OSError):
            reevaluate_spatial(prepared)
    finally:
        prepared.close()
    envelope = json.loads((report_dir / "run_provenance.json").read_text())
    assert envelope["status"] == "failed"
    assert "simulated publication failure" in json.dumps(envelope)


def test_preflight_does_not_read_an_image_that_symlinks_outside_the_root(tmp_path):
    """Containment is decided before the overlay preflight hashes anything."""
    from otuformer.spatial.extractor import prepare_spatial_request

    result = run_extraction(tmp_path, spatial_eval_sample_size=1)
    images = tmp_path / "linked-images"
    images.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(outside)
    (images / "a.png").symlink_to(outside)

    values = base_values(
        tmp_path,
        spatial_input=str(result["out_dir"]),
        input_images_dir=str(images),
        out_dir=str(tmp_path / "read"),
    )
    prepared = prepare_spatial_request(
        values, explicit_options=frozenset({"spatial_input", "input_images_dir"})
    )
    try:
        protected = {str(Path(item.path).resolve()) for item in prepared.protected_files}
        assert str(outside.resolve()) not in protected
    finally:
        prepared.close()


def test_published_artifacts_never_follow_a_symlink_out_of_the_directory(tmp_path):
    """The artifact scan registers this attempt's files, not a planted link."""
    from otuformer.spatial.extractor import _artifact_rows

    out = tmp_path / "out"
    out.mkdir()
    (out / "spatial_results.csv").write_text("a\n", encoding="utf-8")
    (out / "unrelated-regular.txt").write_text("keep", encoding="utf-8")
    (out / "run_provenance.json.part").write_text("stale", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (out / "unrelated.txt").symlink_to(outside)
    rows = _artifact_rows(out, include_log=True)
    assert [row["path"] for row in rows] == ["spatial_results.csv"]
    assert (out / "unrelated-regular.txt").read_text() == "keep"
    assert all(row["sha256"] for row in rows)
