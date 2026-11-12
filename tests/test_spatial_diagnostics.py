"""Numeric diagnostics for stored spatial views."""

from __future__ import annotations

import numpy as np
import pytest

from otuformer.constants import SPATIAL_ATTENTION_LIMITATION
from otuformer.spatial.diagnostics import RESULT_FIELDS, diagnose_views
from otuformer.spatial.records import SpatialRecordError, SpatialView
from otuformer.vision.geometry import GeometryOp, trace_operations


def contract():
    from otuformer.vision.features import FeatureContract, LayerSpec

    return FeatureContract(
        layers=(
            LayerSpec(path="blocks.5", block_index=5, grid_hw=(2, 2), channels=4),
            LayerSpec(path="blocks.11", block_index=11, grid_hw=(2, 2), channels=4),
        ),
        input_hw=(32, 32),
        patch_hw=(16, 16),
        prefix_tokens=1,
        norm="backbone",
        norm_module={"path": "norm", "qualname": "timm.layers.norm.LayerNorm"},
        l2=False,
        epsilon=1e-12,
        normalization={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        model_identity={"architecture": "test"},
    )


def view(
    *,
    image_key="a.png",
    view_id="base",
    role="base",
    seed=0,
    constant=None,
    coverage=1.0,
) -> SpatialView:
    generator = np.random.default_rng(seed)
    layers = {"blocks.5": None, "blocks.11": None}
    for path in layers:
        values = np.zeros((2, 2, 4), dtype=np.float32)
        values[:] = constant if constant is not None else generator.normal(size=(2, 2, 4))
        layers[path] = values.astype(np.float32)
    trace = trace_operations(
        (64, 64),
        [
            GeometryOp(
                "resize",
                {
                    "from_wh": [64, 64],
                    "to_wh": [32, 32],
                    "interpolation": "bicubic",
                    "antialias": True,
                },
            )
        ],
    )
    return SpatialView(
        image_key=image_key,
        view_id=view_id,
        role=role,
        trace=trace,
        features=layers,
        pixel_geometric_coverage=np.full((32, 32), coverage, dtype=np.float32),
        pixel_sampling_valid=np.full((32, 32), coverage > 0.0, dtype=bool),
        geometric_coverage=np.full((2, 2), coverage, dtype=np.float32),
        patch_sampling_valid=np.full((2, 2), coverage > 0.0, dtype=bool),
    )


def test_rows_are_metadata_neutral_and_deterministic():
    views = [view(image_key="a.png"), view(image_key="b.png", view_id="eval-1", role="eval-transform", seed=1)]
    rows, summary = diagnose_views(views, contract())
    assert [row["image_key"] for row in rows] == ["a.png", "a.png", "b.png", "b.png"]
    assert [row["layer"] for row in rows] == ["blocks.5", "blocks.11", "blocks.5", "blocks.11"]
    for row in rows:
        assert set(row) == set(RESULT_FIELDS)
        assert row["finite_count"] == 4
        assert row["nonfinite_count"] == 0
        assert row["dtype"] == "float32"
        assert row["grid_hw"] == "2x2"
        assert row["shape"] == "2x2x4"
        assert row["coverage_full_valid_fraction"] == pytest.approx(1.0)
        assert row["patch_valid_fraction"] == pytest.approx(1.0)
        assert row["support_version"] == "conservative-resampling-v1"
    assert summary["samples"] == 2
    assert summary["views"] == 2
    assert summary["rows"] == 4
    assert summary["layers"] == ["blocks.5", "blocks.11"]
    assert SPATIAL_ATTENTION_LIMITATION in summary["limitations"]
    assert "blocks.5" in summary["averages"]

    # Ordering is a function of the records, not of evaluation time.
    again, _ = diagnose_views(views, contract())
    assert [row["image_key"] for row in again] == [row["image_key"] for row in rows]


def test_constant_and_zero_grids_are_reported_without_invented_accuracy():
    rows, summary = diagnose_views([view(constant=0.0, coverage=0.0)], contract())
    assert all(row["zero_count"] == 4 for row in rows)
    assert all(row["near_zero_norm_count"] == 4 for row in rows)
    assert all(row["norm_max"] == 0.0 for row in rows)
    assert all(row["coverage_min"] == 0.0 for row in rows)
    assert all(row["pixel_valid_fraction"] == 0.0 for row in rows)
    assert all(row["patch_valid_fraction"] == 0.0 for row in rows)
    # Nothing here fabricates a score or a quality value.
    assert "score" not in summary
    assert summary["averages"]["blocks.5"]["zero_count"] == 4

    constant_rows, _ = diagnose_views([view(constant=3.5)], contract())
    assert all(row["channel_variance_mean"] == pytest.approx(0.0) for row in constant_rows)
    assert all(row["norm_min"] == pytest.approx(np.sqrt(4 * 3.5**2)) for row in constant_rows)


def test_nonfinite_features_fail_instead_of_being_summarized():
    bad = view()
    broken = SpatialView(
        image_key=bad.image_key,
        view_id=bad.view_id,
        role=bad.role,
        trace=bad.trace,
        features={
            "blocks.5": np.full((2, 2, 4), np.inf, dtype=np.float32),
            "blocks.11": bad.features["blocks.11"],
        },
        pixel_geometric_coverage=bad.pixel_geometric_coverage,
        pixel_sampling_valid=bad.pixel_sampling_valid,
        geometric_coverage=bad.geometric_coverage,
        patch_sampling_valid=bad.patch_sampling_valid,
    )
    with pytest.raises(SpatialRecordError, match="non-finite feature arrays"):
        diagnose_views([broken], contract())


def test_limitation_text_is_fixed():
    _, summary = diagnose_views([view()], contract())
    assert summary["limitations"][0] == SPATIAL_ATTENTION_LIMITATION
    assert (
        "ViT global attention can still propagate influence from invalid tokens."
        in SPATIAL_ATTENTION_LIMITATION
    )


def test_summary_states_the_stored_plan_status():
    views = [view()]
    _, absent = diagnose_views(views, contract())
    assert absent["evaluation_plan"]["status"] == "absent"

    plan = {
        "protocol": "spatial-geometry-v1",
        "seed": 42,
        "selected_keys": ["a.png"],
        "pairs": [{"pair_id": "tx10"}],
        "query_limit": 64,
    }
    _, present = diagnose_views(views, contract(), plan=plan)
    entry = present["evaluation_plan"]
    assert entry["status"] == "present"
    assert entry["protocol"] == "spatial-geometry-v1"
    assert entry["seed"] == 42
    assert entry["selected_keys"] == 1
    assert entry["pairs"] == 1
