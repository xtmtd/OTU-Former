"""Numeric and coverage diagnostics for stored spatial views.

Reports describe artifact health and geometry support, never learned quality.
Finite zero vectors and zero coverage are reported, not hidden; non-finite
arrays fail. Row order is a deterministic function of the logical records, so
two physical layouts holding the same arrays produce identical rows.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..constants import SPATIAL_ATTENTION_LIMITATION, SPATIAL_SUPPORT_VERSION
from ..vision.features import FeatureContract
from .records import (
    FEATURE_ARRAY_PREFIX,
    PATCH_COVERAGE,
    PATCH_VALID,
    PIXEL_COVERAGE,
    PIXEL_VALID,
    SpatialRecordError,
    SpatialView,
)

RESULT_FIELDS = (
    "sample_id",
    "image_key",
    "view_id",
    "role",
    "layer",
    "shape",
    "grid_hw",
    "dtype",
    "finite_count",
    "nonfinite_count",
    "zero_count",
    "near_zero_norm_count",
    "norm_min",
    "norm_mean",
    "norm_max",
    "channel_variance_mean",
    "coverage_min",
    "coverage_mean",
    "coverage_full_valid_fraction",
    "pixel_valid_fraction",
    "patch_valid_fraction",
    "support_version",
)


def _eligibility_counts(
    plan: Mapping[str, Any], selected: Sequence[str] | None = None
) -> dict[str, Any]:
    """Stored geometric eligibility, restricted to the selected keys.

    The evaluation summary publishes per image/pair/direction counts, both
    fractions and the unavailable reason; diagnostics must publish the same shape
    over the same restriction, or a diagnostic total would include samples the
    report does not cover.
    """
    from .evaluation import eligibility_report

    keys = list(selected) if selected is not None else list(plan.get("selected_keys", []))
    return eligibility_report(plan, keys)


def diagnose_views(
    views: Iterable[SpatialView],
    contract: FeatureContract,
    *,
    plan: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic per sample/view/layer rows and a summary.

    ``plan`` is the product's stored evaluation plan when it has one; the summary
    then states how many keys and query pairs it covers, so a report never implies
    correspondence data a product does not carry.
    """
    rows: list[dict[str, Any]] = []
    samples: dict[str, int] = {}
    view_count = 0
    failures: list[str] = []
    # One pass: the keys come from the stream, so diagnostics never hold every
    # decoded feature/support array of the product at once.
    selected_keys: list[str] = []
    seen_keys: set[str] = set()
    for view in views:
        view_count += 1
        if view.image_key not in seen_keys:
            seen_keys.add(view.image_key)
            selected_keys.append(view.image_key)
        samples[view.image_key] = samples.get(view.image_key, 0) + 1
        coverage = np.asarray(view.pixel_geometric_coverage, dtype=np.float64)
        pixel_valid = np.asarray(view.pixel_sampling_valid, dtype=bool)
        patch_coverage = np.asarray(view.geometric_coverage, dtype=np.float64)
        patch_valid = np.asarray(view.patch_sampling_valid, dtype=bool)
        for layer in contract.layer_paths:
            array = np.asarray(view.features[layer])
            if not np.all(np.isfinite(array)):
                failures.append(f"{view.image_key}:{view.view_id}:{layer}")
                continue
            values = array.astype(np.float64, copy=False)
            flat = values.reshape(-1, values.shape[-1])
            norms = np.linalg.norm(flat, axis=-1)
            rows.append(
                {
                    "sample_id": None,
                    "image_key": view.image_key,
                    "view_id": view.view_id,
                    "role": view.role,
                    "layer": layer,
                    "shape": "x".join(str(int(v)) for v in array.shape),
                    "grid_hw": f"{int(array.shape[0])}x{int(array.shape[1])}",
                    "dtype": str(array.dtype),
                    "finite_count": int(flat.shape[0]),
                    "nonfinite_count": 0,
                    "zero_count": int(np.count_nonzero(norms == 0.0)),
                    "near_zero_norm_count": int(np.count_nonzero(norms <= 1e-12)),
                    "norm_min": float(norms.min()),
                    "norm_mean": float(norms.mean()),
                    "norm_max": float(norms.max()),
                    "channel_variance_mean": float(np.var(flat, axis=-1).mean()),
                    "coverage_min": float(coverage.min()),
                    "coverage_mean": float(coverage.mean()),
                    "coverage_full_valid_fraction": float(
                        np.count_nonzero(coverage >= 1.0) / coverage.size
                    ),
                    "pixel_valid_fraction": float(
                        np.count_nonzero(pixel_valid) / pixel_valid.size
                    ),
                    "patch_valid_fraction": (
                        float(np.count_nonzero(patch_valid) / patch_valid.size)
                        if patch_valid.size
                        else 0.0
                    ),
                    "support_version": SPATIAL_SUPPORT_VERSION,
                }
            )
    if failures:
        raise SpatialRecordError(
            "non-finite feature arrays are not a diagnostic: "
            + ", ".join(failures[:3])
        )
    summary = {
        "schema_version": 1,
        "evaluation_plan": (
            {
                "status": "present",
                "protocol": plan.get("protocol"),
                "seed": plan.get("seed"),
                "selected_keys": len(plan.get("selected_keys", [])),
                "pairs": len(plan.get("pairs", [])),
                "query_limit": plan.get("query_limit"),
                "image_limit": plan.get("image_limit"),
                # The plan separates "no source-valid centre" from "no overlap"
                # from a matching failure. Publishing the stored counts keeps that
                # distinction visible without running the matcher.
                "eligibility": _eligibility_counts(plan, selected_keys),
            }
            if plan is not None
            else {
                "status": "absent",
                "reason": "this product stores no paired views or evaluation plan",
            }
        ),
        "samples": len(samples),
        "views": view_count,
        "layers": list(contract.layer_paths),
        "support_version": SPATIAL_SUPPORT_VERSION,
        "rows": len(rows),
        "counts": {
            "samples": len(samples),
            "views": view_count,
            "layer_rows": len(rows),
        },
        "averages": _averages(rows),
        "limitations": [
            SPATIAL_ATTENTION_LIMITATION,
            "Coverage fractions describe geometric source support, not "
            "foreground or anatomical homology.",
            "Numeric diagnostics say nothing about learned correspondence; "
            "correspondence results are reported separately.",
        ],
    }
    return rows, summary


def _averages(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    layers = sorted({str(row["layer"]) for row in rows})
    averages: dict[str, Any] = {}
    for pattern in layers:
        subset = [row for row in rows if row["layer"] == pattern]
        if not subset:
            continue
        averages[pattern] = {
            "norm_mean": float(
                np.mean([float(row["norm_mean"]) for row in subset])
            ),
            "coverage_mean": float(
                np.mean([float(row["coverage_mean"]) for row in subset])
            ),
            "pixel_valid_fraction_mean": float(
                np.mean([float(row["pixel_valid_fraction"]) for row in subset])
            ),
            "patch_valid_fraction_mean": float(
                np.mean([float(row["patch_valid_fraction"]) for row in subset])
            ),
            "zero_count": int(sum(int(row["zero_count"]) for row in subset)),
            "near_zero_norm_count": int(
                sum(int(row["near_zero_norm_count"]) for row in subset)
            ),
        }
    return averages


def array_role_names() -> tuple[str, ...]:
    return (
        f"{FEATURE_ARRAY_PREFIX}<layer>",
        PIXEL_COVERAGE,
        PIXEL_VALID,
        PATCH_COVERAGE,
        PATCH_VALID,
    )
