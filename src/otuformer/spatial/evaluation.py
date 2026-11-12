"""Frozen correspondence matching and aggregation from stored pairs.

Everything here consumes saved arrays and the saved plan: no model, no image, no
inference and no mutation of the source product. The matcher is deliberately
simple and fixed — cosine similarity over every valid target grid centre — so a
score can be reproduced from the bytes alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..constants import (
    SPATIAL_ATTENTION_LIMITATION,
    SPATIAL_PCK_THRESHOLDS,
    SPATIAL_PROTOCOL,
    SPATIAL_QUERY_LIMIT,
    SPATIAL_SCHEMA_VERSION,
    SPATIAL_SUPPORT_VERSION,
)
from ..data.provenance import sha256_bytes
from .protocol import DIRECTIONS, geometry_signature, grid_center_point
from .records import (
    FILENAME_EVALUATION,
    FILENAME_MATCHES,
    SpatialRecordError,
    SpatialView,
)
from .storage import SpatialReader, write_csv_rows
from ..vision.geometry_support import transform_points

MATCH_FIELDS = (
    "image_key",
    "pair_id",
    "direction",
    "layer",
    "query_index",
    "row",
    "column",
    "match_status",
    "candidate_count",
    "invalid_candidate_count",
    "predicted_row",
    "predicted_column",
    "true_x",
    "true_y",
    "predicted_x",
    "predicted_y",
    "raw_error",
    "normalized_error",
    "quantization_error",
    "pck_0p01",
    "pck_0p02",
    "pck_0p05",
)

PCK_FIELDS = tuple(
    f"pck_0p{str(int(round(threshold * 100))).zfill(2)}"
    for threshold in SPATIAL_PCK_THRESHOLDS
)

MISSING_PLAN_REASON = "missing_paired_views"


def _unit(vectors: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalized vectors plus the mask of usable rows."""
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1)
    usable = norms > epsilon
    safe = np.where(usable, norms, 1.0)
    return values / safe[..., None], usable


def _diagonal(source_wh: Sequence[int]) -> float:
    return float(math.hypot(float(source_wh[0]), float(source_wh[1])))


def _evaluate_pair_direction(
    reference: SpatialView,
    target: SpatialView,
    *,
    pair: Mapping[str, Any],
    direction: str,
    layer: str,
    query_limit: int,
) -> list[dict[str, Any]]:
    """Match every planned query of one pair/direction/layer."""
    entry = pair["directions"][direction]
    source = reference if direction == "reference_to_target" else target
    destination = target if direction == "reference_to_target" else reference
    patch_hw = (int(pair["patch_hw"][0]), int(pair["patch_hw"][1]))
    source_grid = np.asarray(source.features[layer], dtype=np.float64)
    target_grid = np.asarray(destination.features[layer], dtype=np.float64)
    target_valid = np.asarray(destination.patch_sampling_valid, dtype=bool)

    rows, columns = np.nonzero(target_valid)
    candidate_vectors = target_grid[rows, columns] if rows.size else np.zeros((0, 1))
    candidate_unit, candidate_usable = _unit(
        candidate_vectors, float(1e-12)
    )
    candidate_count = int(rows.size)
    invalid_candidates = int(candidate_count - int(candidate_usable.sum()))

    diagonal = _diagonal(reference.trace.source_wh)
    quantized: list[dict[str, Any]] = []
    for index, query in enumerate(entry["queries"]):
        row, column = int(query["row"]), int(query["column"])
        vector = source_grid[row, column]
        norm = float(np.linalg.norm(vector))
        record: dict[str, Any] = {
            "image_key": pair["image_key"],
            "pair_id": pair["pair_id"],
            "direction": direction,
            "layer": layer,
            "query_index": index,
            "row": row,
            "column": column,
            "match_status": "unmatched",
            "candidate_count": candidate_count,
            "invalid_candidate_count": invalid_candidates,
            "predicted_row": "",
            "predicted_column": "",
            "true_x": "",
            "true_y": "",
            "predicted_x": "",
            "predicted_y": "",
            "raw_error": "",
            "normalized_error": "",
            "quantization_error": "",
            **{field: 0 for field in PCK_FIELDS},
        }
        # Ground truth is the continuous mapped point, always in original coords.
        truth_view = np.asarray([query["xy_target"]], dtype=np.float64)
        truth_original = transform_points(
            destination.trace.view_to_source, truth_view
        )[0]
        record["true_x"] = float(truth_original[0])
        record["true_y"] = float(truth_original[1])
        if candidate_count:
            centers = np.stack(
                [
                    np.asarray(
                        [grid_center_point(int(r), int(c), patch_hw)], dtype=np.float64
                    )[0]
                    for r, c in zip(rows.tolist(), columns.tolist())
                ]
            )
            centers_original = transform_points(
                destination.trace.view_to_source, centers
            )
            distance = np.linalg.norm(
                centers_original - truth_original[None, :], axis=-1
            )
            best = int(np.argmin(distance))
            record["quantization_error"] = float(distance[best] / diagonal)
        if norm <= 1e-12:
            record["match_status"] = "invalid_query_feature"
            quantized.append(record)
            continue
        if candidate_count == 0 or not candidate_usable.any():
            record["match_status"] = "no_usable_candidate_features"
            quantized.append(record)
            continue
        similarities = candidate_unit @ (vector / norm)
        similarities = np.where(candidate_usable, similarities, -np.inf)
        best_index = int(np.argmax(similarities))
        predicted_row = int(rows[best_index])
        predicted_column = int(columns[best_index])
        predicted_view = np.asarray(
            [grid_center_point(predicted_row, predicted_column, patch_hw)],
            dtype=np.float64,
        )
        predicted_original = transform_points(
            destination.trace.view_to_source, predicted_view
        )[0]
        raw_error = float(
            np.linalg.norm(predicted_original - truth_original)
        )
        normalized = raw_error / diagonal
        record.update(
            {
                "match_status": "matched",
                "predicted_row": predicted_row,
                "predicted_column": predicted_column,
                "predicted_x": float(predicted_original[0]),
                "predicted_y": float(predicted_original[1]),
                "raw_error": raw_error,
                "normalized_error": float(normalized),
            }
        )
        for threshold, field in zip(SPATIAL_PCK_THRESHOLDS, PCK_FIELDS):
            record[field] = int(normalized <= threshold)
        quantized.append(record)
    if query_limit and len(quantized) > query_limit:
        raise SpatialRecordError(
            f"plan holds more queries than the frozen limit for {pair['image_key']!r}"
            f"/{pair['pair_id']!r}/{direction}: {len(quantized)} > {query_limit}"
        )
    return quantized


def evaluate_spatial(
    reader: SpatialReader, *, selected_keys: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the saved plan for the selected keys, from stored arrays only."""
    plan = reader.evaluation_plan
    if plan is None:
        raise SpatialRecordError(
            f"{MISSING_PLAN_REASON}: this product stores no paired views or "
            "evaluation plan, so correspondence cannot be evaluated"
        )
    planned = list(plan.get("selected_keys", []))
    selected = [key for key in selected_keys if key in set(planned)]
    not_planned = [key for key in selected_keys if key not in set(planned)]
    if not selected:
        raise SpatialRecordError(
            "no stored planned pairs were selected; this product's evaluation plan "
            f"covers {planned[:3]}{'...' if len(planned) > 3 else ''} and the "
            "selection contains none of them"
        )
    query_limit = int(plan.get("query_limit", SPATIAL_QUERY_LIMIT))
    rows: list[dict[str, Any]] = []
    evaluated_keys: list[str] = []
    for pair in plan["pairs"]:
        image_key = pair["image_key"]
        if image_key not in set(selected):
            continue
        if image_key not in evaluated_keys:
            evaluated_keys.append(image_key)
        reference = reader.get_view(image_key, pair["reference_view"])
        target = reader.get_view(image_key, pair["target_view"])
        for direction in DIRECTIONS:
            for layer in reader.contract.layer_paths:
                rows.extend(
                    _evaluate_pair_direction(
                        reference,
                        target,
                        pair=pair,
                        direction=direction,
                        layer=layer,
                        query_limit=query_limit,
                    )
                )
    summary = _aggregate(
        rows,
        plan=plan,
        layer_paths=list(reader.contract.layer_paths),
        selected=selected,
        evaluated=evaluated_keys,
        not_planned=not_planned,
        planned=planned,
    )
    return rows, summary


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    layer_paths: Sequence[str],
    selected: Sequence[str],
    evaluated: Sequence[str],
    not_planned: Sequence[str],
    planned: Sequence[str],
) -> dict[str, Any]:
    """Per-image results first, then a macro average over contributing images."""
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["image_key"], row["pair_id"], row["direction"], row["layer"])
        groups.setdefault(key, []).append(row)

    image_rows: list[dict[str, Any]] = []
    for (image_key, pair_id, direction, layer), group in groups.items():
        finite = [
            float(row["normalized_error"])
            for row in group
            if row["normalized_error"] != ""
        ]
        entry = {
            "image_key": image_key,
            "pair_id": pair_id,
            "direction": direction,
            "layer": layer,
            "queries": len(group),
            "finite_error_queries": len(finite),
            "failed_queries": len(group) - len(finite),
            "error_mean": float(np.mean(finite)) if finite else None,
            "error_median": float(np.median(finite)) if finite else None,
            "raw_error_mean": float(
                np.mean(
                    [float(row["raw_error"]) for row in group if row["raw_error"] != ""]
                )
            )
            if finite
            else None,
            **{
                field: float(
                    np.mean([float(row[field]) for row in group])
                )
                for field in PCK_FIELDS
            },
        }
        image_rows.append(entry)

    def macro(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        contributing = [row for row in subset if row["queries"] > 0]
        finite = [row for row in contributing if row["error_mean"] is not None]
        return {
            "contributing_images": len(contributing),
            "images_with_finite_errors": len(finite),
            "queries": int(sum(row["queries"] for row in contributing)),
            "finite_error_queries": int(
                sum(row["finite_error_queries"] for row in contributing)
            ),
            "failed_queries": int(sum(row["failed_queries"] for row in contributing)),
            "error_mean": float(np.mean([row["error_mean"] for row in finite]))
            if finite
            else None,
            "mean_of_image_medians": float(
                np.mean([row["error_median"] for row in finite])
            )
            if finite
            else None,
            **{
                field: float(np.mean([row[field] for row in contributing]))
                if contributing
                else None
                for field in PCK_FIELDS
            },
        }

    per_layer: dict[str, Any] = {}
    for layer in layer_paths:
        layer_rows = [row for row in image_rows if row["layer"] == layer]
        transforms: dict[str, Any] = {}
        for pair_id in [pair["pair_id"] for pair in plan.get("pairs", [])]:
            pair_rows = [row for row in layer_rows if row["pair_id"] == pair_id]
            directions: dict[str, Any] = {}
            for direction in DIRECTIONS:
                subset = [row for row in pair_rows if row["direction"] == direction]
                if not subset:
                    directions[direction] = {"available": False}
                    continue
                directions[direction] = {"available": True, **macro(subset)}
            both = all(
                directions[direction].get("available") for direction in DIRECTIONS
            )
            combined = None
            if both:
                combined = {
                    "available": True,
                    "error_mean": float(
                        np.mean(
                            [
                                directions[direction]["error_mean"]
                                for direction in DIRECTIONS
                                if directions[direction]["error_mean"] is not None
                            ]
                        )
                    )
                    if all(
                        directions[direction]["error_mean"] is not None
                        for direction in DIRECTIONS
                    )
                    else None,
                    **{
                        field: float(
                            np.mean(
                                [
                                    directions[direction][field]
                                    for direction in DIRECTIONS
                                ]
                            )
                        )
                        for field in PCK_FIELDS
                    },
                }
            transforms[pair_id] = {
                "directions": directions,
                "combined": combined
                if combined is not None
                else {"available": False, "reason": "one direction is unavailable"},
            }
        per_layer[layer] = {
            "transforms": transforms,
            "overall": macro([row for row in layer_rows]),
        }

    return {
        "protocol": SPATIAL_PROTOCOL,
        "schema_version": SPATIAL_SCHEMA_VERSION,
        "support_version": SPATIAL_SUPPORT_VERSION,
        "seed": int(plan.get("seed", 0)),
        "metrics": {
            "pck_thresholds": list(SPATIAL_PCK_THRESHOLDS),
            "error_unit": "euclidean_pixels_over_original_diagonal",
            "aggregation": (
                "per-image mean/median first, then a macro average over contributing "
                "images; PCK is over all geometrically selected queries"
            ),
            "per_layer": per_layer,
        },
        "counts": {
            "selected_keys": len(selected),
            "planned_keys": len(planned),
            "evaluated_keys": len(evaluated),
            "not_in_saved_plan": len(not_planned),
            "queries": len(rows),
            "finite_error_queries": int(
                sum(1 for row in rows if row["normalized_error"] != "")
            ),
            "failed_queries": int(
                sum(1 for row in rows if row["normalized_error"] == "")
            ),
            "invalid_candidate_features": int(
                sum(int(row["invalid_candidate_count"]) for row in rows)
            ),
        },
        "not_in_saved_evaluation_plan": list(not_planned),
        "identity_checks_excluded": (
            "identity forward repeats, transform round trips and store/read equality "
            "are engineering fixtures, not correspondence entries"
        ),
        "limitations": [
            SPATIAL_ATTENTION_LIMITATION,
            "Candidates are grid centres, so prediction error includes the candidate "
            "quantization lower bound reported per query.",
            "Frozen same-image correspondence cannot establish cross-specimen "
            "anatomical identity, part segmentation or biological benefit.",
        ],
    }


def failure_examples(
    rows: Sequence[Mapping[str, Any]], *, per_kind: int = 5
) -> dict[str, Any]:
    """Deterministic worst finite errors plus status examples, per layer/family."""
    examples: dict[str, Any] = {}
    families = sorted({(row["layer"], row["pair_id"]) for row in rows})
    for layer, pair_id in families:
        family = [
            row
            for row in rows
            if row["layer"] == layer and row["pair_id"] == pair_id
        ]
        # Deterministic under any input order: the error is primary, and every
        # tie is broken by stable record fields rather than by list position.
        finite = sorted(
            (row for row in family if row["normalized_error"] != ""),
            key=lambda row: (
                -float(row["normalized_error"]),
                row["image_key"],
                row["direction"],
                row["layer"],
                int(row["query_index"]),
            ),
        )[:per_kind]
        failed = sorted(
            (row for row in family if row["normalized_error"] == ""),
            key=lambda row: (
                row["image_key"],
                row["direction"],
                row["layer"],
                int(row["query_index"]),
            ),
        )[:per_kind]
        examples[f"{layer}|{pair_id}"] = {
            "worst_finite_errors": [
                {
                    field: row[field]
                    for field in (
                        "image_key",
                        "direction",
                        "query_index",
                        "row",
                        "column",
                        "true_x",
                        "true_y",
                        "predicted_x",
                        "predicted_y",
                        "raw_error",
                        "normalized_error",
                    )
                }
                for row in finite
            ],
            "status_examples": [
                {
                    field: row[field]
                    for field in (
                        "image_key",
                        "direction",
                        "query_index",
                        "row",
                        "column",
                        "match_status",
                        "candidate_count",
                    )
                }
                for row in failed
            ],
        }
    return examples


def _verify_overlay_images(
    reader: SpatialReader, images_dir: Path | None
) -> dict[str, Any]:
    """Verify optional raw images before claiming an overlay is available."""
    if images_dir is None:
        return {
            "status": "not_requested",
            "reason": "no image root was supplied; coordinates and numbers are kept",
        }
    root = Path(images_dir).expanduser()
    if not root.is_dir():
        return {"status": "unavailable", "reason": "the image root does not exist"}
    verified: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for entry in reader._index:  # noqa: SLF001 - reader owns its index
        sample = entry["sample"]
        key = sample["image_key"]
        if key in verified or key in missing or key in mismatched:
            continue
        path = root / key
        if not path.is_file():
            missing.append(key)
            continue
        if sha256_bytes(path.read_bytes()) != sample["image_sha256"]:
            mismatched.append(key)
            continue
        verified.append(key)
    available = bool(verified) and not missing and not mismatched
    return {
        "status": "available" if available else "unavailable",
        "verified_images": verified,
        "missing_images": missing,
        "mismatched_images": mismatched,
        "reason": None
        if available
        else "overlay needs every selected image with matching bytes and dimensions",
    }


def write_spatial_evaluation(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    reader: SpatialReader,
    images_dir: Path | None = None,
) -> list[Path]:
    """Write ``spatial_matches.csv`` and ``spatial_evaluation.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matches_path = out_dir / FILENAME_MATCHES
    write_csv_rows(matches_path, rows, MATCH_FIELDS)
    payload = {
        **summary,
        "failure_examples": failure_examples(rows),
        "overlays": _verify_overlay_images(reader, images_dir),
        "source_product": {
            "product_id": reader.product_id,
            "layout": reader.manifest["layout"],
            "verification_scope": reader.verification_scope().to_json(),
        },
        "plan_geometry": {
            pair["pair_id"]: {
                "reference": pair["geometry"]["reference"]["view_id"],
                "target": pair["geometry"]["target"]["view_id"],
                "reference_signature": pair["geometry"]["reference"],
                "target_signature": pair["geometry"]["target"],
            }
            for pair in reader.evaluation_plan.get("pairs", [])
        }
        if reader.evaluation_plan is not None
        else {},
    }
    evaluation_path = out_dir / FILENAME_EVALUATION
    evaluation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return [matches_path, evaluation_path]
