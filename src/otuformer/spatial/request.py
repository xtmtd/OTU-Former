"""Stdlib-only mode/applicability rules for the spatial product selector.

Shared by the CLI constraints, the schema and the core preparation step so an
explicitly supplied option is judged the same way everywhere. An explicitly
supplied option counts even when its value equals the CLI default; an omitted
default for another mode is inactive, not a conflict.
"""

from __future__ import annotations

from typing import Mapping

from ..constants import FEATURE_NORM_CHOICES, OUTPUT_KIND_CHOICES, SPATIAL_LAYOUT_CHOICES

#: Options that only ever describe an embedding vector or its evaluation.
EMBEDDING_ONLY_OPTIONS = (
    "token_mode",
    "topk_patches",
    "attention_pooling_type",
    "attention_pooling_epochs",
    "use_projector_output",
    "metrics_sample_size",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "visualize_class_number",
    "disable_umap",
)

#: Options that only ever describe a spatial product.
SPATIAL_ONLY_OPTIONS = (
    "spatial_input",
    "feature_layers",
    "feature_norm",
    "feature_l2_normalize",
    "spatial_layout",
    "spatial_eval",
    "spatial_eval_sample_size",
)

#: Rejected in artifact read mode: a stored product fixes these contracts.
READ_REJECTED_OPTIONS = (
    "checkpoint",
    "onnx_path",
    "model_name",
    "use_student",
    "feature_layers",
    "feature_norm",
    "feature_l2_normalize",
    "spatial_layout",
    "extract_size",
    "eval_transform",
    "coco_instances",
    "coco_keypoints",
    "coco_panoptic",
    "panoptic_masks_dir",
    "dump_model_structure",
    "spatial_eval_sample_size",
)

MODE_EMBEDDING = "embedding"
MODE_CHECKPOINT = "checkpoint"
MODE_ONNX = "onnx"
MODE_READ = "read"


def spatial_mode(values: Mapping[str, object]) -> str:
    """Resolve the run mode from the request itself, never from a default."""
    output_kind = values.get("output_kind") or "embedding"
    if output_kind == "embedding":
        return MODE_EMBEDDING
    if values.get("spatial_input"):
        return MODE_READ
    if values.get("onnx_path"):
        return MODE_ONNX
    return MODE_CHECKPOINT


def _given(values: Mapping[str, object], name: str) -> bool:
    value = values.get(name)
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    return True


def _explicit(explicit_options: frozenset[str] | set[str], name: str) -> bool:
    return name in explicit_options


def _option(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def validate_extract_mode_options(
    values: Mapping[str, object], *, explicit_options: frozenset[str]
) -> list[str]:
    """Check one extract request against design §3's applicability matrix.

    Returns unresolved notes (conditions that can only be decided against real
    inputs) and raises ``ValueError`` for a deterministic conflict, so a bad
    request is rejected before any output directory is prepared.
    """
    output_kind = values.get("output_kind") or "embedding"
    if output_kind not in OUTPUT_KIND_CHOICES:
        raise ValueError(
            f"unknown --output-kind {output_kind!r}; expected one of "
            f"{OUTPUT_KIND_CHOICES}"
        )
    notes: list[str] = []
    mode = spatial_mode(values)

    if mode == MODE_EMBEDDING:
        if _explicit(explicit_options, "spatial_input"):
            raise ValueError(
                "--spatial-input reads an existing spatial product and requires "
                "--output-kind spatial; add it, or drop --spatial-input for the "
                "default embedding extraction"
            )
        rejected = [
            name for name in SPATIAL_ONLY_OPTIONS if _explicit(explicit_options, name)
        ]
        if rejected:
            raise ValueError(
                "these options are spatial-only and cannot be used with "
                "--output-kind embedding: "
                + ", ".join(_option(name) for name in rejected)
            )
        return notes

    # Spatial modes: embedding-vector machinery has no meaning here.
    rejected = [
        name for name in EMBEDDING_ONLY_OPTIONS if _explicit(explicit_options, name)
    ]
    if rejected:
        raise ValueError(
            "spatial output reads complete backbone-block grids, so these "
            "embedding-only options cannot be used: "
            + ", ".join(_option(name) for name in rejected)
        )

    layout = values.get("spatial_layout") or "per-image"
    if layout not in SPATIAL_LAYOUT_CHOICES:
        raise ValueError(
            f"unknown --spatial-layout {layout!r}; expected one of "
            f"{SPATIAL_LAYOUT_CHOICES}"
        )
    norm = values.get("feature_norm")
    if norm is not None and norm not in FEATURE_NORM_CHOICES:
        raise ValueError(
            f"unknown --feature-norm {norm!r}; expected one of "
            f"{FEATURE_NORM_CHOICES}"
        )

    if mode == MODE_READ:
        rejected = [
            name for name in READ_REJECTED_OPTIONS if _explicit(explicit_options, name)
        ]
        if rejected:
            raise ValueError(
                "artifact read mode inherits the stored representation, so these "
                "options cannot be supplied: "
                + ", ".join(_option(name) for name in rejected)
                + "; the stored product fixes the model, layers, normalization, "
                "size, layout and transform"
            )
        if _explicit(explicit_options, "spatial_eval_sample_size"):
            raise ValueError(
                "--spatial-eval-sample-size cannot be used in read mode; select a "
                "subset with --label-csv instead, which cannot expand the stored "
                "evaluation plan"
            )
        notes.append(
            "read mode inherits the stored evaluation seed; an explicitly "
            "matching --seed is checked against the saved plan"
        )
        return notes

    has_checkpoint = _given(values, "checkpoint")
    has_onnx = _given(values, "onnx_path")
    if has_checkpoint and has_onnx:
        raise ValueError(
            "spatial image extraction runs one backend: pass either --checkpoint "
            "or --onnx-path, not both"
        )
    if not has_checkpoint and not has_onnx:
        raise ValueError(
            "--output-kind spatial needs a feature source: --checkpoint for "
            "checkpoint extraction, --onnx-path for a spatial graph, or "
            "--spatial-input to read an existing product"
        )
    if not _given(values, "input_images_dir"):
        raise ValueError(
            "--input-images-dir is required for embedding and spatial image "
            "extraction; only spatial artifact read mode can omit it"
        )
    if _explicit(explicit_options, "spatial_eval_sample_size") and not _given(
        values, "spatial_eval"
    ):
        raise ValueError(
            "--spatial-eval-sample-size only applies with --spatial-eval"
        )
    if mode == MODE_ONNX:
        for name, why in (
            (
                "dump_model_structure",
                "lists checkpoint modules; a spatial graph already declares its "
                "exported layers",
            ),
            (
                "model_name",
                "cannot relabel a graph; the exported contract fixes the model",
            ),
            (
                "use_student",
                "cannot choose weights in a graph; the exported contract fixes the "
                "weight source",
            ),
        ):
            if _explicit(explicit_options, name):
                raise ValueError(f"{_option(name)} {why}")
    if mode == MODE_CHECKPOINT and _explicit(explicit_options, "use_student"):
        notes.append("--use-student selects the student weights of the checkpoint")
    if mode == MODE_ONNX:
        notes.append(
            "a spatial ONNX graph fixes its layers, normalization and size; an "
            "explicit exact match is verified, a conflict fails"
        )
    return notes


EXPORT_SPATIAL_OPTIONS = (
    "feature_layers",
    "feature_norm",
    "feature_l2_normalize",
    "use_student",
)


def validate_export_mode_options(
    values: Mapping[str, object], *, explicit_options: frozenset[str]
) -> list[str]:
    """Apply design §3's export matrix before any output is prepared.

    ``output_kind="embedding"`` keeps its historical graph, so a spatial-only
    option is rejected as soon as it was supplied — including a value that equals
    the spatial default, which would otherwise look like it took effect.
    """
    output_kind = values.get("output_kind") or "embedding"
    if output_kind not in OUTPUT_KIND_CHOICES:
        raise ValueError(
            f"unknown --output-kind {output_kind!r}; expected one of "
            f"{OUTPUT_KIND_CHOICES}"
        )
    notes: list[str] = []
    if output_kind == "spatial":
        norm = values.get("feature_norm")
        if norm is not None and norm not in FEATURE_NORM_CHOICES:
            raise ValueError(
                f"unknown --feature-norm {norm!r}; expected one of "
                f"{FEATURE_NORM_CHOICES}"
            )
        notes.append(
            "a spatial graph fixes the model, layers, normalization and size; the "
            "export validates every declared output against CPU PyTorch"
        )
        return notes
    supplied = [
        name for name in EXPORT_SPATIAL_OPTIONS if name in explicit_options
    ]
    if supplied:
        raise ValueError(
            "--output-kind embedding cannot accept these spatial-only options: "
            + ", ".join(_option(name) for name in supplied)
            + "; add --output-kind spatial or drop them"
        )
    return notes
