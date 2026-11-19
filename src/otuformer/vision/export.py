"""ONNX export for OTU-Former encoder + projector."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnx
import torch

from otuformer.constants import (
    OUTPUT_KIND_CHOICES,
    SPATIAL_ONNX_METADATA_KEY,
    SPATIAL_SCHEMA_VERSION,
)
from otuformer.utils.checkpoint import (
    load_checkpoint,
    load_checkpoint_encoder,
    resolve_checkpoint,
)
from otuformer.utils.io import write_json
from otuformer.utils.size import resolve_training_image_size, validate_input_size

#: Ordered graph output names for a spatial export.
SPATIAL_OUTPUT_PREFIX = "spatial_"


def spatial_output_name(index: int) -> str:
    return f"{SPATIAL_OUTPUT_PREFIX}{index:03d}"

# Approved from measurement before implementation: CPU FP32 PyTorch and CPU ONNX
# Runtime agree to <=3.7e-06 absolute on real pretrain/fine-tune checkpoints and
# on the synthetic out_dim=64 cases at 32/224/384 px, so these initial
# thresholds hold with a wide margin and were not loosened.
ATOL = 1e-4
RTOL = 1e-3


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    """Same-directory atomic replacement for the export report."""
    temp = path.with_name(path.name + ".tmp")
    write_json(data, temp)
    os.replace(temp, path)


def _remove_temp_files(*paths: Path) -> None:
    for path in paths:
        Path(path).unlink(missing_ok=True)


def _validate_numeric_agreement(
    onnx_path: Path,
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    out_dim: int,
) -> dict[str, Any]:
    """Compare CPU FP32 PyTorch with CPU ONNX Runtime at the export size.

    Returns the report's validation fields. Raises when ONNX Runtime is
    installed but cannot run the graph or the outputs disagree, so the caller
    publishes nothing for this attempt.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return {
            "validated": False,
            "validation_status": "skipped",
            "max_abs_diff": None,
            "output_shape": [1, int(out_dim)],
            "validation_note": (
                "ONNX Runtime is not installed; shape and numeric validation "
                "were skipped."
            ),
        }
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = model(dummy_input)
    if isinstance(reference, tuple):
        reference = reference[0]
    reference_np = reference.detach().to(torch.float32).numpy()
    runtime_out = session.run(None, {"input": dummy_input.numpy()})[0]
    if tuple(runtime_out.shape) != tuple(reference_np.shape):
        raise ValueError(
            "ONNX Runtime output shape "
            f"{tuple(runtime_out.shape)} does not match the PyTorch reference "
            f"{tuple(reference_np.shape)}."
        )
    if not np.allclose(reference_np, runtime_out, atol=ATOL, rtol=RTOL):
        raise ValueError(
            "ONNX Runtime output disagrees with the CPU FP32 PyTorch reference "
            f"beyond atol={ATOL}, rtol={RTOL}."
        )
    return {
        "validated": True,
        "validation_status": "passed",
        "max_abs_diff": float(np.abs(reference_np - runtime_out).max()),
        "output_shape": list(runtime_out.shape),
        "validation_note": (
            f"CPU ONNX Runtime agrees with the CPU FP32 PyTorch reference within "
            f"atol={ATOL}, rtol={RTOL}."
        ),
    }


def _reject_spatial_options_in_embedding(
    *,
    feature_layers: str | None,
    feature_norm: str | None,
    feature_l2_normalize: bool | None,
    use_student: bool,
) -> None:
    """An embedding graph cannot carry a spatial representation contract.

    ``None`` means omitted. An explicitly supplied value counts even when it equals
    the spatial default, because accepting it would look like it took effect.
    """
    supplied = [
        name
        for name, value in (
            ("feature_layers", feature_layers),
            ("feature_norm", feature_norm),
            ("feature_l2_normalize", feature_l2_normalize),
        )
        if value is not None
    ]
    if supplied:
        raise ValueError(
            "--output-kind embedding cannot export these spatial-only options: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in supplied)
            + "; add --output-kind spatial or drop them"
        )
    if use_student:
        raise ValueError(
            "a true student request is not supported for embedding export: the "
            "embedding graph has always exported the teacher/default weights. Drop "
            "--use-student, or request --output-kind spatial where the weight source "
            "is part of the exported feature contract."
        )


def export_to_onnx(
    checkpoint_path: Path,
    out_path: Path,
    imgsz: int | None = None,
    opset: int = 18,
    model_name: str = "vit_tiny_patch16_224",
    *,
    output_kind: str = "embedding",
    feature_layers: str | None = None,
    feature_norm: str | None = None,
    feature_l2_normalize: bool | None = None,
    use_student: bool = False,
    prepared_spatial: Mapping[str, Any] | None = None,
    prepared_embedding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export an encoder graph.

    ``output_kind="embedding"`` (the default, and identical to omitting it) keeps
    the historical encoder+projector graph. ``output_kind="spatial"`` exports one
    named output per selected complete block with a fixed feature contract in the
    graph metadata. Spatial-only keyword values use ``None`` for omitted, so an
    explicit ``feature_norm="backbone"`` is still a spatial request.
    """
    if output_kind not in OUTPUT_KIND_CHOICES:
        raise ValueError(
            f"unknown output kind {output_kind!r}; expected one of "
            f"{OUTPUT_KIND_CHOICES}"
        )
    if output_kind == "embedding":
        _reject_spatial_options_in_embedding(
            feature_layers=feature_layers,
            feature_norm=feature_norm,
            feature_l2_normalize=feature_l2_normalize,
            use_student=use_student,
        )
        return _export_embedding(
            checkpoint_path,
            out_path,
            imgsz=imgsz,
            opset=opset,
            model_name=model_name,
            prepared=prepared_embedding,
        )
    return _export_spatial(
        checkpoint_path,
        out_path,
        imgsz=imgsz,
        opset=opset,
        model_name=model_name,
        feature_layers=feature_layers,
        feature_norm=feature_norm,
        feature_l2_normalize=feature_l2_normalize,
        use_student=use_student,
        prepared=prepared_spatial,
    )


def _prepare_embedding_export(
    checkpoint_path: Path,
    *,
    imgsz: int | None,
    model_name: str,
) -> dict[str, Any]:
    """Resolve the embedding checkpoint, model and sizes, writing nothing.

    Doing this before a caller clears an existing output means an unacceptable
    checkpoint or size cannot destroy a previous graph.
    """
    ckpt = load_checkpoint(checkpoint_path)
    architecture = resolve_checkpoint(ckpt, model_name)
    resolved_name = architecture.model_name
    out_dim = architecture.embedding_dim
    checkpoint_size = resolve_training_image_size(ckpt)

    model, architecture = load_checkpoint_encoder(
        ckpt, architecture, image_size=checkpoint_size
    )
    validate_input_size(checkpoint_size, model, resolved_name)
    # Export the embedding output only; the shared loader builds the encoder
    # with patch tokens enabled for its other consumers.
    model.return_patch_tokens = False
    model.eval()

    size = checkpoint_size if imgsz is None else int(imgsz)
    validate_input_size(size, model, resolved_name)
    return {
        "model_name": resolved_name,
        "out_dim": int(out_dim),
        "checkpoint_size": checkpoint_size,
        "imgsz": size,
        "model": model,
        "architecture": architecture,
    }


def _export_embedding(
    checkpoint_path: Path,
    out_path: Path,
    imgsz: int | None = None,
    opset: int = 18,
    model_name: str = "vit_tiny_patch16_224",
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = prepared or _prepare_embedding_export(
        checkpoint_path, imgsz=imgsz, model_name=model_name
    )
    model_name = prepared["model_name"]
    out_dim = prepared["out_dim"]
    checkpoint_size = prepared["checkpoint_size"]
    imgsz = prepared["imgsz"]
    model = prepared["model"]
    architecture = prepared["architecture"]

    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_path.parent / "export_report.json"

    # Export and validate on this attempt's temporary paths, so a failure never
    # disturbs the previously published model or report.
    temp_onnx = out_path.with_name(f"{out_path.stem}.part{out_path.suffix}")
    temp_data = temp_onnx.with_name(temp_onnx.name + ".data")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            torch.onnx.export(
                model,
                dummy_input,
                str(temp_onnx),
                opset_version=opset,
                input_names=["input"],
                output_names=["embedding"],
                dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
                do_constant_folding=True,
            )
        if temp_data.exists():
            proto = onnx.load(str(temp_onnx), load_external_data=True)
            onnx.save(proto, str(temp_onnx), save_as_external_data=False)
            temp_data.unlink()
        validation = _validate_numeric_agreement(
            temp_onnx, model, dummy_input, out_dim
        )
    except Exception:
        _remove_temp_files(temp_onnx, temp_data)
        raise
    report: dict[str, Any] = {
        "model_name": model_name,
        "out_dim": int(out_dim),
        "input_shape": [1, 3, imgsz, imgsz],
        "opset": opset,
        "register_tokens": int(architecture.actual_register_tokens or 0),
        "atol": ATOL,
        "rtol": RTOL,
        "onnx_path": str(out_path),
        **validation,
        "note": (
            "Exports encoder + projector only. ArcFace head is a training-only classification head "
            "and is not needed for inference. Finetune checkpoints are recommended for export "
            "because the ArcFace head optimizes encoder weights during training, producing better embeddings."
        ),
    }
    # Invalidate any earlier success report before replacing the model, so an
    # interrupted publication can never leave a stale validated=True result.
    _atomic_write_json(
        {
            **report,
            "validated": False,
            "validation_status": "invalidated",
            "max_abs_diff": None,
            "validation_note": (
                "publication in progress; the report is refreshed after the new "
                "model is in place."
            ),
        },
        report_path,
    )
    try:
        os.replace(temp_onnx, out_path)
    finally:
        # A failed publication must not leave this attempt's temporary graph
        # behind in the destination directory.
        _remove_temp_files(temp_onnx, temp_data)
    _atomic_write_json(report, report_path)
    return report


class _SpatialExportModule(torch.nn.Module):
    """Thin module so the exporter sees the encoder's parameters."""

    def __init__(self, encoder: torch.nn.Module, forward) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_forward = forward

    def forward(self, images: torch.Tensor):
        return self.feature_forward(images)


def _contract_metadata_value(contract) -> str:
    import json

    from otuformer.data.provenance import canonical_json_bytes

    return canonical_json_bytes(
        {
            "schema_version": SPATIAL_SCHEMA_VERSION,
            "contract": contract.to_json(),
            "outputs": [spatial_output_name(index) for index in range(len(contract.layers))],
            "layout": "BHWC",
            "per_image_layout": "HWC",
        }
    ).decode("utf-8")


def _attach_contract_metadata(path: Path, contract) -> None:
    entry = None
    proto = onnx.load(str(path), load_external_data=False)
    for existing in proto.metadata_props:
        if existing.key == SPATIAL_ONNX_METADATA_KEY:
            entry = existing
            break
    if entry is None:
        entry = proto.metadata_props.add()
        entry.key = SPATIAL_ONNX_METADATA_KEY
    entry.value = _contract_metadata_value(contract)
    onnx.save(proto, str(path))


def read_spatial_onnx_contract(path: Path) -> Any:
    """Read the exported feature contract, or reject a graph that has none.

    Metadata is treated as an assertion to validate against the graph, not as
    proof that the graph computes it.
    """
    import json

    from otuformer.vision.features import FeatureContract

    proto = onnx.load(str(Path(path)), load_external_data=False)
    for entry in proto.metadata_props:
        if entry.key != SPATIAL_ONNX_METADATA_KEY:
            continue
        payload = json.loads(entry.value)
        if set(payload) != {"schema_version", "contract", "outputs", "layout", "per_image_layout"}:
            raise ValueError(
                "spatial graph metadata does not match version 1 fields: "
                f"{sorted(payload)}"
            )
        if int(payload["schema_version"]) != SPATIAL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported spatial graph schema version {payload['schema_version']}"
            )
        if payload["layout"] != "BHWC":
            raise ValueError(
                f"spatial graph layout must be BHWC, got {payload['layout']!r}"
            )
        if payload["per_image_layout"] != "HWC":
            raise ValueError(
                "spatial graph per-image layout must be HWC, got "
                f"{payload['per_image_layout']!r}"
            )
        contract = FeatureContract.from_json(payload["contract"])
        _check_onnx_graph_contract(proto, contract, declared=payload["outputs"])
        return contract
    raise ValueError(
        "this ONNX graph declares no spatial feature contract "
        f"({SPATIAL_ONNX_METADATA_KEY}); it can only be used by the embedding path"
    )


def _check_onnx_graph_contract(proto, contract, *, declared) -> None:
    """Validate metadata and graph I/O against each other before any run.

    A declared output list, input rank/dtype or output shape that contradicts the
    graph turns every later use into a mysterious runtime failure; checking them
    here makes the graph self-describing or plainly rejected.
    """
    expected_names = [
        spatial_output_name(index) for index in range(len(contract.layers))
    ]
    if list(declared) != expected_names:
        raise ValueError(
            f"spatial graph metadata declares outputs {list(declared)}, but its "
            f"contract produces {expected_names}"
        )
    inputs = list(proto.graph.input)
    if len(inputs) != 1 or inputs[0].name != "input":
        raise ValueError(
            "spatial graph must have exactly one input named 'input', got "
            f"{[item.name for item in inputs]}"
        )
    tensor = inputs[0].type.tensor_type
    dimensions = tensor.shape.dim
    if len(dimensions) != 4:
        raise ValueError(
            f"spatial graph input must be rank 4 (BCHW), got {len(dimensions)}"
        )
    if tensor.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(
            "spatial graph input must be float32, got element type "
            f"{tensor.elem_type}"
        )
    channels = dimensions[1].dim_value
    height, width = (int(v) for v in contract.input_hw)
    if channels != 3:
        raise ValueError(f"spatial graph input must have 3 channels, got {channels}")
    declared_hw = (dimensions[2].dim_value, dimensions[3].dim_value)
    if declared_hw != (height, width):
        raise ValueError(
            f"spatial graph input size {declared_hw} does not match the contract "
            f"canvas {(height, width)}"
        )
    outputs = {output.name: output for output in proto.graph.output}
    if sorted(outputs) != sorted(expected_names):
        raise ValueError(
            f"spatial graph outputs {sorted(outputs)} do not match the metadata "
            f"{expected_names}"
        )
    for layer, name in zip(contract.layers, expected_names):
        output = outputs[name].type.tensor_type
        expected = [
            "batch",
            int(layer.grid_hw[0]),
            int(layer.grid_hw[1]),
            int(layer.channels),
        ]
        actual = [dim.dim_param or int(dim.dim_value) for dim in output.shape.dim]
        if actual != expected:
            raise ValueError(
                f"spatial graph output {name} declares shape {actual}, but its "
                f"contract layer is {expected} (batch, grid_h, grid_w, channels)"
            )
        if output.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(
                f"spatial graph output {name} must be float32, got element type "
                f"{output.elem_type}"
            )


def validate_spatial_onnx_outputs(
    path: Path, forward, contract
) -> dict[str, Any]:
    """Check the graph metadata and every named output against CPU PyTorch.

    Batches 1 and 2 are both required: a fixed-batch graph is not a spatial v1
    graph. Every declared output is compared, so a graph that silently drops or
    reorders a layer fails instead of passing on its first tensor.
    """
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - exercised by the ORT gate test
        raise ValueError(
            "ONNX Runtime is required to validate a spatial graph; without it the "
            "export cannot be marked spatial-validated"
        ) from exc

    session = ort.InferenceSession(str(Path(path)), providers=["CPUExecutionProvider"])
    declared_outputs = [output.name for output in session.get_outputs()]
    expected_outputs = [
        spatial_output_name(index) for index in range(len(contract.layers))
    ]
    if declared_outputs != expected_outputs:
        raise ValueError(
            f"graph outputs {declared_outputs} do not match the feature contract "
            f"{expected_outputs}"
        )
    graph_input = session.get_inputs()[0]
    if graph_input.name != "input":
        raise ValueError(f"graph input must be named 'input', got {graph_input.name!r}")
    batch_dim = graph_input.shape[0]
    if not isinstance(batch_dim, str):
        raise ValueError(
            "spatial v1 requires a dynamic batch dimension; this graph has a fixed "
            f"batch ({batch_dim})"
        )

    height, width = (int(v) for v in contract.input_hw)
    per_batch: dict[str, dict[str, Any]] = {}
    worst_abs = 0.0
    worst_rel = 0.0
    for batch in (1, 2):
        generator = torch.Generator().manual_seed(1000 + batch)
        sample = torch.randn(batch, 3, height, width, generator=generator)
        with torch.no_grad():
            reference = forward(sample)
        if not isinstance(reference, tuple):
            reference = (reference,)
        if len(reference) != len(expected_outputs):
            raise ValueError(
                f"PyTorch reference produced {len(reference)} outputs, expected "
                f"{len(expected_outputs)}"
            )
        produced = session.run(expected_outputs, {"input": sample.numpy()})
        rows: dict[str, Any] = {}
        for name, expected_tensor, actual in zip(expected_outputs, reference, produced):
            expected_np = expected_tensor.detach().to(torch.float32).numpy()
            if actual.dtype != np.float32:
                raise ValueError(f"graph output {name} is {actual.dtype}, expected float32")
            if tuple(actual.shape) != tuple(expected_np.shape):
                raise ValueError(
                    f"graph output {name} has shape {tuple(actual.shape)}, expected "
                    f"{tuple(expected_np.shape)}"
                )
            if not np.all(np.isfinite(actual)):
                raise ValueError(f"graph output {name} contains non-finite values")
            if not np.allclose(expected_np, actual, atol=ATOL, rtol=RTOL):
                difference = np.abs(expected_np - actual)
                raise ValueError(
                    f"graph output {name} disagrees with the CPU FP32 PyTorch "
                    f"reference beyond atol={ATOL}, rtol={RTOL} "
                    f"(max abs {float(difference.max())})"
                )
            absolute = float(np.abs(expected_np - actual).max())
            scale = float(np.abs(expected_np).max()) or 1.0
            rows[name] = {
                "shape": [int(v) for v in actual.shape],
                "dtype": str(actual.dtype),
                "max_abs_diff": absolute,
                "max_rel_diff": absolute / scale,
                "validated": True,
            }
            worst_abs = max(worst_abs, absolute)
            worst_rel = max(worst_rel, absolute / scale)
        per_batch[f"batch_{batch}"] = rows
    return {
        "validated": True,
        "validation_status": "passed",
        "atol": ATOL,
        "rtol": RTOL,
        "outputs": expected_outputs,
        "layout": "BHWC",
        "per_image_layout": "HWC",
        "batch_validation": per_batch,
        "max_abs_diff": worst_abs,
        "max_rel_diff": worst_rel,
        "validation_note": (
            "CPU ONNX Runtime agrees with the CPU FP32 PyTorch reference for every "
            f"declared output at batches 1 and 2 within atol={ATOL}, rtol={RTOL}. "
            "This is a CPU contract, not a CUDA/provider guarantee."
        ),
    }


def _prepare_spatial_export(
    checkpoint_path: Path,
    *,
    imgsz: int | None,
    model_name: str,
    feature_layers: str | None,
    feature_norm: str | None,
    feature_l2_normalize: bool | None,
    use_student: bool,
) -> dict[str, Any]:
    """Resolve the checkpoint capability and layer contract, writing nothing.

    A spatial request can be rejected on its own arguments (an unknown layer, an
    unsupported backbone, a size the contract cannot tile). Doing that here lets a
    caller check the request *before* it clears an existing output, so a rejected
    argument cannot delete an earlier graph.
    """
    from otuformer.constants import IMAGENET_MEAN, IMAGENET_STD
    from otuformer.data.provenance import sha256_bytes
    from otuformer.vision.features import (
        SpatialFeatureForward,
        resolve_feature_contract,
        resolve_feature_layers,
    )

    checkpoint_path = Path(checkpoint_path)
    ckpt = load_checkpoint(checkpoint_path)
    architecture = resolve_checkpoint(
        ckpt, model_name, prefer_student=bool(use_student)
    )
    resolved_model_name = architecture.model_name
    checkpoint_size = resolve_training_image_size(ckpt)
    model, architecture = load_checkpoint_encoder(
        ckpt, architecture, image_size=checkpoint_size
    )
    validate_input_size(checkpoint_size, model, resolved_model_name)
    model.eval()
    size = checkpoint_size if imgsz is None else int(imgsz)
    validate_input_size(size, model, resolved_model_name)
    layers = resolve_feature_layers(model.backbone, feature_layers)
    contract = resolve_feature_contract(
        model,
        size,
        layers,
        norm=feature_norm or "backbone",
        l2=bool(feature_l2_normalize),
        model_identity={
            "architecture": resolved_model_name,
            "checkpoint_sha256": sha256_bytes(checkpoint_path.read_bytes()),
            "weight_source": architecture.source_key,
            "register_tokens": architecture.actual_register_tokens,
            "added_register_tokens": architecture.added_register_tokens,
            "ssl_identity": list(architecture.ssl_identity or ()),
        },
        normalization={
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "space": "rgb01",
        },
    )
    forward = SpatialFeatureForward(model, contract)
    return {
        "model_name": resolved_model_name,
        "architecture": architecture,
        "model": model,
        "size": size,
        "contract": contract,
        "forward": forward,
        "module": _SpatialExportModule(model, forward).eval(),
    }


def _export_spatial(
    checkpoint_path: Path,
    out_path: Path,
    *,
    imgsz: int | None = None,
    opset: int = 18,
    model_name: str = "vit_tiny_patch16_224",
    feature_layers: str | None = None,
    feature_norm: str | None = None,
    feature_l2_normalize: bool | None = None,
    use_student: bool = False,
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if prepared is None:
        prepared = _prepare_spatial_export(
            checkpoint_path,
            imgsz=imgsz,
            model_name=model_name,
            feature_layers=feature_layers,
            feature_norm=feature_norm,
            feature_l2_normalize=feature_l2_normalize,
            use_student=use_student,
        )
    model_name = prepared["model_name"]
    architecture = prepared["architecture"]
    size = int(prepared["size"])
    contract = prepared["contract"]
    forward = prepared["forward"]
    module = prepared["module"]
    output_names = [
        spatial_output_name(index) for index in range(len(contract.layers))
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_path.parent / "export_report.json"
    temp_onnx = out_path.with_name(f"{out_path.stem}.part{out_path.suffix}")
    temp_data = temp_onnx.with_name(temp_onnx.name + ".data")
    dummy = torch.randn(1, 3, size, size)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # The FX/dynamo exporter keeps the batch axis symbolic, but only when
            # it traces without autograd: under grad mode it records a different
            # linear decomposition that bakes a fixed reshape and fails at batch 2.
            with torch.no_grad():
                torch.onnx.export(
                    module,
                    dummy,
                    str(temp_onnx),
                    opset_version=opset,
                    input_names=["input"],
                    output_names=output_names,
                    dynamic_axes={
                        "input": {0: "batch"},
                        **{name: {0: "batch"} for name in output_names},
                    },
                    dynamo=True,
                )
        if temp_data.exists():
            proto = onnx.load(str(temp_onnx), load_external_data=True)
            onnx.save(proto, str(temp_onnx), save_as_external_data=False)
            temp_data.unlink()
        # Metadata is written before the graph is hashed, so the sealed bytes are
        # the ones a reader validates.
        _attach_contract_metadata(temp_onnx, contract)
        round_trip = read_spatial_onnx_contract(temp_onnx)
        if round_trip != contract:
            raise ValueError(
                "the exported graph metadata does not round-trip the feature contract"
            )
        validation = validate_spatial_onnx_outputs(temp_onnx, forward, contract)
    except Exception:
        _remove_temp_files(temp_onnx, temp_data)
        raise
    report: dict[str, Any] = {
        "output_kind": "spatial",
        "model_name": model_name,
        "input_shape": [1, 3, size, size],
        "opset": opset,
        "register_tokens": int(architecture.actual_register_tokens or 0),
        "atol": ATOL,
        "rtol": RTOL,
        "onnx_path": str(out_path),
        "spatial": {
            "schema_version": SPATIAL_SCHEMA_VERSION,
            "metadata_key": SPATIAL_ONNX_METADATA_KEY,
            "feature_contract": contract.to_json(),
            "output_names": output_names,
            **validation,
        },
        "validated": validation["validated"],
        "validation_status": validation["validation_status"],
        "max_abs_diff": validation["max_abs_diff"],
        "validation_note": validation["validation_note"],
        "note": (
            "Exports the selected complete backbone blocks as BHWC grids with a "
            "fixed feature contract in the graph metadata. Channel width is not "
            "the embedding out_dim, and per-image geometry stays outside the graph."
        ),
    }
    _atomic_write_json(
        {
            **report,
            "validated": False,
            "validation_status": "invalidated",
            "max_abs_diff": None,
            "validation_note": (
                "publication in progress; the report is refreshed after the new "
                "model is in place."
            ),
        },
        report_path,
    )
    try:
        os.replace(temp_onnx, out_path)
    finally:
        _remove_temp_files(temp_onnx, temp_data)
    _atomic_write_json(report, report_path)
    return report


class OnnxSpatialForward:
    """CPU ONNX Runtime adapter for a validated spatial graph.

    It owns one session and the graph's own feature contract, accepts normalized
    BCHW FP32 images and returns the ordered BHWC grids. Named outputs are
    requested explicitly, never ``session.run(None, ...)[0]``: a graph that drops
    or reorders a layer must fail here instead of silently returning a subset.
    """

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.path = Path(path)
        self.contract = read_spatial_onnx_contract(self.path)
        self.session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        self.output_names = [
            spatial_output_name(index) for index in range(len(self.contract.layers))
        ]
        declared = [output.name for output in self.session.get_outputs()]
        if declared != self.output_names:
            raise ValueError(
                f"graph outputs {declared} do not match the feature contract "
                f"{self.output_names}"
            )
        graph_input = self.session.get_inputs()[0]
        if graph_input.name != "input":
            raise ValueError(
                f"graph input must be named 'input', got {graph_input.name!r}"
            )
        if not isinstance(graph_input.shape[0], str):
            raise ValueError(
                "spatial v1 requires a dynamic batch dimension; this graph has a "
                f"fixed batch ({graph_input.shape[0]})"
            )
        expected_hw = (int(self.contract.input_hw[1]), int(self.contract.input_hw[0]))
        declared_hw = graph_input.shape[-2:]
        if tuple(declared_hw) != expected_hw:
            raise ValueError(
                f"graph input spatial size {tuple(declared_hw)} does not match the "
                f"declared contract {self.contract.input_hw}"
            )
        self.normalization = dict(self.contract.normalization)

    def __call__(self, images: np.ndarray) -> tuple[np.ndarray, ...]:
        batch = np.asarray(images)
        if batch.dtype != np.float32:
            raise ValueError(
                f"spatial ONNX input must be float32, got {batch.dtype}"
            )
        expected = (None, 3, int(self.contract.input_hw[1]), int(self.contract.input_hw[0]))
        if batch.ndim != 4 or tuple(batch.shape[1:]) != expected[1:]:
            raise ValueError(
                f"spatial ONNX input must be BCHW {expected}, got {batch.shape}"
            )
        produced = self.session.run(self.output_names, {"input": batch})
        outputs: list[np.ndarray] = []
        for name, layer, array in zip(self.output_names, self.contract.layers, produced):
            expected_shape = (
                batch.shape[0],
                int(layer.grid_hw[0]),
                int(layer.grid_hw[1]),
                int(layer.channels),
            )
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"graph output {name} has shape {tuple(array.shape)}, expected "
                    f"{expected_shape}"
                )
            if array.dtype != np.float32:
                raise ValueError(
                    f"graph output {name} is {array.dtype}, expected float32"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"graph output {name} contains non-finite values")
            outputs.append(np.asarray(array, dtype=np.float32))
        return tuple(outputs)

    def close(self) -> None:  # pragma: no cover - ORT sessions own no handle here
        self.session = None


def load_exported_onnx(onnx_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
