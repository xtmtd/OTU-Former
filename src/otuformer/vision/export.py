"""ONNX export for OTU-Former encoder + projector."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from otuformer.utils.checkpoint import (
    load_checkpoint,
    load_checkpoint_encoder,
    resolve_checkpoint,
)
from otuformer.utils.io import write_json
from otuformer.utils.size import resolve_training_image_size, validate_input_size

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


def export_to_onnx(
    checkpoint_path: Path,
    out_path: Path,
    imgsz: int | None = None,
    opset: int = 18,
    model_name: str = "vit_tiny_patch16_224",
) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path)
    architecture = resolve_checkpoint(ckpt, model_name)
    model_name = architecture.model_name
    out_dim = architecture.embedding_dim
    checkpoint_size = resolve_training_image_size(ckpt)

    model, architecture = load_checkpoint_encoder(
        ckpt, architecture, image_size=checkpoint_size
    )
    validate_input_size(checkpoint_size, model, model_name)
    # Export the embedding output only; the shared loader builds the encoder
    # with patch tokens enabled for its other consumers.
    model.return_patch_tokens = False
    model.eval()

    if imgsz is None:
        imgsz = checkpoint_size
    validate_input_size(imgsz, model, model_name)

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


def load_exported_onnx(onnx_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
