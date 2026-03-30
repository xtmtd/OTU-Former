"""ONNX export for OTU-Former encoder + projector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from otuformer.training.model import OTUFormerEncoder
from otuformer.utils.checkpoint import load_checkpoint
from otuformer.utils.io import write_json


def export_to_onnx(
    checkpoint_path: Path,
    out_path: Path,
    imgsz: int = 224,
    opset: int = 17,
) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name", "vit_tiny_patch16_224")
    out_dim = cfg.get("out_dim") or cfg.get("metric_embed_dim", 256)

    model = OTUFormerEncoder(model_name=model_name, out_dim=out_dim)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        do_constant_folding=True,
    )

    validated = False
    output_shape = [1, int(out_dim)]
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(out_path), providers=["CPUExecutionProvider"]
        )
        out = session.run(None, {"input": dummy_input.numpy()})[0]
        output_shape = list(out.shape)
        validated = True
    except Exception:
        validated = False

    report = {
        "model_name": model_name,
        "out_dim": int(out_dim),
        "input_shape": [1, 3, imgsz, imgsz],
        "output_shape": output_shape,
        "opset": opset,
        "validated": validated,
        "onnx_path": str(out_path),
        "note": "Exports encoder + projector only. ArcFace head excluded.",
    }
    write_json(report, out_path.parent / "export_report.json")
    return report


def load_exported_onnx(onnx_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
