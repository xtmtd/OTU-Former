"""ONNX export for OTU-Former encoder + projector."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import onnx
import torch

from otuformer.training.model import OTUFormerEncoder
from otuformer.utils.checkpoint import load_checkpoint
from otuformer.utils.io import write_json


def _infer_backbone_image_size(model: OTUFormerEncoder, fallback: int = 224) -> int:
    patch_embed = getattr(model.backbone, "patch_embed", None)
    if patch_embed is not None:
        img_size = getattr(patch_embed, "img_size", None)
        if isinstance(img_size, (tuple, list)) and len(img_size) >= 1:
            return int(img_size[0])
        if isinstance(img_size, int):
            return int(img_size)
    default_cfg = getattr(model.backbone, "default_cfg", None)
    if isinstance(default_cfg, dict):
        input_size = default_cfg.get("input_size")
        if isinstance(input_size, (tuple, list)) and len(input_size) >= 3:
            return int(input_size[1])
    return int(fallback)


def export_to_onnx(
    checkpoint_path: Path,
    out_path: Path,
    imgsz: int | None = None,
    opset: int = 18,
) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name", "vit_tiny_patch16_224")
    out_dim = cfg.get("out_dim") or cfg.get("metric_embed_dim", 256)

    model = OTUFormerEncoder(model_name=model_name, out_dim=out_dim, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    if imgsz is None:
        imgsz = _infer_backbone_image_size(model, fallback=224)

    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=opset,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
            do_constant_folding=True,
        )

    external_data = out_path.with_name(out_path.name + ".data")
    if external_data.exists():
        proto = onnx.load(str(out_path), load_external_data=True)
        onnx.save(proto, str(out_path), save_as_external_data=False)
        external_data.unlink()

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
        "note": (
            "Exports encoder + projector only. ArcFace head is a training-only classification head "
            "and is not needed for inference. Finetune checkpoints are recommended for export "
            "because the ArcFace head optimizes encoder weights during training, producing better embeddings."
        ),
    }
    write_json(report, out_path.parent / "export_report.json")
    return report


def load_exported_onnx(onnx_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
