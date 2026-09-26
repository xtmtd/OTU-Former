"""export command stub with full options."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import typer

from otuformer.cli import SIZE_EXAMPLES, _parse_size, format_user_command

app = typer.Typer(
    help=(
        "Export a PyTorch checkpoint to ONNX format.\n\n"
        "Converts a trained model checkpoint into a portable ONNX encoder that can be\n"
        "used with ONNX Runtime for faster inference. The exported model outputs\n"
        "embedding vectors directly from image inputs.\n\n"
        "Quick example:\n\n"
        "  otuformer export --checkpoint runs/finetune/best.pt\n"
        "  otuformer export --checkpoint best.pt --imgsz 224 --opset 17\n"
    )
)


@app.callback(invoke_without_command=True)
def export(
    ctx: typer.Context,
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint path."),
    out_dir: Path = typer.Option(
        Path("runs/export"), "--out-dir", help="Output directory."
    ),
    imgsz: str = typer.Option(
        "auto",
        "--imgsz",
        help=(
            "Input image size for ONNX export. 'auto' uses the checkpoint's "
            "recorded training size. "
            f"{SIZE_EXAMPLES}"
        ),
    ),
    opset: int = typer.Option(18, "--opset", help="ONNX opset version."),
    model_name: str = typer.Option(
        "vit_tiny_patch16_224",
        "--model-name",
        help=(
            "Fallback timm backbone for checkpoints that record no model name "
            "(ref-script fine-tune checkpoints store neither config nor args)."
        ),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Clear an existing non-empty output directory."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.vision.export import export_to_onnx
    from otuformer.utils.io import prepare_output_dir

    prepare_output_dir(out_dir, overwrite=overwrite)
    from otuformer.utils.logging import TeeLogger

    tee = TeeLogger(out_dir / "logs" / "export.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        params = {
            "checkpoint": str(checkpoint),
            "out_dir": str(out_dir),
            "overwrite": overwrite,
            "imgsz": imgsz,
            "opset": opset,
            "model_name": model_name,
        }
        print(f"Command: {format_user_command(ctx, params, 'export')}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        onnx_path = out_dir / "encoder.onnx"
        report = export_to_onnx(
            checkpoint_path=checkpoint,
            out_path=onnx_path,
            imgsz=_parse_size(imgsz, stage="--imgsz"),
            opset=opset,
            model_name=model_name,
        )
        typer.echo(f"Export complete: {onnx_path}")
        typer.echo(f"  Model:     {report['model_name']}")
        typer.echo(f"  Embed dim: {report['out_dim']}")
        typer.echo(f"  Validated: {report['validated']}")
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
