"""export command stub with full options."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Export checkpoint to ONNX.")


@app.callback(invoke_without_command=True)
def export(
    ctx: typer.Context,
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint path."),
    out_dir: Path = typer.Option(
        Path("runs/export"), "--out-dir", help="Output directory."
    ),
    imgsz: int = typer.Option(224, "--imgsz", help="Input image size for ONNX export."),
    opset: int = typer.Option(17, "--opset", help="ONNX opset version."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.vision.export import export_to_onnx

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "encoder.onnx"
    report = export_to_onnx(
        checkpoint_path=checkpoint,
        out_path=onnx_path,
        imgsz=imgsz,
        opset=opset,
    )
    typer.echo(f"Export complete: {onnx_path}")
    typer.echo(f"  Model:     {report['model_name']}")
    typer.echo(f"  Embed dim: {report['out_dim']}")
    typer.echo(f"  Validated: {report['validated']}")
