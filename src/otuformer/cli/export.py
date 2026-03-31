"""export command stub with full options."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import click
import typer

app = typer.Typer(help="Export checkpoint to ONNX.")


def _format_user_command(ctx: typer.Context, params: dict[str, object]) -> str:
    parts = ["otuformer", "export"]
    for key, value in params.items():
        source = ctx.get_parameter_source(key)
        if source is not click.core.ParameterSource.COMMANDLINE:
            continue
        option = f"--{key.replace('_', '-')}"
        if value in (None, ""):
            continue
        parts.extend([option, str(value)])
    return " ".join(parts)


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
    from otuformer.utils.logging import TeeLogger

    tee = TeeLogger(out_dir / "logs" / "export.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        params = {
            "checkpoint": str(checkpoint),
            "out_dir": str(out_dir),
            "imgsz": imgsz,
            "opset": opset,
        }
        print(f"Command: {_format_user_command(ctx, params)}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

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
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
