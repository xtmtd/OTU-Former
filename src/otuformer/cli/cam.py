"""cam command stub with full options."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import click
import typer

app = typer.Typer(help="Generate CAM heatmaps.")


def _format_user_command(ctx: typer.Context, params: dict[str, object]) -> str:
    parts = ["otuformer", "cam"]
    for key, value in params.items():
        source = ctx.get_parameter_source(key)
        if source is not click.core.ParameterSource.COMMANDLINE:
            continue
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            parts.extend([option, str(value).lower()])
            continue
        if value in (None, ""):
            continue
        parts.extend([option, str(value)])
    return " ".join(parts)


@app.callback(invoke_without_command=True)
def cam(
    ctx: typer.Context,
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Checkpoint file path."),
    images_dir: Path = typer.Option(..., "--images-dir", help="Directory of images."),
    label_csv: Path | None = typer.Option(
        None, "--label-csv", help="Optional CSV with image,label columns."
    ),
    out_dir: Path = typer.Option(
        Path("runs/cam"), "--out-dir", help="Output directory."
    ),
    cam_method: str = typer.Option("gradcam", "--cam-method", help="CAM method."),
    arch: str | None = typer.Option(None, "--arch", help="Architecture hint."),
    target_layer_name: str | None = typer.Option(
        None, "--target-layer-name", help="Target layer path."
    ),
    image_weight: float = typer.Option(
        0.5, "--image-weight", help="Overlay image weight."
    ),
    fig_format: str = typer.Option("png", "--fig-format", help="Figure format."),
    save_npy: bool = typer.Option(
        False, "--save-npy/--no-save-npy", help="Save raw CAM array as NPY."
    ),
    dump_model_structure: bool = typer.Option(
        False,
        "--dump-model-structure/--no-dump-model-structure",
        help="Dump model layers to file.",
    ),
    max_images: int | None = typer.Option(
        None, "--max-images", help="Maximum images to process."
    ),
    cam_batch_size: int = typer.Option(
        32, "--cam-batch-size", help="Batch size used by CAM backend."
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", help="Data loading workers (reserved)."
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto | cpu | cuda | mps."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.utils.logging import TeeLogger

    out_dir.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(out_dir / "logs" / "cam.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        params = {
            "checkpoint": str(checkpoint),
            "images_dir": str(images_dir),
            "label_csv": str(label_csv) if label_csv is not None else "",
            "out_dir": str(out_dir),
            "cam_method": cam_method,
            "arch": arch or "",
            "target_layer_name": target_layer_name or "",
            "image_weight": image_weight,
            "fig_format": fig_format,
            "save_npy": save_npy,
            "dump_model_structure": dump_model_structure,
            "max_images": max_images if max_images is not None else "",
            "cam_batch_size": cam_batch_size,
            "num_workers": num_workers,
            "device": device,
        }
        print(f"Command: {_format_user_command(ctx, params)}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        from otuformer.vision.cam import run_cam

        run_cam(
            checkpoint=checkpoint,
            images_dir=images_dir,
            out_dir=out_dir,
            label_csv=label_csv,
            cam_method=cam_method,
            arch=arch,
            target_layer_name=target_layer_name,
            image_weight=image_weight,
            fig_format=fig_format,
            save_npy=save_npy,
            dump_model_structure=dump_model_structure,
            max_images=max_images,
            cam_batch_size=max(1, min(cam_batch_size, 8)),
            device=device,
        )
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
    typer.echo(f"CAM heatmaps written to: {out_dir / 'figures'}")
    typer.echo(f"Summary: {out_dir / 'cam_summary.csv'}")
