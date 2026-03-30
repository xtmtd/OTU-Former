"""cam command stub with full options."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

app = typer.Typer(help="Generate CAM heatmaps.")


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
    tee = TeeLogger(out_dir / "cam.log")
    sys.stdout = tee
    try:
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
    finally:
        sys.stdout = tee.terminal
        tee.close()
    typer.echo(f"CAM heatmaps written to: {out_dir / 'figures'}")
    typer.echo(f"Summary: {out_dir / 'cam_summary.csv'}")
