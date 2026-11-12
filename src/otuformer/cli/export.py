"""export command stub with full options."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import typer

from otuformer.cli import SIZE_EXAMPLES, _parse_size, docs_url, format_user_command

app = typer.Typer(
    help=(
        "Export a PyTorch checkpoint to ONNX format.\n\n"
        "Converts a trained model checkpoint into a portable ONNX encoder that can be\n"
        "used with ONNX Runtime for faster inference. The exported model outputs\n"
        "embedding vectors directly from image inputs.\n\n"
        "Quick example:\n\n"
        "  otuformer export --checkpoint runs/finetune/best.pt\n"
        "  otuformer export --checkpoint best.pt --imgsz 224 --opset 17\n"
        "\nDocs:\n"
        f"  {docs_url('export')}\n"
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
    output_kind: str = typer.Option(
        "embedding",
        "--output-kind",
        help=(
            "Graph product: 'embedding' keeps the existing encoder+projector graph "
            "with one 'embedding' output; 'spatial' exports one named output per "
            "selected complete backbone block plus a fixed feature contract."
        ),
        rich_help_panel="Input and output",
    ),
    feature_layers: str | None = typer.Option(
        None,
        "--feature-layers",
        help=(
            "Spatial only. Comma-separated backbone-relative complete block paths "
            "(for example 'blocks.5,blocks.11'). Omitted: the last complete block."
        ),
        rich_help_panel="Model and feature source",
    ),
    feature_norm: str = typer.Option(
        "backbone",
        "--feature-norm",
        help=(
            "Spatial only. Apply the backbone's final norm to each exported block "
            "('backbone') or export the raw block output ('none')."
        ),
        rich_help_panel="Model and feature source",
    ),
    feature_l2_normalize: bool = typer.Option(
        False,
        "--feature-l2-normalize",
        help=(
            "Spatial only. L2-normalize each token vector after the backbone norm "
            "and export that representation."
        ),
        rich_help_panel="Model and feature source",
    ),
    use_student: bool = typer.Option(
        False,
        "--use-student",
        help=(
            "Spatial export only. Export the student weights; the graph records the "
            "weight source it actually used. Rejected for embedding export, whose "
            "graph has always exported the teacher/default weights."
        ),
        rich_help_panel="Model and feature source",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.vision.export import export_to_onnx
    from otuformer.utils.io import prepare_output_dir

    from otuformer.cli import source_is_commandline

    # Explicit values matter here: an embedding graph must reject a spatial-only
    # option even when the value equals the declared default.
    explicit = {
        name
        for name in ("feature_layers", "feature_norm", "feature_l2_normalize", "use_student")
        if source_is_commandline(ctx.get_parameter_source(name))
    }
    try:
        from otuformer.spatial.request import validate_export_mode_options

        validate_export_mode_options(
            {
                "output_kind": output_kind,
                "feature_layers": feature_layers,
                "feature_norm": feature_norm,
                "feature_l2_normalize": feature_l2_normalize,
                "use_student": use_student,
            },
            explicit_options=frozenset(explicit),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    try:
        prepare_output_dir(
            out_dir, overwrite=overwrite, protected_paths=[checkpoint],
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
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
            "output_kind": output_kind,
            "feature_layers": feature_layers or "",
            "feature_norm": feature_norm,
            "feature_l2_normalize": feature_l2_normalize,
            "use_student": use_student,
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
            output_kind=output_kind,
            feature_layers=feature_layers or None,
            feature_norm=feature_norm if output_kind == "spatial" else None,
            feature_l2_normalize=(
                feature_l2_normalize if output_kind == "spatial" else None
            ),
            use_student=use_student,
        )
        typer.echo(f"Export complete: {onnx_path}")
        typer.echo(f"  Model:     {report['model_name']}")
        if report.get("output_kind") == "spatial":
            contract = report["spatial"]["feature_contract"]
            typer.echo(
                "  Layers:    "
                + ", ".join(layer["path"] for layer in contract["layers"])
            )
            typer.echo(f"  Grid:      {contract['layers'][0]['grid_hw']}")
            typer.echo(f"  Outputs:   {', '.join(report['spatial']['output_names'])}")
        else:
            typer.echo(f"  Embed dim: {report['out_dim']}")
        typer.echo(f"  Validated: {report['validated']}")
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
