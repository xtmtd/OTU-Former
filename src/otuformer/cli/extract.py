"""extract command - embeddings extraction."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Extract embeddings from images.")


@app.callback(invoke_without_command=True)
def extract(
    ctx: typer.Context,
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", help="Path to pretrain or finetune checkpoint."
    ),
    input_images_dir: Path = typer.Option(
        ..., "--input-images-dir", help="Image directory or parent directory."
    ),
    out_dir: Path = typer.Option(
        Path("runs/extract"), "--out-dir", help="Output directory."
    ),
    model_name: str = typer.Option(
        "vit_tiny_patch16_224", "--model-name", help="timm backbone."
    ),
    extract_size: int = typer.Option(
        224, "--extract-size", help="Resize/crop size for extraction."
    ),
    use_projector_output: bool = typer.Option(
        False,
        "--use-projector-output/--no-use-projector-output",
        help="Use projector output instead of CLS token.",
    ),
    token_mode: str = typer.Option(
        "cls", "--token-mode", help="Token mode: cls | patch-topk | attention-pool."
    ),
    topk_patches: int = typer.Option(
        20, "--topk-patches", help="Top-K patches for patch-topk mode."
    ),
    attention_pooling_type: str = typer.Option(
        "lightweight",
        "--attention-pooling-type",
        help="Attention pooling type.",
    ),
    attention_pooling_epochs: int = typer.Option(
        20, "--attention-pooling-epochs", help="Epochs for attention query."
    ),
    metrics_sample_size: int = typer.Option(
        10000, "--metrics-sample-size", help="Max samples for metrics."
    ),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader workers."),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto | cpu | cuda | mps."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    prefix: str = typer.Option(
        "OTU", "--prefix", help="OTU name prefix for downstream IDs."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from otuformer.embedding.extractor import extract_embeddings
    from otuformer.utils.io import write_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    df = extract_embeddings(
        checkpoint_path=checkpoint,
        images_dir=input_images_dir,
        model_name=model_name,
        extract_size=extract_size,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        use_projector_output=use_projector_output,
    )
    out_path = out_dir / "embeddings.csv"
    write_csv(df, out_path)
    typer.echo(f"Embeddings saved to: {out_path}")
