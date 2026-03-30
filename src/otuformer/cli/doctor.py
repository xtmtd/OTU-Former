"""doctor command - environment health check."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import typer

app = typer.Typer(help="Check environment and dependency health.")


def _pkg_version(pip_name: str, import_name: str | None = None) -> str:
    try:
        return importlib.metadata.version(pip_name)
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        mod = importlib.import_module(import_name or pip_name)
        return getattr(mod, "__version__", "installed")
    except ImportError:
        return "NOT INSTALLED"


_PACKAGES = [
    ("torch", "torch"),
    ("timm", "timm"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("scikit-bio", "skbio"),
    ("umap-learn", "umap"),
    ("grad-cam", "pytorch_grad_cam"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("tqdm", "tqdm"),
]


@app.callback(invoke_without_command=True)
def doctor(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return

    typer.echo("\n=== otuformer doctor ===\n")
    typer.echo(f"Python: {sys.version.split()[0]}")

    typer.echo("\nPackages:")
    for pip_name, import_name in _PACKAGES:
        typer.echo(f"  {pip_name}: {_pkg_version(pip_name, import_name)}")

    typer.echo("\nDevices:")
    typer.echo("  cpu: available")
    try:
        import torch

        if torch.cuda.is_available():
            typer.echo(f"  cuda: {torch.version.cuda or 'available'}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            typer.echo("  mps: available")
    except ImportError:
        typer.echo("  torch not installed - cannot detect GPU devices")

    typer.echo("")
