"""OTU-Former CLI main entry point."""

from __future__ import annotations

import sys

import typer


def detect_removed_no_phylo(argv: list[str]) -> bool:
    return "diversity" in argv and "--no-phylo" in argv


if detect_removed_no_phylo(sys.argv[1:]):
    import typer as _typer

    _typer.echo("Error: --no-phylo is removed; use --phylo to enable MPD.")
    raise _typer.Exit(code=2)

from otuformer.cli import annotate as _annotate_mod
from otuformer.cli import cam as _cam_mod
from otuformer.cli import cluster as _cluster_mod
from otuformer.cli import diversity as _diversity_mod
from otuformer.cli import doctor as _doctor_mod
from otuformer.cli import export as _export_mod
from otuformer.cli import extract as _extract_mod
from otuformer.cli import finetune as _finetune_mod
from otuformer.cli import pretrain as _pretrain_mod

app = typer.Typer(
    help=(
        "OTU-Former: image-based morphological OTU delineation toolkit.\n\n"
        "Quick start:\n\n"
        "  otuformer pretrain --train-data images.csv --input-images-dir ./images\n\n"
        "  otuformer finetune --checkpoint runs/pretrain/best.pt --train-data labels.csv\n\n"
        "  otuformer extract --checkpoint runs/finetune/best.pt --input-images-dir ./images\n\n"
        "  otuformer cluster --embeddings embeddings.csv\n\n"
        "  otuformer doctor\n"
    ),
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(_doctor_mod.app, name="doctor")
app.add_typer(_pretrain_mod.app, name="pretrain")
app.add_typer(_finetune_mod.app, name="finetune")
app.add_typer(_extract_mod.app, name="extract")
app.add_typer(_cluster_mod.app, name="cluster")
app.add_typer(_annotate_mod.app, name="annotate")
app.add_typer(_diversity_mod.app, name="diversity")
app.add_typer(_cam_mod.app, name="cam")
app.add_typer(_export_mod.app, name="export")
