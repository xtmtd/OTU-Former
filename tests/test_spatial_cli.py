"""Spatial extract CLI: mode matrix, help panels and requested/resolved logs."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from otuformer.cli.main import app

runner = CliRunner()
TIMESTAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ")


def flat(text: str) -> str:
    """Letters/digits only: Rich wraps long messages and inserts box borders."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def write_images(root: Path, keys=("a.png", "nested/b.png")) -> Path:
    for key in keys:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 40), (20, 60, 90)).save(path)
    return root


def write_checkpoint(path: Path, image_size: int = 32) -> Path:
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(out_dim=8, pretrained=False, img_size=image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": "vit_tiny_patch16_224",
                "out_dim": 8,
                "image_size": image_size,
            },
        },
        path,
    )
    return path


def log_text(path: Path) -> str:
    stripped = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TIMESTAMP.match(line)
        stripped.append(line[match.end() :] if match else line)
    return "\n".join(stripped)


def parse_log(path: Path) -> dict:
    """Exactly one request block, plus the separately labeled resolved block."""
    text = log_text(path)
    assert text.count("Command: ") == 1, text
    assert text.count("Parameters:") == 1
    assert text.count("Resolved:") == 1
    command = re.search(r"^Command: (.*)$", text, flags=re.MULTILINE).group(1)
    parameters = text.split("Parameters:", 1)[1].split("-" * 80, 1)[0]
    resolved = text.split("Resolved:", 1)[1]
    return {
        "command": command,
        "parameters": json.loads(parameters),
        "resolved": json.loads(resolved),
        "text": text,
    }


def spatial_argv(tmp_path: Path, images: Path, checkpoint: Path, *extra: str) -> list[str]:
    return [
        "extract",
        "--output-kind",
        "spatial",
        "--checkpoint",
        str(checkpoint),
        "--input-images-dir",
        str(images),
        "--out-dir",
        str(tmp_path / "product"),
        *extra,
    ]


def test_spatial_extraction_logs_requested_and_resolved_separately(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    out = tmp_path / "product"
    result = runner.invoke(
        app,
        spatial_argv(
            tmp_path,
            images,
            checkpoint,
            "--feature-norm",
            "backbone",
            "--spatial-layout",
            "sharded",
        ),
    )
    assert result.exit_code == 0, result.output
    log = parse_log(out / "logs" / "extract.log")

    # Requested values are what was typed, including an auto size and an omitted
    # layer list; they are never backfilled with resolved values.
    assert log["parameters"]["extract_size"] == "auto"
    assert log["parameters"]["feature_layers"] == ""
    assert log["parameters"]["feature_norm"] == "backbone"
    assert log["parameters"]["seed"] == 42
    assert log["parameters"]["spatial_eval"] is False
    assert log["text"].count("Expected") == 0

    # Resolved facts live in their own block and match the contract/provenance.
    assert log["resolved"]["input_size"] == 32
    assert log["resolved"]["feature_contract"]["layers"][0]["path"] == "blocks.11"
    assert log["resolved"]["feature_contract"]["norm"] == "backbone"
    assert log["resolved"]["layout"] == "sharded"
    assert log["resolved"]["eval_transform"] == "whole-specimen-pad"
    envelope = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    assert envelope["status"] == "completed"
    assert envelope["effective"]["feature_contract"] == log["resolved"]["feature_contract"]

    # The command line is replayable, quoting path values.
    argv = shlex.split(log["command"])
    assert argv[0] == "otuformer" and argv[1] == "extract"
    assert "--spatial-layout" in argv and "sharded" in argv


def test_command_log_is_posix_replayable_with_spaces_and_explicit_defaults(tmp_path):
    spaced = tmp_path / "images with space"
    images = write_images(spaced / "nested", keys=("a.png",))
    checkpoint = write_checkpoint(tmp_path / "model with space.pt")
    out = tmp_path / "product out"
    result = runner.invoke(
        app,
        [
            "extract",
            "--output-kind",
            "spatial",
            "--checkpoint",
            str(checkpoint),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--feature-norm",
            "backbone",
            "--spatial-layout",
            "per-image",
        ],
    )
    assert result.exit_code == 0, result.output
    log = parse_log(out / "logs" / "extract.log")
    argv = shlex.split(log["command"])
    assert str(images) in argv and str(out) in argv
    # Explicit default-valued flags survive the round trip.
    assert "--feature-norm" in argv and "backbone" in argv


def test_read_mode_needs_no_images_or_model(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    product = tmp_path / "product"
    extract = runner.invoke(app, spatial_argv(tmp_path, images, checkpoint))
    assert extract.exit_code == 0, extract.output

    checkpoint.unlink()
    import shutil

    shutil.rmtree(images)
    read_out = tmp_path / "read-out"
    result = runner.invoke(
        app,
        [
            "extract",
            "--output-kind",
            "spatial",
            "--spatial-input",
            str(product),
            "--out-dir",
            str(read_out),
        ],
    )
    assert result.exit_code == 0, result.output
    log = parse_log(read_out / "logs" / "extract.log")
    assert log["parameters"]["checkpoint"] == ""
    assert log["resolved"]["mode"] == "read"
    envelope = json.loads((read_out / "run_provenance.json").read_text(encoding="utf-8"))
    assert envelope["source_product"]["product_id"]
    assert not (read_out / "spatial_manifest.json").exists()


def test_explicit_embedding_only_options_fail_before_clearing_output(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("previous run", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            *spatial_argv(tmp_path, images, checkpoint),
            "--out-dir",
            str(out),
            "--overwrite",
            "--token-mode",
            "cls",
        ],
    )
    assert result.exit_code != 0
    assert "embeddingonly" in flat(result.output)
    assert sentinel.read_text(encoding="utf-8") == "previous run"
    assert not (out / "run_provenance.json").exists()


def test_spatial_input_without_spatial_output_kind_is_actionable(tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["extract", "--spatial-input", "somewhere", "--out-dir", str(out)],
    )
    assert result.exit_code != 0
    assert "outputkindspatial" in flat(result.output)
    assert not out.exists()


def test_read_mode_rejects_extraction_overrides_before_writing(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    product = tmp_path / "product"
    assert runner.invoke(app, spatial_argv(tmp_path, images, checkpoint)).exit_code == 0
    before = (product / "spatial_manifest.json").read_bytes()

    read_out = tmp_path / "read-out"
    result = runner.invoke(
        app,
        [
            "extract",
            "--output-kind",
            "spatial",
            "--spatial-input",
            str(product),
            "--out-dir",
            str(read_out),
            "--feature-layers",
            "blocks.0",
        ],
    )
    assert result.exit_code != 0
    assert "cannotbesupplied" in flat(result.output)
    assert (product / "spatial_manifest.json").read_bytes() == before
    assert not read_out.exists()


def test_read_mode_rejects_an_output_inside_the_source_product(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    product = tmp_path / "product"
    assert runner.invoke(app, spatial_argv(tmp_path, images, checkpoint)).exit_code == 0
    before = (product / "spatial_manifest.json").read_bytes()
    result = runner.invoke(
        app,
        [
            "extract",
            "--output-kind",
            "spatial",
            "--spatial-input",
            str(product),
            "--out-dir",
            str(product / "nested"),
            "--overwrite",
        ],
    )
    assert result.exit_code != 0
    assert "sourceproduct" in flat(result.output) or "containstheinput" in flat(result.output)
    assert (product / "spatial_manifest.json").read_bytes() == before
    assert not (product / "nested").exists()


def test_spatial_eval_is_rejected_without_claiming_completion(tmp_path):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    out = tmp_path / "product"
    result = runner.invoke(
        app,
        spatial_argv(tmp_path, images, checkpoint, "--spatial-eval"),
    )
    assert result.exit_code != 0
    assert "spatialevalisnotenabled" in flat(result.output)
    assert not (out / "run_provenance.json").exists()


def test_dump_model_structure_continues_extraction(tmp_path):
    images = write_images(tmp_path / "images", keys=("a.png",))
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    out = tmp_path / "product"
    result = runner.invoke(
        app, spatial_argv(tmp_path, images, checkpoint, "--dump-model-structure")
    )
    assert result.exit_code == 0, result.output
    report = out / "model_layers.txt"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "blocks.11" in text
    assert (out / "spatial_manifest.json").is_file()
    assert (out / "spatial_results.csv").is_file()


def test_failed_spatial_run_records_a_failed_envelope(tmp_path, monkeypatch):
    images = write_images(tmp_path / "images", keys=("a.png",))
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    out = tmp_path / "product"
    # A missing feature layer is rejected during preparation: no output at all.
    result = runner.invoke(
        app,
        spatial_argv(tmp_path, images, checkpoint, "--feature-layers", "blocks.99"),
    )
    assert result.exit_code != 0
    assert not out.exists()

    # A failure inside the run (not during preparation) is recorded as a failed
    # envelope instead of leaving an unreadable half-product.
    from otuformer.spatial import extractor as extractor_module

    def boom(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(extractor_module, "extract_spatial", boom)
    result = runner.invoke(app, spatial_argv(tmp_path, images, checkpoint))
    assert result.exit_code != 0
    envelope = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    assert envelope["status"] in ("failed", "interrupted")
    assert envelope["failure"]["reason"]
    log = out / "logs" / "extract.log"
    assert log.is_file()
    # The log closes normally: the requested block is present and closed.
    assert "Command: " in log_text(log)
    # No completed product is readable.
    from otuformer.spatial.storage import open_spatial_product

    with pytest.raises(Exception):
        open_spatial_product(out)


def test_preflight_rejection_preserves_protected_outputs(tmp_path):
    images = write_images(tmp_path / "images", keys=("a.png",))
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    out = tmp_path / "product"
    out.mkdir()
    keep = out / "keep.txt"
    keep.write_text("older", encoding="utf-8")
    # The checkpoint lives inside the output directory: clearing it would delete
    # the model this run reads.
    inside = out / "model.pt"
    inside.write_bytes(checkpoint.read_bytes())
    result = runner.invoke(
        app,
        [
            "extract",
            "--output-kind",
            "spatial",
            "--checkpoint",
            str(inside),
            "--input-images-dir",
            str(images),
            "--out-dir",
            str(out),
            "--overwrite",
        ],
    )
    assert result.exit_code != 0
    assert keep.read_text(encoding="utf-8") == "older"
    assert inside.is_file()


def test_spatial_help_panels_cover_every_new_option():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    output = result.output
    for panel in (
        "Input and output",
        "Model and feature source",
        "Preprocessing and geometry",
        "Embedding output",
        "Spatial output and storage",
        "Embedding evaluation and visualization",
        "Spatial evaluation and diagnostics",
        "Runtime and reproducibility",
        "Safety",
    ):
        assert panel in output, panel
    for flag in (
        "--output-kind",
        "--spatial-input",
        "--feature-layers",
        "--feature-norm",
        "--feature-l2-normalize",
        "--spatial-layout",
        "--spatial-eval",
        "--spatial-eval-sample-size",
        "--dump-model-structure",
    ):
        assert flag in output, flag
    # Panels appear in the designed order.
    positions = [
        output.index(panel)
        for panel in (
            "Input and output",
            "Model and feature source",
            "Preprocessing and geometry",
            "Embedding output",
            "Spatial output and storage",
            "Embedding evaluation and visualization",
            "Spatial evaluation and diagnostics",
            "Runtime and reproducibility",
            "Safety",
        )
    ]
    assert positions == sorted(positions)
