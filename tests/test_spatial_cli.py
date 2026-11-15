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
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
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


def test_spatial_eval_writes_paired_results_and_plan(tmp_path):
    images = write_images(tmp_path / "images")
    # A 64 px checkpoint so the 4x4 grid has enough centres to sample queries.
    checkpoint = write_checkpoint(tmp_path / "model.pt", image_size=64)
    out = tmp_path / "product"
    result = runner.invoke(
        app,
        spatial_argv(
            tmp_path, images, checkpoint, "--spatial-eval",
            "--spatial-eval-sample-size", "2",
            # A 4x4 patch grid with full-coverage preprocessing keeps queries
            # geometrically eligible, so the report has rows to compare.
            "--extract-size", "64", "--eval-transform", "center-crop",
        ),
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(
        (out / "spatial_evaluation_plan.json").read_text(encoding="utf-8")
    )
    assert plan["protocol"] == "spatial-geometry-v1"
    assert plan["seed"] == 42
    assert plan["selected_keys"]
    evaluation = json.loads((out / "spatial_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["protocol"] == "spatial-geometry-v1"
    assert evaluation["metrics"]["pck_thresholds"] == [0.01, 0.02, 0.05]
    assert (out / "spatial_matches.csv").is_file()
    envelope = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    assert envelope["evaluation"]["status"] == "completed"
    assert envelope["evaluation"]["seed"] == 42

    # Reading the product back reproduces the stored plan without a model.
    checkpoint.unlink()
    import shutil

    shutil.rmtree(images)
    read_out = tmp_path / "read-out"
    read = runner.invoke(
        app,
        [
            "extract", "--output-kind", "spatial", "--spatial-input", str(out),
            "--out-dir", str(read_out), "--spatial-eval",
        ],
    )
    assert read.exit_code == 0, read.output
    import csv

    def first_match(path):
        with path.open(encoding="utf-8") as handle:
            return next(csv.DictReader(handle))

    replayed = first_match(read_out / "spatial_matches.csv")
    stored = first_match(out / "spatial_matches.csv")
    assert replayed["image_key"] == stored["image_key"]
    assert replayed["match_status"] == stored["match_status"]
    assert replayed["normalized_error"] == stored["normalized_error"]
    assert replayed["raw_error"] == stored["raw_error"]


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


@pytest.mark.parametrize("records", [False, True])
def test_embedding_checkpoint_layer_and_dump_are_consumed(tmp_path, records):
    images = write_images(tmp_path / "images")
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    out = tmp_path / "embedding"
    argv = [
        "extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images),
        "--out-dir", str(out), "--feature-layers", "blocks.5",
        "--dump-model-structure", "--device", "cpu", "--num-workers", "0",
        "--disable-umap",
    ]
    if records:
        selector = tmp_path / "selected.csv"
        selector.write_text("image\na.png\nnested/b.png\n", encoding="utf-8")
        argv += ["--label-csv", str(selector)]
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output + str(result.exception)
    assert (out / "embeddings.csv").is_file()
    report = (out / "model_layers.txt").read_text()
    assert "blocks.5" in report and "embedding-cls-eligible" in report
    if records:
        provenance = json.loads((out / "run_provenance.json").read_text())
        assert provenance["effective"]["feature_layer"] == "blocks.5"
        assert any(row["path"] == "model_layers.txt" for row in provenance["artifacts"])


@pytest.mark.parametrize("flags", [
    ["--spatial-eval"], ["--feature-norm", "backbone"],
    ["--feature-layers", "blocks.1,blocks.2"],
    ["--feature-layers", "blocks.1", "--token-mode", "patch-topk"],
    ["--onnx-path", "missing.onnx", "--dump-model-structure"],
])
def test_embedding_inapplicable_flags_fail_before_cleanup(tmp_path, flags):
    out = tmp_path / "preserved"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    result = runner.invoke(app, ["extract", "--out-dir", str(out), "--overwrite", *flags])
    assert result.exit_code != 0
    assert sentinel.read_text() == "keep"
    assert "requires" in result.output or "spatial-only" in result.output


def test_spatial_help_panels_cover_every_new_option():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    import re

    output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
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


@pytest.mark.parametrize("fault", ["hash", "close", "interrupt", "open"])
def test_cli_records_a_failed_publication_instead_of_leaving_it_open(tmp_path, monkeypatch, fault):
    """The CLI's own final hash/publish is part of the attempt's lifecycle."""
    images = write_images(tmp_path / "images", keys=("a.png",))
    checkpoint = write_checkpoint(tmp_path / "model.pt")
    product = tmp_path / "product"
    assert runner.invoke(app, spatial_argv(tmp_path, images, checkpoint)).exit_code == 0

    read_out = tmp_path / "read-out"
    from otuformer.spatial import extractor as module

    from otuformer.utils.logging import TeeLogger

    closed = []
    original_close = module.PreparedSpatialRequest.close

    def close_request(request):
        closed.append(request.mode)
        original_close(request)

    def boom(*args, **kwargs):
        if fault == "interrupt":
            raise KeyboardInterrupt()
        raise OSError("simulated publication failure")

    monkeypatch.setattr(module.PreparedSpatialRequest, "close", close_request)
    if fault == "close":
        original_log_close = TeeLogger.close

        def close_log(logger):
            original_log_close(logger)
            boom()

        monkeypatch.setattr(TeeLogger, "close", close_log)
    elif fault == "open":
        monkeypatch.setattr(TeeLogger, "__init__", boom)
    else:
        monkeypatch.setattr(module, "_artifact_rows", boom)
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
    assert result.exit_code != 0
    envelope = json.loads((read_out / "run_provenance.json").read_text())
    assert closed == ["read"]
    assert envelope["status"] == ("interrupted" if fault == "interrupt" else "failed")
    if fault == "interrupt":
        assert result.exit_code == 130
    else:
        assert "simulated publication failure" in json.dumps(envelope)
