"""Strict-records validation and dependency checks precede destructive cleanup."""

import json

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from otuformer.cli.main import app
from test_output_safety import checkpoint


@pytest.mark.parametrize("kind", ["missing_model", "root_escape", "invalid_csv", "missing_image", "coco_link", "coco_invalid"])
def test_records_failure_preserves_old_output(tmp_path, checkpoint, kind):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "safe.png")
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "old.txt"
    sentinel.write_text("old output")
    pixels = out / "pixels.png"
    Image.new("RGB", (32, 32)).save(pixels)
    selector = tmp_path / "selector.csv"
    pd.DataFrame({"image": ["safe.png"]}).to_csv(selector, index=False)
    argv = ["extract", "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite", "--num-workers", "0"]
    if kind != "missing_model":
        argv += ["--checkpoint", str(checkpoint)]
    if kind in {"coco_link", "coco_invalid"}:
        (images / "linked.png").symlink_to(pixels)
        source = tmp_path / "source.json"
        source.write_text(json.dumps({"images": [{"id": 1, "file_name": "linked.png" if kind == "coco_link" else "safe.png", "width": 32, "height": 32}], "categories": [], "annotations": [{"id": 1, "image_id": 1, "category_id": 9} ] if kind == "coco_invalid" else []}))
        argv += ["--coco-instances", str(source)]
    else:
        if kind == "root_escape":
            (images / "linked.png").symlink_to(pixels)
            pd.DataFrame({"image": ["linked.png"]}).to_csv(selector, index=False)
        elif kind == "invalid_csv":
            selector.write_text("wrong\nsafe.png\n")
        elif kind == "missing_image":
            pd.DataFrame({"image": ["absent.png"]}).to_csv(selector, index=False)
        argv += ["--label-csv", str(selector)]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert sentinel.read_text() == "old output"
    assert pixels.is_file()
    assert sorted(p.name for p in out.iterdir()) == ["old.txt", "pixels.png"]


def test_records_selected_symlink_dependency_cannot_be_cleared(tmp_path, checkpoint):
    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    Image.new("RGB", (32, 32)).save(images / "real.png")
    (out / "middle.png").symlink_to(images / "real.png")
    (images / "linked.png").symlink_to(out / "middle.png")
    selector = tmp_path / "selector.csv"
    pd.DataFrame({"image": ["linked.png"]}).to_csv(selector, index=False)
    result = CliRunner().invoke(app, ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--label-csv", str(selector), "--out-dir", str(out), "--overwrite"])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert (out / "middle.png").is_symlink()
