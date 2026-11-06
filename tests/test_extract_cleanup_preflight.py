"""Strict-records validation and dependency checks precede destructive cleanup."""

import json

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from otuformer.cli.main import app
from test_output_safety import checkpoint


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_legacy_extract_dependency_walk_oserror_is_a_cli_error(
    tmp_path, checkpoint, monkeypatch, error_type
):
    from otuformer import cli_schema
    from otuformer.utils import io

    images = tmp_path / "images"
    images.mkdir()
    source = images / "safe.png"
    Image.new("RGB", (32, 32)).save(source)
    pixels = source.read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    walk_calls = []
    cleanup_calls = []

    def refuse_walk(path):
        walk_calls.append(path)
        raise error_type("input dependency traversal refused by filesystem")

    real_prepare = io.prepare_output_dir

    def observed_prepare(out_dir, **kwargs):
        cleanup_calls.append(out_dir)
        return real_prepare(out_dir, **kwargs)

    monkeypatch.setattr(cli_schema, "input_dependency_positions", refuse_walk)
    monkeypatch.setattr(io, "prepare_output_dir", observed_prepare)
    result = CliRunner().invoke(app, [
        "extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images),
        "--out-dir", str(out), "--overwrite",
    ])
    assert walk_calls == [images]
    assert cleanup_calls == []
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit)
    output = " ".join(result.output.replace("│", " ").split())
    assert "input dependency traversal refused by filesystem" in output
    assert "Traceback" not in output
    assert source.read_bytes() == pixels
    assert sentinel.read_text() == "keep"
    assert set(out.iterdir()) == {sentinel}


@pytest.mark.parametrize("records_path", [False, True])
@pytest.mark.parametrize("error_type", [FileExistsError, PermissionError])
def test_extract_cleanup_oserror_is_a_cli_error(
    tmp_path, checkpoint, monkeypatch, records_path, error_type
):
    from otuformer.utils import io

    images = tmp_path / "images"
    images.mkdir()
    source = images / "safe.png"
    Image.new("RGB", (32, 32)).save(source)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    before = source.read_bytes()
    calls = []

    def refuse_cleanup(out_dir, **kwargs):
        calls.append(out_dir)
        raise error_type("cleanup refused by filesystem")

    monkeypatch.setattr(io, "prepare_output_dir", refuse_cleanup)
    argv = ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite"]
    if records_path:
        selector = tmp_path / "selector.csv"
        pd.DataFrame({"image": ["safe.png"]}).to_csv(selector, index=False)
        argv += ["--label-csv", str(selector)]
    result = CliRunner().invoke(app, argv)
    assert calls == [out]
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit)
    assert "cleanup refused by filesystem" in result.output
    assert "Traceback" not in result.output
    assert source.read_bytes() == before
    assert sentinel.read_text() == "keep"
    assert set(out.iterdir()) == {sentinel}


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("layout", ["equal", "contains"])
def test_legacy_extract_rejects_output_that_contains_image_root(
    tmp_path, checkpoint, overwrite, layout
):
    out = tmp_path / "out"
    images = out if layout == "equal" else out / "images"
    images.mkdir(parents=True)
    source = images / "safe.png"
    Image.new("RGB", (32, 32)).save(source)
    pixels = source.read_bytes()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    argv = ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--out-dir", str(out)]
    if overwrite:
        argv.append("--overwrite")
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    if not overwrite:
        output = " ".join(result.output.replace("│", " ").split())
        assert "would clear the image root" in output
    assert source.read_bytes() == pixels
    assert sentinel.read_text() == "keep"
    assert not (out / "embeddings.csv").exists()


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("records_path", [False, True])
def test_extract_nested_output_follows_documented_data_path_policy(
    tmp_path, checkpoint, overwrite, records_path
):
    images = tmp_path / "images"
    images.mkdir()
    source = images / "safe.png"
    Image.new("RGB", (32, 32)).save(source)
    pixels = source.read_bytes()
    out = images / "run"
    sentinel = out / "keep.txt"
    if overwrite:
        out.mkdir()
        sentinel.write_text("keep")
    argv = ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--out-dir", str(out), "--disable-umap", "--device", "cpu", "--num-workers", "0"]
    if overwrite:
        argv.append("--overwrite")
    if records_path:
        selector = tmp_path / "selector.csv"
        pd.DataFrame({"image": ["safe.png"]}).to_csv(selector, index=False)
        argv += ["--label-csv", str(selector)]
    result = CliRunner().invoke(app, argv)
    assert source.read_bytes() == pixels
    if records_path:
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "is inside protected input" in " ".join(result.output.replace("│", " ").split())
        if overwrite:
            assert sentinel.read_text() == "keep"
        else:
            assert not out.exists()
    else:
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(pd.read_csv(out / "embeddings.csv")) == 1
        assert not sentinel.exists()


@pytest.mark.parametrize("layout", ["direct", "relay", "root"])
def test_legacy_cleanup_guard_matches_schema_dependency_error(tmp_path, layout):
    from otuformer.cli_schema import check_inputs_survive_cleanup, check_output_safety

    out = tmp_path / "out"
    out.mkdir()
    source = out / "input.png"
    source.write_bytes(b"preserve input")
    entry = source if layout == "direct" else out
    if layout == "relay":
        entry = tmp_path / "link.png"
        entry.symlink_to(source)
    for guard in (
        lambda: check_inputs_survive_cleanup(out, [entry]),
        lambda: check_output_safety(out, overwrite=True, protected_paths=[entry]),
    ):
        with pytest.raises(ValueError) as error:
            guard()
        assert str(error.value) == (
            f"output directory {out.resolve()} contains protected input "
            f"dependency {out.resolve()}, which {entry} is reached through"
        )
        assert source.read_bytes() == b"preserve input"


def test_legacy_cleanup_guard_allows_output_below_input_root(tmp_path):
    from otuformer.cli_schema import check_inputs_survive_cleanup

    root = tmp_path / "images"
    out = root / "run"
    out.mkdir(parents=True)
    source = root / "safe.png"
    source.write_bytes(b"preserve input")
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    check_inputs_survive_cleanup(out, [root, source])
    assert source.read_bytes() == b"preserve input"
    assert sentinel.read_text() == "keep"


def test_legacy_cleanup_guard_rejects_unverifiable_chain(tmp_path):
    from otuformer.cli_schema import check_inputs_survive_cleanup

    out = tmp_path / "out"
    out.mkdir()
    source = tmp_path / "loop"
    source.symlink_to(source)
    with pytest.raises(ValueError) as error:
        check_inputs_survive_cleanup(out, [source])
    assert str(error.value) == (
        f"cannot verify output safety: {source} passes through more than 40 symbolic links"
    )


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


@pytest.mark.parametrize("records_path", [False, True])
def test_extract_paths_report_the_same_selected_symlink_conflict(tmp_path, checkpoint, records_path):
    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    Image.new("RGB", (32, 32)).save(images / "real.png")
    (out / "middle.png").symlink_to(images / "real.png")
    (images / "linked.png").symlink_to(out / "middle.png")
    selector = tmp_path / "selector.csv"
    pd.DataFrame({"image": ["linked.png"]}).to_csv(selector, index=False)
    argv = ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite"]
    if records_path:
        argv += ["--label-csv", str(selector)]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    output = " ".join(result.output.replace("│", " ").split())
    assert "contains protected input dependency" in output
    assert "would remove" not in output
    assert (out / "middle.png").is_symlink()
    assert (images / "real.png").is_file()


@pytest.mark.parametrize("overwrite,exists", [(False, False), (True, False), (True, True)])
def test_records_read_once_at_the_required_safety_stage(tmp_path, checkpoint, monkeypatch, overwrite, exists):
    from otuformer.data import reader

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "safe.png")
    selector = tmp_path / "selector.csv"
    pd.DataFrame({"image": ["safe.png"]}).to_csv(selector, index=False)
    out = tmp_path / "out"
    if exists:
        out.mkdir()
        (out / "old.txt").write_text("keep")
    calls = []
    real_read = reader.read_dataset

    def observed_read(*args, **kwargs):
        calls.append("read")
        if overwrite and exists:
            assert (out / "old.txt").read_text() == "keep"
            assert not (out / "logs").exists()
        else:
            assert (out / "logs" / "extract.log").is_file()
            assert (out / "run_provenance.json").is_file()
        return real_read(*args, **kwargs)

    monkeypatch.setattr(reader, "read_dataset", observed_read)
    argv = ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--label-csv", str(selector), "--out-dir", str(out), "--num-workers", "0", "--device", "cpu", "--disable-umap"]
    if overwrite:
        argv.append("--overwrite")
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls == ["read"]


def test_records_occupied_output_is_rejected_without_reading_dataset(tmp_path, checkpoint, monkeypatch):
    from otuformer.data import reader

    images = tmp_path / "images"
    images.mkdir()
    selector = tmp_path / "selector.csv"
    selector.write_text("image\nmissing.png\n")
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "old.txt"
    sentinel.write_text("keep")

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject occupied output before reading the dataset")

    monkeypatch.setattr(reader, "read_dataset", forbidden)
    result = CliRunner().invoke(app, ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--label-csv", str(selector), "--out-dir", str(out)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError), repr(result.exception)
    assert sentinel.read_text() == "keep"
