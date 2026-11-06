import sys
import re
from pathlib import Path

import pandas as pd
import pytest
import torch

from otuformer.utils.checkpoint import load_checkpoint, save_checkpoint
from otuformer.utils.io import (
    prepare_output_dir,
    read_csv,
    read_json,
    write_csv,
    write_json,
)
from otuformer.utils.logging import TeeLogger
from otuformer.utils import logging as logging_mod


def test_tee_logger_writes_to_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger = TeeLogger(log_file)
    original_stdout = sys.stdout
    sys.stdout = logger
    print("hello world")
    sys.stdout = original_stdout
    logger.close()
    content = log_file.read_text()
    lines = content.splitlines()
    assert lines[0].startswith("[") and "otuformer" in lines[0]
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] hello world$", lines[-1]
    )


def test_tee_logger_skips_progress_bars(tmp_path):
    log_file = tmp_path / "test.log"
    logger = TeeLogger(log_file)
    original_stdout = sys.stdout
    sys.stdout = logger
    logger.write("\r[progress bar]")
    logger.write("real line\n")
    sys.stdout = original_stdout
    logger.close()
    content = log_file.read_text()
    assert "[progress bar]" not in content
    assert "real line" in content


def _fake_git_run(monkeypatch, *, toplevel, commit="abc1234", dirty=False):
    from types import SimpleNamespace

    def fake_run(args, **_kwargs):
        if "--show-toplevel" in args:
            stdout = str(toplevel)
        elif "status" in args:
            stdout = " M src/otuformer/utils/logging.py\n" if dirty else ""
        else:
            stdout = commit
        return SimpleNamespace(returncode=0, stdout=stdout + "\n")

    monkeypatch.setattr(logging_mod.subprocess, "run", fake_run)


def _otuformer_checkout(tmp_path, name="otuformer"):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.8.0"\n', encoding="utf-8"
    )
    return root


@pytest.mark.parametrize(
    ("dirty", "expected"), [(False, "abc1234"), (True, "abc1234-dirty")]
)
def test_git_commit_reports_the_otuformer_checkout(
    tmp_path, monkeypatch, dirty, expected
):
    _fake_git_run(monkeypatch, toplevel=_otuformer_checkout(tmp_path), dirty=dirty)
    assert logging_mod._git_commit(tmp_path) == expected


def test_git_commit_rejects_a_foreign_repository(tmp_path, monkeypatch):
    """A package vendored in another repo must not report the host's commit."""
    _fake_git_run(
        monkeypatch, toplevel=_otuformer_checkout(tmp_path, name="host-project")
    )
    assert logging_mod._git_commit(tmp_path) == "unknown"


def test_git_commit_is_unknown_when_git_is_missing(tmp_path, monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(logging_mod.subprocess, "run", fake_run)
    assert logging_mod._git_commit(tmp_path) == "unknown"


def test_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"id": ["a", "b"], "val": [1, 2]})
    p = tmp_path / "test.csv"
    write_csv(df, p)
    df2 = read_csv(p)
    assert list(df2["id"]) == ["a", "b"]


def test_json_roundtrip(tmp_path):
    data = {"key": "value", "n": 42}
    p = tmp_path / "test.json"
    write_json(data, p)
    data2 = read_json(p)
    assert data2["key"] == "value"
    assert data2["n"] == 42


def test_checkpoint_roundtrip(tmp_path):
    state = {"epoch": 5, "model_state_dict": {}, "config": {"out_dim": 256}}
    p = tmp_path / "ckpt.pt"
    save_checkpoint(state, p)
    loaded = load_checkpoint(p)
    assert loaded["epoch"] == 5
    assert loaded["config"]["out_dim"] == 256


def test_load_checkpoint_missing_file():
    with pytest.raises(FileNotFoundError):
        load_checkpoint(Path("/nonexistent/path.pt"))


def test_prepare_output_dir_rejects_existing_content(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / ".DS_Store").write_text("metadata", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_output_dir(out_dir)


def test_prepare_output_dir_overwrite_removes_existing_content(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")

    assert prepare_output_dir(out_dir, overwrite=True) == out_dir
    assert out_dir.exists()
    assert list(out_dir.iterdir()) == []


def test_prepare_output_dir_allows_existing_resume_directory(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "SSL_latest.pth").write_text("checkpoint", encoding="utf-8")

    assert prepare_output_dir(out_dir, allow_existing=True) == out_dir


def test_prepare_output_dir_overwrite_refuses_to_clear_its_inputs(tmp_path):
    """``--overwrite`` must never delete a directory that holds the run's data."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "img_0.jpg").write_text("pixels", encoding="utf-8")
    csv_path = tmp_path / "images.csv"
    csv_path.write_text("image\nimg_0.jpg\n", encoding="utf-8")

    for out_dir, inputs in (
        (images, (images,)),
        (tmp_path, (csv_path,)),
        (tmp_path, (images / "img_0.jpg",)),
    ):
        with pytest.raises(ValueError, match="holds the input"):
            prepare_output_dir(out_dir, overwrite=True, protected_paths=inputs)
        assert (images / "img_0.jpg").exists(), "an input was deleted"
        assert csv_path.exists(), "an input was deleted"


def test_prepare_output_dir_overwrite_allows_an_unrelated_target(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    out_dir = tmp_path / "images" / "run"

    assert (
        prepare_output_dir(out_dir, overwrite=True, protected_paths=(images,))
        == out_dir
    )


def test_prepare_output_dir_refuses_a_symlinked_input(tmp_path):
    """A symlink inside the output directory hides where the pixels really live."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    real = out_dir / "real.jpg"
    real.write_text("pixels", encoding="utf-8")
    link = tmp_path / "link.jpg"
    link.symlink_to(real)

    with pytest.raises(ValueError, match="holds the input"):
        prepare_output_dir(out_dir, overwrite=True, protected_paths=(link,))
    assert real.exists()


def test_prepare_output_dir_refuses_a_symlink_chain_through_the_output(tmp_path):
    """A middle hop is an input dependency, not only the two ends of the chain.

    ``entry -> out/middle -> pixels/real`` resolves outside the output directory
    at both ends, but clearing it deletes the middle link and leaves the run
    reading a dangling one after the old results are gone.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pixels = tmp_path / "pixels"
    pixels.mkdir()
    real = pixels / "real.png"
    real.write_text("pixels", encoding="utf-8")
    middle = out_dir / "middle.png"
    middle.symlink_to(real)
    entry = tmp_path / "images" / "entry.png"
    entry.parent.mkdir()
    entry.symlink_to(middle)

    with pytest.raises(ValueError, match="holds the input"):
        prepare_output_dir(out_dir, overwrite=True, protected_paths=(entry,))
    assert middle.exists() and real.exists()


def test_prepare_output_dir_refuses_a_symlinked_directory_in_the_chain(tmp_path):
    """A symlinked directory on the way is one of the paths the read depends on."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pixels = tmp_path / "pixels"
    pixels.mkdir()
    (pixels / "real.png").write_text("pixels", encoding="utf-8")
    (out_dir / "stage").symlink_to(pixels)
    entry = tmp_path / "images" / "entry.png"
    entry.parent.mkdir()
    entry.symlink_to(out_dir / "stage" / "real.png")

    with pytest.raises(ValueError, match="holds the input"):
        prepare_output_dir(out_dir, overwrite=True, protected_paths=(entry,))
    assert (out_dir / "stage").is_symlink()
    assert (pixels / "real.png").exists()


@pytest.mark.parametrize("in_link_target", [False, True])
def test_prepare_output_dir_preserves_symlink_parent_semantics(tmp_path, in_link_target):
    out_dir = tmp_path / "out"
    (out_dir / "sub").mkdir(parents=True)
    real = out_dir / "real.png"
    real.write_text("pixels", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    (images / "jump").symlink_to(out_dir / "sub", target_is_directory=True)
    source = images / "jump" / ".." / "real.png"
    if in_link_target:
        entry = images / "entry.png"
        entry.symlink_to(source)
        source = entry
    assert source.read_text(encoding="utf-8") == "pixels"

    with pytest.raises(ValueError, match="holds the input"):
        prepare_output_dir(out_dir, overwrite=True, protected_paths=(source,))

    assert real.read_text(encoding="utf-8") == "pixels"
    assert source.exists()


def test_prepare_output_dir_refuses_a_directory_link_inside_a_link_target(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = out_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    pixels = tmp_path / "pixels"
    pixels.mkdir()
    real = pixels / "real.png"
    real.write_text("pixels", encoding="utf-8")
    (out_dir / "middle").symlink_to(pixels, target_is_directory=True)
    (tmp_path / "bridge").symlink_to(out_dir / "middle", target_is_directory=True)
    entry = tmp_path / "entry.png"
    entry.symlink_to(tmp_path / "bridge" / "real.png")
    assert entry.exists()

    with pytest.raises(ValueError, match="holds the input"):
        prepare_output_dir(out_dir, overwrite=True, protected_paths=(entry,))

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert entry.exists() and real.exists()


def test_cleanup_guard_refuses_the_root_input():
    from pathlib import Path

    from otuformer.utils.io import _refuse_clearing_inputs

    root = Path(Path.cwd().anchor)
    with pytest.raises(ValueError, match="holds the input"):
        _refuse_clearing_inputs(root, (root,))


@pytest.mark.parametrize("cycle", [False, True])
def test_prepare_output_dir_refuses_unverifiable_link_chains(tmp_path, cycle):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = out_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    real = images / "real.png"
    real.write_text("pixels", encoding="utf-8")
    links = [images / f"link_{index}.png" for index in range(65)]
    for link, target in zip(links, links[1:]):
        link.symlink_to(target)
    links[-1].symlink_to(links[0] if cycle else real)

    with pytest.raises(ValueError, match="cannot verify output safety"):
        prepare_output_dir(out_dir, overwrite=True, protected_paths=(links[0],))

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert real.exists()
