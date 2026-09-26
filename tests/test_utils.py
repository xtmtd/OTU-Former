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
