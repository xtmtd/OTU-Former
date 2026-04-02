import sys
import re
from pathlib import Path

import pandas as pd
import pytest
import torch

from otuformer.utils.checkpoint import load_checkpoint, save_checkpoint
from otuformer.utils.io import read_csv, read_json, write_csv, write_json
from otuformer.utils.logging import TeeLogger


def test_tee_logger_writes_to_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger = TeeLogger(log_file)
    original_stdout = sys.stdout
    sys.stdout = logger
    print("hello world")
    sys.stdout = original_stdout
    logger.close()
    content = log_file.read_text()
    assert re.match(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] hello world\n$", content)


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
