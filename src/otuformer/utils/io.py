"""CSV and JSON read/write helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from otuformer.utils.paths import input_dependency_positions


def prepare_output_dir(
    out_dir: Path, *, overwrite: bool = False, allow_existing: bool = False,
    protected_paths: Sequence[object] = (),
) -> Path:
    """Create an output directory without silently replacing existing results."""
    if out_dir.exists() and overwrite:
        targets = {out_dir, out_dir.expanduser().resolve()}
        for value in protected_paths:
            if value is None or value == "":
                continue
            for candidate in input_dependency_positions(Path(str(value))):
                if any(candidate == target or target in candidate.parents for target in targets):
                    raise ValueError(
                        f"--overwrite would clear {out_dir}, which holds the input "
                        f"{candidate}; choose another --out-dir or drop --overwrite."
                    )
        shutil.rmtree(out_dir)
    elif out_dir.exists() and not allow_existing and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {out_dir}. "
            "Choose another --out-dir or pass --overwrite."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict[str, Any], path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
