# OTU-Former Plan 1: Project Scaffold + CLI Framework

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the installable `otuformer` package with working CLI entry point, all sub-command stubs, `doctor` command, shared utilities, and `--install-completion` support.

**Architecture:** Typer-based CLI with thin command layer calling core modules. Each sub-command is a separate file in `cli/`. Shared utilities (logging, io, checkpoint) in `utils/`. Package installed via `pyproject.toml` with `hatchling`.

**Tech Stack:** Python ≥ 3.11, Typer ≥ 0.12, hatchling build backend

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package metadata, dependencies, entry point `otuformer = "otuformer.cli.main:app"` |
| `src/otuformer/__init__.py` | Package version |
| `src/otuformer/cli/main.py` | Root Typer app, registers all sub-commands |
| `src/otuformer/cli/pretrain.py` | Stub: argument definitions + `NotImplementedError` |
| `src/otuformer/cli/finetune.py` | Stub |
| `src/otuformer/cli/extract.py` | Stub |
| `src/otuformer/cli/evaluate.py` | Stub |
| `src/otuformer/cli/cluster.py` | Stub |
| `src/otuformer/cli/annotate.py` | Stub |
| `src/otuformer/cli/diversity.py` | Stub |
| `src/otuformer/cli/cam.py` | Stub |
| `src/otuformer/cli/export.py` | Stub |
| `src/otuformer/cli/doctor.py` | Full implementation |
| `src/otuformer/utils/__init__.py` | Empty |
| `src/otuformer/utils/logging.py` | `TeeLogger` class |
| `src/otuformer/utils/io.py` | CSV/JSON read-write helpers |
| `src/otuformer/utils/checkpoint.py` | Checkpoint load/save helpers |
| `tests/test_cli_smoke.py` | Smoke tests: `--help`, `doctor`, stub commands exit cleanly |
| `tests/test_utils.py` | Unit tests for utils/logging.py and utils/io.py |

---

## Task 1: Project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/otuformer/__init__.py`
- Create: `src/otuformer/cli/__init__.py`
- Create: `src/otuformer/utils/__init__.py`
- Create: `src/otuformer/training/__init__.py`
- Create: `src/otuformer/embedding/__init__.py`
- Create: `src/otuformer/delineation/__init__.py`
- Create: `src/otuformer/vision/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "otuformer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "torch>=2.1",
    "timm>=1.0",
    "torchvision>=0.16",
    "numpy>=1.26",
    "pandas>=2.0",
    "scipy>=1.12",
    "scikit-learn>=1.4",
    "scikit-bio>=0.6",
    "umap-learn>=0.5",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "grad-cam>=1.5",
    "onnx>=1.16",
    "onnxruntime>=1.18",
    "tqdm>=4.66",
]

[project.scripts]
otuformer = "otuformer.cli.main:app"

[tool.hatch.build.targets.wheel]
packages = ["src/otuformer"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.coverage.run]
source = ["src/otuformer"]
```

- [ ] **Step 2: Create `src/otuformer/__init__.py`**

```python
"""OTU-Former: image-based morphological OTU delineation toolkit."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create all `__init__.py` stubs**

Create empty `src/otuformer/cli/__init__.py`, `src/otuformer/utils/__init__.py`, `src/otuformer/training/__init__.py`, `src/otuformer/embedding/__init__.py`, `src/otuformer/delineation/__init__.py`, `src/otuformer/vision/__init__.py`.

- [ ] **Step 4: Install package in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: installs without errors, `otuformer` command available.

---

## Task 2: `utils/logging.py` — TeeLogger

**Files:**
- Create: `src/otuformer/utils/logging.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_utils.py
import sys
from pathlib import Path
from otuformer.utils.logging import TeeLogger


def test_tee_logger_writes_to_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger = TeeLogger(log_file)
    original_stdout = sys.stdout
    sys.stdout = logger
    print("hello world")
    sys.stdout = original_stdout
    logger.close()
    assert log_file.read_text() == "hello world\n"


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_utils.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `TeeLogger`**

```python
# src/otuformer/utils/logging.py
"""Logging utilities for OTU-Former CLI."""

from __future__ import annotations

import sys
from pathlib import Path


class TeeLogger:
    """Redirect stdout to both console and file, skipping progress bar lines."""

    SKIP_PATTERNS = ["\r", "|█", "|░", "\x1b[", "[A"]

    def __init__(self, log_path: Path) -> None:
        self.terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("w", encoding="utf-8")
        self._last_was_progress = False

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.terminal.flush()
        is_progress = any(p in message for p in self.SKIP_PATTERNS)
        if not is_progress:
            if self._last_was_progress and message.strip():
                self.log.write("\n")
            self.log.write(message)
            self.log.flush()
            self._last_was_progress = False
        else:
            self._last_was_progress = True

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_utils.py::test_tee_logger_writes_to_file tests/test_utils.py::test_tee_logger_skips_progress_bars -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/utils/logging.py tests/test_utils.py
git commit -m "feat: add TeeLogger utility"
```

---

## Task 3: `utils/io.py` — CSV/JSON helpers

**Files:**
- Modify: `tests/test_utils.py`
- Create: `src/otuformer/utils/io.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_utils.py
import pandas as pd
from otuformer.utils.io import read_csv, write_csv, read_json, write_json


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_utils.py::test_csv_roundtrip tests/test_utils.py::test_json_roundtrip -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `utils/io.py`**

```python
# src/otuformer/utils/io.py
"""CSV and JSON read/write helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV file into DataFrame."""
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False, **kwargs) -> None:
    """Write DataFrame to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)


def read_json(path: Path) -> dict:
    """Read JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: Path, indent: int = 2) -> None:
    """Write dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_utils.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/utils/io.py tests/test_utils.py
git commit -m "feat: add CSV/JSON io helpers"
```

---

## Task 4: `utils/checkpoint.py` — checkpoint helpers

**Files:**
- Create: `src/otuformer/utils/checkpoint.py`
- Modify: `tests/test_utils.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_utils.py
import torch
from otuformer.utils.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    state = {"epoch": 5, "model_state_dict": {}, "config": {"out_dim": 256}}
    p = tmp_path / "ckpt.pt"
    save_checkpoint(state, p)
    loaded = load_checkpoint(p)
    assert loaded["epoch"] == 5
    assert loaded["config"]["out_dim"] == 256


def test_load_checkpoint_missing_file():
    from pathlib import Path
    import pytest
    with pytest.raises(FileNotFoundError):
        load_checkpoint(Path("/nonexistent/path.pt"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_utils.py::test_checkpoint_roundtrip tests/test_utils.py::test_load_checkpoint_missing_file -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `utils/checkpoint.py`**

```python
# src/otuformer/utils/checkpoint.py
"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    """Save training state dict to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load checkpoint from file.

    Args:
        path: Path to checkpoint file.
        map_location: Device to load tensors onto [default: cpu].

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_utils.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/utils/checkpoint.py tests/test_utils.py
git commit -m "feat: add checkpoint save/load helpers"
```

---

## Task 5: `doctor` command — full implementation

**Files:**
- Create: `src/otuformer/cli/doctor.py`
- Create: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write failing smoke test**

```python
# tests/test_cli_smoke.py
from typer.testing import CliRunner
from otuformer.cli.main import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "otuformer" in result.output.lower()


def test_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.output
    assert "torch" in result.output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli_smoke.py -v
```

Expected: FAIL (main.py and doctor.py not yet created).

- [ ] **Step 3: Create `src/otuformer/cli/doctor.py`**

```python
# src/otuformer/cli/doctor.py
"""doctor command — environment health check."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import typer

app = typer.Typer(help="Check environment and dependency health.")


def _pkg_version(pip_name: str, import_name: str | None = None) -> str:
    """Return installed version or 'NOT INSTALLED'."""
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
    """Check environment health: Python, PyTorch, devices, key packages."""
    if ctx.invoked_subcommand is not None:
        return

    typer.echo("\n=== otuformer doctor ===\n")
    typer.echo(f"Python: {sys.version.split()[0]}")

    typer.echo("\nPackages:")
    for pip_name, import_name in _PACKAGES:
        version = _pkg_version(pip_name, import_name)
        typer.echo(f"  {pip_name}: {version}")

    typer.echo("\nDevices:")
    typer.echo("  cpu: available")
    try:
        import torch
        if torch.cuda.is_available():
            typer.echo(f"  cuda: {torch.version.cuda or 'available'}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            typer.echo("  mps: available")
    except ImportError:
        typer.echo("  torch not installed — cannot detect GPU devices")

    typer.echo("")
```

- [ ] **Step 4: Create `src/otuformer/cli/main.py`**

```python
# src/otuformer/cli/main.py
"""OTU-Former CLI — main entry point."""

from __future__ import annotations

import typer

from otuformer.cli import doctor as _doctor_mod

app = typer.Typer(
    help=(
        "OTU-Former: image-based morphological OTU delineation toolkit.\n\n"
        "Quick start:\n\n"
        "  otuformer pretrain  --train-data images.csv --input-images-dir ./images\n\n"
        "  otuformer finetune  --checkpoint runs/pretrain/best.pt --train-data labels.csv\n\n"
        "  otuformer extract   --checkpoint runs/finetune/best.pt --input-images-dir ./images\n\n"
        "  otuformer cluster   --embeddings embeddings.csv\n\n"
        "  otuformer doctor\n"
    ),
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(_doctor_mod.app, name="doctor")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_cli_smoke.py::test_help tests/test_cli_smoke.py::test_doctor -v
```

Expected: PASS.

- [ ] **Step 6: Verify CLI works from terminal**

```bash
otuformer --help
otuformer doctor
```

Expected: help text printed, doctor shows Python version and package list.

- [ ] **Step 7: Commit**

```bash
git add src/otuformer/cli/main.py src/otuformer/cli/doctor.py tests/test_cli_smoke.py
git commit -m "feat: add doctor command and CLI entry point"
```

---

## Task 6: Sub-command stubs (pretrain / finetune / extract / evaluate / cluster / annotate / diversity / cam / export)

Each stub registers all CLI arguments from the spec but raises `typer.Exit` with a "not yet implemented" message. This lets users see the full `--help` for every command immediately.

**Files:**
- Create: `src/otuformer/cli/pretrain.py`
- Create: `src/otuformer/cli/finetune.py`
- Create: `src/otuformer/cli/extract.py`
- Create: `src/otuformer/cli/evaluate.py`
- Create: `src/otuformer/cli/cluster.py`
- Create: `src/otuformer/cli/annotate.py`
- Create: `src/otuformer/cli/diversity.py`
- Create: `src/otuformer/cli/cam.py`
- Create: `src/otuformer/cli/export.py`
- Modify: `src/otuformer/cli/main.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Add stub smoke tests**

```python
# append to tests/test_cli_smoke.py

@pytest.mark.parametrize("cmd", [
    ["pretrain", "--help"],
    ["finetune", "--help"],
    ["extract", "--help"],
    ["evaluate", "--help"],
    ["cluster", "--help"],
    ["annotate", "--help"],
    ["diversity", "--help"],
    ["cam", "--help"],
    ["export", "--help"],
])
def test_subcommand_help(cmd):
    result = runner.invoke(app, cmd)
    assert result.exit_code == 0
```

Add `import pytest` at top of `tests/test_cli_smoke.py`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli_smoke.py::test_subcommand_help -v
```

Expected: FAIL (sub-commands not registered yet).

- [ ] **Step 3: Create `src/otuformer/cli/pretrain.py`**

Full argument list from spec §3.1:

```python
# src/otuformer/cli/pretrain.py
"""pretrain command — SSL self-supervised pre-training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="SSL self-supervised pre-training (teacher-student + masked token regression).")


@app.callback(invoke_without_command=True)
def pretrain(
    ctx: typer.Context,
    train_data: Path = typer.Option(..., "--train-data", help="CSV with 'image' column."),
    input_images_dir: Path = typer.Option(..., "--input-images-dir", help="Root image directory."),
    out_dir: Path = typer.Option(Path("runs/pretrain"), "--out-dir", help="Output directory."),
    model_name: str = typer.Option("vit_small_patch16_224", "--model-name", help="timm backbone."),
    out_dim: int = typer.Option(256, "--out-dim", help="SSL projector output dimension."),
    max_epochs: int = typer.Option(50, "--max-epochs", help="Pretraining epochs."),
    lr: float = typer.Option(5e-4, "--lr", help="Base learning rate."),
    weight_decay: float = typer.Option(0.05, "--weight-decay", help="Weight decay."),
    warmup_epochs: int = typer.Option(3, "--warmup-epochs", help="Warmup epochs."),
    global_crop_size: int = typer.Option(224, "--global-crop-size", help="Global crop resolution."),
    local_crop_size: int = typer.Option(96, "--local-crop-size", help="Local crop resolution."),
    local_crops: int = typer.Option(6, "--local-crops", help="Number of local crops."),
    mask_ratio: float = typer.Option(0.5, "--mask-ratio", help="Masked token ratio."),
    lambda_local: float = typer.Option(1.5, "--lambda-local", help="Weight of local loss."),
    lambda_mask: float = typer.Option(1.0, "--lambda-mask", help="Weight of mask loss."),
    teacher_momentum: float = typer.Option(0.995, "--teacher-momentum", help="Initial EMA momentum."),
    teacher_momentum_end: float = typer.Option(0.999, "--teacher-momentum-end", help="Final EMA momentum."),
    student_temp: float = typer.Option(0.1, "--student-temp", help="Student temperature."),
    teacher_temp_start: float = typer.Option(0.04, "--teacher-temp-start", help="Initial teacher temperature."),
    teacher_temp_end: float = typer.Option(0.07, "--teacher-temp-end", help="Final teacher temperature."),
    disable_cross_view_loss: bool = typer.Option(False, "--disable-cross-view-loss/--no-disable-cross-view-loss", help="Disable cross-view pairing."),
    resume: str = typer.Option("", "--resume", help="Checkpoint path to resume from."),
    log_every_n_steps: int = typer.Option(50, "--log-every-n-steps", help="Log metrics every N iterations."),
    save_every_epochs: int = typer.Option(10, "--save-every-epochs", help="Save checkpoint every N epochs."),
    keep_last_checkpoints: int = typer.Option(10, "--keep-last-checkpoints", help="Keep only last N checkpoints."),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader workers."),
    cpus: int = typer.Option(12, "--cpus", help="CPU threads for PyTorch/MKL."),
    device: str = typer.Option("mps", "--device", help="Device: cpu | cuda | mps."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    """SSL pre-train a ViT encoder.

    \b
    Example:
      otuformer pretrain --train-data images.csv --input-images-dir ./images --out-dir runs/pretrain
    """
    if ctx.invoked_subcommand is not None:
        return
    typer.echo("pretrain: not yet implemented", err=True)
    raise typer.Exit(1)
```

- [ ] **Step 4: Create `src/otuformer/cli/finetune.py`**

Full argument list from spec §3.2:

```python
# src/otuformer/cli/finetune.py
"""finetune command — ArcFace metric learning fine-tuning."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="ArcFace metric learning supervised fine-tuning.")


@app.callback(invoke_without_command=True)
def finetune(
    ctx: typer.Context,
    checkpoint: str = typer.Option("", "--checkpoint", help="Pretrained checkpoint path (auto-detected if empty)."),
    train_data: Path = typer.Option(..., "--train-data", help="CSV with 'image' and 'label' columns."),
    input_images_dir: Path = typer.Option(..., "--input-images-dir", help="Root image directory."),
    out_dir: Path = typer.Option(Path("runs/finetune"), "--out-dir", help="Output directory."),
    model_name: str = typer.Option("vit_small_patch16_224", "--model-name", help="timm backbone (must match pretrain)."),
    metric_embed_dim: int = typer.Option(256, "--metric-embed-dim", help="Embedding dimension for metric learning."),
    finetune_epochs: int = typer.Option(20, "--finetune-epochs", help="Number of ArcFace fine-tuning epochs."),
    finetune_lr: float = typer.Option(1e-4, "--finetune-lr", help="Learning rate for fine-tuning."),
    freeze_ratio: float = typer.Option(0.7, "--freeze-ratio", help="Fraction of transformer blocks to freeze."),
    loss: str = typer.Option("arcface", "--loss", help="Loss function: arcface (proxy_anchor planned)."),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader workers."),
    cpus: int = typer.Option(12, "--cpus", help="CPU threads for PyTorch/MKL."),
    device: str = typer.Option("mps", "--device", help="Device: cpu | cuda | mps."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    log_every_n_steps: int = typer.Option(50, "--log-every-n-steps", help="Log metrics every N iterations."),
    save_every_epochs: int = typer.Option(10, "--save-every-epochs", help="Save checkpoint every N epochs."),
    keep_last_checkpoints: int = typer.Option(10, "--keep-last-checkpoints", help="Keep only last N checkpoints."),
) -> None:
    """Fine-tune with ArcFace metric learning.

    \b
    Example:
      otuformer finetune --checkpoint runs/pretrain/best.pt --train-data labels.csv --input-images-dir ./images
    """
    if ctx.invoked_subcommand is not None:
        return
    typer.echo("finetune: not yet implemented", err=True)
    raise typer.Exit(1)
```

- [ ] **Step 5: Create `src/otuformer/cli/extract.py`**

```python
# src/otuformer/cli/extract.py
"""extract command — embeddings extraction."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Extract embeddings from images using a trained checkpoint.")


@app.callback(invoke_without_command=True)
def extract(
    ctx: typer.Context,
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to pretrain or finetune checkpoint."),
    input_images_dir: Path = typer.Option(..., "--input-images-dir", help="Image directory or parent directory with subdirectories (batch mode)."),
    out_dir: Path = typer.Option(Path("runs/extract"), "--out-dir", help="Output directory."),
    model_name: str = typer.Option("vit_small_patch16_224", "--model-name", help="timm backbone (must match training)."),
    extract_size: int = typer.Option(224, "--extract-size", help="Resize/crop size for extraction."),
    use_projector_output: bool = typer.Option(False, "--use-projector-output/--no-use-projector-output", help="Use projector output instead of CLS token."),
    token_mode: str = typer.Option("cls", "--token-mode", help="Token mode: cls | patch-topk | attention-pool."),
    topk_patches: int = typer.Option(20, "--topk-patches", help="Top-K patches (10/20/30) for patch-topk mode."),
    attention_pooling_type: str = typer.Option("lightweight", "--attention-pooling-type", help="Attention pooling type: lightweight | multihead | gated."),
    attention_pooling_epochs: int = typer.Option(20, "--attention-pooling-epochs", help="Epochs to fine-tune attention query."),
    metrics_sample_size: int = typer.Option(10000, "--metrics-sample-size", help="Max samples for metrics (0=all)."),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader workers."),
    device: str = typer.Option("mps", "--device", help="Device: cpu | cuda | mps."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    prefix: str = typer.Option("OTU", "--prefix", help="OTU name prefix for cluster IDs downstream."),
) -> None:
    """Extract embeddings from images.

    \b
    Single directory:
      otuformer extract --checkpoint best.pt --input-images-dir ./images

    Batch mode (multiple sample sets):
      otuformer extract --checkpoint best.pt --input-images-dir ./dataset
    """
    if ctx.invoked_subcommand is not None:
        return
    typer.echo("extract: not yet implemented", err=True)
    raise typer.Exit(1)
```

- [ ] **Step 6: Create remaining stubs** (`evaluate`, `cluster`, `annotate`, `diversity`, `cam`, `export`)

Follow the same pattern. Each stub:
1. Defines all CLI options from spec §3.4–§3.9
2. Body: `typer.echo("<name>: not yet implemented", err=True); raise typer.Exit(1)`

Key options per command:

**`evaluate.py`:** `--embeddings`, `--labels`, `--out-dir`, `--umap-dims` (default 2), `--umap-n-neighbors` (15), `--umap-min-dist` (0.1), `--umap-metric` (cosine), `--visualize-class-number` (20), `--knn-k` (str default "1,5,10"), `--metrics-sample-size` (10000)

**`cluster.py`:** `--embeddings`, `--out-dir`, `--distance` (cosine), `--prefix` (OTU), `--pca-whitening` (bool flag), `--pca-components` (256), `--local-scaling` (bool flag), `--local-k` (0), `--local-k-strategy` (adaptive), `--cutoff-min` (0.05), `--cutoff-max` (1.0), `--cutoff-step` (0.05), `--custom-cutoffs` (str optional), `--num-bootstraps` (0), `--bootstrap-subsample-ratio` (0.8), `--bootstrap-display-cutoff` (50.0), `--save-distances` (bool flag), `--max-distance-pairs` (1000000), `--labels` (optional path), `--metrics-sample-size` (10000), `--cpus` (8), `--random-state` (42)

**`annotate.py`:** `--assignments`, `--corrections`, `--out-dir`

**`diversity.py`:** `--assignments`, `--out-dir`, `--prefix` (OTU), `--min-abundance` (str default "0,2,5"), `--phylo` (bool flag), `--tree` (optional path)

**`cam.py`:** `--checkpoint`, `--images-dir`, `--label-csv` (optional), `--out-dir`, `--cam-method` (gradcam), `--arch` (optional str), `--target-layer-name` (optional str), `--image-weight` (0.5), `--fig-format` (png), `--save-npy` (bool flag), `--dump-model-structure` (bool flag), `--max-images` (optional int), `--cam-batch-size` (32), `--num-workers` (4), `--device` (mps)

**`export.py`:** `--checkpoint`, `--out-dir`, `--imgsz` (224), `--opset` (17)

- [ ] **Step 7: Register all sub-commands in `main.py`**

```python
# src/otuformer/cli/main.py — updated imports and registrations
from otuformer.cli import (
    doctor as _doctor_mod,
    pretrain as _pretrain_mod,
    finetune as _finetune_mod,
    extract as _extract_mod,
    evaluate as _evaluate_mod,
    cluster as _cluster_mod,
    annotate as _annotate_mod,
    diversity as _diversity_mod,
    cam as _cam_mod,
    export as _export_mod,
)

# ... (keep existing app definition) ...

app.add_typer(_doctor_mod.app, name="doctor")
app.add_typer(_pretrain_mod.app, name="pretrain")
app.add_typer(_finetune_mod.app, name="finetune")
app.add_typer(_extract_mod.app, name="extract")
app.add_typer(_evaluate_mod.app, name="evaluate")
app.add_typer(_cluster_mod.app, name="cluster")
app.add_typer(_annotate_mod.app, name="annotate")
app.add_typer(_diversity_mod.app, name="diversity")
app.add_typer(_cam_mod.app, name="cam")
app.add_typer(_export_mod.app, name="export")
```

- [ ] **Step 8: Run all smoke tests**

```bash
pytest tests/test_cli_smoke.py -v
```

Expected: all PASS (--help for every command exits 0).

- [ ] **Step 9: Verify `--install-completion` works**

```bash
otuformer --install-completion
```

Expected: shell completion instructions printed (Typer built-in).

- [ ] **Step 10: Commit**

```bash
git add src/otuformer/cli/ tests/test_cli_smoke.py
git commit -m "feat: add all CLI sub-command stubs with full argument definitions"
```

---

## Task 7: Git init and final smoke test

- [ ] **Step 1: Initialise git repo (if not already)**

```bash
git init
git add .
git commit -m "chore: initial project scaffold"
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 3: Verify full CLI**

```bash
otuformer --help
otuformer doctor
otuformer pretrain --help
otuformer cluster --help
otuformer diversity --help
```

Expected: all print help without errors.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: plan 1 complete — project scaffold and CLI framework"
```
