# OTU-Former

OTU-Former is an image-based morphological OTU delineation toolkit.
It provides a Typer CLI for pretraining, finetuning, embedding extraction,
evaluation, clustering, annotation, diversity analysis, CAM visualization, and
model export.

## Requirements

- Python 3.11+
- macOS / Linux / Windows (CPU supported; GPU optional)

## Installation

Minimal install:

```bash
pip install -e .
```

For local development and testing (`pytest`):

```bash
pip install -e ".[dev]"
```

Optional `uv` equivalent:

```bash
uv pip install -e .
# or
uv pip install -e ".[dev]"
```

## Quick Start

Check environment and dependencies:

```bash
otuformer doctor
```

View command help:

```bash
otuformer --help
otuformer pretrain --help
```

### Export and Extract with ONNX

Export a checkpoint to ONNX for fast CPU inference:

```bash
otuformer export --checkpoint runs/finetune/finetune_latest.pth --out-dir runs/export
```

Use the exported ONNX model for embedding extraction (2-5x faster on CPU):

```bash
otuformer extract \
  --checkpoint runs/finetune/finetune_latest.pth \
  --input-images-dir examples/images \
  --onnx-path runs/export/encoder.onnx \
  --out-dir runs/extract
```

## Run Tests

```bash
pytest
```

## Project Structure

- `src/otuformer/`: core package and CLI commands
- `tests/`: unit and integration tests
- `examples/`: sample data assets
- `ref/`: reference scripts and notes

## License

This project is licensed under the MIT License. See `LICENSE`.
