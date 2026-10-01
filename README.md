# OTU-Former

[中文文档](README.cn.md) | **English**

An image-based morphological OTU (Operational Taxonomic Unit) delineation toolkit. Provides a unified `otuformer` CLI for self-supervised pretraining, supervised metric-learning finetuning, embedding extraction, UPGMA hierarchical clustering, expert-corrected annotation, community diversity analysis, CAM visualization, and ONNX model export.

## Why OTU-Former

Turning specimen photographs into reproducible OTUs usually means stitching an embedding script, a clustering notebook, and a diversity calculation together by hand. OTU-Former packages that path into one command-line workflow:

- **Images in, morphological barcodes out.** The toolkit consumes standardised specimen images and produces morphological barcodes, morphOTUs, and diversity data: a trained encoder turns each image into a fixed-dimension embedding, and every downstream step works on those vectors. It does not assemble reads, call targets, or infer ortholog groups upstream.
- **Traceable steps.** Each command writes its outputs, logs, and resolved parameters into a predictable directory, so a run can be inspected, resumed, or reproduced.
- **Expert corrections are first-class.** `annotate` folds manual OTU corrections back into the partition and reports the intra-class distances they imply.
- **Diversity from morphology.** `diversity` computes alpha diversity, including Faith's PD from an OTU-centroid tree built from the embeddings.

## Workflow Overview

1. **`doctor`** — check the environment and dependencies.
2. **`pretrain`** — self-supervised contrastive pretraining; no labels required.
3. **`finetune`** — supervised metric-learning finetuning with a selectable objective.
4. **`extract`** — extract embeddings from images.
5. **`cluster`** — UPGMA hierarchical clustering into morphological OTUs.
6. **`annotate`** — apply expert corrections to the partition.
7. **`diversity`** — compute community alpha diversity indices.

`cam` and `export` are side branches: interpretability heatmaps and ONNX export.

## Requirements

- Python 3.11+
- Operating Systems: macOS, Linux, Windows
- Devices: CPU (required); CUDA GPU / Apple Silicon MPS (optional)

## Installation

Recommended: use an isolated Python environment to avoid dependency conflicts with your system/site-packages. Before installation, clone the repository and enter the project directory:

```bash
git clone https://github.com/xtmtd/OTU-Former.git && cd OTU-Former
```

### Isolated Environment

```bash
# Option 1: conda
conda create -n otuformer python=3.11 -y && conda activate otuformer && pip install -e .
# Option 2: uv + venv
uv venv .venv && source .venv/bin/activate && uv pip install -e .
# Option 3: stdlib venv + pip
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

## Agent Skill (chat-driven workflow)

[`skills/otuformer-workflow/`](skills/otuformer-workflow/SKILL.md) is a portable Skill for agent CLIs; `pip install` does not install it. Register or link it per tool — Pi: `--skill <dir>`, a `skills` entry in `.pi/settings.json`, or `.pi/skills/`/`~/.pi/agent/skills/` ([docs](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/skills.md)); Claude Code: `.claude/skills/<name>/` or `~/.claude/skills/<name>/` ([docs](https://code.claude.com/docs/en/skills)); Codex: a skill directory, or a `[[skills.config]]` entry with `path = ".../SKILL.md"` in `~/.codex/config.toml`, packaged as a skill-only plugin for distribution ([docs](https://developers.openai.com/codex/skills)); OpenCode: `.opencode/skills/<name>/` or `~/.config/opencode/skills/<name>/` ([docs](https://opencode.ai/docs/skills/)).
Per-tool rules differ: Pi and Claude Code load project-level skills only after project trust, and Pi keeps the first skill found on a name collision and warns. A copied Skill must carry `LICENSE.txt` and be given the examples root explicitly. After upgrading `otuformer`, refresh the Skill's prose and scripts; its demo uses all 230 `examples/Epidorcus` images under an approved demo root, never your data silently.

```text
You: run OTU-Former on ./images and tell me what to do next | Bot: doctor -> full parameter card -> your approval -> one step -> results
You: extract embeddings with runs/finetune/finetune_latest.pth | Bot: card -> approval -> runs/extract/embeddings.csv
You: show the 230-image demo, then how far did it get? | Bot: all examples under runs/demo; reads logs/artifacts, then recommends continue / redo / stop
```

## Quick Start

```bash
otuformer doctor
otuformer pretrain --input-images-dir ./images --out-dir runs/pretrain
otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv --input-images-dir ./images --out-dir runs/finetune
otuformer extract --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images --out-dir runs/extract
otuformer cluster --embeddings runs/extract/embeddings.csv --out-dir runs/cluster
```

Every command has its own reference document; they are linked from the table at the end of this file, and `otuformer <command> --help` shows the same link.

<a id="doctor-command"></a>
## Doctor Command

Diagnose environment and dependency readiness.

```bash
otuformer doctor
```

The report covers the Python version, the available devices (`cpu`, `cuda`, `mps`), and key package versions with an ok/missing/outdated status.

<a id="update-command"></a>
## Update Command

```bash
otuformer update
otuformer update --check
otuformer update --yes
```

| Option | Meaning |
|--------|---------|
| `--check` | Only show version information. |
| `--yes`, `-y` | Install without prompting. |

Checks published Git tags and installs the latest release only after confirmation.

## Output Safety

Commands refuse a non-empty `--out-dir` by default. Pass `--overwrite` to clear
it deliberately. Training `--resume` runs keep and append to their existing
output directory; `--resume` and `--overwrite` cannot be combined.

## Hugging Face Weights

New pretrain and finetune runs initialize from public timm weights. An
unauthenticated Hugging Face warning is expected when those weights are first
downloaded or absent from the local cache; `HF_TOKEN` is optional and only raises
rate limits. CAM, extract, and export load their supplied local checkpoint and
do not require Hugging Face access.

## Common Behaviours

### Logging

All commands write a log file (`pretrain.log`, `finetune.log`, `extract.log`, `cluster.log`, `annotate.log`, `diversity.log`, `cam.log`, `export.log`) to the `logs/` subdirectory of the output directory, recording the full command line, a timestamp, all parameter values, and runtime output.

### Graceful Shutdown

Press `Ctrl+C` — the current image finishes before exiting; partial results are saved.

### Device Selection

`--device auto` prefers CUDA, then MPS on Apple Silicon, then CPU.

### Shell Completion

Install shell completion for otuformer:

```bash
otuformer --install-completion
```

Supported shells: bash, zsh, fish

### Version

Current version: `0.11.0`.

Show installed version:

```bash
otuformer --version
otuformer -v
```

## Project Structure

See [`src/otuformer/`](src/otuformer/) for the current package layout and the
[master design](docs/superpowers/specs/2026-03-29-otuformer-design.md) for
architectural responsibilities.

## Commands

| Command | Purpose | Documentation |
|---------|---------|---------------|
| `otuformer doctor` | Diagnose environment and dependency status. | [Doctor](#doctor-command) |
| `otuformer update` | Check for and install the latest published version. | [Update](#update-command) |
| `otuformer pretrain` | Self-supervised contrastive pretraining (DINO/iBOT style). | [docs/commands/pretrain.md](docs/commands/pretrain.md) |
| `otuformer finetune` | Supervised metric-learning finetuning (`--loss` selects the objective). | [docs/commands/finetune.md](docs/commands/finetune.md) |
| `otuformer extract` | Extract image embeddings (ONNX-accelerated). | [docs/commands/extract.md](docs/commands/extract.md) |
| `otuformer cluster` | UPGMA hierarchical clustering into morphological OTUs. | [docs/commands/cluster.md](docs/commands/cluster.md) |
| `otuformer annotate` | Apply expert corrections to produce refined OTU annotations. | [docs/commands/annotate.md](docs/commands/annotate.md) |
| `otuformer diversity` | Compute community alpha diversity indices. | [docs/commands/diversity.md](docs/commands/diversity.md) |
| `otuformer cam` | Generate GradCAM and other heatmaps. | [docs/commands/cam.md](docs/commands/cam.md) |
| `otuformer export` | Export model to ONNX format. | [docs/commands/export.md](docs/commands/export.md) |

Two cross-command contracts have their own documents: [Training Augmentation](docs/commands/training-augmentation.md) for the `--augmentation` and `--orientation-policy` behaviour shared by `pretrain` and `finetune`, and [Embedding Metrics](docs/commands/embedding-metrics.md) for the metrics shared by `pretrain`, `finetune`, and `extract`.

## License

From v0.11.0 this project is distributed under the [OTU-Former Research and Commercial Use License Notice](LICENSE): academic, educational, and non-commercial research use is permitted with the notice retained, while commercial use (including enterprise internal use, paid analysis or consulting, and commercial redistribution) requires prior written permission from the copyright holder. This is source-available software, **not OSI-approved open source**; versions published before v0.11.0 remain available under the MIT License.
Outputs you generate (embeddings, OTU/abundance tables, diversity statistics, figures, CAM images) carry no additional restriction from this notice, while checkpoints, exported models, and third-party material keep their own terms.

## Contact

- Email: `xtmtd.zf@gmail.com`

## Citation

If you use OTU-Former in your research, please cite:

```bibtex
@software{otuformer2026,
  author = {Zhang, Feng},
  title = {OTU-Former: Image-based Morphological OTU Delineation Toolkit},
  year = {2026},
  url = {https://github.com/xtmtd/OTU-Former}
}
```

A preprint is available at: https://doi.org/10.64898/2026.04.28.721370
