# Output Safety and Training Resume Design

## Scope

This change makes output handling safe for every OTU-Former command that writes
to `--out-dir`, adds an update command, makes pretraining resume schedules
continuous, and documents how to add images to continued training. It does not
add incremental-learning algorithms, alter model architectures, or add a
learning-rate scheduler to fine-tuning.

The completed cluster partition-table cleanup is included as context: partition
tables are canonical only under `UPGMA/partitions/tables/` and are no longer
copied to the output root.

## Output Directory Safety

The affected commands are `pretrain`, `finetune`, `extract`, `cluster`,
`annotate`, `diversity`, `cam`, and `export`.

Each command gains a boolean `--overwrite` option, defaulting to false.

- A missing output directory is created normally.
- An existing empty output directory is allowed.
- An existing non-empty output directory causes an immediate error unless the
  user passes `--overwrite` or a permitted `--resume` mode applies.
- `--overwrite` removes the complete output directory before the command writes
  any output, then recreates it.
- The error names the occupied directory and tells the user to select another
  path or pass `--overwrite`.

All entries, including dotfiles such as `.DS_Store`, count as directory
contents. Ignoring them would make `--overwrite` silently delete a file that the
preflight had declared harmless. Users can choose a new directory or explicitly
pass `--overwrite`.

`doctor` has no output directory and is unchanged.

## Update Command

Add `otuformer update` with the same user-facing behavior as EntomoKit's update
command. It works for installed packages without requiring local Git history and
updates only to published Git tags.

- It fetches GitHub repository tags for `xtmtd/OTU-Former`, selects the highest
  valid Semantic Version tag, and compares it with `otuformer.__version__`.
- If the installed version is current or newer, it reports that no update is
  needed and exits successfully.
- Without flags, an available update prompts `Proceed with update? [y/N]`.
- `--check` reports the installed and available versions but never installs.
- `--yes` and `-y` skip the confirmation prompt.
- Installation runs the current interpreter's `python -m pip install --upgrade
  git+https://github.com/xtmtd/OTU-Former.git@v<remote-version>`.
- Network errors and pip failures return a nonzero exit code with a concise
  error message.

The implementation uses standard-library `urllib.request`, `subprocess`, and a
small local Semantic Version parser. It does not add a packaging dependency or
write into `--out-dir`. Unlike EntomoKit, it does not use a main-branch
`version.txt`: this repository has no such file, and using main HEAD could
install unreleased commits after reporting the installed release as current.

### Resume Exception

Only `pretrain` and `finetune` accept `--resume`. For these commands:

- `--resume` and `--overwrite` are mutually exclusive and fail validation when
  used together.
- `--resume` permits a non-empty `--out-dir` without deleting it.
- The resumed run continues to append to its corresponding log file.
- The requested resume checkpoint must exist before output initialization.

Resume is intentionally not inferred from a checkpoint named in `--checkpoint`:
finetune `--checkpoint` remains initialization for a new run and therefore
requires a new or empty output directory unless `--overwrite` is explicit.

### Shared Helper

Add one small helper in `otuformer.utils.io` to validate and prepare an output
directory. It receives `out_dir`, `overwrite`, and `allow_existing`. It performs
only directory policy: callers retain responsibility for command-specific
checkpoint and argument validation. This avoids eight divergent safety checks
without introducing a framework.

## Pretrain Resume Semantics

### Checkpoint Metadata

Each pretrain checkpoint records a `schedule` dictionary containing:

- `total_steps`: the original planned number of optimization steps.
- `steps_per_epoch`: the DataLoader length used by that plan.
- `warmup_steps`: the original warmup length in steps.
- `last_lr`: the learning rate used for the final saved optimization step.

Existing checkpoints without this dictionary remain loadable. They use the
current behavior only when resumed with the same `--max-epochs`; an attempted
extension fails with a clear message that the checkpoint lacks schedule
metadata. This is safer than silently introducing an LR discontinuity.

### Interrupted Resume

When `--max-epochs` equals the original checkpoint plan, pretraining rebuilds
the original schedule from saved metadata and indexes it using the saved global
step. It does not recompute warmup from the current command-line values. The LR,
teacher momentum, and teacher temperature remain at their planned positions.

The current dataset must have the same number of batches as the checkpoint for
this same-plan mode. A changed data set or batch size changes the epoch-to-step
mapping of the original schedule, so this mode rejects
`len(loader) != schedule.steps_per_epoch`.

### Extension

When `--max-epochs` is greater than the original planned epochs and greater than
the completed epoch, `--resume` enters extension mode. Extension deliberately
allows a different number of batches so users can train on the union of old and
new images.

- The original plan is considered complete at `schedule.total_steps`; no warmup
  is repeated.
- The extension duration is `(requested_max_epochs - completed_epochs) *
  new_steps_per_epoch`.
- Its first LR is `schedule.last_lr` (or the LR at the saved global step when
  interrupted), then a cosine curve decays monotonically to the normal final LR
  `args.lr * 0.01`.
- The extension teacher momentum and teacher temperature remain at their mature
  endpoint values rather than restarting their original schedules.

Pretrain checkpoints are saved only after a completed epoch. Therefore,
`completed_epochs` is exactly `checkpoint.epoch + 1`; mid-epoch resume is not
supported. `global_step` remains the schedule index and audit value.

If `--max-epochs` is less than or equal to the completed epoch, the command
fails without writing outputs. If it is between the completed epoch and the
original plan, it also fails: decreasing an existing plan is not a supported
resume operation.

The source checkpoint is read before each run; the chosen resume file may be in
the target output directory. New checkpoints retain the original schedule
metadata plus the active continuation metadata so a later resume is equally
continuous.

## Finetune Resume Semantics

Finetune has no learning-rate scheduler: it restores the model, ArcFace loss
state, optimizer state (including its saved fixed learning rate), epoch, and
iteration. Increasing `--finetune-epochs` with `--resume` therefore continues
from the next epoch without an LR reset. `--finetune-lr` is used only when
starting a new fine-tune from `--checkpoint`.

Its validation mirrors the output policy and rejects `--finetune-epochs` less
than or equal to the checkpoint's completed epoch. No new scheduler or checkpoint
metadata is needed.

For correctness, each new fine-tune checkpoint records the sorted class-label
list used to create its ArcFace head. Resume validates the current list before
loading state, so a same-sized but differently mapped label set cannot silently
corrupt training. Existing checkpoints without this list receive a class-count
validation only, with a warning that exact label compatibility cannot be proved.
Users adding images should retain the old labeled records in the CSV. Adding new
classes is outside resume scope and requires a fresh fine-tune initialized from
a checkpoint.

## Documentation

Add a concise README section near the training commands:

- To add unlabeled images for pretraining, build a CSV containing the union of
  old and new images, use `--resume`, keep the original output directory, and
  increase `--max-epochs`.
- To add labeled images for fine-tuning, use the union CSV with the same label
  set and ordering, use `--resume`, and increase `--finetune-epochs`.
- Training only on newly added images is possible as a new initialization run
  but is not recommended because it can forget prior data.

## Checkpoint Inference and CLI Help

`OTUFormerEncoder` gains a `pretrained: bool = True` constructor parameter and
passes it to every `timm.create_model` call, including the fallback branch. New
pretrain and finetune runs keep the default `True`, so they still initialize
from public timm/Hugging Face weights. CAM, embedding extraction, and ONNX
export construct the encoder with `pretrained=False` before loading their local
checkpoint, so these commands work without network access or a Hugging Face
cache.

The public Hugging Face unauthenticated-request warning is expected during new
pretrain and finetune initialization. Internet access is required only when the
pretrained backbone is absent from the local cache; `HF_TOKEN` is optional and
only increases rate limits. This behavior is documented in both READMEs.

Typer renders options in callback parameter order. `--resume` and `--overwrite`
are moved to the final parameters of `pretrain` and `finetune`, with `--resume`
immediately before `--overwrite`. `--overwrite` is moved to the final parameter
of `extract`, `cluster`, `annotate`, `diversity`, `cam`, and `export`. No custom
Typer or Click sorting hook is needed.

## Tests

- Parameterized command tests cover new directory, empty directory, occupied
  directory rejection, and `--overwrite` clearing stale contents for every
  affected command.
- Update tests cover Git tag retrieval, Semantic Version ordering, check-only
  mode, declined confirmation, tag-pinned installation, and network or pip
  failure paths without making real network or pip calls.
- Pretrain and finetune CLI tests cover mutually exclusive `--resume` and
  `--overwrite`.
- Schedule unit tests verify interrupted resume reproduces the saved next LR,
  extension starts at the saved LR and never increases, unsupported epoch
  requests fail, and older checkpoints reject extension requests.
- Fine-tune tests verify extension epoch validation, exact class-label mismatch
  failure before state loading, and a warning plus class-count fallback for old
  checkpoints.
- Checkpoint inference tests verify CAM, extraction, and ONNX export pass
  `pretrained=False`; model tests verify that value reaches timm.
- CLI help tests verify resume and overwrite controls appear after all other
  command-specific options.
- The existing end-to-end cluster test verifies assignment tables exist only at
  the canonical nested path.

## Non-Goals

- Resuming pretraining across changed batch size, dataset cardinality, or crop
  configuration.
- Expanding an ArcFace head for new classes during fine-tune resume.
- Automatic discovery of prior checkpoints or output directories.
- Retaining legacy root-level partition table copies.
