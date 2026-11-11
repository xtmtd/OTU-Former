# Troubleshooting and recovery

Recovery is conversational: report what was observed, keep state in the
conversation and in existing artifacts, and ask before changing anything. There is
no Skill progress database, no automatic rerun, and no implicit cleanup.

## Schema export fails

Read-only inspection may continue, with an explicit limitation:

```text
Parameter source: CLI --help fallback
Complete structured validation: unavailable
```

Do not build a truncated parameter card from memory, and do not treat textual help
as equivalent structured checks. Analysis execution and installation updates stay
blocked until `export_cli_schema.py --command <name>` succeeds again. Diagnose with
`otuformer <command> --help`, the installed package version, and the interpreter
actually used.

## Environment mismatch

The adapter reports the package version, package path, and interpreter that
actually loaded. In proposal mode the launcher is verified separately: a launcher
that reports another version, uses another interpreter, or cannot be inspected
blocks execution instead of being guessed at. Fix the environment (or pass the
matching interpreter/console script) before approving any analysis.

## Doctor reports missing packages

`doctor` can exit zero while dependencies are missing, so read the report instead
of the exit code. Report the actual missing packages and available devices, then
ask whether to install them; installation needs its own approval.

## A step failed

1. Report the exit status, the failing message, and the log path.
2. Do not delete, move, or overwrite the failed output directory.
3. Do not rerun automatically; propose the smallest next action and let the user
   approve it.
4. Prefer a fresh sibling directory for a retry. `--overwrite` requires separate
   approval naming the resolved directory, and supported training `--resume` is the
   only CLI-provided way to continue an existing run.

## Output directory already exists

Non-empty directories are never silently reused. Offer a fresh path, a supported
training `--resume`, or explicit overwrite approval that names the resolved
directory. Never overwrite to get past a compatibility error.

## Long runs

Use the agent's existing process support, and verify process/log evidence before
claiming a run started. Progress must come from the current run's logs and
metrics. On resumption, inspect artifacts and ask which run is intended when it is
ambiguous; reuse verified completed results instead of relaunching.

## Checkpoint and resume problems

Resume compatibility, loss applicability, inherited settings, and pseudo-label
source identity are checked by the CLI preflight when the run starts. Report those
errors as they are; do not work around them by materializing defaults, dropping
flags, or re-initializing from unrelated weights.

Relational-specific resume errors name their own cause and are not workarounds:

- `partial relational target metadata` — the checkpoint has one of the new-style
  keys without the complete canonical bundle. It is not a legacy payload; start a
  new run or resume from an intact checkpoint.
- `cannot change SSL method on resume` — the checkpoint records
the other `--ssl-method`. A method switch needs a new run; resuming within the
recorded method stays supported, including a DINO payload.
- `Unsupported SSL head` / `Unsupported SSL method` — the checkpoint's SSL
  identity is missing or unreadable, so neither method can claim it. Start a new
  run or resume from an intact checkpoint.
- `conflicts with the recorded recipe value` — an explicit relational or
  temperature option differs from the checkpoint. A changed training recipe
  requires a new run.
- `weak teacher augmentation config no longer matches` — the recorded weak
  transform is not what this code derives, including when only the recorded
  normalization differs. Resume with the version that produced the record.
- `overflows the float32 teacher relation reduction` — a recorded endpoint is
  numerically unusable at this run's batch size. Increase the temperature in a
  new run; do not bypass the guard.

Queue-enabled pretraining adds its own enabled-only rejections, none of which is a
workaround target:

- `one CSV row per image identity` — two selected rows share one in-root key.
  Remove the duplicate row, or run with `--relational-queue-size 0` and explain
  that this keeps the legacy batch-only input contract.
- `inside the declared image root` / `decodable images` — the enabled input
  contract needs in-root, readable images. Do not move files or edit the CSV
  mid-investigation on the operator's behalf.
- `Cannot change --batch-size` / `Cannot change --relational-queue-size` — an
  enabled queue resume locks the saved recipe. A different recipe needs a new run.
- `truncated history` / `latest successful batch` — the saved queue state is
  incomplete. Do not truncate, resize or silently recreate history; resume from an
  intact checkpoint or start a new run.
- `requires the saved rng_state` / `records no CUDA RNG` — an enabled queue resume
  never substitutes a fresh random sequence. Report it as an unsupported recovery
  target rather than seeding one.

`--disable-cross-view-loss` was removed: use `--relational-pairing` instead. The
internal historical `same` pairing is only ever inherited from a legacy
checkpoint and is not a request value.

## Ambiguous scientific results

Keep the distinctions the project documents: morphOTUs are not verified species,
label classes are not ecological samples, morphological trees are not established
phylogenies, and raw-feature gradient CAM is not validated class-discriminative
attribution. Preserve real metric names and explain their limits rather than
renaming or rounding them.

## Teaching mode limits

Short demonstration training is procedural only and is not evidence of encoder
quality. If the example assets are missing (for example the Skill was copied
outside the repository), ask for the examples root instead of searching or
downloading data.

## Missing polygon/RLE decoder

`mask_decoder_unavailable` (or a message about the optional decoder) means the
`annotations` extra is not installed in the approved interpreter. Installing it is
a separate approval; do not substitute a missing decoder check with a skip, and do
not re-run the extraction before the validation failure is resolved.

## Shared data-path validation failure

A failed shared-path run publishes `data_validation.json` plus a `failed`
`run_provenance.json`, and no embedding/mapping pair. Report the issue codes and the
affected image keys instead of retrying; missing annotations must not be reread as
biological absence, and an `interrupted` or `running` status is never success.
