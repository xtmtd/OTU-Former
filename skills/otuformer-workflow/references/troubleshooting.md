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
