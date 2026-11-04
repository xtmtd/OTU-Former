# Pretrain Preflight Consistency Design

**Status:** Implemented in the working tree and verified by the tests named in the plan
(uncommitted, `0.11.2`); not accepted and not released. This document remains the binding spec,
and the sections below are the contract the implementation was written against.

**Baseline:** `main` at `0144b1f`, package `0.11.1`. **Target version `0.11.2`**, a patch release: the option surface, the schema and the on-disk format are unchanged, and the only observable difference is that doomed runs are rejected earlier plus the reproducibility note below. Depends on nothing; must land **before** the
[relational pretraining targets design](2026-11-03-relational-pretrain-targets-design.md), which consumes
its resolved backbone, resolved sizes and single resolved patch configuration.

## Why this is a separate change

Pretraining checks several things too late, or against the wrong model. Each defect can make a run that
cannot possibly train either start, or destroy an existing output directory first when `--overwrite` is
passed (`prepare_output_dir` calls `shutil.rmtree`, `utils/io.py:17-18`, and runs at
`cli/pretrain.py:429`, before `run_pretrain` at `:519`). None of these defects involves relational targets,
so they are specified and reviewed here instead of riding along with a breaking recipe change. Mixing them
in would make a regression impossible to attribute and would widen a `feat(pretrain)!` commit beyond its
title.

## Defects in scope

1. **Local crop size is validated against the wrong backbone.** `trainer.py:2375` calls
   `validate_model_name_size(args.model_name, args.local_crop_size)`, while the model actually built uses the
   checkpoint-resolved `model_name` (`:2320`). On resume from a checkpoint whose recorded backbone differs
   from the invocation default, the check validates a backbone that will not be trained.
2. **Blockwise grid viability is checked after output setup.** `_validate_blockwise_grid` and
   `patch_count < 2` run at `trainer.py:2447-2456`. A `patch16` backbone at global size `32` under
   `--patch-loss ibot --masking-strategy blockwise` passes every size check and then fails there, because a
   `2x2` grid admits no rectangle with sides `>= 2` and area `<= floor(0.20 * 4) == 0`.
3. **Backbone capability is checked after output setup.** `validate_pretrain_backbone`
   (`model.py:370-408`) runs at `trainer.py:2438-2441`. A three-block ViT with a divisible, large-enough size
   passes size and grid checks and still cannot supply the final four block outputs that `masked-feature`
   requires.
4. **The `auto` probe uses a placeholder size.** `preflight_pretrain_new_run` builds its architecture probe at
   `224 if image_size is None else int(image_size)` (`trainer.py:2197`). With `--global-crop-size auto` the
   probe therefore validates a size that is not the resolved one.
5. **Patch configuration is resolved twice.** `_resolve_patch_config` runs in `cli/pretrain.py:377` for a
   fresh-run conflict check and again in `trainer.py:2329`. Two resolutions can disagree with nothing
   detecting it, and on resume the CLI check is skipped entirely, so only the trainer's copy exists.

## Goals

- Every one of these conditions fails **before** `prepare_output_dir`, `TeeLogger`, or any checkpoint write.
- Each check runs against the model that will actually be trained and the sizes that were actually resolved.
- Exactly one patch-configuration resolution exists per process; preflight and the trainer read the same object.
- `local_crops == 0` keeps skipping the local size check, as today.

## Non-goals

- No change to any loss, objective, schedule, augmentation profile, checkpoint schema, or CLI option surface.
- No new CLI flags. This is a correctness and ordering fix.
- No general numerical-stability work and no backbone-family expansion (Eva/rotary stay out of scope, per the
  master design §4.9 boundary).
- No change to `--overwrite` semantics themselves; only to when a doomed run is rejected relative to them.

## Contract

**Single resolution point.** A pure resolver produces the patch configuration, the effective global size, and
(when `local_crops > 0`) the effective local size from `args` plus the loaded checkpoint, using the
checkpoint-resolved `model_name` when resuming. It returns fresh data and never mutates the checkpoint. The
CLI preflight calls it before output setup; a direct `run_pretrain` call calls it at the top of the function,
before any trainer-owned output mutation. Both consume the returned object; neither re-derives it.

**Validation on that object.** After resolution, and on the same architecture probe:

- run the trainer's own divisibility check for the global size, and for the local size only when locals are
  used;
- call `validate_pretrain_backbone` with `require_intermediates` set from the resolved patch mode;
- run the existing blockwise/hybrid rectangular-grid check and the `patch_count >= 2` check.

**Probe size.** The fresh-run architecture probe is built at the **resolved** native input size, replacing the
`224` placeholder rather than adding a second probe. The probe keeps `pretrained=False` and stays
RNG-isolated.

**Handoff.** Concretely it stays a `Namespace` attribute, exactly like `resume_state` (`cli/pretrain.py:518`), and every attribute added for this purpose must appear in the saver's exclusion set. The resolved object itself travels in memory only: it is never serialized into `args` and never becomes a checkpoint key. The already-recorded configuration fields keep their existing on-disk shape — the resolved patch fields still land in `config.*`, which is what a resume reads back. Excluding the handoff object and writing those existing fields are both required. The `Patch config:` log line, target construction, capability checking and
saving all read the one object.

## Reproducibility note

`validate_model_name_size` and `resolve_backbone_native_size` construct `timm.create_model(..., pretrained=False)`
(`utils/size.py:114-119`), which consumes the global RNG. Today those calls happen after `_set_seed` and before
the student is created. Isolating every preflight probe under `preserved_rng` is the right behaviour, but it
**changes the student initialization produced by a given `--seed` relative to `0.11.1`**. This must be stated in
the compatibility notes and the release note. What must hold is that all arms started under one code version
begin from identical initial weights; bitwise reproduction of the old version's initialization is explicitly
not promised, and no duplicate probe may be kept to fake it.

## Verification

- A fresh `--overwrite` run with an invalid local crop size, a blockwise-infeasible grid, or a three-block
  backbone under `masked-feature` leaves an output sentinel untouched and raises a CLI parameter error, with
  output setup and logger creation never reached.
- A resume whose checkpoint backbone differs from `args.model_name`, with a local size legal only for the
  checkpoint backbone, validates and trains the checkpoint backbone.
- An `auto` run whose native size is not `224` uses that size for the probe, the dataset, and the saved recipe,
  asserted to be one value.
- The resolver runs exactly once per process on both the CLI path and a direct `run_pretrain` call; a resume
  that omits `--patch-loss` reaches the capability check and the trainer with the inherited mode.
- `local_crops == 0` still skips the local check.

## Constraints

- `main` only. No dependencies, no version change inside this design revision; the implementation targets `0.11.2` as `fix(pretrain): ...`, with no commit or release here.
- Keep English/Chinese documentation and the runtime-derived skill cards aligned; `docs/commands/pretrain.md`
  gains no new option, only corrected behaviour notes where it describes validation.
- Land before the relational targets change so that change can assume this contract.
