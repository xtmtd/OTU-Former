# Pretrain Preflight Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or
> superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reject five pretraining conditions that cannot possibly train, before any output directory, log file,
or checkpoint is touched, using the model and sizes that will actually be trained.

**Spec:** [Pretrain preflight consistency design](../specs/2026-11-04-pretrain-preflight-consistency-design.md).
Baseline `main` at `0144b1f`, package `0.11.1`. **Land before** the relational pretraining targets change.

**Status:** Executed in the working tree: every step below is implemented and its tests are green
(uncommitted, target version `0.11.2`, `fix(pretrain): ...`). Boxes are ticked for implemented and
verified, which is not the same as accepted. No commit, push, tag or release was made.

## Global Constraints

- No CLI option changes, no dependency changes, no loss/schedule/augmentation change, no checkpoint schema change.
- `local_crops == 0` skips the local size check, as today.
- All new checks fail before `prepare_output_dir` / `TeeLogger`; assert that with sentinels, not by inspection.
- Do not keep a duplicate probe to preserve the old RNG consumption; state the reproducibility change instead.

### Task 1: Resolve once, on the resolved backbone

**Files:** Modify `src/otuformer/training/trainer.py` and `src/otuformer/cli/pretrain.py`; test `tests/training/test_pretrain_alignment.py` and `tests/test_cli_smoke.py`.

- [x] Isolate RNG, with its own red/green: wrap `resolve_backbone_native_size` and `validate_model_name_size` (`utils/size.py:63,114`) in `preserved_rng`, and add `test_preflight_probes_do_not_consume_the_seed` in `tests/test_cli_smoke.py` using `CliRunner` for the CLI path and a direct `run_pretrain` call for the other: same `--seed`, `auto` versus explicit size, assert the student `state_dict` is identical. Without this checkbox the goal statement has no test behind it.

**Interfaces:** One pure resolver used by both entry points returns the resolved patch configuration, the
effective global size, and (when locals are used) the effective local size. `run_pretrain` calls it once at
the top; the CLI calls it once and hands the result over. `cli/pretrain.py:377`'s fresh-run `_resolve_patch_config`
call and `trainer.py:2329`'s call both disappear into it.

- [x] Add `test_preflight_resolution_runs_once`: a resume that omits `--patch-loss` reaches both the capability
      check and the trainer with the inherited mode, and an instrumented resolver counts exactly one call.
      Covered by `test_pretrain_resume_trains_the_checkpoint_backbone` (real CLI-to-trainer path, one
      resolution, inherited masked-feature mode) — the first version of this test faked `run_pretrain` and
      was replaced rather than kept as a second test.
- [x] Add `test_preflight_local_size_uses_checkpoint_backbone`: checkpoint backbone differs from
      `args.model_name`, local size legal only for the checkpoint backbone; assert the run validates and trains
      the checkpoint backbone.
- [x] Red: `python -m pytest -q tests/training/test_pretrain_alignment.py -k 'resolve_pretrain_settings or handed_over'`. Expect missing resolver
      or double-resolution failures.
- [x] Implement the resolver; attach its result to the `Namespace` the way `resume_state` is attached, add it to the saver's exclusion set and never write it into saved `args`; delete the two current resolution call sites.
- [x] Green: `python -m pytest -q tests/training/test_pretrain_alignment.py`.

### Task 2: Validate size, grid and backbone capability before output setup

**Files:** Modify `src/otuformer/training/trainer.py`; test `tests/test_cli_smoke.py`.

**Interfaces:** `preflight_pretrain_new_run` builds its probe at the resolved native size instead of `224`, and
the same probe carries the divisibility, grid and capability checks.

- [x] Add three `--overwrite` sentinels that each plant a sentinel file, expect a CLI parameter error, and assert
      the sentinel survives: an invalid local crop size; `patch16` at global size `32` with
      `--patch-loss ibot --masking-strategy blockwise`; a three-block backbone under `--patch-loss masked-feature`. Inject both doubles by monkeypatching `timm.create_model` to return the existing tiny encoder / a three-block encoder with a chosen `default_cfg.input_size`, rather than relying on a real pretrained model.
- [x] Add `test_preflight_resume_failure_leaves_log_unchanged`: a resume whose checkpoint backbone differs from `args.model_name` and whose local size is legal only for `args.model_name` (defect 1) appends nothing to `pretrain.log` and leaves existing outputs byte-identical (resume cannot use `--overwrite`, so the risk here is log pollution, not deletion).
- [x] Add `test_preflight_auto_uses_resolved_native_size`: a test double whose `default_cfg.input_size` is not
      `224`; assert probe, dataset and saved recipe agree on one size.
- [x] Red: `python -m pytest -q tests/test_cli_smoke.py -k 'preflight or overwrite_sentinel'`.
- [x] Implement: resolve sizes first, then run the divisibility checks on the resolved `model_name`, then
      `validate_pretrain_backbone(require_intermediates=<resolved patch mode>)`, then `_validate_blockwise_grid`
      and `patch_count >= 2`. Replace the `224` placeholder with the resolved size. Keep `local_crops == 0`
      exempt from the local check.
- [x] Green: `python -m pytest -q tests/test_cli_smoke.py`.

### Task 3: Record the reproducibility consequence and finish the gates

**Files:** Modify `docs/commands/pretrain.md` and `.cn.md` if they describe validation order;
`docs/superpowers/specs/2026-03-29-otuformer-design.md` §3.1 only if it states the affected behaviour.
Test `tests/test_docs.py`.

- [x] State in the compatibility/version notes that isolating preflight probes under `preserved_rng` changes the
      student initialization a given `--seed` produces relative to `0.11.1`, and that only same-version arm
      consistency is promised.
- [x] Green documentation gate: `python -m pytest -q tests/test_docs.py -k 'pretrain'`.
- [x] Version files for `0.11.2`: `pyproject.toml:7`, `src/otuformer/__init__.py:3`, `README.md:145`, `README.cn.md:144`, `tests/test_version.py:16`, `tests/test_workflow_skill.py:954`.
- [x] Run `python -m pytest -q` and `git diff --check`; record commands, counts and environment limits.
      Unresolved failures are blockers. No commit.

## Handoff to the relational change

The relational targets plan assumes: a resolved backbone, resolved global/local sizes, a single resolved patch
configuration, and every preflight probe RNG-isolated. If any of those is still missing, that plan's
`sources`/`probe` steps are incomplete rather than optional.
