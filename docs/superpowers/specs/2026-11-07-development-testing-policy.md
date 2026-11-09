# Development Testing Policy

Date: 2026-11-07
Status: Active project engineering standard.

## Authority and scope

This policy is required by [master design §8.5](2026-03-29-otuformer-design.md#8-engineering-rules)
and applies to project implementation, review, and acceptance. Its test-selection
rules, regression gates, device boundaries, and evidence requirements must be
followed; they are not optional developer advice. This is the single detailed
testing-policy source. The master design states the obligation and links here;
feature designs identify their affected behavior and required checks without
maintaining competing policy matrices. Keep those references aligned when this
policy changes.

This project runs a large suite. Choose checks by impact while iterating and run
the required wider gates at contract boundaries. `python -m pytest` with no
arguments remains the default full-suite command: do not deselect expensive
tests through default `addopts`, environment defaults, or mocks.

## Start from file paths and node IDs

File paths and node IDs are the first selection mechanism:

```bash
python -m pytest tests/training/test_koleo_shared.py
python -m pytest "tests/training/test_dino_pretrain.py::test_dino_koleo_duplicate_records_keep_per_crop_pool"
```

`-k` is useful *inside* an already narrow scope. It is not a guarantee of cheap
collection across the whole repository, because collection still imports every
module it walks.

## Scope by what the change touches

| Change boundary | Iteration / feature-completion coverage | Full regression trigger |
| --- | --- | --- |
| Documentation or help text only | Documentation pair/contract checks, the relevant rendered help, schema/card/skill checks if affected. No unrelated training run. | Global CLI discovery, packaging, or a cross-command semantic change. |
| CLI callback parameter reordering or panel assignment | Rendered panel/option order, parsing/defaults/explicitness, schema/card/argv behavior, and actual callback callers. Preserve valid name/set-based signature consumers. | Completion of the refactor/help gate; widen earlier if positional bindings or shared discovery change. |
| One loss, regularizer, or method | Formula and gradient tests, the relevant real tiny loop, zero/default parity, supported patch-mode interactions. | Shared step mechanics, changed gradient or update ordering, or milestone acceptance. |
| Configuration, resume, or checkpoint identity | New/legacy/malformed preflight, explicitness and inheritance, save/resume, and the directly affected downstream readers. | Completion of a shared format or identity boundary, before declaring compatibility accepted. |
| Module moves, or shared reporting/runtime | Import and monkeypatch ownership, a representative pretrain and finetune run, migrations/logging, and affected direct callers. | End of a coherent refactor stage, or genuine uncertainty about blast radius. |
| Release, dependency, or packaging | The relevant environment and package checks, authorized separately where they build, install, or download. | Full supported regression before merge or release acceptance; report prerequisites separately. |

## Test structure and shared helpers

Split mixed tests along existing behavioral boundaries. Retain real tiny
integration coverage for each method and recovery path. Reuse lightweight real
model helpers when production-backbone behavior is not under test, and preserve
dedicated production-backbone compatibility checks. Shared helpers and fixtures
must be import-safe and preserve per-test state; do not construct global
torch/model objects or share mutable model instances in `conftest.py`.

File paths and node IDs remain the default selection mechanism. Add markers only
when those are insufficient, and register and validate any added markers. This
policy does not introduce xdist or automatic dependency-based selection.

## Devices for training integration tests

Ordinary real training, gradient, checkpoint, and logging integration tests use
`device="auto"` and the existing runtime resolver: CUDA, then MPS, then CPU.
Keep exact numerical/RNG replay tests and CPU dtype/device contracts explicitly
on CPU. Do not change production defaults, install a global default tensor
device, relax assertions, or silently enable accelerator fallback for tests.

A test whose required operations are unsupported on an accelerator stays on CPU
with the reason next to its device setting. Ordinary tiny masked-pretrain
integration uses native-size global and local views on `auto`. Its separate
patch-mode matrix preserves smaller local crops on CPU, because positional
interpolation needs an antialiased bicubic backward unavailable on MPS. Exact
mask replay and fixed-input comparisons retain their original CPU views. CLI
pretrain smoke tests with smaller local crops retain CPU for the same reason.
Constructor snapshots
compared before/after device placement also stay on CPU. Expected tensors in
accelerator tests must be created on the actual tensor device without dropping
shape or numerical assertions. Report the selected device and any retained
CPU-only scope; a pass on one backend does not certify another.

## Practical rules

- **Do not rerun the full suite after every local repair** just because a
  previous full gate failed. Fix with focused regressions, then rerun the
  required gate on the final candidate. A stale pass does not certify new
  changes.
- **If the affected dependencies are unclear, widen coverage** and say why.
- **Failed or skipped prerequisites are not passed verification.** Report
  unrun MPS/CUDA paths, real-data performance, package installation, or an
  unavailable dependency floor as unverified, never as green.
- **A test that fails only because code moved is a patch-ownership bug.** A
  `monkeypatch.setattr` only intercepts a global lookup in the namespace where
  the call happens. Patch the module the caller reads, not the module that
  happens to define the name.

## Measuring instead of guessing

Do not call a test slow because it is long. For work that requires timing
baseline and final acceptance evidence, reuse a valid baseline or obtain one
necessary baseline with `--durations=25`; capture the same evidence on the final
candidate. Record the interpreter, dependency versions, source import path,
baseline commit and dirty state, selected devices, exact invocation, exit
status, and evidence artifact path. Separate collection/import overhead where
measured from setup/teardown and test-call costs, which `--durations` reports
distinctly. Do not promise a wall-clock budget or speedup before measurement.
