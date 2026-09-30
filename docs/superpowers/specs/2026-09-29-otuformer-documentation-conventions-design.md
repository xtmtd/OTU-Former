# OTU-Former Documentation Conventions — Design

Date: 2026-09-29
Status: Proposed — awaiting operator approval. No README, CLI help, test, or
version change is permitted before approval.
Target release: v0.10.1 (documentation-only patch; no functional change).
Delivery: one uncommitted working-tree change set. The operator commits and
tags; this design does not prescribe a commit split.

## 1. Purpose and scope

The two READMEs are 1047 (`README.md`) and 954 (`README.cn.md`) lines, and
about 60% of that is per-command reference material that grows with every
release. They are hard to navigate and every feature release edits the same two
files, which is why the v0.10.0 register work left `## Overview`'s command list
and the design spec's flag lists drifting from the code.

The immediate goal is to restructure both READMEs into a short front page plus
one reference document per command, following the shape used by the sibling
`phyloAI` project (`README.md` 161 lines + `docs/commands/*.md` + `.zh.md`
twins).

The durable goal is that the rules outlive this change. Sections 2–7 of this
document are the standing conventions: the layer model, the language and naming
rules, the command-document skeleton, the `--help` boundary, the new-module
checklist, and the test corpus contract. Section 8 is the one-time landing plan
for v0.10.1. The master design
([2026-03-29-otuformer-design.md](2026-03-29-otuformer-design.md)) gains a short
`Documentation Conventions` section that states the load-bearing rules and links
here; it does not copy this document.

In scope:

- splitting the command reference out of both READMEs into `docs/commands/`;
- moving the three multi-paragraph, app-level `--help` blocks into those
  documents;
- establishing the conventions above as the standing rules for future modules;
- retargeting the existing documentation assertions onto a documentation corpus;
- the v0.10.1 version bump.

Out of scope:

- any change to CLI behaviour, options, defaults, or output files;
- shortening any existing per-option `--help` string (Section 5.3);
- a `CHANGELOG.md` (Section 9);
- restructuring `docs/superpowers/{plans,specs}` or renaming `README.cn.md`;
- adding CI (the repository has none; the test suite is the gate).

The command reference is migrated, not rewritten: its content is moved without
dropping documented behaviour, and every version-bearing behavioural statement
currently in either README must survive. That set is not a fixed list of release
numbers: it includes `0.2.1` (the `legacy` augmentation reproduction target),
`v0.6.x` (the old `--mask-ratio` default), `v0.7.0`, `v0.7.1`, `v0.7.x`
(fine-tune continuation), `v0.8.0`, `v0.9.0`, and `v0.10.0`. Before implementing,
extract the version-bearing sentences from both READMEs and confirm a
destination file for each. README framing and navigation (`Why`,
`Workflow Overview`, `Quick Start`) may be rewritten for brevity, and the master
design's Section 3 flag lists are deleted in favour of their new home. Neither
is a licence to change a documented flag, default, or behaviour.

## 2. Layer model and link direction

Documentation has three layers with one owner each. The link direction is fixed
so no layer can become a copy of another:

```text
README.md / README.cn.md
    front page: positioning, installation, quick start, cross-cutting behaviour,
    the command table
        └── links to →  docs/commands/<command>.md (+ .cn.md)
                            user-facing usage: flags, defaults, examples, outputs

docs/superpowers/specs/*.md
    design record: purpose, contracts, invariants, decisions
        └── links to →  further design specs (per-version designs)

docs/superpowers/plans/*.md
    execution plans for multi-step work
```

- **README owns navigation and setup.** It may inline installation and
  operational content (Section 6.2) and must not inline analysis-command
  reference prose.
- **`docs/commands/` owns the analysis-command flag surface.** The complete
  parameter reference and the worked examples for `pretrain` … `export` live
  there. README and `--help` carry only minimal quick-start invocations; those are
  non-authoritative signposts and must stay valid. The inline operational
  surfaces for `doctor` and `update` (Section 6.2) are owned by the README, whose
  Update section lists every option the CLI defines, including the `-y` alias of
  `--yes`.
- **Design specs own intent.** They state contracts and link to the per-version
  design that introduced them. They may name flags where a decision or invariant
  requires it — `--mask-ratio auto` resolving to `0.30` is a contract, not a
  default — but they do not duplicate an exhaustive current parameter table.
- **The master design links only to other design specs**, never to
  `docs/commands/`. The reader path from the master design to current flags is
  README → command document, stated once as a sentence in the master design's
  documentation section.

## 3. Language and naming conventions

3.1 **Full bilingual mirror.** Every document under `docs/commands/` has an
English file and a Chinese twin. This is a hard convention for new modules, not
an optional nicety: `docs/commands/<name>.md` requires
`docs/commands/<name>.cn.md` in the same change.

3.2 **Suffix is `.cn.md`** throughout this repository, matching the existing
`README.cn.md`. Do not rename `README.cn.md` and do not adopt phyloAI's `.zh.md`
suffix: the rename would touch `pyproject.toml`, four test files, and external
links, and would fix nothing.

3.3 **Every document opens with a language switch**, immediately after the
title:

```markdown
# otuformer pretrain

[English](pretrain.md) | [中文](pretrain.cn.md)
```

3.4 **Language content split.** The English files carry the full reference. The
Chinese files are a natural Chinese rendering of the same sections in the same
order. Keep literal CLI flags, paths, file names, checkpoint keys, CSV column
names, and code identifiers in English inside the Chinese files, exactly as
`README.cn.md` does today.

3.5 **Section names stay parallel** between the two languages so a diff of a
translated pair is reviewable. Command documents use `Purpose`/`目的`,
`Usage`/`用法`, `Parameters`/`参数`, `Inputs`/`输入`, `Outputs`/`输出`,
`Examples`/`示例`, `Notes`/`说明`, `Version Notes`/`版本注记`. The two reference
documents use additional headings, whose translations are fixed here so the
structure test can normalize both sides instead of guessing at a rendering:

| English | Chinese |
|---|---|
| `Affected Commands` | `适用命令` |
| `Profiles` | `配置` |
| `Orientation Policies` | `朝向策略` |
| `Resume and Inheritance` | `续训与继承` |
| `Metrics` | `指标` |
| `Availability Rules` | `可用性规则` |
| `CSV Fields` | `CSV 字段` |
| `Comparability` | `可比性` |

README headings are fixed here for the same reason: every README section name
appears in both languages below, so the tests can select a parse boundary by the
file's language instead of guessing at a translated heading. Names that already
exist in `README.cn.md` are kept as they are; `Why OTU-Former`, `Workflow
Overview`, `Quick Start`, and `Commands` are new.

| English | Chinese |
|---|---|
| `## Why OTU-Former` | `## 为什么选择 OTU-Former` |
| `## Workflow Overview` | `## 工作流程` |
| `## Requirements` | `## 系统要求` |
| `## Installation` | `## 安装` |
| `## Quick Start` | `## 快速开始` |
| `## Doctor Command` | `## doctor 命令` |
| `## Update Command` | `## update 命令` |
| `## Output Safety` | `## 输出安全` |
| `## Hugging Face Weights` | `## Hugging Face 权重` |
| `## Common Behaviours` | `## 通用行为` |
| `## Project Structure` | `## 项目结构` |
| `## Commands` | `## 命令` |
| `## License` | `## 许可证` |
| `## Contact` | `## 联系方式` |
| `## Citation` | `## 引用` |

## 4. Command document skeleton

Every command document uses this fixed order and never reorders it:

| Section | Content |
|---|---|
| `## Purpose` | What the command does and where it sits in the workflow; one or two short paragraphs. |
| `## Usage` | Minimal invocation, then progressively richer invocations. |
| `## Parameters` | One block per flag (aliases share a block), headed by the flag, then a line with its default and — when the CLI enumerates them — its allowed values, then a description at least as detailed as that option's `--help`. A definition list rather than a table: a table squeezes the flag column and forces a long description into a single cell. |
| `## Inputs` | Required and optional input files, with their required columns or layout. |
| `## Outputs` | Every file and directory written, with a one-line purpose each. |
| `## Examples` | Copy-pasteable examples, each captioned with what it demonstrates. |
| `## Notes` | Contracts and caveats: resume inheritance, validation order, interactions. |
| `## Version Notes` | Version-tagged behaviour, newest first. |

Three sections are required in every command document: `Purpose`, `Usage`, and
`Parameters`. `Inputs`, `Outputs`, `Examples`, and `Notes` are omitted when
genuinely inapplicable. `Version Notes` is required whenever the document has
recoverable command-specific history: always for a new command, and for a
migrated document from its first post-v0.10.1 change (Section 4.3). A migrated
document with none — `cluster`, `annotate`, `diversity`, `cam`, and `export`
today — omits the section rather than inventing entries. Section 7.4 checks the
required set.

4.1 **Each command document's `Parameters` section is authoritative for that
command's CLI syntax, defaults, and allowed values, and is at least as detailed
as generated `--help`.** The CLI is the executable source of truth; the document
is the authoritative prose reference and must contain everything `--help` says
about an option, including the constraints and caveats `--help` states, before
adding anything of its own. The CLI implementation is
the executable source of truth; the table is the authoritative prose reference
and must match generated `--help`. A cross-cutting document owns a shared
behavioural contract and links back to the affected command documents without
duplicating their tabular definitions (Section 4.4). The parameter tables remain
owned by those command documents; no extra anchor is required for them.

4.2 **Headings never carry a version number**, in command and reference documents
alike. `### Embedding Metrics (v0.7.1)` becomes `## Embedding Metrics` in the
metrics reference document, with the version recorded in that document's
`Version Notes`.

4.3 **`Version Notes` is the sole location for command- or reference-specific
user-visible change history within the documentation.** This covers command
documents and the
two reference documents alike: the metrics reference records the v0.7.1
comparability break, and the augmentation reference records the `legacy` 0.2.1
reproduction target. Package versions also appear
legitimately in `pyproject.toml`, the READMEs' `### Version` line, and design
spec titles; those are unaffected. The section lists, newest first, the version
that introduced or changed each user-visible behaviour:

```markdown
## Version Notes

- **v0.10.0** — `--register-tokens` selects `none`, `0`, or `4`; the resolved
  count is recorded in checkpoints as `num_prefix_tokens`.
- **v0.7.0** — `--patch-loss` selects the patch objective ...
```

The rule is forward-looking, not retroactive. The version tags already present
in the READMEs move with their content into the documents that now own it. From
v0.10.1 onward, a change to user-visible behaviour appends an entry to each
document it affects, in the same change; a document a release did not change
gets no entry for that release, and no document is given empty historical
entries that cannot be reconstructed from the repository's design specs.

4.4 **Cross-cutting contracts get a reference document of their own** rather
than being duplicated in the commands that share them. v0.10.1 introduces two:
`docs/commands/training-augmentation.md` owns the `--augmentation` and
`--orientation-policy` behaviour shared by `pretrain` and `finetune`, and
`docs/commands/embedding-metrics.md` owns the embedding-quality metric contract
shared by `pretrain`, `finetune`, and `extract`.

A reference document is not a command document, so it does not use the Section 4
skeleton. It has no `## Parameters` section, and its section list is chosen to fit
its own contract rather than copied from the other reference document. The
augmentation document uses:

```text
Purpose
Usage                 the affected commands' invocations
Affected Commands     which commands the contract covers, with a link to each
Profiles              what each --augmentation value does
Orientation Policies  what each --orientation-policy value does
Resume and Inheritance
Notes                 safety caveats
Version Notes
```

and the metrics document uses:

```text
Purpose
Affected Commands     which commands compute these metrics
Metrics               each metric's definition
Availability Rules    when a metric is unavailable, and what is written instead
CSV Fields            the shared field names and the files that hold them
Version Notes         cross-version rules and their release
```

The tabular definition of each shared flag — flag, default, allowed values, one-
line meaning — stays in the affected command documents' `Parameters` sections and
links here. A reference document necessarily names each profile, policy, or
metric while explaining its behaviour, but it does not duplicate the
command-specific defaults or the compact allowed-values table; that duplication
is what Section 4.1 forbids.

## 5. The `--help` boundary

5.1 **`--help` is an index, not a manual.** It carries the command's one-line
purpose, its examples, and per-option help, plus a URL pointer (Section 5.5) to
the command document for contracts that need more than a sentence. For the
commands whose content is inlined in the README, the pointer targets the README
(Section 6.2).

5.2 **A pointer replaces a long block, not a short one.** A multi-paragraph
block with its own sub-headings belongs in the command document. A per-option
help string — however long — stays in the CLI, because it is read at the moment
of use.

5.3 **Existing per-option help is not shortened.** The only per-option help edits
in v0.10.1 are URL repointing (Section 8.3) and the short warning phrases
retained from removed app-level blocks (Section 5.4, listed in Section 8.3). If a
future change wants to compress a specific per-option help, it lists the current
text and the proposed text for operator approval before applying it; blanket "one
sentence per option" rules are rejected.

5.4 **Safety-relevant warnings stay visible at the point of use.** The full
explanation moves to the document, but the warning itself stays in the CLI: a
phrase that changes what a user would type is retained in the relevant option's
help. v0.10.1 retains exactly four phrases, enumerated in Section 8.3 — that
`color-robust` may suppress diagnostic colour, pattern, or metallic sheen; that
`legacy` reproduces 0.2.1; that `conservative` is experimental; and that omitted
experiment options inherit the finetune#1 resolved values. The rule is "short
warning in the CLI, full explanation in the document", not "move the warning out
of the CLI as well".

5.5 **Help pointers use a stable URL, not a repository-relative path.** The
built wheel packages `src/otuformer` only (`pyproject.toml`
`[tool.hatch.build.targets.wheel]`), so a user who installed through `pip` or
`otuformer update` has no `docs/` tree on disk. A path such as
`docs/commands/pretrain.md` would therefore be dead for exactly those users.
Pointers in `--help` use the published URL form:

```text
https://github.com/xtmtd/OTU-Former/blob/main/docs/commands/<name>.md
```

Define the base once as a module-level constant in `src/otuformer/cli/` and
compose per-document URLs from it. This also repairs a defect that already
exists: today's help text points at "the README", which the wheel does not ship
either. The `main` ref is a deliberate simplification and can show
documentation newer than the installed version; if that ever matters, resolve
the ref from `__version__` with a `main` fallback for untagged installs, but do
not add that now.

## 6. Rules for a new CLI command

6.1 **Checklist.** A new command is not complete until all of the following
hold. An internal module with no CLI surface needs no README row and no command
document; document it in the relevant design spec only when it introduces an
architectural contract or user-visible behaviour. Under the current
architecture, a top-level command is a single-callback sub-Typer app exposing
named options rather than positional arguments (Section 7.6 enforces this):

1. **README command table**: one row added, with a link to the new document. No
   per-command prose is added to the README.
2. **Bilingual pair**: `docs/commands/<name>.md` and
   `docs/commands/<name>.cn.md`, both created in the same change, both opening
   with the language switch line.
3. **Skeleton**: the Section 4 section order, with the required sections present.
4. **Parameter fidelity**: every flag has its own block under `## Parameters`,
   stating its default and, where the CLI enumerates them, its allowed values. The
   CLI is the executable source; the blocks must match generated `--help`, and
   Section 7.6 enforces the option names structurally.
5. **Version note**: a `Version Notes` entry for anything introduced or changed
   in this release.
6. **Help boundary**: the app-level `--help` carries purpose, examples, and a
   document pointer; no multi-paragraph contract block.
7. **Design record**: if the module has a design spec, its per-command section
   states purpose, outputs, key constraints, and links to the relevant design
   specs; it does not repeat the flag list.
8. **Test coverage**: the documentation corpus test (Section 7) covers any
   user-visible contract the module introduces.
9. **Structure and pairing**: `test_every_command_doc_has_a_chinese_twin` and
   `test_command_doc_pairs_are_well_formed` pass.

6.2 **README exceptions (setup and operational content).** Installation,
`doctor`, `update`, and the cross-cutting behaviours (Logging, Graceful
Shutdown, Device Selection, Output Safety, Hugging Face weights, Version, Shell
Completion) stay inline in the READMEs. They are read before a run begins, next
to installation, and are short; splitting them into separate documents would add
a navigation hop and two more files without reducing duplication. This is the
only exception, and it covers setup and operational commands only — never the
analysis commands (`pretrain` … `export`).

6.3 **Command-table links.** Each README links to its own language's documents:
`README.md` → `docs/commands/<name>.md`, `README.cn.md` →
`docs/commands/<name>.cn.md`. Cross-language links would drop a Chinese reader
into an English document and make them switch by hand.

For `doctor` and `update`, which stay inline (Section 6.2), both READMEs carry an
explicit HTML anchor above the heading:

```html
<a id="doctor-command"></a>
## Doctor Command
```

so the table can link `#doctor-command` and `#update-command` from either file.
Both files carry both anchors, each immediately above its own second-level
heading, so the English and Chinese headings may differ in wording while the
anchors stay identical.
GitHub's automatic heading slugs are language- and Unicode-dependent —
`## doctor 命令` does not produce `#doctor-command` — so relying on them would
break the Chinese table. This is the repository's first use of intra-README
anchors; the rule is recorded here so it is not mistaken for a broken link. The
link test (Section 7.8) requires every fragment to correspond to an explicit
`<a id>`, so these two anchors are load-bearing rather than decorative.

## 7. Test corpus contract

Documentation assertions currently read `README.md` and `README.cn.md` by name.
Once the reference moves, those assertions would fail for a reason unrelated to
what they check. The replacement is a corpus helper so the assertions keep
testing the content and not the file layout.

7.1 **`tests/test_docs.py` provides the helpers:**

```python
COMMAND_DOCS = ROOT / "docs" / "commands"

def en_docs() -> list[tuple[Path, str]]:
    # README.md plus every command document that is NOT a .cn.md twin
    paths = [ROOT / "README.md"] + [
        p for p in sorted(COMMAND_DOCS.glob("*.md"))
        if not p.name.endswith(".cn.md")
    ]
    return [(p, p.read_text(encoding="utf-8")) for p in paths]

def cn_docs() -> list[tuple[Path, str]]:
    # README.cn.md plus every .cn.md twin
    paths = [ROOT / "README.cn.md"] + sorted(COMMAND_DOCS.glob("*.cn.md"))
    return [(p, p.read_text(encoding="utf-8")) for p in paths]

def all_docs() -> list[tuple[Path, str]]:
    return en_docs() + cn_docs()
```

The `not p.name.endswith(".cn.md")` filter is load-bearing: `*.md` also matches
`*.cn.md`, so without it an English-side assertion could be satisfied by a
Chinese document and would stop testing anything. Each helper returns
`(path, text)` pairs discovered by glob, so a new document is covered
automatically. `all_docs()` is the union of the two sides.

The same module defines the document **sets**, so the closure checks cannot
disagree about what a command document is:

```python
REFERENCE_DOCS = {"training-augmentation", "embedding-metrics"}

def english_doc_names() -> set[str]:
    # name.removesuffix, never Path.stem: Path("pretrain.cn.md").stem is
    # "pretrain.cn", which would leak Chinese twins into a command-name set.
    return {
        p.name.removesuffix(".md")
        for p in COMMAND_DOCS.glob("*.md")
        if not p.name.endswith(".cn.md")
    }

def command_doc_stems() -> set[str]:
    return english_doc_names() - REFERENCE_DOCS
```

`command_doc_stems()` is the single source for `test_command_doc_covers_every_cli_option`
(Section 7.6) and both closure checks (Section 8.4). It is filtered, never
hardcoded: a new command document joins it automatically, and the two reference
documents are excluded because no single command owns them (Section 4.4).

7.2 **Positive assertions move to the relevant side.** An assertion that read
`README.md` becomes an assertion over `en_docs()`; one that read `README.cn.md`
becomes an assertion over `cn_docs()`. Assertions are retargeted, not rewritten:
the strings they check stay identical unless the string itself moved between
documents, in which case only the side changes.

7.3 **Negative assertions run over `all_docs()`.** Statements of the form "the
removed option `--on-empty-pseudo` must not be documented" are the ones most
likely to be reintroduced in a new file, so they must search the whole corpus,
not one side.

7.4 **Bilingual pairing and structure are enforced.** Two tests guard Section
3.1 and Section 3.5:

- `test_every_command_doc_has_a_chinese_twin` — `docs/commands/*.md` and
  `docs/commands/*.cn.md` correspond one-to-one. Without it, Section 3.1 decays
  into an intention.
- `test_command_doc_pairs_are_well_formed` — for every pair: the title is
  followed by the language-switch line, and that line points at both files of
  the pair; every required section for its kind (Section 4 for command
  documents, each reference document's Section 4.4 skeleton for the other two) is
  present in both files; and the normalized section sequence is identical across
  the pair, using the Section 3.5 translation table.

Pairing alone is too weak: an empty Chinese file, or a twin missing half its
sections, passes a file-existence check.

7.5 **Contract assertions live in the corpus test, not the help tests.** When a
full explanation moves out of `--help` (Section 5.4), its short safety warning
stays asserted in the help test while the complete contract is asserted against
the document that now owns it. The help tests keep asserting flag existence and
the retained short phrase.

7.6 **Option coverage is enforced structurally, not by spot-check.**
`test_command_doc_covers_every_cli_option` is parameterized over
`command_doc_stems()` × {`en`, `cn`} (Section 7.1) — eight names, sixteen files
today — so a Chinese table cannot drift while its English twin stays correct, and
a new command document joins the sweep automatically. For each file it maps the
document to its command, reads the `###` headings of the `## Parameters` list, and
compares that set against the command's real options.

Reading the headings matters: a flag merely mentioned inside a description would
otherwise be mistaken for a documented option. Real options come from
`mod.app.registered_callback.callback`, with `ctx` excluded, and each remaining
parameter's declared option strings read from its `OptionInfo.param_decls`,
falling back to the kebab-cased parameter name when that tuple is empty. Typer
infers `--foo` from a bare `foo: str = typer.Option(...)`, so a future command that
declares no explicit name would otherwise contribute an empty option set; this
project declares every option explicitly today, but the rule is a standing
convention (Section 6.1). Deriving kebab-case from the name alone is not enough,
because an option may declare an alias: `cluster` declares `--label-csv` together
with `--labels`, and `label_csv -> --label-csv` alone would make the documented
`--labels` look like a flag the CLI does not have.
`--label-csv` / `--labels` is the only such alias in the CLI today, but the rule
must handle aliases rather than the single case. Typer's `--help`,
`--install-completion`, and `--show-completion` are injected by Click and never
appear in the callback signature, so they need no filtering.

The comparison is bidirectional: an option missing from the table fails, and a
flag in the table that the CLI does not have fails too.
The two reference documents are excluded because no single command owns them
(Section 4.4); Section 7.7 covers them instead. These tests do not compare
defaults or allowed values; Section 10 states how those are verified. The
introspection follows the existing `_finetune_param_names` precedent in
`tests/test_cli_smoke.py`, and every command is a single-callback Typer app, so
one helper covers all of them.

Two assertions make that architecture explicit rather than assumed, so a future
change fails with a reason instead of silently deriving a wrong option set:
`assert not app.registered_commands`, because a top-level command must be
registered as a sub-Typer app; and `assert isinstance(parameter.default,
OptionInfo)` for every callback parameter, because `ArgumentInfo.param_decls` is
`None` and the kebab-case fallback would otherwise invent a bogus `--path` for a
positional argument. If a positional argument is ever genuinely needed, extend
the skeleton and these rules deliberately rather than loosening the test.

7.7 **Reference documents get literal pair-level checks instead.** A reference
document maps to no single command and carries no `Parameters` section
(Section 4.4), so 7.6 cannot apply. There is no generic "identifier" extraction:
an English/Chinese pair cannot be compared word by word, and backticked spans
include headings and prose. `test_reference_docs_pairs_agree` therefore asserts
explicit literals per document rather than a fuzzy token set.

Both reference documents assert the language-switch line and the normalized
section sequence across the pair (Section 7.4).

`training-augmentation`, where flags are the natural anchor:

- extract `--[a-z0-9-]+` from both files; the English and Chinese flag sets must
  be equal;
- both files must contain the required subset `--augmentation` and
  `--orientation-policy`.

`embedding-metrics`, where the contract is metric names rather than flags. These
identifiers stay in English inside the Chinese document, so one literal set works
on both sides:

```python
required = {
    "pretrain", "finetune", "extract",
    "StratifiedKFold", "kNN_Acc_k1", "Linear_Probing_Acc",
    "Linear_Probing_Balanced_Acc", "mAP", "Recall@k", "Silhouette_Score",
}
```

Two contract points are prose rather than identifiers, so they are asserted per
side with fixed fragments taken from the text this migration moves — not left to
whoever writes the translation:

```python
assert "empty CSV fields" in en_text
assert "not zeros" in en_text
assert "non-query samples" in en_text
assert "singleton" in en_text
assert "denominator" in en_text

assert "空 CSV 字段" in cn_text
assert "而非 0" in cn_text
assert "非查询样本" in cn_text
assert "单例标签" in cn_text
assert "作为分母" in cn_text
```

Every fragment above already occurs in the corresponding README section, so these
assertions constrain the migration instead of inventing new wording.

Because a reference document must not own a parameter table (Section 4.4), the
pair test also asserts the negative: `assert "## Parameters" not in en_text` and
`assert "## 参数" not in cn_text`. Nothing else would notice a reference document
growing a second copy of the flags it shares, which is the one rule the two
reference documents exist to enforce.

Equality is never sufficient on its own — two files that both omit
`--augmentation` would pass it. The assertions are `required <= present` and
`en == cn`, never `present == required`, because a `Usage` example legitimately
names other flags. Without the literal sets a reference document could keep its
headings and command names, drop the whole contract, and still pass.

7.8 **Cross-document links are asserted, not merely resolved.** Section 4.4 makes
reference documents reachable from the commands they serve, and the reverse.
That obligation is currently unverified, so a missing link, a wrong-language
target, or plain-text command names would all pass. Two small checks close it:

- `test_local_document_links_resolve` — for every non-HTTP Markdown link in both
  READMEs and in every `docs/commands/*.md`: split off any `#fragment`; resolve
  the path part against the linking file's own directory, treating an empty path
  as the linking file itself; assert the target file exists; and, when a fragment
  is present, assert the target file contains an explicit `<a id="fragment">`.
  This is what catches a design-spec link left root-relative after its paragraph
  moves into `docs/commands/` (Section 8.2). Fragments are matched against
  explicit anchors rather than a reimplementation of GitHub's heading slugs: slug
  rules are language- and Unicode-dependent, so a slugifier would give false
  results on the Chinese files. Only two fragments exist in the whole set — the
  `doctor` and `update` README anchors of Section 6.3 — so the strict rule costs
  nothing, and it is what makes those anchors load-bearing rather than
  decorative.
- `test_reference_links_are_correct` — assert the required edges, same language
  only: `pretrain` and `finetune` link `training-augmentation` and
  `embedding-metrics`; `extract` links `embedding-metrics`; the Chinese files
  link the `.cn.md` targets, not the English ones; and each reference document's
  `Affected Commands` section links back to the affected command documents in its
  own language. A reference document links the command document itself and not a
  `#parameters` fragment: a command document is mostly its `Parameters` section, so
  an extra anchor there would add boilerplate without improving the landing spot.

Without the second check the first is satisfied by any working link, so a
command document could quietly stop linking the contract that governs it.

## 8. Landing plan for v0.10.1

### 8.1 README outline

Both READMEs are cut to this outline, in this order:

1. Title, language switch.
2. One-paragraph positioning.
3. `## Why OTU-Former` — four to five bullets, including the input/output
   boundary: the toolkit consumes standardised specimen images and produces
   morphological barcodes, morphOTUs, and diversity data; it does not assemble
   reads, call targets, or infer ortholog groups upstream.
4. `## Workflow Overview` — the numbered command order with command names. This
   is the early signpost, since the full table sits lower.
5. `## Requirements`.
6. `## Installation` — including the isolated-environment options.
7. `## Quick Start` — the shortest working invocation.
8. `## Doctor Command`, `## Update Command`, `## Output Safety`,
   `## Hugging Face Weights` — inlined per Section 6.2. The Update section lists
   its options (`--check`; `--yes`, `-y`); it currently omits the `-y` alias. The
   first two carry the `<a id="doctor-command">` / `<a id="update-command">`
   anchors of Section 6.3, which both the link test and the Update option check
   rely on.
9. `## Common Behaviours` — Logging, Graceful Shutdown, Device Selection, Shell
   Completion, Version.
10. `## Project Structure` — a pointer to the real directory, not to the
    design's hand-written tree:

    ```markdown
    See [`src/otuformer/`](src/otuformer/) for the current package layout and the
    [master design](docs/superpowers/specs/2026-03-29-otuformer-design.md) for
    architectural responsibilities.
    ```

    The design's Section 2.1 tree is stale today (Section 8.5), so a README
    pointing at it would replace 43 lines of duplication with a wrong answer.
11. `## Commands` — the table, with one row per command, immediately before
    `## License`.
12. `## License`, `## Contact`, `## Citation`.

Everything else moves out. Target length: at most 200 lines per README.

### 8.2 Document inventory

Ten documents, twenty files: eight command documents and two reference
documents, each with a `.cn.md` twin.

| Document | Absorbs |
|---|---|
| `docs/commands/pretrain.md` | README `### Pretrain Command`; pretrain app-help patch-loss summary |
| `docs/commands/finetune.md` | README `### Finetune Command` |
| `docs/commands/extract.md` | README `### Extract Command`; the enumerated-parameter validation note for `extract` |
| `docs/commands/cluster.md` | README `### Cluster Command` |
| `docs/commands/annotate.md` | README `### Annotate Command` |
| `docs/commands/diversity.md` | README `### Diversity Command`; the app-help metric glossary |
| `docs/commands/cam.md` | README `### CAM Command`; the enumerated-parameter validation note for `cam` |
| `docs/commands/export.md` | README `### Export Command` |
| `docs/commands/training-augmentation.md` | README `### Training Augmentation Profiles`; the augmentation contract from both app helps |
| `docs/commands/embedding-metrics.md` | README `### Embedding Metrics (v0.7.1)` |

`### Embedding Metrics (v0.7.1)` is a shared contract, not an `extract` feature.
It states that `pretrain`, `finetune`, and `extract` compute the same metrics
under the same field names, and it defines the CV fold rule, the singleton
conventions for `mAP` and `Recall@k`, the unavailable-field convention, the
cosine `Silhouette_Score`, and the v0.7.1 comparability break. Filing it under
`extract.md` would make `extract` look like the owner of training-time metrics and
would leave `pretrain` users without the shared rules, so it becomes the second
reference document (Section 4.4). `pretrain.md`, `finetune.md`, and `extract.md`
link it from their `Outputs` and `Notes` sections. The one sentence in that
section unrelated to metrics — that `cam` and `extract` validate enumerated query
parameters natively — moves into `cam.md` and `extract.md` `Notes`.

`doctor` and `update` do not get documents (Section 6.2). `## Common Behaviours`
and `## Output Safety` / `## Hugging Face Weights` stay in the READMEs.

A moved paragraph can carry a link that no longer resolves. The augmentation
section's last sentence links the augmentation design spec as
`docs/superpowers/specs/2026-09-07-otuformer-training-augmentation-design.md`,
which is correct from the repository root but resolves to
`docs/commands/docs/superpowers/...` once the paragraph lives in
`docs/commands/`. Repoint it to
`../superpowers/specs/2026-09-07-otuformer-training-augmentation-design.md` in
both language files, and change only the target, not the sentence
(Section 7.8).

### 8.3 Help edits

Exactly three app-level blocks move out, and six cross-references are repointed.
No CLI behaviour, option, or default changes. Help-text edits are limited to the
three block removals, the URL additions and repointing, the `update` example, and
the four approved per-option phrases listed below.

Blocks removed:

| Site | Change |
|---|---|
| `cli/pretrain.py` app help | Remove the `Augmentation contract:` paragraphs; keep purpose, examples, and the `pretrain` and `training-augmentation` pointers. |
| `cli/finetune.py` app help | Remove the `Augmentation contract:` and `Pseudo-label mode:` paragraphs; keep purpose, both examples, and the same two pointers. |
| `cli/diversity.py` app help | Remove the `Metrics:` glossary paragraph; keep input modes, outputs, the example, and the `diversity` pointer. |

URL pointers added (Section 5.5):

| Site | Target |
|---|---|
| `cli/main.py` root app help | the README |
| `cli/doctor.py`, `cli/update.py` app help | the README anchor for that command — `README.md#doctor-command` / `README.md#update-command` (Section 6.3) — not the bare README |
| The eight analysis commands (`pretrain`, `finetune`, `extract`, `cluster`, `annotate`, `diversity`, `cam`, `export`) | their own command document |
| `pretrain`, `finetune` (in addition) | `training-augmentation` |

Every command gets a pointer, not only the three whose blocks are removed.
Today only `pretrain`'s help names the README, and that mention sits inside the
block being deleted; without this step `extract`, `cluster`, `annotate`, `cam`,
and `export` would have no route from `--help` to their documentation at all.

Reference documents are linked from the command documents that share the
contract, not from `--help`: the metrics contract was never in the help, so no
command gains a second URL for it.

`cli/update.py`'s app help is a single purpose line, so it also gains the minimal
example the Section 5.1 rule requires:

```text
Check for published updates and optionally install the latest version.

Quick example:

  otuformer update --check
```

`cli/doctor.py` already carries a `Quick example:` line and needs no change. This
brings the operational commands into line with Section 5.1; it is not a CLI
behaviour change.

Cross-references repointed:

| Site | Change |
|---|---|
| `cli/pretrain.py:31` | `See the pretrain section of the README` → the `pretrain` documentation URL. |
| `cli/pretrain.py:169`, `:179` | `See 'Augmentation contract' above.` → the `training-augmentation` documentation URL. |
| `cli/pretrain.py:228` | `documented in the README.` → the `pretrain` documentation URL. |
| `cli/finetune.py:486`, `:497` | `See 'Augmentation contract' above.` → the `training-augmentation` documentation URL. |

Add the documentation base URL as one module-level constant in
`src/otuformer/cli/` and use it at all pointer sites. No help text changes beyond
those explicitly listed in this section (Section 5.3).

Per Section 5.4, exactly four short phrases are retained in per-option help. This
is an approved edit to per-option help, not an exception discovered during
implementation:

| Option help | Retained phrase |
|---|---|
| `pretrain --augmentation` | `color-robust` may suppress diagnostic colour, pattern, or metallic sheen |
| `pretrain --augmentation` | `legacy` reproduces 0.2.1 |
| `finetune --augmentation` | `conservative` is experimental |
| `finetune --pseudo-label-from` | `Omitted experiment options inherit finetune#1 resolved values.` |

Without these, three existing help tests would fail and the CLI would silently
lose its only warning about `color-robust` and `legacy`. Neither `finetune
--augmentation` nor `finetune --pseudo-label-from` currently carries its phrase,
so both gain one; `pretrain --augmentation` currently refers to the removed block
and is reworded to carry the two phrases instead.

### 8.4 Test retargeting

Eight tests assert README body text and are retargeted per Section 7:

| Test | File | Retarget |
|---|---|---|
| `test_version_documentation_mentions_patch_loss_migration` | `tests/test_version.py` | both sides |
| `test_v080_readmes_state_subcenters_range` | `tests/test_version.py` | `en_docs()` / `cn_docs()` |
| `test_v080_readmes_document_the_metric_loss_modes` | `tests/test_version.py` | both sides |
| `test_v100_readmes_document_optional_registers` | `tests/test_version.py` | both sides |
| `test_v080_readmes_state_raw_cls_open_set_and_training_only_centers` | `tests/test_version.py` | both sides |
| `test_readmes_document_update_and_continued_training` | `tests/test_cli_smoke.py` | both sides; also assert the Update section documents every option string of `update` exactly. Locate the `<a id="update-command"></a>` anchor, consume its immediately
following level-2 heading, and take the body through the next level-2 heading or
end of file — matching the heading *level* but never its localized text, so one
rule serves both languages. Starting the slice at the anchor without consuming
that heading would yield an empty set, because the heading is itself the first
`## ` after the anchor. Then tokenize complete flags with a boundary-safe pattern
such as `(?<![\w-])--?[a-z][a-z0-9-]*` and compare the extracted set to the
`param_decls` of `update`'s callback as a set equality. A substring test cannot do this: `"-y" in "--yes"` is `True`, and a whole-README search is satisfied by the `conda create … -y` line in the Installation section even when `-y` is undocumented (Section 2) |
| `test_readmes_document_v07_patch_loss_contract` | `tests/test_cli_smoke.py` | both sides |
| `test_readmes_document_sparse_label_workflow` | `tests/test_cli_smoke.py` | `all_docs()` for the negative assertions |

Retaining those short phrases (Section 5.4) keeps two help tests passing without
modification: `test_pretrain_help_warns_about_color_robust_and_legacy`
(`diagnostic`, `metallic`, `0.2.1`, `new run`) and
`test_finetune_help_marks_conservative_experimental` (`experimental`,
`new run`). The corpus test additionally asserts the full explanation, so the
contract is covered on both sides rather than only in the document.

Three help tests change:

| Test | Change |
|---|---|
| `test_pretrain_help_defers_the_full_contract_to_the_readme` | Rename; assert the documentation URL instead of the literal `README`. |
| `test_training_help_documents_augmentation_contract` | It asserts `dorsal` / `ventral` / `lateral`, `in-plane`, and `does not guarantee` on `pretrain` / `finetune` help output, and those strings live only in the removed block. Keep the `--augmentation` / `--orientation-policy`, `invariant` / `sensitive`, profile-name, and `default for a new run: ...` assertions in the help test; move the three removed-block assertions to the `training-augmentation` corpus test. |
| `test_finetune_help_documents_pseudo_and_long_tail_options` | Its `assert "omitted experiment options inherit finetune#1" in normalized` depends on that exact substring, case-sensitively, in the removed `Pseudo-label mode:` block. Retaining the phrase as `Omitted experiment options inherit finetune#1 resolved values.` in `--pseudo-label-from` help breaks it twice: the added `the` and the capital `O`. Normalize case and assert the full phrase instead: `normalized = " ".join(result.output.replace("│", " ").split()).lower()` and `assert "omitted experiment options inherit finetune#1 resolved values" in normalized`. Do not tune the help wording to preserve the old substring. |

Adding a URL to `--help` only helps future commands if a test notices its
absence. A static parameter list cannot notice: a new `otuformer foo` would simply
not be in the list. The guard is registry-driven, and closes three gaps at once:

| Test | Change |
|---|---|
| `test_help` | Also assert the root help carries the README URL. |
| `test_every_registered_command_help_links_documentation` (new, `tests/test_docs.py`) | Parameterized over `sorted(group.name for group in app.registered_groups)`, so a newly registered command is swept automatically. Asserts the command's own documentation URL — its command document, or the README anchor for `doctor` / `update` — and, for `pretrain` / `finetune`, the `training-augmentation` URL. |
| `test_command_set_is_closed` (new, `tests/test_docs.py`) | Parse each README's command table — bounded by that file's own Section 3.5 headings, `## Commands` … `## License` in `README.md` and `## 命令` … `## 许可证` in `README.cn.md`, never the `Workflow Overview` list, which also names commands — into a list of `(command, target)` rows. Before building the mapping, assert `len(rows) == len({command for command, _ in rows})`: `dict(rows)` silently drops a repeated command, so a duplicated `pretrain` row would otherwise pass and "one row per command" would go unverified. Then assert the mapping equals the expected one: every analysis command maps to `docs/commands/<name>.md` in `README.md` and to `docs/commands/<name>.cn.md` in `README.cn.md`, while `doctor` and `update` map to `#doctor-command` / `#update-command` in both. Also assert the key set equals `command_doc_stems() \| {"doctor", "update"}` and equals `{group.name for group in app.registered_groups}`. Comparing only the key set plus "the link resolves elsewhere" would pass three wrong tables: `pretrain` → `pretrain.cn.md` in the English README, `finetune` → `pretrain.md`, and `doctor` → `#update-command`. The stems must come from `command_doc_stems()` (Section 7.1): a naive `Path.stem` over `docs/commands/*.md` yields `pretrain.cn`, `training-augmentation`, and `embedding-metrics`, so the assertion would fail on a correct implementation. This protects the other two obligations of a new command — a command document and a correct README row — which nothing else would catch, because `test_command_doc_covers_every_cli_option` is driven by the documents that already exist. |

URL assertions compare whitespace-stripped output. Measured directly against the
Typer/Rich renderer at width 80, the 84-character `training-augmentation` URL
wraps mid-word (`...training-augmentat` / `ion.md`), so a raw
`expected_url in result.output` fails even when the URL is correct, while
`expected_url in "".join(result.output.split())` passes. Every URL assertion —
the new sweep, `test_help`, and the renamed
`test_pretrain_help_defers_the_full_contract_to_the_readme` — uses the compacted
form.

`tests/test_update.py` changes too (Section 10.8):

| Test | Change |
|---|---|
| `test_update_check_never_runs_pip` | Add `assert f"Current version : {__version__}" in result.output`, importing `__version__` from `otuformer`. Asserting the literal `0.10.1` would add a second place to hand-edit on every bump; `tests/test_version.py` already pins `__version__`, so this test only needs to prove the value is rendered. |

New tests added, all in `tests/test_docs.py`:

- `test_every_command_doc_has_a_chinese_twin` (Section 7.4)
- `test_command_doc_pairs_are_well_formed` (Section 7.4)
- `test_command_doc_covers_every_cli_option` (Section 7.6)
- `test_reference_docs_pairs_agree` (Section 7.7)
- `test_local_document_links_resolve` (Section 7.8)
- `test_reference_links_are_correct` (Section 7.8)
- `test_every_registered_command_help_links_documentation` (this section)
- `test_command_set_is_closed` (this section)

The corpus-based replacements of the assertions moved out of the help tests
(Section 7.5) are retargeted existing assertions, not new coverage, and are not
counted as new tests.

### 8.5 Master design edits

In [2026-03-29-otuformer-design.md](2026-03-29-otuformer-design.md):

- Drop `(v0.1.0)` from the title. Replace the `**Version:** 0.1.0` line with a
  pointer to `pyproject.toml` as the single source of the package version. A
  living master design cannot carry a version number that individual feature
  designs move past.
- Rewrite each Section 3 sub-section to the template: purpose (one to three
  sentences), `**Outputs:**`, `**Key constraints:**`, and
  `**Design history:**` links to the relevant per-version design specs. The
  exhaustive flag bullet lists are removed from the design and are preserved by
  their new home, the `Parameters` sections in `docs/commands/`.

  This is a deletion and relabelling pass only. No sentence is reworded, no
  contract is reinterpreted, no flag's meaning is restated. Where a flag list is
  deleted, the sentence that carried its *design intent* — an inheritance rule, a
  resolved default such as `--mask-ratio auto` resolving to `0.30` — moves into
  `**Key constraints:**` verbatim. A docs patch must not quietly change what a
  historical design decided.

  `doctor` is the one approved exception to the template. It writes no output
  directory (see the output-safety design); its only output is
  the environment diagnostic printed to the console, which its `**Checks:**`
  line already enumerates, and it owns no design history. Reorganizing the same
  fact into `**Outputs:**`, `**Key constraints:**`, and `**Design history:**`
  fields would add nothing, so it keeps its original two lines.
- Replace the duplicated `### Training augmentation contract` block with a link
  to
  [2026-09-07-otuformer-training-augmentation-design.md](2026-09-07-otuformer-training-augmentation-design.md).
  The user-facing profile contract lives in
  `docs/commands/training-augmentation.md`, which the README links to; the master
  design links the design spec, not the command document (Section 2).
- Replace Section 2.1's per-file tree with a directory-level tree, because the
  per-file one had drifted in at least seven ways: it listed
  a `cli/` module that does not exist; omitted
  `cli/update.py`, `vision/export.py`, `constants.py`, `utils/device.py`,
  `utils/paths.py`, and `utils/size.py`, which do; and still labelled
  `embedding/pseudo_label.py` as "planned" although it is implemented. A
  file-by-file list in a design document will drift again, so reduce it to:

  ```text
  src/otuformer/
  ├── cli/          command parsing and orchestration
  ├── training/     pretraining and fine-tuning
  ├── embedding/    extraction, evaluation, pseudo-label logic
  ├── delineation/  distances, trees, partitions, diversity
  ├── vision/       CAM generation and ONNX export
  └── utils/        shared I/O, device, path, size, logging, checkpoint helpers
  ```

  This matches the README rule that no document maintains a per-file tree
  (Section 8.1).
- Add a `## 7. Documentation Conventions` section holding only the load-bearing
  rules — three layers with fixed link direction; `docs/commands/` owns the flag
  surface and is bilingual with `.cn.md`; design specs may name the flags a
  decision or invariant requires but do not duplicate exhaustive current
  parameter tables; documentation assertions read the corpus — plus a link to
  this document. The master design does not link to `docs/commands/`
  (Section 2).
- Delete the `evaluate` sub-section from Section 3 and renumber the following
  sub-sections. There is no `evaluate` command and no design for one: the
  embedding-quality evaluation is delivered through `extract`, and its metric
  contract is recorded by the v0.7.1 metric design plus the command reference
  (Section 7). No separate decision entry is kept, so the command cannot
  resurface as a phantom sub-section.

### 8.6 Version bump

Documentation-only patch, semantic versioning: **0.10.1**. No `CHANGELOG.md` is
created (Section 9).

| File | Change |
|---|---|
| `pyproject.toml` | `version = "0.10.1"` |
| `src/otuformer/__init__.py` | `__version__ = "0.10.1"` |
| `tests/test_version.py` | `assert __version__ == "0.10.1"` |
| `README.md`, `README.cn.md` | the `### Version` line |

No tag is created by this change; the operator tags `v0.10.1` after committing.

## 9. Deliberately not done

- **No `CHANGELOG.md`.** Version history lives in git tags and in the per-release
  design specs under `docs/superpowers/specs/`; a third copy would drift. This is
  a standing decision, not an omission.
- **No `docs/commands/` index page.** The README command table is the index.
- **No `.zh.md` migration.** Section 3.2.
- **No per-option help compression.** Section 5.3.
- **No CI.** The repository has no CI configuration; the test suite remains the
  gate, run manually before the operator commits.
- **No `README.cn.md` rename, no `docs/superpowers` restructuring.** Out of scope
  and unrelated to the problem.

## 10. Verification and acceptance

Implementation is complete when:

1. The full test suite passes, including the retargeted tests and the eight new
   tests in `tests/test_docs.py`.
2. `test_command_doc_covers_every_cli_option` passes for all sixteen command
   documents — eight English, eight Chinese — and
   `test_reference_docs_pairs_agree` passes for both reference pairs:
   every CLI option appears in the corresponding `Parameters` list, and no
   documented option is absent from the CLI. Defaults and allowed values are verified by hand
   against `--help` for each of the eight commands before the change is declared
   complete; that part is deliberately not automated (Section 7.6).
3. No `docs/commands/*.md` lacks its `.cn.md` twin, and
   `test_command_doc_pairs_are_well_formed` passes for every pair.
4. Per README, the command table — bounded by that file's own Section 3.5
   headings — parses to exactly one row per registered command and to the exact
   expected `command → target` mapping: `docs/commands/<name>.md` in `README.md`,
   `docs/commands/<name>.cn.md` in `README.cn.md`, and the two README anchors in
   both. The language switches in all twenty documents resolve in both
   directions. The two anchors are verified by
   rendering both READMEs, not by assuming GitHub's slug for a Chinese heading.
   Every non-HTTP local link in the two READMEs and in all twenty command
   documents resolves from its own directory, every fragment corresponds to an
   explicit `<a id>` in its target, and the command↔reference edges of Section 7.8
   are present in the correct language. The README's Update section documents
   `--check` and `--yes`/`-y`, checked by exact flag-set comparison rather than
   substring search.
5. `--help` renders and shows its documentation URL for the root app and for
   every command: `otuformer --help`, `otuformer doctor --help`,
   `otuformer update --help`, and `otuformer <command> --help` for all eight
   analysis commands. This is a parameterized sweep driven by
   `app.registered_groups`, not a sample of three and not a static list, so a new
   command fails the suite if it omits its URL. URL comparisons use
   whitespace-stripped output (Section 8.4). The three removed blocks are gone
   from `pretrain`, `finetune`, and `diversity`; the four retained phrases are
   present in per-option help; and `update` shows its example line.
6. Both READMEs are at most 200 lines and the `## Commands` table sits
   immediately before `## License`.
7. Every flag in each command document's `Parameters` list states its default, plus
   an allowed-value line wherever the CLI enumerates the values. Every version-bearing behavioural statement
   extracted from the READMEs before implementation has a destination in the
   corpus — including `0.2.1`, `v0.6.x`, `v0.7.0`, `v0.7.1`, `v0.7.x`, `v0.8.0`,
   `v0.9.0`, and `v0.10.0`. A note appears only in the documents whose behaviour
   it describes; `cluster`, `annotate`, `diversity`, `cam`, and `export` have no
   recoverable command-specific history and therefore no `Version Notes` section
   until they change.
8. `otuformer --version` reports `0.10.1`. The `update` unit test, with the
   remote version mocked, asserts it prints `Current version : 0.10.1`. The real
   `otuformer update --check` is not a pre-commit gate: it queries GitHub for the
   latest tag, which does not exist until after this change is committed and
   tagged. Run it by hand after the release.
9. The master design's Section 2.1 shows the directory-level tree and lists no
   individual modules, so it cannot drift file by file. Every directory it names
   exists under `src/otuformer/`.
10. The registered command set, the `docs/commands/` stems plus `doctor` and
    `update`, and the README command table all agree, so a new command cannot be
    added with a URL, a document, or a README row missing.

## 11. Decision Record

- The documentation has three layers with a fixed link direction; the master
  design links to design specs only, and the README links to command documents.
- README is a front page plus setup and operational content; analysis-command
  reference prose lives in `docs/commands/`.
- `doctor` and `update` stay inline in the README next to installation; this is
  the only README exception.
- Documentation is fully bilingual with the `.cn.md` suffix, enforced by a
  pairing test and by a structure test over the pair; `README.cn.md` is not
  renamed. Each README links its own language's documents, and `doctor` /
  `update` use explicit HTML anchors so the anchors work in a Chinese heading.
  README section names come from the Section 3.5 mapping, so a test selects its
  parse boundary by the file's language instead of guessing at a translated
  heading.
- A shared flag is defined once per affected command surface, in that command
  document's `Parameters` section. The two reference documents (augmentation and
  embedding metrics) explain shared behaviour and carry no parameter table.
- The command table sits immediately before `## License`, not at the end of the
  file.
- Headings carry no version numbers. `Version Notes` is the sole home for
  command- or reference-specific user-visible change history inside the
  documentation;
  it is forward-looking from v0.10.1, a document a release did not change gets
  no entry for that release, and a document with no recoverable history omits the
  section rather than inventing entries.
- `--help`: existing per-option help is not shortened. v0.10.1 only repoints URLs
  and retains the four approved short phrases from the removed app-level blocks;
  shortening any other per-option help requires separate operator approval.
- Help pointers use a GitHub URL rather than a repository-relative path, because
  the wheel does not ship `docs/`.
- Safety-relevant warnings stay visible in the CLI at the point of use: the CLI
  keeps a short phrase, the document carries the full explanation. Four phrases
  are retained, and that list is approved rather than discovered.
- URL assertions compare whitespace-stripped help output, because Rich wraps a
  long URL mid-word at the default width.
- Help-pointer, command-document, and README-row coverage is registry-driven, so
  a newly registered command fails the suite until it has all three.
- Documentation assertions read a corpus, so file layout changes cannot break
  content tests; negative assertions search the whole corpus, English-side
  helpers exclude `.cn.md`, option coverage is enforced bidirectionally over both
  sides of every `Parameters` list, and the root and per-command help-pointer
  rule is guarded by the extended help tests.
- The README's `Project Structure` links the real package directory, and the
  master design's tree is reduced to directories, because a hand-written
  file-by-file tree had already drifted.
- The `evaluate` sub-section is deleted rather than recorded: there is no
  `evaluate` command and no design for one, so the master design does not
  mention it at all.
- The README owns the inline `doctor` and `update` surfaces, and lists every
  option the CLI defines for them, including the `-y` alias of `--yes`.
- Local Markdown links must resolve from the linking file's own directory, and
  the command↔reference link edges are asserted per language: a paragraph moved
  into `docs/commands/` has to repoint its relative links. Fragments are matched
  against explicit `<a id>` anchors, never a heading-slug reimplementation, since
  slugs are language-dependent.
- A reference document must not grow a `Parameters` section; the pair test asserts
  its absence, so the single-definition rule of Section 4.1 stays enforced.
- Top-level commands are single-callback sub-Typer apps exposing named options;
  the documentation tests assert that shape instead of assuming it, so a future
  positional argument fails with an explicit message.
- Each README's command table is asserted as a complete `command → target`
  mapping, so a wrong-but-existing target (an English row pointing at a `.cn.md`
  document, or `doctor` pointing at `#update-command`) fails.
- No `CHANGELOG.md`; no `docs/commands/` index page; no CI.
- v0.10.1 is a documentation-only patch; the operator commits and tags.

## 12. References

- [OTU-Former Toolbox Design](2026-03-29-otuformer-design.md)
- [OTU-Former Optional Register Tokens — Design](2026-09-28-otuformer-optional-registers-design.md)
- [OTU-Former Sparse-Label Fine-Tuning Design](2026-09-26-otuformer-sparse-label-pseudolabel-design.md)
- [OTU-Former v0.8.0 Metric-Loss Design](2026-09-24-otuformer-metric-loss-v080-design.md)
- [OTU-Former Training Augmentation Design](2026-09-07-otuformer-training-augmentation-design.md)
- Sibling project structure: `phyloAI` `README.md` + `docs/commands/*.md`
