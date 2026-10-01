# OTU-Former Conversational Workflow Skill - Design

Date: 2026-09-30
Status: Design direction approved in conversation; written specification awaiting
operator review. Creation of this document and recording the subsequently approved
licensing decisions are authorized; implementation is not.
Target release: v0.11.0. Package/runtime versions remain v0.10.1 until separately
approved implementation.

## 1. Purpose and Scope

Add a repository-distributed `otuformer-workflow` Skill so users of an agent CLI
can advance OTU-Former analyses through conversation. The Skill explains each
step, obtains parameters from the actual runtime CLI, presents every parameter,
waits for approval, executes one step, and interprets observed results before
asking about the next step.

Parameter fidelity is the primary acceptance criterion: no omitted parameters,
invented aliases, guessed defaults, or unsupported choices. Teaching mode uses
all 230 images in the repository examples, not a reduced subset.

In scope:

- `skills/otuformer-workflow/SKILL.md` and focused workflow references,
  distributed independently of any particular agent's discovery directory;
- runtime schema discovery for every existing user command;
- a small package-owned schema/validation module and thin Skill script;
- complete parameter cards, explicit approval, output safety, and recovery;
- guided, single-step, and opt-in teaching modes;
- focused tests, bilingual user-facing documentation, and v0.11.0 synchronization
  during approved implementation only;
- adoption of a research and commercial use license for project-owned code from
  v0.11.0, recorded here as a separate licensing workstream subject to Section
  10.1 and its independent approval gate, not implied by Skill approval.

Out of scope:

- a new MCP server, pipeline scheduler, or background job service;
- changes to training algorithms, scientific defaults, or output formats;
- a parallel handwritten parameter registry;
- automatic downloads of examples or model checkpoints;
- bundling images, weights, or the Skill in the Python wheel;
- a report command, persistent Skill progress file, or automatic reruns;
- plan creation, implementation, version edits, commits, or tags at this stage.

## 2. Existing System and Reference Mechanisms

OTU-Former uses Typer. `src/otuformer/cli/main.py` registers ten sub-Typer apps:
`doctor`, `pretrain`, `finetune`, `extract`, `cluster`, `annotate`, `diversity`,
`cam`, `export`, and `update`. Each sub-app declares
`@app.callback(invoke_without_command=True)`. Use that group's callback parameter
set as the command parameter set, excluding context injection; a traversal that
looks only at leaf commands would miss these options. Root `--version`/`-v` and
completion controls are separate operational controls, not parameters of every
analysis command. Root `no_args_is_help=True` governs the bare root invocation;
do not infer every subcommand's no-argument behavior from that root setting.

Analysis commands already provide `--out-dir` and `--overwrite`. Only `pretrain`
and `finetune` provide training `--resume`. Reuse the existing output-directory
behavior in `otuformer.utils.io.prepare_output_dir`; never invent resume flags
for other commands.

The reference projects provide two complementary patterns:

- EntomoKit introspects argparse in `entomokit/cli_schema.py`. Its thin
  `export_cli_schema.py` emits JSON; `param_guard.py` validates and renders all
  parameters from that schema. Its teaching playbook isolates demonstration
  outputs and explicitly switches back to user data.
- PhyloAI introspects Click in `phyloai/mcp/schema_gen.py` and exposes the schema
  through `get_command_schema`. Its Skill includes every parameter in the card,
  including defaults and parameters without translated annotations.

Reuse these principles through OTU-Former's own runtime, without copying argparse
machinery or adding MCP infrastructure.

The master design is architectural context, not the parameter source of truth.
For example, its diversity section describes an older tree-input contract;
the actual CLI also accepts `--embeddings` for an OTU-centroid NJ tree and retains
legacy `--tree` input. Follow the current installed CLI and matching command
documentation rather than restoring outdated design descriptions.

## 3. Architecture and Distribution

| Component | Responsibility |
|---|---|
| `src/otuformer/cli_schema.py` | Discover runtime commands, serialize metadata, validate explicit inputs, and provide shared parameter-card rendering and argv construction without invoking analysis callbacks. |
| `skills/otuformer-workflow/scripts/export_cli_schema.py` | Thin adapter for all commands or one command; expose validation, a complete card, and proposed argv when explicit inputs are supplied. This adapter never executes analysis. |
| `skills/otuformer-workflow/SKILL.md` | Canonical portable Skill: entry modes, approval rules, schema protocol, output safety, teaching triggers, and reference links. |
| `skills/otuformer-workflow/LICENSE.txt` | Full applicable project license carried with the Skill and its helper scripts when copied or packaged independently. |
| Skill references | Workflow dependencies, dialogue templates, teaching procedure, and concise error/recovery guidance; no duplicate exhaustive parameter tables. |

Use Typer's runtime command conversion and resulting parameter objects. Account
for the installed Typer/Click implementation, including vendored Click where
applicable; do not assume externally imported Click classes always match.
Keep `src/otuformer/cli_schema.py` in its proposed package-root location, and
extend `test_cli_modules_do_not_couple_to_external_click` in
`tests/test_cli_smoke.py` to cover it as well as the existing `cli/*.py` files.
The current guard does not include this new path. Its prohibition on external
`import click`, `click.Choice`, and `click.core` applies to the schema module too;
use Typer runtime objects without adding an external-Click coupling exception.
Schema discovery must remain lightweight: no analysis callbacks, model/device
initialization, training imports merely to inspect options, or weight downloads.

The normal schema invocation is:

```bash
python <skill-dir>/scripts/export_cli_schema.py --command pretrain
```

Resolve `<skill-dir>` from the loaded Skill, not an assumed working directory or
developer-specific path. The adapter imports the installed package; a source
checkout may provide `src/` when the package is absent. Report the package version
and location actually loaded. Execution uses the same environment; a mismatch
between schema source and the `otuformer` executable blocks execution until
resolved, rather than validating one version and running another.

The wheel continues to ship package modules only. The repository's
`skills/otuformer-workflow/` is the distribution source, not a claimed automatic
discovery location. Users install or register this directory through their chosen
agent CLI's supported mechanism. Do not add agent-specific discovery directories,
compatibility symlinks, installers, or configuration management to the repository.
The project maintains one canonical Skill. README examples prefer explicit
registration of the checkout directory or a supported link to it, where the agent
supports that mechanism; copying the complete Skill directory is also allowed.
Resolve links to the actual source before repository-relative asset discovery,
without assuming any fixed agent directory depth.

Both READMEs briefly explain installation with common-tool examples (Claude Code,
Codex, OpenCode, and Pi), verified against each tool's current documentation during
implementation. Include its documented destination or explicit loading mechanism
and a link to upstream instructions, not an exhaustive compatibility matrix.
Do not imply one directory or slash-command syntax works for all agents. Skill
installation does not replace installing the OTU-Former Python environment or
obtaining sample data. Do not automatically edit project/global agent settings.
Explain project trust and name-collision warnings in both READMEs. In Pi,
project-level Skill registration requires project trust, including `.pi/skills`,
project `settings.skills`, and ancestor `.agents/skills`. Same-name collisions keep
the first discovered Skill and produce a warning; inspect the actual loaded path
rather than assuming the newly registered copy won. Do not generalize Pi's trust
or collision behavior to other agents without checking their documentation.

OpenCode documents native repo/home Skill discovery. Codex's current official
skills documentation also supports standalone local/repository Skill directories,
while recommending plugins for reusable distribution; deprecation of the
`openai/skills` example repository does not establish removal of that directory
support. Its README entry may illustrate a currently documented standalone setup
and link to skill-only plugin packaging as an alternative, not assert plugin-only
installation or borrow another tool's syntax. Building/distributing a project
plugin is not required for this Skill scope. Recheck upstream instructions for the
versions actually documented at implementation time.

After upgrading OTU-Former, update the canonical Skill checkout and refresh any
copied or plugin-packaged Skill, or reload/re-register a linked Skill as required
by the agent. An already registered link need not be redundantly copied. Runtime
schema reflects the installed CLI, but this does not refresh old workflow prose
or helper scripts. README must state that distinction briefly; no updater or
persistent version-tracking subsystem is added.

Bundled script/reference paths resolve from the loaded Skill directory, even when
copied outside the repository. A copied Skill cannot assume that its ancestors
contain OTU-Former `src/` or `examples/`: prefer the installed package, or require
an explicit source/example root when needed. In the canonical checkout, the
repository-relative examples remain available. Keep path handling independent
of agent-specific directory depth.

### 3.1 Portable Frontmatter and Chat Routing

Use the Agent Skills format. Require `name`, `description`, and, for this
separately distributable Skill, `license: LICENSE.txt`. Bundle the full applicable
license in that file, not a dangling reference to the repository's parent LICENSE
or a symlink that breaks when copied. During implementation, assert its content
matches the applicable root `LICENSE`; changing either notice to the new research
license still requires the independent licensing gate. Do not mislabel a pending
MIT-stage Skill with the proposed future license.

Optional `compatibility` and string-valued `metadata` may be added when needed;
do not add fields speculatively. Match the directory name `otuformer-workflow`,
respect the documented name/description length limits, and keep one portable
Markdown body with Skill-relative references. Copies and any downstream plugin
packages retain `LICENSE.txt` with the Skill and helper scripts.

Omit `allowed-tools` entirely, including read-only preauthorization. The Skill
must not grant itself shell/analysis execution permissions or change the user's
agent permission settings. Some agents apply `allowed-tools` even without prior
workspace trust; a tool preauthorization is not the same as approval of a complete
parameter card. Existing harness permissions remain independent of, and do not
replace, the Skill's explicit per-analysis consent requirement.

Omit `disable-model-invocation` rather than adding a non-core portability field or
setting it to true. The intended behavior is model-selectable loading from chat:
loading the Skill permits guidance and read-only inspection, not automatic
analysis. This deliberately differs from disabling automatic invocation for a
one-shot deployment Skill; the confirmation protocol governs every side effect.
An agent/user configuration can still disable automatic selection, so README
examples also explain that tool's explicit invocation mechanism when available.

The `description` must identify OTU-Former, the supported command/workflow intent,
and both English and Chinese trigger phrases for analysis, training, extraction,
clustering, annotation, diversity, teaching demos, and troubleshooting. The body
remains English, with conversations following the user's language. Description
routing is semantic/model-dependent, not guaranteed keyword matching; test
representative bilingual requests and provide explicit invocation as a fallback.

Do not use Claude-only frontmatter such as `user-invocable`, `argument-hint`,
`context`, or `agent`, nor dynamic shell injection such as !`command`, positional
argument substitution, or fixed harness tool IDs. Use ordinary instructions and
explicitly requested tool calls. Passing a strict Agent Skills frontmatter check
is necessary for portability, but does not prove discovery in every agent.

## 4. Runtime Schema Contract

Discover executable commands dynamically. Unknown commands and incomplete schema
exports are explicit errors. Include root operational controls separately and
identify help/completion controls, rather than silently mixing them into analysis
parameters or dropping them without explanation.

Each command schema records command path, purpose/help, package version/location,
and ordered parameter metadata:

- internal name and exact option spellings, including aliases;
- option versus positional form, if the CLI gains positional arguments;
- required state and declared default, including unset and `None`;
- type, choices, numeric bounds, and path/file constraints when declared;
- Boolean flag semantics and positive/negative spellings when supported;
- value arity, repetition behavior, and complete per-option help;
- whether a restriction is parser-declared, shared-validator-derived, or not
  structurally available.

Do not turn an unset required default into an apparent value. Unsupported
metadata shapes fail explicitly, rather than producing misleading output.

Several current finite choices are strings validated by callbacks/core code,
not Click `Choice` types. Generic introspection cannot automatically recover
these choices or conditional requirements. Implementation must inventory such
restrictions across all commands, reuse existing constants/validators, and expose
shared lightweight metadata where needed. Any required metadata extraction must
retain existing accepted values, defaults, validation behavior, and error
boundaries. Do not parse help prose into authoritative enums or introduce
independent Skill-only copies of accepted values. Mark unavailable restrictions
as unavailable; do not call an affected proposal fully validated.

Structural schema does not replace input-data checks, checkpoint compatibility,
resume checks, or scientific preflight. Never invoke a command callback to
introspect or dry-run parameter validity.

## 5. Parameter Cards and Validation

Before an analysis step, obtain a fresh schema from the execution environment.
Generate the parameter list by iterating the whole schema, not by asking the
agent to select important parameters. Each card shows:

1. Purpose, input/output paths, and exact proposed executable command.
2. Every analysis parameter: aliases, help/meaning, options or type/range,
   required status, proposed value, and declared default.
3. Value provenance: explicit selection, omitted/default, or omitted/inherited.
4. Validation status, schema source/environment, and unresolved conditions.
5. A request for explicit approval or parameter changes.

Follow the user's language for discussion; keep executable commands, option names,
paths, and field names in English. Translations supplement, rather than weaken,
the original help. Long cards may span multiple consecutive sections, but all
parameters must be displayed before approval. A coverage/count check must detect
omissions, including unannotated parameters.

Reject unknown parameters, missing required values, illegal Boolean values,
incompatible types, and invalid declared choices/bounds. Normalize aliases and
reject contradictory duplicate assignments. Reuse runtime parser/type conversion
where possible without invoking callbacks; do not accept coercions the CLI rejects.

**Showing all defaults does not mean passing all defaults.** Training callbacks
inspect `get_parameter_source` to distinguish omission from explicit overrides.
Materializing every displayed default would change checkpoint inheritance and
pseudo-round identity checks. Preserve explicit-input provenance when constructing
argv; omitted options remain omitted. Display inherited values as inherited or
unresolved until verified, not as new-run defaults. Serialize flags according to
the actual CLI, never assume `--flag true` is valid. Execute with an argument list,
not shell interpolation, and show a properly quoted command for review. Shared
package helpers produce the card and proposed argv from the same validated inputs;
the agent executes that reviewed proposal only after approval, without independently
rewriting it. Do not introduce a second parameter definition or execution service.

Check known input combinations before approval: checkpoint/ONNX alternatives,
mutually exclusive diversity inputs, and training resume/overwrite conflicts.
Some callbacks prepare output directories before completing input validation;
launching an invalid analysis as a dry run is therefore unsafe. Make unresolved
conditions explicit, especially before overwrite. Existing CLI/core preflight
remains authoritative, and its failure must be reported as failure.

### 5.1 Schema Failure

If schema export fails, reading `otuformer <command> --help` may support diagnosis
and explanation, with this explicit limitation:

```text
Parameter source: CLI --help fallback
Complete structured validation: unavailable
```

Read-only inspection may continue. Analysis execution and installation updates
are blocked until structured validation is restored. Do not generate a truncated
parameter card from memory or treat textual help as equivalent structured checks.

## 6. Approval and Environment Rules

Run `doctor` before substantive work in a new conversation or when the environment
is unknown. Reading schemas/help/results does not require doctor first. Doctor
has no analysis parameters. Report actual missing packages and available devices:
its current implementation can report missing dependencies with exit code zero,
so zero alone is not proof the requested step can run.

Read-only inspection may proceed without approval. Analysis commands require a
complete approved card. `update` is environment maintenance, not a workflow step;
installation requires separate environment-change approval. Classify an update
proposal by its parameters, not just the command name: `update --check` is
read-only but contacts GitHub; an invocation without `--check` is installation-
capable even if it might find no new version. Include every update option in its
parameter review, and require approval before any installation-capable invocation.
Do not turn a check into installation or silently add `--yes` to bypass consent.
When `--check` is present, the actual CLI returns before installation, including
when `--yes` is also supplied; tests must verify this rather than classify `--yes`
alone as installation.

Approval applies only to the displayed command, inputs, output path, and
execution environment. Changes require a new card and approval. After any step,
summarize observed results and stop; do not automatically execute the next step.

`--overwrite` requires separate explicit approval naming the resolved directory;
general command approval is insufficient. Prefer a fresh directory or supported
training resume. Never delete failed outputs or use overwrite to bypass a
compatibility error without specific approval.

Present compute backends and the proposed device; do not silently select CPU
when an appropriate faster backend exists. Explain possible initial timm weight
downloads before training approval. Do not promise duration solely from image count.

Approval is a Skill behavioral requirement, not something a JSON schema proves.
Do not claim enforcement for agents that ignore the Skill. Prefer `otuformer` for
analyses. Custom fallback analysis scripts need explicit permission and a stated
reason; small approved data-preparation steps are separate from analysis.

## 7. Workflow, Results, and Recovery

The guided dependency path is:

```text
doctor -> pretrain -> finetune -> extract -> cluster -> annotate (optional)
       -> diversity

checkpoint -> cam / export
exported ONNX -> extract (within the actual CLI contract)
```

Arrows express dependencies, not mandatory reruns. Users with compatible artifacts
may start at the relevant stage; single-step requests validate only relevant
prerequisites. Training, annotation, CAM, and export are not mandatory for every
scientific task.

For pseudo-label feedback, preserve original SSL provenance, source eligibility,
experiment inheritance, and resume constraints. Do not create an unapproved
automatic loop or initialize round two from round one's weights against the
implemented contract.

After a step, inspect exit status, logs, and expected artifacts. Report actual
status, paths, key results, warnings, and next-step options. Directory/checkpoint
presence alone is not proof of success. Do not invent unified `result.json`,
`best.pt`, `check_status`, or `read_report` interfaces: current OTU-Former uses
command-specific artifacts and logs, not PhyloAI's unified status/result tools.

Use existing agent process support for long runs, and verify process/log evidence
before claiming launch. Progress must come from current logs/metrics. On resumption,
inspect artifacts and ask which run is intended if ambiguous. Reuse verified
completed results without relaunching. Offer only CLI-supported training resume;
other steps use existing completed results or approved fresh runs. Keep state
in the conversation and existing artifacts; no new Skill progress file.

Scientific interpretation must distinguish morphOTUs from verified species,
label classes from ecological samples, morphological trees from established
phylogenies, and current raw-feature gradient CAM from validated class-discriminative
attribution. Short teaching training is procedural demonstration, not evidence
of encoder quality. Preserve real metric names while explaining their limitations.

## 8. Output Safety

Suggest an unused dedicated root under the user's working directory, for example
`runs/run001/`, with per-step directories. Create it only after approval of its
use. An explicit dedicated user output root is also acceptable after review.

Never write analysis outputs directly into `data/`, `examples/`, source directories,
raw-image directories, or a mixed repository root. Every analysis card selects an
explicit `--out-dir`, even when the CLI provides a default. Do not invent output
parameters for doctor or update.

Resolve paths and symlinks for safety checks. Reject outputs equal to or inside
protected input/asset directories. Reject destructive ancestors that could remove
raw inputs, examples, source, or required checkpoints. Non-empty directories are
not silently reused. Failed/repeated steps use fresh sibling directories by
default; supported training resume may reuse its explicitly approved directory.
Never move or delete earlier results implicitly.

## 9. Teaching Mode

### 9.1 Assets and Entry

Teaching mode is opt-in. Offer it after environment inspection, during CSV/input
explanations, or when input problems block progress. Do not switch away from user
data without explicit selection. State that a demo does not replace user analysis.

Available repository assets:

- `examples/Epidorcus/images/`: 230 JPEG images under two label directories;
- `examples/Epidorcus/figs.csv`: 230 `image,label` rows;
- labels: 102 `Epidorcus_gracilis`, 128 `Epidorcus_tonkinensis`;
- no bundled checkpoint, embeddings, assignments, or correction table.

**Use all 230 images by default, not a reduced subset.** Architecture, device,
batch size, epochs, and metrics are reviewed in the full parameter card. Small
epoch counts may illustrate training without claiming a scientifically validated
encoder. Changes to sample assets require teaching checks to be updated.

Find the default `examples/` relative to the canonical repository Skill location
when that checkout is available, and allow an explicit examples root when the
Skill is installed elsewhere. README copy-install examples must explain how to
provide the checkout's `examples/` root for teaching; no `../../examples` guess
from the copied installation path is allowed. Distributed files must not hardcode this workstation's
absolute project path. If assets are absent or the Skill was copied elsewhere,
ask for the path or explain repository checkout requirements; do not search
unrelated directories or download data automatically.

### 9.2 Step Prerequisites

| Step | Teaching inputs and prerequisites |
|---|---|
| `doctor` | The actual current environment. |
| `pretrain` | All example images; matching CSV optionally selects training/visualization inputs. |
| `finetune` | Example labels/images and a compatible SSL checkpoint from an approved preceding step or explicitly selected existing file. |
| `extract` | All example images plus a compatible checkpoint or ONNX model; labels are optional metric/visualization inputs. |
| `cluster` | Actual generated embeddings; optionally an approved ID-matched label table derived from the example CSV. |
| `annotate` | User-selected partition and an approved teaching correction table; embeddings enable optional distance/tree refresh. |
| `diversity` | Selected raw/annotated assignments or actual OTU table; tree-related options require explicit selection and appropriate inputs. |
| `cam` | Compatible checkpoint and example image directory; explain supported image-count/selection options in its full card. |
| `export` | Compatible checkpoint; inspect the actual ONNX/report artifacts. |
| `update` | Explain or inspect `--check` when requested; do not install merely to demonstrate it. |

If a prerequisite is missing, offer an individually approved preceding step,
an explicitly selected existing artifact, or explanation without execution.
Each prerequisite analysis requires its own full card and approval. Never create
a random-weight checkpoint and present it as trained.

The CSV uses basenames while images are nested. Verify the CLI's existing
resolution and every image-to-label match. Map downstream labels against IDs
actually emitted by extraction; do not assume training names and extracted IDs
are identical. Derived CSV preparation needs approval of source, mapping, and
destination, and writes only under the demo root. Use existing `figs.csv` labels,
not an inferred alternative labeling strategy.

Annotation may derive a clearly named *teaching correction table* from the supplied
labels after reviewing ID-to-label-to-cluster mapping. Use the actual correction
fields `id` (or supported `image`) and `cluster`, not a label-only format. Reject
ambiguous or unmatched IDs. This demonstrates mechanics and is not an expert audit.
The cutoff is user-selected; two supplied classes do not authorize automatically
declaring an optimal partition.

### 9.3 Isolation and Return to User Data

Write demos under an approved root such as `runs/run001/demo/`, with separate step
directories. Keep `examples/` read-only; any derived CSV is written under the demo
root. No image subset is created.

After each demonstration, explain artifacts/limitations and ask whether to show
another step or repeat on user data. Before switching back, restate and confirm
user input, checkpoint, label, and output paths and obtain a fresh full-card approval.
Do not silently reuse demo artifacts in the user's analysis.

Class directories are not automatically ecological sample sets. Explain the
actual extraction `sample` field before interpreting per-sample diversity; species
labels must not silently become community definitions.

## 10. Documentation and Release Integration

During separately approved implementation:

- Briefly link `skills/otuformer-workflow/`, show installation/registration using
  common agent CLIs as examples, and link their official instructions. Installation
  is the user's tool-specific choice, not an additional project subsystem.
  Prefer supported registration/linking examples, keep all Skill files and the
  bundled license together when copying, explain the explicit examples-root
  requirement for copies, and briefly remind users to synchronize the Skill after
  package upgrades. Codex documentation must distinguish standalone discovery
  from optional plugin distribution.
- Include short bilingual question-and-answer examples: a user asks to run from
  images, to extract using an existing checkpoint, or to demonstrate with all
  230 examples; the agent checks prerequisites, presents the full parameter card,
  and asks for approval. A result/progress question demonstrates inspection and
  next-step discussion, not an automatically executed pipeline. The README need
  not reproduce a full training parameter card.
- Keep analysis flag prose in `docs/commands/`; Skill references own conversational
  procedures, not a second parameter reference.
- Any new user-facing reference under `docs/commands/` gets a `.cn.md` twin and
  follows pairing, structure, and `Version Notes` conventions.
- Keep the canonical Skill and developer design/plan in English; conversations
  follow the user's language. The command-document twin convention does not
  require Chinese copies of existing-style developer specs/plans.
- Link this design from the master design and distinguish this CLI Skill from
  future notebook/Python API work. Do not refactor unrelated historical designs.
- Synchronize `pyproject.toml`, `src/otuformer/__init__.py`, version tests, and both
  README current-version lines to `0.11.0`, preserving historical version records.

These are future obligations, not authorized edits in this documentation stage.
No new CLI analysis command or public analysis option is needed.

### 10.1 Research and Commercial Use Licensing

This section records a separately approved policy direction, not authorization
to replace `LICENSE`. Keep licensing preparation, implementation, and acceptance
as a distinct workstream. Approval of the Skill design or implementation plan
must not be interpreted as approval of the license change; Section 12 defines
its independent gate. Keep this policy record in the existing design rather than
creating an additional spec without operator approval.

The operator confirms that both Git author identities (`zf` and `xtmtd`) and the
reference scripts belong to the operator, who holds their copyright. This
confirmation covers project-authored material, not rights to third-party code,
weights, or datasets.

The intended licensing release is v0.11.0, conditional on independent approval
of the exact notice and the related file changes. Model the OTU-Former Research
and Commercial Use License Notice on PhyloAI's notice rather than modifying the
MIT text to add restrictions.
The intended terms are:

- Permit use, copying, modification, and distribution of project-owned code for
  non-commercial academic, educational, and research purposes, with the notice
  retained. Academic or educational status is not an exception for commercial use.
- Require prior written permission from the copyright holder for commercial use,
  including enterprise internal use, paid analysis or consulting, commercial
  redistribution, sublicensing, sale, and integration into commercial products or
  services. Permission to run a command in chat is not a commercial license grant.
- Apply the notice to project-owned source code, including the new Skill and its
  helper scripts. Carry the complete approved notice in the Skill's `LICENSE.txt`
  and reference it through `license: LICENSE.txt` so installation outside the
  repository does not lose the notice. Verify equality with the root license for
  the same release; this is required distribution metadata, not speculative
  frontmatter. Do not claim that this notice replaces third-party permissions
  or grants rights to trademarks, datasets, or sample images.
- Retain the warranty disclaimer and limitation of liability. The final license
  wording must reflect these decisions and be reviewed before release; this design
  records policy, not a legal opinion on enforceability.

Previously published MIT versions retain their original grants. The new notice
must not purport to revoke those grants or restrict commercial use of copies
obtained under MIT. Users may continue to use or fork those versions under MIT;
changing the current license cannot prevent that. Preserve historical license
files in tags/history and clearly state the transition boundary in release-facing
material.

Do not claim ownership of, or impose additional license restrictions on, ordinary
user-generated results merely because the software produced them. This includes
embeddings, OTU/abundance tables, diversity statistics, figures, and CAM images.
Users may publish, share, or use these results commercially subject to any
independent data or third-party rights. This does not exempt commercial software
use that generates those outputs from the prior-permission requirement.

Checkpoints and exported ONNX models are a separate category. Their use and
redistribution remain subject to applicable source-weight and third-party terms;
training or export does not automatically remove those terms. The operator may
publish models under a separately stated model license, but this change does not
assign a blanket new model license or claim rights over every user-trained model.
OTU-Former commercial permission alone does not authorize upstream weights or
training data.

Direct-dependency review found no requirement that the project as a whole remain
MIT. Preserve all applicable upstream copyright/license notices and obligations,
including Apache notices and the MPL-covered portions of `tqdm`. Before release,
check the actual dependency versions and distribution contents; the initial
review was not an exhaustive transitive-dependency or copied-code audit.

Only after the independent licensing gate is approved, replace the current
`LICENSE`, update both README license sections and applicable package metadata/claims, and
verify that source/wheel distributions carry the intended license. Describe the
new release as source-available research-licensed software, not OSI-approved open
source: commercial-use restrictions do not meet the Open Source Definition.
Do not add license keys, telemetry, activation, or runtime commercial-use detection.
No `LICENSE`, README, metadata, or historical release changes are authorized by
permission to record these decisions in this design.

## 11. Verification and Acceptance

Use existing pytest infrastructure and focused tests driven by the real registry:

1. Discover the current ten sub-app callback parameter sets from the live registry,
   not a hardcoded module/file list. Reuse the `param_decls`/callback-option and
   closure-check patterns in `tests/test_docs.py`, including aliases and context
   exclusion, but do not copy its hardcoded `COMMAND_MODULES` dictionary. Its
   parameter extraction is live; its module lookup list is not. Derive this new
   sweep from the runtime registry and use runtime parameters for defaults/types.
   New commands fail workflow
   coverage until guidance is added; root/help/completion controls are separate.
2. Compare schema parameters bidirectionally with the runtime CLI for every
   command: names, aliases, defaults, required state, types, flags, and available
   choices/ranges. Missing/invented parameters fail. Recheck every real signature
   and help surface in the installed distribution, including diversity inputs,
   rather than relying solely on source-checkout or historical design descriptions.
3. Each card covers every schema parameter once and displays default, proposed
   value, and provenance. Invalid unknown/required/enum/Boolean/type and conflicting
   alias inputs block validation without invoking analysis callbacks.
4. Omitted defaults stay omitted in argv; explicit overrides remain explicit.
   Verify training inheritance, Boolean serialization, and argument-safe paths.
5. Export remains free of model/device initialization and heavyweight training
   imports. Extend and run `test_cli_modules_do_not_couple_to_external_click` over
   `src/otuformer/cli_schema.py` as well as `cli/*.py`; the new module must not
   bypass the existing external-Click guard. Non-repository working directories
   work or produce clear errors; environment mismatch never silently validates
   a different installed version.
6. Check schema-failure blocking, overwrite instructions, protected/symlink paths,
   existing results, and supported-only resume behavior. Validation does not write
   or delete analysis output.
7. Validate 230 image files, 230 CSV rows, unambiguous image/label matches,
   portable sample discovery, downstream ID mapping, correction fields, and
   demo output isolation. Do not modify samples or reduce the demonstration set.
8. Review approval, recovery, interpretation, and user/demo switching rules.
   Tests validate instructions/generated data, not compliance of every agent;
   state this limitation honestly.
9. Run relevant CLI startup, documentation, safety/resume, version, and workflow
   tests, followed by the full regression suite before declaring the implemented
   release complete. Replace workstation-specific paths in existing example tests
   only where required for portable teaching verification.
10. Real training or full 230-image demonstrations require separate command
    approval; test approval does not implicitly authorize an entire analysis.
    Report which commands were demonstrated and which were only tested/reviewed.
11. Under the independent licensing gate, verify the v0.11.0 license transition:
    enterprise internal use and paid analysis
    require written permission; ordinary outputs have no added project restrictions;
    models and third-party material retain applicable terms; old MIT grants remain
    intact. Both README license statements and distributed metadata/license files
    agree, with no misleading MIT or OSI-approved claim for the new release. The
    operator reviews the exact legal wording before publication; test assertions
    do not replace that review.
12. Validate portable frontmatter: required name/description and
    `license: LICENSE.txt`, bundled full-license equality with the applicable root
    license, allowed field types/lengths, English/Chinese routing, no `allowed-tools`,
    no invocation disabling, and no harness-specific execution/injection features.
    Test helpers/assets from both a registered/linked checkout and a copied Skill:
    the latter uses an explicit examples root and retains its license file.
    In an isolated checkout/session, follow the README example and verify discovery
    plus explicit loading of the intended Skill. Pi is the required local integration
    smoke test during approved implementation while it is available: explicitly
    register the distribution directory through Pi's documented skill-path mechanism,
    such as a temporary project `settings.skills` entry (project paths are relative
    to `.pi`), not an assumption that root `skills/` is auto-discovered. Record the
    effective project-trust state, registration scope, collision warnings, and actual
    loaded Skill path. Headless/CI sessions cannot show the built-in trust prompt;
    absent an applicable trust decision, the default `ask` skips project resources.
    For untrusted/headless sessions, use an approved isolated user-level registration
    or obtain explicit approval for a one-time `--approve` trust decision. Do not
    automatically grant trust, persist trust decisions, or alter real user/global
    settings. Trust-skipped loading is an unmet test prerequisite, not a Skill
    integration failure or a passing test; rerun with an approved loading mechanism.
    Actual setup/test execution still requires implementation approval. Verify
    `/skill:otuformer-workflow` loads the intended path, not a same-name earlier copy.
    Where automatic selection is enabled, try representative bilingual requests
    and verify loading leads to parameter review without analysis launch.
    Mere file existence/Markdown parsing is insufficient for an integration claim.
    Record exact versions, platform, installation mode, and limitations. Current
    observation: Pi 0.87.1 is installed; the other three tools (Claude Code, Codex,
    and OpenCode) are not on PATH. This is not a completed integration test.
    Those unavailable tools are explicitly unverified, not passed or mandatory local
    blockers; test any that become available. Do not install agents or alter user
    configuration merely to satisfy tests without approval.
13. Test operational classification independently of schema coverage: doctor is
    read-only, `update --check` is read-only/networked and never invokes pip
    (including with `--yes`), and update without `--check` requires installation
    approval. Use mocked network/install calls, not a real update for validation.

## 12. Approval Gates

The agreed design direction authorizes this specification and its requested
revisions only. Next, the operator reviews this file. Skill plan creation requires
approval after review; Skill plan execution requires a separate explicit approval.
Data preparation, analyses, installation, version changes, commits, and tags
remain independently gated.

Licensing has a distinct gate: the operator reviews the exact proposed notice,
transition statement, and affected `LICENSE`/README/package metadata changes and
explicitly approves licensing implementation. Skill design approval, Skill plan
approval, or Skill execution approval alone cannot satisfy this gate. Track license
acceptance separately from Skill acceptance even if both changes eventually ship
in v0.11.0. If licensing approval is pending, Skill work may proceed only when
approved, but publication of the intended combined v0.11.0 release must wait;
do not silently change its licensing scope or replace `LICENSE`.

Do not create a plan, edit `LICENSE` or README/master design, change versions, or
commit this file as part of the present request. Recording licensing policy and
correcting the audited design do not authorize implementation.

## 13. References

- [OTU-Former Master Design](2026-03-29-otuformer-design.md)
- [Documentation Conventions](2026-09-29-otuformer-documentation-conventions-design.md)
- [Output Safety and Resume](2026-07-18-output-safety-and-resume-design.md)
- [Sparse-Label Pseudo-Feedback](2026-09-26-otuformer-sparse-label-pseudolabel-design.md)
- EntomoKit reference: `entomokit-workflow`, `entomokit/cli_schema.py`,
  `entomokit/param_guard.py`, and its teaching playbook.
- PhyloAI reference: `phyloai-workflow`, `phyloai/mcp/schema_gen.py`, its
  dialogue templates, and its Research and Commercial Use License Notice.
- [Agent Skills Specification](https://agentskills.io/specification)
- [Pi Skills Documentation](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/skills.md)
- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)
- [OpenCode Agent Skills](https://opencode.ai/docs/skills/)
- [Codex Skills Entry Point](https://developers.openai.com/codex/skills)
- [OpenAI Build Skills: Standalone and Plugin Distribution](https://learn.chatgpt.com/docs/build-skills)
- [OpenAI Plugin Packaging](https://developers.openai.com/plugins/build/plugins)
- [OpenAI Skills Repository Deprecation Notice](https://github.com/openai/skills)
- [Open Source Definition](https://opensource.org/osd)
- [Mozilla MPL 2.0 FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)
- [timm Code and Pretrained Weight Licensing](https://github.com/huggingface/pytorch-image-models#licenses)
