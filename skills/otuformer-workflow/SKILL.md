---
name: otuformer-workflow
description: >-
  Guide an OTU-Former (image-based morphological OTU delineation) analysis from
  chat: environment check, self-supervised pre-training, fine-tuning, embedding
  extraction, clustering, annotation, diversity, CAM and ONNX export. Use for
  English or Chinese requests such as "run OTU-Former", "train the model",
  "extract embeddings", "cluster the embeddings", "annotate the partition",
  "compute diversity", "teach me with the example images", "why did my run
  fail", 或 "用 OTU-Former 做形态学 OTU 划分", "预训练/微调模型", "提取特征",
  "聚类", "注释分区", "多样性分析", "用示例数据演示", "排查报错". The Skill
  reviews every runtime parameter, asks for approval one step at a time, and
  interprets results before suggesting the next step. Loading it permits
  guidance and read-only inspection only; it never authorizes running an
  analysis.
license: LICENSE.txt
---

# OTU-Former workflow

Help a user advance an OTU-Former analysis through conversation, one approved
step at a time. Never run an analysis because this Skill was loaded.

## Non-negotiable rules

1. **Fresh runtime schema first.** Before proposing any step, read the installed
   CLI through the adapter:
   `python <this-skill-dir>/scripts/export_cli_schema.py --command <name>`.
   Never rely on remembered flags, defaults, or message history.
2. **Show every parameter, then ask.** Present the complete card produced from
   that schema (all parameters, aliases, help, declared default, proposed value,
   provenance, restriction source, unresolved conditions) and the exact proposed
   command. Ask for approval or changes.
3. **Approval is per step.** Approval covers only the displayed command, inputs,
   output directory, and environment. Any change needs a new card and a new
   approval. `--overwrite` needs its own approval naming the resolved directory.
4. **One step, then summarize and recommend.** Run the approved command, inspect
   exit status, logs, and real artifacts, then report a short summary and one
   clear recommendation before moving on (see "After each step" below).
   Directory existence is not proof of success.
5. **Write only to an approved output root.** Never write into `data/`,
   `examples/`, source directories, or a mixed repository root. The adapter
   already protects the loaded package source and a checkout's `examples/`
   root; pass repeatable `--protected-path PATH` for additional read-only inputs
   or asset roots. Every analysis proposal selects an explicit `--out-dir`, and
   overwrite approval never overrides a protected-path rejection.
6. **Improve nothing silently.** Never change defaults, add flags "for safety",
   invent aliases, materialize displayed defaults into argv, or rerun a failed
   step automatically.
7. **Interpretation stays honest.** Distinguish morphOTUs from verified species,
   label classes from ecological samples, morphological trees from established
   phylogenies, and raw-feature gradient CAM from validated attribution.

Read-only inspection (schema, `--help`, logs, existing results) needs no
approval. Analysis, installation, and version changes always do.

## After each step

Report briefly, in this order:

1. **What ran** — the approved command and its exit status.
2. **What it produced** — the real artifact paths you inspected, not just the
   directory that exists.
3. **What it means** — key results with their documented limits, plus warnings
   and unresolved conditions.
4. **What I recommend** — exactly one of:
   - **continue** to `<next step>` and why it is ready now;
   - **redo** this step with `<specific parameter or data changes>` and what
     looked wrong;
   - **stop** and resolve `<blocker>` first (missing dependency, incompatible
     checkpoint, bad inputs).

Then keep the task moving: when the user has already authorized that next step
(or asked for a whole sequence), go straight to that step's parameter card and
approval instead of asking an open "what next?" question. Otherwise ask the
single question: continue, redo, or stop?

One step is still one approval: continuing means presenting the next card, not
inheriting the previous approval. A redo uses a fresh sibling output directory or
a supported `--resume`; `--overwrite` needs its own approval, and a failed step is
never rerun silently.

## Entry modes

- **Guided**: `doctor` first, then follow the dependency graph in
  `references/workflow.md`; users with compatible artifacts may start at any
  stage.
- **Single step**: validate only the prerequisites of the requested step.
- **Teaching**: opt-in only, offered after environment inspection or when input
  problems block progress. Uses all 230 example images; see
  `references/teaching-playbook.md`.

## References

- `references/workflow.md`: per-command guidance, dependencies, artifacts.
- `references/dialog-templates.md`: bilingual dialogue patterns for approval,
  changes, missing prerequisites, collisions, and failures.
- `references/teaching-playbook.md`: demo assets, prerequisites, the tested
  label/correction table recipe, and isolation rules.
- `references/troubleshooting.md`: error and recovery guidance, including the
  schema/help fallback limitation.

## Environment

Run `doctor` before substantive work in a new conversation. Its exit code can be
zero while dependencies are missing, so report the actual packages and devices.
Explain possible initial timm weight downloads before any training approval, and
never promise a duration from image count alone.
