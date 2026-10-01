# Dialogue templates

Keep the user's language; keep commands, option names, paths, and field names in
English. The card is produced from the runtime schema, never from memory.

## Proposing one step

```text
[English]
Before running anything, here is the complete parameter card for the proposed step.

<card from export_cli_schema.py --inputs-json>

Please confirm this exact command, its inputs, the output directory, and the
execution environment, or tell me which parameters to change.

[中文]
在运行之前，这是该步骤的完整参数卡。

<card>

请确认这条命令、输入、输出目录与运行环境，或告诉我需要修改哪些参数。
```

## Parameter changes

```text
[English] Understood: changing <option> from <old> to <new>. Approval applies only
to the exact displayed command, so I will re-generate the card for that change.

[中文] 好的：把 <option> 从 <old> 改成 <new>。批准只对已展示的那一条命令生效，
所以我需要为这次修改重新生成参数卡。
```

## Missing prerequisite

```text
[English] This step needs <artifact> from <step>. I can (a) run that prerequisite
as its own approved step with its own full card, (b) use an existing artifact you
select, or (c) explain the requirement without running anything.

[中文] 该步骤需要 <步骤> 产出的 <artifact>。我可以 (a) 把该前置步骤作为独立步骤、
用完整的参数卡单独申请批准，(b) 使用你指定的已有产物，或 (c) 只解释需求、不执行。
```

## Existing output directory

```text
[English] <resolved path> already contains results. Choose a fresh directory, use
a supported training --resume, or explicitly approve overwriting <resolved path>
(that approval is separate from approval of the command itself).

[中文] <路径> 中已有结果。请选择新目录、使用受支持的训练 --resume，或明确批准覆盖
<路径>（该批准与命令批准是两件事）。
```

## Overwrite consent

```text
[English] Overwriting requires naming the exact directory and is never implied by
approving the command: confirm that <resolved path> may be cleared.

[中文] 覆盖需要指名具体目录，且不会因为批准命令而默认成立：请确认可以清空 <路径>。
```

## Declined or changed consent

```text
[English] Nothing was run. <proposal> stays unexecuted. Tell me what to change and I
will present a new card.

[中文] 未执行任何操作。<方案> 保持未执行状态。告诉我需要修改什么，我会给出新的参数卡。
```

## Schema unavailable

```text
[English] Structured schema export failed, so I will not propose an analysis or an
installation. Parameter source: CLI --help fallback. Complete structured
validation: unavailable. I can still inspect logs, results, and help output.

[中文] 结构化 schema 导出失败，因此我不会提出分析或安装方案。
参数来源：CLI --help 回退。完整结构化校验：不可用。仍然可以阅读日志、结果与帮助信息。
```

## After a step (summary + recommendation)

```text
[English] Ran: <command> (exit <code>).
Produced: <artifact paths actually inspected>.
Meaning: <key results with documented limits>; warnings <list>;
unresolved <conditions>.
Recommendation: continue to <next step> | redo this step with <changes> |
stop and fix <blocker>, because <one-line reason>.
Continue, redo, or stop?

[中文] 已运行：<命令>（退出码 <code>）。
产物：<实际检查过的产物路径>。
结论：<关键结果及其已知局限>；警告 <列表>；未解决 <条件>。
建议：继续 <下一步> ｜ 以 <修改项> 重做本步 ｜ 先解决 <阻塞项>，理由：<一句话>。
继续、重做，还是停止？
```

When the user already authorized the next step (or asked for the whole sequence),
skip the open question and go straight to that step's card:

```text
[English] Next: <step>. Here is its full parameter card; approve it to continue.

[中文] 下一步：<步骤>。这是它的完整参数卡；确认后继续。
```

## Teaching mode

```text
[English] This demonstration uses all 230 example images under an approved demo
root, not your data, and does not replace an analysis of your own samples. Short
training here only illustrates the procedure, not encoder quality.

[中文] 本次演示在获批的 demo 目录下使用全部 230 张示例图片，不使用你的数据，也不能
替代对你样本的分析。短训练仅用于演示流程，不代表编码器质量。
```
