# OTU-Former

**中文** | [English](README.md)

基于图像的形态学 OTU（可操作分类单元）划分工具。提供统一的 `otuformer` CLI，覆盖自监督预训练、监督度量学习微调、嵌入提取、UPGMA 层次聚类、专家校正标注、群落多样性分析、CAM 可视化以及 ONNX 模型导出。

## 为什么选择 OTU-Former

把标本照片变成可复现的 OTU，通常需要手工把嵌入脚本、聚类 notebook 和多样性计算拼在一起。OTU-Former 把这条链路打包成一个命令行工作流：

- **输入图像，输出形态学条码。** 本工具消费标准化的标本图像，产出形态学条码、morphOTU 与多样性数据：训练好的编码器把每张图像变成固定维度的嵌入向量，后续所有步骤都基于这些向量。它不组装 reads、不调用 targets、不推断 ortholog groups。
- **每一步可追溯。** 每个命令都会把输出、日志与解析后的参数写入可预期的目录，便于检查、续跑与复现。
- **专家校正是一等公民。** `annotate` 会把人工 OTU 校正回写到划分结果，并报告其隐含的类内距离。
- **从形态学得到多样性。** `diversity` 计算 alpha 多样性，包括基于嵌入构建的 OTU 质心树得到的 Faith's PD。

## 工作流程

1. **`doctor`** — 检查环境与依赖。
2. **`pretrain`** — 自监督对比预训练，无需标签。
3. **`finetune`** — 监督度量学习微调，目标函数可选。
4. **`extract`** — 从图像提取嵌入向量。
5. **`cluster`** — UPGMA 层次聚类为形态学 OTU。
6. **`annotate`** — 对划分结果应用专家校正。
7. **`diversity`** — 计算群落 alpha 多样性指数。

`cam` 与 `export` 是两条支线：可解释性热图与 ONNX 导出。

## 系统要求

- Python 3.11+
- 操作系统：macOS、Linux、Windows
- 设备：CPU（必选）；CUDA GPU / Apple Silicon MPS（可选）

## 安装

推荐使用隔离的 Python 环境，避免与系统/全局 site-packages 发生依赖冲突。安装前请先克隆仓库并进入项目目录：

```bash
git clone https://github.com/xtmtd/OTU-Former.git && cd OTU-Former
```

### 隔离环境

```bash
# 选项 1：conda
conda create -n otuformer python=3.11 -y && conda activate otuformer && pip install -e .
# 选项 2：uv + venv
uv venv .venv && source .venv/bin/activate && uv pip install -e .
# 选项 3：标准库 venv + pip
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

## 智能体 Skill（对话驱动工作流）

[`skills/otuformer-workflow/`](skills/otuformer-workflow/SKILL.md) 是面向智能体 CLI 的可移植 Skill，`pip install` 不会安装它。按工具注册或链接 —— Pi：`--skill <dir>`、`.pi/settings.json` 的 `skills` 条目，或 `.pi/skills/`、`~/.pi/agent/skills/`（[文档](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/skills.md)）；Claude Code：`.claude/skills/<name>/` 或 `~/.claude/skills/<name>/`（[文档](https://code.claude.com/docs/en/skills)）；Codex：技能目录，或 `~/.codex/config.toml` 中带 `path = ".../SKILL.md"` 的 `[[skills.config]]` 条目（对外分发可打包为仅含 Skill 的 plugin）（[文档](https://developers.openai.com/codex/skills)）；OpenCode：`.opencode/skills/<name>/` 或 `~/.config/opencode/skills/<name>/`（[文档](https://opencode.ai/docs/skills/)）。
各工具规则不同：Pi 与 Claude Code 需要项目信任后才会加载项目级 Skill，且 Pi 在同名冲突时保留先发现者并告警。复制 Skill 时必须一并保留 `LICENSE.txt`，并显式传入 examples 根目录。升级 `otuformer` 后请同步 Skill 正文与脚本；其演示在获批的 demo 目录下使用全部 230 张 `examples/Epidorcus` 图片，不会静默改用你的数据。

```text
你：用 OTU-Former 跑 ./images，并告诉我下一步该做什么 | 机器人：doctor -> 完整参数卡 -> 你确认 -> 只执行一步 -> 报告结果
你：用 runs/finetune/finetune_latest.pth 提取特征 | 机器人：参数卡 -> 确认 -> runs/extract/embeddings.csv
你：跑 230 张示例演示，现在进行到哪一步了？ | 机器人：demo 目录下全部示例；读取日志/产物，并给出继续 / 重做 / 停止的建议
```

## 快速开始

```bash
otuformer doctor
otuformer pretrain --input-images-dir ./images --out-dir runs/pretrain
otuformer finetune --checkpoint runs/pretrain/SSL_latest.pth \
    --train-data labels.csv --input-images-dir ./images --out-dir runs/finetune
otuformer extract --checkpoint runs/finetune/finetune_latest.pth \
    --input-images-dir ./images --out-dir runs/extract
otuformer cluster --embeddings runs/extract/embeddings.csv --out-dir runs/cluster
```

每个命令都有自己的参考文档；见文末命令表中的链接，或运行 `otuformer <命令> --help`。

<a id="doctor-command"></a>
## doctor 命令

诊断环境与依赖是否满足当前功能需求。

```bash
otuformer doctor
```

报告内容涵盖 Python 版本、可用设备（`cpu`、`cuda`、`mps`）以及关键依赖的版本与 ok/missing/outdated 状态。

<a id="update-command"></a>
## update 命令

```bash
otuformer update
otuformer update --check
otuformer update --yes
```

| 选项 | 说明 |
|------|------|
| `--check` | 仅显示版本信息。 |
| `--yes`、`-y` | 不再询问，直接安装。 |

从已发布的 Git tag 检查更新，并在确认后安装最新版本。

## 输出安全

默认情况下，命令拒绝使用非空的 `--out-dir`。请显式使用 `--overwrite`
清空该目录。训练时的 `--resume` 会保留并追加已有输出目录；`--resume`
与 `--overwrite` 不能同时使用。

## Hugging Face 权重

新的 pretrain 和 finetune 会使用公开 timm 权重初始化。首次下载权重或本地
缓存缺失时，出现未认证 Hugging Face 请求的 warning 属于正常现象；`HF_TOKEN`
可选，仅用于提高限额。CAM、extract 和 export 会加载指定的本地 checkpoint，
不需要访问 Hugging Face。

## 通用行为

### 日志

所有命令会在输出目录的 `logs/` 子目录中写入日志文件（`pretrain.log`、`finetune.log`、`extract.log`、`cluster.log`、`annotate.log`、`diversity.log`、`cam.log`、`export.log`），记录完整命令行、时间戳、所有参数值与运行时输出。

### 优雅退出

按 `Ctrl+C` — 当前图像处理完成后退出，保存部分结果。

### 设备选择

`--device auto` 依次优先选择 CUDA、Apple Silicon 上的 MPS，最后回退到 CPU。

### Shell 补全

安装 shell 补全：

```bash
otuformer --install-completion
```

支持的 shell：bash、zsh、fish

### 版本号

当前版本：`0.11.0`。

查看已安装版本：

```bash
otuformer --version
otuformer -v
```

## 项目结构

当前包结构见 [`src/otuformer/`](src/otuformer/)；架构职责见
[总设计文档](docs/superpowers/specs/2026-03-29-otuformer-design.md)。

## 命令

| 命令 | 用途 | 文档 |
|------|------|------|
| `otuformer doctor` | 诊断环境与依赖状态。 | [doctor](#doctor-command) |
| `otuformer update` | 检查并安装最新发布版本。 | [update](#update-command) |
| `otuformer pretrain` | 自监督对比预训练（DINO/iBOT 风格）。 | [docs/commands/pretrain.cn.md](docs/commands/pretrain.cn.md) |
| `otuformer finetune` | 监督度量学习微调（`--loss` 选择目标函数）。 | [docs/commands/finetune.cn.md](docs/commands/finetune.cn.md) |
| `otuformer extract` | 提取图像嵌入向量（支持 ONNX 加速）。 | [docs/commands/extract.cn.md](docs/commands/extract.cn.md) |
| `otuformer cluster` | UPGMA 层次聚类为形态学 OTU。 | [docs/commands/cluster.cn.md](docs/commands/cluster.cn.md) |
| `otuformer annotate` | 应用专家校正，生成精化后的 OTU 标注。 | [docs/commands/annotate.cn.md](docs/commands/annotate.cn.md) |
| `otuformer diversity` | 计算群落 alpha 多样性指数。 | [docs/commands/diversity.cn.md](docs/commands/diversity.cn.md) |
| `otuformer cam` | 生成 GradCAM 等热图。 | [docs/commands/cam.cn.md](docs/commands/cam.cn.md) |
| `otuformer export` | 导出 ONNX 模型。 | [docs/commands/export.cn.md](docs/commands/export.cn.md) |

两份跨命令契约有独立文档：[训练数据增强](docs/commands/training-augmentation.cn.md)（`pretrain` 与 `finetune` 共享的 `--augmentation` 与 `--orientation-policy` 行为）与 [嵌入指标](docs/commands/embedding-metrics.cn.md)（`pretrain`、`finetune` 与 `extract` 共享的指标）。

## 许可证

自 v0.11.0 起，本项目按 [OTU-Former Research and Commercial Use License Notice](LICENSE) 分发：在保留本声明的前提下允许学术、教育与非商业研究使用；商业使用（包括企业内部使用、付费分析或咨询、商业再分发）需事先获得版权持有人的书面许可。这是 source-available 软件，**并非 OSI 认证的开源软件**；v0.11.0 之前发布的版本仍可按 MIT 许可证使用。
你运行软件产生的输出（嵌入、OTU/丰度表、多样性统计、图表、CAM 图像）不因本声明附加任何限制；检查点、导出的模型与第三方材料仍适用各自的条款。

## 联系方式

- 邮箱：`xtmtd.zf@gmail.com`

## 引用

如果您在研究中使用 OTU-Former，请引用：

```bibtex
@software{otuformer2026,
  author = {Zhang, Feng},
  title = {OTU-Former: Image-based Morphological OTU Delineation Toolkit},
  year = {2026},
  url = {https://github.com/xtmtd/OTU-Former}
}
```

预印本可查看：https://doi.org/10.64898/2026.04.28.721370
