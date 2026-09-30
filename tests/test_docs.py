"""Documentation structure, coverage, and link tests.

The command reference lives in ``docs/commands/`` and the READMEs link to it.
These assertions read the documentation corpus rather than a fixed file list, so
a new command document is covered automatically.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from typer.main import get_command

from otuformer.cli import README_URL, docs_url
from otuformer.cli.main import app

ROOT = Path(__file__).resolve().parents[1]
COMMAND_DOCS = ROOT / "docs" / "commands"

REFERENCE_DOCS = {"training-augmentation", "embedding-metrics"}
ANCHOR_COMMANDS = {"doctor": "#doctor-command", "update": "#update-command"}

# Section 3.5 of the documentation conventions: every heading has one fixed
# counterpart, so a pair can be normalized instead of guessed at.
SECTIONS = {
    "Purpose": "目的",
    "Usage": "用法",
    "Parameters": "参数",
    "Inputs": "输入",
    "Outputs": "输出",
    "Examples": "示例",
    "Notes": "说明",
    "Version Notes": "版本注记",
    "Affected Commands": "适用命令",
    "Profiles": "配置",
    "Orientation Policies": "朝向策略",
    "Resume and Inheritance": "续训与继承",
    "Metrics": "指标",
    "Availability Rules": "可用性规则",
    "CSV Fields": "CSV 字段",
    "Comparability": "可比性",
}
CN_TO_EN = {cn: en for en, cn in SECTIONS.items()}

# Sections each document kind must carry (documentation conventions, Section 4
# and Section 4.4). ``Version Notes`` is conditional and therefore not listed.
REQUIRED_COMMON = ("Purpose", "Usage")
REQUIRED = {
    "command": ("Purpose", "Usage", "Parameters"),
    "training-augmentation": (
        "Purpose", "Usage", "Affected Commands", "Profiles",
        "Orientation Policies", "Resume and Inheritance", "Notes", "Version Notes",
    ),
    "embedding-metrics": (
        "Purpose", "Affected Commands", "Metrics", "Availability Rules",
        "CSV Fields", "Version Notes",
    ),
}

# Fixed section order (conventions Section 4 and Section 4.4). Optional sections
# are omitted, so a document's present sections must be a subsequence of this.
CANONICAL = {
    "command": (
        "Purpose", "Usage", "Parameters", "Inputs", "Outputs", "Examples",
        "Notes", "Version Notes",
    ),
    "training-augmentation": (
        "Purpose", "Usage", "Affected Commands", "Profiles",
        "Orientation Policies", "Resume and Inheritance", "Notes", "Version Notes",
    ),
    "embedding-metrics": (
        "Purpose", "Affected Commands", "Metrics", "Availability Rules",
        "CSV Fields", "Version Notes",
    ),
}

# Section 4.3: Version Notes is required whenever a command document has
# recoverable command-specific history, which is always the case for a new
# command. Only the migrated documents listed here may omit the section.
COMMANDS_WITHOUT_HISTORY = {
    "cluster", "annotate", "diversity", "cam", "export", "extract",
}

COMMAND_MODULES = {
    "doctor": "doctor", "pretrain": "pretrain", "finetune": "finetune",
    "extract": "extract", "cluster": "cluster", "annotate": "annotate",
    "diversity": "diversity", "cam": "cam", "export": "export",
    "update": "update",
}

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FLAG_RE = re.compile(r"`(--?[A-Za-z][A-Za-z0-9-]*)`")
TOKEN_RE = re.compile(r"(?<![\w-])--?[a-z][a-z0-9-]*")


# ------------------------------------------------------------------- corpus

def en_docs() -> list[tuple[Path, str]]:
    """README.md plus every command document that is not a ``.cn.md`` twin."""
    paths = [ROOT / "README.md"] + [
        p for p in sorted(COMMAND_DOCS.glob("*.md"))
        if not p.name.endswith(".cn.md")
    ]
    return [(p, p.read_text(encoding="utf-8")) for p in paths]


def cn_docs() -> list[tuple[Path, str]]:
    """README.cn.md plus every ``.cn.md`` twin."""
    paths = [ROOT / "README.cn.md"] + sorted(COMMAND_DOCS.glob("*.cn.md"))
    return [(p, p.read_text(encoding="utf-8")) for p in paths]


def all_docs() -> list[tuple[Path, str]]:
    return en_docs() + cn_docs()


def english_doc_names() -> set[str]:
    """Document names without the extension.

    Uses ``name.removesuffix`` rather than ``Path.stem``: ``Path("pretrain.cn.md").stem``
    is ``"pretrain.cn"``, which would leak Chinese twins into a command-name set.
    """
    return {
        p.name.removesuffix(".md")
        for p in COMMAND_DOCS.glob("*.md")
        if not p.name.endswith(".cn.md")
    }


def command_doc_stems() -> set[str]:
    return english_doc_names() - REFERENCE_DOCS


def registered_commands() -> set[str]:
    return {group.name for group in app.registered_groups}


def sections_of(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        m = re.match(r"^## (.+?)\s*$", line)
        if m:
            out.append(CN_TO_EN.get(m.group(1), m.group(1)))
    return out


def section_text(text: str, heading: str) -> str:
    """Body of one ``##`` section, excluding the heading line."""
    lines = text.splitlines()
    names = set(SECTIONS) | set(CN_TO_EN)
    start = next(
        (i for i, l in enumerate(lines) if re.match(rf"^## ({re.escape(heading)}|{re.escape(SECTIONS.get(heading, heading))})\s*$", l)),
        None,
    )
    if start is None:
        return ""
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            return "\n".join(lines[start + 1:i])
    return "\n".join(lines[start + 1:])


CJK_RE = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def clean_help(text: str) -> str:
    """The same normalization the documents were generated with."""
    text = " ".join((text or "").split())
    text = re.sub(r"documented at\s+https?://\S+\.?",
                  "documented in this document.", text)
    text = re.sub(r"See\s+https?://\S+\.?", "", text)
    text = text.replace(" | ", " / ").replace("|", "/")
    return re.sub(r"\s{2,}", " ", text).strip()


def cli_option_help(name: str) -> dict[str, str]:
    """Primary flag -> full ``--help`` text, straight from the Typer callback."""
    module = __import__(f"otuformer.cli.{COMMAND_MODULES[name]}", fromlist=["app"])
    callback = module.app.registered_callback.callback
    out: dict[str, str] = {}
    for parameter in inspect.signature(callback).parameters.values():
        if parameter.name == "ctx":
            continue
        oi = parameter.default
        decls = tuple(getattr(oi, "param_decls", None) or ()) or (
            f"--{parameter.name.replace('_', '-')}",
        )
        out[decls[0]] = clean_help(getattr(oi, "help", "") or "")
    return out


def _norm_prose(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = text.replace("`", "").replace("*", "")
    return re.sub(r"\s+", "", text).lower().rstrip(".")


def cli_flags(name: str) -> set[str]:
    module = __import__(f"otuformer.cli.{COMMAND_MODULES[name]}", fromlist=["app"])
    callback = module.app.registered_callback.callback
    assert not module.app.registered_commands, (
        f"{name}: a top-level command must be a sub-Typer app, not a "
        "registered command"
    )
    flags: set[str] = set()
    for parameter in inspect.signature(callback).parameters.values():
        if parameter.name == "ctx":
            continue
        from typer.models import OptionInfo

        assert isinstance(parameter.default, OptionInfo), (
            f"{name}: callback parameter {parameter.name!r} is not a named option. "
            "The documentation convention assumes single-callback sub-Typer apps "
            "exposing named options; a positional argument needs an explicit design."
        )
        decls = tuple(parameter.default.param_decls or ())
        if not decls:
            decls = (f"--{parameter.name.replace('_', '-')}",)
        flags.update(decls)
    return flags


def help_output(args: list[str]) -> str:
    return CliRunner().invoke(get_command(app), args).output


def compact(text: str) -> str:
    """Whitespace-free form.

    Rich wraps long URLs mid-word at the default width, so a raw substring test
    fails even when the URL is correct.
    """
    return "".join(text.split())


def readme_pairs() -> list[tuple[str, str, str]]:
    return [
        ("README.md", (ROOT / "README.md").read_text(encoding="utf-8"), "en"),
        ("README.cn.md", (ROOT / "README.cn.md").read_text(encoding="utf-8"), "cn"),
    ]


def command_table(text: str, lang: str) -> list[tuple[str, str]]:
    """``(command, target)`` rows of the README command table."""
    heading = "## Commands" if lang == "en" else "## 命令"
    stop = "## License" if lang == "en" else "## 许可证"
    lines = text.splitlines()
    start = lines.index(heading)
    end = lines.index(stop, start)
    rows = []
    for line in lines[start:end]:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].startswith("`otuformer "):
            continue
        command = cells[0].strip("`").removeprefix("otuformer ").strip()
        m = LINK_RE.search(line)
        assert m, f"command table row has no link: {line}"
        rows.append((command, m.group(1)))
    return rows


# ------------------------------------------------------------------- tests

def test_every_command_doc_has_a_chinese_twin() -> None:
    english = {
        p.name.removesuffix(".md")
        for p in COMMAND_DOCS.glob("*.md")
        if not p.name.endswith(".cn.md")
    }
    chinese = {p.name.removesuffix(".cn.md") for p in COMMAND_DOCS.glob("*.cn.md")}
    assert english, "no command documents found"
    assert english == chinese, (
        f"missing Chinese twins: {sorted(english - chinese)}; "
        f"orphan Chinese documents: {sorted(chinese - english)}"
    )


def _shape(text: str, heading: str) -> tuple[int, int, int, int]:
    """(paragraphs, bullets, fences, tables) of one section.

    The Chinese document is a translation of the English one, so a section must
    carry the same number of blocks in both. Comparing shapes catches a dropped
    or added block, which paragraph text alone cannot.
    """
    body = section_text(text, heading).splitlines()
    paragraphs = bullets = fences = tables = 0
    in_fence = False
    buffer: list[str] = []
    for line in body:
        if line.startswith("```"):
            fences += 1
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.startswith("|--"):
            tables += 1
        if line.lstrip().startswith("- "):
            bullets += 1
        if line.strip():
            buffer.append(line)
        elif buffer:
            paragraphs += 1
            buffer = []
    if buffer:
        paragraphs += 1
    return paragraphs, bullets, fences, tables


def option_headings(text: str) -> list[tuple[str, ...]]:
    """The Parameters definition list, as a sequence of declared flag groups.

    Flags rather than raw headings: the Chinese documents join an alias with
    ``、`` where the English ones use ``,``.
    """
    out = []
    for line in section_text(text, "Parameters").splitlines():
        if line.startswith("### "):
            out.append(tuple(FLAG_RE.findall(line)))
    return out


def option_descriptions(text: str) -> tuple[set[str], dict[str, str]]:
    """(every declared flag, primary flag -> description) from the Parameters list."""
    declared: set[str] = set()
    meanings: dict[str, str] = {}
    for chunk in re.split(r"^### ", section_text(text, "Parameters"), flags=re.MULTILINE)[1:]:
        lines = chunk.splitlines()
        flags = FLAG_RE.findall(lines[0])
        if not flags:
            continue
        declared.update(flags)
        desc = " ".join(
            line for line in lines[1:]
            if line.strip() and not line.startswith(("**Default:**", "**默认值：**"))
        )
        meanings[flags[0]] = desc
    return declared, meanings


@pytest.mark.parametrize("name", sorted(english_doc_names()))
def test_command_doc_pairs_are_well_formed(name: str) -> None:
    en_path = COMMAND_DOCS / f"{name}.md"
    cn_path = COMMAND_DOCS / f"{name}.cn.md"
    en_text = en_path.read_text(encoding="utf-8")
    cn_text = cn_path.read_text(encoding="utf-8")

    switch = f"[English]({name}.md) | [中文]({name}.cn.md)"
    # Section 3.3: the switch line comes immediately after the title.
    for text, path in ((en_text, en_path), (cn_text, cn_path)):
        lede = [line for line in text.splitlines() if line.strip()][:2]
        assert len(lede) == 2 and lede[0].startswith("# "), (
            f"{path.name}: no title line"
        )
        assert lede[1] == switch, (
            f"{path.name}: the language switch is not the line immediately "
            "after the title"
        )

    kind = name if name in REFERENCE_DOCS else "command"
    for text, path in ((en_text, en_path), (cn_text, cn_path)):
        present = sections_of(text)
        missing = [s for s in REQUIRED[kind] if s not in present]
        assert not missing, f"{path.name} is missing required sections: {missing}"
        if kind == "command" and name not in COMMANDS_WITHOUT_HISTORY:
            assert "Version Notes" in present, (
                f"{path.name}: a command with recoverable command-specific "
                "history (always true for a new command) must carry Version Notes"
            )
        # Section 4 / 4.4: the fixed order, with optional sections omitted.
        ordered = [s for s in CANONICAL[kind] if s in present]
        assert present == ordered, (
            f"{path.name}: sections are out of the fixed order — found "
            f"{present}, expected a subsequence of {list(CANONICAL[kind])}"
        )

    en_sections = sections_of(en_text)
    cn_sections = sections_of(cn_text)
    assert en_sections == cn_sections, (
        f"{name}: section sequence differs across the pair\n"
        f"  en: {en_sections}\n  cn: {cn_sections}"
    )
    assert option_headings(en_text) == option_headings(cn_text), (
        f"{name}: the Parameters list covers different options across the pair"
    )

    for section in en_sections:
        en_shape = _shape(en_text, section)
        cn_shape = _shape(cn_text, section)
        assert en_shape == cn_shape, (
            f"{name}.{section}: the Chinese document is not an equivalent "
            f"translation — (paragraphs, bullets, fences, tables) en={en_shape} "
            f"cn={cn_shape}"
        )


@pytest.mark.parametrize("name", sorted(command_doc_stems()))
@pytest.mark.parametrize("lang", ["en", "cn"])
def test_command_doc_covers_every_cli_option(name: str, lang: str) -> None:
    path = COMMAND_DOCS / (f"{name}.md" if lang == "en" else f"{name}.cn.md")
    text = path.read_text(encoding="utf-8")
    documented, meanings = option_descriptions(text)

    expected = cli_flags(name)
    assert expected - documented == set(), (
        f"{path.name} does not document {sorted(expected - documented)}"
    )
    assert documented - expected == set(), (
        f"{path.name} documents flags the CLI does not define: "
        f"{sorted(documented - expected)}"
    )

    # Section 4.1: the reference is at least as detailed as --help. The English
    # document must carry the full option help; the Chinese document must carry a
    # Chinese rendering of it, not the English text verbatim.
    for flag, help_text in cli_option_help(name).items():
        meaning = meanings[flag]
        if lang == "en":
            assert _norm_prose(help_text) in _norm_prose(meaning), (
                f"{path.name}: {flag} is less detailed than --help"
            )
        else:
            assert CJK_RE.search(meaning), (
                f"{path.name}: {flag} has no Chinese rendering"
            )
            assert not _norm_prose(help_text) == _norm_prose(meaning), (
                f"{path.name}: {flag} was left in English"
            )


def test_reference_docs_pairs_agree() -> None:
    literals = {
        "training-augmentation": {"--augmentation", "--orientation-policy"},
        "embedding-metrics": {
            "pretrain", "finetune", "extract",
            "StratifiedKFold", "kNN_Acc_k1", "Linear_Probing_Acc",
            "Linear_Probing_Balanced_Acc", "mAP", "Recall@k", "Silhouette_Score",
        },
    }
    for name in sorted(REFERENCE_DOCS):
        en_text = (COMMAND_DOCS / f"{name}.md").read_text(encoding="utf-8")
        cn_text = (COMMAND_DOCS / f"{name}.cn.md").read_text(encoding="utf-8")

        # A reference document must not duplicate a command parameter table.
        assert "## Parameters" not in en_text, f"{name}.md grew a Parameters table"
        assert "## 参数" not in cn_text, f"{name}.cn.md grew a Parameters table"

        missing_en = sorted(l for l in literals[name] if l not in en_text)
        missing_cn = sorted(l for l in literals[name] if l not in cn_text)
        assert not missing_en, f"{name}.md is missing {missing_en}"
        assert not missing_cn, f"{name}.cn.md is missing {missing_cn}"

    metrics_en = (COMMAND_DOCS / "embedding-metrics.md").read_text(encoding="utf-8")
    metrics_cn = (COMMAND_DOCS / "embedding-metrics.cn.md").read_text(encoding="utf-8")
    for fragment in ("empty CSV fields", "not zeros", "non-query samples",
                     "singleton", "denominator"):
        assert fragment in metrics_en, f"embedding-metrics.md lost {fragment!r}"
    for fragment in ("空 CSV 字段", "而非 0", "非查询样本", "单例标签", "作为分母"):
        assert fragment in metrics_cn, f"embedding-metrics.cn.md lost {fragment!r}"

    aug_en = (COMMAND_DOCS / "training-augmentation.md").read_text(encoding="utf-8")
    aug_cn = (COMMAND_DOCS / "training-augmentation.cn.md").read_text(encoding="utf-8")
    # The biological contract moved out of the app-level help; it must survive
    # here, and the short warnings must stay in the CLI (checked in test_cli_smoke).
    # Prose is hard-wrapped, so match against whitespace-free text.
    aug_en_flat = "".join(aug_en.lower().split())
    aug_cn_flat = "".join(aug_cn.split())
    for phrase in ("dorsal", "ventral", "lateral", "in-plane", "doesnotguarantee"):
        assert phrase in aug_en_flat, f"training-augmentation.md lost {phrase!r}"
    for phrase in ("背面", "腹面", "侧面", "不保证"):
        assert phrase in aug_cn_flat, f"training-augmentation.cn.md lost {phrase!r}"
    en_flags = set(TOKEN_RE.findall(aug_en))
    cn_flags = set(TOKEN_RE.findall(aug_cn))
    assert en_flags == cn_flags, (
        f"training-augmentation flags differ: en-only {sorted(en_flags - cn_flags)}, "
        f"cn-only {sorted(cn_flags - en_flags)}"
    )


@pytest.mark.parametrize("command", sorted(registered_commands()))
def test_every_registered_command_help_links_documentation(command: str) -> None:
    output = compact(help_output([command, "--help"]))
    expected = README_URL + ANCHOR_COMMANDS[command] if command in ANCHOR_COMMANDS \
        else docs_url(command)
    assert expected in output, f"`otuformer {command} --help` does not link {expected}"
    if command in {"pretrain", "finetune"}:
        assert docs_url("training-augmentation") in output, (
            f"`otuformer {command} --help` does not link the augmentation contract"
        )


def test_command_set_is_closed() -> None:
    registered = registered_commands()
    expected_stems = command_doc_stems()

    for name, text, lang in readme_pairs():
        rows = command_table(text, lang)
        commands = [c for c, _ in rows]
        assert len(commands) == len(set(commands)), (
            f"{name}: the command table repeats a command"
        )
        assert set(commands) == expected_stems | set(ANCHOR_COMMANDS), (
            f"{name}: table lists {sorted(set(commands))}, expected "
            f"{sorted(expected_stems | set(ANCHOR_COMMANDS))}"
        )
        suffix = "" if lang == "en" else ".cn"
        expected = {stem: f"docs/commands/{stem}{suffix}.md" for stem in expected_stems}
        expected.update(ANCHOR_COMMANDS)
        assert dict(rows) == expected, (
            f"{name}: command table targets differ\n"
            f"  actual:   {dict(rows)}\n  expected: {expected}"
        )

    assert expected_stems | set(ANCHOR_COMMANDS) == registered, (
        "the registered commands, the command documents, and the README table "
        "have drifted apart"
    )


def test_local_document_links_resolve() -> None:
    for path, text in all_docs():
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, fragment = target.partition("#")
            target_path = (path.parent / file_part).resolve() if file_part else path
            assert target_path.exists(), f"{path.name}: broken link {target!r}"
            if fragment and target_path.is_file():
                target_text = target_path.read_text(encoding="utf-8")
                assert f'<a id="{fragment}">' in target_text, (
                    f"{path.name}: link {target!r} has no explicit anchor "
                    f'<a id="{fragment}"> in {target_path.name}'
                )


def test_reference_links_are_correct() -> None:
    edges = {
        "pretrain": {"training-augmentation", "embedding-metrics"},
        "finetune": {"training-augmentation", "embedding-metrics"},
        "extract": {"embedding-metrics"},
    }
    for stem, refs in edges.items():
        for suffix in ("", ".cn"):
            path = COMMAND_DOCS / f"{stem}{suffix}.md"
            targets = LINK_RE.findall(path.read_text(encoding="utf-8"))
            for ref in refs:
                target = f"{ref}{suffix}.md"
                # A real Markdown link, and to the document itself rather than
                # a ``#parameters`` fragment (conventions Section 7.8).
                assert target in targets, (
                    f"{path.name} has no link to {target} (found {targets})"
                )

    back = {
        "training-augmentation": ("pretrain", "finetune"),
        "embedding-metrics": ("pretrain", "finetune", "extract"),
    }
    for ref, stems in back.items():
        for suffix in ("", ".cn"):
            path = COMMAND_DOCS / f"{ref}{suffix}.md"
            affected = section_text(path.read_text(encoding="utf-8"), "Affected Commands")
            targets = LINK_RE.findall(affected)
            for stem in stems:
                target = f"{stem}{suffix}.md"
                assert target in targets, (
                    f"{path.name} Affected Commands has no link to {target} "
                    f"(found {targets})"
                )
