"""Workflow Skill portability, teaching, and approval contracts (plan Task 4).

These tests validate the instruction artifacts and the derived teaching tables
against the real example assets. They do not train, extract, or otherwise run an
analysis, and a pass here is not evidence that any agent obeys the approval rules.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "otuformer-workflow"
REFERENCES = SKILL / "references"
REFERENCES_FILES = (
    "workflow.md",
    "dialog-templates.md",
    "teaching-playbook.md",
    "troubleshooting.md",
)
EXAMPLES = ROOT / "examples" / "Epidorcus"
IMAGE_DIR = EXAMPLES / "images"
LABEL_CSV = EXAMPLES / "figs.csv"

ALLOWED_FRONTMATTER = {"name", "description", "license", "compatibility", "metadata"}
FORBIDDEN_FRONTMATTER = (
    "allowed-tools",
    "disable-model-invocation",
    "user-invocable",
    "argument-hint",
    "context",
    "agent",
)
FORBIDDEN_BODY_FEATURES = (
    "!`",
    "$ARGUMENTS",
    "mcp__",
    "Bash(",
    "Read(",
    "Write(",
    "Edit(",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def skill_text() -> str:
    return (SKILL / "SKILL.md").read_text(encoding="utf-8")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    assert lines[0] == "---", "SKILL.md must start with frontmatter"
    end = lines.index("---", 1)
    fields: dict[str, str] = {}
    current: str | None = None
    for line in lines[1:end]:
        if line[:1].isspace() and current:
            fields[current] = f"{fields[current]} {line.strip()}".strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            current = key.strip()
            fields[current] = value.strip()
    body = "\n".join(lines[end + 1 :])
    for key, value in list(fields.items()):
        if value.startswith(">"):
            fields[key] = value[1:].strip()
    return fields, body


def runtime_commands() -> list[str]:
    import importlib

    schema = importlib.import_module("otuformer.cli_schema").export_cli_schema()
    return list(schema["commands"])


def test_skill_portable_frontmatter_and_links() -> None:
    text = skill_text()
    fields, body = parse_frontmatter(text)

    assert fields["name"] == "otuformer-workflow" == SKILL.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"])
    assert len(fields["name"]) <= 64
    assert 0 < len(fields["description"]) <= 1024
    assert fields["license"] == "LICENSE.txt"
    assert set(fields) <= ALLOWED_FRONTMATTER

    for key in FORBIDDEN_FRONTMATTER:
        assert key not in fields, f"{key} is not portable"
        assert f"\n{key}:" not in text
    for feature in FORBIDDEN_BODY_FEATURES:
        assert feature not in text, f"harness-specific feature found: {feature}"

    license_file = SKILL / "LICENSE.txt"
    assert license_file.is_file() and not license_file.is_symlink()
    assert license_file.read_bytes() == (ROOT / "LICENSE").read_bytes()

    # every Skill-relative link resolves from the loaded Skill directory
    references = set(re.findall(r"`(references/[A-Za-z0-9._-]+)`", text + body))
    assert references, "SKILL.md must route to reference files"
    for reference in references:
        assert (SKILL / reference).is_file(), reference
    for name in REFERENCES_FILES:
        assert (REFERENCES / name).is_file(), name
    assert (SKILL / "scripts" / "export_cli_schema.py").is_file()

    for referenced in re.findall(r"\(([^)]+\.md)\)", text):
        if referenced.startswith("http"):
            continue
        assert (SKILL / referenced).is_file(), referenced


def test_workflow_guidance_covers_live_commands() -> None:
    headings = [
        line[3:].strip()
        for line in (REFERENCES / "workflow.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    ]
    assert headings == runtime_commands()

    def missing_guidance(commands: list[str]) -> set[str]:
        return set(commands) - set(headings)

    assert missing_guidance(runtime_commands()) == set()
    assert missing_guidance([*runtime_commands(), "brand-new-command"]) == {
        "brand-new-command"
    }

    text = skill_text()
    intents = {
        "training": ("pretrain", "预训练", "fine-tun", "微调"),
        "extraction": ("extract", "提取"),
        "clustering": ("cluster", "聚类"),
        "annotation": ("annotate", "注释", "标注"),
        "diversity": ("diversity", "多样性"),
        "teaching": ("teach", "example images", "演示"),
        "troubleshooting": ("troubleshoot", "fail", "报错", "排查"),
    }
    for intent, terms in intents.items():
        assert any(term.lower() in text.lower() for term in terms), intent
    assert "或" in text, "description must carry Chinese trigger phrases"


def sample_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(EXAMPLES.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(EXAMPLES).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_all_teaching_assets_resolve_without_mutation(tmp_path: Path) -> None:
    from otuformer.training.dataset import _build_recursive_index, _resolve_image_path

    before = sample_digest()

    rows = list(csv.DictReader(LABEL_CSV.open(encoding="utf-8")))
    assert len(rows) == 230
    assert all(set(row) >= {"image", "label"} for row in rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    assert counts == {"Epidorcus_gracilis": 102, "Epidorcus_tonkinensis": 128}

    images = [
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    assert len(images) == 230
    basenames = {path.name for path in images}
    assert len(basenames) == 230
    assert {Path(row["image"]).name for row in rows} == basenames

    by_relative, by_name = _build_recursive_index(IMAGE_DIR)
    for row in rows:
        resolved = _resolve_image_path(IMAGE_DIR, row["image"], by_relative, by_name)
        assert resolved.is_file()
        assert resolved.parent.name == row["label"]
    assert len(by_name) == 230

    # a missing basename returns the unresolved direct path (the runtime's
    # permissive fallback), while an ambiguous basename is rejected outright
    assert not _resolve_image_path(
        IMAGE_DIR, "not-a-real-image.jpg", by_relative, by_name
    ).exists()

    ambiguous_root = tmp_path / "ambiguous"
    for label in ("a", "b"):
        directory = ambiguous_root / label
        directory.mkdir(parents=True)
        (directory / "same.jpg").write_bytes(b"x")
    index = _build_recursive_index(ambiguous_root)
    with pytest.raises(FileNotFoundError):
        _resolve_image_path(ambiguous_root, "same.jpg", *index)

    assert sample_digest() == before

    # the Skill documents the copied-installation requirement explicitly
    teaching = (REFERENCES / "teaching-playbook.md").read_text(encoding="utf-8")
    assert "explicit examples root" in teaching
    assert "../../examples" not in teaching


def load_teaching_recipe() -> dict[str, Any]:
    """Load the exact recipe from its marked code block, not a copy."""
    text = (REFERENCES / "teaching-playbook.md").read_text(encoding="utf-8")
    block = re.search(
        r"<!-- teaching-recipe:start -->\s*```python\n(.*?)\n```\s*<!-- teaching-recipe:end -->",
        text,
        re.DOTALL,
    )
    assert block, "teaching recipe markers are missing"
    namespace: dict[str, Any] = {}
    exec(compile(block.group(1), "teaching-playbook.md", "exec"), namespace)
    assert "prepare_teaching_table" in namespace
    return namespace


def write_embeddings(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "dim_0", "dim_1"])
        for index, image_id in enumerate(ids):
            writer.writerow([image_id, float(index), float(index) + 0.5])


def example_ids() -> list[str]:
    rows = list(csv.DictReader(LABEL_CSV.open(encoding="utf-8")))
    return [Path(row["image"]).name for row in rows]


def test_teaching_id_mapping_recipe(tmp_path: Path) -> None:
    recipe = load_teaching_recipe()["prepare_teaching_table"]
    demo_root = tmp_path / "demo"
    demo_root.mkdir()
    ids = example_ids()
    expected = {
        Path(row["image"]).name: row["label"]
        for row in csv.DictReader(LABEL_CSV.open(encoding="utf-8"))
    }

    embeddings = demo_root / "embeddings.csv"
    write_embeddings(embeddings, ids)

    labels_out = demo_root / "labels.csv"
    assert (
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=labels_out,
            demo_root=demo_root,
        )
        == 230
    )
    written = list(csv.DictReader(labels_out.open(encoding="utf-8")))
    assert len(written) == 230
    assert {row["id"]: row["label"] for row in written} == expected

    # emitted ids are preserved verbatim, never rewritten to a basename
    nested_ids = [f"Epidorcus_gracilis/{image_id}" for image_id in ids]
    nested_embeddings = demo_root / "nested.csv"
    write_embeddings(nested_embeddings, nested_ids)
    nested_out = demo_root / "nested-labels.csv"
    assert (
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=nested_embeddings,
            output_csv=nested_out,
            demo_root=demo_root,
        )
        == 230
    )
    nested_written = {row["id"] for row in csv.DictReader(nested_out.open(encoding="utf-8"))}
    assert nested_written == set(nested_ids)

    # fabricated absolute ids with real example basenames are refused: matching a
    # basename is not proof that the emitted id names a real example image
    fake_ids = [f"/nonexistent/outside/{image_id}" for image_id in ids]
    fake_embeddings = demo_root / "fake.csv"
    write_embeddings(fake_embeddings, fake_ids)
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=fake_embeddings,
            output_csv=demo_root / "fake-labels.csv",
            demo_root=demo_root,
        )

    # a partition that agrees with the labels needs no corrections
    agreeing = demo_root / "agreeing.csv"
    with agreeing.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for image_id in ids:
            writer.writerow([image_id, f"OTU_{expected[image_id]}"])
    clean_out = demo_root / "clean-corrections.csv"
    assert (
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=clean_out,
            demo_root=demo_root,
            assignments_csv=agreeing,
        )
        == 0
    )
    assert list(csv.reader(clean_out.open(encoding="utf-8"))) == [["id", "cluster"]]

    # a partition that disagrees yields label-informed corrections only
    wrong = demo_root / "wrong.csv"
    moved = ids[:10]
    with wrong.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for image_id in ids:
            cluster = f"OTU_{expected[image_id]}"
            if image_id in moved:
                cluster = "OTU_Epidorcus_tonkinensis" if expected[image_id] == "Epidorcus_gracilis" else "OTU_Epidorcus_gracilis"
            writer.writerow([image_id, cluster])
    corrections_out = demo_root / "corrections.csv"
    assert (
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=corrections_out,
            demo_root=demo_root,
            assignments_csv=wrong,
        )
        == len(moved)
    )
    corrections = list(csv.DictReader(corrections_out.open(encoding="utf-8")))
    assert {row["id"] for row in corrections} == set(moved)
    for row in corrections:
        assert row["cluster"] == f"OTU_{expected[row['id']]}"

    # a tied label-to-cluster mapping must be reviewed, then honoured
    tied = demo_root / "tied.csv"
    with tied.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for index, image_id in enumerate(ids):
            label = expected[image_id]
            alternating = index % 2 == 0
            writer.writerow(
                [image_id, "OTU_A" if alternating == (label == "Epidorcus_gracilis") else "OTU_B"]
            )
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=demo_root / "tied-out.csv",
            demo_root=demo_root,
            assignments_csv=tied,
        )
    chosen = demo_root / "chosen.csv"
    chosen_count = recipe(
        examples_root=IMAGE_DIR,
        embeddings_csv=embeddings,
        output_csv=chosen,
        demo_root=demo_root,
        assignments_csv=tied,
        label_to_cluster={
            "Epidorcus_gracilis": "OTU_A",
            "Epidorcus_tonkinensis": "OTU_B",
        },
    )
    assert chosen_count > 0
    for row in csv.DictReader(chosen.open(encoding="utf-8")):
        expected_cluster = "OTU_A" if expected[row["id"]] == "Epidorcus_gracilis" else "OTU_B"
        assert row["cluster"] == expected_cluster
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=demo_root / "bad-mapping.csv",
            demo_root=demo_root,
            assignments_csv=tied,
            label_to_cluster={"unknown-label": "OTU_A"},
        )

    # two real files sharing a basename are ambiguous, even in same-named dirs
    ambiguous_tree = tmp_path / "ambiguous-examples"
    (ambiguous_tree / "Epidorcus_gracilis").mkdir(parents=True)
    (ambiguous_tree / "nested" / "Epidorcus_gracilis").mkdir(parents=True)
    (ambiguous_tree / "Epidorcus_gracilis" / "same.jpg").write_bytes(b"x")
    (ambiguous_tree / "nested" / "Epidorcus_gracilis" / "same.jpg").write_bytes(b"y")
    (ambiguous_tree / "figs.csv").write_text(
        "image,label\nsame.jpg,Epidorcus_gracilis\n", encoding="utf-8"
    )
    ambiguous_embeddings = demo_root / "ambiguous.csv"
    write_embeddings(ambiguous_embeddings, ["same.jpg"])
    with pytest.raises(ValueError):
        recipe(
            examples_root=ambiguous_tree,
            embeddings_csv=ambiguous_embeddings,
            output_csv=demo_root / "ambiguous-out.csv",
            demo_root=demo_root,
            require_all_examples=False,
        )

    # partition ids must be the emitted extraction ids, and each id once only
    prefixed_embeddings = demo_root / "prefixed.csv"
    write_embeddings(prefixed_embeddings, [f"emitted/{image_id}" for image_id in ids])
    renamed_partition = demo_root / "renamed.csv"
    with renamed_partition.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for image_id in ids:
            writer.writerow([f"unrelated/{image_id}", f"OTU_{expected[image_id]}"])
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=prefixed_embeddings,
            output_csv=demo_root / "renamed-out.csv",
            demo_root=demo_root,
            assignments_csv=renamed_partition,
            label_to_cluster={
                "Epidorcus_gracilis": "OTU_Epidorcus_gracilis",
                "Epidorcus_tonkinensis": "OTU_Epidorcus_tonkinensis",
            },
        )

    duplicated_partition = demo_root / "duplicated.csv"
    with duplicated_partition.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for image_id in ids:
            writer.writerow([image_id, f"OTU_{expected[image_id]}"])
        writer.writerow([ids[0], f"OTU_{expected[ids[0]]}"])
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=demo_root / "duplicated-out.csv",
            demo_root=demo_root,
            assignments_csv=duplicated_partition,
        )

    # missing partition fields are rejected before they can become "nan"
    blank = demo_root / "blank-cluster.csv"
    with blank.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for image_id in ids:
            label = expected[image_id]
            writer.writerow(
                [image_id, "" if label == "Epidorcus_gracilis" else "OTU_Epidorcus_tonkinensis"]
            )
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=demo_root / "blank-out.csv",
            demo_root=demo_root,
            assignments_csv=blank,
            label_to_cluster={
                "Epidorcus_gracilis": "nan",
                "Epidorcus_tonkinensis": "OTU_Epidorcus_tonkinensis",
            },
        )

    # a reviewed cluster ID is written exactly as reviewed, not trimmed
    padded = demo_root / "padded.csv"
    with padded.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for index, image_id in enumerate(ids):
            label = expected[image_id]
            alternating = index % 2 == 0
            writer.writerow(
                [
                    image_id,
                    " OTU_A " if alternating == (label == "Epidorcus_gracilis") else "OTU_B",
                ]
            )
    padded_out = demo_root / "padded-out.csv"
    assert (
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=padded_out,
            demo_root=demo_root,
            assignments_csv=padded,
            label_to_cluster={
                "Epidorcus_gracilis": " OTU_A ",
                "Epidorcus_tonkinensis": "OTU_B",
            },
        )
        > 0
    )
    assert " OTU_A " in {row["cluster"] for row in csv.DictReader(padded_out.open(encoding="utf-8"))}

    # an explicit mapping must be a real, non-empty cluster of that label
    for broken in (
        {"Epidorcus_gracilis": None, "Epidorcus_tonkinensis": "OTU_B"},
        {"Epidorcus_gracilis": "   ", "Epidorcus_tonkinensis": "OTU_B"},
        {"Epidorcus_gracilis": "OTU_NOT_IN_PARTITION", "Epidorcus_tonkinensis": "OTU_B"},
        {"not-a-label": "OTU_A"},
    ):
        with pytest.raises(ValueError):
            recipe(
                examples_root=IMAGE_DIR,
                embeddings_csv=embeddings,
                output_csv=demo_root / "bad-mapping.csv",
                demo_root=demo_root,
                assignments_csv=tied,
                label_to_cluster=broken,
            )

    # duplicate embeddings ids are rejected
    duplicated = demo_root / "duplicate.csv"
    write_embeddings(duplicated, [*ids, ids[0]])
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=duplicated,
            output_csv=demo_root / "duplicate-out.csv",
            demo_root=demo_root,
        )

    # unmatched ids are rejected
    unmatched = demo_root / "unmatched.csv"
    write_embeddings(unmatched, [*ids[:-1], "unknown-image.jpg"])
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=unmatched,
            output_csv=demo_root / "unmatched-out.csv",
            demo_root=demo_root,
        )

    # teaching mode requires all 230 example images unless explicitly opted out
    partial_embeddings = demo_root / "partial.csv"
    write_embeddings(partial_embeddings, ids[:-1])
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=partial_embeddings,
            output_csv=demo_root / "partial-out.csv",
            demo_root=demo_root,
        )
    assert (
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=partial_embeddings,
            output_csv=demo_root / "partial-opt-out.csv",
            demo_root=demo_root,
            require_all_examples=False,
        )
        == len(ids) - 1
    )

    # a partition missing an id, adding an unknown id, or conflicting on one id fails
    partial = demo_root / "partial-partition.csv"
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for image_id in ids[:-1]:
            writer.writerow([image_id, "OTU1"])
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=demo_root / "partial-partition-out.csv",
            demo_root=demo_root,
            assignments_csv=partial,
        )

    extra = demo_root / "extra-partition.csv"
    with extra.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for image_id in [*ids, "stranger.jpg"]:
            writer.writerow([image_id, "OTU1"])
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=demo_root / "extra-partition-out.csv",
            demo_root=demo_root,
            assignments_csv=extra,
        )

    conflicted = demo_root / "conflicted.csv"
    with conflicted.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "cluster"])
        for image_id in ids:
            writer.writerow([image_id, "OTU1"])
        writer.writerow([ids[0], "OTU2"])
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=demo_root / "conflicted-out.csv",
            demo_root=demo_root,
            assignments_csv=conflicted,
        )

    # destination safety: outside the demo root, inside examples/, or existing
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=tmp_path / "outside.csv",
            demo_root=demo_root,
        )
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=IMAGE_DIR / "never.csv",
            demo_root=demo_root,
        )
    # the whole examples tree is read-only, not only the image directory
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=EXAMPLES / "never.csv",
            demo_root=EXAMPLES,
        )
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=ROOT / "examples" / "never.csv",
            demo_root=ROOT / "examples",
        )
    with pytest.raises(ValueError):
        recipe(
            examples_root=IMAGE_DIR,
            embeddings_csv=embeddings,
            output_csv=labels_out,
            demo_root=demo_root,
        )

    assert not list(IMAGE_DIR.rglob("never.csv"))


def test_skill_approval_recovery_and_interpretation_contracts() -> None:
    text = skill_text()
    references = "\n".join(
        (REFERENCES / name).read_text(encoding="utf-8") for name in REFERENCES_FILES
    )
    corpus = re.sub(r"\s+", " ", f"{text}\n{references}")

    required_clauses = (
        "fresh schema",
        "approval",
        "one step",
        "overwrite",
        "explicit `--out-dir`",
        "--help fallback",
        "exit code",
        "artifacts",
        "--resume",
        # the post-step loop: summarize, recommend, keep moving on instruction
        "after each step",
        "what ran",
        "what it produced",
        "what it means",
        "what i recommend",
        "continue to",
        "redo this step",
        "stop and fix",
        "`<blocker>`",
        "one step is still one approval",
        "never rerun silently",
        "continue, redo, or stop?",
    )
    for clause in required_clauses:
        assert clause.lower() in corpus.lower(), clause

    # the recommendation table and the bilingual post-step template exist
    workflow = (REFERENCES / "workflow.md").read_text(encoding="utf-8")
    assert "Recommended next step after each command" in workflow
    assert "| After | Recommend | Redo/stop when |" in workflow
    for row in ("| `pretrain` |", "| `extract` |", "| `cluster` |", "| `export` |"):
        assert row in workflow, row
    # the actual inferred `sample` field is explained, not only warned about
    assert "first directory level" in workflow
    assert "sample" in workflow

    # source/examples protection and the explicit extension flag are documented
    skill_body = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "--protected-path" in skill_body

    dialogs_text = (REFERENCES / "dialog-templates.md").read_text(encoding="utf-8")
    assert "After a step (summary + recommendation)" in dialogs_text
    for marker in ("Recommendation:", "重做", "继续、重做，还是停止？"):
        assert marker in dialogs_text, marker

    # doctor exit-zero caveat, update classification, supported-only resume
    assert "exit code can be zero" in corpus
    assert "read-only-networked" in corpus or "read-only" in corpus
    assert "supported training `--resume`" in corpus
    assert "no Skill progress database" in corpus or "no auto" in corpus.lower()

    # interpretation distinctions
    assert "morphOTU" in corpus and "species" in corpus
    assert "label classes" in corpus or "label classes are not" in corpus
    assert "phylogen" in corpus
    assert "CAM" in corpus and "attribution" in corpus

    # bilingual dialogue scenarios
    dialogs = (REFERENCES / "dialog-templates.md").read_text(encoding="utf-8")
    for marker in ("[English]", "[中文]"):
        assert marker in dialogs
    for scenario in ("Parameter changes", "Missing prerequisite", "Existing output directory", "Schema unavailable"):
        assert scenario in dialogs, scenario


# -------------------------------------------- Task 5: README and distribution


def readmes() -> list[tuple[str, str]]:
    return [
        ("README.md", (ROOT / "README.md").read_text(encoding="utf-8")),
        ("README.cn.md", (ROOT / "README.cn.md").read_text(encoding="utf-8")),
    ]


def test_readmes_document_workflow_installation() -> None:
    for name, text in readmes():
        assert len(text.splitlines()) <= 200, f"{name} exceeds the 200-line convention"
        assert "skills/otuformer-workflow/SKILL.md" in text, name
        for tool in ("Pi", "Claude Code", "Codex", "OpenCode"):
            assert tool in text, f"{name}: missing {tool}"
        for link in (
            "github.com/earendil-works/pi",
            "code.claude.com/docs/en/skills",
            "developers.openai.com/codex/skills",
            "opencode.ai/docs/skills",
        ):
            assert link in text, f"{name}: missing link {link}"
        assert "register" in text.lower() or "注册" in text, name
        for path_example in (
            ".pi/settings.json",
            ".claude/skills/",
            "config.toml",
            ".opencode/skills/",
        ):
            assert path_example in text, f"{name}: missing executable example {path_example}"
        assert "LICENSE.txt" in text, f"{name}: copies must carry the license"
        assert "examples" in text, f"{name}: copies need an explicit examples root"
        assert "refresh" in text.lower() or "同步" in text, name
        # Codex standalone discovery must not be described as removed
        assert "plugin" in text.lower()
        assert "no longer" not in text.lower()


def test_readmes_document_pi_trust_and_collisions() -> None:
    for name, text in readmes():
        lowered = text.lower()
        assert "trust" in lowered or "信任" in text, name
        assert "project" in lowered or "项目级" in text, name
        assert "name collision" in lowered or "同名" in text, name
        assert "first" in lowered or "先发现" in text, name
        # the collision rule is Pi's, not a rule shared by every tool
        assert "Pi keeps the first skill" in text or "Pi 在同名冲突时保留先发现者" in text, name


def test_readmes_show_conversational_review() -> None:
    for name, text in readmes():
        assert "parameter card" in text.lower() or "参数卡" in text, name
        assert "approval" in text.lower() or "确认" in text, name
        assert "230" in text, f"{name}: the demo uses all 230 example images"
        assert "doctor" in text, name
        # a conversational example, not an automatic pipeline
        assert "You:" in text or "你：" in text, name
        # the README must not become a second flag registry
        assert "Declared default" not in text
        match = re.search(
            r"## (?:Agent Skill|智能体 Skill).*?\n(?=## )", text, re.DOTALL
        )
        assert match, f"{name}: missing workflow Skill section"
        flags = re.findall(r"--[a-z][a-z0-9-]+", match.group(0))
        assert len(flags) <= 6, f"{name}: Skill section lists too many flags: {flags}"


def build_artifacts(dist: Path) -> tuple[Path, Path | None]:
    """Build a wheel (and an sdist when the backend is importable) read-only.

    The wheel goes through pip's PEP 517 build isolation, so the build backend
    never has to be installed into this environment. The sdist leg runs the
    backend straight out of its own downloaded wheel when it is importable, and
    is reported as unavailable otherwise instead of being called a pass.
    """
    build = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(dist)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert build.returncode == 0, build.stderr
    wheel = next(dist.glob("*.whl"))

    sdist: Path | None = None
    backend_obtainable = False
    backend_env = dist / "backend"
    backend_obtainable = importlib.util.find_spec("hatchling") is not None
    if not backend_obtainable:
        # Use the backend without installing it: download its wheels and run it
        # from an extracted directory, exactly like the isolated wheel build.
        download = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-deps",
                "-q",
                "-d",
                str(backend_env / "wheels"),
                "hatchling",
                "packaging",
                "pathspec",
                "pluggy",
                "trove-classifiers",
            ],
            capture_output=True,
            text=True,
        )
        if download.returncode == 0:
            import zipfile

            library = backend_env / "lib"
            for artifact in sorted((backend_env / "wheels").glob("*.whl")):
                with zipfile.ZipFile(artifact) as archive:
                    archive.extractall(library)
            backend_obtainable = (library / "hatchling").is_dir()

    if backend_obtainable:
        sdist_build = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['hatchling', 'build', '-t', 'sdist', '-d', "
                f"'{dist}']; from hatchling.cli import hatchling; hatchling()",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(backend_env / "lib")},
        )
        assert sdist_build.returncode == 0, sdist_build.stderr
        tarballs = list(dist.glob("*.tar.gz"))
        assert tarballs, f"sdist build produced no archive: {sdist_build.stdout}"
        sdist = tarballs[0]
    return wheel, sdist, backend_obtainable


def test_installed_distribution_schema_and_skill_copy(tmp_path: Path) -> None:
    """Opt-in installed-distribution check (prerequisite-gated, never automatic).

    Creating a disposable interpreter and installing into it is an
    environment-changing operation, so this runs only when explicitly requested
    with ``OTUFORMER_INSTALLED_DISTRIBUTION_CHECK=1``. Nothing here replaces the
    operator's active installation, and a skip is prerequisite-not-met rather
    than a pass.
    """
    import os
    import tarfile
    import venv
    import zipfile

    if os.environ.get("OTUFORMER_INSTALLED_DISTRIBUTION_CHECK") != "1":
        pytest.skip(
            "installed-distribution check requires explicit authorization "
            "(set OTUFORMER_INSTALLED_DISTRIBUTION_CHECK=1)"
        )

    dist = tmp_path / "dist"
    wheel, sdist, backend_obtainable = build_artifacts(dist)
    assert wheel.name.endswith(".whl")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert any(name.endswith("otuformer/cli_schema.py") for name in names)
    assert any(name.endswith("otuformer/cli/constraints.py") for name in names)
    assert not any("skills/" in name for name in names), "the wheel must not ship the Skill"
    assert not any(name.endswith((".jpg", ".pth", ".onnx")) for name in names)
    assert any("licenses/LICENSE" in name for name in names), "the wheel must carry the license"

    if sdist is None:
        assert not backend_obtainable, "the build backend was available but no sdist"
        pytest.skip("sdist leg prerequisite not met: no build backend obtainable")
    if sdist is not None:
        with tarfile.open(sdist) as archive:
            sdist_names = archive.getnames()
        assert any(name.endswith("/LICENSE") for name in sdist_names)
        assert any(name.endswith("skills/otuformer-workflow/SKILL.md") for name in sdist_names)
        assert any(
            name.endswith("skills/otuformer-workflow/LICENSE.txt") for name in sdist_names
        )

    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    isolated_python = environment / "bin" / "python"
    if not isolated_python.exists():
        pytest.skip("isolated interpreter layout is not recognised on this platform")
    install = subprocess.run(
        [
            str(isolated_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            "--no-index",
            str(wheel),
        ],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr

    copied_skill = tmp_path / "copied-skill"
    import shutil as _shutil

    _shutil.copytree(SKILL, copied_skill)
    copied_adapter = copied_skill / "scripts" / "export_cli_schema.py"
    assert copied_adapter.is_file()
    assert (copied_skill / "LICENSE.txt").read_bytes() == (ROOT / "LICENSE").read_bytes()

    for arguments in (
        ["--command", "pretrain"],
        [],
    ):
        check = subprocess.run(
            [str(isolated_python), str(copied_adapter), *arguments],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": ""},
        )
        assert check.returncode == 0, check.stderr
        payload = json.loads(check.stdout)
        assert payload["runtime"]["package_version"] == "0.11.0"
        # the isolated interpreter must load the installed distribution, never
        # the source checkout
        assert str(environment) in payload["runtime"]["package_dir"], payload["runtime"]
        if arguments:
            assert set(payload["commands"]) == {"pretrain"}
        else:
            assert set(payload["commands"]) == set(runtime_commands())

    proposal_inputs = tmp_path / "inputs.json"
    proposal_inputs.write_text(
        json.dumps({"input_images_dir": "images", "out_dir": "runs/x"}), encoding="utf-8"
    )
    launcher = environment / "bin" / "otuformer"
    if not launcher.exists():
        pytest.skip(
            "installed console script missing from the disposable environment; "
            "cannot verify per-command help or cards"
        )
    if launcher.exists():
        # every command's installed help must agree with the installed schema
        installed = json.loads(
            subprocess.run(
                [str(isolated_python), str(copied_adapter)],
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
                env={**os.environ, "PYTHONPATH": ""},
                check=True,
            ).stdout
        )["commands"]
        extras = {
            "extract": {"checkpoint": "placeholder.pth"},
            "diversity": {"assignments": "placeholder.csv"},
            "annotate": {"raw_assignments": "placeholder.csv"},
        }
        for name, entry in installed.items():
            help_result = subprocess.run(
                [str(launcher), name, "--help"],
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
                env={**os.environ, "PYTHONPATH": ""},
            )
            assert help_result.returncode == 0, help_result.stderr
            flat = " ".join(help_result.stdout.split())
            # the rich help column truncates long option names with an ellipsis
            truncated = set(re.findall(r"--[A-Za-z0-9-]+…", flat))
            for item in entry["parameters"]:
                option = item["opts"][0]
                visible = option in flat or any(
                    option.startswith(token.rstrip("…")) for token in truncated
                )
                assert visible, f"{name}: {option} missing from installed help"

            # the installed distribution must produce a complete card per command
            explicit = {
                item["name"]: (
                    "placeholder.csv" if item["type"] == "path" else "placeholder"
                )
                for item in entry["parameters"]
                if item["required"]
            }
            explicit.update(extras.get(name, {}))
            if any(item["name"] == "out_dir" for item in entry["parameters"]):
                explicit["out_dir"] = str(tmp_path / "installed-out" / name)
            card_inputs = tmp_path / f"inputs-{name}.json"
            card_inputs.write_text(json.dumps(explicit), encoding="utf-8")
            card_result = subprocess.run(
                [
                    str(isolated_python),
                    str(copied_adapter),
                    "--command",
                    name,
                    "--inputs-json",
                    str(card_inputs),
                    "--executable",
                    str(launcher),
                ],
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
                env={**os.environ, "PYTHONPATH": ""},
            )
            assert card_result.returncode == 0, card_result.stderr
            card = json.loads(card_result.stdout)["card"]
            for item in entry["parameters"]:
                assert item["opts"][0] in card, f"{name}: {item['opts'][0]} missing from card"
            assert "Output safety conditions:" in card
        proposal = subprocess.run(
            [
                str(isolated_python),
                str(copied_adapter),
                "--command",
                "pretrain",
                "--inputs-json",
                str(proposal_inputs),
                "--executable",
                str(launcher),
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": ""},
        )
        assert proposal.returncode == 0, proposal.stderr
        assert json.loads(proposal.stdout)["launcher"]["verified"] is True
