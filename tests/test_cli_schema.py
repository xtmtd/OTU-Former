"""Runtime schema fidelity tests (workflow design Section 4, plan Task 1).

These tests compare the exported schema against live Typer runtime objects rather
than against a hand-maintained module list, so new commands and options are
covered automatically. They intentionally re-derive the expected default rules
instead of calling the same helper under test.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.main import get_command

from otuformer.cli.main import app

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_NAMES = frozenset({"ctx", "context"})
HEAVY_MODULES = (
    "torch",
    "torchvision",
    "timm",
    "pytorch_grad_cam",
    "otuformer.training.trainer",
    "otuformer.vision.cam",
)


def cli_schema() -> Any:
    """Import lazily so each test fails individually while the module is absent."""
    return importlib.import_module("otuformer.cli_schema")


def runtime_schema() -> dict[str, Any]:
    return cli_schema().export_cli_schema()


def analysis_params(command: Any) -> list[Any]:
    return [
        parameter
        for parameter in command.params
        if getattr(parameter, "name", None) not in CONTEXT_NAMES
        and getattr(parameter, "expose_value", True)
    ]


def expected_default(parameter: Any) -> dict[str, Any]:
    """Expected serialized default shape, derived from the live parameter."""
    default = parameter.default
    if parameter.required and default is None:
        return {"kind": "unset"}
    if default is None:
        return {"kind": "null"}
    if isinstance(default, Path):
        return {"kind": "value", "value": str(default)}
    return {"kind": "value", "value": default}


def test_schema_matches_every_runtime_callback() -> None:
    schema = runtime_schema()
    root = get_command(app)
    registered = [group.name for group in app.registered_groups]

    assert registered, "Typer registry is empty"
    assert set(schema["commands"]) == set(root.commands) == set(registered)

    for name, command in root.commands.items():
        entry = schema["commands"][name]
        assert entry["path"] == name
        live = analysis_params(command)
        exported = entry["parameters"]
        assert [item["name"] for item in exported] == [item.name for item in live]
        for item, parameter in zip(exported, live):
            assert item["opts"] == list(parameter.opts or [])
            assert item["secondary_opts"] == list(parameter.secondary_opts or [])
            assert item["required"] is bool(parameter.required)
            assert item["is_flag"] is bool(parameter.is_flag)
            assert item["nargs"] in (None, 1)
            assert item["multiple"] is bool(parameter.multiple)
            assert item["help"] == ((parameter.help or "").strip() or None)
            assert item["type"] == getattr(parameter.type, "name", None)
            choices = getattr(parameter.type, "choices", None)
            assert item["choices"] == (list(choices) if choices is not None else None)
            assert item["default"] == expected_default(parameter)
            assert isinstance(item["restriction_source"], str)


def test_schema_covers_known_command_aliases_and_shapes() -> None:
    commands = runtime_schema()["commands"]

    assert commands["doctor"]["parameters"] == []
    assert [item["name"] for item in commands["update"]["parameters"]] == ["check", "yes"]
    assert commands["update"]["parameters"][1]["opts"] == ["--yes", "-y"]

    cluster = {item["name"]: item for item in commands["cluster"]["parameters"]}
    assert cluster["label_csv"]["opts"] == ["--label-csv", "--labels"]
    assert cluster["local_k_strategy"]["choices"] == ["adaptive", "sqrt", "log", "fixed"]
    assert cluster["pca_whitening"]["type"] == "text"
    assert cluster["pca_whitening"]["is_flag"] is False

    diversity = {item["name"]: item for item in commands["diversity"]["parameters"]}
    assert "--embeddings" in diversity["embeddings"]["opts"]
    assert diversity["tree"]["opts"] == ["--tree"]

    pretrain = {item["name"]: item for item in commands["pretrain"]["parameters"]}
    assert pretrain["input_images_dir"]["required"] is True
    assert pretrain["input_images_dir"]["default"] == {"kind": "unset"}
    assert pretrain["train_data"]["default"] == {"kind": "null"}
    assert pretrain["out_dir"]["default"] == {"kind": "value", "value": "runs/pretrain"}
    assert pretrain["out_dir"]["path_constraints"] == {
        "exists": False,
        "file_okay": True,
        "dir_okay": True,
        "writable": False,
        "readable": True,
    }
    assert pretrain["resume"]["type"] == "text"


def test_root_controls_are_separate() -> None:
    schema = runtime_schema()
    root = get_command(app)
    entry = schema["root"]

    assert entry["parameters"] == []
    roles = {control["name"]: control["role"] for control in entry["controls"]}
    assert roles["version"] == "version"
    assert roles["install_completion"] == "completion"
    assert roles["show_completion"] == "completion"
    assert roles["help"] == "help"
    assert set(roles) >= {getattr(parameter, "name", "") for parameter in root.params}

    for name, command in schema["commands"].items():
        assert command["controls"] == [], name
        assert command["builtin_help_option"] == "--help"


def test_schema_unknown_and_unsupported_shapes_fail() -> None:
    module = cli_schema()
    with pytest.raises(ValueError):
        module.export_cli_schema("not-a-command")

    class FakeType:
        name = "integer"
        choices = None
        min = None
        max = None

    class FakeOption:
        type = FakeType()
        opts = ["--fake"]
        secondary_opts: list[str] = []
        name = "fake"
        required = False
        default = 1
        is_flag = False
        flag_value = None
        nargs = 1
        multiple = False
        help = "fake"
        expose_value = True

    module.serialize_parameter(FakeOption())

    with_nargs = FakeOption()
    with_nargs.nargs = 2
    with pytest.raises(ValueError):
        module.serialize_parameter(with_nargs)

    repeated = FakeOption()
    repeated.multiple = True
    with pytest.raises(ValueError):
        module.serialize_parameter(repeated)

    class UnknownType:
        name = "unsupported"

    unknown = FakeOption()
    unknown.type = UnknownType()
    with pytest.raises(ValueError):
        module.serialize_parameter(unknown)

    non_finite = FakeOption()
    non_finite.type = type("FakeFloat", (), {"name": "float", "choices": None, "min": None, "max": None})()
    non_finite.default = float("nan")
    with pytest.raises(ValueError):
        module.serialize_parameter(non_finite)

    opaque = FakeOption()
    opaque.default = object()
    with pytest.raises(ValueError):
        module.serialize_parameter(opaque)


def test_registry_derivation_is_not_a_module_list() -> None:
    module = cli_schema()

    sub = typer.Typer()

    @sub.callback(invoke_without_command=True)
    def demo(alpha: int = typer.Option(1, "--alpha")) -> None:  # pragma: no cover
        raise AssertionError("callbacks must never run during schema export")

    local = typer.Typer(no_args_is_help=True)
    local.add_typer(sub, name="demo")

    schema = module.build_schema(get_command(local))
    assert set(schema["commands"]) == {"demo"}
    assert schema["commands"]["demo"]["parameters"][0]["default"] == {
        "kind": "value",
        "value": 1,
    }

    added = typer.Typer()

    @added.callback(invoke_without_command=True)
    def extra(beta: str = typer.Option("b", "--beta")) -> None:  # pragma: no cover
        raise AssertionError("callbacks must never run during schema export")

    local.add_typer(added, name="extra")
    extended = module.build_schema(get_command(local))
    assert set(extended["commands"]) == {"demo", "extra"}
    assert [item["name"] for item in extended["commands"]["extra"]["parameters"]] == [
        "beta"
    ]


def test_schema_export_does_not_import_heavy_modules() -> None:
    script = textwrap.dedent(
        f"""
        import sys

        from otuformer import cli_schema

        schema = cli_schema.export_cli_schema()
        heavy = [name for name in {HEAVY_MODULES!r} if name in sys.modules]
        print("HEAVY=" + ",".join(heavy))
        print("COMMANDS=" + str(len(schema["commands"])))
        print("PARAMS=" + str(sum(len(c["parameters"]) for c in schema["commands"].values())))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT
    )
    assert completed.returncode == 0, completed.stderr
    heavy = completed.stdout.split("HEAVY=", 1)[1].splitlines()[0]
    assert heavy == "", f"schema export pulled in heavy modules: {heavy}"
    assert "COMMANDS=10" in completed.stdout


def test_schema_export_is_read_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    before = sorted(path for path in tmp_path.iterdir())
    runtime_schema()
    assert sorted(path for path in tmp_path.iterdir()) == before


# --------------------------------------------------------------- Task 2: restrictions


def constraints() -> Any:
    return importlib.import_module("otuformer.cli.constraints")


def test_constraints_match_existing_validation() -> None:
    module = constraints()

    # mask ratio: "auto" or an open interval (0, 1)
    assert module.validate_mask_ratio("auto") == "auto"
    assert module.validate_mask_ratio("0.5") == 0.5
    # numeric options accept their numbers as well as the "auto" keyword
    assert module.validate_size_option("224", stage="--global-crop-size") == 224
    assert module.validate_size_option("auto", stage="--extract-size") is None
    schema_module = cli_schema()
    assert schema_module.validate_parameters(
        "pretrain", {"input_images_dir": "i", "out_dir": "runs/a", "global_crop_size": "224"}
    )["explicit_values"]["global_crop_size"] == "224"
    assert schema_module.validate_parameters(
        "pretrain",
        {"input_images_dir": "i", "out_dir": "runs/a", "mask_ratio": "0.5"},
    )["values"]["mask_ratio"] == "0.5"
    assert schema_module.validate_parameters(
        "export", {"checkpoint": "c.pth", "out_dir": "runs/a", "imgsz": "224"}
    )["values"]["imgsz"] == "224"
    with pytest.raises(ValueError):
        schema_module.validate_parameters(
            "pretrain", {"input_images_dir": "i", "out_dir": "runs/a", "global_crop_size": "abc"}
        )
    with pytest.raises(ValueError):
        schema_module.validate_parameters(
            "finetune",
            {
                "checkpoint": "c.pth",
                "train_data": "t.csv",
                "input_images_dir": "i",
                "out_dir": "runs/a",
                "loss": "subcenter-arcface-compact",
                "compact_weight": -1,
            },
        )
    for bad in ("0", "1", "1.0", "0.0", "-0.1", "half", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            module.validate_mask_ratio(bad)

    # patch options and prototype minimum
    module.validate_patch_options("masked-feature", "blockwise", 2)
    with pytest.raises(ValueError):
        module.validate_patch_options("Masked-Feature", "blockwise", 2)
    with pytest.raises(ValueError):
        module.validate_patch_options("none", "spiral", 2)
    with pytest.raises(ValueError):
        module.validate_patch_options("none", "random", 1)

    # sub-centers stay inside the shared 2..8 bound
    assert module.MIN_SUBCENTERS == 2 and module.MAX_SUBCENTERS == 8
    with pytest.raises(ValueError):
        module.validate_constraints(
            "finetune",
            {"loss": "subcenter-arcface", "subcenters": 9},
            explicit_options=frozenset({"subcenters"}),
        )
    with pytest.raises(ValueError):
        module.validate_constraints(
            "finetune",
            {"loss": "subcenter-arcface", "subcenters": 1},
            explicit_options=frozenset({"subcenters"}),
        )

    # pseudo bounds and positivity, including non-finite values
    module.validate_pseudo_options(
        long_tail="none",
        loss="arcface",
        pseudo_label_from="",
        similarity_floor=-1.0,
        min_gap=0.0,
        neighbors=1,
        cap_multiplier=1,
        absolute_cap=1,
    )
    for floor in (1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            module.validate_pseudo_options(
                long_tail="none",
                loss="arcface",
                pseudo_label_from="",
                similarity_floor=floor,
                min_gap=0.1,
                neighbors=5,
                cap_multiplier=1,
                absolute_cap=1,
            )
    for gap in (2.5, float("nan")):
        with pytest.raises(ValueError):
            module.validate_pseudo_options(
                long_tail="none",
                loss="arcface",
                pseudo_label_from="",
                similarity_floor=0.5,
                min_gap=gap,
                neighbors=5,
                cap_multiplier=1,
                absolute_cap=1,
            )
    for keyword in ("neighbors", "cap_multiplier", "absolute_cap"):
        arguments = dict(
            long_tail="none",
            loss="arcface",
            pseudo_label_from="",
            similarity_floor=0.5,
            min_gap=0.1,
            neighbors=5,
            cap_multiplier=1,
            absolute_cap=1,
        )
        arguments[keyword] = 0
        with pytest.raises(ValueError):
            module.validate_pseudo_options(**arguments)

    # CAM and extract choices reuse the canonical sets
    module.validate_cam_options(
        cam_method="eigencam",
        arch="vit",
        fig_format="pdf",
        save_npy="normalized",
        eval_transform="whole-specimen-pad",
        device="mps",
    )
    with pytest.raises(ValueError):
        module.validate_cam_options(
            cam_method="GradCAM",
            arch="vit",
            fig_format="png",
            save_npy="none",
            eval_transform="center-crop",
            device="auto",
        )
    module.validate_eval_transform("center-crop")
    with pytest.raises(ValueError):
        module.validate_eval_transform("Center-Crop")

    # augmentation and orientation sets
    module.validate_augmentation_profile("global-barcode", stage="pretrain")
    module.validate_augmentation_profile("conservative", stage="finetune")
    with pytest.raises(ValueError):
        module.validate_augmentation_profile("conservative", stage="pretrain")
    with pytest.raises(ValueError):
        module.validate_augmentation_profile("global-barcode", stage="finetune")
    module.validate_orientation_policy("sensitive")
    with pytest.raises(ValueError):
        module.validate_orientation_policy("Sensitive")

    # cluster's string-valued booleans keep the runtime's accepted spellings
    assert module.validate_bool_literal("pca-whitening", "YES") is True
    assert module.validate_bool_literal("pca-whitening", "0") is False
    with pytest.raises(ValueError):
        module.validate_bool_literal("pca-whitening", "maybe")


def test_shared_restrictions_match_canonical_sources() -> None:
    from otuformer import constants as canonical

    # canonical sets used by the option declarations and the schema export
    assert canonical.LOSS_CHOICES == (
        "arcface",
        "subcenter-arcface",
        "subcenter-arcface-compact",
        "supcon",
    )
    assert canonical.TOKEN_MODES == ("cls", "patch-topk", "attention-pool")

    # heavy canonical sources agree, without importing them into the schema path
    from otuformer.training.loss import LOSS_REGISTRY
    from otuformer.training.dataset import (
        EVAL_TRANSFORMS,
        FINETUNE_AUGMENTATIONS,
        ORIENTATION_POLICIES,
        PRETRAIN_AUGMENTATIONS,
    )
    from otuformer.embedding.extractor import ATTENTION_POOLING_TYPES
    from otuformer.cli.cam import _CAM_METHOD_CHOICES, _DEVICE_CHOICES

    assert set(canonical.LOSS_CHOICES) == set(LOSS_REGISTRY)
    assert set(canonical.PRETRAIN_AUGMENTATIONS) == set(PRETRAIN_AUGMENTATIONS)
    assert set(canonical.FINETUNE_AUGMENTATIONS) == set(FINETUNE_AUGMENTATIONS)
    assert set(canonical.ORIENTATION_POLICIES) == set(ORIENTATION_POLICIES)
    assert set(canonical.EVAL_TRANSFORM_CHOICES) == set(EVAL_TRANSFORMS)
    assert set(canonical.ATTENTION_POOLING_TYPES) == set(ATTENTION_POOLING_TYPES)
    assert set(canonical.CAM_METHOD_CHOICES) == set(_CAM_METHOD_CHOICES)
    assert set(canonical.CAM_DEVICE_CHOICES) == set(_DEVICE_CHOICES)
    assert canonical.ARCFACE_FAMILY_LOSSES and set(canonical.ARCFACE_FAMILY_LOSSES) <= set(
        canonical.LOSS_CHOICES
    )


def test_conditional_validation_preserves_source_semantics() -> None:
    module = constraints()

    for values in (
        {},
        {"assignments": "a.csv", "otu_table_csv": "b.csv"},
    ):
        with pytest.raises(ValueError):
            module.validate_constraints("diversity", values, explicit_options=frozenset())
    assert (
        module.validate_constraints(
            "diversity",
            {"assignments": "a.csv", "out_dir": "runs/x"},
            explicit_options=frozenset(),
        )
        == []
    )
    legacy = module.validate_constraints(
        "diversity",
        {
            "assignments": "a.csv",
            "embeddings": "e.csv",
            "tree": "t.nwk",
            "out_dir": "runs/x",
        },
        explicit_options=frozenset(),
    )
    assert any("precedence" in note for note in legacy), legacy

    with pytest.raises(ValueError):
        module.validate_constraints("extract", {}, explicit_options=frozenset())
    unresolved_extract = module.validate_constraints(
        "extract",
        {"checkpoint": "best.pt", "out_dir": "runs/x"},
        explicit_options=frozenset(),
    )
    assert any("verified at execution" in note for note in unresolved_extract), (
        unresolved_extract
    )
    both = module.validate_constraints(
        "extract",
        {"checkpoint": "best.pt", "onnx_path": "model.onnx", "out_dir": "runs/x"},
        explicit_options=frozenset(),
    )
    assert any("precedence" in note for note in both), both
    with pytest.raises(ValueError):
        module.validate_constraints(
            "extract",
            {"onnx_path": "model.onnx", "token_mode": "patch-topk", "out_dir": "runs/x"},
            explicit_options=frozenset({"token_mode"}),
        )

    for command in ("pretrain", "finetune"):
        with pytest.raises(ValueError):
            module.validate_constraints(
                command,
                {"resume": "runs/a/ckpt.pth", "overwrite": True, "out_dir": "runs/a"},
                explicit_options=frozenset({"resume", "overwrite"}),
            )
    # resume/overwrite is a training-command rule, and is not a cluster option
    with pytest.raises(ValueError):
        module.validate_constraints(
            "cluster", {"resume": "x"}, explicit_options=frozenset({"resume"})
        )

    # checkpoint-derived settings stay unresolved rather than guessed
    unresolved = module.validate_constraints(
        "finetune",
        {"checkpoint": "best.pt", "out_dir": "runs/x"},
        explicit_options=frozenset(),
    )
    assert any("verified at execution" in note for note in unresolved), unresolved
    # ...and a fresh run really does use the declared default loss
    with pytest.raises(ValueError):
        cli_schema().validate_parameters(
            "finetune",
            {
                "checkpoint": "c.pth",
                "train_data": "t.csv",
                "input_images_dir": "i",
                "out_dir": "runs/a",
                "subcenters": 4,
            },
        )
    # supcon temperature is bounded and applies only to SupCon
    with pytest.raises(ValueError):
        cli_schema().validate_parameters(
            "finetune",
            {
                "checkpoint": "c.pth",
                "train_data": "t.csv",
                "input_images_dir": "i",
                "out_dir": "runs/a",
                "loss": "supcon",
                "supcon_temperature": -1,
            },
        )
    with pytest.raises(ValueError):
        cli_schema().validate_parameters(
            "finetune",
            {
                "checkpoint": "c.pth",
                "train_data": "t.csv",
                "input_images_dir": "i",
                "out_dir": "runs/a",
                "loss": "arcface",
                "supcon_temperature": 0.07,
            },
        )
    # a resume keeps its checkpoint loss, so an explicit sub-center run is allowed
    resumed = cli_schema().validate_parameters(
        "finetune",
        {
            "checkpoint": "c.pth",
            "train_data": "t.csv",
            "input_images_dir": "i",
            "resume": "runs/a/x.pth",
            "out_dir": "runs/a",
            "subcenters": 4,
        },
    )
    assert any("inherited from the resumed" in note for note in resumed["unresolved"])

    # an omitted loss is inherited, so loss-specific options are not rejected
    inherited_loss = module.validate_constraints(
        "finetune",
        {"checkpoint": "best.pt", "subcenters": 4, "out_dir": "runs/x"},
        explicit_options=frozenset({"subcenters"}),
    )
    assert any("applicability depends on the effective loss" in note for note in inherited_loss), (
        inherited_loss
    )
    assert any("effective --loss is not fixed" in note for note in inherited_loss), inherited_loss
    # ...but an explicit conflicting loss still fails
    with pytest.raises(ValueError):
        module.validate_constraints(
            "finetune",
            {"loss": "arcface", "subcenters": 4, "out_dir": "runs/x"},
            explicit_options=frozenset({"loss", "subcenters"}),
        )

    # omitted loss-specific options are not reported as applicability errors
    arcface = module.validate_constraints(
        "finetune",
        {"checkpoint": "best.pt", "loss": "arcface", "out_dir": "runs/x"},
        explicit_options=frozenset(),
    )
    assert not any("does not apply" in note for note in arcface), arcface
    subcenter_unresolved = module.validate_constraints(
        "finetune",
        {"checkpoint": "best.pt", "loss": "subcenter-arcface", "out_dir": "runs/x"},
        explicit_options=frozenset({"loss"}),
    )
    assert any("subcenters defaults" in note for note in subcenter_unresolved), (
        subcenter_unresolved
    )
    with pytest.raises(ValueError):
        module.validate_constraints(
            "finetune",
            {"loss": "arcface", "subcenters": 3, "out_dir": "runs/x"},
            explicit_options=frozenset({"loss", "subcenters"}),
        )
    with pytest.raises(ValueError):
        module.validate_constraints(
            "finetune",
            {"loss": "arcface", "compact_weight": 0.2, "out_dir": "runs/x"},
            explicit_options=frozenset({"loss", "compact_weight"}),
        )
    with pytest.raises(ValueError):
        module.validate_constraints(
            "finetune",
            {"loss": "supcon", "long_tail": "cb-drw", "out_dir": "runs/x"},
            explicit_options=frozenset({"long_tail"}),
        )
    # pseudo rule options follow the CLI wrapper: they are rejected, not ignored
    with pytest.raises(ValueError):
        module.validate_constraints(
            "finetune",
            {"loss": "arcface", "pseudo_similarity_floor": 0.5, "out_dir": "runs/x"},
            explicit_options=frozenset({"pseudo_similarity_floor"}),
        )
    with pytest.raises(ValueError):
        module.validate_constraints(
            "finetune",
            {"resume": "runs/a/x.pth", "pseudo_neighbors": 3, "out_dir": "runs/a"},
            explicit_options=frozenset({"pseudo_neighbors"}),
        )



def test_metadata_does_not_invent_enums() -> None:
    module = constraints()
    schema = runtime_schema()
    entries = {
        name: {item["name"]: item for item in entry["parameters"]}
        for name, entry in schema["commands"].items()
    }

    # free-form names and device spellings stay free-form
    for name in ("model_name", "seed", "lr"):
        assert entries["pretrain"][name]["restriction_source"] == "unavailable"
        assert entries["pretrain"][name]["shared_choices"] is None
    assert module.validate_constraints(
        "pretrain",
        {"model_name": "my-custom-backbone", "device": "cuda:1"},
        explicit_options=frozenset({"model_name", "device"}),
    )
    assert entries["pretrain"]["device"]["restriction_source"] == "unavailable"
    assert entries["pretrain"]["device"]["shared_choices"] is None
    assert any(
        "core device resolver" in note for note in entries["pretrain"]["device"]["unresolved"]
    )

    # cluster keeps accepting runtime-portable strings and reports it as unresolved
    assert module.validate_constraints(
        "cluster",
        {"distance": "braycurtis", "out_dir": "runs/x"},
        explicit_options=frozenset({"distance"}),
    )
    assert entries["cluster"]["distance"]["restriction_source"] == "unavailable"
    assert entries["cluster"]["local_k_strategy"]["restriction_source"] == "parser"

    # shared values are labelled as shared, not as parser-declared
    assert entries["pretrain"]["register_tokens"]["restriction_source"] == "shared-validator"
    assert entries["pretrain"]["register_tokens"]["shared_choices"] == ["none", "0", "4"]
    assert entries["pretrain"]["mask_ratio"]["shared_bounds"] == {
        "min": 0.0,
        "max": 1.0,
        "exclusive": True,
    }
    assert entries["finetune"]["subcenters"]["shared_bounds"] == {"min": 2, "max": 8}
    assert entries["extract"]["token_mode"]["shared_choices"] == [
        "cls",
        "patch-topk",
        "attention-pool",
    ]
    assert entries["cam"]["fig_format"]["shared_choices"] == ["png", "jpg", "pdf"]

    # path roles classify string-valued paths too
    assert entries["pretrain"]["resume"]["path_role"] == "resume_checkpoint"
    assert entries["finetune"]["pseudo_label_from"]["path_role"] == "pseudo_source"
    assert entries["extract"]["onnx_path"]["path_role"] == "onnx_model"
    assert entries["pretrain"]["out_dir"]["path_role"] == "output_dir"
    assert entries["pretrain"]["train_data"]["path_role"] == "label_csv"


# ------------------------------------------------- Task 3: proposals, argv, safety

ADAPTER = ROOT / "skills" / "otuformer-workflow" / "scripts" / "export_cli_schema.py"


def run_adapter(arguments: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ADAPTER), *arguments],
        capture_output=True,
        text=True,
        cwd=str(cwd or ROOT),
    )


def test_normalization_rejects_bad_proposals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Loading the adapter for its duplicate-key hook must not leave bytecode
    # caches inside the distributed Skill directory.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    module = cli_schema()

    with pytest.raises(ValueError):
        module.validate_parameters("pretrain", {"not_a_parameter": 1})
    with pytest.raises(ValueError):
        module.validate_parameters("pretrain", {"--input-images-dir": "x", "out_dir": "y", "seed": None})
    with pytest.raises(ValueError):
        module.validate_parameters("pretrain", {"out_dir": "runs/x"})
    with pytest.raises(ValueError):
        module.validate_parameters(
            "pretrain", {"input_images_dir": "i", "out_dir": "runs/x", "seed": "many"}
        )
    with pytest.raises(ValueError):
        module.validate_parameters(
            "pretrain", {"input_images_dir": "i", "out_dir": "runs/x", "overwrite": "yes"}
        )
    with pytest.raises(ValueError):
        module.validate_parameters(
            "pretrain",
            {"input_images_dir": "i", "out_dir": "runs/x", "patch_loss": "nope"},
        )
    with pytest.raises(ValueError):
        module.validate_parameters(
            "pretrain", {"input_images_dir": "i", "out_dir": "runs/x", "mask_ratio": 1.5}
        )
    with pytest.raises(ValueError):
        module.validate_parameters(
            "pretrain", {"input_images_dir": "i", "out_dir": "runs/x", "overwrite": False}
        )

    # aliases: equal assignments accepted once, contradictory ones rejected
    equal = module.validate_parameters(
        "cluster", {"embeddings": "e.csv", "out_dir": "runs/x", "--labels": "l.csv", "label_csv": "l.csv"}
    )
    assert equal["explicit_options"].count("label_csv") == 1
    with pytest.raises(ValueError):
        module.validate_parameters(
            "cluster", {"embeddings": "e.csv", "out_dir": "runs/x", "--labels": "a.csv", "label_csv": "b.csv"}
        )
    short = module.validate_parameters("update", {"-y": True, "check": True})
    assert short["explicit_options"] == ["check", "yes"]

    # cluster's string-valued booleans are text options, not flags
    text_bool = module.validate_parameters(
        "cluster", {"embeddings": "e.csv", "out_dir": "runs/x", "pca_whitening": "true"}
    )
    assert text_bool["explicit_values"]["pca_whitening"] == "true"
    with pytest.raises(ValueError):
        module.validate_parameters(
            "cluster", {"embeddings": "e.csv", "out_dir": "runs/x", "pca_whitening": "maybe"}
        )

    # failing validation never creates the output directory
    target = tmp_path / "never-created"
    with pytest.raises(ValueError):
        module.validate_parameters(
            "pretrain", {"input_images_dir": "i", "out_dir": str(target), "patch_loss": "nope"}
        )
    assert not target.exists()

    # duplicate JSON keys are rejected through the stdlib hook
    spec = importlib.util.spec_from_file_location("skill_export_cli_schema", ADAPTER)
    assert spec and spec.loader
    module_adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_adapter)
    with pytest.raises(ValueError):
        module_adapter._reject_duplicate_keys([("a", 1), ("a", 2)])
    duplicate = tmp_path / "dup.json"
    duplicate.write_text('{"input_images_dir": "i", "input_images_dir": "j"}', encoding="utf-8")
    completed = run_adapter(
        ["--command", "pretrain", "--inputs-json", str(duplicate)], cwd=tmp_path
    )
    assert completed.returncode == 2, completed.stdout
    assert "duplicate key" in completed.stderr


def test_card_covers_schema_once_and_argv_preserves_omission() -> None:
    module = cli_schema()
    for command, explicit in (
        ("pretrain", {"input_images_dir": "i", "out_dir": "runs/a"}),
        ("cam", {"checkpoint": "c.pth", "images_dir": "i", "out_dir": "runs/a"}),
        ("update", {"check": True}),
    ):
        entry = module.export_cli_schema(command)["commands"][command]
        card = module.render_parameter_card(command, explicit)
        for item in entry["parameters"]:
            spellings = item["opts"] + item["secondary_opts"]
            assert any(spelling in card for spelling in spellings), (command, spellings)
        assert card.count("   meaning:") == len(entry["parameters"]) == id_count(entry)
        assert "provenance: explicit" in card
        assert f"Parameters ({len(entry['parameters'])} of {len(entry['parameters'])} shown)" in card
        assert "Output safety conditions:" in card

    finetune_entry = runtime_schema()["commands"]["finetune"]["parameters"]
    subcenters = next(item for item in finetune_entry if item["name"] == "subcenters")
    assert subcenters["shared_bounds"] == {"min": 2, "max": 8}
    finetune_card = module.render_parameter_card(
        "finetune",
        {"checkpoint": "c.pth", "train_data": "t.csv", "input_images_dir": "i", "out_dir": "runs/a"},
    )
    assert "   restrictions:" in finetune_card
    assert "min" in finetune_card and "2" in finetune_card and "8" in finetune_card
    masked_card = module.render_parameter_card(
        "pretrain", {"input_images_dir": "i", "out_dir": "runs/a"}
    )
    assert "keyword values: auto" in masked_card

    argv = module.build_argv(
        "pretrain", {"input_images_dir": "i", "out_dir": "runs/a"}, executable=["otuformer"]
    )
    assert "--max-epochs" not in argv
    assert "--input-images-dir" in argv and "--out-dir" in argv

    # an explicit value equal to the declared default is still passed explicitly
    same_as_default = module.build_argv(
        "pretrain",
        {"input_images_dir": "i", "out_dir": "runs/a", "max_epochs": 50},
        executable=["otuformer"],
    )
    assert same_as_default[-2:] == ["--max-epochs", "50"]

    # finetune resume marks omitted experiment settings as inherited, not confirmed
    validation = module.validate_parameters(
        "finetune",
        {
            "checkpoint": "c.pth",
            "train_data": "t.csv",
            "input_images_dir": "i",
            "resume": "runs/a/ckpt.pth",
            "out_dir": "runs/a",
        },
    )
    assert validation["provenance"]["loss"] == "omitted-inherited"
    assert validation["provenance"]["device"] == "omitted-default"
    assert any("inherited from the resumed" in note for note in validation["unresolved"])
    pseudo_round = module.validate_parameters(
        "finetune",
        {
            "checkpoint": "c.pth",
            "train_data": "t.csv",
            "input_images_dir": "i",
            "pseudo_label_from": "runs/r1/finetune_latest.pth",
            "out_dir": "runs/x",
        },
    )
    assert pseudo_round["provenance"]["loss"] == "omitted-inherited"


def id_count(entry: dict) -> int:
    return len({item["name"] for item in entry["parameters"]})


def test_argv_round_trips_without_callbacks() -> None:
    """The framework's own parameter sources must see only intended overrides."""
    module = cli_schema()

    sub = typer.Typer()

    @sub.callback(invoke_without_command=True)
    def demo(
        ctx: typer.Context,
        images: str = typer.Option(..., "--images", "--input-images-dir"),
        out_dir: str = typer.Option("runs/demo", "--out-dir"),
        epochs: int = typer.Option(50, "--epochs"),
        flag: bool = typer.Option(False, "--flag/--no-flag"),
    ) -> None:  # pragma: no cover - parsing only, never invoked
        raise AssertionError("callback must not run")

    root = typer.Typer(no_args_is_help=True)
    root.add_typer(sub, name="demo")
    local_app = root

    explicit = {"images": "a b/c'd\u00e9.png", "out_dir": "runs/a", "epochs": 3}
    argv = module.build_argv("pretrain", {"input_images_dir": "x", "out_dir": "runs/a"}, executable=["otuformer"])
    assert argv[:2] == ["otuformer", "pretrain"]

    command = get_command(local_app).commands["demo"]
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        local_app,
        ["demo", "--images", "a b/c'd\u00e9;rm -rf.png", "--out-dir", "runs/a", "--epochs", "3", "--flag"],
    )
    assert result.exit_code != 0  # the callback asserts; parsing must have succeeded

    context = command.make_context("demo", ["--images", "x.png", "--epochs", "3"], resilient_parsing=False)
    sources = {name: context.get_parameter_source(name).name for name in ("images", "out_dir", "epochs", "flag")}
    assert sources["images"] == "COMMANDLINE"
    assert sources["epochs"] == "COMMANDLINE"
    assert sources["out_dir"] == "DEFAULT"
    assert sources["flag"] == "DEFAULT"

    # explicit false is representable through a secondary spelling
    assert module._coerce(
        {"is_flag": True, "secondary_opts": ["--no-flag"], "opts": ["--flag"], "type": "boolean"}, False, "--flag"
    ) is False
    # unrepresentable explicit false fails instead of disappearing
    with pytest.raises(ValueError):
        module._coerce({"is_flag": True, "secondary_opts": [], "opts": ["--overwrite"], "type": "boolean"}, False, "--overwrite")

    # argument-safe transport: an argv list, with a POSIX review string only for display
    awkward = "spaces 'quotes' $(danger) \u00e9"
    proposal = module.build_proposal(
        "cluster", {"embeddings": awkward, "out_dir": "runs/a"}, executable=["otuformer"]
    )
    assert proposal["argv"][3] == awkward
    assert "'spaces" in proposal["review_command"] or "spaces" in proposal["review_command"]
    assert explicit  # keep the awkward-path fixture meaningful


def test_output_checks_are_read_only(tmp_path: Path) -> None:
    module = cli_schema()

    protected_dir = tmp_path / "raw"
    images = protected_dir / "images"
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    output_root = tmp_path / "runs" / "run001"
    output_root.mkdir(parents=True)

    with pytest.raises(ValueError):
        module.check_output_safety(images, protected_paths=[protected_dir])
    with pytest.raises(ValueError):
        module.check_output_safety(protected_dir, protected_paths=[images])
    with pytest.raises(ValueError):
        module.check_output_safety(tmp_path, protected_paths=[source_dir])

    # "link/.." means the parent of the link *target*, not of the link itself
    (tmp_path / "data" / "sub").mkdir(parents=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "link").symlink_to(tmp_path / "data" / "sub")
    link_traversal = tmp_path / "work" / "link" / ".." / "raw"
    with pytest.raises(ValueError):
        module.check_output_safety(link_traversal, protected_paths=[tmp_path / "data" / "raw"])

    # a symlink reappearing inside the non-existent tail is resolved too
    tail_link = tmp_path / "work" / "missing" / ".." / "link" / "raw"
    with pytest.raises(ValueError):
        module.check_output_safety(tail_link, protected_paths=[tmp_path / "data" / "sub" / "raw"])
    assert module._resolve_for_safety(tail_link) == (tmp_path / "data" / "sub" / "raw").resolve()

    # a non-existent component followed by ".." cannot smuggle a protected tree
    traversal = tmp_path / "missing" / ".." / "raw"
    with pytest.raises(ValueError):
        module.check_output_safety(traversal, protected_paths=[protected_dir])
    traversal_ok = tmp_path / "missing" / ".." / "runs"
    assert module.check_output_safety(traversal_ok) is not None

    # a symlink alias of a protected input is still rejected
    alias = tmp_path / "alias"
    alias.symlink_to(protected_dir)
    with pytest.raises(ValueError):
        module.check_output_safety(alias / "out", protected_paths=[protected_dir])

    # non-empty outputs need a fresh path, resume, or explicit overwrite consent
    existing = output_root / "pretrain"
    existing.mkdir()
    (existing / "SSL_latest.pth").write_bytes(b"ckpt")
    conditions = module.check_output_safety(existing)
    assert any("non-empty existing directory" in note for note in conditions)
    overwrite_notes = module.check_output_safety(existing, overwrite=True)
    assert any("overwrite approval" in note for note in overwrite_notes)
    # overwrite never overrides a protected-path rejection
    with pytest.raises(ValueError):
        module.check_output_safety(
            output_root, overwrite=True, protected_paths=[existing / "SSL_latest.pth"]
        )
    # a training resume may reuse its approved existing directory read-only
    resume_notes = module.check_output_safety(existing, resume=True)
    assert any("read-only" in note for note in resume_notes)

    # a resume checkpoint inside that directory is allowed without overwrite...
    checkpoint = existing / "finetune_latest.pth"
    allowed = module.check_output_safety(
        existing,
        protected_paths=[checkpoint],
        resume=True,
        allow_inside=[checkpoint],
    )
    assert any("reused read-only" in note for note in allowed), allowed
    # ...and still refused when an overwrite would delete it
    with pytest.raises(ValueError):
        module.check_output_safety(
            existing,
            protected_paths=[checkpoint],
            overwrite=True,
            allow_inside=[checkpoint],
        )
    resume_proposal = module.build_proposal(
        "finetune",
        {
            "checkpoint": "c.pth",
            "train_data": "t.csv",
            "input_images_dir": "i",
            "resume": str(checkpoint),
            "out_dir": str(existing),
        },
        executable=["otuformer"],
    )
    assert any("reused read-only" in note for note in resume_proposal["safety_conditions"])

    # a real input inside a would-be output is still refused, even for training
    labels_inside = existing / "labels.csv"
    labels_inside.write_text("image,label\n", encoding="utf-8")
    with pytest.raises(ValueError):
        module.build_proposal(
            "finetune",
            {
                "checkpoint": "c.pth",
                "train_data": str(labels_inside),
                "input_images_dir": "i",
                "out_dir": str(existing),
                "overwrite": True,
            },
            executable=["otuformer"],
        )

    before = sorted(path.name for path in existing.iterdir())
    module.check_output_safety(tmp_path / "runs" / "fresh")
    with pytest.raises(ValueError):
        module.check_output_safety(images, protected_paths=[protected_dir])
    assert sorted(path.name for path in existing.iterdir()) == before
    assert not (tmp_path / "runs" / "fresh").exists()

    # analysis proposals must name --out-dir explicitly even though the parser defaults one
    with pytest.raises(ValueError):
        module.build_proposal("cluster", {"embeddings": "e.csv"}, executable=["otuformer"])
    with pytest.raises(ValueError):
        module.build_proposal("pretrain", {"input_images_dir": "i", "out_dir": "runs/a", "overwrite": True, "resume": "runs/a/x.pth"})


def test_adapter_source_and_environment_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = cli_schema()

    # schema-only mode from a non-repository working directory reports what loaded
    completed = run_adapter(["--command", "doctor"], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    runtime = payload["runtime"]
    assert runtime["package_version"]
    assert Path(runtime["package_path"]).is_file()
    assert runtime["interpreter"]
    assert set(payload["commands"]) == {"doctor"}

    # the adapter is read-only and never executes an analysis
    proposal_inputs = tmp_path / "inputs.json"
    proposal_inputs.write_text(
        json.dumps({"input_images_dir": "examples/Epidorcus/images", "out_dir": "runs/never-run"}),
        encoding="utf-8",
    )
    launcher = shutil.which("otuformer")
    if launcher is None:
        pytest.skip("no installed otuformer console script in this environment")
    completed = run_adapter(
        [
            "--command",
            "pretrain",
            "--inputs-json",
            str(proposal_inputs),
            "--executable",
            launcher,
        ],
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["operation"] == "analysis"
    assert payload["argv"][1] == "pretrain"
    assert not (tmp_path / "runs" / "never-run").exists()
    assert not (ROOT / "runs" / "never-run").exists()

    # a copied Skill still resolves helpers and the installed package
    copied = tmp_path / "copied-skill"
    shutil.copytree(ROOT / "skills" / "otuformer-workflow", copied)
    completed = subprocess.run(
        [sys.executable, str(copied / "scripts" / "export_cli_schema.py"), "--command", "update"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert "package_version" in json.loads(completed.stdout)["runtime"]

    # a symlinked Skill resolves the same way
    linked = tmp_path / "linked-skill"
    linked.symlink_to(ROOT / "skills" / "otuformer-workflow")
    completed = subprocess.run(
        [sys.executable, str(linked / "scripts" / "export_cli_schema.py"), "--command", "update"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr

    # launcher verification blocks mismatched and uninspectable launchers
    runtime = module.runtime_identity()
    fake_python = tmp_path / "python3"
    fake_python.write_text("#!/nonexistent/interpreter\n", encoding="utf-8")
    fake_python.chmod(0o755)
    with pytest.raises(ValueError):
        module.verify_launcher([str(fake_python)], source_runtime=runtime)
    uninspectable = tmp_path / "launcher.bin"
    uninspectable.write_bytes(b"\x7fELF\x02\x01\x01\x00binary")
    uninspectable.chmod(0o755)
    with pytest.raises(ValueError):
        module.verify_launcher([str(uninspectable)], source_runtime=runtime)
    with pytest.raises(ValueError):
        module.verify_launcher([str(tmp_path / "missing-launcher")], source_runtime=runtime)

    # a plain interpreter is refused when no module entry point exists
    with pytest.raises(ValueError):
        module.verify_launcher([sys.executable], source_runtime=runtime)

    # the same version from a different installation is refused
    with pytest.raises(ValueError):
        module.verify_launcher(
            ["otuformer"],
            source_runtime={**runtime, "package_path": str(tmp_path / "other" / "__init__.py")},
        )
    # a substring look-alike version no longer passes
    with pytest.raises(ValueError):
        module.verify_launcher(["otuformer"], source_runtime={**runtime, "package_version": "0.1"})

    # a virtual-environment interpreter must be probed through its own path;
    # the module entry point is faked because this distribution has none
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(Path(sys.executable).resolve())
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    _, probe_python, _, _ = module._launch_prefix_and_interpreter(
        [str(venv_bin / "python"), "-m", "otuformer"],
        {**module.runtime_identity(), "interpreter": str(venv_bin / "python")},
    )
    assert probe_python == venv_bin / "python", probe_python
    # without a module entry point the interpreter form is refused
    monkeypatch.undo()
    with pytest.raises(ValueError):
        module._launch_prefix_and_interpreter(
            [str(venv_bin / "python"), "-m", "otuformer"],
            {**module.runtime_identity(), "interpreter": str(venv_bin / "python")},
        )

    # only the standard console-script structure counts: a mention of the module
    # in a comment or string, or any extra statement, is refused
    comment_only = tmp_path / "comment-only"
    comment_only.write_text(
        f"#!{sys.executable}\n# otuformer.cli.main\n"
        "print('otuformer 0.11.0')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        module._verify_console_script(comment_only, "otuformer.cli.main", "app")

    redirecting = tmp_path / "redirecting"
    redirecting.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, '/elsewhere/src')\n"
        "from otuformer.cli.main import app\napp()\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        module._verify_console_script(redirecting, "otuformer.cli.main", "app")

    well_formed = tmp_path / "well-formed"
    well_formed.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "from otuformer.cli.main import app\n"
        "if __name__ == '__main__':\n"
        "    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
        "    sys.exit(app())\n",
        encoding="utf-8",
    )
    module._verify_console_script(well_formed, "otuformer.cli.main", "app")
    with pytest.raises(ValueError):
        module._verify_console_script(well_formed, "otuformer.cli.other", "app")

    # version-like output alone is not evidence of a working launcher
    printer = tmp_path / "otuformer-printer"
    printer.write_text(
        f"#!{sys.executable}\nprint('otuformer 0.11.0')\n", encoding="utf-8"
    )
    printer.chmod(0o755)
    with pytest.raises(ValueError):
        module.verify_launcher([str(printer)])

    # a script that merely behaves like the entry point is not the installed
    # entry point, so it cannot be tied to this installation
    mimic = tmp_path / "otuformer-mimic"
    mimic.write_text(
        f"#!{sys.executable}\nfrom otuformer.cli.main import app\napp()\n",
        encoding="utf-8",
    )
    mimic.chmod(0o755)
    with pytest.raises(ValueError):
        module.verify_launcher([str(mimic)])
    with pytest.raises(ValueError):
        module.verify_launcher([str(mimic), "-c", "print('anything')"])
    with pytest.raises(ValueError):
        module.verify_launcher([str(mimic), "extra", "arguments"])

    # echoed command names are not a command surface, and neither is text that
    # merely embeds the names
    names = " ".join(runtime_schema()["commands"])
    echoer = tmp_path / "otuformer-echo"
    echoer.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"print({names!r} if '--help' in sys.argv else 'otuformer 0.11.0')\n",
        encoding="utf-8",
    )
    echoer.chmod(0o755)
    # echoing the command names satisfies the compatibility check (help text is
    # not provenance)...
    module._verify_cli_surface([str(echoer)])
    # ...but it is still refused as a launcher, because it is not the installed
    # entry point of this distribution
    with pytest.raises(ValueError):
        module.verify_launcher([str(echoer)])
    negated = tmp_path / "otuformer-negated"
    negated.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "print(' '.join('not_' + n + '_command' for n in "
        f"{names!r}.split()) + ' invented-command')\n",
        encoding="utf-8",
    )
    negated.chmod(0o755)
    with pytest.raises(ValueError):
        module._verify_cli_surface([str(negated)])

    # env shebang options cannot be reproduced, so such a launcher is refused
    env_launcher = tmp_path / "otuformer-env"
    env_launcher.write_text(
        "#!/usr/bin/env -S -i python -S\nprint('otuformer 0.11.0')\n", encoding="utf-8"
    )
    env_launcher.chmod(0o755)
    with pytest.raises(ValueError):
        module.verify_launcher([str(env_launcher)])

    # shebang interpreter flags are honoured instead of silently dropped
    flagged = tmp_path / "otuformer-flagged"
    flagged.write_text(
        f"#!{sys.executable} -S\nprint('otuformer 0.11.0')\n", encoding="utf-8"
    )
    flagged.chmod(0o755)
    with pytest.raises(ValueError):
        module.verify_launcher([str(flagged)])

    # a version mismatch is reported, never silently accepted
    mismatched = tmp_path / "otuformer"
    mismatched.write_text(
        f"#!{sys.executable}\nimport sys\nprint('otuformer 99.0.0')\n", encoding="utf-8"
    )
    mismatched.chmod(0o755)
    with pytest.raises(ValueError):
        module.verify_launcher([str(mismatched)], source_runtime=runtime)

    # the real launcher verifies, and same-version/different-interpreter is blocked
    import shutil as _shutil

    real = _shutil.which("otuformer")
    if real is not None:
        assert module.verify_launcher([real])["verified"] is True
        with pytest.raises(ValueError):
            module.verify_launcher(
                [real], source_runtime={**runtime, "interpreter": str(tmp_path / "other-python")}
            )


def test_operation_classes() -> None:
    module = cli_schema()
    assert module.classify_operation("doctor", {}) == "read-only"
    assert module.classify_operation("update", {"check": True}) == "read-only-networked"
    assert module.classify_operation("update", {}) == "installation"
    assert module.classify_operation("pretrain", {"out_dir": "runs/a"}) == "analysis"
    doctor = module.build_proposal("doctor", {}, executable=["otuformer"])
    assert doctor["operation"] == "read-only"
    assert doctor["safety_conditions"] == []


# --------------------------------------------- audit follow-up: shared guards


def test_extract_size_guards_match_core_validators() -> None:
    module = cli_schema()
    base = {"checkpoint": "c.pth", "input_images_dir": "images", "out_dir": "runs/x"}
    for option in ("topk_patches", "attention_pooling_epochs"):
        with pytest.raises(ValueError):
            module.validate_parameters("extract", {**base, option: 0})
        assert (
            module.validate_parameters("extract", {**base, option: 1})["values"][option]
            == 1
        )


def test_pretrain_patch_applicability_matches_core_rules() -> None:
    module = cli_schema()
    base = {"train_data": "e.csv", "input_images_dir": "images", "out_dir": "runs/x"}
    for extra in (
        {"masking_strategy": "hybrid"},
        {"ibot_prototypes": 10},
        {"patch_loss": "none", "mask_ratio": "0.3"},
        {"patch_loss": "none", "masking_strategy": "blockwise"},
        {"patch_loss": "none", "lambda_mask": 0.5},
        {"patch_loss": "masked-feature", "ibot_prototypes": 10},
    ):
        with pytest.raises(ValueError):
            module.validate_parameters("pretrain", {**base, **extra})
    # a resumed run defers to the checkpoint's saved patch mode
    resumed = module.validate_parameters(
        "pretrain", {**base, "resume": "runs/x/SSL_latest.pth", "masking_strategy": "hybrid"}
    )
    assert any("saved patch mode" in note for note in resumed["unresolved"])


def test_pretrain_resume_provenance_is_not_blanket() -> None:
    module = cli_schema()
    validation = module.validate_parameters(
        "pretrain",
        {
            "train_data": "e.csv",
            "input_images_dir": "images",
            "out_dir": "runs/x",
            "resume": "runs/x/SSL_latest.pth",
        },
    )
    provenance = validation["provenance"]
    for name in (
        "device",
        "num_workers",
        "cpus",
        "batch_size",
        "max_epochs",
    ):
        assert provenance[name] == "omitted-default", name
    assert provenance["train_data"] == "explicit"
    for name in ("patch_loss", "augmentation", "orientation_policy", "weight_decay"):
        assert provenance[name] == "omitted-inherited", name


def test_default_protected_paths_cover_source_and_examples() -> None:
    module = cli_schema()
    package_dir = Path(module.runtime_identity()["package_dir"])
    protected = module.default_protected_paths()
    assert package_dir in protected
    checkout_src = package_dir.parent
    if checkout_src.name == "src":
        assert checkout_src in protected
    checkout_examples = package_dir.parent.parent / "examples"
    if checkout_examples.is_dir():
        assert checkout_examples in protected

    base = {"train_data": "e.csv", "input_images_dir": "images", "out_dir": "runs/x"}
    rejected = [package_dir, package_dir.parent, ROOT]
    if checkout_src.name == "src":
        rejected.append(checkout_src / "_audit_scratch")
    for out_dir in rejected:
        with pytest.raises(ValueError):
            module.build_proposal(
                "pretrain",
                {**base, "out_dir": str(out_dir), "overwrite": True},
                executable=["otuformer"],
            )
    # the documented runs/ output root inside the checkout stays usable
    runs_proposal = module.build_proposal(
        "pretrain",
        {**base, "out_dir": str(ROOT / "runs" / "_audit_demo")},
        executable=["otuformer"],
    )
    assert runs_proposal["safety_conditions"]


def test_schema_supports_positional_range_and_path_types(tmp_path: Path) -> None:
    module = cli_schema()
    demo_app = typer.Typer()

    @demo_app.command()
    def demo(
        name: str,
        count: int = typer.Option(1, "--count", min=1, max=10),
        out: Path = typer.Option(".", "--out", file_okay=False),
        src: Path = typer.Option(..., "--src", exists=True),
    ) -> None:  # pragma: no cover - never invoked
        raise AssertionError("callback must not run")

    command = get_command(demo_app)
    entries = {
        item["name"]: item
        for item in (module.serialize_parameter(p) for p in analysis_params(command))
    }
    assert entries["name"]["kind"] == "positional"
    assert entries["name"]["opts"] == ["name"]
    assert entries["count"]["kind"] == "option"
    assert entries["count"]["type"] == "integer range"
    assert entries["count"]["bounds"] == {"min": 1, "max": 10}
    assert entries["out"]["type"] == "directory"
    assert entries["out"]["path_constraints"]["file_okay"] is False

    assert module._coerce(entries["count"], 5, "--count") == 5
    with pytest.raises(ValueError):
        module._coerce(entries["count"], 0, "--count")
    with pytest.raises(ValueError):
        module._coerce(entries["count"], 11, "--count")
    existing_dir = tmp_path / "d"
    existing_dir.mkdir()
    assert module._coerce(entries["out"], str(existing_dir), "--out") == str(existing_dir)
    a_file = tmp_path / "f.txt"
    a_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        module._coerce(entries["out"], str(a_file), "--out")
    with pytest.raises(ValueError):
        module._coerce(entries["src"], str(tmp_path / "missing"), "--src")


def test_build_argv_emits_positional_without_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cli_schema()
    entries = {
        "name": {
            "kind": "positional",
            "opts": ["name"],
            "secondary_opts": [],
            "is_flag": False,
        },
        "count": {
            "kind": "option",
            "opts": ["--count"],
            "secondary_opts": [],
            "is_flag": False,
        },
    }
    monkeypatch.setattr(module, "parameter_index", lambda command: entries)
    monkeypatch.setattr(
        module,
        "validate_parameters",
        lambda command, explicit: {
            "explicit_values": {"name": "a value", "count": 3}
        },
    )
    argv = module.build_argv("demo", {}, executable=["otuformer"])
    assert argv == ["otuformer", "demo", "a value", "--count", "3"]


ADAPTER = ROOT / "skills" / "otuformer-workflow" / "scripts" / "export_cli_schema.py"


def load_adapter_module() -> Any:
    import importlib.util as importlib_util

    spec = importlib_util.spec_from_file_location("otuformer_skill_adapter", ADAPTER)
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_source_fallback_requires_canonical_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = load_adapter_module()
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "otuformer" else real_find_spec(name, *args, **kwargs)
        ),
    )

    # a copied Skill must not guess an unrelated ancestor repository
    copied = tmp_path / "custom" / "copied-skill"
    shutil.copytree(ROOT / "skills" / "otuformer-workflow", copied)
    monkeypatch.setattr(
        adapter, "__file__", str(copied / "scripts" / "export_cli_schema.py")
    )
    with pytest.raises(ValueError):
        adapter._ensure_importable(None)

    # the canonical checkout is still accepted
    monkeypatch.setattr(adapter, "__file__", str(ADAPTER))
    before = list(sys.path)
    try:
        adapter._ensure_importable(None)
        assert str(ROOT / "src") in sys.path
    finally:
        sys.path[:] = before

    # an explicit source root is trusted as given
    checkout = tmp_path / "checkout"
    (checkout / "src" / "otuformer").mkdir(parents=True)
    (checkout / "src" / "otuformer" / "__init__.py").write_text("", encoding="utf-8")
    (checkout / "pyproject.toml").write_text("", encoding="utf-8")
    before = list(sys.path)
    try:
        adapter._ensure_importable(checkout)
        assert str(checkout / "src") in sys.path
    finally:
        sys.path[:] = before
