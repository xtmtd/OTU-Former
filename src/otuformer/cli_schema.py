"""Read-only runtime CLI schema discovery for the OTU-Former workflow Skill.

The module deliberately stays import-light: it never runs analysis callbacks,
never initializes models or devices, never downloads weights, and does not import
the training stack. It also avoids importing the CLI framework directly so the
external-Click coupling guard (``tests/test_cli_smoke.py``) covers it too.

Serialization rules (workflow design Section 4):

* every declared parameter is exported, in declaration order;
* an unset required default is never presented as an apparent value;
* unknown or unsupported metadata shapes raise ``ValueError`` instead of being
  silently dropped or flattened into a misleading value.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
BUILTIN_HELP_OPTION = "--help"

# Root operational controls, classified separately from analysis parameters.
_CONTROL_ROLES: Mapping[str, str] = {
    "version": "version",
    "install_completion": "completion",
    "show_completion": "completion",
}

_CONTEXT_PARAMETER_NAMES = frozenset({"ctx", "context"})
_PARAMETER_TYPE_NAMES = frozenset(
    {
        "integer",
        "integer range",
        "float",
        "float range",
        "text",
        "boolean",
        "path",
        "directory",
        "choice",
    }
)
_PATH_CONSTRAINT_ATTRIBUTES = (
    "exists",
    "file_okay",
    "dir_okay",
    "writable",
    "readable",
)


# ------------------------------------------------------------------ discovery


def root_command() -> Any:
    """Return the converted root command object without running any callback."""
    from otuformer.cli.main import app
    from typer.main import get_command

    return get_command(app)


def runtime_identity() -> dict[str, str]:
    """Identify the package and interpreter actually loaded."""
    import otuformer

    package_file = Path(otuformer.__file__).resolve()
    return {
        "package_version": str(otuformer.__version__),
        "package_path": str(package_file),
        "package_dir": str(package_file.parent),
        "interpreter": sys.executable,
    }


def build_schema(root: Any, command: str | None = None) -> dict[str, Any]:
    """Serialize a converted command tree. ``command`` selects a single command."""
    runtime = runtime_identity()
    commands = {
        name: serialize_command(name, command_object, runtime)
        for name, command_object in getattr(root, "commands", {}).items()
    }
    schema: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runtime": runtime,
        "root": serialize_root(root, runtime),
        "commands": commands,
    }
    if command is None:
        return schema
    if command not in commands:
        known = ", ".join(commands) or "none"
        raise ValueError(f"unknown command: {command!r} (known: {known})")
    return {**schema, "commands": {command: commands[command]}}


def export_cli_schema(command: str | None = None) -> dict[str, Any]:
    """Export the runtime schema for every command, or for one command."""
    return build_schema(root_command(), command)


# ---------------------------------------------------------------- serializers


def _json_value(value: Any) -> Any:
    """Convert a runtime value into a JSON-safe value or fail explicitly."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("unsupported non-finite numeric default")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported value shape: {type(value).__name__}")


def _default_entry(parameter: Any) -> dict[str, Any]:
    if parameter.required and parameter.default is None:
        return {"kind": "unset"}
    if parameter.default is None:
        return {"kind": "null"}
    return {"kind": "value", "value": _json_value(parameter.default)}


def _choices_entry(parameter: Any) -> list[Any] | None:
    choices = getattr(parameter.type, "choices", None)
    if choices is None:
        return None
    return [_json_value(getattr(choice, "value", choice)) for choice in choices]


def _bounds_entry(parameter: Any) -> dict[str, Any] | None:
    bounds: dict[str, Any] = {}
    for key in ("min", "max"):
        value = getattr(parameter.type, key, None)
        if value is not None:
            bounds[key] = _json_value(value)
    for key in ("min_open", "max_open"):
        if getattr(parameter.type, key, False):
            bounds[key] = True
    return bounds or None


def _path_constraints_entry(parameter: Any) -> dict[str, bool] | None:
    if getattr(parameter.type, "name", None) not in ("path", "directory"):
        return None
    return {
        attribute: bool(getattr(parameter.type, attribute, False))
        for attribute in _PATH_CONSTRAINT_ATTRIBUTES
    }


def _parameter_kind(parameter: Any) -> str:
    # A Typer positional also carries its name in ``opts``, so the class shape
    # has to be checked before the option-spelling heuristic.
    if type(parameter).__name__.endswith("Argument"):
        return "positional"
    if list(getattr(parameter, "opts", None) or []):
        return "option"
    raise ValueError(f"unsupported parameter shape: {type(parameter).__name__}")


def _validate_supported_shape(parameter: Any) -> None:
    if getattr(parameter, "nargs", 1) not in (None, 1):
        raise ValueError(f"unsupported value arity on {parameter.name!r}")
    if getattr(parameter, "multiple", False):
        raise ValueError(f"unsupported repeating option {parameter.name!r}")
    if getattr(parameter, "count", False):
        raise ValueError(f"unsupported counting flag {parameter.name!r}")
    flag_value = getattr(parameter, "flag_value", None)
    if flag_value not in (None, True) and getattr(parameter, "is_flag", False):
        raise ValueError(f"unsupported flag value on {parameter.name!r}")


def serialize_parameter(
    parameter: Any, annotation: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Serialize one declared parameter, merged with shared restrictions."""
    _validate_supported_shape(parameter)

    parameter_type = getattr(parameter, "type", None)
    type_name = getattr(parameter_type, "name", None)
    if type_name not in _PARAMETER_TYPE_NAMES:
        raise ValueError(
            f"unsupported parameter type for {parameter.name!r}: {type_name!r}"
        )

    kind = _parameter_kind(parameter)
    opts = [str(option) for option in (getattr(parameter, "opts", None) or [])]
    if kind == "positional" and not opts:
        opts = [str(parameter.name)]

    annotation = dict(annotation or {})
    choices = _choices_entry(parameter)
    bounds = _bounds_entry(parameter)
    path_constraints = _path_constraints_entry(parameter)
    shared_choices = [
        _json_value(choice) for choice in (annotation.get("choices") or [])
    ] or None
    shared_bounds = dict(annotation["bounds"]) if annotation.get("bounds") else None
    keyword_values = [
        _json_value(value) for value in (annotation.get("keyword_values") or [])
    ] or None
    path_role = annotation.get("path_role")

    return {
        "name": str(parameter.name),
        "kind": kind,
        "opts": opts,
        "secondary_opts": [
            str(option) for option in (getattr(parameter, "secondary_opts", None) or [])
        ],
        "required": bool(parameter.required),
        "default": _default_entry(parameter),
        "type": type_name,
        "choices": choices,
        "bounds": bounds,
        "path_constraints": path_constraints,
        "shared_choices": shared_choices,
        "shared_bounds": shared_bounds,
        "keyword_values": keyword_values,
        "path_role": path_role,
        "unresolved": [str(item) for item in annotation.get("unresolved", ())],
        "is_flag": bool(getattr(parameter, "is_flag", False)),
        "flag_value": _json_value(getattr(parameter, "flag_value", None)),
        "nargs": getattr(parameter, "nargs", None),
        "multiple": bool(getattr(parameter, "multiple", False)),
        "help": (getattr(parameter, "help", None) or "").strip() or None,
        "restriction_source": _restriction_source(
            choices,
            bounds,
            path_constraints,
            shared_choices,
            shared_bounds,
            keyword_values,
            path_role,
        ),
    }


def _restriction_source(
    choices: list[Any] | None,
    bounds: dict[str, Any] | None,
    path_constraints: dict[str, bool] | None,
    shared_choices: list[Any] | None = None,
    shared_bounds: dict[str, Any] | None = None,
    keyword_values: list[Any] | None = None,
    path_role: str | None = None,
) -> str:
    """Whether a restriction is parser-declared, shared-validator-derived, or absent."""
    if choices or bounds or path_constraints:
        return "parser"
    if shared_choices or shared_bounds or keyword_values or path_role:
        return "shared-validator"
    return "unavailable"


def _split_runtime_parameters(command_object: Any) -> tuple[list[Any], list[Any], list[str]]:
    parameters: list[Any] = []
    controls: list[Any] = []
    context: list[str] = []
    for parameter in command_object.params:
        name = str(getattr(parameter, "name", ""))
        if name in _CONTEXT_PARAMETER_NAMES:
            context.append(name)
            continue
        if name in _CONTROL_ROLES:
            controls.append(parameter)
            continue
        parameters.append(parameter)
    return parameters, controls, context


def _control_entry(parameter: Any) -> dict[str, Any]:
    return {
        "name": str(parameter.name),
        "role": _CONTROL_ROLES.get(str(parameter.name), "operational"),
        "opts": [str(option) for option in (getattr(parameter, "opts", None) or [])],
        "help": (getattr(parameter, "help", None) or "").strip() or None,
        "declared": True,
    }


def _builtin_help_control() -> dict[str, Any]:
    return {
        "name": "help",
        "role": "help",
        "opts": [BUILTIN_HELP_OPTION],
        "help": "Built-in help option supplied by the CLI framework for every command.",
        "declared": False,
    }


def _shared_constraints(command: str) -> Mapping[str, Mapping[str, Any]]:
    """Supplemental restrictions from the light shared-constraint module.

    A command without shared restrictions (including a test-local synthetic
    command) simply exports its parser-declared metadata.
    """
    from otuformer.cli.constraints import parameter_constraints

    try:
        return parameter_constraints(command)
    except ValueError:
        return {}


def serialize_command(
    name: str, command_object: Any, runtime: Mapping[str, str]
) -> dict[str, Any]:
    """Serialize one command's help, controls, and ordered parameters."""
    parameters, controls, context = _split_runtime_parameters(command_object)
    annotations = _shared_constraints(name)
    return {
        "path": name,
        "name": name,
        "help": (getattr(command_object, "help", None) or "").strip() or None,
        "callback": "group" if hasattr(command_object, "commands") else "command",
        "runtime": dict(runtime),
        "builtin_help_option": BUILTIN_HELP_OPTION,
        "parameters": [
            serialize_parameter(parameter, annotations.get(str(parameter.name)))
            for parameter in parameters
        ],
        "controls": [_control_entry(parameter) for parameter in controls],
        "context_parameters": context,
    }


def serialize_root(root: Any, runtime: Mapping[str, str]) -> dict[str, Any]:
    """Serialize root-level operational controls separately from parameters."""
    parameters, controls, context = _split_runtime_parameters(root)
    if parameters:
        # Root-level analysis parameters would silently bypass per-command
        # guidance, so surface them as controls rather than pretending they are
        # part of a command's parameter set.
        controls = list(parameters) + list(controls)
    return {
        "path": "",
        "name": None,
        "help": (getattr(root, "help", None) or "").strip() or None,
        "runtime": dict(runtime),
        "builtin_help_option": BUILTIN_HELP_OPTION,
        "parameters": [],
        "controls": [_control_entry(parameter) for parameter in controls]
        + [_builtin_help_control()],
        "context_parameters": context,
    }


def command_names() -> Sequence[str]:
    """Names of the runtime's executable commands."""
    return tuple(getattr(root_command(), "commands", {}) or {})


# --------------------------------------------------------------------- proposals
#
# Everything below is read-only: it normalizes explicit inputs, builds the argv
# and card a human reviews, and checks output safety. It never executes a
# command, never creates an output directory, and never grants approval.

_READ_ONLY_COMMANDS = ("doctor",)
_FINETUNE_EXPERIMENT_OPTIONS: tuple[str, ...] | None = None

# Path roles that must never overlap the chosen output directory.
_PROTECTED_PATH_ROLES = frozenset(
    {
        "images_dir",
        "label_csv",
        "checkpoint",
        "resume_checkpoint",
        "pseudo_source",
        "onnx_model",
        "embeddings",
        "assignments",
        "otu_table",
        "corrections",
        "tree",
    }
)

# Settings that ``otuformer.training.trainer.run_pretrain`` resolves from the
# resumed checkpoint when the option is omitted: patch config
# (``_resolve_patch_config``), augmentation/orientation policy
# (``_select_augmentation_profile``/``_select_orientation_policy``), input
# geometry/architecture, and ``weight_decay`` (restored from the saved optimizer
# state). Everything else is read from this invocation, so it stays
# ``omitted-default`` rather than being reported as inherited.
_PRETRAIN_RESUME_INHERITED = frozenset(
    {
        "model_name",
        "register_tokens",
        "global_crop_size",
        "local_crop_size",
        "local_crops",
        "patch_loss",
        "masking_strategy",
        "mask_ratio",
        "ibot_prototypes",
        "augmentation",
        "orientation_policy",
        "weight_decay",
    }
)

_INHERITED_PARAMETERS = {
    "pretrain": _PRETRAIN_RESUME_INHERITED,
    "finetune": (),  # filled from the CLI classification map below
}


def _finetune_experiment_options() -> frozenset[str]:
    """Option names recorded in the finetune cross-run identity."""
    global _FINETUNE_EXPERIMENT_OPTIONS
    if _FINETUNE_EXPERIMENT_OPTIONS is None:
        from otuformer.cli.finetune import PARAM_CLASSIFICATION

        _FINETUNE_EXPERIMENT_OPTIONS = tuple(
            name for name, kind in PARAM_CLASSIFICATION.items() if kind == "experiment"
        )
    return frozenset(_FINETUNE_EXPERIMENT_OPTIONS)


def parameter_index(command: str) -> dict[str, dict[str, Any]]:
    """Schema parameter entries for one command, keyed by internal name."""
    entry = export_cli_schema(command)["commands"][command]
    return {item["name"]: item for item in entry["parameters"]}


def _alias_index(command: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name, entry in parameter_index(command).items():
        for spelling in list(entry["opts"]) + list(entry["secondary_opts"]):
            aliases[str(spelling)] = name
    return aliases


def _check_parser_bounds(key: str, value: Any, bounds: Mapping[str, Any] | None) -> None:
    """Apply the runtime parser's own bounds to a coerced numeric value."""
    if not bounds:
        return
    low, high = bounds.get("min"), bounds.get("max")
    if low is not None and (value <= low if bounds.get("min_open") else value < low):
        raise ValueError(f"{key} must be >= {low}, got {value!r}")
    if high is not None and (value >= high if bounds.get("max_open") else value > high):
        raise ValueError(f"{key} must be <= {high}, got {value!r}")


def _check_path_constraints(
    key: str, value: str, constraints: Mapping[str, Any]
) -> None:
    """Apply the runtime path type's existence/file/dir/access constraints."""
    path = Path(value).expanduser()
    exists = path.exists()
    if constraints.get("exists") and not exists:
        raise ValueError(f"{key} path does not exist: {value!r}")
    if not exists:
        return
    if constraints.get("file_okay") is False and path.is_file():
        raise ValueError(f"{key} must be a directory, got a file: {value!r}")
    if constraints.get("dir_okay") is False and path.is_dir():
        raise ValueError(f"{key} must be a file, got a directory: {value!r}")
    if constraints.get("readable") and not os.access(path, os.R_OK):
        raise ValueError(f"{key} is not readable: {value!r}")
    if constraints.get("writable") and not os.access(path, os.W_OK):
        raise ValueError(f"{key} is not writable: {value!r}")


def _coerce(entry: Mapping[str, Any], value: Any, key: str) -> Any:
    """Convert one explicit JSON value into the CLI's accepted representation."""
    type_name = entry["type"]
    if entry["is_flag"]:
        if not isinstance(value, bool):
            raise ValueError(f"{key} expects true or false, got {value!r}")
        if value is False and not entry["secondary_opts"]:
            raise ValueError(
                f"{key}=false cannot be represented: {entry['opts'][0]} has no "
                "negative spelling; omit the option instead"
            )
        return value
    if type_name == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{key} expects true or false, got {value!r}")
        return value
    if type_name in ("integer", "integer range"):
        if isinstance(value, bool):
            raise ValueError(f"{key} expects an integer, got {value!r}")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float) and value.is_integer():
            parsed = int(value)
        elif isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                raise ValueError(f"{key} expects an integer, got {value!r}") from None
        else:
            raise ValueError(f"{key} expects an integer, got {value!r}")
        _check_parser_bounds(key, parsed, entry.get("bounds"))
        return parsed
    if type_name in ("float", "float range"):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{key} expects a number, got {value!r}")
        try:
            parsed = float(value)
        except ValueError:
            raise ValueError(f"{key} expects a number, got {value!r}") from None
        if not math.isfinite(parsed):
            raise ValueError(f"{key} expects a finite number, got {value!r}")
        _check_parser_bounds(key, parsed, entry.get("bounds"))
        return parsed
    if not isinstance(value, str):
        raise ValueError(f"{key} expects text, got {value!r}")
    choices = entry["choices"] or []
    if choices and value not in choices:
        raise ValueError(
            f"{key} must be one of: {', '.join(map(str, choices))}; got {value!r}"
        )
    constraints = entry.get("path_constraints")
    if constraints:
        _check_path_constraints(key, value, constraints)
    return value


def validate_parameters(command: str, explicit: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize explicit inputs: coerce, resolve aliases, check restrictions.

    Returns canonical values (including declared defaults used for conditional
    checks), the ordered explicit option names, and unresolved conditions.
    ``None`` is not an omitted key and never becomes a merged default.
    """
    entries = parameter_index(command)
    aliases = _alias_index(command)
    explicit_values: dict[str, Any] = {}
    explicit_options: list[str] = []
    sources: dict[str, str] = {}

    for key, raw in explicit.items():
        name = key if key in entries else aliases.get(str(key))
        if name is None:
            raise ValueError(f"unknown parameter for {command}: {key!r}")
        if raw is None:
            raise ValueError(
                f"{key!r} is null; omit the key instead of passing a null value"
            )
        value = _coerce(entries[name], raw, str(key))
        if name in explicit_values:
            if explicit_values[name] != value:
                raise ValueError(
                    f"conflicting values for {name!r} from {sources[name]!r} and {key!r}"
                )
            continue
        explicit_values[name] = value
        sources[name] = str(key)

    missing = [
        name
        for name, entry in entries.items()
        if entry["required"] and name not in explicit_values
    ]
    if missing:
        options = ", ".join(entries[name]["opts"][0] for name in missing)
        raise ValueError(f"missing required parameters for {command}: {options}")

    # Canonical order (declaration order), not caller order, so argv, cards, and
    # explicit-option provenance are deterministic.
    explicit_options = [name for name in entries if name in explicit_values]

    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    unresolved: list[str] = []
    for name, entry in entries.items():
        if name in explicit_values:
            values[name] = explicit_values[name]
            provenance[name] = "explicit"
            continue
        if entry["default"]["kind"] == "value":
            values[name] = entry["default"]["value"]
        provenance[name] = "omitted-default"
        unresolved.extend(entry["unresolved"])

    from otuformer.cli.constraints import validate_constraints

    # Merged values (declared defaults included) let the checker tell an
    # inherited checkpoint setting apart from a declared default.
    unresolved.extend(
        validate_constraints(
            command, values, explicit_options=frozenset(explicit_options)
        )
    )

    inherited_run = bool(explicit_values.get("resume")) or bool(
        explicit_values.get("pseudo_label_from")
    )
    if command in _INHERITED_PARAMETERS and inherited_run:
        inherited = (
            _finetune_experiment_options()
            if command == "finetune"
            else _PRETRAIN_RESUME_INHERITED
        )
        for name in entries:
            if name in explicit_values:
                continue
            if name in inherited:
                provenance[name] = "omitted-inherited"
        unresolved.append(
            "omitted training settings are inherited from the resumed or "
            "pseudo-label source checkpoint unless explicitly overridden; they are "
            "not confirmed effective values"
        )

    return {
        "command": command,
        "values": values,
        "explicit_values": explicit_values,
        "explicit_options": explicit_options,
        "provenance": provenance,
        "unresolved": _dedupe(unresolved),
        "runtime": runtime_identity(),
    }


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def build_argv(
    command: str,
    explicit: Mapping[str, Any],
    *,
    executable: Sequence[str],
) -> list[str]:
    """Build the argv for a validated explicit input set, preserving omission."""
    validation = validate_parameters(command, explicit)
    entries = parameter_index(command)
    argv = [str(part) for part in executable] + [command]
    for name in entries:
        if name not in validation["explicit_values"]:
            continue
        entry = entries[name]
        value = validation["explicit_values"][name]
        if entry["kind"] == "positional":
            # A positional takes the bare value; its ``opts`` entry is only the
            # parameter name, not a command-line spelling.
            argv.append(str(value))
            continue
        option = entry["opts"][0]
        if entry["is_flag"]:
            if value:
                argv.append(option)
            else:
                negative = entry["secondary_opts"][0]
                argv.append(negative)
            continue
        argv.extend([option, str(value)])
    return argv


def classify_operation(command: str, values: Mapping[str, Any]) -> str:
    """Classify what running the proposal would do, independent of approval."""
    if command in _READ_ONLY_COMMANDS:
        return "read-only"
    if command == "update":
        return "read-only-networked" if values.get("check") else "installation"
    return "analysis"


def _resolve_for_safety(path: Path) -> Path:
    """Resolve a path without requiring it to exist, following real symlinks.

    The longest existing prefix is resolved by the filesystem, and only the
    non-existent tail is folded lexically. That ordering matters: for
    ``link/../raw`` where ``link`` points at ``data/sub``, the real target is
    ``data/raw`` (the parent of the link *target*), not ``<link dir>/raw``.
    """
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    tail: list[str] = []
    current = path
    while not current.exists() and current != current.parent:
        tail.append(current.name)
        current = current.parent
    resolved = current.resolve()
    for name in reversed(tail):
        if name == "..":
            resolved = resolved.parent
        elif name in ("", "."):
            continue
        else:
            resolved = resolved / name
            # A symlink can reappear inside the non-existent tail (for example
            # "missing/../link/data"); resolve it as the filesystem would.
            if resolved.exists():
                resolved = resolved.resolve()
    return resolved


def _is_within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def check_output_safety(
    out_dir: Path,
    *,
    protected_paths: Sequence[Path] = (),
    overwrite: bool = False,
    resume: bool = False,
    allow_inside: Sequence[Path] = (),
) -> list[str]:
    """Read-only output-safety check; raises for unsafe overlap.

    ``allow_inside`` names paths that may legitimately live inside the output
    directory as long as no overwrite is requested (for example the checkpoint of
    a supported training resume). Returns consent conditions for existing or
    not-yet-created directories. Never creates, deletes, or writes anything.
    """
    target = _resolve_for_safety(Path(out_dir))
    permitted = [_resolve_for_safety(Path(path)) for path in allow_inside]
    conditions: list[str] = []
    for protected in protected_paths:
        resolved = _resolve_for_safety(Path(protected))
        if _is_within(target, resolved):
            raise ValueError(
                f"output directory {target} is inside protected input {resolved}"
            )
        if _is_within(resolved, target):
            permitted_here = any(
                resolved == candidate or _is_within(resolved, candidate)
                for candidate in permitted
            )
            if permitted_here and not overwrite:
                conditions.append(
                    f"{resolved} lives inside {target} and is reused read-only; "
                    "no overwrite is requested"
                )
                continue
            raise ValueError(
                f"output directory {target} contains protected input {resolved}; "
                "clearing it could remove inputs or results"
            )
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"output path {target} exists and is not a directory")
        if any(target.iterdir()):
            if overwrite:
                conditions.append(
                    f"{target} is a non-empty existing directory and would be "
                    "cleared; this needs separate explicit overwrite approval "
                    "naming this directory"
                )
            elif resume:
                conditions.append(
                    f"{target} is reused read-only as the approved training "
                    "resume directory"
                )
            else:
                conditions.append(
                    f"{target} is a non-empty existing directory; choose a fresh "
                    "path, a supported training resume, or explicitly approve "
                    "overwriting it"
                )
    else:
        conditions.append(f"{target} does not exist yet and is created only after approval")
    return conditions


def default_protected_paths() -> list[Path]:
    """Package source and bundled example assets that outputs must never touch.

    Derived from the loaded package so the default proposal flow protects a
    source checkout's ``src/`` tree and ``examples/`` even when the caller passes
    no ``--protected-path``. The repository root itself is deliberately not
    protected, so the documented ``runs/`` output root stays usable.
    """
    package_dir = Path(runtime_identity()["package_dir"])
    protected = [package_dir]
    # A src layout puts every project source module beside the package; the whole
    # tree, not only the loaded package, is a source directory outputs must avoid.
    if package_dir.parent.name == "src":
        protected.append(package_dir.parent)
    checkout = package_dir.parent.parent
    if (checkout / "pyproject.toml").is_file() and (checkout / "examples").is_dir():
        protected.append(checkout / "examples")
    return protected


def _protected_paths_from_values(
    command: str, explicit_values: Mapping[str, Any]
) -> list[Path]:
    """Inputs that must never overlap the output directory.

    Derived from the runtime schema: any declared path parameter except the
    output directory itself, plus annotated non-path inputs such as a resume
    checkpoint or pseudo-label source. A new input option is therefore protected
    without editing a hand-maintained list here.
    """
    entries = parameter_index(command)
    protected: list[Path] = []
    for name, value in explicit_values.items():
        if value in (None, "") or name == "out_dir":
            continue
        entry = entries.get(name) or {}
        if entry.get("path_role") in _PROTECTED_PATH_ROLES or entry.get("type") == "path":
            protected.append(Path(str(value)))
    return protected


def render_parameter_card(
    command: str,
    explicit: Mapping[str, Any],
    *,
    executable: Sequence[str] = ("otuformer",),
    extra_conditions: Sequence[str] = (),
) -> str:
    """Render the complete review card for one command and explicit input set.

    Parser-declared and shared restrictions are printed for every parameter, and
    output-safety conditions from the proposal are part of the card the operator
    approves.
    """
    validation = validate_parameters(command, explicit)
    entries = parameter_index(command)
    argv = build_argv(command, explicit, executable=executable)
    command_help = export_cli_schema(command)["commands"][command]["help"] or ""
    purpose = command_help.splitlines()[0] if command_help else ""

    lines = [
        f"# Proposed step: {command}",
        f"Purpose: {purpose}",
        f"Schema source: otuformer {validation['runtime']['package_version']} "
        f"({validation['runtime']['package_dir']}) via "
        f"{validation['runtime']['interpreter']}",
        f"Operation: {classify_operation(command, validation['values'])}",
        f"Proposed command: {shlex.join(argv)}",
        "",
        f"Parameters ({len(entries)} of {len(entries)} shown):",
    ]

    shown = 0
    for name, entry in entries.items():
        shown += 1
        default = entry["default"]
        if default["kind"] == "value":
            declared = repr(default["value"])
        else:
            declared = default["kind"]
        if name in validation["explicit_values"]:
            proposed = repr(validation["explicit_values"][name])
        elif entry["required"]:
            proposed = "unset (required)"
        else:
            proposed = "(omitted)"
        spellings = ", ".join(entry["opts"] + entry["secondary_opts"])
        restrictions = []
        if entry["required"]:
            restrictions.append("required")
        if entry["choices"]:
            restrictions.append(f"choices: {', '.join(map(str, entry['choices']))}")
        if entry["shared_choices"]:
            restrictions.append(
                f"accepted values: {', '.join(map(str, entry['shared_choices']))}"
            )
        if entry["keyword_values"]:
            restrictions.append(
                f"keyword values: {', '.join(map(str, entry['keyword_values']))}"
            )
        if entry["bounds"]:
            restrictions.append(f"bounds: {entry['bounds']}")
        if entry["shared_bounds"]:
            restrictions.append(f"bounds: {entry['shared_bounds']}")
        if entry["type"] == "path":
            restrictions.append("path")
        if entry["path_role"]:
            restrictions.append(f"input role: {entry['path_role']}")
        if entry["path_constraints"]:
            constraints = ", ".join(
                f"{key}={value}" for key, value in entry["path_constraints"].items()
            )
            restrictions.append(f"path constraints: {constraints}")
        unresolved = list(entry["unresolved"])
        card_lines = [
            f"{shown}. {spellings}  [{entry['type']}]",
            f"   meaning: {entry['help'] or '(no help text)'}",
            f"   required: {'yes' if entry['required'] else 'no'}; "
            f"declared default: {declared}",
            f"   proposed value: {proposed}; "
            f"provenance: {validation['provenance'][name]}; "
            f"restriction source: {entry['restriction_source']}",
        ]
        if restrictions:
            card_lines.append(f"   restrictions: {'; '.join(restrictions)}")
        lines.extend(card_lines)
        for note in unresolved:
            lines.append(f"   unresolved: {note}")

    if shown != len(entries):
        raise ValueError(
            f"parameter card is incomplete: {shown} of {len(entries)} parameters shown"
        )

    lines.extend(["", f"Validation: {len(entries)} parameters covered."])
    for note in validation["unresolved"]:
        lines.append(f"Condition: {note}")
    lines.append("")
    if extra_conditions:
        lines.append("Output safety conditions:")
        for note in extra_conditions:
            lines.append(f"- {note}")
    else:
        lines.append("Output safety conditions: none reported for this proposal.")
    lines.append(
        "Approval requested: confirm this exact command, inputs, output directory, "
        "and execution environment before the step is run."
    )
    return "\n".join(lines)


def build_proposal(
    command: str,
    explicit: Mapping[str, Any],
    *,
    executable: Sequence[str] = ("otuformer",),
    protected_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Assemble the complete, non-executing proposal an operator reviews."""
    validation = validate_parameters(command, explicit)
    operation = classify_operation(command, validation["values"])
    if operation == "analysis" and not validation["explicit_values"].get("out_dir"):
        raise ValueError(f"{command} proposals must select --out-dir explicitly")

    overwrite = bool(validation["explicit_values"].get("overwrite"))
    resume = bool(validation["explicit_values"].get("resume"))
    safety_conditions: list[str] = []
    if validation["explicit_values"].get("out_dir"):
        protected = (
            list(protected_paths)
            + default_protected_paths()
            + _protected_paths_from_values(command, validation["explicit_values"])
        )
        # A resumed run reads its checkpoint from inside its own output
        # directory; that is a supported read-only reuse, not a destructive
        # ancestor, and is only unsafe when an overwrite would delete it.
        allow_inside = (
            [Path(str(validation["explicit_values"]["resume"]))]
            if resume and validation["explicit_values"].get("resume")
            else []
        )
        safety_conditions = check_output_safety(
            Path(str(validation["explicit_values"]["out_dir"])),
            protected_paths=protected,
            overwrite=overwrite,
            resume=resume,
            allow_inside=allow_inside,
        )

    argv = build_argv(command, explicit, executable=executable)
    return {
        "command": command,
        "operation": operation,
        "runtime": validation["runtime"],
        "schema": export_cli_schema(command),
        "validation": validation,
        "safety_conditions": safety_conditions,
        "card": render_parameter_card(
            command,
            explicit,
            executable=executable,
            extra_conditions=safety_conditions,
        ),
        "argv": argv,
        "review_command": shlex.join(argv),
    }


_VERSION_TOKEN = re.compile(r"\d+(?:\.\d+)+(?:[-+.][0-9A-Za-z.\-]+)?")


def _launch_prefix_and_interpreter(
    executable: Sequence[str], runtime: Mapping[str, str]
) -> tuple[list[str], Path, str | None]:
    """Resolve the launch prefix and the interpreter that will run the package."""
    import importlib.util

    parts = [str(part) for part in executable]
    interpreter = Path(runtime["interpreter"]).resolve()

    if len(parts) == 1:
        candidate = shutil.which(parts[0])
        resolved = str(Path(candidate).resolve()) if candidate else None
        if resolved is None:
            path = Path(parts[0]).expanduser()
            if not path.exists():
                raise ValueError(f"launcher not found: {parts[0]!r}")
            resolved = str(path.resolve())
        shebang = _launcher_interpreter(Path(resolved))
        launcher_interpreter = shebang[0] if shebang else None
        launcher_flags = shebang[1] if shebang else []
        if launcher_interpreter is None:
            # No shebang: accept only the schema's own interpreter, and only when
            # the package exposes a module entry point. Anything else is
            # ambiguous and must not be guessed at.
            if not _same_interpreter(Path(resolved), Path(runtime["interpreter"])):
                raise ValueError(
                    f"cannot inspect launcher {resolved!r}; pass a console script "
                    "such as the installed 'otuformer' entry point, or the "
                    "interpreter that imported this package"
                )
            if importlib.util.find_spec("otuformer.__main__") is None:
                raise ValueError(
                    f"launcher {resolved!r} is a plain interpreter and "
                    "'python -m otuformer' is not available; pass the installed "
                    "'otuformer' console script instead"
                )
            return [resolved, "-m", "otuformer"], interpreter, resolved, []
        if not _same_interpreter(launcher_interpreter, Path(runtime["interpreter"])):
            raise ValueError(
                f"environment mismatch: schema source interpreter "
                f"{runtime['interpreter']} but launcher {resolved} uses "
                f"{launcher_interpreter}; run the schema with the launcher's "
                "interpreter"
            )
        # Provenance: accept only the console entry point installed for this
        # distribution, next to the interpreter we verified. An arbitrary script
        # can print any version or help text, so it is refused rather than
        # reported as verified.
        declared = _installed_console_scripts()
        entry_name = Path(resolved).name
        if not declared:
            raise ValueError(
                "cannot verify the launcher: the installed otuformer distribution "
                f"declares no console scripts ({resolved!r}); install the package "
                "or pass '<python> [flags] -m otuformer'"
            )
        if entry_name not in declared:
            raise ValueError(
                f"{resolved!r} is not a console script declared by the installed "
                f"otuformer distribution ({', '.join(sorted(declared))}); a script "
                "that merely behaves like the CLI cannot be verified"
            )
        target = declared[entry_name]
        module = target.split(":")[0]
        if not module.startswith("otuformer"):
            raise ValueError(
                f"declared entry point {entry_name!r} -> {target!r} is not an "
                "OTU-Former module"
            )
        launcher_dir = Path(resolved).parent.resolve()
        interpreter_dir = Path(runtime["interpreter"]).parent.resolve()
        if launcher_dir != interpreter_dir:
            raise ValueError(
                f"{resolved!r} is not installed next to the schema interpreter "
                f"({interpreter_dir}); it cannot be tied to this installation"
            )
        attribute = target.split(":", 1)[1] if ":" in target else ""
        if not attribute:
            raise ValueError(
                f"declared entry point {entry_name!r} -> {target!r} names no callable"
            )
        _verify_console_script(Path(resolved), module, attribute)
        # probe through the launcher's own interpreter path and flags
        return [resolved], launcher_interpreter, resolved, launcher_flags

    # Explicit launch prefix: only an interpreter module invocation can be
    # verified, and it must keep its own interpreter flags (for example
    # ``-S``/``-E``) so the probe runs in the same startup environment.
    first = Path(parts[0]).expanduser()
    if not first.exists():
        candidate = shutil.which(parts[0])
        if candidate is None:
            raise ValueError(f"launcher not found: {parts[0]!r}")
        first = Path(candidate)
    first_resolved = first.resolve()
    if _launcher_interpreter(first_resolved) is not None:
        # The first element is a script, so any extra arguments are consumed by
        # that script; the interpreter probe cannot reproduce them.
        raise ValueError(
            f"cannot verify launch prefix {parts!r}: its first element is a "
            "script, not an interpreter; pass that script alone, or use "
            "'<python> [flags] -m otuformer'"
        )
    # Probe through the path as written: "venv/bin/python" is usually a symlink
    # to the base interpreter, and running the resolved path would leave the
    # virtual environment.
    if not _same_interpreter(first, Path(runtime["interpreter"])):
        raise ValueError(
            f"environment mismatch: schema source interpreter "
            f"{runtime['interpreter']} but launch prefix {parts[0]!r} uses {first}"
        )
    rest = parts[1:]
    module_flags: list[str] = []
    while rest and rest[0].startswith("-") and rest[0] not in ("-m", "-c"):
        module_flags.append(rest.pop(0))
    if rest[:2] != ["-m", "otuformer"]:
        raise ValueError(
            f"cannot verify launch prefix {parts!r}; pass the installed console "
            "script, or an interpreter prefix of the form "
            "'<python> [flags] -m otuformer'"
        )
    import importlib.util

    if importlib.util.find_spec("otuformer.__main__") is None:
        raise ValueError(
            f"launch prefix {parts!r} uses 'python -m otuformer', but this "
            "distribution has no module entry point; pass the installed console "
            "script instead"
        )
    return parts, first, str(first_resolved), module_flags


def _installed_console_scripts() -> dict[str, str]:
    """Console scripts declared by the installed otuformer distribution."""
    import importlib.metadata as metadata

    try:
        distribution = metadata.distribution("otuformer")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed package
        return {}
    return {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }


def _verify_console_script(path: Path, module: str, attribute: str) -> None:
    """Conservative structural check of a standard console-script entry point.

    Accepts only the pip-generated shape: optional ``import sys``/``import re``,
    one ``from <module> import <attribute>``, and a ``__main__`` guard that calls
    that attribute through ``sys.exit``. Anything else -- extra imports,
    ``sys.path`` or environment manipulation, or a module name that only appears
    in a comment, string, or dead branch -- is refused. A text mention is not
    evidence that the launcher runs this package, so unconfirmable forms are
    blocked rather than reported as verified.
    """
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as error:
        raise ValueError(f"cannot parse launcher {path}: {error}") from None

    def fail(detail: str) -> None:
        raise ValueError(f"launcher {path} is not a standard console script: {detail}")

    imported = False
    called = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name not in {"sys", "re"} for alias in node.names):
                fail(f"unexpected import ({ast.unparse(node)})")
            continue
        if isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            if node.module != module or names != [attribute]:
                fail(
                    f"does not import the declared entry point {module}:{attribute} "
                    f"(found {node.module}:{','.join(names)})"
                )
            imported = True
            continue
        if isinstance(node, ast.If):
            test = node.test
            if not (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                fail("unexpected conditional outside the entry-point guard")
            for inner in node.body:
                if isinstance(inner, ast.Assign):
                    continue  # standard sys.argv[0] fixup
                if (
                    isinstance(inner, ast.Expr)
                    and isinstance(inner.value, ast.Call)
                    and ast.unparse(inner.value.func) == "sys.exit"
                    and len(inner.value.args) == 1
                ):
                    invoked = inner.value.args[0]
                    if not (
                        isinstance(invoked, ast.Call)
                        and ast.unparse(invoked.func) == attribute
                    ):
                        fail("does not call the declared entry point callable")
                    called = True
                    continue
                fail(f"unexpected statement in the entry-point guard ({ast.unparse(inner)})")
            continue
        fail(f"unexpected top-level statement ({ast.unparse(node)})")

    if not imported or not called:
        fail("does not import and call the declared entry point")


def _verify_cli_surface(prefix: Sequence[str]) -> None:
    """Prove the launcher really runs this CLI, not just version-like output.

    This is a compatibility check only: it catches a broken launcher, but help
    text is not provenance (provenance comes from the installed entry-point
    checks below). Command names must appear as whole tokens, so text such as
    ``not_doctor_command`` does not satisfy it.
    """
    completed = subprocess.run(
        [*prefix, "--help"], capture_output=True, text=True, check=False
    )
    reported = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise ValueError(
            f"launcher {prefix[0]!r} --help failed: {reported.strip()[:200]}"
        )
    missing = [
        name
        for name in command_names()
        if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", reported) is None
    ]
    if missing:
        raise ValueError(
            f"launcher {prefix[0]!r} does not expose the expected commands "
            f"({', '.join(missing[:4])}); it is not a working OTU-Former launcher"
        )


def _same_interpreter(left: Path, right: Path) -> bool:
    """Whether two interpreter paths name the same interpreter environment.

    Compared as written first, so a symlinked virtual-environment interpreter is
    not confused with the base interpreter it points at.
    """
    if Path(left) == Path(right):
        return True
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:  # pragma: no cover - defensive
        return False


def _observed_package(python: Path, module_flags: Sequence[str] = ()) -> dict[str, str]:
    """Ask an interpreter which otuformer package it actually imports."""
    probe = (
        "import json, otuformer;"
        "print(json.dumps({'file': str(otuformer.__file__),"
        " 'version': str(otuformer.__version__)}))"
    )
    completed = subprocess.run(
        [str(python), *module_flags, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"cannot inspect the launcher environment via {python}: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    try:
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise ValueError(
            f"unexpected launcher probe output: {completed.stdout.strip()!r}"
        ) from None
    return {"package_path": str(Path(observed["file"]).resolve()), "version": observed["version"]}


def verify_launcher(
    executable: Sequence[str], *, source_runtime: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Verify the launcher that would run an approved proposal.

    For a single-element launcher this accepts only the distribution's declared
    console script, installed next to the verified interpreter, with the standard
    pip-generated structure that imports and calls the declared entry point.
    Arbitrary wrapper scripts are refused rather than reported as verified: their
    text can mention the module without ever running it, and their own import
    path cannot be confirmed from outside. The launcher must also report exactly
    the schema source's package version, expose the same command surface, and use
    an interpreter that imports the same package location. For explicit
    interpreter prefixes (``<python> [flags] -m otuformer``) the package identity
    is inherent to the module invocation.
    """
    runtime = dict(source_runtime or runtime_identity())
    if not list(executable):
        raise ValueError("executable must not be empty")

    prefix, python, resolved, module_flags = _launch_prefix_and_interpreter(
        executable, runtime
    )

    completed = subprocess.run(
        [*prefix, "--version"], capture_output=True, text=True, check=False
    )
    reported = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise ValueError(f"launcher {prefix[0]!r} --version failed: {reported}")
    match = _VERSION_TOKEN.search(reported)
    if match is None:
        raise ValueError(f"launcher {prefix[0]!r} reported no version in {reported!r}")
    if match.group(0) != runtime["package_version"]:
        raise ValueError(
            f"version mismatch: schema source is otuformer "
            f"{runtime['package_version']} but the launcher reports "
            f"{match.group(0)!r}"
        )

    _verify_cli_surface(prefix)

    observed = _observed_package(python, module_flags)
    expected_path = str(Path(runtime["package_path"]).resolve())
    if observed["package_path"] != expected_path:
        raise ValueError(
            f"installation mismatch: schema source loaded {expected_path} but the "
            f"launcher's interpreter loads {observed['package_path']}"
        )
    if observed["version"] != runtime["package_version"]:
        raise ValueError(
            f"version mismatch: schema source is {runtime['package_version']} but "
            f"the launcher's interpreter imports {observed['version']}"
        )

    return {
        "argv": [str(part) for part in executable],
        "argv_prefix": prefix,
        "resolved": resolved,
        "reported_version": reported,
        "package_version": runtime["package_version"],
        "observed_package_path": observed["package_path"],
        "interpreter": str(python),
        "verified": True,
    }


def _launcher_interpreter(path: Path) -> tuple[Path, list[str]] | None:
    """Interpreter and interpreter flags from a console-script shebang.

    The path is kept *unresolved* on purpose: a symlinked virtual-environment
    interpreter must be executed through its own path, otherwise the probe would
    run the base interpreter and report a different environment. Shebang flags
    (for example ``-S``) are preserved so the probe starts exactly like the real
    launcher. Returns ``None`` when the interpreter cannot be determined.
    """
    try:
        with path.open("rb") as handle:
            first = handle.readline(512)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    try:
        command = first[2:].decode().strip()
    except UnicodeDecodeError:
        return None
    parts = shlex.split(command)
    if not parts:
        return None
    if Path(parts[0]).name == "env":
        if len(parts) > 1 and parts[1].startswith("-"):
            # "env" options such as -S/-i change how the interpreter starts; a
            # probe that ignores them would not be equivalent to the real launch,
            # so such a launcher is refused instead of silently mis-verified.
            raise ValueError(
                "shebang uses `env` options that cannot be reproduced faithfully "
                f"({command!r}); pass an explicit '<python> [flags] -m otuformer' "
                "prefix or an installed console script"
            )
        parts = parts[1:]
    if not parts:
        return None
    candidate = Path(parts[0])
    flags = list(parts[1:])
    if not candidate.is_absolute():
        found = shutil.which(parts[0])
        if found is None:
            return None
        candidate = Path(found)
    if not candidate.exists():
        return None
    return candidate, flags
