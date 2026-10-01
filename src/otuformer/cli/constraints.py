"""Shared, framework-free CLI restrictions used by the CLI and the schema export.

Everything here is import-light (no torch, no training import, no CLI framework),
so the read-only schema path can annotate supplemental restrictions without
loading heavy modules or running option callbacks.

Two audiences:

* the CLI keeps its own wrappers, which convert the ``ValueError`` raised here
  into the same ``typer.BadParameter`` boundary and message as before;
* the workflow Skill uses :func:`parameter_constraints` and
  :func:`validate_constraints` to show and check every accepted value without
  keeping a second copy of them.

Values are keyed by the real internal parameter names. Core/checkpoint-dependent
rules are reported as unresolved conditions instead of guessed enums.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Mapping

from otuformer.constants import (
    ATTENTION_POOLING_TYPES,
    CAM_ARCH_CHOICES,
    CAM_DEVICE_CHOICES,
    CAM_FIG_FORMAT_CHOICES,
    CAM_METHOD_CHOICES,
    CAM_SAVE_NPY_CHOICES,
    EVAL_TRANSFORM_CHOICES,
    FINETUNE_AUGMENTATIONS,
    IBOT_PROTOTYPES_MIN,
    LONG_TAIL_MODES,
    LOSS_CHOICES,
    MASKING_STRATEGIES,
    MASK_RATIO_BOUNDS,
    MAX_SUBCENTERS,
    MIN_SUBCENTERS,
    ORIENTATION_POLICIES,
    PATCH_LOSS_MODES,
    PRETRAIN_AUGMENTATIONS,
    PSEUDO_MIN_GAP_BOUNDS,
    PSEUDO_SIMILARITY_BOUNDS,
    REGISTER_TOKEN_CHOICES,
    TOKEN_MODES,
)

_LEGACY_EMBEDDINGS_NOTE = (
    "diversity --embeddings and --tree may be combined; --embeddings takes "
    "precedence for NJ tree construction and --tree stays legacy input"
)

_CONSTRAINTS: dict[str, dict[str, dict[str, Any]]] = {
    "pretrain": {
        "train_data": {"path_role": "label_csv"},
        "input_images_dir": {"path_role": "images_dir"},
        "out_dir": {"path_role": "output_dir"},
        "resume": {
            "path_role": "resume_checkpoint",
            "unresolved": [
                "resume checkpoint compatibility and inheritance are verified by "
                "the CLI preflight when the run starts, not before approval",
            ],
        },
        "augmentation": {"choices": list(PRETRAIN_AUGMENTATIONS)},
        "orientation_policy": {"choices": list(ORIENTATION_POLICIES)},
        "patch_loss": {"choices": list(PATCH_LOSS_MODES)},
        "masking_strategy": {"choices": list(MASKING_STRATEGIES)},
        "register_tokens": {"choices": list(REGISTER_TOKEN_CHOICES)},
        "mask_ratio": {
            "keyword_values": ["auto"],
            "numeric": "float",
            "bounds": {
                "min": MASK_RATIO_BOUNDS[0],
                "max": MASK_RATIO_BOUNDS[1],
                "exclusive": True,
            },
        },
        "ibot_prototypes": {"numeric": "int", "bounds": {"min": IBOT_PROTOTYPES_MIN}},
        "global_crop_size": {
            "keyword_values": ["auto"],
            "numeric": "int",
            "bounds": {"min": 1},
        },
        "local_crop_size": {"numeric": "int", "bounds": {"min": 1}},
        "extract_size": {"keyword_values": ["auto"], "numeric": "int", "bounds": {"min": 1}},
        "device": {
            "unresolved": [
                "accepted device spellings beyond auto/cpu/cuda/mps are decided "
                "by the core device resolver",
            ]
        },
    },
    "finetune": {
        "checkpoint": {
            "path_role": "checkpoint",
            "unresolved": ["checkpoint layout and architecture are verified at execution"],
        },
        "train_data": {"path_role": "label_csv"},
        "input_images_dir": {"path_role": "images_dir"},
        "out_dir": {"path_role": "output_dir"},
        "resume": {
            "path_role": "resume_checkpoint",
            "unresolved": [
                "resume checkpoint compatibility, loss and inheritance are "
                "verified by the CLI preflight when the run starts",
            ],
        },
        "pseudo_label_from": {
            "path_role": "pseudo_source",
            "unresolved": [
                "pseudo-label source eligibility and SSL provenance are verified "
                "against the checkpoint at execution",
            ],
        },
        "augmentation": {"choices": list(FINETUNE_AUGMENTATIONS)},
        "orientation_policy": {"choices": list(ORIENTATION_POLICIES)},
        "loss": {"choices": list(LOSS_CHOICES)},
        "long_tail": {"choices": list(LONG_TAIL_MODES)},
        "subcenters": {
            "numeric": "int",
            "bounds": {"min": MIN_SUBCENTERS, "max": MAX_SUBCENTERS},
            "applies_to": ["subcenter-arcface", "subcenter-arcface-compact"],
        },
        # Bounds mirror the core validator in otuformer.training.loss.
        "compact_weight": {
            "numeric": "float",
            "bounds": {"min": 0.0},
            "applies_to": ["subcenter-arcface-compact"],
        },
        # Mirrors otuformer.training.trainer._validate_temperature: finite > 0.
        "supcon_temperature": {
            "numeric": "float",
            "bounds": {"min": 0.0, "exclusive": True},
            "applies_to": ["supcon"],
        },
        "pseudo_similarity_floor": {
            "numeric": "float",
            "bounds": {
                "min": PSEUDO_SIMILARITY_BOUNDS[0],
                "max": PSEUDO_SIMILARITY_BOUNDS[1],
            },
        },
        "pseudo_min_gap": {
            "numeric": "float",
            "bounds": {"min": PSEUDO_MIN_GAP_BOUNDS[0], "max": PSEUDO_MIN_GAP_BOUNDS[1]},
        },
        "pseudo_neighbors": {"numeric": "int", "bounds": {"min": 1}},
        "pseudo_cap_multiplier": {"numeric": "int", "bounds": {"min": 1}},
        "pseudo_absolute_cap": {"numeric": "int", "bounds": {"min": 1}},
        "extract_size": {"keyword_values": ["auto"], "numeric": "int", "bounds": {"min": 1}},
        "device": {
            "unresolved": [
                "accepted device spellings beyond auto/cpu/cuda/mps are decided "
                "by the core device resolver",
            ]
        },
    },
    "extract": {
        "checkpoint": {
            "path_role": "checkpoint",
            "unresolved": [
                "checkpoint layout and architecture are verified at execution",
            ],
        },
        "onnx_path": {
            "path_role": "onnx_model",
            "unresolved": [
                "ONNX input size and graph layout are verified at execution",
                "ONNX takes precedence over --checkpoint when both are supplied",
            ],
        },
        "input_images_dir": {"path_role": "images_dir"},
        "out_dir": {"path_role": "output_dir"},
        "label_csv": {"path_role": "label_csv"},
        "eval_transform": {"choices": list(EVAL_TRANSFORM_CHOICES)},
        "token_mode": {"choices": list(TOKEN_MODES)},
        "attention_pooling_type": {"choices": list(ATTENTION_POOLING_TYPES)},
        # Mirrors the core guards in otuformer.embedding.extractor, which run
        # after the CLI has already cleared an overwritten --out-dir.
        "topk_patches": {"numeric": "int", "bounds": {"min": 1}},
        "attention_pooling_epochs": {"numeric": "int", "bounds": {"min": 1}},
        "extract_size": {"keyword_values": ["auto"], "numeric": "int", "bounds": {"min": 1}},
        "device": {
            "unresolved": [
                "accepted device spellings beyond auto/cpu/cuda/mps are decided "
                "by the core device resolver",
            ]
        },
    },
    "cluster": {
        "embeddings": {"path_role": "embeddings"},
        "out_dir": {"path_role": "output_dir"},
        "label_csv": {"path_role": "label_csv"},
        "pca_whitening": {"boolean_literal": True},
        "local_scaling": {"boolean_literal": True},
        "save_bootstrap_trees": {"boolean_literal": True},
        "save_distances": {"boolean_literal": True},
        "distance": {
            "unresolved": [
                "supported distance metrics are decided by the core delineation "
                "code, not by the parser",
            ]
        },
        "custom_cutoffs": {
            "unresolved": ["cutoff values are parsed and checked by the core"],
        },
    },
    "annotate": {
        "raw_assignments": {"path_role": "assignments"},
        "corrections": {
            "path_role": "corrections",
            "unresolved": [
                "correction fields and ID mapping are verified against the "
                "selected partition at execution",
            ],
        },
        "embeddings": {"path_role": "embeddings"},
        "out_dir": {"path_role": "output_dir"},
    },
    "diversity": {
        "assignments": {"path_role": "assignments"},
        "otu_table_csv": {"path_role": "otu_table"},
        "out_dir": {"path_role": "output_dir"},
        "embeddings": {"path_role": "embeddings"},
        "tree": {"path_role": "tree"},
        "nj_bootstrap_mode": {
            "unresolved": [
                "documented as 'subsample' or 'bootstrap'; the parser and shared "
                "validators do not enforce this value, the core does",
            ]
        },
        "min_abundance": {
            "unresolved": ["comma-separated thresholds are parsed by the core"],
        },
    },
    "cam": {
        "checkpoint": {
            "path_role": "checkpoint",
            "unresolved": ["checkpoint layout and architecture are verified at execution"],
        },
        "images_dir": {"path_role": "images_dir"},
        "label_csv": {"path_role": "label_csv"},
        "out_dir": {"path_role": "output_dir"},
        "cam_method": {"choices": list(CAM_METHOD_CHOICES)},
        "arch": {"choices": list(CAM_ARCH_CHOICES)},
        "fig_format": {"choices": list(CAM_FIG_FORMAT_CHOICES)},
        "save_npy": {"choices": list(CAM_SAVE_NPY_CHOICES)},
        "eval_transform": {"choices": list(EVAL_TRANSFORM_CHOICES)},
        "device": {"choices": list(CAM_DEVICE_CHOICES)},
    },
    "export": {
        "checkpoint": {
            "path_role": "checkpoint",
            "unresolved": ["checkpoint layout and architecture are verified at execution"],
        },
        "out_dir": {"path_role": "output_dir"},
        "imgsz": {"keyword_values": ["auto"], "numeric": "int", "bounds": {"min": 1}},
    },
    "doctor": {},
    "update": {
        "yes": {
            "unresolved": [
                "network access is required; --yes does not install without "
                "separate environment-change approval",
            ]
        }
    },
}

# Commands whose output is a directory that must exist or be created.
_OUTPUT_DIR_COMMANDS = (
    "pretrain",
    "finetune",
    "extract",
    "cluster",
    "annotate",
    "diversity",
    "cam",
    "export",
)

TRAINING_COMMANDS = ("pretrain", "finetune")


def command_names() -> tuple[str, ...]:
    """Commands that carry shared restrictions."""
    return tuple(_CONSTRAINTS)


def parameter_constraints(command: str) -> dict[str, dict[str, Any]]:
    """Supplemental restrictions for one command, keyed by internal parameter name."""
    if command not in _CONSTRAINTS:
        raise ValueError(f"unknown command: {command!r}")
    return {name: dict(rule) for name, rule in _CONSTRAINTS[command].items()}


# ------------------------------------------------------------------ validators


def validate_size_option(value: object, *, stage: str) -> int | None:
    """``auto`` or a positive integer; ``None`` means auto."""
    if value is None or value == "auto":
        return None
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        raise ValueError(f"{stage} must be 'auto' or a positive integer, got '{value}'")
    return parsed


def validate_mask_ratio(value: object) -> float | str:
    """``auto`` or a float in ``(0, 1)``."""
    if value is None or value == "auto":
        return "auto"
    try:
        ratio = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            f"--mask-ratio must be 'auto' or a float in (0, 1), got '{value}'"
        ) from None
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"--mask-ratio must be 'auto' or a float in (0, 1), got '{value}'"
        )
    return ratio


def validate_patch_options(
    patch_loss: str, masking_strategy: str, ibot_prototypes: int
) -> None:
    """Reject unknown patch-option values before any output directory is touched."""
    if patch_loss not in PATCH_LOSS_MODES:
        raise ValueError(
            f"--patch-loss must be one of {', '.join(PATCH_LOSS_MODES)}, "
            f"got '{patch_loss}'"
        )
    if masking_strategy not in MASKING_STRATEGIES:
        raise ValueError(
            f"--masking-strategy must be one of {', '.join(MASKING_STRATEGIES)}, "
            f"got '{masking_strategy}'"
        )
    if ibot_prototypes < IBOT_PROTOTYPES_MIN:
        raise ValueError(
            f"--ibot-prototypes must be an integer >= {IBOT_PROTOTYPES_MIN}, "
            f"got '{ibot_prototypes}'"
        )


def validate_pretrain_patch_applicability(
    values: Mapping[str, object], *, explicit_options: frozenset[str]
) -> None:
    """Reject pretrain patch options that do not apply to the chosen mode.

    Mirrors the applicability half of ``_resolve_new_patch_config`` in
    ``otuformer.training.trainer`` for a fresh run: the module is import-heavy,
    so the pure rules are restated here for the read-only schema path. Resumed
    runs defer to the checkpoint's saved mode and are handled by the caller.
    """
    mode = str(values.get("patch_loss") or "consistency")
    ratio = values.get("mask_ratio")
    ratio_explicit = "mask_ratio" in explicit_options or ratio not in (None, "auto")
    strategy_explicit = "masking_strategy" in explicit_options
    prototypes_explicit = "ibot_prototypes" in explicit_options
    if mode == "none":
        for option, supplied in (
            ("--mask-ratio", ratio_explicit),
            ("--masking-strategy", strategy_explicit),
            ("--ibot-prototypes", prototypes_explicit),
        ):
            if supplied:
                raise ValueError(
                    f"{option} is not applicable to patch_loss='none'; omit it or "
                    "choose a masking patch loss."
                )
        if "lambda_mask" in explicit_options and float(
            values.get("lambda_mask", 1.0)
        ) != 1.0:
            raise ValueError(
                "--lambda-mask is not applicable to patch_loss='none'; omit it or "
                "reset it to 1.0."
            )
    elif mode == "consistency":
        if strategy_explicit:
            raise ValueError(
                "--masking-strategy is not applicable to patch_loss='consistency'; "
                "consistency selects visible positions and never masks input."
            )
        if prototypes_explicit:
            raise ValueError(
                "--ibot-prototypes is not applicable to patch_loss='consistency'; "
                "omit it or choose ibot."
            )
    elif mode == "masked-feature" and prototypes_explicit:
        raise ValueError(
            "--ibot-prototypes is not applicable to patch_loss='masked-feature'; "
            "omit it or choose ibot."
        )


def validate_augmentation_profile(value: str | None, *, stage: str) -> None:
    """Reject an unknown augmentation profile before any training starts."""
    if value is None:
        return
    allowed = PRETRAIN_AUGMENTATIONS if stage == "pretrain" else FINETUNE_AUGMENTATIONS
    if value not in allowed:
        raise ValueError(
            f"Unknown {stage} augmentation profile '{value}'. "
            f"Choose one of: {', '.join(allowed)}."
        )


def validate_orientation_policy(value: str | None) -> None:
    """Reject an unknown orientation policy before any training starts."""
    if value is None:
        return
    if value not in ORIENTATION_POLICIES:
        raise ValueError(
            f"Unknown orientation policy '{value}'. "
            f"Choose one of: {', '.join(ORIENTATION_POLICIES)}."
        )


def validate_eval_transform(eval_transform: str) -> None:
    """Reject eval-transform values outside the canonical set."""
    if eval_transform not in EVAL_TRANSFORM_CHOICES:
        raise ValueError(
            "--eval-transform must be one of: "
            f"{', '.join(EVAL_TRANSFORM_CHOICES)}; got {eval_transform!r}"
        )


def validate_cam_options(
    *,
    cam_method: str,
    arch: str | None,
    fig_format: str,
    save_npy: str,
    eval_transform: str,
    device: str,
) -> None:
    """Reject CAM options outside their canonical value sets."""
    for option, value, allowed in (
        ("--cam-method", cam_method, CAM_METHOD_CHOICES),
        ("--arch", arch, CAM_ARCH_CHOICES),
        ("--fig-format", fig_format, CAM_FIG_FORMAT_CHOICES),
        ("--save-npy", save_npy, CAM_SAVE_NPY_CHOICES),
        ("--eval-transform", eval_transform, EVAL_TRANSFORM_CHOICES),
        ("--device", device, CAM_DEVICE_CHOICES),
    ):
        if value is None:
            continue
        if value not in allowed:
            raise ValueError(f"{option} must be one of: {', '.join(allowed)}; got {value!r}")


def validate_pseudo_options(
    *,
    long_tail: str,
    loss: str,
    pseudo_label_from: str,
    similarity_floor: float,
    min_gap: float,
    neighbors: int,
    cap_multiplier: int,
    absolute_cap: int,
) -> None:
    """Reject invalid pseudo/long-tail settings before any output exists."""
    if long_tail not in LONG_TAIL_MODES:
        raise ValueError(f"--long-tail must be none or cb-drw, got {long_tail!r}.")
    if not math.isfinite(similarity_floor) or not (
        PSEUDO_SIMILARITY_BOUNDS[0] <= similarity_floor <= PSEUDO_SIMILARITY_BOUNDS[1]
    ):
        raise ValueError(
            "--pseudo-similarity-floor must be a finite value in [-1, 1], "
            f"got {similarity_floor}."
        )
    if not math.isfinite(min_gap) or not (
        PSEUDO_MIN_GAP_BOUNDS[0] <= min_gap <= PSEUDO_MIN_GAP_BOUNDS[1]
    ):
        raise ValueError(
            f"--pseudo-min-gap must be a finite value in [0, 2], got {min_gap}."
        )
    if neighbors < 1:
        raise ValueError(
            f"--pseudo-neighbors must be a positive integer, got {neighbors}."
        )
    if cap_multiplier < 1:
        raise ValueError(
            "--pseudo-cap-multiplier must be a positive integer, got "
            f"{cap_multiplier}."
        )
    if absolute_cap < 1:
        raise ValueError(
            f"--pseudo-absolute-cap must be a positive integer, got {absolute_cap}."
        )
    check_loss_option_combinations(
        long_tail=long_tail, loss=loss, pseudo_label_from=pseudo_label_from
    )


def check_loss_option_combinations(
    *, long_tail: str, loss: str, pseudo_label_from: str
) -> None:
    """Cross-option loss rules shared by the CLI validator and proposal checks."""
    if long_tail == "cb-drw" and loss == "supcon":
        raise ValueError(
            "--long-tail cb-drw is rejected for --loss supcon: SupCon has no "
            "class-level cross-entropy term."
        )
    if pseudo_label_from and loss == "supcon":
        raise ValueError(
            "Pseudo-label feedback supports only ArcFace-family losses; a SupCon "
            "checkpoint cannot be a --pseudo-label-from source."
        )


def validate_diversity_sources(
    assignments: object, otu_table_csv: object
) -> None:
    """Exactly one of ``--assignments`` or ``--otu-table-csv``."""
    if (assignments is None) == (otu_table_csv is None):
        raise ValueError("Provide exactly one of --assignments or --otu-table-csv")


def validate_bool_literal(name: str, value: str) -> bool:
    """Parse the cluster command's string-valued booleans exactly as before."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid value for --{name}: {value!r}. Use true or false.")


# ------------------------------------------------------- constraint validation


def _check_bounds(option: str, value: Any, bounds: Mapping[str, Any]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{option} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{option} must be finite, got {value!r}")
    lower = bounds.get("min")
    upper = bounds.get("max")
    exclusive = bool(bounds.get("exclusive"))
    if lower is not None:
        if value < lower or (exclusive and value == lower):
            limit = "greater than" if exclusive else "at least"
            raise ValueError(f"{option} must be {limit} {lower}, got {value!r}")
    if upper is not None:
        if value > upper or (exclusive and value == upper):
            limit = "less than" if exclusive else "at most"
            raise ValueError(f"{option} must be {limit} {upper}, got {value!r}")


def _coerce_numeric(option: str, value: Any, numeric: str) -> Any:
    """Parse a declared numeric option value, rejecting non-numbers and bools."""
    if isinstance(value, bool):
        raise ValueError(f"{option} expects a number, got {value!r}")
    suffix = "an integer" if numeric == "int" else "a number"
    if numeric == "int":
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                raise ValueError(
                    f"{option} must be 'auto' or {suffix}, got {value!r}"
                ) from None
        raise ValueError(f"{option} must be 'auto' or {suffix}, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            raise ValueError(
                f"{option} must be 'auto' or {suffix}, got {value!r}"
            ) from None
    raise ValueError(f"{option} must be 'auto' or {suffix}, got {value!r}")


def _check_rule(option: str, value: Any, rule: Mapping[str, Any]) -> None:
    """Check one value against its rule.

    ``keyword_values`` add accepted literal spellings (for example ``auto``) to a
    numeric option; they are never a closed enum. Declared numbers are parsed
    before the bounds check, exactly as the CLI parses them.
    """
    choices = list(rule.get("choices") or ())
    keywords = list(rule.get("keyword_values") or ())
    if keywords and value in keywords:
        return
    if choices:
        if value not in choices:
            allowed = choices + keywords
            raise ValueError(
                f"{option} must be one of: {', '.join(map(str, allowed))}; got {value!r}"
            )
        return
    numeric = rule.get("numeric")
    parsed = _coerce_numeric(option, value, numeric) if numeric else value
    bounds = rule.get("bounds")
    if bounds is not None:
        _check_bounds(option, parsed, bounds)


def validate_constraints(
    command: str,
    values: Mapping[str, object],
    *,
    explicit_options: frozenset[str],
) -> list[str]:
    """Pure check of known-invalid inputs; returns unresolved condition strings.

    Raises ``ValueError`` for input the CLI/core rejects deterministically.
    Data-, checkpoint-, and file-dependent checks are returned as unresolved
    strings because they can only be decided against real inputs.
    """
    if command not in _CONSTRAINTS:
        raise ValueError(f"unknown command: {command!r}")
    rules = _CONSTRAINTS[command]
    known = set(rules) | set(_command_parameter_names().get(command, ()))
    unknown = sorted(key for key in values if key not in known)
    if unknown:
        raise ValueError(f"unknown parameters for {command}: {', '.join(unknown)}")

    unresolved: list[str] = []

    for name, value in values.items():
        rule = rules.get(name)
        if rule is None:
            continue
        option = f"--{name.replace('_', '-')}"
        if value is None:
            continue
        _check_rule(option, value, rule)
        if rule.get("boolean_literal") and isinstance(value, str):
            validate_bool_literal(name.replace("_", "-"), value)
        unresolved.extend(str(item) for item in rule.get("unresolved", ()))

    def given(name: str) -> bool:
        return values.get(name) not in (None, "", False)

    if command in TRAINING_COMMANDS:
        if given("resume") and values.get("overwrite"):
            raise ValueError("--resume and --overwrite cannot be used together.")
        if given("resume"):
            unresolved.append(
                "resume runs reuse recorded settings unless an option is explicit"
            )

    if command == "pretrain":
        if given("resume"):
            unresolved.append(
                "patch-option applicability follows the resumed checkpoint's saved "
                "patch mode, which is only known at execution"
            )
        else:
            validate_pretrain_patch_applicability(
                values, explicit_options=explicit_options
            )

    if command == "finetune":
        loss = str(values.get("loss") or "")
        loss_explicit = "loss" in explicit_options
        # A resumed run or a pseudo-label round takes the loss from its source
        # checkpoint; only then is an omitted loss genuinely unresolved. A fresh
        # run uses the declared default, so applicability is checkable here.
        inherited_loss = not loss_explicit and (
            given("resume") or given("pseudo_label_from")
        )
        strict_loss = bool(loss) and not inherited_loss
        for name in ("subcenters", "compact_weight", "supcon_temperature"):
            applies = rules[name]["applies_to"]
            option = f"--{name.replace('_', '-')}"
            if name not in explicit_options:
                continue
            if strict_loss:
                if loss not in applies:
                    raise ValueError(f"{option} does not apply to --loss {loss}.")
            else:
                unresolved.append(
                    f"{option} applicability depends on the effective loss, which "
                    "is inherited from the checkpoint at execution"
                )
        if strict_loss and (
            "subcenters" not in explicit_options
            and loss in rules["subcenters"]["applies_to"]
        ):
            unresolved.append(
                "subcenters defaults to the recorded or declared value at execution"
            )
        if not loss_explicit and inherited_loss:
            unresolved.append(
                "the effective --loss is inherited from the resumed or pseudo-label "
                "source checkpoint, so loss-specific applicability stays unresolved"
            )
        if not loss_explicit and not loss:
            unresolved.append(
                "the effective --loss is not fixed by this proposal, so "
                "loss-specific applicability stays unresolved until the run resolves it"
            )
        check_loss_option_combinations(
            long_tail=str(values.get("long_tail") or "none"),
            loss=loss,
            pseudo_label_from=str(values.get("pseudo_label_from") or ""),
        )
        # Mirrors the CLI wrapper in otuformer.cli.finetune: these two cases are
        # rejected outright, not silently ignored.
        from otuformer.cli.finetune import PSEUDO_RULE_OPTIONS

        supplied = sorted(set(explicit_options) & set(PSEUDO_RULE_OPTIONS))
        if supplied and not given("pseudo_label_from") and not given("resume"):
            raise ValueError(
                "Pseudo rule options require --pseudo-label-from or a finetune#2 "
                f"--resume: {', '.join(supplied)}."
            )
        if supplied and given("resume"):
            raise ValueError(
                "Resume takes the pseudo rule constants from its checkpoint; do "
                f"not resupply: {', '.join(supplied)}."
            )

    if command == "extract":
        checkpoint = values.get("checkpoint")
        onnx_path = values.get("onnx_path")
        if checkpoint in (None, "") and onnx_path in (None, ""):
            raise ValueError(
                "extract requires --checkpoint or --onnx-path."
            )
        if onnx_path not in (None, "") and values.get("token_mode") not in (None, "cls"):
            raise ValueError("ONNX inference only supports --token-mode cls.")

    if command == "diversity":
        validate_diversity_sources(
            values.get("assignments"), values.get("otu_table_csv")
        )
        if given("embeddings") and given("tree"):
            unresolved.append(_LEGACY_EMBEDDINGS_NOTE)

    if command == "cam" and given("checkpoint"):
        unresolved.append("CAM target-layer selection depends on the checkpoint")

    if command == "annotate":
        unresolved.append(
            "annotate requires a selected partition and an approved correction table"
        )

    if command in _OUTPUT_DIR_COMMANDS and not values.get("out_dir"):
        unresolved.append("an explicit --out-dir is required in every proposal")

    if command == "update":
        unresolved.extend(
            str(item) for item in rules.get("yes", {}).get("unresolved", ())
        )

    return unresolved


# Parameter names per command, used only to reject unknown keys. Derived lazily
# from the runtime registry so a new CLI option never becomes an unknown-key
# error here, and so importing this module stays free of CLI-framework setup.
@lru_cache(maxsize=1)
def _command_parameter_names() -> dict[str, tuple[str, ...]]:
    from otuformer.cli_schema import export_cli_schema

    schema = export_cli_schema()
    return {
        command: tuple(item["name"] for item in entry["parameters"])
        for command, entry in schema["commands"].items()
    }
