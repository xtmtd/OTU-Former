"""Shared CLI helpers."""

from __future__ import annotations

import typer

from otuformer.constants import (
    FINETUNE_AUGMENTATIONS,
    ORIENTATION_POLICIES,
    PRETRAIN_AUGMENTATIONS,
)
from otuformer.cli.constraints import (
    validate_augmentation_profile,
    validate_orientation_policy,
    validate_size_option,
)

SIZE_EXAMPLES = "Common examples: 224, 384, 448 (e.g. 518 for patch-14 models)."

# Published documentation. The wheel ships only ``src/otuformer``, so a
# repository-relative path would be dead for anyone who installed via pip or
# ``otuformer update``; help text points at the GitHub URL instead.
DOCS_BASE_URL = "https://github.com/xtmtd/OTU-Former/blob/main/docs"
README_URL = "https://github.com/xtmtd/OTU-Former/blob/main/README.md"


def docs_url(name: str) -> str:
    """Published URL of a command or reference document under docs/commands/."""
    return f"{DOCS_BASE_URL}/commands/{name}.md"

# Shell-completion choices. These are the canonical shared constants, so the
# CLI, the training code, the validators, and the schema export stay in sync by
# construction instead of by a duplicated copy.
_PRETRAIN_AUGMENTATION_CHOICES = PRETRAIN_AUGMENTATIONS
_FINETUNE_AUGMENTATION_CHOICES = FINETUNE_AUGMENTATIONS
_ORIENTATION_POLICY_CHOICES = ORIENTATION_POLICIES


def pretrain_augmentation_choices() -> list[str]:
    """Shell-completion choices for ``pretrain --augmentation``."""
    return list(_PRETRAIN_AUGMENTATION_CHOICES)


def finetune_augmentation_choices() -> list[str]:
    """Shell-completion choices for ``finetune --augmentation``."""
    return list(_FINETUNE_AUGMENTATION_CHOICES)


def orientation_policy_choices() -> list[str]:
    """Shell-completion choices for ``--orientation-policy``."""
    return list(_ORIENTATION_POLICY_CHOICES)


def _parse_size(value: str, *, stage: str) -> int | None:
    """Parse ``auto`` or a positive integer size; ``None`` means auto."""
    try:
        return validate_size_option(value, stage=stage)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None


def _validate_augmentation(value: str | None, *, stage: str) -> None:
    """Reject an unknown augmentation profile before any training starts."""
    try:
        validate_augmentation_profile(value, stage=stage)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--augmentation") from None


def _validate_orientation_policy(value: str | None) -> None:
    """Reject an unknown orientation policy before any training starts."""
    try:
        validate_orientation_policy(value)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--orientation-policy") from None


def source_is_commandline(source: object) -> bool:
    """True when a parameter came from the command line.

    Compares by name so both external Click and Typer's vendored Click
    ``ParameterSource`` enums are accepted.
    """
    return getattr(source, "name", None) == "COMMANDLINE"


def format_user_command(
    ctx: typer.Context, params: dict[str, object], command: str
) -> str:
    """Rebuild the invocation that was actually typed, for the run log.

    Only command-line-sourced parameters are echoed, so a copied line reruns
    the same command instead of its defaults. A boolean flag is written as
    ``--flag`` (never ``--flag true``), matching how Typer parses it.
    """
    parts = ["otuformer", command]
    for key, value in params.items():
        if not source_is_commandline(ctx.get_parameter_source(key)):
            continue
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(option)
            continue
        if value in (None, ""):
            continue
        parts.extend([option, str(value)])
    return " ".join(parts)
