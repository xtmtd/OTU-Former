"""Shared CLI helpers."""

from __future__ import annotations

import typer

SIZE_EXAMPLES = "Common examples: 224, 384, 448 (e.g. 518 for patch-14 models)."

# Published documentation. The wheel ships only ``src/otuformer``, so a
# repository-relative path would be dead for anyone who installed via pip or
# ``otuformer update``; help text points at the GitHub URL instead.
DOCS_BASE_URL = "https://github.com/xtmtd/OTU-Former/blob/main/docs"
README_URL = "https://github.com/xtmtd/OTU-Former/blob/main/README.md"


def docs_url(name: str) -> str:
    """Published URL of a command or reference document under docs/commands/."""
    return f"{DOCS_BASE_URL}/commands/{name}.md"

# Shell-completion choices. These duplicate ``otuformer.training.dataset`` so
# the CLI stays torch-free and Tab completion is instant; a test asserts they
# stay in sync with the dataset constants.
_PRETRAIN_AUGMENTATION_CHOICES = ("global-barcode", "color-robust", "legacy")
_FINETUNE_AUGMENTATION_CHOICES = ("none", "conservative")
_ORIENTATION_POLICY_CHOICES = ("invariant", "sensitive")


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
    if value is None or value == "auto":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        raise typer.BadParameter(
            f"{stage} must be 'auto' or a positive integer, got '{value}'"
        )
    return parsed


def _validate_augmentation(value: str | None, *, stage: str) -> None:
    """Reject an unknown augmentation profile before any training starts.

    The ``otuformer.training.dataset`` import stays lazy so the CLI remains
    torch-free at import time.
    """
    if value is None:
        return
    from otuformer.training.dataset import (
        FINETUNE_AUGMENTATIONS,
        PRETRAIN_AUGMENTATIONS,
    )

    allowed = PRETRAIN_AUGMENTATIONS if stage == "pretrain" else FINETUNE_AUGMENTATIONS
    if value not in allowed:
        choices = ", ".join(allowed)
        raise typer.BadParameter(
            f"Unknown {stage} augmentation profile '{value}'. Choose one of: {choices}.",
            param_hint="--augmentation",
        )


def _validate_orientation_policy(value: str | None) -> None:
    """Reject an unknown orientation policy before any training starts.

    The ``otuformer.training.dataset`` import stays lazy so the CLI remains
    torch-free at import time.
    """
    if value is None:
        return
    from otuformer.training.dataset import ORIENTATION_POLICIES

    if value not in ORIENTATION_POLICIES:
        raise typer.BadParameter(
            f"Unknown orientation policy '{value}'. Choose one of: "
            f"{', '.join(ORIENTATION_POLICIES)}.",
            param_hint="--orientation-policy",
        )


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
