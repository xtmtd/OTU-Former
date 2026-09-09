"""Shared CLI helpers."""

from __future__ import annotations

import typer

SIZE_EXAMPLES = "Common examples: 224, 384, 448 (e.g. 518 for patch-14 models)."


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
