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