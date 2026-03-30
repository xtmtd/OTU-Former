"""Device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve device string with safe fallback and auto selection."""
    requested = (device or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            return torch.device("cpu")
    return torch.device(requested)
