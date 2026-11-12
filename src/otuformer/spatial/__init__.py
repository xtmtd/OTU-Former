"""Explicit spatial products: geometry, complete feature grids and diagnostics.

Importing this package must stay light: no torch, no ONNX Runtime, no CLI. The
submodules are imported on demand by the callers that need them.
"""

from __future__ import annotations

__all__ = [
    "diagnostics",
    "records",
    "storage",
]
