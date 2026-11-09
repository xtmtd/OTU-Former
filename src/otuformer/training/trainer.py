"""Training entry points for pretrain and finetune.

The loops live in ``pretrain_loop`` and ``finetune_loop``. This module
keeps the historical ``otuformer.training.trainer`` import path working for
the two run entry points and for the helpers external code imports; it is a
compatibility surface, not a second implementation. Patching a name here
does not affect a caller that already bound the function from its owner.
"""

from __future__ import annotations

from .finetune_loop import run_finetune
from .pretrain_loop import _freeze_backbone_blocks, run_pretrain

__all__ = [
    "run_pretrain",
    "run_finetune",
    "_freeze_backbone_blocks",
]
