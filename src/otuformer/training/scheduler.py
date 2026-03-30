"""Learning rate, EMA momentum, and teacher temperature schedules."""

from __future__ import annotations

import math


def cosine_lr_schedule(
    epoch: int,
    max_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
    warmup_epochs: int = 0,
) -> float:
    if epoch < warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def ema_momentum_schedule(
    step: int,
    total_steps: int,
    start: float = 0.995,
    end: float = 0.999,
) -> float:
    progress = step / max(total_steps, 1)
    return end - (end - start) * 0.5 * (1 + math.cos(math.pi * progress))


def teacher_temp_schedule(
    epoch: int,
    warmup_epochs: int,
    start: float = 0.04,
    end: float = 0.07,
) -> float:
    if epoch >= warmup_epochs:
        return end
    return start + (end - start) * epoch / max(warmup_epochs, 1)
