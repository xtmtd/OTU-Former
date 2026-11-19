"""One seeded, sorted row-selection rule for evaluation and extraction."""

from __future__ import annotations

import numpy as np


def sorted_sample_indices(count: int, max_samples: int, seed: int) -> np.ndarray:
    """Indices of the same seeded, sorted subset the row cap always produced.

    A nonpositive cap, or a cap at least as large as the population, keeps every
    row in order. Only the index rule is shared: callers keep their own
    population, labels, duplicate policy and any excluded complement.
    """
    if max_samples <= 0 or count <= max_samples:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_samples, replace=False))
