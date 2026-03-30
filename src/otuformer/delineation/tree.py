"""UPGMA tree construction and Newick export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def build_upgma(dist_matrix: np.ndarray) -> np.ndarray:
    condensed = squareform(dist_matrix, checks=False)
    return hierarchy.linkage(condensed, method="average")


def upgma_to_newick(
    z: np.ndarray,
    ids: list[str],
    out_path: Path,
) -> str:
    n = len(ids)
    nodes: dict[int, str] = {i: ids[i] for i in range(n)}
    heights: dict[int, float] = {i: 0.0 for i in range(n)}

    for step, (a, b, dist, _) in enumerate(z):
        a, b = int(a), int(b)
        node_id = n + step
        height = dist / 2.0
        branch_a = height - heights[a]
        branch_b = height - heights[b]
        nodes[node_id] = f"({nodes[a]}:{branch_a:.6f},{nodes[b]}:{branch_b:.6f})"
        heights[node_id] = height

    newick = nodes[n + len(z) - 1] + ";"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(newick, encoding="utf-8")
    return newick


def _extract_clades(z: np.ndarray, ids: list[str]) -> set[frozenset]:
    n = len(ids)
    clusters: dict[int, frozenset] = {i: frozenset([ids[i]]) for i in range(n)}
    clades: set[frozenset] = set()
    for step, (a, b, _, _) in enumerate(z):
        a, b = int(a), int(b)
        merged = clusters[a] | clusters[b]
        if 1 < len(merged) < n:
            clades.add(merged)
        clusters[n + step] = merged
    return clades


def compute_bootstrap_support(
    dist_matrix: np.ndarray,
    ids: list[str],
    n_bootstraps: int = 100,
    subsample_ratio: float = 0.8,
    random_state: int = 42,
) -> dict[frozenset, float]:
    rng = np.random.default_rng(random_state)
    n_samples = dist_matrix.shape[0]
    z_ref = build_upgma(dist_matrix)
    ref_clades = _extract_clades(z_ref, ids)
    counts: dict[frozenset, int] = {c: 0 for c in ref_clades}

    for _ in range(n_bootstraps):
        k = max(2, int(n_samples * subsample_ratio))
        idx = np.sort(rng.choice(n_samples, size=k, replace=False))
        d_boot = dist_matrix[np.ix_(idx, idx)]
        ids_boot = [ids[i] for i in idx]
        z_boot = build_upgma(d_boot)
        boot_clades = _extract_clades(z_boot, ids_boot)
        for clade in ref_clades:
            if clade.issubset(set(ids_boot)) and clade in boot_clades:
                counts[clade] += 1

    return {c: 100.0 * cnt / n_bootstraps for c, cnt in counts.items()}
