"""UPGMA tree construction and Newick export."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    support_dict: dict[frozenset, float] | None = None,
) -> str:
    newick = upgma_to_newick_string(z, ids, support_dict=support_dict)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(newick, encoding="utf-8")
    return newick


def upgma_to_newick_string(
    z: np.ndarray,
    ids: list[str],
    support_dict: dict[frozenset, float] | None = None,
) -> str:
    tree = hierarchy.to_tree(z, rd=False)

    def build(node, parent_dist):
        node_dist = node.dist if node.dist is not None else 0.0
        branch_length = max(parent_dist - node_dist, 0.0)

        if node.left is None and node.right is None:
            return f"{ids[node.id]}:{branch_length:.10f}", {ids[node.id]}

        left_str, left_clade = build(node.left, node_dist)
        right_str, right_clade = build(node.right, node_dist)
        clade = left_clade | right_clade

        support_label = ""
        if support_dict is not None:
            support = support_dict.get(frozenset(clade))
            if support is not None:
                support_label = f"{support:.1f}"

        inner = f"({left_str},{right_str}){support_label}"
        if parent_dist == node_dist:
            return inner, clade
        return f"{inner}:{branch_length:.10f}", clade

    root_dist = tree.dist if tree.dist is not None else 0.0
    newick, _ = build(tree, root_dist)
    return f"{newick};"


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
    x: np.ndarray,
    ids: list[str],
    base_z: np.ndarray,
    distance: str = "cosine",
    support_mode: str = "subsample",
    n_replicates: int = 100,
    subsample_ratio: float = 0.8,
    random_state: int = 42,
    save_trees_path: Path | None = None,
    n_jobs: int = 1,
) -> dict[frozenset, float]:
    from otuformer.delineation.distance import (
        compute_cosine_distances,
        compute_euclidean_distances,
    )

    def ensure_nonzero_rows(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1)
        zero_mask = norms < eps
        if zero_mask.any():
            arr = arr.copy()
            arr[zero_mask, 0] = eps
        return arr

    def bootstrap_once(seed: int):
        rng_local = np.random.default_rng(seed)
        if support_mode == "bootstrap":
            cols = rng_local.choice(x.shape[1], size=x.shape[1], replace=True)
        else:
            cols = rng_local.choice(x.shape[1], size=subset_size, replace=False)
        x_sub = ensure_nonzero_rows(x[:, cols])
        if distance == "euclidean":
            norms = np.linalg.norm(x_sub, axis=1, keepdims=True)
            x_sub = x_sub / np.maximum(norms, 1e-12)
            d_sub = compute_euclidean_distances(x_sub)
        else:
            d_sub = compute_cosine_distances(x_sub)
        z_boot = build_upgma(d_sub)
        return _extract_clades(z_boot, ids), upgma_to_newick_string(z_boot, ids)

    rng = np.random.default_rng(random_state)
    subset_size = max(2, int(round(x.shape[1] * subsample_ratio)))
    ref_clades = _extract_clades(base_z, ids)
    counts: dict[frozenset, int] = {c: 0 for c in ref_clades}
    trees: list[str] = []
    seeds = [int(s) for s in rng.integers(0, 2**32 - 1, size=n_replicates)]

    progress_every = max(1, n_replicates // 20)
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for i, (boot_clades, newick) in enumerate(
                ex.map(bootstrap_once, seeds), start=1
            ):
                if save_trees_path is not None:
                    trees.append(newick)
                for clade in ref_clades:
                    if clade in boot_clades:
                        counts[clade] += 1
                if i % progress_every == 0 or i == n_replicates:
                    pct = int(round(i * 100 / n_replicates))
                    print(f"  Support replicate progress: {i}/{n_replicates} ({pct}%)")
    else:
        for i, seed in enumerate(seeds, start=1):
            boot_clades, newick = bootstrap_once(seed)
            if save_trees_path is not None:
                trees.append(newick)
            for clade in ref_clades:
                if clade in boot_clades:
                    counts[clade] += 1
            if i % progress_every == 0 or i == n_replicates:
                pct = int(round(i * 100 / n_replicates))
                print(f"  Support replicate progress: {i}/{n_replicates} ({pct}%)")

    if save_trees_path is not None:
        save_trees_path.parent.mkdir(parents=True, exist_ok=True)
        save_trees_path.write_text("\n".join(trees) + "\n", encoding="utf-8")

    return {c: 100.0 * cnt / n_replicates for c, cnt in counts.items()}


def build_nj_tree(dist_matrix: np.ndarray, ids: list[str]):
    """Build an unrooted NJ tree from a square distance matrix.

    Negative branch lengths (possible when the matrix is non-additive)
    are clipped to zero.  Returns a ``skbio.TreeNode``.
    """
    from skbio import DistanceMatrix
    from skbio.tree import nj

    # Ensure exact symmetry and zero diagonal
    d = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(d, 0.0)
    dm = DistanceMatrix(d, ids=[str(i) for i in ids])
    tree = nj(dm)
    # Clip negative branch lengths
    for node in tree.traverse():
        if node.length is not None and node.length < 0.0:
            node.length = 0.0
    return tree


def compute_nj_bootstrap_support(
    centroids: np.ndarray,
    ids: list[str],
    support_mode: str = "subsample",
    n_replicates: int = 100,
    subsample_ratio: float = 0.8,
    random_state: int = 42,
    save_trees_path: Path | None = None,
    n_jobs: int = 1,
) -> dict[frozenset, float]:
    """Compute bootstrap support for an NJ tree via majority-rule consensus.

    Generates ``n_replicates`` NJ trees from column-subsampled/bootstrapped
    centroid embeddings, then calls ``skbio.tree.majority_rule`` to obtain
    support values.

    Parameters
    ----------
    centroids:
        L2-normalised OTU centroid embeddings, shape (k, d).
    ids:
        OTU labels matching rows of ``centroids``.
    support_mode:
        ``"subsample"`` — random column subset without replacement.
        ``"bootstrap"`` — column resample with replacement.
    n_replicates:
        Number of bootstrap replicates.
    subsample_ratio:
        Fraction of dimensions used when ``support_mode="subsample"``.
    random_state:
        Seed for reproducibility.
    save_trees_path:
        If given, write replicate Newick strings (one per line) here.
    n_jobs:
        Parallel workers via ``ThreadPoolExecutor``.

    Returns
    -------
    dict mapping each internal clade ``frozenset`` (tip names) to its
    support percentage (0–100).
    """
    from skbio.tree import majority_rule
    from otuformer.delineation.distance import compute_cosine_distances

    def _resample_once(seed: int):
        rng_local = np.random.default_rng(seed)
        if support_mode == "bootstrap":
            cols = rng_local.choice(centroids.shape[1], size=centroids.shape[1], replace=True)
        else:
            cols = rng_local.choice(centroids.shape[1], size=subset_size, replace=False)
        c_sub = centroids[:, cols].copy()
        norms = np.linalg.norm(c_sub, axis=1, keepdims=True)
        c_sub = c_sub / np.maximum(norms, 1e-12)
        d_sub = compute_cosine_distances(c_sub)
        return build_nj_tree(d_sub, ids)

    rng = np.random.default_rng(random_state)
    subset_size = max(2, int(round(centroids.shape[1] * subsample_ratio)))
    seeds = [int(s) for s in rng.integers(0, 2**32 - 1, size=n_replicates)]

    rep_trees = []
    progress_every = max(1, n_replicates // 20)

    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for i, tree in enumerate(ex.map(_resample_once, seeds), start=1):
                rep_trees.append(tree)
                if i % progress_every == 0 or i == n_replicates:
                    pct = int(round(i * 100 / n_replicates))
                    print(f"  NJ bootstrap progress: {i}/{n_replicates} ({pct}%)")
    else:
        for i, seed in enumerate(seeds, start=1):
            rep_trees.append(_resample_once(seed))
            if i % progress_every == 0 or i == n_replicates:
                pct = int(round(i * 100 / n_replicates))
                print(f"  NJ bootstrap progress: {i}/{n_replicates} ({pct}%)")

    if save_trees_path is not None:
        save_trees_path.parent.mkdir(parents=True, exist_ok=True)
        newicks = [str(t) for t in rep_trees]
        save_trees_path.write_text("\n".join(newicks) + "\n", encoding="utf-8")

    # majority_rule returns list of consensus trees; take the first
    consensus_list = majority_rule(rep_trees, cutoff=0.5)
    if not consensus_list:
        return {}
    consensus = consensus_list[0]

    # node.support from majority_rule is a raw count of replicates; convert to percentage
    support: dict[frozenset, float] = {}
    for node in consensus.traverse():
        if node.is_tip() or node.is_root():
            continue
        clade = frozenset(t.name for t in node.tips())
        if node.support is not None:
            support[clade] = float(node.support) / n_replicates * 100.0
    return support
