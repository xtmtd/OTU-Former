"""Alpha diversity metrics from OTU assignment tables."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def filter_by_min_abundance(
    assignments: pd.DataFrame, min_abundance: int
) -> pd.DataFrame:
    if min_abundance <= 0:
        return assignments.copy()
    counts = assignments["cluster"].value_counts()
    keep = counts[counts >= min_abundance].index
    return assignments[assignments["cluster"].isin(keep)].copy()


def _abundance_vector(assignments: pd.DataFrame) -> np.ndarray:
    return np.array(sorted(assignments["cluster"].value_counts().values, reverse=True))


def _fisher_alpha(s: int, n: int, max_iter: int = 100, tol: float = 1e-6) -> float:
    if n <= 0 or s <= 0:
        return 0.0
    alpha = s / max(math.log(1 + n), 1e-12)
    for _ in range(max_iter):
        f = alpha * math.log(1 + n / alpha) - s
        df = math.log(1 + n / alpha) - n / (n + alpha)
        if abs(df) < 1e-12:
            break
        alpha_new = alpha - f / df
        if alpha_new <= 0:
            alpha_new = alpha / 2
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    return float(alpha)


def compute_alpha_diversity(assignments: pd.DataFrame) -> dict[str, float]:
    counts = _abundance_vector(assignments)
    n = int(counts.sum())
    s = int(len(counts))
    if n == 0 or s == 0:
        return {
            "Richness": 0.0,
            "Chao1": 0.0,
            "ACE": 0.0,
            "Margalef": 0.0,
            "Menhinick": 0.0,
            "Shannon": 0.0,
            "Simpson": 0.0,
            "InverseSimpson": 0.0,
            "Brillouin": 0.0,
            "Fisher_alpha": 0.0,
            "Pielou_J": 0.0,
            "Heip_E": 0.0,
            "Berger_Parker": 0.0,
            "Hill_q0": 0.0,
            "Hill_q1": 0.0,
            "Hill_q2": 0.0,
        }

    props = counts / n
    shannon = -float(np.sum(props * np.log(props + 1e-300)))
    simpson = float(1.0 - np.sum(props**2))
    inv_simpson = float(1.0 / np.sum(props**2)) if np.sum(props**2) > 0 else 0.0
    pielou_j = shannon / math.log(s) if s > 1 else 0.0
    heip_e = (math.exp(shannon) - 1) / (s - 1) if s > 1 else 0.0
    berger_parker = float(counts[0] / n)
    margalef = (s - 1) / math.log(n) if n > 1 else 0.0
    menhinick = s / math.sqrt(n) if n > 0 else 0.0

    n1 = int((counts == 1).sum())
    n2 = int((counts == 2).sum())
    chao1 = (
        float(s + (n1 * (n1 - 1)) / (2 * (n2 + 1)))
        if n2 > 0
        else float(s + n1 * (n1 - 1) / 2)
    )

    rare = counts[counts <= 10]
    s_rare = len(rare)
    s_abund = s - s_rare
    n_rare = int(rare.sum())
    f1 = int((counts == 1).sum())
    if n_rare > 0 and s_rare > 0:
        c_ace = 1 - f1 / n_rare
        if c_ace > 0 and n_rare > 1:
            gamma_sq = max(
                0.0,
                s_rare
                / c_ace
                * sum(k * (k - 1) * (counts == k).sum() for k in range(1, 11))
                / (n_rare * (n_rare - 1))
                - 1,
            )
            ace = float(s_abund + s_rare / c_ace + f1 * gamma_sq / c_ace)
        else:
            ace = float(s)
    else:
        ace = float(s)

    brillouin = (
        (math.lgamma(n + 1) - sum(math.lgamma(int(c) + 1) for c in counts)) / n
        if n > 0
        else 0.0
    )
    fisher_alpha = _fisher_alpha(s, n)

    hill_q0 = float(s)
    hill_q1 = float(math.exp(shannon))
    hill_q2 = float(inv_simpson)

    return {
        "Richness": float(s),
        "Chao1": chao1,
        "ACE": ace,
        "Margalef": margalef,
        "Menhinick": menhinick,
        "Shannon": shannon,
        "Simpson": simpson,
        "InverseSimpson": inv_simpson,
        "Brillouin": brillouin,
        "Fisher_alpha": fisher_alpha,
        "Pielou_J": pielou_j,
        "Heip_E": heip_e,
        "Berger_Parker": berger_parker,
        "Hill_q0": hill_q0,
        "Hill_q1": hill_q1,
        "Hill_q2": hill_q2,
    }


def compute_mpd(assignments: pd.DataFrame, tree_newick_path: Path) -> float:
    try:
        from skbio import TreeNode
        from skbio.diversity import alpha_diversity
    except Exception as exc:
        raise ImportError("scikit-bio required for MPD computation") from exc

    tree = TreeNode.read(str(tree_newick_path))
    cluster_counts = assignments["cluster"].value_counts()
    otu_ids = list(cluster_counts.index)
    counts_arr = np.array([cluster_counts.get(o, 0) for o in otu_ids], dtype=int)
    try:
        tree_pruned = tree.shear(otu_ids)
        result = alpha_diversity(
            "faith_pd", counts_arr.reshape(1, -1), otu_ids, tree=tree_pruned
        )
        return float(result[0])
    except Exception:
        return float("nan")


def diversity_table(
    assignments: pd.DataFrame,
    min_abundances: list[int],
    tree_newick_path: Optional[Path] = None,
) -> pd.DataFrame:
    records: dict[str, dict[str, float]] = {}
    for min_ab in min_abundances:
        filtered = filter_by_min_abundance(assignments, min_ab)
        col = f"min_abundance_{min_ab}"
        if len(filtered) == 0:
            records[col] = {}
            continue
        metrics = compute_alpha_diversity(filtered)
        if tree_newick_path is not None:
            try:
                metrics["MPD"] = compute_mpd(filtered, tree_newick_path)
            except Exception:
                metrics["MPD"] = float("nan")
        records[col] = metrics
    df = pd.DataFrame(records)
    df.index.name = "index"
    return df
