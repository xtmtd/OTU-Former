"""Partition scanning, export, and quality metrics for UPGMA linkage trees."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy


def build_partitions_from_linkage(
    z: np.ndarray,
    thresholds: list[float],
) -> OrderedDict[float, np.ndarray]:
    parts: OrderedDict[float, np.ndarray] = OrderedDict()
    for th in sorted(thresholds):
        parts[float(th)] = hierarchy.fcluster(z, t=th, criterion="distance")
    return parts


def export_partition_tables(
    ids: list[str],
    partitions: OrderedDict[float, np.ndarray],
    out_dir: Path,
    prefix: str = "OTU",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for th, labels in partitions.items():
        clusters: dict[int, list[str]] = defaultdict(list)
        for lab, tip in zip(labels, ids):
            clusters[int(lab)].append(tip)

        sorted_clusters = sorted(clusters.items(), key=lambda x: (-len(x[1]), x[1][0]))

        rows_summary = []
        rows_assign = []
        for idx, (_, tips) in enumerate(sorted_clusters, start=1):
            cname = f"{prefix}_{idx}" if prefix else f"OTU_{idx}"
            rows_summary.append(
                {"cluster": cname, "size": len(tips), "members": ";".join(tips)}
            )
            for tip in tips:
                rows_assign.append({"id": tip, "cluster": cname})

        th_tag = f"{th:.4f}".rstrip("0").rstrip(".")
        pd.DataFrame(rows_summary).to_csv(
            out_dir / f"partition_{th_tag}_summary.csv", index=False
        )
        pd.DataFrame(rows_assign).to_csv(
            out_dir / f"partition_{th_tag}_assignments.csv", index=False
        )


def _bcubed_f(true: np.ndarray, pred: np.ndarray) -> float:
    """Legacy scalar BCubed-F (kept for backward compat; prefers vectorised version)."""
    _, _, f = _bcubed_f_full(true, pred)
    return f


def _splitting_lumping(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Mean OTUs per true species (splitting) and true species per OTU (lumping)."""
    from collections import defaultdict

    species_to_otus: dict = defaultdict(set)
    otu_to_species: dict = defaultdict(set)
    for t, p in zip(true, pred):
        species_to_otus[t].add(p)
        otu_to_species[p].add(t)
    splitting = float(np.mean([len(v) for v in species_to_otus.values()]))
    lumping = float(np.mean([len(v) for v in otu_to_species.values()]))
    return splitting, lumping


def _wss(x: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster sum of squares."""
    total = 0.0
    for c in np.unique(labels):
        cluster = x[labels == c]
        if len(cluster) > 0:
            total += float(np.sum((cluster - cluster.mean(axis=0)) ** 2))
    return total


def _silhouette_cluster(
    x: np.ndarray, pred: np.ndarray, max_per_cluster: int = 1000
) -> float:
    """Approximate silhouette score using cluster-stratified sampling."""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    x = normalize(x, norm="l2")
    unique = np.unique(pred)
    if len(unique) < 2:
        return 0.0

    if len(x) <= 5000 or len(unique) <= 5:
        try:
            return float(silhouette_score(x, pred, metric="cosine"))
        except Exception:
            return 0.0

    rng = np.random.default_rng(42)
    idx: list[int] = []
    for c in unique:
        ci = np.where(pred == c)[0]
        n = min(len(ci), max_per_cluster)
        idx.extend(rng.choice(ci, size=n, replace=False).tolist())
    idx_arr = np.array(idx)
    try:
        return float(silhouette_score(x[idx_arr], pred[idx_arr], metric="cosine"))
    except Exception:
        return 0.0


def compute_partition_metrics(
    partitions: OrderedDict[float, np.ndarray],
    labels_true: np.ndarray,
    x: Optional[np.ndarray] = None,
) -> dict[float, dict[str, float]]:
    """Compute full set of partition quality metrics matching ref/embeddings_tree*.py.

    Metrics computed (all per cutoff):
    - NMI, ARI, AMI, V_measure (homogeneity/completeness/v)
    - BCubed_F, BCubed_precision, BCubed_recall
    - purity, n_OTUs, OTU_species_ratio
    - splitting_index, lumping_index
    - silhouette_cluster (if *x* provided)
    - WSS (if *x* provided)
    - silhouette_species (computed once, same for all cutoffs, if *x* provided)
    """
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        homogeneity_completeness_v_measure,
        normalized_mutual_info_score,
    )

    n_true_species = len(np.unique(labels_true))

    # silhouette_species: computed once on true labels (same for all cutoffs)
    sil_species: Optional[float] = None
    if x is not None:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import normalize as sk_normalize

        if len(x) > 1 and n_true_species >= 2:
            try:
                x_norm = sk_normalize(x, norm="l2")
                sil_species = float(
                    silhouette_score(x_norm, labels_true, metric="cosine")
                )
            except Exception:
                sil_species = 0.0

    results: dict[float, dict[str, float]] = {}
    for th, pred in partitions.items():
        h, c, v = homogeneity_completeness_v_measure(labels_true, pred)
        _, _, bcubed_f = _bcubed_f_full(labels_true, pred)
        splitting, lumping = _splitting_lumping(labels_true, pred)
        n_otus = int(len(np.unique(pred)))

        row: dict[str, float] = {
            "NMI": float(normalized_mutual_info_score(labels_true, pred)),
            "ARI": float(adjusted_rand_score(labels_true, pred)),
            "AMI": float(adjusted_mutual_info_score(labels_true, pred)),
            "V_measure": float(v),
            "homogeneity": float(h),
            "completeness": float(c),
            "BCubed_F": float(bcubed_f),
            "purity": float(
                sum(
                    np.bincount(labels_true[pred == c2]).max() for c2 in np.unique(pred)
                )
                / len(labels_true)
            ),
            "n_OTUs": float(n_otus),
            "n_true_species": float(n_true_species),
            "OTU_species_ratio": float(n_otus / max(1, n_true_species)),
            "splitting_index": splitting,
            "lumping_index": lumping,
        }
        if x is not None:
            row["silhouette_cluster"] = _silhouette_cluster(x, pred)
            row["WSS"] = _wss(x, pred)
        if sil_species is not None:
            row["silhouette_species"] = sil_species
        results[th] = row
    return results


def _bcubed_f_full(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """BCubed precision, recall, F-score (vectorised)."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    true_pairs = true[:, None] == true[None, :]
    pred_pairs = pred[:, None] == pred[None, :]
    intersection = true_pairs & pred_pairs
    prec = (intersection.sum(axis=1) / pred_pairs.sum(axis=1)).mean()
    rec = (intersection.sum(axis=1) / true_pairs.sum(axis=1)).mean()
    fscore = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(fscore)


def two_stage_threshold_scan(
    z: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
    coarse_min: float = 0.05,
    coarse_max: float = 1.0,
    coarse_step: float = 0.05,
    fine_step: float = 0.01,
    window_expand: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    coarse = np.arange(coarse_min, coarse_max + 1e-9, coarse_step)
    coarse = np.round(coarse, 6)

    if labels_true is None:
        return coarse, coarse

    parts_coarse = build_partitions_from_linkage(z, coarse.tolist())
    metrics = compute_partition_metrics(parts_coarse, labels_true)

    metric_keys = ["NMI", "ARI", "AMI", "BCubed_F", "V_measure"]
    optima = []
    for key in metric_keys:
        best_th = max(metrics.keys(), key=lambda t: metrics[t].get(key, 0.0))
        optima.append(best_th)

    window_low = max(coarse_min, min(optima) - window_expand)
    window_high = min(coarse_max, max(optima) + window_expand)
    fine = np.arange(window_low, window_high + 1e-9, fine_step)
    fine = np.round(fine, 6)
    return coarse, fine
