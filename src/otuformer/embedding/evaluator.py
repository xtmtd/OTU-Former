"""Embedding quality evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _safe_cv_splits(labels: np.ndarray, max_cv: int = 5) -> int:
    n_classes = len(np.unique(labels))
    return max(1, min(max_cv, n_classes))


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize

    x = normalize(embeddings, norm="l2")
    result: dict[str, float] = {}
    cv = _safe_cv_splits(labels)
    if cv < 2:
        return {f"kNN_Acc_k{k}": 0.0 for k in k_values}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        try:
            scores = cross_val_score(knn, x, labels, cv=cv)
            result[f"kNN_Acc_k{k}"] = float(scores.mean())
        except Exception:
            result[f"kNN_Acc_k{k}"] = 0.0
    return result


def compute_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    x = normalize(embeddings, norm="l2")
    sim = cosine_similarity(x)
    np.fill_diagonal(sim, -np.inf)
    result: dict[str, float] = {}
    for k in k_values:
        k_actual = max(1, min(k, len(labels) - 1))
        correct = sum(
            labels[i] in labels[np.argsort(sim[i])[-k_actual:]]
            for i in range(len(labels))
        )
        result[f"Recall@{k}"] = float(correct / len(labels))
    return result


def compute_map(embeddings: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    x = normalize(embeddings, norm="l2")
    sim = cosine_similarity(x)
    np.fill_diagonal(sim, -np.inf)
    aps: list[float] = []
    for i in range(len(labels)):
        ranked = np.argsort(sim[i])[::-1]
        relevant = labels[ranked] == labels[i]
        n_rel = int(relevant.sum())
        if n_rel == 0:
            continue
        precisions = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        aps.append(float((precisions * relevant).sum() / n_rel))
    return float(np.mean(aps)) if aps else 0.0


def compute_clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    from sklearn.preprocessing import normalize

    unique_labels, label_codes = np.unique(labels, return_inverse=True)
    n_clusters = len(unique_labels)
    x = normalize(embeddings, norm="l2")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(x)

    purity = sum(
        np.bincount(label_codes[pred == c]).max() for c in range(n_clusters)
    ) / len(label_codes)
    if len(x) < 2 or len(np.unique(label_codes)) < 2:
        sil = 0.0
    elif len(label_codes) > n_clusters:
        sil = float(silhouette_score(x, label_codes, metric="cosine"))
    else:
        sil = 0.0
    return {
        "NMI": float(normalized_mutual_info_score(label_codes, pred)),
        "ARI": float(adjusted_rand_score(label_codes, pred)),
        "AMI": float(adjusted_mutual_info_score(label_codes, pred)),
        "Silhouette": sil,
        "Purity": float(purity),
    }


def compute_metric_learning_diagnostics(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    from sklearn.preprocessing import normalize

    norms = np.linalg.norm(embeddings, axis=1)
    unique_labels = np.unique(labels)
    intra_vars: list[float] = []
    inter_dists: list[float] = []
    x = normalize(embeddings, norm="l2")

    for c in unique_labels:
        mask = labels == c
        if mask.sum() > 1:
            class_embs = x[mask]
            sim = class_embs @ class_embs.T
            tri = np.triu_indices(len(class_embs), k=1)
            intra_vars.append(float(1 - sim[tri].mean()))

    for i, c1 in enumerate(unique_labels):
        for c2 in unique_labels[i + 1 :]:
            e1 = x[labels == c1].mean(axis=0)
            e2 = x[labels == c2].mean(axis=0)
            inter_dists.append(float(1 - e1 @ e2))

    return {
        "Intra_Class_Var": float(np.mean(intra_vars)) if intra_vars else 0.0,
        "Inter_Class_Dist": float(np.mean(inter_dists)) if inter_dists else 0.0,
        "Embedding_Norm_Mean": float(norms.mean()),
        "Embedding_Norm_Std": float(norms.std()),
    }


def compute_linear_probing(embeddings: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import normalize

    x = normalize(embeddings, norm="l2")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv = _safe_cv_splits(labels)
    if cv < 2:
        return 0.0
    try:
        scores = cross_val_score(clf, x, labels, cv=cv)
        return float(scores.mean())
    except Exception:
        return 0.0


def run_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray],
    out_path: Path,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    max_classes: int = 20,
    title: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import umap.umap_ as umap
    from sklearn.preprocessing import normalize

    if labels is not None and len(labels) < 2:
        return

    x = normalize(embeddings, norm="l2")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        n_jobs=1,
        verbose=False,
    )
    proj = reducer.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels is not None:
        labels_str = np.asarray(labels).astype(str)
        selected_labels = np.unique(labels_str)
        if max_classes > 0 and len(selected_labels) > max_classes:
            values, counts = np.unique(labels_str, return_counts=True)
            order = np.argsort(counts)[::-1]
            selected_labels = values[order][:max_classes]
            keep_mask = np.isin(labels_str, selected_labels)
            proj = proj[keep_mask]
            labels_str = labels_str[keep_mask]

        if len(selected_labels) <= 10:
            palette = sns.color_palette("tab10", len(selected_labels))
        elif len(selected_labels) <= 20:
            palette = sns.color_palette("tab20", len(selected_labels))
        else:
            palette = sns.color_palette("husl", len(selected_labels))

        cmap = dict(zip(selected_labels.tolist(), palette))
        for lbl in selected_labels:
            mask = labels_str == lbl
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                s=30,
                color=cmap[str(lbl)],
                label=str(lbl),
                alpha=0.85,
                edgecolors="none",
            )

        max_labels_per_col = 30
        n_cols = max(1, int(np.ceil(len(selected_labels) / max_labels_per_col)))
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            title="Label",
            frameon=True,
            fontsize="small",
            title_fontsize="small",
            ncol=n_cols,
            columnspacing=0.8,
            handlelength=1.0,
            borderaxespad=0.2,
        )
        legend._legend_box.align = "left"
        legend_space = min(0.32, 0.08 * n_cols)
        right_rect = 1 - legend_space
        fig.tight_layout(rect=[0, 0, right_rect, 1])
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=30, alpha=0.7, edgecolors="none")
        fig.tight_layout()

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2" if n_components >= 2 else "")
    ax.set_title(title if title is not None else "UMAP")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
