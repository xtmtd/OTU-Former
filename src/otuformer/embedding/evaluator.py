"""Embedding quality evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _make_stratified_cv(
    labels: np.ndarray, max_splits: int = 5
):
    """Return a shuffled StratifiedKFold, or None when stratified CV is impossible.

    The fold count is bounded by the smallest class count (not by the number of
    classes), so a two-class dataset with enough samples per class still gets
    the full ``max_splits`` folds.
    """
    from sklearn.model_selection import StratifiedKFold

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        return None
    min_class_count = int(np.min(counts))
    if min_class_count < 2:
        return None
    return StratifiedKFold(
        n_splits=min(max_splits, min_class_count),
        shuffle=True,
        random_state=42,
    )


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 5, 20],
) -> dict[str, float | None]:
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize

    x = normalize(embeddings, norm="l2")
    result: dict[str, float | None] = {f"kNN_Acc_k{k}": None for k in k_values}
    cv = _make_stratified_cv(labels)
    if cv is None:
        return result
    max_train_size = len(labels) - int(np.ceil(len(labels) / cv.n_splits))
    for k in k_values:
        key = f"kNN_Acc_k{k}"
        if k < 1 or k > max_train_size:
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        try:
            scores = cross_val_score(knn, x, labels, cv=cv)
            result[key] = float(np.nanmean(scores))
        except Exception:
            result[key] = None
    return result


def compute_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float | None]:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    n = len(labels)
    result: dict[str, float | None] = {f"Recall@{k}": None for k in k_values}
    if n < 2:
        return result
    x = normalize(embeddings, norm="l2")
    sim = cosine_similarity(x)
    positions = np.arange(n)
    for k in k_values:
        key = f"Recall@{k}"
        if k < 1 or k > n - 1:
            continue
        correct = 0
        for i in range(n):
            candidates = positions[positions != i]
            ranked = candidates[np.argsort(sim[i, candidates])[::-1][:k]]
            if labels[i] in labels[ranked]:
                correct += 1
        result[key] = float(correct / n)
    return result


def compute_map(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    n = len(labels)
    if n < 2:
        return None
    x = normalize(embeddings, norm="l2")
    sim = cosine_similarity(x)
    positions = np.arange(n)
    aps: list[float] = []
    for i in range(n):
        candidates = positions[positions != i]
        ranked = candidates[np.argsort(sim[i, candidates])[::-1]]
        relevant = labels[ranked] == labels[i]
        n_rel = int(relevant.sum())
        if n_rel == 0:
            continue
        precisions = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        aps.append(float((precisions * relevant).sum() / n_rel))
    if not aps:
        return None
    return float(np.mean(aps))


def compute_clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float | None]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    from sklearn.preprocessing import normalize

    keys = ("NMI", "ARI", "AMI", "Silhouette_Score", "Purity")
    unique_labels, label_codes = np.unique(labels, return_inverse=True)
    n_clusters = len(unique_labels)
    x = normalize(embeddings, norm="l2")
    if (
        n_clusters < 2
        or len(x) < 2
        or np.unique(x, axis=0).shape[0] < n_clusters
    ):
        return {key: None for key in keys}

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(x)

    purity = sum(
        np.bincount(label_codes[pred == c], minlength=n_clusters).max()
        for c in range(n_clusters)
    ) / len(label_codes)
    if len(label_codes) > n_clusters:
        sil: float | None = float(silhouette_score(x, label_codes, metric="cosine"))
    else:
        sil = None
    return {
        "NMI": float(normalized_mutual_info_score(label_codes, pred)),
        "ARI": float(adjusted_rand_score(label_codes, pred)),
        "AMI": float(adjusted_mutual_info_score(label_codes, pred)),
        "Silhouette_Score": sil,
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


def compute_linear_probing_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float | None]:
    """Ordinary and balanced linear-probe accuracy from one shared CV pass."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import normalize

    x = normalize(embeddings, norm="l2")
    result: dict[str, float | None] = {
        "Linear_Probing_Acc": None,
        "Linear_Probing_Balanced_Acc": None,
    }
    cv = _make_stratified_cv(labels)
    if cv is None:
        return result
    clf = LogisticRegression(max_iter=1000, random_state=42)
    try:
        scores = cross_validate(
            clf,
            x,
            labels,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
            },
        )
        result["Linear_Probing_Acc"] = float(np.mean(scores["test_accuracy"]))
        result["Linear_Probing_Balanced_Acc"] = float(
            np.mean(scores["test_balanced_accuracy"])
        )
    except Exception:
        pass
    return result


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

    if len(embeddings) < 3 or (labels is not None and len(labels) < 2):
        return

    x = normalize(embeddings, norm="l2")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(max(2, n_neighbors), len(x) - 1),
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
