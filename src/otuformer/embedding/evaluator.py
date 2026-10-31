"""Embedding quality evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class MetricOutcome:
    """One metric value with the reason it is unavailable, when it is."""

    __slots__ = ("value", "status", "reason", "context")

    STATUSES = ("available", "unavailable", "failed")

    def __init__(
        self,
        value: float | None,
        status: str,
        reason: str | None = None,
        context: dict | None = None,
    ) -> None:
        if status not in self.STATUSES:
            raise ValueError(f"unknown metric status: {status!r}")
        self.value = value
        self.status = status
        self.reason = reason
        self.context = dict(context or {})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MetricOutcome(value={self.value!r}, status={self.status!r}, "
            f"reason={self.reason!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MetricOutcome)
            and (self.value, self.status, self.reason) == (other.value, other.status, other.reason)
        )


def _make_stratified_cv_result(
    labels: np.ndarray, max_splits: int = 5
) -> tuple[object | None, str | None, dict]:
    """Return ``(cv, reason, context)`` for one stratified-CV decision.

    The fold count is bounded by the smallest class count (not by the number of
    classes), so a two-class dataset with enough samples per class still gets
    the full ``max_splits`` folds. The reason is produced here, at the decision
    point, so no caller has to re-derive eligibility.
    """
    from sklearn.model_selection import StratifiedKFold

    unique_labels, counts = np.unique(labels, return_counts=True)
    context = {
        "samples": int(len(labels)),
        "classes": int(len(unique_labels)),
        "min_class_count": int(np.min(counts)) if len(counts) else 0,
        "max_splits": int(max_splits),
    }
    if len(unique_labels) < 2:
        return None, "insufficient_classes", context
    min_class_count = int(np.min(counts))
    if min_class_count < 2:
        return None, "insufficient_class_count", context
    n_splits = min(max_splits, min_class_count)
    context["n_splits"] = n_splits
    context["shuffle"] = True
    context["random_state"] = 42
    return (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        None,
        context,
    )


def _make_stratified_cv(labels: np.ndarray, max_splits: int = 5):
    """Compatibility projection of :func:`_make_stratified_cv_result`."""
    cv, _reason, _context = _make_stratified_cv_result(labels, max_splits)
    return cv


def _finite_mean(scores: np.ndarray) -> float | None:
    """Mean of fold scores, or None when any fold score is non-finite.

    The cross-validation call sites pass ``error_score="raise"`` so a failed
    fold raises instead of being silently scored NaN; this guard additionally
    keeps a non-finite scorer result from reaching the CSV adapters, which only
    convert ``None`` to an empty field.
    """
    values = np.asarray(scores, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return float(np.mean(values))


def _knn_context(len_labels: int, cv) -> dict:
    max_train_size = 0
    if cv is not None:
        max_train_size = len_labels - int(np.ceil(len_labels / cv.n_splits))
    return {
        "family": "knn_accuracy",
        "distance": "cosine",
        "normalization": "l2",
        "estimator": "KNeighborsClassifier",
        "self_exclusion": "cross-validated folds",
        "max_train_size": int(max_train_size),
    }


def _compute_knn_accuracy_result(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 5, 20],
) -> dict[str, MetricOutcome]:
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize

    x = normalize(embeddings, norm="l2")
    cv, reason, cv_context = _make_stratified_cv_result(labels)
    context = {**_knn_context(len(labels), cv), **cv_context}
    outcomes: dict[str, MetricOutcome] = {}
    if cv is None:
        for k in k_values:
            outcomes[f"kNN_Acc_k{k}"] = MetricOutcome(None, "unavailable", reason, context)
        return outcomes
    max_train_size = context["max_train_size"]
    for k in k_values:
        key = f"kNN_Acc_k{k}"
        per_k = {**context, "k": int(k)}
        if k < 1 or k > max_train_size:
            outcomes[key] = MetricOutcome(None, "unavailable", "ineligible_k", per_k)
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        try:
            scores = cross_val_score(knn, x, labels, cv=cv, error_score="raise")
            value = _finite_mean(scores)
        except Exception as exc:
            outcomes[key] = MetricOutcome(
                None, "failed", "evaluation_failed", {**per_k, "error": type(exc).__name__}
            )
            continue
        if value is None:
            outcomes[key] = MetricOutcome(None, "unavailable", "nonfinite_result", per_k)
        else:
            outcomes[key] = MetricOutcome(value, "available", None, per_k)
    return outcomes


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 5, 20],
) -> dict[str, float | None]:
    return {
        key: outcome.value
        for key, outcome in _compute_knn_accuracy_result(embeddings, labels, k_values).items()
    }


def _compute_recall_at_k_result(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 5, 10],
) -> dict[str, MetricOutcome]:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    n = len(labels)
    context = {
        "family": "recall_at_k",
        "similarity": "cosine",
        "normalization": "l2",
        "self_exclusion": "query excluded from candidates",
        "samples": int(n),
    }
    if n < 2:
        return {
            f"Recall@{k}": MetricOutcome(None, "unavailable", "insufficient_samples", context)
            for k in k_values
        }
    x = normalize(embeddings, norm="l2")
    sim = cosine_similarity(x)
    positions = np.arange(n)
    outcomes: dict[str, MetricOutcome] = {}
    for k in k_values:
        key = f"Recall@{k}"
        per_k = {**context, "k": int(k)}
        if k < 1 or k > n - 1:
            outcomes[key] = MetricOutcome(None, "unavailable", "ineligible_k", per_k)
            continue
        correct = 0
        for i in range(n):
            candidates = positions[positions != i]
            ranked = candidates[np.argsort(sim[i, candidates])[::-1][:k]]
            if labels[i] in labels[ranked]:
                correct += 1
        outcomes[key] = MetricOutcome(float(correct / n), "available", None, per_k)
    return outcomes


def compute_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float | None]:
    return {
        key: outcome.value
        for key, outcome in _compute_recall_at_k_result(embeddings, labels, k_values).items()
    }


def _compute_map_result(embeddings: np.ndarray, labels: np.ndarray) -> MetricOutcome:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    n = len(labels)
    context = {
        "family": "map",
        "similarity": "cosine",
        "normalization": "l2",
        "self_exclusion": "query excluded from candidates",
        "samples": int(n),
    }
    if n < 2:
        return MetricOutcome(None, "unavailable", "insufficient_samples", context)
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
        return MetricOutcome(None, "unavailable", "no_relevant_pairs", context)
    return MetricOutcome(float(np.mean(aps)), "available", None, context)


def compute_map(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    return _compute_map_result(embeddings, labels).value


def _compute_clustering_metrics_result(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, MetricOutcome]:
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
    context = {
        "family": "clustering",
        "cluster_estimator": "KMeans",
        "cluster_distance": "euclidean (on l2-normalized vectors)",
        "clustering_input": "l2-normalized vectors",
        "silhouette_distance": "cosine",
        "silhouette_grouping": "ground-truth labels",
        "n_clusters": int(n_clusters),
        "samples": int(len(x)),
        "random_state": 42,
        "n_init": 10,
    }

    def blocked(reason: str) -> dict[str, MetricOutcome]:
        return {key: MetricOutcome(None, "unavailable", reason, context) for key in keys}

    if n_clusters < 2:
        return blocked("insufficient_classes")
    if len(x) < 2:
        return blocked("insufficient_samples")
    distinct = int(np.unique(x, axis=0).shape[0])
    context["distinct_vectors"] = distinct
    if distinct < n_clusters:
        return blocked("insufficient_distinct_vectors")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(x)

    purity = sum(
        np.bincount(label_codes[pred == c], minlength=n_clusters).max()
        for c in range(n_clusters)
    ) / len(label_codes)
    if len(label_codes) > n_clusters:
        sil: float | None = float(silhouette_score(x, label_codes, metric="cosine"))
        silhouette_reason = None
    else:
        sil = None
        silhouette_reason = "insufficient_samples"
    values = {
        "NMI": float(normalized_mutual_info_score(label_codes, pred)),
        "ARI": float(adjusted_rand_score(label_codes, pred)),
        "AMI": float(adjusted_mutual_info_score(label_codes, pred)),
        "Silhouette_Score": sil,
        "Purity": float(purity),
    }
    outcomes: dict[str, MetricOutcome] = {}
    for key, value in values.items():
        if value is None:
            outcomes[key] = MetricOutcome(None, "unavailable", silhouette_reason, context)
        else:
            outcomes[key] = MetricOutcome(value, "available", None, context)
    return outcomes


def compute_clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float | None]:
    return {
        key: outcome.value
        for key, outcome in _compute_clustering_metrics_result(embeddings, labels).items()
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


def _compute_linear_probing_metrics_result(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, MetricOutcome]:
    """Ordinary and balanced linear-probe accuracy from one shared CV pass."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import normalize

    x = normalize(embeddings, norm="l2")
    keys = ("Linear_Probing_Acc", "Linear_Probing_Balanced_Acc")
    cv, reason, cv_context = _make_stratified_cv_result(labels)
    context = {
        "family": "linear_probing",
        "classifier": "LogisticRegression",
        "classifier_distance": "none (probabilistic linear classifier)",
        "normalization": "l2",
        "scoring": ["accuracy", "balanced_accuracy"],
        "random_state": 42,
        **cv_context,
    }
    if cv is None:
        return {key: MetricOutcome(None, "unavailable", reason, context) for key in keys}
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
            error_score="raise",
        )
    except Exception as exc:
        return {
            key: MetricOutcome(
                None, "failed", "evaluation_failed", {**context, "error": type(exc).__name__}
            )
            for key in keys
        }
    outcomes: dict[str, MetricOutcome] = {}
    for key, column in zip(keys, ("test_accuracy", "test_balanced_accuracy")):
        value = _finite_mean(scores[column])
        if value is None:
            outcomes[key] = MetricOutcome(None, "unavailable", "nonfinite_result", context)
        else:
            outcomes[key] = MetricOutcome(value, "available", None, context)
    return outcomes


def compute_linear_probing_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float | None]:
    return {
        key: outcome.value
        for key, outcome in _compute_linear_probing_metrics_result(
            embeddings, labels
        ).items()
    }


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
