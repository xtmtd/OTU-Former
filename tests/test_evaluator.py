import sys
import types
import warnings

import numpy as np
import pytest

from otuformer.embedding.evaluator import (
    compute_clustering_metrics,
    compute_knn_accuracy,
    compute_map,
    compute_metric_learning_diagnostics,
    compute_recall_at_k,
)


def make_data(n: int = 50, dim: int = 32, n_classes: int = 5):
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(n_classes), n // n_classes)
    embeddings = rng.standard_normal((n, dim))
    for c in range(n_classes):
        mask = labels == c
        embeddings[mask] += rng.standard_normal(dim) * 3
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings, labels


def test_knn_accuracy_returns_dict():
    embs, labels = make_data()
    result = compute_knn_accuracy(embs, labels, k_values=[1, 5])
    assert "kNN_Acc_k1" in result
    assert "kNN_Acc_k5" in result
    assert 0.0 <= result["kNN_Acc_k1"] <= 1.0


def test_knn_accuracy_skips_unsupported_k_without_warning():
    embs = np.eye(8, dtype=float)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_knn_accuracy(embs, labels, k_values=[1, 20])

    assert result["kNN_Acc_k1"] >= 0.0
    assert result["kNN_Acc_k20"] == 0.0
    assert not caught


def test_recall_at_k_returns_dict():
    embs, labels = make_data()
    result = compute_recall_at_k(embs, labels, k_values=[1, 5, 10])
    assert "Recall@1" in result
    assert 0.0 <= result["Recall@1"] <= 1.0


def test_map_range():
    embs, labels = make_data()
    score = compute_map(embs, labels)
    assert 0.0 <= score <= 1.0


def test_clustering_metrics_keys():
    embs, labels = make_data()
    result = compute_clustering_metrics(embs, labels)
    for key in ["NMI", "ARI", "AMI", "Silhouette_Score", "Purity"]:
        assert key in result


def test_run_umap_caps_neighbors_to_sample_count(tmp_path, monkeypatch):
    seen = {}

    class FakeUMAP:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def fit_transform(self, x):
            return np.zeros((len(x), 2))

    umap_module = types.ModuleType("umap")
    umap_submodule = types.ModuleType("umap.umap_")
    umap_submodule.UMAP = FakeUMAP
    monkeypatch.setitem(sys.modules, "umap", umap_module)
    monkeypatch.setitem(sys.modules, "umap.umap_", umap_submodule)
    monkeypatch.setattr("matplotlib.pyplot.savefig", lambda *args, **kwargs: None)

    from otuformer.embedding.evaluator import run_umap

    run_umap(np.eye(4), None, tmp_path / "umap.pdf", n_neighbors=15)

    assert seen["n_neighbors"] == 3


def test_metric_learning_diagnostics_keys():
    embs, labels = make_data()
    result = compute_metric_learning_diagnostics(embs, labels)
    for key in [
        "Intra_Class_Var",
        "Inter_Class_Dist",
        "Embedding_Norm_Mean",
        "Embedding_Norm_Std",
    ]:
        assert key in result
